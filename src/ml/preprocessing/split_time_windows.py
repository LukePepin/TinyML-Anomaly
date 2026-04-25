#!/usr/bin/env python3
"""Build windowed chronological train/val/test splits from labeled telemetry CSVs.

Per-source workflow:
1) Sort rows by timestamp.
2) Build sliding windows.
3) Label each window by anomaly fraction threshold.
4) Split windows chronologically into train/val/test.

Outputs are written under a config subfolder, e.g.:
window_runs/ws64_st16_thr0p30/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

FEATURE_COLS = [
    "Accel_X",
    "Accel_Y",
    "Accel_Z",
    "Gyro_X",
    "Gyro_Y",
    "Gyro_Z",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_root = project_root / "data" / "week2"
    results_root = project_root / "results" / "week2"
    parser = argparse.ArgumentParser(description="Build windowed chronological splits")
    parser.add_argument("--baseline-csv", default=str(data_root / "baseline_labeled.csv"))
    parser.add_argument("--adversarial-csv", default=str(data_root / "adversarial_labeled.csv"))
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.30, help="Window labeled 1 if anomaly_fraction >= threshold")
    parser.add_argument("--drop-ambiguous", action="store_true", help="Drop windows where 0.10 < anomaly_fraction < threshold")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--output-root", default=str(results_root / "window_runs"))
    return parser.parse_args()


def check_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")


def config_dir_name(window_size: int, stride: int, threshold: float) -> str:
    thr_text = f"{threshold:.2f}".replace(".", "p")
    return f"ws{window_size}_st{stride}_thr{thr_text}"


def load_and_sort(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = FEATURE_COLS + ["Timestamp", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    bad_ts = int(df["Timestamp"].isna().sum())
    if bad_ts > 0:
        raise ValueError(f"Found {bad_ts} unparseable timestamps in {csv_path}")

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    bad_label = int(df["label"].isna().sum())
    if bad_label > 0:
        raise ValueError(f"Found {bad_label} invalid labels in {csv_path}")

    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df


def build_windows(
    df: pd.DataFrame,
    source_name: str,
    window_size: int,
    stride: int,
    threshold: float,
    drop_ambiguous: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n = len(df)
    if n < window_size:
        return pd.DataFrame()

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        chunk = df.iloc[start:end]
        anomaly_fraction = float((chunk["label"] == 1).mean())

        if drop_ambiguous and (0.10 < anomaly_fraction < threshold):
            continue

        window_label = 1 if anomaly_fraction >= threshold else 0

        row: dict[str, object] = {
            "source": source_name,
            "start_index": int(start),
            "end_index": int(end - 1),
            "start_ts": chunk["Timestamp"].iloc[0].isoformat(),
            "end_ts": chunk["Timestamp"].iloc[-1].isoformat(),
            "anomaly_fraction": anomaly_fraction,
            "window_label": window_label,
        }

        values = chunk[FEATURE_COLS].to_numpy()
        for t in range(window_size):
            for f_idx, feature_name in enumerate(FEATURE_COLS):
                row[f"{feature_name}_t{t:03d}"] = float(values[t, f_idx])

        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def chronological_split(
    windows: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = len(windows)
    if total == 0:
        return windows.copy(), windows.copy(), windows.copy()

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train = windows.iloc[:train_end].copy()
    val = windows.iloc[train_end:val_end].copy()
    test = windows.iloc[val_end:].copy()
    return train, val, test


def label_counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {"0": 0, "1": 0}
    counts = df["window_label"].value_counts().to_dict()
    return {"0": int(counts.get(0, 0)), "1": int(counts.get(1, 0))}


def main() -> None:
    args = parse_args()
    check_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    baseline_path = Path(args.baseline_csv)
    adversarial_path = Path(args.adversarial_csv)

    baseline_df = load_and_sort(baseline_path)
    adversarial_df = load_and_sort(adversarial_path)

    base_windows = build_windows(
        baseline_df,
        source_name="baseline",
        window_size=args.window_size,
        stride=args.stride,
        threshold=args.threshold,
        drop_ambiguous=args.drop_ambiguous,
    )
    adv_windows = build_windows(
        adversarial_df,
        source_name="adversarial",
        window_size=args.window_size,
        stride=args.stride,
        threshold=args.threshold,
        drop_ambiguous=args.drop_ambiguous,
    )

    base_train, base_val, base_test = chronological_split(
        base_windows, args.train_ratio, args.val_ratio, args.test_ratio
    )
    adv_train, adv_val, adv_test = chronological_split(
        adv_windows, args.train_ratio, args.val_ratio, args.test_ratio
    )

    train = pd.concat([base_train, adv_train], ignore_index=True)
    val = pd.concat([base_val, adv_val], ignore_index=True)
    test = pd.concat([base_test, adv_test], ignore_index=True)

    out_root = Path(args.output_root)
    run_dir = out_root / config_dir_name(args.window_size, args.stride, args.threshold)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_path = run_dir / "windowed_train.csv"
    val_path = run_dir / "windowed_val.csv"
    test_path = run_dir / "windowed_test.csv"
    manifest_path = run_dir / "window_manifest.json"

    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    manifest = {
        "config": {
            "window_size": args.window_size,
            "stride": args.stride,
            "threshold": args.threshold,
            "drop_ambiguous": bool(args.drop_ambiguous),
            "ratios": {
                "train": args.train_ratio,
                "val": args.val_ratio,
                "test": args.test_ratio,
            },
        },
        "inputs": {
            "baseline_csv": str(baseline_path),
            "adversarial_csv": str(adversarial_path),
        },
        "sources": {
            "baseline_windows_total": int(len(base_windows)),
            "adversarial_windows_total": int(len(adv_windows)),
            "baseline_splits": {
                "train": int(len(base_train)),
                "val": int(len(base_val)),
                "test": int(len(base_test)),
            },
            "adversarial_splits": {
                "train": int(len(adv_train)),
                "val": int(len(adv_val)),
                "test": int(len(adv_test)),
            },
        },
        "outputs": {
            "train_csv": str(train_path),
            "val_csv": str(val_path),
            "test_csv": str(test_path),
            "manifest": str(manifest_path),
        },
        "split_stats": {
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "test_rows": int(len(test)),
            "train_label_counts": label_counts(train),
            "val_label_counts": label_counts(val),
            "test_label_counts": label_counts(test),
        },
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Windowed chronological split complete")
    print(f"Run dir: {run_dir}")
    print(f"train rows={len(train)} labels={label_counts(train)}")
    print(f"val rows={len(val)} labels={label_counts(val)}")
    print(f"test rows={len(test)} labels={label_counts(test)}")


if __name__ == "__main__":
    main()
