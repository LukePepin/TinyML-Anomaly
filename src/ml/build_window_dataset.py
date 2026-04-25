#!/usr/bin/env python3
"""Build raw 32x6 window dataset for TinyML autoencoder training.

Inputs:
- flat CSV with raw IMU rows (ax, ay, az, gx, gy, gz)

Outputs:
- dataset npz with train/val/test windows (flattened)
- metadata json describing schema and split settings
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_AXIS_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build windowed raw dataset for TinyML")
    parser.add_argument("--input-csv", required=True, help="Path to raw IMU CSV")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output folder for dataset.npz + metadata.json",
    )
    parser.add_argument("--window-size", type=int, default=32, help="Window size (default: 32)")
    parser.add_argument("--stride", type=int, default=8, help="Stride between windows (default: 8)")
    parser.add_argument(
        "--axis-cols",
        default=",".join(DEFAULT_AXIS_COLS),
        help="Comma-separated axis columns (default: ax,ay,az,gx,gy,gz)",
    )
    parser.add_argument(
        "--label-col",
        default="",
        help="Optional point label column where 0=normal,1=anomaly",
    )
    parser.add_argument(
        "--window-label-mode",
        choices=["any", "majority"],
        default="any",
        help="If labels exist, derive window label by any or majority positive",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument(
        "--split-mode",
        choices=["chronological", "stratified"],
        default="chronological",
        help="Split strategy: chronological (default) or stratified when labels exist",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified split")
    return parser.parse_args()


def parse_axis_cols(text: str) -> list[str]:
    cols = [c.strip() for c in text.split(",") if c.strip()]
    if len(cols) != 6:
        raise ValueError("--axis-cols must contain exactly 6 columns")
    return cols


def to_window_label(values: np.ndarray, mode: str) -> int:
    if mode == "any":
        return int((values > 0).any())
    # majority
    return int((values > 0).mean() >= 0.5)


def split_indices_chronological(n: int, train_ratio: float, val_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))
    all_idx = np.arange(n, dtype=np.int32)
    return all_idx[:train_end], all_idx[train_end:val_end], all_idx[val_end:]


def split_indices_stratified(
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = np.unique(y)

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) < 3:
            raise ValueError(
                f"Class {int(cls)} has fewer than 3 windows; cannot create train/val/test split"
            )
        rng.shuffle(cls_idx)

        c_train_end = int(len(cls_idx) * train_ratio)
        c_val_end = c_train_end + int(len(cls_idx) * val_ratio)
        c_train_end = max(1, min(c_train_end, len(cls_idx) - 2))
        c_val_end = max(c_train_end + 1, min(c_val_end, len(cls_idx) - 1))

        train_parts.append(cls_idx[:c_train_end])
        val_parts.append(cls_idx[c_train_end:c_val_end])
        test_parts.append(cls_idx[c_val_end:])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    test_idx = np.concatenate(test_parts)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx.astype(np.int32), val_idx.astype(np.int32), test_idx.astype(np.int32)


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    axis_cols = parse_axis_cols(args.axis_cols)
    df = pd.read_csv(input_csv)

    missing = [c for c in axis_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing axis columns in input CSV: {missing}")

    label_col = args.label_col.strip()
    has_labels = bool(label_col)
    if has_labels and label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in input CSV")

    axis_df = df[axis_cols].apply(pd.to_numeric, errors="coerce")
    valid_mask = axis_df.notna().all(axis=1)
    axis_df = axis_df[valid_mask].reset_index(drop=True)

    label_series = None
    if has_labels:
        label_series = (
            pd.to_numeric(df.loc[valid_mask, label_col], errors="coerce")
            .fillna(0)
            .astype(int)
            .reset_index(drop=True)
        )

    data = axis_df.to_numpy(dtype=np.float32)
    n_rows = data.shape[0]
    ws = args.window_size
    stride = args.stride

    if ws <= 0 or stride <= 0:
        raise ValueError("window-size and stride must be > 0")
    if n_rows < ws:
        raise ValueError(f"Not enough rows for one window: rows={n_rows}, window_size={ws}")

    windows: list[np.ndarray] = []
    window_labels: list[int] = []
    window_start_idx: list[int] = []
    for start in range(0, n_rows - ws + 1, stride):
        end = start + ws
        w = data[start:end, :]
        windows.append(w.reshape(-1))
        window_start_idx.append(start)

        if label_series is not None:
            w_labels = label_series.iloc[start:end].to_numpy()
            window_labels.append(to_window_label(w_labels, args.window_label_mode))

    x = np.asarray(windows, dtype=np.float32)
    starts = np.asarray(window_start_idx, dtype=np.int32)
    y = np.asarray(window_labels, dtype=np.int8) if label_series is not None else None

    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 windows to create train/val/test")

    if args.split_mode == "stratified":
        if y is None:
            raise ValueError("--split-mode stratified requires --label-col")
        train_idx, val_idx, test_idx = split_indices_stratified(
            y=y,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    else:
        train_idx, val_idx, test_idx = split_indices_chronological(
            n=n,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )

    x_train = x[train_idx]
    x_val = x[val_idx]
    x_test = x[test_idx]

    s_train = starts[train_idx]
    s_val = starts[val_idx]
    s_test = starts[test_idx]

    payload = {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "start_train": s_train,
        "start_val": s_val,
        "start_test": s_test,
    }
    if y is not None:
        payload["y_train"] = y[train_idx]
        payload["y_val"] = y[val_idx]
        payload["y_test"] = y[test_idx]

    npz_path = out_dir / "dataset.npz"
    np.savez_compressed(npz_path, **payload)

    meta = {
        "input_csv": str(input_csv),
        "axis_cols": axis_cols,
        "window_size": ws,
        "stride": stride,
        "feature_dim": int(ws * len(axis_cols)),
        "total_rows": int(n_rows),
        "total_windows": int(n),
        "splits": {
            "train": int(len(x_train)),
            "val": int(len(x_val)),
            "test": int(len(x_test)),
        },
        "split_mode": args.split_mode,
        "seed": args.seed,
        "has_window_labels": y is not None,
        "window_label_mode": args.window_label_mode if y is not None else None,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Raw window dataset built")
    print(f"out_dir={out_dir}")
    print(f"feature_dim={meta['feature_dim']}")
    print(f"windows(train/val/test)={len(x_train)}/{len(x_val)}/{len(x_test)}")


if __name__ == "__main__":
    main()
