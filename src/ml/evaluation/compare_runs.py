#!/usr/bin/env python3
"""Compare windowed split runs by reading window_manifest.json files.

Usage:
python compare_window_runs.py --runs-root window_runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    parser = argparse.ArgumentParser(description="Compare window run manifests")
    parser.add_argument("--runs-root", default=str(project_root / "results" / "week2" / "window_runs"))
    parser.add_argument("--out-csv", default="window_runs_comparison.csv")
    parser.add_argument("--out-json", default="window_runs_comparison.json")
    return parser.parse_args()


def safe_ratio(label_counts: dict[str, int]) -> float:
    total = int(label_counts.get("0", 0)) + int(label_counts.get("1", 0))
    if total == 0:
        return 0.0
    return int(label_counts.get("1", 0)) / total


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)

    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")

    manifests = sorted(runs_root.glob("*/window_manifest.json"))
    if not manifests:
        raise RuntimeError(f"No manifests found under {runs_root}")

    records: list[dict[str, object]] = []
    for manifest_path in manifests:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        cfg = payload["config"]
        stats = payload["split_stats"]

        train_counts = stats["train_label_counts"]
        val_counts = stats["val_label_counts"]
        test_counts = stats["test_label_counts"]

        records.append(
            {
                "run_dir": str(manifest_path.parent),
                "window_size": int(cfg["window_size"]),
                "stride": int(cfg["stride"]),
                "threshold": float(cfg["threshold"]),
                "drop_ambiguous": bool(cfg["drop_ambiguous"]),
                "train_rows": int(stats["train_rows"]),
                "val_rows": int(stats["val_rows"]),
                "test_rows": int(stats["test_rows"]),
                "train_label0": int(train_counts.get("0", 0)),
                "train_label1": int(train_counts.get("1", 0)),
                "val_label0": int(val_counts.get("0", 0)),
                "val_label1": int(val_counts.get("1", 0)),
                "test_label0": int(test_counts.get("0", 0)),
                "test_label1": int(test_counts.get("1", 0)),
                "train_label1_ratio": safe_ratio(train_counts),
                "val_label1_ratio": safe_ratio(val_counts),
                "test_label1_ratio": safe_ratio(test_counts),
            }
        )

    df = pd.DataFrame(records).sort_values(["window_size", "stride", "threshold"])

    out_csv = runs_root / args.out_csv
    out_json = runs_root / args.out_json
    df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(records, indent=2), encoding="utf-8")

    print("Comparison complete")
    print(f"runs analyzed={len(df)}")
    print(f"csv={out_csv}")
    print(f"json={out_json}")
    print("\nTop summary:")
    print(df[["window_size", "stride", "threshold", "train_rows", "val_rows", "test_rows", "test_label1_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
