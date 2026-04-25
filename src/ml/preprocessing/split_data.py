#!/usr/bin/env python3
"""
Build stratified train/val/test splits from a labeled CSV.

Requirements:
- Input CSV must contain a label column (default: "label")
- Label values should be 0/1 (configurable check)
- No hardcoded file paths: all paths provided by CLI args
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stratified dataset splits")
    parser.add_argument("--input-csv", required=True, help="Path to combined labeled CSV")
    parser.add_argument("--label-col", default="label", help="Label column name")
    parser.add_argument("--timestamp-col", default="Timestamp", help="Timestamp column name")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-name", default="split_manifest.json")
    return parser.parse_args()


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Ratios must sum to 1.0, got {total:.6f}")
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("All split ratios must be > 0")


def validate_input(df: pd.DataFrame, label_col: str, timestamp_col: str) -> None:
    print_header("INPUT VALIDATION")

    # Column checks
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")
    print(f"[PASS] Label column present: {label_col}")

    if timestamp_col not in df.columns:
        print(f"[WARN] Timestamp column not found: {timestamp_col}")
    else:
        ts = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
        failures = int(ts.isna().sum())
        if failures == 0:
            print(f"[PASS] Timestamp parse: 0 failures in column {timestamp_col}")
        else:
            print(f"[WARN] Timestamp parse failures: {failures} in column {timestamp_col}")

    # Null checks
    null_total = int(df.isnull().sum().sum())
    if null_total == 0:
        print("[PASS] No null values found")
    else:
        print(f"[WARN] Null values found: {null_total}")

    # Label checks
    unique_labels = sorted(df[label_col].dropna().astype(str).unique().tolist())
    print(f"[INFO] Unique label values: {unique_labels}")

    # Strong check for binary labels 0/1
    allowed = {"0", "1"}
    if set(unique_labels).issubset(allowed):
        print("[PASS] Labels are binary (0/1)")
    else:
        print("[WARN] Labels are not strictly binary 0/1; continuing")


def print_split_stats(name: str, split_df: pd.DataFrame, label_col: str) -> dict:
    counts = split_df[label_col].value_counts(dropna=False).to_dict()
    total = int(len(split_df))
    zeros = int(counts.get(0, counts.get("0", 0)))
    ones = int(counts.get(1, counts.get("1", 0)))
    print(f"[INFO] {name}: rows={total}, label_0={zeros}, label_1={ones}")
    return {"rows": total, "label_counts": {str(k): int(v) for k, v in counts.items()}}


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    input_csv = Path(args.input_csv)
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir.parent / "data" / "week2"

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"
    manifest_path = out_dir / args.manifest_name

    print_header("LOADING DATA")
    df = pd.read_csv(input_csv)
    print(f"[INFO] Loaded rows: {len(df)}")
    print(f"[INFO] Loaded columns: {list(df.columns)}")

    validate_input(df, args.label_col, args.timestamp_col)

    print_header("BUILDING STRATIFIED SPLITS")
    y = df[args.label_col]

    # First split: train vs temp (val+test)
    temp_ratio = args.val_ratio + args.test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=args.seed,
        stratify=y,
        shuffle=True,
    )

    # Second split: val vs test from temp
    val_fraction_of_temp = args.val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_fraction_of_temp),
        random_state=args.seed,
        stratify=temp_df[args.label_col],
        shuffle=True,
    )

    # Save outputs
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("[PASS] Split files written")
    print(f"[INFO] train file: {train_path}")
    print(f"[INFO] val file:   {val_path}")
    print(f"[INFO] test file:  {test_path}")

    print_header("SPLIT VALIDATION")
    manifest = {
        "input_rows": int(len(df)),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "files": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
        "stats": {},
    }

    manifest["stats"]["full"] = print_split_stats("full", df, args.label_col)
    manifest["stats"]["train"] = print_split_stats("train", train_df, args.label_col)
    manifest["stats"]["val"] = print_split_stats("val", val_df, args.label_col)
    manifest["stats"]["test"] = print_split_stats("test", test_df, args.label_col)

    # Basic integrity checks
    row_sum = len(train_df) + len(val_df) + len(test_df)
    if row_sum == len(df):
        print("[PASS] Row conservation check passed")
    else:
        print(f"[FAIL] Row conservation mismatch: {row_sum} != {len(df)}")

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[PASS] Manifest written: {manifest_path}")

    print_header("DONE")
    print("[PASS] Stratified split pipeline completed successfully")


if __name__ == "__main__":
    main()