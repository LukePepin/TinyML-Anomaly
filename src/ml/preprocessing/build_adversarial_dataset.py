#!/usr/bin/env python3
"""Build a balanced adversarial dataset via stratified sampling.

This script reads multiple adversarial CSV sources and writes a new sampled CSV
without modifying any original files.

Default behavior is tuned for this repository:
- tinyml-anomaly/data/adversarial1_data.csv
- tinyml-anomaly/data/adversarial2_data.csv
- target rows: 43000

Sampling strategy:
1) Try to allocate an equal share per source (stratum).
2) Cap each source at available rows.
3) Distribute any remainder across sources that still have rows available.
4) Randomly sample rows within each source using a reproducible seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Sequence


def read_csv_rows(csv_path: Path) -> tuple[list[str], list[list[str]]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError(f"CSV has no header: {csv_path}")
        rows = [row for row in reader if row]
    return header, rows


def compute_balanced_allocations(sizes: Sequence[int], target: int) -> list[int]:
    if target <= 0:
        raise ValueError("target must be > 0")
    if not sizes or any(s < 0 for s in sizes):
        raise ValueError("sizes must be non-empty and non-negative")

    total_available = sum(sizes)
    if target > total_available:
        raise ValueError(
            f"Target {target} exceeds total available rows {total_available}."
        )

    n = len(sizes)
    # First pass: equal-share assignment with per-source caps.
    per_source = target // n
    allocations = [min(size, per_source) for size in sizes]

    assigned = sum(allocations)
    remainder = target - assigned

    # Round-robin remainder distribution to preserve balance where possible.
    i = 0
    while remainder > 0:
        idx = i % n
        if allocations[idx] < sizes[idx]:
            allocations[idx] += 1
            remainder -= 1
        i += 1

    return allocations


def write_csv(csv_path: Path, header: list[str], rows: list[list[str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"

    parser = argparse.ArgumentParser(
        description="Build balanced 43k adversarial CSV by stratified sampling"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            str(data_dir / "adversarial1_data.csv"),
            str(data_dir / "adversarial2_data.csv"),
        ],
        help="Input adversarial CSV paths (each path is one stratum)",
    )
    parser.add_argument(
        "--target-rows",
        type=int,
        default=43000,
        help="Total rows to sample into the output file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--out",
        default=str(data_dir / "adversarial_balanced_43k.csv"),
        help="Output CSV path (new derived file)",
    )
    parser.add_argument(
        "--meta-out",
        default=str(data_dir / "adversarial_balanced_43k_metadata.json"),
        help="Output metadata JSON path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_paths = [Path(p) for p in args.inputs]
    out_path = Path(args.out)
    meta_out_path = Path(args.meta_out)

    rng = random.Random(args.seed)

    source_headers: list[list[str]] = []
    source_rows: list[list[list[str]]] = []

    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        header, rows = read_csv_rows(path)
        source_headers.append(header)
        source_rows.append(rows)

    reference_header = source_headers[0]
    for idx, header in enumerate(source_headers[1:], start=1):
        if header != reference_header:
            raise ValueError(
                "Input CSV schemas do not match. "
                f"Header mismatch between {input_paths[0]} and {input_paths[idx]}"
            )

    sizes = [len(rows) for rows in source_rows]
    allocations = compute_balanced_allocations(sizes, args.target_rows)

    sampled_rows: list[list[str]] = []
    per_source_result: list[dict[str, int | str]] = []

    for path, rows, take in zip(input_paths, source_rows, allocations):
        if take > len(rows):
            raise ValueError(f"Allocation exceeds rows for {path}: {take} > {len(rows)}")

        indices = list(range(len(rows)))
        rng.shuffle(indices)
        chosen = [rows[i] for i in indices[:take]]
        sampled_rows.extend(chosen)

        per_source_result.append(
            {
                "file": str(path),
                "rows_available": len(rows),
                "rows_sampled": take,
            }
        )

    rng.shuffle(sampled_rows)

    write_csv(out_path, reference_header, sampled_rows)

    metadata = {
        "output_file": str(out_path),
        "target_rows": args.target_rows,
        "actual_rows": len(sampled_rows),
        "seed": args.seed,
        "strategy": "balanced_stratified_equal_share_with_remainder",
        "sources": per_source_result,
    }

    meta_out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_out_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Balanced adversarial dataset created.")
    print(f"Output CSV: {out_path}")
    print(f"Output metadata: {meta_out_path}")
    for item in per_source_result:
        print(
            f"  - {item['file']}: sampled {item['rows_sampled']} / {item['rows_available']}"
        )


if __name__ == "__main__":
    main()
