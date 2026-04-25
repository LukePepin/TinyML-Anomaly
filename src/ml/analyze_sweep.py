#!/usr/bin/env python3
"""Read consolidated sweep CSV and pick the best setting.

Default ranking:
1) Higher test_f1
2) Higher test_recall
3) Lower test_fpr_normal
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    summaries_root = project_root / "results" / "week2" / "window_sweep_results" / "summaries"
    default_csv = summaries_root / "consolidated_sweep.csv"
    default_out = summaries_root / "best_setting.json"

    parser = argparse.ArgumentParser(description="Find best setting from consolidated sweep CSV")
    parser.add_argument("--csv", default=str(default_csv), help="Path to consolidated_sweep.csv")
    parser.add_argument("--out-json", default=str(default_out), help="Output JSON path")
    parser.add_argument("--status", default="ok", help="Comma-separated statuses to include (default: ok)")
    parser.add_argument("--primary-metric", default="test_f1", help="Primary metric (descending)")
    parser.add_argument("--secondary-metric", default="test_recall", help="Secondary metric (descending)")
    parser.add_argument(
        "--tertiary-low-metric",
        default="test_fpr_normal",
        help="Tertiary metric where lower is better",
    )
    parser.add_argument(
        "--max-fpr",
        type=float,
        help="Optional upper bound for test_fpr_normal",
    )
    return parser.parse_args()


def to_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv).resolve()
    out_json = Path(args.out_json).resolve()
    allowed_status = {s.strip() for s in args.status.split(",") if s.strip()}

    rows = load_rows(csv_path)
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")

    filtered: list[dict[str, str]] = []
    for row in rows:
        status = (row.get("status") or "").strip()
        if allowed_status and status not in allowed_status:
            continue
        if args.max_fpr is not None:
            fpr = to_float(row.get("test_fpr_normal"), default=1.0)
            if fpr > args.max_fpr:
                continue
        filtered.append(row)

    if not filtered:
        raise RuntimeError("No rows remain after filtering")

    primary = args.primary_metric
    secondary = args.secondary_metric
    tertiary = args.tertiary_low_metric

    ranked = sorted(
        filtered,
        key=lambda row: (
            to_float(row.get(primary), default=0.0),
            to_float(row.get(secondary), default=0.0),
            -to_float(row.get(tertiary), default=1.0),
        ),
        reverse=True,
    )
    best = ranked[0]

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(csv_path),
        "row_count": len(rows),
        "filtered_row_count": len(filtered),
        "criteria": {
            "status": sorted(allowed_status),
            "primary_metric_desc": primary,
            "secondary_metric_desc": secondary,
            "tertiary_metric_asc": tertiary,
            "max_fpr": args.max_fpr,
        },
        "best": best,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Best setting selected")
    print(f"source_csv={csv_path}")
    print(f"out_json={out_json}")
    print(
        "best: "
        f"window_config={best.get('window_config')} "
        f"run_tag={best.get('run_tag')} "
        f"test_f1={best.get(primary)} "
        f"test_recall={best.get(secondary)} "
        f"test_fpr_normal={best.get(tertiary)}"
    )


if __name__ == "__main__":
    main()
