#!/usr/bin/env python3
"""Automate window generation + model training sweeps with consolidated outputs.

Default mode:
- Generate multiple window configs with windowed_time_split.py.
- Run train_window_model.py parameter sweeps for each window config.
- Write one consolidated CSV/JSON across all runs.

Compatibility mode:
- If --run-dir is provided, skip window generation and only sweep that folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_float_list(text: str) -> list[float]:
    values = [v.strip() for v in text.split(",") if v.strip()]
    return [float(v) for v in values]


def parse_int_list(text: str) -> list[int]:
    values = [v.strip() for v in text.split(",") if v.strip()]
    return [int(v) for v in values]


def parse_hidden_sets(text: str) -> list[str]:
    return [g.strip() for g in text.split(";") if g.strip()]


def config_dir_name(window_size: int, stride: int, threshold: float) -> str:
    thr_text = f"{threshold:.2f}".replace(".", "p")
    return f"ws{window_size}_st{stride}_thr{thr_text}"


def fmt_float_for_tag(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_data_root = project_root / "data" / "week2"
    default_root = project_root / "results" / "week2" / "window_sweep_results"

    parser = argparse.ArgumentParser(
        description="Automate window config generation + train_window_model sweeps"
    )

    parser.add_argument(
        "--run-dir",
        help="Optional: existing window run folder to sweep (skip window generation)",
    )
    parser.add_argument(
        "--window-split-script",
        default=str(script_dir / "windowed_time_split.py"),
        help="Path to windowed_time_split.py",
    )
    parser.add_argument(
        "--trainer-script",
        default=str(script_dir / "train_window_model.py"),
        help="Path to train_window_model.py",
    )

    parser.add_argument(
        "--baseline-csv",
        default=str(default_data_root / "baseline_labeled.csv"),
        help="Path to baseline labeled CSV",
    )
    parser.add_argument(
        "--adversarial-csv",
        default=str(default_data_root / "adversarial_labeled.csv"),
        help="Path to adversarial labeled CSV",
    )

    parser.add_argument(
        "--window-sizes",
        default="64,128",
        help="Comma-separated window sizes for generation",
    )
    parser.add_argument(
        "--window-strides",
        default="16,32",
        help="Comma-separated strides for generation",
    )
    parser.add_argument(
        "--window-thresholds",
        default="0.30",
        help="Comma-separated window label thresholds for generation",
    )
    parser.add_argument(
        "--drop-ambiguous",
        action="store_true",
        help="Pass --drop-ambiguous to windowed_time_split.py",
    )

    parser.add_argument(
        "--window-configs-root",
        default=str(default_root / "window_configs"),
        help="Output root for generated window config folders",
    )
    parser.add_argument(
        "--summaries-root",
        default=str(default_root / "summaries"),
        help="Output root for consolidated summary files",
    )

    parser.add_argument(
        "--thresholds",
        default="99.5,99.0,97.5,95.0,92.5,90.0",
        help="Comma-separated threshold percentiles",
    )
    parser.add_argument(
        "--max-iters",
        default="400,250",
        help="Comma-separated max_iter values",
    )
    parser.add_argument(
        "--hidden-sets",
        default="512,256,512;256,128,256;128,64,128",
        help="Semicolon-separated hidden layer sets",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed passed to trainer",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip run artifacts that already have eval_report.json and threshold.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan and commands without executing",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop sweep if any run fails",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_window_configs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return [run_dir]

    split_script = Path(args.window_split_script).resolve()
    if not split_script.exists():
        raise FileNotFoundError(f"Window split script not found: {split_script}")

    window_sizes = parse_int_list(args.window_sizes)
    window_strides = parse_int_list(args.window_strides)
    window_thresholds = parse_float_list(args.window_thresholds)

    if not window_sizes:
        raise ValueError("No window sizes provided")
    if not window_strides:
        raise ValueError("No window strides provided")
    if not window_thresholds:
        raise ValueError("No window thresholds provided")

    output_root = Path(args.window_configs_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_dirs: list[Path] = []
    for ws in window_sizes:
        for st in window_strides:
            for wthr in window_thresholds:
                cfg_name = config_dir_name(ws, st, wthr)
                run_dir = output_root / cfg_name
                run_dirs.append(run_dir)

                cmd = [
                    sys.executable,
                    str(split_script),
                    "--baseline-csv",
                    str(Path(args.baseline_csv).resolve()),
                    "--adversarial-csv",
                    str(Path(args.adversarial_csv).resolve()),
                    "--window-size",
                    str(ws),
                    "--stride",
                    str(st),
                    "--threshold",
                    str(wthr),
                    "--output-root",
                    str(output_root),
                ]
                if args.drop_ambiguous:
                    cmd.append("--drop-ambiguous")

                if args.dry_run:
                    print("[dry-run] window command:")
                    print("  " + " ".join(cmd))
                    continue

                print(f"Generating window config: {cfg_name}")
                completed = subprocess.run(cmd, capture_output=True, text=True)
                if completed.returncode != 0:
                    print(completed.stdout)
                    print(completed.stderr)
                    raise RuntimeError(f"Window generation failed for {cfg_name}")

    return run_dirs


def read_window_manifest(run_dir: Path) -> dict[str, object]:
    manifest_path = run_dir / "window_manifest.json"
    if not manifest_path.exists():
        return {
            "window_config": run_dir.name,
            "window_size": None,
            "window_stride": None,
            "window_threshold": None,
            "drop_ambiguous": None,
            "train_rows": None,
            "val_rows": None,
            "test_rows": None,
        }

    payload = load_json(manifest_path)
    cfg = payload.get("config", {})
    stats = payload.get("split_stats", {})
    return {
        "window_config": run_dir.name,
        "window_size": cfg.get("window_size"),
        "window_stride": cfg.get("stride"),
        "window_threshold": cfg.get("threshold"),
        "drop_ambiguous": cfg.get("drop_ambiguous"),
        "train_rows": stats.get("train_rows"),
        "val_rows": stats.get("val_rows"),
        "test_rows": stats.get("test_rows"),
    }


def train_and_collect(
    run_dir: Path,
    trainer: Path,
    thresholds: list[float],
    max_iters: list[int],
    hidden_sets: list[str],
    seed: int,
    resume: bool,
    dry_run: bool,
    stop_on_error: bool,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    model_runs_dir = run_dir / "model_runs"
    if not dry_run:
        model_runs_dir.mkdir(parents=True, exist_ok=True)
    window_info = read_window_manifest(run_dir)

    total = len(thresholds) * len(max_iters) * len(hidden_sets)
    counter = 0

    for threshold in thresholds:
        for max_iter in max_iters:
            for hidden in hidden_sets:
                counter += 1
                hidden_tag = hidden.replace(",", "-")
                run_tag = (
                    f"thr{fmt_float_for_tag(threshold)}_"
                    f"it{max_iter}_h{hidden_tag}_seed{seed}"
                )
                artifact_dir = model_runs_dir / run_tag
                if not dry_run:
                    artifact_dir.mkdir(parents=True, exist_ok=True)

                row: dict[str, object] = {
                    "status": "planned" if dry_run else "ok",
                    "window_config": window_info["window_config"],
                    "window_size": window_info["window_size"],
                    "window_stride": window_info["window_stride"],
                    "window_threshold": window_info["window_threshold"],
                    "drop_ambiguous": window_info["drop_ambiguous"],
                    "window_train_rows": window_info["train_rows"],
                    "window_val_rows": window_info["val_rows"],
                    "window_test_rows": window_info["test_rows"],
                    "run_dir": str(run_dir),
                    "run_tag": run_tag,
                    "threshold_percentile": threshold,
                    "max_iter": max_iter,
                    "hidden_layers": hidden,
                    "seed": seed,
                    "artifact_dir": str(artifact_dir),
                }

                existing_eval = artifact_dir / "eval_report.json"
                existing_thr = artifact_dir / "threshold.json"
                if resume and (not dry_run) and existing_eval.exists() and existing_thr.exists():
                    row["status"] = "skipped"
                    eval_payload = load_json(existing_eval)
                    thr_payload = load_json(existing_thr)
                    test_metrics = eval_payload.get("test_metrics", {})
                    val_metrics = eval_payload.get("validation_metrics", {})
                    row.update(
                        {
                            "threshold": float(thr_payload.get("threshold", 0.0)),
                            "test_f1": float(test_metrics.get("f1", 0.0)),
                            "test_precision": float(test_metrics.get("precision", 0.0)),
                            "test_recall": float(test_metrics.get("recall", 0.0)),
                            "test_fpr_normal": float(test_metrics.get("false_positive_rate_normal", 0.0)),
                            "val_f1": float(val_metrics.get("f1", 0.0)),
                            "val_precision": float(val_metrics.get("precision", 0.0)),
                            "val_recall": float(val_metrics.get("recall", 0.0)),
                            "val_fpr_normal": float(val_metrics.get("false_positive_rate_normal", 0.0)),
                        }
                    )
                    records.append(row)
                    print(f"[{counter}/{total}] skipped {run_dir.name} {run_tag}")
                    continue

                cmd = [
                    sys.executable,
                    str(trainer),
                    "--run-dir",
                    str(run_dir),
                    "--max-iter",
                    str(max_iter),
                    "--hidden-layers",
                    hidden,
                    "--threshold-percentile",
                    str(threshold),
                    "--seed",
                    str(seed),
                ]

                if dry_run:
                    print(f"[dry-run][{counter}/{total}] train command:")
                    print("  " + " ".join(cmd))
                    records.append(row)
                    continue

                print(f"[{counter}/{total}] training {run_dir.name} {run_tag}")
                completed = subprocess.run(cmd, capture_output=True, text=True)

                (artifact_dir / "stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
                (artifact_dir / "stderr.txt").write_text(completed.stderr or "", encoding="utf-8")

                if completed.returncode != 0:
                    row["status"] = "failed"
                    records.append(row)
                    print(f"  -> failed (exit={completed.returncode}), see {artifact_dir}")
                    if stop_on_error:
                        return records
                    continue

                model_dir = run_dir / "model"
                eval_path = model_dir / "eval_report.json"
                threshold_path = model_dir / "threshold.json"

                if eval_path.exists():
                    shutil.copy2(eval_path, artifact_dir / "eval_report.json")
                if threshold_path.exists():
                    shutil.copy2(threshold_path, artifact_dir / "threshold.json")

                if eval_path.exists() and threshold_path.exists():
                    eval_payload = load_json(eval_path)
                    thr_payload = load_json(threshold_path)
                    test_metrics = eval_payload.get("test_metrics", {})
                    val_metrics = eval_payload.get("validation_metrics", {})
                    row.update(
                        {
                            "threshold": float(thr_payload.get("threshold", 0.0)),
                            "test_f1": float(test_metrics.get("f1", 0.0)),
                            "test_precision": float(test_metrics.get("precision", 0.0)),
                            "test_recall": float(test_metrics.get("recall", 0.0)),
                            "test_fpr_normal": float(test_metrics.get("false_positive_rate_normal", 0.0)),
                            "val_f1": float(val_metrics.get("f1", 0.0)),
                            "val_precision": float(val_metrics.get("precision", 0.0)),
                            "val_recall": float(val_metrics.get("recall", 0.0)),
                            "val_fpr_normal": float(val_metrics.get("false_positive_rate_normal", 0.0)),
                        }
                    )

                records.append(row)

    return records


def write_csv(path: Path, records: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    args = parse_args()

    trainer = Path(args.trainer_script).resolve()
    if not trainer.exists():
        raise FileNotFoundError(f"Trainer script not found: {trainer}")

    thresholds = parse_float_list(args.thresholds)
    max_iters = parse_int_list(args.max_iters)
    hidden_sets = parse_hidden_sets(args.hidden_sets)
    if not thresholds:
        raise ValueError("No thresholds provided")
    if not max_iters:
        raise ValueError("No max-iters provided")
    if not hidden_sets:
        raise ValueError("No hidden-sets provided")

    run_dirs = build_window_configs(args)
    run_dirs = [p.resolve() for p in run_dirs]

    total_planned = len(run_dirs) * len(thresholds) * len(max_iters) * len(hidden_sets)
    print(f"Window configs: {len(run_dirs)}")
    print(f"Planned train runs: {total_planned}")

    all_records: list[dict[str, object]] = []
    for run_dir in run_dirs:
        if not run_dir.exists() and (not args.dry_run):
            raise FileNotFoundError(f"Run directory not found after generation: {run_dir}")

        records = train_and_collect(
            run_dir=run_dir,
            trainer=trainer,
            thresholds=thresholds,
            max_iters=max_iters,
            hidden_sets=hidden_sets,
            seed=args.seed,
            resume=args.resume,
            dry_run=args.dry_run,
            stop_on_error=args.stop_on_error,
        )
        all_records.extend(records)
        if args.stop_on_error and any(r.get("status") == "failed" for r in records):
            break

    successful = [r for r in all_records if r.get("status") in {"ok", "skipped"} and "test_f1" in r]
    ranked = sorted(
        successful,
        key=lambda r: (
            float(r.get("test_f1", 0.0)),
            float(r.get("test_recall", 0.0)),
            -float(r.get("test_fpr_normal", 1.0)),
        ),
        reverse=True,
    )

    summaries_root = Path(args.summaries_root).resolve()
    fields = [
        "status",
        "window_config",
        "window_size",
        "window_stride",
        "window_threshold",
        "drop_ambiguous",
        "window_train_rows",
        "window_val_rows",
        "window_test_rows",
        "run_dir",
        "run_tag",
        "threshold_percentile",
        "max_iter",
        "hidden_layers",
        "seed",
        "threshold",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_fpr_normal",
        "val_f1",
        "val_precision",
        "val_recall",
        "val_fpr_normal",
        "artifact_dir",
    ]

    consolidated_csv = summaries_root / "consolidated_sweep.csv"
    consolidated_json = summaries_root / "consolidated_sweep.json"
    ranked_json = summaries_root / "ranked_by_test_f1.json"

    write_csv(consolidated_csv, all_records, fields)
    consolidated_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "window_configs": [str(p) for p in run_dirs],
        "total_records": len(all_records),
        "successful_records": len(successful),
        "records": all_records,
    }
    summaries_root.mkdir(parents=True, exist_ok=True)
    consolidated_json.write_text(json.dumps(consolidated_payload, indent=2), encoding="utf-8")
    ranked_json.write_text(json.dumps(ranked, indent=2), encoding="utf-8")

    print("Sweep complete")
    print(f"consolidated_csv={consolidated_csv}")
    print(f"consolidated_json={consolidated_json}")
    print(f"ranked_json={ranked_json}")

    if ranked:
        best = ranked[0]
        print("Best run:")
        print(
            "  "
            f"window_config={best.get('window_config')} "
            f"test_f1={best.get('test_f1')} "
            f"test_recall={best.get('test_recall')} "
            f"test_precision={best.get('test_precision')} "
            f"test_fpr_normal={best.get('test_fpr_normal')} "
            f"threshold={best.get('threshold')} "
            f"threshold_percentile={best.get('threshold_percentile')} "
            f"hidden_layers={best.get('hidden_layers')} "
            f"max_iter={best.get('max_iter')}"
        )


if __name__ == "__main__":
    main()
