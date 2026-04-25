#!/usr/bin/env python3
"""Prepare Arduino TinyML runtime bundle from exported training artifacts.

This script copies the generated `model_data.h` into the target Arduino sketch
folder and writes `runtime_config.h` containing scaling vectors and threshold.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Arduino TinyML runtime bundle")
    parser.add_argument(
        "--model-header",
        default="",
        help="Optional path to prebuilt model_data.h (if omitted, --model-tflite is required)",
    )
    parser.add_argument(
        "--model-tflite",
        default="",
        help="Optional path to .tflite file to convert into model_data.h",
    )
    parser.add_argument(
        "--array-name",
        default="g_tinyml_model_data",
        help="C symbol name when generating header from --model-tflite",
    )
    parser.add_argument("--scaling-json", required=True, help="Path to scaling.json")
    parser.add_argument("--threshold-json", required=True, help="Path to threshold.json")
    parser.add_argument("--out-sketch-dir", required=True, help="Arduino sketch directory")
    parser.add_argument("--window-size", type=int, default=32, help="Window size")
    parser.add_argument("--axis-count", type=int, default=6, help="Axis count")
    return parser.parse_args()


def c_array_bytes(data: bytes, array_name: str) -> str:
    values = list(data)
    lines = [f"const unsigned char {array_name}[] = {{"]
    row: list[str] = []
    for i, b in enumerate(values):
        row.append(f"0x{b:02x}")
        if len(row) == 12 or i == len(values) - 1:
            lines.append("  " + ", ".join(row) + ",")
            row = []
    lines.append("};")
    lines.append(f"const unsigned int {array_name}_len = {len(values)};")
    return "\n".join(lines) + "\n"


def format_float_array(name: str, values: list[float], cols_per_row: int = 6) -> str:
    lines: list[str] = [f"static const float {name}[{len(values)}] = {{"]
    for i in range(0, len(values), cols_per_row):
        chunk = values[i : i + cols_per_row]
        row = ", ".join(f"{v:.9f}f" for v in chunk)
        lines.append(f"  {row},")
    lines.append("};")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    model_header_arg = args.model_header.strip()
    model_tflite_arg = args.model_tflite.strip()
    scaling_path = Path(args.scaling_json)
    threshold_path = Path(args.threshold_json)
    out_sketch_dir = Path(args.out_sketch_dir)

    if not model_header_arg and not model_tflite_arg:
        raise ValueError("Provide either --model-header or --model-tflite")
    if model_header_arg and model_tflite_arg:
        raise ValueError("Use only one of --model-header or --model-tflite")

    if not scaling_path.exists():
        raise FileNotFoundError(f"scaling json not found: {scaling_path}")
    if not threshold_path.exists():
        raise FileNotFoundError(f"threshold json not found: {threshold_path}")

    out_sketch_dir.mkdir(parents=True, exist_ok=True)

    scaling = json.loads(scaling_path.read_text(encoding="utf-8"))
    threshold = json.loads(threshold_path.read_text(encoding="utf-8"))

    mean = scaling.get("mean")
    std = scaling.get("std")
    input_dim = int(scaling.get("input_dim", 0))
    if not isinstance(mean, list) or not isinstance(std, list):
        raise ValueError("scaling.json must contain 'mean' and 'std' arrays")
    if len(mean) != len(std):
        raise ValueError("mean/std length mismatch in scaling.json")
    if input_dim <= 0:
        input_dim = len(mean)
    if input_dim != len(mean):
        raise ValueError("input_dim does not match mean/std length")

    expected_dim = args.window_size * args.axis_count
    if expected_dim != input_dim:
        raise ValueError(
            f"input_dim mismatch: expected window_size*axis_count={expected_dim}, got {input_dim}"
        )

    anomaly_threshold = float(threshold["threshold"])

    runtime_header = out_sketch_dir / "runtime_config.h"
    runtime_guard = "RUNTIME_CONFIG_H"
    runtime_text = "\n".join(
        [
            f"#ifndef {runtime_guard}",
            f"#define {runtime_guard}",
            "",
            f"static const int kWindowSize = {args.window_size};",
            f"static const int kAxisCount = {args.axis_count};",
            f"static const int kInputDim = {input_dim};",
            f"static const float kAnomalyThreshold = {anomaly_threshold:.9f}f;",
            "",
            format_float_array("kInputMean", [float(v) for v in mean]),
            "",
            format_float_array("kInputStd", [float(v) for v in std]),
            "",
            f"#endif  // {runtime_guard}",
            "",
        ]
    )
    runtime_header.write_text(runtime_text, encoding="utf-8")

    model_dst = out_sketch_dir / "model_data.h"
    if model_header_arg:
        model_header = Path(model_header_arg)
        if not model_header.exists():
            raise FileNotFoundError(f"model header not found: {model_header}")
        shutil.copyfile(model_header, model_dst)
    else:
        model_tflite = Path(model_tflite_arg)
        if not model_tflite.exists():
            raise FileNotFoundError(f"model tflite not found: {model_tflite}")
        tflite_bytes = model_tflite.read_bytes()
        guard = "MODEL_DATA_H"
        model_text = (
            f"#ifndef {guard}\n"
            f"#define {guard}\n\n"
            + c_array_bytes(tflite_bytes, args.array_name)
            + "\n#endif\n"
        )
        model_dst.write_text(model_text, encoding="utf-8")

    print("Arduino TinyML bundle ready")
    print(f"sketch_dir={out_sketch_dir}")
    print(f"model_header={model_dst}")
    print(f"runtime_header={runtime_header}")
    print(f"threshold={anomaly_threshold:.9f}")


if __name__ == "__main__":
    main()
