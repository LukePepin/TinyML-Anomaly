#!/usr/bin/env python3
"""Export trained Keras model to int8 TFLite and C array header.

Inputs:
- tiny_dense_autoencoder.keras
- dataset.npz (for representative data)

Outputs:
- model_fp32.tflite
- model_int8.tflite
- model_data.h
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export tiny dense model to TFLite + C header")
    parser.add_argument("--model", required=True, help="Path to trained .keras model")
    parser.add_argument("--dataset", required=True, help="Path to dataset.npz for representative data")
    parser.add_argument(
        "--scaling-json",
        required=True,
        help="Path to scaling.json from training step",
    )
    parser.add_argument("--out-dir", required=True, help="Output folder for tflite/header")
    parser.add_argument(
        "--header-name",
        default="model_data.h",
        help="Header filename for C array output",
    )
    parser.add_argument(
        "--array-name",
        default="g_tinyml_model_data",
        help="C array symbol name",
    )
    parser.add_argument(
        "--rep-samples",
        type=int,
        default=200,
        help="Representative calibration samples from train split",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    model_path = Path(args.model)
    dataset_path = Path(args.dataset)
    scaling_path = Path(args.scaling_json)
    out_dir = Path(args.out_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not scaling_path.exists():
        raise FileNotFoundError(f"Scaling JSON not found: {scaling_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(model_path)
    ds = np.load(dataset_path)
    x_train = ds["x_train"].astype(np.float32)
    scaling = json.loads(scaling_path.read_text(encoding="utf-8"))
    mean = np.asarray(scaling["mean"], dtype=np.float32)
    std = np.asarray(scaling["std"], dtype=np.float32)

    if x_train.shape[1] != mean.shape[0] or mean.shape[0] != std.shape[0]:
        raise ValueError("Scaling vector size does not match model input feature size")

    # Representative data must match model runtime input distribution.
    x_train_scaled = (x_train - mean) / np.where(std < 1e-6, 1.0, std)
    rng = np.random.default_rng(args.seed)
    sample_count = min(len(x_train_scaled), max(1, args.rep_samples))
    rep_idx = rng.choice(len(x_train_scaled), size=sample_count, replace=False)
    x_rep = x_train_scaled[rep_idx]

    def representative_dataset():
        for i in range(sample_count):
            yield [x_rep[i : i + 1]]

    fp32_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32 = fp32_converter.convert()
    fp32_path = out_dir / "model_fp32.tflite"
    fp32_path.write_bytes(tflite_fp32)

    int8_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    int8_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    int8_converter.representative_dataset = representative_dataset
    int8_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    int8_converter.inference_input_type = tf.int8
    int8_converter.inference_output_type = tf.int8
    tflite_int8 = int8_converter.convert()

    int8_path = out_dir / "model_int8.tflite"
    int8_path.write_bytes(tflite_int8)

    header_path = out_dir / args.header_name
    header_guard = args.header_name.upper().replace(".", "_").replace("-", "_")
    header_text = (
        f"#ifndef {header_guard}\n"
        f"#define {header_guard}\n\n"
        + c_array_bytes(tflite_int8, args.array_name)
        + "\n#endif\n"
    )
    header_path.write_text(header_text, encoding="utf-8")

    print("TinyML artifacts exported")
    print(f"fp32={fp32_path}")
    print(f"int8={int8_path}")
    print(f"header={header_path}")
    print(f"int8_bytes={len(tflite_int8)}")


if __name__ == "__main__":
    main()
