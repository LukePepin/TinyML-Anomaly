#!/usr/bin/env python3
"""Train a small dense autoencoder on raw window vectors and export artifacts.

Inputs:
- dataset.npz from build_raw_window_dataset.py

Outputs:
- tiny_dense_autoencoder.keras
- threshold.json
- scaling.json
- eval_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tiny dense autoencoder on raw windows")
    parser.add_argument("--dataset", required=True, help="Path to dataset.npz")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=99.0,
        help="Percentile over validation normal error",
    )
    parser.add_argument("--dense-1", type=int, default=64, help="First hidden layer size")
    parser.add_argument("--latent", type=int, default=16, help="Latent bottleneck size")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def mse_per_row(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean((y_true - y_pred) ** 2, axis=1)


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate_normal": fpr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def build_model(input_dim: int, dense_1: int, latent: int, learning_rate: float) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(input_dim,), name="raw_window")
    x = tf.keras.layers.Dense(dense_1, activation="relu", name="enc_dense_1")(inp)
    x = tf.keras.layers.Dense(latent, activation="relu", name="latent")(x)
    x = tf.keras.layers.Dense(dense_1, activation="relu", name="dec_dense_1")(x)
    out = tf.keras.layers.Dense(input_dim, activation=None, name="recon")(x)
    model = tf.keras.Model(inp, out, name="tiny_dense_autoencoder")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = np.load(dataset_path)
    x_train = ds["x_train"].astype(np.float32)
    x_val = ds["x_val"].astype(np.float32)
    x_test = ds["x_test"].astype(np.float32)

    y_train = ds["y_train"].astype(np.int32) if "y_train" in ds.files else None
    y_val = ds["y_val"].astype(np.int32) if "y_val" in ds.files else None
    y_test = ds["y_test"].astype(np.int32) if "y_test" in ds.files else None

    input_dim = int(x_train.shape[1])

    if y_train is not None:
        x_train_normal = x_train[y_train == 0]
        if len(x_train_normal) == 0:
            raise ValueError("No normal windows in train split for autoencoder fitting")
    else:
        x_train_normal = x_train

    mean = x_train_normal.mean(axis=0)
    std = x_train_normal.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    x_train_normal_s = (x_train_normal - mean) / std
    x_val_s = (x_val - mean) / std
    x_test_s = (x_test - mean) / std

    model = build_model(
        input_dim=input_dim,
        dense_1=args.dense_1,
        latent=args.latent,
        learning_rate=args.learning_rate,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            mode="min",
            restore_best_weights=True,
        )
    ]

    model.fit(
        x_train_normal_s,
        x_train_normal_s,
        validation_data=(x_val_s, x_val_s),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=callbacks,
    )

    val_pred = model.predict(x_val_s, verbose=0)
    test_pred = model.predict(x_test_s, verbose=0)
    val_err = mse_per_row(x_val_s, val_pred)
    test_err = mse_per_row(x_test_s, test_pred)

    if y_val is not None and np.sum(y_val == 0) > 0:
        val_norm_err = val_err[y_val == 0]
    else:
        val_norm_err = val_err

    threshold = float(np.percentile(val_norm_err, args.threshold_percentile))

    val_pred_labels = (val_err > threshold).astype(np.int32)
    test_pred_labels = (test_err > threshold).astype(np.int32)

    threshold_payload = {
        "threshold": threshold,
        "threshold_percentile": float(args.threshold_percentile),
        "val_normal_error_mean": float(np.mean(val_norm_err)),
        "val_normal_error_std": float(np.std(val_norm_err)),
    }
    (out_dir / "threshold.json").write_text(
        json.dumps(threshold_payload, indent=2),
        encoding="utf-8",
    )

    scaling_payload = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "input_dim": input_dim,
    }
    (out_dir / "scaling.json").write_text(
        json.dumps(scaling_payload),
        encoding="utf-8",
    )

    model_path = out_dir / "tiny_dense_autoencoder.keras"
    model.save(model_path)

    report: dict[str, object] = {
        "input_dim": input_dim,
        "train_rows": int(len(x_train)),
        "val_rows": int(len(x_val)),
        "test_rows": int(len(x_test)),
        "model": {
            "type": "tiny_dense_autoencoder",
            "dense_1": int(args.dense_1),
            "latent": int(args.latent),
            "learning_rate": float(args.learning_rate),
        },
        "threshold": threshold_payload,
    }
    if y_val is not None:
        report["validation_metrics"] = precision_recall_f1(y_val, val_pred_labels)
    if y_test is not None:
        report["test_metrics"] = precision_recall_f1(y_test, test_pred_labels)

    (out_dir / "eval_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Tiny dense training complete")
    print(f"out_dir={out_dir}")
    print(f"input_dim={input_dim}")
    print(f"threshold={threshold:.8f}")


if __name__ == "__main__":
    main()
