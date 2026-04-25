import os
import json
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.metrics import (
    confusion_matrix, 
    precision_recall_fscore_support, 
    roc_curve, 
    auc, 
    precision_recall_curve, 
    average_precision_score
)

def load_npz(path):
    """Load the dataset from an NPZ file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    print(f"Loading dataset from {path}...")
    data = np.load(path)
    return data["x_test"], data["y_test"]

def predict_non_tinyml(x_test, model_dir):
    """Evaluate the standard ML model (Autoencoder)."""
    print("Evaluating Non-TinyML Model (Autoencoder)...")
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    model_path = os.path.join(model_dir, "autoencoder.joblib")
    thresh_path = os.path.join(model_dir, "threshold.json")
    
    if not all(os.path.exists(p) for p in [scaler_path, model_path, thresh_path]):
        print(f"Warning: Missing artifacts in {model_dir}")
        return None, None, None

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    with open(thresh_path) as f:
        threshold = json.load(f)["threshold"]
        
    x_test_s = scaler.transform(x_test)
    y_pred = model.predict(x_test_s)
    err = np.mean((x_test_s - y_pred) ** 2, axis=1)
    
    labels = (err > threshold).astype(np.int32)
    return err, labels, threshold

def predict_tinyml(x_test, tinyml_dir):
    """Evaluate the TinyML quantized model."""
    print("Evaluating TinyML Model (Tiny Dense Autoencoder)...")
    model_path = os.path.join(tinyml_dir, "tiny_dense_autoencoder.keras")
    thresh_path = os.path.join(tinyml_dir, "threshold.json")
    scaling_path = os.path.join(tinyml_dir, "scaling.json")

    if not all(os.path.exists(p) for p in [model_path, thresh_path, scaling_path]):
        print(f"Warning: Missing artifacts in {tinyml_dir}")
        return None, None, None

    model = tf.keras.models.load_model(model_path)
    with open(thresh_path) as f:
        threshold = json.load(f)["threshold"]
    with open(scaling_path) as f:
        scale_data = json.load(f)
        mean = np.array(scale_data["mean"])
        std = np.array(scale_data["std"])
        
    x_test_s = (x_test - mean) / std
    y_pred = model.predict(x_test_s, verbose=0)
    err = np.mean((x_test_s - y_pred) ** 2, axis=1)
    
    labels = (err > threshold).astype(np.int32)
    return err, labels, threshold

def run_comparison(dataset_path, model_dir, tinyml_dir, out_dir):
    """Run the comparison between models and save results."""
    os.makedirs(out_dir, exist_ok=True)
    
    x_test, y_test = load_npz(dataset_path)
    
    res_nt = predict_non_tinyml(x_test, model_dir)
    res_t = predict_tinyml(x_test, tinyml_dir)
    
    if res_nt[0] is None or res_t[0] is None:
        print("Error: Could not complete comparison due to missing models.")
        return

    err_nt, labels_nt, thresh_nt = res_nt
    err_t, labels_t, thresh_t = res_t
    
    # 1. Classification Reports
    report = "=== MODEL COMPARISON REPORT ===\n\n"
    report += "NON-TINYML MODEL\n"
    report += f"Threshold: {thresh_nt:.6f}\n"
    p, r, f, s = precision_recall_fscore_support(y_test, labels_nt, labels=[0, 1])
    report += f"Normal:  P={p[0]:.4f}, R={r[0]:.4f}, F1={f[0]:.4f}, Support={s[0]}\n"
    report += f"Anomaly: P={p[1]:.4f}, R={r[1]:.4f}, F1={f[1]:.4f}, Support={s[1]}\n"
    
    report += "\nTINYML MODEL\n"
    report += f"Threshold: {thresh_t:.6f}\n"
    p, r, f, s = precision_recall_fscore_support(y_test, labels_t, labels=[0, 1])
    report += f"Normal:  P={p[0]:.4f}, R={r[0]:.4f}, F1={f[0]:.4f}, Support={s[0]}\n"
    report += f"Anomaly: P={p[1]:.4f}, R={r[1]:.4f}, F1={f[1]:.4f}, Support={s[1]}\n"
    
    report_path = os.path.join(out_dir, "comparison_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")
    print(report)
    
    # 2. Confusion Matrices
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(confusion_matrix(y_test, labels_nt), annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'], ax=axes[0])
    axes[0].set_title('Non-TinyML Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    sns.heatmap(confusion_matrix(y_test, labels_t), annot=True, fmt='d', cmap='Oranges', cbar=False,
                xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'], ax=axes[1])
    axes[1].set_title('TinyML Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrices.png"), dpi=300)
    
    # 3. Overlaid ROC & PR Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC
    fpr_nt, tpr_nt, _ = roc_curve(y_test, err_nt)
    auc_nt = auc(fpr_nt, tpr_nt)
    fpr_t, tpr_t, _ = roc_curve(y_test, err_t)
    auc_t = auc(fpr_t, tpr_t)
    
    axes[0].plot(fpr_nt, tpr_nt, color='blue', lw=2, label=f'Non-TinyML (AUC = {auc_nt:.2f})')
    axes[0].plot(fpr_t, tpr_t, color='orange', lw=2, label=f'TinyML (AUC = {auc_t:.2f})')
    axes[0].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    axes[0].set_title('ROC Curve Comparison')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend(loc="lower right")
    
    # PR
    precision_nt, recall_nt, _ = precision_recall_curve(y_test, err_nt)
    ap_nt = average_precision_score(y_test, err_nt)
    precision_t, recall_t, _ = precision_recall_curve(y_test, err_t)
    ap_t = average_precision_score(y_test, err_t)
    
    axes[1].plot(recall_nt, precision_nt, color='blue', lw=2, label=f'Non-TinyML (AP = {ap_nt:.2f})')
    axes[1].plot(recall_t, precision_t, color='orange', lw=2, label=f'TinyML (AP = {ap_t:.2f})')
    axes[1].set_title('PR Curve Comparison')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "performance_curves.png"), dpi=300)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare TinyML and Non-TinyML models.")
    parser.add_argument("--dataset", type=str, default="data/dataset.npz", help="Path to evaluation dataset (.npz)")
    parser.add_argument("--model-dir", type=str, default="models/standard", help="Directory containing standard model artifacts")
    parser.add_argument("--tinyml-dir", type=str, default="models/tinyml", help="Directory containing TinyML artifacts")
    parser.add_argument("--out-dir", type=str, default="results/comparison", help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    try:
        run_comparison(args.dataset, args.model_dir, args.tinyml_dir, args.out_dir)
    except Exception as e:
        print(f"Error during comparison: {e}")
