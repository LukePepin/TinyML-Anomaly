import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
import json

# Paths
BASE_DIR = r"C:\Users\lukep\Documents\MVS\backend\ml\anomaly_detection\results\week2\window_sweep_results\window_configs\ws512_st16_thr0p25"
TRAIN_CSV = os.path.join(BASE_DIR, "windowed_train.csv")
TEST_CSV = os.path.join(BASE_DIR, "windowed_test.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "dummy_baseline_results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(filepath):
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    # the label is in 'window_label'
    y = df['window_label'].values
    # We don't actually need full features for a dummy classifier, just shapes 
    # but we extract it realistically for sklearn compatibility
    X = df.iloc[:, 7:].values
    return X, y

def plot_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal (0)', 'Anomaly (1)'],
                yticklabels=['Normal (0)', 'Anomaly (1)'])
    plt.title('Dummy Classifier (Stratified) - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_probs, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_pr_curve(y_true, y_probs, output_path):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AP = {ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    X_train, y_train = load_data(TRAIN_CSV)
    X_test, y_test = load_data(TEST_CSV)
    
    # "Include class split" means using a Stratified strategy based on the class distribution
    # of the training set.
    print("\nTraining Dummy Classifier (strategy='stratified')...")
    clf = DummyClassifier(strategy='stratified', random_state=42)
    clf.fit(X_train, y_train)
    
    print("\nEvaluating on Test dataset...")
    y_pred = clf.predict(X_test)
    y_probs = clf.predict_proba(X_test)[:, 1]
    
    # Explanations
    explanation = (
        "=== Dummy Classifier Explanation ===\n"
        "A Dummy Classifier is a simple baseline model used to gauge the lower-bound performance "
        "of a real machine learning model. In this case, we are using the 'stratified' strategy "
        "(based on include class split).\n\n"
        "What does it do?\n"
        "1. It looks at the percentage of Normal vs. Anomaly samples in your TRAINING data.\n"
        "2. When making a prediction on the test data, it simply guesses randomly, but biased by "
        "that training percentage.\n"
        "3. For example, if your training data has 50% anomalies and 50% normal windows, it flips a fair coin. "
        "If it was 80/20, it uses a weighted coin. It completely ignores all inputs (accelerometer/gyro).\n\n"
        "If your actual autoencoder model cannot significantly beat these metrics (like Precision, Recall, F1), "
        "then the neural network isn't actually learning meaningful patterns from the Arduino sensor data.\n"
        "====================================\n"
    )
    
    # Save text explanation
    with open(os.path.join(OUTPUT_DIR, "model_explanation.txt"), "w") as f:
        f.write(explanation)
        
    print(explanation)
    
    print("=== Classification Report ===")
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly'])
    print(report)
    
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
    
    # Generate Plots
    print("Generating visualizations...")
    plot_confusion_matrix(y_test, y_pred, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plot_roc_curve(y_test, y_probs, os.path.join(OUTPUT_DIR, "roc_curve.png"))
    plot_pr_curve(y_test, y_probs, os.path.join(OUTPUT_DIR, "precision_recall_curve.png"))
    
    print(f"\nAll results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
