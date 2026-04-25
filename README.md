# TinyML Anomaly Detection for Robotic Arms

Autonomous anomaly detection for Industry 4.0 robotic systems using TinyML autoencoders. This project enables a Niryo Ned2 robotic arm to verify its physical motion against commanded motion without cloud connectivity.

## 🚀 Overview

Modern industrial robots often rely on cloud authorization. If connectivity is lost or jammed, they enter a safety stop. This project moves that "authorization" onto the device by using a TinyML autoencoder to detect adversarial motion injections in real-time.

### Key Success Criteria
- **Recall**: ≥ 0.95 (Catch real attacks)
- **Precision**: ≥ 0.85 (Minimize false stops)
- **F2-Score**: ≥ 0.90 (Primary metric)

## 📁 Project Structure

```text
tinyml-anomaly/
├── data/               # Raw and processed datasets (CSV, NPZ)
├── docs/               # Reports, proposals, and media
├── firmware/           # Arduino Nano 33 BLE Sense code
├── ml/                 # ML pipelines and experiment tracking
├── models/             # Exported model artifacts (.keras, .joblib)
├── notebooks/          # Experimentation and EDA
├── results/            # Analysis outputs (plots, metrics)
└── src/                # Core Python source code
    ├── data_collection/# Sensor streaming and logging
    └── analysis/       # Analytic tools and auditing
```

## 🛠️ Workflow

1.  **Data Collection**: Stream 6-axis IMU data from the Niryo wrist via `src/data_collection/stream_niryo.py`.
2.  **Preprocessing**: Use `notebooks/` or `src/analysis/` to window and normalize the data.
3.  **Training**: Train a symmetric undercomplete autoencoder (PyTorch/TensorFlow).
4.  **Export**: Quantize and export the model to C++ headers using `src/ml/export_artifacts.py`.
5.  **Deployment**: Flash the `firmware/` to the Arduino Nano 33 BLE Sense.

## 📊 Analytic Tools

The project includes several tools for auditing and comparing models:
- **Data Auditor**: `src/analysis/data_audit.py` - Verifies schema and data integrity.
- **Model Comparator**: `src/analysis/compare_models.py` - Benchmarks TinyML vs. Full models.
- **Sweep Analyzer**: Finds the best window/stride configurations.

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tinyml-anomaly.git
cd tinyml-anomaly

# Install dependencies
pip install -r requirements.txt
```

## 👥 Contributors
- Luke Pepin
- Mark Litton
- Michael Reinhart
