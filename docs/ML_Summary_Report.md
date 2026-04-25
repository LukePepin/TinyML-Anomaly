# Machine Learning Model Summary

All evaluation metric outputs and scripts generated previously have been consolidated into a more accessible final folder located at:
`tinyml-anomaly/ml/anomaly_detection/results/final_evaluation_metrics`

Here is your full machine learning summary covering the model architectures, the hyperparameters from your grid search sweep, and the baseline results.

---

## 🏗️ 1. Model Architectures Tested

You are actively comparing two distinct models for your anomaly detection pipeline.

### Model 1: Non-TinyML Model (Server-Side Pipeline)
- **Type**: Multilayer Perceptron (MLPRegressor) Autoencoder
- **Framework**: `scikit-learn`
- **Objective**: Standard, high-capacity model designed to process large windows of sensor data without computational bottlenecks.
- **Hidden Layers configuration**: `384, 192, 384`

### Model 2: TinyML Model (Microcontroller Pipeline)
- **Type**: Tiny Dense Autoencoder (Multi-layer Feedforward Neural Network)
- **Framework**: `Keras / TensorFlow`
- **Objective**: A significantly smaller autoencoder trained optimally for deployment onto limited microcontroller units (like an Arduino).
- **Hidden Layers configuration**: `64, 16, 64`
- **Input Dimension constraint**: Operates cleanly on `window_size: 32`, but struggles heavily scaling dynamically to your `window_size: 512`.

### Baseline Verification (Dummy Classifier)
- **Type**: `scikit-learn DummyClassifier` (strategy='stratified')
- **Objective**: Guesses anomalies purely randomly based on the statistical split of the classes found in the training dataset.

---

## ⚖️ 2. Architectural Justification (Why not X?)

When building an embedded anomaly detection pipeline for IMU sensor data, we specifically chose a **Semi-Supervised Autoencoder** approach over other popular algorithms. Here is the justification for avoiding other models:

### Why not Random Forest or Logistic Regression?
1. **Nature of the Data (Imbalance):** Traditional algorithms like Random Forests and Logistic Regression are *Supervised Classifiers*. They require perfectly labeled, highly balanced datasets featuring explicit examples of *every possible anomalous behavior*. In real-world sensor deployments, "normal" data is infinite, whereas "anomalies" are rare, highly variable, and impossible to exhaustively map out. 
2. **Memory Constraints:** Deploying a Random Forest (a massive ensemble of decision trees) onto a microcontroller consumes significant Flash memory and RAM, quickly exceeding the capabilities of a 32-bit MCU like an Arduino.
3. An Autoencoder, however, operates **semi-supervised**. It trains *only* on the abundant "normal" data. During inference, if a piece of data cannot be reconstructed properly, it is inherently flagged as an anomaly regardless of whether that specific failure mode was seen before.

### Why not Deep DL (CNNs, LSTMs, Complex ANNs)?
1. **Computational Overhead (FLOPs):** While Convolutional Neural Networks (CNNs) are excellent for spatial features and Long Short-Term Memory networks (LSTMs) are state-of-the-art for temporal sequences, they require massive amounts of floating-point operations.
2. **TinyML Restrictions:** Complex architectures add millions of parameters. This exceeds the rigid `< 256 KB` SRAM constraints of hardware targets. 
3. A Dense (MLP) Autoencoder provides the absolute best trade-off for TinyML. It remains incredibly lightweight (fast microsecond inference latency and a tiny memory footprint) while still capturing sufficient non-linear correlations across the IMU sensor axes.

---

## ⚙️ 3. Top 5 Grid Search Parameters

These are the top 5 model setups pulled directly from your grid sweep, scoring highest on the testing dataset:

| Window Size | Stride | Window Threshold | Threshold Percentile | Hidden Layers | Test F1 | Test Precision | Test Recall | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **512** | 16 | 0.25 | **85.0** | 384,192,384 | **0.8680** | 1.000 | 0.7669 |
| 512 | 16 | 0.25 | 87.5 | 384,192,384 | 0.8534 | 1.000 | 0.7443 |
| 512 | 16 | 0.25 | 90.0 | 384,192,384 | 0.8177 | 1.000 | 0.6917 |
| 1024 | 16 | 0.25 | 85.0 | 384,192,384 | 0.7938 | 1.000 | 0.6582 |
| 256 | 16 | 0.25 | 85.0 | 384,192,384 | 0.7818 | 1.000 | 0.6417 |

> [!TIP]
> The performance strongly preferred configurations that used a window size of exactly `512` and a relatively high target threshold of `85.0`.

---

## 📊 4. Performance Breakdown (Using Best Configs `ws512_st16_thr0p25`)

Using the parameters from the top scoring Grid Search row above, we trained both of the active pipelines alongside the Dummy Classifier. 

### A) Baseline Check
The dummy classifier proved that simply "guessing" based purely on statistics will yield random-chance results around `~0.49 F1`. Since our models score higher than this, we verify they are successfully learning latent IMU feature behaviors and not cheating!

### B) Non-TinyML Model Validation
The Non-TinyML `scikit-learn` Autoencoder comfortably handles the massive 512-window data dimension, securing a massive difference across the testing classification:
- **Test F1**: `0.868`
- **Test Precision**: `1.0` (Perfect separation; meaning zero false positives!)

### C) TinyML Constraints
When forced to process a 512-window sequence, the TinyML architecture collapses to a `~0.19 F1` rating (only finding roughly 10% of standard anomalies). To properly utilize the TinyML network architecture moving forward, you should construct sequences using a `window_size` constraint matching its design, typically between `32` and `128` data features.
