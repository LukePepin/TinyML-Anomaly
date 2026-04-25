# Arduino Nano TinyML Inference Sketch

This sketch performs true on-device inference using TensorFlow Lite Micro.

## Required generated files

Before compile/upload, generate and place these in this folder:

- `model_data.h`
- `runtime_config.h`

Use (recommended: generate `model_data.h` from fp32 `model_fp32.tflite`):

```powershell
python tinyml-anomaly/ml/anomaly_detection/scripts/generate_tinyml_arduino_bundle.py \
  --model-tflite tinyml-anomaly/ml/anomaly_detection/results/tinyml_raw32_week2/export/model_fp32.tflite \
  --scaling-json tinyml-anomaly/ml/anomaly_detection/results/tinyml_raw32_week2/model/scaling.json \
  --threshold-json tinyml-anomaly/ml/anomaly_detection/results/tinyml_raw32_week2/model/threshold.json \
  --out-sketch-dir tinyml-anomaly/arduino_nano_tinyml_inference \
  --window-size 32 \
  --axis-count 6
```

## Compile/upload

```powershell
.\scripts\Upload-TinyML-ToArduino.ps1 -Port COM9 -SketchPath tinyml-anomaly/arduino_nano_tinyml_inference
```

## Monitor

```powershell
& "C:\Program Files\Arduino CLI\arduino-cli.exe" monitor -p COM9 -c baudrate=115200
```

Expected serial output lines:

- `READY:TINYML_INFERENCE`
- `WARMING_UP`
- `ML_SCORE=<value> THRESH=<value> STATUS=NORMAL|ANOMALY_DETECTED`
