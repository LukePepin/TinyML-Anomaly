// True on-device TinyML inference for Nano 33 BLE.
// Uses ArduTFLite runtime + model_data.h + runtime_config.h generated from training artifacts.

#include <Arduino_LSM9DS1.h>
#include <ArduTFLite.h>

#include "model_data.h"
#include "runtime_config.h"

namespace {

constexpr unsigned long kSampleIntervalMs = 50;  // 20 Hz
constexpr size_t kTensorArenaSize = 110 * 1024;

alignas(16) byte tensor_arena[kTensorArenaSize];

float window_buffer[kWindowSize][kAxisCount];
int write_index = 0;
int sample_count = 0;

float scaled_input[kInputDim];

unsigned long last_sample_ms = 0;

bool read_imu_sample(float* axes6) {
  if (!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable()) {
    return false;
  }

  float ax = 0.0f;
  float ay = 0.0f;
  float az = 0.0f;
  float gx = 0.0f;
  float gy = 0.0f;
  float gz = 0.0f;

  IMU.readAcceleration(ax, ay, az);
  IMU.readGyroscope(gx, gy, gz);

  axes6[0] = ax;
  axes6[1] = ay;
  axes6[2] = az;
  axes6[3] = gx;
  axes6[4] = gy;
  axes6[5] = gz;

  return true;
}

void push_sample(const float* axes6) {
  for (int j = 0; j < kAxisCount; ++j) {
    window_buffer[write_index][j] = axes6[j];
  }
  write_index = (write_index + 1) % kWindowSize;
  if (sample_count < kWindowSize) {
    sample_count++;
  }
}

// Build flattened feature vector oldest->newest with training-time scaling.
void build_scaled_feature_vector() {
  int feat = 0;
  for (int i = 0; i < kWindowSize; ++i) {
    int src = (write_index + i) % kWindowSize;
    for (int j = 0; j < kAxisCount; ++j) {
      float raw = window_buffer[src][j];
      float stdv = kInputStd[feat];
      if (stdv < 1e-6f) {
        stdv = 1.0f;
      }
      float z = (raw - kInputMean[feat]) / stdv;
      scaled_input[feat] = z;
      feat++;
    }
  }
}

float compute_reconstruction_mse() {
  float mse = 0.0f;
  for (int i = 0; i < kInputDim; ++i) {
    float recon = modelGetOutput(i);
    float d = scaled_input[i] - recon;
    mse += d * d;
  }
  return mse / static_cast<float>(kInputDim);
}

void print_status(float mse) {
  Serial.print("ML_SCORE=");
  Serial.print(mse, 6);
  Serial.print(" THRESH=");
  Serial.print(kAnomalyThreshold, 6);
  Serial.print(" STATUS=");
  if (mse > kAnomalyThreshold) {
    Serial.println("ANOMALY_DETECTED");
  } else {
    Serial.println("NORMAL");
  }
}

}  // namespace

void setup() {
  Serial.begin(115200);

  unsigned long start_wait = millis();
  while (!Serial && (millis() - start_wait < 3000)) {
    ;
  }

  if (!IMU.begin()) {
    Serial.println("ERR:IMU_INIT");
    while (true) {
      ;
    }
  }

  if (!modelInit(g_tinyml_model_data, tensor_arena, kTensorArenaSize)) {
    Serial.println("ERR:MODEL_INIT");
    while (true) {
      ;
    }
  }

  Serial.println("READY:TINYML_INFERENCE");
}

void loop() {
  unsigned long now = millis();
  if (now - last_sample_ms < kSampleIntervalMs) {
    return;
  }
  last_sample_ms = now;

  float axes6[kAxisCount];
  if (!read_imu_sample(axes6)) {
    return;
  }

  push_sample(axes6);
  if (sample_count < kWindowSize) {
    Serial.println("WARMING_UP");
    return;
  }

  build_scaled_feature_vector();
  for (int i = 0; i < kInputDim; ++i) {
    if (!modelSetInput(scaled_input[i], i)) {
      Serial.println("ERR:SET_INPUT");
      return;
    }
  }

  if (!modelRunInference()) {
    Serial.println("ERR:INVOKE");
    return;
  }

  float mse = compute_reconstruction_mse();
  print_status(mse);
}
