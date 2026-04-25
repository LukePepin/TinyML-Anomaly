// Nano 33 BLE anomaly-only serial status stream.
// Prints only NORMAL or ANOMALY_DETECTED at ~20 Hz.

#include <Arduino_LSM9DS1.h>
#include <math.h>

const unsigned long SAMPLE_INTERVAL_MS = 50;  // 20 Hz status updates
unsigned long lastSampleMs = 0;
const unsigned long WARMUP_MS = 3000;         // allow IMU to settle at startup
unsigned long bootMs = 0;

// Simple heuristic thresholds for anomaly indication.
// These are not ML inference thresholds.
// We trigger when acceleration magnitude departs from ~1g or when angular rate spikes.
const float ACCEL_MAG_LOW_THRESHOLD = 0.70f;     // g units
const float ACCEL_MAG_HIGH_THRESHOLD = 1.30f;    // g units
const float GYRO_MAG_ANOMALY_THRESHOLD = 12.0f;  // deg/s units

// Hysteresis to avoid rapid toggling.
const float ACCEL_MAG_LOW_CLEAR = 0.78f;
const float ACCEL_MAG_HIGH_CLEAR = 1.22f;
const float GYRO_MAG_CLEAR_THRESHOLD = 8.0f;

const int ANOMALY_ENTER_COUNT = 2;
const int ANOMALY_CLEAR_COUNT = 4;

bool anomalyState = false;
int anomalyHitCount = 0;
int normalHitCount = 0;

void setup() {
  Serial.begin(115200);
  bootMs = millis();

  unsigned long startWait = millis();
  while (!Serial && (millis() - startWait < 3000)) {
    ;
  }

  if (!IMU.begin()) {
    Serial.println("ERR:IMU_INIT");
    while (true) {
      ;
    }
  }

  Serial.println("READY:ANOMALY_STATUS");
}

void loop() {
  unsigned long now = millis();
  if (now - lastSampleMs < SAMPLE_INTERVAL_MS) {
    return;
  }
  lastSampleMs = now;

  if (!IMU.accelerationAvailable() || !IMU.gyroscopeAvailable()) {
    return;
  }

  float ax = 0.0f;
  float ay = 0.0f;
  float az = 0.0f;
  float gx = 0.0f;
  float gy = 0.0f;
  float gz = 0.0f;

  IMU.readAcceleration(ax, ay, az);
  IMU.readGyroscope(gx, gy, gz);

  float accelMag = sqrtf(ax * ax + ay * ay + az * az);
  float gyroMag = sqrtf(gx * gx + gy * gy + gz * gz);

  // Warmup avoids immediate false latch during startup transients.
  if ((millis() - bootMs) < WARMUP_MS) {
    Serial.println("NORMAL");
    return;
  }

  bool anomalyCandidate =
      (accelMag < ACCEL_MAG_LOW_THRESHOLD) ||
      (accelMag > ACCEL_MAG_HIGH_THRESHOLD) ||
      (gyroMag > GYRO_MAG_ANOMALY_THRESHOLD);

  bool normalCandidate =
      (accelMag >= ACCEL_MAG_LOW_CLEAR) &&
      (accelMag <= ACCEL_MAG_HIGH_CLEAR) &&
      (gyroMag <= GYRO_MAG_CLEAR_THRESHOLD);

  if (anomalyCandidate) {
    anomalyHitCount++;
    normalHitCount = 0;
  } else if (normalCandidate) {
    normalHitCount++;
    anomalyHitCount = 0;
  } else {
    // In-between hysteresis zone: keep current state and decay counters.
    if (anomalyHitCount > 0) {
      anomalyHitCount--;
    }
    if (normalHitCount > 0) {
      normalHitCount--;
    }
  }

  if (!anomalyState && anomalyHitCount >= ANOMALY_ENTER_COUNT) {
    anomalyState = true;
    anomalyHitCount = 0;
  }

  if (anomalyState && normalHitCount >= ANOMALY_CLEAR_COUNT) {
    anomalyState = false;
    normalHitCount = 0;
  }

  if (anomalyState) {
    Serial.println("ANOMALY_DETECTED");
  } else {
    Serial.println("NORMAL");
  }
}
