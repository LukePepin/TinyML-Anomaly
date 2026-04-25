// 6-axis IMU telemetry stream for Arduino Nano 33 BLE.
// Output format per line: ax,ay,az,gx,gy,gz
// Timestamp is intentionally added by the host CSV logger.

#include <Arduino_LSM9DS1.h>

const unsigned long SAMPLE_INTERVAL_MS = 20;  // 50 Hz
unsigned long lastSampleMs = 0;

void setup() {
  Serial.begin(115200);

  // Avoid blocking forever when no serial monitor is attached.
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

  // One-time readiness marker; host logger ignores non-CSV lines.
  Serial.println("READY:6AXIS");
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

  // Software E-STOP over Serial threshold
  float magnitude = sqrt(ax*ax + ay*ay + az*az);
  if (magnitude > 4.0f) {
    Serial.println("ESTOP:COLLISION");
  }

  Serial.print(ax, 6);
  Serial.print(',');
  Serial.print(ay, 6);
  Serial.print(',');
  Serial.print(az, 6);
  Serial.print(',');
  Serial.print(gx, 6);
  Serial.print(',');
  Serial.print(gy, 6);
  Serial.print(',');
  Serial.println(gz, 6);
}
