// Phase 2 IMU telemetry stream for Arduino Nano 33 BLE.
// Output format per line: ax,ay,az,gx,gy,gz,mx,my,mz

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
}

void loop() {
  unsigned long now = millis();
  if (now - lastSampleMs < SAMPLE_INTERVAL_MS) {
    return;
  }
  lastSampleMs = now;

  float ax = 0.0f;
  float ay = 0.0f;
  float az = 0.0f;
  float gx = 0.0f;
  float gy = 0.0f;
  float gz = 0.0f;
  float mx = 0.0f;
  float my = 0.0f;
  float mz = 0.0f;

  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(ax, ay, az);
  }
  if (IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(gx, gy, gz);
  }
  if (IMU.magneticFieldAvailable()) {
    IMU.readMagneticField(mx, my, mz);
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
  Serial.print(gz, 6);
  Serial.print(',');
  Serial.print(mx, 6);
  Serial.print(',');
  Serial.print(my, 6);
  Serial.print(',');
  Serial.println(mz, 6);
}