#include <Wire.h>

#define SDA_PIN 15
#define SCL_PIN 14

#define PCA9685_ADDR 0x40

// PCA9685 registers
#define MODE1       0x00
#define PRESCALE    0xFE
#define LED0_ON_L   0x06

static void write8(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(PCA9685_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

static uint8_t read8(uint8_t reg) {
  Wire.beginTransmission(PCA9685_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(PCA9685_ADDR, (uint8_t)1);
  return Wire.available() ? Wire.read() : 0xFF;
}

static void setPWM(uint8_t channel, uint16_t on, uint16_t off) {
  uint8_t reg = LED0_ON_L + 4 * channel;
  Wire.beginTransmission(PCA9685_ADDR);
  Wire.write(reg);
  Wire.write(on & 0xFF);
  Wire.write((on >> 8) & 0xFF);
  Wire.write(off & 0xFF);
  Wire.write((off >> 8) & 0xFF);
  Wire.endTransmission();
}

// 50 Hz servo setup
static void setPWMFreq(float freq_hz) {
  // prescale = round(25MHz / (4096*freq)) - 1
  float prescaleval = 25000000.0;
  prescaleval /= 4096.0;
  prescaleval /= freq_hz;
  prescaleval -= 1.0;
  uint8_t prescale = (uint8_t)(prescaleval + 0.5);

  uint8_t oldmode = read8(MODE1);
  uint8_t sleepmode = (oldmode & 0x7F) | 0x10; // sleep
  write8(MODE1, sleepmode);
  write8(PRESCALE, prescale);
  write8(MODE1, oldmode);
  delay(5);
  write8(MODE1, oldmode | 0xA1); // auto-increment + restart
}

void setup() {
  Serial.begin(115200);
  delay(200);

  Wire.begin(SDA_PIN, SCL_PIN);
  Serial.println("Init PCA9685...");

  // MODE1 reset
  write8(MODE1, 0x00);
  delay(10);

  setPWMFreq(50.0); // standard servos
  Serial.println("PCA9685 ready. Sweeping CH0...");
}

void loop() {
  // Typical servo pulse range ~500–2500 us.
  // Convert microseconds to PCA9685 ticks at 50Hz:
  // ticks = us * 4096 / 20000
  auto usToTicks = [](int us) -> uint16_t {
    return (uint16_t)((us * 4096L) / 20000L);
  };

  // Sweep 700us to 2300us (safe-ish for most servos)
  for (int us = 700; us <= 2300; us += 20) {
    setPWM(0, 0, usToTicks(us));
    delay(15);
  }
  for (int us = 2300; us >= 700; us -= 20) {
    setPWM(0, 0, usToTicks(us));
    delay(15);
  }
}
