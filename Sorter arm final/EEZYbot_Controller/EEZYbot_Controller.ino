/*
 * EEZYbotARM MK2 Controller
 * =========================
 * 
 * Controls 4 servos for the robotic arm.
 * Receives commands via Serial from Python script.
 * 
 * Wiring:
 * - Base (rotation): Pin 9
 * - Shoulder: Pin 10
 * - Elbow: Pin 11
 * - Gripper: Pin 6
 * 
 * Commands:
 * - MOVE,base,shoulder,elbow,gripper,speed
 * - HOME
 * - STATUS
 */

#include <Servo.h>

// ==================== CONFIGURATION ====================

// Servo pins
#define PIN_BASE 9
#define PIN_SHOULDER 10
#define PIN_ELBOW 11
#define PIN_GRIPPER 6

// Servo objects
Servo servoBase;
Servo servoShoulder;
Servo servoElbow;
Servo servoGripper;

// Current positions
int currentBase = 90;
int currentShoulder = 90;
int currentElbow = 90;
int currentGripper = 20;

// Home position
#define HOME_BASE 90
#define HOME_SHOULDER 90
#define HOME_ELBOW 90
#define HOME_GRIPPER 20

// Movement speed (ms between steps)
#define DEFAULT_SPEED 50

// ==================== SETUP ====================

void setup() {
  Serial.begin(115200);
  
  // Attach servos
  servoBase.attach(PIN_BASE);
  servoShoulder.attach(PIN_SHOULDER);
  servoElbow.attach(PIN_ELBOW);
  servoGripper.attach(PIN_GRIPPER);
  
  // Move to home position
  moveToHome();
  
  Serial.println("EEZYbotARM MK2 Ready");
}

// ==================== MOVEMENT FUNCTIONS ====================

void smoothMove(int targetBase, int targetShoulder, int targetElbow, int targetGripper, int speedDelay) {
  /*
   * Smooth movement from current to target positions.
   * All servos move simultaneously with coordinated timing.
   */
  
  // Calculate deltas
  int deltaBase = targetBase - currentBase;
  int deltaShoulder = targetShoulder - currentShoulder;
  int deltaElbow = targetElbow - currentElbow;
  int deltaGripper = targetGripper - currentGripper;
  
  // Find maximum delta (determines number of steps)
  int maxDelta = max(abs(deltaBase), max(abs(deltaShoulder), max(abs(deltaElbow), abs(deltaGripper))));
  
  if (maxDelta == 0) {
    return;  // Already at target
  }
  
  // Move in small steps
  for (int step = 1; step <= maxDelta; step++) {
    // Calculate intermediate positions
    int newBase = currentBase + (deltaBase * step) / maxDelta;
    int newShoulder = currentShoulder + (deltaShoulder * step) / maxDelta;
    int newElbow = currentElbow + (deltaElbow * step) / maxDelta;
    int newGripper = currentGripper + (deltaGripper * step) / maxDelta;
    
    // Write to servos
    servoBase.write(newBase);
    servoShoulder.write(newShoulder);
    servoElbow.write(newElbow);
    servoGripper.write(newGripper);
    
    // Small delay for smooth movement
    delay(speedDelay);
  }
  
  // Update current positions
  currentBase = targetBase;
  currentShoulder = targetShoulder;
  currentElbow = targetElbow;
  currentGripper = targetGripper;
}

void moveToHome() {
  /*
   * Move to home (safe) position.
   */
  smoothMove(HOME_BASE, HOME_SHOULDER, HOME_ELBOW, HOME_GRIPPER, DEFAULT_SPEED);
  Serial.println("Moved to HOME");
}

// ==================== COMMAND PARSING ====================

void processCommand(String cmd) {
  /*
   * Parse and execute command from serial.
   * 
   * Commands:
   * - MOVE,base,shoulder,elbow,gripper,speed
   * - HOME
   * - STATUS
   */
  
  cmd.trim();
  
  if (cmd.startsWith("MOVE,")) {
    // Parse MOVE command
    int commaPos[5];
    int idx = 0;
    
    // Find comma positions
    for (int i = 0; i < cmd.length() && idx < 5; i++) {
      if (cmd.charAt(i) == ',') {
        commaPos[idx++] = i;
      }
    }
    
    if (idx < 5) {
      Serial.println("ERROR: Invalid MOVE command");
      return;
    }
    
    // Extract values
    int base = cmd.substring(commaPos[0] + 1, commaPos[1]).toInt();
    int shoulder = cmd.substring(commaPos[1] + 1, commaPos[2]).toInt();
    int elbow = cmd.substring(commaPos[2] + 1, commaPos[3]).toInt();
    int gripper = cmd.substring(commaPos[3] + 1, commaPos[4]).toInt();
    int speed = cmd.substring(commaPos[4] + 1).toInt();
    
    // Clamp values
    base = constrain(base, 0, 180);
    shoulder = constrain(shoulder, 20, 160);
    elbow = constrain(elbow, 0, 180);
    gripper = constrain(gripper, 20, 90);
    speed = constrain(speed, 1, 100);
    
    // Execute move
    smoothMove(base, shoulder, elbow, gripper, speed);
    
    Serial.print("MOVED: ");
    Serial.print(base);
    Serial.print(",");
    Serial.print(shoulder);
    Serial.print(",");
    Serial.print(elbow);
    Serial.print(",");
    Serial.println(gripper);
  }
  else if (cmd == "HOME") {
    moveToHome();
  }
  else if (cmd == "STATUS") {
    Serial.print("POSITION: ");
    Serial.print(currentBase);
    Serial.print(",");
    Serial.print(currentShoulder);
    Serial.print(",");
    Serial.print(currentElbow);
    Serial.print(",");
    Serial.println(currentGripper);
  }
  else {
    Serial.println("ERROR: Unknown command");
  }
}

// ==================== MAIN LOOP ====================

void loop() {
  // Check for incoming serial commands
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    processCommand(cmd);
  }
  
  // Small delay to prevent serial overflow
  delay(10);
}
