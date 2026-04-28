/*
 * project taiga - turret firmware
 *
 * forked from the stock crunchlabs IRTurret sketch (Copyright (c) 2025 Crunchlabs LLC, MIT)
 * adds a line-based serial command protocol so the cv server on the host PC
 * can drive the turret while the original IR remote still works for manual
 * override.
 *
 * serial protocol (9600 baud, lines terminated by \n, max 63 chars):
 *
 *   YAW <L|R> [ms]       rotate yaw at full speed for `ms` milliseconds
 *                        (default = yawPrecision). direction is L (ccw) or R (cw).
 *   PITCH <angle>        absolute pitch position. clamped to [pitchMin, pitchMax].
 *   PITCH_REL <delta>    relative pitch change. positive = up.
 *   FIRE [count]         fire `count` darts (default 1, clamped to [1, 2]).
 *                        refuses if DISARMED.
 *   HOME                 return servos to the startup pose and reset yaw tracking.
 *   STATUS               reply with "STATE pitch=X darts=Y armed=Z yaw_ms=W".
 *   ARM / DISARM         toggle firmware-level firing lockout.
 *   RELOAD               tell firmware the magazine was refilled (resets dart count).
 *
 * responses:
 *   OK <echo>            command accepted and (for motor moves) completed
 *   ERR <reason>         command rejected
 *   STATE ...            status report (also printed on boot)
 *   LOG ...              free-form info, ignored by the python client
 *
 * the IR library's built-in prints are noisy but always start with something
 * other than "OK "/"ERR "/"STATE " so the host can safely ignore them.
 */

#include <Arduino.h>
#include <Servo.h>
#include <IRremote.hpp>

// --- IR command codes (unchanged from stock) ---
#define IR_LEFT     0x8
#define IR_RIGHT    0x5A
#define IR_UP       0x18
#define IR_DOWN     0x52
#define IR_OK       0x1C
#define IR_STAR     0x16
#define IR_CMD1     0x45
#define IR_CMD2     0x46
#define DECODE_NEC

// --- servos + tuning (unchanged from stock) ---
Servo yawServo;
Servo pitchServo;
Servo rollServo;

int pitchServoVal = 100;

const int pitchMoveSpeed = 8;
const int yawMoveSpeed = 90;      // offset from yawStopSpeed; 0 or 180 = full speed
const int yawStopSpeed = 90;      // continuous servo neutral
const int rollMoveSpeed = 90;
const int rollStopSpeed = 90;

// runtime-tunable via "SET YAW_MS <n>" / "SET ROLL_MS <n>"
//   yawPrecision   - how long each yaw pulse runs. longer = more rotation per
//                    pulse; also overcomes cable drag better. stock was 150ms;
//                    350 gives usable motion with a usb cable attached.
//   rollPrecision  - duration of the fire pulse. stock 158 sometimes doesn't
//                    complete a full 60deg rotation; 250 is more reliable.
int yawPrecision = 350;
int rollPrecision = 250;

const int pitchMax = 150;
const int pitchMin = 33;

// --- firmware state tracked between commands ---
bool armed = true;
int dartsEstimate = 6;            // soft estimate; user triggers RELOAD to reset
long yawCumulativeMs = 0;         // dead-reckoned yaw position from home in ms (+ = ccw, - = cw)

// --- serial input buffer ---
const size_t SERIAL_BUF_CAP = 64;
char serialBuf[SERIAL_BUF_CAP];
size_t serialLen = 0;

// --- helpers, forward decls ---
void homeServos();
void fireOne();
void yawPulse(int ms, bool ccw);
void handleSerialCommand(const char *line);

void setup() {
  Serial.begin(9600);

  yawServo.attach(10);
  pitchServo.attach(11);
  rollServo.attach(12);

  IrReceiver.begin(9, ENABLE_LED_FEEDBACK);

  homeServos();
  Serial.println(F("LOG taiga firmware ready"));
  Serial.print(F("STATE pitch=")); Serial.print(pitchServoVal);
  Serial.print(F(" darts=")); Serial.print(dartsEstimate);
  Serial.print(F(" armed=")); Serial.print(armed ? 1 : 0);
  Serial.print(F(" yaw_ms=")); Serial.print(yawCumulativeMs);
  Serial.print(F(" yaw_pulse=")); Serial.print(yawPrecision);
  Serial.print(F(" roll_pulse=")); Serial.println(rollPrecision);
}

void loop() {
  // --- serial input (non-blocking) ---
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (serialLen > 0) {
        serialBuf[serialLen] = '\0';
        handleSerialCommand(serialBuf);
        serialLen = 0;
      }
    } else if (serialLen < SERIAL_BUF_CAP - 1) {
      serialBuf[serialLen++] = c;
    } else {
      // overflow: drop the line entirely to avoid partial parse
      serialLen = 0;
      Serial.println(F("ERR LINE_TOO_LONG"));
    }
  }

  // --- IR remote (unchanged functionality) ---
  if (IrReceiver.decode()) {
    if (IrReceiver.decodedIRData.protocol != UNKNOWN) {
      switch (IrReceiver.decodedIRData.command) {
        case IR_UP:    if (pitchServoVal + pitchMoveSpeed < pitchMax) {
                         pitchServoVal += pitchMoveSpeed; pitchServo.write(pitchServoVal);
                       } break;
        case IR_DOWN:  if (pitchServoVal - pitchMoveSpeed > pitchMin) {
                         pitchServoVal -= pitchMoveSpeed; pitchServo.write(pitchServoVal);
                       } break;
        case IR_LEFT:  yawPulse(yawPrecision, true); break;
        case IR_RIGHT: yawPulse(yawPrecision, false); break;
        case IR_OK:    if (armed) { fireOne(); } break;
        case IR_STAR:  if (armed) { for (int i = 0; i < 6 && dartsEstimate > 0; i++) fireOne(); } break;
        case IR_CMD1:  homeServos(); break;
      }
    }
    IrReceiver.resume();
  }

  delay(2);
}

// --- serial command dispatch ---

// advance past leading spaces, return pointer
static const char *skipSpaces(const char *s) {
  while (*s == ' ' || *s == '\t') s++;
  return s;
}

// parse an int from `s` into `out`, return pointer past the number or nullptr on failure.
static const char *parseInt(const char *s, long *out) {
  s = skipSpaces(s);
  if (*s == '\0') return nullptr;
  char *end;
  long v = strtol(s, &end, 10);
  if (end == s) return nullptr;
  *out = v;
  return end;
}

// case-insensitive prefix match; returns pointer past the prefix or nullptr
static const char *matchCmd(const char *line, const char *cmd) {
  while (*cmd) {
    char a = *line++, b = *cmd++;
    if (a >= 'a' && a <= 'z') a -= 32;
    if (a != b) return nullptr;
  }
  // next char must be end or whitespace so "PITCH_REL" doesn't match "PITCH"
  if (*line == '\0' || *line == ' ' || *line == '\t') return line;
  return nullptr;
}

void handleSerialCommand(const char *line) {
  line = skipSpaces(line);
  const char *rest;

  if ((rest = matchCmd(line, "YAW"))) {
    rest = skipSpaces(rest);
    if (*rest == '\0') { Serial.println(F("ERR YAW_NEEDS_DIR")); return; }
    char dir = *rest;
    if (dir >= 'a' && dir <= 'z') dir -= 32;
    if (dir != 'L' && dir != 'R') { Serial.println(F("ERR YAW_DIR_LR")); return; }
    long ms = yawPrecision;
    const char *after = parseInt(rest + 1, &ms);
    (void)after;
    if (ms < 10) ms = 10;
    if (ms > 2000) ms = 2000;
    yawPulse((int)ms, dir == 'L');
    Serial.print(F("OK YAW ")); Serial.print(dir); Serial.print(' '); Serial.println(ms);
    return;
  }

  if ((rest = matchCmd(line, "PITCH_REL"))) {
    long delta = 0;
    if (!parseInt(rest, &delta)) { Serial.println(F("ERR PITCH_REL_NEEDS_INT")); return; }
    long target = pitchServoVal + delta;
    if (target < pitchMin) target = pitchMin;
    if (target > pitchMax) target = pitchMax;
    pitchServoVal = (int)target;
    pitchServo.write(pitchServoVal);
    delay(30);
    Serial.print(F("OK PITCH_REL ")); Serial.println(pitchServoVal);
    return;
  }

  if ((rest = matchCmd(line, "PITCH"))) {
    long angle = 0;
    if (!parseInt(rest, &angle)) { Serial.println(F("ERR PITCH_NEEDS_INT")); return; }
    if (angle < pitchMin) angle = pitchMin;
    if (angle > pitchMax) angle = pitchMax;
    pitchServoVal = (int)angle;
    pitchServo.write(pitchServoVal);
    delay(30);
    Serial.print(F("OK PITCH ")); Serial.println(pitchServoVal);
    return;
  }

  if ((rest = matchCmd(line, "FIRE"))) {
    if (!armed) { Serial.println(F("ERR DISARMED")); return; }
    long count = 1;
    parseInt(rest, &count);
    if (count < 1) count = 1;
    if (count > 2) count = 2;
    int fired = 0;
    for (int i = 0; i < count && dartsEstimate > 0; i++) {
      fireOne();
      fired++;
    }
    Serial.print(F("OK FIRE ")); Serial.print(fired);
    Serial.print(F(" darts_left=")); Serial.println(dartsEstimate);
    return;
  }

  if ((rest = matchCmd(line, "HOME"))) {
    homeServos();
    Serial.println(F("OK HOME"));
    return;
  }

  if ((rest = matchCmd(line, "STATUS"))) {
    Serial.print(F("STATE pitch=")); Serial.print(pitchServoVal);
    Serial.print(F(" darts=")); Serial.print(dartsEstimate);
    Serial.print(F(" armed=")); Serial.print(armed ? 1 : 0);
    Serial.print(F(" yaw_ms=")); Serial.print(yawCumulativeMs);
    Serial.print(F(" yaw_pulse=")); Serial.print(yawPrecision);
    Serial.print(F(" roll_pulse=")); Serial.println(rollPrecision);
    Serial.println(F("OK STATUS"));
    return;
  }

  if ((rest = matchCmd(line, "SET"))) {
    rest = skipSpaces(rest);
    // read key token
    const char *keyStart = rest;
    while (*rest && *rest != ' ' && *rest != '\t') rest++;
    size_t keyLen = rest - keyStart;
    long v = 0;
    if (!parseInt(rest, &v)) { Serial.println(F("ERR SET_NEEDS_VALUE")); return; }
    if (keyLen == 6 && strncasecmp(keyStart, "YAW_MS", 6) == 0) {
      if (v < 20 || v > 2000) { Serial.println(F("ERR SET_YAW_MS_RANGE")); return; }
      yawPrecision = (int)v;
      Serial.print(F("OK SET YAW_MS ")); Serial.println(v);
      return;
    }
    if (keyLen == 7 && strncasecmp(keyStart, "ROLL_MS", 7) == 0) {
      if (v < 80 || v > 800) { Serial.println(F("ERR SET_ROLL_MS_RANGE")); return; }
      rollPrecision = (int)v;
      Serial.print(F("OK SET ROLL_MS ")); Serial.println(v);
      return;
    }
    Serial.println(F("ERR SET_UNKNOWN_KEY"));
    return;
  }

  if ((rest = matchCmd(line, "ARM"))) {
    armed = true;
    Serial.println(F("OK ARMED"));
    return;
  }

  if ((rest = matchCmd(line, "DISARM"))) {
    armed = false;
    Serial.println(F("OK DISARMED"));
    return;
  }

  if ((rest = matchCmd(line, "RELOAD"))) {
    dartsEstimate = 6;
    Serial.println(F("OK RELOAD 6"));
    return;
  }

  Serial.print(F("ERR UNKNOWN ")); Serial.println(line);
}

// --- physical motions ---

void yawPulse(int ms, bool ccw) {
  yawServo.write(yawStopSpeed + (ccw ? yawMoveSpeed : -yawMoveSpeed));
  delay(ms);
  yawServo.write(yawStopSpeed);
  yawCumulativeMs += ccw ? ms : -ms;
  delay(5);
}

void fireOne() {
  rollServo.write(rollStopSpeed + rollMoveSpeed);
  delay(rollPrecision);
  rollServo.write(rollStopSpeed);
  if (dartsEstimate > 0) dartsEstimate--;
  delay(5);
}

void homeServos() {
  yawServo.write(yawStopSpeed);
  delay(20);
  rollServo.write(rollStopSpeed);
  delay(100);
  pitchServo.write(100);
  pitchServoVal = 100;
  yawCumulativeMs = 0;
  delay(100);
}
