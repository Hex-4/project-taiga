# project taiga - anti-cat living room defense system

> commissioned by: mom
> target: cat who poops/pees in the living room
> vibe: mark rober meets home alone meets responsible pet ownership

## overview

a security system that detects when a cat enters the living room, tracks it, and
if the cat stops moving for ~10 seconds (the "about to do its business" pose), an
IR dart turret gently persuades it to relocate.

## architecture

```
 [main phone]              [dev pc (later: t430)]          [arduino + turret]
  camera    --(tailscale)-->   CV server      --(usb/serial)-->  aim & fire
  (stream)                     (detection +                      (servo control)
                                tracking)
```

### why this split?

- phone streams camera feed via ip webcam app (zero code on phone side)
- CV server does the brain work (detection, tracking, aim calculation)
- arduino handles physical turret control (keeps IR remote working too)
- **tailscale** connects everything because the home network has client
  isolation enabled and blocks device-to-device traffic. tailscale gives
  every device a stable IP on a virtual network that Just Works.

### deployment plan

- **dev phase**: CV server runs on the current pc (this machine). easier to
  iterate, has more horsepower, already set up.
- **deployment phase**: once it works, migrate to the thinkpad t430 which can
  live in the living room and run headless 24/7. code should be portable.

## components

### 1. camera stream (phone)

- **what**: android app or web page that streams camera feed over local network
- **options (pick one)**:
  - **ip webcam app** (easiest) - free android app, exposes an MJPEG stream over
    http. zero code needed on the phone side. just install and point.
    url looks like `http://192.168.x.x:8080/video`
  - **custom web app** - browser-based, uses getUserMedia API to capture camera
    and streams via WebRTC or websocket to the server. more work but more control.
- **recommendation**: ip webcam app. why write code when someone already did it.
  save the effort for the hard parts.

### 2. CV server (thinkpad t430)

runs on the t430 over local network. this is the brain.

#### detection pipeline

```
camera stream --> frame grab --> cat detection --> tracking --> stillness check --> fire command
                  (opencv)      (yolov8-nano)    (centroid)   (10s timer)        (serial cmd)
```

- **framework**: python + opencv + ultralytics (yolov8)
- **model**: yolov8n (nano) - tiny, fast, runs fine on CPU. COCO dataset already
  includes "cat" as a class (class 15), so zero training needed.
- **detection flow**:
  1. grab frames from MJPEG stream (~10-15 fps is plenty)
  2. run yolov8n inference on each frame
  3. filter detections for class "cat" with confidence > 0.5
  4. track the cat's bounding box center across frames
  5. if the center position stays within a small radius for ~10 seconds:
     - cat is stationary = threat detected
     - calculate turret aim angles
     - send fire command to arduino

#### coordinate mapping (camera <-> turret)

since the camera and turret are in different physical locations, we need a
calibration step to map pixel coordinates to turret aim commands.

**extra complexity**: yaw is in "steps" (timed pulses), not absolute degrees.
so the mapping is: pixel position --> (yaw_steps_from_home, pitch_angle).

- **calibration procedure**:
  1. home the turret
  2. place a target object at several known spots on the floor (like a grid)
  3. for each spot:
     a. note its pixel coordinates in the camera feed
     b. manually aim the turret at it (using the web dashboard or IR remote)
     c. record the yaw steps taken from home + the pitch angle
  4. store these as calibration points
  5. at runtime: interpolate between calibration points to estimate aim commands

- **recommendation**: start with a simple 3x3 or 4x4 grid of calibration points.
  use bilinear interpolation between them. nerf darts have enough spread that
  this should work fine. we can always add more points later if needed.

- **drift correction**: since yaw drifts over time with the continuous servo,
  the system should re-home periodically (e.g., after each engagement or every
  few minutes when idle). homing could use a physical endstop or just assume
  the power-on position.

### 3. turret controller (arduino)

the mark rober hack pack IR turret. stock code is in `turret-stock.ino`.
we'll write custom firmware that adds serial control alongside the existing
IR remote control (keep IR working for manual override / fun).

#### hardware details (from stock code analysis)

- **3 servos**:
  - **yaw** (pin 10) - CONTINUOUS ROTATION servo. this is NOT positional.
    controlled by speed (0-180 where 90=stop, 0=full CW, 180=full CCW) and
    duration. stock uses speed=90 offset from center, duration=150ms per step.
  - **pitch** (pin 11) - standard positional servo. range 33-150 degrees.
    default home position is 100. moves in 8-degree increments in stock code.
  - **roll/fire** (pin 12) - CONTINUOUS ROTATION servo. one dart = one 60-degree
    rotation pulse (~158ms at full speed). 6 darts total in the barrel.
- **IR receiver** on pin 9 (NEC protocol)
- **serial**: 9600 baud, already initialized

#### the yaw problem (continuous rotation)

this is the biggest challenge. since the yaw servo is continuous rotation,
we have NO position feedback. the turret doesn't "know" where it's pointing.

**approach: dead reckoning with home reference**
1. define a "home" yaw position (where it points on power-up / after homing)
2. track yaw position in software by accumulating timed movements
3. to aim at a target yaw angle: calculate delta from current estimated position,
   then rotate for the appropriate duration
4. periodically re-home to correct drift (continuous servos aren't super precise)

**alternative: add a position sensor**
- could add a potentiometer or encoder to the yaw axis for real feedback
- more accurate but requires hardware modification
- consider this if dead reckoning isn't good enough

**practical note**: since nerf darts have significant spread anyway, we don't
need pinpoint accuracy. "roughly in the right direction" is probably fine.
the cat is a big target and the room isn't huge.

#### communication protocol

- **interface**: USB serial from t430 to arduino, 9600 baud
- **proposed commands** (sent from python, newline-terminated):
  ```
  YAW <steps> <direction>   - rotate yaw: steps=number of move pulses, direction=L or R
  PITCH <angle>             - set pitch to absolute angle (33-150)
  FIRE <count>              - fire count darts (1-6)
  HOME                      - return to home position (yaw stop + pitch 100)
  AIM <yaw_steps> <yaw_dir> <pitch_angle>  - combined aim command
  AIMFIRE <yaw_steps> <yaw_dir> <pitch_angle> <dart_count>  - aim then fire
  STATUS                    - report current state (pitch angle, dart count estimate)
  ```
- **responses** (sent from arduino, for acknowledgment):
  ```
  OK:<command>              - command executed successfully
  ERR:<reason>              - command failed (e.g., pitch out of range)
  STATUS:<pitch>,<darts_remaining_estimate>
  ```
- **note on yaw units**: since yaw is continuous rotation, we express it in
  "steps" (each step = one yawPrecision-duration pulse). the python side will
  need to calibrate how many steps = how many degrees of rotation.

#### safety features in firmware
- max 2 darts per FIRE command (ignore higher counts)
- minimum 3-second cooldown between fire commands
- pitch limits enforced (33-150 degrees)
- watchdog: if no command received for 60 seconds, auto-home
- IR remote still works for manual override

### 4. web dashboard (optional but cool)

a simple web UI served from the t430 for monitoring and control.

- **features**:
  - live camera feed with detection overlay (bounding boxes)
  - turret status (current angles, armed/disarmed)
  - event log (timestamps of detections + actions)
  - manual override (aim and fire manually for testing/fun)
  - arm/disarm toggle (don't want it firing when you have guests)
  - calibration interface
- **tech**: flask or fastapi backend, simple html/js frontend
- **not critical for v1** but makes debugging and calibration way easier

## phases

### phase 1: camera + detection (get eyes working) [DONE]
- custom web page (no app needed) on phone uses getUserMedia + websocket
- fastapi server on dev pc ingests jpeg frames from phone
- viewer dashboard shows the stream
- **milestone**: frames flow phone -> server -> viewer

### phase 2: tracking + stillness detection [DONE]
- yolov8n integrated (CAT_CLASS_ID=15). ~40ms warm inference on cpu.
- single-slot latest-frame mailbox + worker gives automatic frame-drop
  backpressure (if detection is slower than phone fps, frames get dropped
  rather than queued)
- StillnessTracker picks the largest cat per frame, tracks centroid over a
  rolling 15s history window, reports stillness_s and stationary flag
- detection json echoed back on the ingest ws so the phone can overlay boxes
- boxes + stillness timer + "THREAT LOCKED" banner on both phone and viewer
- **milestone**: system correctly identifies when cat stops moving for 10s

### phase 3: turret integration
- fork `turret-stock.ino` into `turret-taiga.ino`
- add serial command parser (keep IR remote working too)
- implement yaw step tracking (dead reckoning from home position)
- implement pitch absolute positioning via serial
- implement fire command with dart count + cooldown
- python serial client on t430: send commands, read responses
- test: aim and fire turret from python script
- **milestone**: can reliably control turret from python over serial

### phase 4: coordinate mapping + calibration [DONE]
- click-to-mark calibration UI in /view: toggle mode, click frame to stage a
  pixel, aim turret at the real-world spot, hit capture. records
  (px, py, pitch, yaw_ms) tuples.
- IDW interpolation (inverse-distance weighting, power=2) for aim estimation
  from arbitrary pixels. handles scattered point layouts, degrades gracefully
  to nearest-neighbor when extrapolating.
- calibration persisted to calibration.json (atomic write), survives restarts.
- detection_worker now does aim-then-fire in auto mode: when stationary +
  auto_fire + armed + calibration points exist, it interpolates aim commands,
  moves the turret via `Turret.aim_to()`, settles 300ms, then fires.
- manual "test aim" button in /view to preview an aim without firing.
- if calibration is empty, auto-fire still works but shoots from whatever
  pose the turret happens to be in (preserves the phase 3 behavior).
- **milestone**: turret can aim at detected target from camera feed

### phase 5: polish + safety
- add cooldown timer between fire events
- arm/disarm functionality
- web dashboard for monitoring
- notification system (optional - text mom when cat gets yeeted)
- edge case handling (multiple cats, cat partially in frame, etc.)
- **milestone**: system runs reliably unsupervised

## tech stack summary

| component       | tech                              |
|-----------------|-----------------------------------|
| camera          | ip webcam android app (or custom) |
| CV server       | python 3, opencv, ultralytics     |
| detection model | yolov8n (pre-trained COCO)        |
| turret control  | arduino, servo lib, IRremote, serial |
| communication   | wifi (camera), USB serial (turret)|
| dashboard       | flask/fastapi + vanilla js        |
| OS on t430      | whatever linux is on there        |

## hardware needed

- [x] samsung galaxy s6 edge (camera)
- [x] thinkpad t430 (CV server)
- [x] mark rober hack pack IR turret (the weapon)
- [ ] USB cable (t430 <-> arduino)
- [ ] phone mount/tripod (stable camera position)
- [ ] turret mount (stable, aimed at the living room floor area)

## key technical risks

### the yaw problem (biggest risk)

the continuous rotation yaw servo with no position feedback is the #1 technical
risk. dead reckoning WILL drift over time. mitigations:
1. frequent re-homing (after each fire event)
2. keep yaw movements small and predictable
3. if it's really bad: add a potentiometer to the yaw axis ($2 fix)
4. if that's STILL bad: replace the continuous servo with a standard one
   (may require physical modification to the turret)

### dart count tracking

with 6 darts in the barrel and no sensor to detect empty, we track by counting
fires. the system needs to know when it's out of ammo and alert for reload.
this is a software-only problem and not hard, just important.

## other risks and considerations

- **false positives**: yolov8 might detect other things as cats (unlikely but
  possible). confidence threshold tuning should handle this.
- **cat avoidance learning**: cat might learn to avoid the room entirely (which
  is... actually the goal lol. mission accomplished?)
- **dart safety**: nerf/IR darts are soft, but make sure the turret isn't aimed
  at face height. mount it to shoot downward-ish at floor level.
- **network latency**: wifi stream + processing time means ~0.5-1s delay.
  fine for a cat that's been still for 10 seconds.
- **lighting**: low light conditions might hurt detection accuracy. might need
  a nightlight or IR camera setup if the cat is a nocturnal offender.
- **ethical note**: this is meant to be a gentle deterrent, not punishment.
  one or two soft darts. the cat will be mildly annoyed at worst.

## naming

project codename: **taiga** (it's a cat thing, get it? like the biome but also
sounds like "tiger" if you squint)
