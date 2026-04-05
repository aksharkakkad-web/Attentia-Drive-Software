# ATTENTIA DRIVE MVP — BUILD PLAN v2

**Revised after Codex review. All CRITICAL/HIGH findings addressed.**

**What this document is:** Step-by-step instructions for building the Attentia Drive MVP on a MacBook using Claude Code. Each phase is one prompt you paste into Claude Code. Do not move to the next phase until the current one passes its tests.

**What you need before starting:**
- Akshar's repo cloned: `git clone https://github.com/aksharkakkad-web/Attentia-Drive-Software.git`
- The PRD v2.0.0 added to `docs/PRD_v2.md` in the repo
- This build plan added to `docs/BUILD_PLAN.md` in the repo
- The CLAUDE.md and .claude/ config files in the repo root
- Python 3.10+ installed on your Mac
- A working webcam on your Mac

**The rule:** Do NOT skip phases. Do NOT move on until the test passes. If a test fails, fix it in that phase before moving forward. The repo must import cleanly at the end of every phase — no broken imports, ever.

---

## MVP SIMPLIFICATIONS (vs full PRD)

| PRD says | MVP does | Why | Upgrade later by |
|----------|----------|-----|-----------------|
| BlazeFace + PFLD + custom gaze model | MediaPipe FaceLandmarker | Custom models not ready | Swap Layer 1 perception only |
| .rknn model format | MediaPipe .task + TFLite | .rknn is RK3568-only | Convert ONNX → RKNN on device |
| 4 parallel threads | Single-threaded | Mac is fast enough | Add threading for RK3568 |
| Speed from OBD-II/CAN/GPS | Hardcode URBAN (modifier=1.0) | No OBD-II on Mac. PRD §FM-05 | Implement SpeedSource on device |
| Thermal monitor + watchdog | Skip entirely | Mac won't throttle | Add for RK3568 |
| 10s calibration saved per VIN | 5s calibration every startup, not saved | Simpler for MVP | Add persistence on device |
| Dedicated gaze model (D-A) | Head pose as gaze proxy | See D-A/D-B OVERLAP NOTE below | Plug in trained gaze model |

### D-A / D-B OVERLAP NOTE

Without a dedicated gaze model, gaze direction = head direction. This means:
- D-A fires when head turns >15° for 2.0s (gaze off road zone)
- D-B fires when head turns >30° for 1.5s (head pose breach)
- A 35° head turn fires BOTH D-A and D-B, and inflates the composite score

This is acceptable for MVP because:
- D-A catches moderate inattention (15°+), D-B catches severe turns (30°+)
- Each has its own cooldown so they don't double-beep
- The composite score being inflated doesn't matter if individual thresholds are what trigger alerts
- When the real gaze model arrives, D-A will use eye direction (independent of head) and the overlap disappears

### MVP CONTRACT EXTENSIONS

The PRD's PerceptionBundle does NOT carry head pose angles or EAR values directly — it expects Layer 2 to compute these from landmarks. But MediaPipe gives us these values pre-computed. Rather than wastefully recomputing them, the MVP adds three adapter fields to PerceptionBundle:

- `head_pose_raw: tuple | None` — (pitch, yaw, roll) in degrees from MediaPipe
- `ear_left: float` — left eye aspect ratio from MediaPipe
- `ear_right: float` — right eye aspect ratio from MediaPipe

These fields are marked `# MVP-ONLY — remove when custom models replace MediaPipe`. Layers 2-5 consume these through the SignalFrame contract, never directly from PerceptionBundle.

Additionally, the PRD contracts are missing several fields needed for the full alert system:
- TemporalFeatures: `face_absent_continuous_secs`, `perclos_valid`
- DistractionScore: `composite_threshold_breached`, `face_absent_threshold_breached`
- AlertCommand: `active_classes`

These are added in Phase 1 as documented extensions.

---

## PHASE 0: PROJECT SETUP AND CLEANUP

**What this does:** Deletes old logic modules, fixes ALL broken imports so the repo stays clean, downloads models, verifies environment. At the end of this phase, every file imports cleanly and the remaining tests pass.

**Claude Code prompt — copy this exactly:**
```
I'm setting up the Attentia Drive MVP. Read CLAUDE.md and docs/BUILD_PLAN.md for full context. This is Phase 0.

The repo currently has 101 passing tests. After this phase, fewer tests will exist but everything that remains must import cleanly and pass.

STEP 1 — Delete these modules (we're replacing them):
- src/detection/classifier.py
- src/logic/temporal_smoother.py
- src/logic/hysteresis.py
- src/logic/distraction_reasoner.py
- src/logic/alert_manager.py
- src/logic/object_confirmer.py
- src/stubs/ (entire directory)
- train_classifier.py

STEP 2 — Delete old tests that reference deleted modules:
- tests/test_hysteresis.py
- tests/test_temporal_smoother.py
- tests/test_alert_manager.py
- tests/test_object_confirmer.py
- tests/test_distraction_reasoner.py
- tests/test_pipeline_integration.py

STEP 3 — Fix ALL broken imports. This is critical. The following files import from deleted modules and will crash. Fix every one:

3a) src/pipeline/frame_record.py — This file imports ClassifierResult, Alert, DistractionAssessment, DriverState from deleted modules. REWRITE it as a minimal dataclass with NO imports from src/logic/ or src/detection/classifier. Remove the fields that reference deleted types (classifier_result, driver_state, assessment, alert). Keep: frame_number, timestamp, raw_frame, object_detections, face_result, drowsiness_result, smoothed_probability, inference_time_ms, pipeline_time_ms. Import only from src/detection/face_detector, src/detection/object_detector, src/detection/drowsiness. All removed fields become: Optional[Any] = None.

3b) src/pipeline/pipeline_manager.py — This file imports from 6 deleted modules. DELETE this entire file. It is being replaced by pipeline_manager_v2.py in Phase 7B.

3c) src/output/audio_alerter.py — Imports Alert from deleted alert_manager. REWRITE: remove the import. Change the alert() method to accept any object with level and triggers attributes, or just accept keyword arguments. Keep it simple — this file is being replaced by audio_alerter_v2.py in Phase 7A.

3d) src/output/display.py — Imports DriverState from deleted hysteresis AND FrameRecord from frame_record (which was also broken). Fix: remove the DriverState import. Replace STATE_COLORS dict keys with strings ('ATTENTIVE', 'MONITORING', 'DISTRACTED'). Update the render() method to handle the simplified FrameRecord. The display will be significantly rewritten in Phase 7B anyway — just make it import-clean for now.

3e) src/output/telemetry_logger.py — Imports FrameRecord. After fixing frame_record.py in step 3a, this should work. Verify it imports cleanly.

3f) src/main.py — Imports PipelineManager from the deleted pipeline_manager.py. REWRITE main.py as a simple stub:
    - Keep the argument parser and logging setup
    - In the main() function, print "Pipeline not yet wired — see Phase 7B" and exit
    - This will be properly wired in Phase 7B

STEP 4 — Verify EVERY file in src/ imports cleanly:
python -c "import src.detection.face_detector; print('OK: face_detector')"
python -c "import src.detection.object_detector; print('OK: object_detector')"
python -c "import src.detection.drowsiness; print('OK: drowsiness')"
python -c "import src.detection.base_detector; print('OK: base_detector')"
python -c "import src.pipeline.frame_source; print('OK: frame_source')"
python -c "import src.pipeline.frame_record; print('OK: frame_record')"
python -c "import src.output.display; print('OK: display')"
python -c "import src.output.audio_alerter; print('OK: audio_alerter')"
python -c "import src.output.telemetry_logger; print('OK: telemetry_logger')"
python -c "import src.config_loader; print('OK: config_loader')"
python -c "import src.main; print('OK: main')"

ALL must print OK. If any fails, fix it before proceeding.

STEP 5 — Download models:
mkdir -p models
curl -L -o models/face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

For the phone/object detector, check if models/efficientdet_lite0.tflite exists. If not, download it:
curl -L -o models/efficientdet_lite0.tflite "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite"

STEP 6 — Update requirements.txt to:
opencv-python>=4.8.0
numpy>=1.24.0
PyYAML>=6.0
mediapipe>=0.10.9
tflite-runtime>=2.14.0; platform_system != "Windows"
tensorflow>=2.14.0; platform_system == "Windows"
pytest>=7.0.0

Run: pip install -r requirements.txt

STEP 7 — Verify MediaPipe works:
Write a quick test script that opens the webcam, runs MediaPipe FaceLandmarker on one frame, prints whether a face was detected plus the head pose angles, and closes the webcam.

STEP 8 — Run remaining tests:
pytest tests/test_face_detector.py tests/test_drowsiness.py -v

Both must pass. If any other test files still exist, run them too — nothing should fail.

STEP 9 — Commit:
git add -A
git commit -m "phase 0: project cleanup — all imports clean, remaining tests pass"
```

**Test:** ALL Step 4 imports print OK. MediaPipe detects a face. Remaining tests pass. Zero import errors anywhere.

---

## PHASE 1: DATA CONTRACTS AND CONFIG

**What this does:** Defines all typed data messages between layers and all threshold constants. No logic — just definitions. These contracts are the foundation everything else builds on.

**PRD sections:** §4.1–§4.6, §19

**Claude Code prompt — copy this exactly:**
```
Read docs/PRD_v2.md sections §4 and §19. Read docs/BUILD_PLAN.md "MVP CONTRACT EXTENSIONS" section. This is Phase 1.

CREATE FILE: src/contracts.py

Python dataclasses for all inter-layer messages. Every class needs type hints, default values on ALL fields, and a docstring referencing the PRD section.

IMPORTANT: This file defines MVP contracts that extend the PRD with clearly marked fields. Fields not in the PRD are marked with # MVP-ONLY comments.

Include these dataclasses with field names matching the PRD where applicable:

1. RawFrame (PRD §4.1): timestamp_ns (int, 0), frame_id (int, 0), width (int, 640), height (int, 480), channels (int, 3), data (Optional[np.ndarray], None), source_type (str, 'webcam')

2. FaceDetection (PRD §4.2): present (bool, False), confidence (float, 0.0), bbox_norm (tuple, (0,0,0,0)), face_size_px (int, 0)

3. LandmarkOutput (PRD §4.2): landmarks (Optional[np.ndarray], None), confidence (float, 0.0), pose_valid (bool, False)
   NOTE: PRD specifies 98x2 for PFLD. MediaPipe gives 478x3. The contract uses Optional[np.ndarray] so both shapes work.

4. GazeOutput (PRD §4.2): left_eye_yaw, left_eye_pitch, right_eye_yaw, right_eye_pitch, combined_yaw, combined_pitch (all float, 0.0), confidence (float, 0.0), valid (bool, False)

5. PhoneDetectionOutput (PRD §4.2): detected (bool, False), max_confidence (float, 0.0), bbox_norm (Optional[tuple], None)

6. PerceptionBundle (PRD §4.2 + MVP extensions):
   - timestamp_ns (int, 0), frame_id (int, 0)
   - face (FaceDetection, default FaceDetection())
   - landmarks (Optional[LandmarkOutput], None)
   - gaze (Optional[GazeOutput], None)
   - phone (PhoneDetectionOutput, default PhoneDetectionOutput())
   - phone_result_stale (bool, False)
   - inference_ms (float, 0.0)
   - lstm_hidden_state (Any, None)
   - lstm_reset_occurred (bool, False)
   - head_pose_raw (Optional[tuple], None)  # MVP-ONLY: (pitch, yaw, roll) degrees from MediaPipe. Remove when custom models replace MediaPipe.
   - ear_left (float, 0.0)  # MVP-ONLY: from MediaPipe. Remove when custom models replace MediaPipe.
   - ear_right (float, 0.0)  # MVP-ONLY: from MediaPipe. Remove when custom models replace MediaPipe.

7. HeadPose (PRD §4.3): yaw_deg, pitch_deg, roll_deg (float, 0.0), valid (bool, False), raw_yaw_deg, raw_pitch_deg, raw_roll_deg (float, 0.0)

8. EyeSignals (PRD §4.3): left_EAR, right_EAR, mean_EAR (float, 0.0), baseline_EAR (float, 0.28), close_threshold (float, 0.21), valid (bool, False), calibration_complete (bool, False)

9. GazeWorld (PRD §4.3): yaw_deg, pitch_deg (float, 0.0), on_road (bool, True), valid (bool, False)

10. PhoneSignal (PRD §4.3): detected (bool, False), confidence (float, 0.0), stale (bool, False)

11. SignalFrame (PRD §4.3): timestamp_ns (int, 0), frame_id (int, 0), face_present (bool, False), head_pose (Optional[HeadPose], None), eye_signals (Optional[EyeSignals], None), gaze_world (Optional[GazeWorld], None), phone_signal (PhoneSignal, default), speed_mps (float, 0.0), speed_stale (bool, False), signals_valid (bool, False)

12. TemporalFeatures (PRD §4.4 + extensions):
    - timestamp_ns (int, 0)
    - gaze_off_road_fraction (float, 0.0), gaze_continuous_secs (float, 0.0)
    - head_deviation_mean_deg (float, 0.0), head_continuous_secs (float, 0.0)
    - perclos (float, 0.0), perclos_valid (bool, False)  # EXTENSION: validity flag
    - blink_rate_score (float, 0.0)
    - phone_confidence_mean (float, 0.0), phone_continuous_secs (float, 0.0)
    - face_absent_continuous_secs (float, 0.0)  # EXTENSION: needed for ALT-06
    - speed_zone (str, 'URBAN'), speed_modifier (float, 1.0)
    - frames_valid_in_window (int, 0)
    - thermal_throttle_active (bool, False)

13. DistractionScore (PRD §4.5 + extensions):
    - timestamp_ns (int, 0)
    - composite_score (float, 0.0)
    - component_gaze, component_head, component_perclos, component_blink (float, 0.0)
    - gaze_threshold_breached (bool, False)
    - head_threshold_breached (bool, False)
    - perclos_threshold_breached (bool, False)
    - phone_threshold_breached (bool, False)
    - composite_threshold_breached (bool, False)  # EXTENSION: ALT-05
    - face_absent_threshold_breached (bool, False)  # EXTENSION: ALT-06
    - active_classes (list, field(default_factory=list))  # Which D-classes are active

14. AlertLevel (PRD §4.6): enum with LOW=1, HIGH=2, URGENT=3

15. AlertType (PRD §4.6): enum with VISUAL_INATTENTION='D-A', HEAD_INATTENTION='D-B', DROWSINESS='D-C', PHONE_USE='D-D', FACE_ABSENT='FACE'

16. AlertCommand (PRD §4.6 + extension):
    - alert_id (str, ''), timestamp_ns (int, 0)
    - level (AlertLevel, AlertLevel.HIGH)
    - alert_type (AlertType, AlertType.VISUAL_INATTENTION)
    - composite_score (float, 0.0)
    - suppress_until_ns (int, 0)
    - active_classes (list, field(default_factory=list))  # EXTENSION: all active D-classes

CREATE FILE: src/config_prd.py

ALL constants from PRD §19. Copy every value. Comment each with the PRD section reference.

IMPORTANT — these specific values differ from PRD for MVP:
- CALIBRATION_DURATION_S = 5.0  # MVP override (PRD says 10.0). EAR baseline is collected during this same 5s window, NOT a separate 30s window.

All other values must match PRD §19 exactly. Include every constant listed in §19 — model paths, camera settings, confidence gates, road zone, speed zones, duration thresholds, head pose thresholds, EAR/PERCLOS, Kalman filter, gaze transform, scoring weights, blink detection, temporal buffer, cooldowns, degraded state, calibration.

WRITE TESTS in tests/test_contracts.py:
1. Every dataclass instantiates with no arguments (test all 16)
2. Every dataclass instantiates with specific values
3. AlertLevel enum has LOW, HIGH, URGENT
4. AlertType enum has all 5 members (D-A, D-B, D-C, D-D, FACE)
5. WEIGHT_GAZE + WEIGHT_HEAD + WEIGHT_PERCLOS + WEIGHT_BLINK == 1.0 (pytest.approx)
6. Spot-check 15+ config values match PRD §19 exactly
7. DistractionScore.active_classes defaults to empty list (not None)
8. PerceptionBundle defaults: face.present == False, ear_left == 0.0
9. TemporalFeatures has face_absent_continuous_secs field defaulting to 0.0
10. TemporalFeatures has perclos_valid field defaulting to False
11. DistractionScore has composite_threshold_breached and face_absent_threshold_breached
12. AlertCommand has active_classes field
13. CALIBRATION_DURATION_S == 5.0 (MVP override)

Run: pytest tests/test_contracts.py -v
Then: pytest tests/ -v (all tests including old ones must pass)

Git: git add -A && git commit -m "phase 1: data contracts and PRD config — all tests passing"
```

**Test:** `pytest tests/ -v` — ALL tests pass (new contracts + remaining old tests).

---

## PHASE 2: KALMAN FILTER

**PRD section:** §5.6. No changes from original plan — this phase was clean.

**Claude Code prompt — copy this exactly:**
```
Read docs/PRD_v2.md §5.6. This is Phase 2: Kalman filter.

CREATE FILE: src/logic/kalman_filter.py

1D constant-velocity Kalman filter. Import parameters from config_prd.py.

State: [angle, angular_velocity]. F=[[1,dt],[0,1]], H=[[1,0]]. Q, R, P from config.

Interface: __init__(dt, process_noise=None, measurement_noise=None, initial_covariance=None), update(measurement) -> float, reset(), current_value property, is_initialized property.

First update() initializes state to the measurement value. If measurement is None, do predict-only and return predicted value. After reset(), next update() re-initializes.

Use numpy for matrix math. Handle edge cases: None measurement → predict-only step. After reset → re-initialize on next update.

WRITE TESTS in tests/test_kalman_filter.py:
1. Constant 10.0 for 50 frames → output converges to ~10.0 (within 0.5)
2. Noisy signal (10.0 + noise std=5.0, seed=42, 200 frames) → output std dev at least 60% less than input std dev
3. Reset then new value → converges to new value within 1.0
4. Linear ramp 0→30 over 60 frames → final output within 5.0 of 30.0
5. First frame → output within 1.0 of input

Run: pytest tests/test_kalman_filter.py -v
Then: pytest tests/ -v (all tests must pass)

Git: git add -A && git commit -m "phase 2: kalman filter — all tests passing"
```

---

## PHASE 3: SIGNAL PROCESSOR

**PRD sections:** §5.1–§5.7

**Claude Code prompt — copy this exactly:**
```
Read docs/PRD_v2.md §5.1 through §5.7. This is Phase 3: signal processor.

CREATE FILE: src/logic/signal_processor.py

Takes PerceptionBundle → produces SignalFrame. Uses KalmanFilter from Phase 2. Import thresholds from config_prd.py.

KEY DESIGN: The signal processor reads head pose and EAR from the MVP-ONLY fields on PerceptionBundle (head_pose_raw, ear_left, ear_right). It does NOT import MediaPipe or know anything about MediaPipe. When custom models replace MediaPipe later, the adapter (Phase 7A) will populate these same fields from different sources. The signal processor doesn't change.

Internal state: 5 KalmanFilter instances (head yaw/pitch/roll, gaze yaw/pitch), neutral offsets (default 0.0, 0.0), EAR baseline tracking state, face_absent_count (int).

Main method: process(bundle: PerceptionBundle) -> SignalFrame

When face ABSENT (bundle.face.present is False):
  - Increment face_absent_count
  - If face_absent_count > 10 (LSTM_RESET_ABSENT_FRAMES from config): reset all Kalman filters
  - Return SignalFrame with face_present=False, signals_valid=False
  - Phone signal is STILL extracted (phone detection is independent of face)

When face PRESENT (bundle.face.present is True):
  - Reset face_absent_count to 0
  - Extract head pose from bundle.head_pose_raw → (pitch, yaw, roll)
  - If head_pose_raw is None: set head_pose to None, signals_valid=False
  - Apply Kalman filter to yaw, pitch, roll → filtered values
  - Apply neutral offset correction: corrected_yaw = filtered_yaw - neutral_yaw_offset
  - Store raw values in HeadPose debug fields (raw_yaw_deg, raw_pitch_deg, raw_roll_deg)
  - Compute EAR: mean_EAR = (bundle.ear_left + bundle.ear_right) / 2.0
  - If calibration complete: close_threshold = baseline_EAR * EAR_CALIBRATION_MULTIPLIER (0.75)
  - If calibration not complete: close_threshold = EAR_DEFAULT_CLOSE_THRESHOLD (0.21)
  - Compute gaze: FOR MVP, gaze_world_yaw = corrected_yaw, gaze_world_pitch = corrected_pitch
    (When real gaze model arrives: gaze_world = gaze_camera + 0.7*head. Only this line changes.)
  - Apply Kalman filter to gaze_world values
  - Determine on_road: gaze_world_yaw between ROAD_ZONE_YAW_MIN/MAX AND gaze_world_pitch between ROAD_ZONE_PITCH_MIN/MAX
  - Extract phone signal from bundle.phone
  - Return SignalFrame with all fields populated, signals_valid=True

TWO DIFFERENT EAR THRESHOLDS — get this right:
  - close_threshold (baseline * 0.75): For BLINK detection — "eyes closed enough for a blink"
  - PERCLOS check (baseline * 0.20): For PERCLOS — "eyes 80%+ closed". This is computed in temporal engine (Phase 4), NOT here. The signal processor just provides mean_EAR and baseline_EAR in EyeSignals.

Methods: set_neutral_offsets(yaw, pitch), set_ear_baseline(baseline, threshold), reset_filters()

WRITE TESTS in tests/test_signal_processor.py:
1. Face present, head_pose_raw=(5.0, 10.0, 0.0), no offsets → after 10 frames convergence, filtered angles close to input (within 2.0)
2. Face absent → face_present=False, signals_valid=False
3. Face absent but phone detected → phone_signal.detected=True even with face_present=False
4. Set offsets (5.0, 3.0), feed head_pose_raw=(0.0, 15.0, 0.0) for 10 frames → corrected_yaw ≈ 10.0 (15-5)
5. Gaze on_road: corrected angles (0, 0) → on_road=True
6. Gaze off_road: corrected yaw = 20.0 → on_road=False
7. EAR with set_ear_baseline(0.30, 0.225) → EyeSignals has correct values
8. EAR without calibration → close_threshold=0.21 (default)
9. Face absent 15 frames → Kalman filters reset → new pose converges quickly
10. head_pose_raw is None → signals_valid=False, head_pose=None

Run: pytest tests/test_signal_processor.py -v
Then: pytest tests/ -v

Git: git add -A && git commit -m "phase 3: signal processor — all tests passing"
```

---

## PHASE 4: TEMPORAL ENGINE

**PRD sections:** §5.4, §5.5, §2.3, §FR-3.1–FR-3.5

**Claude Code prompt — copy this exactly:**
```
Read docs/PRD_v2.md §5.4, §5.5, §2.3, §FR-3.1 through FR-3.5. This is Phase 4.

CREATE 4 FILES:

FILE 1: src/logic/duration_timer.py
Stopwatch for continuous conditions. tick(active: bool, delta_seconds: float). If active: add delta to total. If not active: reset to 0.0. Clamp delta to [0.0, 1.0] per tick. Properties: current_duration, reset().

FILE 2: src/logic/perclos_calculator.py (PRD §5.4)
Sliding window (deque maxlen=PERCLOS_WINDOW_FRAMES=60). update(eyes_closed: bool, frame_valid: bool). 

perclos property: count(eyes_closed AND frame_valid) / count(frame_valid). Returns 0.0 if fewer than PERCLOS_MIN_VALID_FRAMES valid frames.
valid property: True if >= PERCLOS_MIN_VALID_FRAMES (30) valid frames in window.

CRITICAL: "eyes_closed" for PERCLOS means mean_EAR <= baseline_EAR * (1.0 - PERCLOS_CLOSURE_FRACTION). With PERCLOS_CLOSURE_FRACTION=0.80, that's baseline * 0.20. The CALLER (temporal_engine) computes this boolean and passes it in.

FILE 3: src/logic/blink_detector.py (PRD §5.5)
Tracks EAR transitions. A blink = EAR drops below close_threshold (baseline*0.75), stays below for BLINK_MIN_FRAMES(2) to BLINK_MAX_FRAMES(10) frames, then rises above. Stays below >10 frames = NOT a blink (extended closure).

blink_rate_hz: blinks per second over a rolling 30-second window.
blink_rate_score per PRD §5.5: below 0.13→drowsy signal, above 0.50→fatigue signal, between→0.0. Clamp to [0,1].

FILE 4: src/logic/temporal_engine.py
Orchestrates all temporal state. process(signal_frame: SignalFrame) -> TemporalFeatures.

Internal components:
- Circular buffer (deque maxlen=CIRCULAR_BUFFER_SIZE=120)
- Duration timers: gaze_timer, head_timer, phone_timer, face_absent_timer
- PERCLOS calculator
- Blink detector

Duration timer logic (use frame timestamps for delta_seconds, NOT fixed 1/30):
- gaze_timer ticks when: gaze_world is not None AND on_road==False AND face_present==True AND signals_valid==True
- head_timer ticks when: head_pose is not None AND (|yaw_deg| > HEAD_YAW_THRESHOLD_DEG OR |pitch_deg| > HEAD_PITCH_THRESHOLD_DEG) AND face_present==True
- phone_timer ticks when: phone_signal.detected==True AND phone_signal.confidence >= PHONE_CONFIDENCE_THRESHOLD
- face_absent_timer ticks when: face_present==False. Resets when face appears.

PERCLOS feeding: if eye_signals is valid and calibration_complete:
  perclos_closed = (mean_EAR <= baseline_EAR * (1.0 - PERCLOS_CLOSURE_FRACTION))
  Feed perclos_closed to calculator with frame_valid=True.
  If eye_signals is None or not valid: feed frame_valid=False.

Compute gaze_off_road_fraction from buffer: frames where gaze off road / total valid frames.
Compute head_deviation_mean_deg from buffer: mean of sqrt(yaw² + pitch²) over valid frames.

Set perclos_valid = perclos_calculator.valid in the output.
Set face_absent_continuous_secs = face_absent_timer.current_duration.

Speed zone: Always 'URBAN', speed_modifier=1.0 for MVP.

Handle None gracefully everywhere: if signal_frame fields are None, skip that computation, don't crash.

WRITE TESTS in tests/test_temporal_engine.py:
1. Duration: 60 ticks at 1/30s each → ~2.0s (pytest.approx(2.0, abs=0.1))
2. Duration: 30 true, 1 false, 30 true → ~1.0s (reset works)
3. PERCLOS: 12/60 closed = 0.20
4. PERCLOS: only 20 valid frames → valid=False, perclos=0.0
5. Blink: 3-frame closure (EAR: 0.30, 0.15, 0.15, 0.15, 0.30 with threshold 0.21) → 1 blink
6. Not-blink: 20-frame closure → 0 blinks
7. Blink score: rate=0.06 → score ≈ 0.54
8. Blink score: rate=0.30 → score = 0.0
9. Full engine: 120 frames gaze off-road → continuous ≈ 4.0s, fraction ≈ 1.0
10. Speed zone = 'URBAN', modifier = 1.0
11. Face absent: face_absent_timer ticks, gaze/head timers at 0.0
12. face_absent_continuous_secs populated correctly (e.g., 60 frames absent at 30fps → ~2.0s)
13. perclos_valid is False when fewer than 30 valid frames
14. None handling: signal_frame with head_pose=None → doesn't crash, head timer stays at 0

Run: pytest tests/test_temporal_engine.py -v
Then: pytest tests/ -v

Git: git add -A && git commit -m "phase 4: temporal engine — all tests passing"
```

---

## PHASE 5: SCORING ENGINE AND ALERT STATE MACHINE

**PRD sections:** §6, §7

**Claude Code prompt — copy this exactly:**
```
Read docs/PRD_v2.md §6 and §7. This is Phase 5.

CREATE FILE: src/logic/scoring_engine.py (PRD §6)

Takes TemporalFeatures → DistractionScore. Import ALL thresholds from config_prd.py.

Composite formula (§6.1):
  F1 = gaze_off_road_fraction
  F2_norm = clamp(head_deviation_mean_deg / 30.0, 0.0, 1.0)
  F3 = perclos (use 0.0 if perclos_valid is False)
  F4 = blink_rate_score
  D_raw = W1*F1 + W2*F2_norm + W3*F3 + W4*F4
  D = D_raw * speed_modifier
  Assert weights sum to 1.0 at startup.

SIX independent threshold checks (§6.2):
  ALT-01: gaze_threshold_breached = gaze_continuous_secs >= T_GAZE_SECONDS (2.0)
  ALT-02: head_threshold_breached = head_continuous_secs >= T_HEAD_SECONDS (1.5)
  ALT-03: perclos_threshold_breached = perclos >= PERCLOS_ALERT_THRESHOLD (0.15) AND perclos_valid
  ALT-04: phone_threshold_breached = phone_continuous_secs >= T_PHONE_SECONDS (1.0)
  ALT-05: composite_threshold_breached = D >= COMPOSITE_ALERT_THRESHOLD (0.55)
  ALT-06: face_absent_threshold_breached = face_absent_continuous_secs >= T_FACE_ABSENT_SECONDS (5.0)

If speed_modifier == 0.0 (PARKED): D = 0.0, all non-phone breaches = False.

Build active_classes: for each breached threshold, add the class string ('D-A', 'D-B', 'D-C', 'D-D').
NOTE on D-A/D-B overlap: both can be in active_classes simultaneously. This is expected for MVP. See BUILD_PLAN "D-A / D-B OVERLAP NOTE".

Interface: __init__(), score(features: TemporalFeatures) -> DistractionScore

CREATE FILE: src/logic/alert_state_machine.py (PRD §7)

Interface: __init__(), update(score: DistractionScore, signals_valid: bool) -> AlertCommand or None

The signals_valid parameter is separate from DistractionScore because DEGRADED state depends on perception health, not scoring.

Internal state:
- cooldown_expiry: dict mapping AlertType → expiry time (time.monotonic()). Empty = not in cooldown for that type.
- degraded_invalid_count: consecutive frames with signals_valid=False
- degraded_recovery_count: consecutive frames with signals_valid=True while in DEGRADED
- is_degraded: bool

Logic:

DEGRADED CHECK (first, before any alert logic):
  If signals_valid is False: increment degraded_invalid_count, reset degraded_recovery_count
  If signals_valid is True: reset degraded_invalid_count, increment degraded_recovery_count if currently degraded
  If degraded_invalid_count >= DEGRADED_TRIGGER_FRAMES (60): enter DEGRADED (is_degraded=True)
  If is_degraded and degraded_recovery_count >= DEGRADED_RECOVERY_FRAMES (30): exit DEGRADED
  If is_degraded: return None (no alerts in DEGRADED — P-04)

ALERT CHECK (only if not DEGRADED):
  Check each breach flag in DistractionScore. For each breached type:
    Map to AlertType:
      gaze_threshold_breached → VISUAL_INATTENTION
      head_threshold_breached → HEAD_INATTENTION
      perclos_threshold_breached → DROWSINESS
      phone_threshold_breached → PHONE_USE
      face_absent_threshold_breached → FACE_ABSENT
      composite_threshold_breached → use the highest-priority active class, or VISUAL_INATTENTION as default
    
    Check if that AlertType is in cooldown (expiry time > current time). If in cooldown: suppress.
    EXCEPTION — PHONE_USE (URGENT): ignores ALL cooldowns. Always fires. (P-01)
    EXCEPTION — FACE_ABSENT: has independent cooldown tracking. Can fire even if other types are in cooldown. (P-03)

  If any non-suppressed breach exists:
    Pick the highest-priority type: PHONE_USE > FACE_ABSENT > others (first breached)
    Emit AlertCommand with:
      alert_id = str(uuid.uuid4())
      level = URGENT for phone, HIGH for everything else
      alert_type = the selected type
      composite_score = score.composite_score
      active_classes = score.active_classes
      suppress_until_ns = int((time.monotonic() + cooldown_for_type) * 1e9)
    Set cooldown for that type: cooldown_expiry[type] = time.monotonic() + cooldown_duration

Cooldown durations from config_prd.py:
  VISUAL_INATTENTION → COOLDOWN_VISUAL (8s)
  HEAD_INATTENTION → COOLDOWN_HEAD (8s)
  DROWSINESS → COOLDOWN_DROWSINESS (12s)
  PHONE_USE → COOLDOWN_PHONE (5s)
  FACE_ABSENT → COOLDOWN_FACE_ABSENT (10s)

WRITE TESTS in tests/test_scoring_and_alerts.py:

SCORING (9 tests):
1. All zeros → composite=0.0, no breaches
2. gaze_off_road_fraction=1.0 only → composite=0.45, composite_threshold_breached=False
3. gaze=1.0 + head_deviation=30.0 → composite=0.75, composite_threshold_breached=True
4. PARKED (modifier=0.0) → composite=0.0, only phone can breach
5. HIGHWAY (modifier=1.4) → composite multiplied
6. gaze_continuous_secs=2.5 → gaze_threshold_breached=True
7. phone_continuous_secs=1.5 → phone_threshold_breached=True
8. perclos=0.20 with perclos_valid=True → perclos_threshold_breached=True
9. perclos=0.20 with perclos_valid=False → perclos_threshold_breached=False (invalid PERCLOS doesn't trigger)

ALERT STATE MACHINE (9 tests):
10. No breaches for 100 updates → no alerts
11. Phone breach → URGENT alert, type=D-D
12. Same phone breach continues → no alert during 5s cooldown
13. After 5+ seconds, phone still breached → new alert fires
14. Gaze breach → HIGH alert, type=D-A
15. While D-A in cooldown, phone breach → phone alert fires (P-01 override)
16. signals_valid=False for 60 updates → DEGRADED, no alerts
17. In DEGRADED, signals_valid=True for 30 updates → recovers, alerts work again
18. Multiple breaches (gaze + head) → one alert with active_classes containing both 'D-A' and 'D-B'

Run: pytest tests/test_scoring_and_alerts.py -v
Then: pytest tests/ -v

Git: git add -A && git commit -m "phase 5: scoring and alerts — all tests passing"
```

---

## PHASE 6: CALIBRATION

**PRD section:** §23

**Claude Code prompt — copy this exactly:**
```
Read docs/PRD_v2.md §23. This is Phase 6.

CREATE FILE: src/logic/calibration.py

5-second startup calibration. Collects head pose (yaw, pitch) and EAR samples while face is visible.

__init__(target_duration_s=CALIBRATION_DURATION_S, fps=30.0)
feed_frame(head_yaw, head_pitch, mean_ear, face_visible) -> bool (True when complete)

Needs at least fps * target_duration_s * 0.90 valid samples (face must be visible 90% of frames).
Computes: neutral_yaw_offset = mean(yaw_samples), neutral_pitch_offset = mean(pitch_samples), baseline_ear = mean(ear_samples), close_threshold = baseline_ear * EAR_CALIBRATION_MULTIPLIER (0.75).

Quality check: std dev of yaw < CALIBRATION_MAX_POSE_STD_DEG (5.0) AND std dev of pitch < 5.0.
On failure (not enough frames, or std dev too high): defaults (0.0 offsets, 0.21 threshold).

Failure timeout: if total frames fed > fps * target_duration_s * 2.0 and still not enough valid frames → fail.

Properties: is_complete, neutral_yaw_offset, neutral_pitch_offset, baseline_ear, close_threshold, quality_ok, status ('collecting'|'complete'|'failed'). reset() method.

TESTS in tests/test_calibration.py:
1. 150 stable frames (yaw=5, pitch=3, ear=0.30) → complete, offsets=(5,3), baseline=0.30, threshold≈0.225
2. 150 noisy frames (yaw std=8) → fails quality, defaults
3. 150 frames with 100 face_visible=False → fails, defaults
4. After failure: offsets=(0,0), threshold=0.21
5. Reset works

Run: pytest tests/test_calibration.py -v
Then: pytest tests/ -v

Git: git add -A && git commit -m "phase 6: calibration — all tests passing"
```

---

## PHASE 7A: ADAPTERS AND OUTPUT MODULES

**What this does:** Bridges MediaPipe output to our contracts, creates audio alerter and event logger. All with tests.

**Claude Code prompt — copy this exactly:**
```
This is Phase 7A: adapters and output modules.

Before creating any files, read these existing files to understand their actual interfaces:
- src/detection/face_detector.py (especially FaceResult dataclass and detect() return type)
- src/detection/object_detector.py (especially ObjectDetection dataclass)
- src/contracts.py (the target contract types)

CREATE FILE: src/pipeline/adapters.py

Function: convert_to_perception_bundle(face_result, object_detections, frame_id, timestamp_ns) -> PerceptionBundle

Parameters:
  face_result: FaceResult from src/detection/face_detector.py (may have face_visible=False)
  object_detections: list of ObjectDetection from src/detection/object_detector.py
  frame_id: int
  timestamp_ns: int

Conversion logic:
  - face_result.face_visible → FaceDetection(present=True/False, confidence=0.95 if visible else 0.0)
  - face_result.landmarks → LandmarkOutput (landmarks=face_result.landmarks, confidence=0.9 if visible, pose_valid=face_visible). NOTE: MediaPipe gives 478x3 landmarks. Store as-is — the contract uses Optional[np.ndarray] which accepts any shape.
  - face_result.head_pose → head_pose_raw=(pitch, yaw, roll) tuple. If head_pose is None, set head_pose_raw=None.
  - face_result.ear_left, ear_right → ear_left, ear_right on PerceptionBundle
  - face_result.gaze_vector → GazeOutput. For MVP: set valid=True if face visible, set combined_yaw/pitch to 0.0 (we use head pose as gaze proxy in signal processor, not this field).
  - object_detections → find "cell phone" with highest confidence. PhoneDetectionOutput(detected=conf>=0.70, max_confidence=conf, bbox_norm=bbox). If no phone found: PhoneDetectionOutput() defaults.

Handle None gracefully: if face_result is None, return PerceptionBundle with all defaults. If face_result has None fields, skip those.

Also create function: wrap_raw_frame(frame: np.ndarray, frame_id: int, timestamp_ns: int) -> RawFrame
  Simple wrapper to satisfy the RawFrame contract from the pipeline.

CREATE FILE: src/output/audio_alerter_v2.py

play_alert(level: AlertLevel) → makes a sound on Mac.
  URGENT: play 3 rapid beeps
  HIGH: play 1 beep
  Implementation: try subprocess.Popen(['afplay', '/System/Library/Sounds/Ping.aiff']). If fails, print('\a'). Non-blocking.

CREATE FILE: src/output/event_logger.py (PRD §9)

JSON-lines rotating log. Uses Python logging with RotatingFileHandler (50MB max, 5 backups).
  log_alert(alert_cmd: AlertCommand, score: DistractionScore)
  log_state_transition(previous: str, new: str, trigger: str, frame_id: int)
  log_calibration(yaw_offset, pitch_offset, baseline_ear)

Each method writes one JSON line matching the formats in PRD §9.

WRITE TESTS in tests/test_adapters.py:
1. face_result with face_visible=True, known head_pose=(10,5,0), ear_left=0.30, ear_right=0.28 → PerceptionBundle has correct fields
2. face_result with face_visible=False → PerceptionBundle.face.present=False, head_pose_raw=None
3. Phone detected ("cell phone" in object_detections with confidence 0.85) → phone.detected=True, max_confidence=0.85
4. No phone → phone.detected=False
5. face_result is None → returns default PerceptionBundle without crashing
6. Event logger writes valid JSON to a temp file (use tempfile)
7. wrap_raw_frame produces correct RawFrame

Run: pytest tests/test_adapters.py -v
Then: pytest tests/ -v

Git: git add -A && git commit -m "phase 7a: adapters and output — all tests passing"
```

---

## PHASE 7B: WIRE THE FULL PIPELINE

**What this does:** Connects everything into the main loop. This is integration — tested manually on webcam.

**Claude Code prompt — copy this exactly:**
```
This is Phase 7B: wiring the full pipeline.

Before writing any code, read these files to understand their interfaces:
- src/pipeline/adapters.py (convert_to_perception_bundle, wrap_raw_frame)
- src/logic/signal_processor.py (process method signature)
- src/logic/temporal_engine.py (process method signature)
- src/logic/scoring_engine.py (score method signature)
- src/logic/alert_state_machine.py (update method signature)
- src/logic/calibration.py (feed_frame, properties)
- src/pipeline/frame_source.py (read method)
- src/detection/face_detector.py (detect method)
- src/detection/object_detector.py (detect method)

CREATE FILE: src/pipeline/pipeline_manager_v2.py

Main loop. Startup:
1. Load config (existing config_loader)
2. Open webcam (existing FrameSource)
3. Init FaceDetector (existing)
4. Init ObjectDetector (existing)
5. Init SignalProcessor
6. Init TemporalEngine
7. Init ScoringEngine
8. Init AlertStateMachine
9. Init Calibration
10. Init AudioAlerterV2, EventLogger
11. frame_id = 0

Main loop (every frame):
1. Read frame from webcam. If None → exit.
2. frame_id += 1, timestamp_ns = time.monotonic_ns()
3. face_result = face_detector.detect(frame)
4. object_detections = object_detector.detect(frame)
5. bundle = convert_to_perception_bundle(face_result, object_detections, frame_id, timestamp_ns)
6. IF calibration not complete:
   - If face_result.face_visible and face_result.head_pose is not None:
     pitch, yaw, roll = face_result.head_pose
     mean_ear = (face_result.ear_left + face_result.ear_right) / 2.0 if face_result.ear_left else 0.0
     done = calibration.feed_frame(yaw, pitch, mean_ear, face_result.face_visible)
     if done:
       signal_processor.set_neutral_offsets(calibration.neutral_yaw_offset, calibration.neutral_pitch_offset)
       signal_processor.set_ear_baseline(calibration.baseline_ear, calibration.close_threshold)
       event_logger.log_calibration(calibration.neutral_yaw_offset, calibration.neutral_pitch_offset, calibration.baseline_ear)
   - Show "CALIBRATING" text on frame via cv2.putText
   - Show frame with cv2.imshow, check for 'q' key
   - Continue to next frame (skip scoring)
7. signal_frame = signal_processor.process(bundle)
8. temporal_features = temporal_engine.process(signal_frame)
9. distraction_score = scoring_engine.score(temporal_features)
10. alert_cmd = alert_state_machine.update(distraction_score, signal_frame.signals_valid)
11. If alert_cmd: audio_alerter.play_alert(alert_cmd.level), event_logger.log_alert(alert_cmd, distraction_score)
12. Draw debug overlay on frame:
    - Top-left: "NOMINAL" / "COOLDOWN" / "DEGRADED" in green/yellow/red
    - Head pose values, EAR, on_road status
    - Duration timers (gaze_secs, head_secs, phone_secs)
    - Composite score bar with threshold line
    - Active classes if any
    - "ALERT!" flash if alert fired in last 1 second
    - During calibration: countdown and sample count
13. cv2.imshow, check 'q' key → exit

Handle exceptions: wrap the per-frame logic in try/except. On exception, log it and continue to next frame.

UPDATE src/main.py:
  Replace the stub from Phase 0. Import PipelineManagerV2. Wire CLI args (--source, --no-display, --log-level). Call pipeline.run().

DO NOT write automated tests for this phase. Test manually:
  python src/main.py --log-level DEBUG

Verify:
1. Webcam opens
2. "CALIBRATING" shows for ~5 seconds
3. After calibration, "NOMINAL" shows
4. No crashes for 30+ seconds of normal use
5. Head pose values update in real-time on overlay
6. Press 'q' → clean exit

Git: git add -A && git commit -m "phase 7b: full pipeline running on webcam"
```

---

## PHASE 8: MANUAL DETECTION TESTING

**Claude Code prompt — copy this exactly:**
```
This is Phase 8. The pipeline should be running. Help me test each detection type.

Run: python src/main.py --log-level DEBUG

I will perform these tests while watching the overlay and DEBUG output:

TEST 1 — Gaze D-A: Look at screen 10s, then turn head left ~20° for 3s.
Expected: gaze off-road detected, D-A alert fires after ~2s.

TEST 2 — Head D-B: Turn head sharply right (>30°) for 2s.
Expected: D-B alert fires after ~1.5s.

TEST 3 — Drowsiness D-C: Close eyes for 5+ seconds.
Expected: PERCLOS rises, D-C alert fires.

TEST 4 — Phone D-D: Hold phone up to camera for 2s.
Expected: D-D alert fires after ~1s.

TEST 5 — Cooldown: Trigger alert, continue behavior.
Expected: no repeat during cooldown.

TEST 6 — Recovery: Return to normal after alert.
Expected: state returns to NOMINAL.

If any test FAILS, debug by checking:
- Calibration offsets (should be small, <10°)
- Kalman filter (if angles seem sluggish, R might be too high)
- Phone confidence (EfficientDet may need PHONE_CONFIDENCE_THRESHOLD lowered to 0.50 in config_prd.py)
- PERCLOS (check that baseline_EAR * 0.20 is the closure check, not baseline * 0.75)
- D-A not triggering separately from D-B: this is expected for MVP since gaze = head pose. D-A triggers at 15° (moderate turn), D-B at 30° (severe turn).
```

---

## PHASE 9: INTEGRATION TESTS AND FINAL CLEANUP

**Claude Code prompt — copy this exactly:**
```
This is Phase 9. Create automated integration tests and verify the full suite.

CREATE FILE: tests/test_pipeline_integration.py

Helper function: make_signal_frame(**overrides) → returns a SignalFrame with sensible defaults (face_present=True, signals_valid=True, all angles at 0, EAR at 0.28, phone not detected) and any fields overridden.

Integration tests that feed synthetic data through signal_processor → temporal_engine → scoring_engine → alert_state_machine. No camera or models needed.

Tests:
1. 90 frames of gaze off-road → alert fires around frame 60 (2.0 seconds at 30fps)
2. 45 frames phone detected (conf=0.85) → alert fires around frame 30 (1.0 second)
3. 100 frames normal driving → no alerts, no breaches
4. Mixed: 60 frames gaze off + 60 frames head turned → both in active_classes
5. Recovery: 90 frames distracted → alert → 60 frames normal → back to normal
6. Cooldown: D-A alert fires, immediately send more D-A breaches → no second alert within 8s
7. Phone override: D-A in cooldown, then phone breach → phone alert fires (P-01)
8. DEGRADED: 60 frames signals_valid=False → no alerts. 30 frames valid → recovery.

Run: pytest tests/ -v --tb=short

ALL tests must pass. Zero failures. Fix any issues.

Git:
git add -A && git commit -m "phase 9: integration tests — full suite passing"
```

---

## DONE

Working MVP on Mac webcam with: all 4 detection types, PRD scoring formula, 5-state alert system, calibration, audio alerts, JSON logging, debug display, full test suite.

### Upgrade to device:
1. Swap MediaPipe → custom models: change adapters.py + detection/. Remove MVP-ONLY fields from PerceptionBundle.
2. Add threading: change pipeline_manager_v2.py only.
3. Add RKNN: change model wrappers only.
4. Add speed: change temporal_engine speed input only.
5. Add thermal/watchdog: add trigger inputs to alert_state_machine.
6. Add calibration persistence: change calibration.py only.

No rewrites. Just upgrades.
