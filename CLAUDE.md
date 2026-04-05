# Attentia Drive — Distraction Detection Engine (MVP)

Aftermarket driver safety device. Detects distracted/drowsy driving using on-device computer vision. Privacy-first — all processing is local, no cloud.

## Tech Stack
- Python 3.10+, macOS (development target)
- MediaPipe FaceLandmarker (face detection, landmarks, head pose, EAR, iris tracking)
- EfficientDet-Lite0 or YOLOv8-nano (phone/object detection, TFLite/ONNX)
- OpenCV (camera, display)
- NumPy, PyYAML
- pytest (testing)

## Commands
python src/main.py                        # Run with webcam
python src/main.py --source video.mp4     # Run with video file
python src/main.py --no-display           # Headless mode
pytest tests/ -v                          # Run all tests
pytest tests/test_<module>.py -v          # Run tests for one module

## Architecture
Six-layer unidirectional pipeline. Data flows one direction only: Layer 0 → 1 → 2 → 3 → 4 → 5.

Camera → Perception (MediaPipe + phone detector) → Signal Processor → Temporal Engine → Scoring Engine → Alert State Machine → Audio + Display

- **Perception**: MediaPipe gives face/landmarks/head pose/EAR/gaze. Phone detector runs independently.
- **Signal Processor** (src/logic/signal_processor.py): Kalman-filters head pose, applies calibration offsets, computes gaze on/off road, extracts clean signals. → Outputs SignalFrame.
- **Temporal Engine** (src/logic/temporal_engine.py): Duration timers per distraction type, PERCLOS sliding window, blink rate tracking. → Outputs TemporalFeatures.
- **Scoring Engine** (src/logic/scoring_engine.py): Weighted composite score (gaze 45%, head 30%, drowsiness 20%, blink 5%). Checks 6 independent alert thresholds. → Outputs DistractionScore.
- **Alert State Machine** (src/logic/alert_state_machine.py): 5 states — NOMINAL / PRE_ALERT / ALERTING / COOLDOWN / DEGRADED. Per-alert-type cooldowns. Phone alerts override all cooldowns.

All data contracts are in src/contracts.py. All thresholds are in src/config_prd.py.

## Key References
- docs/PRD_v2.md — Full product spec. The source of truth for all thresholds, formulas, and behavior.
- docs/BUILD_PLAN.md — Phase-by-phase build instructions.
- docs/ARCHITECTURE.md — Maps each code file to its PRD section.

## Critical Rules

1. **No magic numbers.** Every threshold, weight, and constant comes from src/config_prd.py. Never put a number directly in logic code.

2. **Test before moving on.** Every phase has tests. Run them. If they fail, fix them in that phase. Do not proceed to the next phase with failing tests.

3. **Layers don't reach into each other.** Each layer only knows about its input (the message from the previous layer) and its output. No layer imports another layer's internals.

4. **Plan before coding.** If a task touches 3 or more files AND the task is NOT a phase prompt from docs/BUILD_PLAN.md, write a plan first and wait for user confirmation. Phase prompts already contain the plan — execute them directly.

5. **MediaPipe is temporary.** Layer 1 uses MediaPipe now. Later it will use custom models (BlazeFace + PFLD + gaze model). Layers 2-5 must NEVER depend on MediaPipe-specific data. They only consume the PerceptionBundle from contracts.py.

6. **Run tests after every change.** After creating or modifying any file, immediately run the relevant test file. Do not move on to the next file until tests pass.

7. **Handle None everywhere.** Every function that receives data from a previous layer must handle the case where that data is None, missing, or invalid. Return safe defaults, never crash.

8. **One thing at a time.** Create one file, write its tests, run the tests, confirm they pass. Then move to the next file. Do not batch-create multiple files before testing any of them.

9. **Read before writing.** Before modifying an existing file, read it first. Before creating a file that imports from another module, read that module first to understand its actual interface.

10. **No debug loops.** If a test fails 3 times with the same error, stop and explain the problem to the user instead of trying more fixes. Something fundamental is wrong.

11. **Never leave imports broken.** At the end of every step within a phase, every file in src/ must import cleanly. Run python -c "import src.<module>" to verify. If deleting a module breaks other files, fix those files in the same step.

12. **Mark MVP contract extensions.** Any field added to a PRD contract that is not in the PRD must have a # MVP-ONLY comment explaining what it is and that it should be removed when custom models replace MediaPipe. Never silently add fields.

13. **Ask, don't assume.** If the prompt is ambiguous, if a design decision isn't specified, if there are multiple reasonable ways to implement something, or if you're unsure about any detail — STOP and ask the user before writing code. Never fill gaps with your own assumptions. If the PRD and build plan don't specify it, ask.

## Git Workflow
- Commit to main after each phase passes all tests
- Commit message format: phase N: description — all tests passing
- Run pytest tests/ -v before every commit — never commit with failing tests
- If something breaks badly, git stash to save work and investigate

## Past Mistakes — Do Not Repeat
- **DO NOT** build everything at once and test at the end. Build one piece, test it, confirm it works, then move on.
- **DO NOT** use RKNN, V4L2, or any RK3568-specific code. This is a Mac MVP.
- **DO NOT** implement threading. Single-threaded pipeline only.
- **DO NOT** implement ThermalMonitor or WatchdogManager. Skip for MVP.
- **DO NOT** assume models are working. If perception returns no face, downstream layers must handle it gracefully.
- **DO NOT** create files without tests. Every new .py file in src/logic/ must have a corresponding test file.
- **DO NOT** modify files from previous phases without running ALL tests afterward to check for regressions.
- **DO NOT** refactor or "improve" working code unless explicitly asked. If it passes tests, leave it alone.