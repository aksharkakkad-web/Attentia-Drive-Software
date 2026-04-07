# Attentia Drive

Real-time distracted driving detection using on-device computer vision. Privacy-first: all processing is local, no cloud, no video upload.

## Quick Start

> **Working directory:** All commands below must be run from the `Attentia-Drive-Software/` directory, not the parent repo root.

### Install

```bash
pip install -r requirements.txt
```

**macOS — phone detection (optional):**
Phone detection requires TensorFlow (~500 MB). If you want it enabled, install the macOS requirements instead:

```bash
pip install -r requirements-macos.txt
```

Without this, phone detection is disabled with a warning log on startup and all other features work normally.

### Run with Webcam

```bash
python src/main.py
```

### Run with Video File

```bash
python src/main.py --source path/to/video.mp4
```

### Run Headless (no display window)

```bash
python src/main.py --no-display
```

### Run Tests

```bash
pytest tests/
```

## Architecture

```
 FrameSource (webcam/video)
       |
       v
 +-----+------+----------+
 |             |          |
 Classifier  ObjectDet  FaceDet
 (P(dist))   (phone,    (MediaPipe
              cup, etc)   landmarks)
 |             |          |
 v             v          v
 EMA         N-of-M    Drowsiness
 Smoother    Confirmer  (PERCLOS,
 |             |         blinks,
 v             v         yawns)
 Hysteresis  +-----+------+
 State       |
 Machine     v
 |       DistractionReasoner
 +-----> (fuse signals, triggers)
              |
              v
         AlertManager
         (sustained check, cooldown)
              |
       +------+------+
       |      |      |
       v      v      v
    Display  Audio  Telemetry
    (OpenCV) (beep) (CSV log)
```

## Configuration

All parameters are in `config.yaml`. No code changes needed to tune behavior.

| Section              | Description                                          |
|----------------------|------------------------------------------------------|
| `frame_source`       | Webcam index, video path, resolution, target FPS     |
| `classifier`         | Model path, input size, confidence threshold         |
| `object_detector`    | Model path, target classes, frame skip               |
| `temporal_smoothing` | EMA alpha (responsiveness vs. noise)                 |
| `hysteresis`         | Enter/leave thresholds for state transitions         |
| `object_confirmation`| N-of-M frames for object presence confirmation       |
| `alert_manager`      | Sustained frames, ratio, cooldown between alerts     |
| `display`            | Toggle overlays: bboxes, EMA plot, FPS, state        |
| `telemetry`          | Log file path and logging interval                   |
| `face_detector`      | MediaPipe thresholds: yaw, pitch, EAR                |
| `drowsiness`         | PERCLOS window, EAR/MAR thresholds, yawn detection   |
| `speed_monitor`      | Enable/disable, speed threshold (Phase 4 stub)       |

## Models

Models are **not included**. Place files in the `models/` directory:

- `models/classifier.tflite` — Binary distraction classifier
- `models/efficientdet_lite0.tflite` — EfficientDet-Lite0 object detector
- `models/face_landmarker.task` — MediaPipe FaceLandmarker ([download](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task))

The system runs without models (all detectors return neutral results).

## Testing

```bash
pytest tests/ -v
```

101 tests run without a camera, model files, or display. They test the logic pipeline using synthetic data.

## Phase Roadmap

| Phase   | Status      | Features                                                |
|---------|-------------|---------------------------------------------------------|
| Phase 1 | Complete    | Classifier, EMA smoothing, hysteresis state machine     |
| Phase 2 | Complete    | Object detection, N-of-M confirmation, signal fusion    |
| Phase 3 | Complete    | MediaPipe face mesh, gaze tracking, drowsiness (PERCLOS)|
| Phase 4 | Stubbed     | GPS/OBD-II speed gating, graduated severity levels      |
| Phase 5 | Planned     | Audio alerts, hardware integration, field testing       |
