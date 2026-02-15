# Attentia Drive

Real-time distracted driving detection using on-device computer vision. Privacy-first: all processing is local, no cloud, no video upload.

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

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
 Classifier  ObjectDet  FaceDet(stub)
 (P(dist))   (phone,    (landmarks,
              cup, etc)   gaze, EAR)
 |             |          |
 v             v          v
 EMA         N-of-M    Drowsiness(stub)
 Smoother    Confirmer
 |             |          |
 v             v          v
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
| `face_detector`      | Enable/disable (Phase 3 stub)                        |
| `drowsiness`         | Enable/disable (Phase 3 stub)                        |
| `speed_monitor`      | Enable/disable, speed threshold (Phase 4 stub)       |

## Models

Models are **not included**. Place `.tflite` files in the `models/` directory:

- `models/classifier.tflite` — Binary distraction classifier
- `models/object_detector.tflite` — SSD object detector (COCO-compatible)

The system runs without models (all detectors return neutral results, pipeline shows ATTENTIVE state).

## Testing

```bash
pytest tests/ -v
```

All tests run without a camera, model files, or display. They test the logic pipeline using synthetic data.

## Phase Roadmap

| Phase   | Status      | Features                                                |
|---------|-------------|---------------------------------------------------------|
| Phase 1 | Implemented | Classifier, EMA smoothing, hysteresis state machine     |
| Phase 2 | Implemented | Object detection, N-of-M confirmation, signal fusion    |
| Phase 3 | Stubbed     | MediaPipe face mesh, gaze tracking, drowsiness (PERCLOS)|
| Phase 4 | Stubbed     | GPS/OBD-II speed gating, graduated severity levels      |
| Phase 5 | Planned     | Audio alerts, hardware integration, field testing       |
