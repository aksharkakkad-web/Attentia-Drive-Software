# Architecture — Code to PRD Mapping

## Data Layer
| File | PRD Section | Purpose |
|------|-------------|---------|
| `src/contracts.py` | §4.1–§4.6 | All typed messages between layers |
| `src/config_prd.py` | §19 | All thresholds, weights, and constants |

## Perception (Layer 0–1)
| File | PRD Section | Purpose |
|------|-------------|---------|
| `src/pipeline/frame_source.py` | §8.1 | Webcam/video capture |
| `src/detection/face_detector.py` | §8.2 | MediaPipe face landmarks, head pose, EAR, gaze |
| `src/detection/object_detector.py` | §8.2 | Phone/object detection (EfficientDet or YOLOv8) |
| `src/pipeline/adapters.py` | — | Converts MediaPipe/detector output → PerceptionBundle |

## Signal Processing (Layer 2)
| File | PRD Section | Purpose |
|------|-------------|---------|
| `src/logic/signal_processor.py` | §5.1–§5.7 | Kalman filtering, pose correction, gaze transform, EAR processing |
| `src/logic/kalman_filter.py` | §5.6 | 1D constant-velocity Kalman filter |

## Temporal (Layer 3)
| File | PRD Section | Purpose |
|------|-------------|---------|
| `src/logic/temporal_engine.py` | §8.4 | Orchestrates all temporal logic |
| `src/logic/duration_timer.py` | §6.2, §7 | Stopwatch per distraction condition |
| `src/logic/perclos_calculator.py` | §5.4 | Sliding window eye closure percentage |
| `src/logic/blink_detector.py` | §5.5 | Blink rate and anomaly scoring |

## Scoring (Layer 4)
| File | PRD Section | Purpose |
|------|-------------|---------|
| `src/logic/scoring_engine.py` | §6 | Weighted composite score + threshold checks |

## Alert (Layer 5)
| File | PRD Section | Purpose |
|------|-------------|---------|
| `src/logic/alert_state_machine.py` | §7 | 5-state FSM with per-type cooldowns |

## Calibration
| File | PRD Section | Purpose |
|------|-------------|---------|
| `src/logic/calibration.py` | §23 | Startup neutral pose calibration |

## Output (Layer 6)
| File | PRD Section | Purpose |
|------|-------------|---------|
| `src/output/audio_alerter_v2.py` | §8.7 | System beep on alert |
| `src/output/event_logger.py` | §9 | Structured JSON event log |
| `src/output/display.py` | — | OpenCV debug overlay |

## Pipeline
| File | PRD Section | Purpose |
|------|-------------|---------|
| `src/pipeline/pipeline_manager_v2.py` | §3 | Main loop wiring all layers |
| `src/main.py` | — | Entry point, CLI args |
