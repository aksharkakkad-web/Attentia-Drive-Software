"""All threshold constants for the Attentia Drive detection pipeline.

Every number used in detection logic comes from here. No magic numbers in logic code.
Each constant is commented with its PRD section reference.

PRD section: §19 — Configuration Reference

MVP OVERRIDE: CALIBRATION_DURATION_S = 5.0 (PRD says 10.0).
EAR baseline is collected during this same 5s window, not a separate 30s window.
"""

# ─── MODEL PATHS (RKNN FORMAT — v2.0.0) ──────────────────────────────────────
# PRD §19 — Model paths for production RKNN models on RK3568.
# MVP uses MediaPipe + EfficientDet TFLite instead; paths kept for reference.
BLAZEFACE_MODEL_PATH = 'models/blazeface.rknn'          # PRD §19
PFLD_MODEL_PATH      = 'models/pfld_98pt.rknn'          # PRD §19
GAZE_MODEL_PATH      = 'models/gaze_mobilenetv3_lstm.rknn'  # PRD §19
YOLO_MODEL_PATH      = 'models/yolov8n_phone.rknn'      # PRD §19

# MVP model paths (MediaPipe + EfficientDet TFLite — macOS dev target)
FACE_LANDMARKER_MODEL_PATH = 'models/face_landmarker.task'
EFFICIENTDET_MODEL_PATH    = 'models/efficientdet_lite0.tflite'
YOLO_PHONE_MODEL_PATH      = 'models/yolov8n_phone.onnx'        # PRD §19 — YOLOv8-nano ONNX, single-class phone

# ─── CAMERA / ISP (IMX219 + RK3568 — v2.0.0) ─────────────────────────────────
# PRD §19
V4L2_DEVICE          = '/dev/video0'  # PRD §19
CAPTURE_WIDTH        = 1280           # PRD §19
CAPTURE_HEIGHT       = 720            # PRD §19
CAPTURE_FPS          = 30             # PRD §19
V4L2_PIXEL_FORMAT    = 'NV12'         # PRD §19 — ISP output; VideoSource converts to BGR24
LOW_LIGHT_THRESHOLD  = 30             # PRD §19 — Mean pixel intensity (0-255)
OVEREXPOSE_THRESHOLD = 240            # PRD §19 — Mean pixel intensity (0-255)

# ─── RKNN RUNTIME ─────────────────────────────────────────────────────────────
# PRD §19
RKNN_TARGET_PLATFORM = 'rk3568'    # PRD §19
RKNN_TOOLKIT_VERSION = '2.0.0b0'   # PRD §19 — Must match BSP. Pin this.

# ─── CONFIDENCE GATES ────────────────────────────────────────────────────────
# PRD §19
FACE_CONFIDENCE_GATE     = 0.60  # PRD §19
LANDMARK_CONFIDENCE_GATE = 0.65  # PRD §19

# ─── ROAD ZONE ───────────────────────────────────────────────────────────────
# PRD §19 — Head/gaze angles (degrees) defining the "on-road" window.
ROAD_ZONE_YAW_MIN   = -15.0  # PRD §19
ROAD_ZONE_YAW_MAX   = +15.0  # PRD §19
ROAD_ZONE_PITCH_MIN = -10.0  # PRD §19
ROAD_ZONE_PITCH_MAX = +5.0   # PRD §19 / §2.2

# ─── SPEED ZONES ─────────────────────────────────────────────────────────────
# PRD §19
V_MIN_MPS              = 1.4    # PRD §19 — 5 km/h minimum alerting speed
V_HIGHWAY_MPS          = 13.9   # PRD §19 — 50 km/h highway threshold
HIGHWAY_SCORE_MODIFIER = 1.4    # PRD §19 — Score multiplier at highway speed
SPEED_STALE_THRESHOLD_S = 2.0   # PRD §19 — Declare speed stale if no update in 2s

# ─── SPEED SIGNAL SOURCE (v2.0.0) ────────────────────────────────────────────
# PRD §19
SPEED_SOURCE_PRIORITY = ['OBD2', 'CAN', 'GPS', 'NONE']  # PRD §19 — Try in order
OBD2_PORT      = '/dev/ttyUSB0'  # PRD §19
OBD2_BAUDRATE  = 38400           # PRD §19
OBD2_POLL_HZ   = 10              # PRD §19
CAN_INTERFACE  = 'can0'          # PRD §19
CAN_PGN_SPEED  = 0xFEF1          # PRD §19 — J1939 PGN 65265 Wheel Speed
GPS_PORT       = '/dev/ttyS3'    # PRD §19

# ─── DURATION THRESHOLDS ─────────────────────────────────────────────────────
# PRD §19 — How long a condition must be sustained to trigger an alert.
T_GAZE_SECONDS        = 2.0  # PRD §19
T_HEAD_SECONDS        = 1.5  # PRD §19
T_PHONE_SECONDS       = 1.0  # PRD §19
T_FACE_ABSENT_SECONDS = 5.0  # PRD §19

# ─── HEAD POSE THRESHOLDS ────────────────────────────────────────────────────
# PRD §19
HEAD_YAW_THRESHOLD_DEG   = 30.0  # PRD §19
HEAD_PITCH_THRESHOLD_DEG = 20.0  # PRD §19
PNP_REPROJECTION_ERR_MAX = 8.0   # PRD §19 — pixels; HeadPose.valid=False if exceeded

# ─── EAR / PERCLOS ───────────────────────────────────────────────────────────
# PRD §19
EAR_BASELINE_POPULATION_DEFAULT = 0.28   # PRD §5.2 — Population mean open-eye EAR; used until per-session calibration completes
EAR_DEFAULT_CLOSE_THRESHOLD     = 0.21   # PRD §19 — Used before calibration completes
EAR_CALIBRATION_MULTIPLIER      = 0.75   # PRD §19 — close_threshold = baseline * this
EAR_CALIBRATION_DURATION_S      = 30.0   # PRD §19 — Original EAR baseline collection window
PERCLOS_WINDOW_FRAMES       = 60     # PRD §19 — 2.0s at 30fps
PERCLOS_CLOSURE_FRACTION    = 0.80   # PRD §19 — EAR < close_threshold * this = "closed"
PERCLOS_ALERT_THRESHOLD     = 0.15   # PRD §19
PERCLOS_MIN_VALID_FRAMES    = 30     # PRD §19

# ─── KALMAN FILTER (v2.0.0) ──────────────────────────────────────────────────
# PRD §19 — Parameters for 1D constant-velocity Kalman filter on head pose and gaze.
KALMAN_PROCESS_NOISE_Q     = 0.01  # PRD §19
KALMAN_MEASUREMENT_NOISE_R = 4.0   # PRD §19
KALMAN_INITIAL_COVARIANCE  = 1.0   # PRD §19

# ─── GAZE TRANSFORM ──────────────────────────────────────────────────────────
# PRD §19 — Head-to-world gaze coupling coefficients.
GAZE_HEAD_COUPLING_ALPHA = 0.7  # PRD §19 — Yaw coupling
GAZE_HEAD_COUPLING_BETA  = 0.7  # PRD §19 — Pitch coupling

# ─── LSTM GAZE MODEL (v2.0.0) ────────────────────────────────────────────────
# PRD §19
GAZE_INPUT_RESOLUTION    = 112  # PRD §19 — MobileNetV3 compatible (down from 224)
GAZE_TEMPORAL_FRAMES     = 8    # PRD §19 — LSTM lookback window
LSTM_RESET_ABSENT_FRAMES = 10   # PRD §19 — Reset hidden state after face absent N frames

# ─── PHONE DETECTION ─────────────────────────────────────────────────────────
# PRD §19
PHONE_CONFIDENCE_THRESHOLD     = 0.70  # PRD §19
YOLO_INPUT_RESOLUTION          = 320   # PRD §19
YOLO_INPUT_RESOLUTION_THROTTLE = 256   # PRD §19 — Reduced when thermal throttle active

# ─── SCORING WEIGHTS ─────────────────────────────────────────────────────────
# PRD §19 — Must sum to 1.0. Validated in test_contracts.py.
WEIGHT_GAZE              = 0.45  # PRD §19
WEIGHT_HEAD              = 0.30  # PRD §19
WEIGHT_PERCLOS           = 0.20  # PRD §19
WEIGHT_BLINK             = 0.05  # PRD §19
COMPOSITE_ALERT_THRESHOLD = 0.55  # PRD §19

# ─── BLINK DETECTION ─────────────────────────────────────────────────────────
# PRD §19
BLINK_MIN_FRAMES          = 2     # PRD §19 — 67ms at 30fps
BLINK_MAX_FRAMES          = 10    # PRD §19 — 333ms at 30fps
BLINK_RATE_NORMAL_LOW_HZ  = 0.13  # PRD §19 — 8 blinks/min
BLINK_RATE_NORMAL_HIGH_HZ = 0.50  # PRD §19 — 30 blinks/min

# ─── TEMPORAL BUFFER ─────────────────────────────────────────────────────────
# PRD §19
CIRCULAR_BUFFER_SIZE  = 120  # PRD §19 — 4.0s at 30fps
FEATURE_WINDOW_FRAMES = 60   # PRD §19 — 2.0s at 30fps

# ─── ALERT COOLDOWNS (seconds) ───────────────────────────────────────────────
# PRD §19 — Per-alert-type cooldown durations.
COOLDOWN_VISUAL      = 8.0   # PRD §19
COOLDOWN_HEAD        = 8.0   # PRD §19
COOLDOWN_DROWSINESS  = 12.0  # PRD §19
COOLDOWN_PHONE       = 5.0   # PRD §19
COOLDOWN_FACE_ABSENT = 10.0  # PRD §19
COOLDOWN_COMPOSITE   = 8.0   # PRD §19

# ─── DEGRADED STATE ──────────────────────────────────────────────────────────
# PRD §19
DEGRADED_TRIGGER_FRAMES   = 60  # PRD §19 — 2.0s of invalid frames
DEGRADED_RECOVERY_FRAMES  = 30  # PRD §19 — 1.0s of valid frames to recover
DEGRADED_TRIGGER_LIGHTING = 90  # PRD §19 — Extended trigger during AE convergence

# ─── THREAD / SYNC (v2.0.0) ──────────────────────────────────────────────────
# PRD §19
PHONE_THREAD_TIMEOUT_MS = 5  # PRD §19 — Wait for T-2 before using stale phone result
FRAME_QUEUE_DEPTH       = 2  # PRD §19 — T-0 → T-1/T-2 queue depth (drop oldest on overflow)

# ─── WATCHDOG (v2.0.0) ───────────────────────────────────────────────────────
# PRD §19
WATCHDOG_TIMEOUT_S   = 2.0   # PRD §19
WATCHDOG_HEARTBEAT_S = 0.5   # PRD §19

# ─── THERMAL MONITOR (v2.0.0) ────────────────────────────────────────────────
# PRD §19
THERMAL_WARN_TEMP_C      = 80    # PRD §19
THERMAL_CRITICAL_TEMP_C  = 90    # PRD §19
THERMAL_MONITOR_PATH     = '/sys/class/thermal/thermal_zone0/temp'  # PRD §19
THERMAL_CHECK_INTERVAL_S = 5.0   # PRD §19

# ─── CALIBRATION (v2.0.0) ────────────────────────────────────────────────────
# PRD §19
# MVP OVERRIDE: 5.0s instead of PRD's 10.0s. EAR baseline uses same window.
CALIBRATION_DURATION_S        = 5.0    # PRD says 10.0 — MVP override: faster startup
CALIBRATION_MIN_VALID_FRAMES  = 270    # PRD §19 — 90% of 300 frames
CALIBRATION_MAX_POSE_STD_DEG  = 5.0    # PRD §19 — Reject calibration if std dev >= 5°
NEUTRAL_POSE_FILE             = 'calibration/session_state.json'  # PRD §19
CALIBRATION_REQUIRED_ON_VIN_CHANGE = True  # PRD §19

# ─── LOGGING ─────────────────────────────────────────────────────────────────
# PRD §19
LOG_DIR          = 'logs/'        # PRD §19
LOG_MAX_BYTES    = 52_428_800     # PRD §19 — 50 MB
LOG_BACKUP_COUNT = 5              # PRD §19
