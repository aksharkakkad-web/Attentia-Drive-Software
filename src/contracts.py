"""Typed data contracts for all inter-layer messages.

Each dataclass represents the message passed between two pipeline layers.
All fields have default values so layers can construct safe empty instances.
Fields not in the PRD are marked # MVP-ONLY.

PRD sections: §4.1–§4.6
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

import numpy as np


# ─── Sensor contracts (platform-agnostic inputs) ────────────────────────────

@dataclass
class IMUReading:
    """Inertial measurement reading from the IMU interface.

    MVP-ONLY on Mac: StubImuSource always returns valid=False so downstream
    logic treats the reading as absent. On Pi, Mpu6050ImuSource populates
    all fields. Consumed by ImuSource implementations only; SignalProcessor
    wiring is deferred (see DEV-006).

    See docs/INTERFACES.md §ImuSource for the contract.
    """

    accel_x: float = 0.0         # m/s², body frame
    accel_y: float = 0.0
    accel_z: float = 0.0
    gyro_x: float = 0.0          # °/s, body frame
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    temp_c: float = 0.0          # onboard die temperature
    timestamp_ms: float = 0.0    # monotonic clock
    valid: bool = False          # False on Mac stub or I2C error


# ─── Layer 0 → Layer 1 ──────────────────────────────────────────────────────

@dataclass
class RawFrame:
    """Single camera frame from the video source.

    PRD §4.1 — Layer 0 → Layer 1.
    """

    timestamp_ns: int = 0           # V4L2 buffer timestamp (or Python monotonic fallback)
    frame_id: int = 0               # Monotonically increasing, resets on source restart
    width: int = 640                # Pixels
    height: int = 480               # Pixels
    channels: int = 3               # Always 3 (BGR24)
    data: Optional[np.ndarray] = None  # Shape: (height, width, 3), dtype uint8, BGR
    source_type: str = 'webcam'     # 'imx219_v4l2' | 'webcam' | 'file'


# ─── Layer 1 → Layer 2 ──────────────────────────────────────────────────────

@dataclass
class FaceDetection:
    """Face detection output from BlazeFace (or MediaPipe FaceDetector).

    PRD §4.2 — part of PerceptionBundle.
    """

    present: bool = False
    confidence: float = 0.0         # [0.0, 1.0]
    bbox_norm: tuple = (0, 0, 0, 0)  # (x, y, w, h) normalized to [0, 1]
    face_size_px: int = 0           # Width of bounding box in pixels


@dataclass
class LandmarkOutput:
    """Facial landmark output from PFLD (or MediaPipe FaceLandmarker).

    PRD §4.2 — part of PerceptionBundle.
    NOTE: PRD specifies 98×2 for PFLD. MediaPipe gives 478×3.
    The contract uses Optional[np.ndarray] so both shapes work.
    """

    landmarks: Optional[np.ndarray] = None  # Shape: (98, 2) or (478, 3), normalized [0, 1]
    confidence: float = 0.0                  # [0.0, 1.0]
    pose_valid: bool = False                 # False if face rotation > reliable limit


@dataclass
class GazeOutput:
    """Gaze direction output from the gaze model.

    PRD §4.2 — part of PerceptionBundle.
    """

    left_eye_yaw: float = 0.0       # Degrees (camera space, pre-transform)
    left_eye_pitch: float = 0.0
    right_eye_yaw: float = 0.0
    right_eye_pitch: float = 0.0
    combined_yaw: float = 0.0       # Weighted mean
    combined_pitch: float = 0.0
    confidence: float = 0.0         # [0.0, 1.0]
    valid: bool = False


@dataclass
class PhoneDetectionOutput:
    """Phone/object detection output from YOLO (or EfficientDet).

    PRD §4.2 — part of PerceptionBundle.
    """

    detected: bool = False
    max_confidence: float = 0.0     # [0.0, 1.0], 0.0 if not detected
    bbox_norm: Optional[tuple] = None  # (x, y, w, h) normalized, None if not detected


@dataclass
class PerceptionBundle:
    """All perception outputs for a single frame.

    PRD §4.2 — Layer 1 → Layer 2.
    v2.0.0: Added lstm_hidden_state, lstm_reset_occurred (CHANGE-03),
    phone_result_stale (CHANGE-07).
    """

    timestamp_ns: int = 0
    frame_id: int = 0
    face: FaceDetection = field(default_factory=FaceDetection)
    landmarks: Optional[LandmarkOutput] = None   # None if face not present
    gaze: Optional[GazeOutput] = None            # None if face not present
    phone: PhoneDetectionOutput = field(default_factory=PhoneDetectionOutput)
    phone_result_stale: bool = False             # True if T-2 timed out; last result used
    inference_ms: float = 0.0                    # Total perception inference time
    lstm_hidden_state: Any = None                # Opaque (h_t, c_t) tuple. None on first frame
    lstm_reset_occurred: bool = False            # True if hidden state was reset this frame

    # MVP-ONLY: Pre-computed head pose from MediaPipe. Remove when custom models replace MediaPipe.
    # Layer 2 reads these fields instead of re-deriving from landmarks during MVP.
    head_pose_raw: Optional[tuple] = None  # MVP-ONLY: (pitch, yaw, roll) degrees from MediaPipe
    ear_left: float = 0.0                  # MVP-ONLY: left EAR from MediaPipe
    ear_right: float = 0.0                 # MVP-ONLY: right EAR from MediaPipe


# ─── Layer 2 → Layer 3 ──────────────────────────────────────────────────────

@dataclass
class HeadPose:
    """Kalman-filtered, neutral-pose-corrected head orientation.

    PRD §4.3 — part of SignalFrame.
    v2.0.0 (CHANGE-06, CHANGE-08): All angle values are Kalman-filtered
    and neutral-pose-corrected. Raw values are in debug fields.
    """

    yaw_deg: float = 0.0        # Filtered + corrected. Positive = right.
    pitch_deg: float = 0.0      # Filtered + corrected. Positive = up.
    roll_deg: float = 0.0       # Filtered. Range: [-45, 45]
    valid: bool = False         # False if reprojection error > 8.0 px
    raw_yaw_deg: float = 0.0    # Debug: unfiltered, uncorrected yaw
    raw_pitch_deg: float = 0.0  # Debug: unfiltered, uncorrected pitch
    raw_roll_deg: float = 0.0   # Debug: unfiltered, uncorrected roll


@dataclass
class EyeSignals:
    """Eye aspect ratio signals for drowsiness detection.

    PRD §4.3 — part of SignalFrame.
    """

    left_EAR: float = 0.0              # Eye Aspect Ratio [0.0, ~0.4]
    right_EAR: float = 0.0
    mean_EAR: float = 0.0
    baseline_EAR: float = 0.28         # Per-session calibrated open-eye baseline
    close_threshold: float = 0.21      # baseline_EAR * EAR_CALIBRATION_MULTIPLIER (0.75)
    valid: bool = False
    calibration_complete: bool = False  # True after calibration window of valid data


@dataclass
class GazeWorld:
    """World-space gaze direction after head-coupling transform.

    PRD §4.3 — part of SignalFrame.
    """

    yaw_deg: float = 0.0    # World-space gaze yaw — Kalman-filtered + corrected
    pitch_deg: float = 0.0  # World-space gaze pitch — same
    on_road: bool = True    # True if within ROAD_ZONE
    valid: bool = False


@dataclass
class PhoneSignal:
    """Processed phone detection signal.

    PRD §4.3 — part of SignalFrame.
    """

    detected: bool = False
    confidence: float = 0.0  # [0.0, 1.0]
    stale: bool = False       # True if using T-2 timeout fallback result


@dataclass
class SignalFrame:
    """All processed signals for one frame, ready for temporal analysis.

    PRD §4.3 — Layer 2 → Layer 3.
    v2.0.0 (CHANGE-06): Angles are Kalman-filtered and neutral-corrected.
    """

    timestamp_ns: int = 0
    frame_id: int = 0
    face_present: bool = False
    head_pose: Optional[HeadPose] = None
    eye_signals: Optional[EyeSignals] = None
    gaze_world: Optional[GazeWorld] = None
    phone_signal: PhoneSignal = field(default_factory=PhoneSignal)
    speed_mps: float = 0.0        # From SpeedSource (§22). 0.0 if unavailable.
    speed_stale: bool = False     # True if speed reading > SPEED_STALE_THRESHOLD_S old
    signals_valid: bool = False   # False if any critical signal is unavailable


# ─── Layer 3 → Layer 4 ──────────────────────────────────────────────────────

@dataclass
class TemporalFeatures:
    """Temporal features computed over the rolling window.

    PRD §4.4 — Layer 3 → Layer 4.
    Extensions: perclos_valid (validity flag), face_absent_continuous_secs (for ALT-06).
    """

    timestamp_ns: int = 0
    # F1: Gaze
    gaze_off_road_fraction: float = 0.0   # Fraction of window where gaze was OFF_ROAD
    gaze_continuous_secs: float = 0.0     # Duration of current continuous off-road event
    # F2: Head pose
    head_deviation_mean_deg: float = 0.0  # Euclidean norm of yaw+pitch (corrected angles)
    head_continuous_secs: float = 0.0     # Duration of current continuous head pose breach
    # F3: PERCLOS
    perclos: float = 0.0                  # [0.0, 1.0] — fraction of window eyes closed
    perclos_valid: bool = False           # EXTENSION: False if fewer than MIN_VALID_FRAMES
    # F4: Blink rate
    blink_rate_score: float = 0.0         # [0.0, 1.0], 1.0 = maximally anomalous
    # F5: Phone
    phone_confidence_mean: float = 0.0    # Mean phone confidence in current window
    phone_continuous_secs: float = 0.0    # Duration of current continuous phone detection
    # Extensions
    face_absent_continuous_secs: float = 0.0  # EXTENSION: needed for ALT-06 face-absent alert
    # Context
    speed_zone: str = 'URBAN'             # 'PARKED' | 'URBAN' | 'HIGHWAY'
    speed_modifier: float = 1.0           # 0.0 | 1.0 | 1.4
    frames_valid_in_window: int = 0       # Count of valid frames in buffer
    thermal_throttle_active: bool = False  # True if ThermalMonitor declared throttle state


# ─── Layer 4 → Layer 5 ──────────────────────────────────────────────────────

@dataclass
class DistractionScore:
    """Composite distraction score and per-threshold breach flags.

    PRD §4.5 — Layer 4 → Layer 5.
    Extensions: composite_threshold_breached (ALT-05), face_absent_threshold_breached (ALT-06).
    """

    timestamp_ns: int = 0
    composite_score: float = 0.0          # [0.0, 1.0+] — AFTER speed modifier applied
    # Component scores (pre-modifier)
    component_gaze: float = 0.0           # W1 * F1
    component_head: float = 0.0           # W2 * F2_norm
    component_perclos: float = 0.0        # W3 * F3
    component_blink: float = 0.0          # W4 * F4
    # Threshold breach flags (evaluated independently)
    gaze_threshold_breached: bool = False
    head_threshold_breached: bool = False
    perclos_threshold_breached: bool = False
    phone_threshold_breached: bool = False
    composite_threshold_breached: bool = False  # EXTENSION: ALT-05
    face_absent_threshold_breached: bool = False  # EXTENSION: ALT-06
    active_classes: List[str] = field(default_factory=list)  # e.g. ['D-A', 'D-C']


# ─── Layer 5 → Layer 6 ──────────────────────────────────────────────────────

class AlertLevel(Enum):
    """Alert severity level.

    PRD §4.6.
    """

    LOW = 1     # Advisory — no current use in MVP
    HIGH = 2    # Standard distraction alert
    URGENT = 3  # Phone use alert (highest priority)


class AlertType(Enum):
    """Distraction class that triggered the alert.

    PRD §4.6.
    """

    VISUAL_INATTENTION = 'D-A'
    HEAD_INATTENTION = 'D-B'
    DROWSINESS = 'D-C'
    PHONE_USE = 'D-D'
    FACE_ABSENT = 'FACE'


@dataclass
class AlertCommand:
    """Command to the audio/output layer to fire an alert.

    PRD §4.6 — Layer 5 → Layer 6.
    Extension: active_classes — all active D-classes at time of alert.
    """

    alert_id: str = ''
    timestamp_ns: int = 0
    level: AlertLevel = AlertLevel.HIGH
    alert_type: AlertType = AlertType.VISUAL_INATTENTION
    composite_score: float = 0.0
    suppress_until_ns: int = 0
    active_classes: List[str] = field(default_factory=list)  # EXTENSION
