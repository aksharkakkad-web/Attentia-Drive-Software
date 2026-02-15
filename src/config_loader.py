"""Configuration loader for Attentia Drive.

Reads config.yaml and validates all fields into typed Python dataclasses.
One dataclass per config section, one root AppConfig that holds them all.
Raises clear errors on missing or invalid fields. Provides sensible defaults
if optional fields are missing.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import yaml

logger = logging.getLogger(__name__)


@dataclass
class FrameSourceConfig:
    """Configuration for the frame source (webcam or video file)."""

    type: str = "webcam"
    webcam_index: int = 0
    video_path: str = ""
    target_fps: int = 30
    resolution: Tuple[int, int] = (640, 480)


@dataclass
class ClassifierConfig:
    """Configuration for the distraction classifier model."""

    enabled: bool = True
    model_path: str = "models/classifier.tflite"
    input_size: Tuple[int, int] = (224, 224)
    confidence_threshold: float = 0.7


@dataclass
class ObjectDetectorConfig:
    """Configuration for the TFLite object detector."""

    enabled: bool = True
    model_path: str = "models/object_detector.tflite"
    input_size: Tuple[int, int] = (300, 300)
    confidence_threshold: float = 0.5
    target_classes: List[str] = field(
        default_factory=lambda: ["cell phone", "cup", "bottle", "book", "laptop"]
    )
    frame_skip: int = 3


@dataclass
class TemporalSmoothingConfig:
    """Configuration for EMA temporal smoothing."""

    enabled: bool = True
    alpha: float = 0.3
    window_size: int = 10


@dataclass
class HysteresisConfig:
    """Configuration for the hysteresis state machine thresholds."""

    enter_distracted_threshold: float = 0.75
    leave_distracted_threshold: float = 0.55
    enter_monitoring_threshold: float = 0.40
    leave_monitoring_threshold: float = 0.35


@dataclass
class ObjectConfirmationConfig:
    """Configuration for N-of-M object temporal confirmation."""

    enabled: bool = True
    n: int = 15
    m: int = 30


@dataclass
class AlertManagerConfig:
    """Configuration for the alert manager."""

    sustained_frames: int = 90
    sustained_ratio: float = 0.80
    cooldown_seconds: float = 5.0


@dataclass
class DisplayConfig:
    """Configuration for the OpenCV debug display window."""

    enabled: bool = True
    show_bboxes: bool = True
    show_ema_plot: bool = True
    show_fps: bool = True
    show_state: bool = True
    window_name: str = "Attentia Drive — Debug"


@dataclass
class TelemetryConfig:
    """Configuration for telemetry/logging."""

    enabled: bool = True
    log_file: str = "logs/telemetry.csv"
    log_interval_frames: int = 1


@dataclass
class FaceDetectorConfig:
    """Configuration for the face detector (MediaPipe Face Mesh)."""

    enabled: bool = False
    model_type: str = "mediapipe"
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    refine_landmarks: bool = True
    yaw_threshold: float = 25.0
    pitch_down_threshold: float = 15.0
    pitch_up_threshold: float = 10.0
    ear_threshold: float = 0.21


@dataclass
class DrowsinessConfig:
    """Configuration for the drowsiness detector."""

    enabled: bool = False
    ear_threshold: float = 0.21
    perclos_window_seconds: float = 60.0
    perclos_threshold: float = 0.15
    mar_threshold: float = 0.6
    yawn_min_frames: int = 10


@dataclass
class SpeedMonitorConfig:
    """Configuration for the speed monitor (Phase 4 stub)."""

    enabled: bool = False
    speed_threshold_mph: float = 15.0


@dataclass
class AppConfig:
    """Root configuration holding all subsystem configs."""

    frame_source: FrameSourceConfig = field(default_factory=FrameSourceConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    object_detector: ObjectDetectorConfig = field(default_factory=ObjectDetectorConfig)
    temporal_smoothing: TemporalSmoothingConfig = field(
        default_factory=TemporalSmoothingConfig
    )
    hysteresis: HysteresisConfig = field(default_factory=HysteresisConfig)
    object_confirmation: ObjectConfirmationConfig = field(
        default_factory=ObjectConfirmationConfig
    )
    alert_manager: AlertManagerConfig = field(default_factory=AlertManagerConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    face_detector: FaceDetectorConfig = field(default_factory=FaceDetectorConfig)
    drowsiness: DrowsinessConfig = field(default_factory=DrowsinessConfig)
    speed_monitor: SpeedMonitorConfig = field(default_factory=SpeedMonitorConfig)


def _parse_tuple(value: object, expected_len: int = 2) -> Tuple[int, ...]:
    """Parse a list from YAML into a tuple of ints."""
    if isinstance(value, (list, tuple)):
        if len(value) != expected_len:
            raise ValueError(
                f"Expected {expected_len} elements, got {len(value)}: {value}"
            )
        return tuple(int(v) for v in value)
    raise TypeError(f"Expected list/tuple, got {type(value).__name__}: {value}")


def _build_frame_source_config(data: dict) -> FrameSourceConfig:
    """Build FrameSourceConfig from a raw dict."""
    config = FrameSourceConfig()
    if not data:
        return config

    config.type = str(data.get("type", config.type))
    if config.type not in ("webcam", "video"):
        raise ValueError(
            f"frame_source.type must be 'webcam' or 'video', got '{config.type}'"
        )

    config.webcam_index = int(data.get("webcam_index", config.webcam_index))
    config.video_path = str(data.get("video_path", config.video_path))
    config.target_fps = int(data.get("target_fps", config.target_fps))

    if "resolution" in data:
        config.resolution = _parse_tuple(data["resolution"], 2)

    return config


def _build_classifier_config(data: dict) -> ClassifierConfig:
    """Build ClassifierConfig from a raw dict."""
    config = ClassifierConfig()
    if not data:
        return config

    config.enabled = bool(data.get("enabled", config.enabled))
    config.model_path = str(data.get("model_path", config.model_path))
    config.confidence_threshold = float(
        data.get("confidence_threshold", config.confidence_threshold)
    )

    if "input_size" in data:
        config.input_size = _parse_tuple(data["input_size"], 2)

    return config


def _build_object_detector_config(data: dict) -> ObjectDetectorConfig:
    """Build ObjectDetectorConfig from a raw dict."""
    config = ObjectDetectorConfig()
    if not data:
        return config

    config.enabled = bool(data.get("enabled", config.enabled))
    config.model_path = str(data.get("model_path", config.model_path))
    config.confidence_threshold = float(
        data.get("confidence_threshold", config.confidence_threshold)
    )
    config.frame_skip = int(data.get("frame_skip", config.frame_skip))

    if "input_size" in data:
        config.input_size = _parse_tuple(data["input_size"], 2)

    if "target_classes" in data:
        config.target_classes = [str(c) for c in data["target_classes"]]

    return config


def _build_temporal_smoothing_config(data: dict) -> TemporalSmoothingConfig:
    """Build TemporalSmoothingConfig from a raw dict."""
    config = TemporalSmoothingConfig()
    if not data:
        return config

    config.enabled = bool(data.get("enabled", config.enabled))
    config.alpha = float(data.get("alpha", config.alpha))
    config.window_size = int(data.get("window_size", config.window_size))

    if not 0.0 < config.alpha <= 1.0:
        raise ValueError(
            f"temporal_smoothing.alpha must be in (0, 1], got {config.alpha}"
        )

    return config


def _build_hysteresis_config(data: dict) -> HysteresisConfig:
    """Build HysteresisConfig from a raw dict."""
    config = HysteresisConfig()
    if not data:
        return config

    config.enter_distracted_threshold = float(
        data.get("enter_distracted_threshold", config.enter_distracted_threshold)
    )
    config.leave_distracted_threshold = float(
        data.get("leave_distracted_threshold", config.leave_distracted_threshold)
    )
    config.enter_monitoring_threshold = float(
        data.get("enter_monitoring_threshold", config.enter_monitoring_threshold)
    )
    config.leave_monitoring_threshold = float(
        data.get("leave_monitoring_threshold", config.leave_monitoring_threshold)
    )

    if config.leave_distracted_threshold >= config.enter_distracted_threshold:
        raise ValueError(
            "hysteresis: leave_distracted_threshold must be < enter_distracted_threshold"
        )
    if config.leave_monitoring_threshold >= config.enter_monitoring_threshold:
        raise ValueError(
            "hysteresis: leave_monitoring_threshold must be < enter_monitoring_threshold"
        )

    return config


def _build_object_confirmation_config(data: dict) -> ObjectConfirmationConfig:
    """Build ObjectConfirmationConfig from a raw dict."""
    config = ObjectConfirmationConfig()
    if not data:
        return config

    config.enabled = bool(data.get("enabled", config.enabled))
    config.n = int(data.get("n", config.n))
    config.m = int(data.get("m", config.m))

    if config.n > config.m:
        raise ValueError(
            f"object_confirmation: n ({config.n}) must be <= m ({config.m})"
        )

    return config


def _build_alert_manager_config(data: dict) -> AlertManagerConfig:
    """Build AlertManagerConfig from a raw dict."""
    config = AlertManagerConfig()
    if not data:
        return config

    config.sustained_frames = int(data.get("sustained_frames", config.sustained_frames))
    config.sustained_ratio = float(data.get("sustained_ratio", config.sustained_ratio))
    config.cooldown_seconds = float(
        data.get("cooldown_seconds", config.cooldown_seconds)
    )

    return config


def _build_display_config(data: dict) -> DisplayConfig:
    """Build DisplayConfig from a raw dict."""
    config = DisplayConfig()
    if not data:
        return config

    config.enabled = bool(data.get("enabled", config.enabled))
    config.show_bboxes = bool(data.get("show_bboxes", config.show_bboxes))
    config.show_ema_plot = bool(data.get("show_ema_plot", config.show_ema_plot))
    config.show_fps = bool(data.get("show_fps", config.show_fps))
    config.show_state = bool(data.get("show_state", config.show_state))
    config.window_name = str(data.get("window_name", config.window_name))

    return config


def _build_telemetry_config(data: dict) -> TelemetryConfig:
    """Build TelemetryConfig from a raw dict."""
    config = TelemetryConfig()
    if not data:
        return config

    config.enabled = bool(data.get("enabled", config.enabled))
    config.log_file = str(data.get("log_file", config.log_file))
    config.log_interval_frames = int(
        data.get("log_interval_frames", config.log_interval_frames)
    )

    return config


def _build_face_detector_config(data: dict) -> FaceDetectorConfig:
    """Build FaceDetectorConfig from a raw dict."""
    config = FaceDetectorConfig()
    if not data:
        return config

    config.enabled = bool(data.get("enabled", config.enabled))
    config.model_type = str(data.get("model_type", config.model_type))
    config.min_detection_confidence = float(
        data.get("min_detection_confidence", config.min_detection_confidence)
    )
    config.min_tracking_confidence = float(
        data.get("min_tracking_confidence", config.min_tracking_confidence)
    )
    config.refine_landmarks = bool(
        data.get("refine_landmarks", config.refine_landmarks)
    )
    config.yaw_threshold = float(data.get("yaw_threshold", config.yaw_threshold))
    config.pitch_down_threshold = float(
        data.get("pitch_down_threshold", config.pitch_down_threshold)
    )
    config.pitch_up_threshold = float(
        data.get("pitch_up_threshold", config.pitch_up_threshold)
    )
    config.ear_threshold = float(data.get("ear_threshold", config.ear_threshold))

    return config


def _build_drowsiness_config(data: dict) -> DrowsinessConfig:
    """Build DrowsinessConfig from a raw dict."""
    config = DrowsinessConfig()
    if not data:
        return config

    config.enabled = bool(data.get("enabled", config.enabled))
    config.ear_threshold = float(data.get("ear_threshold", config.ear_threshold))
    config.perclos_window_seconds = float(
        data.get("perclos_window_seconds", config.perclos_window_seconds)
    )
    config.perclos_threshold = float(
        data.get("perclos_threshold", config.perclos_threshold)
    )
    config.mar_threshold = float(data.get("mar_threshold", config.mar_threshold))
    config.yawn_min_frames = int(data.get("yawn_min_frames", config.yawn_min_frames))

    return config


def _build_speed_monitor_config(data: dict) -> SpeedMonitorConfig:
    """Build SpeedMonitorConfig from a raw dict."""
    config = SpeedMonitorConfig()
    if not data:
        return config

    config.enabled = bool(data.get("enabled", config.enabled))
    config.speed_threshold_mph = float(
        data.get("speed_threshold_mph", config.speed_threshold_mph)
    )

    return config


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Load and validate configuration from a YAML file.

    Args:
        config_path: Path to the config.yaml file.

    Returns:
        A fully validated AppConfig dataclass instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If any config values are invalid.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(raw)}")

    logger.info("Loading configuration from %s", path.resolve())

    app_config = AppConfig(
        frame_source=_build_frame_source_config(raw.get("frame_source", {})),
        classifier=_build_classifier_config(raw.get("classifier", {})),
        object_detector=_build_object_detector_config(raw.get("object_detector", {})),
        temporal_smoothing=_build_temporal_smoothing_config(
            raw.get("temporal_smoothing", {})
        ),
        hysteresis=_build_hysteresis_config(raw.get("hysteresis", {})),
        object_confirmation=_build_object_confirmation_config(
            raw.get("object_confirmation", {})
        ),
        alert_manager=_build_alert_manager_config(raw.get("alert_manager", {})),
        display=_build_display_config(raw.get("display", {})),
        telemetry=_build_telemetry_config(raw.get("telemetry", {})),
        face_detector=_build_face_detector_config(raw.get("face_detector", {})),
        drowsiness=_build_drowsiness_config(raw.get("drowsiness", {})),
        speed_monitor=_build_speed_monitor_config(raw.get("speed_monitor", {})),
    )

    logger.info("Configuration loaded successfully")
    return app_config
