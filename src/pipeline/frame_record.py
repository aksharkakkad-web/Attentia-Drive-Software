"""Frame record dataclass holding all intermediate results for one frame.

Aggregates the outputs of all pipeline stages for a single frame, used
for display, logging, and telemetry.

Phase: 1-2 (active).
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.detection.classifier import ClassifierResult
from src.detection.drowsiness import DrowsinessResult
from src.detection.face_detector import FaceResult
from src.detection.object_detector import ObjectDetection
from src.logic.alert_manager import Alert
from src.logic.distraction_reasoner import DistractionAssessment
from src.logic.hysteresis import DriverState


@dataclass
class FrameRecord:
    """All intermediate results for a single pipeline frame."""

    frame_number: int = 0
    timestamp: float = 0.0
    raw_frame: Optional[np.ndarray] = None
    classifier_result: Optional[ClassifierResult] = None
    object_detections: List[ObjectDetection] = field(default_factory=list)
    face_result: Optional[FaceResult] = None
    drowsiness_result: Optional[DrowsinessResult] = None
    smoothed_probability: float = 0.0
    driver_state: DriverState = DriverState.ATTENTIVE
    assessment: Optional[DistractionAssessment] = None
    alert: Optional[Alert] = None
    inference_time_ms: float = 0.0
    pipeline_time_ms: float = 0.0
