"""Frame record dataclass holding all intermediate results for one frame.

Aggregates the outputs of all pipeline stages for a single frame, used
for display, logging, and telemetry.

Phase: 0 (cleaned up — logic fields removed, replaced in Phase 7B).
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

from src.detection.drowsiness import DrowsinessResult
from src.detection.face_detector import FaceResult
from src.detection.object_detector import ObjectDetection


@dataclass
class FrameRecord:
    """All intermediate results for a single pipeline frame."""

    frame_number: int = 0
    timestamp: float = 0.0
    raw_frame: Optional[np.ndarray] = None
    object_detections: List[ObjectDetection] = field(default_factory=list)
    face_result: Optional[FaceResult] = None
    drowsiness_result: Optional[DrowsinessResult] = None
    smoothed_probability: float = 0.0
    inference_time_ms: float = 0.0
    pipeline_time_ms: float = 0.0

    # MVP-ONLY: These fields are placeholders until the new logic layers are wired
    # in Phase 7B. They will be replaced with typed contracts from contracts.py.
    classifier_result: Optional[Any] = None
    driver_state: Optional[Any] = None
    assessment: Optional[Any] = None
    alert: Optional[Any] = None
