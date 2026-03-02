"""Distraction reasoner that fuses all detection signals into a final assessment.

Combines the hysteresis state, confirmed objects, face detection, and drowsiness
signals into a single DistractionAssessment with active triggers and confidence.

Phase: 1-2 (active, with stubs for Phase 3 signals).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List

from src.detection.drowsiness import DrowsinessResult
from src.detection.face_detector import FaceResult
from src.logic.hysteresis import DriverState

logger = logging.getLogger(__name__)


@dataclass
class DistractionAssessment:
    """Final fused distraction assessment for a single frame."""

    state: DriverState = DriverState.ATTENTIVE
    is_distracted: bool = False
    is_monitoring: bool = False
    confidence: float = 0.0
    active_triggers: List[str] = field(default_factory=list)
    confirmed_objects: List[str] = field(default_factory=list)
    face_visible: bool = False
    drowsiness: DrowsinessResult = field(default_factory=DrowsinessResult)
    timestamp: float = 0.0


class DistractionReasoner:
    """Fuses all detection signals into a final DistractionAssessment.

    Trigger logic (Phase 1-2):
    - "distracted_posture": DISTRACTED state with no confirmed objects.
    - "phone_use": "cell phone" in confirmed objects.
    - "eating_drinking": "cup" or "bottle" in confirmed objects.
    - "reading": "book" in confirmed objects.
    - "laptop_use": "laptop" in confirmed objects.

    Future triggers (Phase 3, return False):
    - "gaze_away", "eyes_closed", "drowsy"

    Multi-modal confirmation (Phase 2):
    - If DISTRACTED AND at least one object confirmed -> boost confidence by 0.1.

    Args:
        face_detector_enabled: Whether the face detector is active.
    """

    def __init__(self, face_detector_enabled: bool = False) -> None:
        self._face_detector_enabled = face_detector_enabled

    def assess(
        self,
        driver_state: DriverState,
        smoothed_probability: float,
        confirmed_objects: Dict[str, bool],
        face_result: FaceResult,
        drowsiness_result: DrowsinessResult,
    ) -> DistractionAssessment:
        """Produce a fused DistractionAssessment from all signals.

        Args:
            driver_state: Current state from the hysteresis machine.
            smoothed_probability: EMA-smoothed distraction probability.
            confirmed_objects: Dict of class_name -> confirmed (bool).
            face_result: Face detection result (neutral when stubbed).
            drowsiness_result: Drowsiness detection result (neutral when stubbed).

        Returns:
            DistractionAssessment with fused state, triggers, and confidence.
        """
        confirmed_list = [
            name for name, confirmed in confirmed_objects.items() if confirmed
        ]

        if self._face_detector_enabled and not face_result.face_visible:
            logger.debug(
                "Face not visible with face_detector enabled; "
                "forcing ATTENTIVE state for this frame."
            )
            return DistractionAssessment(
                state=DriverState.ATTENTIVE,
                is_distracted=False,
                is_monitoring=False,
                confidence=0.0,
                active_triggers=[],
                confirmed_objects=confirmed_list,
                face_visible=False,
                drowsiness=drowsiness_result,
                timestamp=time.time(),
            )

        triggers = self._identify_triggers(
            driver_state, confirmed_list, face_result, drowsiness_result
        )

        confidence = smoothed_probability
        if (
            driver_state == DriverState.DISTRACTED
            and len(confirmed_list) > 0
        ):
            confidence = min(1.0, confidence + 0.1)

        # Gaze-based override: if face detector confirms looking away/down,
        # force DISTRACTED regardless of classifier output.
        effective_state = driver_state
        if "gaze_away" in triggers and face_result.face_visible:
            effective_state = DriverState.DISTRACTED
            confidence = max(confidence, 0.85)

        return DistractionAssessment(
            state=effective_state,
            is_distracted=(effective_state == DriverState.DISTRACTED),
            is_monitoring=(effective_state == DriverState.MONITORING),
            confidence=confidence,
            active_triggers=triggers,
            confirmed_objects=confirmed_list,
            face_visible=face_result.face_visible,
            drowsiness=drowsiness_result,
            timestamp=time.time(),
        )

    def _identify_triggers(
        self,
        state: DriverState,
        confirmed_objects: List[str],
        face_result: FaceResult,
        drowsiness_result: DrowsinessResult,
    ) -> List[str]:
        """Identify active distraction triggers.

        Args:
            state: Current driver state.
            confirmed_objects: List of confirmed object class names.
            face_result: Face detection result.
            drowsiness_result: Drowsiness detection result.

        Returns:
            List of trigger name strings.
        """
        triggers: List[str] = []

        if "cell phone" in confirmed_objects:
            triggers.append("phone_use")

        if "cup" in confirmed_objects or "bottle" in confirmed_objects:
            triggers.append("eating_drinking")

        if "book" in confirmed_objects:
            triggers.append("reading")

        if "laptop" in confirmed_objects:
            triggers.append("laptop_use")

        # Face-based triggers (Phase 3)
        if face_result.face_visible and face_result.head_pose is not None:
            pitch, yaw, _roll = face_result.head_pose
            # pitch > 0 = looking up, pitch < 0 = looking down (phone/lap)
            # Three cases:
            #   1. Pure sideways:        abs(yaw) > 40
            #   2. Straight down:        pitch < -10  (sensitive)
            #   3. Diagonal down-left/right (~45°): pitch < -8 AND abs(yaw) > 15
            looking_away = (
                abs(yaw) > 40.0
                or pitch > 25.0
                or pitch < -10.0
                or (pitch < -8.0 and abs(yaw) > 15.0)
            )
            if looking_away:
                triggers.append("gaze_away")

        if face_result.face_visible and face_result.ear_left is not None:
            avg_ear = (face_result.ear_left + face_result.ear_right) / 2.0
            if avg_ear < 0.18:
                triggers.append("eyes_closed")

        if drowsiness_result.is_drowsy:
            triggers.append("drowsy")

        if state == DriverState.DISTRACTED and len(confirmed_objects) == 0:
            triggers.append("distracted_posture")

        return triggers
