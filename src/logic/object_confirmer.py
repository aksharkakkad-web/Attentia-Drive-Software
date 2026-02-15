"""N-of-M temporal confirmation for detected objects.

Confirms an object as present only if it has been detected in at least N of
the last M frames. This filters out transient false positives. Maintains
independent tracking per object class.

Phase: 1-2 (active).
"""

import logging
from collections import deque
from typing import Deque, Dict, List

from src.config_loader import ObjectConfirmationConfig
from src.detection.object_detector import ObjectDetection

logger = logging.getLogger(__name__)


class ObjectTemporalConfirmer:
    """N-of-M temporal confirmation for object detections.

    For each target object class, maintains a sliding window of M frames
    tracking whether the object was detected. An object is confirmed as
    present only if it appeared in at least N of the last M frames.

    Args:
        config: ObjectConfirmationConfig with n and m values.
    """

    def __init__(self, config: ObjectConfirmationConfig) -> None:
        self._n = config.n
        self._m = config.m
        self._buffers: Dict[str, Deque[int]] = {}

    def update(self, detections: List[ObjectDetection]) -> Dict[str, bool]:
        """Update confirmation state with new detections for this frame.

        For each tracked class, appends 1 (present) or 0 (absent) to its
        sliding window buffer.

        Args:
            detections: List of ObjectDetection from the current frame.

        Returns:
            Dict mapping class_name to confirmed (True/False).
        """
        detected_classes = {d.class_name for d in detections}

        all_classes = set(self._buffers.keys()) | detected_classes
        result: Dict[str, bool] = {}

        for class_name in all_classes:
            if class_name not in self._buffers:
                self._buffers[class_name] = deque(maxlen=self._m)

            present = 1 if class_name in detected_classes else 0
            self._buffers[class_name].append(present)

            count = sum(self._buffers[class_name])
            confirmed = count >= self._n
            result[class_name] = confirmed

        return result

    def get_confirmed_objects(self) -> List[str]:
        """Get the list of currently confirmed object class names.

        Returns:
            List of class names that meet the N-of-M threshold.
        """
        confirmed = []
        for class_name, buffer in self._buffers.items():
            if sum(buffer) >= self._n:
                confirmed.append(class_name)
        return confirmed

    def reset(self) -> None:
        """Clear all tracking buffers."""
        self._buffers.clear()
