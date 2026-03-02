"""Alert manager for sustained distraction detection.

Tracks distraction history over a sliding window and fires alerts when the
driver has been distracted for a sustained period. Implements cooldown to
prevent alert fatigue.

Phase: 1-2 (active).
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from src.config_loader import AlertManagerConfig
from src.logic.distraction_reasoner import DistractionAssessment
from src.logic.hysteresis import DriverState

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """An alert fired by the alert manager."""

    level: DriverState
    triggers: List[str] = field(default_factory=list)
    confidence: float = 0.0
    timestamp: float = 0.0


class AlertManager:
    """Manages sustained distraction detection and alert firing.

    Maintains a sliding window of boolean distraction states. Fires an alert
    when the ratio of distracted frames exceeds the sustained_ratio threshold
    and enough time has passed since the last alert (cooldown).

    MONITORING frames do NOT count as distracted in the sustained window.

    Args:
        config: AlertManagerConfig with sustained_frames, sustained_ratio,
                and cooldown_seconds.
    """

    def __init__(self, config: AlertManagerConfig) -> None:
        self._config = config
        self._history: Deque[bool] = deque(maxlen=config.sustained_frames)
        self._last_alert_time: Optional[float] = None

    def update(self, assessment: DistractionAssessment) -> Optional[Alert]:
        """Update the alert manager with a new frame assessment.

        Appends the distraction state to the history window. MONITORING frames
        are recorded as False (not distracted). Checks if sustained distraction
        criteria are met and cooldown has elapsed.

        Args:
            assessment: The DistractionAssessment for the current frame.

        Returns:
            Alert if sustained distraction threshold is met and cooldown elapsed,
            otherwise None.
        """
        is_distracted = assessment.is_distracted
        self._history.append(is_distracted)

        if len(self._history) < self._config.sustained_frames:
            return None

        distracted_count = sum(self._history)
        ratio = distracted_count / len(self._history)

        if ratio < self._config.sustained_ratio:
            return None

        now = time.time()
        if self._last_alert_time is not None:
            elapsed = now - self._last_alert_time
            if elapsed < self._config.cooldown_seconds:
                return None

        self._last_alert_time = now

        alert = Alert(
            level=assessment.state,
            triggers=list(assessment.active_triggers),
            confidence=assessment.confidence,
            timestamp=now,
        )

        logger.warning(
            "ALERT fired: level=%s triggers=%s confidence=%.2f",
            alert.level.name,
            alert.triggers,
            alert.confidence,
        )

        return alert

    def reset(self) -> None:
        """Reset the alert manager state."""
        self._history.clear()
        self._last_alert_time = None
