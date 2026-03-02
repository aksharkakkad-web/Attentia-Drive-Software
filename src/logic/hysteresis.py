"""Hysteresis state machine for driver attention state.

Implements a 3-state machine (ATTENTIVE, MONITORING, DISTRACTED) with
separate enter/leave thresholds to prevent rapid toggling at boundaries.

Transitions:
- ATTENTIVE -> MONITORING: smoothed_prob >= enter_monitoring_threshold
- ATTENTIVE -> DISTRACTED: smoothed_prob >= enter_distracted_threshold
- MONITORING -> DISTRACTED: smoothed_prob >= enter_distracted_threshold
- MONITORING -> ATTENTIVE: smoothed_prob < leave_monitoring_threshold
- DISTRACTED -> MONITORING: smoothed_prob <= leave_distracted_threshold
  (Cannot go directly from DISTRACTED to ATTENTIVE; must pass through MONITORING.)

Phase: 1-2 (active).
"""

import logging
from enum import Enum, auto

from src.config_loader import HysteresisConfig

logger = logging.getLogger(__name__)


class DriverState(Enum):
    """Driver attention state.

    Phase 1-2 uses: ATTENTIVE, MONITORING, DISTRACTED.
    Phase 4 will add graduated severity using: MINOR, MODERATE, SEVERE, CRITICAL.
    """

    ATTENTIVE = auto()
    MONITORING = auto()
    DISTRACTED = auto()
    # Phase 4: graduated severity levels (not used in Phase 1-2 logic)
    MINOR = auto()
    MODERATE = auto()
    SEVERE = auto()
    CRITICAL = auto()


class HysteresisStateMachine:
    """3-state hysteresis machine for driver attention.

    Uses separate enter/leave thresholds to create dead zones that prevent
    rapid state toggling when the probability hovers near a boundary.

    Args:
        config: HysteresisConfig with all threshold values.
    """

    def __init__(self, config: HysteresisConfig) -> None:
        self._config = config
        self._state = DriverState.ATTENTIVE

    def update(self, smoothed_prob: float) -> DriverState:
        """Update the state machine with a new smoothed probability.

        Args:
            smoothed_prob: EMA-smoothed distraction probability in [0, 1].

        Returns:
            The current DriverState after applying transitions.
        """
        if self._state == DriverState.ATTENTIVE:
            if smoothed_prob >= self._config.enter_distracted_threshold:
                self._state = DriverState.DISTRACTED
                logger.debug(
                    "State: ATTENTIVE -> DISTRACTED (prob=%.3f)", smoothed_prob
                )
            elif smoothed_prob >= self._config.enter_monitoring_threshold:
                self._state = DriverState.MONITORING
                logger.debug(
                    "State: ATTENTIVE -> MONITORING (prob=%.3f)", smoothed_prob
                )

        elif self._state == DriverState.MONITORING:
            if smoothed_prob >= self._config.enter_distracted_threshold:
                self._state = DriverState.DISTRACTED
                logger.debug(
                    "State: MONITORING -> DISTRACTED (prob=%.3f)", smoothed_prob
                )
            elif smoothed_prob < self._config.leave_monitoring_threshold:
                self._state = DriverState.ATTENTIVE
                logger.debug(
                    "State: MONITORING -> ATTENTIVE (prob=%.3f)", smoothed_prob
                )

        elif self._state == DriverState.DISTRACTED:
            if smoothed_prob <= self._config.leave_distracted_threshold:
                self._state = DriverState.MONITORING
                logger.debug(
                    "State: DISTRACTED -> MONITORING (prob=%.3f)", smoothed_prob
                )

        return self._state

    @property
    def current_state(self) -> DriverState:
        """The current driver attention state."""
        return self._state

    def reset(self) -> None:
        """Reset the state machine to ATTENTIVE."""
        self._state = DriverState.ATTENTIVE
