"""Severity level mapping stub (Phase 4).

Defines a 5-level severity enum and a mapping function from DriverState
to SeverityLevel. Phase 1-2 uses a simplified 3-level mapping.

# Phase 4: Implement full graduated severity mapping based on trigger type,
# confidence, duration, and multi-modal agreement.
"""

import logging
from enum import Enum, auto
from typing import List

from src.logic.hysteresis import DriverState

logger = logging.getLogger(__name__)


class SeverityLevel(Enum):
    """5-level severity for driver distraction.

    Phase 1-2 uses: ATTENTIVE, MINOR, MODERATE.
    Phase 4 will use all five levels.
    """

    ATTENTIVE = auto()
    MINOR = auto()
    MODERATE = auto()
    SEVERE = auto()
    CRITICAL = auto()


def map_to_severity(
    state: DriverState,
    triggers: List[str],
    confidence: float,
) -> SeverityLevel:
    """Map a DriverState to a SeverityLevel.

    Phase 1-2: Simple mapping ignoring triggers and confidence.
    - ATTENTIVE -> ATTENTIVE
    - MONITORING -> MINOR
    - DISTRACTED -> MODERATE

    # Phase 4: Implement full graduated severity mapping based on trigger type,
    # confidence, duration, and multi-modal agreement. Examples:
    # - DISTRACTED + phone_use + confidence > 0.9 + duration > 5s -> CRITICAL
    # - DISTRACTED + eating_drinking + confidence > 0.8 -> SEVERE
    # - MONITORING + any object confirmed -> MODERATE

    Args:
        state: Current DriverState.
        triggers: List of active trigger names (unused in Phase 1-2).
        confidence: Smoothed distraction confidence (unused in Phase 1-2).

    Returns:
        SeverityLevel corresponding to the driver state.
    """
    if state == DriverState.ATTENTIVE:
        return SeverityLevel.ATTENTIVE
    elif state == DriverState.MONITORING:
        return SeverityLevel.MINOR
    elif state == DriverState.DISTRACTED:
        return SeverityLevel.MODERATE
    else:
        return SeverityLevel.ATTENTIVE
