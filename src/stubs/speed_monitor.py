"""Speed monitor stub (Phase 4).

# Phase 4: Integrate GPS/OBD-II. Gate alerts on speed > threshold.

Currently returns simulated values. Interface ready for GPS/OBD integration.
"""

import logging

logger = logging.getLogger(__name__)


class SpeedMonitor:
    """Vehicle speed monitor (STUB).

    TODO Phase 4 implementation:
    - Integrate with GPS module (serial/USB) for speed data
    - Integrate with OBD-II adapter for vehicle speed PID
    - Gate alerts on speed > speed_threshold_mph
    - Handle GPS signal loss / OBD disconnection gracefully

    # Phase 4: Integrate GPS/OBD-II. Gate alerts on speed > threshold.
    """

    def __init__(self) -> None:
        logger.info("SpeedMonitor: Initialized (Phase 4 stub)")

    def get_speed_mph(self) -> float:
        """Get the current vehicle speed in MPH.

        Phase 4: Will return real GPS/OBD speed.

        Returns:
            Speed in MPH. Currently always 0.0.
        """
        return 0.0

    def is_driving(self) -> bool:
        """Check if the vehicle is currently in motion.

        Phase 4: Will compare speed against threshold.

        Returns:
            True if driving. Currently always True (always active for Phase 1-2).
        """
        return True
