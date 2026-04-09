"""ThermalMonitor interface — CPU temperature abstraction.

Mac has no meaningful thermal concern at desk; NoopThermalMonitor always
returns None. Pi 5 VcgencmdThermalMonitor is deferred to the bring-up phase
(see docs/DEVIATIONS.md DEV-004).

Contract: docs/INTERFACES.md §ThermalMonitor.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ThermalMonitor:
    """Abstract CPU temperature source.

    Implementations must never raise into the pipeline loop. Return None
    (or False from is_available) on any failure mode.
    """

    def open(self) -> bool:
        raise NotImplementedError

    def get_temp_c(self) -> Optional[float]:
        """Current CPU temperature in °C, or None if unavailable.

        Must be O(1) from the caller's perspective; cache internally if the
        backend is slow.
        """
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def is_available(self) -> bool:
        raise NotImplementedError


class NoopThermalMonitor(ThermalMonitor):
    """Mac no-op implementation. Always returns None."""

    def open(self) -> bool:
        logger.debug("NoopThermalMonitor: open (no-op)")
        return True

    def get_temp_c(self) -> Optional[float]:
        return None

    def close(self) -> None:
        return None

    def is_available(self) -> bool:
        return False
