"""ImuSource interface — inertial measurement abstraction.

Mac has no IMU; StubImuSource returns IMUReading(valid=False) forever so
downstream logic can consume IMU readings from day one without branching on
platform. Pi 5 Mpu6050ImuSource is deferred to bring-up (docs/DEVIATIONS.md
DEV-003).

SignalProcessor wiring is also deferred; see DEV-006 — the pipeline currently
reads and discards so the interface exists and is exercised end-to-end.

Contract: docs/INTERFACES.md §ImuSource.
"""

from __future__ import annotations

import logging
import time

from src.contracts import IMUReading

logger = logging.getLogger(__name__)


class ImuSource:
    """Abstract inertial measurement source."""

    def open(self) -> bool:
        raise NotImplementedError

    def read(self) -> IMUReading:
        """Return the latest IMU reading. Must not block, must not raise.

        Stub and failed-hardware cases return IMUReading(valid=False).
        """
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def is_available(self) -> bool:
        raise NotImplementedError


class StubImuSource(ImuSource):
    """Mac stub. Always returns IMUReading(valid=False).

    Exists so the pipeline loop can call read() unconditionally without a
    platform branch. When the real Pi implementation lands, the pipeline
    layer swaps the constructor; no call-site change.
    """

    def open(self) -> bool:
        logger.debug("StubImuSource: open (no hardware)")
        return True

    def read(self) -> IMUReading:
        return IMUReading(timestamp_ms=time.monotonic() * 1000.0, valid=False)

    def close(self) -> None:
        return None

    def is_available(self) -> bool:
        return False
