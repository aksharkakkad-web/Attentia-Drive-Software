"""Rolling FPS counter for pipeline performance measurement.

Uses a sliding window of frame timestamps to compute the current
frames-per-second rate.

Phase: 1-2 (active).
"""

import time
from collections import deque
from typing import Deque


class FPSCounter:
    """Rolling window FPS measurement.

    Tracks the timestamps of the last N frames and computes FPS
    as the number of frames divided by the time span.

    Args:
        window_size: Number of frames to include in the rolling window.
    """

    def __init__(self, window_size: int = 30) -> None:
        self._timestamps: Deque[float] = deque(maxlen=window_size)

    def tick(self) -> None:
        """Record a frame timestamp. Call once per frame."""
        self._timestamps.append(time.time())

    @property
    def fps(self) -> float:
        """Current FPS based on the rolling window.

        Returns:
            Frames per second as a float. Returns 0.0 if fewer than 2 frames.
        """
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed
