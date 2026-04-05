"""Duration timer — stopwatch for continuous distraction conditions.

Tracks how long a condition has been continuously active.
Resets to zero on the first inactive tick.

PRD §FR-3.1
"""


class DurationTimer:
    """Stopwatch that accumulates time while a condition is active.

    Args: None — stateless except for the accumulated duration.
    """

    def __init__(self) -> None:
        self._duration: float = 0.0

    def tick(self, active: bool, delta_seconds: float) -> None:
        """Advance the timer by one frame.

        Args:
            active: Whether the tracked condition is active this frame.
            delta_seconds: Elapsed time since last frame in seconds.
                           Clamped to [0.0, 1.0] to prevent runaway accumulation.
        """
        delta = max(0.0, min(1.0, delta_seconds))
        if active:
            self._duration += delta
        else:
            self._duration = 0.0

    def reset(self) -> None:
        """Explicitly reset duration to zero."""
        self._duration = 0.0

    @property
    def current_duration(self) -> float:
        """Accumulated seconds the condition has been continuously active."""
        return self._duration
