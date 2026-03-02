"""Exponential Moving Average (EMA) temporal smoother.

Smooths raw probability signals to reduce noise and prevent rapid fluctuations.
One instance per signal being smoothed. Stateful: maintains the previous
smoothed value across calls.

Phase: 1-2 (active).
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TemporalSmoother:
    """EMA smoother for a single probability signal.

    Applies the formula: smoothed = alpha * raw + (1 - alpha) * prev_smoothed.
    The first call initializes the smoothed value to the raw input.

    Args:
        alpha: EMA weight in (0, 1]. Higher = more responsive but noisier.
    """

    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._smoothed: Optional[float] = None

    def update(self, raw_value: float) -> float:
        """Apply EMA smoothing to a new raw value.

        Args:
            raw_value: The raw (unsmoothed) value for this frame.

        Returns:
            The smoothed value after applying EMA.
        """
        if self._smoothed is None:
            self._smoothed = raw_value
        else:
            self._smoothed = self._alpha * raw_value + (1.0 - self._alpha) * self._smoothed
        return self._smoothed

    def reset(self) -> None:
        """Clear the smoother state. Next update() will re-initialize."""
        self._smoothed = None

    @property
    def current_value(self) -> float:
        """The current smoothed value, or 0.0 if not yet initialized."""
        return self._smoothed if self._smoothed is not None else 0.0
