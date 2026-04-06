"""Blink detector — EAR transition detector and blink rate anomaly scorer.

A blink is an EAR drop below close_threshold that lasts BLINK_MIN_FRAMES
to BLINK_MAX_FRAMES frames inclusive. Closures longer than BLINK_MAX_FRAMES
are treated as extended closure (drowsiness), NOT a blink.

Blink rate is computed over a rolling 30-second window.
The anomaly score maps rate → [0.0, 1.0]:
  rate < BLINK_RATE_NORMAL_LOW_HZ  → drowsy (interpolate from 0.13→0 :: score 0→1)
  rate > BLINK_RATE_NORMAL_HIGH_HZ → fatigue (interpolate from 0.50→∞ :: clamp 1.0)
  within normal range              → 0.0

PRD §5.5
"""

from src.config_prd import (
    BLINK_MAX_FRAMES,
    BLINK_MIN_FRAMES,
    BLINK_RATE_NORMAL_HIGH_HZ,
    BLINK_RATE_NORMAL_LOW_HZ,
    CAPTURE_FPS,
)

_BLINK_RATE_WINDOW_S = 30.0  # Rolling window length for blink rate (seconds)


class BlinkDetector:
    """Tracks EAR transitions to count blinks and compute blink rate anomaly score.

    PRD §5.5.
    """

    def __init__(self) -> None:
        self._prev_was_above: bool = False  # True once we've seen a frame above threshold
        self._in_blink: bool = False        # True during an active above→below run
        self._below_count: int = 0          # Frames below threshold in current run

        self._blink_timestamps: list = []
        self._current_time_s: float = 0.0
        self._frame_delta: float = 1.0 / CAPTURE_FPS  # Default delta if not provided

    def update(self, mean_ear: float, close_threshold: float, delta_seconds: float = None) -> bool:
        """Process one frame of EAR data.

        A blink requires a full above→below→above EAR transition.
        Frames that begin already below threshold do NOT start a blink counter.

        Args:
            mean_ear: Mean EAR value for this frame.
            close_threshold: Threshold below which eyes are considered closed.
            delta_seconds: Elapsed time since last frame. Defaults to 1/CAPTURE_FPS.

        Returns:
            True if a complete blink was detected on this frame (eye just re-opened).
        """
        if delta_seconds is None:
            delta_seconds = self._frame_delta
        self._current_time_s += max(0.0, min(1.0, delta_seconds))

        blink_detected = False
        is_below = mean_ear < close_threshold

        if is_below:
            if not self._in_blink and self._prev_was_above:
                # Genuine above→below crossover: start counting
                self._in_blink = True
                self._below_count = 1
            elif self._in_blink:
                self._below_count += 1
            # else: started already below without a prior above frame — ignore
        else:
            if self._in_blink:
                # below→above transition: evaluate whether this was a blink
                if BLINK_MIN_FRAMES <= self._below_count <= BLINK_MAX_FRAMES:
                    blink_detected = True
                    self._blink_timestamps.append(self._current_time_s)
                self._in_blink = False
                self._below_count = 0
            self._prev_was_above = True

        # Prune blinks outside the rolling 30s window
        cutoff = self._current_time_s - _BLINK_RATE_WINDOW_S
        self._blink_timestamps = [t for t in self._blink_timestamps if t > cutoff]

        return blink_detected

    @property
    def blink_rate_hz(self) -> float:
        """Blink rate in Hz over the rolling 30-second window."""
        if self._current_time_s <= 0.0:
            return 0.0
        window = min(self._current_time_s, _BLINK_RATE_WINDOW_S)
        return len(self._blink_timestamps) / window

    @property
    def blink_rate_score(self) -> float:
        """Anomaly score [0.0, 1.0] for the current blink rate. PRD §5.5."""
        return self.score_for_rate(self.blink_rate_hz)

    @staticmethod
    def score_for_rate(rate_hz: float) -> float:
        """Compute blink rate anomaly score for a given rate. PRD §5.5.

        Args:
            rate_hz: Blink rate in Hz.

        Returns:
            Anomaly score [0.0, 1.0]:
              0.0 → normal range
              approaching 1.0 → either very low (drowsy) or very high (fatigue)
        """
        if rate_hz < BLINK_RATE_NORMAL_LOW_HZ:
            # Drowsy signal: linearly interpolate from (0.13 → score 0.0) to (0.0 → score 1.0)
            score = 1.0 - (rate_hz / BLINK_RATE_NORMAL_LOW_HZ)
            return max(0.0, min(1.0, score))
        elif rate_hz > BLINK_RATE_NORMAL_HIGH_HZ:
            # Fatigue signal: linearly interpolate from (0.50 → score 0.0) up
            # Use a reasonable upper bound of 2× normal high for max score
            score = (rate_hz - BLINK_RATE_NORMAL_HIGH_HZ) / BLINK_RATE_NORMAL_HIGH_HZ
            return max(0.0, min(1.0, score))
        else:
            return 0.0
