"""PERCLOS calculator — sliding window eye closure fraction.

PERCLOS = fraction of frames in the window where eyes were closed
          (only counting frames where eye signal was valid).

PRD §5.4
"""

from collections import deque

from src.config_prd import PERCLOS_MIN_VALID_FRAMES, PERCLOS_WINDOW_FRAMES


class PerclosCalculator:
    """Sliding window PERCLOS calculator.

    Window length: PERCLOS_WINDOW_FRAMES (60 frames = 2.0s at 30fps).
    Minimum valid frames before a perclos value is reported: PERCLOS_MIN_VALID_FRAMES (30).

    PRD §5.4.
    """

    def __init__(self) -> None:
        # Each entry is (eyes_closed: bool, frame_valid: bool)
        self._window: deque = deque(maxlen=PERCLOS_WINDOW_FRAMES)

    def update(self, eyes_closed: bool, frame_valid: bool) -> None:
        """Add one frame to the sliding window.

        Args:
            eyes_closed: True if mean_EAR <= baseline_EAR * (1 - PERCLOS_CLOSURE_FRACTION).
                         The caller (TemporalEngine) computes this boolean.
            frame_valid: True if eye signals were valid this frame.
        """
        self._window.append((eyes_closed, frame_valid))

    @property
    def perclos(self) -> float:
        """Fraction of valid frames in the window where eyes were closed.

        Returns 0.0 if fewer than PERCLOS_MIN_VALID_FRAMES valid frames are in the window.
        """
        valid_frames = [(closed, valid) for closed, valid in self._window if valid]
        if len(valid_frames) < PERCLOS_MIN_VALID_FRAMES:
            return 0.0
        closed_count = sum(1 for closed, _ in valid_frames if closed)
        return closed_count / len(valid_frames)

    @property
    def valid(self) -> bool:
        """True if >= PERCLOS_MIN_VALID_FRAMES valid frames are in the window."""
        valid_count = sum(1 for _, v in self._window if v)
        return valid_count >= PERCLOS_MIN_VALID_FRAMES
