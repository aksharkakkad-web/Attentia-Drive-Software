"""Audio alerter for distraction and phone-detection alerts.

Two independent triggers share the same beep parameters and the same
_beep_playing guard (so they never overlap):

  1. DriverState.DISTRACTED — onset_delay_seconds then exponentially
     decreasing rollover interval, floored at rollover_min_seconds.
  2. Phone detected (confirmed "cell phone") — same timing parameters.

Rollover decay formula:
    rollover(t) = max(min, initial * exp(-k * t))
    k           = ln(initial / min) / decay_seconds

With defaults (initial=0.5, min=0.2, decay=5s):
    k ≈ 0.183  →  0.50 s at t=0, smoothly → 0.20 s at t=5 s, capped.

Uses winsound.Beep on Windows; falls back to terminal bell on other platforms.
Beep is played in a daemon thread so it never blocks the pipeline.

Phase: 3 (active).
"""

import math
import sys
import threading
import time
from typing import Optional

from src.config_loader import AudioConfig
from src.logic.alert_manager import Alert
from src.logic.hysteresis import DriverState


class AudioAlerter:
    """Timed audio alerter driven by DriverState and phone detection.

    Args:
        config: AudioConfig with timing and tone parameters.
    """

    def __init__(self, config: AudioConfig) -> None:
        self._config = config

        # Pre-compute decay constant k = ln(initial/min) / decay_seconds.
        # Clamped to avoid division by zero or negative k.
        initial = config.rollover_seconds
        minimum = config.rollover_min_seconds
        decay = max(config.rollover_decay_seconds, 1e-6)
        if initial > minimum > 0:
            self._decay_k = math.log(initial / minimum) / decay
        else:
            self._decay_k = 0.0  # No decay; constant rollover.

        # Distraction timer state
        self._distracted_since: Optional[float] = None
        self._last_distracted_beep: Optional[float] = None

        # Phone-detection timer state
        self._phone_since: Optional[float] = None
        self._last_phone_beep: Optional[float] = None

        self._beep_playing: bool = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, state: DriverState, phone_detected: bool = False) -> None:
        """Update alerter state. Call once per frame.

        Args:
            state: Current DriverState from the hysteresis state machine.
            phone_detected: True when a cell phone is confirmed visible.
        """
        if not self._config.enabled:
            return

        now = time.monotonic()
        self._update_distraction(now, state)
        self._update_phone(now, phone_detected)

    def alert(self, alert: Alert) -> None:
        """No-op kept for backward compatibility with old callers."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rollover_at(self, elapsed: float) -> float:
        """Return the current rollover interval for the given elapsed distraction time."""
        if self._decay_k == 0.0:
            return self._config.rollover_seconds
        value = self._config.rollover_seconds * math.exp(-self._decay_k * elapsed)
        return max(self._config.rollover_min_seconds, value)

    def _update_distraction(self, now: float, state: DriverState) -> None:
        if state is DriverState.DISTRACTED:
            if self._distracted_since is None:
                self._distracted_since = now
            elapsed = now - self._distracted_since
            if self._last_distracted_beep is None:
                if elapsed >= self._config.onset_delay_seconds:
                    self._trigger_beep(now, "_last_distracted_beep")
            else:
                rollover = self._rollover_at(elapsed)
                if now - self._last_distracted_beep >= rollover:
                    self._trigger_beep(now, "_last_distracted_beep")
        else:
            self._distracted_since = None
            self._last_distracted_beep = None

    def _update_phone(self, now: float, phone_detected: bool) -> None:
        if phone_detected:
            if self._phone_since is None:
                self._phone_since = now
            elapsed = now - self._phone_since
            if self._last_phone_beep is None:
                if elapsed >= self._config.onset_delay_seconds:
                    self._trigger_beep(now, "_last_phone_beep")
            else:
                rollover = self._rollover_at(elapsed)
                if now - self._last_phone_beep >= rollover:
                    self._trigger_beep(now, "_last_phone_beep")
        else:
            self._phone_since = None
            self._last_phone_beep = None

    def _trigger_beep(self, now: float, timestamp_attr: str) -> None:
        """Trigger a beep in a daemon thread if no beep is already playing.

        Sets the named timestamp attribute before launching so rollover timing
        is based on when the beep was triggered, not when it finishes.
        """
        with self._lock:
            if self._beep_playing:
                return
            self._beep_playing = True

        setattr(self, timestamp_attr, now)

        t = threading.Thread(target=self._beep_worker, daemon=True)
        t.start()

    def _beep_worker(self) -> None:
        """Play a beep and clear the _beep_playing flag when done."""
        try:
            if sys.platform == "win32":
                import winsound
                winsound.Beep(self._config.frequency_hz, self._config.duration_ms)
            else:
                print("\a", end="", flush=True)
        finally:
            with self._lock:
                self._beep_playing = False
