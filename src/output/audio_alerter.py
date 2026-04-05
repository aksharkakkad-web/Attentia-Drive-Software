"""Audio alerter for distraction alerts.

Phase 0: Stub — prints alert to terminal.
Phase 7A: Will be replaced by audio_alerter_v2.py with proper alert types.
"""

from typing import Any


class AudioAlerter:
    """Audio alert output for distraction events.

    Currently prints alerts to the terminal. Interface is designed for
    future audio playback via pygame.mixer or similar.

    # Future: Replace print with audio playback. See assets/ for alert sound files.
    """

    def alert(self, alert: Any) -> None:
        """Fire an audio alert.

        Accepts any object with level and triggers attributes, or None.

        Phase 0: Prints to terminal.
        Phase 7A: Will play a .wav file corresponding to the alert level.

        Args:
            alert: The alert object to output.
        """
        level = getattr(alert, "level", None)
        triggers = getattr(alert, "triggers", [])
        confidence = getattr(alert, "confidence", 0.0)

        level_name = getattr(level, "name", str(level)) if level is not None else "UNKNOWN"
        print(
            f"\U0001f6a8 ALERT: {level_name} | "
            f"Triggers: {triggers} | "
            f"Confidence: {confidence:.2f}"
        )
