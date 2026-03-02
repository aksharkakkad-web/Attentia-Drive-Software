"""Audio alerter for distraction alerts.

Phase 1-2: Prints alert to terminal.
# Future: Replace print with audio playback. See assets/ for alert sound files.

Phase: 1-2 (active, print-only).
"""

from src.logic.alert_manager import Alert


class AudioAlerter:
    """Audio alert output for distraction events.

    Currently prints alerts to the terminal. Interface is designed for
    future audio playback via pygame.mixer or similar.

    # Future: Replace print with audio playback. See assets/ for alert sound files.
    # The alert() method signature accepts an Alert object and can be swapped
    # to play .wav files with different sounds per alert level.
    """

    def alert(self, alert: Alert) -> None:
        """Fire an audio alert.

        Phase 1-2: Prints to terminal.
        Future: Play a .wav file corresponding to the alert level.

        Args:
            alert: The Alert to output.
        """
        print(
            f"[ALERT] {alert.level.name} | "
            f"Triggers: {alert.triggers} | "
            f"Confidence: {alert.confidence:.2f}"
        )
