"""Audio alerter v2 — Mac afplay-based alert sounds.

URGENT: 3 rapid beeps (phone use — highest priority)
HIGH:   1 beep (standard distraction alert)

Non-blocking: each Popen call spawns afplay in the background.
Falls back to the terminal bell character if afplay is unavailable.

PRD §8 — Alert output
"""

import subprocess
from src.contracts import AlertLevel

_SOUND = '/System/Library/Sounds/Ping.aiff'


def play_alert(level: AlertLevel) -> None:
    """Play an audio alert appropriate to the alert severity.

    Args:
        level: AlertLevel.URGENT (3 beeps) or AlertLevel.HIGH (1 beep).
               Other levels are silently ignored.
    """
    if level == AlertLevel.URGENT:
        count = 3
    elif level == AlertLevel.HIGH:
        count = 1
    else:
        return

    for _ in range(count):
        try:
            subprocess.Popen(
                ['afplay', _SOUND],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            print('\a', end='', flush=True)
