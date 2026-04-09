"""AudioSink interface — alert playback abstraction.

Day 1 scope: wrap the existing Mac ``afplay`` subprocess path behind a
proper interface so downstream pipeline code no longer calls subprocess
directly. Day 5 replaces AfplayAudioSink with a SoundDeviceAudioSink that
pre-loads PCM, holds a mutex to prevent overlap, and uses distinct tones
per alert type (see docs/DESK_MVP.md §"Audio cleanup").

Contract: docs/INTERFACES.md §AudioSink.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from enum import Enum
from typing import Optional

from src.contracts import AlertLevel, AlertType

logger = logging.getLogger(__name__)


class SystemEvent(Enum):
    """Non-alert audio events (calibration chimes, system errors).

    Day 1 placeholder — AfplayAudioSink ignores these. Day 5 SoundDeviceAudioSink
    plays distinct chimes per event.
    """

    CALIBRATION_READY = 'calibration_ready'
    CALIBRATION_FAILED = 'calibration_failed'
    SYSTEM_ERROR = 'system_error'


# Tone priority — lower number = higher priority (matches P-01).
# Used by Day 5 mutex logic; Day 1 AfplayAudioSink does not consult priority.
_ALERT_PRIORITY = {
    AlertType.PHONE_USE: 1,
    AlertType.DROWSINESS: 2,
    AlertType.VISUAL_INATTENTION: 3,
    AlertType.HEAD_INATTENTION: 3,
    AlertType.FACE_ABSENT: 4,
}


class AudioSink:
    """Abstract alert playback sink.

    Implementations must be non-blocking: play() returns within ~5 ms and
    actual audio runs on a worker thread or non-blocking callback. Failures
    are logged once and swallowed; never raise into the pipeline loop.
    """

    def open(self) -> bool:
        raise NotImplementedError

    def play(self, alert_type: AlertType, level: Optional[AlertLevel] = None) -> None:
        """Play the tone for this alert type. Non-blocking, never raises."""
        raise NotImplementedError

    def play_system(self, event: SystemEvent) -> None:
        """Play a system event sound (ready chime, calibration failed, etc).

        Day 1 AfplayAudioSink may no-op these."""
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def is_available(self) -> bool:
        raise NotImplementedError


# ─── Mac Day-1 implementation ────────────────────────────────────────────────

_MAC_SOUND = '/System/Library/Sounds/Ping.aiff'


def _play_sequence(count: int, path: str) -> None:
    """Play ``count`` beeps sequentially on a daemon thread."""
    for _ in range(count):
        try:
            subprocess.run(
                ['afplay', path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            print('\a', end='', flush=True)


class AfplayAudioSink(AudioSink):
    """Mac legacy implementation — wraps the existing afplay subprocess path.

    Preserves Day-0 behavior exactly (3 beeps for URGENT, 1 for HIGH, sequential
    via daemon thread) so swapping to this interface is a pure refactor, no
    behavioral delta. Day 5 replaces this with SoundDeviceAudioSink.

    DEV-002: this is the isolated location of the afplay decision. Nothing
    else in the codebase should invoke subprocess for audio.
    """

    def __init__(self, sound_path: str = _MAC_SOUND) -> None:
        self._sound_path = sound_path
        self._opened = False

    def open(self) -> bool:
        self._opened = True
        logger.debug("AfplayAudioSink: open (sound=%s)", self._sound_path)
        return True

    def play(self, alert_type: AlertType, level: Optional[AlertLevel] = None) -> None:
        if not self._opened:
            return
        # Preserve legacy URGENT=3, HIGH=1 count. PHONE_USE always URGENT.
        if level is None:
            level = (
                AlertLevel.URGENT
                if alert_type == AlertType.PHONE_USE
                else AlertLevel.HIGH
            )
        if level == AlertLevel.URGENT:
            count = 3
        elif level == AlertLevel.HIGH:
            count = 1
        else:
            return
        threading.Thread(
            target=_play_sequence,
            args=(count, self._sound_path),
            daemon=True,
        ).start()

    def play_system(self, event: SystemEvent) -> None:
        # Day 1: no distinct system chimes. Day 5 SoundDeviceAudioSink adds them.
        logger.debug("AfplayAudioSink: play_system(%s) — no-op in Day 1", event.value)

    def close(self) -> None:
        self._opened = False

    def is_available(self) -> bool:
        return self._opened


class NullAudioSink(AudioSink):
    """No-op sink for tests and headless runs.

    Records the last event for assertions; makes no sound.
    """

    def __init__(self) -> None:
        self._opened = False
        self.last_alert: Optional[AlertType] = None
        self.last_event: Optional[SystemEvent] = None

    def open(self) -> bool:
        self._opened = True
        return True

    def play(self, alert_type: AlertType, level: Optional[AlertLevel] = None) -> None:
        self.last_alert = alert_type

    def play_system(self, event: SystemEvent) -> None:
        self.last_event = event

    def close(self) -> None:
        self._opened = False

    def is_available(self) -> bool:
        return self._opened
