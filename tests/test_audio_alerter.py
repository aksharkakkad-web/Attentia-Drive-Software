"""Tests for src/output/audio_alerter_v2.py.

All tests patch subprocess.Popen — no audio is played.
PRD §8 — Alert output.
"""

from unittest.mock import MagicMock, call, patch

import pytest

from src.contracts import AlertLevel
from src.output.audio_alerter_v2 import play_alert

_SOUND = '/System/Library/Sounds/Ping.aiff'
_POPEN_KWARGS = dict(stdout=-1, stderr=-1)  # subprocess.DEVNULL == -1


class TestPlayAlertUrgent:
    def test_urgent_triggers_3_popen_calls(self):
        """URGENT → 3 non-blocking Popen calls."""
        with patch('src.output.audio_alerter_v2.subprocess.Popen') as mock_popen:
            play_alert(AlertLevel.URGENT)
        assert mock_popen.call_count == 3

    def test_urgent_calls_afplay_with_correct_sound(self):
        """Each URGENT Popen call uses afplay with Ping.aiff."""
        with patch('src.output.audio_alerter_v2.subprocess.Popen') as mock_popen:
            play_alert(AlertLevel.URGENT)
        for c in mock_popen.call_args_list:
            assert c.args[0] == ['afplay', _SOUND]


class TestPlayAlertHigh:
    def test_high_triggers_1_popen_call(self):
        """HIGH → exactly 1 non-blocking Popen call."""
        with patch('src.output.audio_alerter_v2.subprocess.Popen') as mock_popen:
            play_alert(AlertLevel.HIGH)
        assert mock_popen.call_count == 1

    def test_high_calls_afplay_with_correct_sound(self):
        """HIGH Popen call uses afplay with Ping.aiff."""
        with patch('src.output.audio_alerter_v2.subprocess.Popen') as mock_popen:
            play_alert(AlertLevel.HIGH)
        assert mock_popen.call_args.args[0] == ['afplay', _SOUND]


class TestPlayAlertLow:
    def test_low_triggers_no_popen_calls(self):
        """LOW level is silently ignored — no Popen, no exception."""
        with patch('src.output.audio_alerter_v2.subprocess.Popen') as mock_popen:
            play_alert(AlertLevel.LOW)
        assert mock_popen.call_count == 0


class TestFallback:
    def test_popen_exception_falls_back_without_crashing(self, capsys):
        """When Popen raises, terminal bell is printed and no exception propagates."""
        with patch('src.output.audio_alerter_v2.subprocess.Popen',
                   side_effect=OSError('afplay not found')):
            play_alert(AlertLevel.HIGH)  # must not raise
        captured = capsys.readouterr()
        assert '\a' in captured.out

    def test_urgent_fallback_prints_3_bells(self, capsys):
        """URGENT with failing Popen → 3 terminal bells."""
        with patch('src.output.audio_alerter_v2.subprocess.Popen',
                   side_effect=OSError('afplay not found')):
            play_alert(AlertLevel.URGENT)
        captured = capsys.readouterr()
        assert captured.out.count('\a') == 3
