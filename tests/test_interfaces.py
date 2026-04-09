"""Tests for src/pipeline/interfaces/ — portability boundary scaffolding.

All tests use stubs/nulls. No hardware, no audio, no camera, no IMU.
"""

import time

import pytest

from src.contracts import AlertLevel, AlertType, IMUReading
from src.pipeline.interfaces import (
    AfplayAudioSink,
    AudioSink,
    ColorSpace,
    FrameResult,
    FrameSource,
    ImuSource,
    NoopThermalMonitor,
    NullAudioSink,
    NullFrameSource,
    OpenCVFrameSource,
    StubImuSource,
    SystemEvent,
    ThermalMonitor,
)


# ─── FrameResult ──────────────────────────────────────────────────────────────


class TestFrameResult:
    def test_ok_false_defaults(self):
        r = FrameResult(ok=False)
        assert r.ok is False
        assert r.frame is None
        assert r.timestamp_ms == 0.0
        assert r.color_space == ColorSpace.BGR

    def test_color_space_can_be_set(self):
        r = FrameResult(ok=True, color_space=ColorSpace.RGB)
        assert r.color_space == ColorSpace.RGB


# ─── NullFrameSource ──────────────────────────────────────────────────────────


class TestNullFrameSource:
    def test_open_close_cycle(self):
        src = NullFrameSource()
        assert not src.is_available()
        assert src.open() is True
        assert src.is_available()
        src.close()
        assert not src.is_available()

    def test_read_returns_not_ok(self):
        src = NullFrameSource()
        src.open()
        r = src.read()
        assert r.ok is False
        assert r.frame is None
        src.close()

    def test_fps_default(self):
        src = NullFrameSource()
        assert src.fps == 30.0

    def test_fps_custom(self):
        src = NullFrameSource(target_fps=60.0)
        assert src.fps == 60.0

    def test_read_legacy_returns_tuple(self):
        src = NullFrameSource()
        src.open()
        ok, frame = src.read_legacy()
        assert ok is False
        assert frame is None


# ─── NullAudioSink ────────────────────────────────────────────────────────────


class TestNullAudioSink:
    def test_open_close_cycle(self):
        sink = NullAudioSink()
        assert not sink.is_available()
        assert sink.open() is True
        assert sink.is_available()
        sink.close()
        assert not sink.is_available()

    def test_play_records_last_alert(self):
        sink = NullAudioSink()
        sink.open()
        sink.play(AlertType.PHONE_USE, AlertLevel.URGENT)
        assert sink.last_alert == AlertType.PHONE_USE

    def test_play_system_records_last_event(self):
        sink = NullAudioSink()
        sink.open()
        sink.play_system(SystemEvent.CALIBRATION_READY)
        assert sink.last_event == SystemEvent.CALIBRATION_READY

    def test_play_without_level(self):
        sink = NullAudioSink()
        sink.open()
        sink.play(AlertType.DROWSINESS)
        assert sink.last_alert == AlertType.DROWSINESS


# ─── StubImuSource ────────────────────────────────────────────────────────────


class TestStubImuSource:
    def test_open_returns_true(self):
        src = StubImuSource()
        assert src.open() is True

    def test_is_available_false(self):
        src = StubImuSource()
        assert src.is_available() is False

    def test_read_returns_invalid(self):
        src = StubImuSource()
        reading = src.read()
        assert isinstance(reading, IMUReading)
        assert reading.valid is False
        assert reading.timestamp_ms > 0

    def test_close_is_safe(self):
        src = StubImuSource()
        src.open()
        src.close()  # should not raise


# ─── NoopThermalMonitor ───────────────────────────────────────────────────────


class TestNoopThermalMonitor:
    def test_open_returns_true(self):
        mon = NoopThermalMonitor()
        assert mon.open() is True

    def test_get_temp_c_returns_none(self):
        mon = NoopThermalMonitor()
        assert mon.get_temp_c() is None

    def test_is_available_false(self):
        mon = NoopThermalMonitor()
        assert mon.is_available() is False

    def test_close_is_safe(self):
        mon = NoopThermalMonitor()
        mon.open()
        mon.close()  # should not raise


# ─── AfplayAudioSink (no actual audio — tests logic only) ────────────────────


class TestAfplayAudioSink:
    def test_play_without_open_is_noop(self):
        sink = AfplayAudioSink()
        # Should not raise, should not play
        sink.play(AlertType.PHONE_USE, AlertLevel.URGENT)

    def test_open_close_cycle(self):
        sink = AfplayAudioSink()
        assert sink.open() is True
        assert sink.is_available()
        sink.close()
        assert not sink.is_available()

    def test_play_system_does_not_raise(self):
        sink = AfplayAudioSink()
        sink.open()
        sink.play_system(SystemEvent.SYSTEM_ERROR)
        sink.close()

    def test_play_low_level_is_noop(self):
        """LOW and MEDIUM alerts are below threshold — AfplayAudioSink ignores them."""
        sink = AfplayAudioSink()
        sink.open()
        # Should not raise or spawn threads
        sink.play(AlertType.VISUAL_INATTENTION, AlertLevel.LOW)
        sink.close()


# ─── SystemEvent ──────────────────────────────────────────────────────────────


class TestSystemEvent:
    def test_all_values(self):
        assert SystemEvent.CALIBRATION_READY.value == 'calibration_ready'
        assert SystemEvent.CALIBRATION_FAILED.value == 'calibration_failed'
        assert SystemEvent.SYSTEM_ERROR.value == 'system_error'


# ─── ColorSpace ───────────────────────────────────────────────────────────────


class TestColorSpace:
    def test_values(self):
        assert ColorSpace.BGR.value == 'bgr'
        assert ColorSpace.RGB.value == 'rgb'
