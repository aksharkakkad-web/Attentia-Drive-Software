"""Tests for the drowsiness detector module."""

import pytest

from src.config_loader import DrowsinessConfig
from src.detection.drowsiness import DrowsinessDetector, DrowsinessResult
from src.detection.face_detector import FaceResult


def _make_config(**overrides) -> DrowsinessConfig:
    """Create a DrowsinessConfig with defaults suitable for testing."""
    defaults = dict(
        enabled=True,
        ear_threshold=0.21,
        perclos_window_seconds=60.0,
        perclos_threshold=0.15,
        mar_threshold=0.6,
        yawn_min_frames=10,
    )
    defaults.update(overrides)
    return DrowsinessConfig(**defaults)


def _make_face(ear: float = 0.3, mar: float = 0.2) -> FaceResult:
    """Create a FaceResult with given EAR and MAR."""
    return FaceResult(
        face_visible=True,
        ear_left=ear,
        ear_right=ear,
        mouth_aspect_ratio=mar,
        head_pose=(0.0, 0.0, 0.0),
    )


class TestDrowsinessDetector:
    """Tests for the DrowsinessDetector."""

    def test_disabled_returns_default(self) -> None:
        """Disabled detector returns default DrowsinessResult."""
        config = _make_config(enabled=False)
        detector = DrowsinessDetector(config, fps=30.0)
        result = detector.update(_make_face())
        assert result.is_drowsy is False
        assert result.perclos == 0.0

    def test_no_face_returns_default(self) -> None:
        """No face visible returns default result with accumulated counts."""
        config = _make_config()
        detector = DrowsinessDetector(config, fps=30.0)
        result = detector.update(FaceResult(face_visible=False))
        assert result.is_drowsy is False

    def test_open_eyes_not_drowsy(self) -> None:
        """Normal open eyes -> not drowsy."""
        config = _make_config()
        detector = DrowsinessDetector(config, fps=30.0)
        result = detector.update(_make_face(ear=0.3))
        assert result.is_drowsy is False
        assert result.perclos == 0.0

    def test_closed_eyes_increases_perclos(self) -> None:
        """Eyes below threshold should increase PERCLOS."""
        config = _make_config(perclos_window_seconds=1.0)
        detector = DrowsinessDetector(config, fps=10.0)
        # Feed 5 closed-eye frames out of 10-frame window
        for _ in range(5):
            detector.update(_make_face(ear=0.15))
        for _ in range(5):
            result = detector.update(_make_face(ear=0.3))
        assert result.perclos == pytest.approx(0.5, abs=0.01)

    def test_prolonged_closure_is_drowsy(self) -> None:
        """Eyes closed for > 1.5 seconds -> is_drowsy."""
        config = _make_config()
        detector = DrowsinessDetector(config, fps=10.0)
        # 16 frames at 10 fps = 1.6 seconds of closed eyes
        for _ in range(16):
            result = detector.update(_make_face(ear=0.15))
        assert result.is_drowsy is True
        assert result.eye_closure_duration > 1.5

    def test_blink_detection(self) -> None:
        """Short eye closure (1-3 frames) counts as a blink."""
        config = _make_config()
        detector = DrowsinessDetector(config, fps=30.0)
        # Open eyes
        for _ in range(10):
            detector.update(_make_face(ear=0.3))
        # Blink: 2 frames closed
        detector.update(_make_face(ear=0.15))
        detector.update(_make_face(ear=0.15))
        # Open again (blink registered on opening)
        result = detector.update(_make_face(ear=0.3))
        assert result.blink_count == 1

    def test_long_closure_not_counted_as_blink(self) -> None:
        """Closure > 3 frames is NOT counted as a blink."""
        config = _make_config()
        detector = DrowsinessDetector(config, fps=30.0)
        # 5 frames closed (too long for blink)
        for _ in range(5):
            detector.update(_make_face(ear=0.15))
        # Open again
        result = detector.update(_make_face(ear=0.3))
        assert result.blink_count == 0

    def test_yawn_detection(self) -> None:
        """Sustained high MAR for yawn_min_frames -> is_yawning."""
        config = _make_config(yawn_min_frames=5)
        detector = DrowsinessDetector(config, fps=30.0)
        # 5 frames of yawning
        for _ in range(5):
            result = detector.update(_make_face(ear=0.3, mar=0.8))
        assert result.is_yawning is True
        assert result.yawn_count == 1

    def test_no_yawn_below_threshold(self) -> None:
        """MAR below threshold -> no yawn."""
        config = _make_config()
        detector = DrowsinessDetector(config, fps=30.0)
        for _ in range(20):
            result = detector.update(_make_face(ear=0.3, mar=0.3))
        assert result.is_yawning is False
        assert result.yawn_count == 0

    def test_yawn_count_increments_once_per_yawn(self) -> None:
        """Sustained yawn only counts as 1 yawn event."""
        config = _make_config(yawn_min_frames=3)
        detector = DrowsinessDetector(config, fps=30.0)
        # Long sustained yawn: 10 frames
        for _ in range(10):
            result = detector.update(_make_face(ear=0.3, mar=0.8))
        assert result.yawn_count == 1

    def test_multiple_yawns(self) -> None:
        """Two separate yawns should count as 2."""
        config = _make_config(yawn_min_frames=3)
        detector = DrowsinessDetector(config, fps=30.0)
        # First yawn
        for _ in range(5):
            detector.update(_make_face(ear=0.3, mar=0.8))
        # Break
        for _ in range(5):
            detector.update(_make_face(ear=0.3, mar=0.2))
        # Second yawn
        for _ in range(5):
            result = detector.update(_make_face(ear=0.3, mar=0.8))
        assert result.yawn_count == 2

    def test_drowsiness_confidence_range(self) -> None:
        """Confidence is always in [0, 1]."""
        config = _make_config()
        detector = DrowsinessDetector(config, fps=30.0)
        # Normal frame
        r1 = detector.update(_make_face(ear=0.3))
        assert 0.0 <= r1.drowsiness_confidence <= 1.0
        # Drowsy frame
        for _ in range(50):
            r2 = detector.update(_make_face(ear=0.15, mar=0.8))
        assert 0.0 <= r2.drowsiness_confidence <= 1.0

    def test_perclos_threshold_triggers_drowsy(self) -> None:
        """PERCLOS exceeding threshold -> is_drowsy."""
        config = _make_config(perclos_window_seconds=1.0, perclos_threshold=0.15)
        detector = DrowsinessDetector(config, fps=10.0)
        # All 10 frames closed (PERCLOS = 1.0 >> 0.15)
        for _ in range(10):
            result = detector.update(_make_face(ear=0.15))
        assert result.perclos > config.perclos_threshold
        assert result.is_drowsy is True

    def test_blink_rate_computation(self) -> None:
        """Blink rate should be computed as blinks per minute."""
        config = _make_config()
        detector = DrowsinessDetector(config, fps=30.0)
        # Simulate 30 frames (1 second) with 2 blinks
        for _ in range(10):
            detector.update(_make_face(ear=0.3))
        # Blink 1
        detector.update(_make_face(ear=0.15))
        detector.update(_make_face(ear=0.15))
        detector.update(_make_face(ear=0.3))
        for _ in range(5):
            detector.update(_make_face(ear=0.3))
        # Blink 2
        detector.update(_make_face(ear=0.15))
        detector.update(_make_face(ear=0.3))
        for _ in range(8):
            result = detector.update(_make_face(ear=0.3))
        # 2 blinks in 30 frames at 30fps = 1 second -> 120 blinks/min
        assert result.blink_count == 2
        assert result.blink_rate > 0
