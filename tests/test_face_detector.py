"""Tests for the face detector module."""

import numpy as np
import pytest

from src.config_loader import FaceDetectorConfig
from src.detection.face_detector import FaceDetector, FaceResult, compute_ear, compute_mar


class TestComputeEAR:
    """Tests for the Eye Aspect Ratio computation."""

    def test_open_eye(self) -> None:
        """Open eye should have EAR > 0.2."""
        # Simulate an open eye: wide vertical, normal horizontal
        landmarks = np.zeros((500, 3), dtype=np.float64)
        # p1 (left corner), p4 (right corner): horizontal
        landmarks[0] = [0.0, 0.0, 0.0]
        landmarks[3] = [1.0, 0.0, 0.0]
        # p2 (upper left), p6 (lower left): vertical pair 1
        landmarks[1] = [0.3, -0.15, 0.0]
        landmarks[5] = [0.3, 0.15, 0.0]
        # p3 (upper right), p5 (lower right): vertical pair 2
        landmarks[2] = [0.7, -0.15, 0.0]
        landmarks[4] = [0.7, 0.15, 0.0]

        ear = compute_ear(landmarks, [0, 1, 2, 3, 4, 5])
        assert ear > 0.2

    def test_closed_eye(self) -> None:
        """Closed eye should have EAR close to 0."""
        landmarks = np.zeros((500, 3), dtype=np.float64)
        landmarks[0] = [0.0, 0.0, 0.0]
        landmarks[3] = [1.0, 0.0, 0.0]
        # Very small vertical distances (eyes nearly shut)
        landmarks[1] = [0.3, -0.005, 0.0]
        landmarks[5] = [0.3, 0.005, 0.0]
        landmarks[2] = [0.7, -0.005, 0.0]
        landmarks[4] = [0.7, 0.005, 0.0]

        ear = compute_ear(landmarks, [0, 1, 2, 3, 4, 5])
        assert ear < 0.05

    def test_zero_horizontal(self) -> None:
        """Zero horizontal distance should return 0 (no division error)."""
        landmarks = np.zeros((500, 3), dtype=np.float64)
        # All same point
        ear = compute_ear(landmarks, [0, 1, 2, 3, 4, 5])
        assert ear == 0.0

    def test_symmetric_eye(self) -> None:
        """Symmetric eye landmarks should give consistent EAR."""
        landmarks = np.zeros((500, 3), dtype=np.float64)
        landmarks[0] = [0.0, 0.5, 0.0]
        landmarks[3] = [1.0, 0.5, 0.0]
        landmarks[1] = [0.3, 0.3, 0.0]
        landmarks[5] = [0.3, 0.7, 0.0]
        landmarks[2] = [0.7, 0.3, 0.0]
        landmarks[4] = [0.7, 0.7, 0.0]

        ear = compute_ear(landmarks, [0, 1, 2, 3, 4, 5])
        # Both vertical distances are 0.4, horizontal is 1.0
        # EAR = (0.4 + 0.4) / (2 * 1.0) = 0.4
        assert ear == pytest.approx(0.4, abs=0.01)


class TestComputeMAR:
    """Tests for the Mouth Aspect Ratio computation."""

    def test_closed_mouth(self) -> None:
        """Closed mouth should have low MAR."""
        landmarks = np.zeros((500, 3), dtype=np.float64)
        # Lip landmarks (indices: 13, 312, 82 top; 14, 317, 87 bottom; 78 left, 308 right)
        landmarks[78] = [0.0, 0.5, 0.0]   # left lip
        landmarks[308] = [1.0, 0.5, 0.0]  # right lip
        # Top and bottom very close (closed mouth)
        landmarks[13] = [0.5, 0.49, 0.0]
        landmarks[312] = [0.3, 0.49, 0.0]
        landmarks[82] = [0.7, 0.49, 0.0]
        landmarks[14] = [0.5, 0.51, 0.0]
        landmarks[317] = [0.3, 0.51, 0.0]
        landmarks[87] = [0.7, 0.51, 0.0]

        mar = compute_mar(landmarks)
        assert mar < 0.1

    def test_open_mouth_yawn(self) -> None:
        """Wide open mouth should have high MAR."""
        landmarks = np.zeros((500, 3), dtype=np.float64)
        landmarks[78] = [0.0, 0.5, 0.0]
        landmarks[308] = [1.0, 0.5, 0.0]
        # Large vertical gap (yawn)
        landmarks[13] = [0.5, 0.1, 0.0]
        landmarks[312] = [0.3, 0.1, 0.0]
        landmarks[82] = [0.7, 0.1, 0.0]
        landmarks[14] = [0.5, 0.9, 0.0]
        landmarks[317] = [0.3, 0.9, 0.0]
        landmarks[87] = [0.7, 0.9, 0.0]

        mar = compute_mar(landmarks)
        assert mar > 0.5

    def test_zero_horizontal_returns_zero(self) -> None:
        """Zero horizontal distance should return 0."""
        landmarks = np.zeros((500, 3), dtype=np.float64)
        mar = compute_mar(landmarks)
        assert mar == 0.0


class TestFaceDetectorDisabled:
    """Tests for the face detector when disabled."""

    def test_disabled_returns_empty_result(self) -> None:
        """Disabled face detector returns FaceResult(face_visible=False)."""
        config = FaceDetectorConfig(enabled=False)
        detector = FaceDetector(config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert result.face_visible is False
        assert result.landmarks is None
        assert result.head_pose is None

    def test_disabled_load_model_noop(self) -> None:
        """load_model is a no-op for disabled detector."""
        config = FaceDetectorConfig(enabled=False)
        detector = FaceDetector(config)
        detector.load_model("nonexistent.tflite")  # should not raise


class TestFaceResult:
    """Tests for the FaceResult dataclass."""

    def test_default_values(self) -> None:
        """Default FaceResult has all None/False fields."""
        r = FaceResult()
        assert r.face_visible is False
        assert r.landmarks is None
        assert r.head_pose is None
        assert r.gaze_vector is None
        assert r.ear_left is None
        assert r.ear_right is None
        assert r.mouth_aspect_ratio is None

    def test_populated_values(self) -> None:
        """FaceResult can be constructed with all fields."""
        r = FaceResult(
            face_visible=True,
            head_pose=(10.0, 20.0, 5.0),
            ear_left=0.25,
            ear_right=0.27,
            mouth_aspect_ratio=0.3,
        )
        assert r.face_visible is True
        assert r.head_pose == (10.0, 20.0, 5.0)
        assert r.ear_left == pytest.approx(0.25)
        assert r.mouth_aspect_ratio == pytest.approx(0.3)
