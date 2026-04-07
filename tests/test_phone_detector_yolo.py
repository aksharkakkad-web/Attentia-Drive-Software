"""Tests for src/detection/phone_detector_yolo.py.

All tests use synthetic data and mock onnxruntime — no real model file or
camera required. PRD §FR-1.4 — phone detection every frame.
"""

from dataclasses import dataclass
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config_loader import ObjectDetectorConfig
from src.detection.object_detector import ObjectDetection
from src.detection.phone_detector_yolo import PhoneDetectorYolo


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_config(frame_skip: int = 1) -> ObjectDetectorConfig:
    return ObjectDetectorConfig(
        enabled=True,
        model_path='models/yolov8n_phone.onnx',
        target_classes=['cell phone'],
        frame_skip=frame_skip,
    )


def _make_yolo_output(
    cx: float = 320.0,
    cy: float = 240.0,
    w: float = 100.0,
    h: float = 80.0,
    conf: float = 0.90,
    n_anchors: int = 8400,
) -> List[np.ndarray]:
    """Build a synthetic (1, 5, 8400) YOLO output array with one detection at the given conf."""
    data = np.zeros((1, 5, n_anchors), dtype=np.float32)
    data[0, 0, 0] = cx
    data[0, 1, 0] = cy
    data[0, 2, 0] = w
    data[0, 3, 0] = h
    data[0, 4, 0] = conf
    return [data]


def _make_session_mock(run_return=None):
    """Return a mock onnxruntime.InferenceSession with plausible I/O descriptors."""
    input_meta = MagicMock()
    input_meta.name = 'images'
    input_meta.shape = [1, 3, 640, 640]

    output_meta = MagicMock()
    output_meta.name = 'output0'
    output_meta.shape = [1, 5, 8400]

    session = MagicMock()
    session.get_inputs.return_value = [input_meta]
    session.get_outputs.return_value = [output_meta]
    if run_return is not None:
        session.run.return_value = run_return
    return session


def _blank_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ── Test class ────────────────────────────────────────────────────────────────

class TestPhoneDetectorYoloLoad:
    """Model loading behaviour."""

    def test_loads_successfully_with_mock_session(self):
        """PhoneDetectorYolo is_available() == True when onnxruntime loads the model."""
        session = _make_session_mock()
        with patch('src.detection.phone_detector_yolo.PhoneDetectorYolo._check_model_file', return_value=True), \
             patch('onnxruntime.InferenceSession', return_value=session):
            det = PhoneDetectorYolo(_make_config())
        assert det.is_available() is True

    def test_unavailable_when_model_file_missing(self):
        """is_available() == False if the model file does not exist."""
        with patch('src.detection.phone_detector_yolo.PhoneDetectorYolo._check_model_file', return_value=False):
            det = PhoneDetectorYolo(_make_config())
        assert det.is_available() is False

    def test_unavailable_when_onnxruntime_import_fails(self):
        """is_available() == False if onnxruntime is not installed."""
        import builtins
        real_import = builtins.__import__

        def _block_ort(name, *args, **kwargs):
            if name == 'onnxruntime':
                raise ImportError("no module named onnxruntime")
            return real_import(name, *args, **kwargs)

        with patch('src.detection.phone_detector_yolo.PhoneDetectorYolo._check_model_file', return_value=True), \
             patch('builtins.__import__', side_effect=_block_ort):
            det = PhoneDetectorYolo(_make_config())
        assert det.is_available() is False

    def test_unavailable_when_disabled_in_config(self):
        """is_available() == False when config.enabled is False (load_model never called)."""
        cfg = _make_config()
        cfg.enabled = False
        det = PhoneDetectorYolo(cfg)
        assert det.is_available() is False


class TestPhoneDetectorYoloInference:
    """Inference output shape and content."""

    def _loaded_detector(self, session: MagicMock, frame_skip: int = 1) -> PhoneDetectorYolo:
        with patch('src.detection.phone_detector_yolo.PhoneDetectorYolo._check_model_file', return_value=True), \
             patch('onnxruntime.InferenceSession', return_value=session):
            return PhoneDetectorYolo(_make_config(frame_skip=frame_skip))

    def test_blank_frame_returns_empty_list(self):
        """All-zero output from model → no detections above threshold."""
        zero_output = [np.zeros((1, 5, 8400), dtype=np.float32)]
        session = _make_session_mock(run_return=zero_output)
        det = self._loaded_detector(session)
        result = det.detect(_blank_frame())
        assert result == []

    def test_high_confidence_detection_returned(self):
        """A single anchor with conf=0.90 → one ObjectDetection returned."""
        session = _make_session_mock(run_return=_make_yolo_output(conf=0.90))
        det = self._loaded_detector(session)
        result = det.detect(_blank_frame())
        assert len(result) == 1

    def test_output_class_name_is_cell_phone(self):
        """Returned detection always has class_name == 'cell phone'."""
        session = _make_session_mock(run_return=_make_yolo_output(conf=0.85))
        det = self._loaded_detector(session)
        result = det.detect(_blank_frame())
        assert result[0].class_name == 'cell phone'

    def test_output_confidence_matches_model(self):
        """Returned detection confidence equals the model score."""
        session = _make_session_mock(run_return=_make_yolo_output(conf=0.83))
        det = self._loaded_detector(session)
        result = det.detect(_blank_frame())
        assert result[0].confidence == pytest.approx(0.83, abs=1e-4)

    def test_output_bbox_is_four_int_tuple(self):
        """Returned bbox is a (x1, y1, x2, y2) tuple of four ints."""
        session = _make_session_mock(run_return=_make_yolo_output(conf=0.90))
        det = self._loaded_detector(session)
        result = det.detect(_blank_frame())
        bbox = result[0].bbox
        assert isinstance(bbox, tuple)
        assert len(bbox) == 4
        assert all(isinstance(v, int) for v in bbox)

    def test_output_matches_objectdetection_dataclass(self):
        """Return value is an ObjectDetection instance with the expected fields."""
        session = _make_session_mock(run_return=_make_yolo_output(conf=0.90))
        det = self._loaded_detector(session)
        result = det.detect(_blank_frame())
        assert len(result) == 1
        obj = result[0]
        assert isinstance(obj, ObjectDetection)
        assert hasattr(obj, 'class_name')
        assert hasattr(obj, 'confidence')
        assert hasattr(obj, 'bbox')

    def test_below_threshold_not_returned(self):
        """Detection at conf=0.69 (below PHONE_CONFIDENCE_THRESHOLD=0.70) → []."""
        session = _make_session_mock(run_return=_make_yolo_output(conf=0.69))
        det = self._loaded_detector(session)
        result = det.detect(_blank_frame())
        assert result == []

    def test_at_threshold_returned(self):
        """Detection at conf=0.70 (== threshold) → included."""
        session = _make_session_mock(run_return=_make_yolo_output(conf=0.70))
        det = self._loaded_detector(session)
        result = det.detect(_blank_frame())
        assert len(result) == 1

    def test_returns_empty_when_unavailable(self):
        """detect() returns [] immediately if model is not loaded."""
        with patch('src.detection.phone_detector_yolo.PhoneDetectorYolo._check_model_file', return_value=False):
            det = PhoneDetectorYolo(_make_config())
        result = det.detect(_blank_frame())
        assert result == []


class TestPhoneDetectorYoloFrameSkip:
    """frame_skip caching behaviour — PRD §FR-1.4."""

    def _loaded_detector(self, session: MagicMock, frame_skip: int) -> PhoneDetectorYolo:
        with patch('src.detection.phone_detector_yolo.PhoneDetectorYolo._check_model_file', return_value=True), \
             patch('onnxruntime.InferenceSession', return_value=session):
            return PhoneDetectorYolo(_make_config(frame_skip=frame_skip))

    def test_frame_skip_1_runs_every_call(self):
        """frame_skip=1 → session.run is called on every detect() invocation."""
        zero_output = [np.zeros((1, 5, 8400), dtype=np.float32)]
        session = _make_session_mock(run_return=zero_output)
        det = self._loaded_detector(session, frame_skip=1)

        for _ in range(5):
            det.detect(_blank_frame())

        assert session.run.call_count == 5

    def test_frame_skip_2_skips_every_other_call(self):
        """frame_skip=2 → session.run is called on frames 1, 3, 5 … (every other)."""
        zero_output = [np.zeros((1, 5, 8400), dtype=np.float32)]
        session = _make_session_mock(run_return=zero_output)
        det = self._loaded_detector(session, frame_skip=2)

        for _ in range(6):
            det.detect(_blank_frame())

        # Frames 1, 3, 5 → 3 real inference calls
        assert session.run.call_count == 3

    def test_frame_skip_1_returns_fresh_result_every_call(self):
        """frame_skip=1 → each call reflects the latest model output, not a cache."""
        outputs = [
            [np.zeros((1, 5, 8400), dtype=np.float32)],           # call 1: no detection
            _make_yolo_output(conf=0.91),                          # call 2: detection
        ]
        session = _make_session_mock()
        session.run.side_effect = outputs
        det = self._loaded_detector(session, frame_skip=1)

        result1 = det.detect(_blank_frame())
        result2 = det.detect(_blank_frame())

        assert result1 == []
        assert len(result2) == 1
