"""Tests for src/pipeline/adapters.py and src/output/event_logger.py.

All tests use synthetic data — no camera or model files required.
PRD §4.2, §9.
"""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest

from src.contracts import AlertCommand, AlertLevel, AlertType, DistractionScore
from src.output.event_logger import EventLogger
from src.pipeline.adapters import convert_to_perception_bundle, wrap_raw_frame


# ── Synthetic FaceResult / ObjectDetection stubs ──────────────────────────────

@dataclass
class _FaceResult:
    face_visible: bool = False
    landmarks: Optional[np.ndarray] = None
    head_pose: Optional[Tuple[float, float, float]] = None
    gaze_vector: Optional[Tuple[float, float, float]] = None
    ear_left: Optional[float] = None
    ear_right: Optional[float] = None
    mouth_aspect_ratio: Optional[float] = None


@dataclass
class _ObjectDetection:
    class_name: str = ''
    confidence: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)


# ── Test 1: face visible, known pose and EAR ──────────────────────────────────

class TestConvertFaceVisible:
    def test_face_present_and_head_pose_raw(self):
        """face_visible=True, head_pose=(10,5,0) → face.present=True, head_pose_raw set."""
        fr = _FaceResult(
            face_visible=True,
            head_pose=(10.0, 5.0, 0.0),
            ear_left=0.30,
            ear_right=0.28,
            landmarks=np.zeros((478, 3)),
        )
        bundle = convert_to_perception_bundle(fr, [], frame_id=1, timestamp_ns=1000)
        assert bundle.face.present is True
        assert bundle.face.confidence == pytest.approx(0.95)
        assert bundle.head_pose_raw == (10.0, 5.0, 0.0)
        assert bundle.ear_left == pytest.approx(0.30)
        assert bundle.ear_right == pytest.approx(0.28)
        assert bundle.frame_id == 1
        assert bundle.timestamp_ns == 1000

    def test_landmarks_stored_in_bundle(self):
        """Landmarks are forwarded into LandmarkOutput.landmarks."""
        lm = np.ones((478, 3), dtype=np.float64)
        fr = _FaceResult(face_visible=True, landmarks=lm, head_pose=(0.0, 0.0, 0.0))
        bundle = convert_to_perception_bundle(fr, [], 1, 0)
        assert bundle.landmarks is not None
        assert bundle.landmarks.landmarks is lm
        assert bundle.landmarks.pose_valid is True

    def test_gaze_present_when_face_visible(self):
        """GazeOutput is set (valid=True) when face is visible."""
        fr = _FaceResult(face_visible=True, head_pose=(0.0, 0.0, 0.0))
        bundle = convert_to_perception_bundle(fr, [], 1, 0)
        assert bundle.gaze is not None
        assert bundle.gaze.valid is True


# ── Test 2: face not visible ──────────────────────────────────────────────────

class TestConvertFaceAbsent:
    def test_face_absent_fields(self):
        """face_visible=False → face.present=False, head_pose_raw=None, EAR=0."""
        fr = _FaceResult(face_visible=False)
        bundle = convert_to_perception_bundle(fr, [], frame_id=5, timestamp_ns=500)
        assert bundle.face.present is False
        assert bundle.face.confidence == pytest.approx(0.0)
        assert bundle.head_pose_raw is None
        assert bundle.ear_left == pytest.approx(0.0)
        assert bundle.ear_right == pytest.approx(0.0)
        assert bundle.landmarks is None
        assert bundle.gaze is None


# ── Test 3: phone detected ────────────────────────────────────────────────────

class TestPhoneDetected:
    def test_phone_detected_above_threshold(self):
        """'cell phone' with confidence 0.85 → phone.detected=True, max_confidence=0.85."""
        fr = _FaceResult(face_visible=True, head_pose=(0.0, 0.0, 0.0))
        dets = [_ObjectDetection('cell phone', 0.85, (10, 20, 100, 200))]
        bundle = convert_to_perception_bundle(fr, dets, 1, 0)
        assert bundle.phone.detected is True
        assert bundle.phone.max_confidence == pytest.approx(0.85)
        assert bundle.phone.bbox_norm == (10, 20, 100, 200)

    def test_phone_below_threshold_not_detected(self):
        """'cell phone' with confidence 0.50 < 0.70 → phone.detected=False."""
        fr = _FaceResult(face_visible=True, head_pose=(0.0, 0.0, 0.0))
        dets = [_ObjectDetection('cell phone', 0.50, (0, 0, 0, 0))]
        bundle = convert_to_perception_bundle(fr, dets, 1, 0)
        assert bundle.phone.detected is False
        assert bundle.phone.max_confidence == pytest.approx(0.50)

    def test_highest_confidence_phone_selected(self):
        """Multiple phones → highest-confidence one is used."""
        fr = _FaceResult(face_visible=True, head_pose=(0.0, 0.0, 0.0))
        dets = [
            _ObjectDetection('cell phone', 0.72, (0, 0, 10, 10)),
            _ObjectDetection('cell phone', 0.91, (5, 5, 50, 50)),
        ]
        bundle = convert_to_perception_bundle(fr, dets, 1, 0)
        assert bundle.phone.max_confidence == pytest.approx(0.91)


# ── Test 4: no phone ──────────────────────────────────────────────────────────

class TestNoPhone:
    def test_no_phone_detections(self):
        """Empty object_detections → phone.detected=False, max_confidence=0.0."""
        fr = _FaceResult(face_visible=True, head_pose=(0.0, 0.0, 0.0))
        bundle = convert_to_perception_bundle(fr, [], 1, 0)
        assert bundle.phone.detected is False
        assert bundle.phone.max_confidence == pytest.approx(0.0)

    def test_non_phone_objects_ignored(self):
        """Non-phone objects don't affect phone signal."""
        fr = _FaceResult(face_visible=True, head_pose=(0.0, 0.0, 0.0))
        dets = [_ObjectDetection('cup', 0.95, (0, 0, 50, 50))]
        bundle = convert_to_perception_bundle(fr, dets, 1, 0)
        assert bundle.phone.detected is False


# ── Test 5: face_result is None ───────────────────────────────────────────────

class TestNullFaceResult:
    def test_none_face_result_returns_default_bundle(self):
        """face_result=None → returns default PerceptionBundle without crashing."""
        bundle = convert_to_perception_bundle(None, [], frame_id=3, timestamp_ns=999)
        assert bundle is not None
        assert bundle.face.present is False
        assert bundle.frame_id == 3
        assert bundle.timestamp_ns == 999
        assert bundle.head_pose_raw is None

    def test_none_ear_fields_default_to_zero(self):
        """FaceResult with ear_left=None, ear_right=None → ear values are 0.0."""
        fr = _FaceResult(face_visible=True, head_pose=(0.0, 0.0, 0.0),
                         ear_left=None, ear_right=None)
        bundle = convert_to_perception_bundle(fr, [], 1, 0)
        assert bundle.ear_left == pytest.approx(0.0)
        assert bundle.ear_right == pytest.approx(0.0)


# ── Test 6: event logger writes valid JSON ────────────────────────────────────

class TestEventLogger:
    def _make_logger(self, tmp_path: Path) -> EventLogger:
        return EventLogger(log_dir=str(tmp_path))

    def _read_lines(self, tmp_path: Path):
        files = list(tmp_path.glob('*.jsonl'))
        assert len(files) == 1
        return [json.loads(ln) for ln in files[0].read_text().strip().splitlines()]

    def test_log_alert_writes_valid_json(self, tmp_path):
        """log_alert() writes one valid JSON line with type='alert'."""
        el = self._make_logger(tmp_path)
        alert_cmd = AlertCommand(
            alert_id='abc-123',
            level=AlertLevel.URGENT,
            alert_type=AlertType.PHONE_USE,
            composite_score=0.75,
            active_classes=['D-D'],
        )
        score = DistractionScore(composite_score=0.75)
        el.log_alert(alert_cmd, score)
        lines = self._read_lines(tmp_path)
        assert len(lines) == 1
        rec = lines[0]
        assert rec['type'] == 'alert'
        assert rec['alert_id'] == 'abc-123'
        assert rec['level'] == 'URGENT'
        assert rec['alert_type'] == 'D-D'
        assert 'composite_score' in rec
        assert 'D-D' in rec['active_classes']

    def test_log_calibration_writes_valid_json(self, tmp_path):
        """log_calibration() writes one valid JSON line with type='calibration'."""
        el = self._make_logger(tmp_path)
        el.log_calibration(yaw_offset=5.0, pitch_offset=3.0, baseline_ear=0.30)
        lines = self._read_lines(tmp_path)
        assert lines[0]['type'] == 'calibration'
        assert lines[0]['yaw_offset'] == pytest.approx(5.0)
        assert lines[0]['pitch_offset'] == pytest.approx(3.0)
        assert lines[0]['baseline_ear'] == pytest.approx(0.30)

    def test_log_state_transition_writes_valid_json(self, tmp_path):
        """log_state_transition() writes one valid JSON line with type='state_transition'."""
        el = self._make_logger(tmp_path)
        el.log_state_transition('NOMINAL', 'DEGRADED', 'face_absent', frame_id=100)
        lines = self._read_lines(tmp_path)
        rec = lines[0]
        assert rec['type'] == 'state_transition'
        assert rec['previous'] == 'NOMINAL'
        assert rec['new'] == 'DEGRADED'
        assert rec['trigger'] == 'face_absent'
        assert rec['frame_id'] == 100

    def test_multiple_events_each_on_own_line(self, tmp_path):
        """Each log call produces exactly one JSON line."""
        el = self._make_logger(tmp_path)
        score = DistractionScore()
        alert_cmd = AlertCommand(
            alert_id='x', level=AlertLevel.HIGH,
            alert_type=AlertType.VISUAL_INATTENTION,
        )
        el.log_alert(alert_cmd, score)
        el.log_calibration(0.0, 0.0, 0.28)
        el.log_state_transition('A', 'B', 'test', 1)
        lines = self._read_lines(tmp_path)
        assert len(lines) == 3
        assert all(isinstance(json.loads(json.dumps(ln)), dict) for ln in lines)


# ── Test 7: wrap_raw_frame ────────────────────────────────────────────────────

class TestWrapRawFrame:
    def test_correct_dimensions_and_metadata(self):
        """wrap_raw_frame extracts shape and stores metadata correctly."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        raw = wrap_raw_frame(frame, frame_id=42, timestamp_ns=12345)
        assert raw.width == 640
        assert raw.height == 480
        assert raw.channels == 3
        assert raw.frame_id == 42
        assert raw.timestamp_ns == 12345
        assert raw.data is frame

    def test_source_type_is_webcam(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        raw = wrap_raw_frame(frame, 1, 0)
        assert raw.source_type == 'webcam'
