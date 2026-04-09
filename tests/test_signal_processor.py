"""Tests for src/logic/signal_processor.py.

All tests use synthetic PerceptionBundles — no camera, MediaPipe, or model
files required. PRD §5.1–§5.7.
"""

import pytest

from src.contracts import FaceDetection, PerceptionBundle, PhoneDetectionOutput
from src.logic.signal_processor import SignalProcessor


# ── Helpers ──────────────────────────────────────────────────────────────────

def _bundle_face(
    pitch: float = 0.0,
    yaw: float = 0.0,
    roll: float = 0.0,
    ear_left: float = 0.30,
    ear_right: float = 0.30,
    phone_detected: bool = False,
    phone_conf: float = 0.0,
    timestamp_ns: int = 0,
    frame_id: int = 0,
) -> PerceptionBundle:
    """Synthetic bundle with face present and the given head pose."""
    return PerceptionBundle(
        timestamp_ns=timestamp_ns,
        frame_id=frame_id,
        face=FaceDetection(present=True, confidence=0.95),
        phone=PhoneDetectionOutput(detected=phone_detected, max_confidence=phone_conf),
        head_pose_raw=(pitch, yaw, roll),
        ear_left=ear_left,
        ear_right=ear_right,
    )


def _bundle_no_face(
    phone_detected: bool = False,
    phone_conf: float = 0.0,
) -> PerceptionBundle:
    """Synthetic bundle with face absent."""
    return PerceptionBundle(
        face=FaceDetection(present=False),
        phone=PhoneDetectionOutput(detected=phone_detected, max_confidence=phone_conf),
    )


# ── Test 1: face present, filtered angles converge ────────────────────────────

class TestFacePresent:
    def test_filtered_angles_converge_after_10_frames(self):
        """head_pose_raw=(5.0, 10.0, 0.0) for 10 frames → angles within 2.0 of input."""
        sp = SignalProcessor()
        result = None
        for _ in range(10):
            result = sp.process(_bundle_face(pitch=5.0, yaw=10.0, roll=0.0))
        assert result.face_present is True
        assert result.signals_valid is True
        assert abs(result.head_pose.yaw_deg - 10.0) < 2.0
        assert abs(result.head_pose.pitch_deg - 5.0) < 2.0

    def test_raw_values_stored_in_debug_fields(self):
        sp = SignalProcessor()
        result = sp.process(_bundle_face(pitch=5.0, yaw=10.0, roll=2.0))
        assert result.head_pose.raw_yaw_deg == pytest.approx(10.0)
        assert result.head_pose.raw_pitch_deg == pytest.approx(5.0)
        assert result.head_pose.raw_roll_deg == pytest.approx(2.0)

    def test_head_pose_valid_true_when_face_present(self):
        sp = SignalProcessor()
        result = sp.process(_bundle_face())
        assert result.head_pose.valid is True

    def test_signals_valid_true_when_face_present(self):
        sp = SignalProcessor()
        result = sp.process(_bundle_face())
        assert result.signals_valid is True

    def test_timestamps_forwarded(self):
        sp = SignalProcessor()
        result = sp.process(_bundle_face(timestamp_ns=123456, frame_id=7))
        assert result.timestamp_ns == 123456
        assert result.frame_id == 7


# ── Test 2: face absent ───────────────────────────────────────────────────────

class TestFaceAbsent:
    def test_face_absent_sets_face_present_false(self):
        sp = SignalProcessor()
        result = sp.process(_bundle_no_face())
        assert result.face_present is False

    def test_face_absent_sets_signals_valid_false(self):
        sp = SignalProcessor()
        result = sp.process(_bundle_no_face())
        assert result.signals_valid is False

    def test_face_absent_head_pose_is_none(self):
        sp = SignalProcessor()
        result = sp.process(_bundle_no_face())
        assert result.head_pose is None

    def test_face_absent_gaze_is_none(self):
        sp = SignalProcessor()
        result = sp.process(_bundle_no_face())
        assert result.gaze_world is None


# ── Test 3: phone detected when face absent ───────────────────────────────────

class TestPhoneWithFaceAbsent:
    def test_phone_detected_when_face_absent(self):
        """Phone signal must be extracted even when face is absent."""
        sp = SignalProcessor()
        result = sp.process(_bundle_no_face(phone_detected=True, phone_conf=0.85))
        assert result.face_present is False
        assert result.phone_signal.detected is True
        assert result.phone_signal.confidence == pytest.approx(0.85)

    def test_phone_not_detected_passes_through(self):
        sp = SignalProcessor()
        result = sp.process(_bundle_no_face(phone_detected=False))
        assert result.phone_signal.detected is False

    def test_phone_detected_when_face_present(self):
        sp = SignalProcessor()
        result = sp.process(_bundle_face(phone_detected=True, phone_conf=0.90))
        assert result.phone_signal.detected is True
        assert result.phone_signal.confidence == pytest.approx(0.90)


# ── Test 4: neutral offset correction ────────────────────────────────────────

class TestNeutralOffsets:
    def test_offset_correction_applied_to_yaw(self):
        """set_neutral_offsets(5.0, 3.0) + yaw input 15.0 → corrected ≈ 10.0."""
        sp = SignalProcessor()
        sp.set_neutral_offsets(yaw=5.0, pitch=3.0)
        result = None
        for _ in range(10):
            result = sp.process(_bundle_face(pitch=0.0, yaw=15.0, roll=0.0))
        assert abs(result.head_pose.yaw_deg - 10.0) < 2.0

    def test_offset_correction_applied_to_pitch(self):
        sp = SignalProcessor()
        sp.set_neutral_offsets(yaw=0.0, pitch=3.0)
        result = None
        for _ in range(10):
            result = sp.process(_bundle_face(pitch=10.0, yaw=0.0, roll=0.0))
        assert abs(result.head_pose.pitch_deg - 7.0) < 2.0

    def test_zero_offsets_no_correction(self):
        sp = SignalProcessor()
        sp.set_neutral_offsets(yaw=0.0, pitch=0.0)
        result = None
        for _ in range(10):
            result = sp.process(_bundle_face(pitch=0.0, yaw=10.0, roll=0.0))
        assert abs(result.head_pose.yaw_deg - 10.0) < 2.0

    def test_raw_values_unaffected_by_offset(self):
        """Raw debug fields must store unfiltered, uncorrected values."""
        sp = SignalProcessor()
        sp.set_neutral_offsets(yaw=5.0, pitch=3.0)
        result = sp.process(_bundle_face(pitch=8.0, yaw=20.0, roll=0.0))
        assert result.head_pose.raw_yaw_deg == pytest.approx(20.0)
        assert result.head_pose.raw_pitch_deg == pytest.approx(8.0)


# ── Test 5 & 6: gaze on-road / off-road ──────────────────────────────────────

class TestGazeRoadZone:
    def test_on_road_when_angles_zero(self):
        """Corrected (yaw=0, pitch=0) must be on_road=True."""
        sp = SignalProcessor()
        result = None
        for _ in range(10):
            result = sp.process(_bundle_face(pitch=0.0, yaw=0.0))
        assert result.gaze_world.on_road is True

    def test_off_road_when_yaw_exceeds_limit(self):
        """Corrected yaw=30.0 exceeds ROAD_ZONE_YAW_MAX=25.0 → on_road=False."""
        sp = SignalProcessor()
        result = None
        for _ in range(20):
            result = sp.process(_bundle_face(pitch=0.0, yaw=30.0))
        assert result.gaze_world.on_road is False

    def test_off_road_when_pitch_below_limit(self):
        """Corrected pitch=-15.0 is below ROAD_ZONE_PITCH_MIN=-10.0 → on_road=False."""
        sp = SignalProcessor()
        result = None
        for _ in range(20):
            result = sp.process(_bundle_face(pitch=-15.0, yaw=0.0))
        assert result.gaze_world.on_road is False

    def test_gaze_world_valid_when_face_present(self):
        sp = SignalProcessor()
        result = sp.process(_bundle_face())
        assert result.gaze_world.valid is True


# ── Test 7 & 8: EAR calibration ──────────────────────────────────────────────

class TestEARCalibration:
    def test_ear_with_calibration_uses_provided_threshold(self):
        """set_ear_baseline(0.30, 0.225) → close_threshold=0.225, baseline=0.30."""
        sp = SignalProcessor()
        sp.set_ear_baseline(baseline=0.30, threshold=0.225)
        result = sp.process(_bundle_face(ear_left=0.28, ear_right=0.32))
        assert result.eye_signals.baseline_EAR == pytest.approx(0.30)
        assert result.eye_signals.close_threshold == pytest.approx(0.225)
        assert result.eye_signals.calibration_complete is True

    def test_ear_without_calibration_uses_default_threshold(self):
        """Without calibration: close_threshold must be EAR_DEFAULT_CLOSE_THRESHOLD=0.21."""
        sp = SignalProcessor()
        result = sp.process(_bundle_face(ear_left=0.28, ear_right=0.32))
        assert result.eye_signals.close_threshold == pytest.approx(0.21)
        assert result.eye_signals.calibration_complete is False

    def test_mean_ear_computed_correctly(self):
        sp = SignalProcessor()
        result = sp.process(_bundle_face(ear_left=0.26, ear_right=0.34))
        assert result.eye_signals.mean_EAR == pytest.approx(0.30)

    def test_left_right_ear_forwarded(self):
        sp = SignalProcessor()
        result = sp.process(_bundle_face(ear_left=0.22, ear_right=0.28))
        assert result.eye_signals.left_EAR == pytest.approx(0.22)
        assert result.eye_signals.right_EAR == pytest.approx(0.28)

    def test_eye_signals_valid_when_face_present(self):
        sp = SignalProcessor()
        result = sp.process(_bundle_face())
        assert result.eye_signals.valid is True


# ── Test 9: face absent > 10 frames resets Kalman filters ────────────────────

class TestKalmanResetOnAbsence:
    def test_kalman_reset_after_11_absent_frames(self):
        """After >10 absent frames, filters reset; new pose converges quickly."""
        sp = SignalProcessor()
        # Prime the filters with yaw=50°
        for _ in range(30):
            sp.process(_bundle_face(yaw=50.0, pitch=0.0))

        # 15 absent frames — triggers reset (>10)
        for _ in range(15):
            sp.process(_bundle_no_face())

        # New constant pose at 0° — should converge within a few frames
        result = None
        for _ in range(5):
            result = sp.process(_bundle_face(yaw=0.0, pitch=0.0))

        # Without reset, primed-at-50 filters would drag output far from 0
        # After reset, initialized fresh at 0, should be well within 5°
        assert abs(result.head_pose.yaw_deg) < 5.0

    def test_filters_not_reset_before_threshold(self):
        """Only 5 absent frames (< 10) — filters should NOT reset."""
        sp = SignalProcessor()
        for _ in range(30):
            sp.process(_bundle_face(yaw=30.0, pitch=0.0))

        for _ in range(5):
            sp.process(_bundle_no_face())

        # Filters still primed; first frame back should still show some lag toward 30
        result = sp.process(_bundle_face(yaw=30.0, pitch=0.0))
        # Kalman-filtered value should be above 10 (still influenced by prior state)
        assert result.head_pose.yaw_deg > 10.0


# ── Test 10: head_pose_raw is None ───────────────────────────────────────────

class TestHeadPoseRawNone:
    def test_signals_valid_false_when_head_pose_raw_none(self):
        """Face present but head_pose_raw=None → signals_valid=False."""
        sp = SignalProcessor()
        bundle = PerceptionBundle(
            face=FaceDetection(present=True, confidence=0.90),
            head_pose_raw=None,
        )
        result = sp.process(bundle)
        assert result.signals_valid is False

    def test_head_pose_none_when_head_pose_raw_none(self):
        sp = SignalProcessor()
        bundle = PerceptionBundle(
            face=FaceDetection(present=True, confidence=0.90),
            head_pose_raw=None,
        )
        result = sp.process(bundle)
        assert result.head_pose is None

    def test_face_present_true_even_when_head_pose_raw_none(self):
        sp = SignalProcessor()
        bundle = PerceptionBundle(
            face=FaceDetection(present=True, confidence=0.90),
            head_pose_raw=None,
        )
        result = sp.process(bundle)
        assert result.face_present is True


# ── Supplementary: reset_filters() ───────────────────────────────────────────

class TestResetFilters:
    def test_reset_filters_clears_all_kalman_state(self):
        sp = SignalProcessor()
        for _ in range(30):
            sp.process(_bundle_face(yaw=40.0, pitch=20.0))
        sp.reset_filters()
        result = sp.process(_bundle_face(yaw=0.0, pitch=0.0))
        # After reset, initialized fresh at 0° — must be near 0
        assert abs(result.head_pose.yaw_deg) < 1.0
        assert abs(result.head_pose.pitch_deg) < 1.0


# ── Kalman responsiveness after double-filter removal ─────────────────────────

class TestGazeResponsiveness:
    def test_on_road_recovers_within_15_frames_after_snapping_back(self):
        """After 20 frames at yaw=30° (off-road), snapping back to 0° → on_road within 15 frames.

        Verifies the double-Kalman fix: with a single filter the head-pose estimate
        crosses back inside the ±25° road zone. Higher R (25.0) smooths more aggressively
        so the filter settles in ~10-12 frames vs ~3-4 at R=4.0 — still fast enough
        to avoid false prolonged off-road readings.
        """
        sp = SignalProcessor()
        # Warm up: establish off-road state for 20 frames
        for _ in range(20):
            result = sp.process(_bundle_face(yaw=30.0, pitch=0.0))
        assert result.gaze_world.on_road is False, "Should be off-road after 20 frames at 30°"

        # Snap back to 0° — count how many frames until on_road flips True
        recovered = False
        for i in range(15):
            result = sp.process(_bundle_face(yaw=0.0, pitch=0.0))
            if result.gaze_world.on_road:
                recovered = True
                break

        assert recovered, (
            f"Expected on_road=True within 15 frames of returning to 0°. "
            f"Last yaw: {result.gaze_world.yaw_deg:.2f}° (must be within ±25°)"
        )
