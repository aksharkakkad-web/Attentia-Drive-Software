"""Tests for src/contracts.py and src/config_prd.py.

Verifies:
- Every dataclass instantiates with no arguments and with specific values
- Enum members have correct values
- Config constants match PRD §19
- MVP extensions are present
- Scoring weights sum to 1.0
"""

import pytest

from src.contracts import (
    AlertCommand,
    AlertLevel,
    AlertType,
    DistractionScore,
    EyeSignals,
    FaceDetection,
    GazeOutput,
    GazeWorld,
    HeadPose,
    LandmarkOutput,
    PerceptionBundle,
    PhoneDetectionOutput,
    PhoneSignal,
    RawFrame,
    SignalFrame,
    TemporalFeatures,
)
import src.config_prd as cfg


# ─── Test 1: Every dataclass instantiates with no arguments ──────────────────

class TestDefaultInstantiation:
    def test_raw_frame_default(self):
        r = RawFrame()
        assert r.timestamp_ns == 0
        assert r.frame_id == 0
        assert r.data is None

    def test_face_detection_default(self):
        f = FaceDetection()
        assert f.present is False
        assert f.confidence == 0.0

    def test_landmark_output_default(self):
        l = LandmarkOutput()
        assert l.landmarks is None
        assert l.pose_valid is False

    def test_gaze_output_default(self):
        g = GazeOutput()
        assert g.valid is False
        assert g.combined_yaw == 0.0

    def test_phone_detection_output_default(self):
        p = PhoneDetectionOutput()
        assert p.detected is False
        assert p.bbox_norm is None

    def test_perception_bundle_default(self):
        b = PerceptionBundle()
        assert b.face.present is False
        assert b.ear_left == 0.0
        assert b.head_pose_raw is None

    def test_head_pose_default(self):
        h = HeadPose()
        assert h.valid is False
        assert h.yaw_deg == 0.0

    def test_eye_signals_default(self):
        e = EyeSignals()
        assert e.valid is False
        assert e.calibration_complete is False

    def test_gaze_world_default(self):
        g = GazeWorld()
        assert g.on_road is True
        assert g.valid is False

    def test_phone_signal_default(self):
        p = PhoneSignal()
        assert p.detected is False
        assert p.stale is False

    def test_signal_frame_default(self):
        s = SignalFrame()
        assert s.face_present is False
        assert s.signals_valid is False

    def test_temporal_features_default(self):
        t = TemporalFeatures()
        assert t.gaze_off_road_fraction == 0.0
        assert t.speed_zone == 'URBAN'

    def test_distraction_score_default(self):
        d = DistractionScore()
        assert d.composite_score == 0.0
        assert d.active_classes == []

    def test_alert_command_default(self):
        a = AlertCommand()
        assert a.alert_id == ''
        assert a.level == AlertLevel.HIGH

    def test_all_sixteen_instantiate(self):
        """Smoke test — every contract instantiates without error."""
        RawFrame()
        FaceDetection()
        LandmarkOutput()
        GazeOutput()
        PhoneDetectionOutput()
        PerceptionBundle()
        HeadPose()
        EyeSignals()
        GazeWorld()
        PhoneSignal()
        SignalFrame()
        TemporalFeatures()
        DistractionScore()
        AlertLevel.HIGH
        AlertType.VISUAL_INATTENTION
        AlertCommand()


# ─── Test 2: Dataclasses instantiate with specific values ────────────────────

class TestSpecificValueInstantiation:
    def test_raw_frame_with_values(self):
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        r = RawFrame(timestamp_ns=1000, frame_id=5, data=frame, source_type='file')
        assert r.timestamp_ns == 1000
        assert r.frame_id == 5
        assert r.data is frame
        assert r.source_type == 'file'

    def test_face_detection_with_values(self):
        f = FaceDetection(present=True, confidence=0.92, bbox_norm=(0.1, 0.2, 0.3, 0.4), face_size_px=120)
        assert f.present is True
        assert f.confidence == pytest.approx(0.92)
        assert f.bbox_norm == (0.1, 0.2, 0.3, 0.4)
        assert f.face_size_px == 120

    def test_head_pose_with_values(self):
        h = HeadPose(yaw_deg=10.5, pitch_deg=-5.0, roll_deg=2.0, valid=True,
                     raw_yaw_deg=11.0, raw_pitch_deg=-5.5, raw_roll_deg=2.1)
        assert h.yaw_deg == pytest.approx(10.5)
        assert h.valid is True
        assert h.raw_yaw_deg == pytest.approx(11.0)

    def test_distraction_score_with_values(self):
        d = DistractionScore(
            composite_score=0.7,
            gaze_threshold_breached=True,
            active_classes=['D-A', 'D-C'],
        )
        assert d.composite_score == pytest.approx(0.7)
        assert d.gaze_threshold_breached is True
        assert d.active_classes == ['D-A', 'D-C']

    def test_alert_command_with_values(self):
        a = AlertCommand(
            alert_id='abc-123',
            level=AlertLevel.URGENT,
            alert_type=AlertType.PHONE_USE,
            composite_score=0.85,
        )
        assert a.alert_id == 'abc-123'
        assert a.level == AlertLevel.URGENT
        assert a.alert_type == AlertType.PHONE_USE


# ─── Test 3: AlertLevel enum ─────────────────────────────────────────────────

class TestAlertLevelEnum:
    def test_has_low(self):
        assert AlertLevel.LOW.value == 1

    def test_has_high(self):
        assert AlertLevel.HIGH.value == 2

    def test_has_urgent(self):
        assert AlertLevel.URGENT.value == 3

    def test_ordering(self):
        assert AlertLevel.LOW.value < AlertLevel.HIGH.value < AlertLevel.URGENT.value


# ─── Test 4: AlertType enum ──────────────────────────────────────────────────

class TestAlertTypeEnum:
    def test_visual_inattention(self):
        assert AlertType.VISUAL_INATTENTION.value == 'D-A'

    def test_head_inattention(self):
        assert AlertType.HEAD_INATTENTION.value == 'D-B'

    def test_drowsiness(self):
        assert AlertType.DROWSINESS.value == 'D-C'

    def test_phone_use(self):
        assert AlertType.PHONE_USE.value == 'D-D'

    def test_face_absent(self):
        assert AlertType.FACE_ABSENT.value == 'FACE'

    def test_has_five_members(self):
        assert len(AlertType) == 5


# ─── Test 5: Scoring weights sum to 1.0 ─────────────────────────────────────

class TestScoringWeights:
    def test_weights_sum_to_one(self):
        total = cfg.WEIGHT_GAZE + cfg.WEIGHT_HEAD + cfg.WEIGHT_PERCLOS + cfg.WEIGHT_BLINK
        assert total == pytest.approx(1.0)

    def test_weight_gaze(self):
        assert cfg.WEIGHT_GAZE == pytest.approx(0.45)

    def test_weight_head(self):
        assert cfg.WEIGHT_HEAD == pytest.approx(0.30)

    def test_weight_perclos(self):
        assert cfg.WEIGHT_PERCLOS == pytest.approx(0.20)

    def test_weight_blink(self):
        assert cfg.WEIGHT_BLINK == pytest.approx(0.05)


# ─── Test 6: Spot-check 15+ config values against PRD §19 ───────────────────

class TestConfigValues:
    def test_road_zone_yaw_min(self):
        assert cfg.ROAD_ZONE_YAW_MIN == pytest.approx(-15.0)

    def test_road_zone_yaw_max(self):
        assert cfg.ROAD_ZONE_YAW_MAX == pytest.approx(15.0)

    def test_road_zone_pitch_min(self):
        assert cfg.ROAD_ZONE_PITCH_MIN == pytest.approx(-10.0)

    def test_road_zone_pitch_max(self):
        assert cfg.ROAD_ZONE_PITCH_MAX == pytest.approx(5.0)

    def test_t_gaze_seconds(self):
        assert cfg.T_GAZE_SECONDS == pytest.approx(2.0)

    def test_t_head_seconds(self):
        assert cfg.T_HEAD_SECONDS == pytest.approx(1.5)

    def test_t_phone_seconds(self):
        assert cfg.T_PHONE_SECONDS == pytest.approx(1.0)

    def test_t_face_absent_seconds(self):
        assert cfg.T_FACE_ABSENT_SECONDS == pytest.approx(5.0)

    def test_ear_default_close_threshold(self):
        assert cfg.EAR_DEFAULT_CLOSE_THRESHOLD == pytest.approx(0.21)

    def test_ear_calibration_multiplier(self):
        assert cfg.EAR_CALIBRATION_MULTIPLIER == pytest.approx(0.75)

    def test_perclos_alert_threshold(self):
        assert cfg.PERCLOS_ALERT_THRESHOLD == pytest.approx(0.15)

    def test_perclos_window_frames(self):
        assert cfg.PERCLOS_WINDOW_FRAMES == 60

    def test_composite_alert_threshold(self):
        assert cfg.COMPOSITE_ALERT_THRESHOLD == pytest.approx(0.55)

    def test_highway_score_modifier(self):
        assert cfg.HIGHWAY_SCORE_MODIFIER == pytest.approx(1.4)

    def test_kalman_process_noise(self):
        assert cfg.KALMAN_PROCESS_NOISE_Q == pytest.approx(0.01)

    def test_kalman_measurement_noise(self):
        assert cfg.KALMAN_MEASUREMENT_NOISE_R == pytest.approx(4.0)

    def test_lstm_reset_absent_frames(self):
        assert cfg.LSTM_RESET_ABSENT_FRAMES == 10

    def test_cooldown_drowsiness(self):
        assert cfg.COOLDOWN_DROWSINESS == pytest.approx(12.0)

    def test_face_confidence_gate(self):
        assert cfg.FACE_CONFIDENCE_GATE == pytest.approx(0.60)

    def test_head_yaw_threshold(self):
        assert cfg.HEAD_YAW_THRESHOLD_DEG == pytest.approx(30.0)

    def test_head_pitch_threshold(self):
        assert cfg.HEAD_PITCH_THRESHOLD_DEG == pytest.approx(20.0)


# ─── Test 7: DistractionScore.active_classes defaults to empty list ──────────

class TestActiveClassesDefault:
    def test_active_classes_is_empty_list(self):
        d = DistractionScore()
        assert d.active_classes == []
        assert d.active_classes is not None

    def test_active_classes_are_independent_instances(self):
        """Two instances must not share the same list object."""
        d1 = DistractionScore()
        d2 = DistractionScore()
        d1.active_classes.append('D-A')
        assert d2.active_classes == []


# ─── Test 8: PerceptionBundle defaults ──────────────────────────────────────

class TestPerceptionBundleDefaults:
    def test_face_present_false(self):
        b = PerceptionBundle()
        assert b.face.present is False

    def test_ear_left_zero(self):
        b = PerceptionBundle()
        assert b.ear_left == pytest.approx(0.0)

    def test_ear_right_zero(self):
        b = PerceptionBundle()
        assert b.ear_right == pytest.approx(0.0)

    def test_head_pose_raw_none(self):
        b = PerceptionBundle()
        assert b.head_pose_raw is None

    def test_phone_detected_false(self):
        b = PerceptionBundle()
        assert b.phone.detected is False

    def test_face_instances_are_independent(self):
        """Two bundles must not share the same FaceDetection object."""
        b1 = PerceptionBundle()
        b2 = PerceptionBundle()
        b1.face.present = True
        assert b2.face.present is False


# ─── Test 9: TemporalFeatures.face_absent_continuous_secs ───────────────────

class TestTemporalFeaturesExtensions:
    def test_face_absent_continuous_secs_exists(self):
        t = TemporalFeatures()
        assert hasattr(t, 'face_absent_continuous_secs')

    def test_face_absent_continuous_secs_default(self):
        t = TemporalFeatures()
        assert t.face_absent_continuous_secs == pytest.approx(0.0)


# ─── Test 10: TemporalFeatures.perclos_valid ────────────────────────────────

class TestPerclosValid:
    def test_perclos_valid_exists(self):
        t = TemporalFeatures()
        assert hasattr(t, 'perclos_valid')

    def test_perclos_valid_defaults_false(self):
        t = TemporalFeatures()
        assert t.perclos_valid is False


# ─── Test 11: DistractionScore extension fields ──────────────────────────────

class TestDistractionScoreExtensions:
    def test_composite_threshold_breached_exists(self):
        d = DistractionScore()
        assert hasattr(d, 'composite_threshold_breached')

    def test_composite_threshold_breached_default_false(self):
        d = DistractionScore()
        assert d.composite_threshold_breached is False

    def test_face_absent_threshold_breached_exists(self):
        d = DistractionScore()
        assert hasattr(d, 'face_absent_threshold_breached')

    def test_face_absent_threshold_breached_default_false(self):
        d = DistractionScore()
        assert d.face_absent_threshold_breached is False


# ─── Test 12: AlertCommand.active_classes ────────────────────────────────────

class TestAlertCommandExtension:
    def test_active_classes_exists(self):
        a = AlertCommand()
        assert hasattr(a, 'active_classes')

    def test_active_classes_defaults_to_empty_list(self):
        a = AlertCommand()
        assert a.active_classes == []

    def test_active_classes_are_independent_instances(self):
        a1 = AlertCommand()
        a2 = AlertCommand()
        a1.active_classes.append('D-D')
        assert a2.active_classes == []


# ─── Test 13: CALIBRATION_DURATION_S MVP override ────────────────────────────

class TestCalibrationMvpOverride:
    def test_calibration_duration_is_mvp_value(self):
        """MVP override: 5.0s (PRD says 10.0). See config_prd.py comment."""
        assert cfg.CALIBRATION_DURATION_S == pytest.approx(5.0)
