"""Tests for Phase 4: DurationTimer, PerclosCalculator, BlinkDetector, TemporalEngine.

All tests use synthetic data — no camera, model files, or display required.
PRD §5.4, §5.5, §FR-3.1–FR-3.5.
"""

import pytest

from src.contracts import (
    EyeSignals,
    FaceDetection,
    GazeWorld,
    HeadPose,
    PhoneDetectionOutput,
    PhoneSignal,
    SignalFrame,
)
from src.logic.blink_detector import BlinkDetector
from src.logic.duration_timer import DurationTimer
from src.logic.perclos_calculator import PerclosCalculator
from src.logic.temporal_engine import TemporalEngine


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_frame(
    face_present: bool = True,
    signals_valid: bool = True,
    yaw: float = 0.0,
    pitch: float = 0.0,
    on_road: bool = True,
    phone_detected: bool = False,
    phone_conf: float = 0.0,
    mean_ear: float = 0.30,
    baseline_ear: float = 0.28,
    close_threshold: float = 0.21,
    ear_valid: bool = True,
    ear_calibrated: bool = True,
    timestamp_ns: int = 0,
    frame_id: int = 0,
) -> SignalFrame:
    head_pose = HeadPose(yaw_deg=yaw, pitch_deg=pitch, valid=True) if face_present else None
    gaze_world = GazeWorld(yaw_deg=yaw, pitch_deg=pitch, on_road=on_road, valid=True) if face_present else None
    eye_signals = EyeSignals(
        mean_EAR=mean_ear,
        left_EAR=mean_ear,
        right_EAR=mean_ear,
        baseline_EAR=baseline_ear,
        close_threshold=close_threshold,
        valid=ear_valid,
        calibration_complete=ear_calibrated,
    ) if face_present else None

    return SignalFrame(
        timestamp_ns=timestamp_ns,
        frame_id=frame_id,
        face_present=face_present,
        signals_valid=signals_valid and face_present,
        head_pose=head_pose,
        gaze_world=gaze_world,
        eye_signals=eye_signals,
        phone_signal=PhoneSignal(detected=phone_detected, confidence=phone_conf),
    )


# ── Test 1: Duration timer accumulates correctly ──────────────────────────────

class TestDurationTimer:
    def test_60_ticks_at_1_30th_second_gives_2_seconds(self):
        """60 ticks at 1/30s each → ~2.0s."""
        timer = DurationTimer()
        for _ in range(60):
            timer.tick(True, 1.0 / 30.0)
        assert timer.current_duration == pytest.approx(2.0, abs=0.1)

    def test_reset_on_inactive_tick(self):
        """30 true, 1 false, 30 true → only the last 30 ticks accumulate → ~1.0s."""
        timer = DurationTimer()
        for _ in range(30):
            timer.tick(True, 1.0 / 30.0)
        timer.tick(False, 1.0 / 30.0)  # reset
        for _ in range(30):
            timer.tick(True, 1.0 / 30.0)
        assert timer.current_duration == pytest.approx(1.0, abs=0.1)

    def test_delta_clamped_at_1_second(self):
        """Delta > 1.0 is clamped to 1.0."""
        timer = DurationTimer()
        timer.tick(True, 999.0)
        assert timer.current_duration == pytest.approx(1.0)

    def test_delta_negative_clamped_to_zero(self):
        """Negative delta is clamped to 0.0 — timer does not go backward."""
        timer = DurationTimer()
        timer.tick(True, -5.0)
        assert timer.current_duration == pytest.approx(0.0)

    def test_reset_method(self):
        timer = DurationTimer()
        timer.tick(True, 0.5)
        timer.reset()
        assert timer.current_duration == pytest.approx(0.0)


# ── Test 3 & 4: PERCLOS calculator ───────────────────────────────────────────

class TestPerclosCalculator:
    def test_perclos_12_of_60_closed_equals_0_20(self):
        """12 closed / 60 valid frames = 0.20."""
        pc = PerclosCalculator()
        for i in range(60):
            pc.update(eyes_closed=(i < 12), frame_valid=True)
        assert pc.perclos == pytest.approx(0.20)

    def test_perclos_fewer_than_30_valid_returns_zero(self):
        """Only 20 valid frames → valid=False, perclos=0.0."""
        pc = PerclosCalculator()
        for i in range(20):
            pc.update(eyes_closed=True, frame_valid=True)
        for i in range(40):
            pc.update(eyes_closed=False, frame_valid=False)
        assert pc.valid is False
        assert pc.perclos == pytest.approx(0.0)

    def test_perclos_valid_at_30_frames(self):
        """Exactly 30 valid frames → valid=True."""
        pc = PerclosCalculator()
        for _ in range(30):
            pc.update(eyes_closed=False, frame_valid=True)
        assert pc.valid is True

    def test_perclos_invalid_frames_excluded(self):
        """Invalid frames are not counted in denominator."""
        pc = PerclosCalculator()
        # 30 valid closed frames + 30 invalid frames
        for _ in range(30):
            pc.update(eyes_closed=True, frame_valid=True)
        for _ in range(30):
            pc.update(eyes_closed=False, frame_valid=False)
        # perclos = 30 closed / 30 valid = 1.0
        assert pc.perclos == pytest.approx(1.0)


# ── Test 5 & 6: Blink detector ───────────────────────────────────────────────

class TestBlinkDetector:
    def test_3_frame_closure_counts_as_1_blink(self):
        """EAR: open, closed×3, open → exactly 1 blink detected."""
        bd = BlinkDetector()
        threshold = 0.21
        blinks = []
        # Open
        blinks.append(bd.update(0.30, threshold, 1 / 30))
        # 3 closed frames
        blinks.append(bd.update(0.15, threshold, 1 / 30))
        blinks.append(bd.update(0.15, threshold, 1 / 30))
        blinks.append(bd.update(0.15, threshold, 1 / 30))
        # Re-open → blink detected here
        blinks.append(bd.update(0.30, threshold, 1 / 30))

        assert sum(blinks) == 1

    def test_20_frame_closure_not_a_blink(self):
        """20-frame closure exceeds BLINK_MAX_FRAMES=10 → 0 blinks."""
        bd = BlinkDetector()
        threshold = 0.21
        count = 0
        bd.update(0.30, threshold, 1 / 30)
        for _ in range(20):
            count += bd.update(0.15, threshold, 1 / 30)
        count += bd.update(0.30, threshold, 1 / 30)
        assert count == 0

    def test_blink_at_boundary_min_frames(self):
        """2-frame closure (= BLINK_MIN_FRAMES) is a valid blink."""
        bd = BlinkDetector()
        threshold = 0.21
        bd.update(0.30, threshold, 1 / 30)
        bd.update(0.15, threshold, 1 / 30)
        bd.update(0.15, threshold, 1 / 30)
        result = bd.update(0.30, threshold, 1 / 30)
        assert result is True

    def test_1_frame_closure_not_a_blink(self):
        """1-frame closure (< BLINK_MIN_FRAMES=2) is not a blink."""
        bd = BlinkDetector()
        threshold = 0.21
        bd.update(0.30, threshold, 1 / 30)
        bd.update(0.15, threshold, 1 / 30)
        result = bd.update(0.30, threshold, 1 / 30)
        assert result is False

    def test_no_blink_when_starting_already_below_threshold(self):
        """EAR starts below threshold with no prior above frame — no blink counted.

        A blink requires a genuine above→below crossover. If the very first frames
        are already below close_threshold there is no above→below transition and the
        subsequent rise above threshold must NOT be counted as a blink.
        """
        bd = BlinkDetector()
        threshold = 0.21
        count = 0
        # Start already below — no above frame seen first
        for _ in range(5):
            count += bd.update(0.15, threshold, 1 / 30)
        # Rise back above
        count += bd.update(0.30, threshold, 1 / 30)
        assert count == 0


# ── Test 7 & 8: Blink rate score ─────────────────────────────────────────────

class TestBlinkRateScore:
    def test_rate_below_low_threshold_gives_high_score(self):
        """rate=0.06 Hz < 0.13 → score ≈ 1 - (0.06/0.13) ≈ 0.54."""
        score = BlinkDetector.score_for_rate(0.06)
        assert score == pytest.approx(0.54, abs=0.02)

    def test_rate_in_normal_range_gives_zero(self):
        """rate=0.30 is within [0.13, 0.50] → score = 0.0."""
        score = BlinkDetector.score_for_rate(0.30)
        assert score == pytest.approx(0.0)

    def test_rate_zero_gives_score_1(self):
        """rate=0.0 → maximum drowsy score = 1.0."""
        score = BlinkDetector.score_for_rate(0.0)
        assert score == pytest.approx(1.0)

    def test_rate_above_high_threshold_gives_positive_score(self):
        """rate > 0.50 → positive fatigue score."""
        score = BlinkDetector.score_for_rate(1.00)
        assert score > 0.0

    def test_score_clamped_to_1(self):
        """Very high rate must not exceed 1.0."""
        score = BlinkDetector.score_for_rate(100.0)
        assert score == pytest.approx(1.0)


# ── Test 9: Full engine — 120 frames gaze off-road ────────────────────────────

class TestTemporalEngineGaze:
    def test_120_off_road_frames_gives_continuous_4s_and_fraction_1(self):
        """120 frames gaze off-road at 1/30s each → ~4.0s continuous, fraction ≈ 1.0."""
        te = TemporalEngine()
        result = None
        ns_per_frame = int(1e9 / 30)
        for i in range(120):
            frame = _make_frame(on_road=False, timestamp_ns=i * ns_per_frame)
            result = te.process(frame)
        assert result.gaze_continuous_secs == pytest.approx(4.0, abs=0.15)
        assert result.gaze_off_road_fraction == pytest.approx(1.0, abs=0.01)


# ── Test 10: Speed zone always URBAN ─────────────────────────────────────────

class TestSpeedZone:
    def test_speed_zone_urban_modifier_1(self):
        """MVP always returns URBAN zone with modifier=1.0."""
        te = TemporalEngine()
        result = te.process(_make_frame())
        assert result.speed_zone == 'URBAN'
        assert result.speed_modifier == pytest.approx(1.0)


# ── Test 11: Face absent — gaze/head timers stay at 0 ────────────────────────

class TestFaceAbsentTimers:
    def test_face_absent_ticks_only_face_absent_timer(self):
        """When face is absent, gaze/head timers do not accumulate."""
        te = TemporalEngine()
        ns_per_frame = int(1e9 / 30)
        for i in range(30):
            frame = _make_frame(face_present=False, timestamp_ns=i * ns_per_frame)
            result = te.process(frame)
        assert result.gaze_continuous_secs == pytest.approx(0.0)
        assert result.head_continuous_secs == pytest.approx(0.0)
        assert result.face_absent_continuous_secs > 0.0


# ── Test 12: face_absent_continuous_secs ─────────────────────────────────────

class TestFaceAbsentDuration:
    def test_60_absent_frames_gives_2_seconds(self):
        """60 absent frames at 1/30s each → ~2.0s continuous absence."""
        te = TemporalEngine()
        ns_per_frame = int(1e9 / 30)
        result = None
        for i in range(60):
            frame = _make_frame(face_present=False, timestamp_ns=i * ns_per_frame)
            result = te.process(frame)
        assert result.face_absent_continuous_secs == pytest.approx(2.0, abs=0.15)


# ── Test 13: perclos_valid False when fewer than 30 valid frames ──────────────

class TestPerclosValid:
    def test_perclos_valid_false_with_few_frames(self):
        """First 20 frames with eye signals → perclos_valid=False."""
        te = TemporalEngine()
        result = None
        for i in range(20):
            result = te.process(_make_frame(ear_calibrated=True, ear_valid=True))
        assert result.perclos_valid is False


# ── Test 14: None head_pose doesn't crash, head timer stays 0 ─────────────────

class TestNoneHandling:
    def test_head_pose_none_does_not_crash(self):
        """Signal frame with head_pose=None must not raise, head timer stays 0."""
        te = TemporalEngine()
        frame = SignalFrame(
            face_present=True,
            signals_valid=False,
            head_pose=None,
            gaze_world=None,
            eye_signals=None,
        )
        result = te.process(frame)
        assert result.head_continuous_secs == pytest.approx(0.0)

    def test_all_none_fields_do_not_crash(self):
        """Minimal SignalFrame with all optional fields None processes without error."""
        te = TemporalEngine()
        frame = SignalFrame()  # All defaults: face_present=False
        result = te.process(frame)
        assert result is not None
