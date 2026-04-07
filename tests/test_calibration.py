"""Tests for src/logic/calibration.py.

All tests use synthetic data — no camera or model files required.
PRD §23.
"""

import pytest

from src.logic.calibration import Calibration


# ── Test 1: Stable calibration succeeds ───────────────────────────────────────

class TestStableCalibration:
    def test_150_stable_frames_completes(self):
        """150 frames with constant values → status='complete', quality_ok=True."""
        cal = Calibration(target_duration_s=5.0, fps=30.0)
        done = False
        for _ in range(150):
            done = cal.feed_frame(head_yaw=5.0, head_pitch=3.0, mean_ear=0.30, face_visible=True)
        assert done is True
        assert cal.status == 'complete'
        assert cal.is_complete is True
        assert cal.quality_ok is True

    def test_offsets_correct_on_stable_signal(self):
        """Constant yaw=5, pitch=3 → neutral offsets match exactly."""
        cal = Calibration(target_duration_s=5.0, fps=30.0)
        for _ in range(150):
            cal.feed_frame(5.0, 3.0, 0.30, True)
        assert cal.neutral_yaw_offset == pytest.approx(5.0)
        assert cal.neutral_pitch_offset == pytest.approx(3.0)

    def test_baseline_ear_correct(self):
        """Constant ear=0.30 → baseline_ear=0.30."""
        cal = Calibration(target_duration_s=5.0, fps=30.0)
        for _ in range(150):
            cal.feed_frame(5.0, 3.0, 0.30, True)
        assert cal.baseline_ear == pytest.approx(0.30)

    def test_close_threshold_is_75_percent_of_baseline(self):
        """close_threshold = baseline_ear * 0.75 = 0.30 * 0.75 = 0.225."""
        cal = Calibration(target_duration_s=5.0, fps=30.0)
        for _ in range(150):
            cal.feed_frame(5.0, 3.0, 0.30, True)
        assert cal.close_threshold == pytest.approx(0.225)


# ── Test 2: Noisy calibration fails quality gate ──────────────────────────────

class TestNoisyCalibration:
    def test_yaw_std_8_fails_quality(self):
        """Alternating yaw ±8° from mean → std=8 > 5° threshold → failed."""
        cal = Calibration(target_duration_s=5.0, fps=30.0)
        # Alternate 13.0 / -3.0 → mean=5.0, std=8.0
        for i in range(150):
            yaw = 13.0 if i % 2 == 0 else -3.0
            cal.feed_frame(yaw, 3.0, 0.30, True)
        assert cal.status == 'failed'
        assert cal.quality_ok is False

    def test_noisy_yaw_returns_defaults(self):
        """Failed calibration returns 0.0 offsets and default 0.21 threshold."""
        cal = Calibration(target_duration_s=5.0, fps=30.0)
        for i in range(150):
            yaw = 13.0 if i % 2 == 0 else -3.0
            cal.feed_frame(yaw, 3.0, 0.30, True)
        assert cal.neutral_yaw_offset == pytest.approx(0.0)
        assert cal.close_threshold == pytest.approx(0.21)

    def test_noisy_pitch_fails_quality(self):
        """Alternating pitch ±8° → std=8 > 5° → failed even if yaw is stable."""
        cal = Calibration(target_duration_s=5.0, fps=30.0)
        for i in range(150):
            pitch = 13.0 if i % 2 == 0 else -3.0
            cal.feed_frame(5.0, pitch, 0.30, True)
        assert cal.status == 'failed'


# ── Test 3: Insufficient visible frames fails ─────────────────────────────────

class TestInsufficientFrames:
    def test_100_invisible_out_of_150_fails(self):
        """150 frames total with 100 face_visible=False → only 50 valid → fails."""
        cal = Calibration(target_duration_s=5.0, fps=30.0)
        # First 100 frames: face not visible
        for _ in range(100):
            cal.feed_frame(5.0, 3.0, 0.30, face_visible=False)
        # Next 50 frames: face visible (total: 50 valid out of 150 total)
        done = False
        for _ in range(50):
            done = cal.feed_frame(5.0, 3.0, 0.30, face_visible=True)
        assert done is True
        assert cal.status == 'failed'

    def test_hard_timeout_fails(self):
        """Face never visible → early failure fires at expected_frames (150), not hard timeout (300).

        When face_visible=False every frame, valid_count stays 0. At frame 150
        (== expected_frames), the early-failure check fires before the 300-frame
        hard timeout is ever reached. Feeding all 300 frames still results in
        status='failed'; the extra 150 calls are no-ops because status is already terminal.
        """
        cal = Calibration(target_duration_s=5.0, fps=30.0)
        done = False
        for _ in range(300):
            done = cal.feed_frame(5.0, 3.0, 0.30, face_visible=False)
        assert done is True
        assert cal.status == 'failed'


# ── Test 4: Failed calibration returns safe defaults ─────────────────────────

class TestFailureDefaults:
    def test_after_failure_offsets_are_zero(self):
        """After any failure mode, offsets must be (0.0, 0.0)."""
        cal = Calibration(target_duration_s=5.0, fps=30.0)
        for _ in range(150):
            cal.feed_frame(5.0, 3.0, 0.30, face_visible=False)
        assert cal.neutral_yaw_offset == pytest.approx(0.0)
        assert cal.neutral_pitch_offset == pytest.approx(0.0)

    def test_after_failure_threshold_is_default_0_21(self):
        """After failure, close_threshold = EAR_DEFAULT_CLOSE_THRESHOLD = 0.21."""
        cal = Calibration(target_duration_s=5.0, fps=30.0)
        for _ in range(150):
            cal.feed_frame(5.0, 3.0, 0.30, face_visible=False)
        assert cal.close_threshold == pytest.approx(0.21)

    def test_feed_after_complete_returns_true_without_changing_state(self):
        """Calling feed_frame after completion is a no-op, returns True."""
        cal = Calibration(target_duration_s=5.0, fps=30.0)
        for _ in range(150):
            cal.feed_frame(5.0, 3.0, 0.30, True)
        assert cal.status == 'complete'
        result = cal.feed_frame(99.0, 99.0, 0.01, True)
        assert result is True
        assert cal.status == 'complete'
        assert cal.neutral_yaw_offset == pytest.approx(5.0)  # unchanged


# ── Test 5: Reset works ───────────────────────────────────────────────────────

class TestReset:
    def test_reset_after_complete_returns_to_collecting(self):
        """After completing, reset() returns status to 'collecting'."""
        cal = Calibration(target_duration_s=5.0, fps=30.0)
        for _ in range(150):
            cal.feed_frame(5.0, 3.0, 0.30, True)
        assert cal.status == 'complete'
        cal.reset()
        assert cal.status == 'collecting'
        assert cal.is_complete is False

    def test_reset_clears_offsets(self):
        """After reset, offsets and thresholds return to defaults."""
        cal = Calibration(target_duration_s=5.0, fps=30.0)
        for _ in range(150):
            cal.feed_frame(5.0, 3.0, 0.30, True)
        cal.reset()
        assert cal.neutral_yaw_offset == pytest.approx(0.0)
        assert cal.neutral_pitch_offset == pytest.approx(0.0)
        assert cal.close_threshold == pytest.approx(0.21)

    def test_reset_allows_new_calibration(self):
        """After reset, fresh calibration data overrides previous values."""
        cal = Calibration(target_duration_s=5.0, fps=30.0)
        for _ in range(150):
            cal.feed_frame(5.0, 3.0, 0.30, True)
        cal.reset()
        for _ in range(150):
            cal.feed_frame(10.0, 7.0, 0.25, True)
        assert cal.status == 'complete'
        assert cal.neutral_yaw_offset == pytest.approx(10.0)
        assert cal.neutral_pitch_offset == pytest.approx(7.0)
        assert cal.baseline_ear == pytest.approx(0.25)

    def test_zero_ear_falls_back_to_default_baseline(self):
        """150 frames with mean_ear=0.0 → EAR filtered out; baseline and threshold use defaults."""
        cal = Calibration(target_duration_s=5.0, fps=30.0)
        for _ in range(150):
            cal.feed_frame(5.0, 3.0, mean_ear=0.0, face_visible=True)
        # Pose calibration should still succeed (yaw/pitch are stable)
        assert cal.status == 'complete'
        assert cal.quality_ok is True
        assert cal.neutral_yaw_offset == pytest.approx(5.0)
        assert cal.neutral_pitch_offset == pytest.approx(3.0)
        # EAR defaults because all samples were filtered
        assert cal.baseline_ear == pytest.approx(0.28)
        assert cal.close_threshold == pytest.approx(0.21)

    def test_reset_after_failure_allows_retry(self):
        """After a failed calibration, reset + stable data succeeds."""
        cal = Calibration(target_duration_s=5.0, fps=30.0)
        for _ in range(150):
            cal.feed_frame(5.0, 3.0, 0.30, face_visible=False)
        assert cal.status == 'failed'
        cal.reset()
        for _ in range(150):
            cal.feed_frame(5.0, 3.0, 0.30, True)
        assert cal.status == 'complete'
        assert cal.quality_ok is True


# ── Test 6: Calibration diagnostics ──────────────────────────────────────────

class TestCalibrationDiagnostics:
    def test_failure_reason_insufficient_face_frames(self):
        """Early failure (too few visible frames) sets failure_reason='insufficient_face_frames'."""
        cal = Calibration(target_duration_s=5.0, fps=30.0)
        for _ in range(150):
            cal.feed_frame(5.0, 3.0, 0.30, face_visible=False)
        assert cal.status == 'failed'
        assert cal.failure_reason == 'insufficient_face_frames'

    def test_failure_reason_pose_std_too_high(self):
        """Quality gate failure (std >= 5°) sets failure_reason='pose_std_too_high'."""
        cal = Calibration(target_duration_s=5.0, fps=30.0)
        # Alternating yaw 13°/-3° → mean=5°, std=8° > 5° threshold
        for i in range(150):
            yaw = 13.0 if i % 2 == 0 else -3.0
            cal.feed_frame(yaw, 3.0, 0.30, True)
        assert cal.status == 'failed'
        assert cal.failure_reason == 'pose_std_too_high'

    def test_frame_count_properties_mid_calibration(self):
        """valid_frame_count and total_frame_count track correctly during collection."""
        cal = Calibration(target_duration_s=5.0, fps=30.0)
        # 20 invisible frames then 30 visible — still collecting
        for _ in range(20):
            cal.feed_frame(0.0, 0.0, 0.0, face_visible=False)
        for _ in range(30):
            cal.feed_frame(5.0, 3.0, 0.30, face_visible=True)
        assert cal.total_frame_count == 50
        assert cal.valid_frame_count == 30

    def test_min_valid_frame_count_and_expected_frame_count(self):
        """min_valid_frame_count == 135 and expected_frame_count == 150 for 5s at 30fps."""
        cal = Calibration(target_duration_s=5.0, fps=30.0)
        assert cal.min_valid_frame_count == 135   # 90% of 150
        assert cal.expected_frame_count == 150

    def test_reset_clears_failure_reason(self):
        """reset() clears failure_reason back to ''."""
        cal = Calibration(target_duration_s=5.0, fps=30.0)
        for _ in range(150):
            cal.feed_frame(5.0, 3.0, 0.30, face_visible=False)
        assert cal.failure_reason == 'insufficient_face_frames'
        cal.reset()
        assert cal.failure_reason == ''

    def test_reset_clears_frame_counts(self):
        """reset() resets valid_frame_count and total_frame_count to 0."""
        cal = Calibration(target_duration_s=5.0, fps=30.0)
        for _ in range(50):
            cal.feed_frame(5.0, 3.0, 0.30, face_visible=True)
        cal.reset()
        assert cal.valid_frame_count == 0
        assert cal.total_frame_count == 0

    def test_failure_reason_empty_on_success(self):
        """Successful calibration leaves failure_reason as ''."""
        cal = Calibration(target_duration_s=5.0, fps=30.0)
        for _ in range(150):
            cal.feed_frame(5.0, 3.0, 0.30, True)
        assert cal.status == 'complete'
        assert cal.failure_reason == ''
