"""Tests for src/logic/kalman_filter.py.

All tests use synthetic data — no camera, model file, or display required.
PRD §5.6 validation target: filter SHALL reduce std dev by >= 60% on
noisy signal with injected sigma = 5°.
"""

import numpy as np
import pytest

from src.logic.kalman_filter import KalmanFilter


class TestKalmanConvergence:
    def test_constant_signal_converges(self):
        """Constant 10.0 for 50 frames → output within 0.5 of 10.0."""
        kf = KalmanFilter()
        output = None
        for _ in range(50):
            output = kf.update(10.0)
        assert abs(output - 10.0) < 0.5

    def test_first_frame_near_input(self):
        """First frame output must be within 1.0 of the input measurement."""
        kf = KalmanFilter()
        out = kf.update(25.0)
        assert abs(out - 25.0) < 1.0

    def test_first_frame_negative_input(self):
        kf = KalmanFilter()
        out = kf.update(-15.0)
        assert abs(out - (-15.0)) < 1.0


class TestKalmanNoiseSuppression:
    def test_noisy_signal_std_reduced_by_60_percent(self):
        """PRD §5.6 validation: std dev reduced >= 60% on noisy signal.

        Input: 10.0 + Gaussian noise (sigma=5.0, seed=42), 200 frames.
        Output std must be <= 40% of input std (i.e. >= 60% reduction).
        """
        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, 5.0, 200)
        signal = 10.0 + noise

        kf = KalmanFilter()
        outputs = [kf.update(float(v)) for v in signal]

        input_std = float(np.std(signal))
        output_std = float(np.std(outputs))

        assert output_std <= input_std * 0.40, (
            f"Expected >= 60% reduction, got input_std={input_std:.3f} "
            f"output_std={output_std:.3f} ({(1 - output_std/input_std)*100:.1f}% reduction)"
        )


class TestKalmanReset:
    def test_reset_reinitializes_on_next_update(self):
        """After reset(), the next update re-initializes to the new value."""
        kf = KalmanFilter()
        for _ in range(30):
            kf.update(50.0)

        kf.reset()
        assert not kf.is_initialized

        out = kf.update(0.0)
        assert abs(out - 0.0) < 1.0

    def test_reset_converges_to_new_constant(self):
        """After reset, filter converges to a new constant within reasonable frames."""
        kf = KalmanFilter()
        for _ in range(50):
            kf.update(-20.0)

        kf.reset()
        output = None
        for _ in range(50):
            output = kf.update(20.0)

        assert abs(output - 20.0) < 1.0

    def test_is_initialized_false_after_reset(self):
        kf = KalmanFilter()
        kf.update(5.0)
        assert kf.is_initialized
        kf.reset()
        assert not kf.is_initialized


class TestKalmanRamp:
    def test_linear_ramp_tracks_within_tolerance(self):
        """Linear ramp 0→30 over 60 frames → final output within 10.0 of 30.0.

        Tolerance reflects R=25.0 (high smoothing): the filter lags a slow ramp
        by ~6-7° which is acceptable — sudden real movements are tracked faster
        because the velocity state component updates continuously.
        """
        kf = KalmanFilter()
        values = np.linspace(0.0, 30.0, 60)
        output = None
        for v in values:
            output = kf.update(float(v))
        assert abs(output - 30.0) < 10.0


class TestKalmanPredictOnly:
    def test_none_measurement_before_init_returns_zero(self):
        """Before initialization, None measurement returns 0.0."""
        kf = KalmanFilter()
        out = kf.update(None)
        assert out == pytest.approx(0.0)
        assert not kf.is_initialized

    def test_none_measurement_after_init_predicts(self):
        """After initialization, None advances state without correction."""
        kf = KalmanFilter()
        kf.update(10.0)
        # Several predict-only steps should stay near 10.0 (velocity ≈ 0)
        for _ in range(5):
            out = kf.update(None)
        assert abs(out - 10.0) < 3.0

    def test_none_does_not_corrupt_state(self):
        """Predict-only steps followed by a real measurement still converge."""
        kf = KalmanFilter()
        kf.update(10.0)
        for _ in range(10):
            kf.update(None)
        # Re-anchor at 10.0
        for _ in range(20):
            out = kf.update(10.0)
        assert abs(out - 10.0) < 1.0


class TestKalmanProperties:
    def test_current_value_before_init(self):
        kf = KalmanFilter()
        assert kf.current_value == pytest.approx(0.0)

    def test_current_value_after_update(self):
        kf = KalmanFilter()
        out = kf.update(7.5)
        assert kf.current_value == pytest.approx(out)

    def test_is_initialized_false_before_update(self):
        kf = KalmanFilter()
        assert not kf.is_initialized

    def test_is_initialized_true_after_update(self):
        kf = KalmanFilter()
        kf.update(1.0)
        assert kf.is_initialized


class TestKalmanParameters:
    def test_custom_parameters_accepted(self):
        """Constructor accepts explicit overrides without error."""
        kf = KalmanFilter(dt=1 / 30, process_noise=0.05, measurement_noise=2.0,
                          initial_covariance=0.5)
        out = kf.update(15.0)
        assert abs(out - 15.0) < 1.0

    def test_higher_measurement_noise_smooths_more(self):
        """Higher R → more smoothing (output closer to prior, less to measurement)."""
        rng = np.random.default_rng(7)
        signal = [10.0 + float(rng.normal(0, 5)) for _ in range(100)]

        kf_low_r = KalmanFilter(measurement_noise=1.0)
        kf_high_r = KalmanFilter(measurement_noise=20.0)

        out_low = [kf_low_r.update(v) for v in signal]
        out_high = [kf_high_r.update(v) for v in signal]

        assert np.std(out_high) < np.std(out_low)

    def test_multiple_independent_instances(self):
        """Two KalmanFilter instances maintain independent state."""
        kf1 = KalmanFilter()
        kf2 = KalmanFilter()

        for _ in range(20):
            kf1.update(10.0)
        for _ in range(20):
            kf2.update(-10.0)

        assert kf1.current_value > 0
        assert kf2.current_value < 0
