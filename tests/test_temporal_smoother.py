"""Tests for the EMA temporal smoother."""

import pytest

from src.logic.temporal_smoother import TemporalSmoother


class TestTemporalSmoother:
    """Tests for TemporalSmoother EMA behavior."""

    def test_constant_input_converges(self) -> None:
        """EMA with constant input converges to that constant."""
        smoother = TemporalSmoother(alpha=0.3)
        for _ in range(50):
            result = smoother.update(0.6)
        assert abs(result - 0.6) < 1e-6

    def test_first_value_initializes(self) -> None:
        """First update sets the smoothed value directly."""
        smoother = TemporalSmoother(alpha=0.3)
        result = smoother.update(0.8)
        assert result == 0.8

    def test_step_input_ramps_smoothly(self) -> None:
        """Step input [0]*10 + [1]*10 ramps up without instantaneous jumps."""
        smoother = TemporalSmoother(alpha=0.3)
        results = []

        for _ in range(10):
            results.append(smoother.update(0.0))
        for _ in range(10):
            results.append(smoother.update(1.0))

        # First 10 should all be 0.0
        for v in results[:10]:
            assert v == pytest.approx(0.0, abs=1e-6)

        # After step to 1.0, values should increase gradually
        for i in range(11, len(results)):
            assert results[i] > results[i - 1], (
                f"Expected monotonic increase at index {i}: "
                f"{results[i]} > {results[i-1]}"
            )

        # No instantaneous jump to 1.0
        assert results[10] < 0.5, (
            f"First value after step should be < 0.5, got {results[10]}"
        )

        # Should approach but not reach 1.0 in 10 steps
        assert results[-1] < 1.0
        assert results[-1] > 0.5

    def test_step_down_ramps_smoothly(self) -> None:
        """Step from 1 to 0 ramps down gradually."""
        smoother = TemporalSmoother(alpha=0.3)
        for _ in range(20):
            smoother.update(1.0)

        results = []
        for _ in range(10):
            results.append(smoother.update(0.0))

        # Should decrease monotonically
        for i in range(1, len(results)):
            assert results[i] < results[i - 1]

        # Should approach 0 but not reach it in 10 steps
        assert results[-1] > 0.0
        assert results[-1] < 0.5

    def test_reset_clears_state(self) -> None:
        """After reset, next update re-initializes."""
        smoother = TemporalSmoother(alpha=0.3)
        smoother.update(0.9)
        smoother.update(0.9)
        smoother.reset()

        assert smoother.current_value == 0.0
        result = smoother.update(0.1)
        assert result == 0.1

    def test_current_value_before_any_update(self) -> None:
        """current_value is 0.0 before any update."""
        smoother = TemporalSmoother(alpha=0.3)
        assert smoother.current_value == 0.0

    def test_alpha_one_means_no_smoothing(self) -> None:
        """Alpha=1.0 means no smoothing; output equals input."""
        smoother = TemporalSmoother(alpha=1.0)
        smoother.update(0.5)
        result = smoother.update(0.9)
        assert result == pytest.approx(0.9)

    def test_low_alpha_heavy_smoothing(self) -> None:
        """Low alpha (0.05) results in very heavy smoothing."""
        smoother = TemporalSmoother(alpha=0.05)
        smoother.update(0.0)
        result = smoother.update(1.0)
        assert result < 0.1  # Barely moved

    def test_invalid_alpha_raises(self) -> None:
        """Alpha outside (0, 1] raises ValueError."""
        with pytest.raises(ValueError):
            TemporalSmoother(alpha=0.0)
        with pytest.raises(ValueError):
            TemporalSmoother(alpha=-0.1)
        with pytest.raises(ValueError):
            TemporalSmoother(alpha=1.5)

    def test_ema_formula_correctness(self) -> None:
        """Verify exact EMA formula: s = alpha * x + (1-alpha) * s_prev."""
        alpha = 0.3
        smoother = TemporalSmoother(alpha=alpha)

        v1 = smoother.update(0.5)
        assert v1 == pytest.approx(0.5)

        v2 = smoother.update(0.8)
        expected = alpha * 0.8 + (1 - alpha) * 0.5
        assert v2 == pytest.approx(expected)

        v3 = smoother.update(0.2)
        expected2 = alpha * 0.2 + (1 - alpha) * expected
        assert v3 == pytest.approx(expected2)
