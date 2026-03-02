"""Tests for the hysteresis state machine."""

import pytest

from src.config_loader import HysteresisConfig
from src.logic.hysteresis import DriverState, HysteresisStateMachine


def make_config(
    enter_distracted: float = 0.75,
    leave_distracted: float = 0.55,
    enter_monitoring: float = 0.40,
    leave_monitoring: float = 0.35,
) -> HysteresisConfig:
    """Create a HysteresisConfig with specified thresholds."""
    return HysteresisConfig(
        enter_distracted_threshold=enter_distracted,
        leave_distracted_threshold=leave_distracted,
        enter_monitoring_threshold=enter_monitoring,
        leave_monitoring_threshold=leave_monitoring,
    )


class TestHysteresisStateMachine:
    """Tests for the 3-state hysteresis state machine."""

    def test_starts_attentive(self) -> None:
        """Initial state is ATTENTIVE."""
        sm = HysteresisStateMachine(make_config())
        assert sm.current_state == DriverState.ATTENTIVE

    def test_attentive_to_monitoring(self) -> None:
        """ATTENTIVE -> MONITORING when prob >= enter_monitoring_threshold."""
        sm = HysteresisStateMachine(make_config())
        state = sm.update(0.45)
        assert state == DriverState.MONITORING

    def test_attentive_stays_below_monitoring(self) -> None:
        """Stays ATTENTIVE when prob is below enter_monitoring_threshold."""
        sm = HysteresisStateMachine(make_config())
        state = sm.update(0.30)
        assert state == DriverState.ATTENTIVE

    def test_attentive_to_distracted_directly(self) -> None:
        """ATTENTIVE -> DISTRACTED when prob >= enter_distracted_threshold."""
        sm = HysteresisStateMachine(make_config())
        state = sm.update(0.80)
        assert state == DriverState.DISTRACTED

    def test_monitoring_to_distracted(self) -> None:
        """MONITORING -> DISTRACTED when prob >= enter_distracted_threshold."""
        sm = HysteresisStateMachine(make_config())
        sm.update(0.45)  # -> MONITORING
        state = sm.update(0.80)
        assert state == DriverState.DISTRACTED

    def test_monitoring_to_attentive(self) -> None:
        """MONITORING -> ATTENTIVE when prob < leave_monitoring_threshold."""
        sm = HysteresisStateMachine(make_config())
        sm.update(0.45)  # -> MONITORING
        state = sm.update(0.30)
        assert state == DriverState.ATTENTIVE

    def test_monitoring_stays_in_band(self) -> None:
        """MONITORING stays when prob is between leave_monitoring and enter_distracted."""
        sm = HysteresisStateMachine(make_config())
        sm.update(0.45)  # -> MONITORING
        state = sm.update(0.50)
        assert state == DriverState.MONITORING

    def test_distracted_to_monitoring(self) -> None:
        """DISTRACTED -> MONITORING when prob <= leave_distracted_threshold."""
        sm = HysteresisStateMachine(make_config())
        sm.update(0.80)  # -> DISTRACTED
        state = sm.update(0.50)
        assert state == DriverState.MONITORING

    def test_distracted_stays_above_leave(self) -> None:
        """DISTRACTED stays when prob > leave_distracted_threshold."""
        sm = HysteresisStateMachine(make_config())
        sm.update(0.80)  # -> DISTRACTED
        state = sm.update(0.60)
        assert state == DriverState.DISTRACTED

    def test_distracted_cannot_skip_to_attentive(self) -> None:
        """DISTRACTED must pass through MONITORING; cannot go directly to ATTENTIVE."""
        sm = HysteresisStateMachine(make_config())
        sm.update(0.80)  # -> DISTRACTED

        # Even a very low value goes to MONITORING first
        state = sm.update(0.10)
        assert state == DriverState.MONITORING

        # Then from MONITORING back to ATTENTIVE
        state = sm.update(0.10)
        assert state == DriverState.ATTENTIVE

    def test_full_cycle(self) -> None:
        """Full cycle: ATTENTIVE -> MONITORING -> DISTRACTED -> MONITORING -> ATTENTIVE."""
        sm = HysteresisStateMachine(make_config())

        assert sm.update(0.10) == DriverState.ATTENTIVE
        assert sm.update(0.45) == DriverState.MONITORING
        assert sm.update(0.80) == DriverState.DISTRACTED
        assert sm.update(0.50) == DriverState.MONITORING
        assert sm.update(0.30) == DriverState.ATTENTIVE

    def test_no_rapid_toggling_at_monitoring_boundary(self) -> None:
        """Oscillating input around monitoring threshold does not cause rapid toggling."""
        sm = HysteresisStateMachine(make_config())

        # Enter MONITORING
        sm.update(0.42)
        assert sm.current_state == DriverState.MONITORING

        # Oscillate between 0.36 and 0.39 (between leave and enter)
        # Should stay in MONITORING
        states = []
        for v in [0.36, 0.39, 0.36, 0.39, 0.36, 0.39]:
            states.append(sm.update(v))

        assert all(s == DriverState.MONITORING for s in states)

    def test_no_rapid_toggling_at_distracted_boundary(self) -> None:
        """Oscillating around distracted threshold stays stable."""
        sm = HysteresisStateMachine(make_config())
        sm.update(0.80)  # -> DISTRACTED

        # Oscillate between 0.56 and 0.74 (between leave and enter)
        states = []
        for v in [0.60, 0.70, 0.60, 0.70, 0.60]:
            states.append(sm.update(v))

        assert all(s == DriverState.DISTRACTED for s in states)

    def test_reset(self) -> None:
        """Reset returns to ATTENTIVE."""
        sm = HysteresisStateMachine(make_config())
        sm.update(0.80)
        assert sm.current_state == DriverState.DISTRACTED
        sm.reset()
        assert sm.current_state == DriverState.ATTENTIVE

    def test_exact_threshold_values(self) -> None:
        """Test behavior at exact threshold boundaries."""
        sm = HysteresisStateMachine(make_config())

        # Exactly at enter_monitoring (0.40) -> should enter MONITORING
        state = sm.update(0.40)
        assert state == DriverState.MONITORING

        # Exactly at enter_distracted (0.75) -> should enter DISTRACTED
        state = sm.update(0.75)
        assert state == DriverState.DISTRACTED

        # Exactly at leave_distracted (0.55) -> should leave to MONITORING
        state = sm.update(0.55)
        assert state == DriverState.MONITORING

        # Exactly at leave_monitoring (0.35) -> should stay MONITORING
        # (leave requires STRICTLY less than threshold)
        state = sm.update(0.35)
        assert state == DriverState.MONITORING

        # Just below leave_monitoring -> ATTENTIVE
        state = sm.update(0.34)
        assert state == DriverState.ATTENTIVE
