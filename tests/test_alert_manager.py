"""Tests for the alert manager."""

import time

import pytest

from src.config_loader import AlertManagerConfig
from src.logic.alert_manager import AlertManager
from src.logic.distraction_reasoner import DistractionAssessment
from src.logic.hysteresis import DriverState


def make_assessment(
    is_distracted: bool = False,
    state: DriverState = DriverState.ATTENTIVE,
    confidence: float = 0.0,
    triggers: list = None,
) -> DistractionAssessment:
    """Create a test DistractionAssessment."""
    return DistractionAssessment(
        state=state,
        is_distracted=is_distracted,
        is_monitoring=(state == DriverState.MONITORING),
        confidence=confidence,
        active_triggers=triggers or [],
        confirmed_objects=[],
        timestamp=time.time(),
    )


def make_manager(
    sustained_frames: int = 10,
    sustained_ratio: float = 0.80,
    cooldown_seconds: float = 0.0,
) -> AlertManager:
    """Create an AlertManager with given config."""
    config = AlertManagerConfig(
        sustained_frames=sustained_frames,
        sustained_ratio=sustained_ratio,
        cooldown_seconds=cooldown_seconds,
    )
    return AlertManager(config)


class TestAlertManager:
    """Tests for the AlertManager sustained distraction logic."""

    def test_sustained_distraction_triggers_alert(self) -> None:
        """Alert fires after sustained distraction meets ratio threshold."""
        manager = make_manager(sustained_frames=10, sustained_ratio=0.80)

        # 10 distracted frames (100% ratio)
        alert = None
        for _ in range(10):
            alert = manager.update(
                make_assessment(
                    is_distracted=True,
                    state=DriverState.DISTRACTED,
                    confidence=0.8,
                    triggers=["phone_use"],
                )
            )

        assert alert is not None
        assert alert.level == DriverState.DISTRACTED
        assert alert.confidence == pytest.approx(0.8)

    def test_below_ratio_no_alert(self) -> None:
        """No alert when distracted ratio is below threshold."""
        manager = make_manager(sustained_frames=10, sustained_ratio=0.80)

        # 7 distracted, 3 attentive (70% < 80%)
        for _ in range(7):
            manager.update(
                make_assessment(is_distracted=True, state=DriverState.DISTRACTED)
            )
        for _ in range(3):
            alert = manager.update(
                make_assessment(is_distracted=False, state=DriverState.ATTENTIVE)
            )

        assert alert is None

    def test_cooldown_suppresses_repeated_alerts(self) -> None:
        """Cooldown prevents alerts from firing too frequently."""
        manager = make_manager(
            sustained_frames=5,
            sustained_ratio=0.80,
            cooldown_seconds=1.0,
        )

        # First alert
        for _ in range(5):
            alert = manager.update(
                make_assessment(is_distracted=True, state=DriverState.DISTRACTED)
            )
        assert alert is not None

        # Immediately try to trigger another alert
        for _ in range(5):
            alert = manager.update(
                make_assessment(is_distracted=True, state=DriverState.DISTRACTED)
            )
        assert alert is None  # Suppressed by cooldown

    def test_cooldown_expires(self) -> None:
        """Alert fires again after cooldown period elapses."""
        manager = make_manager(
            sustained_frames=5,
            sustained_ratio=0.80,
            cooldown_seconds=0.1,
        )

        # First alert
        for _ in range(5):
            manager.update(
                make_assessment(is_distracted=True, state=DriverState.DISTRACTED)
            )

        # Wait for cooldown
        time.sleep(0.15)

        # Second alert should fire on the first update after cooldown
        second_alert = None
        for _ in range(5):
            result = manager.update(
                make_assessment(is_distracted=True, state=DriverState.DISTRACTED)
            )
            if result is not None:
                second_alert = result
        assert second_alert is not None

    def test_monitoring_frames_not_counted(self) -> None:
        """MONITORING frames are recorded as not-distracted."""
        manager = make_manager(sustained_frames=10, sustained_ratio=0.80)

        # 8 distracted + 2 monitoring = only 80% but monitoring doesn't count as distracted
        for _ in range(8):
            manager.update(
                make_assessment(is_distracted=True, state=DriverState.DISTRACTED)
            )

        # MONITORING frames: is_distracted is False since MONITORING != DISTRACTED
        for _ in range(2):
            alert = manager.update(
                make_assessment(
                    is_distracted=False,
                    state=DriverState.MONITORING,
                )
            )

        # 8/10 = 80%, which equals sustained_ratio, should fire
        assert alert is not None

    def test_monitoring_blocks_alert(self) -> None:
        """Too many MONITORING frames prevent alert from firing."""
        manager = make_manager(sustained_frames=10, sustained_ratio=0.80)

        # 7 distracted + 3 monitoring = 70% < 80%
        for _ in range(7):
            manager.update(
                make_assessment(is_distracted=True, state=DriverState.DISTRACTED)
            )
        for _ in range(3):
            alert = manager.update(
                make_assessment(is_distracted=False, state=DriverState.MONITORING)
            )

        assert alert is None

    def test_not_enough_frames(self) -> None:
        """No alert when history buffer is not yet full."""
        manager = make_manager(sustained_frames=10, sustained_ratio=0.80)

        for _ in range(9):
            alert = manager.update(
                make_assessment(is_distracted=True, state=DriverState.DISTRACTED)
            )

        assert alert is None  # Only 9 frames, need 10

    def test_reset(self) -> None:
        """Reset clears history and cooldown."""
        manager = make_manager(sustained_frames=5, sustained_ratio=0.80)

        for _ in range(5):
            manager.update(
                make_assessment(is_distracted=True, state=DriverState.DISTRACTED)
            )

        manager.reset()

        # After reset, no alert should fire (history cleared)
        alert = manager.update(
            make_assessment(is_distracted=True, state=DriverState.DISTRACTED)
        )
        assert alert is None

    def test_alert_contains_triggers(self) -> None:
        """Alert includes the triggers from the assessment."""
        manager = make_manager(sustained_frames=3, sustained_ratio=0.80)

        alert = None
        for _ in range(3):
            alert = manager.update(
                make_assessment(
                    is_distracted=True,
                    state=DriverState.DISTRACTED,
                    confidence=0.85,
                    triggers=["phone_use", "distracted_posture"],
                )
            )

        assert alert is not None
        assert "phone_use" in alert.triggers
