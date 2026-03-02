"""End-to-end integration tests for the Attentia Drive logic pipeline.

Feeds synthetic ClassifierResult and ObjectDetection sequences through
the full logic chain (smoother -> hysteresis -> confirmer -> reasoner ->
alert manager) and asserts expected state transitions and alert timing.

Does NOT require a camera, model files, or OpenCV display.
"""

import time

import pytest

from src.config_loader import (
    AlertManagerConfig,
    HysteresisConfig,
    ObjectConfirmationConfig,
    TemporalSmoothingConfig,
)
from src.detection.classifier import ClassifierResult
from src.detection.drowsiness import DrowsinessResult
from src.detection.face_detector import FaceResult
from src.detection.object_detector import ObjectDetection
from src.logic.alert_manager import AlertManager
from src.logic.distraction_reasoner import DistractionReasoner
from src.logic.hysteresis import DriverState, HysteresisStateMachine
from src.logic.object_confirmer import ObjectTemporalConfirmer
from src.logic.temporal_smoother import TemporalSmoother


class LogicPipeline:
    """Simplified logic pipeline for testing (no camera/display/models)."""

    def __init__(
        self,
        alpha: float = 0.3,
        n: int = 3,
        m: int = 5,
        sustained_frames: int = 10,
        sustained_ratio: float = 0.80,
        cooldown_seconds: float = 0.0,
    ) -> None:
        self.smoother = TemporalSmoother(alpha=alpha)
        self.hysteresis = HysteresisStateMachine(HysteresisConfig())
        self.confirmer = ObjectTemporalConfirmer(
            ObjectConfirmationConfig(enabled=True, n=n, m=m)
        )
        self.reasoner = DistractionReasoner(face_detector_enabled=False)
        self.alert_manager = AlertManager(
            AlertManagerConfig(
                sustained_frames=sustained_frames,
                sustained_ratio=sustained_ratio,
                cooldown_seconds=cooldown_seconds,
            )
        )

    def step(
        self,
        raw_prob: float,
        detections: list = None,
    ) -> dict:
        """Run one frame through the logic pipeline.

        Returns dict with smoothed_prob, state, assessment, alert.
        """
        if detections is None:
            detections = []

        smoothed = self.smoother.update(raw_prob)
        state = self.hysteresis.update(smoothed)
        confirmed = self.confirmer.update(detections)

        assessment = self.reasoner.assess(
            driver_state=state,
            smoothed_probability=smoothed,
            confirmed_objects=confirmed,
            face_result=FaceResult(),
            drowsiness_result=DrowsinessResult(),
        )

        alert = self.alert_manager.update(assessment)

        return {
            "smoothed_prob": smoothed,
            "state": state,
            "assessment": assessment,
            "alert": alert,
        }


class TestPipelineIntegration:
    """End-to-end logic pipeline tests."""

    def test_all_zeros_stays_attentive(self) -> None:
        """Zero probability throughout -> stays ATTENTIVE, no alerts."""
        pipeline = LogicPipeline()

        for _ in range(50):
            result = pipeline.step(0.0)
            assert result["state"] == DriverState.ATTENTIVE
            assert result["alert"] is None

    def test_gradual_distraction_transition(self) -> None:
        """Gradual increase in probability -> transitions through states."""
        pipeline = LogicPipeline(alpha=0.5)

        states_seen = set()
        for _ in range(50):
            result = pipeline.step(0.9)
            states_seen.add(result["state"])

        assert DriverState.DISTRACTED in states_seen

    def test_sustained_distraction_fires_alert(self) -> None:
        """Sustained high probability -> alert fires."""
        pipeline = LogicPipeline(
            alpha=0.8,
            sustained_frames=10,
            sustained_ratio=0.80,
        )

        alert_fired = False
        for _ in range(30):
            result = pipeline.step(0.95)
            if result["alert"] is not None:
                alert_fired = True
                break

        assert alert_fired, "Expected alert to fire after sustained distraction"

    def test_brief_distraction_no_alert(self) -> None:
        """Brief spike in probability -> no alert."""
        pipeline = LogicPipeline(
            alpha=0.3,
            sustained_frames=20,
            sustained_ratio=0.80,
        )

        # Brief spike
        for _ in range(5):
            pipeline.step(0.9)

        # Back to normal
        for _ in range(30):
            result = pipeline.step(0.0)

        assert result["alert"] is None

    def test_phone_detection_creates_trigger(self) -> None:
        """Phone detected enough times -> phone_use trigger in assessment."""
        pipeline = LogicPipeline(alpha=0.8, n=3, m=5)

        phone = ObjectDetection(
            class_name="cell phone", confidence=0.9, bbox=(10, 10, 100, 100)
        )

        for _ in range(5):
            result = pipeline.step(0.9, detections=[phone])

        assert "phone_use" in result["assessment"].active_triggers

    def test_object_confirmation_timing(self) -> None:
        """Object must appear in N of M frames to be confirmed."""
        pipeline = LogicPipeline(alpha=0.5, n=3, m=5)

        phone = ObjectDetection(
            class_name="cell phone", confidence=0.9, bbox=(10, 10, 100, 100)
        )

        # Frame 1: phone present, not yet confirmed
        result = pipeline.step(0.5, detections=[phone])
        assert "cell phone" not in result["assessment"].confirmed_objects

        # Frame 2: phone present
        result = pipeline.step(0.5, detections=[phone])
        assert "cell phone" not in result["assessment"].confirmed_objects

        # Frame 3: phone present -> now confirmed (3 of 3)
        result = pipeline.step(0.5, detections=[phone])
        assert "cell phone" in result["assessment"].confirmed_objects

    def test_recovery_from_distraction(self) -> None:
        """After distraction, probability drops -> returns to ATTENTIVE."""
        pipeline = LogicPipeline(alpha=0.5)

        # Drive into distraction
        for _ in range(20):
            pipeline.step(0.9)

        # Recovery
        for _ in range(50):
            result = pipeline.step(0.0)

        assert result["state"] == DriverState.ATTENTIVE

    def test_ema_prevents_instant_state_change(self) -> None:
        """EMA smoothing prevents a single high frame from causing DISTRACTED."""
        pipeline = LogicPipeline(alpha=0.3)

        # 10 frames at 0.0, then 1 frame at 1.0
        for _ in range(10):
            pipeline.step(0.0)

        result = pipeline.step(1.0)
        assert result["state"] != DriverState.DISTRACTED

    def test_multimodal_boost_in_pipeline(self) -> None:
        """DISTRACTED + confirmed phone -> confidence boosted."""
        pipeline = LogicPipeline(alpha=0.9, n=1, m=3)

        phone = ObjectDetection(
            class_name="cell phone", confidence=0.9, bbox=(10, 10, 100, 100)
        )

        # Get to DISTRACTED state
        for _ in range(10):
            result = pipeline.step(0.95, detections=[phone])

        assert result["state"] == DriverState.DISTRACTED
        assert result["assessment"].confidence > result["smoothed_prob"]

    def test_full_scenario(self) -> None:
        """Full scenario: attentive -> phone distraction -> alert -> recovery."""
        pipeline = LogicPipeline(
            alpha=0.5,
            n=3,
            m=5,
            sustained_frames=8,
            sustained_ratio=0.75,
        )

        phone = ObjectDetection(
            class_name="cell phone", confidence=0.9, bbox=(10, 10, 100, 100)
        )

        # Phase 1: Attentive driving (10 frames)
        for _ in range(10):
            result = pipeline.step(0.0)
        assert result["state"] == DriverState.ATTENTIVE

        # Phase 2: Pick up phone (ramp up to distraction)
        alert_fired = False
        for _ in range(30):
            result = pipeline.step(0.95, detections=[phone])
            if result["alert"] is not None:
                alert_fired = True

        assert result["state"] == DriverState.DISTRACTED
        assert "phone_use" in result["assessment"].active_triggers
        assert alert_fired

        # Phase 3: Put phone down (recovery)
        for _ in range(50):
            result = pipeline.step(0.0)

        assert result["state"] == DriverState.ATTENTIVE
