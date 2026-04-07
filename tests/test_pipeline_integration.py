"""Phase 9 — Integration tests for the full Layer 3→4→5 pipeline.

Feeds synthetic SignalFrames through:
    TemporalEngine → ScoringEngine → AlertStateMachine

No camera, model files, or display required. Each test simulates a stream
of frames at 30 fps using monotonically increasing timestamps so that
TemporalEngine's duration timers accumulate at the expected rate.

PRD §4–§7.
"""

from typing import List, Optional
from unittest.mock import patch

import pytest

from src.contracts import (
    AlertCommand,
    AlertType,
    EyeSignals,
    GazeWorld,
    HeadPose,
    PhoneSignal,
    SignalFrame,
)
from src.logic.alert_state_machine import AlertStateMachine
from src.logic.scoring_engine import ScoringEngine
from src.logic.temporal_engine import TemporalEngine


# ── Helpers ───────────────────────────────────────────────────────────────────

_FRAME_NS = 1_000_000_000 // 30  # ~33.33 ms per frame at 30 fps


def make_signal_frame(
    *,
    frame_id: int = 0,
    timestamp_ns: int = 0,
    face_present: bool = True,
    signals_valid: bool = True,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    on_road: bool = True,
    mean_ear: float = 0.28,
    baseline_ear: float = 0.28,
    close_threshold: float = 0.21,
    ear_valid: bool = True,
    ear_calibrated: bool = True,
    phone_detected: bool = False,
    phone_confidence: float = 0.0,
) -> SignalFrame:
    """Build a SignalFrame with sensible defaults for integration tests.

    Defaults: face present, signals valid, gaze on road, eyes open at the
    population baseline, no phone detected.
    """
    if face_present:
        head_pose: Optional[HeadPose] = HeadPose(
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=0.0,
            valid=True,
        )
        gaze_world: Optional[GazeWorld] = GazeWorld(
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            on_road=on_road,
            valid=True,
        )
        eye_signals: Optional[EyeSignals] = EyeSignals(
            left_EAR=mean_ear,
            right_EAR=mean_ear,
            mean_EAR=mean_ear,
            baseline_EAR=baseline_ear,
            close_threshold=close_threshold,
            valid=ear_valid,
            calibration_complete=ear_calibrated,
        )
    else:
        head_pose = None
        gaze_world = None
        eye_signals = None

    return SignalFrame(
        timestamp_ns=timestamp_ns,
        frame_id=frame_id,
        face_present=face_present,
        head_pose=head_pose,
        eye_signals=eye_signals,
        gaze_world=gaze_world,
        phone_signal=PhoneSignal(detected=phone_detected, confidence=phone_confidence),
        signals_valid=signals_valid,
    )


class _Pipeline:
    """Owns the three downstream layers and feeds frames through them.

    Records every emitted alert with the frame index that produced it so
    tests can assert when alerts fire.
    """

    def __init__(self) -> None:
        self.te = TemporalEngine()
        self.se = ScoringEngine()
        self.asm = AlertStateMachine()
        self.alerts: List[tuple] = []  # list of (frame_idx, AlertCommand)
        self.scores: List = []
        self.features: List = []

    def feed(self, sf: SignalFrame, frame_idx: int) -> Optional[AlertCommand]:
        tf = self.te.process(sf)
        ds = self.se.score(tf)
        alert = self.asm.update(ds, sf.signals_valid)
        self.features.append(tf)
        self.scores.append(ds)
        if alert is not None:
            self.alerts.append((frame_idx, alert))
        return alert


def _stream(
    n_frames: int,
    pipe: _Pipeline,
    start_frame: int = 0,
    start_ns: int = 0,
    monotonic_start: float = 0.0,
    **frame_kwargs,
) -> None:
    """Push n_frames identical frames through the pipeline.

    Mocks `time.monotonic` inside the AlertStateMachine so that one second
    of wall-clock time advances per 30 frames — this matches the timer
    accumulation in TemporalEngine and lets cooldowns expire deterministically.
    """
    for i in range(n_frames):
        idx = start_frame + i
        ts = start_ns + (i + 1) * _FRAME_NS
        sf = make_signal_frame(
            frame_id=idx,
            timestamp_ns=ts,
            **frame_kwargs,
        )
        mono = monotonic_start + (i + 1) / 30.0
        with patch(
            'src.logic.alert_state_machine.time.monotonic',
            return_value=mono,
        ):
            pipe.feed(sf, idx)


# ── Test 1: gaze off-road for 90 frames triggers alert ~frame 60 ─────────────

class TestGazeOffRoadAlert:
    def test_gaze_off_road_alert_at_2_seconds(self):
        pipe = _Pipeline()
        _stream(90, pipe, on_road=False)

        assert pipe.alerts, "expected at least one alert"
        first_idx, first_alert = pipe.alerts[0]
        assert first_alert.alert_type == AlertType.VISUAL_INATTENTION
        # 2.0s at 30 fps = frame 60. Allow ±2 frames of tolerance.
        assert 58 <= first_idx <= 62, f"alert fired at frame {first_idx}, expected ~60"


# ── Test 2: phone detected for 45 frames triggers alert ~frame 30 ────────────

class TestPhoneAlert:
    def test_phone_alert_at_1_second(self):
        pipe = _Pipeline()
        _stream(45, pipe, phone_detected=True, phone_confidence=0.85)

        assert pipe.alerts, "expected phone alert"
        first_idx, first_alert = pipe.alerts[0]
        assert first_alert.alert_type == AlertType.PHONE_USE
        # 1.0s at 30 fps = frame 30. Allow ±2 frames.
        assert 28 <= first_idx <= 32, f"phone alert fired at frame {first_idx}, expected ~30"


# ── Test 3: 100 normal frames produce no alerts and no breaches ──────────────

class TestNormalDriving:
    def test_no_alerts_when_normal(self):
        pipe = _Pipeline()
        _stream(100, pipe)

        assert pipe.alerts == [], "no alerts expected for normal driving"
        for ds in pipe.scores:
            assert ds.gaze_threshold_breached is False
            assert ds.head_threshold_breached is False
            assert ds.phone_threshold_breached is False
            assert ds.composite_threshold_breached is False
            assert ds.face_absent_threshold_breached is False


# ── Test 4: gaze off + head turned → both classes active ────────────────────

class TestMixedGazeAndHead:
    def test_gaze_and_head_both_active(self):
        pipe = _Pipeline()
        # 60 frames gaze off-road only (gaze timer hits 2.0s → D-A breach)
        _stream(60, pipe, on_road=False)
        # 60 frames head turned (yaw=45 > 30) AND still off-road
        _stream(
            60,
            pipe,
            start_frame=60,
            start_ns=60 * _FRAME_NS,
            monotonic_start=60 / 30.0,
            on_road=False,
            yaw_deg=45.0,
        )

        # Find a frame where both classes are active simultaneously
        both_active = any(
            ('D-A' in ds.active_classes and 'D-B' in ds.active_classes)
            for ds in pipe.scores
        )
        assert both_active, (
            "expected at least one frame with both D-A and D-B in active_classes"
        )


# ── Test 5: recovery — distracted → alert → normal → no longer breached ─────

class TestRecovery:
    def test_recovery_after_distraction(self):
        pipe = _Pipeline()
        _stream(90, pipe, on_road=False)
        assert pipe.alerts, "expected an alert during the distracted phase"

        before_count = len(pipe.alerts)

        _stream(
            60,
            pipe,
            start_frame=90,
            start_ns=90 * _FRAME_NS,
            monotonic_start=90 / 30.0,
        )

        # The most recent score must show no gaze breach (timer reset).
        last = pipe.scores[-1]
        assert last.gaze_threshold_breached is False
        assert last.composite_threshold_breached is False
        assert 'D-A' not in last.active_classes
        assert pipe.features[-1].gaze_continuous_secs == pytest.approx(0.0)
        # Recovery period must not produce new gaze alerts.
        assert len(pipe.alerts) == before_count


# ── Test 6: D-A cooldown — second breach within 8s suppressed ────────────────

class TestVisualCooldown:
    def test_no_second_visual_alert_within_cooldown(self):
        pipe = _Pipeline()
        # Fire the first D-A alert (~frame 60)
        _stream(90, pipe, on_road=False)
        first_alerts = list(pipe.alerts)
        assert first_alerts, "expected first D-A alert"
        assert first_alerts[0][1].alert_type == AlertType.VISUAL_INATTENTION

        # Brief recovery to reset the gaze timer (1s = 30 frames normal)
        _stream(
            30,
            pipe,
            start_frame=90,
            start_ns=90 * _FRAME_NS,
            monotonic_start=90 / 30.0,
        )

        # Trigger another D-A breach. Total elapsed mono time at end of this
        # block ≈ 4s + 2s = 6s, which is still inside the 8s D-A cooldown.
        before_count = len(pipe.alerts)
        _stream(
            120,
            pipe,
            start_frame=120,
            start_ns=120 * _FRAME_NS,
            monotonic_start=120 / 30.0,
            on_road=False,
        )

        new_visual_alerts = [
            a for (i, a) in pipe.alerts[before_count:]
            if a.alert_type == AlertType.VISUAL_INATTENTION
        ]
        assert new_visual_alerts == [], (
            "no second VISUAL_INATTENTION alert expected within 8s cooldown"
        )


# ── Test 7: phone breach overrides D-A cooldown (P-01) ───────────────────────

class TestPhoneOverridesCooldown:
    def test_phone_fires_while_visual_in_cooldown(self):
        pipe = _Pipeline()
        # Fire D-A alert (~frame 60), putting VISUAL_INATTENTION into cooldown
        _stream(90, pipe, on_road=False)
        assert pipe.alerts and pipe.alerts[0][1].alert_type == AlertType.VISUAL_INATTENTION

        before_count = len(pipe.alerts)

        # While still in D-A cooldown, raise a phone breach.
        # 45 frames phone detected → phone timer hits 1.0s → P-01 fires.
        _stream(
            45,
            pipe,
            start_frame=90,
            start_ns=90 * _FRAME_NS,
            monotonic_start=90 / 30.0,
            phone_detected=True,
            phone_confidence=0.85,
        )

        new_alerts = pipe.alerts[before_count:]
        phone_alerts = [a for (_i, a) in new_alerts if a.alert_type == AlertType.PHONE_USE]
        assert phone_alerts, "phone alert must fire even while D-A is in cooldown"


# ── Test 8: DEGRADED — 60 invalid frames suppress alerts, 30 valid recover ───

class TestDegradedRecovery:
    def test_degraded_then_recovery(self):
        pipe = _Pipeline()

        # 60 invalid frames — no alerts allowed; ASM enters DEGRADED.
        _stream(60, pipe, signals_valid=False)
        assert pipe.alerts == [], "no alerts allowed during DEGRADED entry"
        assert pipe.asm._is_degraded is True

        # 30 consecutive valid frames → recovery.
        _stream(
            30,
            pipe,
            start_frame=60,
            start_ns=60 * _FRAME_NS,
            monotonic_start=60 / 30.0,
        )
        assert pipe.asm._is_degraded is False
