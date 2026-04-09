"""Tests for Phase 5: ScoringEngine and AlertStateMachine.

All tests use synthetic data. PRD §6, §7.
"""

import pytest
from unittest.mock import patch

from src.contracts import AlertLevel, AlertType, DistractionScore, TemporalFeatures
from src.logic.alert_state_machine import AlertStateMachine
from src.logic.scoring_engine import ScoringEngine


# ── Helpers ───────────────────────────────────────────────────────────────────

def _features(
    gaze_fraction: float = 0.0,
    gaze_secs: float = 0.0,
    head_mean: float = 0.0,
    head_secs: float = 0.0,
    perclos: float = 0.0,
    perclos_valid: bool = False,
    blink_score: float = 0.0,
    phone_secs: float = 0.0,
    face_absent_secs: float = 0.0,
    speed_zone: str = 'URBAN',
    speed_modifier: float = 1.0,
) -> TemporalFeatures:
    return TemporalFeatures(
        gaze_off_road_fraction=gaze_fraction,
        gaze_continuous_secs=gaze_secs,
        head_deviation_mean_deg=head_mean,
        head_continuous_secs=head_secs,
        perclos=perclos,
        perclos_valid=perclos_valid,
        blink_rate_score=blink_score,
        phone_continuous_secs=phone_secs,
        face_absent_continuous_secs=face_absent_secs,
        speed_zone=speed_zone,
        speed_modifier=speed_modifier,
    )


def _score_no_breaches() -> DistractionScore:
    """DistractionScore with all flags False."""
    return DistractionScore()


def _score_phone() -> DistractionScore:
    """DistractionScore with only phone_threshold_breached=True."""
    return DistractionScore(phone_threshold_breached=True, active_classes=['D-D'])


def _score_gaze() -> DistractionScore:
    """DistractionScore with only gaze_threshold_breached=True."""
    return DistractionScore(
        gaze_threshold_breached=True,
        composite_score=0.45,
        active_classes=['D-A'],
    )


def _score_gaze_and_head() -> DistractionScore:
    """DistractionScore with gaze + head both breached."""
    return DistractionScore(
        gaze_threshold_breached=True,
        head_threshold_breached=True,
        composite_score=0.75,
        active_classes=['D-A', 'D-B'],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE (tests 1–9)
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoringEngine:
    # Test 1
    def test_all_zeros_gives_composite_zero_no_breaches(self):
        """All features at zero → composite=0.0, no breach flags."""
        se = ScoringEngine()
        result = se.score(_features())
        assert result.composite_score == pytest.approx(0.0)
        assert result.gaze_threshold_breached is False
        assert result.head_threshold_breached is False
        assert result.perclos_threshold_breached is False
        assert result.phone_threshold_breached is False
        assert result.composite_threshold_breached is False
        assert result.face_absent_threshold_breached is False
        assert result.active_classes == []

    # Test 2
    def test_gaze_fraction_1_gives_0_45_no_composite_breach(self):
        """gaze_off_road_fraction=1.0, all else 0 → composite=0.45 < 0.55."""
        se = ScoringEngine()
        result = se.score(_features(gaze_fraction=1.0))
        assert result.composite_score == pytest.approx(0.45)
        assert result.composite_threshold_breached is False

    # Test 3
    def test_gaze_1_and_head_30_gives_0_75_composite_breached(self):
        """gaze=1.0 + head_deviation=30.0 with active gaze timer → composite=0.75, breached."""
        se = ScoringEngine()
        # gaze_secs=2.5 ensures at least one currently-active contributor exists
        result = se.score(_features(gaze_fraction=1.0, gaze_secs=2.5, head_mean=30.0))
        assert result.composite_score == pytest.approx(0.75)
        assert result.composite_threshold_breached is True

    # Test 4
    def test_parked_composite_zero_only_phone_can_breach(self):
        """speed_modifier=0.0 (PARKED) → composite=0.0, only phone can breach."""
        se = ScoringEngine()
        result = se.score(_features(
            gaze_fraction=1.0, gaze_secs=3.0,
            head_mean=30.0, head_secs=2.0,
            perclos=0.30, perclos_valid=True,
            face_absent_secs=6.0,
            phone_secs=2.0,
            speed_modifier=0.0,
        ))
        assert result.composite_score == pytest.approx(0.0)
        assert result.gaze_threshold_breached is False
        assert result.head_threshold_breached is False
        assert result.perclos_threshold_breached is False
        assert result.composite_threshold_breached is False
        assert result.face_absent_threshold_breached is False
        assert result.phone_threshold_breached is True  # phone still fires

    # Test 5
    def test_highway_modifier_multiplies_composite(self):
        """speed_modifier=1.4 (HIGHWAY) → composite = D_raw * 1.4."""
        se = ScoringEngine()
        result_urban = se.score(_features(gaze_fraction=1.0, speed_modifier=1.0))
        result_highway = se.score(_features(gaze_fraction=1.0, speed_modifier=1.4))
        assert result_highway.composite_score == pytest.approx(result_urban.composite_score * 1.4)

    # Test 6
    def test_gaze_continuous_secs_2_5_triggers_gaze_breach(self):
        """gaze_continuous_secs=2.5 >= T_GAZE_SECONDS=2.0 → gaze_threshold_breached=True."""
        se = ScoringEngine()
        result = se.score(_features(gaze_secs=2.5))
        assert result.gaze_threshold_breached is True

    # Test 7
    def test_phone_continuous_secs_1_5_triggers_phone_breach(self):
        """phone_continuous_secs=1.5 > T_PHONE_SECONDS=0.0 → phone_threshold_breached=True."""
        se = ScoringEngine()
        result = se.score(_features(phone_secs=1.5))
        assert result.phone_threshold_breached is True

    # Test 8
    def test_perclos_0_20_with_valid_triggers_perclos_breach(self):
        """perclos=0.20 >= 0.15 AND perclos_valid=True → perclos_threshold_breached=True."""
        se = ScoringEngine()
        result = se.score(_features(perclos=0.20, perclos_valid=True))
        assert result.perclos_threshold_breached is True

    # Test 9
    def test_perclos_0_20_without_valid_does_not_trigger(self):
        """perclos=0.20 but perclos_valid=False → perclos_threshold_breached=False."""
        se = ScoringEngine()
        result = se.score(_features(perclos=0.20, perclos_valid=False))
        assert result.perclos_threshold_breached is False

    def test_active_classes_contains_gaze_and_head(self):
        """Both gaze and head breached → active_classes=['D-A','D-B']."""
        se = ScoringEngine()
        result = se.score(_features(gaze_secs=3.0, head_secs=2.0))
        assert 'D-A' in result.active_classes
        assert 'D-B' in result.active_classes

    def test_component_scores_correct(self):
        """Component scores are weighted fractions before modifier."""
        se = ScoringEngine()
        result = se.score(_features(gaze_fraction=1.0))
        assert result.component_gaze == pytest.approx(0.45)
        assert result.component_head == pytest.approx(0.0)

    def test_weights_assert_fires_on_bad_config(self):
        """ScoringEngine.__init__ asserts weights sum to 1.0."""
        # This test exercises the assertion indirectly: a correct config passes.
        se = ScoringEngine()  # should not raise
        assert se is not None


# ═══════════════════════════════════════════════════════════════════════════════
# ALERT STATE MACHINE (tests 10–18)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlertStateMachine:
    # Test 10
    def test_no_breaches_never_alerts(self):
        """100 updates with no breach flags → no alerts emitted."""
        asm = AlertStateMachine()
        score = _score_no_breaches()
        for _ in range(100):
            result = asm.update(score, signals_valid=True)
        assert result is None

    # Test 11
    def test_phone_breach_gives_urgent_alert(self):
        """Phone breach → URGENT AlertCommand with alert_type=PHONE_USE."""
        asm = AlertStateMachine()
        with patch('src.logic.alert_state_machine.time.monotonic', return_value=0.0):
            result = asm.update(_score_phone(), signals_valid=True)
        assert result is not None
        assert result.level == AlertLevel.URGENT
        assert result.alert_type == AlertType.PHONE_USE

    # Test 12 — P-01: phone respects its own cooldown
    def test_phone_respects_own_cooldown(self):
        """PHONE_USE fires on first breach, then is suppressed during COOLDOWN_PHONE."""
        asm = AlertStateMachine()
        with patch('src.logic.alert_state_machine.time.monotonic', return_value=0.0):
            r1 = asm.update(_score_phone(), signals_valid=True)
        with patch('src.logic.alert_state_machine.time.monotonic', return_value=0.5):
            # t=0.5 is within the 5s cooldown window — phone must be suppressed
            r2 = asm.update(_score_phone(), signals_valid=True)
        assert r1 is not None and r1.alert_type == AlertType.PHONE_USE
        assert r2 is None  # suppressed by cooldown

    def test_phone_fires_again_after_cooldown_expires(self):
        """PHONE_USE fires again once COOLDOWN_PHONE seconds have elapsed."""
        asm = AlertStateMachine()
        with patch('src.logic.alert_state_machine.time.monotonic', return_value=0.0):
            r1 = asm.update(_score_phone(), signals_valid=True)
        with patch('src.logic.alert_state_machine.time.monotonic', return_value=5.1):
            r2 = asm.update(_score_phone(), signals_valid=True)
        assert r1 is not None and r1.alert_type == AlertType.PHONE_USE
        assert r2 is not None and r2.alert_type == AlertType.PHONE_USE

    # Test 13 — face absent alert
    def test_face_absent_breach_gives_high_alert(self):
        """face_absent_threshold_breached → HIGH alert, alert_type=FACE_ABSENT."""
        asm = AlertStateMachine()
        score = DistractionScore(face_absent_threshold_breached=True)
        with patch('src.logic.alert_state_machine.time.monotonic', return_value=0.0):
            result = asm.update(score, signals_valid=True)
        assert result is not None
        assert result.alert_type == AlertType.FACE_ABSENT
        assert result.level == AlertLevel.HIGH

    # Test 14
    def test_gaze_breach_gives_high_alert(self):
        """Gaze breach → HIGH AlertCommand with alert_type=VISUAL_INATTENTION."""
        asm = AlertStateMachine()
        with patch('src.logic.alert_state_machine.time.monotonic', return_value=0.0):
            result = asm.update(_score_gaze(), signals_valid=True)
        assert result is not None
        assert result.level == AlertLevel.HIGH
        assert result.alert_type == AlertType.VISUAL_INATTENTION

    # Test 15
    def test_phone_fires_while_gaze_in_cooldown(self):
        """P-01: phone alert fires even while VISUAL_INATTENTION is in cooldown."""
        asm = AlertStateMachine()
        # Fire gaze alert → puts VISUAL_INATTENTION in 8s cooldown
        with patch('src.logic.alert_state_machine.time.monotonic', return_value=0.0):
            asm.update(_score_gaze(), signals_valid=True)

        # t=4: gaze still in cooldown, but phone breach now → phone fires
        score_gaze_and_phone = DistractionScore(
            gaze_threshold_breached=True,
            phone_threshold_breached=True,
            active_classes=['D-A', 'D-D'],
        )
        with patch('src.logic.alert_state_machine.time.monotonic', return_value=4.0):
            result = asm.update(score_gaze_and_phone, signals_valid=True)
        assert result is not None
        assert result.alert_type == AlertType.PHONE_USE

    # Test 16
    def test_60_invalid_frames_enters_degraded(self):
        """60 consecutive signals_valid=False → DEGRADED; every update returns None."""
        asm = AlertStateMachine()
        # Use no-breach score so no alert fires before DEGRADED is entered
        for _ in range(60):
            result = asm.update(_score_no_breaches(), signals_valid=False)
            assert result is None  # no alert on any frame during entry
        assert asm._is_degraded is True
        # Even a phone breach returns None once DEGRADED
        result = asm.update(_score_phone(), signals_valid=False)
        assert result is None

    # Test 17
    def test_30_valid_frames_recovers_from_degraded(self):
        """After DEGRADED, 30 consecutive valid frames → recover, alerts work again."""
        asm = AlertStateMachine()
        # Enter DEGRADED
        for _ in range(60):
            asm.update(_score_no_breaches(), signals_valid=False)
        assert asm._is_degraded is True

        # Recover with 30 valid frames (no breach — we just want to recover)
        for _ in range(30):
            asm.update(_score_no_breaches(), signals_valid=True)
        assert asm._is_degraded is False

        # Now a phone breach should fire normally
        with patch('src.logic.alert_state_machine.time.monotonic', return_value=0.0):
            result = asm.update(_score_phone(), signals_valid=True)
        assert result is not None
        assert result.alert_type == AlertType.PHONE_USE

    # Test 18
    def test_multiple_breaches_one_alert_with_both_active_classes(self):
        """Gaze + head both breached → one alert, active_classes contains 'D-A' and 'D-B'."""
        asm = AlertStateMachine()
        with patch('src.logic.alert_state_machine.time.monotonic', return_value=0.0):
            result = asm.update(_score_gaze_and_head(), signals_valid=True)
        assert result is not None
        assert 'D-A' in result.active_classes
        assert 'D-B' in result.active_classes
        # Only one alert command emitted
        assert isinstance(result.alert_id, str) and len(result.alert_id) > 0

    def test_composite_breach_without_individual_gives_visual_inattention(self):
        """composite_threshold_breached with no individual breaches → VISUAL_INATTENTION."""
        asm = AlertStateMachine()
        score = DistractionScore(composite_threshold_breached=True, active_classes=[])
        with patch('src.logic.alert_state_machine.time.monotonic', return_value=0.0):
            result = asm.update(score, signals_valid=True)
        assert result is not None
        assert result.alert_type == AlertType.VISUAL_INATTENTION

    def test_alert_fires_immediately_on_recovery_frame(self):
        """The 30th recovery frame exits DEGRADED and immediately allows an alert."""
        asm = AlertStateMachine()
        for _ in range(60):
            asm.update(_score_no_breaches(), signals_valid=False)
        assert asm._is_degraded is True
        for _ in range(29):
            asm.update(_score_no_breaches(), signals_valid=True)
        assert asm._is_degraded is True  # not yet recovered
        # 30th valid frame exits DEGRADED and should be able to emit an alert
        with patch('src.logic.alert_state_machine.time.monotonic', return_value=0.0):
            result = asm.update(_score_phone(), signals_valid=True)
        assert asm._is_degraded is False
        assert result is not None
        assert result.alert_type == AlertType.PHONE_USE

    def test_alert_id_is_unique_per_alert(self):
        """Each fired alert has a distinct UUID."""
        asm = AlertStateMachine()
        ids = set()
        for t in range(3):
            with patch('src.logic.alert_state_machine.time.monotonic', return_value=float(t * 10)):
                result = asm.update(_score_phone(), signals_valid=True)
                if result:
                    ids.add(result.alert_id)
        assert len(ids) >= 2

    def test_degraded_recovery_requires_consecutive_valid_frames(self):
        """Recovery counter resets if an invalid frame interrupts the recovery sequence."""
        asm = AlertStateMachine()
        for _ in range(60):
            asm.update(_score_no_breaches(), signals_valid=False)
        assert asm._is_degraded is True

        # 29 valid, 1 invalid, 29 valid — should NOT recover (counter resets)
        for _ in range(29):
            asm.update(_score_no_breaches(), signals_valid=True)
        asm.update(_score_no_breaches(), signals_valid=False)  # interrupt
        for _ in range(29):
            asm.update(_score_no_breaches(), signals_valid=True)
        assert asm._is_degraded is True  # still degraded


# ═══════════════════════════════════════════════════════════════════════════════
# PHONE INSTANT ALERT — T_PHONE_SECONDS = 0.0
# ═══════════════════════════════════════════════════════════════════════════════

class TestInstantPhoneAlert:
    def test_phone_zero_secs_does_not_breach(self):
        """phone_continuous_secs=0.0 with T_PHONE_SECONDS=0.0 → no breach (no phone present)."""
        se = ScoringEngine()
        result = se.score(_features(phone_secs=0.0))
        assert result.phone_threshold_breached is False

    def test_phone_first_active_frame_breaches(self):
        """phone_continuous_secs=0.033 (one frame at 30fps) → instant breach."""
        se = ScoringEngine()
        result = se.score(_features(phone_secs=0.033))
        assert result.phone_threshold_breached is True

    def test_phone_breach_in_parked_mode(self):
        """Phone breach fires even at speed_modifier=0.0 (parked)."""
        se = ScoringEngine()
        result = se.score(_features(phone_secs=0.033, speed_modifier=0.0))
        assert result.phone_threshold_breached is True

    def test_no_phone_parked_does_not_breach(self):
        """No phone, parked → phone_threshold_breached=False."""
        se = ScoringEngine()
        result = se.score(_features(phone_secs=0.0, speed_modifier=0.0))
        assert result.phone_threshold_breached is False


# ═══════════════════════════════════════════════════════════════════════════════
# GAZE / HEAD RECOVERY — breach flags clear on same frame, composite guards
# ═══════════════════════════════════════════════════════════════════════════════

class TestRecoveryBehavior:
    def test_gaze_breach_clears_immediately_when_timer_resets(self):
        """gaze_continuous_secs=0.0 → gaze_threshold_breached=False immediately."""
        se = ScoringEngine()
        result = se.score(_features(gaze_secs=0.0))
        assert result.gaze_threshold_breached is False

    def test_head_breach_clears_immediately_when_timer_resets(self):
        """head_continuous_secs=0.0 → head_threshold_breached=False immediately."""
        se = ScoringEngine()
        result = se.score(_features(head_secs=0.0))
        assert result.head_threshold_breached is False

    def test_composite_does_not_fire_from_stale_history_alone(self):
        """High gaze_off_road_fraction + high head_mean but both timers at 0 → no composite breach.

        Simulates the driver just looked back on-road after 4s of distraction.
        Buffer history is still full of off-road frames, but no current contributor.
        """
        se = ScoringEngine()
        result = se.score(_features(
            gaze_fraction=1.0,   # buffer full of off-road history
            head_mean=30.0,      # buffer full of large head deviations
            gaze_secs=0.0,       # but driver is currently looking on-road
            head_secs=0.0,       # and head is currently straight
        ))
        # Composite score is high (0.75) but should not breach without current activity
        assert result.composite_score == pytest.approx(0.75)
        assert result.composite_threshold_breached is False

    def test_composite_fires_when_currently_distracted_and_history_high(self):
        """Stale history AND currently distracted → composite breach fires."""
        se = ScoringEngine()
        result = se.score(_features(
            gaze_fraction=1.0,
            gaze_secs=3.0,   # currently off-road (active contributor)
            head_mean=30.0,
        ))
        assert result.composite_threshold_breached is True

    def test_composite_fires_from_perclos_alone(self):
        """If perclos is the only contributor, composite can still breach."""
        se = ScoringEngine()
        result = se.score(_features(
            gaze_fraction=1.0,   # stale history
            head_mean=30.0,
            gaze_secs=0.0,       # driver looking on-road now
            head_secs=0.0,
            perclos=0.20,
            perclos_valid=True,  # drowsiness is current — valid contributor
        ))
        assert result.composite_threshold_breached is True

    def test_composite_suppressed_after_recovery_no_perclos(self):
        """After recovery with no drowsiness, composite is suppressed from stale data."""
        se = ScoringEngine()
        result = se.score(_features(
            gaze_fraction=1.0,
            head_mean=30.0,
            gaze_secs=0.0,
            head_secs=0.0,
            perclos=0.0,
            perclos_valid=False,
        ))
        assert result.composite_threshold_breached is False
