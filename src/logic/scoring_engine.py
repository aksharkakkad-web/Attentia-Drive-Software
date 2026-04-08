"""Scoring Engine — Layer 4 of the detection pipeline.

Takes TemporalFeatures → DistractionScore.

Composite formula (PRD §6.1):
  F1       = gaze_off_road_fraction
  F2_norm  = clamp(head_deviation_mean_deg / HEAD_YAW_THRESHOLD_DEG, 0, 1)
  F3       = perclos  (0.0 if perclos_valid is False)
  F4       = blink_rate_score
  D_raw    = W1*F1 + W2*F2_norm + W3*F3 + W4*F4
  D        = D_raw * speed_modifier

Six independent threshold checks produce breach flags (PRD §6.2).

PRD §6
"""

from src.config_prd import (
    COMPOSITE_ALERT_THRESHOLD,
    HEAD_YAW_THRESHOLD_DEG,
    PERCLOS_ALERT_THRESHOLD,
    T_FACE_ABSENT_SECONDS,
    T_GAZE_SECONDS,
    T_HEAD_SECONDS,
    T_PHONE_SECONDS,
    WEIGHT_BLINK,
    WEIGHT_GAZE,
    WEIGHT_HEAD,
    WEIGHT_PERCLOS,
)
from src.contracts import DistractionScore, TemporalFeatures


class ScoringEngine:
    """Computes a weighted composite distraction score and six breach flags.

    PRD §6.
    """

    def __init__(self) -> None:
        total = WEIGHT_GAZE + WEIGHT_HEAD + WEIGHT_PERCLOS + WEIGHT_BLINK
        assert abs(total - 1.0) < 1e-6, (
            f"Scoring weights must sum to 1.0, got {total:.6f}"
        )

    def score(self, features: TemporalFeatures) -> DistractionScore:
        """Compute DistractionScore from TemporalFeatures.

        Args:
            features: Temporal output for this frame.

        Returns:
            DistractionScore with composite score and all breach flags.
        """
        speed_modifier = features.speed_modifier
        parked = speed_modifier == 0.0

        # ── Component factors ──────────────────────────────────────────────────

        f1 = features.gaze_off_road_fraction
        f2_norm = max(0.0, min(1.0, features.head_deviation_mean_deg / HEAD_YAW_THRESHOLD_DEG))
        f3 = features.perclos if features.perclos_valid else 0.0
        f4 = features.blink_rate_score

        d_raw = WEIGHT_GAZE * f1 + WEIGHT_HEAD * f2_norm + WEIGHT_PERCLOS * f3 + WEIGHT_BLINK * f4

        # ── Composite score ────────────────────────────────────────────────────

        d = 0.0 if parked else d_raw * speed_modifier

        # ── Threshold breach flags ─────────────────────────────────────────────
        # Phone breach is independent of speed zone (PRD §6.2 ALT-04).
        # When T_PHONE_SECONDS == 0.0 (instant alert), use strict > to avoid
        # the false positive 0.0 >= 0.0 that fires when no phone is present.
        if T_PHONE_SECONDS == 0.0:
            phone_threshold_breached = features.phone_continuous_secs > 0.0
        else:
            phone_threshold_breached = features.phone_continuous_secs >= T_PHONE_SECONDS

        if parked:
            # PARKED: all non-phone breaches suppressed
            gaze_threshold_breached = False
            head_threshold_breached = False
            perclos_threshold_breached = False
            composite_threshold_breached = False
            face_absent_threshold_breached = False
        else:
            gaze_threshold_breached = features.gaze_continuous_secs >= T_GAZE_SECONDS
            head_threshold_breached = features.head_continuous_secs >= T_HEAD_SECONDS
            perclos_threshold_breached = (
                features.perclos >= PERCLOS_ALERT_THRESHOLD and features.perclos_valid
            )
            # Composite requires at least one currently-active contributor.
            # gaze_off_road_fraction and head_deviation_mean are buffer averages
            # that can stay elevated for up to 4s after recovery — without this
            # guard, composite keeps firing from stale history alone.
            currently_distracted = (
                features.gaze_continuous_secs > 0.0
                or features.head_continuous_secs > 0.0
                or perclos_threshold_breached
            )
            composite_threshold_breached = (
                d >= COMPOSITE_ALERT_THRESHOLD and currently_distracted
            )
            face_absent_threshold_breached = (
                features.face_absent_continuous_secs >= T_FACE_ABSENT_SECONDS
            )

        # ── Active classes ─────────────────────────────────────────────────────
        active_classes = []
        if gaze_threshold_breached:
            active_classes.append('D-A')
        if head_threshold_breached:
            active_classes.append('D-B')
        if perclos_threshold_breached:
            active_classes.append('D-C')
        if phone_threshold_breached:
            active_classes.append('D-D')

        return DistractionScore(
            timestamp_ns=features.timestamp_ns,
            composite_score=d,
            component_gaze=WEIGHT_GAZE * f1,
            component_head=WEIGHT_HEAD * f2_norm,
            component_perclos=WEIGHT_PERCLOS * f3,
            component_blink=WEIGHT_BLINK * f4,
            gaze_threshold_breached=gaze_threshold_breached,
            head_threshold_breached=head_threshold_breached,
            perclos_threshold_breached=perclos_threshold_breached,
            phone_threshold_breached=phone_threshold_breached,
            composite_threshold_breached=composite_threshold_breached,
            face_absent_threshold_breached=face_absent_threshold_breached,
            active_classes=active_classes,
        )
