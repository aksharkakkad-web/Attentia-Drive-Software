"""Alert State Machine — Layer 5 of the detection pipeline.

Takes DistractionScore + signals_valid → AlertCommand | None.

States: NOMINAL → PRE_ALERT → ALERTING → COOLDOWN → DEGRADED.
Per-alert-type cooldowns. PHONE_USE has the highest priority and is checked
first, but it respects its own COOLDOWN_PHONE window (P-01).
FACE_ABSENT has independent cooldown tracking (P-03).
No alerts in DEGRADED state (P-04).

PRD §7
"""

import time
import uuid
from typing import Dict, Optional

from src.config_prd import (
    COOLDOWN_DROWSINESS,
    COOLDOWN_FACE_ABSENT,
    COOLDOWN_HEAD,
    COOLDOWN_PHONE,
    COOLDOWN_VISUAL,
    DEGRADED_RECOVERY_FRAMES,
    DEGRADED_TRIGGER_FRAMES,
)
from src.contracts import AlertCommand, AlertLevel, AlertType, DistractionScore

# Cooldown duration per alert type (seconds)
_COOLDOWN: Dict[AlertType, float] = {
    AlertType.VISUAL_INATTENTION: COOLDOWN_VISUAL,
    AlertType.HEAD_INATTENTION: COOLDOWN_HEAD,
    AlertType.DROWSINESS: COOLDOWN_DROWSINESS,
    AlertType.PHONE_USE: COOLDOWN_PHONE,
    AlertType.FACE_ABSENT: COOLDOWN_FACE_ABSENT,
}

# Priority-ordered mapping from active_class string to AlertType
_CLASS_PRIORITY = ['D-D', 'D-A', 'D-B', 'D-C']
_CLASS_TO_ALERT = {
    'D-D': AlertType.PHONE_USE,
    'D-A': AlertType.VISUAL_INATTENTION,
    'D-B': AlertType.HEAD_INATTENTION,
    'D-C': AlertType.DROWSINESS,
}


class AlertStateMachine:
    """Manages alert suppression, cooldowns, and the DEGRADED state.

    PRD §7.
    """

    def __init__(self) -> None:
        self._cooldown_expiry: Dict[AlertType, float] = {}
        self._degraded_invalid_count: int = 0
        self._degraded_recovery_count: int = 0
        self._is_degraded: bool = False

    def update(
        self,
        score: DistractionScore,
        signals_valid: bool,
    ) -> Optional[AlertCommand]:
        """Evaluate score and return an AlertCommand, or None to suppress.

        Args:
            score: DistractionScore for this frame.
            signals_valid: True if perception signals are healthy this frame.

        Returns:
            AlertCommand to fire, or None if suppressed/degraded.
        """
        # ── DEGRADED CHECK (always first) ──────────────────────────────────────
        if not signals_valid:
            self._degraded_invalid_count += 1
            self._degraded_recovery_count = 0
        else:
            self._degraded_invalid_count = 0
            if self._is_degraded:
                self._degraded_recovery_count += 1

        if self._degraded_invalid_count >= DEGRADED_TRIGGER_FRAMES:
            self._is_degraded = True

        if self._is_degraded and self._degraded_recovery_count >= DEGRADED_RECOVERY_FRAMES:
            self._is_degraded = False
            self._degraded_recovery_count = 0

        if self._is_degraded:
            return None  # P-04: no alerts in DEGRADED

        # ── ALERT CHECK ───────────────────────────────────────────────────────
        now = time.monotonic()

        # Build candidates in priority order: PHONE > FACE_ABSENT > others
        candidates = []

        if score.phone_threshold_breached:
            candidates.append(AlertType.PHONE_USE)
        if score.face_absent_threshold_breached:
            candidates.append(AlertType.FACE_ABSENT)
        if score.gaze_threshold_breached:
            candidates.append(AlertType.VISUAL_INATTENTION)
        if score.head_threshold_breached:
            candidates.append(AlertType.HEAD_INATTENTION)
        if score.perclos_threshold_breached:
            candidates.append(AlertType.DROWSINESS)
        if score.composite_threshold_breached:
            composite_type = self._composite_alert_type(score.active_classes)
            if composite_type not in candidates:
                candidates.append(composite_type)

        # Select the highest-priority candidate not suppressed by its own cooldown.
        # P-01: PHONE_USE has the highest priority (first in candidates) and its
        # cooldown is checked independently — it fires even while other alert types
        # are in cooldown, but it respects its own COOLDOWN_PHONE window to prevent
        # rapid-fire alerts from noisy detection.
        selected: Optional[AlertType] = None
        for alert_type in candidates:
            expiry = self._cooldown_expiry.get(alert_type, 0.0)
            if now >= expiry:
                selected = alert_type
                break

        if selected is None:
            return None

        # Emit alert and set cooldown for the selected type
        cooldown = _COOLDOWN[selected]
        self._cooldown_expiry[selected] = now + cooldown

        level = AlertLevel.URGENT if selected == AlertType.PHONE_USE else AlertLevel.HIGH

        return AlertCommand(
            alert_id=str(uuid.uuid4()),
            timestamp_ns=score.timestamp_ns,
            level=level,
            alert_type=selected,
            composite_score=score.composite_score,
            suppress_until_ns=int((now + cooldown) * 1e9),
            active_classes=list(score.active_classes),
        )

    @staticmethod
    def _composite_alert_type(active_classes: list) -> AlertType:
        """Map active_classes to the highest-priority AlertType for a composite breach."""
        for cls in _CLASS_PRIORITY:
            if cls in active_classes:
                return _CLASS_TO_ALERT[cls]
        return AlertType.VISUAL_INATTENTION  # default
