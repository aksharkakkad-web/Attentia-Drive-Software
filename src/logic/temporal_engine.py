"""Temporal Engine — Layer 3 of the detection pipeline.

Orchestrates duration timers, PERCLOS calculator, and blink detector
over a rolling circular buffer of SignalFrames. Produces TemporalFeatures.

PRD §5.4, §5.5, §FR-3.1–FR-3.5
"""

import math
from collections import deque
from typing import Optional

from src.config_prd import (
    CAPTURE_FPS,
    HEAD_PITCH_THRESHOLD_DEG,
    HEAD_YAW_THRESHOLD_DEG,
    PERCLOS_CLOSURE_FRACTION,
    PHONE_CONFIDENCE_THRESHOLD,
    CIRCULAR_BUFFER_SIZE,
)
from src.contracts import SignalFrame, TemporalFeatures
from src.logic.blink_detector import BlinkDetector
from src.logic.duration_timer import DurationTimer
from src.logic.perclos_calculator import PerclosCalculator


class TemporalEngine:
    """Converts a stream of SignalFrames into temporal features.

    Internal state:
    - Circular buffer of SignalFrames (maxlen=CIRCULAR_BUFFER_SIZE=120).
    - Four DurationTimers: gaze, head, phone, face_absent.
    - PerclosCalculator: 60-frame sliding window.
    - BlinkDetector: rolling 30s blink rate.

    PRD §5.4, §5.5, §FR-3.1–FR-3.5.

    Speed zone: always URBAN / 1.0 for MVP (no OBD-II/CAN/GPS on Mac).
    PRD §FM-05 — will be replaced when SpeedSource is implemented on device.
    """

    def __init__(self, fps: float = CAPTURE_FPS) -> None:
        self._buffer: deque = deque(maxlen=CIRCULAR_BUFFER_SIZE)

        self._gaze_timer = DurationTimer()
        self._head_timer = DurationTimer()
        self._phone_timer = DurationTimer()
        self._face_absent_timer = DurationTimer()

        self._perclos = PerclosCalculator()
        self._blink = BlinkDetector()

        self._last_timestamp_ns: Optional[int] = None
        self._default_delta: float = 1.0 / fps

    # ── Main processing method ─────────────────────────────────────────────────

    def process(self, signal_frame: SignalFrame) -> TemporalFeatures:
        """Process one SignalFrame → TemporalFeatures.

        Args:
            signal_frame: Output from SignalProcessor for this frame.

        Returns:
            TemporalFeatures with all temporal signals populated.
        """
        # Compute frame delta from timestamps
        delta = self._compute_delta(signal_frame.timestamp_ns)

        # Add to circular buffer
        self._buffer.append(signal_frame)

        # ── Timer ticking ──────────────────────────────────────────────────────

        # gaze_timer: off-road gaze, face present, signals valid
        gaze_active = (
            signal_frame.face_present
            and signal_frame.signals_valid
            and signal_frame.gaze_world is not None
            and not signal_frame.gaze_world.on_road
        )
        self._gaze_timer.tick(gaze_active, delta)

        # head_timer: head pose exceeds threshold, face present
        head_active = False
        if signal_frame.face_present and signal_frame.head_pose is not None:
            head_active = (
                abs(signal_frame.head_pose.yaw_deg) > HEAD_YAW_THRESHOLD_DEG
                or abs(signal_frame.head_pose.pitch_deg) > HEAD_PITCH_THRESHOLD_DEG
            )
        self._head_timer.tick(head_active, delta)

        # phone_timer: phone detected with confidence >= threshold
        phone_active = (
            signal_frame.phone_signal.detected
            and signal_frame.phone_signal.confidence >= PHONE_CONFIDENCE_THRESHOLD
        )
        self._phone_timer.tick(phone_active, delta)

        # face_absent_timer: face not present
        self._face_absent_timer.tick(not signal_frame.face_present, delta)

        # ── PERCLOS feeding ────────────────────────────────────────────────────

        eye = signal_frame.eye_signals
        if eye is not None and eye.valid and eye.calibration_complete:
            perclos_closed = eye.mean_EAR <= eye.baseline_EAR * (1.0 - PERCLOS_CLOSURE_FRACTION)
            self._perclos.update(perclos_closed, frame_valid=True)
        else:
            self._perclos.update(eyes_closed=False, frame_valid=False)

        # ── Blink detection ────────────────────────────────────────────────────

        if eye is not None and eye.valid:
            self._blink.update(eye.mean_EAR, eye.close_threshold, delta)

        # ── Buffer statistics ──────────────────────────────────────────────────

        gaze_off_road_fraction = self._compute_gaze_fraction()
        head_deviation_mean = self._compute_head_deviation_mean()
        frames_valid = sum(1 for f in self._buffer if f.signals_valid)

        # ── Phone confidence mean ──────────────────────────────────────────────
        phone_confs = [
            f.phone_signal.confidence
            for f in self._buffer
            if f.phone_signal.detected
        ]
        phone_confidence_mean = (
            sum(phone_confs) / len(phone_confs) if phone_confs else 0.0
        )

        return TemporalFeatures(
            timestamp_ns=signal_frame.timestamp_ns,
            gaze_off_road_fraction=gaze_off_road_fraction,
            gaze_continuous_secs=self._gaze_timer.current_duration,
            head_deviation_mean_deg=head_deviation_mean,
            head_continuous_secs=self._head_timer.current_duration,
            perclos=self._perclos.perclos,
            perclos_valid=self._perclos.valid,
            blink_rate_score=self._blink.blink_rate_score,
            phone_confidence_mean=phone_confidence_mean,
            phone_continuous_secs=self._phone_timer.current_duration,
            face_absent_continuous_secs=self._face_absent_timer.current_duration,
            speed_zone='URBAN',
            speed_modifier=1.0,
            frames_valid_in_window=frames_valid,
            thermal_throttle_active=False,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _compute_delta(self, timestamp_ns: int) -> float:
        """Compute frame delta in seconds from consecutive timestamps.

        Returns 0.0 for duplicate or backward timestamps to prevent timer corruption.
        """
        if self._last_timestamp_ns is None:
            self._last_timestamp_ns = timestamp_ns
            return self._default_delta
        if timestamp_ns <= self._last_timestamp_ns:
            # Duplicate or backward timestamp — clamp to 0.0, leave baseline unchanged
            return 0.0
        delta = (timestamp_ns - self._last_timestamp_ns) / 1e9
        self._last_timestamp_ns = timestamp_ns
        return max(self._default_delta, min(1.0, delta))

    def _compute_gaze_fraction(self) -> float:
        """Fraction of valid frames in the buffer where gaze was off-road."""
        valid_frames = [f for f in self._buffer if f.signals_valid and f.gaze_world is not None]
        if not valid_frames:
            return 0.0
        off_road = sum(1 for f in valid_frames if not f.gaze_world.on_road)
        return off_road / len(valid_frames)

    def _compute_head_deviation_mean(self) -> float:
        """Mean Euclidean head deviation (sqrt(yaw²+pitch²)) over valid frames."""
        deviations = []
        for f in self._buffer:
            if f.signals_valid and f.head_pose is not None:
                dev = math.sqrt(f.head_pose.yaw_deg ** 2 + f.head_pose.pitch_deg ** 2)
                deviations.append(dev)
        if not deviations:
            return 0.0
        return sum(deviations) / len(deviations)
