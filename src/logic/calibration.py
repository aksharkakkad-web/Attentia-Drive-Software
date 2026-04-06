"""Startup calibration — 5-second window to establish neutral pose and EAR baseline.

Collects head pose (yaw, pitch) and EAR samples while the face is visible.
On success: supplies neutral pose offsets and a per-session EAR threshold
            to SignalProcessor via set_neutral_offsets() / set_ear_baseline().
On failure: safe defaults (0.0 offsets, population EAR threshold 0.21).

Completion rules:
  1. Success: >= fps * target_duration_s * 0.90 valid frames AND pose std < 5°.
  2. Early fail: total_frames reaches fps * target_duration_s but valid count < minimum.
  3. Hard-timeout fail: total_frames > fps * target_duration_s * 2.0.

PRD §23
"""

import math
from typing import List

from src.config_prd import (
    CALIBRATION_DURATION_S,
    CALIBRATION_MAX_POSE_STD_DEG,
    EAR_BASELINE_POPULATION_DEFAULT,
    EAR_CALIBRATION_MULTIPLIER,
    EAR_DEFAULT_CLOSE_THRESHOLD,
)

_MIN_VALID_FRACTION = 0.90
_TIMEOUT_MULTIPLIER = 2.0


class Calibration:
    """5-second startup calibration for neutral head pose and EAR baseline.

    PRD §23.

    Args:
        target_duration_s: Calibration window length in seconds.
        fps: Camera frame rate used to compute frame counts.
    """

    def __init__(
        self,
        target_duration_s: float = CALIBRATION_DURATION_S,
        fps: float = 30.0,
    ) -> None:
        self._target_duration_s = target_duration_s
        self._fps = fps

        # Derived frame counts
        self._expected_frames: int = int(fps * target_duration_s)
        self._min_valid_frames: int = int(fps * target_duration_s * _MIN_VALID_FRACTION)
        self._timeout_frames: int = int(fps * target_duration_s * _TIMEOUT_MULTIPLIER)

        self._yaw_samples: List[float] = []
        self._pitch_samples: List[float] = []
        self._ear_samples: List[float] = []
        self._total_frames: int = 0

        self._status: str = 'collecting'
        self._neutral_yaw_offset: float = 0.0
        self._neutral_pitch_offset: float = 0.0
        self._baseline_ear: float = EAR_BASELINE_POPULATION_DEFAULT
        self._close_threshold: float = EAR_DEFAULT_CLOSE_THRESHOLD
        self._quality_ok: bool = False

    # ── Public interface ──────────────────────────────────────────────────────

    def feed_frame(
        self,
        head_yaw: float,
        head_pitch: float,
        mean_ear: float,
        face_visible: bool,
    ) -> bool:
        """Feed one frame of sensor data.

        Args:
            head_yaw: Head yaw angle in degrees (from SignalProcessor).
            head_pitch: Head pitch angle in degrees.
            mean_ear: Mean Eye Aspect Ratio for this frame.
            face_visible: True if the face was detected this frame.

        Returns:
            True when calibration has reached a terminal state
            (either 'complete' or 'failed'). False while still collecting.
        """
        if self._status != 'collecting':
            return True

        self._total_frames += 1

        if face_visible:
            self._yaw_samples.append(head_yaw)
            self._pitch_samples.append(head_pitch)
            # Only collect EAR samples that are physiologically plausible.
            # mean_ear=0.0 indicates a sensor dropout, not a closed eye.
            if mean_ear > 0.05:
                self._ear_samples.append(mean_ear)

        n_valid = len(self._yaw_samples)

        # Success path: enough valid frames accumulated
        if n_valid >= self._min_valid_frames:
            return self._finalize()

        # Early failure: target duration elapsed but insufficient visible frames
        if self._total_frames >= self._expected_frames and n_valid < self._min_valid_frames:
            self._fail()
            return True

        # Hard timeout: twice the target duration without enough valid frames
        if self._total_frames >= self._timeout_frames:
            self._fail()
            return True

        return False

    def reset(self) -> None:
        """Reset to initial collecting state."""
        self._yaw_samples = []
        self._pitch_samples = []
        self._ear_samples = []
        self._total_frames = 0
        self._status = 'collecting'
        self._neutral_yaw_offset = 0.0
        self._neutral_pitch_offset = 0.0
        self._baseline_ear = EAR_BASELINE_POPULATION_DEFAULT
        self._close_threshold = EAR_DEFAULT_CLOSE_THRESHOLD
        self._quality_ok = False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_complete(self) -> bool:
        """True once calibration has reached a terminal state (success or failure)."""
        return self._status != 'collecting'

    @property
    def status(self) -> str:
        """Current status: 'collecting' | 'complete' | 'failed'."""
        return self._status

    @property
    def quality_ok(self) -> bool:
        """True if calibration succeeded with pose std below threshold."""
        return self._quality_ok

    @property
    def neutral_yaw_offset(self) -> float:
        """Mean yaw from calibration window (degrees). 0.0 on failure."""
        return self._neutral_yaw_offset

    @property
    def neutral_pitch_offset(self) -> float:
        """Mean pitch from calibration window (degrees). 0.0 on failure."""
        return self._neutral_pitch_offset

    @property
    def baseline_ear(self) -> float:
        """Mean open-eye EAR from calibration window. Population default on failure."""
        return self._baseline_ear

    @property
    def close_threshold(self) -> float:
        """EAR close threshold = baseline_ear * 0.75. Default 0.21 on failure."""
        return self._close_threshold

    # ── Private helpers ───────────────────────────────────────────────────────

    def _finalize(self) -> bool:
        """Compute results from collected samples and apply quality gate."""
        n = len(self._yaw_samples)
        yaw_mean = sum(self._yaw_samples) / n
        pitch_mean = sum(self._pitch_samples) / n
        # EAR samples may be fewer than pose samples (invalid values filtered in feed_frame)
        ear_mean = sum(self._ear_samples) / len(self._ear_samples) if self._ear_samples else 0.0

        yaw_std = math.sqrt(sum((y - yaw_mean) ** 2 for y in self._yaw_samples) / n)
        pitch_std = math.sqrt(sum((p - pitch_mean) ** 2 for p in self._pitch_samples) / n)

        if (
            yaw_std < CALIBRATION_MAX_POSE_STD_DEG
            and pitch_std < CALIBRATION_MAX_POSE_STD_DEG
        ):
            self._neutral_yaw_offset = yaw_mean
            self._neutral_pitch_offset = pitch_mean
            if self._ear_samples:
                self._baseline_ear = ear_mean
                self._close_threshold = ear_mean * EAR_CALIBRATION_MULTIPLIER
            # else: no valid EAR samples — keep population defaults (0.28 / 0.21)
            self._quality_ok = True
            self._status = 'complete'
        else:
            self._fail()

        return True

    def _fail(self) -> None:
        """Enter failed state with safe defaults."""
        self._neutral_yaw_offset = 0.0
        self._neutral_pitch_offset = 0.0
        self._baseline_ear = EAR_BASELINE_POPULATION_DEFAULT
        self._close_threshold = EAR_DEFAULT_CLOSE_THRESHOLD
        self._quality_ok = False
        self._status = 'failed'
