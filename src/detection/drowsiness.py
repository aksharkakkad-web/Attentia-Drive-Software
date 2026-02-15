"""Drowsiness detector using EAR-based PERCLOS, blink counting, and yawn detection.

Consumes FaceResult per frame and maintains rolling state to compute:
- PERCLOS (percentage of eye closure over a time window)
- Blink rate (blinks per minute)
- Yawn detection (sustained high MAR)
- Fused drowsiness confidence score
"""

import logging
from collections import deque
from dataclasses import dataclass

from src.config_loader import DrowsinessConfig
from src.detection.face_detector import FaceResult

logger = logging.getLogger(__name__)


@dataclass
class DrowsinessResult:
    """Result from the drowsiness detector."""

    eye_closure_duration: float = 0.0  # Seconds eyes have been closed
    blink_rate: float = 0.0  # Blinks per minute
    is_drowsy: bool = False
    drowsiness_confidence: float = 0.0  # [0, 1]
    perclos: float = 0.0  # Percentage of eye closure [0, 1]
    is_yawning: bool = False
    yawn_count: int = 0
    blink_count: int = 0


class DrowsinessDetector:
    """Drowsiness detector consuming face landmarks.

    Tracks EAR over time for PERCLOS and blink detection,
    MAR for yawn detection, and fuses into a drowsiness score.

    Args:
        config: DrowsinessConfig with thresholds.
        fps: Expected frames per second for time-based calculations.
    """

    def __init__(self, config: DrowsinessConfig, fps: float = 30.0) -> None:
        self._config = config
        self._fps = max(1.0, fps)

        window_frames = int(config.perclos_window_seconds * self._fps)
        self._ear_history: deque = deque(maxlen=max(window_frames, 1))

        self._eyes_closed_frames = 0
        self._blink_count = 0
        self._yawn_count = 0
        self._yawn_consecutive = 0
        self._in_blink = False
        self._in_yawn = False
        self._total_frames = 0

        logger.info(
            "DrowsinessDetector: Initialized (fps=%.1f, PERCLOS window=%d frames)",
            self._fps, window_frames,
        )

    def update(self, face_result: FaceResult) -> DrowsinessResult:
        """Update drowsiness state from the latest face result.

        Args:
            face_result: Face detection result for the current frame.

        Returns:
            DrowsinessResult with computed metrics.
        """
        if not self._config.enabled:
            return DrowsinessResult()

        if not face_result.face_visible or face_result.ear_left is None:
            return DrowsinessResult(
                blink_count=self._blink_count,
                yawn_count=self._yawn_count,
            )

        self._total_frames += 1
        avg_ear = (face_result.ear_left + face_result.ear_right) / 2.0
        eyes_closed = avg_ear < self._config.ear_threshold

        # Track EAR for PERCLOS
        self._ear_history.append(eyes_closed)

        # Eye closure duration
        if eyes_closed:
            self._eyes_closed_frames += 1
        else:
            self._eyes_closed_frames = 0

        eye_closure_duration = self._eyes_closed_frames / self._fps

        # Blink detection: short closure (< 3 frames below threshold)
        if eyes_closed and not self._in_blink:
            self._in_blink = True
            self._blink_start_frames = 1
        elif eyes_closed and self._in_blink:
            self._blink_start_frames += 1
        elif not eyes_closed and self._in_blink:
            if self._blink_start_frames <= 3:
                self._blink_count += 1
            self._in_blink = False

        # Blink rate (blinks per minute)
        elapsed_seconds = self._total_frames / self._fps
        if elapsed_seconds > 0:
            blink_rate = (self._blink_count / elapsed_seconds) * 60.0
        else:
            blink_rate = 0.0

        # PERCLOS
        if len(self._ear_history) > 0:
            perclos = sum(self._ear_history) / len(self._ear_history)
        else:
            perclos = 0.0

        # Yawn detection
        is_yawning = False
        if face_result.mouth_aspect_ratio is not None:
            if face_result.mouth_aspect_ratio > self._config.mar_threshold:
                self._yawn_consecutive += 1
                if self._yawn_consecutive >= self._config.yawn_min_frames:
                    is_yawning = True
                    if not self._in_yawn:
                        self._yawn_count += 1
                        self._in_yawn = True
            else:
                self._yawn_consecutive = 0
                self._in_yawn = False

        # Fusion: weighted combination
        perclos_weight = 0.5
        closure_weight = 0.3
        yawn_weight = 0.2

        perclos_score = min(1.0, perclos / max(self._config.perclos_threshold, 1e-6))
        closure_score = min(1.0, eye_closure_duration / 2.0)  # 2 sec -> 1.0
        yawn_score = 1.0 if is_yawning else 0.0

        drowsiness_confidence = (
            perclos_weight * perclos_score
            + closure_weight * closure_score
            + yawn_weight * yawn_score
        )
        drowsiness_confidence = min(1.0, drowsiness_confidence)

        is_drowsy = (
            perclos > self._config.perclos_threshold
            or eye_closure_duration > 1.5
            or is_yawning
        )

        return DrowsinessResult(
            eye_closure_duration=eye_closure_duration,
            blink_rate=blink_rate,
            is_drowsy=is_drowsy,
            drowsiness_confidence=drowsiness_confidence,
            perclos=perclos,
            is_yawning=is_yawning,
            yawn_count=self._yawn_count,
            blink_count=self._blink_count,
        )
