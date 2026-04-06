"""Pipeline Manager v2 — full pipeline wiring for Phase 7B.

Connects all layers in order:
  FrameSource → FaceDetector + ObjectDetector → adapters →
  SignalProcessor → TemporalEngine → ScoringEngine →
  AlertStateMachine → AudioAlerterV2 + EventLogger + display

Calibration runs during the first ~5 seconds before scoring begins.

PRD §4 — Pipeline architecture.
"""

import logging
import time
from typing import Optional

import cv2
import numpy as np

from src.config_loader import (
    FaceDetectorConfig,
    FrameSourceConfig,
    ObjectDetectorConfig,
)
from src.config_prd import (
    CALIBRATION_DURATION_S,
    CAPTURE_FPS,
    COOLDOWN_VISUAL,
    DEGRADED_RECOVERY_FRAMES,
    DEGRADED_TRIGGER_FRAMES,
)
from src.contracts import AlertCommand, AlertLevel, SignalFrame, TemporalFeatures
from src.detection.face_detector import FaceDetector
from src.detection.object_detector import ObjectDetector
from src.logic.alert_state_machine import AlertStateMachine
from src.logic.calibration import Calibration
from src.logic.scoring_engine import ScoringEngine
from src.logic.signal_processor import SignalProcessor
from src.logic.temporal_engine import TemporalEngine
from src.output.audio_alerter_v2 import play_alert
from src.output.event_logger import EventLogger
from src.pipeline.adapters import convert_to_perception_bundle
from src.pipeline.frame_source import FrameSource

logger = logging.getLogger(__name__)

# Overlay colours (BGR)
_GREEN  = (0, 200, 0)
_YELLOW = (0, 200, 200)
_RED    = (0, 0, 220)
_WHITE  = (255, 255, 255)
_CYAN   = (220, 200, 0)
_ORANGE = (0, 140, 255)

_WINDOW = 'Attentia Drive'
_ALERT_FLASH_S = 1.0  # seconds to flash "ALERT!" after firing


class PipelineManagerV2:
    """Wires all pipeline layers into a single blocking run loop.

    Args:
        source: Override frame source — None (webcam), 'webcam', or video path.
        display: Show OpenCV debug window.
    """

    def __init__(
        self,
        source: Optional[str] = None,
        display: bool = True,
    ) -> None:
        self._display = display

        # ── Frame source ───────────────────────────────────────────────────────
        if source and source != 'webcam':
            fs_config = FrameSourceConfig(type='video', video_path=source)
        else:
            fs_config = FrameSourceConfig(type='webcam', webcam_index=0)
        self._frame_source = FrameSource(fs_config)

        # ── Detectors ─────────────────────────────────────────────────────────
        face_config = FaceDetectorConfig(enabled=True)
        self._face_detector = FaceDetector(face_config)

        obj_config = ObjectDetectorConfig(enabled=True)
        self._object_detector = ObjectDetector(obj_config)

        # ── Pipeline layers ────────────────────────────────────────────────────
        self._signal_processor = SignalProcessor()
        self._temporal_engine  = TemporalEngine()
        self._scoring_engine   = ScoringEngine()
        self._alert_sm         = AlertStateMachine()
        self._calibration      = Calibration()

        # ── Output modules ─────────────────────────────────────────────────────
        self._event_logger = EventLogger()

        # ── Runtime state ──────────────────────────────────────────────────────
        self._frame_id: int = 0
        self._last_alert_time: float = 0.0
        self._last_alert_cmd: Optional[AlertCommand] = None

        # Lightweight degraded tracking for overlay (mirrors ASM internal logic)
        self._invalid_streak: int = 0
        self._recovery_streak: int = 0
        self._display_degraded: bool = False

    # ── Public entry point ─────────────────────────────────────────────────────

    def run(self) -> None:
        """Block until the user presses 'q' or the frame source ends."""
        logger.info("Pipeline started — press 'q' to quit")

        try:
            self._loop()
        finally:
            self._frame_source.release()
            cv2.destroyAllWindows()
            logger.info("Pipeline stopped")

    # ── Main loop ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while True:
            ok, frame = self._frame_source.read()
            if not ok or frame is None:
                logger.info("Frame source exhausted — stopping")
                break

            self._frame_id += 1
            timestamp_ns = time.monotonic_ns()

            try:
                self._process_frame(frame, timestamp_ns)
            except Exception:
                logger.exception("Frame %d: unhandled exception — continuing", self._frame_id)

            if self._display:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def _process_frame(self, frame: np.ndarray, timestamp_ns: int) -> None:
        # ── Detection ──────────────────────────────────────────────────────────
        face_result      = self._face_detector.detect(frame)
        object_detections = self._object_detector.detect(frame)

        # ── Adapter ────────────────────────────────────────────────────────────
        bundle = convert_to_perception_bundle(
            face_result, object_detections, self._frame_id, timestamp_ns
        )

        # ── Calibration phase ──────────────────────────────────────────────────
        if not self._calibration.is_complete:
            if face_result.face_visible and face_result.head_pose is not None:
                pitch, yaw, roll = face_result.head_pose
                if face_result.ear_left is not None and face_result.ear_right is not None:
                    mean_ear = (face_result.ear_left + face_result.ear_right) / 2.0
                else:
                    mean_ear = 0.0
                done = self._calibration.feed_frame(yaw, pitch, mean_ear, True)
                if done:
                    self._signal_processor.set_neutral_offsets(
                        self._calibration.neutral_yaw_offset,
                        self._calibration.neutral_pitch_offset,
                    )
                    self._signal_processor.set_ear_baseline(
                        self._calibration.baseline_ear,
                        self._calibration.close_threshold,
                    )
                    self._event_logger.log_calibration(
                        self._calibration.neutral_yaw_offset,
                        self._calibration.neutral_pitch_offset,
                        self._calibration.baseline_ear,
                    )
                    self._event_logger.log_state_transition(
                        'CALIBRATING', 'NOMINAL', 'calibration_complete', self._frame_id
                    )
                    logger.info(
                        "Calibration complete — yaw_offset=%.2f pitch_offset=%.2f baseline_ear=%.3f",
                        self._calibration.neutral_yaw_offset,
                        self._calibration.neutral_pitch_offset,
                        self._calibration.baseline_ear,
                    )

            if self._display:
                self._draw_calibrating(frame)
                cv2.imshow(_WINDOW, frame)
            return

        # ── Active pipeline ────────────────────────────────────────────────────
        signal_frame      = self._signal_processor.process(bundle)
        temporal_features = self._temporal_engine.process(signal_frame)
        distraction_score = self._scoring_engine.score(temporal_features)
        alert_cmd         = self._alert_sm.update(distraction_score, signal_frame.signals_valid)

        # Update degraded display state
        self._update_degraded_display(signal_frame.signals_valid)

        # Handle alert
        if alert_cmd is not None:
            play_alert(alert_cmd.level)
            self._event_logger.log_alert(alert_cmd, distraction_score)
            self._last_alert_time = time.monotonic()
            self._last_alert_cmd = alert_cmd
            logger.info(
                "ALERT: %s %s score=%.3f",
                alert_cmd.level.name,
                alert_cmd.alert_type.value,
                distraction_score.composite_score,
            )

        if self._display:
            self._draw_overlay(frame, signal_frame, temporal_features, distraction_score, alert_cmd)
            cv2.imshow(_WINDOW, frame)

    # ── Degraded display tracking ──────────────────────────────────────────────

    def _update_degraded_display(self, signals_valid: bool) -> None:
        if not signals_valid:
            self._invalid_streak += 1
            self._recovery_streak = 0
            if self._invalid_streak >= DEGRADED_TRIGGER_FRAMES:
                self._display_degraded = True
        else:
            self._invalid_streak = 0
            if self._display_degraded:
                self._recovery_streak += 1
                if self._recovery_streak >= DEGRADED_RECOVERY_FRAMES:
                    self._display_degraded = False
                    self._recovery_streak = 0

    # ── Display helpers ────────────────────────────────────────────────────────

    def _draw_calibrating(self, frame: np.ndarray) -> None:
        """Draw calibration overlay."""
        h, w = frame.shape[:2]
        # Semi-transparent dark strip at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Count of collected frames (each call to feed_frame with visible=True)
        target = int(CAPTURE_FPS * CALIBRATION_DURATION_S)
        # We don't have direct access to sample count; show time estimate instead
        cv2.putText(frame, 'CALIBRATING — hold still',
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, _YELLOW, 2)
        cv2.putText(frame, f'Target: {target} frames at {CAPTURE_FPS} fps (~{CALIBRATION_DURATION_S:.0f}s)',
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1)

    def _draw_overlay(
        self,
        frame: np.ndarray,
        sf: SignalFrame,
        tf: TemporalFeatures,
        ds,
        alert_cmd: Optional[AlertCommand],
    ) -> None:
        """Draw full debug overlay onto frame in-place."""
        h, w = frame.shape[:2]

        # ── Background strip ──────────────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 180), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # ── State label ───────────────────────────────────────────────────────
        now = time.monotonic()
        in_cooldown = (now - self._last_alert_time) < COOLDOWN_VISUAL
        if self._display_degraded:
            state_label, state_color = 'DEGRADED', _RED
        elif in_cooldown:
            state_label, state_color = 'COOLDOWN', _YELLOW
        else:
            state_label, state_color = 'NOMINAL', _GREEN
        cv2.putText(frame, state_label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, state_color, 2)

        # ── Head pose ─────────────────────────────────────────────────────────
        if sf.head_pose is not None:
            yaw   = sf.head_pose.yaw_deg
            pitch = sf.head_pose.pitch_deg
            roll  = sf.head_pose.roll_deg
            cv2.putText(frame,
                        f'Head  yaw={yaw:+.1f}  pitch={pitch:+.1f}  roll={roll:+.1f}',
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, _WHITE, 1)
        else:
            cv2.putText(frame, 'Head  ---', (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, _YELLOW, 1)

        # ── EAR / gaze ────────────────────────────────────────────────────────
        if sf.eye_signals is not None:
            ear_l = sf.eye_signals.left_EAR
            ear_r = sf.eye_signals.right_EAR
        else:
            ear_l = ear_r = 0.0
        on_road = sf.gaze_world.on_road if sf.gaze_world is not None else True
        road_str = 'ON ROAD' if on_road else 'OFF ROAD'
        road_col = _GREEN if on_road else _RED
        cv2.putText(frame,
                    f'EAR  L={ear_l:.3f}  R={ear_r:.3f}',
                    (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.52, _WHITE, 1)
        cv2.putText(frame, road_str, (w - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, road_col, 2)

        # ── Timers ────────────────────────────────────────────────────────────
        timer_txt = (
            f'gaze={tf.gaze_continuous_secs:.1f}s  '
            f'head={tf.head_continuous_secs:.1f}s  '
            f'phone={tf.phone_continuous_secs:.1f}s'
        )
        cv2.putText(frame, timer_txt, (10, 106),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, _CYAN, 1)

        # ── Composite score bar ───────────────────────────────────────────────
        bar_x, bar_y, bar_w, bar_h = 10, 120, 280, 16
        score_clamped = min(ds.composite_score, 1.0)
        fill_w = int(bar_w * score_clamped)
        bar_color = _RED if score_clamped >= 0.55 else _YELLOW if score_clamped >= 0.35 else _GREEN
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        # Threshold line at 0.55
        thresh_x = bar_x + int(bar_w * 0.55)
        cv2.line(frame, (thresh_x, bar_y - 2), (thresh_x, bar_y + bar_h + 2), _WHITE, 1)
        cv2.putText(frame, f'Score: {ds.composite_score:.3f}',
                    (bar_x + bar_w + 8, bar_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, _WHITE, 1)

        # ── Active classes ────────────────────────────────────────────────────
        if ds.active_classes:
            classes_str = '  '.join(ds.active_classes)
            cv2.putText(frame, f'Active: {classes_str}',
                        (10, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.52, _ORANGE, 1)

        # ── ALERT! flash ──────────────────────────────────────────────────────
        if now - self._last_alert_time < _ALERT_FLASH_S and self._last_alert_cmd is not None:
            level_str = self._last_alert_cmd.level.name
            atype_str = self._last_alert_cmd.alert_type.value
            cv2.putText(frame, f'ALERT!  {level_str}  {atype_str}',
                        (w // 2 - 140, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, _RED, 3)

        # ── Frame id ──────────────────────────────────────────────────────────
        cv2.putText(frame, f'#{self._frame_id}',
                    (w - 80, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)
