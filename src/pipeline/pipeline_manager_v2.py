"""Pipeline Manager v2 — full pipeline wiring for Phase 7B.

Connects all layers in order:
  FrameSource → FaceDetector + ObjectDetector → adapters →
  SignalProcessor → TemporalEngine → ScoringEngine →
  AlertStateMachine → AudioAlerterV2 + EventLogger + display

Calibration runs during the first ~5 seconds before scoring begins.

PRD §4 — Pipeline architecture.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config_loader import (
    AppConfig,
    FaceDetectorConfig,
    FrameSourceConfig,
    ObjectDetectorConfig,
    load_config,
)
from src.config_prd import (
    COOLDOWN_VISUAL,
    DEGRADED_RECOVERY_FRAMES,
    DEGRADED_TRIGGER_FRAMES,
    YOLO_PHONE_MODEL_PATH,
)
from src.contracts import AlertCommand, SignalFrame, TemporalFeatures
from src.detection.face_detector import FaceDetector
from src.detection.object_detector import ObjectDetector
from src.detection.phone_detector_yolo import PhoneDetectorYolo
from src.logic.alert_state_machine import AlertStateMachine
from src.logic.calibration import Calibration
from src.logic.scoring_engine import ScoringEngine
from src.logic.signal_processor import SignalProcessor
from src.logic.temporal_engine import TemporalEngine
from src.output.event_logger import EventLogger
from src.pipeline.adapters import convert_to_perception_bundle
from src.pipeline.interfaces import (
    AfplayAudioSink,
    AudioSink,
    FrameSource,
    ImuSource,
    NoopThermalMonitor,
    OpenCVFrameSource,
    StubImuSource,
    ThermalMonitor,
)
from src.pipeline.stage_timer import StageTimer

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
        source: Override frame source — None (use config), 'webcam', or video path.
        display: Show OpenCV debug window.
        config_path: Path to config.yaml (default: 'config.yaml').
    """

    def __init__(
        self,
        source: Optional[str] = None,
        display: bool = True,
        config_path: str = 'config.yaml',
        save_frames: bool = False,
    ) -> None:
        self._display = display

        # ── Debug frame saver ──────────────────────────────────────────────────
        self._save_frames = save_frames
        self._save_dir: Optional[Path] = None
        self._save_log = None  # open file handle for session_log.jsonl
        self._frames_since_save: int = 0
        if save_frames:
            session_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            self._save_dir = Path('debug_frames') / session_ts
            self._save_dir.mkdir(parents=True, exist_ok=True)
            self._save_log = open(self._save_dir / 'session_log.jsonl', 'w')
            logger.info("Frame saver: writing to %s", self._save_dir)

        # ── Load config ────────────────────────────────────────────────────────
        try:
            config: AppConfig = load_config(config_path)
        except FileNotFoundError:
            logger.warning("config.yaml not found at '%s' — using defaults", config_path)
            config = AppConfig()

        # ── Frame source ───────────────────────────────────────────────────────
        # --source CLI arg overrides config.yaml frame_source section
        if source is not None and source != 'webcam':
            fs_config = FrameSourceConfig(type='video', video_path=source)
        elif source == 'webcam':
            fs_config = FrameSourceConfig(
                type='webcam',
                webcam_index=config.frame_source.webcam_index,
                resolution=config.frame_source.resolution,
                target_fps=config.frame_source.target_fps,
            )
        else:
            # Use config entirely (webcam or video as specified in config.yaml)
            fs_config = config.frame_source
        self._frame_source: FrameSource = OpenCVFrameSource(fs_config)
        if not self._frame_source.open():
            logger.error("Pipeline: frame source failed to open")

        # ── Portability interface sinks ────────────────────────────────────────
        # DEV-007: AfplayAudioSink wraps the existing afplay subprocess path.
        # Day 5 replaces with SoundDeviceAudioSink (preloaded PCM + mutex).
        self._audio_sink: AudioSink = AfplayAudioSink()
        if not self._audio_sink.open():
            logger.warning("Pipeline: audio sink failed to open — alerts will be silent")

        # DEV-006: Mac has no IMU; StubImuSource returns valid=False.
        # SignalProcessor wiring of IMU data is deferred — pipeline currently
        # reads and discards so the interface is exercised end-to-end.
        self._imu_source: ImuSource = StubImuSource()
        if not self._imu_source.open():
            logger.warning("Pipeline: IMU source failed to open — IMU data unavailable")

        self._thermal_monitor: ThermalMonitor = NoopThermalMonitor()
        if not self._thermal_monitor.open():
            logger.warning("Pipeline: thermal monitor failed to open — throttling disabled")

        # ── Per-stage timing instrumentation ──────────────────────────────────
        self._stage_timer = StageTimer(report_interval_s=2.0)

        # ── Detectors ─────────────────────────────────────────────────────────
        # Face detection always enabled for Phase 7B — override config flag
        face_config = config.face_detector
        face_config.enabled = True
        self._face_detector = FaceDetector(face_config)

        obj_config = config.object_detector
        obj_config.enabled = True

        # YOLOv8-nano ONNX phone detector.
        # PRD §FR-1.4 specifies every-frame detection; MVP override: frame_skip=2
        # halves YOLO cost (~17ms saved/frame) with no perceptible accuracy loss
        # on a laptop. Revert to frame_skip=1 for production RKNN deployment.
        from dataclasses import replace as _replace
        yolo_config = _replace(
            obj_config,
            model_path=YOLO_PHONE_MODEL_PATH,
            target_classes=['cell phone'],
            frame_skip=2,  # MVP override — revert to 1 for production
        )
        self._phone_detector_yolo = PhoneDetectorYolo(yolo_config)

        if self._phone_detector_yolo.is_available():
            # YOLO handles phones; ObjectDetector handles all other target classes
            # (cup, bottle, book, laptop) with no phone path active.
            non_phone_classes = [c for c in obj_config.target_classes if c != 'cell phone']
            non_phone_config = _replace(obj_config, target_classes=non_phone_classes)
            self._object_detector = ObjectDetector(non_phone_config)
            logger.info("Phone detection: YOLOv8-nano ONNX (primary)")
        else:
            # YOLO failed — ObjectDetector handles all targets including phone.
            self._object_detector = ObjectDetector(obj_config)
            logger.warning(
                "Phone detection: YOLOv8 ONNX unavailable — falling back to ObjectDetector"
            )

        # ── Pipeline layers ────────────────────────────────────────────────────
        fps = float(config.frame_source.target_fps)
        self._signal_processor = SignalProcessor(dt=1.0 / fps)
        self._temporal_engine  = TemporalEngine(fps=fps)
        self._scoring_engine   = ScoringEngine()
        self._alert_sm         = AlertStateMachine()
        self._calibration      = Calibration(fps=fps)

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

        # Calibration outcome — None until calibration finishes, then 'ok'/'failed'
        self._cal_status: Optional[str] = None

    # ── Public entry point ─────────────────────────────────────────────────────

    def run(self) -> None:
        """Block until the user presses 'q' or the frame source ends."""
        logger.info("Pipeline started — press 'q' to quit")

        try:
            self._loop()
        finally:
            self._frame_source.close()
            self._audio_sink.close()
            self._imu_source.close()
            self._thermal_monitor.close()
            cv2.destroyAllWindows()
            if self._save_log is not None:
                self._save_log.close()
                logger.info("Frame saver: session closed — %s", self._save_dir)
            logger.info("Pipeline stopped")

    # ── Main loop ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while True:
            with self._stage_timer.measure('capture'):
                result = self._frame_source.read()
            if not result.ok or result.frame is None:
                logger.info("Frame source exhausted — stopping")
                break
            frame = result.frame

            # Read IMU every frame; data is discarded until DEV-006 wires it
            # into SignalProcessor. Exercising the interface prevents bit-rot.
            with self._stage_timer.measure('imu'):
                _ = self._imu_source.read()

            self._frame_id += 1
            timestamp_ns = time.monotonic_ns()

            try:
                with self._stage_timer.measure('frame_total'):
                    self._process_frame(frame, timestamp_ns)
            except Exception:
                logger.exception("Frame %d: unhandled exception — continuing", self._frame_id)

            self._stage_timer.maybe_report()

            if self._display:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def _process_frame(self, frame: np.ndarray, timestamp_ns: int) -> None:
        # ── Detection ──────────────────────────────────────────────────────────
        face_result       = self._face_detector.detect(frame)
        # YOLO handles phones every other frame (frame_skip=2). ObjectDetector handles
        # all other classes (or all classes when YOLO is unavailable). No merging
        # needed — target_classes on each detector are mutually exclusive at init.
        phone_detections  = self._phone_detector_yolo.detect(frame)
        other_detections  = self._object_detector.detect(frame)
        object_detections = phone_detections + other_detections

        if logger.isEnabledFor(logging.DEBUG) and object_detections:
            summary = ', '.join(
                f"{d.class_name}={d.confidence:.2f}" for d in object_detections
            )
            logger.debug("Frame %d objects: %s", self._frame_id, summary)

        # ── Adapter ────────────────────────────────────────────────────────────
        bundle = convert_to_perception_bundle(
            face_result, object_detections, self._frame_id, timestamp_ns
        )

        # ── Calibration phase ──────────────────────────────────────────────────
        if not self._calibration.is_complete:
            # Always feed calibration — even when face is absent — so the
            # internal frame count advances and the hard timeout can fire.
            if face_result.face_visible and face_result.head_pose is not None:
                pitch, yaw, roll = face_result.head_pose
                if face_result.ear_left is not None and face_result.ear_right is not None:
                    mean_ear = (face_result.ear_left + face_result.ear_right) / 2.0
                else:
                    mean_ear = 0.0
                done = self._calibration.feed_frame(yaw, pitch, mean_ear, face_visible=True)
            else:
                done = self._calibration.feed_frame(0.0, 0.0, 0.0,
                                                     face_visible=face_result.face_visible)

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
                    "Calibration %s — yaw_offset=%.2f pitch_offset=%.2f baseline_ear=%.3f",
                    self._calibration.status,
                    self._calibration.neutral_yaw_offset,
                    self._calibration.neutral_pitch_offset,
                    self._calibration.baseline_ear,
                )
                self._cal_status = 'ok' if self._calibration.status == 'complete' else 'failed'
                if self._calibration.status == 'failed':
                    logger.warning(
                        "Calibration failed — reason: %s | valid_frames=%d min_required=%d",
                        self._calibration.failure_reason,
                        self._calibration.valid_frame_count,
                        self._calibration.min_valid_frame_count,
                    )

            if self._display or self._save_frames:
                self._draw_calibrating(
                    frame,
                    valid=self._calibration.valid_frame_count,
                    min_valid=self._calibration.min_valid_frame_count,
                    total=self._calibration.total_frame_count,
                    expected=self._calibration.expected_frame_count,
                )
                if self._display:
                    cv2.imshow(_WINDOW, frame)
            if self._save_frames:
                self._frames_since_save += 1
                if self._frames_since_save >= 30:
                    self._frames_since_save = 0
                    frame_name = f'frame_{self._frame_id:06d}_cal.jpg'
                    cv2.imwrite(str(self._save_dir / frame_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
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
            self._audio_sink.play(alert_cmd.alert_type, alert_cmd.level)
            self._event_logger.log_alert(alert_cmd, distraction_score)
            self._last_alert_time = time.monotonic()
            self._last_alert_cmd = alert_cmd
            logger.info(
                "ALERT: %s %s score=%.3f",
                alert_cmd.level.name,
                alert_cmd.alert_type.value,
                distraction_score.composite_score,
            )

        if self._display or self._save_frames:
            self._draw_overlay(frame, signal_frame, temporal_features, distraction_score)
            if self._display:
                cv2.imshow(_WINDOW, frame)

        if self._save_frames:
            self._maybe_save_frame(frame, signal_frame, temporal_features, distraction_score, alert_cmd)

    # ── Frame saver ────────────────────────────────────────────────────────────

    def _maybe_save_frame(self, frame, signal_frame, temporal_features, distraction_score, alert_cmd) -> None:
        """Save one JPEG + one JSONL log line per second."""
        self._frames_since_save += 1
        fps_approx = 30
        if self._frames_since_save < fps_approx:
            return
        self._frames_since_save = 0

        frame_name = f'frame_{self._frame_id:06d}.jpg'
        cv2.imwrite(str(self._save_dir / frame_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])

        hp = signal_frame.head_pose
        gw = signal_frame.gaze_world
        es = signal_frame.eye_signals
        record = {
            'frame': self._frame_id,
            'ts': datetime.now().isoformat(),
            'face_present': signal_frame.face_present,
            'signals_valid': signal_frame.signals_valid,
            'head_yaw': round(hp.yaw_deg, 2) if hp else None,
            'head_pitch': round(hp.pitch_deg, 2) if hp else None,
            'on_road': gw.on_road if gw else None,
            'mean_ear': round(es.mean_EAR, 3) if es else None,
            'phone_detected': signal_frame.phone_signal.detected,
            'phone_conf': round(signal_frame.phone_signal.confidence, 3),
            'gaze_secs': round(temporal_features.gaze_continuous_secs, 2),
            'head_secs': round(temporal_features.head_continuous_secs, 2),
            'phone_secs': round(temporal_features.phone_continuous_secs, 2),
            'perclos': round(temporal_features.perclos, 3),
            'composite': round(distraction_score.composite_score, 3),
            'breaches': {
                'gaze': distraction_score.gaze_threshold_breached,
                'head': distraction_score.head_threshold_breached,
                'phone': distraction_score.phone_threshold_breached,
                'perclos': distraction_score.perclos_threshold_breached,
                'composite': distraction_score.composite_threshold_breached,
            },
            'alert': alert_cmd.alert_type.value if alert_cmd else None,
            'image': frame_name,
        }
        self._save_log.write(json.dumps(record) + '\n')
        self._save_log.flush()

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

    def _draw_calibrating(
        self,
        frame: np.ndarray,
        valid: int,
        min_valid: int,
        total: int,
        expected: int,
    ) -> None:
        """Draw calibration overlay with frame progress."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, 'CALIBRATING - hold still',
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, _YELLOW, 2)
        cv2.putText(frame,
                    f'Valid: {valid}/{min_valid} | Total: {total}/{expected}',
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1)

    def _draw_overlay(
        self,
        frame: np.ndarray,
        sf: SignalFrame,
        tf: TemporalFeatures,
        ds,
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

        # ── Calibration quality indicator ─────────────────────────────────────
        if self._cal_status == 'ok':
            cv2.putText(frame, 'CAL OK', (w - 100, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _GREEN, 1)
        elif self._cal_status == 'failed':
            cv2.putText(frame, 'CAL FAIL', (w - 110, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _RED, 1)

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
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
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
