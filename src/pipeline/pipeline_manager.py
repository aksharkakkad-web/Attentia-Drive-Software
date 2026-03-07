"""Pipeline manager that orchestrates the full Attentia Drive processing chain.

Runs the main loop: frame -> detection -> reasoning -> alerting -> display.
Handles graceful shutdown on keyboard interrupt or end of video.

Phase: 1-2 (active).
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import numpy as np

from src.config_loader import AppConfig
from src.detection.classifier import ClassifierResult, DistractionClassifier
from src.detection.drowsiness import DrowsinessDetector, DrowsinessResult
from src.detection.face_detector import FaceDetector, FaceResult
from src.detection.object_detector import ObjectDetection, ObjectDetector
from src.logic.alert_manager import AlertManager
from src.logic.distraction_reasoner import DistractionReasoner
from src.logic.hysteresis import HysteresisStateMachine
from src.logic.object_confirmer import ObjectTemporalConfirmer
from src.logic.temporal_smoother import TemporalSmoother
from src.output.audio_alerter import AudioAlerter
from src.output.display import Display
from src.output.telemetry_logger import TelemetryLogger
from src.pipeline.frame_record import FrameRecord
from src.pipeline.frame_source import FrameSource
from src.utils.fps_counter import FPSCounter
from src.utils.timer import Timer

logger = logging.getLogger(__name__)


class PipelineManager:
    """Orchestrates the full Attentia Drive processing pipeline.

    Instantiates all modules (frame source, detectors, logic, output) and
    runs the main processing loop. Times the full pipeline per frame.

    Args:
        config: Root AppConfig with all subsystem configurations.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config

        self._frame_source = FrameSource(config.frame_source)
        self._classifier = DistractionClassifier(config.classifier)
        self._object_detector = ObjectDetector(config.object_detector)
        self._face_detector = FaceDetector(config.face_detector)
        self._drowsiness_detector = DrowsinessDetector(
            config=config.drowsiness,
            fps=config.frame_source.target_fps,
        )

        self._smoother = TemporalSmoother(config.temporal_smoothing.alpha)
        self._hysteresis = HysteresisStateMachine(config.hysteresis)
        self._object_confirmer = ObjectTemporalConfirmer(config.object_confirmation)
        self._reasoner = DistractionReasoner(
            face_detector_enabled=config.face_detector.enabled
        )
        self._alert_manager = AlertManager(config.alert_manager)

        self._display = Display(config.display)
        self._audio_alerter = AudioAlerter(config.audio)
        self._telemetry = TelemetryLogger(config.telemetry)
        self._fps_counter = FPSCounter()

        self._frame_number = 0
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="inference")

    def run(self) -> None:
        """Run the main processing loop.

        Reads frames, runs all pipeline stages, outputs results.
        Exits cleanly on end of video, quit key, or keyboard interrupt.
        """
        if not self._frame_source.is_opened:
            logger.error("PipelineManager: Frame source not available. Exiting.")
            return

        logger.info("PipelineManager: Starting pipeline...")

        try:
            while True:
                with Timer() as pipeline_timer:
                    record = self._process_frame()

                if record is None:
                    logger.info("PipelineManager: End of input. Exiting.")
                    break

                record.pipeline_time_ms = pipeline_timer.elapsed_ms

                self._output(record)

                self._fps_counter.tick()

                if self._config.display.enabled:
                    key = self._display.wait_key()
                    if key == ord("q"):
                        logger.info("PipelineManager: Quit key pressed. Exiting.")
                        break

        except KeyboardInterrupt:
            logger.info("PipelineManager: Keyboard interrupt. Shutting down.")
        finally:
            self._shutdown()

    def _process_frame(self) -> Optional[FrameRecord]:
        """Process a single frame through all pipeline stages.

        Returns:
            FrameRecord with all results, or None if no frame available.
        """
        success, frame = self._frame_source.read()
        if not success or frame is None:
            return None

        self._frame_number += 1
        timestamp = time.time()

        with Timer() as inference_timer:
            # Classifier and face detector run in parallel (both release the GIL).
            classifier_future = self._executor.submit(self._classifier.detect, frame)
            face_future = self._executor.submit(self._face_detector.detect, frame)

            # Object detector runs on main thread (frame-skipped, cheap most frames).
            object_detections = self._object_detector.detect(frame)

            classifier_result = classifier_future.result()
            face_result = face_future.result()

            drowsiness_result = self._drowsiness_detector.update(face_result)

        classifier_prob = classifier_result.distracted_probability
        gaze_score = self._compute_gaze_distraction_score(face_result)

        if face_result.face_visible:
            # Face is the strong signal; down-weight the noisy ImageNet classifier
            combined_prob = max(gaze_score, classifier_prob * 0.3)
        else:
            combined_prob = classifier_prob

        if self._config.temporal_smoothing.enabled:
            smoothed_prob = self._smoother.update(combined_prob)
        else:
            smoothed_prob = combined_prob

        driver_state = self._hysteresis.update(smoothed_prob)

        if self._config.object_confirmation.enabled:
            confirmed_objects = self._object_confirmer.update(object_detections)
        else:
            confirmed_objects = {d.class_name: True for d in object_detections}

        assessment = self._reasoner.assess(
            driver_state=driver_state,
            smoothed_probability=smoothed_prob,
            confirmed_objects=confirmed_objects,
            face_result=face_result,
            drowsiness_result=drowsiness_result,
        )

        alert = self._alert_manager.update(assessment)

        return FrameRecord(
            frame_number=self._frame_number,
            timestamp=timestamp,
            raw_frame=frame,
            classifier_result=classifier_result,
            object_detections=object_detections,
            face_result=face_result,
            drowsiness_result=drowsiness_result,
            smoothed_probability=smoothed_prob,
            driver_state=driver_state,
            assessment=assessment,
            alert=alert,
            inference_time_ms=inference_timer.elapsed_ms,
            pipeline_time_ms=0.0,
        )

    def _compute_gaze_distraction_score(self, face_result: FaceResult) -> float:
        """Map head pose angles to a 0.0-1.0 distraction score.

        Uses config thresholds: yaw_threshold, pitch_down_threshold, pitch_up_threshold.
        Score ramps linearly from 0 at threshold to 1.0 at 2x threshold.

        Args:
            face_result: Face detection result.

        Returns:
            Gaze-based distraction score in [0.0, 1.0].
        """
        if not face_result.face_visible or face_result.head_pose is None:
            return 0.0

        pitch, yaw, roll = face_result.head_pose
        cfg = self._config.face_detector

        yaw_score = 0.0
        abs_yaw = abs(yaw)
        if abs_yaw > cfg.yaw_threshold:
            yaw_score = min(1.0, (abs_yaw - cfg.yaw_threshold) / cfg.yaw_threshold)

        pitch_score = 0.0
        if pitch > cfg.pitch_down_threshold:
            pitch_score = min(1.0, (pitch - cfg.pitch_down_threshold) / cfg.pitch_down_threshold)
        elif pitch < -cfg.pitch_up_threshold:
            pitch_score = min(1.0, (-pitch - cfg.pitch_up_threshold) / cfg.pitch_up_threshold)

        return max(yaw_score, pitch_score)

    def _output(self, record: FrameRecord) -> None:
        """Run all output stages for a frame record.

        Args:
            record: The completed FrameRecord for this frame.
        """
        if self._config.display.enabled:
            self._display.render(record, self._fps_counter.fps)

        phone_detected = (
            record.assessment is not None
            and "cell phone" in record.assessment.confirmed_objects
        )
        self._audio_alerter.update(record.driver_state, phone_detected)

        if self._config.telemetry.enabled:
            if record.frame_number % self._config.telemetry.log_interval_frames == 0:
                self._telemetry.log(record)

    def _shutdown(self) -> None:
        """Clean up all resources."""
        self._executor.shutdown(wait=True)
        self._frame_source.release()
        self._display.destroy()
        self._telemetry.close()
        logger.info("PipelineManager: Shutdown complete.")
