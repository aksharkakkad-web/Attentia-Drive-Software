"""Debug display window for Attentia Drive.

Renders the camera frame with overlays including state, confidence,
bounding boxes, EMA plot, FPS counter, active triggers, and alert indicator.

Phase: 1-2 (active).
"""

import logging
import time
from collections import deque
from typing import Deque, List, Tuple

import cv2
import numpy as np

from src.config_loader import DisplayConfig
from src.pipeline.frame_record import FrameRecord

logger = logging.getLogger(__name__)

def _draw_contour(
    frame: np.ndarray, landmarks: np.ndarray,
    indices: List[int], w: int, h: int, color: Tuple[int, int, int],
) -> None:
    """Draw a polyline contour from landmark indices."""
    pts = []
    for idx in indices:
        if idx < len(landmarks):
            pts.append((int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)))
    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i + 1], color, 1)


STATE_COLORS = {
    "ATTENTIVE": (0, 200, 0),    # Green (BGR)
    "MONITORING": (0, 200, 255),  # Yellow (BGR)
    "DISTRACTED": (0, 0, 255),    # Red (BGR)
}


class Display:
    """OpenCV debug display window.

    Renders the camera frame with:
    - Object detection bounding boxes (green=confirmed, yellow=detected)
    - Current DriverState as large text
    - Smoothed probability as a number and horizontal bar
    - Mini rolling EMA chart (last 150 frames)
    - FPS counter
    - Active triggers list
    - Alert indicator (flashes "ALERT" in red for 1 second)

    Args:
        config: DisplayConfig with display preferences.
    """

    def __init__(self, config: DisplayConfig) -> None:
        self._config = config
        self._ema_history: Deque[float] = deque(maxlen=150)
        self._alert_flash_time: float = 0.0

    def render(self, record: FrameRecord, fps: float = 0.0) -> None:
        """Render the debug overlay onto the frame and display it.

        Args:
            record: FrameRecord with all pipeline results.
            fps: Current FPS measurement.
        """
        if not self._config.enabled or record.raw_frame is None:
            return

        frame = record.raw_frame.copy()

        self._ema_history.append(record.smoothed_probability)

        if self._config.show_bboxes and record.assessment is not None:
            self._draw_bboxes(frame, record)

        if self._config.show_state and record.assessment is not None:
            self._draw_state(frame, record)

        self._draw_probability_bar(frame, record.smoothed_probability)

        if record.assessment is not None:
            self._draw_triggers(frame, record.assessment.active_triggers)

        if self._config.show_ema_plot:
            self._draw_ema_plot(frame)

        if self._config.show_fps:
            self._draw_fps(frame, fps, record.pipeline_time_ms)

        if record.face_result is not None and record.face_result.face_visible:
            self._draw_face_info(frame, record)

        if record.drowsiness_result is not None:
            self._draw_drowsiness_info(frame, record)

        if record.alert is not None:
            self._alert_flash_time = time.time()

        if time.time() - self._alert_flash_time < 1.0:
            self._draw_alert_flash(frame)

        cv2.imshow(self._config.window_name, frame)

    def wait_key(self, delay: int = 1) -> int:
        """Wait for a key press.

        Args:
            delay: Wait time in milliseconds.

        Returns:
            Key code, or -1 if no key pressed.
        """
        return cv2.waitKey(delay) & 0xFF

    def destroy(self) -> None:
        """Destroy the display window."""
        if self._config.enabled:
            cv2.destroyAllWindows()

    def _draw_bboxes(self, frame: np.ndarray, record: FrameRecord) -> None:
        """Draw object detection bounding boxes on the frame."""
        confirmed_set = set(record.assessment.confirmed_objects) if record.assessment else set()

        for det in record.object_detections:
            is_confirmed = det.class_name in confirmed_set
            color = (0, 200, 0) if is_confirmed else (0, 200, 255)
            x1, y1, x2, y2 = det.bbox

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{det.class_name} {det.confidence:.2f}"
            if is_confirmed:
                label += " [OK]"

            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - 6),
                (x1 + label_size[0], y1),
                color,
                -1,
            )
            cv2.putText(
                frame, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
            )

    def _draw_state(self, frame: np.ndarray, record: FrameRecord) -> None:
        """Draw the current driver state as large text."""
        state = record.driver_state
        if state is None:
            return
        state_str = state if isinstance(state, str) else getattr(state, "name", str(state))
        color = STATE_COLORS.get(state_str, (255, 255, 255))

        cv2.putText(
            frame, state_str, (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3,
        )

    def _draw_probability_bar(self, frame: np.ndarray, prob: float) -> None:
        """Draw smoothed probability as a number and horizontal bar."""
        h, w = frame.shape[:2]

        bar_x = 10
        bar_y = 55
        bar_w = 200
        bar_h = 20

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)

        fill_w = int(bar_w * min(1.0, prob))
        if prob < 0.4:
            bar_color = (0, 200, 0)
        elif prob < 0.75:
            bar_color = (0, 200, 255)
        else:
            bar_color = (0, 0, 255)

        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)

        cv2.putText(
            frame, f"P(dist): {prob:.3f}", (bar_x + bar_w + 10, bar_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    def _draw_triggers(self, frame: np.ndarray, triggers: List[str]) -> None:
        """Draw the list of active triggers."""
        y = 100
        for trigger in triggers:
            cv2.putText(
                frame, f"- {trigger}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1,
            )
            y += 20

    def _draw_ema_plot(self, frame: np.ndarray) -> None:
        """Draw a mini rolling EMA chart in the bottom-right corner."""
        h, w = frame.shape[:2]
        plot_w = 200
        plot_h = 80
        plot_x = w - plot_w - 10
        plot_y = h - plot_h - 10

        overlay = frame[plot_y:plot_y + plot_h, plot_x:plot_x + plot_w].copy()
        cv2.rectangle(overlay, (0, 0), (plot_w, plot_h), (40, 40, 40), -1)
        cv2.addWeighted(
            overlay, 0.7,
            frame[plot_y:plot_y + plot_h, plot_x:plot_x + plot_w], 0.3,
            0,
            frame[plot_y:plot_y + plot_h, plot_x:plot_x + plot_w],
        )

        threshold_y = int(plot_h * (1.0 - 0.75))
        cv2.line(
            frame,
            (plot_x, plot_y + threshold_y),
            (plot_x + plot_w, plot_y + threshold_y),
            (0, 0, 200), 1,
        )

        if len(self._ema_history) >= 2:
            points = []
            data = list(self._ema_history)
            for i, val in enumerate(data):
                px = plot_x + int(i * plot_w / max(len(data) - 1, 1))
                py = plot_y + int(plot_h * (1.0 - min(1.0, val)))
                points.append((px, py))

            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], (0, 255, 0), 1)

        cv2.putText(
            frame, "EMA", (plot_x + 5, plot_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1,
        )

    def _draw_fps(self, frame: np.ndarray, fps: float, pipeline_ms: float) -> None:
        """Draw FPS counter and pipeline latency."""
        h, w = frame.shape[:2]
        text = f"FPS: {fps:.1f} | {pipeline_ms:.1f}ms"
        cv2.putText(
            frame, text, (w - 220, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,
        )

    def _draw_face_info(self, frame: np.ndarray, record: FrameRecord) -> None:
        """Draw face detection info: head pose, EAR, landmarks."""
        h, w = frame.shape[:2]
        face = record.face_result

        # Head pose text (right side)
        if face.head_pose is not None:
            pitch, yaw, roll = face.head_pose
            pose_text = f"Yaw:{yaw:.0f} Pitch:{pitch:.0f} Roll:{roll:.0f}"
            cv2.putText(
                frame, pose_text, (w - 280, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
            )

        # EAR values
        if face.ear_left is not None:
            ear_text = f"EAR L:{face.ear_left:.2f} R:{face.ear_right:.2f}"
            cv2.putText(
                frame, ear_text, (w - 280, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
            )

        # Face landmark contours (eyes, eyebrows, lips, face outline)
        if face.landmarks is not None:
            # Left eye contour
            _draw_contour(frame, face.landmarks, [
                33, 7, 163, 144, 145, 153, 154, 155, 133,
                173, 157, 158, 159, 160, 161, 246, 33,
            ], w, h, (0, 255, 0))
            # Right eye contour
            _draw_contour(frame, face.landmarks, [
                362, 382, 381, 380, 374, 373, 390, 249, 263,
                466, 388, 387, 386, 385, 384, 398, 362,
            ], w, h, (0, 255, 0))
            # Lips outer
            _draw_contour(frame, face.landmarks, [
                61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61,
            ], w, h, (0, 200, 200))
            # Nose bridge
            for idx in [1, 4, 5, 6]:
                if idx < len(face.landmarks):
                    px = int(face.landmarks[idx][0] * w)
                    py = int(face.landmarks[idx][1] * h)
                    cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)
            # Iris centers (if available)
            for idx in [468, 473]:
                if idx < len(face.landmarks):
                    px = int(face.landmarks[idx][0] * w)
                    py = int(face.landmarks[idx][1] * h)
                    cv2.circle(frame, (px, py), 3, (255, 255, 0), -1)

    def _draw_drowsiness_info(self, frame: np.ndarray, record: FrameRecord) -> None:
        """Draw drowsiness info: PERCLOS bar, blink rate, yawn/drowsy indicators."""
        h, w = frame.shape[:2]
        dr = record.drowsiness_result

        # PERCLOS progress bar (top-right area)
        bar_x = w - 180
        bar_y = 85
        bar_w = 120
        bar_h = 12

        cv2.putText(
            frame, "PERCLOS", (bar_x - 2, bar_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1,
        )
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        fill = int(bar_w * min(1.0, dr.perclos / 0.3))  # scale: 0.3 = full bar
        if fill > 0:
            color = (0, 200, 0) if dr.perclos < 0.15 else (0, 0, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)

        perclos_text = f"{dr.perclos:.1%}"
        cv2.putText(
            frame, perclos_text, (bar_x + bar_w + 5, bar_y + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1,
        )

        # Blink rate
        blink_text = f"Blinks: {dr.blink_count} ({dr.blink_rate:.0f}/min)"
        cv2.putText(
            frame, blink_text, (w - 280, 115),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1,
        )

        # Drowsy flash indicator
        if dr.is_drowsy:
            cv2.putText(
                frame, "DROWSY!", (w // 2 - 80, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
            )

        # Yawn indicator
        if dr.is_yawning:
            cv2.putText(
                frame, "YAWN", (w // 2 - 50, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2,
            )

    def _draw_alert_flash(self, frame: np.ndarray) -> None:
        """Flash an ALERT indicator on screen."""
        h, w = frame.shape[:2]
        cv2.putText(
            frame, "ALERT!", (w // 2 - 80, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4,
        )
