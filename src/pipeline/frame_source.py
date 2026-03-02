"""Unified frame source for webcam capture and video file replay.

Provides a single interface for reading frames regardless of whether the
source is a live webcam or a recorded video file. Video replay respects
original FPS timing for real-time playback.

Phase: 1-2 (active).
"""

import logging
import queue
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config_loader import FrameSourceConfig

logger = logging.getLogger(__name__)


class FrameSource:
    """Unified frame source for webcam and video file playback.

    For video replay: respects original FPS timing so playback is real-time.
    For webcam: no artificial delay.
    Handles end-of-video gracefully by returning (False, None).

    Args:
        config: FrameSourceConfig specifying source type and parameters.
    """

    def __init__(self, config: FrameSourceConfig) -> None:
        self._config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_video = config.type == "video"
        self._video_fps: float = config.target_fps
        self._last_frame_time: float = 0.0
        self._frame_interval: float = 0.0

        self._open()

        self._threaded = config.threaded and not self._is_video
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None

        if self._threaded and self._cap is not None and self._cap.isOpened():
            self._stop_event.clear()
            self._capture_thread = threading.Thread(
                target=self._capture_loop, name="FrameCapture", daemon=True
            )
            self._capture_thread.start()
            logger.info("FrameSource: Background capture thread started")

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                break
            ret, frame = self._cap.read()
            if not ret:
                break
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def _open(self) -> None:
        """Open the video capture source."""
        if self._is_video:
            video_path = self._config.video_path
            if not video_path:
                logger.error("FrameSource: video mode but no video_path specified")
                return

            self._cap = cv2.VideoCapture(video_path)
            if not self._cap.isOpened():
                logger.error(
                    "FrameSource: Failed to open video file '%s'", video_path
                )
                self._cap = None
                return

            self._video_fps = self._cap.get(cv2.CAP_PROP_FPS)
            if self._video_fps <= 0:
                self._video_fps = self._config.target_fps
            self._frame_interval = 1.0 / self._video_fps

            logger.info(
                "FrameSource: Opened video '%s' at %.1f FPS",
                video_path,
                self._video_fps,
            )
        else:
            self._cap = cv2.VideoCapture(self._config.webcam_index)
            if not self._cap.isOpened():
                logger.error(
                    "FrameSource: Failed to open webcam index %d",
                    self._config.webcam_index,
                )
                self._cap = None
                return

            w, h = self._config.resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

            logger.info(
                "FrameSource: Opened webcam %d at %dx%d",
                self._config.webcam_index,
                w,
                h,
            )

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame from the source.

        For video replay: sleeps to maintain original FPS timing.
        For webcam: returns immediately.

        Returns:
            Tuple of (success, frame). frame is None if read failed.
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None

        if self._threaded:
            try:
                frame = self._frame_queue.get(timeout=0.5)
                return True, frame
            except queue.Empty:
                return False, None

        if self._is_video and self._last_frame_time > 0:
            elapsed = time.time() - self._last_frame_time
            sleep_time = self._frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        ret, frame = self._cap.read()
        self._last_frame_time = time.time()

        if not ret:
            return False, None

        return True, frame

    def release(self) -> None:
        """Release the video capture resource."""
        if self._capture_thread is not None:
            self._stop_event.set()
            self._capture_thread.join(timeout=1.0)
            self._capture_thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("FrameSource: Released")

    @property
    def fps(self) -> float:
        """Actual FPS of the source (from video metadata or target FPS)."""
        if self._is_video:
            return self._video_fps
        return float(self._config.target_fps)

    @property
    def is_opened(self) -> bool:
        """Whether the capture source is currently open."""
        return self._cap is not None and self._cap.isOpened()
