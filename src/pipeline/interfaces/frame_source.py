"""FrameSource interface — camera/video input abstraction.

Day 1 scope: move the existing cv2.VideoCapture + video-replay code behind
a proper interface so downstream pipeline code consumes frames via a
method call, not a library. Pi 5 Picamera2FrameSource is deferred to
bring-up (see docs/DEVIATIONS.md).

Contract: docs/INTERFACES.md §FrameSource.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config_loader import FrameSourceConfig

logger = logging.getLogger(__name__)


class ColorSpace(Enum):
    """Color space of the frame returned by a FrameSource.

    Consumers must check this field rather than assuming BGR or RGB.
    OpenCVFrameSource returns BGR; a future Picamera2FrameSource returns RGB.
    """

    BGR = 'bgr'
    RGB = 'rgb'


@dataclass
class FrameResult:
    """Result of a FrameSource.read() call.

    ok=False means the source is exhausted (video ended) or a transient
    read failure occurred. Callers must handle ok=False without raising.
    """

    ok: bool
    frame: Optional[np.ndarray] = None
    timestamp_ms: float = 0.0
    color_space: ColorSpace = ColorSpace.BGR


class FrameSource:
    """Abstract camera/video source.

    Implementations must never raise into the pipeline loop; return
    FrameResult(ok=False) on any failure mode.
    """

    def open(self) -> bool:
        raise NotImplementedError

    def read(self) -> FrameResult:
        raise NotImplementedError

    # Legacy tuple form for callers that haven't been migrated yet.
    def read_legacy(self) -> Tuple[bool, Optional[np.ndarray]]:
        r = self.read()
        return r.ok, r.frame

    def close(self) -> None:
        raise NotImplementedError

    def is_available(self) -> bool:
        raise NotImplementedError

    @property
    def fps(self) -> float:
        raise NotImplementedError


# ─── Mac Day-1 implementation ────────────────────────────────────────────────


class OpenCVFrameSource(FrameSource):
    """Unified OpenCV frame source for webcam capture and video file replay.

    For video replay: respects original FPS timing so playback is real-time.
    For webcam: drains one queued frame before reading so the returned frame
    is always the freshest available (prevents stale-buffer lag when a slow
    cycle leaves a frame in OpenCV's internal buffer).

    This is the ONLY place in the codebase that may call cv2.VideoCapture.
    """

    def __init__(self, config: FrameSourceConfig) -> None:
        self._config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_video = config.type == "video"
        self._video_fps: float = float(config.target_fps)
        self._last_frame_time: float = 0.0
        self._frame_interval: float = 0.0

    def open(self) -> bool:
        if self._is_video:
            video_path = self._config.video_path
            if not video_path:
                logger.error("OpenCVFrameSource: video mode but no video_path specified")
                return False

            self._cap = cv2.VideoCapture(video_path)
            if not self._cap.isOpened():
                logger.error(
                    "OpenCVFrameSource: Failed to open video file '%s'", video_path
                )
                self._cap = None
                return False

            self._video_fps = self._cap.get(cv2.CAP_PROP_FPS)
            if self._video_fps <= 0:
                self._video_fps = float(self._config.target_fps)
            self._frame_interval = 1.0 / self._video_fps

            logger.info(
                "OpenCVFrameSource: Opened video '%s' at %.1f FPS",
                video_path,
                self._video_fps,
            )
            return True

        self._cap = cv2.VideoCapture(self._config.webcam_index)
        if not self._cap.isOpened():
            logger.error(
                "OpenCVFrameSource: Failed to open webcam index %d",
                self._config.webcam_index,
            )
            self._cap = None
            return False

        w, h = self._config.resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        logger.info(
            "OpenCVFrameSource: Opened webcam %d at %dx%d",
            self._config.webcam_index,
            w,
            h,
        )
        return True

    def read(self) -> FrameResult:
        if self._cap is None or not self._cap.isOpened():
            return FrameResult(ok=False)

        if self._is_video and self._last_frame_time > 0:
            elapsed = time.time() - self._last_frame_time
            sleep_time = self._frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        elif not self._is_video:
            # Webcam: discard one buffered frame so we always process the
            # freshest frame rather than a stale one from a slow previous cycle.
            self._cap.grab()

        ret, frame = self._cap.read()
        self._last_frame_time = time.time()

        if not ret:
            return FrameResult(ok=False)

        return FrameResult(ok=True, frame=frame, timestamp_ms=self._last_frame_time * 1000.0)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("OpenCVFrameSource: Released")

    def is_available(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        if self._is_video:
            return self._video_fps
        return float(self._config.target_fps)


class NullFrameSource(FrameSource):
    """No-op source for tests and headless runs. Always returns ok=False."""

    def __init__(self, target_fps: float = 30.0) -> None:
        self._opened = False
        self._fps = target_fps

    def open(self) -> bool:
        self._opened = True
        return True

    def read(self) -> FrameResult:
        return FrameResult(ok=False)

    def close(self) -> None:
        self._opened = False

    def is_available(self) -> bool:
        return self._opened

    @property
    def fps(self) -> float:
        return self._fps
