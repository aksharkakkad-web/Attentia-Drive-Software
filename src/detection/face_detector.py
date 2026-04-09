"""Face detector module using MediaPipe FaceLandmarker (tasks API).

Extracts 478 facial landmarks, computes head pose (pitch/yaw/roll),
Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and gaze vector.
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config_loader import FaceDetectorConfig
from src.detection.base_detector import BaseDetector

logger = logging.getLogger(__name__)

# Landmark indices for EAR computation (MediaPipe Face Mesh)
_LEFT_EYE = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Landmark indices for MAR computation (inner lip)
_LEFT_LIP = [78]
_RIGHT_LIP = [308]
# Vertical lip landmarks for better MAR
_LIP_TOP_INNER = [13, 312, 82]
_LIP_BOTTOM_INNER = [14, 317, 87]

# Iris landmark indices
_LEFT_IRIS_CENTER = 468
_RIGHT_IRIS_CENTER = 473

# Default model path
_DEFAULT_MODEL_PATH = "models/face_landmarker.task"


@dataclass
class FaceResult:
    """Result from the face detector."""

    face_visible: bool = False
    landmarks: Optional[np.ndarray] = None  # shape (478, 3) normalized coords
    head_pose: Optional[Tuple[float, float, float]] = None  # (pitch, yaw, roll) degrees
    gaze_vector: Optional[Tuple[float, float, float]] = None  # MVP: (x_norm, y_norm, 0.0) iris offset normalized by eye width. Signed; positive x = iris right of eye center.
    ear_left: Optional[float] = None
    ear_right: Optional[float] = None
    mouth_aspect_ratio: Optional[float] = None


def compute_ear(landmarks: np.ndarray, eye_indices: list) -> float:
    """Compute Eye Aspect Ratio from 6 landmark points.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Args:
        landmarks: Array of landmark coordinates, shape (N, 3).
        eye_indices: 6 indices [p1, p2, p3, p4, p5, p6].

    Returns:
        Eye Aspect Ratio value.
    """
    pts = landmarks[eye_indices]
    vertical_1 = np.linalg.norm(pts[1] - pts[5])
    vertical_2 = np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])
    if horizontal < 1e-6:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def compute_mar(landmarks: np.ndarray) -> float:
    """Compute Mouth Aspect Ratio from lip landmarks.

    MAR = avg(vertical distances) / horizontal distance

    Args:
        landmarks: Array of landmark coordinates, shape (N, 3).

    Returns:
        Mouth Aspect Ratio value.
    """
    top_pts = landmarks[_LIP_TOP_INNER]
    bottom_pts = landmarks[_LIP_BOTTOM_INNER]
    left = landmarks[_LEFT_LIP[0]]
    right = landmarks[_RIGHT_LIP[0]]

    vertical_sum = 0.0
    for t, b in zip(top_pts, bottom_pts):
        vertical_sum += np.linalg.norm(t - b)
    avg_vertical = vertical_sum / len(top_pts)

    horizontal = np.linalg.norm(left - right)
    if horizontal < 1e-6:
        return 0.0
    return avg_vertical / horizontal


def _rotation_matrix_to_euler(rmat: np.ndarray) -> Tuple[float, float, float]:
    """Extract pitch, yaw, roll (degrees) from a 3x3 rotation matrix.

    Uses ZYX (yaw-pitch-roll) convention.
    """
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = 0.0
    return (math.degrees(pitch), math.degrees(yaw), math.degrees(roll))


class FaceDetector(BaseDetector):
    """Face detector using MediaPipe FaceLandmarker (tasks API).

    Extracts facial landmarks, head pose, EAR, MAR, and gaze vector.
    Returns FaceResult(face_visible=False) when no face detected or disabled.
    """

    def __init__(self, config: FaceDetectorConfig) -> None:
        super().__init__()
        self._config = config
        self._landmarker = None
        self._frame_timestamp_ms = 0

        if config.enabled:
            model_path = Path(_DEFAULT_MODEL_PATH)
            if not model_path.exists():
                logger.warning(
                    "FaceDetector: Model file not found at '%s'. "
                    "Download from: https://storage.googleapis.com/mediapipe-models/"
                    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
                    model_path.resolve(),
                )
                return

            try:
                from mediapipe.tasks.python import BaseOptions
                from mediapipe.tasks.python.vision.face_landmarker import (
                    FaceLandmarker,
                    FaceLandmarkerOptions,
                )

                options = FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(model_path)),
                    running_mode=FaceLandmarkerOptions.running_mode.__class__("VIDEO"),
                    num_faces=1,
                    min_face_detection_confidence=config.min_detection_confidence,
                    min_tracking_confidence=config.min_tracking_confidence,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=True,
                )
                self._landmarker = FaceLandmarker.create_from_options(options)
                self._model_loaded = True
                logger.info("FaceDetector: MediaPipe FaceLandmarker initialized")
            except ImportError:
                logger.warning(
                    "FaceDetector: mediapipe not installed. "
                    "Install with: pip install mediapipe"
                )
            except Exception as e:
                logger.warning("FaceDetector: Failed to initialize MediaPipe: %s", e)

    def load_model(self, path: str) -> None:
        """No-op for MediaPipe (model loaded in __init__)."""
        pass

    def detect(self, frame: np.ndarray) -> FaceResult:
        """Detect face landmarks and compute head pose, EAR, MAR, gaze.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            FaceResult with all computed fields.
        """
        if not self._config.enabled or self._landmarker is None:
            return FaceResult()

        h, w = frame.shape[:2]

        try:
            from mediapipe import Image, ImageFormat

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

            self._frame_timestamp_ms += 33  # ~30fps increment
            result = self._landmarker.detect_for_video(
                mp_image, self._frame_timestamp_ms
            )
        except Exception as e:
            logger.debug("FaceDetector: Detection failed: %s", e)
            return FaceResult()

        if not result.face_landmarks:
            return FaceResult()

        face_lms = result.face_landmarks[0]

        # Convert to numpy arrays
        landmarks_norm = np.array(
            [(lm.x, lm.y, lm.z) for lm in face_lms],
            dtype=np.float64,
        )
        landmarks_px = np.array(
            [(lm.x * w, lm.y * h, lm.z * w) for lm in face_lms],
            dtype=np.float64,
        )

        # Head pose from MediaPipe's transformation matrix
        head_pose = (0.0, 0.0, 0.0)
        if result.facial_transformation_matrixes:
            mat = result.facial_transformation_matrixes[0]  # 4x4
            rmat = mat[:3, :3]
            head_pose = _rotation_matrix_to_euler(rmat)

        # EAR
        ear_left = compute_ear(landmarks_px, _LEFT_EYE)
        ear_right = compute_ear(landmarks_px, _RIGHT_EYE)

        # MAR
        mar = compute_mar(landmarks_px)

        # Gaze vector from iris landmarks
        gaze = self._compute_gaze(landmarks_px)

        return FaceResult(
            face_visible=True,
            landmarks=landmarks_norm,
            head_pose=head_pose,
            gaze_vector=gaze,
            ear_left=ear_left,
            ear_right=ear_right,
            mouth_aspect_ratio=mar,
        )

    def _compute_gaze(
        self, landmarks_px: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        """Compute iris offset from eye center, normalized by eye width.

        MVP: Returns signed horizontal/vertical iris offset as a fraction of
        eye width. A value of (+0.5, 0.0, 0.0) means the iris is shifted one
        half-eye-width to the screen's right (driver's left). This preserves
        magnitude (unlike a unit vector) so downstream can estimate how far
        the eye has rotated.

        Returns:
            (x_norm, y_norm, 0.0) or None if iris landmarks unavailable.
        """
        if landmarks_px.shape[0] <= _RIGHT_IRIS_CENTER:
            return None

        left_iris = landmarks_px[_LEFT_IRIS_CENTER]
        right_iris = landmarks_px[_RIGHT_IRIS_CENTER]

        left_eye_pts = landmarks_px[_LEFT_EYE]
        right_eye_pts = landmarks_px[_RIGHT_EYE]
        left_eye_center = np.mean(left_eye_pts, axis=0)
        right_eye_center = np.mean(right_eye_pts, axis=0)

        # Eye width = distance between outer/inner corners (indices 0 and 3 of EAR set)
        left_eye_width = float(np.linalg.norm(left_eye_pts[0] - left_eye_pts[3]))
        right_eye_width = float(np.linalg.norm(right_eye_pts[0] - right_eye_pts[3]))
        eye_width = (left_eye_width + right_eye_width) / 2.0
        if eye_width < 1e-6:
            return None

        left_offset = left_iris - left_eye_center
        right_offset = right_iris - right_eye_center
        avg_offset = (left_offset + right_offset) / 2.0

        x_norm = float(avg_offset[0] / eye_width)
        y_norm = float(avg_offset[1] / eye_width)
        return (x_norm, y_norm, 0.0)
