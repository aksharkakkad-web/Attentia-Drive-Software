"""Pipeline adapters — Layer 1 → Layer 2 bridge.

Converts MediaPipe FaceResult and TFLite ObjectDetection lists into the
typed PerceptionBundle contract that SignalProcessor expects.

This is the only place in the codebase that knows about both the detection
layer's output types (FaceResult, ObjectDetection) and the pipeline contracts.
All downstream layers only see PerceptionBundle.

PRD §4.2 — MVP adapter
"""

from typing import List, Optional

import numpy as np

from src.config_prd import PHONE_CONFIDENCE_THRESHOLD
from src.contracts import (
    FaceDetection,
    GazeOutput,
    LandmarkOutput,
    PerceptionBundle,
    PhoneDetectionOutput,
    RawFrame,
)


def convert_to_perception_bundle(
    face_result,
    object_detections: List,
    frame_id: int,
    timestamp_ns: int,
) -> PerceptionBundle:
    """Convert MediaPipe + TFLite outputs to a PerceptionBundle.

    Args:
        face_result: FaceResult from FaceDetector.detect(). None → all defaults.
        object_detections: List[ObjectDetection] from ObjectDetector.detect().
        frame_id: Monotonically increasing frame counter.
        timestamp_ns: Frame timestamp in nanoseconds.

    Returns:
        PerceptionBundle ready for SignalProcessor.process().
    """
    if face_result is None:
        return PerceptionBundle(frame_id=frame_id, timestamp_ns=timestamp_ns)

    visible: bool = face_result.face_visible

    # ── FaceDetection ──────────────────────────────────────────────────────────
    face = FaceDetection(
        present=visible,
        confidence=0.95 if visible else 0.0,
    )

    # ── LandmarkOutput ─────────────────────────────────────────────────────────
    landmarks: Optional[LandmarkOutput] = None
    if visible and face_result.landmarks is not None:
        landmarks = LandmarkOutput(
            landmarks=face_result.landmarks,  # (478, 3) normalized — stored as-is
            confidence=0.9,
            pose_valid=True,
        )

    # ── GazeOutput (MVP: head-pose proxy, not eye gaze) ───────────────────────
    gaze: Optional[GazeOutput] = None
    if visible:
        gaze = GazeOutput(valid=True)  # combined_yaw/pitch stay 0.0 — MVP proxy

    # ── MVP-ONLY: head_pose_raw and EAR ───────────────────────────────────────
    head_pose_raw = None
    if visible and face_result.head_pose is not None:
        head_pose_raw = face_result.head_pose  # (pitch, yaw, roll) tuple

    ear_left: float = float(face_result.ear_left) if face_result.ear_left is not None else 0.0
    ear_right: float = float(face_result.ear_right) if face_result.ear_right is not None else 0.0

    # ── Phone detection ────────────────────────────────────────────────────────
    phone = _extract_phone(object_detections or [])

    return PerceptionBundle(
        timestamp_ns=timestamp_ns,
        frame_id=frame_id,
        face=face,
        landmarks=landmarks,
        gaze=gaze,
        phone=phone,
        head_pose_raw=head_pose_raw,
        ear_left=ear_left,
        ear_right=ear_right,
    )


def wrap_raw_frame(
    frame: np.ndarray,
    frame_id: int,
    timestamp_ns: int,
) -> RawFrame:
    """Wrap a raw BGR numpy array in a RawFrame contract.

    Args:
        frame: BGR image, shape (H, W, 3), dtype uint8.
        frame_id: Monotonically increasing frame counter.
        timestamp_ns: Frame timestamp in nanoseconds.

    Returns:
        RawFrame ready for the pipeline.
    """
    h, w = frame.shape[:2]
    return RawFrame(
        timestamp_ns=timestamp_ns,
        frame_id=frame_id,
        width=w,
        height=h,
        channels=3,
        data=frame,
        source_type='webcam',
    )


def _extract_phone(object_detections: List) -> PhoneDetectionOutput:
    """Find the highest-confidence 'cell phone' detection."""
    phones = [d for d in object_detections if d.class_name == 'cell phone']
    if not phones:
        return PhoneDetectionOutput()
    best = max(phones, key=lambda d: d.confidence)
    return PhoneDetectionOutput(
        detected=best.confidence >= PHONE_CONFIDENCE_THRESHOLD,
        max_confidence=best.confidence,
        bbox_norm=best.bbox,  # MVP: pixel bbox stored in bbox_norm field
    )
