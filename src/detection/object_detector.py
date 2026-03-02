"""Object detector module for detecting distraction-related objects.

Detects objects like phones, cups, bottles, books, and laptops in the
driver-facing camera frame using a TFLite SSD model. Returns a list of
detected objects with class name, confidence, and bounding box.

If the model file is missing, returns an empty list (neutral).
Respects frame_skip to reduce compute: caches results on skipped frames.

Label mapping priority:
1. Embedded label map from TFLite model metadata (most reliable).
2. Fallback unified map covering all three common COCO labeling schemes.

Phase: 1-2 (active).
"""

import logging
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# EfficientDet-Lite anchor generation parameters (standard values).
_EFF_STRIDES = [8, 16, 32, 64, 128]
_EFF_ANCHOR_SCALE = 4.0
_EFF_NUM_SCALES = 3
_EFF_ASPECT_RATIOS = [1.0, 2.0, 0.5]   # width/height
_EFF_BOX_SCALE_FACTORS = [10.0, 10.0, 5.0, 5.0]
_EFF_NMS_IOU = 0.5

from src.config_loader import ObjectDetectorConfig
from src.detection.base_detector import BaseDetector

logger = logging.getLogger(__name__)

# Target class names we care about (lowercased for matching).
TARGET_NAMES = {"cell phone", "cup", "bottle", "book", "laptop"}

# Fallback unified target-class label map covering ALL three common COCO schemes.
# Used only when the model does not contain an embedded label map.
#
# COCO 91-class (1-indexed, gaps):  Used by TF1 Object Detection API.
# COCO 90-class (0-indexed, gaps):  Used by EfficientDet-Lite / TF2 Task Library.
# COCO 80-class (0-indexed, contiguous):  Used by YOLO-style frameworks.
FALLBACK_TARGET_LABELS: Dict[int, str] = {
    # COCO 80-class (0-indexed contiguous)
    39: "bottle",
    41: "cup",
    63: "laptop",
    67: "cell phone",
    73: "laptop",  # 73 = "book" in 80-class, "laptop" in 91-class. Both are targets.
    # COCO 91-class (1-indexed with gaps)
    44: "bottle",
    47: "cup",
    77: "cell phone",
    84: "book",
    # COCO 90-class (0-indexed with gaps, EfficientDet-Lite)
    43: "bottle",
    46: "cup",
    72: "laptop",
    76: "cell phone",
    83: "book",
}


@dataclass
class ObjectDetection:
    """A single detected object."""

    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)


def _load_embedded_labelmap(model_path: str) -> Optional[Dict[int, str]]:
    """Try to read a label map embedded in the TFLite model metadata.

    TFLite models with metadata are ZIP archives containing a flatbuffer
    and associated files (like labelmap.txt). If present, the label map
    gives us the exact class-ID-to-name mapping for this specific model.

    Args:
        model_path: Path to the .tflite file.

    Returns:
        Dict mapping class_id -> class_name, or None if no metadata found.
    """
    try:
        with zipfile.ZipFile(model_path, "r") as z:
            for name in z.namelist():
                if "label" in name.lower() and name.endswith(".txt"):
                    content = z.read(name).decode("utf-8")
                    lines = content.strip().split("\n")
                    label_map: Dict[int, str] = {}
                    for i, line in enumerate(lines):
                        line = line.strip()
                        if line and line != "???":
                            label_map[i] = line
                    return label_map
    except (zipfile.BadZipFile, Exception):
        pass
    return None


def _build_target_filter(
    label_map: Optional[Dict[int, str]],
    target_classes: List[str],
) -> Dict[int, str]:
    """Build a class_id -> target_name filter from a label map.

    If we have an embedded label map, scan it for target class names.
    Otherwise fall back to the hardcoded FALLBACK_TARGET_LABELS.

    Args:
        label_map: Full label map from model metadata, or None.
        target_classes: List of target class name strings from config.

    Returns:
        Dict mapping class_id -> class_name for target classes only.
    """
    target_set = {t.lower() for t in target_classes}

    if label_map is not None:
        result: Dict[int, str] = {}
        for class_id, name in label_map.items():
            normalized = name.strip().lower()
            if normalized in target_set:
                result[class_id] = normalized
        if result:
            return result
        logger.warning(
            "ObjectDetector: Embedded label map found but no target classes matched. "
            "Target classes: %s. Falling back to hardcoded map.",
            target_classes,
        )

    return {
        cid: name
        for cid, name in FALLBACK_TARGET_LABELS.items()
        if name in target_set
    }


class ObjectDetector(BaseDetector):
    """TFLite SSD object detector for distraction-related objects.

    Filters detections by target_classes and confidence_threshold from config.
    Implements frame_skip: only runs inference every N frames and caches results.

    On model load, attempts to read the embedded label map from the TFLite
    metadata. If found, uses the exact class IDs from the model. Otherwise
    falls back to a hardcoded map covering three common COCO label schemes.

    Auto-detects the model's input size from the input tensor shape, so it
    works with any model resolution (300x300, 320x320, etc.).
    """

    def __init__(self, config: ObjectDetectorConfig) -> None:
        super().__init__()
        self._config = config
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._frame_count: int = 0
        self._cached_detections: List[ObjectDetection] = []
        self._target_filter: Dict[int, str] = {}
        self._model_input_hw: Tuple[int, int] = tuple(config.input_size)

        # Output tensor indices, resolved by shape at load time
        self._boxes_idx: int = 0
        self._classes_idx: int = 1
        self._scores_idx: int = 2
        self._num_det_idx: int = 3

        # Raw EfficientDet format (2 outputs: class logits + box encodings)
        self._raw_efficientdet: bool = False
        self._anchors: Optional[np.ndarray] = None  # [N, 4] (cy, cx, ah, aw) normalised
        self._eff_class_logits_idx: int = 0
        self._eff_boxes_idx: int = 1

        if config.enabled:
            self.load_model(config.model_path)

    def load_model(self, path: str) -> None:
        """Load a TFLite SSD object detection model.

        Tries tflite_runtime first, falls back to tensorflow.lite on Windows.
        Reads embedded label map if available. Auto-detects input size and
        resolves output tensor indices by shape.

        Args:
            path: Path to the .tflite model file.
        """
        if not self._check_model_file(path):
            return

        try:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                from tensorflow.lite.python.interpreter import Interpreter

            self._interpreter = Interpreter(model_path=path, num_threads=self._config.num_threads)
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()

            # Auto-detect input size from model
            input_shape = self._input_details[0]["shape"]
            self._model_input_hw = (int(input_shape[1]), int(input_shape[2]))

            supported = self._resolve_output_indices()
            if not supported:
                self._model_loaded = False
                return

            # Try to load embedded label map
            embedded_map = _load_embedded_labelmap(path)
            self._target_filter = _build_target_filter(
                embedded_map, self._config.target_classes
            )

            self._model_loaded = True

            source = "embedded metadata" if embedded_map else "fallback hardcoded"
            inp = self._input_details[0]
            logger.info(
                "ObjectDetector: Model loaded from '%s' | "
                "input=%s %s | %d outputs | label source: %s | "
                "target IDs: %s",
                path,
                self._model_input_hw,
                inp["dtype"],
                len(self._output_details),
                source,
                dict(self._target_filter),
            )
        except Exception:
            logger.exception(
                "ObjectDetector: Failed to load model from '%s'", path
            )
            self._model_loaded = False

    def _resolve_output_indices(self) -> bool:
        """Resolve output tensor indices by shape instead of assuming fixed order.

        Handles two formats:

        1. Post-processed SSD (4 outputs):
           boxes [1,N,4], classes [1,N], scores [1,N], num_detections [1]

        2. Raw EfficientDet (2 rank-3 outputs):
           class_logits [1,N,C], raw_boxes [1,N,4]
           Requires anchor generation + decoding + NMS.

        Returns:
            True if a supported layout was detected, False otherwise.
        """
        n = len(self._output_details)
        shapes = [tuple(d["shape"]) for d in self._output_details]

        # Detect raw EfficientDet format: exactly 2 rank-3 outputs
        rank3_indices = [i for i, s in enumerate(shapes) if len(s) == 3]
        if len(rank3_indices) == 2:
            i0, i1 = rank3_indices
            s0, s1 = shapes[i0], shapes[i1]
            # One output has last_dim==4 (boxes), the other has last_dim>4 (class logits)
            if s0[-1] == 4 and s1[-1] > 4:
                self._eff_boxes_idx = i0
                self._eff_class_logits_idx = i1
            elif s1[-1] == 4 and s0[-1] > 4:
                self._eff_boxes_idx = i1
                self._eff_class_logits_idx = i0
            else:
                logger.warning("ObjectDetector: Unrecognised 2-output layout %s", shapes)
                return False

            input_h, input_w = self._model_input_hw
            self._anchors = self._generate_efficientdet_anchors(input_h, input_w)
            self._raw_efficientdet = True
            num_classes = shapes[self._eff_class_logits_idx][-1]
            logger.info(
                "ObjectDetector: Raw EfficientDet format detected "
                "(%d anchors, %d classes). Using anchor decoding + NMS.",
                len(self._anchors), num_classes,
            )
            return True

        # Standard post-processed SSD: resolve by shape
        for i, s in enumerate(shapes):
            if len(s) == 3 and s[-1] == 4:
                self._boxes_idx = i
            elif len(s) == 1:
                self._num_det_idx = i

        rank2_indices = [i for i, s in enumerate(shapes) if len(s) == 2]
        if len(rank2_indices) == 2:
            name0 = self._output_details[rank2_indices[0]].get("name", "")
            name1 = self._output_details[rank2_indices[1]].get("name", "")
            if "class" in name1.lower() and "class" not in name0.lower():
                self._scores_idx = rank2_indices[0]
                self._classes_idx = rank2_indices[1]
            else:
                self._classes_idx = rank2_indices[0]
                self._scores_idx = rank2_indices[1]

        max_idx = max(self._boxes_idx, self._classes_idx,
                      self._scores_idx, self._num_det_idx)
        if max_idx >= n:
            logger.warning(
                "ObjectDetector: Unsupported model output layout "
                "(%d outputs, shapes=%s). Object detection will be disabled.",
                n, shapes,
            )
            return False
        return True

    @staticmethod
    def _generate_efficientdet_anchors(input_h: int, input_w: int) -> np.ndarray:
        """Generate EfficientDet-Lite0 anchor boxes for the given input size.

        Produces anchors in [cy, cx, ah, aw] normalised format (values in [0, 1]).
        Iteration order matches the TFLite model: stride → y → x → scale → ratio.

        Args:
            input_h: Model input height in pixels.
            input_w: Model input width in pixels.

        Returns:
            Float32 array of shape [num_anchors, 4].
        """
        anchors: List[List[float]] = []
        for stride in _EFF_STRIDES:
            feat_h = math.ceil(input_h / stride)
            feat_w = math.ceil(input_w / stride)
            for fy in range(feat_h):
                for fx in range(feat_w):
                    cy = (fy + 0.5) * stride / input_h
                    cx = (fx + 0.5) * stride / input_w
                    for s_idx in range(_EFF_NUM_SCALES):
                        base = _EFF_ANCHOR_SCALE * stride * (2 ** (s_idx / _EFF_NUM_SCALES))
                        for ratio in _EFF_ASPECT_RATIOS:
                            ah = base / input_h / math.sqrt(ratio)
                            aw = base / input_w * math.sqrt(ratio)
                            anchors.append([cy, cx, ah, aw])
        return np.array(anchors, dtype=np.float32)

    def detect(self, frame: np.ndarray) -> List[ObjectDetection]:
        """Run object detection on a frame, respecting frame_skip.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of ObjectDetection for target classes above confidence threshold.
        """
        if not self._model_loaded or self._interpreter is None:
            return []

        self._frame_count += 1

        if self._frame_count % self._config.frame_skip != 1 and self._frame_count != 1:
            return self._cached_detections

        try:
            detections = self._run_inference(frame)
            self._cached_detections = detections
            return detections
        except Exception:
            logger.exception("ObjectDetector: Inference failed")
            return self._cached_detections

    def _run_inference(self, frame: np.ndarray) -> List[ObjectDetection]:
        """Route to the appropriate inference path based on model format."""
        if self._raw_efficientdet:
            return self._run_efficientdet_inference(frame)
        return self._run_ssd_inference(frame)

    def _run_efficientdet_inference(self, frame: np.ndarray) -> List[ObjectDetection]:
        """Run inference for raw EfficientDet format with anchor decoding + NMS.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            Filtered list of ObjectDetection.
        """
        model_h, model_w = self._model_input_hw
        frame_h, frame_w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (model_w, model_h)).astype(np.uint8)
        input_data = np.expand_dims(resized, axis=0)

        self._interpreter.set_tensor(self._input_details[0]["index"], input_data)
        self._interpreter.invoke()

        # [N, C] class logits and [N, 4] encoded box deltas
        class_logits = self._interpreter.get_tensor(
            self._output_details[self._eff_class_logits_idx]["index"]
        )[0].astype(np.float32)
        raw_boxes = self._interpreter.get_tensor(
            self._output_details[self._eff_boxes_idx]["index"]
        )[0].astype(np.float32)

        # Sigmoid class probabilities
        class_probs = 1.0 / (1.0 + np.exp(-np.clip(class_logits, -88, 88)))

        # Decode boxes: [N, 4] → normalised [y_min, x_min, y_max, x_max]
        sf = _EFF_BOX_SCALE_FACTORS
        ay, ax, ah, aw = (self._anchors[:, k] for k in range(4))
        cy = raw_boxes[:, 0] / sf[0] * ah + ay
        cx = raw_boxes[:, 1] / sf[1] * aw + ax
        h  = np.exp(np.clip(raw_boxes[:, 2] / sf[2], -6, 6)) * ah
        w  = np.exp(np.clip(raw_boxes[:, 3] / sf[3], -6, 6)) * aw
        decoded = np.stack([cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2], axis=1)
        decoded = np.clip(decoded, 0.0, 1.0)

        threshold = self._config.confidence_threshold
        num_classes = class_probs.shape[1]
        results: List[ObjectDetection] = []

        for class_id, class_name in self._target_filter.items():
            if class_id >= num_classes:
                continue
            scores = class_probs[:, class_id]
            mask = scores >= threshold
            if not mask.any():
                continue

            sel_scores = scores[mask]
            sel_boxes = decoded[mask]  # [M, 4] normalised [y_min, x_min, y_max, x_max]

            # cv2.dnn.NMSBoxes expects [x, y, w, h] in pixel coords
            cv_boxes = [
                [int(b[1] * frame_w), int(b[0] * frame_h),
                 int((b[3] - b[1]) * frame_w), int((b[2] - b[0]) * frame_h)]
                for b in sel_boxes
            ]
            indices = cv2.dnn.NMSBoxes(
                cv_boxes, sel_scores.tolist(), threshold, _EFF_NMS_IOU
            )
            if len(indices) == 0:
                continue

            for idx in indices.flatten():
                b = sel_boxes[idx]
                bbox = (
                    int(b[1] * frame_w), int(b[0] * frame_h),
                    int(b[3] * frame_w), int(b[2] * frame_h),
                )
                results.append(ObjectDetection(
                    class_name=class_name,
                    confidence=float(sel_scores[idx]),
                    bbox=bbox,
                ))

        return results

    def _run_ssd_inference(self, frame: np.ndarray) -> List[ObjectDetection]:
        """Run inference for standard post-processed SSD format.

        Converts BGR->RGB before inference. Uses model's own input size.
        Logs all raw detections at DEBUG level for diagnostics.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            Filtered list of ObjectDetection.
        """
        model_h, model_w = self._model_input_hw
        frame_h, frame_w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (model_w, model_h))
        input_data = np.expand_dims(resized, axis=0)

        input_dtype = self._input_details[0]["dtype"]
        if input_dtype == np.float32:
            input_data = input_data.astype(np.float32) / 255.0
        else:
            input_data = input_data.astype(input_dtype)

        self._interpreter.set_tensor(self._input_details[0]["index"], input_data)
        self._interpreter.invoke()

        boxes = self._interpreter.get_tensor(
            self._output_details[self._boxes_idx]["index"]
        )[0]
        classes = self._interpreter.get_tensor(
            self._output_details[self._classes_idx]["index"]
        )[0]
        scores = self._interpreter.get_tensor(
            self._output_details[self._scores_idx]["index"]
        )[0]
        num_detections = int(
            self._interpreter.get_tensor(
                self._output_details[self._num_det_idx]["index"]
            )[0]
        )

        if logger.isEnabledFor(logging.DEBUG):
            for i in range(num_detections):
                cid = int(classes[i])
                score = float(scores[i])
                target = self._target_filter.get(cid)
                logger.debug(
                    "ObjectDetector raw[%d]: class_id=%d score=%.3f target='%s'",
                    i, cid, score, target or "---",
                )

        results: List[ObjectDetection] = []
        for i in range(num_detections):
            confidence = float(scores[i])
            if confidence < self._config.confidence_threshold:
                continue

            class_id = int(classes[i])
            class_name = self._target_filter.get(class_id)

            if class_name is None:
                continue

            y_min, x_min, y_max, x_max = boxes[i]
            bbox = (
                int(x_min * frame_w),
                int(y_min * frame_h),
                int(x_max * frame_w),
                int(y_max * frame_h),
            )

            results.append(
                ObjectDetection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=bbox,
                )
            )

        return results
