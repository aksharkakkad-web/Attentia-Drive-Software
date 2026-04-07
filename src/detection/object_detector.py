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
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

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
        # True when the model uses raw-anchor format: [1,N,C] scores + [1,N,4] boxes
        # instead of the standard 4-tensor postprocessed SSD format.
        self._raw_output_format: bool = False

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

            self._interpreter = Interpreter(model_path=path)
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()

            # Auto-detect input size from model
            input_shape = self._input_details[0]["shape"]
            self._model_input_hw = (int(input_shape[1]), int(input_shape[2]))

            self._resolve_output_indices()
            self._raw_output_format = self._detect_raw_output_format()

            # Try to load embedded label map
            embedded_map = _load_embedded_labelmap(path)
            self._target_filter = _build_target_filter(
                embedded_map, self._config.target_classes
            )

            self._model_loaded = True

            source = "embedded metadata" if embedded_map else "fallback hardcoded"
            inp = self._input_details[0]
            fmt = "raw-anchor" if self._raw_output_format else "postprocessed-SSD"
            logger.info(
                "ObjectDetector: Model loaded from '%s' | "
                "input=%s %s | %d outputs | format=%s | label source: %s | "
                "target IDs: %s",
                path,
                self._model_input_hw,
                inp["dtype"],
                len(self._output_details),
                fmt,
                source,
                dict(self._target_filter),
            )
        except ImportError:
            logger.warning(
                "ObjectDetector: TFLite runtime not installed — phone detection disabled. "
                "Install tensorflow (macOS) or tflite-runtime (Linux) to enable."
            )
            self._model_loaded = False
        except Exception:
            logger.exception(
                "ObjectDetector: Failed to load model from '%s'", path
            )
            self._model_loaded = False

    def _resolve_output_indices(self) -> None:
        """Resolve output tensor indices by shape instead of assuming fixed order.

        Standard SSD output shapes:
        - boxes:          [1, N, 4]
        - classes:        [1, N]
        - scores:         [1, N]
        - num_detections: [1]
        """
        for i, detail in enumerate(self._output_details):
            shape = tuple(detail["shape"])
            if len(shape) == 3 and shape[-1] == 4:
                self._boxes_idx = i
            elif len(shape) == 1:
                self._num_det_idx = i

        rank2_indices = []
        for i, detail in enumerate(self._output_details):
            shape = tuple(detail["shape"])
            if len(shape) == 2:
                rank2_indices.append(i)

        if len(rank2_indices) == 2:
            name0 = self._output_details[rank2_indices[0]].get("name", "")
            name1 = self._output_details[rank2_indices[1]].get("name", "")
            if "class" in name1.lower() and "class" not in name0.lower():
                self._scores_idx = rank2_indices[0]
                self._classes_idx = rank2_indices[1]
            else:
                self._classes_idx = rank2_indices[0]
                self._scores_idx = rank2_indices[1]

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

    def _detect_raw_output_format(self) -> bool:
        """Return True if the model uses raw-anchor format instead of 4-tensor postprocessed SSD.

        Raw-anchor models expose 2 tensors:
          - [1, N, C]  class logits/scores  (C > 4)
          - [1, N, 4]  box predictions
        Postprocessed SSD models expose 4 tensors (boxes, classes, scores, num_detections).
        """
        if len(self._output_details) != 2:
            return False
        shapes = [tuple(d["shape"]) for d in self._output_details]
        has_class_tensor = any(len(s) == 3 and s[-1] > 4 for s in shapes)
        has_box_tensor   = any(len(s) == 3 and s[-1] == 4 for s in shapes)
        return has_class_tensor and has_box_tensor

    def _run_inference(self, frame: np.ndarray) -> List[ObjectDetection]:
        """Run the actual TFLite inference and parse results.

        Converts BGR->RGB before inference. Uses model's own input size.
        Dispatches to _run_inference_raw() for raw-anchor models.
        Logs all raw detections at DEBUG level for diagnostics.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            Filtered list of ObjectDetection.
        """
        if self._raw_output_format:
            return self._run_inference_raw(frame)

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

    def _run_inference_raw(self, frame: np.ndarray) -> List[ObjectDetection]:
        """Inference for raw-anchor output format: [1,N,C] scores + [1,N,4] boxes.

        Applies sigmoid over class dimension and picks the highest-scoring anchor
        per target class. Box decoding requires anchor priors (not available here),
        so bbox is returned as (0,0,0,0) — sufficient for MVP phone detection which
        only uses detected/max_confidence, not the bbox.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            At most one ObjectDetection per target class above the confidence threshold.
        """
        model_h, model_w = self._model_input_hw

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

        # Find the class-score tensor [1, N, C] and ignore the box tensor
        scores_raw = None
        for detail in self._output_details:
            tensor = self._interpreter.get_tensor(detail["index"])
            if len(tensor.shape) == 3 and tensor.shape[-1] > 4:
                scores_raw = tensor[0]  # [N, C]
                break

        if scores_raw is None:
            return []

        # Sigmoid → class probabilities
        scores_prob = 1.0 / (1.0 + np.exp(-scores_raw.astype(np.float32)))  # [N, C]

        results: List[ObjectDetection] = []
        for class_id, class_name in self._target_filter.items():
            if class_id >= scores_prob.shape[1]:
                continue
            max_score = float(np.max(scores_prob[:, class_id]))
            logger.debug(
                "ObjectDetector raw-anchor: class_id=%d name='%s' max_score=%.3f",
                class_id, class_name, max_score,
            )
            if max_score >= self._config.confidence_threshold:
                results.append(
                    ObjectDetection(class_name=class_name, confidence=max_score, bbox=(0, 0, 0, 0))
                )

        return results
