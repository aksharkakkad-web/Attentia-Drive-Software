"""YOLOv8-nano phone detector backed by ONNX Runtime.

Loads a single-class YOLOv8-nano ONNX model trained for phone detection
(mAP50 = 0.977) and returns ObjectDetection results in the same shape as
the generic ObjectDetector so the pipeline can drop one in for the other.

Model contract (verified at load time):
  input  'images':  (1, 3, 640, 640) float32, RGB, normalized to [0, 1]
  output 'output0': (1, 5, 8400)     each prediction = [cx, cy, w, h, conf]

Inference pipeline:
  BGR frame  →  resize to 640x640 (direct, no letterbox)
             →  RGB  →  /255.0  →  CHW  →  NCHW
             →  ONNX session.run
             →  filter by PHONE_CONFIDENCE_THRESHOLD
             →  cv2.dnn.NMSBoxes (IoU 0.5)
             →  rescale boxes to original frame size
             →  List[ObjectDetection]

Phase: 1-2 (active) — replaces the EfficientDet TFLite path for phones.
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np

from src.config_loader import ObjectDetectorConfig
from src.config_prd import PHONE_CONFIDENCE_THRESHOLD
from src.detection.base_detector import BaseDetector
from src.detection.object_detector import ObjectDetection

logger = logging.getLogger(__name__)

# Model contract — set by training, not config.
_MODEL_INPUT_HW: Tuple[int, int] = (640, 640)
_NMS_IOU_THRESHOLD: float = 0.5
_PHONE_CLASS_NAME: str = "cell phone"


class PhoneDetectorYolo(BaseDetector):
    """YOLOv8-nano single-class phone detector via ONNX Runtime.

    Honours the same ObjectDetectorConfig as ObjectDetector so it can be
    swapped in transparently:
      * config.enabled        — gates load
      * config.model_path     — overridden to the ONNX file
      * config.frame_skip     — only run inference every Nth frame
      * config.confidence_threshold is ignored; phones use the
        PHONE-specific threshold from config_prd.py (PRD §19).
    """

    def __init__(self, config: ObjectDetectorConfig) -> None:
        super().__init__()
        self._config = config
        self._frame_count: int = 0
        self._cached_detections: List[ObjectDetection] = []

        self._session = None
        self._input_name: str = ""
        self._confidence_threshold: float = float(PHONE_CONFIDENCE_THRESHOLD)

        if config.enabled:
            self.load_model(config.model_path)

    # ── Loading ────────────────────────────────────────────────────────────────

    def load_model(self, path: str) -> None:
        """Load the ONNX model and validate its I/O shape contract."""
        if not self._check_model_file(path):
            return

        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning(
                "PhoneDetectorYolo: onnxruntime not installed — phone detection disabled."
            )
            return

        try:
            self._session = ort.InferenceSession(
                path, providers=["CPUExecutionProvider"]
            )
            inputs = self._session.get_inputs()
            outputs = self._session.get_outputs()
            if not inputs or not outputs:
                raise RuntimeError("ONNX model exposes no inputs/outputs")

            self._input_name = inputs[0].name
            input_shape = tuple(inputs[0].shape)
            output_shape = tuple(outputs[0].shape)

            self._model_loaded = True
            logger.info(
                "PhoneDetectorYolo: ONNX backend loaded from '%s' | "
                "input=%s output=%s | conf_threshold=%.2f | iou=%.2f",
                path,
                input_shape,
                output_shape,
                self._confidence_threshold,
                _NMS_IOU_THRESHOLD,
            )
        except Exception:
            logger.exception(
                "PhoneDetectorYolo: failed to load ONNX model '%s'", path
            )
            self._session = None
            self._model_loaded = False

    # ── Public detect ──────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[ObjectDetection]:
        """Run YOLO inference on a frame, respecting frame_skip caching.

        Args:
            frame: BGR image as a numpy array (H, W, 3).

        Returns:
            List of ObjectDetection for "cell phone" above confidence threshold.
        """
        if not self._model_loaded or self._session is None:
            return []

        self._frame_count += 1

        if (self._frame_count - 1) % self._config.frame_skip != 0:
            return self._cached_detections

        try:
            detections = self._run_inference(frame)
            self._cached_detections = detections
            return detections
        except Exception:
            logger.exception("PhoneDetectorYolo: inference failed")
            return self._cached_detections

    # ── Inference internals ────────────────────────────────────────────────────

    def _run_inference(self, frame: np.ndarray) -> List[ObjectDetection]:
        frame_h, frame_w = frame.shape[:2]
        model_h, model_w = _MODEL_INPUT_HW

        # Preprocess: BGR → RGB → resize → normalize → CHW → NCHW
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (model_w, model_h))
        input_data = resized.astype(np.float32) / 255.0
        input_data = np.transpose(input_data, (2, 0, 1))      # (3, 640, 640)
        input_data = np.expand_dims(input_data, axis=0)       # (1, 3, 640, 640)

        # Inference: output is (1, 5, 8400) → transpose to (8400, 5)
        outputs = self._session.run(None, {self._input_name: input_data})
        preds = outputs[0]
        if preds.ndim == 3 and preds.shape[1] == 5:
            preds = preds[0].T  # (8400, 5)
        elif preds.ndim == 3 and preds.shape[2] == 5:
            preds = preds[0]    # already (8400, 5)
        else:
            logger.warning(
                "PhoneDetectorYolo: unexpected output shape %s — skipping frame",
                preds.shape,
            )
            return []

        # Filter by confidence first (cheap) before NMS
        scores = preds[:, 4]
        keep_mask = scores >= self._confidence_threshold
        if not np.any(keep_mask):
            return []

        kept = preds[keep_mask]
        kept_scores = scores[keep_mask]

        # Convert (cx, cy, w, h) in 640x640 space to xywh in original-frame
        # pixels (cv2.dnn.NMSBoxes expects xywh: top-left + width + height).
        scale_x = frame_w / float(model_w)
        scale_y = frame_h / float(model_h)

        cx = kept[:, 0] * scale_x
        cy = kept[:, 1] * scale_y
        bw = kept[:, 2] * scale_x
        bh = kept[:, 3] * scale_y
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0

        boxes_xywh = np.stack([x1, y1, bw, bh], axis=1).tolist()
        scores_list = kept_scores.tolist()

        # NMS — pass score_threshold=0.0 because we already pre-filtered above;
        # cv2.dnn.NMSBoxes uses strict >, so passing the real threshold would
        # silently drop detections at exactly the threshold value.
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
            scores_list,
            0.0,
            _NMS_IOU_THRESHOLD,
        )
        if indices is None or len(indices) == 0:
            return []

        # cv2.dnn.NMSBoxes returns either a 2D array or a flat list depending
        # on the OpenCV build — flatten to be safe.
        indices = np.array(indices).flatten()

        results: List[ObjectDetection] = []
        for idx in indices:
            x, y, w, h = boxes_xywh[idx]
            x_min = int(max(0, round(x)))
            y_min = int(max(0, round(y)))
            x_max = int(min(frame_w, round(x + w)))
            y_max = int(min(frame_h, round(y + h)))
            if x_max <= x_min or y_max <= y_min:
                continue
            results.append(
                ObjectDetection(
                    class_name=_PHONE_CLASS_NAME,
                    confidence=float(scores_list[idx]),
                    bbox=(x_min, y_min, x_max, y_max),
                )
            )

        if logger.isEnabledFor(logging.DEBUG):
            for d in results:
                logger.debug(
                    "PhoneDetectorYolo: %s conf=%.3f bbox=%s",
                    d.class_name, d.confidence, d.bbox,
                )

        return results
