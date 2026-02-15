"""Distraction classifier module.

Loads a TFLite classification model and returns the probability that the
driver is distracted. Supports two model formats:

1. Binary classifier (1-2 outputs): Interprets output directly as P(distracted).
2. ImageNet 1001-class model: Sums probabilities of distraction-related classes
   (cell phone, iPod, laptop, book, remote, etc.) as a proxy distraction score.

If the model file is missing, returns 0.0 (neutral).

Phase: 1-2 (active).
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config_loader import ClassifierConfig
from src.detection.base_detector import BaseDetector

logger = logging.getLogger(__name__)

# ImageNet class indices (0-indexed, 1001-class with background at 0) that
# correspond to objects/activities associated with driver distraction.
# When an ImageNet model is detected, we sum these class probabilities.
IMAGENET_DISTRACTION_CLASSES = {
    # Handheld electronics
    487: "cell phone / cellular telephone",
    548: "entertainment center",  # screen-like
    664: "monitor",
    681: "notebook / laptop",
    508: "computer keyboard",
    # Handheld objects
    605: "iPod",
    # Reading material
    623: "letter opener",  # often near paper/documents
    472: "book jacket",
    921: "book",  # not a real ImageNet class but included for safety
    # Food and drink
    504: "coffee mug",
    968: "cup",
    898: "water bottle",
    737: "pop bottle / soda bottle",
    907: "water jug",
    # Grooming
    515: "comb",
    587: "hair spray",
    # Smoking
    808: "cigarette",
    # Generic distraction posture indicators
    # (person looking down, hand raised, etc. — these are heuristic)
}

# Broader set: anything handheld or screen-related
IMAGENET_HANDHELD_OBJECTS = {
    487, 605, 681, 508, 664,  # electronics
    504, 968, 898, 737, 907,  # drink containers
    472,                       # book-like
    515, 587,                  # grooming
}


@dataclass
class ClassifierResult:
    """Result from the distraction classifier."""

    distracted_probability: float = 0.0
    raw_logits: Optional[List[float]] = None


class DistractionClassifier(BaseDetector):
    """Classifier that estimates P(distracted) from a driver-facing frame.

    Auto-detects model format based on output shape:
    - 1 output: sigmoid binary classifier, output is P(distracted) directly.
    - 2 outputs: softmax binary classifier, output[1] is P(distracted).
    - 1001 outputs: ImageNet classifier, sums distraction-related class probs.

    Loads a .tflite model and runs inference. Preprocesses frames to the
    configured input size with BGR→RGB conversion and normalization.
    """

    def __init__(self, config: ClassifierConfig) -> None:
        super().__init__()
        self._config = config
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._output_size: int = 0
        self._is_imagenet: bool = False

        if config.enabled:
            self.load_model(config.model_path)

    def load_model(self, path: str) -> None:
        """Load a TFLite classifier model.

        Tries tflite_runtime first, falls back to tensorflow.lite on Windows.
        Auto-detects whether the model is binary or ImageNet multi-class.

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

            output_shape = self._output_details[0]["shape"]
            self._output_size = int(output_shape[-1])
            self._is_imagenet = self._output_size > 100

            self._model_loaded = True

            if self._is_imagenet:
                logger.warning(
                    "DistractionClassifier: Model has %d output classes — "
                    "detected as ImageNet model, NOT a binary distraction classifier. "
                    "Using distraction-related class probability sum as proxy. "
                    "For best results, use a dedicated binary distraction model.",
                    self._output_size,
                )
            else:
                logger.info(
                    "DistractionClassifier: Binary model loaded (%d outputs) from '%s'",
                    self._output_size,
                    path,
                )
        except Exception:
            logger.exception(
                "DistractionClassifier: Failed to load model from '%s'", path
            )
            self._model_loaded = False

    def detect(self, frame: np.ndarray) -> ClassifierResult:
        """Run distraction classification on a frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            ClassifierResult with distracted_probability in [0, 1].
        """
        if not self._model_loaded or self._interpreter is None:
            return ClassifierResult(distracted_probability=0.0)

        try:
            input_frame = self._preprocess(frame)
            self._interpreter.set_tensor(
                self._input_details[0]["index"], input_frame
            )
            self._interpreter.invoke()
            output = self._interpreter.get_tensor(
                self._output_details[0]["index"]
            )

            raw_logits = output.flatten().tolist()

            if self._is_imagenet:
                probability = self._imagenet_distraction_score(raw_logits)
            elif len(raw_logits) == 1:
                probability = float(raw_logits[0])
            elif len(raw_logits) == 2:
                probability = float(raw_logits[1])
            else:
                probability = float(raw_logits[1]) if len(raw_logits) >= 2 else 0.0

            probability = max(0.0, min(1.0, probability))

            return ClassifierResult(
                distracted_probability=probability,
                raw_logits=raw_logits,
            )
        except Exception:
            logger.exception("DistractionClassifier: Inference failed")
            return ClassifierResult(distracted_probability=0.0)

    def _imagenet_distraction_score(self, logits: List[float]) -> float:
        """Compute a distraction proxy score from ImageNet class probabilities.

        Sums the probabilities of ImageNet classes related to handheld objects,
        electronics, food/drink containers, and other distraction indicators.
        The sum is clamped to [0, 1] and scaled to be useful as a probability.

        Args:
            logits: Full softmax probability vector (1001 elements).

        Returns:
            Distraction proxy probability in [0, 1].
        """
        distraction_sum = 0.0
        for class_idx in IMAGENET_HANDHELD_OBJECTS:
            if class_idx < len(logits):
                distraction_sum += logits[class_idx]

        # Scale up since individual ImageNet probs are small when spread
        # across 1000 classes. A score of 0.05 across distraction classes
        # is actually quite significant.
        scaled = min(1.0, distraction_sum * 5.0)
        return scaled

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a frame for the classifier model.

        Converts BGR→RGB, resizes to the configured input_size, and applies
        MobileNetV2 preprocessing (scale to [-1, 1]) for binary models, or
        [0, 1] normalization for ImageNet models.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            Preprocessed frame as float32 array with batch dimension.
        """
        h, w = self._config.input_size
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (w, h))
        resized = resized.astype(np.float32)

        if self._is_imagenet:
            # ImageNet models expect [0, 1]
            resized /= 255.0
        else:
            # Binary MobileNetV2 models expect [-1, 1]
            resized = (resized / 127.5) - 1.0

        return np.expand_dims(resized, axis=0)
