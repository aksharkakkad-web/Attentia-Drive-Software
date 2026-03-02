"""Abstract base class for all detectors in Attentia Drive.

All detectors (classifier, object detector, face detector) inherit from this.
If a model file is missing, the detector returns a neutral/no-detection result
and logs a warning. The pipeline must never crash due to a missing model file.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """Abstract base class for all detectors.

    Subclasses must implement load_model() and detect(). The is_available()
    method returns True only if the model was loaded successfully.
    """

    def __init__(self) -> None:
        self._model_loaded: bool = False

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load a model from the given file path.

        Args:
            path: Path to the model file (.tflite, .onnx, .pt, etc.).
        """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Any:
        """Run detection on a single frame.

        Args:
            frame: BGR image as a numpy array (H, W, 3).

        Returns:
            Subclass-specific result type.
        """

    def is_available(self) -> bool:
        """Check if the detector model is loaded and ready.

        Returns:
            True if the model was loaded successfully.
        """
        return self._model_loaded

    def _check_model_file(self, path: str) -> bool:
        """Check if a model file exists at the given path.

        Args:
            path: Path to check.

        Returns:
            True if the file exists.
        """
        model_path = Path(path)
        if not model_path.exists():
            logger.warning(
                "%s: Model file not found at '%s'. "
                "Detector will return neutral results.",
                self.__class__.__name__,
                model_path.resolve(),
            )
            return False
        return True
