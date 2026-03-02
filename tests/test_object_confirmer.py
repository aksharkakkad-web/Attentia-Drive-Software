"""Tests for the N-of-M object temporal confirmer."""

import pytest

from src.config_loader import ObjectConfirmationConfig
from src.detection.object_detector import ObjectDetection
from src.logic.object_confirmer import ObjectTemporalConfirmer


def make_detection(class_name: str, confidence: float = 0.8) -> ObjectDetection:
    """Create a test ObjectDetection."""
    return ObjectDetection(
        class_name=class_name,
        confidence=confidence,
        bbox=(10, 10, 100, 100),
    )


def make_confirmer(n: int = 15, m: int = 30) -> ObjectTemporalConfirmer:
    """Create an ObjectTemporalConfirmer with given N and M."""
    config = ObjectConfirmationConfig(enabled=True, n=n, m=m)
    return ObjectTemporalConfirmer(config)


class TestObjectTemporalConfirmer:
    """Tests for N-of-M temporal confirmation."""

    def test_confirmation_at_threshold(self) -> None:
        """Object present in exactly N of M frames is confirmed."""
        confirmer = make_confirmer(n=15, m=30)

        # Present for 15 frames
        for _ in range(15):
            confirmer.update([make_detection("cell phone")])

        # Absent for 15 frames
        for _ in range(15):
            confirmer.update([])

        confirmed = confirmer.get_confirmed_objects()
        assert "cell phone" in confirmed

    def test_below_threshold_not_confirmed(self) -> None:
        """Object present in N-1 of M frames is NOT confirmed."""
        confirmer = make_confirmer(n=15, m=30)

        # Present for 14 frames
        for _ in range(14):
            confirmer.update([make_detection("cell phone")])

        # Absent for 16 frames
        for _ in range(16):
            confirmer.update([])

        confirmed = confirmer.get_confirmed_objects()
        assert "cell phone" not in confirmed

    def test_object_unconfirms_after_absence(self) -> None:
        """Object un-confirms after enough absent frames."""
        confirmer = make_confirmer(n=15, m=30)

        # Present for 30 frames (fully confirmed)
        for _ in range(30):
            confirmer.update([make_detection("cell phone")])
        assert "cell phone" in confirmer.get_confirmed_objects()

        # Absent for 16 frames (enough to drop below threshold)
        for _ in range(16):
            confirmer.update([])

        # Should no longer be confirmed (14 of last 30)
        assert "cell phone" not in confirmer.get_confirmed_objects()

    def test_multiple_classes_independent(self) -> None:
        """Multiple object classes are tracked independently."""
        confirmer = make_confirmer(n=3, m=5)

        # Phone present for all 5 frames, cup for only 2
        for i in range(5):
            detections = [make_detection("cell phone")]
            if i < 2:
                detections.append(make_detection("cup"))
            confirmer.update(detections)

        confirmed = confirmer.get_confirmed_objects()
        assert "cell phone" in confirmed
        assert "cup" not in confirmed

    def test_empty_detections(self) -> None:
        """No detections means nothing confirmed."""
        confirmer = make_confirmer(n=3, m=5)
        for _ in range(10):
            confirmer.update([])
        assert confirmer.get_confirmed_objects() == []

    def test_update_returns_confirmation_dict(self) -> None:
        """update() returns a dict of class_name -> confirmed."""
        confirmer = make_confirmer(n=2, m=3)

        confirmer.update([make_detection("cell phone")])
        result = confirmer.update([make_detection("cell phone")])

        assert isinstance(result, dict)
        assert result["cell phone"] is True

    def test_reset_clears_all(self) -> None:
        """Reset clears all tracking buffers."""
        confirmer = make_confirmer(n=2, m=3)

        for _ in range(3):
            confirmer.update([make_detection("cell phone")])
        assert "cell phone" in confirmer.get_confirmed_objects()

        confirmer.reset()
        assert confirmer.get_confirmed_objects() == []

    def test_sliding_window_behavior(self) -> None:
        """Old frames drop out of the sliding window."""
        confirmer = make_confirmer(n=3, m=5)

        # Present for 3 frames -> confirmed
        for _ in range(3):
            confirmer.update([make_detection("bottle")])
        assert "bottle" in confirmer.get_confirmed_objects()

        # Absent for 3 frames -> drops to 2 of 5 -> not confirmed
        for _ in range(3):
            confirmer.update([])
        assert "bottle" not in confirmer.get_confirmed_objects()

    def test_n_equals_one(self) -> None:
        """N=1 means any single detection confirms."""
        confirmer = make_confirmer(n=1, m=5)
        confirmer.update([make_detection("book")])
        assert "book" in confirmer.get_confirmed_objects()

    def test_n_equals_m(self) -> None:
        """N=M means object must be present in every frame."""
        confirmer = make_confirmer(n=5, m=5)

        for _ in range(4):
            confirmer.update([make_detection("laptop")])
        assert "laptop" not in confirmer.get_confirmed_objects()

        confirmer.update([make_detection("laptop")])
        assert "laptop" in confirmer.get_confirmed_objects()
