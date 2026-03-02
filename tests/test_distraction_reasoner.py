"""Tests for the distraction reasoner."""

import pytest

from src.detection.drowsiness import DrowsinessResult
from src.detection.face_detector import FaceResult
from src.logic.distraction_reasoner import DistractionReasoner
from src.logic.hysteresis import DriverState


class TestDistractionReasoner:
    """Tests for the DistractionReasoner signal fusion."""

    def test_phone_trigger(self) -> None:
        """Phone confirmed -> 'phone_use' in active triggers."""
        reasoner = DistractionReasoner(face_detector_enabled=False)
        assessment = reasoner.assess(
            driver_state=DriverState.DISTRACTED,
            smoothed_probability=0.8,
            confirmed_objects={"cell phone": True},
            face_result=FaceResult(),
            drowsiness_result=DrowsinessResult(),
        )
        assert "phone_use" in assessment.active_triggers

    def test_eating_drinking_trigger(self) -> None:
        """Cup or bottle confirmed -> 'eating_drinking' in triggers."""
        reasoner = DistractionReasoner(face_detector_enabled=False)

        assessment = reasoner.assess(
            driver_state=DriverState.MONITORING,
            smoothed_probability=0.5,
            confirmed_objects={"cup": True},
            face_result=FaceResult(),
            drowsiness_result=DrowsinessResult(),
        )
        assert "eating_drinking" in assessment.active_triggers

    def test_reading_trigger(self) -> None:
        """Book confirmed -> 'reading' in triggers."""
        reasoner = DistractionReasoner(face_detector_enabled=False)
        assessment = reasoner.assess(
            driver_state=DriverState.MONITORING,
            smoothed_probability=0.5,
            confirmed_objects={"book": True},
            face_result=FaceResult(),
            drowsiness_result=DrowsinessResult(),
        )
        assert "reading" in assessment.active_triggers

    def test_laptop_trigger(self) -> None:
        """Laptop confirmed -> 'laptop_use' in triggers."""
        reasoner = DistractionReasoner(face_detector_enabled=False)
        assessment = reasoner.assess(
            driver_state=DriverState.MONITORING,
            smoothed_probability=0.5,
            confirmed_objects={"laptop": True},
            face_result=FaceResult(),
            drowsiness_result=DrowsinessResult(),
        )
        assert "laptop_use" in assessment.active_triggers

    def test_distracted_posture_trigger(self) -> None:
        """DISTRACTED with no objects -> 'distracted_posture' trigger."""
        reasoner = DistractionReasoner(face_detector_enabled=False)
        assessment = reasoner.assess(
            driver_state=DriverState.DISTRACTED,
            smoothed_probability=0.8,
            confirmed_objects={},
            face_result=FaceResult(),
            drowsiness_result=DrowsinessResult(),
        )
        assert "distracted_posture" in assessment.active_triggers

    def test_distracted_with_object_no_posture_trigger(self) -> None:
        """DISTRACTED with confirmed object -> no 'distracted_posture'."""
        reasoner = DistractionReasoner(face_detector_enabled=False)
        assessment = reasoner.assess(
            driver_state=DriverState.DISTRACTED,
            smoothed_probability=0.8,
            confirmed_objects={"cell phone": True},
            face_result=FaceResult(),
            drowsiness_result=DrowsinessResult(),
        )
        assert "distracted_posture" not in assessment.active_triggers

    def test_multimodal_boost(self) -> None:
        """DISTRACTED + confirmed object -> confidence boosted by 0.1."""
        reasoner = DistractionReasoner(face_detector_enabled=False)
        assessment = reasoner.assess(
            driver_state=DriverState.DISTRACTED,
            smoothed_probability=0.8,
            confirmed_objects={"cell phone": True},
            face_result=FaceResult(),
            drowsiness_result=DrowsinessResult(),
        )
        assert assessment.confidence == pytest.approx(0.9)

    def test_multimodal_boost_capped_at_one(self) -> None:
        """Confidence boost is capped at 1.0."""
        reasoner = DistractionReasoner(face_detector_enabled=False)
        assessment = reasoner.assess(
            driver_state=DriverState.DISTRACTED,
            smoothed_probability=0.95,
            confirmed_objects={"cell phone": True},
            face_result=FaceResult(),
            drowsiness_result=DrowsinessResult(),
        )
        assert assessment.confidence == pytest.approx(1.0)

    def test_no_boost_without_distracted(self) -> None:
        """No confidence boost if not in DISTRACTED state."""
        reasoner = DistractionReasoner(face_detector_enabled=False)
        assessment = reasoner.assess(
            driver_state=DriverState.MONITORING,
            smoothed_probability=0.5,
            confirmed_objects={"cell phone": True},
            face_result=FaceResult(),
            drowsiness_result=DrowsinessResult(),
        )
        assert assessment.confidence == pytest.approx(0.5)

    def test_face_not_visible_forces_attentive(self) -> None:
        """When face detector enabled and face not visible -> force ATTENTIVE."""
        reasoner = DistractionReasoner(face_detector_enabled=True)
        assessment = reasoner.assess(
            driver_state=DriverState.DISTRACTED,
            smoothed_probability=0.9,
            confirmed_objects={"cell phone": True},
            face_result=FaceResult(face_visible=False),
            drowsiness_result=DrowsinessResult(),
        )
        assert assessment.state == DriverState.ATTENTIVE
        assert assessment.is_distracted is False
        assert assessment.confidence == 0.0

    def test_face_visible_does_not_override(self) -> None:
        """When face is visible, state is not overridden."""
        reasoner = DistractionReasoner(face_detector_enabled=True)
        assessment = reasoner.assess(
            driver_state=DriverState.DISTRACTED,
            smoothed_probability=0.8,
            confirmed_objects={},
            face_result=FaceResult(face_visible=True),
            drowsiness_result=DrowsinessResult(),
        )
        assert assessment.state == DriverState.DISTRACTED

    def test_face_detector_disabled_no_override(self) -> None:
        """When face detector disabled, face_visible=False does NOT override."""
        reasoner = DistractionReasoner(face_detector_enabled=False)
        assessment = reasoner.assess(
            driver_state=DriverState.DISTRACTED,
            smoothed_probability=0.8,
            confirmed_objects={},
            face_result=FaceResult(face_visible=False),
            drowsiness_result=DrowsinessResult(),
        )
        assert assessment.state == DriverState.DISTRACTED

    def test_assessment_fields(self) -> None:
        """Assessment has all expected fields populated."""
        reasoner = DistractionReasoner(face_detector_enabled=False)
        assessment = reasoner.assess(
            driver_state=DriverState.ATTENTIVE,
            smoothed_probability=0.1,
            confirmed_objects={},
            face_result=FaceResult(),
            drowsiness_result=DrowsinessResult(),
        )
        assert assessment.state == DriverState.ATTENTIVE
        assert assessment.is_distracted is False
        assert assessment.is_monitoring is False
        assert assessment.confidence == pytest.approx(0.1)
        assert assessment.active_triggers == []
        assert assessment.confirmed_objects == []
        assert assessment.timestamp > 0

    def test_multiple_triggers(self) -> None:
        """Multiple objects confirmed -> multiple triggers."""
        reasoner = DistractionReasoner(face_detector_enabled=False)
        assessment = reasoner.assess(
            driver_state=DriverState.DISTRACTED,
            smoothed_probability=0.8,
            confirmed_objects={"cell phone": True, "cup": True},
            face_result=FaceResult(),
            drowsiness_result=DrowsinessResult(),
        )
        assert "phone_use" in assessment.active_triggers
        assert "eating_drinking" in assessment.active_triggers
        assert "distracted_posture" not in assessment.active_triggers

    def test_unconfirmed_objects_not_in_triggers(self) -> None:
        """Objects with confirmed=False are not in triggers."""
        reasoner = DistractionReasoner(face_detector_enabled=False)
        assessment = reasoner.assess(
            driver_state=DriverState.DISTRACTED,
            smoothed_probability=0.8,
            confirmed_objects={"cell phone": False, "cup": False},
            face_result=FaceResult(),
            drowsiness_result=DrowsinessResult(),
        )
        assert "phone_use" not in assessment.active_triggers
        assert "eating_drinking" not in assessment.active_triggers
        assert "distracted_posture" in assessment.active_triggers

    def test_gaze_away_trigger_yaw(self) -> None:
        """Head yaw > 40 degrees -> 'gaze_away' trigger."""
        reasoner = DistractionReasoner(face_detector_enabled=True)
        face = FaceResult(
            face_visible=True,
            head_pose=(0.0, 45.0, 0.0),
            ear_left=0.3,
            ear_right=0.3,
        )
        assessment = reasoner.assess(
            driver_state=DriverState.MONITORING,
            smoothed_probability=0.5,
            confirmed_objects={},
            face_result=face,
            drowsiness_result=DrowsinessResult(),
        )
        assert "gaze_away" in assessment.active_triggers

    def test_gaze_away_trigger_pitch(self) -> None:
        """Head pitch > 25 degrees (looking down) -> 'gaze_away' trigger."""
        reasoner = DistractionReasoner(face_detector_enabled=True)
        face = FaceResult(
            face_visible=True,
            head_pose=(30.0, 0.0, 0.0),
            ear_left=0.3,
            ear_right=0.3,
        )
        assessment = reasoner.assess(
            driver_state=DriverState.MONITORING,
            smoothed_probability=0.5,
            confirmed_objects={},
            face_result=face,
            drowsiness_result=DrowsinessResult(),
        )
        assert "gaze_away" in assessment.active_triggers

    def test_no_gaze_away_within_threshold(self) -> None:
        """Head yaw < 40 and pitch < 25 -> no 'gaze_away' trigger."""
        reasoner = DistractionReasoner(face_detector_enabled=True)
        face = FaceResult(
            face_visible=True,
            head_pose=(10.0, 20.0, 0.0),
            ear_left=0.3,
            ear_right=0.3,
        )
        assessment = reasoner.assess(
            driver_state=DriverState.MONITORING,
            smoothed_probability=0.5,
            confirmed_objects={},
            face_result=face,
            drowsiness_result=DrowsinessResult(),
        )
        assert "gaze_away" not in assessment.active_triggers

    def test_eyes_closed_trigger(self) -> None:
        """Average EAR < 0.18 -> 'eyes_closed' trigger."""
        reasoner = DistractionReasoner(face_detector_enabled=True)
        face = FaceResult(
            face_visible=True,
            head_pose=(0.0, 0.0, 0.0),
            ear_left=0.15,
            ear_right=0.15,
        )
        assessment = reasoner.assess(
            driver_state=DriverState.MONITORING,
            smoothed_probability=0.5,
            confirmed_objects={},
            face_result=face,
            drowsiness_result=DrowsinessResult(),
        )
        assert "eyes_closed" in assessment.active_triggers

    def test_no_eyes_closed_above_threshold(self) -> None:
        """Average EAR > 0.18 -> no 'eyes_closed' trigger."""
        reasoner = DistractionReasoner(face_detector_enabled=True)
        face = FaceResult(
            face_visible=True,
            head_pose=(0.0, 0.0, 0.0),
            ear_left=0.25,
            ear_right=0.25,
        )
        assessment = reasoner.assess(
            driver_state=DriverState.MONITORING,
            smoothed_probability=0.5,
            confirmed_objects={},
            face_result=face,
            drowsiness_result=DrowsinessResult(),
        )
        assert "eyes_closed" not in assessment.active_triggers

    def test_drowsy_trigger(self) -> None:
        """DrowsinessResult.is_drowsy=True -> 'drowsy' trigger."""
        reasoner = DistractionReasoner(face_detector_enabled=True)
        face = FaceResult(face_visible=True, head_pose=(0.0, 0.0, 0.0), ear_left=0.3, ear_right=0.3)
        drowsy = DrowsinessResult(is_drowsy=True, drowsiness_confidence=0.8)
        assessment = reasoner.assess(
            driver_state=DriverState.MONITORING,
            smoothed_probability=0.5,
            confirmed_objects={},
            face_result=face,
            drowsiness_result=drowsy,
        )
        assert "drowsy" in assessment.active_triggers

    def test_no_drowsy_trigger_when_not_drowsy(self) -> None:
        """DrowsinessResult.is_drowsy=False -> no 'drowsy' trigger."""
        reasoner = DistractionReasoner(face_detector_enabled=True)
        face = FaceResult(face_visible=True, head_pose=(0.0, 0.0, 0.0), ear_left=0.3, ear_right=0.3)
        drowsy = DrowsinessResult(is_drowsy=False)
        assessment = reasoner.assess(
            driver_state=DriverState.MONITORING,
            smoothed_probability=0.5,
            confirmed_objects={},
            face_result=face,
            drowsiness_result=drowsy,
        )
        assert "drowsy" not in assessment.active_triggers
