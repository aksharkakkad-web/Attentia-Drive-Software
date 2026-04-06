"""Signal processor — Layer 2 of the detection pipeline.

Takes a PerceptionBundle and produces a SignalFrame with Kalman-filtered,
neutral-pose-corrected head pose, EAR-based eye signals, world-space gaze,
and phone signal.

KEY DESIGN: Reads head pose and EAR from the MVP-ONLY fields on
PerceptionBundle (head_pose_raw, ear_left, ear_right). Does NOT import
MediaPipe. When custom models replace MediaPipe, the adapter (Phase 7A)
will populate these same fields — this layer does not change.

PRD §5.1–§5.7
"""

from src.config_prd import (
    CAPTURE_FPS,
    EAR_BASELINE_POPULATION_DEFAULT,
    EAR_DEFAULT_CLOSE_THRESHOLD,
    LSTM_RESET_ABSENT_FRAMES,
    ROAD_ZONE_PITCH_MAX,
    ROAD_ZONE_PITCH_MIN,
    ROAD_ZONE_YAW_MAX,
    ROAD_ZONE_YAW_MIN,
)
from src.contracts import (
    EyeSignals,
    GazeWorld,
    HeadPose,
    PerceptionBundle,
    PhoneSignal,
    SignalFrame,
)
from src.logic.kalman_filter import KalmanFilter


class SignalProcessor:
    """Converts raw perception output into clean, calibrated signals.

    Internal state:
    - Five KalmanFilter instances: head yaw/pitch/roll, gaze world yaw/pitch.
    - Neutral pose offsets (yaw, pitch) — zero until calibration provides them.
    - EAR baseline and close threshold — population defaults until calibrated.
    - face_absent_count — triggers Kalman reset after LSTM_RESET_ABSENT_FRAMES.

    PRD §5.1–§5.7.

    Args:
        dt: Frame time step in seconds. Default 1/CAPTURE_FPS from config_prd.
    """

    def __init__(self, dt: float = 1.0 / CAPTURE_FPS) -> None:
        # Five independent Kalman filters (PRD §5.6)
        self._kf_yaw = KalmanFilter(dt=dt)
        self._kf_pitch = KalmanFilter(dt=dt)
        self._kf_roll = KalmanFilter(dt=dt)
        self._kf_gaze_yaw = KalmanFilter(dt=dt)
        self._kf_gaze_pitch = KalmanFilter(dt=dt)

        # Neutral pose offsets (PRD §5.7) — set by calibration
        self._neutral_yaw_offset: float = 0.0
        self._neutral_pitch_offset: float = 0.0

        # EAR calibration state (PRD §5.2)
        self._baseline_EAR: float = EAR_BASELINE_POPULATION_DEFAULT
        self._close_threshold: float = EAR_DEFAULT_CLOSE_THRESHOLD
        self._calibration_complete: bool = False

        # Face absence tracking (PRD §5.6 reset condition)
        self._face_absent_count: int = 0

    # ── Configuration methods ─────────────────────────────────────────────────

    def set_neutral_offsets(self, yaw: float, pitch: float) -> None:
        """Set neutral pose offsets from calibration. PRD §5.7.

        Args:
            yaw: Neutral yaw offset in degrees.
            pitch: Neutral pitch offset in degrees.
        """
        self._neutral_yaw_offset = yaw
        self._neutral_pitch_offset = pitch

    def set_ear_baseline(self, baseline: float, threshold: float) -> None:
        """Set per-session EAR baseline from calibration. PRD §5.2.

        After this call, calibration_complete=True and EyeSignals will use
        the provided threshold instead of the population default (0.21).

        Args:
            baseline: Mean open-eye EAR measured during calibration window.
            threshold: Pre-computed close threshold (typically baseline * 0.75).
        """
        self._baseline_EAR = baseline
        self._close_threshold = threshold
        self._calibration_complete = True

    def reset_filters(self) -> None:
        """Reset all five Kalman filters to uninitialized state.

        Called internally when face has been absent > LSTM_RESET_ABSENT_FRAMES.
        Can also be called externally on source restart.
        """
        self._kf_yaw.reset()
        self._kf_pitch.reset()
        self._kf_roll.reset()
        self._kf_gaze_yaw.reset()
        self._kf_gaze_pitch.reset()

    # ── Main processing method ────────────────────────────────────────────────

    def process(self, bundle: PerceptionBundle) -> SignalFrame:
        """Process one PerceptionBundle → SignalFrame.

        Phone signal is always extracted regardless of face presence.
        Head pose, EAR, and gaze are only computed when face is present.

        Args:
            bundle: Perception output for this frame.

        Returns:
            SignalFrame with all signals populated (or None for unavailable fields).
        """
        # Phone signal is always extracted — independent of face (PRD §5.3)
        phone_signal = PhoneSignal(
            detected=bundle.phone.detected,
            confidence=bundle.phone.max_confidence,
            stale=bundle.phone_result_stale,
        )

        # ── Face absent path ──────────────────────────────────────────────────
        if not bundle.face.present:
            self._face_absent_count += 1
            if self._face_absent_count > LSTM_RESET_ABSENT_FRAMES:
                self.reset_filters()
            return SignalFrame(
                timestamp_ns=bundle.timestamp_ns,
                frame_id=bundle.frame_id,
                face_present=False,
                signals_valid=False,
                phone_signal=phone_signal,
            )

        # ── Face present path ─────────────────────────────────────────────────
        self._face_absent_count = 0

        # No head pose data → cannot compute anything meaningful
        if bundle.head_pose_raw is None:
            return SignalFrame(
                timestamp_ns=bundle.timestamp_ns,
                frame_id=bundle.frame_id,
                face_present=True,
                signals_valid=False,
                phone_signal=phone_signal,
            )

        # Unpack head pose: (pitch, yaw, roll) from MediaPipe MVP-ONLY field
        raw_pitch, raw_yaw, raw_roll = bundle.head_pose_raw

        # Kalman filter each angle independently (PRD §5.6)
        filtered_yaw = self._kf_yaw.update(raw_yaw)
        filtered_pitch = self._kf_pitch.update(raw_pitch)
        filtered_roll = self._kf_roll.update(raw_roll)

        # Neutral pose offset correction (PRD §5.7)
        corrected_yaw = filtered_yaw - self._neutral_yaw_offset
        corrected_pitch = filtered_pitch - self._neutral_pitch_offset

        head_pose = HeadPose(
            yaw_deg=corrected_yaw,
            pitch_deg=corrected_pitch,
            roll_deg=filtered_roll,
            valid=True,
            raw_yaw_deg=raw_yaw,
            raw_pitch_deg=raw_pitch,
            raw_roll_deg=raw_roll,
        )

        # EAR computation (PRD §5.2)
        mean_EAR = (bundle.ear_left + bundle.ear_right) / 2.0
        # Use stored threshold directly — set by set_ear_baseline() or default 0.21
        close_threshold = self._close_threshold

        eye_signals = EyeSignals(
            left_EAR=bundle.ear_left,
            right_EAR=bundle.ear_right,
            mean_EAR=mean_EAR,
            baseline_EAR=self._baseline_EAR,
            close_threshold=close_threshold,
            valid=True,
            calibration_complete=self._calibration_complete,
        )

        # MVP gaze: world-space gaze = corrected head pose (PRD §5.3 MVP path)
        # When real gaze model arrives: gaze_world = gaze_camera + 0.7*head.
        # Only this step changes — everything downstream is already correct.
        gaze_world_yaw = self._kf_gaze_yaw.update(corrected_yaw)
        gaze_world_pitch = self._kf_gaze_pitch.update(corrected_pitch)

        on_road = (
            ROAD_ZONE_YAW_MIN <= gaze_world_yaw <= ROAD_ZONE_YAW_MAX
            and ROAD_ZONE_PITCH_MIN <= gaze_world_pitch <= ROAD_ZONE_PITCH_MAX
        )

        gaze_world = GazeWorld(
            yaw_deg=gaze_world_yaw,
            pitch_deg=gaze_world_pitch,
            on_road=on_road,
            valid=True,
        )

        return SignalFrame(
            timestamp_ns=bundle.timestamp_ns,
            frame_id=bundle.frame_id,
            face_present=True,
            head_pose=head_pose,
            eye_signals=eye_signals,
            gaze_world=gaze_world,
            phone_signal=phone_signal,
            signals_valid=True,
        )
