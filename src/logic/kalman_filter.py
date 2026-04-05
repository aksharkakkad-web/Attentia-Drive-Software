"""1D constant-velocity Kalman filter for head pose and gaze smoothing.

Applied independently to each angle output (head yaw, pitch, roll,
gaze world yaw, gaze world pitch) to suppress PnP jitter before
the values reach duration timers in Layer 3.

PRD §5.6 — Kalman Filter for Pose and Gaze Smoothing.
"""

from typing import Optional

import numpy as np

from src.config_prd import (
    KALMAN_INITIAL_COVARIANCE,
    KALMAN_MEASUREMENT_NOISE_R,
    KALMAN_PROCESS_NOISE_Q,
)


class KalmanFilter:
    """1D constant-velocity Kalman filter.

    State vector: [angle, angular_velocity]
    Transition:   F = [[1, dt], [0, 1]]
    Observation:  H = [1, 0]  (observe angle only)

    PRD §5.6.

    Args:
        dt: Time step in seconds (default 1/30 for 30fps).
        process_noise: Q scalar — overrides config_prd default if provided.
        measurement_noise: R scalar — overrides config_prd default if provided.
        initial_covariance: P0 scalar — overrides config_prd default if provided.
    """

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        process_noise: Optional[float] = None,
        measurement_noise: Optional[float] = None,
        initial_covariance: Optional[float] = None,
    ) -> None:
        self._dt = dt
        self._q = process_noise if process_noise is not None else KALMAN_PROCESS_NOISE_Q
        self._r = measurement_noise if measurement_noise is not None else KALMAN_MEASUREMENT_NOISE_R
        self._p0 = initial_covariance if initial_covariance is not None else KALMAN_INITIAL_COVARIANCE

        # State transition matrix (constant velocity)
        self._F = np.array([[1.0, dt], [0.0, 1.0]])
        # Observation matrix (angle only)
        self._H = np.array([[1.0, 0.0]])
        # Process noise covariance
        self._Q = self._q * np.eye(2)
        # Measurement noise covariance (1x1)
        self._R = np.array([[self._r]])

        self._x: Optional[np.ndarray] = None  # State [angle, velocity]
        self._P: Optional[np.ndarray] = None  # Error covariance

    # ── Public interface ──────────────────────────────────────────────────────

    def update(self, measurement: Optional[float]) -> float:
        """Run one filter step.

        If not yet initialized and measurement is not None: initializes
        state to the measurement with zero velocity.

        If not yet initialized and measurement is None: returns 0.0
        (no state to predict from).

        If initialized and measurement is None: predict-only step —
        advances state without a correction, returns predicted angle.

        If initialized and measurement is provided: full predict + update.

        Args:
            measurement: Observed angle in degrees, or None for predict-only.

        Returns:
            Filtered angle in degrees.
        """
        if self._x is None:
            if measurement is None:
                return 0.0
            # First measurement — initialize state
            self._x = np.array([float(measurement), 0.0])
            self._P = self._p0 * np.eye(2)
            return float(self._x[0])

        # Predict step
        x_pred = self._F @ self._x
        P_pred = self._F @ self._P @ self._F.T + self._Q

        if measurement is None:
            # Predict-only — no correction
            self._x = x_pred
            self._P = P_pred
            return float(self._x[0])

        # Update step
        z = np.array([[float(measurement)]])
        y = z - self._H @ x_pred.reshape(2, 1)          # Innovation
        S = self._H @ P_pred @ self._H.T + self._R      # Innovation covariance
        K = P_pred @ self._H.T @ np.linalg.inv(S)       # Kalman gain
        self._x = x_pred + (K @ y).flatten()
        self._P = (np.eye(2) - K @ self._H) @ P_pred

        return float(self._x[0])

    def reset(self) -> None:
        """Reset filter state. Next update() will re-initialize from scratch.

        Call when face is lost, face has been absent > LSTM_RESET_ABSENT_FRAMES,
        or frame source restarts. PRD §5.6.
        """
        self._x = None
        self._P = None

    @property
    def current_value(self) -> float:
        """Current filtered angle estimate, or 0.0 if not initialized."""
        if self._x is None:
            return 0.0
        return float(self._x[0])

    @property
    def is_initialized(self) -> bool:
        """True if the filter has received at least one measurement."""
        return self._x is not None
