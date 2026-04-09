# Known Issues — Attentia Drive MVP

Remaining limitations at MVP. None block the demo; all are documented so the
production rewrite knows what to fix. Listed in rough order of user impact.

## 1. MediaPipe head pose is unreliable on laptop webcams

**Symptom:** MediaPipe's `facial_transformation_matrixes` occasionally reports
yaw values of ±30–50° when the driver is looking straight ahead. Single-frame
yaw spikes are smoothed by the Kalman filter (`KALMAN_MEASUREMENT_NOISE_R=25`,
see `src/config_prd.py`) but sustained misreads still occur with oblique camera
angles and poor lighting.

**Cause:** MediaPipe's head-pose output uses an internal generic 3D face model
and assumes a canonical camera intrinsic. A real laptop webcam with unknown
focal length, unknown principal point, and an off-axis mount gives the solver
an ill-posed problem.

**Production fix:** RK3568 builds use IMX219 with fixed, known intrinsics
(`CAPTURE_WIDTH/HEIGHT`, calibrated principal point) and solve head pose via
`cv2.solvePnP` against a per-session face model. That path is sketched by
`PNP_REPROJECTION_ERR_MAX` in `config_prd.py` but not wired on MVP. Reverting
`ROAD_ZONE_YAW_MIN/MAX` from ±25° to ±15° is gated on this fix.

## 2. Mac pipeline runs at ~11 FPS, not 30

**Symptom:** Observed effective FPS on an M-series MacBook is 11–15, not the
30 FPS target. Duration timers (`T_GAZE_SECONDS=2.0s`) still fire correctly
because they use wall-clock seconds, but the visual response is less snappy
and PERCLOS has fewer samples per second than the PRD assumes.

**Cause:** Single-threaded pipeline (CLAUDE.md rule: no threads on MVP). Per
frame: MediaPipe FaceLandmarker (~35 ms), EfficientDet-Lite0 TFLite every
other frame (~25 ms when it runs), draw overlay, imshow. Biggest single cost
is MediaPipe CPU inference on the Mac.

**Production fix:** RK3568 runs RKNN-compiled models on the NPU (expected
~5× faster) and pipelines Layer 1 across two cores per the v2.0.0 PRD
(`FRAME_QUEUE_DEPTH=2`, `PHONE_THREAD_TIMEOUT_MS=5`). We intentionally did not
backport either optimization to MVP.

## 3. Iris-based gaze is a geometric proxy, not a real model

**Symptom:** Gaze world yaw/pitch is computed as
`corrected_head + iris_offset * IRIS_GAZE_*_SCALE_DEG`. This catches obvious
eye drift but is not a calibrated eyeball model: the scale constants are
ballpark values chosen from MediaPipe iris geometry, not measured per-driver.

**Cause:** The real gaze path in the PRD uses a learned
MobileNetV3+LSTM model (`GAZE_MODEL_PATH`, `GAZE_TEMPORAL_FRAMES=8`) that
ingests per-eye crops. That model is not built for MVP.

**Production fix:** Replace the iris-offset path in
`src/logic/signal_processor.py` with the LSTM gaze model. The `kf_gaze_yaw`/
`kf_gaze_pitch` filters are already instantiated and wired for this swap.

## 4. Kalman `dt` is hardcoded at construction

**Symptom:** All five Kalman filters are built with
`dt = 1.0 / CAPTURE_FPS = 1/30 s` and never updated. At the true ~11 FPS the
filter's predict step under-advances its internal state, which slightly biases
the velocity component of the constant-velocity model.

**Cause:** `SignalProcessor.__init__` takes `dt` once. The frame interval is
not measured on the hot path.

**Production fix:** Measure actual frame interval in `PipelineManager.process`
and either (a) pass it to each filter as a per-call `dt`, or (b) retune `Q`
to accept the variable-rate behavior. Low priority — bias is small and state
is bounded.

## 5. Phone detector wiring is hardcoded in `PipelineManager`

**Symptom:** The `PipelineManager._init_phone_detector` path hardcodes the
YOLO-vs-EfficientDet fallback and the `frame_skip=2` override, instead of
reading them from `config.yaml`. Changing phone-detector behavior requires a
code edit.

**Cause:** MVP convenience — we wanted to toggle things during debugging
without re-writing config.

**Production fix:** Move the fallback logic into a small factory that reads
`object_detector.model_path` and `object_detector.frame_skip` from config.

## 6. Calibration UX is silent on failure

**Symptom:** If calibration fails (face absent, pose not stable), the overlay
briefly shows `CAL FAIL` in the top-right during the first post-calibration
frame and then keeps running as if nothing happened. There is no retry button
and no prompt telling the driver what went wrong.

**Cause:** MVP has no interactive UI state beyond the OpenCV window.

**Production fix:** On-device builds have a dedicated LCD and buttons; the
UX layer there will gate pipeline activation on a successful calibration and
surface the failure reason from `CalibrationManager.failure_reason`.

## 7. `config.yaml: face_detector.refine_landmarks` is a dead letter

**Symptom:** The config flag exists but has no effect. MediaPipe's `tasks`
API `FaceLandmarker` does not support `refine_landmarks` (that was the old
`mp.solutions.face_mesh` API), and `face_detector.py` does not read the
flag either.

**Production fix:** Delete the field from `config.yaml` and
`FaceDetectorConfig`. Kept for now only to avoid churn before push.
