# INTERFACES — Portability Boundary Contracts

**Purpose:** Defines the four interfaces that isolate the codebase from hardware specifics. Any code touching camera, audio, IMU, or thermal MUST go through one of these interfaces. No direct hardware calls in logic or pipeline code.

**Enforced by:** `.claude/rules/portability.md`

**Related:** `docs/HARDWARE_TARGET.md`, `docs/DEVIATIONS.md`, `docs/DESK_MVP.md`

---

## Why four interfaces

Four and only four things differ between a MacBook at Rishit's desk and a Raspberry Pi 5 in a vehicle:

1. **How frames enter the system** — webcam vs CSI camera
2. **How audio leaves the system** — CoreAudio vs I2S amplifier
3. **How inertial data enters the system** — absent vs MPU-6050 on I2C
4. **How thermal state is observed** — irrelevant vs critical on BCM2712

Everything else is pure computation and ports without change. If you find yourself writing a fifth boundary, stop and ask — something is probably wrong with the design.

---

## Interface 1: FrameSource

### Purpose
Abstracts the source of video frames. Replaces direct `cv2.VideoCapture` calls and direct `picamera2.Picamera2` calls.

### Contract

```python
class FrameSource:
    def open(self) -> bool:
        """Initialize hardware. Return True on success, False on failure.
        Failure must not raise. Log the reason."""

    def read(self) -> FrameResult:
        """Return the next frame. Must not block longer than ~1 frame period.
        If no frame is available, return FrameResult(ok=False).
        For video sources: sleeps to maintain original FPS timing.
        For webcam: drains one buffered frame before reading so the
        returned frame is always the freshest available."""

    def close(self) -> None:
        """Release hardware. Idempotent. Must not raise."""

    @property
    def fps(self) -> float:
        """Return target frame rate. Actual rate may differ."""
```

### FrameResult dataclass

```python
class ColorSpace(Enum):
    BGR = 'bgr'    # OpenCV native
    RGB = 'rgb'    # picamera2 native

@dataclass
class FrameResult:
    ok: bool                                    # False on end-of-source or transient failure
    frame: np.ndarray | None = None             # shape (H, W, 3), dtype uint8
    timestamp_ms: float = 0.0                   # wall-clock milliseconds (time.time() * 1000)
    color_space: ColorSpace = ColorSpace.BGR     # tells consumers which channel order they got
```

**Color space note:** Each implementation returns its backend's native format and sets `color_space` accordingly. `OpenCVFrameSource` returns `BGR`; a future `Picamera2FrameSource` returns `RGB`. Consumers must check `color_space` before converting — e.g., `FaceDetector` (MediaPipe, wants RGB) already does `cv2.cvtColor` on entry and should branch on this field when Pi support lands.

**Timestamp note:** `timestamp_ms` is wall-clock (`time.time() * 1000`). The pipeline currently ignores this field and stamps frames with its own `time.monotonic_ns()`. Future work may unify these, but Day 1 scaffolding preserves existing pipeline timing behavior exactly.

### Implementations

| Name | Platform | Backend | Notes |
|---|---|---|---|
| `OpenCVFrameSource` | Mac (and Linux desktop) | `cv2.VideoCapture` | Returns BGR. Handles device index and video file paths. Webcam mode drains one buffered frame per read to prevent stale-buffer lag. |
| `Picamera2FrameSource` | Raspberry Pi 5 | `picamera2.Picamera2` | Configures `main={"size": (W, H), "format": "RGB888"}`. Returns RGB. Downstream must handle both color spaces. |
| `NullFrameSource` | Any (test only) | always returns `ok=False` | For headless tests and CI. No frames, no hardware. |

### Rules
- Frames are numpy arrays. No references to `cv2.Mat` or `libcamera.Stream` escape this interface.
- If the camera fails mid-run, `read()` returns `FrameResult(ok=False)` and logs once. The pipeline handles the invalid frame gracefully via existing `signals_valid` propagation.
- Color space is NOT normalized at this layer. Each implementation returns its backend's native format. Consumers handle conversion.

---

## Interface 2: AudioSink

### Purpose
Abstracts alert playback. Replaces direct `afplay` subprocess calls, direct `aplay` calls, and direct `pygame.mixer` usage.

### Contract

```python
class AudioSink:
    def open(self) -> bool:
        """Initialize audio backend. Return True on success.
        Day 5 target: pre-load all alert tones into memory at open()."""

    def play(self, alert_type: AlertType, level: AlertLevel | None = None) -> None:
        """Play the tone for this alert type. Non-blocking.
        Must not raise, must not block the pipeline.
        Day 1 (AfplayAudioSink): spawns afplay subprocess on daemon thread.
        Day 5 target (SoundDeviceAudioSink): preempts lower-priority tones
        via mutex, pre-loaded PCM, returns within 5 ms."""

    def play_system(self, event: SystemEvent) -> None:
        """Play a system event sound (ready chime, calibration failed,
        degraded mode entered). Separate from alert tones.
        Day 1: no-op in AfplayAudioSink. Day 5: distinct chimes."""

    def close(self) -> None:
        """Release audio backend. Idempotent."""

    def is_available(self) -> bool:
        """Return True if audio backend is functional."""
```

### Tone mapping

| AlertType | Tone name | Character | Priority |
|---|---|---|---|
| `PHONE_USE` | Tone B | Urgent triple-beep, highest pitch | 1 (highest) |
| `DROWSINESS` | Tone C | Descending two-tone | 2 |
| `GAZE_OFF_ROAD` / `HEAD_INATTENTION` | Tone A | Low double-beep | 3 |
| `COMPOSITE` | Tone A | Same as gaze/head | 3 |
| `DEGRADED` / system-health | Tone D | Single long beep | 4 (lowest) |

System events get a distinct chime, not an alert tone:
- `CALIBRATION_READY` — ascending two-note chime
- `CALIBRATION_FAILED` — soft error tone
- `SYSTEM_ERROR` — triple low tone

### Implementations

| Name | Platform | Backend | Notes |
|---|---|---|---|
| `SoundDeviceAudioSink` | Mac + Pi | `sounddevice` with pre-loaded PCM | Same code runs on both. On Pi it writes to the ALSA device created by `dtoverlay=max98357a`. |
| `AfplayAudioSink` | Mac only | `subprocess.run(["afplay", ...])` | Legacy fallback. Do not use for new work — only kept for emergency rollback. |
| `NullAudioSink` | Any (test only) | logs events, makes no sound | For tests and `--no-audio` flag. |

### Rules (Day 5 target — not all met by Day 1 `AfplayAudioSink`)
- **Day 5:** All tones pre-loaded at `open()`. No subprocess launches at `play()` time.
- **Day 5:** A mutex prevents overlap. New tone of equal-or-higher priority preempts current; lower priority is dropped.
- **Day 5:** `play()` returns within 5 ms. Actual playback on a non-blocking audio callback.
- Tone priority maps directly to existing alert priority in `src/logic/alert_state_machine.py`. Phone is always #1 (matches P-01 rule).
- **Day 1 (current):** `AfplayAudioSink` spawns `afplay` subprocess per alert on a daemon thread. No mutex, no preemption, no pre-loaded tones. Acceptable for scaffolding; see DEV-007.

---

## Interface 3: ImuSource

### Purpose
Abstracts inertial measurement input. Exists on Mac only as a stub — but it must exist so that the `SignalProcessor` and related logic can consume IMU data from day one. This eliminates a whole class of retrofit work during Pi bring-up.

### Contract

```python
class ImuSource:
    def open(self) -> bool:
        """Initialize IMU. Return True on success.
        Mac stub always returns True. Pi must wake the MPU-6050
        from sleep (write 0x00 to register 0x6B) and verify
        WHO_AM_I returns 0x68."""

    def read(self) -> IMUReading:
        """Return the latest IMU reading. Must not block.
        On Mac stub, always returns IMUReading(valid=False).
        On Pi, returns the most recent accel + gyro + temp."""

    def close(self) -> None:
        """Release IMU. Idempotent."""

    def is_available(self) -> bool:
        """Return True if IMU is physically present and responding."""
```

### IMUReading dataclass

```python
@dataclass
class IMUReading:
    accel_x: float          # m/s², body frame
    accel_y: float
    accel_z: float
    gyro_x: float           # °/s, body frame
    gyro_y: float
    gyro_z: float
    temp_c: float           # onboard die temperature, for drift compensation
    timestamp_ms: float     # monotonic clock
    valid: bool             # False on Mac stub or I2C error
```

### Implementations

| Name | Platform | Backend | Notes |
|---|---|---|---|
| `StubImuSource` | Mac (and any platform without IMU) | none | `read()` always returns `IMUReading(valid=False)`. Pipeline behavior is identical to not having IMU logic at all. |
| `Mpu6050ImuSource` | Raspberry Pi 5 | `mpu6050-raspberrypi` library or raw `smbus2` | Wake from sleep on init. Configure ±4g accel, ±500°/s gyro. Read on demand from pipeline loop (no threading required for MVP). Handle I2C errors by returning `valid=False`, never raise. |

### Downstream consumers (future, Pi phase)

Once a real IMU is present, the following features become available. All are deferred but the interface must support them:

1. **Mount-drift correction** — integrate gyro over time to detect camera rotation since calibration. If drift > 3°, trigger soft recalibration.
2. **Vibration gating** — if accel magnitude exceeds a threshold (pothole, engine start), flag the frame's head-pose data as unreliable to suppress false alerts.
3. **Gravity vector fusion** — use the accel gravity vector as an absolute reference for pitch and roll, reducing PnP ambiguity and gimbal lock.
4. **Motion state estimation** — crude substitute for OBD/CAN speed. Stationary vehicle = relaxed thresholds; detected motion = tightened thresholds.

### Rules
- `SignalProcessor` checks `imu_reading.valid` before using IMU data. When `valid=False`, behavior is identical to today's head-pose-only pipeline.
- IMU failures mid-run are non-fatal. Return `valid=False` and continue.
- The contract is additive — adding IMU features does not change behavior for the no-IMU case.

---

## Interface 4: ThermalMonitor

### Purpose
Abstracts CPU temperature monitoring. Exists on Mac only as a no-op — but the interface must exist so downstream throttling logic doesn't branch on platform at call sites.

### Contract

```python
class ThermalMonitor:
    def open(self) -> bool:
        """Initialize thermal source. Return True on success."""

    def get_temp_c(self) -> float | None:
        """Return current CPU temperature in Celsius.
        Return None if unavailable (Mac no-op, or transient failure).
        Must be cheap to call — cache internally if backend is slow."""

    def close(self) -> None:
        """Release thermal source. Idempotent."""

    def is_available(self) -> bool:
        """Return True if thermal data can be read."""
```

### Implementations

| Name | Platform | Backend | Notes |
|---|---|---|---|
| `NoopThermalMonitor` | Mac | none | `get_temp_c()` always returns `None`. `is_available()` returns `False`. |
| `VcgencmdThermalMonitor` | Raspberry Pi 5 | `vcgencmd measure_temp` subprocess, cached 1 s | Parses `"temp=64.3'C"` format. Alternative backend: read `/sys/class/thermal/thermal_zone0/temp` (integer millidegrees C) — faster, preferred. |

### Downstream consumer (future, Pi phase)

A `ThermalThrottler` component in the pipeline checks `get_temp_c()` every N frames. If above `THERMAL_WARN_TEMP_C` (already defined in `config_prd.py:163`), drop YOLO input resolution to `YOLO_INPUT_RESOLUTION_THROTTLE`. If above `THERMAL_CRITICAL_TEMP_C`, frame-skip phone detection. The config values exist; only the wiring is missing.

### Rules
- `get_temp_c()` must be O(1) from the caller's perspective. If the backend is slow (subprocess), cache internally and refresh in the background or on a 1 s interval.
- `None` return is non-fatal. Throttling logic treats `None` as "assume safe temperature."
- No panic, no raise, no logging flood on repeated failures.

---

## Directory layout (when code scaffolding lands)

```
src/pipeline/interfaces/
    __init__.py
    frame_source.py         # FrameSource + FrameResult + implementations
    audio_sink.py           # AudioSink + AlertType mapping + implementations
    imu_source.py           # ImuSource + IMUReading + StubImuSource
    thermal_monitor.py      # ThermalMonitor + NoopThermalMonitor

src/contracts.py            # IMUReading added here, reachable from both interfaces and logic
```

The real Pi implementations live in the same files, gated by a runtime import check:

```python
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
```

This way the Mac development environment never needs Pi-only packages installed.

---

## Banned patterns (detected during review)

The following patterns indicate an interface violation:

- `cv2.VideoCapture(...)` anywhere outside `OpenCVFrameSource`
- `picamera2.Picamera2(...)` anywhere outside `Picamera2FrameSource`
- `subprocess.run(["afplay", ...])` anywhere outside `AfplayAudioSink`
- `subprocess.run(["aplay", ...])` anywhere in logic/pipeline code (use `AudioSink`)
- `pygame.mixer.*` anywhere outside an AudioSink implementation
- `import mpu6050` or `from mpu6050 import *` outside `Mpu6050ImuSource`
- `smbus2.SMBus(...)` outside `Mpu6050ImuSource`
- `vcgencmd` or reads from `/sys/class/thermal/*` outside `VcgencmdThermalMonitor`
- Any path starting with `/Users/`, `/Volumes/`, or `C:\\` outside test fixtures
- Any file in `src/logic/` importing from `src/pipeline/interfaces/` (logic is hardware-agnostic — interfaces are consumed by the pipeline layer only, not by logic)

Violations are grounds for rejecting a change. Every violation must either be fixed or logged as a deviation with an explicit reason.
