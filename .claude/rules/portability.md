# Portability Rules

**Applies to:** All code in the desk polish phase and beyond. Enforced during review.

**Why this exists:** The development target is a MacBook webcam, but the production target is a Raspberry Pi 5 + OV5647 + MPU-6050 + MAX98357. Polish that works on Mac but cannot port to Pi is debt, not progress. These rules prevent that debt from accumulating.

**Related:** `docs/DESK_MVP.md`, `docs/HARDWARE_TARGET.md`, `docs/INTERFACES.md`, `docs/DEVIATIONS.md`

---

## The four interfaces

All hardware access goes through one of these four interfaces in `src/pipeline/interfaces/`:

1. **`FrameSource`** — camera input (`OpenCVFrameSource` on Mac, `Picamera2FrameSource` on Pi)
2. **`AudioSink`** — alert audio output (`SoundDeviceAudioSink` on both)
3. **`ImuSource`** — inertial data (`StubImuSource` on Mac, `Mpu6050ImuSource` on Pi)
4. **`ThermalMonitor`** — CPU temperature (`NoopThermalMonitor` on Mac, `VcgencmdThermalMonitor` on Pi)

Any code that needs camera, audio, IMU, or thermal data **must** consume it via the interface. Not via a direct library call. Not via a subprocess. Not via a file read. Via the interface.

Full contracts are in `docs/INTERFACES.md`.

---

## Banned patterns

The following patterns indicate an interface violation. If review finds them, the change is rejected until fixed or logged as a deviation with explicit justification.

### Camera
- ❌ `cv2.VideoCapture(...)` anywhere outside `OpenCVFrameSource`
- ❌ `picamera2.Picamera2(...)` anywhere outside `Picamera2FrameSource`
- ❌ Direct reads from `/dev/video*`
- ❌ `libcamera` bindings used outside an interface implementation
- ✅ `frame_source.get_frame()` — correct

### Audio
- ❌ `subprocess.run(["afplay", ...])` anywhere outside `AfplayAudioSink` (legacy, do not extend)
- ❌ `subprocess.run(["aplay", ...])` anywhere in logic or pipeline code
- ❌ `subprocess.run(["say", ...])`
- ❌ `pygame.mixer.*` outside an `AudioSink` implementation
- ❌ `simpleaudio.*` outside an `AudioSink` implementation
- ❌ Direct `sounddevice.play(...)` calls outside `SoundDeviceAudioSink`
- ❌ Hard-coded alert sound file paths in logic code
- ✅ `audio_sink.play(alert_type)` — correct

### IMU
- ❌ `import mpu6050` or `from mpu6050 import ...` outside `Mpu6050ImuSource`
- ❌ `smbus2.SMBus(...)` outside an IMU implementation
- ❌ `import board; import busio` (CircuitPython I2C) outside an IMU implementation
- ❌ Raw `/dev/i2c-1` reads
- ✅ `imu_source.read()` — correct

### Thermal
- ❌ `subprocess.run(["vcgencmd", ...])` outside `VcgencmdThermalMonitor`
- ❌ Direct reads from `/sys/class/thermal/thermal_zone*/temp` outside `VcgencmdThermalMonitor`
- ❌ `psutil.sensors_temperatures()` in logic code (if it gets added later, it goes behind the interface)
- ✅ `thermal_monitor.get_temp_c()` — correct

### Filesystem paths
- ❌ Any path literal starting with `/Users/` outside test fixtures
- ❌ Any path literal starting with `/Volumes/`
- ❌ Any path literal starting with `C:\\` or `D:\\`
- ❌ Hard-coded `/boot/firmware/config.txt` reads in application code
- ✅ Paths loaded from config (`config/desk.yaml` or `config/pi5.yaml`) or computed from `__file__`

### Platform detection
- ❌ `import platform; if platform.system() == "Darwin":` in logic or pipeline code
- ❌ `if sys.platform == "darwin":` in logic or pipeline code
- ❌ `os.uname()` branches in logic
- ✅ Platform differences are handled by choosing a different interface implementation via config, not by branching in shared code
- ⚠️ Interface implementations themselves may use platform detection internally (that's their whole job) — just not the callers

### Threading and process model
- ❌ `threading.Thread(...)` in logic code (threading lives in the pipeline and interface layers)
- ❌ `multiprocessing.Process(...)` anywhere in the MVP (overkill; worker threads are sufficient)
- ❌ `asyncio.*` in logic code (pipeline is synchronous except for specific worker threads)
- ✅ Worker threads for async inference (phone detector) owned by the pipeline layer
- ✅ Thread-safe latest-result-wins pattern for async results (not queues)

---

## Required practices

### Rule P-1 — All hardware access goes through an interface
If you need to read a camera frame, play a sound, read an IMU, or read CPU temperature, you call a method on one of the four interface objects. Period. No exceptions without a deviation entry.

### Rule P-2 — Log every Mac-specific decision
Any commit that introduces a Mac-specific shortcut, workaround, or design decision must add an entry to `docs/DEVIATIONS.md` in the same commit. The entry must include:
- What was decided
- Why (Mac constraint, laziness, deferred to Pi phase, etc.)
- How it's isolated (which interface)
- What the Pi migration task is
- Rough effort estimate

A commit that modifies hardware-adjacent code without updating `DEVIATIONS.md` is considered incomplete.

### Rule P-3 — Run the desk runbook before merging perception/signals/scoring/alert/output changes
Any change that could affect subjective feel must be validated via `docs/DESK_RUNBOOK.md`. Note any regressions in the commit message. If any step goes from PASS to FAIL, do not commit — fix first.

### Rule P-4 — Two-config discipline
No hardcoded thresholds outside `src/config_prd.py`, `config/desk.yaml`, and `config/pi5.yaml`. If you find yourself wanting to tweak a number at a call site, the correct answer is to add it to the config schema and load it.

### Rule P-5 — Tests must not require hardware
All tests in `tests/` must pass on a machine with no camera, no audio device, no IMU, no thermal sensor. Use `NullFrameSource`, `NullAudioSink`, `StubImuSource`, `NoopThermalMonitor` in tests. This is a pre-existing rule from `.claude/rules/testing.md`; this doc reinforces it for the interface layer.

### Rule P-6 — Logic layer is hardware-agnostic
Nothing in `src/logic/*.py` imports from `src/pipeline/interfaces/*`. Logic consumes data contracts (`PerceptionBundle`, `SignalFrame`, `TemporalFeatures`, `DistractionScore`, `IMUReading`). It does not consume interfaces. The pipeline layer is the glue that connects interfaces to logic.

### Rule P-7 — Interface implementations must fail safely
All four interfaces must handle hardware failure without raising into the pipeline loop. On failure, return a "valid=False" result, log once, and continue. The pipeline's existing `signals_valid` propagation handles the rest.

---

## When a deviation is acceptable

A deviation is acceptable when:
1. The Mac path is obviously better for development (e.g., using OpenCV for webcam since picamera2 doesn't exist on macOS)
2. The Pi migration is cheap and explicit (a few hours, fully documented)
3. The interface isolates the decision (no leakage into logic)
4. It's logged in `DEVIATIONS.md` with a migration task

A deviation is **not** acceptable when:
1. It leaks platform-specific code into logic layers
2. It blocks or makes harder any PRD-required feature
3. It's undocumented
4. It's a workaround for a test failure rather than a hardware difference
5. It introduces a "TODO: fix on Pi" comment in code — use `DEVIATIONS.md` instead, always

---

## Review checklist

Before merging any change touching hardware-adjacent code, confirm:

- [ ] No banned patterns present (search the diff)
- [ ] All new hardware access goes through an interface
- [ ] `DEVIATIONS.md` updated if any Mac-specific decision was made
- [ ] Runbook steps affected by the change still pass
- [ ] No hardcoded paths, platform checks, or library imports leaked into logic
- [ ] Tests run on a machine with no hardware (use stubs)
- [ ] `STATUS.md` updated if measurable targets changed

If any checkbox is unchecked, the change is not ready.
