# DEVIATIONS — Mac-specific decisions and Pi 5 migration log

**Purpose:** Every Mac-specific shortcut, workaround, or design decision that requires work to port to Raspberry Pi 5 is logged here. This is the running spec for the Pi bring-up phase.

**Rule:** Any commit touching hardware-adjacent code (camera, audio, sensor, thermal, filesystem paths) must update this file in the same commit. A commit that introduces a deviation without logging it is considered incomplete. See `.claude/rules/portability.md`.

**How to use this doc:**
- When Pi bring-up begins, execute these entries top-to-bottom. Each one becomes a task.
- When a deviation is fully migrated to Pi, mark status `resolved` and keep the entry for history.
- If a deviation turns out to be a non-issue on Pi, mark status `not-needed` with a reason.

---

## Entry template

```
## DEV-XXX: <short title>
- **Decision:** <what was done>
- **Reason:** <why this Mac-specific path was taken>
- **Portability:** <interface or abstraction that isolates the decision>
- **Pi migration task:** <concrete work to replace the Mac path with the Pi path>
- **Effort to port:** <rough hours>
- **Files touched:** <list>
- **Status:** open | in-progress | resolved | not-needed
- **Logged:** <date>
```

---

## Active deviations

### DEV-001: OpenCV VideoCapture used for camera input on Mac
- **Decision:** Mac uses `cv2.VideoCapture(0)` to read from the laptop webcam. Pi will use `picamera2` to read from the OV5647 via CSI.
- **Reason:** Mac has no CSI camera; OpenCV is the standard Mac webcam path. `picamera2` does not run on macOS.
- **Portability:** Will be abstracted behind `FrameSource` interface in `src/pipeline/interfaces/frame_source.py`. Mac implementation: `OpenCVFrameSource`. Pi implementation: `Picamera2FrameSource`.
- **Pi migration task:** Implement `Picamera2FrameSource` using `picamera2.Picamera2.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"})`. Verify frame format matches what `FaceDetector` expects (RGB vs BGR — MediaPipe wants RGB). Swap implementation via `config/pi5.yaml`.
- **Effort to port:** ~2–3 hours
- **Files touched:** `src/pipeline/frame_source.py` (existing), `src/pipeline/pipeline_manager_v2.py` (existing)
- **Status:** open (interface not yet created)
- **Logged:** 2026-04-08

---

### DEV-002: `afplay` subprocess for audio alerts on Mac
- **Decision:** Mac uses `subprocess.run(["afplay", path])` to play alert sounds. Pi will use ALSA `aplay -D hw:0,0` or `sounddevice` with pre-loaded PCM.
- **Reason:** `afplay` is the macOS built-in audio player; no extra dependencies on Mac. It is also erratic — subprocess launch has 50–200 ms jitter, tones can overlap, and there is no way to cancel an in-flight playback.
- **Portability:** Will be abstracted behind `AudioSink` interface in `src/pipeline/interfaces/audio_sink.py`. Both Mac and Pi implementations will use `sounddevice` with pre-loaded PCM for determinism. A mutex prevents overlapping playback. One distinct tone per alert type.
- **Pi migration task:** Verify `sounddevice` works with the MAX98357 I2S output (`dtoverlay=max98357a` creates an ALSA device the same library can target). Fallback: `subprocess.run(["aplay", "-D", "hw:0,0", path])`. Keep the interface identical; only the constructor differs.
- **Effort to port:** ~1–2 hours
- **Files touched:** `src/output/audio_alerter.py`, `src/output/audio_alerter_v2.py` (both existing)
- **Status:** open
- **Logged:** 2026-04-08

---

### DEV-003: No IMU consumer exists in the pipeline
- **Decision:** Mac has no IMU. The pipeline will gain an `ImuSource` interface with a `StubImuSource` implementation that always returns `valid=False`. On Pi, a real `Mpu6050ImuSource` will read from I2C address 0x68.
- **Reason:** IMU is a Pi-only hardware feature. But the pipeline must consume it from day one so that downstream logic (`SignalProcessor`, drift correction, vibration gating) is written with IMU awareness from the start. Otherwise the Pi phase requires retrofitting, which is exactly what this doc exists to prevent.
- **Portability:** New contract `IMUReading{ accel_xyz, gyro_xyz, temp_c, timestamp_ms, valid }` in `src/contracts.py`. New interface `ImuSource` in `src/pipeline/interfaces/imu_source.py`. `SignalProcessor` consumes the reading if `valid=True`, otherwise ignores it. Behavior on Mac with stub is identical to today's behavior.
- **Pi migration task:** Implement `Mpu6050ImuSource` using `mpu6050-raspberrypi`. On init, write 0x00 to register 0x6B (wake from sleep — this is mandatory every boot). Set accel range to ±4g, gyro range to ±500°/s. Handle I2C errors gracefully (return `valid=False` not raise). Expose temperature for drift compensation.
- **Additional Pi work:** Mount-drift correction using gyro integration (auto-recalibrate if camera rotated > 3° since startup). Vibration gating (if accel magnitude > threshold, mark signals invalid for that frame to suppress false head-pose alerts). Gravity vector fusion for absolute pitch/roll reference.
- **Effort to port:** ~4–6 hours for basic reader; ~1 day for fusion features
- **Files touched:** `src/contracts.py`, `src/pipeline/interfaces/imu_source.py` (new), `src/logic/signal_processor.py`, `src/pipeline/pipeline_manager_v2.py`
- **Status:** open
- **Logged:** 2026-04-08

---

### DEV-004: No thermal monitor exists anywhere in the code
- **Decision:** Mac has no thermal monitor and doesn't need one at desk. Pi 5 BCM2712 throttles around 80°C under sustained MediaPipe + YOLO load. We will add a `ThermalMonitor` interface now with a `NoopThermalMonitor` for Mac and defer the real `VcgencmdThermalMonitor` to Pi.
- **Reason:** Thermal handling is a Pi-only concern but the interface must exist so downstream code can query temperature without knowing the platform. Otherwise the pipeline gains a platform check at call sites, which violates portability.
- **Portability:** New interface `ThermalMonitor` in `src/pipeline/interfaces/thermal_monitor.py`. Method: `get_temp_c() -> float | None` (None = unavailable). Mac returns None. Pi reads `vcgencmd measure_temp` or `/sys/class/thermal/thermal_zone0/temp`.
- **Pi migration task:** Implement `VcgencmdThermalMonitor`. Subprocess call is ~5 ms; cache result for 1 s to avoid per-frame overhead. Downstream use: if temp > `THERMAL_WARN_TEMP_C` (from `config_prd.py`), drop YOLO input resolution to `YOLO_INPUT_RESOLUTION_THROTTLE`. If > `THERMAL_CRITICAL_TEMP_C`, frame-skip phone detection. Both thresholds already defined in `config_prd.py` lines 161–166 but never wired.
- **Effort to port:** ~2–3 hours for monitor + throttle integration
- **Files touched:** `src/pipeline/interfaces/thermal_monitor.py` (new), `src/pipeline/pipeline_manager_v2.py`, `src/detection/phone_detector_yolo.py`
- **Status:** open
- **Logged:** 2026-04-08

---

### DEV-005: `config_prd.py` values are tuned for car context, not desk
- **Decision:** All thresholds in `src/config_prd.py` assume a dash-mounted camera in a moving vehicle at ~80 cm from the driver. At a desk with a laptop webcam at ~50 cm and different mount angle, several of these values cause false positives or false negatives. We will add a `config/desk.yaml` overlay that relaxes specific thresholds for desk mode, selected via a `--mode` CLI flag.
- **Reason:** The PRD values are correct for the eventual product. They are wrong for a developer sitting at a desk trying to evaluate whether the system feels responsive. Two configs solve this cleanly; one codebase with profile selection.
- **Portability:** Both YAML files ship in the repo. `config/desk.yaml` is the Mac default. `config/pi5.yaml` inherits PRD values and adds Pi-specific paths (camera driver, audio device, IMU address, thermal sysfs). The `--mode` flag selects which overlay applies on top of `config_prd.py`.
- **Pi migration task:** Create `config/pi5.yaml` with Pi-specific hardware paths and PRD-matching thresholds. Verify all threshold values in PRD still make sense for Pi 5 + OV5647 at 80 cm dash mount. Any adjustments vs PRD become their own DEV entries.
- **Effort to port:** ~1 hour for the YAML file; adjustments discovered during bring-up logged separately.
- **Files touched:** `config/desk.yaml` (new), `config/pi5.yaml` (new), `src/config_loader.py` (existing), `src/main.py` (add `--mode` flag)
- **Status:** open
- **Logged:** 2026-04-08

---

### DEV-006: `ImuSource` exists but is not wired into `SignalProcessor`
- **Decision:** Day 1 scaffolding creates `ImuSource` / `StubImuSource` and the pipeline loop calls `imu_source.read()` every frame — but the returned `IMUReading` is discarded. `SignalProcessor` does not yet consume IMU data; vibration gating, gravity-vector fusion, and mount-drift correction are all deferred.
- **Reason:** On Mac the stub always returns `valid=False`, so consuming it in `SignalProcessor` would be dead code paths that we cannot meaningfully test until real hardware exists. Exercising the interface every frame (read-and-discard) is enough to catch API drift without forcing speculative logic now. The logic layer is locked during desk polish anyway (Rule 14 / `.claude/rules/portability.md`).
- **Portability:** Fully isolated — the interface is consumed, just not its output. When the real `Mpu6050ImuSource` lands, `SignalProcessor.process(bundle, imu_reading)` gets a second parameter and the pipeline passes it through. No call-site change outside `pipeline_manager_v2.py` and `signal_processor.py`.
- **Pi migration task:** Part of DEV-003. In addition to implementing `Mpu6050ImuSource`, extend `SignalProcessor.process()` to accept an `IMUReading` and implement vibration gating (drop frames when `|accel| > threshold`), mount-drift correction (auto-recalibrate neutral offsets if gyro-integrated rotation exceeds 3°), and gravity-vector fusion for absolute pitch/roll.
- **Effort to port:** Folded into DEV-003's estimate.
- **Files touched:** `src/pipeline/pipeline_manager_v2.py` (reads IMU), `src/logic/signal_processor.py` (future consumer).
- **Status:** open
- **Logged:** 2026-04-08

---

### DEV-007: `AfplayAudioSink` wraps legacy `afplay` subprocess instead of using `sounddevice`
- **Decision:** Day 1 of desk polish wraps the pre-existing `afplay` subprocess code behind an `AudioSink` interface (`AfplayAudioSink`) instead of immediately rewriting to `sounddevice` with pre-loaded PCM. Behavior is a pure refactor — same 3-beep URGENT / 1-beep HIGH pattern on a daemon thread, same Ping.aiff path. No mutex, no priority handling, no distinct per-alert tones.
- **Reason:** Day 1 is scaffolding, not behavior change. Wrapping first isolates every `afplay` call site behind the interface so Day 5 can swap implementations without touching the pipeline. Rewriting audio *and* introducing the interface in the same day would bundle a refactor with a behavior change, which makes regression diagnosis hard.
- **Portability:** Fully isolated — `AfplayAudioSink` is the only place in the codebase that may call `subprocess.run(["afplay", ...])` (enforced by `.claude/rules/portability.md`). Day 5's `SoundDeviceAudioSink` replaces the class entirely; no call-site change.
- **Pi migration task:** Superseded by DEV-002 — the Day 5 `SoundDeviceAudioSink` is the real portable implementation and works on both Mac and Pi.
- **Future state:** Desk MVP plan includes `PRE_ALERT_HOLD_S` (a short hold between `PRE_ALERT` and `ALERTING` so the audio stack has predictable scheduling headroom). Not yet wired — the alert state machine has no `PRE_ALERT` hold timer and `config_prd.py` has no such constant. Logged here so Day 5's audio rewrite can revisit whether it's still needed once `SoundDeviceAudioSink` has deterministic sub-10 ms launch latency. Do not add the constant until the supporting logic exists.
- **Effort to port:** ~0 hours — Day 5 (`SoundDeviceAudioSink`) is the migration path and it's already on the desk polish plan.
- **Files touched:** `src/pipeline/interfaces/audio_sink.py` (new), `src/pipeline/pipeline_manager_v2.py`, deletion of `src/output/audio_alerter_v2.py` and its test file.
- **Status:** open
- **Logged:** 2026-04-08

---

## Resolved deviations

*(none yet)*

---

## Not-needed deviations

*(none yet)*

---

## Notes for the Pi bring-up phase

When Pi bring-up begins:
1. Read every `open` entry above, top to bottom
2. Each one becomes a task in the Pi phase plan
3. Execute them in dependency order (interfaces first, then implementations)
4. Mark resolved as you go
5. Any new Pi-specific deviations discovered during bring-up get their own DEV-XXX entries
6. The Pi phase is complete when every entry is `resolved` or `not-needed` and `docs/DESK_RUNBOOK.md` passes 10/10 on actual hardware
