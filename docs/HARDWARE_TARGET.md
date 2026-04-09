# HARDWARE_TARGET — Production Hardware Reference

**Status:** Active — this is the inner contract for all portability work
**Target platform:** Raspberry Pi 5 (16 GB) + peripherals
**Used by:** `docs/DESK_MVP.md`, `docs/INTERFACES.md`, `docs/DEVIATIONS.md`

---

## Purpose of this doc

This is the production hardware the system will ultimately run on. **Any code added to this repo must be portable to this hardware without rework.** If a change cannot be portable, document the gap in `docs/DEVIATIONS.md` with a migration task and effort estimate.

The current development target is the MacBook webcam (see `docs/DESK_MVP.md`). But every decision made on Mac is evaluated against this doc. When Mac and Pi disagree, Pi wins — because Pi is where the product actually lives.

---

## HARDWARE DEVELOPER REFERENCE
### Raspberry Pi 5 — Peripheral Integration Guide
### Camera · Speaker · Amplifier · IMU
**Platform:** Raspberry Pi 5 (16 GB) | **Audience:** AI / App Developers | **OS:** Raspberry Pi OS 64-bit (Bookworm)

---

## 1. Platform — Raspberry Pi 5 (16 GB + Heatsink Fan)

### Overview
The Raspberry Pi 5 is the primary compute host. All peripherals connect through its GPIO, CSI, and I2S interfaces. The 16 GB RAM variant supports memory-intensive AI workloads (local inference, camera frame buffering, sensor fusion). The active heatsink/fan is critical — the BCM2712 throttles under sustained load without it.

### Key Specs
- **SoC:** BCM2712, Arm Cortex-A76 quad-core @ 2.4 GHz
- **RAM:** 16 GB LPDDR4X
- **GPIO:** 40-pin header — 3.3V logic ONLY (not 5V tolerant)
- **Camera:** 2× CSI/DSI MIPI connectors (22-pin, 4-lane each)
- **I2C:** i2c-1 (GPIO2/3), i2c-3, i2c-6, i2c-10
- **I2S/PCM:** GPIO18 (BCLK), GPIO19 (LRC/FS), GPIO20 (DOUT), GPIO21 (DIN)
- **USB:** 2× USB 3.0 + 2× USB 2.0
- **Power:** 5V/5A via USB-C (25W minimum)
- **Temp range:** 0°C – 85°C (throttles ~80°C without active cooling)

### Critical Constraints
- ALL GPIO pins are 3.3V. Applying 5V to any GPIO pin permanently damages the SoC
- Max current per GPIO pin: 16 mA (8 mA recommended)
- Total GPIO bank max: 50 mA
- Pi 5 uses the new RP1 I/O controller — older Pi 4 HATs and some drivers are NOT compatible
- Camera connector is 22-pin — older 15-pin OV5647 cables need a 15-to-22 pin FFC adapter
- I2S audio pins (GPIO18–21) conflict with PWM audio — do not load both overlays simultaneously

### OS Setup
```bash
sudo apt update && sudo apt full-upgrade -y
sudo raspi-config  # → Interface Options → I2C → Enable; do NOT enable Legacy Camera
sudo apt install -y python3-pip python3-smbus i2c-tools libcamera-apps alsa-utils
pip3 install picamera2 smbus2 mpu6050-raspberrypi pygame
```

### Thermal Monitoring
```bash
watch -n 1 vcgencmd measure_temp     # Real-time CPU temp
vcgencmd get_throttled               # 0x0 = healthy; bit 2 = throttled, bit 0 = undervoltage
```
Keep CPU below 75°C under sustained load. Fan spins up automatically around 60°C.

---

## 2. Camera — 5MP OV5647 / M12 FOV90 with IR Filter

### Overview
A 5MP camera based on the OmniVision OV5647 sensor with a fixed-focus M12 lens (90° FOV) and integrated IR-cut filter. Connects via the Pi 5's CSI ribbon connector. Controlled entirely through libcamera — do NOT use legacy raspicam drivers on Bookworm.

### Hardware Specs
- **Sensor:** OmniVision OV5647
- **Max still resolution:** 2592 × 1944 (5MP)
- **Video:** 1080p @ 30 fps, 720p @ 60 fps, 480p @ 90 fps
- **Lens:** Fixed M12, 90° horizontal FOV, fixed focus (~0.5 m to ∞)
- **IR Filter:** Built-in IR-cut filter (NOT night-vision capable)
- **Interface:** MIPI CSI-2, 2-lane, 22-pin ribbon (Pi 5 specific)
- **Pixel formats:** SBGGR10 (raw Bayer), YUYV, RGB888, MJPEG, H.264
- **Power:** 3.3V via CSI connector (no separate wiring)
- **Min illuminance:** ~1 lux

### Physical Connection
**NOTE:** Pi 5 uses a 22-pin CSI connector. Most OV5647 modules ship with a 15-pin cable. You need a 15-to-22 pin FFC adapter or Pi 5 replacement cable before connecting.

- Lift the plastic latch on CAM0 (nearest USB ports), insert cable (blue side facing away from board), press latch down
- CAM0 = camera index 0 by default; CAM1 requires additional dtoverlay config

### Verification & CLI
```bash
libcamera-hello --list-cameras       # Confirm camera detected
libcamera-hello -t 5000              # 5-second live preview
libcamera-still -o photo.jpg         # Full-res still capture
libcamera-still --width 1920 --height 1080 -o photo_1080p.jpg
libcamera-vid -t 10000 -o video.h264 # 10-second video
```

### Python (picamera2)
```python
from picamera2 import Picamera2

cam = Picamera2()

# Still capture
cam.configure(cam.create_still_configuration())
cam.start()
cam.capture_file("photo.jpg")
cam.stop()

# Video / AI frame pipeline
cam.configure(cam.create_video_configuration(
    main={"size": (1920, 1080), "format": "RGB888"}
))
cam.start()
frame = cam.capture_array()  # numpy array, shape (1080, 1920, 3), dtype uint8

# Runtime controls
cam.set_controls({
    "Brightness": 0.1,       # -1.0 to 1.0
    "Contrast": 1.2,
    "ExposureTime": 10000,   # microseconds
    "AnalogueGain": 4.0      # 1.0 to 16.0
})
cam.stop()
```

### Limitations
- Fixed focus — no autofocus; objects must be within the lens working range
- IR-cut filter blocks >~650 nm — IR LEDs invisible, no night vision
- Rolling shutter — fast motion causes skew/jello artifacts
- No hardware ISP — all post-processing is CPU (~15–25% at 1080p30)
- Max still res (2592×1944) not available in video mode
- No HDR support
- Significant noise below ~5 lux

---

## 3. Audio — MAX98357 I2S Amplifier + 3W/8Ω Speaker

### Overview
The MAX98357A Class D digital amplifier receives I2S audio directly from the Pi's GPIO header, converts it to analog, and drives the speaker. No separate DAC or analog audio jack needed. The speaker connects to the amp via JST-PH 2.0 connector.

### MAX98357A Specs
- **IC:** Maxim MAX98357A, Class D (filterless)
- **Interface:** I2S digital audio
- **Output:** 3.2W @ 8Ω with 5V supply
- **Efficiency:** ~92%
- **SNR:** 77 dBA | **THD+N:** 0.015% @ 1W
- **Sample rates:** 8 kHz – 96 kHz
- **Bit depth:** 16-bit and 32-bit
- **Supply:** 2.5V – 5.5V (use 5V rail for full power)
- **Gain:** 3, 6, 9, 12, or 15 dB — set via SD_MODE pin (hardware only, no software control)
- **Channel:** Mono output

### Speaker Specs
- **Power:** 3W continuous
- **Impedance:** 8Ω
- **Connector:** JST-PH 2.0 (2-pin, polarity-sensitive)
- **Frequency:** ~200 Hz – 20 kHz

### Wiring — MAX98357 to Pi 5

| Amp Pin | Pi Pin | Notes |
|---|---|---|
| VIN | Pin 2 or 4 (5V) | Must be 5V; lower voltage reduces max output |
| GND | Pin 6/9/14/25 (GND) | Common ground |
| LRC (FS) | GPIO19 — Pin 35 | I2S Frame Select |
| BCLK | GPIO18 — Pin 12 | I2S Bit Clock |
| DIN | GPIO21 — Pin 40 | I2S Data In |
| SD_MODE | Leave floating | Float = 9 dB gain, L+R mixed mono (default) |

### Gain Configuration (SD_MODE Pin)

| SD_MODE Connection | Gain | Channel |
|---|---|---|
| GND | 3 dB | Left only |
| Float / open (default) | 9 dB | Left + Right mixed |
| 100kΩ to GND | 6 dB | Left only |
| 100kΩ to 3.3V | 12 dB | Right only |
| 3.3V | 15 dB | Right only |

### OS Config (`/boot/firmware/config.txt`)
```bash
dtoverlay=max98357a
# Reboot after editing, then verify:
aplay -l                    # Should show I2S audio device
speaker-test -c 1 -t wav    # Test tone through speaker
amixer sset Master 80%      # Set volume
```

### Python Playback
```python
import subprocess

# Via aplay (most reliable)
subprocess.run(["aplay", "-D", "hw:0,0", "audio.wav"])

# Via pygame
import pygame
pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
pygame.mixer.music.load("audio.wav")
pygame.mixer.music.play()

# Volume control
subprocess.run(["amixer", "sset", "Master", "80%"])
```

### Limitations
- Mono only — for stereo you need two MAX98357 modules (one per channel)
- No software gain control — gain is hardware-only via SD_MODE pin strapping
- Filterless Class D output — keep speaker wire short (<20 cm) to reduce EMI
- JST-PH 2.0 is polarity-sensitive — verify +/− before connecting
- Do NOT probe speaker terminals with oscilloscope ground — output is BTL (bridged), not ground-referenced
- High gain (12–15 dB) with full software volume may cause clipping — tune software volume first
- GPIO18–21 are shared with PWM audio — do not load both dtoverlays

---

## 4. IMU — MPU-6050 6-Axis Accelerometer & Gyroscope

### Overview
The MPU-6050 is a 6-axis IMU combining a 3-axis accelerometer and 3-axis gyroscope in one package. It communicates over I2C and includes a built-in Digital Motion Processor (DMP) for onboard sensor fusion. One unit is installed in this build.

### Hardware Specs
- **Gyroscope range:** ±250, ±500, ±1000, ±2000 °/s (configurable)
- **Accelerometer range:** ±2g, ±4g, ±8g, ±16g (configurable)
- **ADC resolution:** 16-bit (both axes)
- **Interface:** I2C, up to 400 kHz (Fast Mode)
- **I2C address:** 0x68 (AD0 low, default) or 0x69 (AD0 high)
- **Supply:** Module accepts 3.3V or 5V (onboard regulator on most breakout boards)
- **Logic level:** 3.3V (I2C pins)
- **Temperature sensor:** Built-in ±1°C accuracy (use for drift compensation)
- **DMP:** Built-in Digital Motion Processor (quaternion output, tap detection, orientation)
- **Output data rate:** up to 8 kHz (gyro), 1 kHz (accel)
- **Current:** 3.9 mA active, 10 µA sleep
- **Interrupt pin:** INT (active high, configurable)

### Wiring — MPU-6050 to Pi 5

| Module Pin | Pi Pin | Notes |
|---|---|---|
| VCC | Pin 1 (3.3V) or Pin 2 (5V) | Module regulator handles either |
| GND | Pin 6 (GND) | Common ground |
| SDA | GPIO2 — Pin 3 | I2C1 data |
| SCL | GPIO3 — Pin 5 | I2C1 clock |
| AD0 | GND | Sets address to 0x68 (default). Tie to 3.3V for 0x69 |
| INT | Any GPIO (e.g. GPIO17) | Optional — for DMP/FIFO interrupt-driven reads |
| XDA / XCL | Not connected | Auxiliary I2C bus (not needed here) |

**NOTE:** Only ONE MPU-6050 is installed. A single I2C bus supports max 2 MPU-6050s (0x68 and 0x69). More units require an I2C multiplexer (e.g. TCA9548A) or a separate I2C bus.

### Verification & Wake
```bash
# Scan I2C bus — MPU-6050 should appear at address 0x68
sudo i2cdetect -y 1

# CRITICAL: MPU-6050 boots in SLEEP mode — must be woken on every power cycle
# Write 0x00 to register 0x6B (PWR_MGMT_1)
sudo i2cset -y 1 0x68 0x6B 0x00

# Confirm wake — read WHO_AM_I register (0x75), should return 0x68
sudo i2cget -y 1 0x68 0x75
```

### Python Integration
```python
from mpu6050 import mpu6050

sensor = mpu6050(0x68)  # use 0x69 if AD0 is pulled high

# Accelerometer — returns dict with x, y, z in m/s²
accel = sensor.get_accel_data()
print(f"Accel X: {accel['x']:.3f} m/s²")

# Gyroscope — returns dict with x, y, z in °/s
gyro = sensor.get_gyro_data()
print(f"Gyro Z: {gyro['z']:.3f} °/s")

# Onboard temperature (use for drift compensation)
temp = sensor.get_temp()
print(f"Temp: {temp:.1f} °C")

# Set accelerometer range
sensor.set_accel_range(mpu6050.ACCEL_RANGE_4G)   # ±4g

# Set gyroscope range
sensor.set_gyro_range(mpu6050.GYRO_RANGE_500DEG)  # ±500°/s
```

### Key Registers (for raw I2C access)

| Register | Address | Description |
|---|---|---|
| PWR_MGMT_1 | 0x6B | Write 0x00 to wake from sleep |
| ACCEL_CONFIG | 0x1C | Set accelerometer full-scale range |
| GYRO_CONFIG | 0x1B | Set gyroscope full-scale range |
| ACCEL_XOUT_H | 0x3B | Accel X high byte (X: 3B-3C, Y: 3D-3E, Z: 3F-40) |
| GYRO_XOUT_H | 0x43 | Gyro X high byte (X: 43-44, Y: 45-46, Z: 47-48) |
| TEMP_OUT_H | 0x41 | Temperature high byte |
| INT_ENABLE | 0x38 | Bit 0 = data ready interrupt enable |
| WHO_AM_I | 0x75 | Identity — always returns 0x68 |
| SMPLRT_DIV | 0x19 | Sample rate = 8 kHz / (1 + value) |
| CONFIG | 0x1A | DLPF (low-pass filter) config |

### Limitations
- Boots in sleep mode — must write 0x00 to register 0x6B on every startup
- Gyroscope has temperature-dependent drift — use onboard temp sensor for compensation in precision applications
- No magnetometer — cannot determine absolute heading/yaw without external compass (e.g. HMC5883L on auxiliary I2C)
- Raw 16-bit values must be divided by sensitivity scale factor to get physical units (libraries handle this)
- DMP firmware loading is complex — use library-based access unless you need gesture detection at very low latency
- Max I2C clock: 400 kHz — do not configure Pi I2C bus above this

---

## 5. System Integration

### GPIO Allocation Summary

| GPIO | Pin # | Peripheral | Function |
|---|---|---|---|
| GPIO2 | Pin 3 | MPU-6050 | I2C1 SDA |
| GPIO3 | Pin 5 | MPU-6050 | I2C1 SCL |
| GPIO18 | Pin 12 | MAX98357 | I2S Bit Clock (BCLK) |
| GPIO19 | Pin 35 | MAX98357 | I2S Frame Select (LRC) |
| GPIO21 | Pin 40 | MAX98357 | I2S Data In (DIN) |
| CSI0 | CSI Connector | OV5647 Camera | MIPI CSI-2 dedicated |
| 5V Rail | Pin 2/4 | MAX98357 | Amplifier power |
| 3.3V Rail | Pin 1 | MPU-6050 | IMU VCC |
| GND | Pin 6/9/14 | All peripherals | Common ground |

### `/boot/firmware/config.txt` (Combined)
```bash
# Camera (libcamera auto-detects OV5647 — no legacy overlay needed)
camera_auto_detect=1

# I2S amplifier
dtoverlay=max98357a

# I2C
dtparam=i2c_arm=on
dtparam=i2c_arm_baudrate=400000

# GPU memory (reduce if headless)
gpu_mem=128
```

### Power Budget

| Component | Typical Draw |
|---|---|
| Raspberry Pi 5 (idle) | ~600 mA @ 5V |
| Raspberry Pi 5 (full load) | ~2500 mA @ 5V |
| Heatsink fan | ~200 mA @ 5V |
| OV5647 camera | ~250 mA @ 5V |
| MAX98357 @ 1W audio | ~300 mA @ 5V |
| MPU-6050 | ~4 mA @ 3.3V |
| **Total worst case** | **~3.4 A** |

Use a genuine Raspberry Pi 27W USB-C supply (5.1V/5A). Underpowering causes random crashes, SD corruption, and brown-out resets.

### Recommended Startup Order
1. Boot Pi — firmware initializes CSI, I2C, I2S automatically
2. Wake MPU-6050 — write 0x00 to register 0x6B
3. Initialize Picamera2 — configure resolution and format
4. Initialize ALSA / pygame mixer
5. Start application loop

### Common Gotchas
- All GPIO is 3.3V — no 5V signals on data lines, ever
- Camera needs 22-pin FFC cable for Pi 5 — 15-pin won't fit without adapter
- MPU-6050 must be explicitly woken from sleep every boot
- MAX98357 I2S pins conflict with PWM audio — do not load both overlays
- Speaker JST-PH 2.0 is polarity-sensitive — reverse polarity causes phase issues (no damage to amp, but very quiet/distorted output)
- Keep I2S speaker wire short (<20 cm) to reduce EMI
- After any OS update, re-run raspi-config and verify interfaces are still enabled

---

## 6. Quick Reference

### Required Python Packages
```bash
pip3 install picamera2              # Camera control
pip3 install mpu6050-raspberrypi    # IMU
pip3 install pygame                 # Audio playback
pip3 install sounddevice numpy      # Low-level audio / numpy
pip3 install smbus2                 # Raw I2C
```

### Required System Packages
```bash
sudo apt install -y libcamera-apps i2c-tools python3-smbus alsa-utils python3-pygame
```

### Diagnostic One-Liners
```bash
libcamera-hello --list-cameras      # Camera detected?
sudo i2cdetect -y 1                 # I2C scan (expect 0x68 for IMU)
aplay -l                            # Audio devices listed?
vcgencmd measure_temp               # CPU temperature
vcgencmd get_throttled              # Throttle/undervoltage flags (0x0 = OK)
dmesg | grep -iE 'ov5647|i2s|mpu'   # Kernel messages for peripherals
speaker-test -c 1 -t wav            # Audio test tone
```

---

## Implications for code architecture

Summary of constraints this hardware places on the codebase:

1. **Camera access must be abstracted.** Mac uses OpenCV `VideoCapture`; Pi uses `picamera2`. Both go behind `FrameSource` interface. See `docs/INTERFACES.md`.
2. **Audio must be abstracted.** Mac uses `sounddevice` or `afplay`; Pi uses ALSA `aplay -D hw:0,0` or `pygame.mixer` with `dtoverlay=max98357a`. Both go behind `AudioSink` interface.
3. **IMU must be present in the contract from day one.** Mac has no IMU (`StubImuSource`, `valid=False`); Pi has `Mpu6050ImuSource`. Both implement `ImuSource` interface.
4. **Thermal monitoring must be in the contract.** Mac no-op; Pi reads `vcgencmd measure_temp` or `/sys/class/thermal/thermal_zone0/temp`. Both implement `ThermalMonitor` interface.
5. **No night mode, ever.** OV5647 has an IR-cut filter. Do not spend time on IR or low-light handling beyond graceful degradation.
6. **Fixed focus ~0.5 m to ∞.** Design for face distance ≥ 0.5 m. At a desk, the user's face may be at the near focus limit; expect some softness.
7. **Rolling shutter.** Fast head turns skew the image. IMU fusion helps compensate.
8. **90° FOV is wide.** Face occupies a smaller fraction of the frame than typical dash cams. Consider cropping to center ROI before running MediaPipe on Pi to save CPU.
9. **No threading restriction.** Pi 5 has 4 Cortex-A76 cores. Async inference on a worker thread is expected, not forbidden. Earlier single-thread-only guidance was RK3568-era and is superseded.
10. **Thermal throttling is real on Pi 5.** Sustained MediaPipe + YOLO will push the SoC toward 80°C without active cooling. Thermal monitor stub must exist on Mac so real impl drops in cleanly on Pi.
11. **No GPU.** VideoCore VII on Pi 5 is not used for ML inference. All inference is CPU (ONNX Runtime ARM64). Budget accordingly: YOLOv8-nano @ 320×320 ≈ 30–50 ms, MediaPipe FaceLandmarker ≈ 40–60 ms.
12. **Power is limited.** 27W USB-C supply; peripherals eat ~750 mA of that. Don't assume infinite compute headroom.

---

## Deviation log

All Mac-specific decisions that require Pi migration work are tracked in `docs/DEVIATIONS.md`. Every commit touching hardware-adjacent code must update that log.
