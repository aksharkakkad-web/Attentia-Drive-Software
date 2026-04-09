# DESK_MVP — Crisp on Mac, Portable to Pi 5

**Status:** Active
**Supersedes for current phase:** docs/PRD_v2.md (the PRD still governs the eventual product; this doc governs the current desk-polish phase only)
**Owner:** Rishit
**Created:** 2026-04-08

---

## Goal

The system feels responsive, accurate, and premium on a MacBook webcam within 3 seconds of launch — **and every change made here is portable to Raspberry Pi 5 + OV5647 + MPU-6050 + MAX98357 without rework.**

That second half is non-negotiable. Polish that doesn't port is debt, not progress.

---

## Why this doc exists

The PRD (`docs/PRD_v2.md`) is a car-production spec. It is correct for the eventual product but wrong for "does this feel good at my desk right now." Following the PRD literally produces an MVP that is technically compliant but subjectively dead: 8-second cooldowns, narrow road zone tuned for a dash mount, gaze-equals-head shortcut, minimal HUD, `afplay` subprocess audio.

This doc replaces the PRD **only for the current desk-polish phase**. When the system is crisp on Mac and the `DEVIATIONS.md` log is complete, we return to PRD-driven development for the Pi 5 bring-up phase.

---

## Measurable success criteria

All numbers measured on MacBook webcam with `python src/main.py --mode desk`:

| Metric | Target | Measurement method |
|---|---|---|
| End-to-end frame latency (capture → display) | **p95 < 80 ms** | Per-stage timing instrumentation in pipeline loop |
| Effective FPS | **≥ 25** | FPS counter, rolling 2s average |
| Alert-to-audio latency (trigger → sound) | **< 200 ms** | Timestamp delta from alert fire to `AudioSink.play()` return |
| Calibration duration | **≤ 3 s** | Startup timer |
| Startup to first nominal frame | **< 5 s** | From process launch to first `AlertState.NOMINAL` emission |
| Runbook pass rate | **10/10** | `docs/DESK_RUNBOOK.md` executed manually |
| Deviations logged | **100%** of Mac-specific decisions | Every commit touching hardware-adjacent code updates `DEVIATIONS.md` |

---

## Definition of "feels crisp"

Numerical targets above are necessary but not sufficient. The subjective bar:

1. Overlay tracks my actual movement with no visible lag
2. Gaze indicator follows my **eyes**, not just my head
3. Head pose angles match my actual head orientation — no weird offsets, no wobble
4. Each alert type has a distinct sound I can identify without looking
5. Audio never overlaps, never cuts off mid-tone, never has 50–200 ms jitter
6. Calibration has a visible countdown and a ready chime
7. When I do something non-distracted (glance at keyboard, check speedo equivalent), system stays silent
8. When I do something distracted, alert fires fast enough that I feel the cause-and-effect
9. HUD shows system state clearly: CALIBRATING / NOMINAL / PRE_ALERT / ALERTING / DEGRADED

If Rishit sits down, runs the command, and within 5 seconds thinks "yeah, that's a product" — we're done.

---

## Explicit non-goals (out of scope this phase)

These are deferred and must not consume time in this phase. Listed explicitly so we stop worrying about them:

- ❌ Night mode / IR handling — OV5647 has an IR-cut filter, hardware can't do it
- ❌ OBD / CAN / GPS speed gating — no vehicle interface on desk or Pi MVP
- ❌ VIN detection / multi-driver identity
- ❌ Thermal throttling implementation (interface stub only; real impl is Pi phase)
- ❌ Privacy / encryption of telemetry
- ❌ On-device bring-up (Pi 5 hardware phase)
- ❌ Model retraining, swapping, or quantization
- ❌ Any modification to layers 3–6 (scoring, alerts, temporal). They're correct. Leave them.
- ❌ Rewriting the PRD
- ❌ Refactoring working code that passes tests

---

## In-scope changes this phase

The only things that should change in code this phase:

1. **Portability scaffolding** — four interfaces in `src/pipeline/interfaces/`: `FrameSource`, `AudioSink`, `ImuSource`, `ThermalMonitor`. Mac implementations real, Pi implementations deferred, stubs where needed.
2. **Config profile split** — `config/desk.yaml` and `config/pi5.yaml` overlays on top of `src/config_prd.py`. `--mode` flag in `main.py`.
3. **Async phone detector** — YOLO in a worker thread, latest-result-wins, display never blocks on inference.
4. **Iris-based rough gaze** — consume MediaPipe iris landmarks (already in FaceLandmarker output), compute eye-center-relative offset, fuse with head pose in `signal_processor.py`. Keep head-only gaze path as fallback behind config flag.
5. **HUD polish** — rewrite `src/output/display.py`: state badge, smoothed bars, stable bounding box, FPS counter, calibration countdown.
6. **Audio cleanup** — `MacAudioSink` uses pre-loaded PCM via `sounddevice`, one distinct tone per alert type, mutex so no overlap, ready chime on calibration success.
7. **IMU stub integration** — `IMUReading` contract, `StubImuSource` (always returns `valid=False` on Mac), pipeline loop passes reading into `SignalProcessor`. Real `Mpu6050ImuSource` deferred to Pi phase.
8. **Per-stage timing instrumentation** — p50/p95 log every 2 seconds for capture / face / phone / signal / temporal / scoring / display stages.

Nothing else. If a change doesn't fall under one of these eight, it's out of scope.

---

## Desk-mode threshold deltas vs PRD

These land in `config/desk.yaml`. Values here are documentation — source of truth is the YAML file.

| Threshold | PRD value | Desk value | Reason |
|---|---|---|---|
| Road zone yaw | ±15° | ±25° | Laptop camera is ~50 cm from face, not 80 cm dash mount; wider zone matches actual geometry |
| Road zone pitch | −10° to +5° | −20° to +10° | Laptop camera is below eyeline, pitch baseline is offset |
| Alert cooldowns | 8 s | 2 s | Demo must feel reactive; car needs non-nagging |
| Calibration window | 5 s | 3 s | Shorter feedback loop for iteration |
| Phone alert cooldown | already immediate | unchanged | Phone priority P-01 preserved |
| EAR close threshold | 0.75 × baseline | 0.70 × baseline | Laptop lighting is typically worse than car interior, eyes appear more closed |

---

## Portability rule (hard constraint)

**No macOS-only APIs. No Apple-Silicon-specific threading. No file paths starting with `/Users/`, `/Volumes/`, or macOS-specific sysfs.** All hardware access goes through the four interfaces defined in `docs/INTERFACES.md`.

A change that works on Mac but cannot port to Raspberry Pi 5 is considered broken, even if all tests pass.

Every Mac-specific decision — even a good one — must be logged in `docs/DEVIATIONS.md` in the same commit that introduces it. No exceptions.

---

## Definition of done

The desk-polish phase is complete when **all** of these are true:

1. `python src/main.py --mode desk` launches and reaches nominal within 5 seconds
2. All 10 steps of `docs/DESK_RUNBOOK.md` pass on Rishit's MacBook
3. All measurable success criteria above are met, confirmed by instrumentation
4. `docs/DEVIATIONS.md` has a complete log of every Mac-specific decision with Pi migration tasks
5. `docs/STATUS.md` shows current latency numbers and runbook pass matrix
6. All existing tests still pass (`pytest tests/ -v`)
7. `git status` is clean; phase is committed

Only then do we start Pi 5 bring-up.

---

## Key references

- `docs/HARDWARE_TARGET.md` — production hardware spec (Pi 5 + peripherals)
- `docs/INTERFACES.md` — the four portability boundary contracts
- `docs/DEVIATIONS.md` — running log of Mac-specific decisions and Pi migration tasks
- `docs/DESK_RUNBOOK.md` — manual acceptance test
- `docs/STATUS.md` — current latency / pass matrix snapshot
- `docs/PRD_v2.md` — canonical product spec (still governs Pi 5 phase and beyond)
- `.claude/rules/portability.md` — enforcement rules for portable code
