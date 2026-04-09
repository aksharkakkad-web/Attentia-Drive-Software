# STATUS — Desk MVP Phase Snapshot

**Phase:** Desk polish (Day 1 scaffolding complete)
**Target:** Crisp on MacBook webcam, portable to Raspberry Pi 5
**Last updated:** 2026-04-08

---

## Quick state

| | |
|---|---|
| **Current day in 7-day plan** | Day 1 complete (scaffolding landed) |
| **Portability interfaces created** | 4 / 4 (`FrameSource`, `AudioSink`, `ImuSource`, `ThermalMonitor`) |
| **Config profiles created** | 2 / 2 (`config/desk.yaml` populated, `config/pi5.yaml` empty placeholder) |
| **Async phone detector** | Not started (Day 2) |
| **Iris-based gaze** | Not started (Day 3–4) |
| **HUD polish** | Not started (Day 5) |
| **Audio cleanup** | Legacy `afplay` wrapped behind `AfplayAudioSink`; real rewrite deferred to Day 5 |
| **IMU stub integration** | Interface + `StubImuSource` wired into pipeline loop (read-and-discard; DEV-006) |
| **Per-stage timing instrumentation** | `StageTimer` logs `capture` / `imu` / `frame_total` p50/p95 every 2 s |
| **Runbook pass rate** | Unrun (target 10/10) |
| **Deviations logged** | 7 (all open — DEV-001..005 pre-existing, DEV-006 IMU wiring deferred, DEV-007 afplay wrapped) |
| **Existing tests passing** | 278 / 278 (was 285; 7 removed with deletion of `audio_alerter_v2.py` and its unit test file) |

---

## Measurable targets vs current

| Metric | Target | Current | Source |
|---|---|---|---|
| End-to-end frame latency (p95) | < 80 ms | **unmeasured** | Will come from per-stage timing instrumentation (Day 1) |
| Effective FPS | ≥ 25 | **unmeasured** | FPS counter already exists at `src/utils/fps_counter.py` but not reported in HUD |
| Alert-to-audio latency | < 200 ms | **unmeasured** | Subject to `afplay` subprocess jitter (50–200 ms) — will fix via `AudioSink` rewrite |
| Calibration duration | ≤ 3 s | currently 5 s (PRD value) | Will land in `config/desk.yaml` |
| Startup to first nominal | < 5 s | **unmeasured** | |

First full measurement run happens after Day 1 scaffolding lands (timing instrumentation is part of Day 1).

---

## Runbook pass matrix

All 10 steps currently **unrun**. Run via `docs/DESK_RUNBOOK.md` after each code change from Day 2 onward.

| # | Step | Status | Notes |
|---|---|---|---|
| 1 | Launch and calibration | — | Needs countdown + ready chime (Day 5 audio work, Day 5 HUD work) |
| 2 | Sit still, nominal baseline | — | Likely passes today; verify after Day 1 |
| 3 | Slow head turn, no sustained | — | Depends on latency (Day 2) and desk config (Day 1) |
| 4 | Sustained left look | — | Depends on desk config cooldown (Day 1) and tone mapping (Day 5) |
| 5 | Phone pickup | — | Depends on async phone detector (Day 2) for responsiveness |
| 6 | Eyes closed drowsiness | — | Depends on calibration quality; needs runbook attention |
| 7 | Cover camera (degraded) | — | Needs distinct tone D (Day 5) |
| 8 | Small head tilt (no-alert) | — | Depends on desk-mode road zone (Day 1) |
| 9 | **Glance with eyes only** | — | **Requires iris-based gaze (Day 3–4) — this is the primary gaze-decoupling test** |
| 10 | Feel test (subjective) | — | Final integration check — only meaningful after all other days land |

---

## Open deviations

See `docs/DEVIATIONS.md` for full entries.

| ID | Title | Status | Effort to port |
|---|---|---|---|
| DEV-001 | OpenCV VideoCapture used for camera input on Mac | open | ~2–3 hrs |
| DEV-002 | `afplay` subprocess for audio alerts on Mac | open | ~1–2 hrs |
| DEV-003 | No IMU consumer exists in the pipeline | open | ~4–6 hrs (basic), +1 day (fusion) |
| DEV-004 | No thermal monitor exists anywhere in the code | open | ~2–3 hrs |
| DEV-005 | `config_prd.py` tuned for car, not desk | open | ~1 hr |

**Total Pi migration effort currently estimated:** ~12–18 hours of focused work after Mac desk polish is complete, plus whatever new deviations accumulate during the polish phase.

---

## Critical code hotspots for the current phase

Files that will be touched during Days 1–7. Listed so we know what's in scope before we start.

| File | Day | Purpose |
|---|---|---|
| `src/main.py` | 1 | Add `--mode {desk,pi5}` CLI flag, route config loading |
| `src/config_loader.py` | 1 | Load YAML overlay on top of `config_prd.py` |
| `config/desk.yaml` (new) | 1 | Desk-mode threshold overrides |
| `config/pi5.yaml` (new) | 1 | Pi-mode hardware paths (real impls deferred) |
| `src/pipeline/interfaces/frame_source.py` (new) | 1 | `FrameSource` + `OpenCVFrameSource` + `FrameResult` |
| `src/pipeline/interfaces/audio_sink.py` (new) | 1 | `AudioSink` + `SoundDeviceAudioSink` stub |
| `src/pipeline/interfaces/imu_source.py` (new) | 1 | `ImuSource` + `StubImuSource` + `IMUReading` contract |
| `src/pipeline/interfaces/thermal_monitor.py` (new) | 1 | `ThermalMonitor` + `NoopThermalMonitor` |
| `src/contracts.py` | 1 | Add `IMUReading` dataclass |
| `src/pipeline/pipeline_manager_v2.py` | 1, 2 | Route through interfaces; add per-stage timing; async phone detector integration |
| `src/pipeline/frame_source.py` (existing) | 1 | Move `cv2.VideoCapture` into `OpenCVFrameSource` |
| `src/detection/phone_detector_yolo.py` | 2 | Wrapper or controller for async thread execution |
| `src/detection/face_detector.py` | 3–4 | Expose iris landmarks from MediaPipe output |
| `src/logic/signal_processor.py` | 3–4, 6 | Consume iris gaze fusion; consume `IMUReading` from stub |
| `src/output/display.py` | 5 | Full HUD rewrite: state badge, smoothed bars, countdown, FPS, bounding box |
| `src/output/audio_alerter.py`, `audio_alerter_v2.py` | 5 | Route through `AudioSink`; pre-loaded PCM; mutex; distinct tone per alert |

Files that will **not** be touched (locked for this phase):
- `src/logic/temporal_engine.py`
- `src/logic/scoring_engine.py`
- `src/logic/alert_state_machine.py`
- `src/logic/calibration.py` (configuration-only changes via desk.yaml)
- `src/logic/kalman_filter.py`
- `src/logic/blink_detector.py`
- `src/logic/perclos_calculator.py`
- `src/logic/duration_timer.py`
- `src/config_prd.py` (still the canonical PRD values)

If a change during this phase requires modifying one of the locked files, stop and escalate — it probably means the scope slipped.

---

## Active risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Existing tests break during interface scaffolding | Medium | High | Run full pytest suite after each file added |
| Async phone detector introduces race conditions | Medium | High | Single-writer / latest-result-wins pattern, not a queue |
| Iris gaze fusion feels worse than head-only gaze | Low | Medium | Keep head-only path behind config flag for rollback |
| Desk audio polish doesn't port cleanly to Pi | Low | Medium | `sounddevice` works on both platforms — use it for both from day one |
| New `IMUReading` contract breaks existing contract tests | Medium | Low | Add as optional field; default factory; existing tests unaffected |
| CLAUDE.md "no threading" rule contradicts Day 2 work | High | High (blocks Day 2) | **NEEDS RESOLUTION** — see CLAUDE.md Past Mistakes lines 74–76. Rules 14–17 proposal depends on this. |

---

## Notes

- This doc is short on purpose. It should fit in one glance. Update it at the end of every work session.
- "unmeasured" is not a placeholder — it's information. It means we don't know, and the instrumentation to know doesn't exist yet. That is itself a status.
- When a row changes from unmeasured to measured, record the date and the number. Historical values help spot regressions.
