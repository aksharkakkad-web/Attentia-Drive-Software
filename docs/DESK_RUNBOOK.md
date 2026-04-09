# DESK_RUNBOOK — Manual Acceptance Test

**Purpose:** The scripted set of physical actions you perform at your desk to decide if the system feels crisp. This is the subjective half of the definition of done for `docs/DESK_MVP.md`.

**When to run:** Before committing any change that touches perception, signals, scoring, alerting, or output. After Day 1, run it every day. On the final day, all 10 steps must pass.

**Environment:**
- MacBook with built-in webcam
- `python src/main.py --mode desk` running
- Room with normal indoor lighting (not dim, not sun-glare)
- You sitting ~50 cm from the screen in a normal working posture
- Speakers/headphones enabled, system volume at 60%

**Total time:** ~5 minutes for a full run.

---

## How to score

Each step is **PASS / FAIL / PARTIAL**.

- **PASS** — system behaves exactly as described, feels natural, no weird lag or wrong tones
- **PARTIAL** — system does the right thing eventually but it feels off (lag, wrong angle, wrong tone, overlapping audio)
- **FAIL** — system does the wrong thing, crashes, or does nothing

**Definition of done:** all 10 steps PASS. A single PARTIAL or FAIL blocks the desk-polish phase from being called done.

Record results in `docs/STATUS.md` after each run.

---

## The 10 steps

### Step 1 — Launch and calibration
1. Close any other camera-using apps
2. Run `python src/main.py --mode desk`
3. Look straight ahead at the screen, keep your face centered, don't move, don't blink hard

**Expected:**
- Display window appears within 2 seconds
- HUD shows `CALIBRATING` badge with a visible countdown (3, 2, 1)
- Ready chime plays when calibration completes
- Badge switches to `NOMINAL`
- No errors, no crashes, no missing-model warnings

**Pass if:** The whole sequence feels like turning on a product. You know when to look, you know when it's ready, it sounds intentional.

**Fail if:** No countdown, no chime, you don't know when it's listening, weird delay between commands and effect.

---

### Step 2 — Sit still, nominal baseline
1. Keep looking at the screen
2. Don't move or do anything for 10 seconds

**Expected:**
- Badge stays `NOMINAL` the entire time
- HUD overlays (head pose bars, gaze indicator, EAR bar) show minimal movement
- No alerts fire
- FPS counter stable around 25+
- No audio plays

**Pass if:** The system is completely quiet and the HUD is stable. No jitter on the bars.

**Fail if:** Any spurious alert fires, bars jitter wildly even though you're still, FPS dips below 20.

---

### Step 3 — Slow head turn, no sustained look
1. Slowly turn your head left for ~1 second, then back to center
2. Slowly turn right for ~1 second, then back to center
3. Repeat both sides once more

**Expected:**
- Head pose overlay tracks your movement smoothly, no lag, no snap-back
- `PRE_ALERT` may briefly flash on extreme turns but clears immediately when you return
- No full alert fires (you never sustained long enough)
- No audio

**Pass if:** The overlay feels glued to your head — when you move, it moves in the same frame you see yourself move on screen.

**Fail if:** Visible lag between your head and the overlay, overlay snaps to positions instead of tracking, alert fires even though you never sustained.

---

### Step 4 — Sustained left look (the primary gaze test)
1. Turn your head fully left and **hold it** for ~3 seconds
2. Return to center

**Expected:**
- Within ~0.5 s: badge goes to `PRE_ALERT`
- Within ~2 s total: badge goes to `ALERTING`, **Tone A** (low double-beep) plays
- When you return to center: badge returns to `NOMINAL` after the desk-mode cooldown (~2 s)
- Overlay correctly shows your head yaw throughout

**Pass if:** You can predict when the alert will fire based on what your eyes are seeing in the overlay. Tone A is distinct and identifiable. The return to nominal feels responsive.

**Fail if:** Alert fires too early or too late, wrong tone plays, overlay shows wrong angle, cooldown takes forever (indicates desk config not loaded).

---

### Step 5 — Phone pickup (the priority test)
1. Have your phone in your pocket or off to the side
2. In one motion, pick it up and bring it in front of your face for ~1 second
3. Put it back down

**Expected:**
- Within ~0.5–1.0 s of the phone entering frame: **Tone B** (urgent triple-beep) plays
- Tone B must be **audibly distinct** from Tone A
- Phone alert fires **even if** a previous alert is in cooldown — phone overrides everything (P-01 rule)
- When phone leaves frame, badge returns to nominal

**Pass if:** You can tell by sound alone that this is a phone alert, not a gaze alert. The response feels fast enough to be useful in a real car.

**Fail if:** No alert fires, same tone as gaze, significant delay (>1.5 s), false positive on non-phone objects like your hand or a water bottle.

---

### Step 6 — Eyes closed (drowsiness)
1. Look straight ahead
2. Close your eyes and keep them closed for ~2.5 seconds
3. Open them

**Expected:**
- Within ~1.5–2 s: **Tone C** (descending two-tone) plays
- Tone C is distinct from A and B
- Badge shows `ALERTING` with drowsiness indicator
- When you open your eyes, badge returns to nominal

**Pass if:** The drowsiness detection triggers at the expected duration and you can identify the tone is about eye closure, not gaze.

**Fail if:** Doesn't trigger (bad EAR baseline from calibration), triggers immediately on blinks, wrong tone.

---

### Step 7 — Cover the camera (degraded mode)
1. Cover the webcam completely with your hand or a piece of paper for ~3 seconds
2. Uncover it

**Expected:**
- Within ~1 s of covering: badge goes to `DEGRADED`
- Within ~2 s: **Tone D** (single long beep) plays — distinct from A, B, C
- No other alerts fire while degraded (they are suppressed)
- When you uncover, system returns to `NOMINAL` (possibly re-calibrating if face was absent long enough)

**Pass if:** The system clearly communicates that it can't see and stops pretending to monitor. Tone D is distinguishable as a "system problem" sound, not a "driver problem" sound.

**Fail if:** System silently fails (pretends to work), keeps firing distraction alerts while blind, doesn't recover when uncovered.

---

### Step 8 — Small head tilt (no-alert boundary test)
1. Tilt your head slightly (maybe 5–10°) and hold for 3 seconds
2. Return to center

**Expected:**
- No alert fires
- Head pose overlay reflects the tilt accurately
- Badge stays `NOMINAL`

**Pass if:** The desk-mode road zone (±25°) is wide enough that natural small movements don't trigger false positives.

**Fail if:** Alert fires on normal movement — desk config isn't loaded, or road zone is still using car values.

---

### Step 9 — Glance with eyes only (THE gaze-decoupling test)
1. Keep your head **completely still**, facing the screen
2. Move only your eyes to look at your keyboard for ~3 seconds
3. Move your eyes back to the screen

**Expected:**
- If iris-based gaze is working: alert fires because your gaze is off-road (Tone A)
- If gaze is still head-only: NO alert fires because your head didn't move — **this is a FAIL**

**This is the single most important test for whether the iris gaze fix landed.** If gaze is still hardcoded to head pose (`signal_processor.py:200-201`), this step fails silently and the system feels "smart" for the wrong reasons.

**Pass if:** System detects eyes-off-road with head still. Tone A plays.

**Fail if:** System stays `NOMINAL` while you're clearly looking at the keyboard. Means gaze-head decoupling is not implemented (DESK_MVP Day 3–4 work not done).

---

### Step 10 — Feel test (subjective overall)
1. Do a mix of the previous steps — look around naturally for ~30 seconds like you're driving
2. Periodically do one of the distractions above

**Expected:**
- The system feels **alive** — HUD reacts, bars move, state badge updates
- Alerts fire when you expect them to
- Alerts do NOT fire when you're doing normal things
- Audio never overlaps
- Audio never gets cut off mid-tone
- You can always tell which alert is which just by the sound
- No lag between your actions and the display
- The whole thing feels like it could be in a car showroom

**Pass if:** If someone walked in and saw this running, they would believe it's a real product. Not a prototype. A product.

**Fail if:** Anywhere in this 30 seconds you think "yeah, that's broken" or "that felt slow" or "I can't tell what just beeped."

---

## After the run

1. Record each step's PASS / PARTIAL / FAIL in `docs/STATUS.md`
2. For any FAIL, write one line describing what you saw and which file likely needs the fix
3. For any PARTIAL, note what made it feel off
4. If all 10 PASS — you are done with desk polish, commit the phase, and begin Pi bring-up

---

## Notes

- **Lighting matters.** Run the runbook in the same lighting each time for consistent results.
- **Face distance matters.** ~50 cm from screen. Closer than 40 cm and the camera may lose focus (desk is at the near focus limit for the future OV5647 too).
- **Don't optimize the runbook.** If a step is too hard to pass, the answer is to fix the code, not to relax the step. The only exception is if a step is testing something explicitly out of scope per `DESK_MVP.md` non-goals.
- **This runbook transfers to Pi.** Every one of these 10 steps will be re-run on the Pi 5 during bring-up. The acceptance bar is identical. If the Mac version passes and the Pi version doesn't, the `DEVIATIONS.md` migration was incomplete.
