# Ball detection (experimental)

Kick-local visual ball re-detection and kit-aware touch attribution, independent of
`referee_signals`. Learns appearance from ATD + homography, then runs only around kick frames.

**Constraint:** gated match files under `data/` are optional. Unit tests are fully synthetic.

Goalkeeper ATD slots are **excluded** from kit learning, ROI seeds, and touch candidates
(`data_correction.goalkeeper_trackers.DEFAULT_TRACKERS` = `24`, `46`).

---

## Stages

| Stage | Status | What |
|---|---|---|
| **A** Appearance banks | **done** | Ball + outfield kit HSV histograms from clean ATD projections |
| **B** Kick-window ball detect | **done** | Motion/appearance blobs in a ROI around each kick |
| **C** Touch / team attribution | **done** | Score nearby outfield players (distance + kit) at contact |
| **D** Wire into possession | **done** | `lh_balldet` / `lh_balldet_gk_dead` in `possesion_algorithms` |

---

## Quick commands

```bash
# tests (no data/)
python -m ball_detection.tests.test_stage_a
python -m ball_detection.tests.test_stage_b
python -m ball_detection.tests.test_stage_c

# with data/
python -m ball_detection.build_banks
python -m ball_detection.run_detect --max-kicks 50
python -m ball_detection.run_attribute
python -m ball_detection.run_preview --max-kicks 10
```

Outputs under `ball_detection/export/`:

| File | Stage |
|---|---|
| `appearance_banks.json` | A |
| `kick_ball_tracks.json` | B |
| `touch_attributions.json` | C |
| `ball_detection.json` | B+C sidecar for the match viewer |
| `preview/kick_*.png` | QA stills |

`run_attribute` also copies the sidecar to `match_viewer/export/ball_detection.json`.

---

## Visualization

Three complementary views:

### 1. Match viewer (interactive)

After Stage C, open the tactical export as usual. The HTML loads `ball_detection.json` automatically.

- Layer **ball detection (kick windows)** — pixel ball blobs + contact marker + team label
- Candidates / step events → **ball detection** — jump between contact frames
- HUD shows team + confidence on the contact frame

```bash
python match_viewer/match_viewer.py --export --serve
# ensure ball_detection/export/ball_detection.json (or match_viewer/export/…) exists
```

### 2. Preview PNGs (fast QA)

```bash
python -m ball_detection.run_preview --max-kicks 20
# -> ball_detection/export/preview/kick_XXXXXX.png
```

Draws contact circle, team/confidence HUD, and top candidate dots.

### 3. OpenCV overlay helper

`ball_detection.overlay.draw_track_on_frame` / `draw_attribution` for custom scripts or a future OpenCV path in the viewer (same idea as referee overlays, but self-contained here).

**Recommended loop:** build banks → detect a short graded window → attribute → `run_preview` for a handful of kicks → only then scrub those frames in the match viewer.

---

## Stage C scoring

For each kick `contact` (pixel):

1. Unproject to pitch (when `H` is valid)
2. Consider **outfield** players within `max_dist_m` / `max_dist_px`
3. Score = distance proximity + kit histogram match to Stage A banks (if video frame available)
4. Emit `team` (ATD team of winner), `kit_vote`, `confidence`, top candidates

Fallback without banks/video: distance-only (`method=dist_only`).

---

## Package layout

```
ball_detection/
  geometry.py harvest.py appearance.py banks.py gk.py   # Stage A
  kicks.py roi.py detect.py                             # Stage B
  attribute.py                                          # Stage C
  results.py viewer_sidecar.py overlay.py               # IO + viz
  build_banks.py run_detect.py run_attribute.py run_preview.py
  tests/
  export/
```

---

## Tuning notes (shoes, continuity, RAM, fewer kicks)

```bash
# First 5 min of GT, memory-budgeted parallelism, every 2nd kick:
python -m ball_detection.build_banks --first 11185 --last 18684
python -m ball_detection.run_detect --first 11185 --last 18684 \
  --max-memory-mb 4096 --workers 4 --kick-stride 2 \
  --kick-threshold 6.0
```

| Flag | Default | Meaning |
|---|---|---|
| `--max-memory-mb` | `4096` | RAM budget for **concurrent** kick windows |
| `--workers` | auto | Requested parallel windows (always capped by the budget) |
| `--max-kicks` | all | Cap after stride (omit = process every selected kick) |
| `--kick-stride` | `1` | Keep every N-th detected kick only |
| `--kick-threshold` | `4.0` | ATD accel peak (m/s²); **higher → fewer kicks** |
| `--max-link-dist-px` | `28` | Hard continuity (no teleports) |
| `--foot-exclude-px` | `14` | Suppress shoe-like blobs near feet |
| `--min-bright-frac` | `0.30` | Require white/light ball pixels |

Rough window cost at 1920×1080 with `pre+post+1 = 21` frames ≈ **120–180 MiB** per in-flight window (with overhead). With `--max-memory-mb 4096` that is typically ~20–30 workers unless you lower `--workers`.

### `lh_balldet` — when a kick-team from ball_detection is used

Possession only adopts Stage C’s team when **nobody is inside `t_p`** at a kick **and** confidence clears the threshold:

```bash
python -m possesion_algorithms.run lh_balldet \
  --param bd_min_confidence=0.5 \
  --param attributions_json=ball_detection/export/touch_attributions.json

# stricter / looser matching of ATD kick frame ↔ Stage C kick_frame (frames):
python -m possesion_algorithms.run lh_balldet --param bd_match_tol=2
```

| Param | Default | Meaning |
|---|---|---|
| `bd_min_confidence` | `0.3` | Min Stage C confidence to trust the kick team |
| `bd_match_tol` | `3` | Max \|ATD kick − attribution kick_frame\| in frames |
| `attributions_json` | `ball_detection/export/touch_attributions.json` | Stage C output |
| `t_p` / `kick_win` / … | same as `lh_alt` | Base LH-alt behaviour |

---

## Stage D — possession algorithms

| name | behaviour |
|---|---|
| `lh_balldet` | LH-alt + if a kick has **no** player within `t_p`, adopt Stage C team (if confidence ≥ `bd_min_confidence`) |
| `lh_balldet_gk_dead` | same + corrected keepers + dead-ball mask |

```bash
python -m possesion_algorithms.run lh_balldet
python -m possesion_algorithms.run lh_balldet_gk_dead
# optional: --param attributions_json=ball_detection/export/touch_attributions.json
#           --param bd_min_confidence=0.3
```
