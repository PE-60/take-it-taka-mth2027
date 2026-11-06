# Referee flag signals (phase 1)

Stein-style **color histogram / HSV contour** detection of assistant referees and their flags on the **tactical** video, plus Law 6 heuristics from flag-vs-body geometry. Results overlay in `match_viewer`.

Match video is confidential and **not** in this repo. Use synthetic tests locally; run the CLI on your gated `data/tactical/` copy.

## Defaults (from your notes)

- AR kit: neon yellow shirts, black shorts
- Flag: yellow + orange panels
- One AR near top touchline, one near bottom; either may be cropped
- Tactical: 1920×1080 @ 25 fps

## Quick start

```bash
# unit tests (no video)
python -m referee_signals.tests.test_pipeline

# calibrate HSV on a real frame (needs video + display)
python -m referee_signals.calibrate_colors \
  --video data/tactical/BAR-ATH_tactical.mp4 --frame 5000 \
  --out export/seeds_tactical.json

# run detection on a window
python -m referee_signals \
  --video data/tactical/BAR-ATH_tactical.mp4 \
  --seeds export/seeds_tactical.json \
  --start 0 --end 2500 \
  --out export/referee_signals_tactical.json

# export / serve viewer (picks up export/referee_signals_tactical.json automatically)
python match_viewer.py --export --serve
# or: python match_viewer.py --referee-signals export/referee_signals_tactical.json --export
```

## JSON schema

See `schema.py`. Pixel-space `bbox_xyxy` / `flag_centroid`; absolute tactical frame index `f`.

## Phase 2 — hand pose

`pose.py` stubs MediaPipe-based Law 5 center-ref signals. Not required for phase 1.

## Phase 3 — broadcast

`broadcast.py` documents adapter hooks. Do not merge into tactical exports until tactical QA fails (`should_use_broadcast`).
