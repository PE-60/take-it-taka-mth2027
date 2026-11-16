# Match viewer

Tools for looking at the BAR-ATH match the way the data sees it: every tracked player, the
ball, the homography, Metrica's smart tags, the public ground truth and two possession
heuristics, on the tactical video or the broadcast video. Two outputs from one script:

- a rendered video with everything burned in (for a big screen), or
- a precomputed export plus an interactive browser viewer (for analysis and annotation).

```
match_viewer.py         the script: render, export, serve
match_viewer.html       the viewer, one self contained file
MATCH_VIEWER.md         this file
dev/                    only needed to change the viewer:
  viewer/               the module sources, viewer/STORIES.md has the architecture and stories
  tests/                unit tests (node --test) and the end to end script
  build_viewer.py       inlines the modules into match_viewer.html
```

Only `match_viewer.py` and `match_viewer.html` need to be next to each other. They can sit in
the FC Barcelona folder next to `data/`, or in a subfolder of it; the script looks in its own
folder and one level up. Run the script from the folder it is in.

## Setup

Python 3.9+ with `pip`. The script installs its own packages (opencv-python, pandas, numpy,
pillow) into your user profile on first run, no admin needed.

The data bundle sits next to the script in the repository's layout:

```
data/
  tactical/   BAR-ATH_tactical.mp4  _atd.xml  _atd.txt  _homography.csv  _smart_tagging.xml
  broadcast/  BAR-ATH_broadcast.mp4 _atd.xml  _atd.txt  _homography.csv
```

The public ground truth is found inside the extracted hackathon repository
(`.../evaluation/gt/BAR-ATH_possession_gt_public.csv`); any folder name works.
`MTAH_DATA_DIR` is honoured if you keep the data elsewhere.

## Commands

The two sources take the same commands. `--source tactical` is the default; `--mode gt`
(the public 5-minute window) exists only for the tactical source because the ground truth is
on that timeline.

| what | tactical | broadcast |
|---|---|---|
| interactive viewer, labelled window | `python match_viewer.py --mode gt --export --serve` | n/a |
| interactive viewer, whole match | `python match_viewer.py --mode full --export --serve` | `python match_viewer.py --source broadcast --mode full --export --serve` |
| reopen the viewer | `python match_viewer.py --serve` | same |
| rendered video, labelled window | `python match_viewer.py --mode gt` | n/a |
| rendered video, whole match | `python match_viewer.py --mode full` | `python match_viewer.py --source broadcast --mode full` |
| check parsing | `python match_viewer.py --inspect` | `python match_viewer.py --source broadcast --inspect` |

Outputs: `export/<source>_<mode>/` (data.bin + meta.json) and `<source>_overlay_<mode>.mp4`.
In the browser, switch exports with `match_viewer.html?d=export/tactical_full` etc.

The filtered ball: run `python -m ballq run --source all` first (see `ballq/README.md`); the export
then picks up `out/<source>/` on its own and the viewer shows the detector ball and the filtered
ball side by side, with the probability map painted on the pitch (darker = more likely) and a
legend entry for every mark and row. `--ballq DIR` points elsewhere, `--ballq none` leaves it out.
Our dead-ball intervals: `--dual` writes `export/dead_ball.json` itself
and the export adds the `DEAD OURS`, `SUBMISSION` and `SUB ≠ GT` rows and the submission score.

Modifiers for either source: `--pred file.csv` (tactical only, compares a submission),
`--start N --end M`, `--radius 1.0` (possession ring, metres), `--second-half-start N`
(broadcast only), `--port 8765`, `--layout overlay`, `--ui 0.6`, `--scale 0.5`, `--shape`,
`--trail 1.5`, `--no-players --no-ball --no-pitch --no-minimap --no-tags --no-timeline --no-regions`,
and explicit paths `--video --gt --atd --atd-meta --homography --tags --out`.

## The viewer

Layout: header (clock, play, speed, shortcuts), the video with ground-anchored marks, the
timeline, and a side panel.

### Video marks
Players as thin rings with slot numbers; the L&H holder gets a ground disc and its distance to
the ball; the possession ring is tinted by whoever is inside it; the closest opponent gets a
line to the ball; the ball is a translucent real-size disc with a fading trail and dotted
future path; kicks, teleports, vanishes and appearances flash and fade. Hover a player for
speeds and distance; click to pin (3 s trail, live readout); shift-click two for a distance.
`Z` opens a magnifier that follows the cursor. `L` loops the last two seconds.

### Timeline
Top: the overview of the whole export. GT is always first; `overview` picks what follows,
from strips (PRED, L&H, SEGMENT) and per-bin bands (restarts, kicks, ball lost, L&H share,
player count off, homography bad). Click it to jump. Below: row groups, filterable with `rows`:

- possession: GT, PRED, NAIVE, L&H (raw ball), L&H FILTERED, DISAGREE (green where only the
  filtered heuristic matches GT, red where only the raw one does), SEGMENT (my reading), MINE
  (derived label), SEGMENT B
- events: RESTART (set-piece windows by type, GT dead to alive ticks; glyphs above the rows),
  KICK, ANN EVENTS, ANN B, BOOKMARK
- data quality: BALL LOST, BALL SPEED, BALL AIR, BALL JUMP, PLAYER JUMP, DROPOUT, VANISH/APPEAR,
  COUNT, BOLT, HOMOGRAPHY, ANN DATA
- ball filter (when `ballq` output is in the export): BALL MOVED (metres between detector and
  filtered mean), BALL SD, TRUST (how little the detector was believed), BALL AIR FLAG,
  WHITE BLEND, HOM LOW, RESTART PRIOR
- slots: one thin row per track slot (click the header to expand, click a row label to pin)

Click to seek, drag to zoom, wheel to zoom at the cursor, Esc to reset. `,` and `.` step
through the event type chosen in the bar, including the candidate queue.

### Candidates and the queue
The Candidates list is model-driven candidate selection: frames where GT disagrees with the
L&H heuristic or with your prediction, where the raw and the filtered L&H disagree, where the
filter moved the ball more than 2 m, GT transitions, restarts, kicks, lost-ball stretches,
teleports, vanish/appear events, or ambiguous rings. Each candidate has a status:
`Enter` done, `Backspace` skipped, `!` flagged; each advances to the next unhandled one.
Progress is shown in the header of the panel and saved per export and list.

### Annotation
Everything is a fixed vocabulary with one key each.

- Possession reading (segment, runs until the next one you set): `H` `A` team; `1` controlled,
  `2` pass/flight, `3` loose; `D` or `4` dead; `X` or `0` clear. MINE is derived: team for
  alive states, D for dead, X when unset.
- Events (point at this frame): `P` pass, `S` shot, `T` tackle, `I` interception, `C` clearance,
  `J` deflection, `F` foul, `O` out, `K` restart taken.
- Data errors (intervals): `N` true ball position (placed at the tracked ball; arrows nudge
  10 cm, Shift 50 cm, click to move), `M` player missing (click, then `H`/`A`), `G` ghost,
  `W` wrong team, `V` identity swap (two clicks), `Y` lag, `U` homography off, `R` GT doubt,
  `Q` unclear. The key opens the interval; the same key or `Enter` closes it; `Esc` discards.
  Arming a tool pauses the video.

The note box attaches to the next annotation or bookmark. `B` bookmarks. `Ctrl+Z` undoes.
`?` lists every shortcut.

Exports: annotations as JSON (re-importable, schema included) or CSV with durations;
bookmarks CSV; and a "human reading" CSV (`frame, team, state, derived_label, annotator`).
Import a colleague's JSON as layer B to see their rows next to yours and read team, state and
event agreement.

### Rule 1
The challenge forbids hand labels and hand corrections on this game for the submission,
training or validation. The annotation layer exists to characterise where heuristics fail and
what the data gets wrong, for the technical document and for feature design, and to label
other games. The human-reading export is deliberately not in submission format.

## Development

```
node --test dev/tests/*.test.mjs                   # unit tests for the pure modules
python dev/build_viewer.py                         # rebuild match_viewer.html from dev/viewer/
python match_viewer.py --serve --port 8780 &       # then:
node dev/tests/e2e.mjs http://127.0.0.1:8780       # end to end stories (needs playwright)
```

Module boundaries are described in `dev/viewer/STORIES.md`. Pure modules never touch the DOM;
`dev/viewer/app.js` is the only file that does. Edit the sources and rebuild; the built html is
never edited by hand.
