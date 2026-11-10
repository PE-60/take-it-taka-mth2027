# Dead-ball intervals from the synchronised recordings

```
no view   = broadcast homography missing  ∩  broadcast player count ≠ 22
ball lost = tactical ball missing         ∩  broadcast ball missing
baseline  = no view  ∪  (the ball-lost intervals that overlap a no-view interval)

then, in this order:
1. no counterpart   frames without a counterpart in the other video are dead
2. off pitch        the tactical ball off the pitch is dead; where the tactical ball is
                    missing, the broadcast ball off the pitch is dead
3. still ball       a gap between two dead intervals is dead when the ball stays within
                    1.5 m of where it is first seen in the gap, for the whole gap
```

**Baseline.** A ball-lost interval joins the dead ball whole when it shares at least one frame with
a no-view interval; a ball-lost interval on its own is not dead ball. `run.py --max-gap N` also
accepts ball-lost intervals that end or start within N frames of a no-view interval.

**Step 1** catches what the broadcast does not show: on the tactical timeline, before the broadcast
starts, halftime, and after it ends. "No counterpart" means the sync map gives none, or gives a
frame off the end of the other video; a counterpart that exists but is outside a partial export
does not count.

**Step 2** uses the viewer's BALL OFF PITCH row for each recording: the ball is tracked with x or
y outside [-0.002, 1.002] (fractions of the pitch), on a run of more than 1 s of consecutive frames;
a frame without the ball ends the run. On a frame where the tactical ball is tracked, the tactical
row decides, even when the broadcast says otherwise. Only where the tactical ball is missing does
the broadcast row count. Broadcast frames without a tactical counterpart use the broadcast row.
`--off-pitch-margin` and `--off-pitch-min-s` change the 0.002 and the 1 s.

**Step 3** runs on the dead intervals after steps 1 and 2 (so halftime, the match ends and the
ball out of play count as neighbours) and only between two of them, never before the first or after the last. The ball
position is the tactical ATD ball, in metres; the broadcast ball is used only for a gap in which
the tactical one is never seen. A gap in which neither sees the ball is left alone: there is no
evidence it was still. `--still-radius M` changes the 1.5 m.

The full method runs on the tactical timeline; broadcast frames with a tactical counterpart take its
result, so both timelines agree. `dead_ball.json` keeps every stage as a term: `no_view`,
`ball_lost` (all of them), `ball_lost_kept`, `baseline`, `no_counterpart` (step 1), `off_pitch`
(step 2) and `still_gap` (step 3). Leave out a step with `--skip-counterpart-step`,
`--skip-off-pitch-step` or `--skip-still-step`. The CLI prints precision, recall and IoU against the
public ground truth for the baseline and for the full method, and the step 3 gap counts (filled,
ball moving, ball never seen).

The two recordings are put on one timeline with the kick-off anchors from
`match_viewer/synchronize_videos.py` (`match_viewer/export/sync.json`). The mapping rule is the
same one as `SyncMap` in `match_dual.html` (a test checks this against the page's own code): from
an anchor up to the next one, `broadcast = tactical + offset`, and where a segment would run into
the next one the frames have no counterpart.

The three signals are gathered the way `match_viewer.py` gathers them, through its own loaders:

| signal | definition | viewer row |
|---|---|---|
| ball missing | `~isfinite(load_atd(...)["ball"][:, 0])` | BALL LOST |
| player count ≠ 22 | `load_atd(...)["count"] != 22` (track slots with a position) | COUNT ≠ 22 |
| homography missing | no CSV row, `valid == 0`, or a non-finite matrix | HOMOGRAPHY red, "homography none" |
| ball position (step 3) | `load_atd(...)["ball"][:, :2]` × pitch size (metres) | ball on the minimap |
| ball off pitch (step 2) | ball x or y ∉ [-0.002, 1.002] for > 1 s in a row | BALL OFF PITCH |

The baseline terms are evaluated on both timelines. On the tactical timeline the broadcast signals
are pulled across through the sync. On the broadcast timeline the tactical ball is pulled across. A
frame with no counterpart on the other recording takes "not missing" for that recording's signals,
so for the baseline missing data is never dead-ball evidence on its own (`--no-counterpart-missing`
flips this); step 1 is what makes a missing counterpart dead.

## Run

From the repo root:

```bash
python match_viewer/synchronize_videos.py                  # 1. -> match_viewer/export/sync.json
python match_viewer/match_viewer.py --dual --mode full --export --serve
                                                           # 2. export both, compute dead_ball.json, open the dual view
```

`--dual` computes `dead_ball.json` whenever it is missing or older than `sync.json`, either export
or the `dead_ball_investigation/*.py` files;
`--dead-ball` forces it. When the viewers have no dead-ball data they say so: the dual view shows
"no dead-ball rows" next to the sync offset (hover it for the reason), the normal view in its title line.

Or step by step:

```bash
python dead_ball_investigation/run.py                      # raw data files, parses both ATDs (minutes)
python dead_ball_investigation/run.py --from-exports       # reuse export/tactical_full + export/broadcast_full (seconds)
python match_viewer/match_viewer.py --dual --serve         # dual view (reuses exports, sync and dead_ball.json)
python match_viewer/match_viewer.py --source tactical --mode full --serve   # normal view
```

Every file in the package runs as a file from any directory (`python dead_ball_investigation/<file>.py`).
`python -m dead_ball_investigation` also works, but only from the repo root, because `-m` finds
packages through the current directory. Internal imports are absolute (`dead_ball_investigation.sync`);
do not turn them into bare `from sync import ...`, which breaks the package import used by the tests
and by code that does `import dead_ball_investigation`.

### Copying into another checkout

The folder must sit at the repo root, next to `match_viewer/` and `games.py`; the paths to the data,
the exports and `sync.json` are found from there. Copy the whole folder (replace any older copy, and
delete its `__pycache__/`), together with `match_viewer/match_viewer.py`, `match_viewer/match_viewer.html`
and `match_viewer/match_dual.html`, which hold the `--dead-ball` flag and the dead-ball rows.

The output is `match_viewer/export/dead_ball.json`: `[first, last]` inclusive video-frame intervals
on each timeline for the union and each term, the frames that have a counterpart, the sync anchors
used, per-timeline stats, and the Smart Tagging instances with their spans mapped onto the broadcast.
The CLI also prints precision and recall of the derived dead ball against `D` in the public
5-minute ground truth.

## In the viewers

`match_viewer.html` loads `export/dead_ball.json` on start. With it, you get:

- a **dead ball (sync)** row group: `DEAD BALL` (the full method, in the GT dead-ball grey),
  `DEAD (BASELINE)`, `NO VIEW (HOM ∅ ∩ ≠22)`, `BALL LOST T∩B` (every ball-lost interval),
  `NO COUNTERPART` (step 1), `OFF PITCH T (ELSE B)` (step 2) and `STILL-BALL GAP` (step 3);
- a `DEAD BALL` overview strip, a "dead ball" row preset, "dead ball (derived)" in step events and
  in the candidate queue (interval starts, labelled by the terms that make them), and a line in the
  frame info panel saying why a frame is dead.

The dual view (`match_dual.html`) shows the same rows for both panes, mapped through the live sync.
If you change the anchors there, the readout says the dead-ball rows were computed with other
anchors. Download the new `sync.json` and rerun the CLI.

## Use from code

```python
from dead_ball_investigation import Params, SyncMap, SyncedMatch, annotate_dead_ball, load_source, load_tags

match = SyncedMatch(load_source("tactical"), load_source("broadcast"), SyncMap.from_json(), load_tags())
res = annotate_dead_ball(match)          # full method; Params(...) as a second argument changes it
res["tactical"].intervals()              # [[a, b], ...] tactical frames
res["tactical"].intervals("still_gap")   # one term
res["tactical"].gaps                     # step 3 counts
annotate_dead_ball(match, Params(counterpart_step=False, off_pitch_step=False, still_step=False))   # baseline only
match.timeline("tactical")               # every signal of both recordings on tactical frames
```

## Tests

```bash
python -m unittest dead_ball_investigation.tests.test_dead_ball
```

All synthetic: the baseline's truth table and overlap rule, all three steps (halftime cut; the
off-pitch margin, the 1 s minimum and the tactical-else-broadcast fallback; still, moving and unseen
gaps, the radius), the sync map (including parity with `match_dual.html`, run through
`node` if it is installed), agreement of both timelines, partial exports, and the `match_viewer`
loaders on generated ATD and homography files.
