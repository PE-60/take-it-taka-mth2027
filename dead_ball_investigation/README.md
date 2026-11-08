# Dead-ball intervals from the synchronised recordings

```
no view   = broadcast homography missing  ∩  broadcast player count ≠ 22
ball lost = tactical ball missing         ∩  broadcast ball missing
dead ball = no view  ∪  (the ball-lost intervals that overlap a no-view interval)
```

A ball-lost interval joins the dead ball whole when it shares at least one frame with a no-view
interval; a ball-lost interval on its own is not dead ball. `run.py --max-gap N` also accepts
ball-lost intervals that end or start within N frames of a no-view interval. `dead_ball.json` keeps
every ball-lost interval in `terms.ball_lost` and the accepted ones in `terms.ball_lost_kept`, so the
viewers' BALL LOST T∩B row shows all of them and DEAD BALL only the accepted ones.

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

The formula is evaluated on both timelines. On the tactical timeline the broadcast signals are pulled
across through the sync. On the broadcast timeline the tactical ball is pulled across. A frame with
no counterpart on the other recording takes "not missing" for that recording's signals, so missing
data is never dead-ball evidence on its own. `--no-counterpart-missing` flips this. The broadcast-only
"no view" term still applies to broadcast frames that have no tactical counterpart (pre-match,
halftime studio).

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

- a **dead ball (sync)** row group: `DEAD BALL` (the union, in the GT dead-ball grey),
  `NO VIEW (HOM ∅ ∩ ≠22)`, `BALL LOST T∩B` and `NO COUNTERPART`;
- a `DEAD BALL` overview strip, a "dead ball" row preset, "dead ball (derived)" in step events and
  in the candidate queue (interval starts, labelled by which term fired), and a line in the frame info panel.

The dual view (`match_dual.html`) shows the same rows for both panes, mapped through the live sync.
If you change the anchors there, the readout says the dead-ball rows were computed with other
anchors. Download the new `sync.json` and rerun the CLI.

## Use from code

```python
from dead_ball_investigation import SyncMap, SyncedMatch, annotate_dead_ball, load_source, load_tags

match = SyncedMatch(load_source("tactical"), load_source("broadcast"), SyncMap.from_json(), load_tags())
res = annotate_dead_ball(match)
res["tactical"].intervals()              # [[a, b], ...] tactical frames
res["tactical"].intervals("no_view")     # one term
match.timeline("tactical")               # every signal of both recordings on tactical frames
```

## Tests

```bash
python -m unittest dead_ball_investigation.tests.test_dead_ball
```

All synthetic: the formula's truth table, the sync map (including parity with `match_dual.html`,
run through `node` if it is installed), both timelines, the no-counterpart policy, and the
`match_viewer` loaders on generated ATD and homography files.
