# The control model: ball position and possession from tracking

Run with `python -m ballq submit`. Code in `ballq/control.py`, parameters under `control:` in
`ballq/config.yaml`.

## 1. Problem

The tracking data gives, 25 times a second, a position for every player and one for the ball. The
ball position is the weak part:

- it is missing when the detector loses the ball (behind a player, against a white shirt);
- it is wrong while the ball is in the air, because a camera cannot tell height: the ball is drawn
  where the line of sight meets the grass, which is beyond the ball;
- it sometimes jumps to something that is not the ball.

We want two things per frame that other work can rely on: where the ball is and how sure we are,
and which team has it.

## 2. Definitions

**Possession zone.** A circle of 1 m around a player. The ball is *at* a player when it is inside
his zone and he is the nearest player. (Vidal-Codina et al.; Link and Hoernig call it the sphere
of influence.)

**Visit.** A stretch of consecutive frames in which the ball is at the same player.

**Touch.** A visit during which the player acted on the ball. We cannot see feet, so we infer it
from what the ball did. A visit is a touch if either holds:

- the ball's velocity changed across the visit by at least 3 m/s (it was struck, stopped or
  deflected; this is Link and Hoernig's "ball action");
- the ball stayed at the player for at least 0.5 s (he is carrying it).

A visit with neither is the ball **passing by**: a pass that rolls past an opponent's feet, or a
high ball whose projection crosses a player. It is ignored. A pass at 12 m/s crosses a 2 m zone in
0.17 s without changing velocity, so it fails both tests by a wide margin.

**State.** At every frame the ball is **dead** (play stopped), **held** (inside a touch) or
**free** (rolling or flying between touches).

**Possession.** The team of the last player who touched the ball. It does not change while the
ball is free.

The one principle behind all of this, from Maksai et al.: *the state of the ball can only change
when a player interacts with it.* Nothing that happens to the ball between two touches, real or
measured, can move possession.

## 3. Method

**Step 1. Clean the ball track.** Remove points the ball could not have reached from the previous
kept point at 60 m/s. That is faster than any real ball, so this only removes the detector jumping
to something else. Then smooth each seen stretch (Savitzky-Golay, 7 frames, as Vidal-Codina et al.).

**Step 2. Find the visits.** For each frame take the nearest player; if the ball is within 1 m, the
ball is at him. Join consecutive frames at the same player into a visit (a gap of up to 3 frames is
allowed).

**Step 3. Decide touch or pass-by.** For each visit measure

- `dv`: the ball's velocity after the visit minus its velocity before it, in m/s. Before and after
  are taken from the nearest seen frames within a second, so a ball that disappears into a player
  and comes out in another direction still shows its change;
- `duration`: the length of the visit.

Turn them into one confidence `p` between 0 and 1 that the player acted: `p` is 0.5 when `dv` is
exactly 3 m/s or the duration exactly 0.5 s, grows toward 1 beyond that, and both kinds of evidence
add up. If an opponent is also within 1 m of the ball (a duel), `p` is multiplied by 0.7. Visits
with `p` below 0.5 are pass-bys.

**Step 4. Assign states and possession.** Walk through the touches in time order.

- Inside your dead-ball intervals the state is dead and nobody has the ball.
- A touch by the other team is a **gain**: possession changes on the first frame of that touch.
- A touch by the same team is a **keep**: possession stays and its confidence rises.
- **After a dead ball** the player already at the ball when play resumes (within 1 s) is the
  **taker**. He does not count, however often he touches it. The first touch by anyone else takes
  possession, and takes it from the restart frame on, not from the moment of the touch.
- While the ball is not seen it keeps the state it had: a ball that vanished at a player's feet is
  still held by him.

**Step 5. Correct a ball in the air.** While the ball is in the air its drawn position drifts away
from the camera and drifts back as it lands. The drift is smooth, so no speed test can catch it.
But physics gives the answer directly: a ball in flight moves in a vertical plane at steady
horizontal speed, so over the ground it travels the straight line from where it was struck to
where it is next touched, at an even pace. So between two touches, if the seen track bows more
than 1.5 m away from that straight line (over a distance of at least 8 m), the ball was in the air
and we use the line instead.

**Step 6. Write the ball position** with a spread in metres and where it came from:

| source | position | spread |
|---|---|---|
| `seen` | the cleaned detector point | 0.3 m, more where the homography is poor |
| `flight` | the straight line between the two touches | 1.5 m |
| `held` | the holder's position, when the ball is hidden at him | 1.0 m |
| `interp` | a straight line between two sightings of a free ball, up to 3 s apart | 0.5 m at the ends, growing to at most 15 m in the middle |
| `unknown` | none | none |

## 4. Worked example

Home player 7 has the ball at his feet for two seconds, plays a 40 m pass along the ground past
away player 21, and home player 9 controls it.

| what happens | visit | dv | duration | p | result |
|---|---|---|---|---|---|
| 7 dribbles, then strikes | at 7 | 12 m/s | 2.0 s | 1.00 | touch, **gain** for home |
| ball rolls past 21, 0.6 m from him | at 21 | 0.1 m/s | 0.16 s | 0.20 | **passed by**, ignored |
| 9 stops the ball | at 9 | 12 m/s | 1.5 s | 1.00 | touch, **keep** for home |

Label for every frame: H. If instead 21 had stopped the ball, his visit would have `dv` 12 m/s and
become a gain for away on the frame the ball reached him.

Now the same pass lofted: the detector's track swings 6 m to the side in mid-flight and returns.
It no longer passes near 21 at all, the two touches are the same, and `ball_track` reports the
straight line from 7 to 9 with `source = flight`, keeping the detector's point in `raw_x`, `raw_y`.

## 5. Inputs

| input | file | what is read | used for |
|---|---|---|---|
| tracking | `*_atd.txt`, `*_atd.xml` | x, y of every player slot and of the ball per frame; each slot's team | everything |
| homography | `*_homography.csv` | reprojection error per frame | the spread of a seen ball |
| dead ball | `dead_ball.json` | `timelines.<source>.dead_ball`: pairs `[first, last]` of frames | when play is stopped; taken as given |
| settings | `ballq/config.yaml`, `control:` | radii and thresholds named above | every threshold |

Not used: smart tags (the restart type is unreliable there), shirt colours, the video. The ground
truth is read only to print a score next to the result.

## 6. Outputs

In `out/<source>/`, as plain CSV files (`possession_frames.csv` and so on). Read any of them with
pandas, Excel, or `ballq.contract.read_table("out/tactical/<name>")`.

### `possession_frames`: who has the ball. One row per frame.

| column | values | meaning |
|---|---|---|
| `frame` | integer | video frame |
| `state` | `dead`, `held`, `free` | what the ball is doing |
| `holder_slot` | shirt number, or empty | the player holding it, when `held` |
| `team` | `H`, `A`, or empty | team in possession; kept while the ball is free |
| `p_team` | 0 to 1 | confidence in `team`: the `p` of the gaining touch, raised by each later touch of the same team |
| `p_home`, `p_away`, `p_dead` | 0 to 1 | the same as three probabilities, summing to 1 wherever a team is known |
| `event` | `restart`, `gain`, or empty | marked on the frame it happens |
| `restart_zone` | `corner`, `touchline`, `goal_area`, `centre`, `other`, `unseen` | where the ball is when play resumes, from its position only |
| `label` | `H`, `A`, `D`, `X` | the submission label for this frame |

### `ball_track`: where the ball is. One row per frame.

| column | values | meaning |
|---|---|---|
| `frame` | integer | video frame |
| `x`, `y` | 0 to 1 | best position in pitch units, as in the ATD; empty when unknown |
| `sd_m` | metres | spread around that position (table in step 6) |
| `source` | `seen`, `flight`, `held`, `interp`, `unknown` | how the position was obtained |
| `raw_x`, `raw_y` | 0 to 1 | the detector's point, untouched |

### `touches`: the evidence. One row per visit, pass-bys included.

| column | values | meaning |
|---|---|---|
| `frame_start`, `frame_end` | integers | the visit |
| `slot`, `team` | shirt number, `H` or `A` | the player |
| `duration_s` | seconds | how long the ball was at him |
| `dv_ms` | m/s | change of the ball's velocity across the visit |
| `duel` | true or false | an opponent was within 1 m of the ball |
| `p` | 0 to 1 | confidence that he acted on the ball |
| `kind` | `gain`, `keep`, `taker`, `passed_by` | what the visit became |

### `control_manifest.json`: what this run was. One per run.

Model and version, the settings used and their hash, input file names and sizes, frame range, row
counts, the column list of each table, and a summary: label and state counts, touch kinds, share
of frames by ball source, number of gains, restarts by zone. Read it first.

### `out/submission_control.csv`: the file for the scorer.

`frame,label` for every frame: `label` from `possession_frames`, after one more rule. The scorer
counts whole possessions, so a spell of under a second with the same team before and after is
folded into that team (`ballq/smooth.py`).

## 7. Which output to use

| you want | use |
|---|---|
| the team in possession, with a confidence | `possession_frames`: `team`, `p_team` |
| who is on the ball | `possession_frames`: `state`, `holder_slot` |
| where the ball is, and how sure | `ball_track`: `x`, `y`, `sd_m`, `source` |
| passes, receptions, interceptions | `touches` with `kind` `gain` or `keep`: two consecutive rows are a pass and its reception |
| restarts and where they were taken | `possession_frames` where `event` is `restart` |
| why a decision was made | `touches`, including `passed_by`: `dv_ms`, `duration_s` and `p` |

## 8. Comparing against an algorithm that uses the raw ball

The question is whether this model labels possession better than a method that reads the
detector's ball point as it is. Both must be judged the same way, on the same frames, with the
same dead-ball intervals, so that the only difference is how the ball is used.

**The reference.** `--quick` also writes `out/submission_lh_raw.csv`: Link and Hoernig's rule on
the raw ball point, with the same dead-ball intervals and the same short-spell rule. Any other
method can join as long as it writes the scorer's format, `frame,label` for every frame.

**The command.**

```
python -m ballq submit --quick --also path/to/other_algorithm.csv
```

Each file is checked with the organisers' `validate_submission.py` and all of them are scored
together by `run_evaluation.py`. The board and one report per file land in `out/evaluation/`.

**What to read, in this order.**

| column | what it tells you | better is |
|---|---|---|
| `spellF1` | whole possessions matched to the true ones; the ranking metric | higher |
| `poss`, `dur` | how many possessions were predicted and their mean length; the public window has 17 of 14.2 s | close to the truth |
| `frag` | predicted possessions per true possession; above 1 means true possessions are being cut up | close to 1.0 |
| `transF1`, `lat_f` | changes of team found within a second, and how late on average | higher, close to 0 |
| `macroF1` | frames right, per label | higher, but it barely separates methods |

A raw-ball method typically shows high `macroF1` with low `spellF1`, many more possessions than
the truth and `frag` near 2: it is right most of the time and changes its mind too often. This
model should show fewer possessions and `frag` near 1. If its `poss` falls well below the truth it
is missing touches; lower `dv_touch_ms`.

**Reading the difference honestly.** The public window holds 17 possessions, so one possession
moves `spellF1` by about six points. A gap under 0.04 is noise. Do not tune thresholds until the
gap closes on this window: a setting fitted to 17 possessions will not hold on the full match.

**Finding where they differ.** Per frame, join the two files on `frame` and keep the rows where
the labels differ; for each, `touches.csv` shows what this model saw there (`dv_ms`, `duration_s`,
`p`, `kind`). In the viewer, `python match_viewer/match_viewer.py --pred out/submission_control.csv`
shows a submission as the PRED row beside GT and the raw L&H row, and the candidate list
"GT ≠ prediction" steps through the disagreements.

```python
import pandas as pd
a = pd.read_csv("out/submission_control.csv"); b = pd.read_csv("out/submission_lh_raw.csv")
diff = a.merge(b, on="frame", suffixes=("_control", "_raw")).query("label_control != label_raw")
touches = pd.read_csv("out/tactical/touches.csv")
```

## 9. What the tables promise

- One row per frame in the two frame tables, in order, with no gaps; join them on `frame`.
- Unknown is stated, never left blank or guessed: `label` X, `source` unknown, empty `team`.
- The detector's values are kept beside ours; nothing is overwritten.
- The tables are checked against this schema before they are written (`control.validate_outputs`).
  A run that breaks the schema writes nothing.
- The same input and settings give the same output. There is no randomness and no training.
- No hand labels go in.

## 10. Known limits

- A header or a volley is a touch above the ground, where the drawn ball is displaced; it can be
  missed, and possession then changes at the next touch on the ground.
- The taker rule assumes the ball is seen at the taker when play resumes. If it is not, the first
  touch after the restart is treated as the receiver.
- Thresholds are set from the papers and from physics, not fitted to this match.

## References

- Maksai, Wang, Fua. What Players do with the Ball: A Physically Constrained Interaction Modeling. CVPR 2016. https://arxiv.org/abs/1511.06181
- Vidal-Codina, Evans, El Fakir, Billingham. Automatic event detection in football using tracking data. 2022. https://arxiv.org/abs/2202.00804
- Link, Hoernig. Individual ball possession in soccer. PLoS ONE 12(7), 2017. https://doi.org/10.1371/journal.pone.0179953
