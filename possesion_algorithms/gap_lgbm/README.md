# LightGBM possession algorithms (`gap_lgbm` family)

Four possession algorithms built on top of `lh_alternative_V2`. Each one uses LightGBM to decide
possession where `lh_alt` can't see the ball, and optionally to check `lh_alt` where it can.
All four plug into the normal runner (`possesion_algorithms/run.py`) and the match viewer, like
every other algorithm.

| Algorithm | One line | Status |
|---|---|---|
| `gap_lgbm` | `lh_alt` labels where the ball is seen; LightGBM + HMM fills the ball-missing gaps | Run on the real match. Good catches; sometimes adds a short spell right before a dead ball, and disagrees with `lh_alt` where a dead ball was missed |
| `gap_lgbm_dead` | `gap_lgbm` + a dead-vs-alive model; ball-missing frames and the start of play can become H, A **or D** | Built and tested on synthetic data; **not yet compared on the real match** |
| `gap_lgbm_double` | `gap_lgbm_dead` + a second, ball-aware LightGBM that can overrule `lh_alt` where the ball is seen | Built and tested on synthetic data; **not yet compared on the real match** |
| `lgbm_shape` | Test only: LightGBM labels every alive frame from team shape alone (no ball) | Experiment, to see how much team shape knows by itself |

**Which one to use:** `gap_lgbm_double` is the most complete version of the plan, and
`gap_lgbm_dead` is the same without the second model. Nobody has compared them on the public
5 minutes yet, so that's the first thing to do (see "Next steps").

---

## 1. Setup

```powershell
pip install -r requirements-full.txt        # includes lightgbm
```

Each algorithm needs three inputs. Skip any you already have:

```powershell
# tactical tracking export -> match_viewer/export/tactical_full/
python match_viewer/match_viewer.py --export --source tactical

# dead-ball intervals -> dead_ball_investigation/dead_ball.json
#   (needs the broadcast export too: same command with --source broadcast; sync.json is in the repo)
python dead_ball_investigation/run.py --from-exports

# corrected goalkeeper teams -> data_correction/goalkeeper_trackers/goalkeeper_tracks.csv
python data_correction/goalkeeper_trackers/run.py
```

`match_viewer.py` has no `--range` option, even though the error message in
`possesion_algorithms/inputs.py` suggests `--range all`. That hint is wrong; leave the option out.

## 2. Running, viewing, scoring

```powershell
# train + predict + score on the public 5 minutes (each writes evaluation/results/<name>.csv + .meta.json)
python possesion_algorithms/run.py gap_lgbm gap_lgbm_dead gap_lgbm_double lh_alt_gk_dead --from-exports

# compare everything in evaluation/results/
python evaluation/run_evaluation.py

# look at it: open the viewer, then "algorithms ▾" -> add the CSVs (refresh after a new run)
python match_viewer/match_viewer.py --serve
```

- **One command trains and predicts.** The model is not saved: each run retrains from scratch
  and writes only the predictions.
- **Training takes a few minutes** from the exports. Team features and teacher labels are cached
  in `gap_lgbm/cache/` (git-ignored), so later runs that only change model settings take about
  a minute.
- **The run prints a CV table.** It is also stored in `<name>.meta.json` under `"report"`, with
  the HMM settings chosen, the top 20 features and counts of what changed versus `lh_alt`.
- **Changing settings:** `--param key=value` (repeatable). The output name then gets a suffix,
  e.g. `gap_lgbm_double__use_hmm=True`, or set your own with `--name`.
- **`run.py --list`** shows every setting with its default.

## 3. How it works

### Shared ideas

- **Teacher.** Training labels come from `lh_alternative_V2` on the corrected goalkeepers, run on
  each alive interval of `dead_ball.json` (the same as `lh_alt_gk_dead`). No ground truth is used
  for training; the public 5 minutes are only for checking.
- **Pretend gaps.** To train for ball-missing gaps, stretches of play where the ball *was* seen
  are treated as gaps: the ball is hidden and the model must predict the teacher's labels.
  - Gap lengths are drawn from the real gaps' lengths.
  - Half of them start just after a kick.
  - Each keeps 1 s of visible ball on both sides.
- **Team features (no ball inside a gap).** Each team is described in its own attacking direction:
  - centroid; most advanced players; second-deepest player;
  - length, width, convex hull;
  - speeds and velocity towards goal;
  - pressure: opponents within 3 m / 5 m of each player;
  - changes of these over the last 0.5 s and 1 s;
  - team-neutral cues: share of players standing still, speed spread, players near the lines.

  Gap context: the team before and after the gap, time since its start and until its end, and the
  ball's position and velocity on the last frame seen before the gap and the first frame after it.
- **Mirroring.** Every frame becomes two rows, "does H have it" and "does A have it", with the
  teams swapped. The model therefore treats both teams alike, and P(H) is the average of the two.
- **Grouped CV.** Frames are grouped into 3-minute blocks (fold = block mod 5, so every fold has
  both halves). Early stopping uses 20% of the *training* blocks, never the held-out fold.
- **HMM.** Turns per-frame probabilities into possessions with a cost for each team change, so a
  possession doesn't break up every time a probability is near 0.5. Its settings are tuned in CV.
- **Only the tactical camera** is supported, because the goalkeeper correction only exists for it.

### `gap_lgbm`

- Frames where the ball is seen keep the teacher's label.
- Each ball-missing gap that the teacher enters with a team gets P(H) per frame from the team
  model, then a two-state HMM. The HMM starts on the team before the gap and ends on the team
  after it.
- Gaps at the start of play (before the teacher's first holder) keep the teacher's empty label.
- Dead balls are D from `dead_ball.json`.

### `gap_lgbm_dead`

Adds a dead-vs-alive model and a three-state HMM (H, A, D).

- **Dead model:**
  - **Labels:** the detector's *sure* dead frames (`off_pitch`, `still_gap`, `baseline` terms),
    minus 1 s at each end.
  - **Features:** team-neutral only. The detector's own inputs are deliberately left out, so the
    model is independent evidence rather than a copy of the detector.
  - **Training windows:** some anywhere in play, most straddling the start or end of a dead ball.
- **What gets decoded:** every ball-missing frame, alive or dead, plus visible frames at the start
  of play before the teacher's first holder. These can be H, A or D. Frames where the ball is seen
  stay fixed: the teacher's team in play, D in dead intervals.
- **Rules in the HMM:**
  - a dead ball lasts at least 1 s;
  - the team that did *not* have the ball before a stoppage restarts with probability `q`,
    estimated from the teacher;
  - frames the detector calls dead get a bonus towards D (`detector_logodds`), so the model needs
    real evidence to overrule it.
- **Scope:** D can only be added, extended or shortened where the ball is missing.

### `gap_lgbm_double`

`gap_lgbm_dead` + a second LightGBM for the frames where the ball is seen.

- **Ball-aware model features:** the team features, plus:
  - team of the closest player, now and in 1 s;
  - distance from each team's nearest player to the ball, now and in 1 s;
  - players of each team within 1 m, 3 m and 5 m of the ball;
  - ball position and velocity in the team's attacking direction;
  - attacking direction;
  - kicks: now, within ±3 frames, time since the last kick and until the next.
- **Trained on the teacher's labels, used out of fold only.** Each frame is predicted by a model
  that didn't see its 3-minute block, so it can't memorise the teacher's individual mistakes.
- **Nothing is fixed except the visible dead frames.** Every frame between fixed dead balls is
  labelled from the models, so the teacher can be overruled.
- **`use_hmm` (default `false`):**
  - `false`: each frame takes its most likely state (H if P(H) > 0.5 where the ball is seen;
    H/A/D from the gap and dead models where it isn't).
  - `true`: one three-state HMM decodes everything, with a switching cost, the 1 s minimum for
    dead balls and the restart rule.

  On synthetic data, turning the HMM off gave ~6x too many possessions versus ~2x with it, at
  the same frame accuracy. Since spellF1 punishes fragmentation, **compare both** on the public
  5 minutes (`--param use_hmm=true --name gap_lgbm_double_hmm`).

### `lgbm_shape` (test)

- Team features only: no ball, no gap context.
- Trained on the teacher's labels where the ball is seen.
- Labels every alive frame through a two-state HMM per alive interval. Nothing of the teacher's
  output is kept.
- Use it to see how much possession team shape alone recovers. It is not meant as a submission.

## 4. Settings worth knowing

| Setting | Algorithms | Default | Meaning |
|---|---|---|---|
| `seed` | all | 0 | All random sampling. Keep it fixed for a reproducible submission |
| `p_switch`, `tau` | all | 0 (tuned) | HMM team-change cost; how strongly per-frame probabilities count (lower = smoother). Set e.g. `p_switch=0.001` to force fewer changes |
| `num_leaves`, `min_data_in_leaf`, `feature_fraction`, `bagging_fraction`, `learning_rate` | all | 15, 300, 0.8, 0.8, 0.05 | LightGBM. Deliberately conservative |
| `n_folds`, `block_s` | all | 5, 180 | CV folds and block length (s) |
| `coverage`, `kick_bias`, `stride` | gap models | 1.0, 0.5, 4 | Pretend-gap frames per visible frame; share starting at a kick; train on every 4th frame |
| `t_p`, `t_a`, `kick_win`, `min_hold_s`, `horizon_s` | all | `lh_alt` defaults | The teacher's own settings (changing them rebuilds the cache) |
| `corrected_gk` | all | true | Swap in the corrected goalkeepers (tactical only) |
| `detector_logodds` | dead, double | 1.0 | Bonus for D on detector-dead frames; 0 = pure model |
| `min_dead_s` | dead, double | 1.0 | Minimum dead-ball length |
| `p_stop`, `p_restart`, `q` | dead, double | 0 (tuned / estimated) | HMM: alive→dead and dead→alive costs; how likely the other team restarts |
| `use_hmm`, `tau_ball` | double | false, 0 | Final HMM on/off; HMM temperature on ball-seen frames |
| `dead_ball_json` | dead, double | the repo default | These two read `dead_ball.json` themselves, so `run.py --dead-ball-json` doesn't reach them. Use this setting instead |

## 5. Reproducibility

RULES.md requires rerunning the repository to reproduce **the same file**.
- **Sampling:** every random step (pretend gaps, windows, folds, subsampling) is seeded from
  `seed`.
- **LightGBM:** runs with `deterministic=True` and `force_row_wise=True`.
- **Checked:** two separate runs give byte-identical labels.

If you submit one of these, write down the exact command, including every `--param`, and the
seed.

## 6. Code map

| File | What |
|---|---|
| `algorithms/gap_lgbm.py` | The four algorithm classes (`GapLGBM`, `GapLGBMDead`, `GapLGBMDouble`, `LGBMShape`) and the goalkeeper swap |
| `gap_lgbm/intervals.py` | Target gaps, pretend-gap and window samplers |
| `gap_lgbm/features.py` | Team features (no ball), attacking direction, cache-friendly chunked computation |
| `gap_lgbm/model.py` | Team model: mirrored dataset, gap-edge ball features, grouped CV, baselines, two-state HMM |
| `gap_lgbm/pipeline.py` | Teacher labels, cache, `train_team_model`, `gap_lgbm` end to end |
| `gap_lgbm/dead_model.py` | Dead-vs-alive model: labels from the detector, features, CV |
| `gap_lgbm/hmm3.py` | Three-state HMM with minimum dead length and restart rule |
| `gap_lgbm/pipeline_dead.py` | `gap_lgbm_dead` end to end; shared dead-model parts |
| `gap_lgbm/ball_features.py`, `double_model.py`, `pipeline_double.py` | `gap_lgbm_double`: ball features, out-of-fold ball model, decoding |
| `gap_lgbm/shape_model.py` | `lgbm_shape` |
| `tests/test_gap_lgbm.py` | Synthetic tests (no match data): `python -m unittest discover possesion_algorithms/tests` |

## 7. Limitations and next steps

**Limitations:**
- **The teacher sets the ceiling.** Every model learns from `lh_alt`'s labels. Random `lh_alt`
  mistakes can get smoothed away; systematic ones (e.g. taking the receiver as the first holder
  after a restart) are learned too.
- **The dead labels come from our own detector.** The dead model can find missed dead balls that
  look like the ones it learned from, but CV only measures agreement with the detector.
- **Only the public 5 minutes are a real check,** and they contain few stoppages. Treat score
  differences there as indications and confirm by eye in the viewer.
- **Synthetic tests check the plumbing, not the quality.** The synthetic match is far easier than
  the real one.

**Next steps:**
1. Run `gap_lgbm`, `gap_lgbm_dead`, `gap_lgbm_double` (with and without `use_hmm`) and
   `lh_alt_gk_dead`. Compare spellF1 / macroF1 on the public 5 minutes, and check in the viewer
   where they disagree.
2. Try `p_switch=0.001` on `gap_lgbm`, which may remove the short spells before dead balls.
3. If `lh_alt`'s restart labels are systematically wrong, fix that in `lh_alt` itself (or via
   Smart Tagging restart events, if they exist). The models can't unlearn it.
4. Fix the wrong `--range all` hint in `possesion_algorithms/inputs.py`.

## 8. Committing

- **Commit:** the code (`algorithms/gap_lgbm.py`, `gap_lgbm/`, `tests/test_gap_lgbm.py`), this
  README, `requirements-full.txt` and `.gitignore`.
- **Don't commit generated data:**
  - `evaluation/results/*.csv` and `gap_lgbm/cache/` are git-ignored;
  - **`evaluation/results/*.meta.json` is *not* ignored.** The new runs' `.meta.json` files
    (`gap_lgbm*.meta.json`, `lgbm_shape*.meta.json`) contain counts and scores derived from the
    match data, so leave them out of commits.
