# lgbm_shape and the combined submission

These files drop into the team repository (take-it-taka-mth2027) at the same paths. They are
the LightGBM possession code plus two small scripts that combine its output with the L&H labels.
They do not run alone: `lgbm_shape` uses the repository's runner, the teacher algorithm
(`lh_alternative_V2`), the dead-ball intervals and the corrected goalkeepers.

## What is in here

| Path | What it is |
|---|---|
| `possesion_algorithms/gap_lgbm/` | the LightGBM models (unchanged, except `shape_model.py`) |
| `possesion_algorithms/gap_lgbm/shape_model.py` | changed: keeps the per-frame probabilities |
| `possesion_algorithms/algorithms/gap_lgbm.py` | changed: writes them to `evaluation/results/lgbm_shape_proba.csv` |
| `possesion_algorithms/tests/test_gap_lgbm.py` | the existing tests |
| `combine_labels.py` | builds the submission from the L&H labels and the lgbm_shape labels |
| `combine_proba.py` | experimental: same idea, using the probabilities instead of a duration |

## Step 1: inputs (skip what you already have)

```
pip install -r requirements-full.txt
python match_viewer/match_viewer.py --export --source tactical
python dead_ball_investigation/run.py --from-exports
python data_correction/goalkeeper_trackers/run.py
```

## Step 2: run lgbm_shape

```
python possesion_algorithms/run.py lgbm_shape --from-exports
```

Writes, in `evaluation/results/`:
- `lgbm_shape.csv`: frame,label for all 199,319 frames
- `lgbm_shape.meta.json`: settings, cross-validation table, HMM settings used
- `lgbm_shape_proba.csv`: frame, p_home, p_home_oof (empty on dead-ball frames)

`p_home` comes from the final model. `p_home_oof` comes from the cross-validation fold that did
not train on that frame, so it is the honest one to use for decisions.

Time: the model step took about 20 s in the last run (see `report.seconds` in the meta file).
The first run also builds the team features, which takes a few minutes; they are cached in
`gap_lgbm/cache/`, so later runs take about a minute.

Seeds: every random step uses `seed` (default 0) and LightGBM runs with `deterministic=True`,
so the same data, code and LightGBM version give the same file. Pin the LightGBM version in
`requirements-full.txt`.

## Step 3: build the submission

`LH.csv` below is the L&H label file from the ball-detection run
(`lh_balldet_gk_dead__bd_match_tol15__bd_min_confidence0.5.csv`).

```
python combine_labels.py LH.csv evaluation/results/lgbm_shape.csv submission.csv --max-s 2.5
python evaluation/validate_submission.py submission.csv
```

What it does, in order:
1. Restart gap: an X stretch that follows a dead ball takes the team that comes next.
2. Short-run veto: a team run shorter than 2.5 s is given to the other team when lgbm_shape
   labels more than half of its frames as the other team.

No randomness. On the files of 17 November it fills 6,917 frames, overrules 43 runs and scores
spellF1 0.7030 on the public window (0.6000 with `--max-s 0`, which skips step 2).

## Optional: the probability version

```
python combine_proba.py LH.csv evaluation/results/lgbm_shape_proba.csv out.csv --conf 0.7
```

A team run is given to the other team when the model's mean probability for the other team over
the run is at least `--conf`. Add `--max-s 2.5` to limit it to short runs. This has only been
tried on made-up probabilities, so score it before trusting it.
