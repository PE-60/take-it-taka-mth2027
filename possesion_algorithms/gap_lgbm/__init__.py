"""Possession inside ball-missing gaps, learned from pretend gaps where the ball was visible.

    intervals  target gaps (alive, ball missing in both recordings) with the team before and
               after each one; visible runs; the pretend-gap sampler
    features   per-frame team features with the ball left out on purpose, so they look the
               same inside a real gap as inside a pretend one (the ball is used only at the
               seen frames just outside a gap, by model.edge_features)
    model      the dataset from team T's point of view (every frame twice, T = H and T = A),
               LightGBM with grouped CV over time blocks, the baselines, and a two-state HMM
               with fixed ends that turns per-frame probabilities into few team changes
    pipeline   wires it together: teacher labels (lh_alternative_V2) on visible frames,
               pretend gaps, CV report, final model, decoded target gaps
    dead_model P(dead ball) per frame from team-neutral features, trained on the sure parts
               of the dead-ball detector and windows that straddle stoppages
    hmm3       three-state (H, A, dead) HMM with a minimum dead-ball length and a restart prior
    pipeline_dead  gap_lgbm_dead: both models and hmm3 over every frame the ball leaves unclear

Wired into the runner by ``algorithms.gap_lgbm`` (``gap_lgbm`` and ``gap_lgbm_dead``). Train and
score with

    python possesion_algorithms/run.py gap_lgbm gap_lgbm_dead --from-exports

The CV table is printed during the run and stored in evaluation/results/gap_lgbm.meta.json.
"""
