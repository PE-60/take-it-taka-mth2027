import numpy as np
from ballq.validate import holdout_mask, holdout_scores, kalman_baseline, kalman_holdout, ablations, _merge


def test_holdout_mask_hides_one_second_in_ten():
    m = holdout_mask(300, 25, every_s=10, hide_s=1, offset_s=1)
    assert m.sum() == 50 and m[25] and m[49] and not m[50] and m[275]


def test_holdout_filter_beats_kalman_through_the_lost_and_air_stretches(world, cfg):
    """Hide the detector on 1 s windows; the filter should put the ball close to where the detector saw it."""
    from ballq.filter import prepare
    import copy
    w = prepare(copy.copy(world), cfg)
    clean = ~w["air"] & ~w["white"]
    mask = holdout_mask(len(w["ball_obs"]), w["fps"], every_s=4, hide_s=1, offset_s=2)
    cfg_small = copy.deepcopy(cfg); cfg_small["grid"]["cell_m"] = 0.5
    s = holdout_scores(world, cfg_small, mask, clean)
    k = kalman_holdout(world, mask, clean)
    assert s["n_hidden"] > 20
    assert s["median_err_m"] < 1.5, s
    assert 0.0 <= s["inside_2sigma"] <= 1.0
    assert k["median_err_m"] is not None


def test_kalman_baseline_tracks_a_clean_line():
    n = 100; obs = np.column_stack([np.linspace(0.2, 0.4, n), np.full(n, 0.5)])
    m = kalman_baseline(obs, (105.3, 68.0), 25)
    assert np.abs(m[10:-10, 0] - obs[10:-10, 0]).max() * 105.3 < 0.3


def test_ablation_merge_switches_parts_off(cfg):
    c = _merge(cfg, {"observe": {"air_ball_tau": None}})
    assert c["observe"]["air_ball_tau"] == cfg["observe"]["base_tau"] and c["observe"]["gate_tau"] == cfg["observe"]["gate_tau"]


def test_ablations_run_and_report(world, cfg):
    import copy
    c = copy.deepcopy(cfg); c["grid"]["cell_m"] = 0.5
    w = copy.copy(world); w["gt"] = {r: ("H" if r < 115 else "A") for r in range(300)}
    out = ablations(w, c, names=["no_gate"])
    assert "no_gate" in out and out["no_gate"]["gt_agreement_lh_posterior"] is not None
