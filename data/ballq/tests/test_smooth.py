import numpy as np
from ballq.smooth import absorb_short_spells, runs, spell_stats


def test_short_spell_between_same_team_is_absorbed_but_real_changes_stay():
    fps = 25
    lab = ["H"] * 100 + ["A"] * 10 + ["H"] * 100 + ["A"] * 100 + ["H"] * 5 + ["D"] * 50 + ["A"] * 8 + ["H"] * 100
    out = absorb_short_spells(lab, fps, 1.0)
    assert list(out[100:110]) == ["H"] * 10, "a 0.4 s interception that came straight back is absorbed"
    assert list(out[210:310]) == ["A"] * 100, "the real change to A is untouched"
    assert list(out[310:315]) == ["H"] * 5, "a short run next to a dead ball is kept"
    assert list(out[365:373]) == ["A"] * 8, "a short run after a dead ball is kept"


def test_x_inside_a_possession_does_not_protect_a_blip():
    lab = ["H"] * 50 + ["X"] * 5 + ["A"] * 6 + ["X"] * 5 + ["H"] * 50
    out = absorb_short_spells(lab, 25, 1.0)
    assert set(out[55:61]) == {"H"}


def test_runs_and_stats():
    lab = list("HHHAAXXHD")
    assert runs(lab) == [(0, 3, "H"), (3, 5, "A"), (5, 7, "X"), (7, 8, "H"), (8, 9, "D")]
    s = spell_stats(lab, 1)
    assert s["possessions"] == 3 and s["min_s"] == 1.0
