"""Unsupervised building blocks for possession from tracking geometry.

    features   per-frame geometry of ball and both teams, in two blocks: anti-symmetric
               (sign flips when H and A swap: who is nearer, who pushes up, who runs at the
               ball) and symmetric (ball speed, who is moving at all, how many are tracked)
    anomaly    tracking that cannot be right: a ball nobody plays, teleporting slots, a
               goalkeeper filed under the wrong team
    pca        robust scaling, winsorized per-block PCA, and a scale-only path for the
               possession axis (nan-mean of anti features — PC1 is too brittle alone)
    hmm        a sticky Gaussian hidden Markov model whose H and A states are mirror images,
               fitted by EM, decoded by Viterbi
    synthetic  an artificial match with ground truth, for tests and demos (the real data is
               proprietary and never read here)

Wired together by ``algorithms.geometry_hmm`` (``geom_hmm`` / ``geom_hmm_dead``).
Inspect on real or synthetic data with ``python possesion_algorithms/inspect_geom_hmm.py``.
Everything is numpy; nothing is learned from labels.
"""
