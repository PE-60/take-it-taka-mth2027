"""Possession algorithms: tracking in, one H/A/D/X label per tactical frame out.

Every algorithm is a PossessionAlgorithm subclass in possesion_algorithms/algorithms/ (any module
there is picked up). Besides its labelling rule it declares

    dead_ball   True:  label only the alive intervals of dead_ball_investigation/dead_ball.json,
                       each one on its own, and write D on the dead balls
                False: label the whole video
    source      "tactical"            the tactical tracking
                "broadcast"           the broadcast tracking, synced onto the tactical timeline
                "tactical+broadcast"  tactical, falling back to the synced broadcast on frames
                                      where the tactical lost the ball (or has none)

and run.py writes evaluation/results/<name>.csv, the submission format run_evaluation.py scores,
plus <name>.meta.json describing the run. The match viewers list those CSVs under "algorithms ▾"
on their timeline, to compare them with each other and with the ground truth.

    python possesion_algorithms/run.py --list
    python possesion_algorithms/run.py --all --from-exports
    python possesion_algorithms/inspect_geom_hmm.py --synthetic          # no match data
    python possesion_algorithms/inspect_geom_hmm.py --from-exports       # real match + public GT
    python -m unittest discover possesion_algorithms/tests

What is here:
    base.py          PossessionAlgorithm, Tracking (what label() gets), the source names
    inputs.py        loading tracking (raw files or viewer exports), sync, dead-ball mask
    runner.py        labelling a match, writing / validating / scoring the CSV
    run.py           the command line
    algorithms/      lh / lh_v2 heuristics, and geom_hmm (unsupervised geometry HMM)
    unsupervised/    features, anomaly, pca, hmm, synthetic — building blocks for geom_hmm
"""


from possesion_algorithms import paths  # noqa: F401  (puts the repo root on sys.path)
from possesion_algorithms.base import (BROADCAST, SOURCES, TACTICAL, TACTICAL_THEN_BROADCAST,  # noqa: F401
                                       PossessionAlgorithm, Tracking)
