#!/usr/bin/env python3
"""Run possession algorithms and write their submissions to evaluation/results/.

    python possesion_algorithms/run.py --list                     # what there is
    python possesion_algorithms/run.py lh lh_dead                 # from the raw data (parses the ATDs)
    python possesion_algorithms/run.py --all --from-exports       # every algorithm, from the viewer exports
    python possesion_algorithms/run.py lh --param t_p=1.5         # -> evaluation/results/lh__t_p1.5.csv
    python possesion_algorithms/run.py lh --dead-ball on --source tactical+broadcast --name lh_mine

Each run writes <name>.csv (frame,label for every tactical frame) and <name>.meta.json, checks the
CSV the way evaluation/run_evaluation.py does and prints its score on the public ground truth.
Algorithms that use the dead-ball intervals read dead_ball_investigation/dead_ball.json (make it
with dead_ball_investigation/run.py); broadcast ones go through match_viewer/export/sync.json.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

if __package__ in (None, ""):  # run as a file: make the package importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from possesion_algorithms.algorithms import registry  # noqa: E402
from possesion_algorithms.base import BROADCAST, SOURCES, TACTICAL, TACTICAL_THEN_BROADCAST  # noqa: E402
from possesion_algorithms.inputs import MatchInputs  # noqa: E402
from possesion_algorithms.paths import DEAD_BALL_JSON, DEFAULT_SYNC, RESULTS_DIR  # noqa: E402
from possesion_algorithms.runner import run  # noqa: E402

SHORT = {TACTICAL: "t", BROADCAST: "b", TACTICAL_THEN_BROADCAST: "tb"}


def parse_value(text: str, default):
    if isinstance(default, bool):
        if text.lower() in ("1", "true", "yes", "on"):
            return True
        if text.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"not a boolean: {text!r}")
    if isinstance(default, int):
        return int(text)
    if isinstance(default, float):
        return float(text)
    return text


def parse_params(cls, pairs):
    out = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--param wants key=value, got {pair!r}")
        if key not in cls.params:
            raise SystemExit(f"{cls.name} has no parameter {key!r}; it has {sorted(cls.params)}")
        try:
            out[key] = parse_value(value, cls.params[key])
        except ValueError as e:
            raise SystemExit(f"--param {pair}: {e}")
    return out


def variant(cls, dead_ball, source):
    """cls with dead_ball / source changed (None = keep)."""
    attrs = {}
    if dead_ball is not None and dead_ball != cls.dead_ball:
        attrs["dead_ball"] = dead_ball
    if source is not None and source != cls.source:
        attrs["source"] = source
    return type(cls.__name__, (cls,), attrs) if attrs else cls


def auto_name(cls, algo, params):
    """<name>[__dead_<src>|__whole_<src>][__<key><value>...] for runs that differ from the defaults."""
    name = cls.name
    if algo.dead_ball != cls.dead_ball or algo.source != cls.source:
        name += f"__{'dead' if algo.dead_ball else 'whole'}_{SHORT[algo.source]}"
    for k in sorted(params):
        name += f"__{k}{params[k]}"
    return name


def list_algorithms(algs):
    w = max((len(n) for n in algs), default=4)
    print(f"{'name':<{w}}  dead ball  source              params")
    for name, cls in algs.items():
        params = ", ".join(f"{k}={v}" for k, v in cls.params.items())
        print(f"{name:<{w}}  {'yes' if cls.dead_ball else 'no':<9}  {cls.source:<18}  {params}")
        if cls.description:
            print(f"{'':<{w}}  {cls.description}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="algorithms to run (see --list)")
    ap.add_argument("--list", action="store_true", help="list the algorithms and exit")
    ap.add_argument("--all", action="store_true", help="run every algorithm")
    ap.add_argument("--from-exports", action="store_true",
                    help="read the viewer exports in match_viewer/export/ instead of the raw data (fast, "
                         "but only the exported frames get labels)")
    ap.add_argument("--tactical-export", help="tactical export folder (implies reading it instead of the raw data)")
    ap.add_argument("--broadcast-export", help="broadcast export folder (same)")
    ap.add_argument("--sync", default=DEFAULT_SYNC, help="sync anchors (default match_viewer/export/sync.json)")
    ap.add_argument("--dead-ball-json", default=DEAD_BALL_JSON,
                    help="dead-ball intervals (default dead_ball_investigation/dead_ball.json)")
    ap.add_argument("--out-dir", default=RESULTS_DIR, help="where the CSVs go (default evaluation/results/)")
    ap.add_argument("--param", action="append", metavar="KEY=VALUE", help="override a parameter (repeatable)")
    ap.add_argument("--dead-ball", choices=("on", "off"), help="override the algorithm's dead-ball mode")
    ap.add_argument("--source", choices=SOURCES, help="override the algorithm's tracking source")
    ap.add_argument("--name", help="output name (one algorithm only); default the algorithm's name, "
                                   "plus a suffix for each override")
    ap.add_argument("--no-score", action="store_true", help="skip scoring on the public ground truth")
    args = ap.parse_args(argv)

    algs = registry()
    if args.list:
        list_algorithms(algs)
        return 0
    names = list(algs) if args.all else args.names
    if not names:
        ap.error("name some algorithms, or --all (see --list)")
    unknown = [n for n in names if n not in algs]
    if unknown:
        ap.error(f"unknown algorithm(s) {unknown}; have {list(algs)}")
    if args.name and len(names) != 1:
        ap.error("--name needs exactly one algorithm")
    if args.param and len(names) != 1:
        ap.error("--param needs exactly one algorithm")

    exports = {s: p for s, p in ((TACTICAL, args.tactical_export), (BROADCAST, args.broadcast_export)) if p}
    inputs = MatchInputs(from_exports=args.from_exports, exports=exports, sync_path=args.sync,
                         dead_ball_path=args.dead_ball_json)
    dead = None if args.dead_ball is None else args.dead_ball == "on"
    failed = 0
    for name in names:
        cls = algs[name]
        params = parse_params(cls, args.param)
        algo = variant(cls, dead, args.source)(**params)
        out_name = args.name or auto_name(cls, algo, params)
        print(f"\n=== {out_name}: dead ball {'on' if algo.dead_ball else 'off'}, source {algo.source}, "
              f"params {algo.params}")
        t0 = time.time()
        try:
            meta = run(algo, inputs, args.out_dir, out_name, do_score=not args.no_score)
        except (IOError, ValueError) as e:
            print(f"{out_name}: FAILED\n  {e}")
            failed += 1
            continue
        c = meta["counts"]
        print(f"wrote {os.path.relpath(os.path.join(args.out_dir, out_name + '.csv'))} in {time.time() - t0:.1f} s"
              f"  H {c['H']}  A {c['A']}  D {c['D']}  X {c['X']}")
        s = meta.get("public_gt")
        if s:
            print(f"public GT: spellF1 {s['spell_f1_mean']:.3f}  macroF1 {s['macro_f1']:.3f}  "
                  f"accuracy {s['accuracy']:.3f}  ({s['n_graded']} graded frames)")
    if failed:
        print(f"\n{failed} of {len(names)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
