"""python -m ballq run      --source tactical [--start N --end M] [--data data] [--config ballq/config.yaml] [--out out]
   python -m ballq validate --source tactical [--start N --end M] ...   (hold-out, calibration, Kalman baseline, ablations)
run writes out/<source>/ball_frames, ball_players, ball_grid, slot_colours and ball_quality.json;
validate adds a "validation" section to that ball_quality.json."""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import yaml

from . import __version__
from .grid import Grid
from .io import load_world
from .filter import run_chunked
from .possession import lh_point, lh_posterior
from .contract import (FRAME_COLUMNS, PLAYER_COLUMNS, GRID_COLUMNS, COLOUR_COLUMNS, FLAG_BITS,
                       validate_frames, validate_players, write_table, config_hash, write_quality)
from . import colours as C
from .kitdb import FileKitDB, HttpKitDB, lookup_kits, resolve_kits


HERE = os.path.dirname(os.path.abspath(__file__))          # the ballq package folder
HOME = os.path.dirname(HERE)                                # where match_viewer.py sits


def find_data_root():
    """the folder that holds data/ (or the raw files): the folder above ballq/, its parent, the
    current folder, or MTAH_DATA_DIR; the first one with an ATD txt under it wins"""
    roots = [HOME, os.path.dirname(HOME), os.getcwd()] + ([os.environ["MTAH_DATA_DIR"]] if os.environ.get("MTAH_DATA_DIR") else [])
    for r in roots:
        if glob.glob(os.path.join(r, "**", "*_atd*.txt"), recursive=True):
            return r
    return HOME


def find(root, pattern, required=True):
    """first match under root; files inside data/ win, rendered overlays and exports never count"""
    hits = [p for p in glob.glob(os.path.join(root, "**", pattern), recursive=True)
            if "overlay" not in os.path.basename(p).lower() and (os.sep + "export" + os.sep) not in p and (os.sep + "out" + os.sep) not in p]
    hits.sort(key=lambda p: ((os.sep + "data" + os.sep) not in p, len(p)))
    if hits:
        return hits[0]
    if required:
        sys.exit(f"could not find {pattern} under {root}")
    return None


def paths_for(data_root, source):
    return {"atd_txt": find(data_root, f"*{source}_atd*.txt"), "atd_xml": find(data_root, f"*{source}_atd*.xml"),
            "homography": find(data_root, f"*{source}_homography*.csv", False),
            "tags": find(data_root, "*smart_tagging*.xml", False) if source == "tactical" else None,
            "gt": find(data_root, "*possession_gt_public.csv", False) if source == "tactical" else None,
            "video": find(data_root, f"*{source}*.mp4", False)}


def git_commit():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def write_grid_bin(sparse, grid, out_dir):
    offsets = np.zeros(len(sparse) + 1, np.uint64)
    with open(os.path.join(out_dir, "ball_grid.bin"), "wb") as fh:
        for r, (flat, p) in enumerate(sparse):
            ii, jj = np.unravel_index(flat, grid.shape)
            rec = np.column_stack([ii, jj, np.clip(np.round(p * 65535), 0, 65535)]).astype(np.uint16)
            fh.write(rec.tobytes()); offsets[r + 1] = offsets[r] + rec.nbytes
    offsets.tofile(os.path.join(out_dir, "ball_grid_index.bin"))


def cmd_colours(args, cfg, paths, world, out_dir):
    """Which team wears the light kit, from the video (option 2) and a kit database (option 3).
    Writes slot_colours; returns (white team or None, summary dict for the quality file)."""
    ccfg = cfg["colours"]
    summary = {"override": ccfg.get("override_white_team"), "video": None, "database": None}
    if ccfg.get("override_white_team"):
        print(f"white team from config: {ccfg['override_white_team']}")
        return ccfg["override_white_team"], summary
    team_of = dict(zip(world["slot_ids"], world["team"]))
    measured = {"H": None, "A": None}
    if paths.get("video") and "homography" in world:
        idx, H, valid = world["homography"]
        n = len(world["ball_obs"]); rng = np.linspace(0, n - 1, min(n, ccfg["frames_sampled"])).astype(int)
        hom = lambda r: (H[idx[world["start"] + r]] if (world["start"] + r) in idx and valid[idx[world["start"] + r]] else None)
        t0 = time.time()
        lab = C.measure_slots(paths["video"], rng, world["xy"], hom, world["start"], world["slot_ids"], world["pitch"], ccfg,
                              progress=lambda i, t: print(f"  colours {i}/{t}", end="\r"))
        res = C.classify_kits(lab, team_of, ccfg["white_lightness_min"], ccfg["white_chroma_max"])
        rows = [{"slot": s, "team": team_of[s], "L": v["L"], "a": v["a"], "b": v["b"], "d_own": v["d_own"], "d_other": v["d_other"],
                 "mislabel": v["mislabel"], "third": v["third"], "n_samples": v["n"]} for s, v in res["slots"].items()]
        p = write_table(pd.DataFrame(rows, columns=COLOUR_COLUMNS), os.path.join(out_dir, "slot_colours"))
        measured = res["team_lab"]
        summary["video"] = {"team_lab": measured, "white_team": res["white_team"], "mislabel": sorted(s for s, v in res["slots"].items() if v["mislabel"]),
                            "third": sorted(s for s, v in res["slots"].items() if v["third"])}
        print(f"\nvideo: team colours (Lab) {measured}  light kit: {res['white_team'] or 'none'}  mislabel {summary['video']['mislabel']}  third kit {summary['video']['third']}  ({time.time() - t0:.0f} s) -> {p}")
    else:
        print("no video or homography: shirt colours not measured")
    providers = [FileKitDB(ccfg.get("kit_db_file") or os.path.join(os.path.dirname(args.config), "kit_colours.yaml"))]
    if ccfg.get("kit_db_http", True):
        providers.append(HttpKitDB())
    db_kits = lookup_kits(world["team_names"], providers)
    final = resolve_kits(db_kits, measured, ccfg["white_lightness_min"], ccfg["white_chroma_max"])
    summary["database"] = {"kits": db_kits, "kit_used": final["kit_used"]}
    summary.update({"white_team": final["white_team"], "decided_by": final["source"], "notes": final["notes"]})
    print(f"database: {db_kits}  kit used {final['kit_used']}\nlight team: {final['white_team'] or 'none'} (decided by {final['source']})")
    for note in final["notes"]:
        print("  note:", note)
    return final["white_team"], summary


def cmd_run(args):
    cfg_text = open(args.config).read(); cfg = yaml.safe_load(cfg_text)
    paths = paths_for(args.data, args.source)
    for k, v in paths.items():
        print(f"{k:11s}: {v or '(none)'}")
    start, end = args.start, args.end
    if end is None:
        import cv2
        cap = cv2.VideoCapture(paths["video"]) if paths.get("video") else None
        end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1 if cap and cap.isOpened() else start + 10 ** 6
    world = load_world(paths, cfg, start, end)
    end = start + len(world["ball_obs"]) - 1
    out_dir = os.path.join(args.out, args.source); os.makedirs(out_dir, exist_ok=True)
    print(f"{args.source}: frames {start}..{end}, {world['xy'].shape[1]} slots, teams {world['team_names']}")

    white, col_res = cmd_colours(args, cfg, paths, world, out_dir)
    world["white_team"] = white

    grid = Grid(world["pitch"], cfg["grid"]["cell_m"])
    print(f"grid {grid.shape[1]} x {grid.shape[0]} cells of {grid.cell_m} m; chunks of {cfg['grid']['chunk_frames']} frames")
    t0 = time.time()
    res = run_chunked(grid, world, cfg, 0, len(world["ball_obs"]) - 1, progress=lambda i, t: print(f"  filter {i}/{t}  {time.time() - t0:.0f} s", end="\r"))
    print()
    n, S = world["xy"].shape[:2]
    pc = cfg["possession"]
    mass_tp = res["mass_within"][1.0] if pc["tp_m"] == 1.0 else res["mass_within"][2.0]
    lh_raw = lh_point(world["xy"], world["ball_obs"], world["team"], world["pitch"], world["fps"], pc, world["kicks"])
    lh_post = lh_posterior(world["xy"], res["mean"], mass_tp, world["team"], world["pitch"], world["fps"], pc, world["kicks"])

    # ---- ball_frames
    sd_x = np.sqrt(res["cov"][:, 0, 0]); sd_y = np.sqrt(res["cov"][:, 1, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(sd_x * sd_y > 0, res["cov"][:, 0, 1] / (sd_x * sd_y), 0.0)
    flags = np.zeros(n, int)
    lost = ~np.isfinite(world["ball_obs"][:, 0])
    flags |= np.where(lost, FLAG_BITS["lost"], 0); flags |= np.where(world["air"], FLAG_BITS["air"], 0); flags |= np.where(world["white"], FLAG_BITS["white"], 0)
    flags |= np.where(np.nan_to_num(world["hom_err"], nan=0) > cfg["observe"]["homography_bad_m"], FLAG_BITS["hom_low"], 0); flags |= np.where(world["dead"], FLAG_BITS["dead"], 0)
    for (r, _, _) in world["restarts"]:
        flags[r] |= FLAG_BITS["restart_prior"]
    holder = lh_post["holder"]
    holder_p = np.array([mass_tp[r, holder[r]] if holder[r] >= 0 else 0.0 for r in range(n)])
    p_carried = np.clip(res["mass_within"][2.0].max(axis=1), 0, 1)
    frames = pd.DataFrame({
        "frame": np.arange(start, end + 1), "mean_x": res["mean"][:, 0], "mean_y": res["mean"][:, 1], "sd_x": sd_x, "sd_y": sd_y, "corr": corr,
        "entropy_bits": res["entropy"], "p_observed_trusted": world["tau"], "p_air": world["air"].astype(float), "p_dead": world["dead"].astype(float),
        "p_carried": p_carried, "holder_slot": [world["slot_ids"][h] if h >= 0 else "" for h in holder], "holder_p": holder_p, "source_flags": flags,
        "lh_raw_team": lh_raw["team"], "lh_post_team": lh_post["team"], "raw_x": world["ball_obs"][:, 0], "raw_y": world["ball_obs"][:, 1],
        "speed_frames": world["speed_frames"], "kick": [int(r in world["kicks"]) for r in range(n)],
    })
    validate_frames(frames)
    p1 = write_table(frames, os.path.join(out_dir, "ball_frames"))

    # ---- ball_players: one row per (frame, slot) for slots within 5 m of the mean or with any mass within 2 m
    m1, m2 = res["mass_within"][1.0], res["mass_within"][2.0]
    dist = np.hypot((res["mean"][:, None, 0] - world["xy"][:, :, 0]) * world["pitch"][0], (res["mean"][:, None, 1] - world["xy"][:, :, 1]) * world["pitch"][1])
    keep = np.isfinite(dist) & ((dist <= 5.0) | (m2 >= 0.001))
    rr, kk = np.nonzero(keep)
    players = pd.DataFrame({"frame": start + rr, "slot": np.array(world["slot_ids"])[kk], "team": world["team"][kk], "dist_mean_m": dist[rr, kk],
                            "p_within_1m": m1[rr, kk], "p_within_2m": m2[rr, kk], "p_holder": (holder[rr] == kk) * holder_p[rr]}, columns=PLAYER_COLUMNS)
    validate_players(players)
    p2 = write_table(players, os.path.join(out_dir, "ball_players"))

    # ---- ball_grid (sparse: cells with mass >= grid.mass_min_out)
    parts = []
    for r in range(n):
        flat, p = res["sparse"][r]
        ii, jj = np.unravel_index(flat, grid.shape)
        xs, ys = grid.centre(ii, jj)
        parts.append(np.column_stack([np.full(len(flat), start + r), xs, ys, p]))
    p3 = write_table(pd.DataFrame(np.concatenate(parts), columns=GRID_COLUMNS).astype({"frame": int}), os.path.join(out_dir, "ball_grid"))
    # the same cells as a compact binary for the viewer: per cell uint16 row, uint16 col, uint16 mass x 65535,
    # plus a uint64 byte offset per frame (n + 1 entries) so one frame can be read with a range request
    write_grid_bin(res["sparse"], grid, out_dir)

    # ---- quality
    gt = world["gt"] or {}
    def agree(team_arr):
        hits = [(gt.get(start + r) == team_arr[r]) for r in range(n) if gt.get(start + r) in ("H", "A")]
        return float(np.mean(hits)) if hits else None
    q = {"ballq_version": __version__, "config_hash": config_hash(cfg_text), "git_commit": git_commit(), "source": args.source,
         "grid": {"cell_m": grid.cell_m, "nrows": grid.nrows, "ncols": grid.ncols, "pitch_m": list(world["pitch"])},
         "frames": [start, end], "n_frames": n, "frames_observed": int((~lost).sum()), "frac_lost": float(lost.mean()),
         "frac_air_flag": float(world["air"].mean()), "frac_white_flag": float(world["white"].mean()), "white_team": white, "colours": col_res,
         "mean_entropy_bits": float(res["entropy"].mean()), "median_sd_m": float(np.median(np.hypot(sd_x, sd_y))),
         "gt_agreement_lh_raw": agree(lh_raw["team"]), "gt_agreement_lh_posterior": agree(lh_post["team"]),
         "frac_filtered_speed_over_cap": float(np.mean(np.hypot(np.diff(res["mean"][:, 0]) * world["pitch"][0], np.diff(res["mean"][:, 1]) * world["pitch"][1]) * world["fps"] > cfg["motion"]["speed_cap_ms"])),
         "run_seconds": round(time.time() - t0, 1), "outputs": [p1, p2, p3]}
    write_quality(os.path.join(out_dir, "ball_quality.json"), q)
    print(json.dumps({k: v for k, v in q.items() if k not in ("outputs",)}, indent=1))
    print("wrote", out_dir)


def cmd_validate(args):
    """no hand labels: hide the detector for a second in every ten and see how well the filter fills it,
    compare with a Kalman smoother, check the stated uncertainty, and switch parts off one by one"""
    from .validate import holdout_mask, holdout_scores, kalman_holdout, ablations
    from .filter import prepare
    import copy
    cfg_text = open(args.config).read(); cfg = yaml.safe_load(cfg_text)
    paths = paths_for(args.data, args.source)
    end = args.end if args.end is not None else args.start + 25 * 60 * 5          # five minutes by default: the filter runs 5 times here
    world = load_world(paths, cfg, args.start, end)
    out_dir = os.path.join(args.out, args.source); os.makedirs(out_dir, exist_ok=True)
    qp = os.path.join(out_dir, "ball_quality.json")
    q = json.load(open(qp)) if os.path.exists(qp) else {}
    world["white_team"] = (q.get("colours") or {}).get("white_team") or cfg["colours"].get("override_white_team")
    w = prepare(copy.copy(world), cfg)
    clean = ~w["air"] & ~w["white"] & ~w["dead"]
    mask = holdout_mask(len(w["ball_obs"]), w["fps"])
    t0 = time.time()
    print(f"hold-out: hiding {int(mask.sum())} of {len(mask)} frames ...")
    hold = holdout_scores(world, cfg, mask, clean)
    kal = kalman_holdout(world, mask, clean)
    print(f"  grid filter  median {hold['median_err_m']:.2f} m  p90 {hold['p90_err_m']:.2f} m  inside 1/2 sigma {hold['inside_1sigma']:.2f}/{hold['inside_2sigma']:.2f} (expect 0.39/0.87)")
    print(f"  kalman       median {kal['median_err_m']:.2f} m  p90 {kal['p90_err_m']:.2f} m")
    abl = ablations(world, cfg, names=args.ablations.split(",") if args.ablations else None,
                    progress=lambda name, v: print(f"  ablation {name:18s} GT agreement {v['gt_agreement_lh_posterior']}  entropy {v['mean_entropy_bits']:.2f}"))
    q["validation"] = {"frames": [args.start, end], "holdout": hold, "kalman_holdout": kal, "ablations": abl, "seconds": round(time.time() - t0, 1)}
    write_quality(qp, q)
    print("validation written to", qp)


def cmd_inspect(args):
    """read every input once and say what was found; no filtering. Exit code 1 when something required is missing."""
    from . import io as bio
    cfg = yaml.safe_load(open(args.config))
    paths = paths_for(args.data, args.source)
    ok = True
    print(f"data root  : {args.data}")
    for k, v in paths.items():
        print(f"{k:11s}: {v or '(none)'}")
    if not paths.get("video"):
        print("  ! no video: shirt colours cannot be measured, frame count unknown (pass --end)")
    if args.source == "tactical" and not paths.get("tags"):
        print("  ! no smart tagging xml: no dead-ball windows, no restart priors")
    if args.source == "tactical" and not paths.get("gt"):
        print("  ! no GT csv: agreement numbers will be empty (fine for the submission)")
    # ATD
    try:
        meta = bio.parse_atd_meta(paths["atd_xml"])
        print(f"\nATD xml    : teams {[(t[0], t[1]) for t in meta['teams']]}, {len(meta['players'])} tracked slots, {len(meta['channels'])} channels, attack first half {meta['attack']}")
        first = last = None; n_lines = 0
        with open(paths["atd_txt"]) as fh:
            for line in fh:
                s = line.find(":")
                if s > 0 and line[:s].isdigit():
                    f = int(line[:s]) - 1; n_lines += 1
                    first = f if first is None else first; last = f
        print(f"ATD txt    : {n_lines} frames, video frames {first}..{last} ({(last - first + 1) / cfg['fps'] / 60:.1f} min at {cfg['fps']} fps)")
        w = bio.read_atd(paths["atd_txt"], paths["atd_xml"], first, min(last, first + 3000))
        seen = np.isfinite(w["xy"][:, :, 0]).mean(); ball = np.isfinite(w["ball_obs"][:, 0]).mean()
        print(f"             first 3000 frames: {100 * seen:.0f} % of player slots filled, ball present on {100 * ball:.0f} % of frames, teams {w['team_names']}, sides {dict(zip(*(a.tolist() for a in np.unique(w['team'], return_counts=True))))}")
    except Exception as e:  # noqa: BLE001
        ok = False; print(f"  ! ATD failed: {e}")
    # homography
    if paths.get("homography"):
        try:
            idx, H, err, valid = bio.read_homography(paths["homography"])
            print(f"homography : {len(idx)} rows, frames {min(idx)}..{max(idx)}, valid {100 * valid.mean():.0f} %, median reprojection error {np.nanmedian(err[valid == 1]):.2f} m, above {cfg['observe']['homography_bad_m']} m on {100 * np.mean(err[valid == 1] > cfg['observe']['homography_bad_m']):.0f} %")
        except Exception as e:  # noqa: BLE001
            ok = False; print(f"  ! homography failed: {e}")
    # tags
    if paths.get("tags"):
        try:
            tags = bio.read_tags(paths["tags"], cfg["fps"])
            codes = {}
            for t in tags:
                codes[t["code"]] = codes.get(t["code"], 0) + 1
            sp = [t for t in tags if t["code"].upper().startswith("SET PIECE")]
            print(f"tags       : {len(tags)} instances, codes {codes}")
            print(f"             {len(sp)} set pieces -> dead-ball windows; kick-offs for the restart prior: {sum(1 for t in sp if t.get('type') and 'kick' in t['type'].lower() and 'off' in t['type'].lower())}")
        except Exception as e:  # noqa: BLE001
            ok = False; print(f"  ! tags failed: {e}")
    # GT
    if paths.get("gt"):
        try:
            gt = bio.read_gt(paths["gt"]); labels = {}
            for v in gt.values():
                labels[v] = labels.get(v, 0) + 1
            print(f"GT         : {len(gt)} frames, {min(gt)}..{max(gt)}, labels {labels} (read for agreement only)")
        except Exception as e:  # noqa: BLE001
            ok = False; print(f"  ! GT failed: {e}")
    # video
    if paths.get("video"):
        import cv2
        cap = cv2.VideoCapture(paths["video"])
        if cap.isOpened():
            print(f"video      : {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames, {cap.get(cv2.CAP_PROP_FPS):.2f} fps, {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        else:
            ok = False; print("  ! video could not be opened")
    # kit database
    try:
        providers = [FileKitDB(cfg["colours"].get("kit_db_file") or os.path.join(HERE, "kit_colours.yaml"))]
        names = {"H": meta["teams"][0][1], "A": meta["teams"][1][1]}
        local = lookup_kits(names, providers)
        print(f"kit db     : local file -> {local}")
        if cfg["colours"].get("kit_db_http", True):
            print(f"             TheSportsDB -> {lookup_kits(names, [HttpKitDB()])}")
    except Exception as e:  # noqa: BLE001
        print(f"  ! kit database failed: {e}")
    print("\nall inputs readable" if ok else "\nsome inputs failed, see lines marked !")
    if not ok:
        sys.exit(1)


def cmd_submit(args):
    """frame,label for every tactical frame: D from the dead-ball intervals, else the L&H team from ball_frames"""
    from .submit import read_dead_intervals, dead_mask, build_labels, score_against_gt, write_submission
    from .contract import read_table
    from .io import read_gt
    import cv2
    paths = paths_for(args.data, "tactical")
    cap = cv2.VideoCapture(paths["video"]) if paths.get("video") else None
    n = args.frames or (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap and cap.isOpened() else None)
    if not n:
        sys.exit("cannot tell the frame count: pass --frames 199319")
    dead_path = args.dead
    if not dead_path:
        # any json or csv with "dead" in its name under the data root, nearest to the top first
        # match_viewer/export/dead_ball.json (written by match_viewer.py --dual) first, then any *dead*.json or csv
        cands = [p for p in glob.glob(os.path.join(args.data, "**", "*dead*.*"), recursive=True)
                 if p.lower().endswith((".json", ".csv")) and (os.sep + "out" + os.sep) not in p and os.path.basename(p) != "dead_ball_investigation"]
        cands.sort(key=lambda p: (not p.endswith(os.path.join("match_viewer", "export", "dead_ball.json")), p.count(os.sep), len(p)))
        dead_path = cands[0] if cands else None
        if not dead_path:
            print(f"  ! no dead-ball file: looked for *dead*.json or *dead*.csv under {args.data}; save the team json as dead_ball.json there or pass --dead PATH")
    intervals = read_dead_intervals(dead_path, "tactical") if dead_path else []
    dead = dead_mask(intervals, n)
    print(f"dead ball  : {dead_path or '(none)'}  {len(intervals)} intervals, {int(dead.sum())} frames ({100 * dead.mean():.1f} %)")
    # the control model: tracking + dead-ball intervals, no filter, no video (ballq/control.py)
    from .control import run_control
    cfg0 = yaml.safe_load(open(args.config))
    t0 = time.time()
    world0 = load_world(paths, cfg0, 0, n - 1)
    touches, poss, track = run_control(world0, dead, cfg0["control"])
    cdir = os.path.join(args.out, "tactical"); os.makedirs(cdir, exist_ok=True)
    from .control import validate_outputs, manifest
    validate_outputs(touches, poss, track)
    written_tables = [write_table(df, os.path.join(cdir, name_)) for name_, df in (("touches", touches), ("possession_frames", poss), ("ball_track", track))]
    inputs = {k: {"file": os.path.basename(v), "bytes": os.path.getsize(v)} for k, v in paths.items() if v and k in ("atd_txt", "atd_xml", "homography")}
    inputs["dead_ball"] = {"file": os.path.basename(dead_path), "intervals": len(intervals)} if dead_path else None
    man = manifest(touches, poss, track, cfg0["control"], inputs, config_hash(open(args.config).read()), __version__)
    man["files"] = [os.path.basename(p) for p in written_tables]
    write_quality(os.path.join(cdir, "control_manifest.json"), man)
    kinds = touches.kind.value_counts().to_dict()
    print(f"control    : {n} frames in {time.time() - t0:.0f} s; zone runs {len(touches)}: {kinds}; ball seen {100 * (track.source == 'seen').mean():.0f} %, held {100 * (track.source == 'held').mean():.0f} %, interpolated {100 * (track.source == 'interp').mean():.0f} %, unknown {100 * (track.source == 'unknown').mean():.0f} % -> {cdir}")
    control_frames = pd.DataFrame({"frame": poss.frame, "control": poss.team.replace("", "X")})
    if args.quick:
        # the base Link and Hoernig on the raw ball over the whole match: tracking only, no filter, no video
        from .possession import lh_point
        from . import features as F
        cfg = yaml.safe_load(open(args.config))
        t0 = time.time()
        world = load_world(paths, cfg, 0, n - 1)
        speed = F.ball_speed_frames(world["ball_obs"], world["pitch"], world["fps"])
        kicks = set(F.kicks_from_accel(speed, world["fps"], cfg["features"]["kick_accel_ms2"], cfg["features"]["kick_min_gap_s"], accel=F.ball_accel_xy(world["ball_obs"], world["pitch"], world["fps"])))
        lh = lh_point(world["xy"], world["ball_obs"], world["team"], world["pitch"], world["fps"], cfg["possession"], kicks)
        frames = pd.DataFrame({"frame": np.arange(n), "lh_raw_team": lh["team"], "lh_post_team": lh["team"]})
        print(f"quick      : base L&H on the raw ball, {n} frames in {time.time() - t0:.0f} s (tp {cfg['possession']['tp_m']} m, ta {cfg['possession']['ta_m']} m)")
        names = (("control", "control"), ("lh_raw", "lh_raw_team"))
    else:
        stem = os.path.join(args.out, "tactical", "ball_frames")
        try:
            frames = read_table(stem)
        except FileNotFoundError:
            frames = None
        names = (("control", "control"),) + ((("lh_filtered", "lh_post_team"), ("lh_raw", "lh_raw_team")) if frames is not None else ())
    if frames is not None and not args.quick:
        lo, hi = int(frames["frame"].min()), int(frames["frame"].max())
        print(f"ball_frames: frames {lo}..{hi} ({len(frames)} rows){'' if len(frames) >= n - 1 else '  ! not the whole match: frames outside get X'}")
    gt = read_gt(paths["gt"]) if paths.get("gt") else None
    os.makedirs(args.out, exist_ok=True)
    written = []
    from .smooth import absorb_short_spells, spell_stats
    for name, col in names:
        labels = build_labels(control_frames if col == "control" else frames, n, dead, col)
        if args.min_spell_s > 0:
            before = spell_stats(labels, 25)
            labels = absorb_short_spells(labels, 25, args.min_spell_s)
            after = spell_stats(labels, 25)
            print(f"{name:12s}: spells {before['possessions']} (mean {before['mean_s']:.1f} s, {before.get('under_1s', 0)} under 1 s) -> {after['possessions']} (mean {after['mean_s']:.1f} s) after absorbing runs under {args.min_spell_s} s between the same team")
        p = write_submission(labels, os.path.join(args.out, f"submission_{name}.csv")); written.append(p)
        counts = {k: int((labels == k).sum()) for k in "HADX"}
        line = f"{name:12s}: {counts}"
        if gt:
            s = score_against_gt(labels, gt)
            if s:
                line += f"  | GT frames {s['frames']}: accuracy {100 * s['accuracy']:.1f} %, abstained {100 * s['abstained']:.1f} %, " + ", ".join(f"{k} recall {100 * s[k]['recall']:.0f} % precision {100 * s[k]['precision']:.0f} %" for k in "HAD")
        print(line + f"  -> {p}")
    # the official checks, when the repository is next to the data
    val = find(args.data, "validate_submission.py", required=False)
    if val:
        passed = []
        for p in written:
            print(f"\n$ python {os.path.relpath(val, args.data)} {os.path.basename(p)}")
            if subprocess.call([sys.executable, val, p]) == 0:
                passed.append(p)
        ev = os.path.join(os.path.dirname(val), "run_evaluation.py")
        if passed and os.path.exists(ev) and not args.no_score:
            # the official scorer over a folder holding just our files; its scoreboard and reports land in out/evaluation
            import shutil
            res_dir = os.path.join(args.out, "submissions"); ev_out = os.path.join(args.out, "evaluation")
            shutil.rmtree(res_dir, ignore_errors=True); os.makedirs(res_dir); os.makedirs(ev_out, exist_ok=True)
            for p in passed:
                shutil.copy(p, res_dir)
            for p in args.also or []:                                      # other people's submissions, scored on the same board
                if subprocess.call([sys.executable, val, p]) == 0:
                    shutil.copy(p, res_dir)
            print(f"\n$ python {os.path.relpath(ev, args.data)} --results {os.path.relpath(res_dir, args.data)} --out {os.path.relpath(ev_out, args.data)}")
            subprocess.call([sys.executable, ev, "--results", res_dir, "--out", ev_out], cwd=os.path.dirname(ev))
            print(f"\nscoreboard and reports in {ev_out}")
    else:
        print("validate_submission.py not found under the data root; run it from the challenge repository on the files above")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ballq")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = {"--source": dict(choices=["tactical", "broadcast", "all"], default="tactical", help="'all' runs tactical then broadcast"), "--data": dict(default=None, help="folder that holds data/ (found on its own when left out)"),
              "--out": dict(default=os.path.join(HOME, "out"), help="default: out/ next to match_viewer.py, where the viewer looks"),
              "--config": dict(default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")), "--start": dict(type=int, default=0), "--end": dict(type=int)}
    r = sub.add_parser("run", help="filter a source and write the contract files")
    v = sub.add_parser("validate", help="hold-out, calibration, Kalman baseline and ablations on a window")
    v.add_argument("--ablations", help="comma separated subset of: no_holder,no_gate,no_air_flag,no_restart_prior")
    i = sub.add_parser("inspect", help="read every input once and report what was found")
    for p in (r, v, i):
        for flag, kw in common.items():
            p.add_argument(flag, **kw)
    s = sub.add_parser("submit", help="run the control model and write out/submission_control.csv (frame,label) with the dead-ball intervals as D; validates and scores it")
    s.add_argument("--dead", help="json or csv with the dead-ball intervals (default: dead_ball.json next to match_viewer.py)")
    s.add_argument("--frames", type=int, help="rows to write (default: tactical video frame count)")
    s.add_argument("--quick", action="store_true", help="also write the base L&H on the raw ball as submission_lh_raw.csv, for comparison")
    s.add_argument("--no-score", action="store_true", help="skip the official run_evaluation.py after validation")
    s.add_argument("--also", nargs="*", metavar="CSV", help="other submission files (frame,label) to validate and put on the same scoreboard")
    s.add_argument("--min-spell-s", type=float, default=1.0, help="absorb team runs shorter than this when the same team is on both sides (0 = off)")
    for flag in ("--data", "--out", "--config"):
        s.add_argument(flag, **common[flag])
    t = sub.add_parser("test", help="run the unit tests (installs pytest if needed)")
    args = ap.parse_args(argv)
    if args.cmd == "test":
        from .__main__ import ensure_packages
        ensure_packages({"pytest": "pytest"})
        sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q", os.path.join(HERE, "tests")]))
    if args.data is None:
        args.data = find_data_root()
    if args.cmd == "submit":
        return cmd_submit(args)
    for src in (["tactical", "broadcast"] if args.source == "all" else [args.source]):
        args.source = src
        {"run": cmd_run, "validate": cmd_validate, "inspect": cmd_inspect}[args.cmd](args)


if __name__ == "__main__":
    main()
