"""What leaves the pipeline, and the checks on it. Tables are plain CSV."""
import json
import hashlib
import numpy as np
import pandas as pd

FRAME_COLUMNS = ["frame", "mean_x", "mean_y", "sd_x", "sd_y", "corr", "entropy_bits", "p_observed_trusted",
                 "p_air", "p_dead", "p_carried", "holder_slot", "holder_p", "source_flags"]
# also written, for comparison in the viewer: the raw detector point, both L&H teams, frame speed, kick flag
FRAME_EXTRA = ["lh_raw_team", "lh_post_team", "raw_x", "raw_y", "speed_frames", "kick"]
PLAYER_COLUMNS = ["frame", "slot", "team", "dist_mean_m", "p_within_1m", "p_within_2m", "p_holder"]
GRID_COLUMNS = ["frame", "cell_x", "cell_y", "p"]
COLOUR_COLUMNS = ["slot", "team", "L", "a", "b", "d_own", "d_other", "mislabel", "third", "n_samples"]
FLAG_BITS = {"lost": 1, "air": 2, "white": 4, "hom_low": 8, "restart_prior": 16, "dead": 32}


def validate_frames(df):
    missing = [c for c in FRAME_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"ball_frames missing columns {missing}")
    for c in ("p_observed_trusted", "p_air", "p_dead", "p_carried", "holder_p"):
        v = df[c].to_numpy(dtype=float)
        if np.any((v < -1e-9) | (v > 1 + 1e-9)):
            raise ValueError(f"{c} outside 0..1")
    if df["frame"].duplicated().any():
        raise ValueError("duplicate frames")
    if not df["frame"].is_monotonic_increasing:
        raise ValueError("frames not sorted")
    if df[["mean_x", "mean_y", "entropy_bits"]].isna().any().any():
        raise ValueError("NaN in mean or entropy: a frame with no information must have high entropy, not NaN")


def validate_players(df):
    for c in ("p_within_1m", "p_within_2m", "p_holder"):
        v = df[c].to_numpy(dtype=float)
        if np.any((v < -1e-9) | (v > 1 + 1e-9)):
            raise ValueError(f"{c} outside 0..1")


def write_table(df, path_stem):
    """plain CSV, so anyone can open it; returns the path written"""
    p = str(path_stem) + ".csv"; df.to_csv(p, index=False); return p


def read_table(path_stem):
    import os
    for ext in (".csv", ".csv.gz", ".parquet"):
        p = str(path_stem) + ext
        if os.path.exists(p):
            return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
    raise FileNotFoundError(path_stem)


def config_hash(cfg_text):
    return hashlib.sha256(cfg_text.encode()).hexdigest()[:12]


def write_quality(path, payload):
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
