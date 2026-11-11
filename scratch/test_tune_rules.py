"""Test physics-based dead ball and persistence rules."""

import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = "E:/more-than-a-hack-2027"
sys.path.insert(0, PROJECT_ROOT)

import games
from src.utils.config import load_config
from src.utils.determinism import set_seed
from src.tracking.atd_loader import load_team_mapping, parse_tactical_atd
from src.geometry.homography import load_homography
from src.tracking.smart_tagging import parse_smart_tagging, build_frame_priors
from src.fusion.kalman_fusion import fuse_ball_tracking
from src.features.feature_builder import build_frame_features
import evaluation.possession_metrics as pm

TOTAL_FRAMES = 199319

# Load assets
cfg = load_config()
game = "BAR-ATH"
meta_p, raw_p, homo_p, st_p, _ = games.require(["atd_meta", "atd_raw", "homography", "smart_tagging", "video"])
half_blocks = games.half_video_blocks(game)

team_of_slot, _, _ = load_team_mapping(meta_p)
atd_data = parse_tactical_atd(raw_p, team_of_slot)
homo_matrices, reproj_errors, _ = load_homography(homo_p)
st_events = parse_smart_tagging(st_p)
st_priors = build_frame_priors(st_events)

# Load fused tracking
fused_table = pq.read_table("cache/tactical/fused_tracking.parquet")
fused_ball = {col: fused_table[col].to_numpy() for col in fused_table.column_names}

features = build_frame_features(fused_ball, atd_data, st_priors, half_blocks)

# Load GT for scoring
gt_df = pm.read_label_csv("evaluation/gt/BAR-ATH_possession_gt_public.csv")
gt_codes = pm.label_codes(gt_df)

def evaluate_predictions(labels):
    sub_df = pd.DataFrame({"frame": np.arange(TOTAL_FRAMES), "label": labels})
    pred_codes = pm.label_codes(sub_df)
    m = pm.evaluate(gt_codes, pred_codes)
    runs = pm.valid_runs(pred_codes)
    spells = pm.possession_spells(pred_codes, runs)
    n_alive = int(np.sum((pred_codes == pm.H) | (pred_codes == pm.A)))
    dur = n_alive / 25.0 / len(spells) if spells else 0.0
    return {
        "macro_f1": m["macro_f1"],
        "spell_f1": m["spell_f1_mean"],
        "alive_dead_f1": m["alive_dead_f1"],
        "n_spells": len(spells),
        "mean_dur": dur,
        "n_h": int(np.sum(pred_codes == pm.H)),
        "n_a": int(np.sum(pred_codes == pm.A)),
        "n_d": int(np.sum(pred_codes == pm.D)),
    }

print("Setup ready.")

