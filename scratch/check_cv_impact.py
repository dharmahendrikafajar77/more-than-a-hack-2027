import sys
sys.path.insert(0, ".")
import numpy as np
import pyarrow.parquet as pq
import games
from src.utils.config import load_config
from src.tracking.atd_loader import load_team_mapping, parse_tactical_atd
from src.geometry.homography import load_homography
from src.tracking.smart_tagging import parse_smart_tagging, build_frame_priors
from src.fusion.kalman_fusion import fuse_ball_tracking
from src.features.feature_builder import build_frame_features
from src.possession.dead_ball import compute_dead_ball_scores
from src.possession.control_model import compute_emission_probabilities
from src.temporal.hmm_viterbi import decode_full_match
import evaluation.possession_metrics as pm

cfg = load_config()
meta_p, raw_p, homo_p, st_p = games.require(["atd_meta", "atd_raw", "homography", "smart_tagging"])
half_blocks = games.half_video_blocks("BAR-ATH")
team_of_slot, _, _ = load_team_mapping(meta_p)
atd_data = parse_tactical_atd(raw_p, team_of_slot)
_, reproj_err, _ = load_homography(homo_p)
st_events = parse_smart_tagging(st_p)
st_priors = build_frame_priors(st_events)

atd_vis_in_halves = sum(int(np.sum(atd_data["ball_visible"][sf:ef+1])) for _, sf, ef in half_blocks)
print("ATD visible inside half_blocks:", atd_vis_in_halves)

# Check how many CV detections happen in public window 11185..18684
cv = pq.read_table("cache/tactical/cv_ball.parquet").to_pandas()
cv_pub = int(cv["detected"].values[11185:18685].sum())
atd_pub_miss = int((~atd_data["ball_visible"][11185:18685]).sum())
print(f"Public window missing ATD frames: {atd_pub_miss}, CV detected in public window: {cv_pub}")

# Run full E6 WITHOUT CV
n = 199319
fused_nocv = fuse_ball_tracking(
    atd_ball_pos_m=atd_data["ball_pos_m"],
    atd_ball_vis=atd_data["ball_visible"],
    cv_ball_pos_m=np.full((n, 2), np.nan, dtype=np.float32),
    cv_ball_conf=np.zeros(n, dtype=np.float32),
    cv_detected=np.zeros(n, dtype=bool),
    reproj_err=reproj_err,
    half_blocks=half_blocks,
    max_coast_frames=cfg["fusion"]["max_coast_frames"],
    use_rts=cfg["fusion"]["use_rts_smoother"],
)
vals, cnts = np.unique(fused_nocv["fused_source"], return_counts=True)
print("No-CV fused_source:", dict(zip(vals.tolist(), cnts.tolist())))

feat_nocv = build_frame_features(fused_nocv, atd_data, st_priors, half_blocks)
sd_nocv = compute_dead_ball_scores(feat_nocv, half_blocks, params=cfg["dead_ball"])
em_nocv = compute_emission_probabilities(feat_nocv, sd_nocv, half_blocks, params=cfg["possession"])
lbls_nocv = decode_full_match(
    em_nocv, half_blocks,
    transition_matrix=np.array(cfg["temporal"]["transition_matrix"]),
    initial_priors=np.array(cfg["temporal"]["initial_priors"]),
    smoothing_window=cfg["temporal"]["smoothing_window"],
)

gt_codes = pm.label_codes(pm.read_label_csv("evaluation/gt/BAR-ATH_possession_gt_public.csv"))
pred_codes = pm.label_codes(lbls_nocv)
m = pm.evaluate(gt_codes, pred_codes)
spells = pm.possession_spells(pred_codes, pm.valid_runs(pred_codes))
n_alive = int(np.sum((pred_codes == pm.H) | (pred_codes == pm.A)))
print(
    f"Full System WITHOUT CV -> Macro-F1: {m['macro_f1']:.4f}, "
    f"Spell-F1: {m['spell_f1_mean']:.4f}, Alive/Dead-F1: {m['alive_dead_f1']:.4f}, "
    f"Full-Match Poss: {len(spells)}, Dur: {n_alive/25.0/len(spells):.1f}s"
)
