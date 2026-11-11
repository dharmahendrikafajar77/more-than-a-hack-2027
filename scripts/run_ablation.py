"""Ablation Study Suite for Team 'Mama Doain Aku Sukses Ya'
More Than a Hack 2027: Phase 1 (Ball Possession)

Compares incremental system components:
  E0: Reference Baseline (baseline_smooth50)
  E1: ATD Kalman Only (no CV)
  E2: E1 + CV Ball Candidate Detection
  E3: E2 + RTS Backward Smoother
  E4: E3 + Intelligent Dead-Ball Engine
  E5: E4 + Controlled Possession Model (Anti-Deflection Dwell)
  E6: Full System (HMM Viterbi Temporal Decoder)
"""

import os
import sys
import copy
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import games
from src.utils.config import load_config
from src.utils.determinism import set_seed
from src.tracking.atd_loader import load_team_mapping, parse_tactical_atd
from src.geometry.homography import load_homography
from src.tracking.smart_tagging import parse_smart_tagging, build_frame_priors
from src.fusion.kalman_fusion import fuse_ball_tracking
from src.features.feature_builder import build_frame_features
from src.possession.dead_ball import compute_dead_ball_scores
from src.possession.control_model import compute_emission_probabilities
from src.temporal.hmm_viterbi import decode_full_match
import evaluation.possession_metrics as pm

TOTAL_FRAMES = 199319
RESULTS_DIR = os.path.join(PROJECT_ROOT, "evaluation", "results")


def evaluate_against_public_gt(pred_csv_path):
    """Evaluate a submission CSV against the public 5-minute ground truth."""
    gt_path = os.path.join(PROJECT_ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    if not os.path.isfile(gt_path):
        return None
        
    gt_df = pm.read_label_csv(gt_path)
    gt_codes = pm.label_codes(gt_df)
    
    pred_df = pm.read_label_csv(pred_csv_path)
    pred_codes = pm.label_codes(pred_df)
    
    metrics = pm.evaluate(gt_codes, pred_codes)
    return metrics


def run_ablation_suite():
    """Run all ablation variants and compile comparative analysis table."""
    set_seed(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    base_config = load_config()
    game = base_config.get("environment", {}).get("game", "BAR-ATH")
    cache_dir = os.path.join(PROJECT_ROOT, base_config.get("environment", {}).get("cache_dir", "cache/tactical"))
    half_blocks = games.half_video_blocks(game)
    
    # 1. Load pre-requisites
    print("Loading ATD tracking, homography, and smart tagging for ablations...")
    meta_p, raw_p, homo_p, st_p, _ = games.require(
        ["atd_meta", "atd_raw", "homography", "smart_tagging", "video"], game=game
    )
    team_of_slot, _, _ = load_team_mapping(meta_p)
    atd_data = parse_tactical_atd(raw_p, team_of_slot)
    homo_matrices, reproj_errors, homo_valid = load_homography(homo_p)
    st_events = parse_smart_tagging(st_p)
    st_priors = build_frame_priors(st_events)
    
    # Check CV ball parquet
    cv_parquet = os.path.join(cache_dir, "cv_ball.parquet")
    if os.path.isfile(cv_parquet):
        cv_df = pq.read_table(cv_parquet).to_pandas()
    else:
        print("Warning: cv_ball.parquet not found, using empty CV detection")
        cv_df = pd.DataFrame({
            "pitch_x_m": np.full(TOTAL_FRAMES, np.nan, dtype=np.float32),
            "pitch_y_m": np.full(TOTAL_FRAMES, np.nan, dtype=np.float32),
            "cv_conf": np.zeros(TOTAL_FRAMES, dtype=np.float32),
            "detected": np.zeros(TOTAL_FRAMES, dtype=bool),
        })
        
    cv_pos_m = np.stack([cv_df["pitch_x_m"].values, cv_df["pitch_y_m"].values], axis=-1)
    cv_conf = cv_df["cv_conf"].values
    cv_detected = cv_df["detected"].values
    
    records = []
    
    # -------------------------------------------------------------
    # E0: Baseline Reference (baseline_smooth50)
    # -------------------------------------------------------------
    e0_path = os.path.join(RESULTS_DIR, "baseline_smooth50.csv")
    if os.path.isfile(e0_path):
        m0 = evaluate_against_public_gt(e0_path)
        sub0 = pm.read_label_csv(e0_path)
        c0 = pm.label_codes(sub0)
        sp0 = pm.possession_spells(c0, pm.valid_runs(c0))
        n_alive0 = int(np.sum((c0 == pm.H) | (c0 == pm.A)))
        dur0 = n_alive0 / 25.0 / len(sp0) if sp0 else 0.0
        records.append({
            "Experiment": "E0: Baseline Floor (smooth50)",
            "N Possessions": len(sp0),
            "Mean Dur (s)": round(dur0, 1),
            "Public Macro-F1": round(m0["macro_f1"], 4),
            "Public Spell-F1": round(m0["spell_f1_mean"], 4),
            "Public Alive/Dead F1": round(m0["alive_dead_f1"], 4),
        })
        
    # -------------------------------------------------------------
    # Helper to evaluate & record any variant
    # -------------------------------------------------------------
    def record_variant(name, labels, filename):
        path = os.path.join(RESULTS_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write("frame,label\n")
            f.writelines(f"{i},{l}\n" for i, l in enumerate(labels))
            
        sub = pm.read_label_csv(path)
        codes = pm.label_codes(sub)
        spells = pm.possession_spells(codes, pm.valid_runs(codes))
        n_alive = int(np.sum((codes == pm.H) | (codes == pm.A)))
        dur = n_alive / 25.0 / len(spells) if spells else 0.0
        
        m = evaluate_against_public_gt(path)
        records.append({
            "Experiment": name,
            "N Possessions": len(spells),
            "Mean Dur (s)": round(dur, 1),
            "Public Macro-F1": round(m["macro_f1"], 4),
            "Public Spell-F1": round(m["spell_f1_mean"], 4),
            "Public Alive/Dead F1": round(m["alive_dead_f1"], 4),
        })
        print(f"Recorded {name} -> Spell-F1: {m['spell_f1_mean']:.4f}, Macro-F1: {m['macro_f1']:.4f}")

    # -------------------------------------------------------------
    # E1: ATD Kalman Only (no CV, no RTS, simple proximity)
    # -------------------------------------------------------------
    print("\nRunning E1: ATD Kalman only...")
    fused_e1 = fuse_ball_tracking(
        atd_ball_pos_m=atd_data["ball_pos_m"],
        atd_ball_vis=atd_data["ball_visible"],
        cv_ball_pos_m=np.full_like(cv_pos_m, np.nan),
        cv_ball_conf=np.zeros_like(cv_conf),
        cv_detected=np.zeros_like(cv_detected),
        reproj_err=reproj_errors,
        half_blocks=half_blocks,
        max_coast_frames=0,
        use_rts=False
    )
    feat_e1 = build_frame_features(fused_e1, atd_data, st_priors, half_blocks)
    # Simple nearest player thresholding
    lbls_e1 = np.full(TOTAL_FRAMES, "D", dtype="<U1")
    for _, sf, ef in half_blocks:
        nh = feat_e1["nearest_home_dist_m"][sf:ef+1]
        na = feat_e1["nearest_away_dist_m"][sf:ef+1]
        bvis = feat_e1["ball_valid"][sf:ef+1]
        h_closer = (nh <= na) & (nh <= 2.5) & bvis
        a_closer = (na < nh) & (na <= 2.5) & bvis
        blk = np.full(ef - sf + 1, "D", dtype="<U1")
        blk[h_closer] = "H"
        blk[a_closer] = "A"
        lbls_e1[sf:ef+1] = blk
    record_variant("E1: ATD Kalman (No CV, No RTS, Raw Proximity)", lbls_e1, "ablation_e1.csv")

    # -------------------------------------------------------------
    # E2: E1 + CV Ball Candidate Detection
    # -------------------------------------------------------------
    print("\nRunning E2: E1 + CV Ball Detection...")
    fused_e2 = fuse_ball_tracking(
        atd_ball_pos_m=atd_data["ball_pos_m"],
        atd_ball_vis=atd_data["ball_visible"],
        cv_ball_pos_m=cv_pos_m,
        cv_ball_conf=cv_conf,
        cv_detected=cv_detected,
        reproj_err=reproj_errors,
        half_blocks=half_blocks,
        max_coast_frames=0,
        use_rts=False
    )
    feat_e2 = build_frame_features(fused_e2, atd_data, st_priors, half_blocks)
    lbls_e2 = np.full(TOTAL_FRAMES, "D", dtype="<U1")
    for _, sf, ef in half_blocks:
        nh = feat_e2["nearest_home_dist_m"][sf:ef+1]
        na = feat_e2["nearest_away_dist_m"][sf:ef+1]
        bvis = feat_e2["ball_valid"][sf:ef+1]
        h_closer = (nh <= na) & (nh <= 2.5) & bvis
        a_closer = (na < nh) & (na <= 2.5) & bvis
        blk = np.full(ef - sf + 1, "D", dtype="<U1")
        blk[h_closer] = "H"
        blk[a_closer] = "A"
        lbls_e2[sf:ef+1] = blk
    record_variant("E2: E1 + CV Ball Candidate Detection", lbls_e2, "ablation_e2.csv")

    # -------------------------------------------------------------
    # E3: E2 + RTS Backward Smoother
    # -------------------------------------------------------------
    print("\nRunning E3: E2 + RTS Backward Smoother...")
    fused_e3 = fuse_ball_tracking(
        atd_ball_pos_m=atd_data["ball_pos_m"],
        atd_ball_vis=atd_data["ball_visible"],
        cv_ball_pos_m=cv_pos_m,
        cv_ball_conf=cv_conf,
        cv_detected=cv_detected,
        reproj_err=reproj_errors,
        half_blocks=half_blocks,
        max_coast_frames=25,
        use_rts=True
    )
    feat_e3 = build_frame_features(fused_e3, atd_data, st_priors, half_blocks)
    lbls_e3 = np.full(TOTAL_FRAMES, "D", dtype="<U1")
    for _, sf, ef in half_blocks:
        nh = feat_e3["nearest_home_dist_m"][sf:ef+1]
        na = feat_e3["nearest_away_dist_m"][sf:ef+1]
        bvis = feat_e3["ball_valid"][sf:ef+1]
        h_closer = (nh <= na) & (nh <= 2.5) & bvis
        a_closer = (na < nh) & (na <= 2.5) & bvis
        blk = np.full(ef - sf + 1, "D", dtype="<U1")
        blk[h_closer] = "H"
        blk[a_closer] = "A"
        lbls_e3[sf:ef+1] = blk
    record_variant("E3: E2 + RTS Backward Smoother", lbls_e3, "ablation_e3.csv")

    # -------------------------------------------------------------
    # E4: E3 + Intelligent Dead-Ball Engine
    # -------------------------------------------------------------
    print("\nRunning E4: E3 + Intelligent Dead-Ball Engine...")
    score_d_e4 = compute_dead_ball_scores(feat_e3, half_blocks, params=base_config.get("dead_ball"))
    lbls_e4 = np.full(TOTAL_FRAMES, "D", dtype="<U1")
    for _, sf, ef in half_blocks:
        nh = feat_e3["nearest_home_dist_m"][sf:ef+1]
        na = feat_e3["nearest_away_dist_m"][sf:ef+1]
        sd = score_d_e4[sf:ef+1]
        is_d = sd > 0.0
        blk = np.full(ef - sf + 1, "D", dtype="<U1")
        h_mask = (~is_d) & (nh <= na) & (nh <= 2.5)
        a_mask = (~is_d) & (na < nh) & (na <= 2.5)
        blk[h_mask] = "H"
        blk[a_mask] = "A"
        lbls_e4[sf:ef+1] = blk
    record_variant("E4: E3 + Intelligent Dead-Ball Engine", lbls_e4, "ablation_e4.csv")

    # -------------------------------------------------------------
    # E5: E4 + Controlled Possession Model (Anti-Deflection Dwell)
    # -------------------------------------------------------------
    print("\nRunning E5: E4 + Controlled Possession Model...")
    emiss_e5 = compute_emission_probabilities(feat_e3, score_d_e4, half_blocks, params=base_config.get("possession"))
    # Argmax on probabilities (no HMM)
    lbls_e5 = np.full(TOTAL_FRAMES, "X", dtype="<U1")
    p_stack = np.stack([emiss_e5["p_home"], emiss_e5["p_away"], emiss_e5["p_dead"]], axis=-1)
    argmax_states = np.argmax(p_stack, axis=-1)
    state_letters = np.array(["H", "A", "D"])
    for _, sf, ef in half_blocks:
        lbls_e5[sf:ef+1] = state_letters[argmax_states[sf:ef+1]]
    record_variant("E5: E4 + Controlled Possession (Anti-Deflection, Argmax)", lbls_e5, "ablation_e5.csv")

    # -------------------------------------------------------------
    # E6: Full System (HMM Viterbi Temporal Sequence Decoder)
    # -------------------------------------------------------------
    print("\nRunning E6: Full System (HMM Viterbi Decoder)...")
    trans_mat = np.array(base_config.get("temporal", {}).get("transition_matrix"))
    init_priors = np.array(base_config.get("temporal", {}).get("initial_priors"))
    smoothing_w = base_config.get("temporal", {}).get("smoothing_window", 60)
    lbls_e6 = decode_full_match(
        emiss_e5,
        half_blocks,
        transition_matrix=trans_mat,
        initial_priors=init_priors,
        smoothing_window=smoothing_w,
    )
    record_variant("E6: Mama Doain Aku Sukses Ya (Full HMM Viterbi)", lbls_e6, "Mama_Doain_Aku_Sukses_Ya.csv")

    # -------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------
    df_results = pd.DataFrame(records)
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS (More Than a Hack 2027)")
    print("=" * 80)
    print(df_results.to_string(index=False))
    
    # Save markdown summary manually without requiring external tabulate
    ablation_md_path = os.path.join(PROJECT_ROOT, "outputs", "ablation_results.md")
    cols = list(df_results.columns)
    md_lines = ["# Ablation Study: Team Mama Doain Aku Sukses Ya\n\n"]
    md_lines.append("| " + " | ".join(cols) + " |\n")
    md_lines.append("| " + " | ".join(["---"] * len(cols)) + " |\n")
    for _, row in df_results.iterrows():
        md_lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |\n")
        
    with open(ablation_md_path, "w", encoding="utf-8") as f:
        f.writelines(md_lines)
        
    print(f"\nAblation table saved to: {os.path.relpath(ablation_md_path, PROJECT_ROOT)}")
    return df_results


if __name__ == "__main__":
    run_ablation_suite()
