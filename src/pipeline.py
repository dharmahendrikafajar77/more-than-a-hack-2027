"""Master Pipeline for Team 'Mama Doain Aku Sukses Ya'
More Than a Hack 2027: Phase 1 (Ball Possession)
"""

import os
import sys
import time
import shutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import games
from src.utils.config import load_config
from src.utils.determinism import set_seed, compute_file_sha256, save_manifest
from src.tracking.atd_loader import load_team_mapping, parse_tactical_atd
from src.geometry.homography import load_homography
from src.tracking.smart_tagging import parse_smart_tagging, build_frame_priors
from src.video.decoder import VideoChunkDecoder
from src.cv.ball_detector import extract_cv_ball_table
from src.fusion.kalman_fusion import fuse_ball_tracking
from src.features.feature_builder import build_frame_features
from src.possession.dead_ball import compute_dead_ball_scores
from src.possession.control_model import compute_emission_probabilities
from src.temporal.hmm_viterbi import decode_full_match
from src.reporting.storytelling import generate_all_storytelling_artifacts
import evaluation.possession_metrics as pm

TOTAL_FRAMES = 199319


def run_pipeline(config_path=None, output_path=None, force_recompute=False, generate_report=True):
    """Execute the end-to-end algorithmic and CV possession pipeline."""
    start_time = time.time()
    
    # 1. Load config and lock seeds
    config = load_config(config_path)
    seed = config.get("environment", {}).get("seed", 42)
    set_seed(seed)
    
    team_name = config.get("team", {}).get("name", "Mama Doain Aku Sukses Ya")
    team_slug = config.get("team", {}).get("slug", "Mama_Doain_Aku_Sukses_Ya")
    game = config.get("environment", {}).get("game", "BAR-ATH")
    cache_dir = os.path.join(PROJECT_ROOT, config.get("environment", {}).get("cache_dir", "cache/tactical"))
    output_dir = os.path.join(PROJECT_ROOT, config.get("environment", {}).get("output_dir", "outputs"))
    use_cache = config.get("environment", {}).get("use_cache", True) and not force_recompute
    
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    if output_path is None:
        primary_output_csv = os.path.join(output_dir, f"{team_slug}.csv")
    else:
        primary_output_csv = os.path.abspath(output_path)
        
    eval_results_dir = os.path.join(PROJECT_ROOT, "evaluation", "results")
    os.makedirs(eval_results_dir, exist_ok=True)
    mirrored_output_csv = os.path.join(eval_results_dir, f"{team_slug}.csv")
    
    print("=" * 72)
    print(f"MORE THAN A HACK 2027 - PHASE 1: BALL POSSESSION")
    print(f"Team: {team_name} ({team_slug})")
    print(f"Master Output: {os.path.relpath(primary_output_csv, PROJECT_ROOT)}")
    print("=" * 72)
    
    # 2. Locate match assets
    print("\n[1/7] Locating match assets via games.py...")
    asset_names = ["atd_meta", "atd_raw", "homography", "smart_tagging", "video"]
    meta_p, raw_p, homo_p, st_p, video_p = games.require(asset_names, game=game)
    half_blocks = games.half_video_blocks(game)
    print(f"  Tactical match blocks: {half_blocks}")
    
    # 3. Stream tracking data & homography
    print("\n[2/7] Parsing ATD tracking, homography, and smart tagging...")
    team_of_slot, home_name, away_name = load_team_mapping(meta_p)
    print(f"  Teams: Home={home_name}, Away={away_name} (Slots: {len(team_of_slot)})")
    
    atd_data = parse_tactical_atd(raw_p, team_of_slot, total_frames=TOTAL_FRAMES)
    atd_vis_count = int(np.sum(atd_data["ball_visible"]))
    print(f"  ATD Ball visible on {atd_vis_count}/{TOTAL_FRAMES} frames ({atd_vis_count*100.0/TOTAL_FRAMES:.1f}%)")
    
    homo_matrices, reproj_errors, homo_valid = load_homography(homo_p)
    print(f"  Homography loaded: {len(homo_matrices)} matrices, mean reproj err: {np.mean(reproj_errors):.2f}px")
    
    st_events = parse_smart_tagging(st_p, fps=config.get("geometry", {}).get("fps", 25.0))
    st_priors = build_frame_priors(st_events, total_frames=TOTAL_FRAMES)
    print(f"  Smart Tagging: {len(st_events)} instances parsed and transformed into directional priors")
    
    # 4. CV Ball Candidate Detection
    print("\n[3/7] Processing CV ball candidate detection...")
    cv_cfg = config.get("cv", {})
    cv_enabled = cv_cfg.get("enabled", True)
    cv_parquet_path = os.path.join(cache_dir, "cv_ball.parquet")
    
    decoder = None
    if cv_enabled and (not use_cache or not os.path.isfile(cv_parquet_path)):
        print(f"  Opening tactical video stream for ROI inspection...")
        decoder = VideoChunkDecoder(video_p)
        
    cv_df = extract_cv_ball_table(
        decoder=decoder,
        homography_matrices=homo_matrices,
        reprojection_errors=reproj_errors,
        atd_ball_pos_m=atd_data["ball_pos_m"],
        atd_ball_vis=atd_data["ball_visible"],
        player_mean_spd=atd_data["mean_player_speed"],
        output_parquet_path=cv_parquet_path,
        sample_rate=1,
        cv_enabled=cv_enabled,
    )
    if decoder is not None:
        decoder.release()
        
    cv_det_count = int(np.sum(cv_df["detected"].values))
    print(f"  CV Ball detections available: {cv_det_count} frames")
    
    # 5. Kalman Filter & RTS Smoother Fusion
    print("\n[4/7] Fusing ATD tracking + CV detections via Kalman & RTS smoother...")
    fusion_cfg = config.get("fusion", {})
    fused_parquet_path = os.path.join(cache_dir, "fused_tracking.parquet")
    
    if use_cache and os.path.isfile(fused_parquet_path):
        fused_table = pq.read_table(fused_parquet_path)
        if len(fused_table) == TOTAL_FRAMES:
            fused_df = fused_table.to_pandas()
            fused_ball = {col: fused_df[col].values for col in fused_df.columns}
            print(f"  Loaded fused ball trajectory from cache ({fused_parquet_path})")
        else:
            fused_ball = None
    else:
        fused_ball = None
        
    if fused_ball is None:
        cv_pos_m = np.stack([cv_df["pitch_x_m"].values, cv_df["pitch_y_m"].values], axis=-1)
        fused_ball = fuse_ball_tracking(
            atd_ball_pos_m=atd_data["ball_pos_m"],
            atd_ball_vis=atd_data["ball_visible"],
            cv_ball_pos_m=cv_pos_m,
            cv_ball_conf=cv_df["cv_conf"].values,
            cv_detected=cv_df["detected"].values,
            reproj_err=reproj_errors,
            half_blocks=half_blocks,
            max_coast_frames=fusion_cfg.get("max_coast_frames", 25),
            use_rts=fusion_cfg.get("use_rts_smoother", True),
        )
        # Save to parquet cache
        fused_df = pd.DataFrame(fused_ball)
        table = pa.Table.from_pandas(fused_df)
        pq.write_table(table, fused_parquet_path)
        print(f"  Saved fused ball trajectory to cache ({fused_parquet_path})")
        
    valid_fused = int(np.sum(~np.isnan(fused_ball["fused_x"])))
    print(f"  Total valid ball frames after fusion: {valid_fused}/{TOTAL_FRAMES} ({valid_fused*100.0/TOTAL_FRAMES:.1f}%)")
    
    # 6. Feature Engineering & Dead-Ball Engine
    print("\n[5/7] Building kinematic & spatial features and dead-ball evidence...")
    features = build_frame_features(fused_ball, atd_data, st_priors, half_blocks)
    dead_ball_params = config.get("dead_ball", {})
    score_d = compute_dead_ball_scores(features, half_blocks, params=dead_ball_params)
    
    # 7. Controlled Possession & HMM Viterbi Sequence Decoding
    print("\n[6/7] Computing controlled possession emissions & running Viterbi decoding...")
    possession_params = config.get("possession", {})
    emissions = compute_emission_probabilities(features, score_d, half_blocks, params=possession_params)
    
    temporal_cfg = config.get("temporal", {})
    trans_mat = np.array(temporal_cfg.get("transition_matrix")) if "transition_matrix" in temporal_cfg else None
    init_priors = np.array(temporal_cfg.get("initial_priors")) if "initial_priors" in temporal_cfg else None
    
    smoothing_w = temporal_cfg.get("smoothing_window", 60)
    labels = decode_full_match(
        emissions,
        half_blocks,
        transition_matrix=trans_mat,
        initial_priors=init_priors,
        smoothing_window=smoothing_w,
    )
    
    # 8. Write Submissions & Validation
    print("\n[7/7] Writing submission file & validating...")
    with open(primary_output_csv, "w", encoding="utf-8") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{lbl}\n" for i, lbl in enumerate(labels))
        
    # Mirror to evaluation/results
    shutil.copyfile(primary_output_csv, mirrored_output_csv)
    
    # Validate
    sub_df = pm.read_label_csv(primary_output_csv)
    val_reason = pm.validate_submission(sub_df, TOTAL_FRAMES)
    if val_reason is not None:
        raise ValueError(f"Submission validation failed: {val_reason}")
        
    codes = pm.label_codes(sub_df)
    n_spells, mean_dur = pm.possession_spells(codes, pm.valid_runs(codes)), 0.0
    n_alive = int(np.sum((codes == pm.H) | (codes == pm.A)))
    if n_spells:
        mean_dur = n_alive / 25.0 / len(n_spells)
        
    sha256 = compute_file_sha256(primary_output_csv)
    
    print(f"\nVALIDATION PASSED:")
    print(f"  Rows: {len(sub_df)}")
    print(f"  H frames: {int(np.sum(codes == pm.H))} ({np.mean(codes == pm.H)*100.0:.1f}%)")
    print(f"  A frames: {int(np.sum(codes == pm.A))} ({np.mean(codes == pm.A)*100.0:.1f}%)")
    print(f"  D frames: {int(np.sum(codes == pm.D))} ({np.mean(codes == pm.D)*100.0:.1f}%)")
    print(f"  X frames: {int(np.sum(codes == pm.X))} ({np.mean(codes == pm.X)*100.0:.1f}%)")
    print(f"  Possession spells: {len(n_spells)}")
    print(f"  Mean possession duration: {mean_dur:.1f} s")
    print(f"  File SHA-256: {sha256}")

    # 9. Automated Data Storytelling Report & Charts
    output_files = {
        "submission": primary_output_csv,
        "mirrored_submission": mirrored_output_csv,
    }
    report_cfg = config.get("reporting", {})
    should_report = generate_report and report_cfg.get("generate_storytelling", True)
    story_res = None
    if should_report:
        print("\n[8/8] Generating Data Storytelling report (Markdown, Chart PNG, PDF)...")
        fps_val = float(config.get("geometry", {}).get("fps", 25.0))
        eval_json = os.path.join(PROJECT_ROOT, "evaluation", "reports", f"{team_slug}.json")
        story_res = generate_all_storytelling_artifacts(
            codes=codes,
            half_blocks=half_blocks,
            output_dir=output_dir,
            team_slug=team_slug,
            home_name=home_name,
            away_name=away_name,
            fps=fps_val,
            sha256=sha256,
            eval_report_path=eval_json,
        )
        st_verdict = story_res["stats"]["verdict"]
        st_teams = story_res["stats"]["teams"]
        print(
            f"  Match Story Verdict: {st_verdict['winner_name']} dominates "
            f"({st_teams['H']['alive_share_pct']:.2f}% vs {st_teams['A']['alive_share_pct']:.2f}%)"
        )
        print(f"  Storytelling MD:  {os.path.relpath(story_res['markdown_path'], PROJECT_ROOT)}")
        print(f"  Storytelling PNG: {os.path.relpath(story_res['chart_path'], PROJECT_ROOT)}")
        print(f"  Storytelling PDF: {os.path.relpath(story_res['pdf_path'], PROJECT_ROOT)}")
        output_files["storytelling_md"] = story_res["markdown_path"]
        output_files["storytelling_png"] = story_res["chart_path"]
        output_files["storytelling_pdf"] = story_res["pdf_path"]
    
    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    save_manifest(
        manifest_path=manifest_path,
        config_dict=config,
        output_files=output_files,
        extra_metadata={
            "n_spells": len(n_spells),
            "mean_duration_s": float(mean_dur),
            "execution_time_s": float(time.time() - start_time),
            "sha256": sha256,
        }
    )
    print(f"  Manifest written to: {os.path.relpath(manifest_path, PROJECT_ROOT)}")
    print(f"Done in {time.time() - start_time:.1f} seconds.\n")
    
    return {
        "output_path": primary_output_csv,
        "sha256": sha256,
        "n_spells": len(n_spells),
        "mean_duration_s": mean_dur,
        "storytelling": story_res,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run full possession inference pipeline.")
    parser.add_argument("--config", default="configs/final.yaml", help="Path to YAML configuration")
    parser.add_argument("--output", default=None, help="Target submission CSV path")
    parser.add_argument("--force-recompute", action="store_true", help="Force recomputation ignoring cache")
    parser.add_argument("--no-report", action="store_true", help="Skip Data Storytelling report generation")
    args = parser.parse_args()
    
    run_pipeline(
        config_path=args.config,
        output_path=args.output,
        force_recompute=args.force_recompute,
        generate_report=not args.no_report,
    )
