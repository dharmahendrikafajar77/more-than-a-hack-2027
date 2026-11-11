"""Build comprehensive per-frame feature arrays for possession and dead-ball modeling."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.geometry.homography import distance_to_boundaries, PITCH_LENGTH, PITCH_WIDTH

TOTAL_FRAMES = 199319
HOME = 0
AWAY = 1


def build_frame_features(fused_ball, atd_data, smart_tagging_priors, half_blocks):
    """Compute spatial relations, kinematics, boundaries, and tactical context per frame."""
    ball_x = fused_ball["fused_x"]
    ball_y = fused_ball["fused_y"]
    ball_speed = fused_ball["fused_speed"]
    ball_conf = fused_ball["fused_conf"]
    ball_valid = ~np.isnan(ball_x)
    
    player_pos_m = atd_data["player_pos_m"]  # (199319, 46, 2)
    player_active = atd_data["player_active"]  # (199319, 46)
    player_speed_mps = atd_data["player_speed_mps"]  # (199319, 46)
    mean_player_speed = atd_data["mean_player_speed"]
    team_of_slot = atd_data["team_of_slot"]  # (46,)
    
    home_mask = (team_of_slot == HOME)
    away_mask = (team_of_slot == AWAY)
    
    # Distance to boundaries
    dist_goalline, dist_touchline, outside_margin_m = distance_to_boundaries(
        np.nan_to_num(ball_x, nan=52.65),
        np.nan_to_num(ball_y, nan=34.0)
    )
    # If ball invalid, boundary margin defaults to 0
    outside_margin_m = np.where(ball_valid, outside_margin_m, 0.0)
    
    # Ball-to-player distance metrics
    nearest_home_dist_m = np.full(TOTAL_FRAMES, 50.0, dtype=np.float32)
    nearest_away_dist_m = np.full(TOTAL_FRAMES, 50.0, dtype=np.float32)
    home_players_within_2m = np.zeros(TOTAL_FRAMES, dtype=np.int8)
    away_players_within_2m = np.zeros(TOTAL_FRAMES, dtype=np.int8)
    home_players_within_4m = np.zeros(TOTAL_FRAMES, dtype=np.int8)
    away_players_within_4m = np.zeros(TOTAL_FRAMES, dtype=np.int8)
    
    frac_players_walking = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    own_half_separation = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    
    # Process only graded gameplay frames for efficiency
    for half_num, start_f, end_f in half_blocks:
        for vf in range(start_f, end_f + 1):
            act_slots = player_active[vf]
            if not act_slots.any():
                continue
                
            speeds = player_speed_mps[vf, act_slots]
            valid_speeds = speeds[~np.isnan(speeds)]
            if len(valid_speeds) > 0:
                frac_players_walking[vf] = np.mean(valid_speeds < 1.3)
                
            # Team half separation (kickoff detection)
            h_slots = act_slots & home_mask
            a_slots = act_slots & away_mask
            if h_slots.any() and a_slots.any():
                hx = player_pos_m[vf, h_slots, 0]
                ax = player_pos_m[vf, a_slots, 0]
                # In half 1: H attacks left->right (own half x < 52.65), A attacks right->left (own half x > 52.65)
                # In half 2: H attacks right->left (own half x > 52.65), A attacks left->right (own half x < 52.65)
                if half_num == 1:
                    h_own = np.mean(hx <= 55.0)
                    a_own = np.mean(ax >= 50.0)
                else:
                    h_own = np.mean(hx >= 50.0)
                    a_own = np.mean(ax <= 55.0)
                own_half_separation[vf] = 0.5 * (h_own + a_own)
                
            # Ball distances if ball valid
            if ball_valid[vf]:
                bx, by = ball_x[vf], ball_y[vf]
                if h_slots.any():
                    d_h = np.hypot(player_pos_m[vf, h_slots, 0] - bx,
                                   player_pos_m[vf, h_slots, 1] - by)
                    nearest_home_dist_m[vf] = float(np.min(d_h))
                    home_players_within_2m[vf] = int(np.sum(d_h <= 2.0))
                    home_players_within_4m[vf] = int(np.sum(d_h <= 4.0))
                    
                if a_slots.any():
                    d_a = np.hypot(player_pos_m[vf, a_slots, 0] - bx,
                                   player_pos_m[vf, a_slots, 1] - by)
                    nearest_away_dist_m[vf] = float(np.min(d_a))
                    away_players_within_2m[vf] = int(np.sum(d_a <= 2.0))
                    away_players_within_4m[vf] = int(np.sum(d_a <= 4.0))

    dist_margin_m = nearest_away_dist_m - nearest_home_dist_m  # Positive -> Home closer
    
    features = {
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_speed": ball_speed,
        "ball_conf": ball_conf,
        "ball_valid": ball_valid,
        "nearest_home_dist_m": nearest_home_dist_m,
        "nearest_away_dist_m": nearest_away_dist_m,
        "dist_margin_m": dist_margin_m,
        "home_within_2m": home_players_within_2m,
        "away_within_2m": away_players_within_2m,
        "home_within_4m": home_players_within_4m,
        "away_within_4m": away_players_within_4m,
        "mean_player_speed": mean_player_speed,
        "frac_players_walking": frac_players_walking,
        "own_half_separation": own_half_separation,
        "outside_margin_m": outside_margin_m,
        "dist_touchline_m": dist_touchline,
        "dist_goalline_m": dist_goalline,
        "prior_h": smart_tagging_priors["prior_h"],
        "prior_a": smart_tagging_priors["prior_a"],
        "is_set_piece": smart_tagging_priors["is_set_piece"],
        "is_kickoff": smart_tagging_priors["is_kickoff"],
        "is_goal_window": smart_tagging_priors["is_goal_window"],
    }
    return features

