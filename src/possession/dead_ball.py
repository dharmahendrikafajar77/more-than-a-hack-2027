"""Intelligent Dead-Ball Engine combining physical, kinematic, and tactical context."""

import numpy as np

TOTAL_FRAMES = 199319


def compute_dead_ball_scores(features, half_blocks, params=None):
    """Compute per-frame log-evidence score S_D for dead ball / stoppage."""
    p = params or {}
    out_margin_weight = p.get("out_margin_weight", 3.0)
    stillness_weight = p.get("stillness_weight", 4.0)
    goal_stoppage_weight = p.get("goal_stoppage_weight", 4.5)
    kickoff_stoppage_weight = p.get("kickoff_stoppage_weight", 6.0)
    
    score_d = np.full(TOTAL_FRAMES, -10.0, dtype=np.float32)
    
    outside_m = features["outside_margin_m"]
    mean_speed = features["mean_player_speed"]
    frac_walking = features["frac_players_walking"]
    own_half_sep = features["own_half_separation"]
    is_goal = features["is_goal_window"]
    is_kickoff = features["is_kickoff"]
    ball_valid = features["ball_valid"]
    nh_dist = features["nearest_home_dist_m"]
    na_dist = features["nearest_away_dist_m"]
    
    for half_num, start_f, end_f in half_blocks:
        # Detect actual kickoff frame inside kickoff window algorithmically
        # (when mean player speed rises or ball starts moving)
        kickoff_done = False
        
        for vf in range(start_f, end_f + 1):
            s_d = -1.0  # Base live play bias
            
            # 1. Out of pitch physical evidence
            if outside_m[vf] > 0.0:
                s_d += 6.0 + outside_m[vf] * out_margin_weight
                
            # 2. Collective player stillness (stoppage)
            if mean_speed[vf] < 1.35 and frac_walking[vf] > 0.65:
                s_d += 5.0 + (1.35 - mean_speed[vf]) * stillness_weight
                if mean_speed[vf] < 1.0:
                    s_d += 3.0
                    
            # 3. Missing ball during low player speed -> stoppage
            if not ball_valid[vf] and mean_speed[vf] < 1.5:
                s_d += 5.0
                
            # 4. Pre-kickoff state detection (handles start of half 1 and half 2)
            if not kickoff_done:
                if is_kickoff[vf] or (own_half_sep[vf] >= 0.80 and mean_speed[vf] < 1.45):
                    s_d += kickoff_stoppage_weight
                if is_kickoff[vf] and mean_speed[vf] >= 1.6 and (nh_dist[vf] < 2.0 or na_dist[vf] < 2.0):
                    kickoff_done = True
            else:
                # Can re-arm kickoff detection if a goal occurs
                if is_goal[vf] and own_half_sep[vf] >= 0.80 and mean_speed[vf] < 1.3:
                    kickoff_done = False
                    
            # 5. Stoppage after goal (celebrations, VAR, resets)
            if is_goal[vf] and mean_speed[vf] < 1.4:
                s_d += goal_stoppage_weight
                
            # 6. Active gameplay suppressor: if players are sprinting near valid ball, suppress D
            if ball_valid[vf] and mean_speed[vf] >= 1.8 and (nh_dist[vf] < 2.5 or na_dist[vf] < 2.5):
                s_d -= 5.0
                
            score_d[vf] = s_d
            
    return score_d

