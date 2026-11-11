"""Controlled Possession Model and Emission Probabilities P(H), P(A), P(D)."""

import numpy as np

TOTAL_FRAMES = 199319


def softmax(x, axis=-1):
    """Numerically stable softmax over specified axis."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def compute_emission_probabilities(features, score_d, half_blocks, params=None):
    """Compute per-frame emission probabilities P(H), P(A), P(D).
    
    Applies physical control radius, dwell consistency, anti-deflection logic,
    and Smart Tagging directional priors.
    """
    p = params or {}
    control_radius = p.get("control_radius_m", 2.5)
    dwell_threshold = p.get("dwell_threshold_frames", 25)
    st_weight = p.get("st_prior_weight", 1.2)
    state_persistence = p.get("persistence_weight", 2.0)
    
    nh_dist = features["nearest_home_dist_m"]
    na_dist = features["nearest_away_dist_m"]
    prior_h = features["prior_h"]
    prior_a = features["prior_a"]
    ball_valid = features["ball_valid"]
    
    score_h = np.full(TOTAL_FRAMES, -10.0, dtype=np.float32)
    score_a = np.full(TOTAL_FRAMES, -10.0, dtype=np.float32)
    
    for half_num, start_f, end_f in half_blocks:
        current_possessor = "NONE"  # 'H' or 'A'
        opponent_candidate = "NONE"
        opponent_candidate_frames = 0
        loose_frames = 0
        
        for vf in range(start_f, end_f + 1):
            sd = score_d[vf]
            
            # If dead ball is dominant, reset active possession state
            if sd > 1.0:
                current_possessor = "NONE"
                opponent_candidate = "NONE"
                opponent_candidate_frames = 0
                loose_frames = 0
                score_h[vf] = -5.0
                score_a[vf] = -5.0
                continue
                
            dh = nh_dist[vf]
            da = na_dist[vf]
            
            # Base proximity score via exponential decay
            sh = np.exp(-dh / control_radius) * 4.0
            sa = np.exp(-da / control_radius) * 4.0
            
            # Margin bonus
            margin = da - dh  # Positive = Home closer
            sh += np.clip(margin / 2.0, -2.0, 2.0)
            sa -= np.clip(margin / 2.0, -2.0, 2.0)
            
            # Smart Tagging prior
            sh += prior_h[vf] * st_weight
            sa += prior_a[vf] * st_weight
            
            # Anti-deflection / Dwell consistency logic:
            if ball_valid[vf]:
                if dh < control_radius and dh + 0.5 < da:
                    frame_team = "H"
                elif da < control_radius and da + 0.5 < dh:
                    frame_team = "A"
                else:
                    frame_team = "NONE"
            else:
                frame_team = "NONE"
                
            if current_possessor == "NONE":
                if frame_team != "NONE":
                    current_possessor = frame_team
            else:
                other_team = "A" if current_possessor == "H" else "H"
                if frame_team == other_team:
                    if opponent_candidate == other_team:
                        opponent_candidate_frames += 1
                    else:
                        opponent_candidate = other_team
                        opponent_candidate_frames = 1
                        
                    # Turnover occurs only after sustained dwell frames
                    if opponent_candidate_frames >= dwell_threshold:
                        current_possessor = other_team
                        opponent_candidate_frames = 0
                else:
                    opponent_candidate_frames = max(0, opponent_candidate_frames - 1)
                    
                # Loose ball clears persistence after 35 frames (1.4s)
                if frame_team == "NONE":
                    loose_frames += 1
                    if loose_frames > 35:
                        current_possessor = "NONE"
                else:
                    loose_frames = 0
                    
            # Apply state persistence to preserve spell continuity
            if current_possessor == "H":
                sh += state_persistence
            elif current_possessor == "A":
                sa += state_persistence
                
            score_h[vf] = sh
            score_a[vf] = sa
            
    # Combine [S_H, S_A, S_D] into probabilities via softmax
    logits = np.stack([score_h, score_a, score_d], axis=-1)  # (N, 3)
    probs = softmax(logits, axis=-1)
    
    return {
        "p_home": probs[:, 0].astype(np.float32),
        "p_away": probs[:, 1].astype(np.float32),
        "p_dead": probs[:, 2].astype(np.float32),
        "score_h": score_h,
        "score_a": score_a,
        "score_d": score_d,
    }

