"""Test sweep of dwell and smoothing window."""

import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scratch.test_tune_rules import features, half_blocks, evaluate_predictions, TOTAL_FRAMES
import evaluation.possession_metrics as pm

outside_m = features['outside_margin_m']
mean_speed = features['mean_player_speed']
frac_walking = features['frac_players_walking']
own_half_sep = features['own_half_separation']
is_goal = features['is_goal_window']
is_kickoff = features['is_kickoff']
ball_valid = features['ball_valid']
nh_dist = features['nearest_home_dist_m']
na_dist = features['nearest_away_dist_m']
prior_h = features['prior_h']
prior_a = features['prior_a']

print(f"{'Dwell':<6} {'Smooth':<7} {'Macro-F1':<10} {'Spell-F1':<10} {'Poss':<6} {'Dur (s)':<8} {'Alive/Dead F1':<13}")
print("-" * 65)

for dwell_w in [15, 25, 35]:
    for smooth_w in [35, 50, 75]:
        labels = np.full(TOTAL_FRAMES, 'X', dtype='<U1')
        
        for _, sf, ef in half_blocks:
            n_b = ef - sf + 1
            codes = np.full(n_b, pm.D, dtype=np.int8)
            
            # Determine dead ball
            out_pitch = outside_m[sf:ef+1] > 0.0
            stillness = (mean_speed[sf:ef+1] < 1.35) & (frac_walking[sf:ef+1] > 0.65)
            lost_ball_slow = (~ball_valid[sf:ef+1]) & (mean_speed[sf:ef+1] < 1.5)
            pre_kick = (own_half_sep[sf:ef+1] >= 0.80) & (mean_speed[sf:ef+1] < 1.45)
            goal_celeb = is_goal[sf:ef+1] & (mean_speed[sf:ef+1] < 1.4)
            is_dead = out_pitch | stillness | lost_ball_slow | pre_kick | goal_celeb
            
            # Live play: nearest player within 3.0m
            nh = nh_dist[sf:ef+1]
            na = na_dist[sf:ef+1]
            bvis = ball_valid[sf:ef+1]
            
            curr_team = pm.D
            cand_team = pm.D
            cand_cnt = 0
            block_labels = np.full(n_b, pm.D, dtype=np.int8)
            
            for i in range(n_b):
                if is_dead[i]:
                    curr_team = pm.D
                    cand_team = pm.D
                    cand_cnt = 0
                    block_labels[i] = pm.D
                    continue
                    
                if bvis[i]:
                    if nh[i] <= 3.0 and nh[i] + 0.3 < na[i]:
                        inst_team = pm.H
                    elif na[i] <= 3.0 and na[i] + 0.3 < nh[i]:
                        inst_team = pm.A
                    else:
                        inst_team = curr_team
                else:
                    inst_team = curr_team
                    
                if curr_team == pm.D:
                    if inst_team in (pm.H, pm.A):
                        curr_team = inst_team
                elif inst_team != curr_team and inst_team in (pm.H, pm.A):
                    if cand_team == inst_team:
                        cand_cnt += 1
                        if cand_cnt >= dwell_w:
                            curr_team = inst_team
                            cand_cnt = 0
                    else:
                        cand_team = inst_team
                        cand_cnt = 1
                else:
                    cand_cnt = max(0, cand_cnt - 1)
                    
                block_labels[i] = curr_team
                
            w = smooth_w
            h_cum = np.concatenate(([0], np.cumsum(block_labels == pm.H)))
            a_cum = np.concatenate(([0], np.cumsum(block_labels == pm.A)))
            d_cum = np.concatenate(([0], np.cumsum(block_labels == pm.D)))
            
            idx = np.arange(n_b)
            lo = np.maximum(0, idx - w)
            hi = np.minimum(n_b, idx + w + 1)
            
            h_cnt = h_cum[hi] - h_cum[lo]
            a_cnt = a_cum[hi] - a_cum[lo]
            d_cnt = d_cum[hi] - d_cum[lo]
            
            final_block = np.full(n_b, 'D', dtype='<U1')
            h_win = (h_cnt >= a_cnt) & (h_cnt >= d_cnt) & (h_cnt > 0)
            a_win = (a_cnt > h_cnt) & (a_cnt >= d_cnt) & (a_cnt > 0)
            final_block[h_win] = 'H'
            final_block[a_win] = 'A'
            
            labels[sf:ef+1] = final_block
            
        res = evaluate_predictions(labels)
        print(f"{dwell_w:<6} {smooth_w:<7} {res['macro_f1']:<10.4f} {res['spell_f1']:<10.4f} {res['n_spells']:<6} {res['mean_dur']:<8.1f} {res['alive_dead_f1']:<13.4f}")

