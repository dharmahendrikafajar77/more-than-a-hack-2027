"""High-performance streaming parser for Metrica Automatic Tracking Data (ATD)."""

import os
import xml.etree.ElementTree as ET
import numpy as np

# Official pitch dimensions and FPS
PITCH_LENGTH = 105.3
PITCH_WIDTH = 68.0
FPS = 25.0
TOTAL_FRAMES = 199319
N_SLOTS = 46

HOME = 0
AWAY = 1


def load_team_mapping(xml_path):
    """Parse team metadata from EPTS XML. Returns (team_of_slot, home_name, away_name)."""
    root = ET.parse(xml_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) < 2:
        raise ValueError(f"Expected at least 2 teams in {xml_path}, got {len(teams)}")
    
    home_id = teams[0].get("id")
    home_name = teams[0].findtext("Name") or "Barcelona"
    away_name = teams[1].findtext("Name") or "Atletic Club"
    
    players = list(root.iter("Player"))
    if not players:
        raise ValueError(f"No player slots found in {xml_path}")
        
    team_of_slot = np.array([
        HOME if p.get("teamId") == home_id else AWAY
        for p in players
    ], dtype=np.int8)
    
    return team_of_slot, home_name, away_name


def parse_tactical_atd(raw_path, team_of_slot, total_frames=TOTAL_FRAMES):
    """Stream raw ATD text file into pre-allocated NumPy arrays.
    
    video_frame = frame_id - 1
    Coordinates scaled to metres (105.3 x 68.0).
    """
    ball_pos_m = np.full((total_frames, 2), np.nan, dtype=np.float32)
    ball_visible = np.zeros(total_frames, dtype=bool)
    
    player_pos_m = np.full((total_frames, N_SLOTS, 2), np.nan, dtype=np.float32)
    player_speed_mps = np.full((total_frames, N_SLOTS), np.nan, dtype=np.float32)
    player_active = np.zeros((total_frames, N_SLOTS), dtype=bool)
    
    active_player_count = np.zeros(total_frames, dtype=np.int16)
    mean_player_speed = np.zeros(total_frames, dtype=np.float32)
    
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                continue
                
            frame_id = int(parts[0])
            vf = frame_id - 1
            if vf < 0 or vf >= total_frames:
                continue
                
            # 1. Parse ball
            ball_str = parts[2]
            if ball_str[0] != "N":  # Not 'NaN'
                b_parts = ball_str.split(",")
                ball_pos_m[vf, 0] = float(b_parts[0]) * PITCH_LENGTH
                ball_pos_m[vf, 1] = float(b_parts[1]) * PITCH_WIDTH
                ball_visible[vf] = True
                
            # 2. Parse players
            player_strs = parts[1].split(";")
            active_cnt = 0
            speeds = []
            
            for s_idx, p_str in enumerate(player_strs):
                if s_idx >= N_SLOTS or p_str[0] == "N":
                    continue
                    
                p_items = p_str.split(",")
                px = float(p_items[0]) * PITCH_LENGTH
                py = float(p_items[1]) * PITCH_WIDTH
                player_pos_m[vf, s_idx, 0] = px
                player_pos_m[vf, s_idx, 1] = py
                player_active[vf, s_idx] = True
                active_cnt += 1
                
                if len(p_items) > 2 and p_items[2][0] != "N":
                    spd = float(p_items[2])
                    player_speed_mps[vf, s_idx] = spd
                    speeds.append(spd)
                    
            active_player_count[vf] = active_cnt
            if speeds:
                mean_player_speed[vf] = sum(speeds) / len(speeds)

    # 3. Compute ball velocity from finite differences where consecutive frames visible
    ball_speed_mps = np.full(total_frames, np.nan, dtype=np.float32)
    ball_vx = np.full(total_frames, np.nan, dtype=np.float32)
    ball_vy = np.full(total_frames, np.nan, dtype=np.float32)
    
    consec_mask = ball_visible[:-1] & ball_visible[1:]
    diffs = ball_pos_m[1:] - ball_pos_m[:-1]
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    
    ball_speed_mps[1:][consec_mask] = (dists[consec_mask] * FPS).astype(np.float32)
    ball_vx[1:][consec_mask] = (diffs[consec_mask, 0] * FPS).astype(np.float32)
    ball_vy[1:][consec_mask] = (diffs[consec_mask, 1] * FPS).astype(np.float32)
    
    return {
        "ball_pos_m": ball_pos_m,
        "ball_speed_mps": ball_speed_mps,
        "ball_vx": ball_vx,
        "ball_vy": ball_vy,
        "ball_visible": ball_visible,
        "player_pos_m": player_pos_m,
        "player_speed_mps": player_speed_mps,
        "player_active": player_active,
        "active_player_count": active_player_count,
        "mean_player_speed": mean_player_speed,
        "team_of_slot": team_of_slot,
    }

