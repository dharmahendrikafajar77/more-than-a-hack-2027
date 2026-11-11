"""SportsCode XML Parser for Tactical Smart Tagging."""

import xml.etree.ElementTree as ET
import numpy as np

FPS = 25.0
TOTAL_FRAMES = 199319

# Offensive codes where 'Team' is the attacking/possessing team
OFFENSIVE_CODES = {
    "BUILD UP",
    "PROGRESSION",
    "CREATING CHANCES",
    "ATTACKING TRANSITION",
    "BALL IN FINAL THIRD",
    "BALL IN THE BOX",
    "LONG BALL",
    "GOALS",
}

# Defensive codes where 'Team' is defending, so OPPONENT is likely in possession
DEFENSIVE_CODES = {
    "DEFENDING IN ATTACKING THIRD",
    "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN DEFENSIVE THIRD",
    "DEFENSIVE TRANSITION",
}

SET_PIECE_CODES = {"SET PIECES"}


def normalize_team(team_str):
    """Normalize club names to H, A, or NONE."""
    if not team_str:
        return "NONE"
    t = team_str.strip().lower()
    if "barcelona" in t:
        return "H"
    elif "atletic" in t or "athletic" in t:
        return "A"
    return "NONE"


def opponent_team(team_code):
    """Opponent of H is A, and vice-versa."""
    if team_code == "H":
        return "A"
    elif team_code == "A":
        return "H"
    return "NONE"


def parse_smart_tagging(xml_path, fps=FPS):
    """Parse Smart Tagging XML into structured interval dictionaries."""
    root = ET.parse(xml_path).getroot()
    events = []
    
    for inst in root.iter("instance"):
        start_s = float(inst.findtext("start") or 0.0)
        end_s = float(inst.findtext("end") or 0.0)
        code = (inst.findtext("code") or "").strip().upper()
        
        tags = {}
        for lbl in inst.findall("label"):
            grp = (lbl.findtext("group") or "").strip()
            txt = (lbl.findtext("text") or "").strip()
            if grp:
                tags[grp] = txt
                
        raw_team = tags.get("Team")
        norm_team = normalize_team(raw_team)
        
        events.append({
            "code": code,
            "start_s": start_s,
            "end_s": end_s,
            "start_frame": int(round(start_s * fps)),
            "end_frame": int(round(end_s * fps)),
            "raw_team": raw_team,
            "team": norm_team,
            "half": tags.get("Half"),
            "type": (tags.get("Type") or "").strip().lower(),
            "tags": tags,
        })
        
    return events


def build_frame_priors(events, total_frames=TOTAL_FRAMES):
    """Compute frame-level prior weights for H, A, and Dead-Ball from Smart Tagging."""
    prior_h = np.zeros(total_frames, dtype=np.float32)
    prior_a = np.zeros(total_frames, dtype=np.float32)
    is_set_piece = np.zeros(total_frames, dtype=bool)
    is_kickoff = np.zeros(total_frames, dtype=bool)
    is_goal_window = np.zeros(total_frames, dtype=bool)
    
    for ev in events:
        sf = max(0, min(total_frames, ev["start_frame"]))
        ef = max(0, min(total_frames, ev["end_frame"]))
        if sf >= ef:
            continue
            
        code = ev["code"]
        team = ev["team"]
        
        if code in OFFENSIVE_CODES:
            if team == "H":
                prior_h[sf:ef] += 1.0
            elif team == "A":
                prior_a[sf:ef] += 1.0
                
        elif code in DEFENSIVE_CODES:
            opp = opponent_team(team)
            if opp == "H":
                prior_h[sf:ef] += 1.0
            elif opp == "A":
                prior_a[sf:ef] += 1.0
                
        elif code in SET_PIECE_CODES:
            is_set_piece[sf:ef] = True
            if "kick off" in ev["type"]:
                is_kickoff[sf:ef] = True
                
        elif code == "GOALS":
            is_goal_window[sf:ef] = True
            
    # Normalize priors to [0, 1] range softly
    prior_h = np.clip(prior_h / 3.0, 0.0, 1.0)
    prior_a = np.clip(prior_a / 3.0, 0.0, 1.0)
    
    return {
        "prior_h": prior_h,
        "prior_a": prior_a,
        "is_set_piece": is_set_piece,
        "is_kickoff": is_kickoff,
        "is_goal_window": is_goal_window,
    }

