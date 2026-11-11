"""Unit tests for tracking alignment and smart tagging parsing."""

import pytest
import numpy as np
import games
from src.tracking.atd_loader import load_team_mapping, parse_tactical_atd, TOTAL_FRAMES, N_SLOTS
from src.tracking.smart_tagging import parse_smart_tagging, build_frame_priors


def test_asset_paths_exist():
    """Verify all 5 tactical assets exist and can be located by games.require."""
    paths = games.require(['video', 'atd_meta', 'atd_raw', 'homography', 'smart_tagging'])
    assert len(paths) == 5
    for p in paths:
        assert p and p.endswith(('.mp4', '.xml', '.txt', '.csv'))


def test_team_mapping():
    """Verify 46 slots mapped to Home (0) and Away (1)."""
    meta_path = games.asset_paths()["atd_meta"]
    team_of_slot, home_name, away_name = load_team_mapping(meta_path)
    
    assert len(team_of_slot) == N_SLOTS
    assert set(np.unique(team_of_slot)).issubset({0, 1})
    assert "Barcelona" in home_name
    assert "Atletic" in away_name or "Athletic" in away_name


def test_smart_tagging_kickoffs():
    """Verify Smart Tagging contains kick-offs for both halves."""
    st_path = games.asset_paths()["smart_tagging"]
    events = parse_smart_tagging(st_path)
    assert len(events) == 638
    
    priors = build_frame_priors(events)
    assert len(priors["prior_h"]) == TOTAL_FRAMES
    assert len(priors["is_kickoff"]) == TOTAL_FRAMES
    
    # 1st half kickoff frame (11185) and 2nd half kickoff frame (113742) must be inside kickoff tags
    assert priors["is_kickoff"][11185] == True
    assert priors["is_kickoff"][113742] == True

