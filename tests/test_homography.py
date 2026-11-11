"""Unit tests for homography loading and projection mathematics."""

import pytest
import numpy as np
import games
from src.geometry.homography import (
    load_homography,
    project,
    unproject,
    unproject_to_metres,
    is_inside_pitch,
    distance_to_boundaries,
    PITCH_LENGTH,
    PITCH_WIDTH,
)


def test_homography_loading_and_validity():
    """Verify tactical homography has exactly 199,319 rows and is 100% valid."""
    path = games.asset_paths()["homography"]
    matrices, error, valid = load_homography(path)
    
    assert len(matrices) == 199319
    assert valid.all()
    assert np.median(error) < 1.0


def test_project_unproject_roundtrip():
    """Verify unproject(project(x, y)) returns (x, y) with high numeric precision."""
    path = games.asset_paths()["homography"]
    matrices, _, _ = load_homography(path)
    
    # Test on kickoff frame 11185
    H = matrices[11185]
    test_pts = [
        (0.5, 0.5),
        (0.0, 0.0),
        (1.0, 1.0),
        (0.25, 0.75),
    ]
    
    for x_orig, y_orig in test_pts:
        u_px, v_px = project(H, x_orig, y_orig)
        assert not np.isnan(u_px) and not np.isnan(v_px)
        
        x_rec, y_rec = unproject(H, u_px, v_px)
        assert np.isclose(x_orig, x_rec, atol=1e-5)
        assert np.isclose(y_orig, y_rec, atol=1e-5)


def test_pitch_boundaries():
    """Verify pitch bounds and boundary distance computations."""
    # Center spot (52.65, 34.0)
    assert is_inside_pitch(52.65, 34.0)
    d_goal, d_touch, out_m = distance_to_boundaries(52.65, 34.0)
    assert np.isclose(d_goal, 52.65, atol=1e-3)
    assert np.isclose(d_touch, 34.0, atol=1e-3)
    assert out_m == 0.0
    
    # Outside pitch: x = -2.0, y = 70.0
    assert not is_inside_pitch(-2.0, 70.0)
    d_goal, d_touch, out_m = distance_to_boundaries(-2.0, 70.0)
    assert out_m == 2.0  # max(2.0, 70 - 68 = 2.0)

