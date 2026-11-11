"""Homography loader and projection transformations between pixels and metric pitch."""

import os
import numpy as np
import pandas as pd

PITCH_LENGTH = 105.3
PITCH_WIDTH = 68.0

MATRIX_COLUMNS = [f"h{row}{col}" for row in range(3) for col in range(3)]


def load_homography(csv_path):
    """Load per-frame homography matrices (N, 3, 3), reprojection error, and valid flags."""
    df = pd.read_csv(csv_path)
    matrices = df[MATRIX_COLUMNS].values.astype(np.float64).reshape(-1, 3, 3)
    reprojection_error = df["reprojection_error"].values.astype(np.float32)
    valid = df["valid"].values.astype(bool)
    
    return matrices, reprojection_error, valid


def project(matrix, x_norm, y_norm):
    """Project normalised pitch coordinates (0..1) to image pixels (u, v)."""
    x = np.asarray(x_norm, dtype=np.float64)
    y = np.asarray(y_norm, dtype=np.float64)
    
    denom = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    u = (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denom
    v = (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denom
    return u, v


def unproject(matrix, u_px, v_px):
    """Unproject image pixels (u, v) back to normalised pitch coordinates (0..1)."""
    inv_h = np.linalg.inv(matrix)
    return project(inv_h, u_px, v_px)


def unproject_to_metres(matrix, u_px, v_px):
    """Unproject image pixels directly to metric pitch coordinates (x_m, y_m)."""
    x_norm, y_norm = unproject(matrix, u_px, v_px)
    return x_norm * PITCH_LENGTH, y_norm * PITCH_WIDTH


def is_inside_pitch(x_m, y_m, margin_m=0.0):
    """Check if pitch coordinate (x_m, y_m) is within pitch bounds [-margin, length+margin]."""
    x = np.asarray(x_m)
    y = np.asarray(y_m)
    return (
        (x >= -margin_m) & (x <= PITCH_LENGTH + margin_m) &
        (y >= -margin_m) & (y <= PITCH_WIDTH + margin_m)
    )


def distance_to_boundaries(x_m, y_m):
    """Compute distance (metres) to touchlines and goal lines."""
    x = np.asarray(x_m, dtype=np.float32)
    y = np.asarray(y_m, dtype=np.float32)
    
    dist_goalline = np.minimum(np.abs(x), np.abs(PITCH_LENGTH - x))
    dist_touchline = np.minimum(np.abs(y), np.abs(PITCH_WIDTH - y))
    
    # Positive if outside pitch bounds
    outside_margin = np.maximum(
        np.maximum(-x, x - PITCH_LENGTH),
        np.maximum(-y, y - PITCH_WIDTH)
    )
    outside_margin = np.maximum(0.0, outside_margin)
    
    return dist_goalline, dist_touchline, outside_margin

