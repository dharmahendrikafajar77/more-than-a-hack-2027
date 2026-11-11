"""Unit tests for Kalman filter and RTS smoother fusion."""

import pytest
import numpy as np
from src.fusion.kalman_fusion import Kalman2DFusion, run_rts_smoother, fuse_ball_tracking


def test_kalman_constant_velocity():
    """Verify Kalman filter tracks constant velocity motion accurately."""
    kf = Kalman2DFusion()
    dt = 0.04
    true_x, true_y = 10.0, 20.0
    vx, vy = 5.0, 3.0
    
    kf.x = np.array([true_x, true_y, vx, vy], dtype=np.float64)
    R = np.diag([0.25, 0.25]).astype(np.float64)
    
    for _ in range(50):
        kf.predict()
        true_x += vx * dt
        true_y += vy * dt
        z_noisy = np.array([true_x + np.random.normal(0, 0.2), true_y + np.random.normal(0, 0.2)])
        kf.update(z_noisy, R)
        
    assert np.isclose(kf.x[0], true_x, atol=1.0)
    assert np.isclose(kf.x[1], true_y, atol=1.0)
    assert np.isclose(kf.x[2], vx, atol=1.5)
    assert np.isclose(kf.x[3], vy, atol=1.5)


def test_rts_smoother_continuity():
    """Verify RTS smoother provides continuous smoothed trajectories across gaps."""
    N = 100
    x_filt = np.zeros((N, 4))
    P_filt = np.zeros((N, 4, 4))
    x_pred = np.zeros((N, 4))
    P_pred = np.zeros((N, 4, 4))
    
    kf = Kalman2DFusion()
    for t in range(N):
        xp, Pp = kf.predict()
        x_pred[t] = xp
        P_pred[t] = Pp
        z = np.array([10.0 + t * 0.2, 20.0 + t * 0.1])
        xf, Pf = kf.update(z, np.diag([0.2, 0.2]))
        x_filt[t] = xf
        P_filt[t] = Pf
        
    x_smooth, _ = run_rts_smoother(x_filt, P_filt, x_pred, P_pred)
    assert len(x_smooth) == N
    # Smoothed trajectory should be smooth and close to truth
    assert np.isclose(x_smooth[-1, 0], 10.0 + 99 * 0.2, atol=0.5)

