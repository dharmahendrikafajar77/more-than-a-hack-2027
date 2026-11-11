"""Confidence-aware 2D Kalman Filter and RTS Smoother for ATD + CV Ball Fusion."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PITCH_LENGTH = 105.3
PITCH_WIDTH = 68.0
FPS = 25.0
DT = 1.0 / FPS
TOTAL_FRAMES = 199319

# State: [x, y, vx, vy]^T
F = np.array([
    [1.0, 0.0, DT,  0.0],
    [0.0, 1.0, 0.0, DT ],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

H_obs = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
], dtype=np.float64)

# Process noise covariance (ball acceleration / bounces)
Q = np.diag([0.15, 0.15, 4.0, 4.0]).astype(np.float64)

# Measurement noise covariance
R_ATD = np.diag([0.25, 0.25]).astype(np.float64)  # ~0.5m std dev
R_CV_BASE = np.diag([0.64, 0.64]).astype(np.float64)  # ~0.8m std dev


class Kalman2DFusion:
    """Kalman Filter with dynamic physical outlier gating and multi-source fusion."""
    
    def __init__(self, max_speed_mps=45.0):
        self.max_speed_mps = max_speed_mps
        self.x = np.array([52.65, 34.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([10.0, 10.0, 50.0, 50.0]).astype(np.float64)
        self.last_valid_frame = -1
        
    def predict(self):
        """Standard Kalman state prediction step."""
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x.copy(), self.P.copy()
        
    def update(self, z_obs, R_meas):
        """Standard Kalman measurement update step."""
        y_residual = z_obs - H_obs @ self.x
        S = H_obs @ self.P @ H_obs.T + R_meas
        K = self.P @ H_obs.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y_residual
        I_KH = np.eye(4) - K @ H_obs
        self.P = I_KH @ self.P @ I_KH.T + K @ R_meas @ K.T
        return self.x.copy(), self.P.copy()


def run_rts_smoother(x_filt, P_filt, x_pred, P_pred):
    """Rauch-Tung-Striebel (RTS) backward smoothing pass."""
    N = len(x_filt)
    x_smooth = np.empty_like(x_filt)
    P_smooth = np.empty_like(P_filt)
    
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]
    
    for t in range(N - 2, -1, -1):
        P_p = P_pred[t + 1]
        # Invert predicted covariance stably
        try:
            inv_Pp = np.linalg.inv(P_p)
        except np.linalg.LinAlgError:
            inv_Pp = np.linalg.pinv(P_p)
            
        C = P_filt[t] @ F.T @ inv_Pp
        x_smooth[t] = x_filt[t] + C @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + C @ (P_smooth[t + 1] - P_p) @ C.T
        
    return x_smooth, P_smooth


def fuse_ball_tracking(atd_ball_pos_m, atd_ball_vis,
                       cv_ball_pos_m, cv_ball_conf, cv_detected,
                       reproj_err, half_blocks,
                       max_coast_frames=25, use_rts=True):
    """Fuse ATD tracking and CV detection across the two match halves."""
    fused_x = np.full(TOTAL_FRAMES, np.nan, dtype=np.float32)
    fused_y = np.full(TOTAL_FRAMES, np.nan, dtype=np.float32)
    fused_vx = np.full(TOTAL_FRAMES, np.nan, dtype=np.float32)
    fused_vy = np.full(TOTAL_FRAMES, np.nan, dtype=np.float32)
    fused_speed = np.full(TOTAL_FRAMES, np.nan, dtype=np.float32)
    fused_conf = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    fused_source = np.zeros(TOTAL_FRAMES, dtype=np.int8)  # 0: missing, 1: ATD, 2: CV, 3: both, 4: coast
    
    for half_num, start_f, end_f in half_blocks:
        n_block = end_f - start_f + 1
        kf = Kalman2DFusion()
        
        # Initialize at first valid observation in half
        init_pos = None
        for k in range(start_f, end_f + 1):
            if atd_ball_vis[k]:
                init_pos = atd_ball_pos_m[k]
                break
            elif cv_detected[k]:
                init_pos = cv_ball_pos_m[k]
                break
                
        if init_pos is not None:
            kf.x = np.array([init_pos[0], init_pos[1], 0.0, 0.0], dtype=np.float64)
            kf.P = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float64)
            
        x_filt = np.zeros((n_block, 4), dtype=np.float64)
        P_filt = np.zeros((n_block, 4, 4), dtype=np.float64)
        x_pred = np.zeros((n_block, 4), dtype=np.float64)
        P_pred = np.zeros((n_block, 4, 4), dtype=np.float64)
        
        sources_block = np.zeros(n_block, dtype=np.int8)
        conf_block = np.zeros(n_block, dtype=np.float32)
        valid_block = np.zeros(n_block, dtype=bool)
        
        coast_count = 0
        
        for idx, vf in enumerate(range(start_f, end_f + 1)):
            xp, Pp = kf.predict()
            x_pred[idx] = xp
            P_pred[idx] = Pp
            
            atd_ok = atd_ball_vis[vf]
            cv_ok = cv_detected[vf] and (cv_ball_conf[vf] >= 0.40)
            
            if atd_ok and cv_ok:
                z_atd = atd_ball_pos_m[vf].astype(np.float64)
                z_cv = cv_ball_pos_m[vf].astype(np.float64)
                
                # Check agreement
                dist = np.hypot(z_atd[0] - z_cv[0], z_atd[1] - z_cv[1])
                r_scale = 1.0 + float(reproj_err[vf]) * 0.5
                R_cv = R_CV_BASE * r_scale / max(0.1, float(cv_ball_conf[vf]))
                
                if dist < 3.0:  # Close agreement -> precision fusion
                    # Weighted average measurement
                    W_atd = np.linalg.inv(R_ATD)
                    W_cv = np.linalg.inv(R_cv)
                    W_sum = W_atd + W_cv
                    z_fused = np.linalg.inv(W_sum) @ (W_atd @ z_atd + W_cv @ z_cv)
                    R_fused = np.linalg.inv(W_sum)
                    xf, Pf = kf.update(z_fused, R_fused)
                    sources_block[idx] = 3
                    conf_block[idx] = 1.0
                else:  # Disagree -> pick the one closer to Kalman prediction
                    d_atd = np.hypot(z_atd[0] - xp[0], z_atd[1] - xp[1])
                    d_cv = np.hypot(z_cv[0] - xp[0], z_cv[1] - xp[1])
                    if d_atd <= d_cv:
                        xf, Pf = kf.update(z_atd, R_ATD)
                        sources_block[idx] = 1
                        conf_block[idx] = 0.85
                    else:
                        xf, Pf = kf.update(z_cv, R_cv)
                        sources_block[idx] = 2
                        conf_block[idx] = float(cv_ball_conf[vf]) * 0.8
                coast_count = 0
                valid_block[idx] = True
                
            elif atd_ok:
                z_atd = atd_ball_pos_m[vf].astype(np.float64)
                # Gating: check outlier jump
                d_jump = np.hypot(z_atd[0] - xp[0], z_atd[1] - xp[1])
                implied_speed = d_jump * FPS
                if implied_speed <= 45.0 or coast_count > 5:
                    xf, Pf = kf.update(z_atd, R_ATD)
                    sources_block[idx] = 1
                    conf_block[idx] = 0.85
                    coast_count = 0
                    valid_block[idx] = True
                else:  # Outlier jump -> coast
                    xf, Pf = xp, Pp
                    sources_block[idx] = 4
                    conf_block[idx] = max(0.1, 0.6 - coast_count * 0.05)
                    coast_count += 1
                    
            elif cv_ok:
                z_cv = cv_ball_pos_m[vf].astype(np.float64)
                r_scale = 1.0 + float(reproj_err[vf]) * 0.5
                R_cv = R_CV_BASE * r_scale / max(0.1, float(cv_ball_conf[vf]))
                xf, Pf = kf.update(z_cv, R_cv)
                sources_block[idx] = 2
                conf_block[idx] = float(cv_ball_conf[vf]) * 0.75
                coast_count = 0
                valid_block[idx] = True
                
            else:  # Neither available
                xf, Pf = xp, Pp
                coast_count += 1
                if coast_count <= max_coast_frames:
                    sources_block[idx] = 4
                    conf_block[idx] = max(0.05, 0.5 - coast_count * (0.45 / max_coast_frames))
                    valid_block[idx] = True
                else:
                    sources_block[idx] = 0
                    conf_block[idx] = 0.0
                    valid_block[idx] = False
                    
            x_filt[idx] = xf
            P_filt[idx] = Pf
            
        # RTS smoothing over the half block
        if use_rts and n_block > 2:
            x_out, P_out = run_rts_smoother(x_filt, P_filt, x_pred, P_pred)
        else:
            x_out = x_filt
            
        # Assign to global arrays
        for idx, vf in enumerate(range(start_f, end_f + 1)):
            if valid_block[idx]:
                fused_x[vf] = float(x_out[idx, 0])
                fused_y[vf] = float(x_out[idx, 1])
                fused_vx[vf] = float(x_out[idx, 2])
                fused_vy[vf] = float(x_out[idx, 3])
                fused_speed[vf] = float(np.hypot(x_out[idx, 2], x_out[idx, 3]))
                fused_conf[vf] = conf_block[idx]
                fused_source[vf] = sources_block[idx]
                
    return {
        "fused_x": fused_x,
        "fused_y": fused_y,
        "fused_vx": fused_vx,
        "fused_vy": fused_vy,
        "fused_speed": fused_speed,
        "fused_conf": fused_conf,
        "fused_source": fused_source,
    }

