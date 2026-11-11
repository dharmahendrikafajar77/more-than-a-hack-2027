"""Visual Ball Detector and Candidate Filter with Homography-Guided ROI."""

import os
import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.geometry.homography import (
    project,
    unproject,
    unproject_to_metres,
    is_inside_pitch,
    PITCH_LENGTH,
    PITCH_WIDTH,
)

TOTAL_FRAMES = 199319


class BallCandidateDetector:
    """Detect ball candidates in tactical video frames with geometry filtering."""
    
    def __init__(self, homography_matrices, reprojection_errors=None):
        self.H = homography_matrices
        self.reprojection_errors = reprojection_errors
        
    def detect_candidate_in_crop(self, crop_bgr, roi_x, roi_y, matrix):
        """Analyze high-resolution ROI crop for white/bright ball-like contours."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
            
        h_crop, w_crop = crop_bgr.shape[:2]
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        
        # In 1080p tactical view, a soccer ball is roughly 6 to 14 pixels in diameter
        # Apply top-hat / adaptive thresholding for small bright circular features
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        _, thresh = cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_candidate = None
        best_score = -1.0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10 or area > 180:  # Ball pixel area filter
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / max(1, h)
            if aspect_ratio < 0.6 or aspect_ratio > 1.6:
                continue
                
            # Pixel center in full image coordinates
            u_px = float(roi_x + x + w / 2.0)
            v_px = float(roi_y + y + h / 2.0)
            
            # Project to pitch metres
            x_m, y_m = unproject_to_metres(matrix, u_px, v_px)
            if not is_inside_pitch(x_m, y_m, margin_m=3.0):
                continue
                
            # Circularity metric
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Brightness contrast in crop
            mean_val = float(np.mean(gray[y:y+h, x:x+w]))
            score = circularity * 0.5 + min(1.0, mean_val / 200.0) * 0.5
            
            if score > best_score:
                best_score = score
                best_candidate = {
                    "u_px": u_px,
                    "v_px": v_px,
                    "pitch_x_m": float(x_m),
                    "pitch_y_m": float(y_m),
                    "bbox_w": float(w),
                    "bbox_h": float(h),
                    "confidence": float(min(1.0, score)),
                }
                
        return best_candidate


def extract_cv_ball_table(decoder, homography_matrices, reprojection_errors,
                           atd_ball_pos_m, atd_ball_vis, player_mean_spd,
                           output_parquet_path, sample_rate=1, cv_enabled=True):
    """Generate or retrieve cv_ball.parquet table for all tactical frames."""
    if os.path.isfile(output_parquet_path):
        table = pq.read_table(output_parquet_path)
        if len(table) == TOTAL_FRAMES:
            return table.to_pandas()

    if not cv_enabled or decoder is None:
        df = pd.DataFrame({
            "video_frame": np.arange(TOTAL_FRAMES, dtype=np.int32),
            "detected": np.zeros(TOTAL_FRAMES, dtype=bool),
            "u_px": np.full(TOTAL_FRAMES, np.nan, dtype=np.float32),
            "v_px": np.full(TOTAL_FRAMES, np.nan, dtype=np.float32),
            "pitch_x_m": np.full(TOTAL_FRAMES, np.nan, dtype=np.float32),
            "pitch_y_m": np.full(TOTAL_FRAMES, np.nan, dtype=np.float32),
            "cv_conf": np.zeros(TOTAL_FRAMES, dtype=np.float32),
            "bbox_w": np.zeros(TOTAL_FRAMES, dtype=np.float32),
            "bbox_h": np.zeros(TOTAL_FRAMES, dtype=np.float32),
            "reproj_err": reprojection_errors if reprojection_errors is not None else np.zeros(TOTAL_FRAMES, dtype=np.float32)
        })
        os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_parquet_path)
        return df

    detector = BallCandidateDetector(homography_matrices, reprojection_errors)
    
    detected = np.zeros(TOTAL_FRAMES, dtype=bool)
    u_px = np.full(TOTAL_FRAMES, np.nan, dtype=np.float32)
    v_px = np.full(TOTAL_FRAMES, np.nan, dtype=np.float32)
    pitch_x_m = np.full(TOTAL_FRAMES, np.nan, dtype=np.float32)
    pitch_y_m = np.full(TOTAL_FRAMES, np.nan, dtype=np.float32)
    cv_conf = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    bbox_w = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    bbox_h = np.zeros(TOTAL_FRAMES, dtype=np.float32)
    
    # Process frames where ATD ball is missing during active play
    # Graded intervals: 11185..90651 and 110808..190197
    graded_active_missing = []
    for a, b in ((11185, 90651), (110808, 190197)):
        for vf in range(a, b + 1, sample_rate):
            if not atd_ball_vis[vf] and player_mean_spd[vf] >= 1.5:
                graded_active_missing.append(vf)
                
    # Stream frames and detect in contiguous chunks for high decoding speed
    last_known_m = np.array([52.65, 34.0], dtype=np.float32)
    
    if graded_active_missing:
        diffs = np.diff(graded_active_missing)
        split_pts = np.where(diffs > 1)[0] + 1
        contiguous_segments = np.split(graded_active_missing, split_pts)
        n_segs = len(contiguous_segments)
        
        print(f"Extracting CV ball detections: {len(graded_active_missing)} frames across {n_segs} segments...")
        
        for s_idx, seg in enumerate(contiguous_segments):
            start_vf = int(seg[0])
            num_vf = len(seg)
            
            # Update last known position from ATD prior to segment if possible
            if start_vf > 0 and atd_ball_vis[start_vf - 1]:
                last_known_m = atd_ball_pos_m[start_vf - 1]
                
            for vf, frame in decoder.stream_chunk(start_vf, num_vf):
                if frame is None:
                    continue
                    
                matrix = homography_matrices[vf]
                # Project last known metric pos to pixels for ROI center
                x_norm = last_known_m[0] / PITCH_LENGTH
                y_norm = last_known_m[1] / PITCH_WIDTH
                center_u, center_v = project(matrix, x_norm, y_norm)
                
                if np.isnan(center_u) or np.isnan(center_v):
                    continue
                    
                # Crop 256x256 ROI around expected position
                crop_size = 256
                x1 = max(0, int(center_u - crop_size // 2))
                y1 = max(0, int(center_v - crop_size // 2))
                x2 = min(decoder.width, x1 + crop_size)
                y2 = min(decoder.height, y1 + crop_size)
                
                crop = frame[y1:y2, x1:x2]
                cand = detector.detect_candidate_in_crop(crop, x1, y1, matrix)
                
                if cand is not None and cand["confidence"] >= 0.40:
                    detected[vf] = True
                    u_px[vf] = cand["u_px"]
                    v_px[vf] = cand["v_px"]
                    pitch_x_m[vf] = cand["pitch_x_m"]
                    pitch_y_m[vf] = cand["pitch_y_m"]
                    cv_conf[vf] = cand["confidence"]
                    bbox_w[vf] = cand["bbox_w"]
                    bbox_h[vf] = cand["bbox_h"]
                    last_known_m = np.array([cand["pitch_x_m"], cand["pitch_y_m"]], dtype=np.float32)
                    
            if (s_idx + 1) % 25 == 0 or s_idx == n_segs - 1:
                print(f"  Processed {s_idx + 1}/{n_segs} segments (detected so far: {np.sum(detected)})")

    df = pd.DataFrame({
        "video_frame": np.arange(TOTAL_FRAMES, dtype=np.int32),
        "detected": detected,
        "u_px": u_px,
        "v_px": v_px,
        "pitch_x_m": pitch_x_m,
        "pitch_y_m": pitch_y_m,
        "cv_conf": cv_conf,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "reproj_err": reprojection_errors if reprojection_errors is not None else np.zeros(TOTAL_FRAMES, dtype=np.float32)
    })
    
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_parquet_path)
    return df
