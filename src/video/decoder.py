"""In-memory streaming video chunk decoder without disk image extraction."""

import cv2
import numpy as np

EXPECTED_TACTICAL_FRAMES = 199319
EXPECTED_FPS = 25.0


class VideoChunkDecoder:
    """Stream video frames in sequential memory chunks via OpenCV."""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video at: {video_path}")
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    def get_info(self):
        """Return video properties dict."""
        return {
            "total_frames": self.total_frames,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "valid_length": self.total_frames == EXPECTED_TACTICAL_FRAMES
        }
        
    def stream_chunk(self, start_frame, num_frames):
        """Read a contiguous chunk of frames starting from start_frame.
        
        Yields (video_frame_index, frame_bgr) without writing to disk.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        curr_frame = start_frame
        
        for _ in range(num_frames):
            if curr_frame >= self.total_frames:
                break
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break
            yield curr_frame, frame
            curr_frame += 1
            
    def read_single_frame(self, frame_index):
        """Seek and read a single frame."""
        if frame_index < 0 or frame_index >= self.total_frames:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        return frame if ret else None
        
    def release(self):
        """Release underlying OpenCV capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.release()

