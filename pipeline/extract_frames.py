"""
Extract evenly-spaced frames from a video file using OpenCV.
Returns a list of numpy arrays (BGR images).
"""
import cv2
import numpy as np
from typing import List

MAX_FRAMES = 60   # cap to keep processing fast


def extract_frames(video_path: str, max_frames: int = MAX_FRAMES) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError("Video contains no frames")

    # Pick evenly spaced frame indices
    step = max(1, total // max_frames)
    indices = set(range(0, total, step))

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            frames.append(frame)
        idx += 1

    cap.release()
    return frames[:max_frames]
