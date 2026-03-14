"""
Run MediaPipe Pose on a list of frames using the new Tasks API (mediapipe 0.10.30+).
Returns a list of landmark dicts, one per frame (None if no person detected).
"""
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerOptions
from typing import List, Optional, Dict, Any
import urllib.request
import os

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_PATH = "/tmp/pose_landmarker.task"


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def _landmark_to_dict(lm) -> Dict[str, float]:
    return {
        "x": lm.x,
        "y": lm.y,
        "z": lm.z,
        "visibility": lm.visibility if hasattr(lm, "visibility") else 1.0
    }


def run_pose_estimation(
    frames: List[np.ndarray],
) -> List[Optional[Dict[str, Any]]]:
    _ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    results = []
    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                results.append(None)
                continue

            lms = result.pose_landmarks[0]
            named = {
                LANDMARK_NAMES[i]: _landmark_to_dict(lms[i])
                for i in range(min(len(lms), len(LANDMARK_NAMES)))
            }
            results.append(named)

    return results
