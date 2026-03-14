"""
Run MediaPipe Pose on a list of frames.
Returns a list of landmark dicts, one per frame (None if no person detected).
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Optional, Dict, Any

mp_pose = mp.solutions.pose

# Named landmark indices for readability
LANDMARKS = {
    "nose":            0,
    "left_shoulder":  11, "right_shoulder": 12,
    "left_elbow":     13, "right_elbow":    14,
    "left_wrist":     15, "right_wrist":    16,
    "left_hip":       23, "right_hip":      24,
    "left_knee":      25, "right_knee":     26,
    "left_ankle":     27, "right_ankle":    28,
    "left_heel":      29, "right_heel":     30,
    "left_foot":      31, "right_foot":     32,
}


def _landmark_to_dict(lm) -> Dict[str, float]:
    return {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}


def run_pose_estimation(
    frames: List[np.ndarray],
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> List[Optional[Dict[str, Any]]]:
    results = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if not result.pose_landmarks:
                results.append(None)
                continue

            lms = result.pose_landmarks.landmark
            named = {
                name: _landmark_to_dict(lms[idx])
                for name, idx in LANDMARKS.items()
            }
            results.append(named)

    return results
