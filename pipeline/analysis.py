"""
Biomechanics analysis engine.
Takes pose landmarks per frame and computes exercise-specific metrics.
Returns a structured analysis dict ready to be passed to the LLM coach.
"""
import math
from typing import List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _angle(a, b, c) -> float:
    """
    Returns the angle at point B formed by A-B-C (in degrees).
    Each point is a dict with 'x', 'y'.
    """
    ax, ay = a["x"] - b["x"], a["y"] - b["y"]
    cx, cy = c["x"] - b["x"], c["y"] - b["y"]
    dot = ax * cx + ay * cy
    mag = (math.sqrt(ax**2 + ay**2) * math.sqrt(cx**2 + cy**2)) + 1e-9
    return math.degrees(math.acos(max(-1, min(1, dot / mag))))


def _midpoint(a, b) -> Dict[str, float]:
    return {"x": (a["x"] + b["x"]) / 2, "y": (a["y"] + b["y"]) / 2}


def _visible(lm: Dict) -> bool:
    return lm.get("visibility", 0) > 0.5


# ---------------------------------------------------------------------------
# Per-exercise analysis
# ---------------------------------------------------------------------------

def _analyze_squat(frames: List[Optional[Dict]]) -> Dict[str, Any]:
    valid = [f for f in frames if f]
    if not valid:
        return {"error": "No pose detected in any frame"}

    # Compute knee and hip angles per frame
    knee_angles, hip_angles = [], []
    knee_cave_flags = []

    for lm in valid:
        try:
            # Left side
            lka = _angle(lm["left_hip"], lm["left_knee"], lm["left_ankle"])
            lha = _angle(lm["left_shoulder"], lm["left_hip"], lm["left_knee"])
            # Right side
            rka = _angle(lm["right_hip"], lm["right_knee"], lm["right_ankle"])
            rha = _angle(lm["right_shoulder"], lm["right_hip"], lm["right_knee"])

            knee_angles.append((lka + rka) / 2)
            hip_angles.append((lha + rha) / 2)

            # Knee cave: knee x should track between hip and ankle x
            # If knee is significantly inside ankle, flag it
            l_cave = lm["left_knee"]["x"] > lm["left_ankle"]["x"] + 0.05
            r_cave = lm["right_knee"]["x"] < lm["right_ankle"]["x"] - 0.05
            knee_cave_flags.append(l_cave or r_cave)
        except KeyError:
            continue

    min_knee_angle = min(knee_angles) if knee_angles else None
    min_hip_angle  = min(hip_angles)  if hip_angles  else None

    # Depth: parallel = knee angle ~90°, below parallel < 90°
    depth_reached = min_knee_angle is not None and min_knee_angle <= 95
    knee_cave_detected = sum(knee_cave_flags) > len(knee_cave_flags) * 0.3

    flags = []
    if not depth_reached:
        flags.append("insufficient_depth")
    if knee_cave_detected:
        flags.append("knee_cave")
    if min_hip_angle is not None and min_hip_angle < 50:
        flags.append("excessive_forward_lean")

    return {
        "min_knee_angle_deg":   round(min_knee_angle, 1) if min_knee_angle else None,
        "min_hip_angle_deg":    round(min_hip_angle, 1)  if min_hip_angle  else None,
        "depth_reached":        depth_reached,
        "knee_cave_detected":   knee_cave_detected,
        "flags":                flags,
        "frames_analyzed":      len(valid),
    }


def _analyze_bench(frames: List[Optional[Dict]]) -> Dict[str, Any]:
    valid = [f for f in frames if f]
    if not valid:
        return {"error": "No pose detected in any frame"}

    elbow_angles, wrist_deviations = [], []

    for lm in valid:
        try:
            lea = _angle(lm["left_shoulder"], lm["left_elbow"], lm["left_wrist"])
            rea = _angle(lm["right_shoulder"], lm["right_elbow"], lm["right_wrist"])
            elbow_angles.append((lea + rea) / 2)

            # Wrist extension: wrist y significantly higher than elbow y (in image coords)
            l_ext = lm["left_wrist"]["y"] - lm["left_elbow"]["y"]
            r_ext = lm["right_wrist"]["y"] - lm["right_elbow"]["y"]
            wrist_deviations.append((l_ext + r_ext) / 2)
        except KeyError:
            continue

    min_elbow = min(elbow_angles) if elbow_angles else None
    avg_wrist_dev = sum(wrist_deviations) / len(wrist_deviations) if wrist_deviations else None

    flags = []
    if min_elbow is not None and min_elbow < 70:
        flags.append("excessive_elbow_flare")
    if avg_wrist_dev is not None and avg_wrist_dev > 0.04:
        flags.append("excessive_wrist_extension")

    return {
        "min_elbow_angle_deg":  round(min_elbow, 1)       if min_elbow      else None,
        "avg_wrist_deviation":  round(avg_wrist_dev, 4)   if avg_wrist_dev  else None,
        "flags":                flags,
        "frames_analyzed":      len(valid),
    }


def _analyze_deadlift(frames: List[Optional[Dict]]) -> Dict[str, Any]:
    valid = [f for f in frames if f]
    if not valid:
        return {"error": "No pose detected in any frame"}

    hip_angles, back_angles = [], []
    hip_rise_flags = []

    prev_hip_y, prev_shoulder_y = None, None

    for lm in valid:
        try:
            ha = _angle(lm["left_shoulder"], lm["left_hip"], lm["left_knee"])
            hip_angles.append(ha)

            # Back angle: angle between shoulder-hip vector and vertical
            sh = lm["left_shoulder"]
            hi = lm["left_hip"]
            dx, dy = sh["x"] - hi["x"], sh["y"] - hi["y"]
            back_angle = math.degrees(math.atan2(abs(dx), abs(dy)))
            back_angles.append(back_angle)

            # Hip rise: if hip y decreases faster than shoulder y, hips rising early
            hip_y = _midpoint(lm["left_hip"], lm["right_hip"])["y"]
            shoulder_y = _midpoint(lm["left_shoulder"], lm["right_shoulder"])["y"]

            if prev_hip_y is not None:
                hip_delta = prev_hip_y - hip_y
                shoulder_delta = prev_shoulder_y - shoulder_y
                if hip_delta > 0.02 and hip_delta > shoulder_delta * 1.5:
                    hip_rise_flags.append(True)
                else:
                    hip_rise_flags.append(False)

            prev_hip_y = hip_y
            prev_shoulder_y = shoulder_y
        except KeyError:
            continue

    flags = []
    if hip_rise_flags and sum(hip_rise_flags) > len(hip_rise_flags) * 0.25:
        flags.append("hips_rising_early")
    if back_angles:
        max_back = max(back_angles)
        if max_back > 45:
            flags.append("excessive_back_angle")

    return {
        "min_hip_angle_deg":    round(min(hip_angles), 1)   if hip_angles   else None,
        "max_back_angle_deg":   round(max(back_angles), 1)  if back_angles  else None,
        "hips_rising_early":    bool(hip_rise_flags and sum(hip_rise_flags) > len(hip_rise_flags) * 0.25),
        "flags":                flags,
        "frames_analyzed":      len(valid),
    }


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def analyze_lift(exercise: str, landmarks: List[Optional[Dict]]) -> Dict[str, Any]:
    analyzers = {
        "squat":     _analyze_squat,
        "bench":     _analyze_bench,
        "deadlift":  _analyze_deadlift,
    }
    return analyzers[exercise](landmarks)
