"""
gaze_features.py

Extracts normalized gaze features from MediaPipe FaceMesh landmarks.

Goal:
- Produce a robust, calibration-friendly gaze feature vector:
  gx, gy in approximately [-0.5, +0.5] (centered ratios)
- No ML. No screen mapping. No filtering. No mouse output.

Works with:
- mediapipe.solutions.face_mesh.FaceMesh(refine_landmarks=True)

Author: EyeTalk
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


# MediaPipe FaceMesh landmark indices (common + stable)
# Left eye (from viewer perspective): corners 33 (outer), 133 (inner)
# Right eye: corners 362 (outer), 263 (inner)
# Eyelids: left upper 159, left lower 145; right upper 386, right lower 374
L_EYE_OUTER = 33
L_EYE_INNER = 133
L_EYE_UPPER = 159
L_EYE_LOWER = 145

R_EYE_OUTER = 362
R_EYE_INNER = 263
R_EYE_UPPER = 386
R_EYE_LOWER = 374

# Iris center landmarks (with refine_landmarks=True)
# Some implementations use 468/473 as "iris center". Others average the iris ring points.
L_IRIS_CENTER = 468
R_IRIS_CENTER = 473

# Iris ring points (optional averaging for stability)
# Left iris ring: 469-472 + 474-477
L_IRIS_RING = [469, 470, 471, 472, 474, 475, 476, 477]
# Right iris ring: 474? No — right ring is 474-477? In MediaPipe, both rings exist but indices differ.
# To avoid index confusion across versions, we default to center landmarks.
# If you want ring averaging for right iris too, enable it after verifying indices in your pipeline.


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _safe_div(num: float, den: float, eps: float = 1e-6) -> Optional[float]:
    if abs(den) < eps:
        return None
    return num / den


def _lm_xy(landmarks, idx: int, w: int, h: int) -> np.ndarray:
    """Return landmark pixel coords (x,y) as float array."""
    p = landmarks[idx]
    return np.array([p.x * w, p.y * h], dtype=np.float32)


@dataclass
class EyeGazeFeatures:
    valid: bool

    # Combined gaze features (centered ratios)
    gx: float  # left(-) to right(+), around 0
    gy: float  # up(-) to down(+), around 0 (note image coords: y increases downward)

    # Per-eye centered ratios (optional debug/diagnostics)
    gx_l: float
    gy_l: float
    gx_r: float
    gy_r: float

    # Extra diagnostics
    confidence: float  # simple heuristic 0..1
    debug_points: Dict[str, Tuple[int, int]]


def compute_gaze_features(
    face_landmarks,
    frame_w: int,
    frame_h: int,
    *,
    mirrored_input: bool = False,
    use_left_iris_ring_avg: bool = False,
) -> EyeGazeFeatures:
    """
    Compute normalized gaze features from landmarks.

    Parameters
    ----------
    face_landmarks: sequence of MediaPipe landmarks (results.multi_face_landmarks[0].landmark)
    frame_w, frame_h: input frame dimensions in pixels
    mirrored_input: set True if your preview is mirrored (selfie view).
                    This flips horizontal direction so moving eyes right moves cursor right.
    use_left_iris_ring_avg: if True, use average of left iris ring points instead of 468.

    Returns
    -------
    EyeGazeFeatures with valid flag and normalized gaze values.
    """

    dbg: Dict[str, Tuple[int, int]] = {}

    try:
        # --- Eye reference points (pixel coordinates) ---
        l_outer = _lm_xy(face_landmarks, L_EYE_OUTER, frame_w, frame_h)
        l_inner = _lm_xy(face_landmarks, L_EYE_INNER, frame_w, frame_h)
        l_upper = _lm_xy(face_landmarks, L_EYE_UPPER, frame_w, frame_h)
        l_lower = _lm_xy(face_landmarks, L_EYE_LOWER, frame_w, frame_h)

        r_outer = _lm_xy(face_landmarks, R_EYE_OUTER, frame_w, frame_h)
        r_inner = _lm_xy(face_landmarks, R_EYE_INNER, frame_w, frame_h)
        r_upper = _lm_xy(face_landmarks, R_EYE_UPPER, frame_w, frame_h)
        r_lower = _lm_xy(face_landmarks, R_EYE_LOWER, frame_w, frame_h)

        # --- Iris centers ---
        if use_left_iris_ring_avg:
            pts = np.stack([_lm_xy(face_landmarks, i, frame_w, frame_h) for i in L_IRIS_RING], axis=0)
            l_iris = pts.mean(axis=0)
        else:
            l_iris = _lm_xy(face_landmarks, L_IRIS_CENTER, frame_w, frame_h)

        r_iris = _lm_xy(face_landmarks, R_IRIS_CENTER, frame_w, frame_h)

        # Debug points (int for drawing)
        def to_int(p: np.ndarray) -> Tuple[int, int]:
            return int(p[0]), int(p[1])

        dbg["l_outer"] = to_int(l_outer)
        dbg["l_inner"] = to_int(l_inner)
        dbg["l_upper"] = to_int(l_upper)
        dbg["l_lower"] = to_int(l_lower)
        dbg["l_iris"] = to_int(l_iris)

        dbg["r_outer"] = to_int(r_outer)
        dbg["r_inner"] = to_int(r_inner)
        dbg["r_upper"] = to_int(r_upper)
        dbg["r_lower"] = to_int(r_lower)
        dbg["r_iris"] = to_int(r_iris)

        # --- Normalize iris position inside each eye box ---
        # X axis: inner<->outer; Y axis: upper<->lower
        # For left eye: outer is left side, inner is right side (in image coordinates).
        # For right eye: outer is right side, inner is left side.
        # We'll compute ratios using (min,max) to avoid sign mistakes.

        # Left X ratio (0..1): from outer->inner
        l_dx = float(l_inner[0] - l_outer[0])
        l_rx = _safe_div(float(l_iris[0] - l_outer[0]), l_dx)
        # Left Y ratio (0..1): from upper->lower
        l_dy = float(l_lower[1] - l_upper[1])
        l_ry = _safe_div(float(l_iris[1] - l_upper[1]), l_dy)

        # Right X ratio (0..1): from inner->outer (because right eye is mirrored in indices)
        r_dx = float(r_outer[0] - r_inner[0])
        r_rx = _safe_div(float(r_iris[0] - r_inner[0]), r_dx)
        # Right Y ratio (0..1): from upper->lower
        r_dy = float(r_lower[1] - r_upper[1])
        r_ry = _safe_div(float(r_iris[1] - r_upper[1]), r_dy)

        if l_rx is None or l_ry is None or r_rx is None or r_ry is None:
            return EyeGazeFeatures(
                valid=False,
                gx=0.0, gy=0.0,
                gx_l=0.0, gy_l=0.0, gx_r=0.0, gy_r=0.0,
                confidence=0.0,
                debug_points=dbg,
            )

        # Clamp ratios to reduce wild spikes when eyelids/corners are misdetected
        l_rx = _clamp(l_rx, 0.0, 1.0)
        l_ry = _clamp(l_ry, 0.0, 1.0)
        r_rx = _clamp(r_rx, 0.0, 1.0)
        r_ry = _clamp(r_ry, 0.0, 1.0)

        # Center around 0: -0.5..+0.5
        gx_l = float(l_rx - 0.5)
        gy_l = float(l_ry - 0.5)
        gx_r = float(r_rx - 0.5)
        gy_r = float(r_ry - 0.5)

        # Combine eyes (average)
        gx = (gx_l + gx_r) * 0.5
        gy = (gy_l + gy_r) * 0.5

        # Mirror correction if your camera preview is mirrored
        if mirrored_input:
            gx = -gx
            gx_l = -gx_l
            gx_r = -gx_r

        # Simple confidence heuristic:
        # based on eye box sizes (avoid near-zero height/width)
        l_w = abs(l_dx)
        r_w = abs(r_dx)
        l_h = abs(l_dy)
        r_h = abs(r_dy)

        # Expected: widths/ heights should be > some pixels depending on resolution
        # This is deliberately mild; runtime controller can add stronger gating.
        conf = 1.0
        if l_w < 15 or r_w < 15:
            conf *= 0.5
        if l_h < 8 or r_h < 8:
            conf *= 0.5

        return EyeGazeFeatures(
            valid=True,
            gx=gx, gy=gy,
            gx_l=gx_l, gy_l=gy_l,
            gx_r=gx_r, gy_r=gy_r,
            confidence=float(_clamp(conf, 0.0, 1.0)),
            debug_points=dbg,
        )

    except Exception:
        # If MediaPipe output shape changes or indices are missing
        return EyeGazeFeatures(
            valid=False,
            gx=0.0, gy=0.0,
            gx_l=0.0, gy_l=0.0, gx_r=0.0, gy_r=0.0,
            confidence=0.0,
            debug_points=dbg,
        )
