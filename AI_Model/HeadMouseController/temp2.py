"""
Filename: eye_tracker_improved.py
Creator/Author: Basel Mohamed Mostafa Sayed (with DeepSeek enhancements)
Date: 2/7/2026

Description:
    Enhanced eye tracking with smooth cursor control using modular components.
    Features multi-point calibration, improved smoothing, and better user experience.
"""

# Import necessary libraries
import cv2
import numpy as np
import os
import mediapipe as mp
import time
import math
import sys
from scipy.spatial.transform import Rotation as Rscipy
from collections import deque
import pyautogui
import threading

# --- Project-level imports (parent / root) ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import your modules
from Filters.filterManager import FilterManager
from PyautoguiController.mouse_controller import MouseController
from VisionModule.camera import IrisCamera
from VisionModule.face_utils import FaceMeshDetector
from VisionModule.head_tracking import HeadTracker

# Screen and mouse control setup
MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
CENTER_X = MONITOR_WIDTH // 2
CENTER_Y = MONITOR_HEIGHT // 2
mouse_control_enabled = False
filter_length = 10
gaze_length = 350

# --- Orbit camera state for the debug view ---
orbit_yaw   = -151.0          # radians, left/right
orbit_pitch = 00.0          # radians, up/down
orbit_radius = 1500.0       # distance from head center
orbit_fov_deg = 50.0       # horizontal FOV for projection

# --- Debug-view world freeze (pivot fixed after center calibration) ---
debug_world_frozen = False
orbit_pivot_frozen = None  # world-space point the debug camera orbits (monitor center at calib)

# Stored gaze markers on the monitor plane (as (a,b) in plane coords)
gaze_markers = []

# --- 3D monitor plane state (world space) ---
monitor_corners = None   # list of 4 world points (p0..p3)
monitor_center_w = None  # world center of the plane
monitor_normal_w = None  # world normal
units_per_cm = None      # world units per centimeter (computed at calibration)

# Calibration offsets for screen mapping
calibration_offset_yaw = 0
calibration_offset_pitch = 0

# --- Multi-point calibration state ---
calibration_mode = 'none'  # 'none', 'center', 'multi_point'
calibration_points = []
current_calibration_point = 0
calibration_samples = []

# Eye sphere tracking variables
left_sphere_locked = False
left_sphere_local_offset = None
left_calibration_nose_scale = None

right_sphere_locked = False
right_sphere_local_offset = None
right_calibration_nose_scale = None

# Use your filter manager
gaze_filter = FilterManager(filter_type="ema", alpha=0.3)

# Use your mouse controller
mouse_controller = MouseController(update_interval=0.01, enable_failsafe=False)
mouse_controller.start()

# Initialize MediaPipe FaceMesh (keep original for compatibility)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === Open webcam using your camera module ===
camera = IrisCamera(0, 640, 480)
w, h = camera.get_resolution()

# === Nose-only landmark indices ===
nose_indices = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241, 
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
                3, 248]

# ===== File writing for screen position =====
screen_position_file = "screen_position.txt"

def write_screen_position(x, y):
    """Write screen position to file, overwriting the same line"""
    with open(screen_position_file, 'w') as f:
        f.write(f"{x},{y}\n")

def _rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0],
                     [0, ca, -sa],
                     [0, sa,  ca]], dtype=float)

def _rot_y(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ ca, 0, sa],
                     [  0, 1,  0],
                     [-sa, 0, ca]], dtype=float)

def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def _focal_px(width, fov_deg):
    return 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)

def create_monitor_plane(head_center, R_final, face_landmarks, w, h, 
                         forward_hint=None, gaze_origin=None, gaze_dir=None):
    """
    Build a 60cm x 40cm plane 50cm in front of the face, in world units.
    """
    # 1) Estimate scale from chin<->forehead distance
    try:
        lm_chin = face_landmarks[152]
        lm_fore = face_landmarks[10]
        chin_w = np.array([lm_chin.x * w,  lm_chin.y * h,  lm_chin.z * w], dtype=float)
        fore_w = np.array([lm_fore.x * w,  lm_fore.y * h,  lm_fore.z * w], dtype=float)
        face_h_units = np.linalg.norm(fore_w - chin_w)
        upc = face_h_units / 15.0  # units per cm
    except Exception:
        upc = 5.0
    
    # 2) Monitor geometry in world units
    dist_cm = 50.0
    mon_w_cm, mon_h_cm = 60.0, 40.0
    half_w = (mon_w_cm * 0.5) * upc
    half_h = (mon_h_cm * 0.5) * upc

    # Head forward vector
    head_forward = -R_final[:, 2]
    if forward_hint is not None:
        head_forward = forward_hint / np.linalg.norm(forward_hint)

    # Use gaze ray intersection
    if gaze_origin is not None and gaze_dir is not None:
        gaze_dir = gaze_dir / np.linalg.norm(gaze_dir)
        plane_point = head_center + head_forward * (50.0 * upc)
        plane_normal = head_forward

        denom = np.dot(plane_normal, gaze_dir)
        if abs(denom) > 1e-6:
            t = np.dot(plane_normal, plane_point - gaze_origin) / denom
            center_w = gaze_origin + t * gaze_dir
        else:
            center_w = head_center + head_forward * (50.0 * upc)
    else:
        center_w = head_center + head_forward * (50.0 * upc)

    # Compute right/up using head orientation
    world_up = np.array([0, -1, 0], dtype=float)
    head_right = np.cross(world_up, head_forward)
    head_right /= np.linalg.norm(head_right)
    head_up = np.cross(head_forward, head_right)
    head_up /= np.linalg.norm(head_up)

    # Corners
    p0 = center_w - head_right * half_w - head_up * half_h
    p1 = center_w + head_right * half_w - head_up * half_h
    p2 = center_w + head_right * half_w + head_up * half_h
    p3 = center_w - head_right * half_w + head_up * half_h

    normal_w = head_forward / (np.linalg.norm(head_forward) + 1e-9)
    return [p0, p1, p2, p3], center_w, normal_w, upc

def compute_scale(points_3d):
    # Use average pairwise distance for robustness
    n = len(points_3d)
    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points_3d[i] - points_3d[j])
            total += dist
            count += 1
    return total / count if count > 0 else 1.0

def draw_gaze(frame, eye_center, iris_center, eye_radius, color, gaze_length):
    # Gaze vector
    gaze_direction = iris_center - eye_center
    if np.linalg.norm(gaze_direction) > 0:
        gaze_direction /= np.linalg.norm(gaze_direction)
    gaze_endpoint = eye_center + gaze_direction * gaze_length

    cv2.line(frame, tuple(int(v) for v in eye_center[:2]), 
             tuple(int(v) for v in gaze_endpoint[:2]), color, 2)

    # Segment points
    iris_offset = eye_center + gaze_direction * (1.2 * eye_radius)

    # Back segment (behind iris)
    cv2.line(frame, (int(eye_center[0]), int(eye_center[1])),
             (int(iris_offset[0]), int(iris_offset[1])), color, 1)

    # Front segment (on top of iris)
    cv2.line(frame, (int(iris_offset[0]), int(iris_offset[1])),
             (int(gaze_endpoint[0]), int(gaze_endpoint[1])), color, 1)

def draw_wireframe_cube(frame, center, R, size=80):
    # Given a center and rotation matrix, draw a cube aligned to that orientation
    right = R[:, 0]
    up = -R[:, 1]
    forward = -R[:, 2]

    hw, hh, hd = size * 1, size * 1, size * 1

    def corner(x_sign, y_sign, z_sign):
        return (center +
                x_sign * hw * right +
                y_sign * hh * up +
                z_sign * hd * forward)

    # 8 corners of the cube
    corners = [corner(x, y, z) for x in [-1, 1] for y in [1, -1] for z in [-1, 1]]
    projected = [(int(pt[0]), int(pt[1])) for pt in corners]

    # Edges connecting the corners
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for i, j in edges:
        cv2.line(frame, projected[i], projected[j], (255, 128, 0), 2)

def compute_and_draw_coordinate_box(frame, face_landmarks, indices, ref_matrix_container, color=(0, 255, 0), size=80):
    # Extract 3D positions of selected landmarks
    points_3d = np.array([
        [face_landmarks[i].x * w, face_landmarks[i].y * h, face_landmarks[i].z * w]
        for i in indices
    ])

    # Compute the average position as the center of this substructure
    center = np.mean(points_3d, axis=0)

    # Draw the raw 2D landmark points
    for i in indices:
        x, y = int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)
        cv2.circle(frame, (x, y), 3, color, -1)

    # PCA-based orientation
    centered = points_3d - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, np.argsort(-eigvals)]  # Sort by descending eigenvalue

    # Ensure the orientation matrix is right-handed
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1

    # Convert to Euler angles and re-construct rotation matrix
    r = Rscipy.from_matrix(eigvecs)
    roll, pitch, yaw = r.as_euler('zyx', degrees=False)
    R_final = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()

    # Stabilize rotation with reference matrix to avoid flipping
    if ref_matrix_container[0] is None:
        ref_matrix_container[0] = R_final.copy()
    else:
        R_ref = ref_matrix_container[0]
        for i in range(3):
            if np.dot(R_final[:, i], R_ref[:, i]) < 0:
                R_final[:, i] *= -1

    # Draw cube and orientation axes on the image
    draw_wireframe_cube(frame, center, R_final, size)

    # Draw X (green), Y (blue), Z (red) axes
    axis_length = size * 1.2
    axis_dirs = [R_final[:, 0], -R_final[:, 1], -R_final[:, 2]]
    axis_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

    for i in range(3):
        end_pt = center + axis_dirs[i] * axis_length
        cv2.line(frame, (int(center[0]), int(center[1])), 
                 (int(end_pt[0]), int(end_pt[1])), axis_colors[i], 2)

    return center, R_final, points_3d

def convert_gaze_to_screen_coordinates(combined_gaze_direction, calibration_offset_yaw, calibration_offset_pitch):
    """
    Convert 3D gaze direction vector to 2D screen coordinates
    """
    # Reference forward direction (camera looking straight ahead)
    reference_forward = np.array([0, 0, -1])

    # Normalize the gaze direction
    avg_direction = combined_gaze_direction
    if np.linalg.norm(avg_direction) > 0:
        avg_direction = avg_direction / np.linalg.norm(avg_direction)

    # Horizontal (yaw) angle from reference (project onto XZ plane)
    xz_proj = np.array([avg_direction[0], 0, avg_direction[2]])
    if np.linalg.norm(xz_proj) > 0:
        xz_proj = xz_proj / np.linalg.norm(xz_proj)
        yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
        if avg_direction[0] < 0:
            yaw_rad = -yaw_rad  # left is negative
    else:
        yaw_rad = 0

    # Vertical (pitch) angle from reference (project onto YZ plane)
    yz_proj = np.array([0, avg_direction[1], avg_direction[2]])
    if np.linalg.norm(yz_proj) > 0:
        yz_proj = yz_proj / np.linalg.norm(yz_proj)
        pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
        if avg_direction[1] > 0:
            pitch_rad = -pitch_rad  # up is positive
    else:
        pitch_rad = 0

    # Convert to degrees
    yaw_deg = np.degrees(yaw_rad)
    pitch_deg = np.degrees(pitch_rad)

    # Convert left rotations to 0-180 (from old script logic)
    if yaw_deg < 0:
        yaw_deg = -(yaw_deg)
    elif yaw_deg > 0:
        yaw_deg = - yaw_deg

    raw_yaw_deg = yaw_deg
    raw_pitch_deg = pitch_deg
    
    # Specify degrees at which screen border will be reached
    yawDegrees = 5 * 3  # x degrees left or right
    pitchDegrees = 2.0 * 2.5  # x degrees up or down

    # Apply calibration offsets
    yaw_deg += calibration_offset_yaw
    pitch_deg += calibration_offset_pitch

    # Map to full screen resolution
    screen_x = int(((yaw_deg + yawDegrees) / (2 * yawDegrees)) * MONITOR_WIDTH)
    screen_y = int(((pitchDegrees - pitch_deg) / (2 * pitchDegrees)) * MONITOR_HEIGHT)

    # Clamp screen position to monitor bounds
    screen_x = max(10, min(screen_x, MONITOR_WIDTH - 10))
    screen_y = max(10, min(screen_y, MONITOR_HEIGHT - 10))

    # Apply smoothing filter
    smooth_x, smooth_y = gaze_filter.apply_filter(screen_x, screen_y)
    
    return int(smooth_x), int(smooth_y), raw_yaw_deg, raw_pitch_deg

def start_multi_point_calibration():
    """Start multi-point calibration process."""
    global calibration_mode, calibration_points, current_calibration_point, calibration_samples
    global calibration_offset_yaw, calibration_offset_pitch
    
    calibration_mode = 'multi_point'
    calibration_points = [
        (0.5, 0.5),   # Center
        (0.1, 0.1),   # Top-left
        (0.9, 0.1),   # Top-right
        (0.1, 0.9),   # Bottom-left
        (0.9, 0.9)    # Bottom-right
    ]
    current_calibration_point = 0
    calibration_samples = []
    
    print("\n=== MULTI-POINT CALIBRATION ===")
    print("Look at each point and press SPACE when ready")
    print(f"Point 1/{len(calibration_points)}: Center")
    
    return calibration_points[current_calibration_point]

def update_calibration_with_sample(screen_x, screen_y, raw_yaw, raw_pitch):
    """Update calibration with current gaze sample."""
    global calibration_mode, calibration_points, current_calibration_point, calibration_samples
    
    if calibration_mode != 'multi_point':
        return False
    
    target_x_ratio, target_y_ratio = calibration_points[current_calibration_point]
    target_x = int(target_x_ratio * MONITOR_WIDTH)
    target_y = int(target_y_ratio * MONITOR_HEIGHT)
    
    # Store sample
    calibration_samples.append({
        'target': (target_x, target_y),
        'gaze': (screen_x, screen_y),
        'raw_yaw': raw_yaw,
        'raw_pitch': raw_pitch,
        'timestamp': time.time()
    })
    
    return True

def finish_calibration_point():
    """Finish current calibration point and move to next."""
    global calibration_mode, calibration_points, current_calibration_point
    
    if calibration_mode != 'multi_point':
        return False
    
    current_calibration_point += 1
    
    if current_calibration_point >= len(calibration_points):
        complete_multi_point_calibration()
        return False
    else:
        print(f"\nPoint {current_calibration_point + 1}/{len(calibration_points)}")
        if current_calibration_point == 1:
            print("Look at TOP-LEFT corner")
        elif current_calibration_point == 2:
            print("Look at TOP-RIGHT corner")
        elif current_calibration_point == 3:
            print("Look at BOTTOM-LEFT corner")
        elif current_calibration_point == 4:
            print("Look at BOTTOM-RIGHT corner")
        return True

def complete_multi_point_calibration():
    """Complete multi-point calibration and compute offsets."""
    global calibration_mode, calibration_offset_yaw, calibration_offset_pitch
    
    if len(calibration_samples) < 5:
        print("Not enough calibration samples collected")
        calibration_mode = 'none'
        return
    
    # Calculate average offsets
    yaw_offsets = []
    pitch_offsets = []
    
    for sample in calibration_samples:
        target_x_ratio, target_y_ratio = calibration_points[calibration_samples.index(sample) % len(calibration_points)]
        target_x = int(target_x_ratio * MONITOR_WIDTH)
        target_y = int(target_y_ratio * MONITOR_HEIGHT)
        
        # Calculate needed offsets to reach target
        yaw_offsets.append(-sample['raw_yaw'])
        pitch_offsets.append(-sample['raw_pitch'])
    
    # Use average offset
    calibration_offset_yaw = np.mean(yaw_offsets)
    calibration_offset_pitch = np.mean(pitch_offsets)
    
    print("\n=== CALIBRATION COMPLETE ===")
    print(f"Yaw offset: {calibration_offset_yaw:.2f}°")
    print(f"Pitch offset: {calibration_offset_pitch:.2f}°")
    print("Calibration applied. Mouse control ready.")
    
    calibration_mode = 'none'

def render_debug_view_orbit(
    h, w,
    head_center3d=None,
    sphere_world_l=None, scaled_radius_l=None,
    sphere_world_r=None, scaled_radius_r=None,
    iris3d_l=None, iris3d_r=None,
    left_locked=False, right_locked=False,
    landmarks3d=None,
    combined_dir=None,
    gaze_len=430,
    monitor_corners=None,
    monitor_center=None,
    monitor_normal=None,
    gaze_markers=None,
):
    if head_center3d is None:
        return

    debug = np.zeros((h, w, 3), dtype=np.uint8)

    # --- Choose orbit pivot ---
    head_w = np.asarray(head_center3d, dtype=float)

    global debug_world_frozen, orbit_pivot_frozen
    if debug_world_frozen and orbit_pivot_frozen is not None:
        pivot_w = np.asarray(orbit_pivot_frozen, dtype=float)
    else:
        if monitor_center is not None:
            pivot_w = (head_w + np.asarray(monitor_center, dtype=float)) * 0.5
        else:
            pivot_w = head_w

    # --- Camera pose (orbit around pivot_w) ---
    f_px = _focal_px(w, orbit_fov_deg)
    cam_offset = _rot_y(orbit_yaw) @ (_rot_x(orbit_pitch) @ np.array([0.0, 0.0, orbit_radius]))
    cam_pos = pivot_w + cam_offset

    up_world = np.array([0.0, -1.0, 0.0])
    fwd = _normalize(pivot_w - cam_pos)
    right = _normalize(np.cross(fwd, up_world))
    up = _normalize(np.cross(right, fwd))
    V = np.stack([right, up, fwd], axis=0)

    def project_point(P):
        Pw = np.asarray(P, dtype=float)
        Pc = V @ (Pw - cam_pos)
        if Pc[2] <= 1e-3:
            return None
        x = f_px * (Pc[0] / Pc[2]) + w * 0.5
        y = -f_px * (Pc[1] / Pc[2]) + h * 0.5
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return (int(x), int(y)), Pc[2]

    # --- helper draws ---
    def draw_poly_3d(pts, color=(0, 200, 255), thickness=2):
        projs = [project_point(p) for p in pts]
        if any(p is None for p in projs): return
        p2 = [p[0] for p in projs]
        for a, b in zip(p2, p2[1:] + [p2[0]]):
            cv2.line(debug, a, b, color, thickness)

    def draw_cross_3d(P, size=12, color=(255, 0, 255), thickness=2):
        res = project_point(P)
        if res is None: return
        (x, y), _ = res
        cv2.line(debug, (x - size, y), (x + size, y), color, thickness)
        cv2.line(debug, (x, y - size), (x, y + size), color, thickness)

    def draw_arrow_3d(P0, P1, color=(0, 200, 255), thickness=3):
        a = project_point(P0); b = project_point(P1)
        if a is None or b is None: return
        p0, p1 = a[0], b[0]
        cv2.line(debug, p0, p1, color, thickness)
        v = np.array([p1[0]-p0[0], p1[1]-p0[1]], dtype=float)
        n = np.linalg.norm(v)
        if n > 1e-3:
            v /= n
            l = np.array([-v[1], v[0]])
            ah = 10
            a1 = (int(p1[0] - v[0]*ah + l[0]*ah*0.6), int(p1[1] - v[1]*ah + l[1]*ah*0.6))
            a2 = (int(p1[0] - v[0]*ah - l[0]*ah*0.6), int(p1[1] - v[1]*ah - l[1]*ah*0.6))
            cv2.line(debug, p1, a1, color, thickness)
            cv2.line(debug, p1, a2, color, thickness)

    # --- Landmarks ---
    if landmarks3d is not None:
        for P in landmarks3d:
            res = project_point(P)
            if res is not None:
                cv2.circle(debug, res[0], 0, (200, 200, 200), -1)

    # --- Head center ---
    draw_cross_3d(head_w, size=12, color=(255, 0, 255), thickness=2)
    hc2d = project_point(head_w)
    if hc2d is not None:
        cv2.putText(debug, "Head Center", (hc2d[0][0] + 12, hc2d[0][1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

    # --- Eyes + per-eye gaze ---
    left_dir = None
    right_dir = None

    if left_locked and sphere_world_l is not None:
        res = project_point(sphere_world_l)
        if res is not None:
            (cx, cy), z = res
            r_px = max(2, int((scaled_radius_l if scaled_radius_l else 6) * f_px / max(z, 1e-3)))
            cv2.circle(debug, (cx, cy), r_px, (255, 255, 25), 1)
            if iris3d_l is not None:
                left_dir = np.asarray(iris3d_l) - np.asarray(sphere_world_l)
                if np.linalg.norm(left_dir) > 0:
                    left_dir = left_dir / np.linalg.norm(left_dir)
                p1 = project_point(np.asarray(sphere_world_l) + left_dir * gaze_len)
                if p1 is not None:
                    cv2.line(debug, (cx, cy), p1[0], (155, 155, 25), 1)

    if right_locked and sphere_world_r is not None:
        res = project_point(sphere_world_r)
        if res is not None:
            (cx, cy), z = res
            r_px = max(2, int((scaled_radius_r if scaled_radius_r else 6) * f_px / max(z, 1e-3)))
            cv2.circle(debug, (cx, cy), r_px, (25, 255, 255), 1)
            if iris3d_r is not None:
                right_dir = np.asarray(iris3d_r) - np.asarray(sphere_world_r)
                if np.linalg.norm(right_dir) > 0:
                    right_dir = right_dir / np.linalg.norm(right_dir)
                p1 = project_point(np.asarray(sphere_world_r) + right_dir * gaze_len)
                if p1 is not None:
                    cv2.line(debug, (cx, cy), p1[0], (25, 155, 155), 1)

    if left_locked and right_locked and sphere_world_l is not None and sphere_world_r is not None:
        origin_mid = (np.asarray(sphere_world_l) + np.asarray(sphere_world_r)) / 2.0
        if combined_dir is None and (left_dir is not None or right_dir is not None):
            parts = []
            if left_dir is not None:  
                parts.append(left_dir)
            if right_dir is not None: 
                parts.append(right_dir)
            if parts:
                combined_dir = np.mean(parts, axis=0)
                if np.linalg.norm(combined_dir) > 0:
                    combined_dir = combined_dir / np.linalg.norm(combined_dir)
        if combined_dir is not None:
            p0 = project_point(origin_mid)
            p1 = project_point(origin_mid + combined_dir * (gaze_len * 1.2))
            if p0 is not None and p1 is not None:
                cv2.line(debug, p0[0], p1[0], (155, 200, 10), 2)

    # --- Monitor plane ---
    if monitor_corners is not None:
        def draw_poly(points, color, thickness):
            projs = [project_point(p) for p in points]
            if any(p is None for p in projs): return
            p2 = [p[0] for p in projs]
            for a, b in zip(p2, p2[1:] + [p2[0]]):
                cv2.line(debug, a, b, color, thickness)
        draw_poly(monitor_corners, (0, 200, 255), 2)

    # --- Calibration UI overlay for multi-point ---
    if calibration_mode == 'multi_point':
        target_ratio = calibration_points[current_calibration_point]
        target_x = int(target_ratio[0] * w)
        target_y = int(target_ratio[1] * h)
        
        # Draw target circle
        cv2.circle(debug, (target_x, target_y), 20, (0, 0, 255), 2)
        cv2.circle(debug, (target_x, target_y), 5, (0, 0, 255), -1)
        
        # Draw instruction
        cv2.putText(debug, "CALIBRATION MODE", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(debug, f"Point {current_calibration_point + 1}/{len(calibration_points)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(debug, "Look at red circle, press SPACE", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Head/Eye Debug", debug)

# Buffers for gaze smoothing
combined_gaze_directions = deque(maxlen=filter_length)

# Reference matrices
R_ref_nose = [None]
R_ref_forehead = [None]

print("Eye Tracking System Initialized")
print("\nControls:")
print("  F7: Toggle mouse control (use 't' key instead)")
print("  C: Calibrate/lock eye spheres")
print("  S: Single-point calibration (center only)")
print("  M: Multi-point calibration (5 points)")
print("  SPACE: Next calibration point (in multi-point mode)")
print("  X: Add gaze marker")
print("  Q: Quit")

# F7 toggle state
last_f7_toggle = 0
f7_debounce = 0.5  # seconds

# Main loop
while camera.is_opened():
    ret = camera.get_frame()
    if ret is None:
        continue
    
    frame = ret
    
    combined_dir = None
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        # Iris indices
        left_iris_idx = 468
        right_iris_idx = 473
        left_iris = face_landmarks[left_iris_idx]
        right_iris = face_landmarks[right_iris_idx]

        # Compute and draw stabilized coordinate frame from nose region
        head_center, R_final, nose_points_3d = compute_and_draw_coordinate_box(
            frame,
            face_landmarks,
            nose_indices,
            R_ref_nose,
            color=(0, 255, 0),
            size=80
        )

        base_radius = 20

        x_iris_l = int(left_iris.x * w)
        y_iris_l = int(left_iris.y * h)
        
        # Left eye visualization
        if not left_sphere_locked:
            cv2.circle(frame, (x_iris_l, y_iris_l), 10, (255, 25, 25), 2)
        else:
            current_nose_scale = compute_scale(nose_points_3d)
            scale_ratio = current_nose_scale / left_calibration_nose_scale if left_calibration_nose_scale else 1.0
            scaled_offset = left_sphere_local_offset * scale_ratio
            sphere_world_l = head_center + R_final @ scaled_offset
            x_sphere_l, y_sphere_l = int(sphere_world_l[0]), int(sphere_world_l[1])
            scaled_radius_l = int(base_radius * scale_ratio)
            cv2.circle(frame, (x_sphere_l, y_sphere_l), scaled_radius_l, (255, 255, 25), 2)

        x_iris_r = int(right_iris.x * w)
        y_iris_r = int(right_iris.y * h)
        
        # Right eye visualization
        if not right_sphere_locked:
            cv2.circle(frame, (x_iris_r, y_iris_r), 10, (25, 255, 25), 2)
        else:
            current_nose_scale = compute_scale(nose_points_3d)
            scale_ratio_r = current_nose_scale / right_calibration_nose_scale if right_calibration_nose_scale else 1.0
            scaled_offset_r = right_sphere_local_offset * scale_ratio_r
            sphere_world_r = head_center + R_final @ scaled_offset_r
            x_sphere_r, y_sphere_r = int(sphere_world_r[0]), int(sphere_world_r[1])
            scaled_radius_r = int(base_radius * scale_ratio_r)
            cv2.circle(frame, (x_sphere_r, y_sphere_r), scaled_radius_r, (25, 255, 255), 2)

        iris_3d_left = np.array([left_iris.x * w, left_iris.y * h, left_iris.z * w])
        iris_3d_right = np.array([right_iris.x * w, right_iris.y * h, right_iris.z * w])
        
        if left_sphere_locked and right_sphere_locked:
            # Draw gaze rays
            draw_gaze(frame, sphere_world_l, iris_3d_left, scaled_radius_l, (55, 255, 0), 130)   
            draw_gaze(frame, sphere_world_r, iris_3d_right, scaled_radius_r, (55, 255, 0), 130)  

            # Calculate individual gaze directions
            left_gaze_dir = iris_3d_left - sphere_world_l
            right_gaze_dir = iris_3d_right - sphere_world_r
            
            if np.linalg.norm(left_gaze_dir) > 0:
                left_gaze_dir = left_gaze_dir / np.linalg.norm(left_gaze_dir)
            if np.linalg.norm(right_gaze_dir) > 0:
                right_gaze_dir = right_gaze_dir / np.linalg.norm(right_gaze_dir)
            
            # Combine gaze directions
            raw_combined_direction = (left_gaze_dir + right_gaze_dir) / 2
            
            # Update direction buffer for smoothing
            combined_gaze_directions.append(raw_combined_direction)

            # Smoothed direction
            if len(combined_gaze_directions) > 0:
                avg_combined_direction = np.mean(combined_gaze_directions, axis=0)
                if np.linalg.norm(avg_combined_direction) > 0:
                    avg_combined_direction = avg_combined_direction / np.linalg.norm(avg_combined_direction)

                combined_dir = avg_combined_direction

                # Convert gaze to screen coordinates
                screen_x, screen_y, raw_yaw, raw_pitch = convert_gaze_to_screen_coordinates(
                    avg_combined_direction, 
                    calibration_offset_yaw, 
                    calibration_offset_pitch
                )

                # Update calibration if in multi-point mode
                if calibration_mode == 'multi_point':
                    update_calibration_with_sample(screen_x, screen_y, raw_yaw, raw_pitch)

                # Update mouse target if control is enabled
                if mouse_control_enabled:
                    mouse_controller.set_target(screen_x, screen_y)

                # Write screen position to file
                write_screen_position(screen_x, screen_y)

                # Draw combined gaze ray for visualization
                combined_origin = (sphere_world_l + sphere_world_r) / 2
                if combined_dir is not None:
                    combined_target = combined_origin + combined_dir * gaze_length
                    cv2.line(frame, (int(combined_origin[0]), int(combined_origin[1])),
                            (int(combined_target[0]), int(combined_target[1])),
                            (255, 255, 10), 3)

                # Draw screen position on frame
                cv2.putText(frame, f"Screen: ({screen_x}, {screen_y})", 
                          (frame.shape[1] // 2 - 100, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw all landmark points
        for idx, lm in enumerate(face_landmarks):
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

        # Build 3D landmarks
        landmarks3d = None
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            landmarks3d = np.array([[p.x * w, p.y * h, p.z * w] for p in lm], dtype=float)

        # Render debug view
        render_debug_view_orbit(
            h, w,
            head_center3d=head_center,
            sphere_world_l=sphere_world_l if left_sphere_locked else None,
            scaled_radius_l=scaled_radius_l if left_sphere_locked else None,
            sphere_world_r=sphere_world_r if right_sphere_locked else None,
            scaled_radius_r=scaled_radius_r if right_sphere_locked else None,
            iris3d_l=iris_3d_left,
            iris3d_r=iris_3d_right,
            left_locked=left_sphere_locked,
            right_locked=right_sphere_locked,
            landmarks3d=landmarks3d,
            combined_dir=combined_dir,
            gaze_len=5230,
            monitor_corners=monitor_corners,
            monitor_center=monitor_center_w,
            monitor_normal=monitor_normal_w,
            gaze_markers=gaze_markers
        )

    # Draw status on main frame
    mouse_status = "MOUSE: ON" if mouse_control_enabled else "MOUSE: OFF"
    mouse_color = (0, 255, 0) if mouse_control_enabled else (0, 0, 255)
    cv2.putText(frame, mouse_status, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, mouse_color, 2)
    
    eye_status = "EYES: LOCKED" if (left_sphere_locked and right_sphere_locked) else "EYES: UNLOCKED"
    eye_color = (0, 255, 0) if (left_sphere_locked and right_sphere_locked) else (0, 120, 255)
    cv2.putText(frame, eye_status, (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
    
    cal_status = "CALIBRATING" if calibration_mode == 'multi_point' else "READY"
    cal_color = (0, 255, 255) if calibration_mode == 'multi_point' else (255, 255, 255)
    cv2.putText(frame, f"STATUS: {cal_status}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, cal_color, 2)

    cv2.imshow("Integrated Eye Tracking", frame)

    # Handle keyboard input - FIXED VERSION
    key = cv2.waitKey(1) & 0xFF
    
    # Use 't' key instead of F7 (since F7 is hard to detect)
    if key == ord('t'):
        current_time = time.time()
        if current_time - last_f7_toggle > f7_debounce:
            mouse_control_enabled = not mouse_control_enabled
            print(f"[Mouse Control] {'Enabled' if mouse_control_enabled else 'Disabled'}")
            last_f7_toggle = current_time
    
    elif key == ord('q'):
        break
    
    elif key == ord('c') and not (left_sphere_locked and right_sphere_locked):
        # Only lock if we have the necessary data
        if 'head_center' in locals() and 'R_final' in locals() and 'iris_3d_left' in locals() and 'iris_3d_right' in locals():
            current_nose_scale = compute_scale(nose_points_3d)
            
            # Lock LEFT eye
            left_sphere_local_offset = R_final.T @ (iris_3d_left - head_center)
            camera_dir_world = np.array([0, 0, 1])
            camera_dir_local = R_final.T @ camera_dir_world
            left_sphere_local_offset += base_radius * camera_dir_local
            left_calibration_nose_scale = current_nose_scale
            left_sphere_locked = True

            # Lock RIGHT eye
            right_sphere_local_offset = R_final.T @ (iris_3d_right - head_center)
            right_sphere_local_offset += base_radius * camera_dir_local
            right_calibration_nose_scale = current_nose_scale
            right_sphere_locked = True

            # Create 3D monitor plane at calibration
            sphere_world_l_calib = head_center + R_final @ left_sphere_local_offset
            sphere_world_r_calib = head_center + R_final @ right_sphere_local_offset

            # Estimate forward gaze direction
            left_dir  = iris_3d_left  - sphere_world_l_calib
            right_dir = iris_3d_right - sphere_world_r_calib
            if np.linalg.norm(left_dir) > 0:
                left_dir = left_dir / np.linalg.norm(left_dir)
            if np.linalg.norm(right_dir) > 0:
                right_dir = right_dir / np.linalg.norm(right_dir)
            forward_hint = (left_dir + right_dir) * 0.5
            if np.linalg.norm(forward_hint) > 0:
                forward_hint = forward_hint / np.linalg.norm(forward_hint)
            else:
                forward_hint = None

            gaze_origin = (sphere_world_l_calib + sphere_world_r_calib) / 2
            gaze_dir = forward_hint

            monitor_corners, monitor_center_w, monitor_normal_w, units_per_cm = create_monitor_plane(
                head_center, R_final, face_landmarks, w, h,
                forward_hint=forward_hint,
                gaze_origin=gaze_origin,
                gaze_dir=gaze_dir
            )

            debug_world_frozen = True
            orbit_pivot_frozen = monitor_center_w.copy()
            print("[Debug View] World pivot frozen at monitor center.")
            print(f"[Monitor] units_per_cm={units_per_cm:.3f}")

            print("[Both Spheres Locked] Eye sphere calibration complete.")
            print("Now press 'M' for multi-point calibration or 'S' for single-point calibration.")
    
    elif key == ord('s') and left_sphere_locked and right_sphere_locked:
        # Original single-point calibration (center only)
        if 'combined_dir' in locals() and combined_dir is not None:
            _, _, raw_yaw, raw_pitch = convert_gaze_to_screen_coordinates(
                combined_dir, 0, 0
            )
            
            calibration_offset_yaw = 0 - raw_yaw
            calibration_offset_pitch = 0 - raw_pitch
            
            print(f"[Single-Point Calibration] Offset Yaw: {calibration_offset_yaw:.2f}, Offset Pitch: {calibration_offset_pitch:.2f}")
    
    elif key == ord('m') and left_sphere_locked and right_sphere_locked:
        # Start multi-point calibration
        start_multi_point_calibration()
    
    elif key == 32 and calibration_mode == 'multi_point':  # Space bar (ASCII 32)
        # Advance to next calibration point
        if not finish_calibration_point():
            print("Calibration complete!")
    
    elif key == ord('x'):
        # Drop a marker (original functionality)
        if (monitor_corners is not None and monitor_center_w is not None and 
            monitor_normal_w is not None and left_sphere_locked and right_sphere_locked):
            current_nose_scale = compute_scale(nose_points_3d)
            scale_ratio_l = current_nose_scale / left_calibration_nose_scale if left_calibration_nose_scale else 1.0
            scale_ratio_r = current_nose_scale / right_calibration_nose_scale if right_calibration_nose_scale else 1.0
            sphere_world_l_now = head_center + R_final @ (left_sphere_local_offset * scale_ratio_l)
            sphere_world_r_now = head_center + R_final @ (right_sphere_local_offset * scale_ratio_r)

            if combined_dir is not None:
                D = combined_dir
                O = (sphere_world_l_now + sphere_world_r_now) * 0.5
                C = np.asarray(monitor_center_w, dtype=float)
                N = _normalize(np.asarray(monitor_normal_w, dtype=float))
                denom = float(np.dot(N, D))
                if abs(denom) > 1e-6:
                    t = float(np.dot(N, (C - O)) / denom)
                    if t > 0.0:
                        P = O + t * D
                        p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in monitor_corners]
                        u = p1 - p0
                        v = p3 - p0
                        u_len2 = float(np.dot(u, u))
                        v_len2 = float(np.dot(v, v))
                        if u_len2 > 1e-9 and v_len2 > 1e-9:
                            wv = P - p0
                            a = float(np.dot(wv, u) / u_len2)
                            b = float(np.dot(wv, v) / v_len2)
                            if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
                                gaze_markers.append((a, b))
                                print(f"[Marker] Added at a={a:.3f}, b={b:.3f}")

# Cleanup
camera.stop_recording()
mouse_controller.stop()
cv2.destroyAllWindows()