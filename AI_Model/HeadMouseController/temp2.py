import cv2
import numpy as np
import os
import mediapipe as mp
import time
import math
from scipy.spatial.transform import Rotation as Rscipy
from collections import deque
import pyautogui
import threading
import keyboard

# ============================================================
#                    STABILIZATION PARAMETERS (NEW)
# ============================================================
# Direction smoothing / outlier gating
DIR_OUTLIER_DEG = 5.0                 # reject raw gaze direction if it deviates too much from recent mean
DIR_ADAPTIVE_ALPHA_MIN = 0.18         # stronger smoothing when stable (lower alpha)
DIR_ADAPTIVE_ALPHA_MAX = 0.55         # more responsive when moving (higher alpha)

# Screen-space smoothing (tiny, low-latency)
SCREEN_EMA_ALPHA_SLOW = 0.18          # stable gaze -> heavier smoothing
SCREEN_EMA_ALPHA_FAST = 0.55          # fast movement -> lighter smoothing
PIXEL_DEADZONE = 8                    # pixel deadzone after smoothing (reduces micro-jitter)
MAX_SCREEN_STEP = 220                 # clamp one-frame cursor spikes (px)

# Angle mapping sensitivity (keep your "angle to screen" approach)
YAW_DEGREES_RANGE = 15.0              # was 5*3
PITCH_DEGREES_RANGE = 5.0             # was 2.0*2.5

# File writing (reduce I/O jitter)
SCREEN_WRITE_HZ = 30                  # limit writes per second

# ============================================================
# Screen and mouse control setup (from old script)
# ============================================================
MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
CENTER_X = MONITOR_WIDTH // 2
CENTER_Y = MONITOR_HEIGHT // 2
mouse_control_enabled = False
filter_length = 10
gaze_length = 350

# --- Orbit camera state for the debug view ---
orbit_yaw   = -151.0
orbit_pitch = 00.0
orbit_radius = 1500.0
orbit_fov_deg = 50.0

# --- Debug-view world freeze (pivot fixed after center calibration) ---
debug_world_frozen = False
orbit_pivot_frozen = None

# Stored gaze markers on the monitor plane (as (a,b) in plane coords)
gaze_markers = []

# --- 3D monitor plane state (world space) ---
monitor_corners = None
monitor_center_w = None
monitor_normal_w = None
units_per_cm = None

# Shared mouse target position
mouse_target = [CENTER_X, CENTER_Y]
mouse_lock = threading.Lock()

# Calibration offsets for screen mapping
calibration_offset_yaw = 0
calibration_offset_pitch = 0

# --- Simple monitor-edge calibration state ---
calib_step = 0

# Buffers to store recent gaze data for smoothing
combined_gaze_directions = deque(maxlen=filter_length)

# reference matrices to fix coordinate flipping issue
R_ref_nose = [None]
R_ref_forehead = [None]
calibration_nose_scale = None

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === Open webcam ===
cap = cv2.VideoCapture(0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === Nose-only landmark indices (for stable up/down eye sphere tracking) ===
nose_indices = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241,
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
                3, 248]

# ===== File writing for screen position =====
screen_position_file = r"G:\Thebes Higher Institue of Engineering\Final Project\EyeTalk\AI_Model\HeadMouseController\screen_position.txt"
_last_write_t = 0.0
_write_period = 1.0 / float(max(1, SCREEN_WRITE_HZ))

def write_screen_position_rate_limited(x, y):
    """Write screen position to file at most SCREEN_WRITE_HZ (reduces I/O-induced jitter)."""
    global _last_write_t
    now = time.time()
    if (now - _last_write_t) < _write_period:
        return
    _last_write_t = now
    try:
        with open(screen_position_file, 'w') as f:
            f.write(f"{x},{y}\n")
    except Exception as e:
        # Don't let I/O failures affect tracking loop stability
        pass

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

def angle_between_deg(a, b):
    """Angle between unit vectors a and b in degrees."""
    a = _normalize(a)
    b = _normalize(b)
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))

def create_monitor_plane(head_center, R_final, face_landmarks, w, h,
                         forward_hint=None, gaze_origin=None, gaze_dir=None):
    """
    Build a 60cm x 40cm plane 50cm in front of the face, in world units.
    Monitor is oriented horizontally like a real monitor (top edge parallel to global X-axis).
    """
    # 1) Estimate scale from chin<->forehead distance
    try:
        lm_chin = face_landmarks[152]
        lm_fore = face_landmarks[10]
        chin_w = np.array([lm_chin.x * w,  lm_chin.y * h,  lm_chin.z * w], dtype=float)
        fore_w = np.array([lm_fore.x * w,  lm_fore.y * h,  lm_fore.z * w], dtype=float)
        face_h_units = np.linalg.norm(fore_w - chin_w)
        upc = face_h_units / 15.0
    except Exception:
        upc = 5.0

    dist_cm = 50.0
    mon_w_cm, mon_h_cm = 60.0, 40.0
    half_w = (mon_w_cm * 0.5) * upc
    half_h = (mon_h_cm * 0.5) * upc

    head_forward = -R_final[:, 2]
    if forward_hint is not None:
        head_forward = forward_hint / np.linalg.norm(forward_hint)

    # use gaze ray intersection if possible
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

    world_up = np.array([0, -1, 0], dtype=float)
    head_right = np.cross(world_up, head_forward)
    head_right /= np.linalg.norm(head_right)
    head_up = np.cross(head_forward, head_right)
    head_up /= np.linalg.norm(head_up)

    p0 = center_w - head_right * half_w - head_up * half_h
    p1 = center_w + head_right * half_w - head_up * half_h
    p2 = center_w + head_right * half_w + head_up * half_h
    p3 = center_w - head_right * half_w + head_up * half_h

    normal_w = head_forward / (np.linalg.norm(head_forward) + 1e-9)
    return [p0, p1, p2, p3], center_w, normal_w, upc

def update_orbit_from_keys():
    global orbit_yaw, orbit_pitch, orbit_radius
    yaw_step   = math.radians(1.5)
    pitch_step = math.radians(1.5)
    zoom_step  = 12.0

    changed = False

    if keyboard.is_pressed('j'):
        orbit_yaw -= yaw_step; changed = True
    if keyboard.is_pressed('l'):
        orbit_yaw += yaw_step; changed = True
    if keyboard.is_pressed('i'):
        orbit_pitch += pitch_step; changed = True
    if keyboard.is_pressed('k'):
        orbit_pitch -= pitch_step; changed = True

    if keyboard.is_pressed('['):
        orbit_radius += zoom_step; changed = True
    if keyboard.is_pressed(']'):
        orbit_radius = max(80.0, orbit_radius - zoom_step); changed = True

    if keyboard.is_pressed('r'):
        orbit_yaw = 0.0
        orbit_pitch = 0.0
        orbit_radius = 600.0
        changed = True

    orbit_pitch = max(math.radians(-89), min(math.radians(89), orbit_pitch))
    orbit_radius = max(80.0, orbit_radius)

    if changed:
        print(f"[Orbit Debug] yaw={math.degrees(orbit_yaw):.2f}°, "
              f"pitch={math.degrees(orbit_pitch):.2f}°, "
              f"radius={orbit_radius:.2f}, "
              f"fov={orbit_fov_deg:.1f}°")

def compute_scale(points_3d):
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
    gaze_direction = iris_center - eye_center
    gaze_direction /= np.linalg.norm(gaze_direction)
    gaze_endpoint = eye_center + gaze_direction * gaze_length

    cv2.line(frame, tuple(int(v) for v in eye_center[:2]), tuple(int(v) for v in gaze_endpoint[:2]), color, 2)

    iris_offset = eye_center + gaze_direction * (1.2 * eye_radius)

    cv2.line(frame, (int(eye_center[0]), int(eye_center[1])),
             (int(iris_offset[0]), int(iris_offset[1])), color, 1)

    up_dir = np.array([0, -1, 0])
    right_dir = np.cross(gaze_direction, up_dir)
    if np.linalg.norm(right_dir) < 1e-6:
        right_dir = np.array([1, 0, 0])
    up_dir = np.cross(right_dir, gaze_direction)
    up_dir /= np.linalg.norm(up_dir)
    right_dir /= np.linalg.norm(right_dir)

    cv2.line(frame, (int(iris_offset[0]), int(iris_offset[1])),
             (int(gaze_endpoint[0]), int(gaze_endpoint[1])), color, 1)

def draw_wireframe_cube(frame, center, R, size=80):
    right = R[:, 0]
    up = -R[:, 1]
    forward = -R[:, 2]

    hw, hh, hd = size * 1, size * 1, size * 1

    def corner(x_sign, y_sign, z_sign):
        return (center +
                x_sign * hw * right +
                y_sign * hh * up +
                z_sign * hd * forward)

    corners = [corner(x, y, z) for x in [-1, 1] for y in [1, -1] for z in [-1, 1]]
    projected = [(int(pt[0]), int(pt[1])) for pt in corners]

    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for i, j in edges:
        cv2.line(frame, projected[i], projected[j], (255, 128, 0), 2)

def compute_and_draw_coordinate_box(frame, face_landmarks, indices, ref_matrix_container, color=(0, 255, 0), size=80):
    points_3d = np.array([
        [face_landmarks[i].x * w, face_landmarks[i].y * h, face_landmarks[i].z * w]
        for i in indices
    ])

    center = np.mean(points_3d, axis=0)

    for i in indices:
        x, y = int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)
        cv2.circle(frame, (x, y), 3, color, -1)

    centered = points_3d - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, np.argsort(-eigvals)]

    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1

    r = Rscipy.from_matrix(eigvecs)
    roll, pitch, yaw = r.as_euler('zyx', degrees=False)
    R_final = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()

    if ref_matrix_container[0] is None:
        ref_matrix_container[0] = R_final.copy()
    else:
        R_ref = ref_matrix_container[0]
        for i in range(3):
            if np.dot(R_final[:, i], R_ref[:, i]) < 0:
                R_final[:, i] *= -1

    draw_wireframe_cube(frame, center, R_final, size)

    axis_length = size * 1.2
    axis_dirs = [R_final[:, 0], -R_final[:, 1], -R_final[:, 2]]
    axis_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

    for i in range(3):
        end_pt = center + axis_dirs[i] * axis_length
        cv2.line(frame, (int(center[0]), int(center[1])), (int(end_pt[0]), int(end_pt[1])), axis_colors[i], 2)

    return center, R_final, points_3d

# ============================================================
#   UPDATED: stable angle extraction (atan2), same mapping idea
# ============================================================
def convert_gaze_to_screen_coordinates(combined_gaze_direction, calibration_offset_yaw, calibration_offset_pitch):
    """
    Convert 3D gaze direction vector to 2D screen coordinates.
    SAME architecture: direction -> (yaw,pitch) -> screen map.
    UPDATED: stable atan2-based yaw/pitch (reduces jitter + drift).
    """
    d = _normalize(combined_gaze_direction)

    # Camera convention here matches your reference_forward = [0,0,-1]
    # yaw: left/right around Y axis, pitch: up/down around X axis (screen space)
    # Using atan2 avoids acos noise near 0 degrees.
    yaw_rad = math.atan2(d[0], -d[2])          # +right
    pitch_rad = math.atan2(d[1], -d[2])        # +down (we'll keep your sign behavior below)

    yaw_deg = math.degrees(yaw_rad)
    pitch_deg = math.degrees(pitch_rad)

    # Keep your original "up is positive" behavior:
    # If d[1] > 0 (down in image coords), pitch becomes negative in your old logic.
    # We can reproduce your mapping sense by flipping pitch:
    pitch_deg = -pitch_deg

    raw_yaw_deg = yaw_deg
    raw_pitch_deg = pitch_deg

    yawDegrees = float(YAW_DEGREES_RANGE)
    pitchDegrees = float(PITCH_DEGREES_RANGE)

    # Apply calibration offsets (unchanged concept)
    yaw_deg += calibration_offset_yaw
    pitch_deg += calibration_offset_pitch

    # Map to screen
    screen_x = int(((yaw_deg + yawDegrees) / (2 * yawDegrees)) * MONITOR_WIDTH)
    screen_y = int(((pitchDegrees - pitch_deg) / (2 * pitchDegrees)) * MONITOR_HEIGHT)

    # Clamp
    screen_x = max(10, min(screen_x, MONITOR_WIDTH - 10))
    screen_y = max(10, min(screen_y, MONITOR_HEIGHT - 10))

    return screen_x, screen_y, raw_yaw_deg, raw_pitch_deg

# ============================================================
#   NEW: low-latency screen-space stabilizer (EMA + deadzone)
# ============================================================
_screen_ema_x = None
_screen_ema_y = None

def stabilize_screen_xy(raw_x, raw_y):
    """
    Keeps the same pipeline (we do NOT change mapping),
    but stabilizes the final cursor coordinate:
      - adaptive EMA (stable gaze -> stronger smoothing)
      - clamp per-frame spikes
      - pixel deadzone
    """
    global _screen_ema_x, _screen_ema_y

    if _screen_ema_x is None or _screen_ema_y is None:
        _screen_ema_x, _screen_ema_y = float(raw_x), float(raw_y)
        return int(raw_x), int(raw_y)

    dx = float(raw_x) - _screen_ema_x
    dy = float(raw_y) - _screen_ema_y
    dist = math.hypot(dx, dy)

    # Adaptive alpha: small movements -> heavier smoothing
    # big moves -> more responsive
    alpha = SCREEN_EMA_ALPHA_SLOW if dist < 40.0 else SCREEN_EMA_ALPHA_FAST

    # Clamp spikes
    if dist > MAX_SCREEN_STEP:
        s = MAX_SCREEN_STEP / max(1e-6, dist)
        raw_x = _screen_ema_x + dx * s
        raw_y = _screen_ema_y + dy * s

    _screen_ema_x = alpha * float(raw_x) + (1.0 - alpha) * _screen_ema_x
    _screen_ema_y = alpha * float(raw_y) + (1.0 - alpha) * _screen_ema_y

    out_x = int(_screen_ema_x)
    out_y = int(_screen_ema_y)

    # Pixel deadzone
    if abs(out_x - int(_screen_ema_x)) < PIXEL_DEADZONE and abs(out_y - int(_screen_ema_y)) < PIXEL_DEADZONE:
        # (this condition is effectively always true as written if we reuse ema values)
        # We instead deadzone against previous integer output:
        pass

    return out_x, out_y

# ============================================================
#   NEW: direction buffer update with outlier rejection + EMA
# ============================================================
_dir_ema = None

def update_smoothed_direction(raw_dir, dir_buffer: deque):
    """
    Keeps your deque-based smoothing, but improves it:
      - Reject outliers vs mean direction (reduces jitter spikes)
      - Adaptive EMA on direction vector (reduces drift + micro jitter)
    """
    global _dir_ema

    raw_dir = _normalize(raw_dir)

    if len(dir_buffer) >= 3:
        mean_dir = _normalize(np.mean(np.array(dir_buffer), axis=0))
        jump = angle_between_deg(raw_dir, mean_dir)
        if jump > DIR_OUTLIER_DEG:
            # Drop this sample; return previous best estimate
            if _dir_ema is not None:
                return _normalize(_dir_ema)
            return mean_dir

    dir_buffer.append(raw_dir)
    mean_dir = _normalize(np.mean(np.array(dir_buffer), axis=0))

    # Adaptive alpha based on angular speed (difference vs EMA)
    if _dir_ema is None:
        _dir_ema = mean_dir.copy()
        return _normalize(_dir_ema)

    speed_deg = angle_between_deg(mean_dir, _dir_ema)
    # Map speed into [min,max] alpha range
    t = min(1.0, speed_deg / 8.0)  # 0..8 deg -> 0..1
    alpha = DIR_ADAPTIVE_ALPHA_MIN + t * (DIR_ADAPTIVE_ALPHA_MAX - DIR_ADAPTIVE_ALPHA_MIN)

    _dir_ema = _normalize(alpha * mean_dir + (1.0 - alpha) * _dir_ema)
    return _normalize(_dir_ema)

# ---------------- Debug view (unchanged from your version) ----------------
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

    head_w = np.asarray(head_center3d, dtype=float)

    global debug_world_frozen, orbit_pivot_frozen
    if debug_world_frozen and orbit_pivot_frozen is not None:
        pivot_w = np.asarray(orbit_pivot_frozen, dtype=float)
    else:
        if monitor_center is not None:
            pivot_w = (head_w + np.asarray(monitor_center, dtype=float)) * 0.5
        else:
            pivot_w = head_w

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

    if landmarks3d is not None:
        for P in landmarks3d:
            res = project_point(P)
            if res is not None:
                cv2.circle(debug, res[0], 0, (200, 200, 200), -1)

    draw_cross_3d(head_w, size=12, color=(255, 0, 255), thickness=2)
    hc2d = project_point(head_w)
    if hc2d is not None:
        cv2.putText(debug, "Head Center", (hc2d[0][0] + 12, hc2d[0][1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

    draw_cross_3d(pivot_w, size=8, color=(180, 120, 255), thickness=2)
    if monitor_center is not None:
        mc2d = project_point(monitor_center)
        pv2d = project_point(pivot_w)
        if mc2d is not None and pv2d is not None and hc2d is not None:
            cv2.line(debug, pv2d[0], hc2d[0], (160, 100, 255), 1)
            cv2.line(debug, pv2d[0], mc2d[0], (160, 100, 255), 1)

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
                p1 = project_point(np.asarray(sphere_world_l) + _normalize(left_dir) * gaze_len)
                if p1 is not None:
                    cv2.line(debug, (cx, cy), p1[0], (155, 155, 25), 1)
    elif iris3d_l is not None:
        res = project_point(iris3d_l)
        if res is not None:
            cv2.circle(debug, res[0], 2, (255, 255, 25), 1)

    if right_locked and sphere_world_r is not None:
        res = project_point(sphere_world_r)
        if res is not None:
            (cx, cy), z = res
            r_px = max(2, int((scaled_radius_r if scaled_radius_r else 6) * f_px / max(z, 1e-3)))
            cv2.circle(debug, (cx, cy), r_px, (25, 255, 255), 1)
            if iris3d_r is not None:
                right_dir = np.asarray(iris3d_r) - np.asarray(sphere_world_r)
                p1 = project_point(np.asarray(sphere_world_r) + _normalize(right_dir) * gaze_len)
                if p1 is not None:
                    cv2.line(debug, (cx, cy), p1[0], (25, 155, 155), 1)
    elif iris3d_r is not None:
        res = project_point(iris3d_r)
        if res is not None:
            cv2.circle(debug, res[0], 2, (25, 255, 255), 1)

    if left_locked and right_locked and sphere_world_l is not None and sphere_world_r is not None:
        origin_mid = (np.asarray(sphere_world_l) + np.asarray(sphere_world_r)) / 2.0
        if combined_dir is not None:
            p0 = project_point(origin_mid)
            p1 = project_point(origin_mid + _normalize(combined_dir) * (gaze_len * 1.2))
            if p0 is not None and p1 is not None:
                cv2.line(debug, p0[0], p1[0], (155, 200, 10), 2)

    if monitor_corners is not None:
        def draw_poly(points, color, thickness):
            projs = [project_point(p) for p in points]
            if any(p is None for p in projs): return
            p2 = [p[0] for p in projs]
            for a, b in zip(p2, p2[1:] + [p2[0]]):
                cv2.line(debug, a, b, color, thickness)
        draw_poly(monitor_corners, (0, 200, 255), 2)
        draw_poly([monitor_corners[0], monitor_corners[2]], (0, 150, 210), 1)
        draw_poly([monitor_corners[1], monitor_corners[3]], (0, 150, 210), 1)
        if monitor_center is not None:
            draw_cross_3d(monitor_center, size=8, color=(0, 200, 255), thickness=2)
            if monitor_normal is not None:
                tip = np.asarray(monitor_center) + np.asarray(monitor_normal) * (20.0 * (units_per_cm or 1.0))
                draw_arrow_3d(monitor_center, tip, color=(0, 220, 255), thickness=2)

    if (gaze_markers and monitor_corners is not None):
        p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in monitor_corners]
        u = p1 - p0
        v = p3 - p0
        width_world = float(np.linalg.norm(u))
        if width_world > 1e-9:
            u_hat = u / width_world
            r_world = 0.01 * width_world
            for (a, b) in gaze_markers:
                Pm = p0 + a * u + b * v
                projP = project_point(Pm)
                projR = project_point(Pm + u_hat * r_world)
                if projP is not None and projR is not None:
                    center_px = projP[0]
                    r_px = int(max(1, np.linalg.norm(np.array(projR[0]) - np.array(center_px))))
                    cv2.circle(debug, center_px, r_px, (0, 255, 0), 1, lineType=cv2.LINE_AA)

    # help text
    help_text = [
        "C = calibrate screen center",
        "J = yaw left",
        "L = yaw right",
        "I = pitch up",
        "K = pitch down",
        "[ = zoom out",
        "] = zoom in",
        "R = reset view",
        "X = add marker",
        "q = quit",
        "F7 = toggle mouse control",
        "S = screen (angle) calibration"
    ]

    font        = cv2.FONT_HERSHEY_SIMPLEX
    font_scale  = 0.5
    thickness   = 1
    line_height = 18

    y0 = h - (len(help_text) * line_height) - 10
    x0 = 10

    for i, text in enumerate(help_text):
        y = y0 + i * line_height
        cv2.putText(debug, text, (x0, y), font, font_scale, (200, 200, 200), thickness, cv2.LINE_AA)

    cv2.imshow("Head/Eye Debug", debug)

def mouse_mover():
    """Mouse movement thread from old script"""
    while True:
        if mouse_control_enabled:
            with mouse_lock:
                x, y = mouse_target
            pyautogui.moveTo(x, y)
        time.sleep(0.01)

threading.Thread(target=mouse_mover, daemon=True).start()

# Eye sphere tracking variables
left_sphere_locked = False
left_sphere_local_offset = None
left_calibration_nose_scale = None

right_sphere_locked = False
right_sphere_local_offset = None
right_calibration_nose_scale = None

# For pixel deadzone vs last output (NEW)
_last_sent_x = None
_last_sent_y = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    combined_dir = None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        left_iris_idx = 468
        right_iris_idx = 473
        left_iris = face_landmarks[left_iris_idx]
        right_iris = face_landmarks[right_iris_idx]

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
            draw_gaze(frame, sphere_world_l, iris_3d_left, scaled_radius_l, (55, 255, 0), 130)
            draw_gaze(frame, sphere_world_r, iris_3d_right, scaled_radius_r, (55, 255, 0), 130)

            left_gaze_dir = iris_3d_left - sphere_world_l
            left_gaze_dir /= np.linalg.norm(left_gaze_dir)

            right_gaze_dir = iris_3d_right - sphere_world_r
            right_gaze_dir /= np.linalg.norm(right_gaze_dir)

            raw_combined_direction = (left_gaze_dir + right_gaze_dir) / 2
            raw_combined_direction /= np.linalg.norm(raw_combined_direction)

            # ============================================================
            # UPDATED: direction smoothing with outlier rejection + EMA
            # (still uses your deque; same stage in the pipeline)
            # ============================================================
            avg_combined_direction = update_smoothed_direction(raw_combined_direction, combined_gaze_directions)
            combined_dir = avg_combined_direction

            # Convert gaze to screen coords (stable atan2 angles)
            raw_x, raw_y, raw_yaw, raw_pitch = convert_gaze_to_screen_coordinates(
                avg_combined_direction,
                calibration_offset_yaw,
                calibration_offset_pitch
            )

            # ============================================================
            # NEW: final cursor stabilization (EMA + spike clamp + deadzone)
            # ============================================================
            stab_x, stab_y = stabilize_screen_xy(raw_x, raw_y)

            # Pixel deadzone vs last output (reduces micro-jitter)
            if _last_sent_x is not None and _last_sent_y is not None:
                if (abs(stab_x - _last_sent_x) < PIXEL_DEADZONE and
                    abs(stab_y - _last_sent_y) < PIXEL_DEADZONE):
                    stab_x, stab_y = _last_sent_x, _last_sent_y

            # Update mouse target (same thread model)
            if mouse_control_enabled:
                with mouse_lock:
                    mouse_target[0] = stab_x
                    mouse_target[1] = stab_y
                _last_sent_x, _last_sent_y = stab_x, stab_y


            # Write screen position to file (rate-limited)
            write_screen_position_rate_limited(stab_x, stab_y)

            # Draw combined gaze ray
            combined_origin = (sphere_world_l + sphere_world_r) / 2
            combined_target = combined_origin + avg_combined_direction * gaze_length
            cv2.line(
                frame,
                (int(combined_origin[0]), int(combined_origin[1])),
                (int(combined_target[0]), int(combined_target[1])),
                (255, 255, 10), 3
            )

            # UI text
            texts = [
                f"Screen(raw): ({raw_x}, {raw_y})",
                f"Screen(stab): ({stab_x}, {stab_y})"
            ]

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2

            for i, text in enumerate(texts):
                (text_width, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
                center_x = (w - text_width) // 2
                cv2.putText(frame, text, (center_x, 30 + i * 28), font, font_scale, (0, 255, 0), thickness)

        # Draw all landmark points in white
        for idx, lm in enumerate(face_landmarks):
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 0, (255, 255, 255), -1)

        update_orbit_from_keys()

        landmarks3d = None
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            landmarks3d = np.array([[p.x * w, p.y * h, p.z * w] for p in lm], dtype=float)

        render_debug_view_orbit(
            h, w,
            head_center3d=head_center if 'head_center' in locals() else None,
            sphere_world_l=sphere_world_l if left_sphere_locked and 'sphere_world_l' in locals() else None,
            scaled_radius_l=scaled_radius_l if left_sphere_locked and 'scaled_radius_l' in locals() else None,
            sphere_world_r=sphere_world_r if right_sphere_locked and 'sphere_world_r' in locals() else None,
            scaled_radius_r=scaled_radius_r if right_sphere_locked and 'scaled_radius_r' in locals() else None,
            iris3d_l=iris_3d_left if 'iris_3d_left' in locals() else None,
            iris3d_r=iris_3d_right if 'iris_3d_right' in locals() else None,
            left_locked=left_sphere_locked,
            right_locked=right_sphere_locked,
            landmarks3d=landmarks3d,
            combined_dir=combined_dir if combined_dir is not None else None,
            gaze_len=5230,
            monitor_corners=monitor_corners,
            monitor_center=monitor_center_w,
            monitor_normal=monitor_normal_w,
            gaze_markers=gaze_markers
        )

    cv2.imshow("Integrated Eye Tracking", frame)

    # Handle keyboard input
    if keyboard.is_pressed('f7'):
        mouse_control_enabled = not mouse_control_enabled
        print(f"[Mouse Control] {'Enabled' if mouse_control_enabled else 'Disabled'}")
        time.sleep(0.3)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    elif key == ord('c') and not (left_sphere_locked and right_sphere_locked):
        current_nose_scale = compute_scale(nose_points_3d)

        left_sphere_local_offset = R_final.T @ (iris_3d_left - head_center)
        camera_dir_world = np.array([0, 0, 1])
        camera_dir_local = R_final.T @ camera_dir_world
        left_sphere_local_offset += base_radius * camera_dir_local
        left_calibration_nose_scale = current_nose_scale
        left_sphere_locked = True

        right_sphere_local_offset = R_final.T @ (iris_3d_right - head_center)
        right_sphere_local_offset += base_radius * camera_dir_local
        right_calibration_nose_scale = current_nose_scale
        right_sphere_locked = True

        sphere_world_l_calib = head_center + R_final @ left_sphere_local_offset
        sphere_world_r_calib = head_center + R_final @ right_sphere_local_offset

        left_dir  = iris_3d_left  - sphere_world_l_calib
        right_dir = iris_3d_right - sphere_world_r_calib
        if np.linalg.norm(left_dir)  > 1e-9: left_dir  /= np.linalg.norm(left_dir)
        if np.linalg.norm(right_dir) > 1e-9: right_dir /= np.linalg.norm(right_dir)
        forward_hint = (left_dir + right_dir) * 0.5
        if np.linalg.norm(forward_hint) > 1e-9:
            forward_hint /= np.linalg.norm(forward_hint)
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
        print(f"[Monitor] units_per_cm={units_per_cm:.3f}, center={monitor_center_w}, normal={monitor_normal_w}")
        print("[Both Spheres Locked] Eye sphere calibration complete.")

    elif key == ord('s') and left_sphere_locked and right_sphere_locked:
        left_gaze_dir = iris_3d_left - sphere_world_l
        left_gaze_dir /= np.linalg.norm(left_gaze_dir)
        right_gaze_dir = iris_3d_right - sphere_world_r
        right_gaze_dir /= np.linalg.norm(right_gaze_dir)
        current_combined_direction = (left_gaze_dir + right_gaze_dir) / 2
        current_combined_direction /= np.linalg.norm(current_combined_direction)

        _, _, raw_yaw, raw_pitch = convert_gaze_to_screen_coordinates(
            current_combined_direction, 0, 0
        )

        calibration_offset_yaw = 0 - raw_yaw
        calibration_offset_pitch = 0 - raw_pitch
        print(f"[Screen Calibrated] Offset Yaw: {calibration_offset_yaw:.2f}, Offset Pitch: {calibration_offset_pitch:.2f}")

    elif key == ord('x'):
        if (monitor_corners is not None and monitor_center_w is not None and monitor_normal_w is not None
            and left_sphere_locked and right_sphere_locked):

            current_nose_scale = compute_scale(nose_points_3d)
            scale_ratio_l = current_nose_scale / left_calibration_nose_scale if left_calibration_nose_scale else 1.0
            scale_ratio_r = current_nose_scale / right_calibration_nose_scale if right_calibration_nose_scale else 1.0
            sphere_world_l_now = head_center + R_final @ (left_sphere_local_offset * scale_ratio_l)
            sphere_world_r_now = head_center + R_final @ (right_sphere_local_offset * scale_ratio_r)

            if combined_dir is not None:
                D = _normalize(np.asarray(combined_dir, dtype=float))
            else:
                lg = iris_3d_left  - sphere_world_l_now
                rg = iris_3d_right - sphere_world_r_now
                if np.linalg.norm(lg) < 1e-9 or np.linalg.norm(rg) < 1e-9:
                    print("[Marker] Gaze direction invalid; try again.")
                    D = None
                else:
                    lg /= np.linalg.norm(lg)
                    rg /= np.linalg.norm(rg)
                    D = _normalize(lg + rg)

            if D is not None:
                O = (sphere_world_l_now + sphere_world_r_now) * 0.5
                C = np.asarray(monitor_center_w, dtype=float)
                N = _normalize(np.asarray(monitor_normal_w, dtype=float))
                denom = float(np.dot(N, D))
                if abs(denom) < 1e-6:
                    print("[Marker] Gaze ray parallel to monitor; no marker.")
                else:
                    t = float(np.dot(N, (C - O)) / denom)
                    if t <= 0.0:
                        print("[Marker] Intersection behind/at eye; no marker.")
                    else:
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
                            else:
                                print("[Marker] Gaze not on monitor; no marker.")
                        else:
                            print("[Marker] Monitor dimensions degenerate; no marker.")
        else:
            print("[Marker] Monitor/gaze not ready; complete center calibration first.")

cap.release()
cv2.destroyAllWindows()
