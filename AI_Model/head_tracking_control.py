"""
Filename: head_tracking_control.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 2/1/2026

Description:
    Head Tracking Mouse Control System (Config-driven, FULL SCRIPT)

    Main Features:
      - All tuning now lives in config.yaml (including per-profile overrides)
      - Continuous yaw/pitch [-180,+180] + wrap-safe subtraction
      - Calibration is robust (median), UI progress, persistent save/load
      - Cursor-space filtering (EMA/Kalman) controlled from config
      - Profile switching preserves state (no HeadTracker re-instantiation)
      - Edge-trigger hotkeys (no debounce sleeps)
      - Absolute + Relative modes

Hotkeys:
  F7 : Toggle mouse control
  C  : Calibrate (uses calibration.samples)
  P  : Cycle profiles (FAST -> BALANCED -> SMOOTH)
  M  : Toggle mode (absolute <-> relative)
  Q  : Quit
"""

import os
import json
import math
import time
import threading
from dataclasses import dataclass, asdict
from collections import deque

import cv2
import numpy as np
import yaml
import keyboard

# -----------------------------
# Module Imports
# -----------------------------
try:
    from VisionModule.camera import IrisCamera, print_camera_info
    from VisionModule.face_utils import FaceMeshDetector
    from VisionModule.head_tracking import HeadTracker
    from PyautoguiController.mouse_controller import MouseController
except Exception:
    from camera import IrisCamera, print_camera_info
    from face_utils import FaceMeshDetector
    from head_tracking import HeadTracker
    from mouse_controller import MouseController

from Filters.filterManager import FilterManager


# -----------------------------
# Constants
# -----------------------------
FACE_OUTLINE_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]

PROFILE_ORDER = ["FAST", "BALANCED", "SMOOTH"]


# -----------------------------
# Config utilities
# -----------------------------
def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (returns new dict)."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str = "config.yaml") -> dict:
    """
    Loads config.yaml and merges missing keys with defaults.
    If missing, returns defaults (so code still runs).
    """
    defaults = {
        "tracking": {
            "yaw_range": 20,
            "pitch_range": 10,
            "filter_length": 8,
            "invert_x": True,
            "invert_y": True,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
        },
        "smoothing": {"process_interval": 0.01},
        "control": {
            "default_mode": "absolute",
            "start_mouse_enabled": True,
            "movement_threshold_px": 0.5,
            "pixel_deadzone_radius": 2.0,
            "relative": {
                "max_speed_px_per_s": 1800.0,
                "deadzone_deg": 1.5,
                "expo": 1.6,
            },
        },
        "filter": {
            "type": "kalman",
            "ema": {"alpha": 0.55},
            "kalman": {"dt": 1.0, "process_noise": 0.002, "measurement_noise": 0.08},
        },
        "profiles": {"active": "BALANCED"},
        "calibration": {"file": "head_calibration.json", "autosave": True, "samples": 30},
        "mouse": {"update_interval": 0.01},
        "display": {"show_landmarks": True, "show_cube": True, "show_gaze_ray": True},
    }

    try:
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        return deep_merge(defaults, user_cfg)
    except FileNotFoundError:
        print("[Config] config.yaml not found. Using defaults.")
        return defaults
    except Exception as e:
        print(f"[Config] Failed reading config.yaml: {e}. Using defaults.")
        return defaults


def get_effective_config(cfg: dict, profile_name: str) -> dict:
    """Applies profiles.<PROFILE> overrides on top of base config."""
    profiles = cfg.get("profiles", {}) or {}
    prof_block = (profiles.get(profile_name, {}) or {})
    return deep_merge(cfg, prof_block)


# -----------------------------
# Edge-trigger key helper (no debounce sleeps)
# -----------------------------
class EdgeKey:
    def __init__(self):
        self.prev = False

    def rising(self, pressed_now: bool) -> bool:
        fired = (not self.prev) and pressed_now
        self.prev = pressed_now
        return fired


# -----------------------------
# Angle math (continuous, wrap-safe)
# -----------------------------
def wrap_deg(angle: float) -> float:
    a = (angle + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def angle_diff_deg(a: float, b: float) -> float:
    return wrap_deg(a - b)


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def dist2(x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + dy * dy


# -----------------------------
# Calibration (robust median + persistence)
# -----------------------------
@dataclass
class HeadCalibration:
    center_yaw_deg: float = 0.0
    center_pitch_deg: float = 0.0
    calibrated: bool = False


class CalibrationManager:
    def __init__(self):
        self.cal = HeadCalibration()
        self._calibrating = False
        self._target_n = 30
        self._yaw_samples = []
        self._pitch_samples = []

    @property
    def is_calibrating(self) -> bool:
        return self._calibrating

    def start(self, num_samples: int = 30):
        self._calibrating = True
        self._target_n = max(10, int(num_samples))
        self._yaw_samples.clear()
        self._pitch_samples.clear()
        print(f"[Calibration] Collecting {self._target_n} samples...")

    def add_sample(self, yaw_deg: float, pitch_deg: float):
        if not self._calibrating:
            return
        self._yaw_samples.append(wrap_deg(yaw_deg))
        self._pitch_samples.append(wrap_deg(pitch_deg))
        if len(self._yaw_samples) >= self._target_n:
            self.finish()

    def progress(self):
        return len(self._yaw_samples), self._target_n

    def finish(self):
        if not self._yaw_samples:
            self._calibrating = False
            return

        ys = sorted(self._yaw_samples)
        ps = sorted(self._pitch_samples)
        mid = len(ys) // 2

        if len(ys) % 2 == 1:
            cy = ys[mid]
            cp = ps[mid]
        else:
            cy = (ys[mid - 1] + ys[mid]) / 2.0
            cp = (ps[mid - 1] + ps[mid]) / 2.0

        self.cal.center_yaw_deg = wrap_deg(cy)
        self.cal.center_pitch_deg = wrap_deg(cp)
        self.cal.calibrated = True
        self._calibrating = False

        yaw_std = float(np.std(self._yaw_samples))
        pitch_std = float(np.std(self._pitch_samples))
        print("[Calibration] Complete.")
        print(f"  Center yaw={self.cal.center_yaw_deg:.2f}°, pitch={self.cal.center_pitch_deg:.2f}°")
        print(f"  Stability std: yaw={yaw_std:.2f}°, pitch={pitch_std:.2f}°")

    def save(self, path: str) -> bool:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(self.cal), f, indent=2)
            return True
        except Exception as e:
            print(f"[Calibration] Save failed: {e}")
            return False

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.cal.center_yaw_deg = wrap_deg(float(data.get("center_yaw_deg", 0.0)))
            self.cal.center_pitch_deg = wrap_deg(float(data.get("center_pitch_deg", 0.0)))
            self.cal.calibrated = bool(data.get("calibrated", False))
            return True
        except Exception as e:
            print(f"[Calibration] Load failed: {e}")
            return False


# -----------------------------
# HeadTracker buffer resize (preserve state; no re-instantiation)
# -----------------------------
def resize_headtracker_buffers(head_tracker: HeadTracker, new_len: int):
    new_len = int(max(1, new_len))
    if getattr(head_tracker, "filter_length", None) == new_len:
        return

    # Preserve last values as seed
    try:
        last_origin = head_tracker.ray_origins[-1] if len(head_tracker.ray_origins) else np.array([0.0, 0.0, 0.0], dtype=float)
        last_dir = head_tracker.ray_directions[-1] if len(head_tracker.ray_directions) else np.array([0.0, 0.0, -1.0], dtype=float)
    except Exception:
        last_origin = np.array([0.0, 0.0, 0.0], dtype=float)
        last_dir = np.array([0.0, 0.0, -1.0], dtype=float)

    # Preserve old offsets (even if unused)
    yaw_off = getattr(head_tracker, "calibration_offset_yaw", 0.0)
    pit_off = getattr(head_tracker, "calibration_offset_pitch", 0.0)

    head_tracker.filter_length = new_len
    head_tracker.ray_origins = deque(maxlen=new_len)
    head_tracker.ray_directions = deque(maxlen=new_len)

    for _ in range(new_len):
        head_tracker.ray_origins.append(np.array(last_origin, dtype=float))
        head_tracker.ray_directions.append(np.array(last_dir, dtype=float))

    head_tracker.calibration_offset_yaw = yaw_off
    head_tracker.calibration_offset_pitch = pit_off


# -----------------------------
# Control core (mapping + cursor-space filtering)
# -----------------------------
class HeadControlCore:
    def __init__(self, screen_w: int, screen_h: int, yaw_range_deg: float, pitch_range_deg: float):
        self.screen_w = int(screen_w)
        self.screen_h = int(screen_h)

        # Absolute sensitivity (config-driven)
        self.yaw_range = float(yaw_range_deg)
        self.pitch_range = float(pitch_range_deg)

        # Inversion (config-driven)
        self.invert_x = True
        self.invert_y = True

        # Modes
        self.mode = "absolute"  # absolute | relative

        # Gating
        self.movement_threshold_px = 0.5
        self.pixel_deadzone_radius = 2.0

        # Relative mode tuning
        self.max_speed_px_per_s = 1800.0
        self.relative_deadzone_deg = 1.5
        self.relative_expo = 1.6

        # Integrator state (relative)
        self._cursor_x = self.screen_w / 2.0
        self._cursor_y = self.screen_h / 2.0

        self._last_sent_x = None
        self._last_sent_y = None
        self._last_t = None

        # Filter (config-driven)
        self.filter = FilterManager(filter_type="kalman", dt=1.0, process_noise=0.002, measurement_noise=0.08)

        # Calibration
        self.cal_mgr = CalibrationManager()

    def toggle_mode(self):
        self.mode = "relative" if self.mode == "absolute" else "absolute"
        self._cursor_x = self.screen_w / 2.0
        self._cursor_y = self.screen_h / 2.0
        self._last_sent_x = None
        self._last_sent_y = None
        try:
            self.filter.reset()
        except Exception:
            pass

    def start_calibration(self, num_samples: int):
        self.cal_mgr.start(num_samples)

    def update(self, yaw_deg: float, pitch_deg: float, now: float):
        yaw = wrap_deg(yaw_deg)
        pitch = wrap_deg(pitch_deg)

        if self.cal_mgr.is_calibrating:
            self.cal_mgr.add_sample(yaw, pitch)

        # Auto-initialize center if not calibrated (pragmatic)
        if not self.cal_mgr.cal.calibrated:
            self.cal_mgr.cal.center_yaw_deg = yaw
            self.cal_mgr.cal.center_pitch_deg = pitch
            self.cal_mgr.cal.calibrated = True

        dyaw = angle_diff_deg(yaw, self.cal_mgr.cal.center_yaw_deg)
        dpitch = angle_diff_deg(pitch, self.cal_mgr.cal.center_pitch_deg)

        # Apply inversion (you confirmed this was needed)
        if self.invert_x:
            dyaw = -dyaw
        if self.invert_y:
            dpitch = -dpitch

        # dt
        if self._last_t is None:
            dt = 1.0 / 60.0
        else:
            dt = max(1e-4, now - self._last_t)
        self._last_t = now

        # Map -> raw
        if self.mode == "absolute":
            nx = clamp(dyaw / self.yaw_range, -1.0, 1.0)
            ny = clamp(dpitch / self.pitch_range, -1.0, 1.0)
            raw_x = (self.screen_w / 2.0) + nx * (self.screen_w / 2.0)
            raw_y = (self.screen_h / 2.0) + ny * (self.screen_h / 2.0)
        else:
            # Relative (joystick-like)
            dz = self.relative_deadzone_deg
            ax = 0.0 if abs(dyaw) < dz else dyaw
            ay = 0.0 if abs(dpitch) < dz else dpitch

            nx = clamp(ax / self.yaw_range, -1.0, 1.0)
            ny = clamp(ay / self.pitch_range, -1.0, 1.0)

            expo = max(1.0, float(self.relative_expo))
            nx = math.copysign(abs(nx) ** expo, nx)
            ny = math.copysign(abs(ny) ** expo, ny)

            vx = nx * self.max_speed_px_per_s
            vy = ny * self.max_speed_px_per_s

            self._cursor_x += vx * dt
            self._cursor_y += vy * dt
            raw_x, raw_y = self._cursor_x, self._cursor_y

        # Clamp raw
        raw_x = clamp(raw_x, 0.0, float(self.screen_w - 1))
        raw_y = clamp(raw_y, 0.0, float(self.screen_h - 1))

        # Filter in cursor space
        fx, fy = self.filter.apply_filter(raw_x, raw_y)

        # Pixel deadzone AFTER filtering
        if self.pixel_deadzone_radius > 0.0 and self._last_sent_x is not None and self._last_sent_y is not None:
            if dist2(fx, fy, self._last_sent_x, self._last_sent_y) < (self.pixel_deadzone_radius ** 2):
                fx, fy = self._last_sent_x, self._last_sent_y

        # Movement threshold gating
        if self._last_sent_x is None or self._last_sent_y is None:
            should_send = True
        else:
            should_send = dist2(fx, fy, self._last_sent_x, self._last_sent_y) >= (self.movement_threshold_px ** 2)

        if should_send:
            self._last_sent_x, self._last_sent_y = fx, fy

        dbg = {
            "yaw": yaw,
            "pitch": pitch,
            "dyaw": dyaw,
            "dpitch": dpitch,
            "raw_x": raw_x,
            "raw_y": raw_y,
            "fx": fx,
            "fy": fy,
            "dt": dt,
            "mode": self.mode,
            "calibrating": self.cal_mgr.is_calibrating,
            "calib_progress": self.cal_mgr.progress() if self.cal_mgr.is_calibrating else None,
        }
        return should_send, float(fx), float(fy), dbg


# -----------------------------
# Apply config to runtime
# -----------------------------
def apply_effective_config(eff: dict, head_tracker: HeadTracker, core: HeadControlCore):
    # Tracking
    core.yaw_range = float(eff["tracking"]["yaw_range"])
    core.pitch_range = float(eff["tracking"]["pitch_range"])
    core.invert_x = bool(eff["tracking"].get("invert_x", True))
    core.invert_y = bool(eff["tracking"].get("invert_y", True))

    resize_headtracker_buffers(head_tracker, int(eff["tracking"]["filter_length"]))

    # Control
    core.mode = str(eff["control"].get("default_mode", core.mode)).lower()
    core.movement_threshold_px = float(eff["control"]["movement_threshold_px"])
    core.pixel_deadzone_radius = float(eff["control"]["pixel_deadzone_radius"])

    rel = eff["control"].get("relative", {}) or {}
    core.max_speed_px_per_s = float(rel.get("max_speed_px_per_s", core.max_speed_px_per_s))
    core.relative_deadzone_deg = float(rel.get("deadzone_deg", core.relative_deadzone_deg))
    core.relative_expo = float(rel.get("expo", core.relative_expo))

    # Filter
    fcfg = eff.get("filter", {}) or {}
    ftype = str(fcfg.get("type", "kalman")).lower()

    if ftype == "ema":
        alpha = float((fcfg.get("ema", {}) or {}).get("alpha", 0.55))
        core.filter = FilterManager(filter_type="ema", alpha=alpha)
    else:
        kcfg = fcfg.get("kalman", {}) or {}
        dt = float(kcfg.get("dt", 1.0))
        pn = float(kcfg.get("process_noise", 0.002))
        mn = float(kcfg.get("measurement_noise", 0.08))
        core.filter = FilterManager(filter_type="kalman", dt=dt, process_noise=pn, measurement_noise=mn)

    try:
        core.filter.reset()
    except Exception:
        pass

    # Loop interval
    process_interval = float((eff.get("smoothing", {}) or {}).get("process_interval", 0.01))
    return process_interval


# -----------------------------
# Main
# -----------------------------
def main():
    print("Head Tracking Mouse Control System (Config-driven)")
    print("=================================================")

    cfg = load_config("config.yaml")

    # Display
    SHOW_LANDMARKS = bool(cfg["display"].get("show_landmarks", True))
    SHOW_CUBE = bool(cfg["display"].get("show_cube", True))
    SHOW_GAZE_RAY = bool(cfg["display"].get("show_gaze_ray", True))

    # Calibration config
    cal_cfg = cfg.get("calibration", {}) or {}
    CAL_FILE = str(cal_cfg.get("file", "head_calibration.json"))
    CAL_AUTOSAVE = bool(cal_cfg.get("autosave", True))
    CAL_SAMPLES = int(cal_cfg.get("samples", 30))

    # Starting profile
    active_profile = str((cfg.get("profiles", {}) or {}).get("active", "BALANCED")).upper()
    if active_profile not in PROFILE_ORDER:
        active_profile = "BALANCED"
    profile_idx = PROFILE_ORDER.index(active_profile)

    # 1) Camera selection
    print("\nScanning for cameras...")
    cameras = print_camera_info()
    if not cameras:
        print("No cameras found. Please connect a camera and try again.")
        return

    camera_ports = list(cameras.keys())
    if len(camera_ports) > 1:
        print(f"\nAvailable cameras: {camera_ports}")
        cam_id = int(input(f"Select camera port (default {camera_ports[0]}): ") or camera_ports[0])
    else:
        cam_id = camera_ports[0]
        print(f"\nUsing camera at port {cam_id}")

    # 2) Init modules
    print("\nInitializing components...")
    try:
        camera = IrisCamera(cam_id, cam_width=640, cam_height=480)
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return

    face_detector = FaceMeshDetector(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=float(cfg["tracking"]["min_detection_confidence"]),
        min_tracking_confidence=float(cfg["tracking"]["min_tracking_confidence"]),
    )

    head_tracker = HeadTracker(filter_length=int(cfg["tracking"]["filter_length"]))

    mouse_controller = MouseController(
        update_interval=float(cfg["mouse"]["update_interval"]),
        enable_failsafe=False,
    )
    mouse_controller.start()
    screen_w, screen_h = mouse_controller.get_screen_size()

    core = HeadControlCore(
        screen_w, screen_h,
        yaw_range_deg=float(cfg["tracking"]["yaw_range"]),
        pitch_range_deg=float(cfg["tracking"]["pitch_range"]),
    )

    # Load calibration if present
    if core.cal_mgr.load(CAL_FILE):
        print(f"[Calibration] Loaded from {CAL_FILE}")

    # Apply active profile effective config
    current_profile_name = PROFILE_ORDER[profile_idx]
    eff = get_effective_config(cfg, current_profile_name)
    PROCESS_INTERVAL = apply_effective_config(eff, head_tracker, core)

    # Startup mouse enabled
    mouse_control_enabled = bool((eff.get("control", {}) or {}).get("start_mouse_enabled", True))

    # Thread lock for mouse updates
    mouse_lock = threading.Lock()

    # Controls
    print("\nHotkeys:")
    print("  F7: Toggle mouse control")
    print("  C : Calibrate")
    print("  P : Cycle profiles (FAST/BALANCED/SMOOTH)")
    print("  M : Toggle mode (absolute/relative)")
    print("  Q : Quit")
    print(f"\nActive profile: {current_profile_name}")
    print(f"Mode: {core.mode}")
    print(f"Invert: X={core.invert_x}, Y={core.invert_y}")
    print(f"Config file: config.yaml")
    print(f"Calibration file: {CAL_FILE}")

    last_processed_time = time.time()

    # Edge-trigger keys
    key_f7 = EdgeKey()
    key_c = EdgeKey()
    key_p = EdgeKey()
    key_m = EdgeKey()

    prev_calibrating = False

    try:
        while camera.is_opened():
            now = time.time()

            # Frame rate limiting
            if now - last_processed_time < PROCESS_INTERVAL:
                time.sleep(0.001)
                continue
            last_processed_time = now

            frame = camera.get_frame()
            if frame is None:
                continue

            cam_width, cam_height = camera.get_resolution()

            landmarks_frame, landmarks_list_3d = face_detector.get_landmarks_as_pixels(frame)

            yaw_deg = 0.0
            pitch_deg = 0.0
            screen_x = None
            screen_y = None

            if landmarks_list_3d:
                landmarks_dict = {idx: (x, y, z) for idx, x, y, z in landmarks_list_3d}

                if SHOW_LANDMARKS:
                    for idx, x, y, z in landmarks_list_3d:
                        color = (155, 155, 155) if idx in FACE_OUTLINE_INDICES else (255, 25, 10)
                        cv2.circle(landmarks_frame, (x, y), 3, color, -1)
                        cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

                key_points_indices = {
                    "left": 234,
                    "right": 454,
                    "top": 10,
                    "bottom": 152,
                    "front": 1
                }

                missing = [name for name, idx in key_points_indices.items() if idx not in landmarks_dict]
                if not missing:
                    # Extract key points (3D)
                    key_points_3d = {}
                    for name, idx in key_points_indices.items():
                        x, y, z = landmarks_dict[idx]
                        key_points_3d[name] = np.array([float(x), float(y), float(z)], dtype=float)

                    orientation = head_tracker.calculate_head_orientation(key_points_3d)

                    # Update smoothing buffers (HeadTracker internal averaging)
                    head_tracker.update_smoothing_buffers(orientation["center"], orientation["forward_axis"])
                    smoothed_origin, smoothed_direction = head_tracker.get_smoothed_orientation()

                    # Compute yaw/pitch (continuous degrees)
                    yaw_deg, pitch_deg = head_tracker.calculate_yaw_pitch(smoothed_direction)
                    yaw_deg = wrap_deg(yaw_deg)
                    pitch_deg = wrap_deg(pitch_deg)

                    # Control -> cursor target
                    should_send, fx, fy, dbg = core.update(yaw_deg, pitch_deg, now=now)
                    screen_x, screen_y = fx, fy

                    if mouse_control_enabled and should_send:
                        with mouse_lock:
                            mouse_controller.set_target(screen_x, screen_y)

                    # Visualization: cube
                    if SHOW_CUBE:
                        cube_corners = head_tracker.generate_head_cube(
                            orientation["center"],
                            orientation["right_axis"],
                            orientation["up_axis"],
                            orientation["forward_axis"],
                            orientation["half_width"],
                            orientation["half_height"]
                        )
                        face_detector.draw_head_cube(frame, cube_corners)

                    # Visualization: ray
                    if SHOW_GAZE_RAY:
                        ray_origin_2d = smoothed_origin[:2]
                        ray_dir_2d = smoothed_direction[:2]
                        nrm = np.linalg.norm(ray_dir_2d)
                        if nrm > 0:
                            ray_dir_2d = ray_dir_2d / nrm
                        face_detector.draw_simple_gaze_ray(frame, ray_origin_2d, ray_dir_2d, length=200)
                        face_detector.draw_simple_gaze_ray(landmarks_frame, ray_origin_2d, ray_dir_2d, length=200)

                    # Draw key points
                    for name, idx in key_points_indices.items():
                        x, y, z = landmarks_dict[idx]
                        cv2.circle(frame, (x, y), 4, (0, 0, 0), -1)
                        cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)

            # UI overlays
            cv2.putText(frame, f"Yaw: {yaw_deg:+.1f}, Pitch: {pitch_deg:+.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f"Mouse: {'ON' if mouse_control_enabled else 'OFF'}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if mouse_control_enabled else (0, 0, 255), 2)

            cv2.putText(frame, f"Profile: {current_profile_name}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f"Mode: {core.mode.upper()}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if screen_x is not None and screen_y is not None:
                cv2.putText(frame, f"Screen: ({screen_x:.0f}, {screen_y:.0f})",
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if core.cal_mgr.is_calibrating:
                c, n = core.cal_mgr.progress()
                cv2.putText(frame, f"CALIBRATING: {c}/{n}",
                            (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            fps = camera.get_frame_rate()
            cv2.putText(frame, f"FPS: {fps}", (10, cam_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Head Tracking Mouse Control", frame)
            cv2.imshow("Facial Landmarks", landmarks_frame)

            # ---------
            # Keyboard handling (edge-triggered)
            # ---------
            if key_f7.rising(keyboard.is_pressed("f7")):
                mouse_control_enabled = not mouse_control_enabled
                print(f"[Mouse Control] {'Enabled' if mouse_control_enabled else 'Disabled'}")

            if key_c.rising(keyboard.is_pressed("c")):
                if not core.cal_mgr.is_calibrating:
                    core.start_calibration(CAL_SAMPLES)

            if key_p.rising(keyboard.is_pressed("p")):
                profile_idx = (profile_idx + 1) % len(PROFILE_ORDER)
                current_profile_name = PROFILE_ORDER[profile_idx]
                eff = get_effective_config(cfg, current_profile_name)
                PROCESS_INTERVAL = apply_effective_config(eff, head_tracker, core)
                print(f"[Profile] Switched to {current_profile_name} (process_interval={PROCESS_INTERVAL})")

            if key_m.rising(keyboard.is_pressed("m")):
                core.toggle_mode()
                print(f"[Mode] Switched to {core.mode}")

            # Auto-save calibration when it finishes
            if prev_calibrating and (not core.cal_mgr.is_calibrating) and CAL_AUTOSAVE:
                if core.cal_mgr.save(CAL_FILE):
                    print(f"[Calibration] Saved to {CAL_FILE}")
            prev_calibrating = core.cal_mgr.is_calibrating

            # Quit
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q") or cv2.getWindowProperty("Head Tracking Mouse Control", cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        print("\nShutting down...")
        try:
            camera.stop_recording()
        except Exception:
            pass
        try:
            face_detector.release()
        except Exception:
            pass
        try:
            mouse_controller.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
