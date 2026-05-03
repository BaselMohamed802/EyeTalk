import os
import sys

# --- Project-level imports (parent / root) ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Optional, Tuple, Dict, Any
from Filters.filterManager import FilterManager
from .gaze_calibration import GazeCalibration

def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v

def _dist2(x1, y1, x2, y2) -> float:
    dx = x1 - x2
    dy = y1 - y2
    return dx*dx + dy*dy

class GazeControlCore:
    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = int(screen_w)
        self.screen_h = int(screen_h)

        self.cal = GazeCalibration()

        self.filter = FilterManager(filter_type="kalman", dt=1.0, process_noise=0.003, measurement_noise=0.10)

        self.deadzone_px = 3.0
        self.movement_threshold_px = 0.8
        self.min_confidence = 0.5

        self.last_sent: Optional[Tuple[float, float]] = None

    def configure_filter(self, filter_type: str, ema_alpha: float, k_dt: float, k_q: float, k_r: float):
        if filter_type.lower() == "ema":
            self.filter = FilterManager(filter_type="ema", alpha=float(ema_alpha))
        else:
            self.filter = FilterManager(filter_type="kalman", dt=float(k_dt), process_noise=float(k_q), measurement_noise=float(k_r))
        self.filter.reset()

    def update(self, yaw_rad: float, pitch_rad: float, confidence: float) -> Tuple[bool, float, float, Dict[str, Any]]:
        dbg = {"confidence": float(confidence)}

        if confidence < self.min_confidence:
            if self.last_sent is None:
                return False, self.screen_w/2, self.screen_h/2, {"gated": True, **dbg}
            return False, self.last_sent[0], self.last_sent[1], {"gated": True, **dbg}

        x, y = self.cal.predict(yaw_rad, pitch_rad)

        x = _clamp(x, 0.0, self.screen_w - 1.0)
        y = _clamp(y, 0.0, self.screen_h - 1.0)

        fx, fy = self.filter.apply_filter(x, y)

        if self.last_sent is not None and self.deadzone_px > 0:
            if _dist2(fx, fy, self.last_sent[0], self.last_sent[1]) < (self.deadzone_px ** 2):
                fx, fy = self.last_sent

        should_send = (self.last_sent is None) or (_dist2(fx, fy, self.last_sent[0], self.last_sent[1]) >= (self.movement_threshold_px ** 2))

        if should_send:
            self.last_sent = (float(fx), float(fy))

        dbg.update({"x": x, "y": y, "fx": float(fx), "fy": float(fy)})
        return should_send, float(fx), float(fy), dbg