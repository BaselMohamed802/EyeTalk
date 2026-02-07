"""
gaze_calibration.py

Simple non-ML calibration for webcam gaze control.

We calibrate a normalized gaze space (gx, gy) -> screen space by applying:
  gx_n = (gx - offset_x) * scale_x
  gy_n = (gy - offset_y) * scale_y
(optional gamma shaping in runtime)

Calibration modes:
- 1-point: center only -> set offsets.
- 3-point: center + left + right -> offsets + scale_x.
- 5-point: center + left + right + up + down -> offsets + scale_x + scale_y.

All values are "lightweight parameters" (not an ML model).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import json
import time
import os


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class GazeCalibration:
    # Offsets in gaze-feature space (center reference)
    offset_x: float = 0.0
    offset_y: float = 0.0

    # Gains that map gaze-feature delta to normalized [-1..1] range
    # (runtime maps [-1..1] to screen width/height)
    scale_x: float = 2.0
    scale_y: float = 2.0

    # Optional nonlinearity (keep = 1.0 to disable)
    gamma_x: float = 1.0
    gamma_y: float = 1.0

    # Metadata
    method: str = "1point"  # "1point" | "3point" | "5point"
    created_at: float = 0.0
    updated_at: float = 0.0

    # --- transient captured samples (not saved) ---
    _center_gx: Optional[float] = None
    _center_gy: Optional[float] = None
    _left_gx: Optional[float] = None
    _right_gx: Optional[float] = None
    _up_gy: Optional[float] = None
    _down_gy: Optional[float] = None

    def reset(self, *, method: Optional[str] = None) -> None:
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.scale_x = 2.0
        self.scale_y = 2.0
        self.gamma_x = 1.0
        self.gamma_y = 1.0
        if method is not None:
            self.method = method
        self._clear_samples()
        now = time.time()
        self.created_at = now
        self.updated_at = now

    def _clear_samples(self) -> None:
        self._center_gx = None
        self._center_gy = None
        self._left_gx = None
        self._right_gx = None
        self._up_gy = None
        self._down_gy = None

    # ---------- capture points ----------
    def capture_center(self, gx: float, gy: float) -> None:
        self._center_gx = float(gx)
        self._center_gy = float(gy)
        self.offset_x = float(gx)
        self.offset_y = float(gy)
        self.method = "1point"
        self.updated_at = time.time()

    def capture_left(self, gx: float) -> None:
        self._left_gx = float(gx)
        self.updated_at = time.time()

    def capture_right(self, gx: float) -> None:
        self._right_gx = float(gx)
        self.updated_at = time.time()

    def capture_up(self, gy: float) -> None:
        self._up_gy = float(gy)
        self.updated_at = time.time()

    def capture_down(self, gy: float) -> None:
        self._down_gy = float(gy)
        self.updated_at = time.time()

    # ---------- compute scales ----------
    def finalize_3point(self, *, target_edge_norm: float = 0.90) -> bool:
        """
        Use center + left + right to compute scale_x.
        target_edge_norm means: when user looks at edge, normalized gaze should be about +/-target_edge_norm.
        Returns True if success.
        """
        if self._center_gx is None or self._center_gy is None:
            return False
        if self._left_gx is None or self._right_gx is None:
            return False

        cx = self._center_gx
        lx = self._left_gx
        rx = self._right_gx

        # Compute half-span from center to edges (use robust min of left/right)
        span_left = abs(lx - cx)
        span_right = abs(rx - cx)
        span = min(span_left, span_right)

        if span < 1e-4:
            return False

        # Want: (gx - offset_x) * scale_x ≈ +/- target_edge_norm at edges
        self.scale_x = float(target_edge_norm / span)

        # keep y scale default unless user finalizes 5-point later
        self.method = "3point"
        self.updated_at = time.time()
        return True

    def finalize_5point(self, *, target_edge_norm: float = 0.90) -> bool:
        """
        Use center + left + right + up + down to compute scale_x and scale_y.
        Returns True if success.
        """
        ok3 = self.finalize_3point(target_edge_norm=target_edge_norm)
        if not ok3:
            return False
        if self._up_gy is None or self._down_gy is None or self._center_gy is None:
            return False

        cy = self._center_gy
        uy = self._up_gy
        dy = self._down_gy

        span_up = abs(uy - cy)
        span_down = abs(dy - cy)
        span = min(span_up, span_down)

        if span < 1e-4:
            return False

        self.scale_y = float(target_edge_norm / span)
        self.method = "5point"
        self.updated_at = time.time()
        return True

    # ---------- apply calibration ----------
    def apply(self, gx: float, gy: float) -> tuple[float, float]:
        """
        Apply offset + scale only.
        Output is in roughly [-1..1] range (runtime will clamp).
        """
        x = (float(gx) - self.offset_x) * self.scale_x
        y = (float(gy) - self.offset_y) * self.scale_y
        return x, y

    # ---------- persistence ----------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # remove transient samples
        for k in list(d.keys()):
            if k.startswith("_"):
                d.pop(k, None)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GazeCalibration":
        cal = GazeCalibration()
        cal.offset_x = float(d.get("offset_x", 0.0))
        cal.offset_y = float(d.get("offset_y", 0.0))
        cal.scale_x = float(d.get("scale_x", 2.0))
        cal.scale_y = float(d.get("scale_y", 2.0))
        cal.gamma_x = float(d.get("gamma_x", 1.0))
        cal.gamma_y = float(d.get("gamma_y", 1.0))
        cal.method = str(d.get("method", "1point"))
        cal.created_at = float(d.get("created_at", 0.0))
        cal.updated_at = float(d.get("updated_at", 0.0))
        return cal

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load_json(path: str) -> Optional["GazeCalibration"]:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return GazeCalibration.from_dict(d)
