from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import time

@dataclass
class GazeResult:
    ok: bool
    yaw_rad: float = 0.0
    pitch_rad: float = 0.0
    confidence: float = 0.0
    face: Optional[Dict[str, float]] = None
    ts: float = 0.0
    raw: Optional[Dict[str, Any]] = None

class GazeBackendBase:
    def detect(self, frame_bgr: np.ndarray) -> GazeResult:
        raise NotImplementedError