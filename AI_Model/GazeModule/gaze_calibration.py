from dataclasses import dataclass, asdict
from typing import List, Tuple
import json
import numpy as np

@dataclass
class GazeCalibModel:
    Ax: List[float]
    Ay: List[float]
    calibrated: bool = False

def _phi(yaw: float, pitch: float) -> np.ndarray:
    return np.array([1.0, yaw, pitch, yaw*yaw, yaw*pitch, pitch*pitch], dtype=float)

class GazeCalibration:
    def __init__(self):
        self.model = GazeCalibModel(Ax=[0]*6, Ay=[0]*6, calibrated=False)

    def fit(self, samples: List[Tuple[float, float, float, float]]):
        if len(samples) < 12:
            raise ValueError("Need more samples to fit calibration (recommend >= 12).")

        X = np.vstack([_phi(y, p) for (y, p, sx, sy) in samples])
        tx = np.array([sx for (_, _, sx, _) in samples], dtype=float)
        ty = np.array([sy for (_, _, _, sy) in samples], dtype=float)

        Ax, *_ = np.linalg.lstsq(X, tx, rcond=None)
        Ay, *_ = np.linalg.lstsq(X, ty, rcond=None)

        self.model = GazeCalibModel(Ax=Ax.tolist(), Ay=Ay.tolist(), calibrated=True)

    def predict(self, yaw_rad: float, pitch_rad: float) -> Tuple[float, float]:
        if not self.model.calibrated:
            raise RuntimeError("Gaze calibration model not calibrated.")
        v = _phi(float(yaw_rad), float(pitch_rad))
        x = float(np.dot(np.array(self.model.Ax, dtype=float), v))
        y = float(np.dot(np.array(self.model.Ay, dtype=float), v))
        return x, y

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.model), f, indent=2)

    def load(self, path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.model = GazeCalibModel(
                Ax=list(d.get("Ax", [0]*6)),
                Ay=list(d.get("Ay", [0]*6)),
                calibrated=bool(d.get("calibrated", False)),
            )
            return self.model.calibrated
        except Exception:
            return False