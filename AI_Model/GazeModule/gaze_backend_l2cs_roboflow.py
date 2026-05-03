import base64
import time
import cv2
import numpy as np
import requests

from .gaze_backend_base import GazeBackendBase, GazeResult

class RoboflowL2CSBackend(GazeBackendBase):
    def __init__(self, host: str, port: int, endpoint: str, api_key: str,
                 timeout_s: float = 0.35, jpeg_quality: int = 70):
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.jpeg_quality = int(max(30, min(95, jpeg_quality)))
        self.url = f"http://{host}:{port}{endpoint}?api_key={api_key}"

    def detect(self, frame_bgr: np.ndarray) -> GazeResult:
        ts = time.time()
        try:
            ok, enc = cv2.imencode(
                ".jpg",
                frame_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not ok:
                return GazeResult(ok=False, ts=ts)

            img_b64 = base64.b64encode(enc).decode("utf-8")
            resp = requests.post(
                self.url,
                json={"api_key": self.api_key, "image": {"type": "base64", "value": img_b64}},
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()

            preds = (data[0].get("predictions") if isinstance(data, list) and data else None) or []
            if not preds:
                return GazeResult(ok=False, ts=ts, raw={"no_face": True})

            g = preds[0]
            face = g.get("face", None)
            conf = 0.0
            if isinstance(face, dict):
                conf = float(face.get("confidence", 0.0))

            return GazeResult(
                ok=True,
                yaw_rad=float(g.get("yaw", 0.0)),
                pitch_rad=float(g.get("pitch", 0.0)),
                confidence=conf,
                face=face if isinstance(face, dict) else None,
                ts=ts,
                raw=g,
            )
        except Exception as e:
            return GazeResult(ok=False, ts=ts, raw={"error": str(e)})