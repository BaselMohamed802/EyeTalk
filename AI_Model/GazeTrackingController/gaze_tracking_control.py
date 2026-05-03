import os
import time
import yaml
import cv2
import pyautogui
import sys

# --- Project-level imports (parent / root) ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from VisionModule.camera import IrisCamera
from PyautoguiController.mouse_controller import MouseController

from GazeModule.gaze_backend_l2cs_roboflow import RoboflowL2CSBackend
from GazeModule.gaze_control_core import GazeControlCore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_PATH = os.path.join(ROOT, "GazeTrackingController", "config.yaml")

def load_cfg():
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def main():
    cfg = load_cfg()

    cam_id = int(cfg["camera"]["id"])
    cam_w = int(cfg["camera"]["width"])
    cam_h = int(cfg["camera"]["height"])

    gcfg = cfg["gaze"]
    rfcfg = gcfg["roboflow"]

    api_key = "FcYyztnZpboFQJnLzko6"
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY environment variable is not set.")

    backend = RoboflowL2CSBackend(
        host=str(rfcfg["host"]),
        port=int(rfcfg["port"]),
        endpoint=str(rfcfg["endpoint"]),
        api_key=api_key,
        timeout_s=float(rfcfg["timeout_s"]),
        jpeg_quality=int(rfcfg["jpeg_quality"]),
    )

    screen_w, screen_h = pyautogui.size()
    core = GazeControlCore(screen_w, screen_h)

    core.min_confidence = float(gcfg.get("confidence_gate", 0.50))
    core.deadzone_px = float(gcfg["control"]["deadzone_px"])
    core.movement_threshold_px = float(gcfg["control"]["movement_threshold_px"])

    # Filter settings
    sm = gcfg["smoothing"]
    k = sm["kalman"]
    core.configure_filter(
        filter_type=str(sm["filter"]),
        ema_alpha=float(sm["ema_alpha"]),
        k_dt=float(k["dt"]),
        k_q=float(k["process_noise"]),
        k_r=float(k["measurement_noise"]),
    )

    # Load calibration if exists
    calib_path = os.path.join(ROOT, gcfg["calibration"]["file"])
    loaded = core.cal.load(calib_path)
    print(f"[Gaze] Calibration loaded: {loaded} ({calib_path})")

    mouse = MouseController(update_interval=0.01, enable_failsafe=False)
    mouse.start()
    mouse_enabled = True

    inference_hz = float(gcfg.get("inference_hz", 15.0))
    min_dt = 1.0 / max(1.0, inference_hz)
    last_infer = 0.0

    print("[Gaze] Controls: q=quit | m=toggle mouse | c=calibrate (stub)")

    with IrisCamera(cam_id, cam_w, cam_h) as cam:
        while True:
            frame = cam.get_frame()
            if frame is None:
                continue

            now = time.time()
            if (now - last_infer) >= min_dt:
                last_infer = now

                res = backend.detect(frame)
                if not res.ok:
                    print("[Gaze][backend fail]", res.raw)
                else:
                    if res.raw:
                        print("[Gaze][raw]", res.raw if isinstance(res.raw, dict) else type(res.raw))
                if res.ok and core.cal.model.calibrated:
                    should_send, x, y, dbg = core.update(res.yaw_rad, res.pitch_rad, res.confidence)
                    if mouse_enabled and should_send:
                        mouse.set_position(int(x), int(y))

                # debug overlay
                txt = "NO FACE"
                if res.ok:
                    txt = f"yaw={res.yaw_rad:+.3f} rad  pitch={res.pitch_rad:+.3f} rad  conf={res.confidence:.2f}"
                cv2.putText(frame, txt, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow("EyeTalk - Gaze Mode", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("m"):
                mouse_enabled = not mouse_enabled
                print(f"[Gaze] mouse_enabled={mouse_enabled}")
            if key == ord("c"):
                print("[Gaze] Calibration routine will be added next (Phase: Gaze calibration UI).")

    mouse.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
