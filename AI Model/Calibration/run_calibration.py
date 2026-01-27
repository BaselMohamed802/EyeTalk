"""
Filename: run_calibration.py
Author: Basel Mohamed Mostafa Sayed

Purpose:
    Phase 3 Runner (Calibration)
    - Runs the calibration UI
    - Captures iris centers from MediaPipe (VisionModule)
    - Collects stable samples (CalibrationSampler/Coordinator)
    - Trains regression model (CalibrationRegression)
    - Saves model -> calibration_model.pkl

How it works:
    CalibrationTimingController drives which screen target is active + countdown timing.
    CalibrationCoordinator collects N stable samples per target and outputs calibration pairs.
    CalibrationRegression trains mapping (eye_x, eye_y) -> (screen_x, screen_y).
"""

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np


# -----------------------------------------------------------------------------
# Path setup (so imports work when running from project root)
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


# -----------------------------------------------------------------------------
# Imports from your modules
# -----------------------------------------------------------------------------
from VisionModule.camera import IrisCamera
from VisionModule.face_utils import FaceMeshDetector

from Calibration.calibration_logic import create_default_config, CalibrationTimingController
from Calibration.calibration_sampler import CalibrationCoordinator
from Calibration.calibration_ui import CalibrationUI
from Calibration.calibration_model import CalibrationRegression


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _safe_tuple2(t) -> Optional[tuple]:
    """Return (x,y) as floats if valid else None."""
    if t is None:
        return None
    try:
        x, y = t
        if x is None or y is None:
            return None
        return float(x), float(y)
    except Exception:
        return None


def _now() -> float:
    return time.time()


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run Phase 3 calibration and save regression model.")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device index.")
    parser.add_argument("--model-out", type=str, default="calibration_model.pkl", help="Output model filename/path.")
    parser.add_argument("--model-type", type=str, default="linear", choices=["linear", "polynomial", "ridge"],
                        help="Regression model type.")
    parser.add_argument("--poly-degree", type=int, default=2, help="Polynomial degree (if polynomial).")
    parser.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge regularization alpha (if ridge).")
    parser.add_argument("--debug", action="store_true", help="Print extra debug logs.")

    args = parser.parse_args()

    # 1) Build default calibration config (points, timings, sampling params)
    config = create_default_config()

    # 2) UI
    ui = CalibrationUI()

    # 3) Timing controller (drives current point + countdown)
    timing = CalibrationTimingController(config)

    # 4) Coordinator (collects stable eye samples -> calibration pairs)
    #    IMPORTANT: point_order must match config target points (same exact list).
    coordinator = CalibrationCoordinator(
        point_order=config.target_points,
        samples_per_point=config.samples_per_point,
        stability_threshold=config.stability_threshold,
        min_stable_samples=config.min_stable_samples,
    )

    # 5) Vision modules
    camera = IrisCamera(camera_id=args.camera_id)
    detector = FaceMeshDetector()

    # ----- Callbacks: keep UI synced with timing state -----
    def on_phase_changed(phase: str):
        ui.update_phase(phase)

    def on_point_started(idx: int):
        # idx is 0-based
        pt = config.target_points[idx]
        ui.update_target(pt[0], pt[1], idx + 1, len(config.target_points))

    def on_countdown_changed(seconds_remaining: int):
        ui.update_countdown(seconds_remaining)

    def on_complete():
        ui.update_phase("complete")
        ui.update_countdown(0)

    timing.set_callbacks(
        on_phase_changed=on_phase_changed,
        on_point_started=on_point_started,
        on_countdown_changed=on_countdown_changed,
        on_complete=on_complete,
    )

    # Initialize UI with first point
    ui.update_phase("ready")
    ui.update_target(config.target_points[0][0], config.target_points[0][1], 1, len(config.target_points))

    # Start timing
    timing.start()
    ui.update_phase("running")

    # Main loop
    last_t = _now()

    print("\n[CalibrationRunner] Starting calibration...")
    print(f"[CalibrationRunner] Points: {len(config.target_points)} | Samples/point: {config.samples_per_point}")
    print(f"[CalibrationRunner] Output model: {args.model_out}")

    try:
        while not timing.is_complete():
            t = _now()
            dt = t - last_t
            last_t = t

            # Update timing state + UI (via callbacks)
            timing.update(dt)

            # Current target on screen (normalized 0..1)
            point_idx = timing.get_current_point_index()
            if point_idx is None:
                # Not started yet
                time.sleep(0.005)
                continue

            screen_pos = config.target_points[point_idx]  # (screen_x_norm, screen_y_norm)

            # Get frame
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            # Get iris centers
            centers = detector.get_iris_centers(frame)
            left = _safe_tuple2(centers.get("left"))
            right = _safe_tuple2(centers.get("right"))

            if left is None or right is None:
                # No valid iris detection this frame
                time.sleep(0.001)
                continue

            # Feed sample to coordinator (it will decide stability + when a point is complete)
            coordinator.add_iris_sample(
                screen_position=screen_pos,
                left_iris=left,
                right_iris=right,
            )

            # Keep UI responsive
            ui.update()

            if args.debug:
                if coordinator.get_progress().get("total_pairs", 0) % 10 == 0:
                    print("[Debug] Progress:", coordinator.get_progress())

            time.sleep(0.001)

        # Done collecting
        print("[CalibrationRunner] Collection complete.")
        ui.update_phase("training")

        results = coordinator.get_results()
        pairs = results.get("pairs", [])

        print(f"[CalibrationRunner] Total calibration pairs: {len(pairs)}")
        if len(pairs) < 4:
            raise RuntimeError("Not enough calibration pairs collected to train a model (need >= 4).")

        # Train regression
        model = CalibrationRegression(
            model_type=args.model_type,
            polynomial_degree=args.poly_degree,
            regularization=args.ridge_alpha,
        )
        model.add_calibration_pairs(pairs)

        ok = model.train()
        if not ok:
            raise RuntimeError("Model training failed. Check your calibration point distribution and stability settings.")

        # Save model
        out_path = args.model_out
        model.save_model(out_path)

        ui.update_phase("complete")
        ui.update_countdown(0)

        print(f"[CalibrationRunner] Saved model to: {out_path}")
        print("[CalibrationRunner] Training metrics:")
        print(model.training_metrics)

        # Keep UI visible briefly
        t0 = _now()
        while _now() - t0 < 1.0:
            ui.update()
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[CalibrationRunner] Interrupted by user.")
    finally:
        # Cleanup
        try:
            ui.close()
        except Exception:
            pass
        try:
            camera.stop_recording()
        except Exception:
            pass
        try:
            detector.release()
        except Exception:
            pass

        print("[CalibrationRunner] Clean exit.")


if __name__ == "__main__":
    main()
