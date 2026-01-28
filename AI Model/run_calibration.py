"""
Filename: run_calibration.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 1/27/2026

Purpose:
    Phase 3 "glue" script:
    - Shows Calibration UI (dots)
    - Drives timing (settle/sample per point)
    - Captures frames from camera
    - Extracts iris centers via FaceMeshDetector
    - Samples pairs via CalibrationSampler
    - Trains regression model via CalibrationRegression
    - Saves calibration_model.pkl
"""

import os
import sys
import time

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

# Phase 3 modules
try:
    from Calibration.calibration_ui import CalibrationWindow
    from Calibration.calibration_logic import CalibrationTimingController
    from Calibration.calibration_sampler import CalibrationSampler, CalibrationCoordinator
    from Calibration.calibration_model import CalibrationRegression
except ImportError:
    # Try alternative import path
    sys.path.insert(0, os.path.join(ROOT_DIR, 'AI Model'))
    from Calibration.calibration_ui import CalibrationWindow
    from Calibration.calibration_logic import CalibrationTimingController
    from Calibration.calibration_sampler import CalibrationSampler, CalibrationCoordinator
    from Calibration.calibration_model import CalibrationRegression

# Vision modules
try:
    from VisionModule.camera import IrisCamera
    from VisionModule.face_utils import FaceMeshDetector
except ImportError:
    # Try alternative import path
    sys.path.insert(0, os.path.join(ROOT_DIR, 'AI Model'))
    from VisionModule.camera import IrisCamera
    from VisionModule.face_utils import FaceMeshDetector


def run_calibration(
    model_out_path: str = "calibration_model.pkl",
    camera_id: int = 0,
    num_points: int = 9,
    settle_duration: float = 1.0,        # Increased from 0.8s
    sample_duration: float = 2.0,        # Increased from 1.2s - more samples!
    min_samples_per_point: int = 15,     # Increased from 8 - more data!
    filter_type_for_runtime: str = "ema",
):
    app = QApplication(sys.argv)

    # -------------------------
    # Init UI
    # -------------------------
    window = CalibrationWindow(num_points=num_points)
    window.show()

    # Use the UI-generated points (normalized screen coords)
    calibration_points = window.calibration_points
    if not calibration_points or len(calibration_points) < 4:
        raise RuntimeError("CalibrationWindow did not generate enough calibration points.")

    # -------------------------
    # Init camera + detector
    # -------------------------
    camera = IrisCamera(cam_id=camera_id)
    detector = FaceMeshDetector()

    # -------------------------
    # Init timing controller with UI reference
    # -------------------------
    controller = CalibrationTimingController(
        num_points=num_points,
        settling_delay=settle_duration,
        ui=window
    )

    # -------------------------
    # Init sampler with increased sampling parameters
    # -------------------------
    sampler = CalibrationSampler(
        sampling_duration=sample_duration,
        min_samples=min_samples_per_point,
        max_samples=min_samples_per_point * 5  # Increased to allow more samples
    )

    # -------------------------
    # Create coordinator to connect controller and sampler
    # -------------------------
    coordinator = CalibrationCoordinator(controller, sampler)

    # -------------------------
    # Main tick loop (Qt timer)
    # -------------------------
    timer = QTimer()
    timer.setInterval(15)  # ~66 Hz UI + sampling loop
    
    # Track if we're in the finalization phase
    finalizing = False
    last_point_shown = False

    def finish_and_exit():
        nonlocal finalizing
        finalizing = True
        
        try:
            # Small delay to ensure last point was shown
            time.sleep(0.5)
            
            # Get calibration pairs from sampler
            pairs = sampler.get_calibration_pairs()
            print(f"\n[RunCalib] Total pairs collected: {len(pairs)}")
            
            if len(pairs) < 4:
                print("[RunCalib] ERROR: Not enough calibration pairs collected (minimum 4 required)")
                print(f"[RunCalib] Collected only {len(pairs)} pairs.")
                print("[RunCalib] Model NOT saved.")
                
                # Show what we got for debugging
                if pairs:
                    print("[RunCalib] Collected pairs:")
                    for i, (screen_pos, eye_pos) in enumerate(pairs):
                        print(f"  {i+1}: Screen {screen_pos} -> Eye {eye_pos}")
                return

            # Train regression model
            reg = CalibrationRegression(model_type="linear")
            reg.add_calibration_pairs(pairs)

            ok = reg.train()
            if not ok:
                print("[RunCalib] Regression training failed. Model NOT saved.")
            else:
                saved = reg.save_model(model_out_path)
                if saved:
                    print(f"[RunCalib] Model saved -> {model_out_path}")
                    
                    # Optional: Show model metrics
                    metrics = reg.training_metrics
                    if metrics:
                        print(f"[RunCalib] Model Performance:")
                        print(f"  X-axis R²: {metrics.r2_x:.4f}")
                        print(f"  Y-axis R²: {metrics.r2_y:.4f}")
                        print(f"  X-axis MAE: {metrics.mae_x:.4f}")
                        print(f"  Y-axis MAE: {metrics.mae_y:.4f}")
                        print(f"  Max Error X: {metrics.max_error_x:.4f}")
                        print(f"  Max Error Y: {metrics.max_error_y:.4f}")
                        
                    # Optional: Show visualization
                    try:
                        reg.visualize_calibration_data(show_plot=False)
                    except:
                        print("[RunCalib] Could not generate visualization (matplotlib not installed?)")
                else:
                    print("[RunCalib] Model training ok but save failed.")

        except Exception as e:
            print(f"[RunCalib] Error during finalization: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Release resources no matter what
            try:
                camera.stop_recording()
            except Exception:
                pass
            try:
                detector.release()
            except Exception:
                pass

            window.close()
            app.quit()

    def tick():
        nonlocal last_point_shown
        
        # Allow window close to abort safely
        if not window.isVisible():
            print("[RunCalib] Window closed by user. Exiting.")
            timer.stop()
            finish_and_exit()
            return
            
        # Check if we're finalizing
        if finalizing:
            return

        # Update coordinator (which updates both controller and sampler)
        coordinator.update()
        
        # Get current point info
        point_info = controller.get_current_point_info()
        
        # Check if we're at the last point and it's been shown
        if point_info and point_info['index'] == num_points - 1:
            if not last_point_shown:
                print(f"\n[RunCalib] Last point ({num_points}) displayed")
                last_point_shown = True
        
        # Sample only if we're in sampling phase and have a current point
        if point_info and point_info.get('sampling_allowed'):
            frame = camera.get_frame()
            if frame is None:
                print("[RunCalib] No frame from camera")
                return

            # Get iris centers
            iris = detector.get_iris_centers(frame)
            if iris and iris.get("left") and iris.get("right"):
                # Add sample to coordinator
                success = coordinator.add_iris_sample(
                    left_iris=iris["left"],
                    right_iris=iris["right"]
                )
                
                # Get current stats
                stats = sampler.get_current_stats()
                if stats.get('is_sampling'):
                    progress = stats['sampling_progress'] * 100
                    print(f"\r[RunCalib] Point {point_info['index']+1}/{num_points}: "
                          f"{stats['samples_collected']} samples ({progress:.0f}%)", 
                          end="", flush=True)
            else:
                # Only show "no iris" message occasionally to avoid spam
                if time.time() % 1.0 < 0.1:  # Every ~1 second
                    print(f"\r[RunCalib] No iris detected - keep looking at the dot", end="", flush=True)
        
        # Check if calibration is complete
        # We need to check if controller has finished AND sampler is done
        if controller.current_point >= num_points - 1:
            # Check if we're at the last point AND not sampling anymore
            point_info = controller.get_current_point_info()
            if not point_info or (not sampler.is_sampling and not controller.waiting_for_fixation):
                # All points done and sampling complete
                print(f"\n[RunCalib] All {num_points} points completed!")
                timer.stop()
                
                # Show completion message on UI briefly
                window.setWindowTitle("Calibration Complete - Processing...")
                
                # Small delay to let user see last point
                QTimer.singleShot(500, finish_and_exit)

    timer.timeout.connect(tick)

    # -------------------------
    # Start
    # -------------------------
    print("[RunCalib] Starting calibration...")
    print(f"[RunCalib] Points: {num_points}")
    print(f"[RunCalib] Settle time: {settle_duration}s")
    print(f"[RunCalib] Sample time: {sample_duration}s (increased for better accuracy)")
    print(f"[RunCalib] Min samples per point: {min_samples_per_point}")
    print("[RunCalib] Look at each dot as it appears. Press ESC to cancel.")
    print("=" * 60)
    
    # Start the calibration process
    coordinator.start_calibration()
    
    # Start the timer
    timer.start()
    
    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    # You can adjust these parameters based on your needs:
    
    # For QUICK calibration (faster but less accurate):
    # run_calibration(
    #     model_out_path="calibration_model.pkl",
    #     camera_id=1,
    #     num_points=9,
    #     settle_duration=0.8,
    #     sample_duration=1.5,
    #     min_samples_per_point=10,
    # )
    
    # For STANDARD calibration (good balance):
    run_calibration(
        model_out_path="calibration_model.pkl",
        camera_id=1,
        num_points=9,
        settle_duration=1.0,
        sample_duration=2.5,
        min_samples_per_point=15,
    )
    
    # For HIGH QUALITY calibration (slower but most accurate):
    # run_calibration(
    #     model_out_path="calibration_model_high_quality.pkl",
    #     camera_id=1,
    #     num_points=16,  # More points!
    #     settle_duration=1.2,
    #     sample_duration=2.5,
    #     min_samples_per_point=20,
    # )