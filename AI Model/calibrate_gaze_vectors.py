"""
Filename: calibrate_gaze_vectors.py
Creator/Author: Basel Mohamed Mostafa Sayed + ChatGPT
Date: 1/27/2026

Purpose:
    Calibrate the 3D gaze vector system.
    Collects gaze vectors at known screen positions to calibrate screen plane.
"""

import os
import sys
import time
import numpy as np
import pickle
import cv2
from datetime import datetime

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

# Import existing modules
try:
    from Calibration.calibration_ui import CalibrationWindow
    from VisionModule.camera import IrisCamera
    from VisionModule.face_utils import FaceMeshDetector
    from gaze_vector_mapper import GazeVectorMapper, SimpleGazeVectorCalibrator
except ImportError:
    sys.path.insert(0, os.path.join(ROOT_DIR, 'AI Model'))
    from Calibration.calibration_ui import CalibrationWindow
    from VisionModule.camera import IrisCamera
    from VisionModule.face_utils import FaceMeshDetector
    
    # Import gaze vector mapper
    try:
        from gaze_vector_mapper import GazeVectorMapper, SimpleGazeVectorCalibrator
    except ImportError:
        print("ERROR: gaze_vector_mapper.py not found")
        print("Please save the gaze_vector_mapper.py file first")
        sys.exit(1)


class GazeVectorCalibration:
    """
    Calibrate 3D gaze vector system.
    Collects gaze vectors at known screen positions.
    """
    
    def __init__(self, camera_id: int = 1, num_points: int = 9):
        self.camera_id = camera_id
        self.num_points = num_points
        
        # Initialize components
        self.camera = None
        self.detector = None
        self.gaze_mapper = None
        self.calibrator = SimpleGazeVectorCalibrator()
        
        # Calibration data
        self.calibration_data = []  # List of (gaze_vector, screen_point)
        self.calibration_complete = False
        
    def initialize(self):
        """Initialize camera and detector."""
        print("[GazeCalibration] Initializing...")
        
        try:
            self.camera = IrisCamera(cam_id=self.camera_id)
            self.detector = FaceMeshDetector(refine_landmarks=True)  # IMPORTANT: 3D landmarks
            self.gaze_mapper = GazeVectorMapper()
            
            print("[GazeCalibration] Initialization successful")
            return True
            
        except Exception as e:
            print(f"[GazeCalibration] Initialization failed: {e}")
            return False
    
    def collect_calibration_point(self, screen_point: tuple, 
                                 settle_time: float = 1.0,
                                 collect_time: float = 2.0) -> bool:
        """
        Collect gaze vectors at a known screen point.
        
        Args:
            screen_point: Normalized screen coordinates (x, y) in [0, 1]
            settle_time: Time to settle gaze (seconds)
            collect_time: Time to collect data (seconds)
            
        Returns:
            True if successful
        """
        print(f"[GazeCalibration] Collecting data for screen point: {screen_point}")
        
        gaze_vectors = []
        start_time = time.time()
        
        print(f"[GazeCalibration] Settling for {settle_time}s...")
        time.sleep(settle_time)
        
        print(f"[GazeCalibration] Collecting gaze vectors for {collect_time}s...")
        collection_end = time.time() + collect_time
        
        samples_collected = 0
        successful_samples = 0
        
        while time.time() < collection_end:
            # Get frame
            frame = self.camera.get_frame()
            if frame is None:
                continue
            
            # Process frame
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.face_mesh.process(img_rgb)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Calculate gaze vector
                gaze_vector, debug_info = self.gaze_mapper.calculate_gaze_vector(
                    face_landmarks.landmark
                )
                
                if gaze_vector is not None:
                    gaze_vectors.append(gaze_vector)
                    successful_samples += 1
                
                samples_collected += 1
            
            # Small delay to not overload CPU
            time.sleep(0.01)
        
        # Average the collected gaze vectors
        if gaze_vectors:
            avg_gaze_vector = np.mean(gaze_vectors, axis=0)
            avg_gaze_vector = avg_gaze_vector / np.linalg.norm(avg_gaze_vector)  # Normalize
            
            # Add to calibration data
            self.calibration_data.append((avg_gaze_vector, screen_point))
            self.calibrator.add_calibration_point(avg_gaze_vector, screen_point)
            
            print(f"[GazeCalibration] Point collected: {screen_point}")
            print(f"  Samples: {successful_samples}/{samples_collected}")
            print(f"  Avg Gaze Vector: {avg_gaze_vector.tolist()}")
            print(f"  Total points: {len(self.calibration_data)}")
            
            return True
        else:
            print(f"[GazeCalibration] Failed to collect gaze vectors for point {screen_point}")
            return False
    
    def calibrate(self) -> bool:
        """Perform calibration from collected data."""
        if len(self.calibration_data) < 3:
            print(f"[GazeCalibration] Need at least 3 points, have {len(self.calibration_data)}")
            return False
        
        print("[GazeCalibration] Performing calibration...")
        
        # Calibrate the gaze mapper
        success = self.calibrator.calibrate(self.gaze_mapper)
        
        if success:
            print("[GazeCalibration] Calibration successful!")
            print(f"  Screen Normal: {self.gaze_mapper.screen_normal.tolist()}")
            print(f"  Screen Center: {self.gaze_mapper.screen_center.tolist()}")
            print(f"  Screen Distance: {self.gaze_mapper.screen_distance}")
            
            self.calibration_complete = True
            return True
        else:
            print("[GazeCalibration] Calibration failed")
            return False
    
    def save_calibration(self, filename: str = "gaze_vector_calibration.pkl") -> bool:
        """Save calibration data."""
        if not self.calibration_complete:
            print("[GazeCalibration] Not calibrated, cannot save")
            return False
        
        return self.gaze_mapper.save_calibration(filename)
    
    def close(self):
        """Release resources."""
        if self.camera:
            self.camera.stop_recording()
        if self.detector:
            self.detector.release()
        print("[GazeCalibration] Resources released")


def run_gaze_vector_calibration(
    output_path: str = "gaze_vector_calibration.pkl",
    camera_id: int = 1,
    num_points: int = 9,
    settle_duration: float = 1.0,
    collect_duration: float = 2.0
):
    """
    Run complete gaze vector calibration with UI.
    
    Args:
        output_path: Path to save calibration
        camera_id: Camera ID
        num_points: Number of calibration points
        settle_duration: Time to settle at each point
        collect_duration: Time to collect data at each point
    """
    app = QApplication(sys.argv)
    
    # Initialize UI
    window = CalibrationWindow(num_points=num_points)
    window.show()
    
    calibration_points = window.calibration_points
    if not calibration_points:
        raise RuntimeError("Failed to generate calibration points")
    
    # Initialize calibration system
    calibration = GazeVectorCalibration(camera_id=camera_id, num_points=num_points)
    if not calibration.initialize():
        print("Failed to initialize calibration system")
        return
    
    print("=" * 70)
    print("3D GAZE VECTOR CALIBRATION")
    print("=" * 70)
    print(f"Points: {num_points}")
    print(f"Settle time: {settle_duration}s")
    print(f"Collection time: {collect_duration}s")
    print("=" * 70)
    print("Look at each dot as it appears")
    print("Keep your head still during collection")
    print("=" * 70)
    
    # Setup timer
    timer = QTimer()
    timer.setInterval(16)  # ~60 Hz
    
    current_point = 0
    state = "SHOWING_POINT"  # SHOWING_POINT, SETTLING, COLLECTING
    state_start_time = 0
    
    def finish_calibration():
        print("\n" + "=" * 70)
        print("CALIBRATION COLLECTION COMPLETE")
        print("=" * 70)
        
        # Perform calibration
        success = calibration.calibrate()
        
        if success:
            # Save calibration
            calibration.save_calibration(output_path)
            print(f"[Calibration] Saved to {output_path}")
            
            # Show calibration results
            print("\nCalibration Results:")
            print(f"  Points collected: {len(calibration.calibration_data)}")
            print(f"  Screen Normal: {calibration.gaze_mapper.screen_normal.tolist()}")
            print(f"  Screen Distance: {calibration.gaze_mapper.screen_distance:.3f}m")
            
            # Test the calibration
            print("\nTesting calibration...")
            test_gaze_vector = np.array([0, 0, -1])  # Looking straight
            test_head = np.array([0, 0, 0])  # Head at origin
            
            intersection = calibration.gaze_mapper.intersect_with_screen(
                test_gaze_vector, test_head
            )
            if intersection is not None:
                print(f"  Test intersection: {intersection.tolist()}")
        
        else:
            print("[Calibration] Calibration failed")
        
        print("=" * 70)
        
        # Cleanup
        calibration.close()
        window.close()
        app.quit()
    
    def tick():
        nonlocal current_point, state, state_start_time
        
        # Check if window closed
        if not window.isVisible():
            timer.stop()
            finish_calibration()
            return
        
        # Check if done
        if current_point >= num_points:
            timer.stop()
            finish_calibration()
            return
        
        # State machine
        if state == "SHOWING_POINT":
            # Show current point
            screen_point = calibration_points[current_point]
            window.show_calibration_point(current_point)
            
            print(f"\n[Calibration] Point {current_point + 1}/{num_points}")
            print(f"  Screen: ({screen_point[0]:.2f}, {screen_point[1]:.2f})")
            
            state = "SETTLING"
            state_start_time = time.time()
        
        elif state == "SETTLING":
            # Wait for settle time
            if time.time() - state_start_time >= settle_duration:
                state = "COLLECTING"
                state_start_time = time.time()
                print(f"  Collecting gaze vectors...")
        
        elif state == "COLLECTING":
            # Collect data for this point
            screen_point = calibration_points[current_point]
            
            # Collect data (simplified - in real implementation, this would be incremental)
            if time.time() - state_start_time >= collect_duration:
                # Actually collect the data
                success = calibration.collect_calibration_point(
                    screen_point, 
                    settle_time=0,  # Already settled
                    collect_time=0.5  # Quick collection for demonstration
                )
                
                if success:
                    # Move to next point
                    current_point += 1
                    state = "SHOWING_POINT"
                else:
                    # Retry or skip
                    print(f"  Failed to collect point {current_point + 1}, skipping...")
                    current_point += 1
                    state = "SHOWING_POINT"
    
    timer.timeout.connect(tick)
    
    # Start timer
    timer.start()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    run_gaze_vector_calibration(
        output_path="gaze_vector_calibration.pkl",
        camera_id=1,
        num_points=9,
        settle_duration=1.0,
        collect_duration=2.0
    )