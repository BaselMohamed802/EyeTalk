"""
Filename: working_3d_calibration.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 1/28/2026

Description:
    WORKING 3D calibration with proper timing and sampling.
    This fixes the issues with simple_calibration_runner.py
"""

import cv2
import sys
import time
import threading
from PySide6.QtWidgets import QApplication
from VisionModule.camera import IrisCamera
from VisionModule.face_utils import FaceMeshDetector
from gaze_vector_mapper import GazeVectorMapper, GazeVectorCalibrator
from Calibration.calibration_ui import CalibrationWindow


class Working3DCalibration:
    """Working 3D calibration with proper timing."""
    
    def __init__(self, camera_id=0, num_points=9):
        self.camera_id = camera_id
        self.num_points = num_points
        
        # Initialize components
        self.camera = None
        self.face_detector = None
        self.gaze_mapper = None
        self.calibrator = None
        self.ui = None
        
        # State
        self.is_running = False
        self.current_point = 0
        self.samples_per_point = 30  # Collect 30 samples per point
        self.current_samples = 0
        
        # Threading
        self.calibration_thread = None
        
        print(f"\n{'='*60}")
        print("WORKING 3D GAZE VECTOR CALIBRATION")
        print(f"{'='*60}")
    
    def initialize(self):
        """Initialize all components."""
        print("\nInitializing...")
        
        # Initialize camera
        self.camera = IrisCamera(self.camera_id)
        if not self.camera.is_opened():
            print(f"Failed to open camera {self.camera_id}")
            return False
        
        # Initialize face mesh detector
        self.face_detector = FaceMeshDetector(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize gaze mapper and calibrator
        self.gaze_mapper = GazeVectorMapper()
        self.calibrator = GazeVectorCalibrator()
        
        print("Initialization complete!")
        return True
    
    def start_calibration(self):
        """Start the calibration process."""
        # Start Qt application
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # Create and show calibration UI
        self.ui = CalibrationWindow(num_points=self.num_points)
        self.ui.show()
        
        # Start calibration thread
        self.is_running = True
        self.calibration_thread = threading.Thread(target=self._calibration_loop)
        self.calibration_thread.start()
        
        print("\nCalibration started!")
        print("Look at each dot as it appears.")
        print("Press SPACE to advance to next point.")
        print("Press ESC to cancel calibration.")
        
        # Run Qt event loop
        app.exec()
        
        # Cleanup
        self.is_running = False
        if self.calibration_thread:
            self.calibration_thread.join()
    
    def _calibration_loop(self):
        """Main calibration loop (runs in thread)."""
        last_point = -1
        
        while self.is_running and self.ui and self.ui.isVisible():
            # Get current point from UI
            current_point = self.ui.current_point
            
            # Check if point changed
            if current_point != last_point:
                last_point = current_point
                self.current_samples = 0
                
                # Get screen position for current point
                if current_point < len(self.ui.calibration_points):
                    x_norm, y_norm = self.ui.calibration_points[current_point]
                    
                    # Start calibration at this point
                    self.calibrator.start_point((x_norm, y_norm))
                    print(f"\n[Point {current_point + 1}/{self.num_points}]")
                    print(f"Screen position: ({x_norm:.3f}, {y_norm:.3f})")
                    print("Collecting samples...")
            
            # Get frame from camera
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Process frame with face mesh
            processed_frame, landmarks_list = self.face_detector.draw_face_mesh(
                frame,
                draw_face=False,
                draw_iris=True,
                draw_tesselation=False,
                draw_contours=False
            )
            
            # Check if we have 3D landmarks
            if (hasattr(self.face_detector, 'results') and 
                self.face_detector.results.multi_face_landmarks):
                
                face_landmarks = self.face_detector.results.multi_face_landmarks[0]
                landmarks_3d = face_landmarks.landmark
                
                # Add calibration sample
                success = self.calibrator.add_sample(landmarks_3d)
                
                if success:
                    self.current_samples += 1
                    
                    # Show progress
                    if self.current_samples % 10 == 0:
                        print(f"  Collected {self.current_samples}/{self.samples_per_point} samples")
                    
                    # Check if we have enough samples for this point
                    if self.current_samples >= self.samples_per_point:
                        print(f"✓ Point {current_point + 1} complete: {self.current_samples} samples")
                        
                        # If not last point, wait for user to press SPACE
                        # (UI handles this automatically)
            
            # Show camera feed (optional)
            # cv2.imshow('Calibration - 3D Mode', processed_frame)
            # cv2.waitKey(1)
            
            # Small delay to prevent CPU overload
            time.sleep(0.01)
        
        # Calibration loop ended
        self._finish_calibration()
    
    def _finish_calibration(self):
        """Finish calibration and train model."""
        print("\n" + "="*60)
        print("Calibration data collection complete!")
        
        # Check if we have enough samples
        if len(self.calibrator.samples) >= 4:
            print(f"Training with {len(self.calibrator.samples)} samples...")
            
            # Train the gaze mapper
            success = self.calibrator.calibrate(self.gaze_mapper)
            
            if success:
                print("✓ 3D calibration successful!")
                
                # Save calibration
                filename = "3d_gaze_calibration_working.pkl"
                if self.gaze_mapper.save_calibration(filename):
                    print(f"✓ Calibration saved to {filename}")
                
                # Show calibration summary
                self._show_calibration_summary()
                
                return self.gaze_mapper
            else:
                print("✗ 3D calibration failed!")
        else:
            print(f"✗ Not enough samples: {len(self.calibrator.samples)} (need at least 4)")
        
        print("="*60)
        return None
    
    def _show_calibration_summary(self):
        """Show calibration summary."""
        status = self.gaze_mapper.get_status()
        
        print("\n" + "="*50)
        print("CALIBRATION SUMMARY")
        print("="*50)
        print(f"Is calibrated: {status['is_calibrated']}")
        
        if status['plane_normal'] is not None:
            normal = status['plane_normal']
            print(f"Screen plane normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
        
        print(f"Samples collected: {len(self.calibrator.samples)}")
        print("="*50)
    
    def cleanup(self):
        """Clean up resources."""
        if self.camera:
            self.camera.stop_recording()
        
        if self.face_detector:
            self.face_detector.release()
        
        cv2.destroyAllWindows()
        print("\nCleanup complete!")


def run_working_3d_calibration():
    """Run the working 3D calibration."""
    # Create and run calibration
    calibration = Working3DCalibration(camera_id=1, num_points=9)
    
    if calibration.initialize():
        try:
            # Start calibration
            calibration.start_calibration()
            
            # Get the trained mapper
            gaze_mapper = calibration.gaze_mapper
            
            if gaze_mapper and gaze_mapper.is_calibrated:
                print("\n" + "="*60)
                print("3D CALIBRATION SUCCESSFUL!")
                print("="*60)
                print("\nYour gaze mapper is ready to use!")
                print("\nTo use it in your application:")
                print("1. Load calibration: gaze_mapper.load_calibration('3d_gaze_calibration_working.pkl')")
                print("2. For each frame: gaze_mapper.update(landmarks_3d)")
                print("3. Get prediction: coords = gaze_mapper.get_screen_coordinates()")
                print("="*60)
                
                return gaze_mapper
            else:
                print("\nCalibration failed or was cancelled.")
                return None
                
        except KeyboardInterrupt:
            print("\nCalibration interrupted by user.")
        finally:
            calibration.cleanup()
    else:
        print("Failed to initialize calibration.")
    
    return None


def quick_test_prediction(gaze_mapper):
    """Quick test of prediction after calibration."""
    if not gaze_mapper or not gaze_mapper.is_calibrated:
        print("No calibrated gaze mapper available.")
        return
    
    print("\n" + "="*60)
    print("QUICK PREDICTION TEST")
    print("="*60)
    print("This will test gaze prediction for 10 seconds.")
    print("Look around the screen to see predictions.")
    print("Press 'q' to quit early.")
    
    # Quick setup for testing
    camera = IrisCamera(0)
    face_detector = FaceMeshDetector(refine_landmarks=True)
    
    start_time = time.time()
    test_duration = 10  # seconds
    
    while time.time() - start_time < test_duration:
        frame = camera.get_frame()
        if frame is None:
            continue
        
        # Get landmarks
        _, landmarks_list = face_detector.draw_face_mesh(
            frame, draw_face=False, draw_iris=True
        )
        
        if (hasattr(face_detector, 'results') and 
            face_detector.results.multi_face_landmarks):
            
            face_landmarks = face_detector.results.multi_face_landmarks[0]
            landmarks_3d = face_landmarks.landmark
            
            # Update gaze mapper
            if gaze_mapper.update(landmarks_3d):
                # Get prediction
                coords = gaze_mapper.get_screen_coordinates()
                
                if coords:
                    u, v = coords
                    print(f"Prediction: ({u:.3f}, {v:.3f})")
        
        # Show frame
        cv2.imshow('Test Prediction', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    camera.stop_recording()
    face_detector.release()
    cv2.destroyAllWindows()
    
    print("\nTest complete!")
    print("="*60)


if __name__ == "__main__":
    print("WORKING 3D GAZE VECTOR CALIBRATION")
    print("\nThis version fixes the sampling issues.")
    print("\nInstructions:")
    print("1. Look at each white dot as it appears")
    print("2. The system will automatically collect 30 samples per point")
    print("3. Press SPACE to advance to next point")
    print("4. Press ESC to cancel at any time")
    
    input("\nPress Enter to start calibration...")
    
    # Run calibration
    gaze_mapper = run_working_3d_calibration()
    
    # Ask if user wants to test
    if gaze_mapper:
        test = input("\nTest prediction? (y/n): ").strip().lower()
        if test == 'y':
            quick_test_prediction(gaze_mapper)
    
    print("\nDone!")