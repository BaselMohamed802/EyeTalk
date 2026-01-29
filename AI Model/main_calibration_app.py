"""
Filename: main_calibration_app.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 1/28/2026

Description:
    Main application for running calibration with the integrated system.
    This ties together camera, face detection, and calibration.
"""

import sys
import cv2
import time
import argparse
from PySide6.QtWidgets import QApplication
import mediapipe as mp

# Import our modules
from VisionModule.camera import IrisCamera, list_available_cameras
from VisionModule.face_utils import FaceMeshDetector
from calibration_integrated import IntegratedCalibrationSystem, HybridCalibrationSystem
from Calibration.calibration_ui import CalibrationWindow


class CalibrationApplication:
    """Main calibration application."""
    
    def __init__(self, mode="2d", camera_id=0, num_points=9):
        """
        Initialize calibration application.
        
        Args:
            mode: "2d" or "3d"
            camera_id: Camera device ID
            num_points: Number of calibration points
        """
        self.mode = mode.lower()
        self.camera_id = camera_id
        self.num_points = num_points
        
        # Initialize components
        self.camera = None
        self.face_detector = None
        self.calibration_system = None
        self.ui = None
        
        # State
        self.is_running = False
        self.is_calibrating = False
        self.calibration_complete = False
        
        # Statistics
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"EYE TRACKING CALIBRATION APPLICATION")
        print(f"{'='*60}")
        print(f"Mode: {self.mode.upper()}")
        print(f"Camera: {camera_id}")
        print(f"Points: {num_points}")
        print(f"{'='*60}")
    
    def initialize(self):
        """Initialize all components."""
        print("\nInitializing components...")
        
        try:
            # Initialize camera
            print("Initializing camera...")
            self.camera = IrisCamera(self.camera_id)
            if not self.camera.is_opened():
                print(f"Failed to open camera {self.camera_id}")
                return False
            
            # Initialize face mesh detector
            print("Initializing face mesh detector...")
            self.face_detector = FaceMeshDetector(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Initialize calibration system
            print(f"Initializing {self.mode.upper()} calibration system...")
            self.calibration_system = IntegratedCalibrationSystem(
                mode=self.mode,
                num_points=self.num_points
            )
            
            # Initialize UI (will be shown later)
            print("UI ready (will show when calibration starts)")
            
            print("\nInitialization complete!")
            return True
            
        except Exception as e:
            print(f"Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def start_calibration(self):
        """Start the calibration process."""
        if not self.calibration_system:
            print("Calibration system not initialized")
            return
        
        print(f"\nStarting {self.mode.upper()} calibration...")
        
        # Setup UI
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        self.ui = CalibrationWindow(num_points=self.num_points)
        self.ui.show()
        
        # Start calibration system
        self.calibration_system.start_calibration()
        
        self.is_calibrating = True
        self.is_running = True
        
        print("\nCalibration started!")
        print("Look at each dot as it appears.")
        print("Press SPACE to move to next point.")
        print("Press ESC to cancel.")
    
    def process_frame(self, frame):
        """
        Process a single frame.
        
        Returns:
            Processed frame and whether calibration is complete
        """
        if not self.is_running or not self.is_calibrating:
            return frame, False
        
        # Update FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = current_time
        
        # Process frame with face mesh
        processed_frame, landmarks_list = self.face_detector.draw_face_mesh(
            frame,
            draw_face=False,
            draw_iris=True,
            draw_tesselation=False,
            draw_contours=False
        )
        
        # Extract iris centers for 2D mode
        iris_centers = self.face_detector.get_iris_centers(frame)
        
        # Add sample to calibration system
        if self.mode == "2d" and iris_centers:
            # For 2D mode: use iris centers
            self.calibration_system.add_sample(
                iris_centers_2d=iris_centers
            )
        
        elif self.mode == "3d" and landmarks_list:
            # For 3D mode: need 3D landmarks
            # Note: MediaPipe face_mesh returns 3D landmarks when we process
            # We need to access the raw results
            if hasattr(self.face_detector, 'results') and self.face_detector.results.multi_face_landmarks:
                face_landmarks = self.face_detector.results.multi_face_landmarks[0]
                landmarks_3d = face_landmarks.landmark  # These are 3D landmarks!
                
                self.calibration_system.add_sample(
                    landmarks_3d=landmarks_3d
                )
        
        # Update calibration system
        self.calibration_system.update()
        
        # Add FPS display
        cv2.putText(processed_frame, f"FPS: {self.fps}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add mode display
        cv2.putText(processed_frame, f"Mode: {self.mode.upper()}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add calibration status
        if self.is_calibrating:
            cv2.putText(processed_frame, "CALIBRATING", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Check if UI is closed (calibration complete)
        if self.ui and not self.ui.isVisible():
            self.finish_calibration()
            return processed_frame, True
        
        return processed_frame, False
    
    def finish_calibration(self):
        """Finish calibration and train model."""
        print("\n" + "="*60)
        print("Calibration data collection complete!")
        print("Training model...")
        print("="*60)
        
        self.is_calibrating = False
        
        # Train the model
        success = self.calibration_system.calibrate()
        
        if success:
            print("\n✓ Calibration successful!")
            self.calibration_complete = True
            
            # Save calibration
            filename = f"calibration_{self.mode}.pkl"
            if self.calibration_system.save_calibration(filename):
                print(f"✓ Calibration saved to {filename}")
            
            # Show calibration info
            status = self.calibration_system.get_status()
            print("\nCalibration Status:")
            print(f"Mode: {status['mode'].upper()}")
            print(f"Points: {status['num_points']}")
            print(f"Calibrated: {status['is_calibrated']}")
            
        else:
            print("\n✗ Calibration failed!")
        
        print("="*60)
    
    def run_live_preview(self):
        """Run live preview without calibration."""
        print("\nRunning live preview...")
        print("Press 'q' to quit")
        print("Press 'c' to start calibration")
        
        self.is_running = True
        
        while self.is_running:
            # Get frame from camera
            frame = self.camera.get_frame()
            if frame is None:
                print("Failed to get frame")
                break
            
            # Process frame
            processed_frame, _ = self.process_frame(frame)
            
            # Show frame
            cv2.imshow('Eye Tracking - Live Preview', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.start_calibration()
                # Once calibration starts, we need to handle Qt events
                # This gets more complex with mixed OpenCV/Qt
                print("Note: Calibration UI uses Qt, may need separate window")
                break
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        
        if self.camera:
            self.camera.stop_recording()
        
        if self.face_detector:
            self.face_detector.release()
        
        cv2.destroyAllWindows()
        print("Cleanup complete!")
    
    def run_test_prediction(self):
        """Test prediction after calibration."""
        if not self.calibration_complete:
            print("Cannot test: calibration not complete")
            return
        
        print("\n" + "="*60)
        print("Testing prediction...")
        print("="*60)
        
        test_count = 0
        max_tests = 100
        
        while test_count < max_tests:
            # Get frame
            frame = self.camera.get_frame()
            if frame is None:
                break
            
            # Process for landmarks
            _, landmarks_list = self.face_detector.draw_face_mesh(
                frame, draw_face=False, draw_iris=True
            )
            
            if landmarks_list:
                if self.mode == "2d":
                    # Get iris centers for 2D prediction
                    iris_centers = self.face_detector.get_iris_centers(frame)
                    if iris_centers.get('left') and iris_centers.get('right'):
                        # Average both eyes
                        left_x, left_y = iris_centers['left']
                        right_x, right_y = iris_centers['right']
                        eye_x = (left_x + right_x) / 2
                        eye_y = (left_y + right_y) / 2
                        
                        # Predict
                        prediction = self.calibration_system.predict((eye_x, eye_y))
                        
                        if prediction:
                            screen_x, screen_y = prediction
                            print(f"Prediction: ({screen_x:.3f}, {screen_y:.3f})")
                
                elif self.mode == "3d":
                    # Get 3D landmarks for 3D prediction
                    if hasattr(self.face_detector, 'results') and self.face_detector.results.multi_face_landmarks:
                        face_landmarks = self.face_detector.results.multi_face_landmarks[0]
                        landmarks_3d = face_landmarks.landmark
                        
                        # Predict
                        prediction = self.calibration_system.predict(landmarks_3d)
                        
                        if prediction:
                            screen_x, screen_y = prediction
                            print(f"Prediction: ({screen_x:.3f}, {screen_y:.3f})")
            
            test_count += 1
            
            # Show frame
            cv2.imshow('Test Prediction', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Eye Tracking Calibration')
    parser.add_argument('--mode', type=str, default='2d',
                       choices=['2d', '3d'],
                       help='Calibration mode: 2d (regression) or 3d (gaze vectors)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--points', type=int, default=9,
                       choices=[4, 9, 16],
                       help='Number of calibration points')
    parser.add_argument('--list-cameras', action='store_true',
                       help='List available cameras and exit')
    
    args = parser.parse_args()
    
    if args.list_cameras:
        print("Available cameras:")
        cameras = list_available_cameras()
        if not cameras:
            print("No cameras found")
        return
    
    # Create and run application
    app = CalibrationApplication(
        mode=args.mode,
        camera_id=args.camera,
        num_points=args.points
    )
    
    if app.initialize():
        # Run calibration
        print("\nStarting calibration workflow...")
        app.run_live_preview()
        
        # After calibration, you could run test prediction
        # app.run_test_prediction()
    else:
        print("Failed to initialize application")


if __name__ == "__main__":
    main()