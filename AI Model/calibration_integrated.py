"""
Filename: calibration_integrated.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 1/28/2026

Description:
    Integrated calibration system supporting both:
    1. 2D Regression-based approach (original)
    2. 3D Gaze Vector approach (new)
    
    This integrates with existing timing controller, sampler, and UI.
"""

# Import necessary libraries
import sys
import os
import time
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

# Import existing modules
from Calibration.calibration_logic import CalibrationTimingController
from Calibration.calibration_sampler import CalibrationSampler, CalibrationCoordinator
from Calibration.calibration_model import CalibrationRegression
from Calibration.calibration_ui import CalibrationWindow

# Import the new gaze vector mapper
from gaze_vector_mapper import GazeVectorMapper, GazeVectorCalibrator, GazeRay


class IntegratedCalibrationSystem:
    """
    Integrated calibration system that supports both 2D and 3D approaches.
    
    Can run either:
    - Mode 1: Traditional 2D regression (uses iris pixel positions)
    - Mode 2: 3D gaze vectors (uses geometric intersection)
    """
    
    def __init__(self, mode: str = "2d", num_points: int = 9):
        """
        Initialize integrated calibration system.
        
        Args:
            mode: "2d" for regression, "3d" for gaze vectors
            num_points: Number of calibration points
        """
        self.mode = mode.lower()
        self.num_points = num_points
        
        # Initialize components based on mode
        if self.mode == "2d":
            # Traditional 2D regression pipeline
            self.timing_controller = CalibrationTimingController(num_points=num_points)
            self.sampler = CalibrationSampler()
            self.regression_model = CalibrationRegression()
            self.coordinator = CalibrationCoordinator(self.timing_controller, self.sampler)
            
            # Connect regression model to sampler
            def on_calibration_pair_ready(point_index, screen_pos, eye_pos):
                print(f"[2D Mode] Adding calibration pair for point {point_index}")
                self.regression_model.add_calibration_pair(screen_pos, eye_pos)
            
            self.sampler.on_calibration_pair_ready = on_calibration_pair_ready
            
        elif self.mode == "3d":
            # New 3D gaze vector pipeline
            self.gaze_mapper = GazeVectorMapper()
            self.gaze_calibrator = GazeVectorCalibrator()
            
            # We still use timing controller for point sequencing
            self.timing_controller = CalibrationTimingController(num_points=num_points)
            
            # But we need custom sampling logic for 3D
            self.current_point_samples = []  # Store 3D samples for current point
            self.calibration_samples_3d = []  # All collected 3D samples
            
        else:
            raise ValueError(f"Invalid mode: {mode}. Use '2d' or '3d'")
        
        # Common UI
        self.ui = None
        
        # Calibration state
        self.is_calibrated = False
        self.calibration_data = None
        
        print(f"\n{'='*60}")
        print(f"INTEGRATED CALIBRATION SYSTEM - {self.mode.upper()} MODE")
        print(f"{'='*60}")
        print(f"Points: {num_points}")
        print(f"{'='*60}")
    
    def setup_ui(self):
        """Setup the calibration UI."""
        self.ui = CalibrationWindow(num_points=self.num_points)
        return self.ui
    
    def start_calibration(self):
        """Start the calibration process."""
        print(f"\n[Integrated] Starting calibration in {self.mode.upper()} mode...")
        
        if self.mode == "2d":
            # Start 2D calibration pipeline
            self.coordinator.start_calibration()
            
        elif self.mode == "3d":
            # Start 3D calibration pipeline
            self._setup_3d_calibration()
            self.timing_controller.start_calibration()
            
            # Connect timing controller callbacks for 3D
            def on_point_started(point_index, x_norm, y_norm):
                print(f"\n[3D Mode] Starting point {point_index + 1}")
                print(f"[3D Mode] Screen position: ({x_norm:.3f}, {y_norm:.3f})")
                
                # Start collecting 3D samples for this point
                self.current_point_samples = []
                self.gaze_calibrator.start_point((x_norm, y_norm))
            
            def on_sampling_start(point_index):
                print(f"[3D Mode] Sampling allowed for point {point_index}")
                # In 3D mode, we'll collect samples during the main loop
            
            self.timing_controller.on_point_started = on_point_started
            self.timing_controller.on_sampling_start = on_sampling_start
            
            def on_calibration_complete():
                print("\n[3D Mode] All points displayed - ready to calibrate")
                # We'll calibrate after all points are shown
            
            self.timing_controller.on_calibration_complete = on_calibration_complete
    
    def _setup_3d_calibration(self):
        """Setup callbacks for 3D calibration."""
        # We'll handle 3D calibration differently
        pass
    
    def add_sample(self, landmarks_3d: Optional[List] = None, 
                   iris_centers_2d: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Add a calibration sample.
        
        Args:
            landmarks_3d: MediaPipe 3D landmarks (required for 3D mode)
            iris_centers_2d: Dictionary with 'left' and 'right' iris centers (pixels)
                           (required for 2D mode)
        
        Returns:
            True if sample was added successfully
        """
        if self.mode == "2d":
            # 2D mode: add iris centers to sampler
            if iris_centers_2d is None:
                return False
            
            left_iris = iris_centers_2d.get('left')
            right_iris = iris_centers_2d.get('right')
            
            if left_iris is None or right_iris is None:
                return False
            
            return self.coordinator.add_iris_sample(left_iris, right_iris)
        
        elif self.mode == "3d":
            # 3D mode: add 3D landmarks to calibrator
            if landmarks_3d is None:
                return False
            
            # Check if we're currently sampling
            point_info = self.timing_controller.get_current_point_info()
            if not point_info or not point_info['sampling_allowed']:
                return False
            
            # Add sample to 3D calibrator
            success = self.gaze_calibrator.add_sample(landmarks_3d)
            
            if success:
                # Store the sample with point info
                sample_data = {
                    'landmarks': landmarks_3d,
                    'point_index': point_info['index'],
                    'screen_uv': (point_info['x_norm'], point_info['y_norm']),
                    'timestamp': time.time()
                }
                self.current_point_samples.append(sample_data)
                
                # Check if we have enough samples for this point
                if len(self.current_point_samples) >= 30:  # Collect ~30 samples per point
                    self._finalize_3d_point()
            
            return success
        
        return False
    
    def _finalize_3d_point(self):
        """Finalize sampling for current 3D point."""
        if not self.current_point_samples:
            return
        
        point_info = self.timing_controller.get_current_point_info()
        if not point_info:
            return
        
        print(f"\n[3D Mode] Collected {len(self.current_point_samples)} samples for point {point_info['index'] + 1}")
        
        # Store samples
        self.calibration_samples_3d.extend(self.current_point_samples)
        
        # Request advance to next point
        self.timing_controller.advance_to_next_point()
    
    def update(self):
        """Update calibration state."""
        if self.mode == "2d":
            # Update 2D pipeline
            self.coordinator.update()
        elif self.mode == "3d":
            # Update 3D timing controller
            self.timing_controller.update()
    
    def calibrate(self) -> bool:
        """
        Perform calibration with collected data.
        
        Returns:
            True if calibration successful
        """
        print(f"\n[Integrated] Calibrating {self.mode.upper()} model...")
        
        if self.mode == "2d":
            # Train 2D regression model
            success = self.regression_model.train()
            if success:
                self.is_calibrated = True
                self.calibration_data = self.regression_model.get_model_info()
                
                # Save model
                self.regression_model.save_model("calibration_model_2d.pkl")
            
            return success
        
        elif self.mode == "3d":
            # Convert samples to format needed by gaze calibrator
            if not self.calibration_samples_3d:
                print("[3D Mode] No calibration samples collected")
                return False
            
            # Calibrate gaze mapper
            success = self.gaze_calibrator.calibrate(self.gaze_mapper)
            
            if success:
                self.is_calibrated = True
                self.calibration_data = self.gaze_mapper.get_status()
                
                # Save calibration
                self.gaze_mapper.save_calibration("gaze_calibration_3d.pkl")
            
            return success
        
        return False
    
    def predict(self, input_data) -> Optional[Tuple[float, float]]:
        """
        Predict screen coordinates from input data.
        
        Args:
            input_data: For 2D mode: (eye_x, eye_y) tuple
                       For 3D mode: landmarks_3d list
            
        Returns:
            Normalized screen coordinates (x, y) or None
        """
        if not self.is_calibrated:
            print("[Integrated] Model not calibrated")
            return None
        
        if self.mode == "2d":
            # 2D prediction
            if not isinstance(input_data, tuple) or len(input_data) != 2:
                print("[2D Mode] Invalid input: expected (eye_x, eye_y)")
                return None
            
            eye_x, eye_y = input_data
            return self.regression_model.predict(eye_x, eye_y)
        
        elif self.mode == "3d":
            # 3D prediction
            if not isinstance(input_data, list):
                print("[3D Mode] Invalid input: expected landmarks_3d list")
                return None
            
            # Update gaze mapper with new landmarks
            if self.gaze_mapper.update(input_data):
                return self.gaze_mapper.get_screen_coordinates()
            
            return None
        
        return None
    
    def get_pixel_coordinates(self, input_data, screen_width: int, screen_height: int) -> Optional[Tuple[int, int]]:
        """
        Get pixel coordinates on screen.
        
        Args:
            input_data: Input for prediction
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            
        Returns:
            (pixel_x, pixel_y) or None
        """
        coords = self.predict(input_data)
        if coords is None:
            return None
        
        x_norm, y_norm = coords
        pixel_x = int(x_norm * screen_width)
        pixel_y = int(y_norm * screen_height)
        
        return pixel_x, pixel_y
    
    def get_status(self) -> Dict[str, Any]:
        """Get calibration status."""
        status = {
            'mode': self.mode,
            'is_calibrated': self.is_calibrated,
            'num_points': self.num_points,
            'calibration_data': self.calibration_data
        }
        
        if self.mode == "2d" and hasattr(self, 'regression_model'):
            status['regression_info'] = self.regression_model.get_model_info()
        
        elif self.mode == "3d" and hasattr(self, 'gaze_mapper'):
            status['gaze_mapper_status'] = self.gaze_mapper.get_status()
        
        return status
    
    def save_calibration(self, filename: Optional[str] = None) -> bool:
        """Save calibration to file."""
        if not self.is_calibrated:
            return False
        
        if filename is None:
            filename = f"calibration_{self.mode}.pkl"
        
        if self.mode == "2d":
            return self.regression_model.save_model(filename)
        elif self.mode == "3d":
            return self.gaze_mapper.save_calibration(filename)
        
        return False
    
    def load_calibration(self, filename: str) -> bool:
        """Load calibration from file."""
        if self.mode == "2d":
            success = self.regression_model.load_model(filename)
            if success:
                self.is_calibrated = True
                self.calibration_data = self.regression_model.get_model_info()
            return success
        
        elif self.mode == "3d":
            success = self.gaze_mapper.load_calibration(filename)
            if success:
                self.is_calibrated = True
                self.calibration_data = self.gaze_mapper.get_status()
            return success
        
        return False


class HybridCalibrationSystem:
    """
    Hybrid system that can switch between 2D and 3D approaches.
    Also supports running both and comparing results.
    """
    
    def __init__(self, num_points: int = 9):
        """Initialize hybrid system with both 2D and 3D."""
        self.num_points = num_points
        
        # Initialize both systems
        self.system_2d = IntegratedCalibrationSystem(mode="2d", num_points=num_points)
        self.system_3d = IntegratedCalibrationSystem(mode="3d", num_points=num_points)
        
        # Active system
        self.active_system = None  # "2d", "3d", or "both"
        
        # Comparison data
        self.comparison_results = None
        
        print(f"\n{'='*60}")
        print("HYBRID CALIBRATION SYSTEM")
        print(f"{'='*60}")
        print("Supports: 2D Regression, 3D Gaze Vectors, or Both")
        print(f"{'='*60}")
    
    def set_active_mode(self, mode: str):
        """
        Set active calibration mode.
        
        Args:
            mode: "2d", "3d", or "both"
        """
        mode = mode.lower()
        if mode not in ["2d", "3d", "both"]:
            raise ValueError("Mode must be '2d', '3d', or 'both'")
        
        self.active_system = mode
        print(f"[Hybrid] Active mode set to: {mode.upper()}")
    
    def start_calibration(self):
        """Start calibration in active mode."""
        if self.active_system is None:
            print("[Hybrid] No active mode set")
            return
        
        if self.active_system == "2d":
            self.system_2d.start_calibration()
        elif self.active_system == "3d":
            self.system_3d.start_calibration()
        elif self.active_system == "both":
            print("[Hybrid] Starting both 2D and 3D calibration...")
            self.system_2d.start_calibration()
            self.system_3d.start_calibration()
    
    def add_sample(self, landmarks_3d: Optional[List] = None,
                   iris_centers_2d: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Add sample to active system(s).
        """
        if self.active_system is None:
            return False
        
        results = []
        
        if self.active_system in ["2d", "both"]:
            result_2d = self.system_2d.add_sample(
                landmarks_3d=landmarks_3d,
                iris_centers_2d=iris_centers_2d
            )
            results.append(("2D", result_2d))
        
        if self.active_system in ["3d", "both"]:
            result_3d = self.system_3d.add_sample(
                landmarks_3d=landmarks_3d,
                iris_centers_2d=iris_centers_2d
            )
            results.append(("3D", result_3d))
        
        # Return True if any system accepted the sample
        return any(result for _, result in results)
    
    def update(self):
        """Update active system(s)."""
        if self.active_system in ["2d", "both"]:
            self.system_2d.update()
        
        if self.active_system in ["3d", "both"]:
            self.system_3d.update()
    
    def calibrate_all(self) -> Dict[str, bool]:
        """Calibrate both systems and compare."""
        results = {}
        
        if self.active_system in ["2d", "both"]:
            print("\n[Hybrid] Calibrating 2D system...")
            results['2d'] = self.system_2d.calibrate()
        
        if self.active_system in ["3d", "both"]:
            print("\n[Hybrid] Calibrating 3D system...")
            results['3d'] = self.system_3d.calibrate()
        
        # Compare if both were calibrated
        if self.active_system == "both" and results.get('2d') and results.get('3d'):
            self._compare_calibrations()
        
        return results
    
    def _compare_calibrations(self):
        """Compare 2D and 3D calibration results."""
        print("\n[Hybrid] Comparing 2D vs 3D calibration...")
        
        # This would involve running both models on test data
        # For now, just report basic info
        print("Comparison functionality to be implemented")
        print("Would test both models on validation data")
    
    def predict(self, input_data, mode: Optional[str] = None) -> Optional[Tuple[float, float]]:
        """
        Predict using specified or active mode.
        
        Args:
            input_data: Input for prediction
            mode: "2d" or "3d" (uses active mode if None)
        """
        if mode is None:
            mode = self.active_system
        
        if mode == "2d":
            return self.system_2d.predict(input_data)
        elif mode == "3d":
            return self.system_3d.predict(input_data)
        
        return None


def run_calibration_demo():
    """Run a demonstration of the integrated calibration system."""
    import cv2
    import mediapipe as mp
    
    print("Integrated Calibration System Demo")
    print("="*60)
    
    # Ask user for mode
    print("\nSelect calibration mode:")
    print("1. 2D Regression (traditional)")
    print("2. 3D Gaze Vectors (new)")
    print("3. Both (compare)")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        mode = "2d"
    elif choice == "2":
        mode = "3d"
    elif choice == "3":
        mode = "both"
    else:
        print("Invalid choice, defaulting to 2D")
        mode = "2d"
    
    # Initialize system
    if mode == "both":
        system = HybridCalibrationSystem(num_points=9)
        system.set_active_mode("both")
    else:
        system = IntegratedCalibrationSystem(mode=mode, num_points=9)
    
    print(f"\nStarting {mode.upper()} calibration demo...")
    
    # Setup UI
    ui = system.setup_ui()
    
    # Note: In a real implementation, you would:
    # 1. Start the UI
    # 2. Capture frames from camera
    # 3. Detect faces and extract landmarks
    # 4. Add samples during calibration
    # 5. Train the model
    
    print("\nDemo setup complete!")
    print("To run actual calibration:")
    print("1. Start the UI with ui.show()")
    print("2. Capture camera frames")
    print("3. Extract landmarks with MediaPipe")
    print("4. Call system.add_sample() with landmarks")
    print("5. Call system.calibrate() when done")
    
    return system


if __name__ == "__main__":
    # Run demo
    system = run_calibration_demo()
    
    print("\n" + "="*60)
    print("Integrated calibration system ready!")
    print("="*60)