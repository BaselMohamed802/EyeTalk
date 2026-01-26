"""
Filename: runtime_tracker.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 1/26/2026

Description:
    Pure runtime eye tracking module - the "brain stem" of the system.
    
    Responsibilities:
    1. Load and use pre-trained regression model
    2. Map iris coordinates to normalized screen coordinates (0-1)
    3. Apply smoothing filters to screen coordinates only
    4. Handle screen resolution scaling
    5. Provide safety bounds and error handling
    
    Design Constraints:
    - NEVER calls train() or modifies the model
    - NEVER stores calibration data
    - NEVER touches MediaPipe directly
    - NEVER filters iris coordinates directly
    - ALWAYS filters predicted screen coordinates
    - Pure stateless mapping at runtime
"""

# Import necessary libraries
import numpy as np
from typing import Tuple, Optional, Dict, Any
import pickle
import sys
import os
import time

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Try to import filter manager
from Filters.filterManager import FilterManager

class RuntimeEyeTracker:
    """
    Pure runtime eye tracking module.
    
    This is the core mapping engine that converts iris coordinates
    to screen coordinates using a pre-trained regression model.
    """
    
    def __init__(self, 
                 model_path: str = "calibration_model.pkl",
                 filter_type: str = "ema",
                 screen_width: int = 1920,
                 screen_height: int = 1080,
                 safety_margin: float = 0.05):
        """
        Initialize the runtime tracker with a pre-trained model.
        
        Args:
            model_path: Path to the saved regression model
            filter_type: Type of filter for screen coordinates ("ema" or "kalman")
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            safety_margin: Safety margin (0-0.5) to keep predictions within bounds
        """
        # Screen properties
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.safety_margin = safety_margin
        
        # State
        self.model_loaded = False
        self.tracking_active = False
        self.last_prediction = None
        self.prediction_count = 0
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'failed_predictions': 0
        }

        # Load regression model
        print(f"[Runtime] Initializing runtime tracker...")
        print(f"[Runtime] Screen: {screen_width}x{screen_height}")
        print(f"[Runtime] Safety margin: {safety_margin}")
        
        if not self._load_model(model_path):
            raise RuntimeError(f"Failed to load model from {model_path}")
        
        # Initialize filters for screen coordinates
        self._init_filters(filter_type)
        
        print(f"[Runtime] Runtime tracker initialized successfully")
        print(f"[Runtime] Model type: {self.model_info.get('model_type', 'unknown')}")
        print(f"[Runtime] Filter type: {filter_type}")
    
    def _load_model(self, model_path: str) -> bool:
        """
        Load a pre-trained regression model from file.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
            
            # Extract model information
            self.model_info = {
                'model_type': model_data.get('model_type', 'linear'),
                'is_trained': model_data.get('is_trained', False),
                'training_metrics': model_data.get('training_metrics')
            }
            
            # Load model components
            self.model_x = model_data.get('model_x')
            self.model_y = model_data.get('model_y')
            
            # Basic validation
            if not self.model_info['is_trained']:
                print("[Runtime] Warning: Loaded model is not marked as trained")
            
            if self.model_x is None or self.model_y is None:
                print("[Runtime] Error: Model components not found in file")
                return False
            
            # Check if scikit-learn is available for sklearn models
            if (not isinstance(self.model_x, dict) or not isinstance(self.model_y, dict)):
                try:
                    import sklearn
                    self.sklearn_available = True
                except ImportError:
                    print("[Runtime] Warning: scikit-learn not available but model appears to be sklearn-based")
                    self.sklearn_available = False
            
            self.model_loaded = True
            print(f"[Runtime] Model loaded successfully from {model_path}")
            
            # Print model info
            if self.model_info['training_metrics']:
                metrics = self.model_info['training_metrics']
                if hasattr(metrics, '__dict__'):
                    r2_x = metrics.__dict__.get('r2_x', 0)
                    r2_y = metrics.__dict__.get('r2_y', 0)
                    print(f"[Runtime] Model R² scores: X={r2_x:.3f}, Y={r2_y:.3f}")
            
            return True
            
        except FileNotFoundError:
            print(f"[Runtime] Error: Model file not found at {model_path}")
            return False
        except Exception as e:
            print(f"[Runtime] Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _init_filters(self, filter_type: str):
        """Initialize filters for screen coordinates."""
        try:
            # Initialize filters for X and Y coordinates separately
            self.filter_x = FilterManager(filter_type=filter_type, alpha=0.3)
            self.filter_y = FilterManager(filter_type=filter_type, alpha=0.3)
            
            print(f"[Runtime] Filters initialized: {filter_type}")
        except Exception as e:
            print(f"[Runtime] Warning: Failed to initialize filters: {e}")
            print("[Runtime] Continuing without filtering")
            self.filter_x = None
            self.filter_y = None
    
    def start_tracking(self):
        """Start the tracking process."""
        self.tracking_active = True
        self.stats['total_predictions'] = 0
        self.stats['failed_predictions'] = 0
        print("[Runtime] Tracking started")
    
    def stop_tracking(self):
        """Stop the tracking process."""
        self.tracking_active = False
        print(f"[Runtime] Tracking stopped. Stats: {self.stats}")
    
    def predict_screen_coordinates(self, 
                                  left_iris: Tuple[float, float],
                                  right_iris: Tuple[float, float],
                                  use_average: bool = True) -> Tuple[float, float]:
        """
        Predict screen coordinates from iris coordinates.
        
        Args:
            left_iris: (x, y) coordinates of left iris (pixels)
            right_iris: (x, y) coordinates of right iris (pixels)
            use_average: Whether to average both eyes or use dominant eye
            
        Returns:
            (screen_x_norm, screen_y_norm) normalized coordinates (0-1)
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not self.tracking_active:
            print("[Runtime] Warning: Tracking not active, prediction may be unstable")
        
        # Calculate eye position
        if use_average:
            eye_x = (left_iris[0] + right_iris[0]) / 2.0
            eye_y = (left_iris[1] + right_iris[1]) / 2.0
        else:
            # Use left eye as dominant (can be configured)
            eye_x, eye_y = left_iris
        
        # Validate input
        if not self._validate_iris_coordinates(eye_x, eye_y):
            self.stats['failed_predictions'] += 1
            # Return last known good position or center
            if self.last_prediction:
                return self.last_prediction
            return 0.5, 0.5  # Screen center as fallback
        
        try:
            raw_screen_x, raw_screen_y = self._raw_regression_predict(eye_x, eye_y)

            # Filter FIRST
            if self.filter_x and self.filter_y:
                filtered_x = self.filter_x.apply(raw_screen_x)
                filtered_y = self.filter_y.apply(raw_screen_y)
            else:
                filtered_x, filtered_y = raw_screen_x, raw_screen_y

            # Apply safety margins AFTER filtering
            safe_x = self._apply_safety_margin(filtered_x, axis='x')
            safe_y = self._apply_safety_margin(filtered_y, axis='y')

            final_x = np.clip(safe_x, 0.0, 1.0)
            final_y = np.clip(safe_y, 0.0, 1.0)
            
            # Update state
            self.last_prediction = (final_x, final_y)
            self.prediction_count += 1
            self.stats['total_predictions'] += 1
            
            return final_x, final_y
            
        except Exception as e:
            self.stats['failed_predictions'] += 1
            print(f"[Runtime] Prediction error: {e}")
            
            # Return last known good position or center
            if self.last_prediction:
                return self.last_prediction
            return 0.5, 0.5
    
    def _raw_regression_predict(self, eye_x: float, eye_y: float) -> Tuple[float, float]:
        """
        Raw regression prediction without filtering or safety margins.
        
        Args:
            eye_x: Iris X coordinate
            eye_y: Iris Y coordinate
            
        Returns:
            Raw predicted screen coordinates (0-1)
        """
        # Prepare input
        X_input = np.array([[eye_x, eye_y]])

        if not isinstance(self.model_x, dict) and not self.sklearn_available:
            raise RuntimeError(
                "Sklearn-based model loaded but scikit-learn is not installed"
            )
        
        if hasattr(self, 'sklearn_available') and self.sklearn_available and \
           not isinstance(self.model_x, dict):
            # Use sklearn model
            screen_x = float(self.model_x.predict(X_input)[0])
            screen_y = float(self.model_y.predict(X_input)[0])
        elif isinstance(self.model_x, dict) and isinstance(self.model_y, dict):
            # Use simple linear model
            coeff_x = self.model_x['coeff']
            coeff_y = self.model_y['coeff']
            
            # Add bias term
            X_with_bias = np.array([[eye_x, eye_y, 1.0]])
            
            screen_x = float(X_with_bias @ coeff_x)
            screen_y = float(X_with_bias @ coeff_y)
        else:
            raise RuntimeError("Model type not recognized")
        
        return screen_x, screen_y
    
    def _apply_safety_margin(self, coordinate: float, axis: str = 'x') -> float:
        """
        Apply safety margin to keep coordinates within bounds.
        
        Args:
            coordinate: Input coordinate (0-1)
            axis: 'x' or 'y'
            
        Returns:
            Coordinate with safety margin applied
        """
        margin = self.safety_margin
        
        # Map from [0, 1] to [margin, 1-margin]
        if coordinate < 0:
            coordinate = 0
        elif coordinate > 1:
            coordinate = 1
        
        # Apply non-linear scaling to reduce sensitivity at edges
        # This keeps the center region more responsive while being careful at edges
        if coordinate < 0.5:
            # Map [0, 0.5] to [margin, 0.5]
            scaled = margin + (coordinate * (0.5 - margin) / 0.5)
        else:
            # Map [0.5, 1] to [0.5, 1-margin]
            scaled = 0.5 + ((coordinate - 0.5) * (0.5 - margin) / 0.5)
        
        return scaled
    
    def _validate_iris_coordinates(self, eye_x: float, eye_y: float) -> bool:
        """
        Validate iris coordinates before prediction.
        
        Args:
            eye_x: Iris X coordinate
            eye_y: Iris Y coordinate
            
        Returns:
            True if coordinates are valid, False otherwise
        """
        # Prevent Negative
        if eye_x < 0 or eye_y < 0:
            return False

        # Check for NaN or infinity
        if np.isnan(eye_x) or np.isnan(eye_y) or np.isinf(eye_x) or np.isinf(eye_y):
            return False
        
        # Assuming camera resolution up to 1920x1080
        if eye_x > 2000 or eye_y > 2000:
            return False
        
        # Check for zero or very small values (camera error)
        if eye_x < 1 or eye_y < 1:
            return False

        return True
    
    def convert_to_pixel_coordinates(self, screen_x_norm: float, screen_y_norm: float) -> Tuple[int, int]:
        """
        Convert normalized screen coordinates to pixel coordinates.
        
        Args:
            screen_x_norm: Normalized X coordinate (0-1)
            screen_y_norm: Normalized Y coordinate (0-1)
            
        Returns:
            (pixel_x, pixel_y) integer pixel coordinates
        """
        pixel_x = int(screen_x_norm * self.screen_width)
        pixel_y = int(screen_y_norm * self.screen_height)
        
        # Clamp to screen bounds
        pixel_x = max(0, min(pixel_x, self.screen_width - 1))
        pixel_y = max(0, min(pixel_y, self.screen_height - 1))
        
        return pixel_x, pixel_y
    
    def get_screen_coordinates(self, 
                              left_iris: Tuple[float, float],
                              right_iris: Tuple[float, float]) -> Dict[str, Any]:
        """
        Get screen coordinates with additional information.
        
        Args:
            left_iris: (x, y) coordinates of left iris
            right_iris: (x, y) coordinates of right iris
            
        Returns:
            Dictionary with screen coordinates and metadata
        """
        # Get normalized coordinates
        screen_x_norm, screen_y_norm = self.predict_screen_coordinates(
            left_iris, right_iris
        )
        
        # Convert to pixel coordinates
        pixel_x, pixel_y = self.convert_to_pixel_coordinates(screen_x_norm, screen_y_norm)
        
        return {
            'normalized': (screen_x_norm, screen_y_norm),
            'pixels': (pixel_x, pixel_y),
            'screen_resolution': (self.screen_width, self.screen_height),
            'prediction_count': self.prediction_count,
            'stats': self.stats.copy()
        }
    
    def reset_filters(self):
        """Reset all filters to initial state."""
        if self.filter_x:
            self.filter_x.reset()
        if self.filter_y:
            self.filter_y.reset()
        print("[Runtime] Filters reset")
    
    def update_screen_resolution(self, width: int, height: int):
        """Update screen resolution for pixel coordinate conversion."""
        self.screen_width = width
        self.screen_height = height
        print(f"[Runtime] Screen resolution updated to {width}x{height}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the runtime tracker."""
        return {
            'model_loaded': self.model_loaded,
            'tracking_active': self.tracking_active,
            'model_info': self.model_info,
            'prediction_count': self.prediction_count,
            'last_prediction': self.last_prediction,
            'stats': self.stats.copy(),
            'screen_resolution': (self.screen_width, self.screen_height),
            'safety_margin': self.safety_margin
        }


# ============================================================================
# INTEGRATION HELPER CLASS
# ============================================================================

class EyeTrackingRuntime:
    """
    High-level runtime interface for eye tracking.
    
    This class provides a clean API for integrating the runtime tracker
    with other components (camera, face detection, cursor control).
    """
    
    def __init__(self, 
                 model_path: str = "calibration_model.pkl",
                 camera_id: int = 0,
                 screen_width: int = None,
                 screen_height: int = None,
                 filter_type: str = "ema"):
        """
        Initialize eye tracking runtime.
        
        Args:
            model_path: Path to pre-trained regression model
            camera_id: Camera ID for iris detection
            screen_width: Screen width (autodetect if None)
            screen_height: Screen height (autodetect if None)
            filter_type: Filter type for smoothing
        """
        # Auto-detect screen resolution if not provided
        if screen_width is None or screen_height is None:
            screen_width, screen_height = self._get_screen_resolution()
        
        # Initialize runtime tracker
        self.tracker = RuntimeEyeTracker(
            model_path=model_path,
            filter_type=filter_type,
            screen_width=screen_width,
            screen_height=screen_height
        )
        
        # Initialize camera and face mesh detector (lazy loading)
        self.camera_id = camera_id
        self.camera = None
        self.face_mesh_detector = None
        
        print(f"[EyeTracking] Runtime initialized")
        print(f"[EyeTracking] Screen: {screen_width}x{screen_height}")
        print(f"[EyeTracking] Model: {model_path}")
    
    def _get_screen_resolution(self) -> Tuple[int, int]:
        """Get screen resolution."""
        try:
            # Try multiple methods to get screen resolution
            try:
                import tkinter as tk
                root = tk.Tk()
                width = root.winfo_screenwidth()
                height = root.winfo_screenheight()
                root.destroy()
                return width, height
            except:
                pass
            
            try:
                from screeninfo import get_monitors
                monitor = get_monitors()[0]
                return monitor.width, monitor.height
            except ImportError:
                print("[EyeTracking] Install screeninfo for better resolution detection: pip install screeninfo")
            
            # Default to common resolutions
            return 1920, 1080
            
        except Exception as e:
            print(f"[EyeTracking] Error detecting screen resolution: {e}")
            return 1920, 1080  # Fallback to FHD
    
    def start(self):
        """Start eye tracking."""
        # Initialize camera if not already
        if self.camera is None:
            try:
                from camera import IrisCamera
                self.camera = IrisCamera(self.camera_id)
                print(f"[EyeTracking] Camera {self.camera_id} initialized")
            except ImportError:
                print("[EyeTracking] Camera module not available")
                self.camera = None
        
        # Initialize face mesh detector if not already
        if self.face_mesh_detector is None:
            try:
                from face_utils import FaceMeshDetector
                self.face_mesh_detector = FaceMeshDetector()
                print("[EyeTracking] Face mesh detector initialized")
            except ImportError:
                print("[EyeTracking] Face utils module not available")
                self.face_mesh_detector = None
        
        # Start tracking
        self.tracker.start_tracking()
        print("[EyeTracking] Tracking started")
    
    def stop(self):
        """Stop eye tracking."""
        self.tracker.stop_tracking()
        
        # Release resources
        if self.camera:
            self.camera.stop_recording()
            self.camera = None
        
        if self.face_mesh_detector:
            self.face_mesh_detector.release()
            self.face_mesh_detector = None
        
        print("[EyeTracking] Tracking stopped")
    
    def get_gaze_position(self) -> Optional[Dict[str, Any]]:
        """
        Get current gaze position from camera.
        
        Returns:
            Dictionary with gaze information or None if failed
        """
        if not self.tracker.tracking_active:
            print("[EyeTracking] Warning: Tracking not active")
            return None
        
        if self.camera is None or self.face_mesh_detector is None:
            print("[EyeTracking] Error: Camera or detector not initialized")
            return None
        
        try:
            # Get frame from camera
            frame = self.camera.get_frame()
            if frame is None:
                return None
            
            # Get iris centers
            iris_centers = self.face_mesh_detector.get_iris_centers(frame)
            
            if not iris_centers.get('left') or not iris_centers.get('right'):
                return None
            
            # Get screen coordinates
            left_iris = iris_centers['left']
            right_iris = iris_centers['right']
            
            result = self.tracker.get_screen_coordinates(left_iris, right_iris)
            
            # Add iris information
            result['iris_left'] = left_iris
            result['iris_right'] = right_iris
            result['timestamp'] = time.time() if 'time' in globals() else 0
            
            return result
            
        except Exception as e:
            print(f"[EyeTracking] Error getting gaze position: {e}")
            return None
    
    def update(self):
        """
        Update tracking state.
        Call this periodically in your main loop.
        """
        # Currently just a placeholder for future state updates
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        status = self.tracker.get_status()
        status['camera_initialized'] = self.camera is not None
        status['detector_initialized'] = self.face_mesh_detector is not None
        return status
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit."""
        self.stop()


# ============================================================================
# SIMPLE TEST/DEMO
# ============================================================================

if __name__ == "__main__":
    import time
    
    print("Runtime Tracker Test")
    print("=" * 50)
    
    # Create a simple test without camera
    tracker = RuntimeEyeTracker(
        model_path="calibration_model.pkl",
        filter_type="ema",
        screen_width=1920,
        screen_height=1080
    )
    
    tracker.start_tracking()
    
    print("\nTesting with simulated iris coordinates...")
    print("(These would normally come from your camera)")
    
    # Simulate some test points
    test_points = [
        (320, 240),   # Top-left
        (960, 540),   # Center
        (1600, 800),  # Bottom-right
    ]
    
    for i, (eye_x, eye_y) in enumerate(test_points):
        # Simulate both eyes at same position (for testing)
        result = tracker.get_screen_coordinates(
            left_iris=(eye_x, eye_y),
            right_iris=(eye_x, eye_y)
        )
        
        print(f"\nTest {i + 1}:")
        print(f"  Iris: ({eye_x}, {eye_y})")
        print(f"  Normalized: ({result['normalized'][0]:.3f}, {result['normalized'][1]:.3f})")
        print(f"  Pixels: {result['pixels']}")
    
    print("\n" + "=" * 50)
    print("Test complete")
    print(f"Total predictions: {tracker.stats['total_predictions']}")
    print(f"Failed predictions: {tracker.stats['failed_predictions']}")
    
    tracker.stop_tracking()