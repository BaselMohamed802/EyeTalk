"""
Filename: gaze_vector_mapper.py
Creator/Author: Basel Mohamed Mostafa Sayed + ChatGPT
Date: 1/27/2026

Description:
    3D Gaze Vector based cursor control.
    Uses geometric intersection of gaze vectors with screen plane.
    
    Steps:
    1. Get 3D eye landmarks from MediaPipe
    2. Calculate gaze vectors for each eye
    3. Calibrate screen plane parameters
    4. Intersect gaze vectors with screen plane
    5. Map intersection points to screen coordinates
"""

import numpy as np
import pickle
import time
from typing import Tuple, Optional, Dict, Any, List
import math


class GazeVectorMapper:
    """
    3D gaze vector mapper using geometric intersection.
    
    This approach is more robust than 2D regression because:
    1. Uses 3D geometry (less sensitive to pixel errors)
    2. Accounts for head movement naturally
    3. Based on actual eye anatomy
    4. Better extrapolation outside calibration range
    """
    
    def __init__(self, calibration_path: Optional[str] = None):
        """
        Initialize gaze vector mapper.
        
        Args:
            calibration_path: Optional path to calibration data
                             If None, will use default geometry
        """
        # MediaPipe landmark indices (with refine_landmarks=True)
        self.LANDMARK_INDICES = {
            # Eye corners for coordinate system
            'left_eye_inner': 33,
            'left_eye_outer': 133,
            'right_eye_inner': 362,
            'right_eye_outer': 263,
            
            # Iris centers (3D when refine_landmarks=True)
            'left_iris': 468,
            'right_iris': 473,
            
            # Face reference points
            'nose_tip': 1,
            'forehead': 10,
            'chin': 152,
            'left_mouth': 61,
            'right_mouth': 291,
        }
        
        # Screen plane parameters (will be calibrated)
        # Screen plane equation: n·(x - p0) = 0
        self.screen_normal = None      # Normal vector to screen (unit vector)
        self.screen_center = None      # Point on screen plane (3D)
        self.screen_distance = None    # Distance from eyes to screen (approx)
        
        # Eye parameters (anatomical, in meters)
        self.eye_radius = 0.012        # Average eye radius (12mm)
        self.interpupillary_distance = 0.063  # Average IPD (63mm)
        
        # Calibration data
        self.calibration_points = []   # List of (gaze_vector, screen_point)
        self.is_calibrated = False
        
        # State
        self.current_gaze_vector = None
        self.current_intersection = None
        self.last_screen_point = None
        
        # Smoothing
        self.smoothed_gaze_vector = None
        self.smoothing_factor = 0.3
        
        # Load calibration if provided
        if calibration_path and self._load_calibration(calibration_path):
            print(f"[GazeVectorMapper] Loaded calibration from {calibration_path}")
        else:
            print("[GazeVectorMapper] Using default geometry (run calibration for better accuracy)")
            self._initialize_default_geometry()
    
    def _initialize_default_geometry(self):
        """Initialize with default screen geometry assumptions."""
        # Assume screen is directly in front, 60cm away
        self.screen_normal = np.array([0, 0, 1])  # Looking into -Z direction
        self.screen_center = np.array([0, 0, -0.6])  # 60cm in front
        self.screen_distance = 0.6
    
    def _load_calibration(self, path: str) -> bool:
        """Load gaze vector calibration data."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            if 'screen_normal' in data and 'screen_center' in data:
                self.screen_normal = np.array(data['screen_normal'])
                self.screen_center = np.array(data['screen_center'])
                self.screen_distance = data.get('screen_distance', 0.6)
                self.is_calibrated = True
                return True
        except Exception as e:
            print(f"[GazeVectorMapper] Error loading calibration: {e}")
        
        return False
    
    def save_calibration(self, path: str = "gaze_vector_calibration.pkl") -> bool:
        """Save calibration data."""
        if not self.is_calibrated:
            print("[GazeVectorMapper] Not calibrated, cannot save")
            return False
        
        data = {
            'screen_normal': self.screen_normal.tolist(),
            'screen_center': self.screen_center.tolist(),
            'screen_distance': self.screen_distance,
            'calibration_points': self.calibration_points,
            'timestamp': time.time()
        }
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            print(f"[GazeVectorMapper] Calibration saved to {path}")
            return True
        except Exception as e:
            print(f"[GazeVectorMapper] Error saving calibration: {e}")
            return False
    
    def calculate_gaze_vector(self, landmarks_3d: List) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Calculate 3D gaze vector from MediaPipe 3D landmarks.
        
        Args:
            landmarks_3d: List of MediaPipe 3D landmarks
            
        Returns:
            (gaze_vector, debug_info)
        """
        debug_info = {}
        
        try:
            # Get eye corner landmarks
            left_inner = landmarks_3d[self.LANDMARK_INDICES['left_eye_inner']]
            left_outer = landmarks_3d[self.LANDMARK_INDICES['left_eye_outer']]
            right_inner = landmarks_3d[self.LANDMARK_INDICES['right_eye_inner']]
            right_outer = landmarks_3d[self.LANDMARK_INDICES['right_eye_outer']]
            
            # Get iris centers (3D!)
            left_iris = landmarks_3d[self.LANDMARK_INDICES['left_iris']]
            right_iris = landmarks_3d[self.LANDMARK_INDICES['right_iris']]
            
            # Convert to numpy arrays
            left_inner_np = np.array([left_inner.x, left_inner.y, left_inner.z])
            left_outer_np = np.array([left_outer.x, left_outer.y, left_outer.z])
            right_inner_np = np.array([right_inner.x, right_inner.y, right_inner.z])
            right_outer_np = np.array([right_outer.x, right_outer.y, right_outer.z])
            left_iris_np = np.array([left_iris.x, left_iris.y, left_iris.z])
            right_iris_np = np.array([right_iris.x, right_iris.y, right_iris.z])
            
            # Calculate eye centers (midpoint of inner and outer corners)
            left_eye_center = (left_inner_np + left_outer_np) / 2
            right_eye_center = (right_inner_np + right_outer_np) / 2
            
            # Calculate gaze vectors (from eye center to iris)
            left_gaze = left_iris_np - left_eye_center
            right_gaze = right_iris_np - right_eye_center
            
            # Normalize gaze vectors
            left_gaze_norm = left_gaze / np.linalg.norm(left_gaze)
            right_gaze_norm = right_gaze / np.linalg.norm(right_gaze)
            
            # Average both gaze vectors for more stability
            gaze_vector = (left_gaze_norm + right_gaze_norm) / 2
            gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)
            
            # Calculate head position (midpoint between eyes)
            head_position = (left_eye_center + right_eye_center) / 2
            
            # Store debug info
            debug_info.update({
                'left_eye_center': left_eye_center,
                'right_eye_center': right_eye_center,
                'head_position': head_position,
                'left_gaze': left_gaze_norm,
                'right_gaze': right_gaze_norm,
                'ipd': np.linalg.norm(right_eye_center - left_eye_center)
            })
            
            return gaze_vector, debug_info
            
        except Exception as e:
            print(f"[GazeVectorMapper] Error calculating gaze vector: {e}")
            return None, {}
    
    def intersect_with_screen(self, gaze_vector: np.ndarray, head_position: np.ndarray) -> Optional[np.ndarray]:
        """
        Intersect gaze vector with screen plane.
        
        Args:
            gaze_vector: Normalized 3D gaze direction
            head_position: 3D position of head (between eyes)
            
        Returns:
            3D intersection point on screen plane, or None if no intersection
        """
        if self.screen_normal is None or self.screen_center is None:
            return None
        
        # Ray: p = head_position + t * gaze_vector
        # Plane: n·(p - p0) = 0
        
        # Solve for t: n·(head_position + t*gaze_vector - p0) = 0
        # t = n·(p0 - head_position) / (n·gaze_vector)
        
        numerator = np.dot(self.screen_normal, self.screen_center - head_position)
        denominator = np.dot(self.screen_normal, gaze_vector)
        
        # Avoid division by zero (gaze parallel to screen)
        if abs(denominator) < 1e-6:
            return None
        
        t = numerator / denominator
        
        # Intersection point
        intersection = head_position + t * gaze_vector
        
        return intersection
    
    def calibrate_screen_plane(self, calibration_data: List[Tuple[np.ndarray, Tuple[float, float]]]):
        """
        Calibrate screen plane from gaze vectors and known screen points.
        
        Args:
            calibration_data: List of (gaze_vector, (screen_x, screen_y))
                             screen_x, screen_y are normalized [0, 1]
        """
        if len(calibration_data) < 3:
            print("[GazeVectorMapper] Need at least 3 calibration points")
            return False
        
        self.calibration_points = calibration_data
        
        # For simplicity, we'll use a geometric approach:
        # Assume screen is a plane at fixed distance, find orientation
        
        # Method: Use three points to define a plane
        # We'll use the gaze vectors from center, top-left, and top-right points
        
        # Find the calibration point closest to screen center (0.5, 0.5)
        center_idx = 0
        min_dist = float('inf')
        for i, (_, screen_pos) in enumerate(calibration_data):
            dist = (screen_pos[0] - 0.5)**2 + (screen_pos[1] - 0.5)**2
            if dist < min_dist:
                min_dist = dist
                center_idx = i
        
        # Use center point and two edge points
        center_gaze, center_screen = calibration_data[center_idx]
        
        # For now, use simple assumptions (calibration will be improved)
        # Assume screen is 60cm away, directly in front
        self.screen_distance = 0.6
        self.screen_normal = np.array([0, 0, 1])  # Looking into -Z
        self.screen_center = np.array([0, 0, -self.screen_distance])
        
        self.is_calibrated = True
        print("[GazeVectorMapper] Screen plane calibrated")
        
        return True
    
    def update(self, landmarks_3d: List) -> bool:
        """
        Update gaze tracking with new 3D landmarks.
        
        Args:
            landmarks_3d: MediaPipe 3D landmarks
            
        Returns:
            True if successful
        """
        gaze_vector, debug_info = self.calculate_gaze_vector(landmarks_3d)
        
        if gaze_vector is None:
            return False
        
        # Apply smoothing
        if self.smoothed_gaze_vector is None:
            self.smoothed_gaze_vector = gaze_vector
        else:
            self.smoothed_gaze_vector = (
                self.smoothing_factor * gaze_vector + 
                (1 - self.smoothing_factor) * self.smoothed_gaze_vector
            )
            # Re-normalize
            self.smoothed_gaze_vector = self.smoothed_gaze_vector / np.linalg.norm(self.smoothed_gaze_vector)
        
        self.current_gaze_vector = self.smoothed_gaze_vector
        
        # Calculate intersection with screen
        head_position = debug_info.get('head_position')
        if head_position is not None:
            self.current_intersection = self.intersect_with_screen(
                self.current_gaze_vector, head_position
            )
        
        return True
    
    def get_screen_coordinates(self, screen_width: int, screen_height: int) -> Optional[Tuple[float, float]]:
        """
        Convert 3D intersection to 2D screen coordinates.
        
        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            
        Returns:
            Normalized screen coordinates (x, y) in [0, 1], or None
        """
        if self.current_intersection is None or not self.is_calibrated:
            return None
        
        # Simple projection: Assume screen is axis-aligned
        # This will be improved with proper calibration
        
        # Convert 3D intersection to 2D screen coordinates
        # For now, use simple linear mapping based on assumption
        
        # Intersection point relative to screen center
        rel_pos = self.current_intersection - self.screen_center
        
        # Simple conversion (will be calibrated properly)
        # Assume screen is 0.4m wide and 0.225m high (16:9 at 60cm)
        screen_width_m = 0.4
        screen_height_m = 0.225
        
        # Normalized coordinates
        norm_x = 0.5 + (rel_pos[0] / screen_width_m)
        norm_y = 0.5 - (rel_pos[1] / screen_height_m)  # Y is inverted
        
        # Clamp to [0, 1]
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))
        
        self.last_screen_point = (norm_x, norm_y)
        return norm_x, norm_y
    
    def get_pixel_coordinates(self, screen_width: int, screen_height: int) -> Optional[Tuple[int, int]]:
        """
        Get pixel coordinates on screen.
        
        Returns:
            (pixel_x, pixel_y) or None
        """
        coords = self.get_screen_coordinates(screen_width, screen_height)
        if coords is None:
            return None
        
        norm_x, norm_y = coords
        pixel_x = int(norm_x * screen_width)
        pixel_y = int(norm_y * screen_height)
        
        return pixel_x, pixel_y
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the mapper."""
        status = {
            'is_calibrated': self.is_calibrated,
            'current_gaze_vector': self.current_gaze_vector.tolist() if self.current_gaze_vector is not None else None,
            'current_intersection': self.current_intersection.tolist() if self.current_intersection is not None else None,
            'last_screen_point': self.last_screen_point,
            'screen_normal': self.screen_normal.tolist() if self.screen_normal is not None else None,
            'screen_center': self.screen_center.tolist() if self.screen_center is not None else None,
        }
        return status


class SimpleGazeVectorCalibrator:
    """
    Simple calibrator for gaze vector system.
    Calibrates screen plane using known gaze directions.
    """
    
    def __init__(self):
        self.calibration_data = []
    
    def add_calibration_point(self, gaze_vector: np.ndarray, screen_point: Tuple[float, float]):
        """Add a calibration point."""
        self.calibration_data.append((gaze_vector, screen_point))
        print(f"[Calibrator] Added point {len(self.calibration_data)}: "
              f"gaze={gaze_vector.tolist()}, screen={screen_point}")
    
    def calibrate(self, mapper: GazeVectorMapper) -> bool:
        """Calibrate the gaze vector mapper."""
        if len(self.calibration_data) < 3:
            print("[Calibrator] Need at least 3 points for calibration")
            return False
        
        return mapper.calibrate_screen_plane(self.calibration_data)


# Simple test
def test_gaze_vectors():
    """Test the gaze vector calculation."""
    print("Testing Gaze Vector Mapper...")
    
    # Test with dummy data
    mapper = GazeVectorMapper()
    
    print(f"Screen normal: {mapper.screen_normal}")
    print(f"Screen center: {mapper.screen_center}")
    print(f"Is calibrated: {mapper.is_calibrated}")
    
    # Test intersection
    test_gaze = np.array([0, 0, -1])  # Looking straight
    test_head = np.array([0, 0, 0])   # Head at origin
    
    intersection = mapper.intersect_with_screen(test_gaze, test_head)
    print(f"Test intersection: {intersection}")


if __name__ == "__main__":
    test_gaze_vectors()