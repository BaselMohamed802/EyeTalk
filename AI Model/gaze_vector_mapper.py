"""
Filename: gaze_vector_mapper.py
Creator/Author: Basel Mohamed Mostafa Sayed + ChatGPT
Date: 1/27/2026
Updated: 1/28/2026

Description:
    3D Gaze Vector based cursor control.
    Uses geometric intersection of gaze vectors with screen plane.
    
    Key improvements:
    1. Uses proper calibration to estimate screen plane (not assumed metrics)
    2. Handles MediaPipe's relative coordinate system properly
    3. Estimates plane+basis from calibration points via SVD
    4. Uses both iris origin and direction for each eye
    
    Steps:
    1. Get 3D eye landmarks from MediaPipe
    2. Calculate gaze vectors with proper origins
    3. Calibrate screen plane and basis from known points
    4. Intersect gaze vectors with screen plane
    5. Map intersection points to screen coordinates using learned basis
"""

import numpy as np
import pickle
import time
from typing import Tuple, Optional, Dict, Any, List
import math
from dataclasses import dataclass


@dataclass
class GazeRay:
    """Represents a gaze ray with origin and direction."""
    origin: np.ndarray  # 3D point where ray starts (eye center)
    direction: np.ndarray  # 3D normalized gaze direction


@dataclass 
class CalibrationSample3D:
    """3D calibration sample with complete ray information."""
    ray: GazeRay  # Origin + direction
    screen_uv: Tuple[float, float]  # Known screen position (normalized 0-1)
    timestamp: float


class GazeVectorMapper:
    """
    3D gaze vector mapper using geometric intersection.
    
    This approach is more robust than 2D regression because:
    1. Uses 3D geometry (less sensitive to pixel errors)
    2. Accounts for head movement naturally
    3. Based on actual eye anatomy
    4. Better extrapolation outside calibration range
    
    Key improvement: Learns screen plane and basis from calibration,
    doesn't assume metric measurements.
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
            
            # Iris ring landmarks (4 per eye) - for better center estimation
            'left_iris_ring': [468, 469, 470, 471],
            'right_iris_ring': [473, 474, 475, 476],
            
            # Face reference points for coordinate frame
            'nose_tip': 1,
            'forehead': 10,
            'chin': 152,
            'left_ear': 234,  # Approximate
            'right_ear': 454,  # Approximate
        }
        
        # Screen plane parameters (will be calibrated)
        # Screen plane equation: n·(x - p0) = 0
        self.plane_normal = None      # Normal vector to screen (unit vector)
        self.plane_point = None       # Any point on the plane
        
        # Screen coordinate system on the plane
        self.screen_origin = None     # Point on plane corresponding to (0,0)
        self.screen_u_axis = None     # Direction for increasing x
        self.screen_v_axis = None     # Direction for increasing y
        self.screen_width = None      # Scale in u direction
        self.screen_height = None     # Scale in v direction
        
        # Calibration data
        self.calibration_samples: List[CalibrationSample3D] = []
        self.is_calibrated = False
        
        # State
        self.current_ray: Optional[GazeRay] = None
        self.current_intersection = None
        self.last_screen_uv = None
        
        # Smoothing
        self.smoothed_direction = None
        self.smoothing_alpha = 0.3
        
        # Eye anatomical estimates (relative to MediaPipe scale)
        self.eyeball_depth_factor = 0.3  # Eyeball center behind eyelid midpoint
        
        # Load calibration if provided
        if calibration_path and self._load_calibration(calibration_path):
            print(f"[GazeVectorMapper] Loaded calibration from {calibration_path}")
        else:
            print("[GazeVectorMapper] Using default geometry (run calibration for better accuracy)")
            self._initialize_default_geometry()
    
    def _initialize_default_geometry(self):
        """Initialize with default screen geometry assumptions."""
        # In MediaPipe coordinates, assume screen is in front
        # This will be replaced by proper calibration
        self.plane_normal = np.array([0, 0, -1])  # Plane facing camera
        self.plane_point = np.array([0, 0, -2])   # Some distance in front
        
        # Default screen coordinate system (aligned with plane)
        self.screen_origin = np.array([-1, 1, -2])  # Top-left
        self.screen_u_axis = np.array([1, 0, 0])    # Right
        self.screen_v_axis = np.array([0, -1, 0])   # Down
        self.screen_width = 2.0
        self.screen_height = 2.0
    
    def _load_calibration(self, path: str) -> bool:
        """Load gaze vector calibration data."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            if ('plane_normal' in data and 'plane_point' in data and 
                'screen_origin' in data and 'screen_u_axis' in data):
                self.plane_normal = np.array(data['plane_normal'])
                self.plane_point = np.array(data['plane_point'])
                self.screen_origin = np.array(data['screen_origin'])
                self.screen_u_axis = np.array(data['screen_u_axis'])
                self.screen_v_axis = np.array(data['screen_v_axis'])
                self.screen_width = data['screen_width']
                self.screen_height = data['screen_height']
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
            'plane_normal': self.plane_normal.tolist(),
            'plane_point': self.plane_point.tolist(),
            'screen_origin': self.screen_origin.tolist(),
            'screen_u_axis': self.screen_u_axis.tolist(),
            'screen_v_axis': self.screen_v_axis.tolist(),
            'screen_width': self.screen_width,
            'screen_height': self.screen_height,
            'calibration_samples': [
                (sample.ray.origin.tolist(), 
                 sample.ray.direction.tolist(), 
                 sample.screen_uv, 
                 sample.timestamp)
                for sample in self.calibration_samples
            ],
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
    
    def calculate_gaze_ray(self, landmarks_3d: List) -> Optional[GazeRay]:
        """
        Calculate 3D gaze ray from MediaPipe 3D landmarks.
        
        Improved: Uses iris ring for better center, estimates eyeball position.
        
        Args:
            landmarks_3d: List of MediaPipe 3D landmarks
            
        Returns:
            GazeRay object with origin and direction, or None if failed
        """
        try:
            # Get eye corner landmarks
            left_inner = landmarks_3d[self.LANDMARK_INDICES['left_eye_inner']]
            left_outer = landmarks_3d[self.LANDMARK_INDICES['left_eye_outer']]
            right_inner = landmarks_3d[self.LANDMARK_INDICES['right_eye_inner']]
            right_outer = landmarks_3d[self.LANDMARK_INDICES['right_eye_outer']]
            
            # Convert to numpy
            left_inner_np = np.array([left_inner.x, left_inner.y, left_inner.z])
            left_outer_np = np.array([left_outer.x, left_outer.y, left_outer.z])
            right_inner_np = np.array([right_inner.x, right_inner.y, right_inner.z])
            right_outer_np = np.array([right_outer.x, right_outer.y, right_outer.z])
            
            # Calculate eye centers (midpoint of inner and outer corners)
            left_eye_center = (left_inner_np + left_outer_np) / 2
            right_eye_center = (right_inner_np + right_outer_np) / 2
            
            # Estimate eyeball center (behind eyelid)
            # Use face forward direction (nose to forehead) as reference
            nose_tip = landmarks_3d[self.LANDMARK_INDICES['nose_tip']]
            forehead = landmarks_3d[self.LANDMARK_INDICES['forehead']]
            face_forward = np.array([forehead.x - nose_tip.x, 
                                     forehead.y - nose_tip.y, 
                                     forehead.z - nose_tip.z])
            face_forward = face_forward / np.linalg.norm(face_forward)
            
            # Eyeball center is behind eyelid along face forward direction
            left_eyeball_center = left_eye_center - self.eyeball_depth_factor * face_forward
            right_eyeball_center = right_eye_center - self.eyeball_depth_factor * face_forward
            
            # Calculate iris centers from iris ring landmarks
            left_iris_centers = []
            for idx in self.LANDMARK_INDICES['left_iris_ring']:
                lm = landmarks_3d[idx]
                left_iris_centers.append([lm.x, lm.y, lm.z])
            
            right_iris_centers = []
            for idx in self.LANDMARK_INDICES['right_iris_ring']:
                lm = landmarks_3d[idx]
                right_iris_centers.append([lm.x, lm.y, lm.z])
            
            left_iris_center = np.mean(left_iris_centers, axis=0)
            right_iris_center = np.mean(right_iris_centers, axis=0)
            
            # Calculate gaze directions (from eyeball center to iris center)
            left_gaze_dir = left_iris_center - left_eyeball_center
            right_gaze_dir = right_iris_center - right_eyeball_center
            
            # Normalize
            left_gaze_dir = left_gaze_dir / np.linalg.norm(left_gaze_dir)
            right_gaze_dir = right_gaze_dir / np.linalg.norm(right_gaze_dir)
            
            # Average for combined gaze
            combined_direction = (left_gaze_dir + right_gaze_dir) / 2
            combined_direction = combined_direction / np.linalg.norm(combined_direction)
            
            # Use midpoint between eyeballs as ray origin
            combined_origin = (left_eyeball_center + right_eyeball_center) / 2
            
            return GazeRay(origin=combined_origin, direction=combined_direction)
            
        except Exception as e:
            print(f"[GazeVectorMapper] Error calculating gaze ray: {e}")
            return None
    
    def intersect_ray_with_plane(self, ray: GazeRay) -> Optional[np.ndarray]:
        """
        Intersect gaze ray with screen plane.
        
        Args:
            ray: GazeRay with origin and direction
            
        Returns:
            3D intersection point on plane, or None if no intersection
        """
        if self.plane_normal is None or self.plane_point is None:
            return None
        
        # Ray: P = O + t * D
        # Plane: n·(P - P0) = 0
        
        # Solve for t: n·(O + tD - P0) = 0
        # t = n·(P0 - O) / (n·D)
        
        numerator = np.dot(self.plane_normal, self.plane_point - ray.origin)
        denominator = np.dot(self.plane_normal, ray.direction)
        
        # Avoid division by zero (ray parallel to plane)
        if abs(denominator) < 1e-10:
            return None
        
        t = numerator / denominator
        
        # Only accept intersections in front of the ray
        if t < 0:
            return None
        
        # Intersection point
        intersection = ray.origin + t * ray.direction
        
        return intersection
    
    def plane_coordinates_to_uv(self, point_on_plane: np.ndarray) -> Tuple[float, float]:
        """
        Convert a point on the plane to normalized screen coordinates.
        
        Args:
            point_on_plane: 3D point on the screen plane
            
        Returns:
            Normalized screen coordinates (u, v) in [0, 1]
        """
        if (self.screen_origin is None or self.screen_u_axis is None or 
            self.screen_v_axis is None):
            return 0.5, 0.5  # Center as fallback
        
        # Vector from screen origin to point
        v_to_point = point_on_plane - self.screen_origin
        
        # Project onto u and v axes
        u = np.dot(v_to_point, self.screen_u_axis) / self.screen_width
        v = np.dot(v_to_point, self.screen_v_axis) / self.screen_height
        
        # Clamp to [0, 1]
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))
        
        return u, v
    
    def calibrate_from_samples(self, samples: List[CalibrationSample3D]) -> bool:
        """
        Calibrate screen plane and coordinate system from samples.
        
        Uses iterative SVD-based plane fitting and corner-based basis estimation.
        
        Args:
            samples: List of calibration samples
            
        Returns:
            True if calibration successful
        """
        if len(samples) < 4:
            print(f"[GazeVectorMapper] Need at least 4 points, got {len(samples)}")
            return False
        
        self.calibration_samples = samples
        
        print(f"[GazeVectorMapper] Calibrating from {len(samples)} samples...")
        
        # Step 1: Find best-fit plane using iterative refinement
        if not self._estimate_plane_iterative(samples):
            print("[GazeVectorMapper] Plane estimation failed")
            return False
        
        # Step 2: Estimate screen coordinate system from corner points
        if not self._estimate_screen_coordinate_system(samples):
            print("[GazeVectorMapper] Screen coordinate system estimation failed")
            return False
        
        self.is_calibrated = True
        print("[GazeVectorMapper] Calibration successful!")
        self._print_calibration_summary()
        
        return True
    
    def _estimate_plane_iterative(self, samples: List[CalibrationSample3D], 
                                 max_iterations: int = 10, 
                                 tolerance: float = 1e-6) -> bool:
        """
        Iteratively estimate plane using SVD.
        
        Alternates between:
        1. Intersecting rays with current plane to get 3D points
        2. Fitting new plane to those points via SVD
        
        Args:
            samples: Calibration samples
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            True if successful
        """
        # Initial guess: plane facing camera, some distance away
        self.plane_normal = np.array([0, 0, -1])
        self.plane_point = np.array([0, 0, -2])
        
        for iteration in range(max_iterations):
            # Step 1: Intersect all rays with current plane
            points_3d = []
            valid_samples = []
            
            for sample in samples:
                intersection = self.intersect_ray_with_plane(sample.ray)
                if intersection is not None:
                    points_3d.append(intersection)
                    valid_samples.append(sample)
            
            if len(points_3d) < 3:
                print(f"[GazeVectorMapper] Only {len(points_3d)} valid intersections")
                return False
            
            points_array = np.array(points_3d)
            
            # Step 2: Fit new plane to these points using SVD
            # Center the points
            centroid = np.mean(points_array, axis=0)
            centered = points_array - centroid
            
            # SVD to find normal (smallest singular vector)
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            new_normal = Vt[-1]  # Last row of V^T is the smallest singular vector
            
            # Ensure normal points toward camera (negative z in MediaPipe coords)
            if new_normal[2] > 0:
                new_normal = -new_normal
            
            # Check for convergence
            if iteration > 0:
                angle_change = np.arccos(np.clip(
                    np.dot(self.plane_normal, new_normal), -1.0, 1.0
                ))
                if angle_change < tolerance:
                    print(f"[GazeVectorMapper] Plane converged after {iteration+1} iterations")
                    break
            
            # Update plane
            self.plane_normal = new_normal
            self.plane_point = centroid
            
            if iteration == max_iterations - 1:
                print(f"[GazeVectorMapper] Plane estimation reached max iterations")
        
        return True
    
    def _estimate_screen_coordinate_system(self, samples: List[CalibrationSample3D]) -> bool:
        """
        Estimate screen coordinate system from corner calibration points.
        
        Requires at least 4 samples including corners.
        
        Args:
            samples: Calibration samples
            
        Returns:
            True if successful
        """
        # Find corner samples
        corners = {
            'tl': None,  # top-left (0,0)
            'tr': None,  # top-right (1,0)
            'bl': None,  # bottom-left (0,1)
            'br': None   # bottom-right (1,1)
        }
        
        for sample in samples:
            u, v = sample.screen_uv
            
            # Check if this is a corner (within threshold)
            threshold = 0.2
            if u < threshold and v < threshold:
                corners['tl'] = sample
            elif u > 1 - threshold and v < threshold:
                corners['tr'] = sample
            elif u < threshold and v > 1 - threshold:
                corners['bl'] = sample
            elif u > 1 - threshold and v > 1 - threshold:
                corners['br'] = sample
        
        # We need at least 3 corners to define the coordinate system
        corner_count = sum(1 for c in corners.values() if c is not None)
        if corner_count < 3:
            print(f"[GazeVectorMapper] Need at least 3 corners, found {corner_count}")
            
            # Fallback: use extreme points
            return self._estimate_coords_from_extremes(samples)
        
        # Intersect corner rays with plane to get 3D corner points
        corner_points = {}
        for key, sample in corners.items():
            if sample is not None:
                intersection = self.intersect_ray_with_plane(sample.ray)
                if intersection is not None:
                    corner_points[key] = intersection
        
        if len(corner_points) < 3:
            print(f"[GazeVectorMapper] Could only intersect {len(corner_points)} corners")
            return self._estimate_coords_from_extremes(samples)
        
        # Define coordinate system
        # Use top-left as origin if available
        if 'tl' in corner_points:
            self.screen_origin = corner_points['tl']
        else:
            # Use available corner as origin
            first_key = next(iter(corner_points.keys()))
            self.screen_origin = corner_points[first_key]
        
        # Determine u-axis (right) and v-axis (down)
        if 'tr' in corner_points and 'tl' in corner_points:
            self.screen_u_axis = corner_points['tr'] - corner_points['tl']
            self.screen_width = np.linalg.norm(self.screen_u_axis)
            self.screen_u_axis = self.screen_u_axis / self.screen_width
        elif 'br' in corner_points and 'bl' in corner_points:
            # Use bottom edge
            self.screen_u_axis = corner_points['br'] - corner_points['bl']
            self.screen_width = np.linalg.norm(self.screen_u_axis)
            self.screen_u_axis = self.screen_u_axis / self.screen_width
        
        if 'bl' in corner_points and 'tl' in corner_points:
            self.screen_v_axis = corner_points['bl'] - corner_points['tl']
            self.screen_height = np.linalg.norm(self.screen_v_axis)
            self.screen_v_axis = self.screen_v_axis / self.screen_height
        elif 'br' in corner_points and 'tr' in corner_points:
            # Use right edge
            self.screen_v_axis = corner_points['br'] - corner_points['tr']
            self.screen_height = np.linalg.norm(self.screen_v_axis)
            self.screen_v_axis = self.screen_v_axis / self.screen_height
        
        # Ensure orthogonality (u and v may not be perfectly perpendicular)
        # Make v orthogonal to u
        if self.screen_u_axis is not None and self.screen_v_axis is not None:
            v_parallel = np.dot(self.screen_v_axis, self.screen_u_axis) * self.screen_u_axis
            self.screen_v_axis = self.screen_v_axis - v_parallel
            self.screen_v_axis = self.screen_v_axis / np.linalg.norm(self.screen_v_axis)
            
            # Ensure right-handed coordinate system
            cross = np.cross(self.screen_u_axis, self.screen_v_axis)
            if np.dot(cross, self.plane_normal) < 0:
                self.screen_v_axis = -self.screen_v_axis
        
        return True
    
    def _estimate_coords_from_extremes(self, samples: List[CalibrationSample3D]) -> bool:
        """
        Fallback: estimate coordinate system from extreme points.
        
        Args:
            samples: Calibration samples
            
        Returns:
            True if successful
        """
        print("[GazeVectorMapper] Using fallback coordinate estimation from extremes")
        
        # Intersect all rays with plane
        intersections = []
        valid_samples = []
        
        for sample in samples:
            intersection = self.intersect_ray_with_plane(sample.ray)
            if intersection is not None:
                intersections.append(intersection)
                valid_samples.append(sample)
        
        if len(intersections) < 3:
            return False
        
        # Find bounding box of intersection points
        points_array = np.array(intersections)
        min_coords = np.min(points_array, axis=0)
        max_coords = np.max(points_array, axis=0)
        
        # Use center of bounding box as origin
        self.screen_origin = (min_coords + max_coords) / 2
        
        # Use principal components as axes
        centered = points_array - self.screen_origin
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        # First two principal components as axes
        self.screen_u_axis = Vt[0]
        self.screen_v_axis = Vt[1]
        
        # Ensure v-axis is orthogonal to u-axis
        v_parallel = np.dot(self.screen_v_axis, self.screen_u_axis) * self.screen_u_axis
        self.screen_v_axis = self.screen_v_axis - v_parallel
        self.screen_v_axis = self.screen_v_axis / np.linalg.norm(self.screen_v_axis)
        
        # Determine scale from bounding box
        # Project all points onto axes to find extents
        u_coords = np.dot(points_array - self.screen_origin, self.screen_u_axis)
        v_coords = np.dot(points_array - self.screen_origin, self.screen_v_axis)
        
        self.screen_width = np.max(u_coords) - np.min(u_coords)
        self.screen_height = np.max(v_coords) - np.min(v_coords)
        
        # Adjust origin to be at min u, min v
        self.screen_origin = self.screen_origin + \
                           np.min(u_coords) * self.screen_u_axis + \
                           np.min(v_coords) * self.screen_v_axis
        
        return True
    
    def update(self, landmarks_3d: List) -> bool:
        """
        Update gaze tracking with new 3D landmarks.
        
        Args:
            landmarks_3d: MediaPipe 3D landmarks
            
        Returns:
            True if successful
        """
        ray = self.calculate_gaze_ray(landmarks_3d)
        
        if ray is None:
            return False
        
        # Apply smoothing to direction
        if self.smoothed_direction is None:
            self.smoothed_direction = ray.direction
        else:
            self.smoothed_direction = (
                self.smoothing_alpha * ray.direction + 
                (1 - self.smoothing_alpha) * self.smoothed_direction
            )
            self.smoothed_direction = self.smoothed_direction / np.linalg.norm(self.smoothed_direction)
        
        # Create smoothed ray
        smoothed_ray = GazeRay(origin=ray.origin, direction=self.smoothed_direction)
        self.current_ray = smoothed_ray
        
        # Calculate intersection with screen plane
        if self.is_calibrated:
            self.current_intersection = self.intersect_ray_with_plane(smoothed_ray)
        
        return True
    
    def get_screen_coordinates(self) -> Optional[Tuple[float, float]]:
        """
        Get normalized screen coordinates from current gaze.
        
        Returns:
            Normalized screen coordinates (u, v) in [0, 1], or None
        """
        if (not self.is_calibrated or self.current_intersection is None or
            self.screen_origin is None):
            return None
        
        u, v = self.plane_coordinates_to_uv(self.current_intersection)
        self.last_screen_uv = (u, v)
        
        return u, v
    
    def get_pixel_coordinates(self, screen_width: int, screen_height: int) -> Optional[Tuple[int, int]]:
        """
        Get pixel coordinates on screen.
        
        Returns:
            (pixel_x, pixel_y) or None
        """
        coords = self.get_screen_coordinates()
        if coords is None:
            return None
        
        u, v = coords
        pixel_x = int(u * screen_width)
        pixel_y = int(v * screen_height)
        
        return pixel_x, pixel_y
    
    def _print_calibration_summary(self):
        """Print calibration summary."""
        print("\n" + "="*50)
        print("CALIBRATION SUMMARY")
        print("="*50)
        print(f"Plane normal: {self.plane_normal}")
        print(f"Plane point: {self.plane_point}")
        print(f"Screen origin: {self.screen_origin}")
        print(f"U axis: {self.screen_u_axis}")
        print(f"V axis: {self.screen_v_axis}")
        print(f"Screen width (scale): {self.screen_width:.3f}")
        print(f"Screen height (scale): {self.screen_height:.3f}")
        print(f"Number of samples: {len(self.calibration_samples)}")
        print("="*50)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the mapper."""
        status = {
            'is_calibrated': self.is_calibrated,
            'current_ray': {
                'origin': self.current_ray.origin.tolist() if self.current_ray else None,
                'direction': self.current_ray.direction.tolist() if self.current_ray else None,
            },
            'current_intersection': self.current_intersection.tolist() if self.current_intersection is not None else None,
            'last_screen_uv': self.last_screen_uv,
            'plane_normal': self.plane_normal.tolist() if self.plane_normal is not None else None,
            'plane_point': self.plane_point.tolist() if self.plane_point is not None else None,
        }
        return status


class GazeVectorCalibrator:
    """
    Calibrator for 3D gaze vector system.
    
    Collects calibration samples and calibrates the mapper.
    """
    
    def __init__(self):
        self.samples: List[CalibrationSample3D] = []
        self.current_screen_uv = None
    
    def start_point(self, screen_uv: Tuple[float, float]):
        """
        Start calibration at a specific screen position.
        
        Args:
            screen_uv: Normalized screen coordinates (u, v) for this point
        """
        self.current_screen_uv = screen_uv
        print(f"[GazeVectorCalibrator] Starting calibration at screen position {screen_uv}")
    
    def add_sample(self, landmarks_3d: List) -> bool:
        """
        Add a calibration sample at current screen position.
        
        Args:
            landmarks_3d: MediaPipe 3D landmarks
            
        Returns:
            True if sample added successfully
        """
        if self.current_screen_uv is None:
            print("[GazeVectorCalibrator] No current screen position set")
            return False
        
        # Create temporary mapper to calculate ray
        temp_mapper = GazeVectorMapper()
        ray = temp_mapper.calculate_gaze_ray(landmarks_3d)
        
        if ray is None:
            return False
        
        # Create and store sample
        sample = CalibrationSample3D(
            ray=ray,
            screen_uv=self.current_screen_uv,
            timestamp=time.time()
        )
        
        self.samples.append(sample)
        print(f"[GazeVectorCalibrator] Added sample {len(self.samples)}")
        
        return True
    
    def calibrate(self, mapper: GazeVectorMapper) -> bool:
        """
        Calibrate the mapper with collected samples.
        
        Args:
            mapper: GazeVectorMapper to calibrate
            
        Returns:
            True if calibration successful
        """
        if len(self.samples) < 4:
            print(f"[GazeVectorCalibrator] Need at least 4 samples, have {len(self.samples)}")
            return False
        
        return mapper.calibrate_from_samples(self.samples)
    
    def reset(self):
        """Reset calibrator for new session."""
        self.samples = []
        self.current_screen_uv = None
        print("[GazeVectorCalibrator] Reset")


# Test function
def test_gaze_vectors():
    """Test the gaze vector calculation."""
    print("Testing Gaze Vector Mapper...")
    
    # Test with default initialization
    mapper = GazeVectorMapper()
    
    print(f"Plane normal: {mapper.plane_normal}")
    print(f"Plane point: {mapper.plane_point}")
    print(f"Is calibrated: {mapper.is_calibrated}")
    
    # Test with dummy ray
    test_ray = GazeRay(
        origin=np.array([0, 0, 0]),
        direction=np.array([0, 0, -1])
    )
    
    intersection = mapper.intersect_ray_with_plane(test_ray)
    print(f"Test intersection: {intersection}")
    
    if intersection is not None:
        uv = mapper.plane_coordinates_to_uv(intersection)
        print(f"Test screen coordinates: {uv}")


if __name__ == "__main__":
    test_gaze_vectors()