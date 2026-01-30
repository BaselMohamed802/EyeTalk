"""
Filename: head_tracking.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 1/30/2026

Description:
    This file contains the HeadTracker class for 3D head pose estimation,
    orientation calculation, and screen mapping for gaze control.
"""

import numpy as np
import math
from collections import deque
from typing import Optional, Tuple, List, Dict

class HeadTracker:
    def __init__(self, filter_length: int = 8):
        """
        Initialize the HeadTracker with smoothing buffers.
        
        Args:
            filter_length: Number of frames to average for smoothing
        """
        self.filter_length = filter_length
        self.ray_origins = deque(maxlen=filter_length)
        self.ray_directions = deque(maxlen=filter_length)
        
        # Calibration offsets
        self.calibration_offset_yaw = 0
        self.calibration_offset_pitch = 0
        
        # Key facial landmark indices for head tracking
        self.LANDMARK_INDICES = {
            "left": 234,     # Left temple
            "right": 454,    # Right temple
            "top": 10,       # Forehead top
            "bottom": 152,   # Chin bottom
            "front": 1,      # Nose tip
            "nose_bridge": 168,  # Nose bridge (alternative forward reference)
        }
        
        # Face outline indices for visualization
        self.FACE_OUTLINE_INDICES = [
            10, 338, 297, 332, 284, 251, 389, 356,
            454, 323, 361, 288, 397, 365, 379, 378,
            400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21,
            54, 103, 67, 109
        ]
    
    def extract_key_points(self, landmarks: List[Tuple[int, int, int]], 
                           img_width: int, img_height: int) -> Dict[str, np.ndarray]:
        """
        Extract key 3D points from facial landmarks.
        
        Args:
            landmarks: List of (index, x, y) facial landmarks
            img_width: Image width
            img_height: Image height
            
        Returns:
            Dictionary of 3D points for key facial features
        """
        # Convert landmarks list to dictionary for easy access
        landmark_dict = {idx: (x, y) for idx, x, y in landmarks}
        
        key_points = {}
        for name, idx in self.LANDMARK_INDICES.items():
            if idx in landmark_dict:
                x, y = landmark_dict[idx]
                # Create 3D point (z is estimated as 0 initially)
                key_points[name] = np.array([x, y, 0])
        
        return key_points
    
    def calculate_head_orientation(self, key_points: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calculate head orientation axes from key facial points.
        
        Args:
            key_points: Dictionary of key facial 3D points
            
        Returns:
            Dictionary containing:
                - right_axis: Left-to-right axis
                - up_axis: Bottom-to-top axis  
                - forward_axis: Forward direction
                - center: Head center point
        """
        # Extract points
        left = key_points["left"]
        right = key_points["right"]
        top = key_points["top"]
        bottom = key_points["bottom"]
        front = key_points["front"]
        
        # Calculate axes
        right_axis = (right - left)
        if np.linalg.norm(right_axis) > 0:
            right_axis /= np.linalg.norm(right_axis)
        
        up_axis = (top - bottom)
        if np.linalg.norm(up_axis) > 0:
            up_axis /= np.linalg.norm(up_axis)
        
        # Forward axis is cross product of right and up axes
        forward_axis = np.cross(right_axis, up_axis)
        if np.linalg.norm(forward_axis) > 0:
            forward_axis /= np.linalg.norm(forward_axis)
        
        # Flip to point forward from face
        forward_axis = -forward_axis
        
        # Calculate head center
        center = (left + right + top + bottom + front) / 5
        
        return {
            'right_axis': right_axis,
            'up_axis': up_axis,
            'forward_axis': forward_axis,
            'center': center,
            'half_width': np.linalg.norm(right - left) / 2,
            'half_height': np.linalg.norm(top - bottom) / 2
        }
    
    def calculate_yaw_pitch(self, forward_axis: np.ndarray) -> Tuple[float, float]:
        """
        Calculate yaw (left-right) and pitch (up-down) angles from forward axis.
        
        Args:
            forward_axis: 3D forward direction vector
            
        Returns:
            Tuple of (yaw_deg, pitch_deg) in degrees
        """
        # Reference forward direction (looking straight at camera)
        reference_forward = np.array([0, 0, -1])
        
        # Calculate yaw (horizontal rotation)
        xz_proj = np.array([forward_axis[0], 0, forward_axis[2]])
        if np.linalg.norm(xz_proj) > 0:
            xz_proj /= np.linalg.norm(xz_proj)
            yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
            if forward_axis[0] < 0:
                yaw_rad = -yaw_rad
        else:
            yaw_rad = 0
        
        # Calculate pitch (vertical rotation)
        yz_proj = np.array([0, forward_axis[1], forward_axis[2]])
        if np.linalg.norm(yz_proj) > 0:
            yz_proj /= np.linalg.norm(yz_proj)
            pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
            if forward_axis[1] > 0:
                pitch_rad = -pitch_rad
        else:
            pitch_rad = 0
        
        # Convert to degrees
        yaw_deg = np.degrees(yaw_rad)
        pitch_deg = np.degrees(pitch_rad)
        
        return yaw_deg, pitch_deg
    
    def convert_angles_to_360(self, yaw_deg: float, pitch_deg: float) -> Tuple[float, float]:
        """
        Convert angles from -180/180 range to 0-360 range.
        
        Args:
            yaw_deg: Yaw angle in degrees (-180 to 180)
            pitch_deg: Pitch angle in degrees (-180 to 180)
            
        Returns:
            Tuple of (yaw_360, pitch_360) in 0-360 range
        """
        # Convert yaw: 90° = left, 180° = center, 270° = right
        if yaw_deg < 0:
            yaw_360 = abs(yaw_deg)  # 0-180 for left rotations
        else:
            yaw_360 = 360 - yaw_deg if yaw_deg < 180 else 180  # 180-360 for right rotations
        
        # Convert pitch: 90° = down, 180° = center, 270° = up
        if pitch_deg < 0:
            pitch_360 = 360 + pitch_deg  # 270-360 for down rotations
        else:
            pitch_360 = pitch_deg  # 0-180 for up rotations
        
        return yaw_360, pitch_360
    
    def map_to_screen(self, yaw_deg: float, pitch_deg: float, 
                      screen_width: int, screen_height: int,
                      yaw_range: float = 20, pitch_range: float = 10) -> Tuple[int, int]:
        """
        Map head orientation angles to screen coordinates.
        
        Args:
            yaw_deg: Yaw angle in degrees (0-360 format)
            pitch_deg: Pitch angle in degrees (0-360 format)
            screen_width: Monitor width in pixels
            screen_height: Monitor height in pixels
            yaw_range: Degrees of head rotation to cover full screen width
            pitch_range: Degrees of head rotation to cover full screen height
            
        Returns:
            Tuple of (screen_x, screen_y) pixel coordinates
        """
        # Apply calibration offsets
        yaw_calibrated = yaw_deg + self.calibration_offset_yaw
        pitch_calibrated = pitch_deg + self.calibration_offset_pitch
        
        # Map to screen coordinates
        # Yaw: (180 - yaw_range) to (180 + yaw_range) maps to 0 to screen_width
        screen_x = int(((yaw_calibrated - (180 - yaw_range)) / (2 * yaw_range)) * screen_width)
        
        # Pitch: (180 - pitch_range) to (180 + pitch_range) maps to screen_height to 0
        screen_y = int(((180 + pitch_range - pitch_calibrated) / (2 * pitch_range)) * screen_height)
        
        return screen_x, screen_y
    
    def clamp_to_screen(self, x: int, y: int, 
                        screen_width: int, screen_height: int,
                        margin: int = 10) -> Tuple[int, int]:
        """
        Clamp coordinates to stay within screen bounds.
        
        Args:
            x: X coordinate
            y: Y coordinate
            screen_width: Screen width
            screen_height: Screen height
            margin: Minimum margin from edges
            
        Returns:
            Clamped (x, y) coordinates
        """
        x = max(margin, min(x, screen_width - margin))
        y = max(margin, min(y, screen_height - margin))
        return x, y
    
    def update_smoothing_buffers(self, center: np.ndarray, forward_axis: np.ndarray):
        """
        Update smoothing buffers with current head position and orientation.
        
        Args:
            center: Head center point
            forward_axis: Forward direction vector
        """
        self.ray_origins.append(center)
        self.ray_directions.append(forward_axis)
    
    def get_smoothed_orientation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get smoothed head position and orientation from buffers.
        
        Returns:
            Tuple of (avg_origin, avg_direction)
        """
        if len(self.ray_origins) == 0:
            return np.array([0, 0, 0]), np.array([0, 0, -1])
        
        avg_origin = np.mean(self.ray_origins, axis=0)
        avg_direction = np.mean(self.ray_directions, axis=0)
        
        # Normalize direction
        norm = np.linalg.norm(avg_direction)
        if norm > 0:
            avg_direction /= norm
        
        return avg_origin, avg_direction
    
    def calibrate(self, current_yaw: float, current_pitch: float):
        """
        Calibrate the tracker by setting current position as center.
        
        Args:
            current_yaw: Current yaw angle
            current_pitch: Current pitch angle
        """
        self.calibration_offset_yaw = 180 - current_yaw
        self.calibration_offset_pitch = 180 - current_pitch
    
    def generate_head_cube(self, center: np.ndarray, right_axis: np.ndarray,
                          up_axis: np.ndarray, forward_axis: np.ndarray,
                          half_width: float, half_height: float,
                          depth: float = 80) -> List[np.ndarray]:
        """
        Generate 3D cube points representing head volume.
        
        Args:
            center: Head center point
            right_axis: Right direction axis
            up_axis: Up direction axis
            forward_axis: Forward direction axis
            half_width: Half width of head
            half_height: Half height of head
            depth: Depth of cube (forward-backward)
            
        Returns:
            List of 8 cube corner points in 3D
        """
        def corner(x_sign, y_sign, z_sign):
            return (center
                    + x_sign * half_width * right_axis
                    + y_sign * half_height * up_axis
                    + z_sign * depth * forward_axis)
        
        cube_corners = [
            corner(-1, 1, -1),   # top-left-front
            corner(1, 1, -1),    # top-right-front
            corner(1, -1, -1),   # bottom-right-front
            corner(-1, -1, -1),  # bottom-left-front
            corner(-1, 1, 1),    # top-left-back
            corner(1, 1, 1),     # top-right-back
            corner(1, -1, 1),    # bottom-right-back
            corner(-1, -1, 1)    # bottom-left-back
        ]
        
        return cube_corners
    
    @staticmethod
    def project_to_2d(point_3d: np.ndarray) -> Tuple[int, int]:
        """
        Project 3D point to 2D screen coordinates (ignoring Z).
        
        Args:
            point_3d: 3D point as numpy array
            
        Returns:
            Tuple of (x, y) pixel coordinates
        """
        return int(point_3d[0]), int(point_3d[1])