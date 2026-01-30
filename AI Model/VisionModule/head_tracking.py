"""
Filename: head_tracking.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: [Current Date]

Description:
    Simple head tracking calculations without complex type hints.
"""

import numpy as np
import math
from collections import deque

class HeadTracker:
    def __init__(self, filter_length=8):
        self.filter_length = filter_length
        self.ray_origins = deque(maxlen=filter_length)
        self.ray_directions = deque(maxlen=filter_length)
        self.calibration_offset_yaw = 0
        self.calibration_offset_pitch = 0
        
    def extract_key_points(self, landmarks, img_width, img_height):
        """Extract key points from landmarks."""
        # Convert landmarks to dictionary
        landmark_dict = {}
        for idx, x, y in landmarks:
            landmark_dict[idx] = (x, y)
        
        # Define which landmarks we need
        indices = {
            "left": 234,
            "right": 454, 
            "top": 10,
            "bottom": 152,
            "front": 1
        }
        
        key_points = {}
        for name, idx in indices.items():
            if idx in landmark_dict:
                x, y = landmark_dict[idx]
                # Create 3D point (z=0 for now)
                key_points[name] = np.array([float(x), float(y), 0.0])
        
        return key_points
    
    def calculate_head_orientation(self, key_points):
        """Calculate head orientation from key points."""
        # Get points
        left = np.array(key_points["left"], dtype=float)
        right = np.array(key_points["right"], dtype=float)
        top = np.array(key_points["top"], dtype=float)
        bottom = np.array(key_points["bottom"], dtype=float)
        front = np.array(key_points["front"], dtype=float)
        
        # Calculate width and height
        width = np.linalg.norm(right - left)
        height = np.linalg.norm(top - bottom)
        
        # Calculate axes (normalized)
        right_axis = (right - left) / width if width > 0 else np.array([1.0, 0.0, 0.0])
        up_axis = (top - bottom) / height if height > 0 else np.array([0.0, 1.0, 0.0])
        
        # Forward axis (cross product, normalized)
        forward_axis = np.cross(right_axis, up_axis)
        forward_norm = np.linalg.norm(forward_axis)
        if forward_norm > 0:
            forward_axis = forward_axis / forward_norm
        
        # Flip to point forward from face
        forward_axis = -forward_axis
        
        # Center of head
        center = (left + right + top + bottom + front) / 5.0
        
        return {
            'right_axis': right_axis,
            'up_axis': up_axis,
            'forward_axis': forward_axis,
            'center': center,
            'half_width': width / 2.0,
            'half_height': height / 2.0
        }
    
    def update_smoothing_buffers(self, center, forward_axis):
        """Update smoothing buffers."""
        self.ray_origins.append(center)
        self.ray_directions.append(forward_axis)
    
    def get_smoothed_orientation(self):
        """Get smoothed head orientation."""
        if len(self.ray_origins) == 0:
            return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])
        
        # Calculate averages
        avg_origin = np.mean(list(self.ray_origins), axis=0)
        avg_direction = np.mean(list(self.ray_directions), axis=0)
        
        # Normalize direction
        norm = np.linalg.norm(avg_direction)
        if norm > 0:
            avg_direction = avg_direction / norm
        
        return avg_origin, avg_direction
    
    def calculate_yaw_pitch(self, forward_axis):
        """Calculate yaw and pitch angles."""
        reference = np.array([0.0, 0.0, -1.0])
        
        # Yaw (left-right rotation)
        xz_proj = np.array([forward_axis[0], 0.0, forward_axis[2]])
        xz_norm = np.linalg.norm(xz_proj)
        if xz_norm > 0:
            xz_proj = xz_proj / xz_norm
            yaw_rad = math.acos(np.clip(np.dot(reference, xz_proj), -1.0, 1.0))
            if forward_axis[0] < 0:
                yaw_rad = -yaw_rad
        else:
            yaw_rad = 0
        
        # Pitch (up-down rotation)
        yz_proj = np.array([0.0, forward_axis[1], forward_axis[2]])
        yz_norm = np.linalg.norm(yz_proj)
        if yz_norm > 0:
            yz_proj = yz_proj / yz_norm
            pitch_rad = math.acos(np.clip(np.dot(reference, yz_proj), -1.0, 1.0))
            if forward_axis[1] > 0:
                pitch_rad = -pitch_rad
        else:
            pitch_rad = 0
        
        # Convert to degrees
        return math.degrees(yaw_rad), math.degrees(pitch_rad)
    
    def convert_angles_to_360(self, yaw_deg, pitch_deg):
        """Convert angles to 0-360 range."""
        # Yaw: 90° = left, 180° = center, 270° = right
        if yaw_deg < 0:
            yaw_360 = abs(yaw_deg)
        else:
            yaw_360 = 360 - yaw_deg if yaw_deg < 180 else 180
        
        # Pitch: 90° = down, 180° = center, 270° = up
        if pitch_deg < 0:
            pitch_360 = 360 + pitch_deg
        else:
            pitch_360 = pitch_deg
        
        return yaw_360, pitch_360
    
    def map_to_screen(self, yaw_deg, pitch_deg, screen_width, screen_height, 
                     yaw_range=20, pitch_range=10):
        """Map angles to screen coordinates."""
        # Apply calibration
        yaw_calibrated = yaw_deg + self.calibration_offset_yaw
        pitch_calibrated = pitch_deg + self.calibration_offset_pitch
        
        # Map to screen
        screen_x = int(((yaw_calibrated - (180 - yaw_range)) / (2 * yaw_range)) * screen_width)
        screen_y = int(((180 + pitch_range - pitch_calibrated) / (2 * pitch_range)) * screen_height)
        
        return screen_x, screen_y
    
    def clamp_to_screen(self, x, y, screen_width, screen_height, margin=10):
        """Clamp coordinates to screen."""
        x = max(margin, min(x, screen_width - margin))
        y = max(margin, min(y, screen_height - margin))
        return x, y
    
    def generate_head_cube(self, center, right_axis, up_axis, forward_axis, half_width, half_height):
        """Generate 3D cube points for visualization."""
        depth = 80.0
        
        # Helper function to calculate corners
        def corner(x_sign, y_sign, z_sign):
            return (center + 
                    x_sign * half_width * right_axis +
                    y_sign * half_height * up_axis +
                    z_sign * depth * forward_axis)
        
        # 8 corners of the cube
        return [
            corner(-1, 1, -1),   # top-left-front
            corner(1, 1, -1),    # top-right-front
            corner(1, -1, -1),   # bottom-right-front
            corner(-1, -1, -1),  # bottom-left-front
            corner(-1, 1, 1),    # top-left-back
            corner(1, 1, 1),     # top-right-back
            corner(1, -1, 1),    # bottom-right-back
            corner(-1, -1, 1)    # bottom-left-back
        ]
    
    def calibrate(self, yaw_deg, pitch_deg):
        """Calibrate to current head position."""
        self.calibration_offset_yaw = 180 - yaw_deg
        self.calibration_offset_pitch = 180 - pitch_deg
        print(f"Calibrated: yaw_offset={self.calibration_offset_yaw:.1f}, pitch_offset={self.calibration_offset_pitch:.1f}")