"""
Filename: relative_gaze_mapper.py
Creator/Author: Basel Mohamed Mostafa Sayed + ChatGPT
Date: 1/27/2026

Description:
    Relative gaze mapper for intuitive cursor control.
    Uses enhanced calibration data to map relative eye movement to cursor movement.
    
    Key features:
    1. Relative movement mapping (eye movement delta → cursor movement)
    2. Head stabilization using face center
    3. Dynamic sensitivity adjustment
    4. Smooth acceleration curves
"""

import numpy as np
import pickle
import time
from typing import Tuple, Optional, Dict, Any
import math


class RelativeGazeMapper:
    """
    Maps relative eye movement to cursor movement for intuitive control.
    
    Instead of absolute regression mapping, this uses:
    1. Neutral position as reference point
    2. Eye movement delta from neutral
    3. Head position stabilization
    4. Non-linear sensitivity curves
    """
    
    def __init__(self, calibration_path: str = "enhanced_calibration.pkl"):
        """
        Initialize relative gaze mapper with enhanced calibration data.
        
        Args:
            calibration_path: Path to enhanced calibration data
        """
        # Load calibration data
        self.calib_data = self._load_calibration(calibration_path)
        if not self.calib_data:
            raise ValueError(f"Failed to load calibration from {calibration_path}")
        
        # State variables
        self.neutral_position = None
        self.eye_x_range = None
        self.eye_y_range = None
        self.eye_x_center = None
        self.eye_y_center = None
        
        # Current tracking state
        self.current_eye_pos = None
        self.current_face_center = None
        self.last_cursor_pos = None
        
        # Smoothing and filtering
        self.smoothed_eye_x = None
        self.smoothed_eye_y = None
        self.smoothing_factor = 0.3
        
        # Sensitivity parameters (tunable)
        self.sensitivity_x = 1.0
        self.sensitivity_y = 1.0
        self.max_speed = 1000.0  # pixels per second
        self.dead_zone = 0.1  # Percentage of range where no movement occurs
        
        # Statistics
        self.total_updates = 0
        self.start_time = time.time()
        
        # Initialize from calibration data
        self._initialize_from_calibration()
        
        print(f"[RelativeMapper] Initialized with {calibration_path}")
        print(f"[RelativeMapper] Eye range: X={self.eye_x_range[1]-self.eye_x_range[0]:.1f}px, "
              f"Y={self.eye_y_range[1]-self.eye_y_range[0]:.1f}px")
        print(f"[RelativeMapper] Neutral position: {self.neutral_position}")
    
    def _load_calibration(self, path: str) -> Optional[Dict[str, Any]]:
        """Load enhanced calibration data."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            print(f"[RelativeMapper] Loaded calibration data: {len(data.get('avg_eye_x', []))} samples")
            return data
        except Exception as e:
            print(f"[RelativeMapper] Error loading calibration: {e}")
            return None
    
    def _initialize_from_calibration(self):
        """Initialize mapper parameters from calibration data."""
        # Get eye movement range
        if self.calib_data.get('eye_x_range') and self.calib_data.get('eye_y_range'):
            self.eye_x_range = self.calib_data['eye_x_range']
            self.eye_y_range = self.calib_data['eye_y_range']
            
            # Calculate center of range
            self.eye_x_center = (self.eye_x_range[0] + self.eye_x_range[1]) / 2.0
            self.eye_y_center = (self.eye_y_range[0] + self.eye_y_range[1]) / 2.0
            
            # Use calibration's neutral position if available
            if self.calib_data.get('neutral_position'):
                self.neutral_position = self.calib_data['neutral_position']
            else:
                # Fallback to center of range
                self.neutral_position = (self.eye_x_center, self.eye_y_center)
            
            print(f"[RelativeMapper] Eye center: ({self.eye_x_center:.1f}, {self.eye_y_center:.1f})")
        
        # Calculate dynamic sensitivity based on eye movement range
        self._calculate_dynamic_sensitivity()
    
    def _calculate_dynamic_sensitivity(self):
        """Calculate sensitivity based on eye movement range."""
        if not self.eye_x_range or not self.eye_y_range:
            return
        
        x_range = self.eye_x_range[1] - self.eye_x_range[0]
        y_range = self.eye_y_range[1] - self.eye_y_range[0]
        
        # Base sensitivity: smaller range = higher sensitivity
        # This makes the system work for different people
        base_sensitivity = 2.0  # Adjust this to change overall sensitivity
        
        if x_range > 0:
            self.sensitivity_x = base_sensitivity * (50.0 / x_range)  # Normalize to ~50px range
        if y_range > 0:
            self.sensitivity_y = base_sensitivity * (50.0 / y_range)
        
        # Clamp sensitivities
        self.sensitivity_x = max(0.5, min(5.0, self.sensitivity_x))
        self.sensitivity_y = max(0.5, min(5.0, self.sensitivity_y))
        
        print(f"[RelativeMapper] Dynamic sensitivity: X={self.sensitivity_x:.2f}, Y={self.sensitivity_y:.2f}")
    
    def update_eye_position(self, 
                           left_eye: Tuple[float, float],
                           right_eye: Tuple[float, float],
                           face_center: Optional[Tuple[float, float]] = None) -> bool:
        """
        Update current eye position with optional head stabilization.
        
        Args:
            left_eye: (x, y) coordinates of left iris
            right_eye: (x, y) coordinates of right iris
            face_center: (x, y) coordinates of face center for stabilization
            
        Returns:
            True if update successful
        """
        # Calculate average eye position
        avg_x = (left_eye[0] + right_eye[0]) / 2.0
        avg_y = (left_eye[1] + right_eye[1]) / 2.0
        
        # Apply head stabilization if face center available
        if face_center and self.current_face_center:
            # Adjust eye position based on head movement
            head_dx = face_center[0] - self.current_face_center[0]
            head_dy = face_center[1] - self.current_face_center[1]
            
            # Compensate for head movement (reduces effect of small head motions)
            compensation_factor = 0.3  # How much to compensate (0=none, 1=full)
            avg_x -= head_dx * compensation_factor
            avg_y -= head_dy * compensation_factor
        
        # Apply smoothing
        if self.smoothed_eye_x is None or self.smoothed_eye_y is None:
            self.smoothed_eye_x = avg_x
            self.smoothed_eye_y = avg_y
        else:
            self.smoothed_eye_x = (self.smoothing_factor * avg_x + 
                                  (1 - self.smoothing_factor) * self.smoothed_eye_x)
            self.smoothed_eye_y = (self.smoothing_factor * avg_y + 
                                  (1 - self.smoothing_factor) * self.smoothed_eye_y)
        
        # Store current positions
        self.current_eye_pos = (self.smoothed_eye_x, self.smoothed_eye_y)
        self.current_face_center = face_center
        
        self.total_updates += 1
        return True
    
    def calculate_cursor_movement(self, 
                                 screen_width: int,
                                 screen_height: int) -> Tuple[int, int]:
        """
        Calculate cursor movement from current eye position.
        
        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            
        Returns:
            (dx, dy) cursor movement in pixels
        """
        if not self.current_eye_pos or not self.neutral_position:
            return 0, 0
        
        current_x, current_y = self.current_eye_pos
        neutral_x, neutral_y = self.neutral_position
        
        # Calculate normalized deviation from neutral
        if self.eye_x_range and self.eye_y_range:
            x_range = self.eye_x_range[1] - self.eye_x_range[0]
            y_range = self.eye_y_range[1] - self.eye_y_range[0]
            
            if x_range > 0 and y_range > 0:
                # Normalize deviation to [-1, 1] range
                norm_dx = (current_x - neutral_x) / (x_range / 2.0)
                norm_dy = (current_y - neutral_y) / (y_range / 2.0)
                
                # Apply dead zone (no movement near center)
                if abs(norm_dx) < self.dead_zone:
                    norm_dx = 0
                if abs(norm_dy) < self.dead_zone:
                    norm_dy = 0
                
                # Apply non-linear response curve (cubic for finer control)
                # This gives more precision near center, faster movement at edges
                norm_dx = self._apply_response_curve(norm_dx)
                norm_dy = self._apply_response_curve(norm_dy)
                
                # Apply sensitivity
                norm_dx *= self.sensitivity_x
                norm_dy *= self.sensitivity_y
                
                # Convert to pixel movement
                dx = int(norm_dx * screen_width * 0.5)  # Max movement is half screen
                dy = int(norm_dy * screen_height * 0.5)
                
                # Limit maximum speed
                max_move = self.max_speed / 60.0  # Assuming 60 FPS
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance > max_move:
                    scale = max_move / distance
                    dx = int(dx * scale)
                    dy = int(dy * scale)
                
                return dx, dy
        
        # Fallback: simple relative movement
        dx = int((current_x - neutral_x) * self.sensitivity_x * 0.5)
        dy = int((current_y - neutral_y) * self.sensitivity_y * 0.5)
        
        return dx, dy
    
    def _apply_response_curve(self, value: float) -> float:
        """
        Apply non-linear response curve for better control.
        
        Args:
            value: Input value in [-1, 1]
            
        Returns:
            Curved output value
        """
        # Cubic curve for more precision near center
        return value ** 3
    
    def calculate_absolute_cursor_position(self,
                                          screen_width: int,
                                          screen_height: int,
                                          current_cursor_pos: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate absolute cursor position from eye movement.
        
        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            current_cursor_pos: Current cursor position (x, y)
            
        Returns:
            New cursor position (x, y)
        """
        # Calculate relative movement
        dx, dy = self.calculate_cursor_movement(screen_width, screen_height)
        
        # Apply to current position
        new_x = current_cursor_pos[0] + dx
        new_y = current_cursor_pos[1] + dy
        
        # Clamp to screen bounds
        new_x = max(0, min(new_x, screen_width - 1))
        new_y = max(0, min(new_y, screen_height - 1))
        
        self.last_cursor_pos = (new_x, new_y)
        return new_x, new_y
    
    def reset_neutral_position(self):
        """Reset neutral position to current eye position."""
        if self.current_eye_pos:
            self.neutral_position = self.current_eye_pos
            print(f"[RelativeMapper] Neutral position reset to: {self.neutral_position}")
    
    def adjust_sensitivity(self, x_factor: float, y_factor: float):
        """
        Adjust sensitivity factors.
        
        Args:
            x_factor: Multiply X sensitivity by this factor
            y_factor: Multiply Y sensitivity by this factor
        """
        self.sensitivity_x *= x_factor
        self.sensitivity_y *= y_factor
        
        # Clamp to reasonable values
        self.sensitivity_x = max(0.1, min(10.0, self.sensitivity_x))
        self.sensitivity_y = max(0.1, min(10.0, self.sensitivity_y))
        
        print(f"[RelativeMapper] Sensitivity adjusted: X={self.sensitivity_x:.2f}, Y={self.sensitivity_y:.2f}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the mapper."""
        elapsed = time.time() - self.start_time
        fps = self.total_updates / elapsed if elapsed > 0 else 0
        
        return {
            'neutral_position': self.neutral_position,
            'current_eye_position': self.current_eye_pos,
            'sensitivity': (self.sensitivity_x, self.sensitivity_y),
            'total_updates': self.total_updates,
            'fps': fps,
            'eye_x_range': self.eye_x_range,
            'eye_y_range': self.eye_y_range,
            'last_cursor_position': self.last_cursor_pos
        }


class HybridGazeMapper:
    """
    Hybrid gaze mapper that combines relative movement with absolute mapping.
    
    Uses the best of both approaches:
    1. Relative movement for fine control
    2. Absolute mapping for large gaze shifts
    3. Automatic switching based on gaze speed
    """
    
    def __init__(self, calibration_path: str = "enhanced_calibration.pkl"):
        """
        Initialize hybrid gaze mapper.
        
        Args:
            calibration_path: Path to enhanced calibration data
        """
        # Initialize both mappers
        self.relative_mapper = RelativeGazeMapper(calibration_path)
        
        # Load linear mapping from calibration for absolute positioning
        self.calib_data = self.relative_mapper.calib_data
        self.absolute_mapping = self.calib_data.get('screen_mapping')
        
        # State
        self.mode = "relative"  # "relative" or "absolute"
        self.last_gaze_speed = 0
        self.gaze_speed_history = []
        
        # Mode switching parameters
        self.speed_threshold = 50.0  # pixels/second to switch to absolute
        self.relative_priority = 0.7  # Weight for relative mode (0-1)
        
        print(f"[HybridMapper] Initialized with {calibration_path}")
        print(f"[HybridMapper] Starting in {self.mode} mode")
    
    def update_eye_position(self,
                           left_eye: Tuple[float, float],
                           right_eye: Tuple[float, float],
                           face_center: Optional[Tuple[float, float]] = None) -> bool:
        """
        Update eye position in both mappers.
        
        Returns:
            True if successful
        """
        return self.relative_mapper.update_eye_position(left_eye, right_eye, face_center)
    
    def calculate_cursor_position(self,
                                 screen_width: int,
                                 screen_height: int,
                                 current_cursor_pos: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate cursor position using hybrid approach.
        
        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            current_cursor_pos: Current cursor position
            
        Returns:
            New cursor position
        """
        # Calculate positions using both methods
        if self.mode == "relative":
            # Use relative mapper
            new_x, new_y = self.relative_mapper.calculate_absolute_cursor_position(
                screen_width, screen_height, current_cursor_pos
            )
        else:
            # Use absolute mapping from calibration
            if (self.absolute_mapping and 
                self.relative_mapper.current_eye_pos):
                
                eye_x, eye_y = self.relative_mapper.current_eye_pos
                
                # Apply linear mapping from calibration
                screen_x = (self.absolute_mapping['x_slope'] * eye_x + 
                           self.absolute_mapping['x_intercept'])
                screen_y = (self.absolute_mapping['y_slope'] * eye_y + 
                           self.absolute_mapping['y_intercept'])
                
                # Clamp to [0, 1]
                screen_x = max(0.0, min(1.0, screen_x))
                screen_y = max(0.0, min(1.0, screen_y))
                
                # Convert to pixels
                new_x = int(screen_x * screen_width)
                new_y = int(screen_y * screen_height)
            else:
                # Fallback to relative
                new_x, new_y = current_cursor_pos
        
        # Update mode based on gaze speed
        self._update_mode()
        
        return new_x, new_y
    
    def _update_mode(self):
        """Update mapping mode based on gaze speed."""
        if not self.relative_mapper.current_eye_pos:
            return
        
        # Calculate gaze speed (simplified)
        # In a real implementation, you'd track position over time
        
        # For now, use a simple heuristic based on distance from neutral
        if self.relative_mapper.neutral_position:
            current_x, current_y = self.relative_mapper.current_eye_pos
            neutral_x, neutral_y = self.relative_mapper.neutral_position
            
            distance = math.sqrt((current_x - neutral_x)**2 + 
                                (current_y - neutral_y)**2)
            
            # If far from neutral, switch to absolute mode for large movements
            if distance > 20:  # pixels threshold
                self.mode = "absolute"
            else:
                self.mode = "relative"
    
    def reset_neutral(self):
        """Reset neutral position."""
        self.relative_mapper.reset_neutral_position()
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of hybrid mapper."""
        status = self.relative_mapper.get_status()
        status['mode'] = self.mode
        status['absolute_mapping_available'] = self.absolute_mapping is not None
        
        if self.absolute_mapping:
            status['absolute_mapping_r2'] = (
                self.absolute_mapping.get('r2_x', 0),
                self.absolute_mapping.get('r2_y', 0)
            )
        
        return status


# Simple test function
def test_mapper():
    """Test the relative gaze mapper."""
    print("Testing Relative Gaze Mapper...")
    
    try:
        mapper = RelativeGazeMapper("enhanced_calibration.pkl")
        
        # Simulate some eye positions
        test_positions = [
            # (left_eye, right_eye)
            ((380, 170), (300, 168)),  # Near neutral
            ((400, 170), (280, 168)),  # Right movement
            ((360, 170), (320, 168)),  # Left movement
            ((380, 180), (300, 178)),  # Down movement
            ((380, 160), (300, 158)),  # Up movement
        ]
        
        for i, (left_eye, right_eye) in enumerate(test_positions):
            mapper.update_eye_position(left_eye, right_eye)
            dx, dy = mapper.calculate_cursor_movement(1920, 1080)
            
            print(f"Test {i+1}:")
            print(f"  Eye: L{left_eye}, R{right_eye}")
            print(f"  Avg: ({(left_eye[0]+right_eye[0])/2:.1f}, {(left_eye[1]+right_eye[1])/2:.1f})")
            print(f"  Movement: dx={dx}, dy={dy}")
            print()
        
        print("Mapper test complete!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_mapper()