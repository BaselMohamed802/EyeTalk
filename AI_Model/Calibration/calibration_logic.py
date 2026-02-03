"""
Filename: calibration_logic.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 1/24/2026

Description:
    Calibration timing logic controller.
    Manages fixation timing and declares when sampling is allowed
    
    Responsibilities:
    1. Show calibration points
    2. Wait for fixation settling delay
    3. Signal when sampling may begin

"""

# Import necessary libraries
import time
from typing import Callable, Optional


class CalibrationTimingController:
    """
    Controller responsible for fixation timing and 
    sampling authorization during calibration.
    """
    def __init__(self, 
                 num_points: int = 9,
                 settling_delay: float = 0.5,
                 ui=None):
        """
        Initialize timing controller.
        
        Args:
            num_points: Number of calibration points
            settling_delay: Time to wait after showing point (seconds)
            ui: Optional UI object (for testing integration)
        """
        self.num_points = num_points
        self.settling_delay = settling_delay
        self.fixation_start_time = None
        self.waiting_for_fixation = False
        
        # Store UI reference (if provided)
        self.ui = ui
        
        # Calibration state
        self.current_point = -1  # -1 means not started
        self.sampling_allowed = False
        
        # Callbacks for external components
        self.on_point_started = None
        self.on_sampling_start = None
        self.on_calibration_complete = None
        
        # Generate calibration points (normalized coordinates)
        self.calibration_points = self._generate_calibration_points(num_points)
        
        print(f"Timing controller initialized with {num_points} points")
        print(f"Settling delay: {settling_delay}s")
    
    def _generate_calibration_points(self, num_points: int, margin_screen_edge: float = 0.1):
        """Generate normalized calibration points (0-1)."""
        if num_points == 9:
            rows, cols = 3, 3
        elif num_points == 16:
            rows, cols = 4, 4
        else:
            rows, cols = 3, 3

        # Assign margin        
        margin = margin_screen_edge
        
        points = []
        for i in range(rows):
            for j in range(cols):
                x_norm = margin + j * ((1 - 2 * margin) / (cols - 1))
                y_norm = margin + i * ((1 - 2 * margin) / (rows - 1))
                points.append((x_norm, y_norm))
        
        return points
    
    def start_calibration(self):
        """Start the calibration process."""
        print(f"\n[Controller] Calibration started with {self.num_points} points")
        self.current_point = -1
        self._advance_to_next_point()
    
    def _advance_to_next_point(self):
        """Advance to the next calibration point."""
        next_point = self.current_point + 1
        
        if next_point < self.num_points:
            self._start_point(next_point)
        else:
            self._finish_calibration()
    
    def _start_point(self, point_index: int):
        """
        Start calibration at a specific point.
        
        Args:
            point_index: Index of point to start (0-based)
        """
        self.current_point = point_index
        self.sampling_allowed = False
        
        # Get point coordinates
        x_norm, y_norm = self.calibration_points[point_index]
        
        # Show the point (if UI exists)
        if self.ui and hasattr(self.ui, 'show_calibration_point'):
            success = self.ui.show_calibration_point(point_index)
            if not success:
                print(f"[Warning] UI failed to show point {point_index}")
        
        print(f"\n[Controller] Point {point_index + 1}/{self.num_points}")
        print(f"[Controller] Coordinates: ({x_norm:.2f}, {y_norm:.2f}) normalized")
        print(f"[Controller] Waiting {self.settling_delay}s for user to settle...")
        
        # Notify that point has started
        if self.on_point_started:
            self.on_point_started(point_index, x_norm, y_norm)
        
        # Wait for settling delay, then start sampling
        self.fixation_start_time = time.time()
        self.waiting_for_fixation = True


    def _signal_sampling_ready(self):
        """
        Signal that sampling can start at current point.

        This signal is emitted when the settling delay has passed and the user is ready to start sampling.
        The sampling_start callback will be called after this signal is emitted.
        """
        print("\n[Controller] [Signal] SAMPLING CAN START")
        self.sampling_allowed = True

        if self.on_sampling_start:
            self.on_sampling_start(self.current_point)
    
    def _finish_calibration(self):
        """Finish the calibration process."""
        print("\n" + "="*50)
        print("[Controller] CALIBRATION TIMING COMPLETE")
        print("[Controller] All points have been displayed")
        print("="*50)
        
        # Notify that calibration is complete
        if self.on_calibration_complete:
            self.on_calibration_complete()
    
    def get_current_point_info(self):
        """
        Get information about current calibration point.
        
        Returns:
            dict: Point information or None if no active point
        """
        if self.current_point < 0 or self.current_point >= len(self.calibration_points):
            return None
        
        x_norm, y_norm = self.calibration_points[self.current_point]
                
        return {
            'index': self.current_point,
            'total_points': self.num_points,
            'x_norm': x_norm,
            'y_norm': y_norm,
            'sampling_allowed': self.sampling_allowed
        }

    
    def advance_to_next_point(self):
        """
        Advance to the next calibration point.

        This method triggers the timing controller to advance to the next point, which may
        involve showing a new calibration point, waiting for the settling delay, and
        starting sampling.

        This method can be called by the UI to manually advance to the next calibration
        point when the user presses a key or when the UI decides to advance.
        """
        self._advance_to_next_point()
    
    def update(self):
        """
        Update the timing controller state.

        If waiting for fixation, check if the settling delay has passed.
        If so, signal that sampling can start and stop waiting for fixation.
        """
        if self.waiting_for_fixation:
            elapsed = time.time() - self.fixation_start_time
            if elapsed >= self.settling_delay:
                self.waiting_for_fixation = False
                self._signal_sampling_ready()
