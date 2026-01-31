"""
Filename: run_cursor_control.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 1/28/2026

Description:
    Main runtime cursor control module using pyautogui.
    Uses the trained calibration model to control mouse cursor with eye gaze.
    
    Features:
    1. Load pre-trained calibration model
    2. Real-time iris tracking with MediaPipe
    3. Smooth cursor movement with EMA/Kalman filtering
    4. Performance monitoring and status display
    5. Keyboard shortcuts for control (Pause/Resume/Exit)
"""

import os
import sys
import time
import numpy as np
import cv2
from typing import Tuple

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import eye tracking modules
try:
    from MainControllers.gaze_runtime_controller import EyeTrackingRuntime
except ImportError:
    # Try alternative import path
    sys.path.insert(0, os.path.join(ROOT_DIR, 'AI Model'))
    from gaze_runtime_controller import EyeTrackingRuntime

# Import pyautogui for cursor control
try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    print("ERROR: pyautogui not installed!")
    print("Install with: pip install pyautogui")
    sys.exit(1)

# Import keyboard for shortcuts
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    print("WARNING: keyboard module not installed. Install with: pip install keyboard")
    print("You can still use the system but keyboard shortcuts won't work.")
    KEYBOARD_AVAILABLE = False


class GazeCursorController:
    """
    Main controller for gaze-based cursor control using pyautogui.
    """
    
    def __init__(self,
                 model_path: str = "calibration_model.pkl",
                 camera_id: int = 1,
                 filter_type: str = "ema",
                 smoothing_factor: float = 0.3,
                 min_movement_threshold: float = 0.005,
                 max_cursor_speed: float = 1000.0,
                 enable_smoothing: bool = True):
        """
        Initialize gaze cursor controller with simple defaults.
        
        Args:
            model_path: Path to trained calibration model
            camera_id: Camera ID for iris detection (default: 1)
            filter_type: Filter type for smoothing ("ema" or "kalman")
            smoothing_factor: Smoothing factor (0-1, higher = smoother)
            min_movement_threshold: Minimum normalized movement to trigger cursor update
            max_cursor_speed: Maximum cursor speed in pixels/second
            enable_smoothing: Enable cursor movement smoothing
        """
        # Configuration - using your successful calibration settings
        self.camera_id = camera_id
        self.filter_type = filter_type
        self.smoothing_factor = smoothing_factor
        self.min_movement_threshold = min_movement_threshold
        self.max_cursor_speed = max_cursor_speed
        self.enable_smoothing = enable_smoothing
        
        # State
        self.is_running = False
        self.is_paused = False
        self.tracking_active = False
        self.last_gaze_position = None
        self.last_cursor_position = None
        self.smoothed_gaze_x = None
        self.smoothed_gaze_y = None
        self.frame_count = 0
        self.start_time = 0
        
        # Performance metrics
        self.metrics = {
            'total_frames': 0,
            'tracked_frames': 0,
            'lost_frames': 0,
            'cursor_updates': 0,
            'avg_fps': 0.0,
            'tracking_accuracy': 0.0,
            'avg_processing_time': 0.0
        }
        
        # Get screen info
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"[CursorControl] Screen: {self.screen_width}x{self.screen_height}")
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"ERROR: Calibration model not found at {model_path}")
            print("Please run calibration first: python run_calibration.py")
            sys.exit(1)
        
        # Initialize eye tracker
        print(f"[CursorControl] Initializing gaze cursor controller...")
        print(f"[CursorControl] Model: {model_path}")
        print(f"[CursorControl] Camera: {camera_id}")
        print(f"[CursorControl] Filter: {filter_type}")
        print(f"[CursorControl] Smoothing: {smoothing_factor}")
        
        try:
            self.eye_tracker = EyeTrackingRuntime(
                model_path=model_path,
                camera_id=camera_id,
                screen_width=self.screen_width,
                screen_height=self.screen_height,
                filter_type=filter_type
            )
        except Exception as e:
            print(f"ERROR: Failed to initialize eye tracker: {e}")
            sys.exit(1)
        
        # Setup keyboard shortcuts if available
        if KEYBOARD_AVAILABLE:
            self._setup_keyboard_shortcuts()
        
        print("[CursorControl] Initialization complete!")
        print("[CursorControl] Controls:")
        print("  - P: Pause/Resume tracking")
        print("  - R: Reset smoothing")
        print("  - C: Calibrate center (move cursor to center)")
        print("  - ESC: Exit program")
        print("=" * 60)
    
    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for control."""
        keyboard.add_hotkey('p', self.toggle_pause)
        keyboard.add_hotkey('r', self.reset_smoothing)
        keyboard.add_hotkey('c', self.calibrate_center)
        keyboard.add_hotkey('esc', self.stop)
        print("[CursorControl] Keyboard shortcuts enabled")
    
    def toggle_pause(self):
        """Toggle pause state."""
        self.is_paused = not self.is_paused
        status = "PAUSED" if self.is_paused else "RESUMED"
        print(f"[CursorControl] Tracking {status}")
    
    def reset_smoothing(self):
        """Reset smoothing filters."""
        self.smoothed_gaze_x = None
        self.smoothed_gaze_y = None
        self.last_gaze_position = None
        if hasattr(self.eye_tracker, 'tracker') and hasattr(self.eye_tracker.tracker, 'reset_filters'):
            self.eye_tracker.tracker.reset_filters()
        print("[CursorControl] Smoothing reset")
    
    def calibrate_center(self):
        """Move cursor to screen center."""
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        pyautogui.moveTo(center_x, center_y, duration=0.1)
        print(f"[CursorControl] Cursor moved to center: ({center_x}, {center_y})")
    
    def start(self):
        """Start the gaze cursor control system."""
        if self.is_running:
            print("[CursorControl] Already running")
            return
        
        print("[CursorControl] Starting gaze cursor control...")
        print("[CursorControl] Look directly at the camera")
        print("[CursorControl] Keep your head still for best results")
        
        # Start eye tracker
        try:
            self.eye_tracker.start()
        except Exception as e:
            print(f"ERROR: Failed to start eye tracker: {e}")
            return
        
        # Reset metrics
        self.start_time = time.time()
        self.frame_count = 0
        self.metrics = {
            'total_frames': 0,
            'tracked_frames': 0,
            'lost_frames': 0,
            'cursor_updates': 0,
            'avg_fps': 0.0,
            'tracking_accuracy': 0.0,
            'avg_processing_time': 0.0
        }
        
        self.is_running = True
        self.tracking_active = True
        
        print("[CursorControl] Gaze cursor control started!")
    
    def stop(self):
        """Stop the gaze cursor control system."""
        if not self.is_running:
            return
        
        print("\n[CursorControl] Stopping gaze cursor control...")
        
        self.is_running = False
        self.tracking_active = False
        
        # Stop eye tracker
        try:
            self.eye_tracker.stop()
        except Exception as e:
            print(f"Warning: Error stopping eye tracker: {e}")
        
        # Calculate final metrics
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.metrics['avg_fps'] = self.frame_count / elapsed
            if self.metrics['total_frames'] > 0:
                self.metrics['tracking_accuracy'] = (
                    self.metrics['tracked_frames'] / self.metrics['total_frames'] * 100
                )
        
        print("[CursorControl] Gaze cursor control stopped")
        print(f"[CursorControl] Final metrics:")
        print(f"  Total frames: {self.metrics['total_frames']}")
        print(f"  Tracked frames: {self.metrics['tracked_frames']} ({self.metrics['tracking_accuracy']:.1f}%)")
        print(f"  Cursor updates: {self.metrics['cursor_updates']}")
        print(f"  Average FPS: {self.metrics['avg_fps']:.1f}")
        
        # Close any OpenCV windows
        cv2.destroyAllWindows()
    
    def _apply_smoothing(self, x: float, y: float) -> Tuple[float, float]:
        """
        Apply exponential smoothing to gaze coordinates.
        
        Args:
            x: Raw x coordinate
            y: Raw y coordinate
            
        Returns:
            Smoothed (x, y) coordinates
        """
        if not self.enable_smoothing or self.smoothing_factor <= 0:
            return x, y
        
        if self.smoothed_gaze_x is None or self.smoothed_gaze_y is None:
            self.smoothed_gaze_x = x
            self.smoothed_gaze_y = y
            return x, y
        
        # Exponential moving average
        self.smoothed_gaze_x = (self.smoothing_factor * x + 
                               (1 - self.smoothing_factor) * self.smoothed_gaze_x)
        self.smoothed_gaze_y = (self.smoothing_factor * y + 
                               (1 - self.smoothing_factor) * self.smoothed_gaze_y)
        
        return self.smoothed_gaze_x, self.smoothed_gaze_y
    
    def _should_update_cursor(self, new_pos: Tuple[float, float]) -> bool:
        """
        Check if cursor should be updated based on movement threshold.
        
        Args:
            new_pos: New gaze position (normalized)
            
        Returns:
            True if cursor should be updated
        """
        if self.last_gaze_position is None:
            return True
        
        dx = abs(new_pos[0] - self.last_gaze_position[0])
        dy = abs(new_pos[1] - self.last_gaze_position[1])
        
        # Update if movement exceeds threshold
        return (dx > self.min_movement_threshold or 
                dy > self.min_movement_threshold)
    
    def update(self):
        """
        Update gaze tracking and cursor position.
        Returns True if successful, False otherwise.
        """
        if not self.is_running or self.is_paused:
            return False
        
        frame_start_time = time.time()
        self.frame_count += 1
        self.metrics['total_frames'] += 1
        
        # Get gaze position from tracker
        try:
            result = self.eye_tracker.get_gaze_position()
        except Exception as e:
            print(f"[CursorControl] Error getting gaze position: {e}")
            self.metrics['lost_frames'] += 1
            return False
        
        if result is None:
            self.metrics['lost_frames'] += 1
            return False
        
        self.metrics['tracked_frames'] += 1
        
        # Get normalized and pixel coordinates
        normalized = result['normalized']
        pixels = result['pixels']
        
        # Apply smoothing
        if self.enable_smoothing:
            smoothed_norm = self._apply_smoothing(normalized[0], normalized[1])
            # Convert smoothed normalized back to pixels
            pixels = (
                int(smoothed_norm[0] * self.screen_width),
                int(smoothed_norm[1] * self.screen_height)
            )
        
        # Check if we should update cursor
        if self._should_update_cursor(normalized):
            # Limit cursor speed
            current_pos = pyautogui.position()
            dx = pixels[0] - current_pos[0]
            dy = pixels[1] - current_pos[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance > 0:
                # Calculate maximum movement per frame (assuming 60 FPS)
                max_move_per_frame = self.max_cursor_speed / 60.0
                
                if distance > max_move_per_frame:
                    # Scale down movement
                    scale = max_move_per_frame / distance
                    target_x = current_pos[0] + int(dx * scale)
                    target_y = current_pos[1] + int(dy * scale)
                else:
                    target_x, target_y = pixels
                
                # Move cursor with pyautogui
                try:
                    pyautogui.moveTo(target_x, target_y, duration=0)
                    self.metrics['cursor_updates'] += 1
                    self.last_gaze_position = normalized
                    self.last_cursor_position = (target_x, target_y)
                except Exception as e:
                    print(f"[CursorControl] Error moving cursor: {e}")
        
        # Calculate processing time
        processing_time = time.time() - frame_start_time
        self.metrics['avg_processing_time'] = (
            self.metrics['avg_processing_time'] * 0.9 + processing_time * 0.1
        )
        
        # Update FPS
        elapsed_total = time.time() - self.start_time
        if elapsed_total > 0:
            self.metrics['avg_fps'] = self.frame_count / elapsed_total
        
        return True
    
    def run_continuous(self):
        """
        Run continuous gaze cursor control with default settings.
        """
        self.start()
        
        print("[CursorControl] Starting continuous tracking...")
        print("[CursorControl] Move your eyes to control the cursor!")
        
        target_fps = 60.0
        frame_time = 1.0 / target_fps
        last_status_time = time.time()
        status_interval = 2.0  # Print status every 2 seconds
        
        try:
            while self.is_running:
                frame_start = time.time()
                
                # Update gaze tracking
                success = self.update()
                
                # Print status occasionally
                current_time = time.time()
                if current_time - last_status_time > status_interval:
                    fps = self.metrics['avg_fps']
                    accuracy = self.metrics['tracking_accuracy']
                    cursor_pos = self.last_cursor_position or pyautogui.position()
                    
                    print(f"[Status] FPS: {fps:.1f} | "
                          f"Accuracy: {accuracy:.1f}% | "
                          f"Cursor: {cursor_pos} | "
                          f"Processing: {self.metrics['avg_processing_time']*1000:.1f}ms")
                    last_status_time = current_time
                
                # Sleep to maintain target FPS
                processing_time = time.time() - frame_start
                sleep_time = max(0, frame_time - processing_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n[CursorControl] Interrupted by user")
        except Exception as e:
            print(f"\n[CursorControl] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()


def main():
    """Main function to run gaze cursor control with simple defaults."""
    
    # Display startup banner
    print("=" * 60)
    print("Gaze Cursor Control System - Windows 11")
    print("=" * 60)
    print("Default Settings:")
    print("  Model: calibration_model.pkl")
    print("  Camera: 1")
    print("  Filter: ema")
    print("  Smoothing: 0.3")
    print("  Max speed: 1000 px/sec")
    print("=" * 60)
    
    # Check if model exists
    model_path = "calibration_model.pkl"
    if not os.path.exists(model_path):
        print(f"ERROR: Calibration model not found at {model_path}")
        print("Please run calibration first: python run_calibration.py")
        print("Or place your model file in the current directory.")
        sys.exit(1)
    
    # Check pyautogui
    if not PYAUTOGUI_AVAILABLE:
        print("ERROR: pyautogui is not installed!")
        print("Install with: pip install pyautogui")
        sys.exit(1)
    
    # Disable pyautogui failsafe (optional, be careful!)
    pyautogui.FAILSAFE = False
    
    # Countdown before starting
    print("\nStarting in 3 seconds...")
    print("Please look directly at the camera")
    print("Keep your head still for best results")
    
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    print("GO!")
    print("=" * 60)
    
    # Create and run cursor controller with default settings
    controller = GazeCursorController(
        model_path=model_path,
        camera_id=1,           # Based on your successful calibration
        filter_type="ema",     # Exponential Moving Average filter
        smoothing_factor=0.3,  # Good balance of responsiveness and smoothness
        min_movement_threshold=0.005,  # Small threshold for precise control
        max_cursor_speed=1000.0,       # Reasonable maximum speed
        enable_smoothing=True          # Enable smoothing by default
    )
    
    # Run continuous control
    controller.run_continuous()


if __name__ == "__main__":
    # Add current directory to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Run main function
    main()