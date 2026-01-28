"""
Filename: run_gaze_vector_control_fixed.py
Creator/Author: Basel Mohamed Mostafa Sayed + ChatGPT
Date: 1/27/2026

Description:
    Fixed version with center calibration to remove bias.
"""

import os
import sys
import time
import numpy as np
import cv2
import pyautogui
from typing import Optional, Tuple, List

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import gaze vector mapper
try:
    from gaze_vector_mapper import GazeVectorMapper
except ImportError:
    print("ERROR: gaze_vector_mapper.py not found")
    sys.exit(1)

# Import existing modules
try:
    from VisionModule.camera import IrisCamera
    from VisionModule.face_utils import FaceMeshDetector
except ImportError:
    sys.path.insert(0, os.path.join(ROOT_DIR, 'AI Model'))
    from VisionModule.camera import IrisCamera
    from VisionModule.face_utils import FaceMeshDetector


class GazeVectorCursorControl:
    """
    Fixed version with center calibration to remove bias.
    """
    
    def __init__(self,
                 camera_id: int = 1,
                 calibration_path: Optional[str] = None,
                 show_visualization: bool = True,
                 target_fps: int = 60):
        """
        Initialize gaze vector cursor control.
        
        Args:
            camera_id: Camera ID
            calibration_path: Path to calibration data (optional)
            show_visualization: Show visualization
            target_fps: Target frames per second
        """
        # Configuration
        self.camera_id = camera_id
        self.calibration_path = calibration_path
        self.show_visualization = show_visualization
        self.target_fps = target_fps
        
        # Calibration state
        self.neutral_gaze_vector = None  # Gaze vector when looking straight ahead
        self.gaze_bias = np.array([0.0, 0.0, 0.0])  # Bias correction
        self.is_center_calibrated = False
        
        # Coordinate inversion flags
        self.invert_x = False  # Start with no inversion
        self.invert_y = False
        
        # State
        self.is_running = False
        self.is_paused = False
        self.frame_count = 0
        self.start_time = 0
        self.fps = 0
        self.fps_frame_count = 0
        self.fps_last_update_time = 0
        
        # Performance metrics
        self.metrics = {
            'total_frames': 0,
            'processed_frames': 0,
            'failed_frames': 0,
            'cursor_updates': 0,
            'avg_fps': 0.0,
            'current_fps': 0.0
        }
        
        # Get screen info
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"[GazeVectorControl] Screen: {self.screen_width}x{self.screen_height}")
        
        # Initialize gaze vector mapper with calibration if available
        print("[GazeVectorControl] Initializing 3D gaze vector mapper...")
        try:
            self.gaze_mapper = GazeVectorMapper(
                calibration_path=calibration_path
            )
            if calibration_path and os.path.exists(calibration_path):
                print(f"[GazeVectorControl] Loaded calibration from {calibration_path}")
            else:
                print("[GazeVectorControl] Using default geometry")
        except Exception as e:
            print(f"ERROR: Failed to initialize gaze mapper: {e}")
            sys.exit(1)
        
        # Initialize camera and detector
        print(f"[GazeVectorControl] Initializing camera {camera_id}...")
        try:
            self.camera = IrisCamera(cam_id=camera_id)
            self.detector = FaceMeshDetector(refine_landmarks=True)
            
            self.cam_width, self.cam_height = self.camera.get_resolution()
            print(f"[GazeVectorControl] Camera resolution: {self.cam_width}x{self.cam_height}")
            
        except Exception as e:
            print(f"ERROR: Failed to initialize camera/detector: {e}")
            sys.exit(1)
        
        # Initialize keyboard control
        self._init_keyboard()
        
        print("[GazeVectorControl] Initialization complete!")
        print("[GazeVectorControl] FIRST: Press 'C' to calibrate center (look straight ahead)")
        print("[GazeVectorControl] THEN: Use X/Y keys to fix inversion if needed")
        print("=" * 60)
    
    def _init_keyboard(self):
        """Initialize keyboard controls."""
        try:
            import keyboard
            
            keyboard.add_hotkey('p', self.toggle_pause)
            keyboard.add_hotkey('r', self.recenter_cursor)
            keyboard.add_hotkey('c', self.calibrate_center)  # IMPORTANT: Calibrate center
            keyboard.add_hotkey('x', self.toggle_invert_x)
            keyboard.add_hotkey('y', self.toggle_invert_y)
            keyboard.add_hotkey('b', self.calibrate_bias)  # Additional bias calibration
            keyboard.add_hotkey('esc', self.stop)
            
            print("[GazeVectorControl] Keyboard controls enabled")
            print("  C: Calibrate center (look straight, press C)")
            print("  B: Fine-tune bias calibration")
            print("  X: Toggle X-axis inversion")
            print("  Y: Toggle Y-axis inversion")
            print("  R: Recenter cursor")
            print("  P: Pause/Resume")
            print("  ESC: Exit")
            
        except ImportError:
            print("[GazeVectorControl] Keyboard module not available")
    
    def toggle_pause(self):
        """Toggle pause state."""
        self.is_paused = not self.is_paused
        status = "PAUSED" if self.is_paused else "RESUMED"
        print(f"[GazeVectorControl] {status}")
    
    def recenter_cursor(self):
        """Move cursor to screen center."""
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        pyautogui.moveTo(center_x, center_y, duration=0.1)
        print(f"[GazeVectorControl] Cursor recentered: ({center_x}, {center_y})")
    
    def toggle_invert_x(self):
        """Toggle X-axis inversion."""
        self.invert_x = not self.invert_x
        status = "INVERTED" if self.invert_x else "NORMAL"
        print(f"[GazeVectorControl] X-axis: {status}")
    
    def toggle_invert_y(self):
        """Toggle Y-axis inversion."""
        self.invert_y = not self.invert_y
        status = "INVERTED" if self.invert_y else "NORMAL"
        print(f"[GazeVectorControl] Y-axis: {status}")
    
    def calibrate_center(self):
        """
        Calibrate the center gaze position.
        Look straight ahead at the screen center and press C.
        """
        print("[GazeVectorControl] Calibrating center gaze...")
        print("Look straight ahead at the CENTER of the screen")
        print("Keep your head still for 3 seconds...")
        
        # Collect gaze vectors while user looks straight ahead
        gaze_vectors = []
        start_time = time.time()
        
        while time.time() - start_time < 3.0:  # Collect for 3 seconds
            frame = self.camera.get_frame()
            if frame is None:
                continue
            
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.face_mesh.process(img_rgb)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                gaze_vector, _ = self.gaze_mapper.calculate_gaze_vector(face_landmarks.landmark)
                if gaze_vector is not None:
                    gaze_vectors.append(gaze_vector)
            
            time.sleep(0.03)  # ~30 Hz
        
        if gaze_vectors:
            # Calculate average gaze vector when looking straight ahead
            self.neutral_gaze_vector = np.mean(gaze_vectors, axis=0)
            self.neutral_gaze_vector = self.neutral_gaze_vector / np.linalg.norm(self.neutral_gaze_vector)
            
            # The neutral gaze vector should ideally be [0, 0, -1] (looking straight into screen)
            # Calculate bias as deviation from ideal
            ideal_neutral = np.array([0.0, 0.0, -1.0])
            self.gaze_bias = self.neutral_gaze_vector - ideal_neutral
            
            self.is_center_calibrated = True
            
            print(f"[GazeVectorControl] Center calibration complete!")
            print(f"  Neutral gaze: [{self.neutral_gaze_vector[0]:.3f}, "
                  f"{self.neutral_gaze_vector[1]:.3f}, {self.neutral_gaze_vector[2]:.3f}]")
            print(f"  Gaze bias: [{self.gaze_bias[0]:.3f}, "
                  f"{self.gaze_bias[1]:.3f}, {self.gaze_bias[2]:.3f}]")
            
            # Recenter cursor after calibration
            self.recenter_cursor()
        else:
            print("[GazeVectorControl] Failed to calibrate center - no gaze vectors collected")
    
    def calibrate_bias(self):
        """
        Fine-tune bias calibration.
        Use this if cursor is still off-center after initial calibration.
        """
        if not self.is_center_calibrated:
            print("[GazeVectorControl] Please run center calibration (C) first")
            return
        
        print("[GazeVectorControl] Fine-tuning bias...")
        print("Look at where the cursor SHOULD be (screen center)")
        print("Current cursor position will be used to adjust bias")
        
        # Get current gaze vector
        frame = self.camera.get_frame()
        if frame is None:
            return
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.face_mesh.process(img_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            current_gaze, _ = self.gaze_mapper.calculate_gaze_vector(face_landmarks.landmark)
            
            if current_gaze is not None:
                # Adjust bias based on current gaze
                # If cursor is off-center, we need to adjust the bias
                adjustment_factor = 0.3  # Small adjustments
                self.gaze_bias = self.gaze_bias * (1 - adjustment_factor) + current_gaze * adjustment_factor
                
                print(f"[GazeVectorControl] Bias adjusted")
                print(f"  New bias: [{self.gaze_bias[0]:.3f}, "
                      f"{self.gaze_bias[1]:.3f}, {self.gaze_bias[2]:.3f}]")
    
    def _apply_bias_correction(self, gaze_vector: np.ndarray) -> np.ndarray:
        """
        Apply bias correction to gaze vector.
        
        Args:
            gaze_vector: Raw gaze vector
            
        Returns:
            Corrected gaze vector
        """
        if not self.is_center_calibrated or self.neutral_gaze_vector is None:
            return gaze_vector
        
        # Simple bias subtraction
        corrected = gaze_vector.copy()
        
        # Method 1: Subtract neutral gaze (make it the new zero)
        corrected = corrected - self.neutral_gaze_vector
        
        # Method 2: Alternatively, rotate toward ideal
        # For now, just use subtraction
        
        # Renormalize
        norm = np.linalg.norm(corrected)
        if norm > 0:
            corrected = corrected / norm
        
        return corrected
    
    def start(self):
        """Start the gaze vector control system."""
        if self.is_running:
            return
        
        print("[GazeVectorControl] Starting 3D gaze vector system...")
        print("[GazeVectorControl] FIRST: Look straight ahead and press 'C' to calibrate")
        
        # Reset metrics
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0
        self.fps_frame_count = 0
        self.fps_last_update_time = self.start_time
        
        self.metrics = {
            'total_frames': 0,
            'processed_frames': 0,
            'failed_frames': 0,
            'cursor_updates': 0,
            'avg_fps': 0.0,
            'current_fps': 0.0
        }
        
        self.is_running = True
        print("[GazeVectorControl] System started!")
    
    def stop(self):
        """Stop the system."""
        if not self.is_running:
            return
        
        print("\n[GazeVectorControl] Stopping system...")
        self.is_running = False
        
        # Release resources
        self.camera.stop_recording()
        self.detector.release()
        
        # Calculate final metrics
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.metrics['avg_fps'] = self.frame_count / elapsed
        
        print("[GazeVectorControl] System stopped")
        print(f"[GazeVectorControl] Final metrics:")
        print(f"  Total frames: {self.metrics['total_frames']}")
        print(f"  Processed frames: {self.metrics['processed_frames']}")
        print(f"  Cursor updates: {self.metrics['cursor_updates']}")
        print(f"  Average FPS: {self.metrics['avg_fps']:.1f}")
        
        cv2.destroyAllWindows()
    
    def _calculate_fps(self):
        """Calculate current FPS."""
        current_time = time.time()
        
        # Increment frame counter for FPS calculation
        self.fps_frame_count += 1
        
        # Update FPS every second
        if current_time - self.fps_last_update_time >= 1.0:
            self.fps = self.fps_frame_count / (current_time - self.fps_last_update_time)
            self.metrics['current_fps'] = self.fps
            self.fps_frame_count = 0
            self.fps_last_update_time = current_time
    
    def _gaze_to_screen_coordinates(self, gaze_vector: np.ndarray) -> Tuple[float, float]:
        """
        Convert gaze vector to screen coordinates with bias correction.
        
        Args:
            gaze_vector: 3D gaze direction
            
        Returns:
            Normalized screen coordinates (x, y) in [0, 1]
        """
        # Apply bias correction
        corrected_gaze = self._apply_bias_correction(gaze_vector)
        
        # Extract horizontal and vertical components
        # In MediaPipe coordinates:
        # - X: Positive = right, Negative = left
        # - Y: Positive = down, Negative = up  
        # - Z: Positive = toward camera, Negative = away from camera
        
        horizontal = corrected_gaze[0]  # Looking left/right
        vertical = corrected_gaze[1]    # Looking up/down
        
        # Scale factors - adjust these for sensitivity
        # Larger values = more screen movement for same eye movement
        scale_x = 2.0
        scale_y = 2.0
        
        # Map to normalized screen coordinates
        # When looking straight: horizontal=0, vertical=0 -> center (0.5, 0.5)
        norm_x = 0.5 + horizontal * scale_x
        norm_y = 0.5 + vertical * scale_y
        
        # Apply inversion if needed
        if self.invert_x:
            norm_x = 1.0 - norm_x
        
        if self.invert_y:
            norm_y = 1.0 - norm_y
        
        # Clamp to valid range
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))
        
        return norm_x, norm_y
    
    def process_frame(self) -> bool:
        """
        Process a single frame for gaze tracking.
        
        Returns:
            True if successful, False otherwise
        """
        self.frame_count += 1
        self.metrics['total_frames'] += 1
        
        # Calculate FPS
        self._calculate_fps()
        
        # Get camera frame
        frame = self.camera.get_frame()
        if frame is None:
            self.metrics['failed_frames'] += 1
            return False
        
        # Process frame for face mesh (with 3D landmarks)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.face_mesh.process(img_rgb)
        
        if not results.multi_face_landmarks:
            self.metrics['failed_frames'] += 1
            return False
        
        # Get first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Calculate gaze vector
        gaze_vector, debug_info = self.gaze_mapper.calculate_gaze_vector(face_landmarks.landmark)
        
        if gaze_vector is None:
            self.metrics['failed_frames'] += 1
            return False
        
        # Convert to screen coordinates
        norm_x, norm_y = self._gaze_to_screen_coordinates(gaze_vector)
        
        # Convert to pixels
        pixel_x = int(norm_x * self.screen_width)
        pixel_y = int(norm_y * self.screen_height)
        
        # Apply exponential smoothing for smoother cursor movement
        if not hasattr(self, 'smoothed_x'):
            self.smoothed_x = pixel_x
            self.smoothed_y = pixel_y
        else:
            smoothing_factor = 0.2  # Lower = smoother, higher = more responsive
            self.smoothed_x = int(self.smoothed_x * (1 - smoothing_factor) + pixel_x * smoothing_factor)
            self.smoothed_y = int(self.smoothed_y * (1 - smoothing_factor) + pixel_y * smoothing_factor)
        
        # Move cursor
        pyautogui.moveTo(self.smoothed_x, self.smoothed_y, duration=0)
        self.metrics['cursor_updates'] += 1
        self.metrics['processed_frames'] += 1
        
        # Show visualization if enabled
        if self.show_visualization:
            self._show_visualization(frame, face_landmarks, gaze_vector, (pixel_x, pixel_y))
        
        return True
    
    def _show_visualization(self, frame, face_landmarks, gaze_vector, cursor_pos):
        """Show visualization with gaze information."""
        display_frame = frame.copy()
        height, width, _ = display_frame.shape
        
        # Draw face mesh
        self.detector.draw_face_mesh(display_frame, draw_iris=True)
        
        # Display information
        y_offset = 30
        info_lines = [
            f"FPS: {self.fps:.1f}",
            f"Cursor: {cursor_pos}",
            f"Center Calibrated: {self.is_center_calibrated}",
            f"Invert X: {self.invert_x}, Invert Y: {self.invert_y}",
            f"Gaze Vector: [{gaze_vector[0]:.3f}, {gaze_vector[1]:.3f}, {gaze_vector[2]:.3f}]",
        ]
        
        if self.is_center_calibrated and self.neutral_gaze_vector is not None:
            info_lines.append(f"Neutral Gaze: [{self.neutral_gaze_vector[0]:.3f}, "
                            f"{self.neutral_gaze_vector[1]:.3f}, {self.neutral_gaze_vector[2]:.3f}]")
        
        for line in info_lines:
            cv2.putText(display_frame, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        # Draw gaze direction on camera frame
        center_x, center_y = width // 2, height // 2
        scale = 100
        
        # Project gaze vector to 2D for display
        gaze_end_x = int(center_x + gaze_vector[0] * scale * 50)
        gaze_end_y = int(center_y + gaze_vector[1] * scale * 50)
        
        cv2.arrowedLine(display_frame, 
                       (center_x, center_y),
                       (gaze_end_x, gaze_end_y),
                       (0, 0, 255), 3)
        
        # Draw center marker
        cv2.circle(display_frame, (center_x, center_y), 5, (0, 255, 0), -1)
        
        # Show frame
        cv2.imshow("3D Gaze Vector Tracking", display_frame)
        cv2.waitKey(1)
    
    def run(self):
        """Run the gaze vector control system."""
        self.start()
        
        print("[GazeVectorControl] Running 3D gaze vector tracking...")
        print("[GazeVectorControl] REMEMBER: Press 'C' to calibrate center first!")
        
        frame_time = 1.0 / self.target_fps
        last_status_time = time.time()
        status_interval = 2.0
        
        try:
            while self.is_running and not self.is_paused:
                frame_start = time.time()
                
                # Process frame
                success = self.process_frame()
                
                # Print status occasionally
                current_time = time.time()
                if current_time - last_status_time > status_interval:
                    cursor_pos = pyautogui.position()
                    
                    print(f"[Status] FPS: {self.fps:.1f} | "
                          f"Cursor: {cursor_pos} | "
                          f"Center Calibrated: {self.is_center_calibrated} | "
                          f"Invert: X={self.invert_x}, Y={self.invert_y}")
                    last_status_time = current_time
                
                # Maintain target FPS
                processing_time = time.time() - frame_start
                sleep_time = max(0, frame_time - processing_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n[GazeVectorControl] Interrupted by user")
        except Exception as e:
            print(f"\n[GazeVectorControl] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()


def main():
    """Main function to run gaze vector control."""
    
    print("=" * 70)
    print("3D GAZE VECTOR CURSOR CONTROL - WITH CENTER CALIBRATION")
    print("=" * 70)
    print("FIXES: Bias correction for accurate centering")
    print("=" * 70)
    
    # Check for calibration file
    calibration_path = "gaze_vector_calibration.pkl"
    use_calibration = os.path.exists(calibration_path)
    
    if use_calibration:
        print(f"Using calibration: {calibration_path}")
    else:
        print("No calibration found. Using default geometry.")
    
    print("=" * 70)
    print("IMPORTANT INSTRUCTIONS:")
    print("1. Start the program")
    print("2. Look straight ahead at screen CENTER")
    print("3. Press 'C' to calibrate center")
    print("4. Use X/Y keys if inversion is wrong")
    print("5. Use 'B' to fine-tune if still off-center")
    print("=" * 70)
    
    # Check dependencies
    try:
        import pyautogui
    except ImportError:
        print("ERROR: pyautogui required")
        sys.exit(1)
    
    # Disable pyautogui failsafe
    pyautogui.FAILSAFE = False
    
    # Countdown
    print("\nStarting in 3 seconds...")
    
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    print("GO!")
    print("=" * 70)
    
    # Create and run controller
    controller = GazeVectorCursorControl(
        camera_id=1,
        calibration_path=calibration_path if use_calibration else None,
        show_visualization=True,
        target_fps=60
    )
    
    # Run system
    controller.run()


if __name__ == "__main__":
    # Add current directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Run main function
    main()