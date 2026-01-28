"""
Filename: run_relative_cursor_control.py
Creator/Author: Basel Mohamed Mostafa Sayed + ChatGPT
Date: 1/27/2026

Description:
    Main application for relative gaze cursor control.
    Uses the relative gaze mapper for intuitive cursor movement.
    
    Features:
    1. Relative eye movement mapping (not absolute regression)
    2. Head stabilization
    3. Adjustable sensitivity
    4. Real-time feedback
    5. Keyboard controls
"""

import os
import sys
import time
import numpy as np
import cv2
from typing import Optional, Tuple

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import gaze mapper
try:
    from relative_gaze_mapper import RelativeGazeMapper, HybridGazeMapper
except ImportError:
    # Try creating the module if it doesn't exist
    print("Creating relative_gaze_mapper module...")
    # The module should be created by the previous code

# Import existing modules
try:
    from VisionModule.camera import IrisCamera
    from VisionModule.face_utils import FaceMeshDetector
except ImportError:
    sys.path.insert(0, os.path.join(ROOT_DIR, 'AI Model'))
    from VisionModule.camera import IrisCamera
    from VisionModule.face_utils import FaceMeshDetector

# Import cursor control
try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    print("ERROR: pyautogui not installed!")
    print("Install with: pip install pyautogui")
    sys.exit(1)

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    print("WARNING: keyboard module not installed.")
    KEYBOARD_AVAILABLE = False


class RelativeGazeCursorControl:
    """
    Main controller for relative gaze cursor control.
    """
    
    def __init__(self,
                 calibration_path: str = "enhanced_calibration.pkl",
                 camera_id: int = 1,
                 use_hybrid: bool = True,
                 enable_head_stabilization: bool = True,
                 show_camera_feed: bool = False,
                 show_debug_info: bool = True):
        """
        Initialize relative gaze cursor control.
        
        Args:
            calibration_path: Path to enhanced calibration data
            camera_id: Camera ID
            use_hybrid: Use hybrid mapper (relative + absolute)
            enable_head_stabilization: Use head position for stabilization
            show_camera_feed: Show camera feed with overlays
            show_debug_info: Show debug information on screen
        """
        # Configuration
        self.calibration_path = calibration_path
        self.camera_id = camera_id
        self.use_hybrid = use_hybrid
        self.enable_head_stabilization = enable_head_stabilization
        self.show_camera_feed = show_camera_feed
        self.show_debug_info = show_debug_info
        
        # State
        self.is_running = False
        self.is_paused = False
        self.frame_count = 0
        self.start_time = 0
        
        # Performance metrics
        self.metrics = {
            'total_frames': 0,
            'tracked_frames': 0,
            'lost_frames': 0,
            'cursor_updates': 0,
            'avg_fps': 0.0,
            'tracking_accuracy': 0.0
        }
        
        # Get screen info
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"[RelativeControl] Screen: {self.screen_width}x{self.screen_height}")
        
        # Check if calibration exists
        if not os.path.exists(calibration_path):
            print(f"ERROR: Enhanced calibration not found at {calibration_path}")
            print("Please run enhanced_calibration.py first")
            sys.exit(1)
        
        # Initialize gaze mapper
        print(f"[RelativeControl] Initializing gaze mapper...")
        try:
            if use_hybrid:
                self.gaze_mapper = HybridGazeMapper(calibration_path)
                print("[RelativeControl] Using HYBRID gaze mapper")
            else:
                self.gaze_mapper = RelativeGazeMapper(calibration_path)
                print("[RelativeControl] Using RELATIVE gaze mapper")
        except Exception as e:
            print(f"ERROR: Failed to initialize gaze mapper: {e}")
            sys.exit(1)
        
        # Initialize camera and detector
        print(f"[RelativeControl] Initializing camera {camera_id}...")
        try:
            self.camera = IrisCamera(cam_id=camera_id)
            self.detector = FaceMeshDetector()
        except Exception as e:
            print(f"ERROR: Failed to initialize camera/detector: {e}")
            sys.exit(1)
        
        # Setup keyboard shortcuts
        if KEYBOARD_AVAILABLE:
            self._setup_keyboard_shortcuts()
        
        print("[RelativeControl] Initialization complete!")
        print("[RelativeControl] Controls:")
        print("  - P: Pause/Resume tracking")
        print("  - R: Reset neutral position (look straight, press R)")
        print("  - +: Increase sensitivity")
        print("  - -: Decrease sensitivity")
        print("  - C: Center cursor")
        print("  - ESC: Exit")
        print("=" * 60)
    
    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts."""
        keyboard.add_hotkey('p', self.toggle_pause)
        keyboard.add_hotkey('r', self.reset_neutral)
        keyboard.add_hotkey('+', self.increase_sensitivity)
        keyboard.add_hotkey('-', self.decrease_sensitivity)
        keyboard.add_hotkey('c', self.center_cursor)
        keyboard.add_hotkey('esc', self.stop)
        print("[RelativeControl] Keyboard shortcuts enabled")
    
    def toggle_pause(self):
        """Toggle pause state."""
        self.is_paused = not self.is_paused
        status = "PAUSED" if self.is_paused else "RESUMED"
        print(f"[RelativeControl] Tracking {status}")
    
    def reset_neutral(self):
        """Reset neutral position to current eye position."""
        self.gaze_mapper.reset_neutral()
        print("[RelativeControl] Neutral position reset")
    
    def increase_sensitivity(self):
        """Increase cursor sensitivity."""
        self.gaze_mapper.adjust_sensitivity(1.2, 1.2)
        print("[RelativeControl] Sensitivity increased")
    
    def decrease_sensitivity(self):
        """Decrease cursor sensitivity."""
        self.gaze_mapper.adjust_sensitivity(0.8, 0.8)
        print("[RelativeControl] Sensitivity decreased")
    
    def center_cursor(self):
        """Move cursor to screen center."""
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        pyautogui.moveTo(center_x, center_y, duration=0.1)
        print(f"[RelativeControl] Cursor centered: ({center_x}, {center_y})")
    
    def start(self):
        """Start the relative gaze control system."""
        if self.is_running:
            print("[RelativeControl] Already running")
            return
        
        print("[RelativeControl] Starting relative gaze control...")
        print("[RelativeControl] Look straight ahead to set neutral position")
        print("[RelativeControl] Then move your eyes to control the cursor")
        
        # Reset metrics
        self.start_time = time.time()
        self.frame_count = 0
        self.metrics = {
            'total_frames': 0,
            'tracked_frames': 0,
            'lost_frames': 0,
            'cursor_updates': 0,
            'avg_fps': 0.0,
            'tracking_accuracy': 0.0
        }
        
        self.is_running = True
        print("[RelativeControl] System started!")
    
    def stop(self):
        """Stop the system."""
        if not self.is_running:
            return
        
        print("\n[RelativeControl] Stopping system...")
        self.is_running = False
        
        # Release resources
        self.camera.stop_recording()
        self.detector.release()
        
        # Calculate final metrics
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.metrics['avg_fps'] = self.frame_count / elapsed
            if self.metrics['total_frames'] > 0:
                self.metrics['tracking_accuracy'] = (
                    self.metrics['tracked_frames'] / self.metrics['total_frames'] * 100
                )
        
        print("[RelativeControl] System stopped")
        print(f"[RelativeControl] Final metrics:")
        print(f"  Total frames: {self.metrics['total_frames']}")
        print(f"  Tracked frames: {self.metrics['tracked_frames']} ({self.metrics['tracking_accuracy']:.1f}%)")
        print(f"  Cursor updates: {self.metrics['cursor_updates']}")
        print(f"  Average FPS: {self.metrics['avg_fps']:.1f}")
        
        # Close any OpenCV windows
        cv2.destroyAllWindows()
    
    def get_face_center(self, frame, face_landmarks):
        """Extract face center from landmarks."""
        if not face_landmarks:
            return None
        
        img_height, img_width, _ = frame.shape
        
        try:
            # Use nose tip and eye centers for better stabilization
            key_indices = [1, 4, 6, 168, 197]  # Nose and central points
            
            points = []
            for idx in key_indices:
                if idx < len(face_landmarks.landmark):
                    lm = face_landmarks.landmark[idx]
                    x = int(lm.x * img_width)
                    y = int(lm.y * img_height)
                    points.append((x, y))
            
            if points:
                center_x = np.mean([p[0] for p in points])
                center_y = np.mean([p[1] for p in points])
                return (center_x, center_y)
        except Exception as e:
            print(f"[FaceCenter] Error: {e}")
        
        return None
    
    def update(self):
        """
        Update gaze tracking and cursor position.
        Returns True if successful, False otherwise.
        """
        if not self.is_running or self.is_paused:
            return False
        
        frame_start = time.time()
        self.frame_count += 1
        self.metrics['total_frames'] += 1
        
        # Get camera frame
        frame = self.camera.get_frame()
        if frame is None:
            self.metrics['lost_frames'] += 1
            return False
        
        # Process frame for face mesh
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.detector.results = self.detector.face_mesh.process(img_rgb)
        
        iris_data = None
        face_center = None
        
        if self.detector.results.multi_face_landmarks:
            # Get iris centers
            iris_data = self.detector.get_iris_centers(frame)
            
            # Get face center for stabilization
            if self.enable_head_stabilization:
                face_center = self.get_face_center(frame, 
                                                  self.detector.results.multi_face_landmarks[0])
        
        # Check if we have iris data
        if iris_data and iris_data.get('left') and iris_data.get('right'):
            self.metrics['tracked_frames'] += 1
            
            # Update gaze mapper with new eye positions
            self.gaze_mapper.update_eye_position(
                left_eye=iris_data['left'],
                right_eye=iris_data['right'],
                face_center=face_center
            )
            
            # Get current cursor position
            current_cursor = pyautogui.position()
            
            # Calculate new cursor position
            new_x, new_y = self.gaze_mapper.calculate_cursor_position(
                screen_width=self.screen_width,
                screen_height=self.screen_height,
                current_cursor_pos=current_cursor
            )
            
            # Move cursor if position changed
            if new_x != current_cursor[0] or new_y != current_cursor[1]:
                pyautogui.moveTo(new_x, new_y, duration=0)
                self.metrics['cursor_updates'] += 1
            
            # Show camera feed with overlays if enabled
            if self.show_camera_feed:
                self._display_camera_feed(frame, iris_data, face_center)
            
            # Update FPS
            elapsed_total = time.time() - self.start_time
            if elapsed_total > 0:
                self.metrics['avg_fps'] = self.frame_count / elapsed_total
            
            return True
        else:
            self.metrics['lost_frames'] += 1
            return False
    
    def _display_camera_feed(self, frame, iris_data, face_center):
        """Display camera feed with gaze overlays."""
        display_frame = frame.copy()
        
        # Draw iris centers
        if iris_data:
            self.detector.draw_iris_centers(display_frame, iris_data, 
                                           with_text=True, size=5)
        
        # Draw face center
        if face_center:
            cv2.circle(display_frame, (int(face_center[0]), int(face_center[1])),
                      8, (0, 255, 255), 2)  # Yellow circle for face center
        
        # Draw neutral position marker
        mapper_status = self.gaze_mapper.get_status()
        if mapper_status.get('neutral_position'):
            neutral = mapper_status['neutral_position']
            # Convert to frame coordinates (assuming 640x480 camera)
            # This is approximate - adjust based on your camera resolution
            cam_width, cam_height = self.camera.get_resolution()
            neutral_x = int(neutral[0] * cam_width / self.screen_width)
            neutral_y = int(neutral[1] * cam_height / self.screen_height)
            cv2.circle(display_frame, (neutral_x, neutral_y),
                      10, (0, 255, 0), 2)  # Green circle for neutral
        
        # Add debug info
        if self.show_debug_info:
            info = [
                f"FPS: {self.metrics['avg_fps']:.1f}",
                f"Tracked: {self.metrics['tracked_frames']}/{self.metrics['total_frames']}",
                f"Mode: {mapper_status.get('mode', 'N/A')}",
                f"Sensitivity: {mapper_status.get('sensitivity', (0, 0))[0]:.2f}",
            ]
            
            for i, text in enumerate(info):
                cv2.putText(display_frame, text, (10, 30 + i*30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("Gaze Tracking", display_frame)
        cv2.waitKey(1)
    
    def run_continuous(self, target_fps: float = 60.0):
        """Run continuous gaze control."""
        self.start()
        
        print("[RelativeControl] Starting continuous control...")
        print("[RelativeControl] Move your eyes to control the cursor!")
        
        frame_time = 1.0 / target_fps
        last_status_time = time.time()
        status_interval = 2.0
        
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
                    cursor_pos = pyautogui.position()
                    mapper_status = self.gaze_mapper.get_status()
                    
                    print(f"[Status] FPS: {fps:.1f} | "
                          f"Accuracy: {accuracy:.1f}% | "
                          f"Cursor: {cursor_pos} | "
                          f"Mode: {mapper_status.get('mode', 'N/A')} | "
                          f"Sens: {mapper_status.get('sensitivity', (0, 0))[0]:.2f}")
                    last_status_time = current_time
                
                # Maintain target FPS
                processing_time = time.time() - frame_start
                sleep_time = max(0, frame_time - processing_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n[RelativeControl] Interrupted by user")
        except Exception as e:
            print(f"\n[RelativeControl] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()


def main():
    """Main function to run relative gaze cursor control."""
    
    print("=" * 70)
    print("RELATIVE GAZE CURSOR CONTROL")
    print("=" * 70)
    print("A new approach to eye-controlled cursor movement")
    print("=" * 70)
    print("Key features:")
    print("  1. Relative eye movement mapping (not absolute)")
    print("  2. Head stabilization for better accuracy")
    print("  3. Dynamic sensitivity adjustment")
    print("  4. Hybrid mode (combines relative + absolute)")
    print("=" * 70)
    
    # Check if enhanced calibration exists
    calib_path = "enhanced_calibration.pkl"
    if not os.path.exists(calib_path):
        print(f"ERROR: Enhanced calibration not found at {calib_path}")
        print("Please run enhanced_calibration.py first")
        sys.exit(1)
    
    # Check dependencies
    if not PYAUTOGUI_AVAILABLE:
        print("ERROR: pyautogui is required")
        sys.exit(1)
    
    # Disable pyautogui failsafe
    pyautogui.FAILSAFE = False
    
    # Countdown
    print("\nStarting in 3 seconds...")
    print("Look straight ahead at the camera")
    print("After starting, move your eyes to control the cursor")
    
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    print("GO!")
    print("=" * 70)
    
    # Create and run controller
    controller = RelativeGazeCursorControl(
        calibration_path=calib_path,
        camera_id=1,  # Based on your setup
        use_hybrid=True,  # Try hybrid mode first
        enable_head_stabilization=True,
        show_camera_feed=False,  # Set to True for debugging
        show_debug_info=True
    )
    
    # Run continuous control
    controller.run_continuous(target_fps=60.0)


if __name__ == "__main__":
    # Add current directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Run main function
    main()