"""
Filename: head_tracking_control.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 1/30/2026

Description:
    Head tracking mouse control system using custom modules.
    Recreates the exact functionality of the original code.
    Enhanced with thread safety, movement thresholds, config files,
    enhanced calibration, and smoothing profiles for reduced jitter.
"""

import cv2
import numpy as np
import time
import keyboard
import threading  # Added for thread safety
import yaml  # Added for configuration
from collections import deque

# Import your modules
from VisionModule.camera import IrisCamera, print_camera_info
from VisionModule.face_utils import FaceMeshDetector
from VisionModule.head_tracking import HeadTracker
from PyautoguiController.mouse_controller import MouseController


# ========== ADDED CLASSES ==========
class EnhancedCalibrator:
    def __init__(self, head_tracker):
        self.head_tracker = head_tracker
        self.calibration_samples = []
        self.is_calibrating = False
        
    def start_calibration(self, num_samples=30):
        """Start collecting calibration samples"""
        self.calibration_samples = []
        self.is_calibrating = True
        self.num_samples = num_samples
        print(f"Calibration: Look straight ahead for {num_samples/10} seconds...")
        
    def add_sample(self, yaw, pitch):
        """Add a calibration sample"""
        if self.is_calibrating and len(self.calibration_samples) < self.num_samples:
            self.calibration_samples.append((yaw, pitch))
            
            if len(self.calibration_samples) >= self.num_samples:
                self.finish_calibration()
                
    def finish_calibration(self):
        """Calculate calibration offsets from collected samples"""
        if not self.calibration_samples:
            return
            
        yaw_samples = [s[0] for s in self.calibration_samples]
        pitch_samples = [s[1] for s in self.calibration_samples]
        
        # Calculate median (more robust than mean for outliers)
        median_yaw = np.median(yaw_samples)
        median_pitch = np.median(pitch_samples)
        
        # Set calibration offsets
        self.head_tracker.calibrate(median_yaw, median_pitch)
        
        # Calculate calibration quality
        yaw_std = np.std(yaw_samples)
        pitch_std = np.std(pitch_samples)
        
        print(f"Calibration complete!")
        print(f"  Center: Yaw={median_yaw:.1f}°, Pitch={median_pitch:.1f}°")
        print(f"  Stability: Yaw_std={yaw_std:.2f}°, Pitch_std={pitch_std:.2f}°")
        
        if yaw_std > 1.0 or pitch_std > 1.0:
            print("  Warning: High variance detected. Try to hold your head still.")
            
        self.is_calibrating = False


class SmoothingProfile:
    def __init__(self, name, filter_length, movement_threshold, 
                 process_interval, deadzone_radius):
        self.name = name
        self.filter_length = filter_length
        self.movement_threshold = movement_threshold
        self.process_interval = process_interval
        self.deadzone_radius = deadzone_radius
        
class SmoothingProfiles:
    FAST = SmoothingProfile(
        name="Fast",
        filter_length=4,
        movement_threshold=0.3,
        process_interval=0.005,
        deadzone_radius=1.0
    )
    
    BALANCED = SmoothingProfile(
        name="Balanced",
        filter_length=8,
        movement_threshold=0.5,
        process_interval=0.01,
        deadzone_radius=2.0
    )
    
    SMOOTH = SmoothingProfile(
        name="Smooth",
        filter_length=12,
        movement_threshold=0.8,
        process_interval=0.02,
        deadzone_radius=3.0
    )
    
    @classmethod
    def get_all_profiles(cls):
        return [cls.FAST, cls.BALANCED, cls.SMOOTH]
    
    @classmethod
    def cycle_profile(cls, current_profile):
        profiles = cls.get_all_profiles()
        current_index = profiles.index(current_profile) if current_profile in profiles else 0
        next_index = (current_index + 1) % len(profiles)
        return profiles[next_index]
# ====================================


# Load configuration
def load_config():
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("config.yaml not found, using defaults")
        config = {
            'tracking': {
                'yaw_range': 20,
                'pitch_range': 10,
                'filter_length': 8,
                'min_detection_confidence': 0.5,
                'min_tracking_confidence': 0.5
            },
            'smoothing': {
                'movement_threshold': 0.5,
                'process_interval': 0.01,
                'deadzone_radius': 2.0
            },
            'mouse': {
                'update_interval': 0.01
            },
            'display': {
                'show_landmarks': True,
                'show_cube': True,
                'show_gaze_ray': True
            }
        }
    return config


FACE_OUTLINE_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]


def apply_deadzone(x, y, last_x, last_y, deadzone_radius):
    """Apply deadzone to filter out tiny movements"""
    if last_x is None or last_y is None:
        return x, y
    
    distance = np.sqrt((x - last_x)**2 + (y - last_y)**2)
    if distance < deadzone_radius:
        return last_x, last_y  # Return previous position
    return x, y


def main():
    print("Head Tracking Mouse Control System")
    print("==================================")
    
    # Load configuration
    config = load_config()
    
    # Extract values from config
    YAW_RANGE = config['tracking']['yaw_range']
    PITCH_RANGE = config['tracking']['pitch_range']
    FILTER_LENGTH = config['tracking']['filter_length']
    MIN_DETECTION_CONFIDENCE = config['tracking']['min_detection_confidence']
    MIN_TRACKING_CONFIDENCE = config['tracking']['min_tracking_confidence']
    
    # Initialize with balanced profile
    current_profile = SmoothingProfiles.BALANCED
    
    # Extract values from profile
    MOVEMENT_THRESHOLD = current_profile.movement_threshold
    PROCESS_INTERVAL = current_profile.process_interval
    DEADZONE_RADIUS = current_profile.deadzone_radius
    
    # 1. List available cameras
    print("\nScanning for cameras...")
    cameras = print_camera_info()
    
    if not cameras:
        print("No cameras found. Please connect a camera and try again.")
        return
    
    # 2. Select camera
    camera_ports = list(cameras.keys())
    if len(camera_ports) > 1:
        print(f"\nAvailable cameras: {camera_ports}")
        cam_id = int(input(f"Select camera port (default {camera_ports[0]}): ") or camera_ports[0])
    else:
        cam_id = camera_ports[0]
        print(f"\nUsing camera at port {cam_id}")
    
    # 3. Initialize components
    print("\nInitializing components...")
    
    # Initialize camera
    try:
        camera = IrisCamera(cam_id, cam_width=640, cam_height=480)
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return
    
    # Initialize face mesh detector
    face_detector = FaceMeshDetector(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )
    
    # Initialize head tracker
    head_tracker = HeadTracker(filter_length=FILTER_LENGTH)
    
    # Initialize enhanced calibrator
    calibrator = EnhancedCalibrator(head_tracker)
    
    # Initialize mouse controller
    mouse_controller = MouseController(update_interval=config['mouse']['update_interval'], 
                                      enable_failsafe=False)
    mouse_controller.start()
    
    # Thread lock for mouse position updates
    mouse_lock = threading.Lock()
    
    print("\nControls:")
    print("  F7: Toggle mouse control")
    print("  C: Calibrate (collect 30 samples for accurate center)")
    print("  P: Cycle smoothing profiles (Fast/Balanced/Smooth)")
    print("  Q: Quit")
    print(f"\nStarting tracking with {current_profile.name} profile...")
    
    # Movement tracking variables
    mouse_control_enabled = True
    last_screen_x = None
    last_screen_y = None
    last_processed_time = time.time()
    
    # For storing raw angles before calibration
    raw_yaw = 0
    raw_pitch = 0
    
    try:
        while camera.is_opened():
            current_time = time.time()
            
            # Skip processing if too soon (frame rate limiting)
            if current_time - last_processed_time < PROCESS_INTERVAL:
                time.sleep(0.001)  # Small sleep to prevent CPU hog
                continue
                
            last_processed_time = current_time
            
            # Get frame
            frame = camera.get_frame()
            if frame is None:
                continue
            
            # Get camera resolution
            cam_width, cam_height = camera.get_resolution()
            
            # Get face landmarks with 3D coordinates
            landmarks_frame, landmarks_list_3d = face_detector.get_landmarks_as_pixels(frame)
            
            if landmarks_list_3d:
                # Convert landmarks to dictionary for easier access
                landmarks_dict = {}
                for idx, x, y, z in landmarks_list_3d:
                    landmarks_dict[idx] = (x, y, z)
                
                # Create landmark visualization (like original code)
                for idx, x, y, z in landmarks_list_3d:
                    # Color coding like original code
                    if idx in FACE_OUTLINE_INDICES:
                        color = (155, 155, 155)  # Gray for outline
                    else:
                        color = (255, 25, 10)    # Red for other landmarks
                    
                    cv2.circle(landmarks_frame, (x, y), 3, color, -1)
                    cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)
                
                # Get key points for head tracking
                key_points_indices = {
                    "left": 234,
                    "right": 454,
                    "top": 10,
                    "bottom": 152,
                    "front": 1
                }
                
                # Check if we have all required key points
                missing_points = [name for name, idx in key_points_indices.items() 
                                if idx not in landmarks_dict]
                
                if not missing_points:
                    # Extract key points with 3D coordinates
                    key_points_3d = {}
                    for name, idx in key_points_indices.items():
                        x, y, z = landmarks_dict[idx]
                        key_points_3d[name] = np.array([float(x), float(y), float(z)])
                    
                    # Calculate head orientation
                    orientation = head_tracker.calculate_head_orientation(key_points_3d)
                    
                    # Update smoothing buffers
                    head_tracker.update_smoothing_buffers(
                        orientation['center'],
                        orientation['forward_axis']
                    )
                    
                    # Get smoothed orientation
                    smoothed_origin, smoothed_direction = head_tracker.get_smoothed_orientation()
                    
                    # Calculate angles
                    yaw_deg, pitch_deg = head_tracker.calculate_yaw_pitch(smoothed_direction)
                    
                    # Convert to 360-degree range
                    yaw_360, pitch_360 = head_tracker.convert_angles_to_360(yaw_deg, pitch_deg)
                    
                    # Store raw angles for calibration
                    raw_yaw = yaw_360
                    raw_pitch = pitch_360
                    
                    # Add to calibration if calibrating
                    if calibrator.is_calibrating:
                        calibrator.add_sample(raw_yaw, raw_pitch)
                    
                    # Map to screen coordinates
                    screen_width, screen_height = mouse_controller.get_screen_size()
                    screen_x, screen_y = head_tracker.map_to_screen(
                        yaw_360, pitch_360, 
                        screen_width, screen_height,
                        YAW_RANGE, PITCH_RANGE
                    )
                    
                    # Clamp to screen
                    screen_x, screen_y = head_tracker.clamp_to_screen(
                        screen_x, screen_y, screen_width, screen_height
                    )
                    
                    # Apply deadzone filter
                    screen_x, screen_y = apply_deadzone(
                        screen_x, screen_y, 
                        last_screen_x, last_screen_y, 
                        DEADZONE_RADIUS
                    )
                    
                    # Check if we should update mouse position
                    should_update = False
                    if last_screen_x is None or last_screen_y is None:
                        should_update = True
                    else:
                        # Calculate distance moved
                        distance = np.sqrt((screen_x - last_screen_x)**2 + (screen_y - last_screen_y)**2)
                        
                        # Only update if movement exceeds threshold
                        if distance > MOVEMENT_THRESHOLD:
                            should_update = True
                    
                    # Update mouse position if enabled
                    if mouse_control_enabled and should_update:
                        # Store last position
                        last_screen_x = screen_x
                        last_screen_y = screen_y
                        
                        # Thread-safe update
                        with mouse_lock:
                            mouse_controller.set_target(screen_x, screen_y)
                    
                    # Generate and draw head cube
                    cube_corners = head_tracker.generate_head_cube(
                        orientation['center'],
                        orientation['right_axis'],
                        orientation['up_axis'],
                        orientation['forward_axis'],
                        orientation['half_width'],
                        orientation['half_height']
                    )
                    
                    # Draw cube on frame
                    face_detector.draw_head_cube(frame, cube_corners)
                    
                    # Draw gaze ray (convert 3D to 2D for drawing)
                    ray_origin_2d = smoothed_origin[:2]  # Just x, y
                    ray_dir_2d = smoothed_direction[:2]  # Just x, y
                    
                    # Normalize 2D direction for consistent ray length
                    ray_dir_2d_norm = np.linalg.norm(ray_dir_2d)
                    if ray_dir_2d_norm > 0:
                        ray_dir_2d = ray_dir_2d / ray_dir_2d_norm
                    
                    # Draw on both frames
                    face_detector.draw_simple_gaze_ray(frame, ray_origin_2d, ray_dir_2d, length=200)
                    face_detector.draw_simple_gaze_ray(landmarks_frame, ray_origin_2d, ray_dir_2d, length=200)
                    
                    # Draw key points in pink (like original code)
                    for name, idx in key_points_indices.items():
                        if idx in landmarks_dict:
                            x, y, z = landmarks_dict[idx]
                            cv2.circle(frame, (x, y), 4, (0, 0, 0), -1)  # Black border
                            cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)  # Pink center
                    
                    # Display info
                    cv2.putText(frame, f"Yaw: {yaw_360:.1f}°, Pitch: {pitch_360:.1f}°", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Mouse: {'ON' if mouse_control_enabled else 'OFF'}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                               (0, 255, 0) if mouse_control_enabled else (0, 0, 255), 2)
                    cv2.putText(frame, f"Profile: {current_profile.name}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Screen: ({screen_x:.0f}, {screen_y:.0f})", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show FPS
            fps = camera.get_frame_rate()
            cv2.putText(frame, f"FPS: {fps}", (10, cam_height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display windows
            cv2.imshow("Head Tracking Mouse Control", frame)
            cv2.imshow("Facial Landmarks", landmarks_frame)
            
            # Handle keyboard input
            if keyboard.is_pressed('f7'):
                mouse_control_enabled = not mouse_control_enabled
                status = "Enabled" if mouse_control_enabled else "Disabled"
                print(f"[Mouse Control] {status}")
                time.sleep(0.3)  # Debounce
            
            if keyboard.is_pressed('c'):
                if not calibrator.is_calibrating:
                    calibrator.start_calibration(num_samples=30)
                time.sleep(0.3)  # Debounce
            
            if keyboard.is_pressed('p'):  # Press 'P' to cycle profiles
                current_profile = SmoothingProfiles.cycle_profile(current_profile)
                
                # Update settings
                FILTER_LENGTH = current_profile.filter_length
                MOVEMENT_THRESHOLD = current_profile.movement_threshold
                PROCESS_INTERVAL = current_profile.process_interval
                DEADZONE_RADIUS = current_profile.deadzone_radius
                
                # Reinitialize head tracker with new filter length
                head_tracker = HeadTracker(filter_length=FILTER_LENGTH)
                
                print(f"[Profile] Switched to {current_profile.name}")
                time.sleep(0.3)
            
            # Check for 'q' key or window close
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty("Head Tracking Mouse Control", cv2.WND_PROP_VISIBLE) < 1:
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Cleanup
        print("\nShutting down...")
        camera.stop_recording()
        face_detector.release()
        mouse_controller.stop()
        cv2.destroyAllWindows()
        print("Cleanup complete.")


if __name__ == "__main__":
    main()