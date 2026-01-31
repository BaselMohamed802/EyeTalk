"""
Filename: head_tracking_control.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 1/30/2026

Description:
    Head tracking mouse control system using custom modules.
    Recreates the exact functionality of the original code.
"""

import cv2
import numpy as np
import time
import keyboard
from collections import deque

# Import your modules
from VisionModule.camera import IrisCamera, print_camera_info
from VisionModule.face_utils import FaceMeshDetector
from VisionModule.head_tracking import HeadTracker
from PyautoguiController.mouse_controller import MouseController

# Constants from original code
FACE_OUTLINE_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]

# Head tracking parameters (from original code)
YAW_RANGE = 20  # degrees
PITCH_RANGE = 10  # degrees
FILTER_LENGTH = 8

def main():
    print("Head Tracking Mouse Control System")
    print("==================================")
    
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
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize head tracker
    head_tracker = HeadTracker(filter_length=FILTER_LENGTH)
    
    # Initialize mouse controller
    mouse_controller = MouseController(update_interval=0.01, enable_failsafe=False)
    mouse_controller.start()
    
    print("\nControls:")
    print("  F7: Toggle mouse control")
    print("  C: Calibrate (set current head position as center)")
    print("  Q: Quit")
    print("\nStarting tracking...")
    
    # Mouse control state
    mouse_control_enabled = True
    
    # For storing raw angles before calibration
    raw_yaw = 0
    raw_pitch = 0
    
    try:
        while camera.is_opened():
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
                    
                    # DEBUG: Print raw angles
                    # print(f"Raw angles: yaw={yaw_deg:.1f}, pitch={pitch_deg:.1f}")
                    
                    # Convert to 360-degree range
                    yaw_360, pitch_360 = head_tracker.convert_angles_to_360(yaw_deg, pitch_deg)
                    
                    # Store raw angles for calibration
                    raw_yaw = yaw_360
                    raw_pitch = pitch_360
                    
                    # DEBUG: Print converted angles
                    # print(f"Converted: yaw={yaw_360:.1f}, pitch={pitch_360:.1f}")
                    
                    # Map to screen coordinates
                    screen_width, screen_height = mouse_controller.get_screen_size()
                    screen_x, screen_y = head_tracker.map_to_screen(
                        yaw_360, pitch_360, 
                        screen_width, screen_height,
                        YAW_RANGE, PITCH_RANGE
                    )
                    
                    # DEBUG: Print screen coordinates
                    # print(f"Screen before clamp: ({screen_x}, {screen_y})")
                    
                    # Clamp to screen
                    screen_x, screen_y = head_tracker.clamp_to_screen(
                        screen_x, screen_y, screen_width, screen_height
                    )
                    
                    # DEBUG: Print final coordinates
                    # print(f"Screen final: ({screen_x}, {screen_y})")
                    
                    # Update mouse position if enabled
                    if mouse_control_enabled:
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
                    cv2.putText(frame, f"Screen: ({screen_x}, {screen_y})", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
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
                head_tracker.calibrate(raw_yaw, raw_pitch)
                time.sleep(0.3)  # Debounce
            
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