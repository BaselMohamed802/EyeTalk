"""
Filename: head_tracking_app.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: [Current Date]

Description:
    Head tracking mouse control - simplified version based on original code.
"""

import cv2
import numpy as np
import math
import keyboard
import time
import pyautogui
import threading
from collections import deque
from VisionModule.camera import IrisCamera
from VisionModule.face_utils import FaceMeshDetector

# Disable fail-safe for now
pyautogui.FAILSAFE = False

# Get screen info
MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
CENTER_X = MONITOR_WIDTH // 2
CENTER_Y = MONITOR_HEIGHT // 2

# Mouse control thread
mouse_target = [CENTER_X, CENTER_Y]
mouse_lock = threading.Lock()
mouse_enabled = True
filter_length = 8

# Face outline indices for drawing
FACE_OUTLINE_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]

# Key landmark indices
LANDMARKS = {
    "left": 234,
    "right": 454,
    "top": 10,
    "bottom": 152,
    "front": 1,
}

# Calibration
calibration_offset_yaw = 0
calibration_offset_pitch = 0

# Smoothing buffers
ray_origins = deque(maxlen=filter_length)
ray_directions = deque(maxlen=filter_length)

def mouse_mover():
    """Thread for moving mouse cursor."""
    while True:
        if mouse_enabled:
            with mouse_lock:
                x, y = mouse_target
            pyautogui.moveTo(x, y)
        time.sleep(0.01)

def landmark_to_np(landmark, w, h):
    """Convert MediaPipe landmark to numpy array."""
    return np.array([landmark.x * w, landmark.y * h, landmark.z * w])

def main():
    global mouse_target, mouse_enabled, calibration_offset_yaw, calibration_offset_pitch
    
    print("Starting Head Tracking Mouse Control...")
    print(f"Screen: {MONITOR_WIDTH}x{MONITOR_HEIGHT}")
    print("Controls: F7=toggle mouse, C=calibrate, Q=quit")
    
    # Initialize camera
    camera = IrisCamera(1, 640, 480)
    
    # Initialize face mesh detector
    face_detector = FaceMeshDetector(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Start mouse thread
    mouse_thread = threading.Thread(target=mouse_mover, daemon=True)
    mouse_thread.start()
    
    # Main loop
    frame_count = 0
    
    while camera.is_opened():
        # Get frame
        frame = camera.get_frame()
        if frame is None:
            continue
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            
            # Create blank frame for landmarks
            landmarks_frame = np.zeros_like(frame)
            
            # Draw all landmarks
            for i, landmark in enumerate(face_landmarks):
                pt = landmark_to_np(landmark, w, h)
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < w and 0 <= y < h:
                    color = (155, 155, 155) if i in FACE_OUTLINE_INDICES else (255, 25, 10)
                    cv2.circle(landmarks_frame, (x, y), 3, color, -1)
            
            # Extract key points
            key_points = {}
            for name, idx in LANDMARKS.items():
                pt = landmark_to_np(face_landmarks[idx], w, h)
                key_points[name] = pt
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(frame, (x, y), 4, (0, 0, 0), -1)
            
            # Get points
            left = key_points["left"]
            right = key_points["right"]
            top = key_points["top"]
            bottom = key_points["bottom"]
            front = key_points["front"]
            
            # Calculate axes
            right_axis = (right - left)
            right_norm = np.linalg.norm(right_axis)
            if right_norm > 0:
                right_axis = right_axis / right_norm
            
            up_axis = (top - bottom)
            up_norm = np.linalg.norm(up_axis)
            if up_norm > 0:
                up_axis = up_axis / up_norm
            
            # Forward axis
            forward_axis = np.cross(right_axis, up_axis)
            forward_norm = np.linalg.norm(forward_axis)
            if forward_norm > 0:
                forward_axis = forward_axis / forward_norm
            
            forward_axis = -forward_axis  # Flip to face forward
            
            # Center
            center = (left + right + top + bottom + front) / 5.0
            
            # Update smoothing buffers
            ray_origins.append(center)
            ray_directions.append(forward_axis)
            
            # Get smoothed values
            if ray_origins:
                avg_origin = np.mean(list(ray_origins), axis=0)
                avg_direction = np.mean(list(ray_directions), axis=0)
                avg_norm = np.linalg.norm(avg_direction)
                if avg_norm > 0:
                    avg_direction = avg_direction / avg_norm
            
            # Calculate angles
            reference_forward = np.array([0, 0, -1])
            
            # Yaw
            xz_proj = np.array([avg_direction[0], 0, avg_direction[2]])
            if np.linalg.norm(xz_proj) > 0:
                xz_proj = xz_proj / np.linalg.norm(xz_proj)
                yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
                if avg_direction[0] < 0:
                    yaw_rad = -yaw_rad
            else:
                yaw_rad = 0
            
            # Pitch
            yz_proj = np.array([0, avg_direction[1], avg_direction[2]])
            if np.linalg.norm(yz_proj) > 0:
                yz_proj = yz_proj / np.linalg.norm(yz_proj)
                pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
                if avg_direction[1] > 0:
                    pitch_rad = -pitch_rad
            else:
                pitch_rad = 0
            
            # Convert to degrees
            yaw_deg = math.degrees(yaw_rad)
            pitch_deg = math.degrees(pitch_rad)
            
            # Convert to 0-360 range
            if yaw_deg < 0:
                yaw_deg = abs(yaw_deg)
            elif yaw_deg < 180:
                yaw_deg = 360 - yaw_deg
            
            if pitch_deg < 0:
                pitch_deg = 360 + pitch_deg
            
            raw_yaw_deg = yaw_deg
            raw_pitch_deg = pitch_deg
            
            # Apply calibration
            yaw_deg += calibration_offset_yaw
            pitch_deg += calibration_offset_pitch
            
            # Map to screen
            yawDegrees = 20
            pitchDegrees = 10
            
            screen_x = int(((yaw_deg - (180 - yawDegrees)) / (2 * yawDegrees)) * MONITOR_WIDTH)
            screen_y = int(((180 + pitchDegrees - pitch_deg) / (2 * pitchDegrees)) * MONITOR_HEIGHT)
            
            # Clamp to screen
            screen_x = max(10, min(screen_x, MONITOR_WIDTH - 10))
            screen_y = max(10, min(screen_y, MONITOR_HEIGHT - 10))
            
            # Update mouse target
            if mouse_enabled and frame_count > 5:
                with mouse_lock:
                    mouse_target[0] = screen_x
                    mouse_target[1] = screen_y
            
            # Draw head cube
            half_width = np.linalg.norm(right - left) / 2
            half_height = np.linalg.norm(top - bottom) / 2
            depth = 80
            
            # Generate cube corners
            def corner(x_sign, y_sign, z_sign):
                return (center + 
                        x_sign * half_width * right_axis +
                        y_sign * half_height * up_axis +
                        z_sign * depth * forward_axis)
            
            cube_corners = [
                corner(-1, 1, -1), corner(1, 1, -1),
                corner(1, -1, -1), corner(-1, -1, -1),
                corner(-1, 1, 1), corner(1, 1, 1),
                corner(1, -1, 1), corner(-1, -1, 1)
            ]
            
            # Draw cube wireframe
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]
            
            for i, j in edges:
                start = (int(cube_corners[i][0]), int(cube_corners[i][1]))
                end = (int(cube_corners[j][0]), int(cube_corners[j][1]))
                cv2.line(frame, start, end, (255, 125, 35), 2)
            
            # Draw gaze ray
            ray_length = 2.5 * depth
            ray_end = avg_origin - avg_direction * ray_length
            cv2.line(frame, 
                    (int(avg_origin[0]), int(avg_origin[1])),
                    (int(ray_end[0]), int(ray_end[1])),
                    (15, 255, 0), 3)
            
            # Draw gaze ray on landmarks frame too
            cv2.line(landmarks_frame,
                    (int(avg_origin[0]), int(avg_origin[1])),
                    (int(ray_end[0]), int(ray_end[1])),
                    (15, 255, 0), 3)
            
            # Display info
            cv2.putText(frame, f"Yaw: {yaw_deg:.1f}°, Pitch: {pitch_deg:.1f}°",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Mouse: {screen_x}, {screen_y}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display mouse status
        status = "ON" if mouse_enabled else "OFF"
        color = (0, 255, 0) if mouse_enabled else (0, 0, 255)
        cv2.putText(frame, f"Mouse: {status}", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display FPS
        fps = camera.get_frame_rate()
        cv2.putText(frame, f"FPS: {fps}", 
                   (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Show frames
        cv2.imshow("Head Tracking", frame)
        cv2.imshow("Facial Landmarks", landmarks_frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if keyboard.is_pressed('f7'):
            mouse_enabled = not mouse_enabled
            print(f"Mouse control: {'ENABLED' if mouse_enabled else 'DISABLED'}")
            time.sleep(0.3)
        
        elif key == ord('c'):
            calibration_offset_yaw = 180 - raw_yaw_deg
            calibration_offset_pitch = 180 - raw_pitch_deg
            print(f"Calibrated: Yaw offset={calibration_offset_yaw:.1f}, Pitch offset={calibration_offset_pitch:.1f}")
        
        elif key == ord('q'):
            break
    
    # Cleanup
    print("Cleaning up...")
    face_detector.release()
    camera.stop_recording()
    cv2.destroyAllWindows()
    print("Application closed")

if __name__ == "__main__":
    main()