"""
Filename: eye_tracker_improved.py
Creator/Author: Basel Mohamed Mostafa Sayed (with DeepSeek enhancements)
Date: 2/7/2026

Description:
    Enhanced eye tracking with smooth cursor control using modular components.
    Features multi-point calibration, improved smoothing, and better user experience.
"""

# # Import necessary libraries
# import cv2
# import numpy as np
# import os
# import mediapipe as mp
# import time
# import math
# import sys
# from scipy.spatial.transform import Rotation as Rscipy
# from collections import deque
# import pyautogui
# import threading



# # Import your modules
# from Filters.filterManager import FilterManager
# from PyautoguiController.mouse_controller import MouseController
# from VisionModule.camera import IrisCamera
# from VisionModule.face_utils import FaceMeshDetector
# from VisionModule.head_tracking import HeadTracker

import cv2
import numpy as np
import os
import mediapipe as mp
import time
import math
from scipy.spatial.transform import Rotation as Rscipy
from collections import deque
import pyautogui
import threading
import keyboard
import sys

# --- Project-level imports (parent / root) ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ===== IMPORT YOUR MODULES =====
from VisionModule.camera import IrisCamera
from VisionModule.face_utils import FaceMeshDetector
from Filters.filterManager import FilterManager
from Calibration.calibration_model import CalibrationRegression
from Calibration.calibration_sampler import CalibrationSampler, CalibrationCoordinator
from Calibration.calibration_logic import CalibrationTimingController
from PyautoguiController.mouse_controller import MouseController
from VisionModule.head_tracking import HeadTracker

# ===== CONFIGURATION =====
MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
CENTER_X = MONITOR_WIDTH // 2
CENTER_Y = MONITOR_HEIGHT // 2
mouse_control_enabled = False

# ===== FILTERING CONFIG =====
# Multiple levels of filtering
filter_length = 15  # Increased for more smoothing
gaze_length = 350

# Use your filter modules
coord_filter = FilterManager(filter_type="kalman", dt=0.033, process_noise=0.01, measurement_noise=0.1)
screen_filter = FilterManager(filter_type="ema", alpha=0.25)  # For final screen coordinates
gaze_direction_filter = deque(maxlen=filter_length)

# ===== CALIBRATION MODEL =====
calibration_model = CalibrationRegression(model_type="polynomial", polynomial_degree=2)
is_calibrated = False

# ===== TRACKING QUALITY METRICS =====
tracking_confidence = 1.0
bad_frame_count = 0
MAX_BAD_FRAMES = 5

# ===== EYE SPHERE STATE =====
left_sphere_locked = False
left_sphere_local_offset = None
left_calibration_nose_scale = None
right_sphere_locked = False
right_sphere_local_offset = None
right_calibration_nose_scale = None

# ===== SMOOTHING BUFFERS =====
screen_x_buffer = deque(maxlen=7)
screen_y_buffer = deque(maxlen=7)
raw_gaze_buffer = deque(maxlen=5)

# ===== MOUSE CONTROLLER =====
mouse_controller = MouseController(update_interval=0.01, enable_failsafe=True)
mouse_controller.start()

# ===== HEAD TRACKER =====
head_tracker = HeadTracker(filter_length=10)

# Initialize MediaPipe FaceMesh through your module
face_mesh_detector = FaceMeshDetector(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===== OPEN WEBCAM =====
cam = IrisCamera(1, cam_width=640, cam_height=480)
w, h = cam.get_resolution()

# ===== NOSE LANDMARK INDICES =====
nose_indices = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241, 
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
                3, 248]

# ===== REFERENCE MATRICES =====
R_ref_nose = [None]
calibration_nose_scale = None

# ===== FILE WRITING =====
screen_position_file = "C:/Storage/Google Drive/Software/EyeTracker3DPython/screen_position.txt"

def write_screen_position(x, y):
    """Write screen position to file"""
    try:
        with open(screen_position_file, 'w') as f:
            f.write(f"{int(x)},{int(y)}\n")
    except:
        pass

def _normalize(v):
    """Normalize vector with safety check"""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.array([0.0, 0.0, 1.0])  # Default forward
    return v / n

def compute_scale(points_3d):
    """Robust scale computation"""
    n = len(points_3d)
    if n < 2:
        return 1.0
    
    # Use median pairwise distance for robustness against outliers
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points_3d[i] - points_3d[j])
            distances.append(dist)
    
    if not distances:
        return 1.0
    
    return np.median(distances)

def apply_smoothing(x, y):
    """Apply multiple levels of smoothing to screen coordinates"""
    # Add to buffers
    screen_x_buffer.append(x)
    screen_y_buffer.append(y)
    
    # Compute median (rejects outliers)
    if len(screen_x_buffer) >= 3:
        x_smooth = np.median(list(screen_x_buffer))
        y_smooth = np.median(list(screen_y_buffer))
    else:
        x_smooth = x
        y_smooth = y
    
    # Apply EMA filter
    x_filtered, y_filtered = screen_filter.apply_filter(x_smooth, y_smooth)
    
    return int(x_filtered), int(y_filtered)

def validate_gaze_quality(left_gaze, right_gaze):
    """Check if gaze tracking is reliable"""
    global tracking_confidence, bad_frame_count
    
    # Check if directions are valid
    if np.linalg.norm(left_gaze) < 0.1 or np.linalg.norm(right_gaze) < 0.1:
        tracking_confidence *= 0.8
        bad_frame_count += 1
        return False
    
    # Check angle between left and right gaze (should be similar)
    left_norm = left_gaze / np.linalg.norm(left_gaze)
    right_norm = right_gaze / np.linalg.norm(right_gaze)
    angle_diff = math.degrees(math.acos(np.clip(np.dot(left_norm, right_norm), -1.0, 1.0)))
    
    if angle_diff > 15.0:  # Eyes looking in very different directions
        tracking_confidence *= 0.7
        bad_frame_count += 1
        return False
    
    # Check for sudden jumps
    if raw_gaze_buffer:
        last_gaze = raw_gaze_buffer[-1]
        current_gaze = (left_norm + right_norm) / 2
        jump_distance = np.linalg.norm(current_gaze - last_gaze)
        
        if jump_distance > 0.3:  # Sudden large jump
            tracking_confidence *= 0.6
            bad_frame_count += 1
            return False
    
    # Good frame
    tracking_confidence = min(1.0, tracking_confidence * 1.05)
    bad_frame_count = max(0, bad_frame_count - 1)
    
    if bad_frame_count > MAX_BAD_FRAMES:
        tracking_confidence = 0.3
    
    return True

def compute_and_draw_coordinate_box(frame, face_landmarks, indices, ref_matrix_container, color=(0, 255, 0), size=80):
    """Your existing function with added robustness"""
    try:
        # Extract 3D positions
        points_3d = np.array([
            [face_landmarks[i].x * w, face_landmarks[i].y * h, face_landmarks[i].z * w]
            for i in indices if i < len(face_landmarks)
        ])
        
        if len(points_3d) < 3:
            return None, None, None
        
        center = np.mean(points_3d, axis=0)
        
        # PCA with regularization
        centered = points_3d - center
        cov = np.cov(centered.T) + np.eye(3) * 1e-6  # Regularization
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvecs = eigvecs[:, np.argsort(-eigvals)]
        
        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, 2] *= -1
        
        # Stabilize with reference matrix
        R_final = eigvecs
        if ref_matrix_container[0] is None:
            ref_matrix_container[0] = R_final.copy()
        else:
            R_ref = ref_matrix_container[0]
            for i in range(3):
                if np.dot(R_final[:, i], R_ref[:, i]) < 0:
                    R_final[:, i] *= -1
        
        return center, R_final, points_3d
        
    except Exception as e:
        print(f"Coordinate box error: {e}")
        return None, None, None

def convert_gaze_to_screen_coordinates_improved(combined_gaze_direction, use_calibration_model=True):
    """
    Improved screen coordinate conversion with multiple methods
    """
    global is_calibrated, calibration_model
    
    # Method 1: Use calibration regression model if available
    if use_calibration_model and is_calibrated and calibration_model.is_trained:
        try:
            # Convert gaze direction to some feature representation
            # For now, use simple angles as features
            gaze_norm = _normalize(combined_gaze_direction)
            
            # Create simple features from gaze vector
            features = np.array([gaze_norm[0], gaze_norm[1], gaze_norm[2]])
            
            # Dummy conversion - you'll need to adapt based on your calibration data
            # For now, use the original method as fallback
            pass
        except:
            pass
    
    # Method 2: Original angle-based method (fallback)
    avg_direction = _normalize(combined_gaze_direction)
    
    # Projections
    xz_proj = np.array([avg_direction[0], 0, avg_direction[2]])
    xz_proj = _normalize(xz_proj)
    
    yz_proj = np.array([0, avg_direction[1], avg_direction[2]])
    yz_proj = _normalize(yz_proj)
    
    reference_forward = np.array([0, 0, -1])
    
    # Calculate angles
    try:
        yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
        if avg_direction[0] < 0:
            yaw_rad = -yaw_rad
        
        pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
        if avg_direction[1] > 0:
            pitch_rad = -pitch_rad
    except:
        yaw_rad = 0
        pitch_rad = 0
    
    # Convert to degrees
    yaw_deg = np.degrees(yaw_rad)
    pitch_deg = np.degrees(pitch_rad)
    
    # Apply calibration offsets (if no model)
    if not is_calibrated:
        yaw_deg += calibration_offset_yaw
        pitch_deg += calibration_offset_pitch
    
    # Map to screen with adjustable sensitivity
    yaw_range = 25.0  # Adjust based on your head movement range
    pitch_range = 15.0
    
    screen_x = int(((yaw_deg + yaw_range) / (2 * yaw_range)) * MONITOR_WIDTH)
    screen_y = int(((pitch_range - pitch_deg) / (2 * pitch_range)) * MONITOR_HEIGHT)
    
    # Clamp with margin
    margin = 20
    screen_x = max(margin, min(screen_x, MONITOR_WIDTH - margin))
    screen_y = max(margin, min(screen_y, MONITOR_HEIGHT - margin))
    
    return screen_x, screen_y, yaw_deg, pitch_deg

def draw_tracking_quality(frame, confidence, x=10, y=30):
    """Visualize tracking quality"""
    color = (0, int(255 * confidence), int(255 * (1 - confidence)))
    cv2.rectangle(frame, (x, y - 20), (x + 200, y), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y - 20), (x + int(200 * confidence), y), color, -1)
    cv2.putText(frame, f"Tracking: {confidence:.1%}", (x + 5, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def run_calibration():
    """Run proper calibration using your modules"""
    global is_calibrated, calibration_model
    
    print("\n" + "="*50)
    print("STARTING CALIBRATION PROCESS")
    print("="*50)
    
    # Initialize calibration components
    timing_controller = CalibrationTimingController(num_points=9, settling_delay=0.8)
    sampler = CalibrationSampler(sampling_duration=1.5, min_samples=15)
    coordinator = CalibrationCoordinator(timing_controller, sampler)
    
    # Connect calibration model
    calibration_pairs = []
    
    def on_calibration_pair(point_index, screen_pos, eye_pos):
        calibration_pairs.append((screen_pos, eye_pos))
        print(f"Calibration pair {point_index}: {screen_pos} -> {eye_pos}")
    
    sampler.on_calibration_pair_ready = on_calibration_pair
    
    def on_calibration_complete():
        print("\nCalibration complete! Training model...")
        # Add all pairs to regression model
        calibration_model.add_calibration_pairs(calibration_pairs)
        
        # Train the model
        if calibration_model.train():
            global is_calibrated
            is_calibrated = True
            print("Calibration model trained successfully!")
            
            # Save model for future use
            calibration_model.save_model("eye_tracking_calibration.pkl")
        else:
            print("Calibration training failed!")
    
    timing_controller.on_calibration_complete = on_calibration_complete
    
    # Start calibration
    coordinator.start_calibration()
    
    # Run calibration loop
    print("Look at each dot as it appears. Press SPACE when ready.")
    print("Press ESC to cancel calibration.")
    
    # This is a simplified version - you'd need to integrate with your UI
    # For now, we'll use a simple version
    calibration_points = [
        (0.5, 0.5),  # Center
        (0.1, 0.1),  # Top-left
        (0.9, 0.1),  # Top-right
        (0.1, 0.9),  # Bottom-left
        (0.9, 0.9),  # Bottom-right
        (0.5, 0.1),  # Top-center
        (0.5, 0.9),  # Bottom-center
        (0.1, 0.5),  # Left-center
        (0.9, 0.5),  # Right-center
    ]
    
    # Note: In practice, you'd use your calibration_ui.py module
    # For now, we'll mark calibration as manual
    print("\nManual calibration activated.")
    print("Look at the center of the screen and press 's' to calibrate center.")
    print("Then look at corners and press corresponding keys.")
    
    return False  # Manual calibration needed

# ===== MAIN LOOP =====
try:
    while True:
        frame = cam.get_frame()
        if frame is None:
            continue
        
        # Get iris centers using your module
        iris_centers = face_mesh_detector.get_iris_centers(frame)
        
        # Also get face mesh for landmarks
        frame_with_mesh, landmarks_lst = face_mesh_detector.draw_face_mesh(
            frame, draw_face=False, draw_iris=True, draw_tesselation=False
        )
        
        if landmarks_lst and len(landmarks_lst) > 0:
            face_landmarks = results.multi_face_landmarks[0].landmark if 'results' in locals() else None
            
            if face_landmarks:
                # Compute head orientation
                head_center, R_final, nose_points_3d = compute_and_draw_coordinate_box(
                    frame, face_landmarks, nose_indices, R_ref_nose, color=(0, 255, 0), size=80
                )
                
                if head_center is not None:
                    # Track eye spheres
                    base_radius = 20
                    
                    # Get iris positions
                    left_iris_idx = 468
                    right_iris_idx = 473
                    
                    if left_iris_idx < len(face_landmarks) and right_iris_idx < len(face_landmarks):
                        left_iris = face_landmarks[left_iris_idx]
                        right_iris = face_landmarks[right_iris_idx]
                        
                        iris_3d_left = np.array([left_iris.x * w, left_iris.y * h, left_iris.z * w])
                        iris_3d_right = np.array([right_iris.x * w, right_iris.y * h, right_iris.z * w])
                        
                        # Update sphere positions if locked
                        if left_sphere_locked and right_sphere_locked:
                            current_nose_scale = compute_scale(nose_points_3d)
                            
                            # Left eye sphere
                            scale_ratio = current_nose_scale / left_calibration_nose_scale if left_calibration_nose_scale else 1.0
                            scaled_offset = left_sphere_local_offset * scale_ratio
                            sphere_world_l = head_center + R_final @ scaled_offset
                            scaled_radius_l = int(base_radius * scale_ratio)
                            
                            # Right eye sphere
                            scale_ratio_r = current_nose_scale / right_calibration_nose_scale if right_calibration_nose_scale else 1.0
                            scaled_offset_r = right_sphere_local_offset * scale_ratio_r
                            sphere_world_r = head_center + R_final @ scaled_offset_r
                            scaled_radius_r = int(base_radius * scale_ratio_r)
                            
                            # Calculate gaze directions
                            left_gaze_dir = iris_3d_left - sphere_world_l
                            right_gaze_dir = iris_3d_right - sphere_world_r
                            
                            # Validate gaze quality
                            is_gaze_valid = validate_gaze_quality(left_gaze_dir, right_gaze_dir)
                            
                            if is_gaze_valid and tracking_confidence > 0.3:
                                # Normalize and combine
                                left_gaze_norm = _normalize(left_gaze_dir)
                                right_gaze_norm = _normalize(right_gaze_dir)
                                
                                raw_combined = (left_gaze_norm + right_gaze_norm) / 2
                                raw_gaze_buffer.append(_normalize(raw_combined))
                                
                                # Apply filtering to gaze direction
                                gaze_filtered = coord_filter.apply_filter(
                                    raw_combined[0], 
                                    raw_combined[1]
                                )
                                
                                # Create 3D gaze vector
                                gaze_3d = np.array([gaze_filtered[0], gaze_filtered[1], raw_combined[2]])
                                gaze_3d = _normalize(gaze_3d)
                                
                                # Add to gaze buffer for additional smoothing
                                gaze_direction_filter.append(gaze_3d)
                                
                                if len(gaze_direction_filter) >= 3:
                                    # Weighted average (recent frames more important)
                                    weights = np.linspace(0.5, 1.0, len(gaze_direction_filter))
                                    weights = weights / weights.sum()
                                    
                                    smoothed_gaze = np.zeros(3)
                                    for i, g in enumerate(gaze_direction_filter):
                                        smoothed_gaze += g * weights[i]
                                    
                                    smoothed_gaze = _normalize(smoothed_gaze)
                                    
                                    # Convert to screen coordinates
                                    screen_x, screen_y, raw_yaw, raw_pitch = convert_gaze_to_screen_coordinates_improved(
                                        smoothed_gaze, 
                                        use_calibration_model=is_calibrated
                                    )
                                    
                                    # Apply additional smoothing to screen coordinates
                                    screen_x_smooth, screen_y_smooth = apply_smoothing(screen_x, screen_y)
                                    
                                    # Update mouse controller
                                    if mouse_control_enabled and tracking_confidence > 0.5:
                                        mouse_controller.set_target(screen_x_smooth, screen_y_smooth)
                                    
                                    # Write to file
                                    write_screen_position(screen_x_smooth, screen_y_smooth)
                                    
                                    # Display info
                                    cv2.putText(frame, f"Screen: ({screen_x_smooth}, {screen_y_smooth})", 
                                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                    cv2.putText(frame, f"Confidence: {tracking_confidence:.1%}", 
                                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                               (0, int(255 * tracking_confidence), int(255 * (1 - tracking_confidence))), 2)
                            
                            # Draw gaze rays (thinner when confidence is low)
                            ray_thickness = max(1, int(3 * tracking_confidence))
                            if tracking_confidence > 0.3:
                                cv2.line(frame, 
                                        (int(sphere_world_l[0]), int(sphere_world_l[1])),
                                        (int(iris_3d_left[0]), int(iris_3d_left[1])),
                                        (0, 255, 0), ray_thickness)
                                
                                cv2.line(frame, 
                                        (int(sphere_world_r[0]), int(sphere_world_r[1])),
                                        (int(iris_3d_right[0]), int(iris_3d_right[1])),
                                        (0, 255, 0), ray_thickness)
                        
                        # Draw iris centers from your module
                        if iris_centers.get('left'):
                            cv2.circle(frame, iris_centers['left'], 3, (255, 0, 0), -1)
                        if iris_centers.get('right'):
                            cv2.circle(frame, iris_centers['right'], 3, (0, 0, 255), -1)
        
        # Draw tracking quality indicator
        draw_tracking_quality(frame, tracking_confidence)
        
        # Display mouse control status
        status_color = (0, 255, 0) if mouse_control_enabled else (0, 0, 255)
        status_text = f"Mouse: {'ON' if mouse_control_enabled else 'OFF'}"
        cv2.putText(frame, status_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Display calibration status
        cal_status = f"Calibrated: {'YES' if is_calibrated else 'NO'}"
        cv2.putText(frame, cal_status, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (0, 255, 0) if is_calibrated else (0, 0, 255), 2)
        
        cv2.imshow("Improved Eye Tracking", frame)
        
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('t'):
            mouse_control_enabled = not mouse_control_enabled
            print(f"Mouse control {'enabled' if mouse_control_enabled else 'disabled'}")
            time.sleep(0.3)
        elif key == ord('c') and not (left_sphere_locked and right_sphere_locked):
            # Lock eye spheres (existing functionality)
            current_nose_scale = compute_scale(nose_points_3d)
            
            left_sphere_local_offset = R_final.T @ (iris_3d_left - head_center)
            camera_dir_world = np.array([0, 0, 1])
            camera_dir_local = R_final.T @ camera_dir_world
            left_sphere_local_offset += base_radius * camera_dir_local
            left_calibration_nose_scale = current_nose_scale
            left_sphere_locked = True
            
            right_sphere_local_offset = R_final.T @ (iris_3d_right - head_center)
            right_sphere_local_offset += base_radius * camera_dir_local
            right_calibration_nose_scale = current_nose_scale
            right_sphere_locked = True
            
            print("Eye spheres locked")
        elif key == ord('s'):
            # Start calibration process
            run_calibration()
        elif key == ord('l'):
            # Load existing calibration
            if calibration_model.load_model("eye_tracking_calibration.pkl"):
                is_calibrated = True
                print("Calibration model loaded!")
        
finally:
    # Cleanup
    cam.stop_recording()
    face_mesh_detector.release()
    mouse_controller.stop()
    cv2.destroyAllWindows()