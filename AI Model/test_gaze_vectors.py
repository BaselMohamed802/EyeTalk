"""
Filename: test_3d_gaze.py
Creator/Author: Basel Mohamed Mostafa Sayed + ChatGPT
Date: 1/27/2026

Purpose:
    Test 3D gaze vector calculation with MediaPipe Face Mesh.
    Understand the 3D coordinate system and landmark positions.
"""

import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Face Mesh with 3D landmarks
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Key landmark indices for gaze calculation
# MediaPipe provides 468 3D face landmarks
LEFT_EYE_INNER_CORNER = 33
LEFT_EYE_OUTER_CORNER = 133
RIGHT_EYE_INNER_CORNER = 362
RIGHT_EYE_OUTER_CORNER = 263
LEFT_IRIS_CENTER = 468  # When refine_landmarks=True
RIGHT_IRIS_CENTER = 473

# Face center/reference points
NOSE_TIP = 1
FOREHEAD_CENTER = 10
CHIN = 152

def test_3d_landmarks():
    """Test and display 3D landmark positions."""
    
    # Initialize camera
    cap = cv2.VideoCapture(1)
    
    # Initialize Face Mesh with 3D
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,  # REQUIRED for iris landmarks
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        print("Testing 3D gaze vectors...")
        print("Look at different screen positions")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get 3D coordinates of key landmarks
                    landmarks_3d = face_landmarks.landmark
                    
                    # Extract key points
                    left_inner = landmarks_3d[LEFT_EYE_INNER_CORNER]
                    left_outer = landmarks_3d[LEFT_EYE_OUTER_CORNER]
                    right_inner = landmarks_3d[RIGHT_EYE_INNER_CORNER]
                    right_outer = landmarks_3d[RIGHT_EYE_OUTER_CORNER]
                    
                    # Get iris centers (3D)
                    left_iris = landmarks_3d[LEFT_IRIS_CENTER]
                    right_iris = landmarks_3d[RIGHT_IRIS_CENTER]
                    
                    # Calculate eye centers (approximate)
                    left_eye_center = (
                        (left_inner.x + left_outer.x) / 2,
                        (left_inner.y + left_outer.y) / 2,
                        (left_inner.z + left_outer.z) / 2
                    )
                    
                    right_eye_center = (
                        (right_inner.x + right_outer.x) / 2,
                        (right_inner.y + right_outer.y) / 2,
                        (right_inner.z + right_outer.z) / 2
                    )
                    
                    # Calculate gaze vectors (from eye center to iris)
                    left_gaze_vector = (
                        left_iris.x - left_eye_center[0],
                        left_iris.y - left_eye_center[1],
                        left_iris.z - left_eye_center[2]
                    )
                    
                    right_gaze_vector = (
                        right_iris.x - right_eye_center[0],
                        right_iris.y - right_eye_center[1],
                        right_iris.z - right_eye_center[2]
                    )
                    
                    # Display 3D information
                    height, width, _ = frame.shape
                    y_offset = 30
                    
                    # Show eye centers
                    cv2.putText(frame, f"Left Eye Center (3D): ({left_eye_center[0]:.3f}, {left_eye_center[1]:.3f}, {left_eye_center[2]:.3f})", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_offset += 20
                    
                    cv2.putText(frame, f"Right Eye Center (3D): ({right_eye_center[0]:.3f}, {right_eye_center[1]:.3f}, {right_eye_center[2]:.3f})", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_offset += 30
                    
                    # Show gaze vectors
                    cv2.putText(frame, f"Left Gaze Vector: ({left_gaze_vector[0]:.4f}, {left_gaze_vector[1]:.4f}, {left_gaze_vector[2]:.4f})", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                    y_offset += 20
                    
                    cv2.putText(frame, f"Right Gaze Vector: ({right_gaze_vector[0]:.4f}, {right_gaze_vector[1]:.4f}, {right_gaze_vector[2]:.4f})", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                    y_offset += 30
                    
                    # Calculate gaze angles (for debugging)
                    left_gaze_angle_x = np.arctan2(left_gaze_vector[0], abs(left_gaze_vector[2])) * 180 / np.pi
                    left_gaze_angle_y = np.arctan2(left_gaze_vector[1], abs(left_gaze_vector[2])) * 180 / np.pi
                    
                    cv2.putText(frame, f"Left Gaze Angles: X={left_gaze_angle_x:.1f}°, Y={left_gaze_angle_y:.1f}°", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
                    y_offset += 20
                    
                    # Draw 3D visualization points on 2D image
                    # Convert 3D to 2D for visualization
                    def project_3d_to_2d(point_3d, img_width, img_height):
                        x = int(point_3d[0] * img_width)
                        y = int(point_3d[1] * img_height)
                        return (x, y)
                    
                    # Draw eye centers
                    left_center_2d = project_3d_to_2d(left_eye_center, width, height)
                    right_center_2d = project_3d_to_2d(right_eye_center, width, height)
                    
                    cv2.circle(frame, left_center_2d, 5, (0, 255, 0), -1)
                    cv2.circle(frame, right_center_2d, 5, (0, 255, 0), -1)
                    
                    # Draw iris centers
                    left_iris_2d = project_3d_to_2d((left_iris.x, left_iris.y, left_iris.z), width, height)
                    right_iris_2d = project_3d_to_2d((right_iris.x, right_iris.y, right_iris.z), width, height)
                    
                    cv2.circle(frame, left_iris_2d, 3, (0, 0, 255), -1)
                    cv2.circle(frame, right_iris_2d, 3, (0, 0, 255), -1)
                    
                    # Draw gaze vectors (scaled for visualization)
                    scale = 50
                    left_gaze_end = (
                        left_center_2d[0] + int(left_gaze_vector[0] * scale * 100),  # Scale up for visibility
                        left_center_2d[1] + int(left_gaze_vector[1] * scale * 100)
                    )
                    right_gaze_end = (
                        right_center_2d[0] + int(right_gaze_vector[0] * scale * 100),
                        right_center_2d[1] + int(right_gaze_vector[1] * scale * 100)
                    )
                    
                    cv2.arrowedLine(frame, left_center_2d, left_gaze_end, (255, 0, 0), 2)
                    cv2.arrowedLine(frame, right_center_2d, right_gaze_end, (255, 0, 0), 2)
            
            # Show frame
            cv2.imshow('3D Gaze Vector Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_3d_landmarks()