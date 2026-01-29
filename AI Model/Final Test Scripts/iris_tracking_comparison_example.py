"""
Filename: example.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 12/5/2025

Description:
    Simple iris detection and drawing test with filters.
"""

import sys
import os

# Add the parent directory to Python path to import from Filters
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from VisionModule.camera import IrisCamera
from VisionModule.face_utils import FaceMeshDetector
from Filters.filterManager import FilterManager 
import cv2

# Initialize filter manager
right_eye_filter = FilterManager(filter_type="ema", alpha=0.2)
left_eye_filter = FilterManager(filter_type="ema", alpha=0.2)

# Initialize face mesh detector
face_mesh = FaceMeshDetector(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

with IrisCamera(0) as camera:
    while True:
        frame = camera.get_frame()
        if frame is None:
            break

        # Add FPS counter
        fps = camera.get_frame_rate()
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        # Get iris centers - RAW
        iris_centers_raw = face_mesh.get_iris_centers(frame)
        
        # Check if both eyes are detected
        if iris_centers_raw['left'] and iris_centers_raw['right']:
            iris_right_raw = iris_centers_raw['right']
            iris_left_raw = iris_centers_raw['left']
            
            # Apply filters
            iris_right_filtered = right_eye_filter.apply_filter(*iris_right_raw)
            iris_left_filtered = left_eye_filter.apply_filter(*iris_left_raw)
            
            # Convert filtered coordinates to integers
            iris_centers_filtered = {
                'right': (int(iris_right_filtered[0]), int(iris_right_filtered[1])),
                'left': (int(iris_left_filtered[0]), int(iris_left_filtered[1]))
            }
            
            # Draw raw iris centers (small, default colors)
            frame = face_mesh.draw_iris_centers(frame, iris_centers_raw, with_text=True)
            
            # Draw filtered iris centers (bigger, different color)
            frame = face_mesh.draw_iris_centers(frame, iris_centers_filtered, 
                                               with_text=False, color=(0, 255, 255), size=6)
            
            # Print to console
            print(f"Raw - Left: {iris_centers_raw['left']}, Right: {iris_centers_raw['right']}")
            print(f"Filtered - Left: {iris_centers_filtered['left']}, Right: {iris_centers_filtered['right']}")
            
            # Display coordinates on screen
            y_offset = 60
            cv2.putText(frame, f"Raw L: {iris_centers_raw['left']}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Raw R: {iris_centers_raw['right']}", (10, y_offset + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Filtered L: {iris_centers_filtered['left']}", (10, y_offset + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Filtered R: {iris_centers_filtered['right']}", (10, y_offset + 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        else:
            cv2.putText(frame, "No face/iris detected", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            # Reset filters when no face is detected
            right_eye_filter.reset()
            left_eye_filter.reset()
        
        cv2.imshow('Iris Detection with Filters', frame)
        
        # Terminate when 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
face_mesh.release()