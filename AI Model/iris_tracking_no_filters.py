"""
Filename: iris_tracking_no_filters.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 12/6/2025

Description:
    Simple iris detection and drawing test WITHOUT filters.
"""

import sys
import os

# Add the parent directory to Python path to import from Filters
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from VisionModule.camera import IrisCamera
from VisionModule.face_utils import FaceMeshDetector
import cv2

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
        
        # Check if both eyes are detected (person is not dead LOL - XD???)
        if iris_centers_raw['left'] and iris_centers_raw['right']:
            iris_right_raw = iris_centers_raw['right']
            iris_left_raw = iris_centers_raw['left']
            
            # Draw raw iris centers (small, default colors)
            frame = face_mesh.draw_iris_centers(frame, iris_centers_raw, with_text=True, color=(0, 0, 255), size=1)
            
            # Print to console
            print(f"Raw - Left: {iris_centers_raw['left']}, Right: {iris_centers_raw['right']}")
            
            # Display coordinates on screen
            y_offset = 60
            cv2.putText(frame, f"Raw L: {iris_centers_raw['left']}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Raw R: {iris_centers_raw['right']}", (10, y_offset + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        else:
            cv2.putText(frame, "No face/iris detected", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('Iris Detection with Filters', frame)
        
        # Terminate when 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
face_mesh.release()