"""
Filename: example.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 12/5/2025

Description:
    Iris detection with filter testing.
"""

import sys
import os

# Add the parent directory to Python path to import from Filters
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from camera import IrisCamera
from face_utils import FaceMeshDetector
from Filters.filterManager import FilterManager 
import cv2

# Initialize filters for left and right eyes
left_filter = FilterManager(filter_type="kalman", dt=1.0, process_noise=1e-3, measurement_noise=1e-1)
right_filter = FilterManager(filter_type="ema", alpha=0.25)

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
        
        # Get iris centers
        iris_centers = face_mesh.get_iris_centers(frame)
        
        if iris_centers['left'] and iris_centers['right']:
            # Get raw coordinates
            raw_left = iris_centers['left']
            raw_right = iris_centers['right']
            
            # Apply filters
            filtered_left = left_filter.apply_filter(*raw_left)
            filtered_right = right_filter.apply_filter(*raw_right)
            
            # Draw raw coordinates (red)
            cv2.circle(frame, raw_left, 3, (0, 0, 255), -1)  # Red
            cv2.circle(frame, raw_right, 3, (0, 0, 255), -1)  # Red
            
            # Draw filtered coordinates (green)
            cv2.circle(frame, (int(filtered_left[0]), int(filtered_left[1])), 5, (0, 255, 0), -1)  # Green
            cv2.circle(frame, (int(filtered_right[0]), int(filtered_right[1])), 5, (0, 255, 0), -1)  # Green
            
            # Add text
            cv2.putText(frame, f"Raw L: {raw_left}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Filtered L: {filtered_left}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Raw R: {raw_right}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Filtered R: {filtered_right}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display FPS
        fps = camera.get_frame_rate()
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Filter Comparison: Red=Raw, Green=Filtered", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
face_mesh.release()