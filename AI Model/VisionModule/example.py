"""
Filename: example.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 12/5/2025

Description:
    Simple iris detection and drawing test.
"""

from camera import IrisCamera
from face_utils import FaceMeshDetector
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
        
        # Get Face Mesh with Iris
        frame, landmarks = face_mesh.draw_face_mesh(
            frame, 
            draw_face=True, 
            draw_iris=False,
            draw_tesselation=False,
            draw_contours=False
        )
        
        # Get and draw iris centers
        iris_centers = face_mesh.get_iris_centers(frame)
        if iris_centers:
            frame = face_mesh.draw_iris_centers(frame, iris_centers)       
            # Print to console & display on screen
            print(f"Left: {iris_centers['left']}, Right: {iris_centers['right']}")
            cv2.putText(frame, f"Left: {iris_centers['left']}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Right: {iris_centers['right']}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        else:
            cv2.putText(frame, "No face/iris detected", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('Iris Detection', frame)
        
        # Terminate when 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
face_mesh.release()