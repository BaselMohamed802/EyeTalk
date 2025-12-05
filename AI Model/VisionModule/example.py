"""
Filename: example.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 12/4/2025

Description:
    Test demo file to test the Vision Module.
"""

# Import necessary modules 
from camera import IrisCamera, print_camera_info
from face_utils import FaceDetector
import cv2

# Test the ports
print_camera_info()

# Get Video capture
with IrisCamera(0) as camera:
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        # Add FPS and show webcam
        fps = camera.get_frame_rate()
        cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Show faces
        frame, bboxes = FaceDetector(min_detection_confidence=0.8, min_tracking_confidence=0.8).findFaces(frame, draw=True, print_score=True)

        # Show webcam
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            camera.stop_recording()
            cv2.destroyAllWindows()
            FaceDetector().release()
            break



