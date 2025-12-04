"""
Filename: example.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 12/4/2025

Description:
    Test demo file to test the Vision Module.
"""

# Import necessary modules
from camera import IrisCamera, print_camera_info
import cv2

# Test the ports
print_camera_info()

# Get Video capture
with IrisCamera(0) as camera:
    while True:
        frame = camera.get_frame()
        if frame is None:
            break

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break



