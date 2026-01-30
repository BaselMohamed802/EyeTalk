"""
Filename: test_camera.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 1/23/2026

Description:
    Simple script to only test the camera if it is working correctly.
    This is to help debugging in case of an error.
"""

# Import necessary libraries.
import cv2
import sys
import os

from camera import IrisCamera, print_camera_info

# Print camera information first
print("=" * 50)
print("Testing Camera Module")
print("=" * 50)
print_camera_info()
print("=" * 50)

# Try different camera IDs
camera_ids = [0, 1, 2, 3]

for cam_id in camera_ids:
    try:
        print(f"\nTrying camera ID: {cam_id}")
        with IrisCamera(cam_id) as camera:
            print(f"✓ Camera {cam_id} opened successfully")
            print(f"  Resolution: {camera.get_resolution()}")
            
            # Try to get a few frames
            for i in range(30):  # Try 30 frames
                frame = camera.get_frame()
                if frame is not None:
                    print(f"  ✓ Frame {i+1}: Success (shape: {frame.shape})")
                    
                    # Show the first successful frame
                    cv2.imshow(f"Camera {cam_id} Test", frame)
                    cv2.waitKey(100)  # Show for 100ms
                    
                    # Show FPS
                    fps = camera.get_frame_rate()
                    cv2.putText(frame, f"Camera {cam_id} | FPS: {fps}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow(f"Camera {cam_id} Test", frame)
                    
                    # Check if user wants to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        print("\nTest stopped by user")
                        sys.exit(0)
                        
                else:
                    print(f"  ✗ Frame {i+1}: Failed")
                    break
            
            cv2.destroyAllWindows()
            print(f"✓ Camera {cam_id} test completed successfully")
            
    except Exception as e:
        print(f"✗ Camera {cam_id} failed: {e}")
        continue

print("\n" + "=" * 50)
print("Camera testing completed")
print("=" * 50)