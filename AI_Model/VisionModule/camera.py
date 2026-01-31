"""
Filename: camera.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 12/4/2025

Description:
    This file is the module for working with the camera using OpenCV.
    It contains all the necessary functions for capturing frames from the camera and processing them.
"""

# Import necessary libraries
import cv2
import time

def list_available_cameras(max_to_test=10):
    """
    Scan for available camera ports and return working cameras.
    
    Args:
        max_to_test (int): Maximum number of ports to test
        
    Returns:
        dict: Dictionary with camera information by port number
    """
    available_cameras = {}
    
    for port in range(max_to_test):
        cap = cv2.VideoCapture(port)
        
        if not cap.isOpened():
            continue
        ret, frame = cap.read()
        
        if ret:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            available_cameras[port] = {
                'port': port,
                'resolution': (width, height),
                'fps': fps,
                'name': f'Camera {port}'
            }
            
            print(f"✓ Camera found at port {port}: {width}x{height} ({fps} FPS)")
        else:
            print(f"⚠ Camera at port {port} opens but cannot read frames")
        
        cap.release()
    
    return available_cameras


def list_camera_ports(max_to_test=10):
    """
    Simplified version that just returns working camera ports.
    
    Args:
        max_to_test (int): Maximum number of ports to test
        
    Returns:
        list: List of working camera port numbers
    """
    working_ports = []
    
    for port in range(max_to_test):
        cap = cv2.VideoCapture(port)
        
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                working_ports.append(port)
            cap.release()
    
    return working_ports

# Camera class
class IrisCamera:
    def __init__(self, cam_id, cam_width=640, cam_height=480):
        try:
            self.cam = cv2.VideoCapture(cam_id)
        except:
            raise Exception(f"Camera with [{cam_id}] not found.")
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

        # Store resolution
        self.cam_width = cam_width
        self.cam_height = cam_height

    def get_frame(self):
        """
        Capture a single frame from the camera.
        
        Returns:
            tuple: (success, frame) where success is boolean and frame is numpy array
        """
        if not self.cam.isOpened():
            return False, None
        
        ret, frame = self.cam.read()
        return frame if ret else None

    def stop_recording(self):
        """
        Stop recording from the camera. This function will release the camera resource.
        """
        if self.cam.isOpened():
            self.cam.release()
        else:
            return None

    def is_opened(self):
        """
        Check if the camera is opened or not.
        
        Returns:
            bool: True if camera is opened, False otherwise.
        """
        return self.cam.isOpened()
    
    def get_resolution(self):
        """
        Get the resolution of the camera.
        
        Returns:
            tuple: (width, height) containing the resolution of the camera.
        """
        return (self.cam_width, self.cam_height)
    
    def get_frame_rate(self):
        """
        Get the current frame rate of the camera.
        
        Returns:
            float: The current frame rate of the camera.
        """
        # Store previous time as instance variable if not exists
        if not hasattr(self, 'pTime'):
            self.pTime = time.time()
            return 0
        
        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime
        return int(fps)
    
    def __enter__(self):
        """
        Context manager entry point for the camera object.
        Returns the camera object itself.
        """
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Context manager exit point for the camera object.
        Releases the camera resource.
        """
        self.stop_recording()

def print_camera_info():
    """Print information about all available cameras."""
    print("Scanning for available cameras...")
    cameras = list_available_cameras()
    
    if not cameras:
        print("No cameras found.")
        return
    
    print(f"\nFound {len(cameras)} camera(s):")
    for port, info in cameras.items():
        print(f"  Port {port}: {info['resolution'][0]}x{info['resolution'][1]} "
              f"(~{info['fps']:.1f} FPS)")
    
    return cameras