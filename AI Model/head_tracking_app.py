"""
Filename: head_tracking_app.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: [Current Date]

Description:
    Main application for head-tracking mouse control.
    Uses the modular components for camera, face detection, head tracking, and mouse control.
"""

import cv2
import keyboard
import time
from VisionModule.camera import IrisCamera
from VisionModule.face_utils import FaceMeshDetector
from VisionModule.head_tracking import HeadTracker
from PyautoguiController.mouse_controller import MouseController

class HeadTrackingApp:
    def __init__(self, camera_id=0, resolution=(640, 480)):
        """
        Initialize head tracking application.
        
        Args:
            camera_id: Camera device ID
            resolution: Camera resolution (width, height)
        """
        # Initialize components
        self.camera = IrisCamera(camera_id, *resolution)
        self.face_detector = FaceMeshDetector(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.head_tracker = HeadTracker(filter_length=8)
        self.mouse_controller = MouseController(update_interval=0.01)
        
        # Get screen info
        self.screen_width, self.screen_height = self.mouse_controller.get_screen_size()
        self.screen_center_x = self.screen_width // 2
        self.screen_center_y = self.screen_height // 2
        
        # Control flags
        self.mouse_control_enabled = True
        
        # Tracking parameters
        self.yaw_range = 20  # degrees for full screen width
        self.pitch_range = 10  # degrees for full screen height
        
        # Start mouse controller thread
        self.mouse_controller.start()
    
    def run(self):
        """Main application loop."""
        print("Starting Head Tracking Application...")
        print("Press 'F7' to toggle mouse control")
        print("Press 'C' to calibrate (center current position)")
        print("Press 'Q' to quit")
        
        while self.camera.is_opened():
            # Capture frame
            frame = self.camera.get_frame()
            if frame is None:
                continue
            
            # Get facial landmarks
            _, landmarks_list = self.face_detector.draw_face_mesh(
                frame, 
                draw_face=False, 
                draw_iris=False
            )
            
            if landmarks_list:
                landmarks = landmarks_list[0]
                
                # Extract key points for head tracking
                key_points = self.head_tracker.extract_key_points(
                    landmarks, 
                    frame.shape[1], 
                    frame.shape[0]
                )
                
                if key_points and len(key_points) >= 5:
                    # Calculate head orientation
                    orientation = self.head_tracker.calculate_head_orientation(key_points)
                    
                    # Update smoothing buffers
                    self.head_tracker.update_smoothing_buffers(
                        orientation['center'], 
                        orientation['forward_axis']
                    )
                    
                    # Get smoothed orientation
                    avg_origin, avg_direction = self.head_tracker.get_smoothed_orientation()
                    
                    # Calculate head angles
                    yaw_raw, pitch_raw = self.head_tracker.calculate_yaw_pitch(avg_direction)
                    yaw_360, pitch_360 = self.head_tracker.convert_angles_to_360(yaw_raw, pitch_raw)
                    
                    # Map to screen coordinates
                    screen_x, screen_y = self.head_tracker.map_to_screen(
                        yaw_360, pitch_360,
                        self.screen_width, self.screen_height,
                        self.yaw_range, self.pitch_range
                    )
                    
                    # Clamp to screen
                    screen_x, screen_y = self.head_tracker.clamp_to_screen(
                        screen_x, screen_y,
                        self.screen_width, self.screen_height
                    )
                    
                    # Update mouse position
                    if self.mouse_control_enabled:
                        self.mouse_controller.set_target(screen_x, screen_y)
                    
                    # Generate and draw head cube
                    cube_corners = self.head_tracker.generate_head_cube(
                        orientation['center'],
                        orientation['right_axis'],
                        orientation['up_axis'],
                        orientation['forward_axis'],
                        orientation['half_width'],
                        orientation['half_height']
                    )
                    frame = self.face_detector.draw_head_cube(frame, cube_corners)
                    
                    # Draw gaze ray
                    frame = self.face_detector.draw_gaze_ray(
                        frame, avg_origin, avg_direction, length=200
                    )
                    
                    # Draw face outline points
                    outline_points = self.face_detector.get_face_outline_points(frame)
                    for x, y in outline_points:
                        cv2.circle(frame, (x, y), 2, (155, 155, 155), -1)
                    
                    # Display angles
                    cv2.putText(frame, f"Yaw: {yaw_360:.1f}°, Pitch: {pitch_360:.1f}°",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"Screen: ({screen_x}, {screen_y})",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display mouse control status
            status_color = (0, 255, 0) if self.mouse_control_enabled else (0, 0, 255)
            status_text = "ENABLED" if self.mouse_control_enabled else "DISABLED"
            cv2.putText(frame, f"Mouse Control: {status_text}",
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Display FPS
            fps = self.camera.get_frame_rate()
            cv2.putText(frame, f"FPS: {fps}", 
                       (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Show frame
            cv2.imshow("Head Tracking", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if keyboard.is_pressed('f7'):
                self.mouse_control_enabled = not self.mouse_control_enabled
                print(f"[Mouse Control] {'Enabled' if self.mouse_control_enabled else 'Disabled'}")
                time.sleep(0.3)
            
            elif key == ord('c'):
                # Calibrate
                if 'yaw_360' in locals() and 'pitch_360' in locals():
                    self.head_tracker.calibrate(yaw_360, pitch_360)
                    print(f"[Calibrated] Current position set as center")
            
            elif key == ord('q'):
                break
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        print("Cleaning up...")
        self.mouse_controller.stop()
        self.face_detector.release()
        self.camera.stop_recording()
        cv2.destroyAllWindows()
        print("Application terminated.")

if __name__ == "__main__":
    # List available cameras
    from VisionModule.camera import print_camera_info
    cameras = print_camera_info()
    
    # Use first available camera or specify ID
    camera_id = 1 if not cameras else list(cameras.keys())[0]
    
    # Create and run application
    app = HeadTrackingApp(camera_id=camera_id, resolution=(640, 480))
    app.run()