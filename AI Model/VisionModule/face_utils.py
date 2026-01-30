"""
Filename: face_utils.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 12/4/2025

Description:
    This file contains Two main classes:
    1. FaceDetector: This class provides a set of helping functions for working with 
                     face detection with google mediapipe.
    2. FaceMeshDetector: This class provides a set of helping functions for working with 
                         face mesh detection with google mediapipe.
"""

# Import necessary libraries
import cv2
import mediapipe as mp
import numpy as np
from head_tracking import HeadTracker

# Face Detection Class
class FaceDetector():
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the FaceDetector class.

        Args:
            min_detection_confidence (float, optional): Minimum confidence value
                ([0.0, 1.0]) from the face detection model for the detection to be
                considered successful. Defaults to 0.5.
            min_tracking_confidence (float, optional): Minimum confidence value
                ([0.0, 1.0]) from the landmark tracker model for the face to be
                considered tracked successfully. Defaults to 0.5.
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            self.min_detection_confidence, 
            self.min_tracking_confidence)

    def findFaces(self, img, draw=True, print_score=False):
        """
        Find faces in an image using Google MediaPipe Face Detection.

        Args:
            img (numpy.ndarray): Input image.
            draw (bool, optional): Whether to draw bounding boxes around detected faces. Defaults to True.
            print_score (bool, optional): Whether to print scores of detected faces. Defaults to False.

        Returns:
            tuple: Tuple containing the processed image and a list of bounding boxes and scores.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(imgRGB)

        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if print_score and draw:
                    img = self.draw_target(img, bbox, l=30, t=5, rt=1)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                if draw:
                    img = self.draw_target(img, bbox, l=30, t=5, rt=1)
        return img, bboxs
    
    def draw_target(self, img, bbox, l=30, t=3, rt=1):
        """
        Draw a bounding box and lines (To make it more visible) around a detected face in an image.

        Args:
            img (numpy.ndarray): Input image.
            bbox (tuple): Tuple containing the bounding box coordinates (x, y, w, h).
            l (int, optional): Length of the lines. Defaults to 30.
            t (int, optional): Thickness of the lines. Defaults to 5.
            rt (int, optional): Thickness of the bounding box. Defaults to 1.

        Returns:
            numpy.ndarray: Processed image with drawn bounding box and lines.
        """
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        # Top Left x, y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)

        # Top Right x1, y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)

        # Bottom Left x, y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)

        # Bottom Right x1, y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        return img
    def release(self):
        """
        Release the face detection model resources.
        """
        self.face_detection.close()

# Face Mesh class
class FaceMeshDetector:
    def __init__(self, static_image_mode=False, max_num_faces=1,
                 refine_landmarks=True, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Initialize the Face Mesh Detector.

        Args:
            static_image_mode (bool, optional): If set to True, use a static image mode. 
                Defaults to False.
            max_num_faces (int, optional): Maximum number of faces to detect. Defaults to 1.
            refine_landmarks (bool, optional): Whether to refine the detected facial landmarks.
                MUST be True for iris detection! Defaults to True.
            min_detection_confidence (float, optional): Minimum confidence value ([0.0, 1.0]) 
                from the face detection model. Defaults to 0.5.
            min_tracking_confidence (float, optional): Minimum confidence value ([0.0, 1.0]) 
                from the tracking model. Defaults to 0.5.
        """
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=self.static_image_mode,
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Iris Landmark indices (ONLY available when refine_landmarks=True)
        self.LEFT_IRIS_IDS = [474, 475, 476, 477]
        self.RIGHT_IRIS_IDS = [469, 470, 471, 472]

    def draw_face_mesh(self, img, draw_face=True, draw_iris=False, 
                       draw_tesselation=True, draw_contours=False):
        """
        Draw a face mesh, face contours, and/or iris on an image using Google MediaPipe Face Mesh.

        Args:
            img (numpy.ndarray): Input image.
            draw_face (bool, optional): Whether to draw the face mesh. Defaults to True.
            draw_iris (bool, optional): Whether to draw the iris. Defaults to False.
            draw_tesselation (bool, optional): Whether to draw the face mesh tessellation. 
                Defaults to True.
            draw_contours (bool, optional): Whether to draw the face contours. Defaults to False.

        Returns:
            tuple: Tuple containing the processed image and a list of lists of landmarks coordinates.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(img_rgb)

        landmarks_lst = []
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                # Get the landmarks coordinates (unnormalized)
                img_height, img_width, _ = img.shape
                landmarks = []

                for idx, landmark in enumerate(face_landmarks.landmark):
                    x = int(landmark.x * img_width)
                    y = int(landmark.y * img_height)
                    landmarks.append((idx, x, y))
                
                landmarks_lst.append(landmarks)

                # Draw face mesh if requested
                if draw_face:
                    # Draw the face mesh
                    if draw_tesselation:
                        self.mp_drawing.draw_landmarks(
                            image=img,
                            landmark_list=face_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing_styles
                            .get_default_face_mesh_tesselation_style()
                        )

                    # Draw the face contours
                    if draw_contours:
                        self.mp_drawing.draw_landmarks(
                            image=img,
                            landmark_list=face_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing_styles
                            .get_default_face_mesh_contours_style()
                        )

                # Draw iris if requested
                if draw_iris and self.refine_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_IRISES,  # Use predefined iris connections
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                            color=(0, 255, 0), thickness=1
                        )
                    )

        return img, landmarks_lst
    
    def get_iris_landmarks(self, img):
        """
        Extract iris landmarks from the given image.

        Args:
            img: A 3-channel color image represented as a numpy array.

        Returns:
            A dictionary containing the left and right iris landmarks. 
            Each landmark is represented as a tuple of (idx, x, y).
        """
        if not self.refine_landmarks:
            print("Warning: refine_landmarks=False. Iris landmarks not available.")
            return {'left': [], 'right': []}
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(img_rgb)

        iris_data = {'left': [], 'right': []}
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                img_height, img_width, _ = img.shape

                # Extract left iris landmarks (indices 468-478)
                for idx in self.LEFT_IRIS_IDS:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * img_width)
                    y = int(landmark.y * img_height)
                    iris_data['left'].append((idx, x, y))

                # Extract right iris landmarks (indices 473-478 and 468-473)
                for idx in self.RIGHT_IRIS_IDS:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * img_width)
                    y = int(landmark.y * img_height)
                    iris_data['right'].append((idx, x, y))

        return iris_data
    
    def get_iris_centers(self, img):
        """
        Calculate the center of the left and right iris from the given image.

        Args:
            img: A 3-channel color image represented as a numpy array.

        Returns:
            A dictionary containing the center of the left and right iris as tuples of (x, y) coordinates.
        """
        iris_data = self.get_iris_landmarks(img)
        centers = {}
        
        for eye in ['left', 'right']:
            if iris_data[eye]:
                points = iris_data[eye]
                if len(points) > 0:
                    x_coords = [p[1] for p in points]
                    y_coords = [p[2] for p in points]
                    center_x = int(np.mean(x_coords))
                    center_y = int(np.mean(y_coords))
                    centers[eye] = (center_x, center_y)
                else:
                    centers[eye] = None
            else:
                centers[eye] = None

        return centers
    
    def draw_iris_centers(self, img, centers, with_text=False, color=None, size=3):
        """
        Draw the center of the left and right iris on the given image.
        
        Args:
            img: A 3-channel color image represented as a numpy array.
            centers: A dictionary containing the center of the left and right iris as tuples of (x, y) coordinates.
            with_text (bool): Whether to add text labels.
            color (tuple): Custom BGR color tuple. If None, uses default colors.
            size (int): Circle size.
            
        Returns:
            The image with the center of the left and right iris drawn on it.
        """
        # Default colors
        left_color = color or (0, 255, 0)  # Green
        right_color = color or (0, 0, 255)  # Red
        
        if centers.get('left'):
            cv2.circle(img, centers['left'], size, left_color, 2)
            if with_text:
                cv2.putText(img, 'L', (centers['left'][0] + 5, centers['left'][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)
        
        if centers.get('right'):
            cv2.circle(img, centers['right'], size, right_color, 2)
            if with_text:
                cv2.putText(img, 'R', (centers['right'][0] + 5, centers['right'][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)
        
        return img
    
    def get_head_key_points(self, img):
            """
            Extract key points for head tracking from the given image.
            
            Args:
                img: A 3-channel color image
                
            Returns:
                Dictionary with key head points and their 3D coordinates
            """
            # Define key landmark indices for head tracking
            HEAD_KEY_POINTS = {
                "left": 234,     # Left temple
                "right": 454,    # Right temple
                "top": 10,       # Forehead top
                "bottom": 152,   # Chin bottom
                "front": 1,      # Nose tip
                "nose_bridge": 168,  # Nose bridge
                "left_eye_outer": 33,   # Left eye outer corner
                "right_eye_outer": 263, # Right eye outer corner
                "left_eye_inner": 133,  # Left eye inner corner
                "right_eye_inner": 362, # Right eye inner corner
            }
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.face_mesh.process(img_rgb)
            
            key_points = {}
            if self.results.multi_face_landmarks:
                face_landmarks = self.results.multi_face_landmarks[0]
                img_height, img_width, _ = img.shape
                
                for name, idx in HEAD_KEY_POINTS.items():
                    landmark = face_landmarks.landmark[idx]
                    # Convert to 3D point (z coordinate from landmark.z)
                    key_points[name] = np.array([
                        landmark.x * img_width,
                        landmark.y * img_height,
                        landmark.z * img_width  # Using width as z scale
                    ])
            
            return key_points
    
    def draw_head_cube(self, img, cube_corners, edges=None):
        """
        Draw a 3D cube representing head orientation.
        
        Args:
            img: Input image
            cube_corners: List of 8 cube corner points as numpy arrays
            edges: List of edge connections (optional, uses default if None)
            
        Returns:
            Image with cube drawn
        """
        if edges is None:
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # front face
                (4, 5), (5, 6), (6, 7), (7, 4),  # back face
                (0, 4), (1, 5), (2, 6), (3, 7)   # sides
            ]
        
        # Convert 3D points to 2D for drawing
        cube_2d = [HeadTracker.project_to_2d(pt) for pt in cube_corners]
        
        # Draw edges
        for i, j in edges:
            cv2.line(img, cube_2d[i], cube_2d[j], (255, 125, 35), 2)
        
        # Draw corners
        for pt in cube_2d:
            cv2.circle(img, pt, 3, (0, 255, 255), -1)
        
        return img
    
    def draw_gaze_ray(self, img, origin, direction, length=200, color=(15, 255, 0)):
        """
        Draw a gaze direction ray from head center.
        
        Args:
            img: Input image
            origin: Ray origin point (3D)
            direction: Ray direction vector (3D)
            length: Ray length in pixels
            color: Ray color in BGR
            
        Returns:
            Image with ray drawn
        """
        # Calculate ray end point
        end_point = origin - direction * length
        
        # Convert to 2D
        start_2d = HeadTracker.project_to_2d(origin)
        end_2d = HeadTracker.project_to_2d(end_point)
        
        # Draw ray
        cv2.line(img, start_2d, end_2d, color, 3)
        
        # Draw origin point
        cv2.circle(img, start_2d, 5, (0, 0, 255), -1)
        
        return img
    
    def get_face_outline_points(self, img):
        """
        Get face outline points for visualization.
        
        Args:
            img: Input image
            
        Returns:
            List of (x, y) points for face outline
        """
        FACE_OUTLINE_INDICES = [
            10, 338, 297, 332, 284, 251, 389, 356,
            454, 323, 361, 288, 397, 365, 379, 378,
            400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21,
            54, 103, 67, 109
        ]
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(img_rgb)
        
        outline_points = []
        if self.results.multi_face_landmarks:
            face_landmarks = self.results.multi_face_landmarks[0]
            img_height, img_width, _ = img.shape
            
            for idx in FACE_OUTLINE_INDICES:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * img_width)
                y = int(landmark.y * img_height)
                outline_points.append((x, y))
        
        return outline_points

    def release(self):
        """
        Release the FaceMesh object resources.
        """
        self.face_mesh.close()