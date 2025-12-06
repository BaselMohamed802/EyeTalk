"""
Filename: kalmanFilter.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 12/6/2025

Description:
    This file contains the code for the Kalman filter, which is a linear 
    estimation technique used to predict and update the state of a system.
    It will take the Iris center coordinates (Both eyes) and apply a Kalman filter to them to smoothen the movement.
"""

import numpy as np

class KalmanFilter:
    def __init__(self, dt=1.0, process_noise=1e-3, measurement_noise=1e-1):
        """
        Initialize Kalman Filter for 2D position and velocity tracking.
        
        Args:
            dt (float): Time step (default 1.0 for discrete frames)
            process_noise (float): Process noise covariance
            measurement_noise (float): Measurement noise covariance
        """
        self.dt = dt
        
        # State: [x, y, vx, vy]
        self.state = np.zeros(4)
        self.covariance = np.eye(4)
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Process noise covariance
        self.Q = np.eye(4) * process_noise
        
        # Measurement matrix (we only measure position, not velocity)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise
        
        self.initialized = False
    
    def apply_kalman(self, x, y):
        """
        Apply Kalman filter to new measurement.
        
        Args:
            x (float): Measured x-coordinate
            y (float): Measured y-coordinate
            
        Returns:
            tuple: Filtered (x, y) coordinates
        """
        measurement = np.array([x, y])
        
        if not self.initialized:
            # Initialize state with first measurement
            self.state = np.array([x, y, 0, 0])
            self.covariance = np.eye(4) * 1000  # Large initial uncertainty
            self.initialized = True
            return x, y
        
        # --- Prediction step ---
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        
        # --- Update step ---
        # Innovation (measurement residual)
        y_residual = measurement - self.H @ self.state
        
        # Innovation covariance
        S = self.H @ self.covariance @ self.H.T + self.R
        
        # Kalman gain
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        
        # Update state estimate
        self.state = self.state + K @ y_residual
        
        # Update covariance estimate
        self.covariance = (np.eye(4) - K @ self.H) @ self.covariance
        
        # Return filtered position (first two elements of state)
        return float(self.state[0]), float(self.state[1])
    
    def reset(self):
        """Reset filter to initial state."""
        self.state = np.zeros(4)
        self.covariance = np.eye(4)
        self.initialized = False