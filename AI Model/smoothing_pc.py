"""
Filename: smoothing_pc.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: [Current Date]

Description:
    Advanced smoothing algorithms optimized for PC.
"""

import numpy as np
from collections import deque
import time
import math

class ExponentialSmoothing:
    """Double exponential smoothing optimized for high FPS."""
    def __init__(self, alpha=0.25, beta=0.1):
        self.alpha = alpha  # Smoothing factor (lower = more smoothing)
        self.beta = beta    # Trend factor
        self.level = None
        self.trend = None
        self.last_time = time.perf_counter()
        
    def update(self, value):
        current_time = time.perf_counter()
        dt = max(0.001, current_time - self.last_time)  # Avoid division by zero
        self.last_time = current_time
        
        if self.level is None:
            self.level = value
            self.trend = 0
            return value
        
        last_level = self.level
        
        # Smooth the level
        self.level = self.alpha * value + (1 - self.alpha) * (self.level + self.trend * dt)
        
        # Smooth the trend
        if dt > 0:
            self.trend = self.beta * (self.level - last_level) / dt + (1 - self.beta) * self.trend
        
        return self.level

class AdaptiveLowPassFilter:
    """Adaptive filter that adjusts smoothing based on speed."""
    def __init__(self, cutoff_freq=5.0, dt=0.01):
        self.cutoff_freq = cutoff_freq
        self.dt = dt
        self.alpha = self.calculate_alpha(cutoff_freq, dt)
        self.filtered_value = None
        self.speed_buffer = deque(maxlen=10)
        
    @staticmethod
    def calculate_alpha(cutoff_freq, dt):
        """Calculate alpha for low-pass filter."""
        rc = 1.0 / (2 * math.pi * cutoff_freq)
        return dt / (rc + dt)
    
    def update_cutoff(self, speed):
        """Adjust cutoff frequency based on movement speed."""
        # More smoothing when moving fast
        if speed > 500:  # Fast movement
            self.cutoff_freq = 2.0
        elif speed > 200:  # Medium movement
            self.cutoff_freq = 3.5
        else:  # Slow movement
            self.cutoff_freq = 5.0
        
        self.alpha = self.calculate_alpha(self.cutoff_freq, self.dt)
    
    def update(self, value, speed=0):
        """Update filter with new value."""
        self.update_cutoff(speed)
        
        if self.filtered_value is None:
            self.filtered_value = value
            return value
        
        self.filtered_value = self.alpha * value + (1 - self.alpha) * self.filtered_value
        return self.filtered_value

class PCVelocitySmoother:
    """Velocity-based smoothing optimized for high-FPC PC."""
    def __init__(self, position_history=8):
        self.position_history = position_history
        self.positions = deque(maxlen=position_history)
        self.timestamps = deque(maxlen=position_history)
        self.last_smoothed = None
        
    def smooth(self, x, y):
        """Smooth position based on velocity."""
        current_time = time.perf_counter()
        
        self.positions.append((x, y))
        self.timestamps.append(current_time)
        
        if len(self.positions) < 3:
            return x, y
        
        # Calculate velocity
        dx = x - self.positions[0][0]
        dy = y - self.positions[0][1]
        dt = current_time - self.timestamps[0]
        
        if dt > 0:
            velocity = math.sqrt(dx*dx + dy*dy) / dt
            
            # Adaptive weights based on velocity
            if velocity > 800:  # Very fast
                weights = [0.1, 0.15, 0.2, 0.25, 0.3]
            elif velocity > 400:  # Fast
                weights = [0.2, 0.25, 0.3, 0.15, 0.1]
            elif velocity > 100:  # Medium
                weights = [0.3, 0.3, 0.2, 0.1, 0.1]
            else:  # Slow
                weights = [0.4, 0.3, 0.2, 0.1, 0.0]
            
            # Ensure we have enough positions
            num_weights = min(len(weights), len(self.positions))
            weights = weights[:num_weights]
            positions = list(self.positions)[-num_weights:]
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                
                # Weighted average
                smoothed_x = sum(p[0] * w for p, w in zip(positions, weights))
                smoothed_y = sum(p[1] * w for p, w in zip(positions, weights))
                
                self.last_smoothed = (smoothed_x, smoothed_y)
                return int(smoothed_x), int(smoothed_y)
        
        return x, y

class AngleSmoother:
    """Specialized smoother for head tracking angles."""
    def __init__(self, smoothing_factor=0.3):
        self.smoothing_factor = smoothing_factor
        self.last_yaw = None
        self.last_pitch = None
        self.max_angle_change = 8.0  # Max degrees per frame
        
    def smooth(self, yaw, pitch):
        if self.last_yaw is None:
            self.last_yaw = yaw
            self.last_pitch = pitch
            return yaw, pitch
        
        # Apply exponential smoothing
        smoothed_yaw = self.smoothing_factor * yaw + (1 - self.smoothing_factor) * self.last_yaw
        smoothed_pitch = self.smoothing_factor * pitch + (1 - self.smoothing_factor) * self.last_pitch
        
        # Limit rate of change to prevent jumps
        delta_yaw = smoothed_yaw - self.last_yaw
        delta_pitch = smoothed_pitch - self.last_pitch
        
        if abs(delta_yaw) > self.max_angle_change:
            smoothed_yaw = self.last_yaw + np.sign(delta_yaw) * self.max_angle_change
        
        if abs(delta_pitch) > self.max_angle_change:
            smoothed_pitch = self.last_pitch + np.sign(delta_pitch) * self.max_angle_change
        
        self.last_yaw = smoothed_yaw
        self.last_pitch = smoothed_pitch
        
        return smoothed_yaw, smoothed_pitch