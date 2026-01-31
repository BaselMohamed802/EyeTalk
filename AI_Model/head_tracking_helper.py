"""
Filename: head_tracking_helper.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 1/31/2026

Description:
    Helper script with classes and functions for head tracking calibration,
    And smoothing profiles for the main head tracking application.
"""

# Import necessary libraries
import numpy as np

class EnhancedCalibrator:
    def __init__(self, head_tracker):
        """
        Initialize EnhancedCalibrator with a HeadTracker object.
        
        Args:
            head_tracker: HeadTracker object to calibrate
        """

        self.head_tracker = head_tracker
        self.calibration_samples = []
        self.is_calibrating = False
        
    def start_calibration(self, num_samples=30):
        """Start collecting calibration samples"""
        self.calibration_samples = []
        self.is_calibrating = True
        self.num_samples = num_samples
        print(f"Calibration: Look straight ahead for {num_samples/10} seconds...")
        
    def add_sample(self, yaw, pitch):
        """Add a calibration sample"""
        if self.is_calibrating and len(self.calibration_samples) < self.num_samples:
            self.calibration_samples.append((yaw, pitch))
            
            if len(self.calibration_samples) >= self.num_samples:
                self.finish_calibration()
                
    def finish_calibration(self):
        """Calculate calibration offsets from collected samples"""
        if not self.calibration_samples:
            return
            
        yaw_samples = [s[0] for s in self.calibration_samples]
        pitch_samples = [s[1] for s in self.calibration_samples]
        
        # Calculate median (more robust than mean for outliers)
        median_yaw = np.median(yaw_samples)
        median_pitch = np.median(pitch_samples)
        
        # Set calibration offsets
        self.head_tracker.calibrate(median_yaw, median_pitch)
        
        # Calculate calibration quality
        yaw_std = np.std(yaw_samples)
        pitch_std = np.std(pitch_samples)
        
        print(f"Calibration complete!")
        print(f"  Center: Yaw={median_yaw:.1f}°, Pitch={median_pitch:.1f}°")
        print(f"  Stability: Yaw_std={yaw_std:.2f}°, Pitch_std={pitch_std:.2f}°")
        
        if yaw_std > 1.0 or pitch_std > 1.0:
            print("  Warning: High variance detected. Try to hold your head still.")
            
        self.is_calibrating = False


class SmoothingProfile:
    def __init__(self, name, filter_length, movement_threshold, 
                 process_interval, deadzone_radius):
        """
        Initialize a SmoothingProfile object with parameters for the head tracking controller.

        Args:
            name: Name of the smoothing profile
            filter_length: Length of the moving average filter
            movement_threshold: Minimum movement required to update cursor position (in pixels)
            process_interval: Minimum time between updates to the cursor position (in seconds)
            deadzone_radius: Radius of the deadzone around the cursor (in pixels)
        """
        self.name = name
        self.filter_length = filter_length
        self.movement_threshold = movement_threshold
        self.process_interval = process_interval
        self.deadzone_radius = deadzone_radius
        
class SmoothingProfiles:
    FAST = SmoothingProfile(
        name="Fast",
        filter_length=4,
        movement_threshold=0.3,
        process_interval=0.005,
        deadzone_radius=1.0
    )
    
    BALANCED = SmoothingProfile(
        name="Balanced",
        filter_length=8,
        movement_threshold=0.5,
        process_interval=0.01,
        deadzone_radius=2.0
    )
    
    SMOOTH = SmoothingProfile(
        name="Smooth",
        filter_length=12,
        movement_threshold=0.8,
        process_interval=0.02,
        deadzone_radius=3.0
    )
    
    @classmethod
    def get_all_profiles(cls):
        
        """
        Return a list of all available smoothing profiles.
        
        Returns:
            list[SmoothingProfile]: A list of all available smoothing profiles.
        """
        return [cls.FAST, cls.BALANCED, cls.SMOOTH]
    
    @classmethod
    def cycle_profile(cls, current_profile):
        
        """
        Cycle through the available smoothing profiles and return the next one.
        
        Args:
            current_profile (SmoothingProfile): The current smoothing profile.
        
        Returns:
            SmoothingProfile: The next smoothing profile in the cycle.
        """
        profiles = cls.get_all_profiles()
        current_index = profiles.index(current_profile) if current_profile in profiles else 0
        next_index = (current_index + 1) % len(profiles)
        return profiles[next_index]