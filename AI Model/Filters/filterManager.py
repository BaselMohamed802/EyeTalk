"""
Filename: filterManager.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 12/6/2025

Description:
    Filter manager for applying different smoothing filters to iris coordinates.
    Supports Kalman and EMA filters.
"""

# Import necessary libraries
import sys
import os

# Add the Filters directory to Python path so we can import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from EMAFilter import EMAFilter
    from kalmanFilter import KalmanFilter
except ImportError:
    # For relative imports when running from parent directory
    from .EMAFilter import EMAFilter
    from .kalmanFilter import KalmanFilter

# Filter Manager Class
class FilterManager:
    def __init__(self, filter_type="ema", **kwargs):
        """
        Initialize filter manager.
        
        Args:
            filter_type (str): Type of filter to use ("kalman" or "ema")
            **kwargs: Additional filter parameters:
                - For EMA: alpha (default 0.35)
                - For Kalman: dt, process_noise, measurement_noise
        """
        self.filter_type = filter_type.lower()
        
        if self.filter_type == "kalman":
            self.filter = KalmanFilter(**kwargs)
        elif self.filter_type == "ema":
            self.filter = EMAFilter(**kwargs)
        else:
            raise ValueError(f"Invalid filter type: {filter_type}. Use 'kalman' or 'ema'.")
    
    def apply_filter(self, x, y):
        """
        Apply selected filter to input coordinates.
        
        Args:
            x (float): x-coordinate
            y (float): y-coordinate
            
        Returns:
            tuple: Filtered (x, y) coordinates
        """
        try:
            if self.filter_type == "kalman":
                return self.filter.apply_kalman(x, y)
            elif self.filter_type == "ema":
                return self.filter.apply_ema(x, y)
        except AttributeError as e:
            raise ValueError(f"Filter method not found. Error: {e}")
    
    def reset(self):
        """Reset the current filter."""
        if hasattr(self.filter, 'reset'):
            self.filter.reset()
    
    def switch_filter(self, new_filter_type, **kwargs):
        """
        Switch to a different filter type.
        
        Args:
            new_filter_type (str): New filter type ("kalman" or "ema")
            **kwargs: Filter parameters
        """
        old_filter = self.filter_type
        self.filter_type = new_filter_type.lower()
        
        if self.filter_type == "kalman":
            self.filter = KalmanFilter(**kwargs)
        elif self.filter_type == "ema":
            self.filter = EMAFilter(**kwargs)
        else:
            self.filter_type = old_filter
            raise ValueError(f"Invalid filter type: {new_filter_type}. Use 'kalman' or 'ema'.")