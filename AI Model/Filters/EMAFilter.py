"""
Filename: EMAFilter.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 12/6/2025

Description:
    This file contains the code for the EMA filter, which is a simple exponential moving average (EMA) filter.
    It will take the Iris center coordinates (Both eyes) and apply an EMA filter to them to smoothen the movement.
"""

# Import necessary libraries
import numpy as np

# EMA Filter class
class EMAFilter:
    def __init__(self, alpha=0.35):
        """
        Initialize the EMA filter object.

        Args:
            alpha (float, optional): The smoothing factor of the EMA filter. Defaults to 0.35.
        """
        if not (0 <= alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1")

        self.alpha = alpha
        self.initialized = False
        self.prev_x = 0
        self.prev_y = 0

    def apply_ema(self, x, y):
        """
        Apply the EMA filter to the input coordinates.

        Args:
            x (float): The x-coordinate of the input.
            y (float): The y-coordinate of the input.

        Returns:
            tuple: A tuple containing the filtered x and y coordinates.
        """
        if not self.initialized:
            self.prev_x = x
            self.prev_y = y
            self.initialized = True

        filtered_x = self.alpha * x + (1 - self.alpha) * self.prev_x
        filtered_y = self.alpha * y + (1 - self.alpha) * self.prev_y

        self.prev_x = filtered_x
        self.prev_y = filtered_y

        return filtered_x, filtered_y
    
    def reset(self):
        """
        Reset the EMA filter to its initial state.

        This is useful when switching between different sources of input data.
        """
        self.initialized = False
        self.prev_x = 0
        self.prev_y = 0