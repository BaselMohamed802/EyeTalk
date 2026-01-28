"""
Filename: cursor_controller.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 1/28/2026

Description:
    Simple cursor controller using pyautogui.
    Minimal version for compatibility.
"""

import pyautogui

class CursorController:
    """Simple cursor controller using pyautogui."""
    
    def __init__(self):
        """Initialize cursor controller."""
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"[CursorController] Screen: {self.screen_width}x{self.screen_height}")
    
    def move_to(self, x: int, y: int):
        """
        Move cursor to specified coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        # Clamp to screen bounds
        x = max(0, min(x, self.screen_width - 1))
        y = max(0, min(y, self.screen_height - 1))
        
        # Move cursor
        pyautogui.moveTo(x, y, duration=0)
    
    def get_position(self):
        """
        Get current cursor position.
        
        Returns:
            (x, y) tuple of current position
        """
        return pyautogui.position()