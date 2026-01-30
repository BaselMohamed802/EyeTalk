"""
Filename: mouse_controller.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 1/30/2026

Description:
    Threaded mouse controller for smooth cursor movement.
"""

import pyautogui
import threading
import time
from typing import Tuple, Optional

class MouseController:
    def __init__(self, update_interval: float = 0.01):
        """
        Initialize threaded mouse controller.
        
        Args:
            update_interval: Time between mouse updates in seconds
        """
        self.update_interval = update_interval
        self.mouse_target = [0, 0]
        self.mouse_lock = threading.Lock()
        self.enabled = True
        self.running = False
        self.thread = None
        
        # Get screen size
        self.screen_width, self.screen_height = pyautogui.size()
    
    def start(self):
        """Start the mouse control thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._mouse_mover, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop the mouse control thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def _mouse_mover(self):
        """Main mouse movement loop."""
        while self.running:
            if self.enabled:
                with self.mouse_lock:
                    x, y = self.mouse_target
                # Move mouse
                pyautogui.moveTo(x, y)
            time.sleep(self.update_interval)
    
    def set_target(self, x: int, y: int):
        """
        Set new mouse target position.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        # Clamp to screen bounds
        x = max(0, min(x, self.screen_width - 1))
        y = max(0, min(y, self.screen_height - 1))
        
        with self.mouse_lock:
            self.mouse_target[0] = x
            self.mouse_target[1] = y
    
    def get_target(self) -> Tuple[int, int]:
        """Get current mouse target position."""
        with self.mouse_lock:
            return tuple(self.mouse_target)
    
    def enable(self):
        """Enable mouse control."""
        self.enabled = True
    
    def disable(self):
        """Disable mouse control."""
        self.enabled = False
    
    def toggle(self) -> bool:
        """Toggle mouse control on/off."""
        self.enabled = not self.enabled
        return self.enabled
    
    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen dimensions."""
        return self.screen_width, self.screen_height
    
    def get_center(self) -> Tuple[int, int]:
        """Get screen center coordinates."""
        return self.screen_width // 2, self.screen_height // 2