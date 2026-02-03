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
from typing import Tuple

class MouseController:
    def __init__(self, update_interval: float = 0.01, enable_failsafe: bool = False):
        """
        Initialize threaded mouse controller.
        
        Args:
            update_interval: Time between mouse updates in seconds
            enable_failsafe: Whether to enable PyAutoGUI's fail-safe feature
        """
        self.update_interval = update_interval
        self.mouse_target = [0, 0]
        self.mouse_lock = threading.Lock()
        self.enabled = True
        self.running = False
        self.thread = None
        
        # Get screen size
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Set fail-safe
        pyautogui.FAILSAFE = enable_failsafe
        
        # Initialize mouse to center of screen
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        self.set_target(center_x, center_y)
    
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
                try:
                    pyautogui.moveTo(x, y)
                except pyautogui.FailSafeException:
                    # Handle fail-safe exception gracefully
                    print("[WARNING] Fail-safe triggered. Mouse control paused.")
                    time.sleep(1)  # Wait before resuming
            time.sleep(self.update_interval)
    
    def set_target(self, x: int, y: int):
        """
        Set new mouse target position.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        # Clamp to screen bounds with margin
        margin = 10
        x = max(margin, min(x, self.screen_width - margin))
        y = max(margin, min(y, self.screen_height - margin))
        
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