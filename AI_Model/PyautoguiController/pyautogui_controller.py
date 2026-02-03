"""
Filename: pyautogui_controller.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 1/30/2026

Description:
    Simple pyautogui controller for cursor movement functions and some minimal keyboard functions,
    and screen utilities.
"""

# Import necessary libraries
import pyautogui
import time
from typing import Optional, Tuple
from PIL import Image

class PyAutoGUIController:
    """A Controller class for PyAutoGUI automation tasks."""

    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"[PyAutoGUI] Screen: {self.screen_width}x{self.screen_height}")
    
    def move_mouse(self, x: int, y: int, duration: float = 0.0) -> bool:
        """
        Move mouse cursor to position.
        
        Returns:
            bool: True if successful, False if out of bounds
        """
        try:
            # Clamp to screen bounds
            x = max(0, min(x, self.screen_width - 1))
            y = max(0, min(y, self.screen_height - 1))
            pyautogui.moveTo(x, y, duration=max(0.0, duration))
            return True
        except Exception as e:
            print(f"Error moving mouse: {e}")
            return False
    
    def move_mouse_relative(self, x: int, y: int, duration: float = 0.0) -> None:
        """
        Move the mouse cursor by a given relative offset.

        Args:
            x (int): The x-coordinate offset to move by.
            y (int): The y-coordinate offset to move by.
            duration (float, optional): The duration of the move in seconds. Defaults to 0.0.

        Returns:
            None
        """
        # Note: Relative movement doesn't need screen clamping in the same way
        # But we should check that the move won't go off-screen
        current_x, current_y = pyautogui.position()
        new_x = max(0, min(current_x + x, self.screen_width - 1))
        new_y = max(0, min(current_y + y, self.screen_height - 1))
        
        # Calculate relative movement to the clamped position
        rel_x = new_x - current_x
        rel_y = new_y - current_y
        
        pyautogui.moveRel(rel_x, rel_y, duration=max(0.0, duration))
        
    @property
    def screen_size(self) -> Tuple[int, int]:
        """Get current screen dimensions."""
        return pyautogui.size()

    @property
    def mouse_position(self) -> Tuple[int, int]:
        """Get current mouse position."""
        return pyautogui.position()
    
    def set_fail_safe(self) -> None:
        """
        Enable pyautogui's fail-safe mode.

        When fail-safe mode is enabled, the mouse will move to (0, 0) when an exception is raised in user code.

        Returns:
            None
        """
        pyautogui.FAILSAFE = True

    def disable_fail_safe(self) -> None:
        """
        Disable pyautogui's fail-safe mode.

        Returns:
            None
        """
        pyautogui.FAILSAFE = False

    def drag_mouse(self, start_x: int, start_y: int, end_x: int, end_y: int, 
                   duration: float = 0.5, button: str = 'left') -> bool:
        """Drag mouse from start to end position."""
        # First move to start position
        if not self.move_mouse(start_x, start_y):
            return False
        
        # Clamp end position to screen bounds
        end_x = max(0, min(end_x, self.screen_width - 1))
        end_y = max(0, min(end_y, self.screen_height - 1))
        
        try:
            pyautogui.dragTo(end_x, end_y, duration=duration, button=button)
            return True
        except Exception as e:
            print(f"Error dragging mouse: {e}")
            return False

    def type_text(self, text: str, interval: float = 0.1) -> None:
        """Type text with specified interval between keystrokes."""
        pyautogui.write(text, interval=interval)

    def press_key(self, key: str, presses: int = 1, interval: float = 0.1) -> None:
        """Press a key multiple times."""
        pyautogui.press(key, presses=presses, interval=interval)

    def key_down(self, key: str) -> None:
        """Hold a key down."""
        pyautogui.keyDown(key)

    def key_up(self, key: str) -> None:
        """Release a key."""
        pyautogui.keyUp(key)

    def click_mouse(self, x: int, 
                    y: int, 
                    clicks_num: int = 1, 
                    interval_bet_clicks: float = 0.0, 
                    button: str = 'left') -> bool:
        """
        Simulate a mouse click at the given position.

        Args:
            x (int): The x-coordinate of the click.
            y (int): The y-coordinate of the click.
            clicks_num (int, optional): The number of clicks to simulate. Defaults to 1.
            interval_bet_clicks (float, optional): The time in seconds to wait between clicks. Defaults to 0.0.
            button (str, optional): The mouse button to use. Can be 'left', 'middle', or 'right'. Defaults to 'left'.

        Returns:
            bool: True if click was successful
        """
        try:
            # Clamp coordinates to screen bounds
            x = max(0, min(x, self.screen_width - 1))
            y = max(0, min(y, self.screen_height - 1))
            
            pyautogui.click(x, y, clicks=clicks_num, 
                           interval=interval_bet_clicks, button=button)
            return True
        except Exception as e:
            print(f"Error clicking mouse: {e}")
            return False

    def scroll_mouse(self, x: Optional[int] = None, 
                     y: Optional[int] = None, 
                     scroll_amount: int = 1) -> None:
        """
        Simulate a mouse scroll event.

        Args:
            x (int, optional): The x-coordinate of the scroll event.
            y (int, optional): The y-coordinate of the scroll event.
            scroll_amount (int, optional): The amount to scroll. Defaults to 1.

        Returns:
            None
        """
        # If x and y are provided, clamp them
        if x is not None and y is not None:
            x = max(0, min(x, self.screen_width - 1))
            y = max(0, min(y, self.screen_height - 1))
            pyautogui.scroll(scroll_amount, x=x, y=y)
        else:
            pyautogui.scroll(scroll_amount)

    def keyboard_hotkey(self, *keys: str) -> None:
        """
        Simulate a keyboard hotkey event.

        Args:
            *keys: Variable number of keys to press simultaneously

        Returns:
            None
        """
        pyautogui.hotkey(*keys)

    # ===================================================================
    #                          Screenshot Functionality
    # ===================================================================

    def take_screenshot(self, region: Optional[Tuple[int, int, int, int]] = None, 
                        save_path: Optional[str] = None) -> Optional[Image.Image]:
        """
        Take a screenshot.
        
        Args:
            region: (left, top, width, height) tuple
            save_path: Path to save the screenshot
        
        Returns:
            PIL Image object if no save_path provided
        """
        screenshot = pyautogui.screenshot(region=region)
        if save_path:
            screenshot.save(save_path)
            return None
        return screenshot

    def locate_on_screen(self, image_path: str, confidence: float = 0.8, 
                         region: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[int, int, int, int]]:
        """
        Locate an image on screen.
        
        Returns:
            (left, top, width, height) if found, None otherwise
        """
        try:
            return pyautogui.locateOnScreen(image_path, confidence=confidence, region=region)
        except pyautogui.ImageNotFoundException:
            return None
    
    def wait_for_image(self, image_path: str, timeout: float = 10.0, 
                       confidence: float = 0.8) -> Optional[Tuple[int, int, int, int]]:
        """Wait for an image to appear on screen."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            location = self.locate_on_screen(image_path, confidence=confidence)
            if location:
                return location
            time.sleep(0.5)
        return None
        
    # ===================================================================
    #                          Utility Functions
    # ===================================================================

    def alert(self, message: str, title: str = "Alert") -> None:
        """Show an alert box."""
        pyautogui.alert(message, title)

    def get_pixel_color(self, x: int, y: int) -> Tuple[int, int, int]:
        """Get RGB color of a pixel."""
        return pyautogui.pixel(x, y)

    def is_pixel_color(self, x: int, y: int, expected_color: Tuple[int, int, int], 
                       tolerance: int = 10) -> bool:
        """Check if pixel color matches expected color within tolerance."""
        actual_color = self.get_pixel_color(x, y)
        return all(abs(a - e) <= tolerance for a, e in zip(actual_color, expected_color))