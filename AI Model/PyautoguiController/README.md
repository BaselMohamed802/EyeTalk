# PyAutoGUI Controller

A comprehensive Python wrapper class for PyAutoGUI automation tasks with enhanced functionality, error handling, and convenience methods.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Features](#core-features)
- [Mouse Operations](#mouse-operations)
- [Keyboard Operations](#keyboard-operations)
- [Screen Utilities](#screen-utilities)
- [Image Recognition](#image-recognition)
- [Error Handling](#error-handling)
- [Advanced Examples](#advanced-examples)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)

## Installation

### Prerequisites
- Python 3.7+
- Required packages:

```bash
pip install pyautogui pillow opencv-python
```

### Install the Module
Copy the `pyautogui_controller.py` file to your project directory, or install it as a package:

```python
# Direct usage
from pyautogui_controller import PyAutoGUIController
```

## Quick Start

```python
from pyautogui_controller import PyAutoGUIController

# Initialize controller
controller = PyAutoGUIController()

# Move and click
controller.move_mouse(100, 100, duration=0.5)
controller.click_mouse(100, 100)

# Type text
controller.type_text("Hello World!", interval=0.1)

# Take a screenshot
controller.take_screenshot(save_path="screenshot.png")
```

## Core Features

### Initialization
```python
controller = PyAutoGUIController()
# Output: [PyAutoGUI] Screen: 1920x1080
```

### Screen Properties
```python
# Get screen dimensions
width, height = controller.screen_size
print(f"Screen: {width}x{height}")

# Get current mouse position
x, y = controller.mouse_position
print(f"Mouse at: ({x}, {y})")
```

## Mouse Operations

### Basic Movement
```python
# Absolute movement
controller.move_mouse(500, 300, duration=1.0)  # Smooth movement over 1 second

# Relative movement
controller.move_mouse_relative(100, -50, duration=0.5)  # Move right 100px, up 50px

# Movement with bounds checking (automatically clamped to screen)
controller.move_mouse(2000, 2000)  # Will clamp to screen edge
```

### Click Operations
```python
# Single click
controller.click_mouse(100, 100)

# Double click
controller.click_mouse(200, 200, clicks_num=2, interval_bet_clicks=0.1)

# Right click
controller.click_mouse(300, 300, button='right')

# Middle click
controller.click_mouse(400, 400, button='middle')

# Multiple clicks with interval
controller.click_mouse(500, 500, clicks_num=3, interval_bet_clicks=0.2)
```

### Drag and Drop
```python
# Drag from start to end position
controller.drag_mouse(100, 100, 300, 300, duration=1.0)

# Drag with different button
controller.drag_mouse(200, 200, 400, 400, duration=0.5, button='right')
```

### Scrolling
```python
# Scroll up at current position
controller.scroll_mouse(scroll_amount=5)

# Scroll down
controller.scroll_mouse(scroll_amount=-5)

# Scroll at specific position
controller.scroll_mouse(500, 500, scroll_amount=10)
```

## Keyboard Operations

### Text Input
```python
# Type text with custom interval
controller.type_text("Hello, World!", interval=0.05)

# Type with default interval (0.1 seconds)
controller.type_text("This is a test.")
```

### Key Presses
```python
# Press single key
controller.press_key('enter')

# Press multiple times
controller.press_key('tab', presses=3, interval=0.2)

# Hold and release keys (for games or special operations)
controller.key_down('shift')
controller.key_down('a')
controller.key_up('a')
controller.key_up('shift')
```

### Hotkeys/Keyboard Shortcuts
```python
# Standard hotkeys
controller.keyboard_hotkey('ctrl', 'c')  # Copy
controller.keyboard_hotkey('ctrl', 'v')  # Paste
controller.keyboard_hotkey('ctrl', 's')  # Save
controller.keyboard_hotkey('alt', 'tab') # Switch windows
controller.keyboard_hotkey('win', 'd')   # Show desktop

# Complex hotkeys
controller.keyboard_hotkey('ctrl', 'shift', 'esc')  # Task Manager
```

## Screen Utilities

### Screenshots
```python
# Full screen screenshot
screenshot = controller.take_screenshot()
screenshot.show()  # Display the image

# Save screenshot to file
controller.take_screenshot(save_path="fullscreen.png")

# Screenshot of specific region (left, top, width, height)
controller.take_screenshot(region=(100, 100, 500, 300), save_path="region.png")
```

### Pixel Color Operations
```python
# Get pixel color at position
color = controller.get_pixel_color(500, 300)
print(f"RGB Color: {color}")  # Output: (255, 0, 0) for red

# Check if pixel matches expected color
is_red = controller.is_pixel_color(500, 300, (255, 0, 0), tolerance=5)
print(f"Is pixel red? {is_red}")

# Wait for specific color (useful for loading screens)
import time
def wait_for_color(x, y, expected_color, timeout=10):
    start_time = time.time()
    while time.time() - start_time < timeout:
        if controller.is_pixel_color(x, y, expected_color):
            return True
        time.sleep(0.1)
    return False
```

### Alerts
```python
# Show alert box
controller.alert("Process completed successfully!", title="Success")

# Simple notification
controller.alert("Please check the configuration.")
```

## Image Recognition

### Basic Image Finding
```python
# Find an image on screen
button_location = controller.locate_on_screen("button.png", confidence=0.9)

if button_location:
    left, top, width, height = button_location
    print(f"Found button at: ({left}, {top})")
    
    # Click on the found image
    controller.click_mouse(left + width//2, top + height//2)
else:
    print("Button not found")
```

### Waiting for Images
```python
# Wait for image to appear (with timeout)
loading_complete = controller.wait_for_image("loading_complete.png", timeout=30)

if loading_complete:
    print("Loading complete!")
    # Proceed with next steps
else:
    print("Timeout waiting for loading to complete")
```

### Pattern Matching with Confidence
```python
# Adjust confidence for fuzzy matching
# Lower confidence for matching similar elements
login_button = controller.locate_on_screen("login_button.png", confidence=0.7)

# Higher confidence for exact matches
exact_icon = controller.locate_on_screen("exact_icon.png", confidence=0.95)
```

### Region-Specific Search
```python
# Search only in specific area (improves performance)
# (left, top, width, height)
search_region = (100, 100, 800, 600)
result = controller.locate_on_screen("item.png", region=search_region, confidence=0.8)
```

## Error Handling

### Fail-Safe Mode
```python
# Enable fail-safe (move mouse to corner to abort)
controller.set_fail_safe()

# Disable for uninterrupted automation
controller.disable_fail_safe()

# Recommended: Use fail-safe for production, disable for testing
```

### Checking Operation Success
```python
# Most methods return boolean for success
success = controller.move_mouse(100, 100)
if success:
    print("Movement successful")
else:
    print("Movement failed")

# Click with error handling
if controller.click_mouse(200, 200):
    print("Click successful")
```

### Exception Handling in Automation Scripts
```python
try:
    # Attempt to find and click button
    location = controller.locate_on_screen("submit_button.png")
    if location:
        controller.click_mouse(location[0], location[1])
    else:
        print("Button not found, trying alternative...")
        # Fallback strategy
        
except Exception as e:
    print(f"Automation error: {e}")
    # Take screenshot for debugging
    controller.take_screenshot(save_path=f"error_{int(time.time())}.png")
```

## Advanced Examples

### Automated Form Filling
```python
def fill_form():
    controller = PyAutoGUIController()
    
    # Click name field
    controller.click_mouse(200, 300)
    controller.type_text("John Doe")
    
    # Tab to next field
    controller.press_key('tab')
    controller.type_text("john@example.com")
    
    # Tab to checkbox and click
    controller.press_key('tab')
    controller.click_mouse(250, 400)
    
    # Click submit
    controller.click_mouse(300, 500)
    print("Form submitted successfully")

fill_form()
```

### Game Automation
```python
def automate_game_actions():
    controller = PyAutoGUIController()
    controller.disable_fail_safe()  # Disable for uninterrupted gaming
    
    while True:
        # Check for enemy (red health bar)
        if controller.is_pixel_color(500, 100, (255, 50, 50), tolerance=20):
            # Attack
            controller.press_key('space')
            time.sleep(0.5)
        
        # Check for loot (gold color)
        if controller.is_pixel_color(600, 200, (255, 215, 0), tolerance=30):
            controller.press_key('e')  # Pick up
            time.sleep(1)
        
        # Move forward
        controller.key_down('w')
        time.sleep(0.1)
        controller.key_up('w')
        
        time.sleep(0.5)  # Main loop delay
```

### GUI Testing Automation
```python
class GUITester:
    def __init__(self):
        self.controller = PyAutoGUIController()
        
    def test_login_flow(self):
        """Test complete login flow"""
        print("Starting login flow test...")
        
        # Step 1: Find and click username field
        username_field = self.controller.wait_for_image("username_field.png", timeout=5)
        if username_field:
            self.controller.click_center(username_field)
            self.controller.type_text("testuser")
            print("✓ Username entered")
        
        # Step 2: Enter password
        self.controller.press_key('tab')
        self.controller.type_text("testpass123")
        print("✓ Password entered")
        
        # Step 3: Click login button
        login_btn = self.controller.wait_for_image("login_button.png", timeout=3)
        if login_btn:
            self.controller.click_center(login_btn)
            print("✓ Login button clicked")
            
            # Step 4: Verify success
            success_msg = self.controller.wait_for_image("login_success.png", timeout=10)
            if success_msg:
                print("✓ Login successful!")
                return True
        
        print("✗ Login test failed")
        return False

# Run the test
tester = GUITester()
tester.test_login_flow()
```

### Web Scraping Automation
```python
def automate_web_scraping():
    controller = PyAutoGUIController()
    
    # Open browser (assuming browser is already open)
    controller.keyboard_hotkey('alt', 'tab')
    time.sleep(1)
    
    # Navigate through pages
    for page in range(1, 6):
        print(f"Scraping page {page}")
        
        # Take screenshot of page
        controller.take_screenshot(save_path=f"page_{page}.png")
        
        # Scroll down
        for _ in range(3):
            controller.scroll_mouse(scroll_amount=-500)
            time.sleep(0.5)
        
        # Click next page if not last page
        if page < 5:
            next_button = controller.wait_for_image("next_button.png", timeout=5)
            if next_button:
                controller.click_center(next_button)
                time.sleep(2)  # Wait for page load
```

## Best Practices

### 1. **Add Delays Between Actions**
```python
import time

def safe_automation():
    controller = PyAutoGUIController()
    
    # Always add small delays between actions
    controller.click_mouse(100, 100)
    time.sleep(0.5)  # Wait for UI response
    
    controller.type_text("Hello")
    time.sleep(0.3)
    
    controller.press_key('enter')
    time.sleep(1)  # Wait for action to complete
```

### 2. **Use Coordinates Relatively**
```python
# Instead of hardcoding coordinates:
def click_relative_to_window(window_x, window_y, offset_x, offset_y):
    """Click at position relative to window top-left"""
    controller.click_mouse(window_x + offset_x, window_y + offset_y)
```

### 3. **Implement Retry Logic**
```python
def robust_click(image_path, max_attempts=3):
    """Try clicking with retries"""
    for attempt in range(max_attempts):
        location = controller.locate_on_screen(image_path)
        if location:
            if controller.click_center(location):
                return True
        time.sleep(1)  # Wait before retry
    return False
```

### 4. **Log Everything**
```python
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def logged_automation():
    controller = PyAutoGUIController()
    
    logging.info("Starting automation")
    if controller.move_mouse(100, 100):
        logging.info("Mouse moved to (100, 100)")
    
    # Take screenshot on error
    try:
        controller.click_mouse(200, 200)
    except Exception as e:
        logging.error(f"Click failed: {e}")
        controller.take_screenshot(save_path="error_screenshot.png")
```

### 5. **Create Configuration**
```python
class AutomationConfig:
    DEFAULT_DELAY = 0.5
    CLICK_DELAY = 0.3
    TYPE_DELAY = 0.1
    CONFIDENCE_LEVEL = 0.8
    SCREENSHOT_DIR = "screenshots"
    
class AutomatedTask:
    def __init__(self):
        self.controller = PyAutoGUIController()
        self.config = AutomationConfig()
```

## Troubleshooting

### Common Issues and Solutions

1. **Images not found**
   ```python
   # Increase confidence or check image quality
   controller.locate_on_screen("image.png", confidence=0.7)
   
   # Use region to limit search area
   controller.locate_on_screen("image.png", region=(0, 0, 500, 500))
   ```

2. **Mouse moves too fast**
   ```python
   # Increase duration for smooth movement
   controller.move_mouse(100, 100, duration=1.0)
   ```

3. **Script running too fast**
   ```python
   # Add pauses between actions
   import time
   time.sleep(0.5)  # Half second pause
   ```

4. **Screen resolution issues**
   ```python
   # Always use relative coordinates or image recognition
   # Hardcoded coordinates may fail on different screens
   ```

## API Reference

### Core Methods
| Method | Description | Returns |
|--------|-------------|---------|
| `move_mouse(x, y, duration)` | Move mouse to coordinates | `bool` |
| `move_mouse_relative(x, y, duration)` | Move mouse by offset | `None` |
| `click_mouse(x, y, clicks, interval, button)` | Click at position | `bool` |
| `drag_mouse(start_x, start_y, end_x, end_y, duration, button)` | Drag mouse | `bool` |
| `scroll_mouse(x, y, scroll_amount)` | Scroll mouse wheel | `None` |

### Keyboard Methods
| Method | Description |
|--------|-------------|
| `type_text(text, interval)` | Type text |
| `press_key(key, presses, interval)` | Press key |
| `key_down(key)` | Hold key down |
| `key_up(key)` | Release key |
| `keyboard_hotkey(*keys)` | Press hotkey |

### Screen Methods
| Method | Description | Returns |
|--------|-------------|---------|
| `take_screenshot(region, save_path)` | Take screenshot | `Image` or `None` |
| `get_pixel_color(x, y)` | Get pixel color | `(R, G, B)` |
| `is_pixel_color(x, y, expected_color, tolerance)` | Compare colors | `bool` |

### Image Recognition
| Method | Description | Returns |
|--------|-------------|---------|
| `locate_on_screen(image_path, confidence, region)` | Find image | `(left, top, width, height)` or `None` |
| `wait_for_image(image_path, timeout, confidence)` | Wait for image | Location or `None` |

### Properties
| Property | Description |
|----------|-------------|
| `screen_size` | Screen dimensions `(width, height)` |
| `mouse_position` | Current mouse position `(x, y)` |

## Contributing

Feel free to extend this controller with additional functionality:

1. Add window management features
2. Implement OCR capabilities
3. Add support for multiple monitors
4. Create gesture recognition
5. Add macro recording/playback

## License

This module is provided as-is. Modify and use according to your project needs.

---

**Note**: Always test automation scripts in a controlled environment. Use fail-safe mode (`controller.set_fail_safe()`) when running important automations to quickly abort by moving the mouse to a corner.