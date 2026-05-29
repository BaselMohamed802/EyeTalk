"""
Filename: dwell_detector.py
Description: 
    Dwell-to-click module. 
    Fires a click if the provided X/Y coordinates stay within a small 
    pixel radius for a specified amount of time.
"""

import time

class DwellDetector:
    def __init__(self, dwell_time_sec=2.0, tolerance_px=30.0):
        """
        Parameters:
        - dwell_time_sec: Seconds the cursor must remain still to click.
        - tolerance_px: How much the cursor is allowed to jiggle (in pixels) 
                        while still counting as "still".
        """
        self.dwell_time_sec = float(dwell_time_sec)
        self.tolerance_px = float(tolerance_px)
        
        self.anchor_x = None
        self.anchor_y = None
        self.start_time = None
        self.click_fired = False

    def process(self, current_x: float, current_y: float) -> bool:
        """
        Feed this the current screen_x and screen_y every frame.
        Returns True exactly once when the dwell time is reached.
        """
        if current_x is None or current_y is None:
            return False

        # Initialize anchor on first frame
        if self.anchor_x is None or self.anchor_y is None:
            self.anchor_x = current_x
            self.anchor_y = current_y
            self.start_time = time.time()
            return False

        # Calculate Euclidean distance from the anchor point
        dist = ((current_x - self.anchor_x)**2 + (current_y - self.anchor_y)**2)**0.5

        if dist > self.tolerance_px:
            # The user moved the cursor outside the tolerance zone.
            # Reset the anchor to the new position and restart the timer.
            self.anchor_x = current_x
            self.anchor_y = current_y
            self.start_time = time.time()
            self.click_fired = False
            return False

        # If the cursor is still inside the tolerance zone and hasn't clicked yet
        if not self.click_fired:
            elapsed = time.time() - self.start_time
            if elapsed >= self.dwell_time_sec:
                self.click_fired = True  # Prevent rapid continuous clicking
                return True              # FIRE CLICK!

        return False