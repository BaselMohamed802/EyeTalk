"""
Filename: calibration_ui.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 1/23/2026

Description:
    Calibration UI ONLY - using PySide6 (Qt for Python).
    Responsibilities:
    1. Create a screen-sized window
    2. Draw one calibration dot at a known screen position
    3. Allow switching between dots programmatically (Using space bar to move onto the next dot)
"""

# Import necessary libraries
import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget
)
from PySide6.QtCore import Qt, QTimer, QPoint
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont
)


class CalibrationDotWidget(QWidget):
    def __init__(self, dot_position=(0.5, 0.5), dot_size=50):
        """
        Initialize the calibration dot widget.
        
        Args:
            dot_position: Tuple (x, y) normalized coordinates (0-1)
            dot_size: Size of the dot in pixels
        """
        super().__init__()
        
        # Window properties - FULL SCREEN
        self.setWindowFlags(
            Qt.FramelessWindowHint | 
            Qt.WindowStaysOnTopHint | 
            Qt.Tool
        )
        
        # Make it a solid color background
        self.setStyleSheet("background-color: black;")
        
        # Dot properties (will be updated when we know screen size)
        self.dot_size = dot_size
        self.dot_color = QColor(255, 255, 255)  # White dot
        self.ring_color = QColor(255, 0, 0)     # Red ring
        
        # Initialize position
        self.dot_position = QPoint(0, 0)
        self.set_dot_position(dot_position[0], dot_position[1])
        
        # Dot animation properties
        self.pulse_animation = False
        self.pulse_size = 0
        self.pulse_growing = True
        self.pulse_speed = 2
        
    def set_dot_position(self, x_norm, y_norm):
        """
        Set the dot position in normalized coordinates (0-1).
        
        Args:
            x_norm: Normalized x position (0 = left, 1 = right)
            y_norm: Normalized y position (0 = top, 1 = bottom)
        """
        # Store normalized position
        self.dot_position_norm = (x_norm, y_norm)
        
        # Convert to actual screen coordinates
        screen_width = self.width()
        screen_height = self.height()
        
        # Handle case where widget isn't sized yet
        if screen_width == 0 or screen_height == 0:
            screen_width = 1920  # Default fallback to FHD.
            screen_height = 1080
        
        self.dot_position = QPoint(
            int(x_norm * screen_width),
            int(y_norm * screen_height)
        )
        self.update()
        
    def resizeEvent(self, event):
        """Handle widget resize - update dot position."""
        super().resizeEvent(event)
        # Recalculate dot position based on new size
        if hasattr(self, 'dot_position_norm'):
            self.set_dot_position(self.dot_position_norm[0], 
                                 self.dot_position_norm[1])
        
    def enable_pulse_animation(self, enable=True):
        """Enable or disable the pulsing animation."""
        self.pulse_animation = enable
        if enable:
            self.start_pulse_animation()
    
    def start_pulse_animation(self):
        """Start the pulsing animation timer."""
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(self.update_pulse)
        self.pulse_timer.start(50)  # Update every 50ms
    
    def update_pulse(self):
        """Update the pulse animation."""
        if self.pulse_growing:
            self.pulse_size += self.pulse_speed
            if self.pulse_size > 20:
                self.pulse_growing = False
        else:
            self.pulse_size -= self.pulse_speed
            if self.pulse_size < 0:
                self.pulse_growing = True
        
        self.update()
    
    def paintEvent(self, event):
        """Paint the calibration dot."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw the calibration dot
        self.draw_calibration_dot(painter)
        
        # Draw instructions (subtle, at the bottom)
        self.draw_instructions(painter)
    
    def draw_calibration_dot(self, painter):
        """Draw the main calibration dot with ring."""
        # Draw outer ring (red)
        ring_pen = QPen(self.ring_color)
        ring_pen.setWidth(4)
        painter.setPen(ring_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(
            self.dot_position.x() - (self.dot_size // 2 + 10),
            self.dot_position.y() - (self.dot_size // 2 + 10),
            self.dot_size + 20,
            self.dot_size + 20
        )
        
        # Draw inner white dot
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self.dot_color))
        painter.drawEllipse(
            self.dot_position.x() - (self.dot_size // 2),
            self.dot_position.y() - (self.dot_size // 2),
            self.dot_size,
            self.dot_size
        )
        
        # Draw pulsing animation if enabled
        if self.pulse_animation and self.pulse_size > 0:
            pulse_pen = QPen(QColor(0, 255, 0, 100))  # Semi-transparent green
            pulse_pen.setWidth(2)
            painter.setPen(pulse_pen)
            painter.setBrush(Qt.NoBrush)
            
            current_size = self.dot_size + self.pulse_size
            painter.drawEllipse(
                self.dot_position.x() - (current_size // 2),
                self.dot_position.y() - (current_size // 2),
                current_size,
                current_size
            )
    
    def draw_instructions(self, painter):
        """Draw minimal instructions at the bottom."""
        painter.setPen(QColor(200, 50, 50)) # Hex: #c83232
        font = QFont("Arial", 14)
        painter.setFont(font)
        
        instructions = [
            "Look at the white dot that appears on the screen.",
            "Press ESC to exit calibration at any point.",
            "Press SPACE to move to next point."
        ]
        
        screen_height = self.height()
        for i, text in enumerate(instructions):
            text_width = painter.fontMetrics().horizontalAdvance(text)
            painter.drawText(
                (self.width() - text_width) // 2,
                screen_height - 50 - (len(instructions) - i - 1) * 30,
                text
            )


class CalibrationWindow(QMainWindow):
    def __init__(self, num_points=9, margin_screen_edge=0.1):
        """
        Initialize the CalibrationWindow with a specified number of points.

        Args:
            num_points (int, optional): Number of calibration points. Defaults to 9.

        Initializes the window, generates the calibration points, sets up the UI, and shows the first point.
        """
        super().__init__()
        
        try:
            self.num_points = num_points
        except ValueError:
            self.num_points = 9
        except Exception as e:
            print(f"Error Occured, defaulted to 9 points: {e}")
            self.num_points = 9
            
        self.current_point = 0

        self.margin_screen_edge = margin_screen_edge
        
        # Generate calibration points (normalized coordinates 0-1)
        self.calibration_points = self.generate_calibration_points(num_points)
        
        # Setup UI
        self.init_ui()
        
        # Show first calibration point
        self.show_calibration_point(0)
    
    def generate_calibration_points(self, num_points):
        """
        Generate a grid of calibration points.
        
        Args:
            num_points: Number of points (9 or 16 recommended)
            
        Returns:
            List of tuples (x_norm, y_norm) where values are 0-1
        """
        if num_points == 9:
            rows, cols = 3, 3
        elif num_points == 16:
            rows, cols = 4, 4
        else:
            rows, cols = 3, 3
        
        points = []
        for i in range(rows):
            for j in range(cols):
                x_norm = self.margin_screen_edge + j * ((1 - 2 * self.margin_screen_edge) / (cols - 1))
                y_norm = self.margin_screen_edge + i * ((1 - 2 * self.margin_screen_edge) / (rows - 1))
                points.append((x_norm, y_norm))
        
        return points
    
    def init_ui(self):
        """Initialize the UI."""
        # Get screen geometry
        screen = QApplication.primaryScreen().geometry()
        
        # Set window to full screen
        self.setGeometry(screen)
        self.setWindowTitle("Eye Tracking Calibration")
        
        # Create central widget with initial position at center
        self.central_widget = CalibrationDotWidget(dot_position=(0.5, 0.5))
        self.setCentralWidget(self.central_widget)
        
        # Window properties
        self.setWindowFlags(
            Qt.FramelessWindowHint | 
            Qt.WindowStaysOnTopHint
        )
        
        # Make it truly full screen
        self.showFullScreen()
    
    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_Escape:
            print("Calibration cancelled by user")
            self.close()
        elif event.key() == Qt.Key_Space:
            print(f"Moving from point {self.current_point + 1} to next")
            self.next_calibration_point()
        else:
            super().keyPressEvent(event)
    
    def show_calibration_point(self, point_index):
        """
        Show a specific calibration point.
        
        Args:
            point_index: Index of the point to show (0-based)
        """
        if 0 <= point_index < len(self.calibration_points):
            self.current_point = point_index
            x_norm, y_norm = self.calibration_points[point_index]
            
            # Update the dot position
            self.central_widget.set_dot_position(x_norm, y_norm)
            
            # Enable pulse animation during countdown
            self.central_widget.enable_pulse_animation(True)
            
            # Update window title with current point info
            self.setWindowTitle(
                f"Calibration - Point {point_index + 1}/{len(self.calibration_points)}"
            )
            
            print(f"\n--- Calibration Point {point_index + 1} ---")
            print(f"Position: ({x_norm:.2f}, {y_norm:.2f}) normalized")
            
            # Convert to actual screen coordinates for reference
            screen_width = self.width()
            screen_height = self.height()
            actual_x = int(x_norm * screen_width)
            actual_y = int(y_norm * screen_height)
            print(f"Screen coordinates: ({actual_x}, {actual_y})")
            
            return True
        return False
    
    def next_calibration_point(self):
        """Move to the next calibration point."""
        next_point = self.current_point + 1
        if next_point < len(self.calibration_points):
            self.show_calibration_point(next_point)
        else:
            print("\n" + "="*50)
            print("CALIBRATION UI COMPLETE!")
            print("All points have been displayed.")
            print("="*50)
            self.close()
    
    def closeEvent(self, event):
        """Handle window close event."""
        print("\nCalibration UI closed")
        event.accept()