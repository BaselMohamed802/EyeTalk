"""
Filename: calibration_ui_test.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 1/23/2026

Description:
    Simple script to only test the calibration UI if it is working correctly.
    This is to help debugging in case of an error.
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
import os

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from calibration_ui import CalibrationWindow, CalibrationDotWidget

def main():
    """Main function to run the calibration UI."""
    print("="*60)
    print("CALIBRATION UI - STEP 1")
    print("="*60)
    print("\nThis UI has exactly three responsibilities:")
    print("1. Create a screen-sized window")
    print("2. Draw one calibration dot at a known screen position")
    print("3. Allow switching between dots programmatically")
    print("\nControls:")
    print("- SPACE: Move to next calibration point")
    print("- ESC: Exit calibration")
    print("="*60 + "\n")
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create and show calibration window
    window = CalibrationWindow(num_points=9, margin_screen_edge=0.05)
    
    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()