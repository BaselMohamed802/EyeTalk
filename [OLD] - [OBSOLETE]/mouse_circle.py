import sys
import pyautogui
from PySide6 import QtWidgets, QtGui, QtCore

class CursorOverlay(QtWidgets.QWidget):
    def __init__(self, radius=40):
        super().__init__()
        self.radius = radius
        self.diameter = (self.radius * 2) + 20 # Extra padding for the glow
        
        # Window Setup
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint |
            QtCore.Qt.WindowType.WindowStaysOnTopHint |
            QtCore.Qt.WindowType.Tool |
            QtCore.Qt.WindowType.WindowTransparentForInput # Clicks pass through the circle
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(self.diameter, self.diameter)

        # Smooth Follow Variables
        self.target_x, self.target_y = pyautogui.position()
        self.current_x, self.current_y = self.target_x, self.target_y

        # Animation Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_logic)
        self.timer.start(10) # ~100 FPS for buttery smoothness

    def update_logic(self):
        # Get mouse position
        tx, ty = pyautogui.position()
        
        # Easing formula: Current + (Target - Current) * EaseFactor
        # This makes the circle "float" toward the mouse
        ease = 0.25 
        self.current_x += (tx - self.current_x) * ease
        self.current_y += (ty - self.current_y) * ease

        self.move(int(self.current_x - self.diameter/2), 
                  int(self.current_y - self.diameter/2))
        self.update() # Triggers paintEvent

    def paintEvent(self, event):
        """This is where the 'Friendly' UI is drawn."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        center = QtCore.QPointF(self.diameter / 2, self.diameter / 2)
        
        # 1. Create a Soft Radial Gradient Glow
        gradient = QtGui.QRadialGradient(center, self.radius)
        gradient.setColorAt(0.0, QtGui.QColor(0, 255, 150, 40))  # Center (faint green)
        gradient.setColorAt(0.7, QtGui.QColor(0, 255, 150, 100)) # Middle ring
        gradient.setColorAt(1.0, QtGui.QColor(0, 255, 150, 0))   # Edge (transparent)

        # 2. Draw the Glow
        painter.setBrush(gradient)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawEllipse(center, self.radius, self.radius)

        # 3. Draw a Crisp, Thin Inner Border
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 180)) # Semi-transparent white
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawEllipse(center, self.radius * 0.7, self.radius * 0.7)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    overlay = CursorOverlay(radius=50)
    overlay.show()
    sys.exit(app.exec())