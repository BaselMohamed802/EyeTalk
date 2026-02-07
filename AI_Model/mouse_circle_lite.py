import sys
import pyautogui
from PySide6 import QtWidgets, QtGui, QtCore

class LightCursorOverlay(QtWidgets.QWidget):
    def __init__(self, radius=40):
        super().__init__()
        self.radius = radius
        self.diameter = self.radius * 2 + 10
        
        # 1. Essential Window Flags for low overhead
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint |
            QtCore.Qt.WindowType.WindowStaysOnTopHint |
            QtCore.Qt.WindowType.Tool |
            QtCore.Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(self.diameter, self.diameter)

        # 2. Pre-render the circle once to save CPU cycles
        self.cached_pixmap = self.render_circle_pixmap()
        
        self.label = QtWidgets.QLabel(self)
        self.label.setPixmap(self.cached_pixmap)

        # 3. Use a slightly slower timer (30-60 FPS is plenty for a Pi)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_position)
        self.timer.start(16) # ~60 FPS

    def render_circle_pixmap(self):
        """Draws the 'Friendly' UI once and caches it."""
        pixmap = QtGui.QPixmap(self.diameter, self.diameter)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        
        center = QtCore.QPointF(self.diameter / 2, self.diameter / 2)
        
        # Simple solid but soft-colored ring (less math than gradients)
        pen = QtGui.QPen(QtGui.QColor(0, 255, 150, 150)) # Seafoam with 60% alpha
        pen.setWidth(4)
        painter.setPen(pen)
        painter.drawEllipse(center, self.radius, self.radius)
        
        painter.end()
        return pixmap

    def update_position(self):
        # Direct movement (no easing physics) to save CPU
        x, y = pyautogui.position()
        self.move(int(x - self.diameter/2), int(y - self.diameter/2))

if __name__ == "__main__":
    # Raspberry Pi performance tip: Enable hardware acceleration via ENV
    import os
    os.environ["QT_QUICK_BACKEND"] = "software" # Force software if GPU is busy
    
    app = QtWidgets.QApplication(sys.argv)
    overlay = LightCursorOverlay(radius=40)
    overlay.show()
    sys.exit(app.exec())