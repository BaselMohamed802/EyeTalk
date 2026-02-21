from PySide6.QtWidgets import QApplication, QStackedWidget
from PySide6.QtCore import Signal
import sys
from MainWindow_EN import Widget_EN
from MainWindow_AR import Widget_AR

app = QApplication(sys.argv)

# Function to adjust the Language Layout 
def switch_to_arabic():
    global current_language
    current_language = 1
    stack.setCurrentIndex(1)

def switch_to_english():
    global current_language
    current_language = 0
    stack.setCurrentIndex(0)
 

current_language = 0  # Change this to switch: 0=English, 1=Arabic

# Create stacked widget
stack = QStackedWidget()

stack.setWindowTitle("EyeTalk")

# Add both keyboards
english_widget = Widget_EN()
arabic_widget = Widget_AR()

stack.addWidget(english_widget)  # Index 0
stack.addWidget(arabic_widget)   # Index 1

english_widget.Language_Change_req.connect(switch_to_arabic)
arabic_widget.Language_Change_req.connect(switch_to_english)

stack.setFixedSize(1024, 768)
stack.show()

app.exec()
