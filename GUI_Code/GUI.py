import sys
from PySide6.QtWidgets import QApplication
from MainWindow_AR import Widget_AR
from MainWindow_EN import Widget_EN



# 1. Create application instance FIRST
app = QApplication(sys.argv)  

# 2. Create main window ( to switch from EN to AR GUI Just change Widget_EN to Widget_AR)
Main_window = Widget_AR()  
Main_window.setFixedSize(1024, 768)

# 3. Show window
Main_window.show()    

# 4. Execute application
app.exec()