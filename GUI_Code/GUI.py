import sys
from PySide6.QtWidgets import QApplication

from MainWindow import Main_GUI_Window



# 1. Create application instance FIRST
app = QApplication(sys.argv)  

# 2. Create main window
Main_window = Main_GUI_Window()  


# 3. Show window
Main_window.show()    

# 4. Execute application
app.exec()