
from PySide6.QtWidgets import QApplication, QMainWindow, QMenuBar, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QLineEdit
from PySide6.QtGui import QAction



class Main_GUI_Window(QMainWindow):
    def __init__(self,):
        super().__init__()
        
        # Giving the application Name and size
        self.setWindowTitle("Eye Talks")
        self.resize(600, 800)
        
        #Creating the Menu Bar
        Menu_Bar = self.menuBar()
        
        # Declaring the Help Option
        Help_Menu = Menu_Bar.addMenu("Help")
        Help_Bar_Action = Help_Menu.addAction("Bro, I need Help too")
        
        # Create central widget with layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QVBoxLayout()
        Label = QLabel()
        Line_Edit = QLineEdit()
        

        Layout_Box = QHBoxLayout()
        Layout_Box.addWidget(Label)
        Layout_Box.addWidget(Line_Edit)

        Buttons_Layout = QHBoxLayout()


        # Emergency_button 1
        Button_EM = QPushButton("PAIN")
        Button_EM.setFixedSize(100, 80)
        Buttons_Layout.addWidget(Button_EM, 5)
        Button_EM.clicked.connect(self.PAIN_Button_Pressed)
        layout.addWidget(Button_EM)
        
        # Emergency_button 2
        Button_EM2 = QPushButton("Need Water")
        Button_EM2.setFixedSize(100, 80)
        Buttons_Layout.addWidget(Button_EM2)
        Button_EM2.clicked.connect(self.WATER_Button_Pressed)
        layout.addWidget(Button_EM2)
        
        # Center the Button names
        layout.addStretch()  
        
        # Set layout to central widget
        central_widget.setLayout(layout)

    def PAIN_Button_Pressed(self):
        print("PAIN")
    def WATER_Button_Pressed(self):
        print("Need Water")






