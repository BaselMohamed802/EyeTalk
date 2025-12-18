# Eye Talks - Assistive Communication Application

## Project Structure
```
Eye_Talks/
├── `GUI.py` # Application entry point
├── `MainWindow.py` # Main window implementation
├── `Button_Holder.py` # Custom button component
└── `README.md` # Documentation
```

## Architecture Overview
```
┌─────────────────────────────────────────────────────────┐
│                    QApplication                          │
│                    (Event Loop)                          │
└──────────────────────────┬──────────────────────────────┘
                           │
                  ┌────────▼─────────┐
                  │  Main_GUI_Window │
                  │  (QMainWindow)   │
                  └────────┬─────────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐
│  Menu Bar    │    │  Layouts     │    │  Widgets    │
│  • Help Menu │    │  • QVBox     │    │  • QLabel   │
│  • Actions   │    │  • QHBox     │    │  • QLineEdit│
└──────────────┘    │              │    │  • Buttons  │
                    └──────────────┘    └─────────────┘
```

## 📝 Code Documentation

### 1. `GUI.py` - Application Entry Point
``` python
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
```
## Key Responsibilities:

- Initializes the Qt application framework

- Creates the main window instance

- Starts the event loop

- Manages application lifecycle

**Critical Implementation Details:**

`QApplication` must be instantiated before any widgets

`sys.argv` passes command-line arguments to Qt

`app.exec()` starts the main event loop (blocks until window closes)

# 2. MainWindow.py - Main Application Window
## Class Definition & Initialization
```python
from PySide6.QtWidgets import (QApplication, QMainWindow, QMenuBar, 
                              QWidget, QHBoxLayout, QVBoxLayout, 
                              QPushButton, QLabel, QLineEdit)
from PySide6.QtGui import QAction

class Main_GUI_Window(QMainWindow):
    def __init__(self):
        super().__init__()  # Initialize parent QMainWindow
        
        # Window Configuration
        self.setWindowTitle("Eye Talks")  # Sets window title
        self.resize(600, 800)             # Sets initial window size: 600x800 pixels
```
## Menu System Implementation  
```python
# Creating the Menu Bar
Menu_Bar = self.menuBar()  # Gets/Creates the window's menu bar

# Declaring the Help Option
Help_Menu = Menu_Bar.addMenu("Help")  # Creates "Help" dropdown menu
Help_Bar_Action = Help_Menu.addAction("You Can contact...")  # Adds menu item
```
## Menu Bar Components:

`menuBar():` Returns the window's menu bar (creates it if it doesn't exist)

`addMenu():` Creates a new dropdown menu

`addAction():` Adds a clickable item to the menu

# Layout System
```python
# Create central widget with layout
central_widget = QWidget()
self.setCentralWidget(central_widget)

# Create main vertical layout
layout = QVBoxLayout()

# Create horizontal layout for input components
Layout_Box = QHBoxLayout()
Label = QLabel()          # Empty label (can be configured)
Line_Edit = QLineEdit()   # Text input field
Layout_Box.addWidget(Label)
Layout_Box.addWidget(Line_Edit)

# Create horizontal layout for buttons
Buttons_Layout = QHBoxLayout()
```
## Layout Types:

`QVBoxLayout:` Vertical arrangement (top to bottom)

`QHBoxLayout:` Horizontal arrangement (left to right)

Nested Layouts: Combine for complex arrangements

## Button Implementation
```python
# Emergency_button 1
Button_EM = QPushButton("PAIN")
Button_EM.setFixedSize(100, 80)          # Fixed size: 100x80 pixels
Button_EM.clicked.connect(self.PAIN_Button_Pressed)  # Connect to handler
Buttons_Layout.addWidget(Button_EM, 5)   # Add to layout with stretch factor 5
layout.addWidget(Button_EM)              # Also add to main layout (duplicate)

# Emergency_button 2
Button_EM2 = QPushButton("Need Water")
Button_EM2.setFixedSize(100, 80)
Button_EM2.clicked.connect(self.WATER_Button_Pressed)
Buttons_Layout.addWidget(Button_EM2)
layout.addWidget(Button_EM2)
```
## Button Properties:

`setFixedSize(width, height):` Prevents button resizing

`clicked.connect():` Connects button press to handler function

`addWidget(widget, stretch):` Adds widget with optional stretch factor

## Event Handlers
```python
def PAIN_Button_Pressed(self):
    print("PAIN")  # Placeholder - should trigger actual action
    
def WATER_Button_Pressed(self):
    print("Need Water")  # Placeholder - should trigger actual action
```

## Final Layout Assembly
```Python
# Center the Button names
layout.addStretch()  # Adds expanding space to push widgets up

# Set layout to central widget
central_widget.setLayout(layout)
```
# Layout Completion:

`addStretch(): `Creates a flexible space that expands

`setLayout():` Applies the layout to the widget
