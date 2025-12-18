
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PySide6.QtGui import QAction


class Main_GUI_Window(QWidget):
    """
        Main widget class that implements a virtual keyboard.
        Allows users to type text by clicking letter buttons.

    """ 
    def __init__(self,):
        """Initialize the widget and set up the virtual keyboard interface."""
        super().__init__() # Initialize parent QWidget class
     
        # ============================================
        # WINDOW SETUP
        # ============================================
        self.setWindowTitle("EyeTalk")  # Set window title
        self.current_text = ""  # Variable to store the typed text (starts empty)

        
        # ============================================
        # TEXT DISPLAY AREA
        # ============================================
        
        # Label to show "Your Text:" prompt
        Display_Label = QLabel("Your Text:")
        
        # Label that displays the typed text (updates as user types)
        self.text_holder_label = QLabel("")
        self.text_holder_label.setStyleSheet("""
        QLabel {
            padding: 15px;
            font-size: 20px;
            min-width: 300px;
            min-height: 100px;
            border-radius: 8px;
        }
    """)
    
        # ============================================
        # KEYBOARD BUTTONS - ROW 1 (QWERTYUIOP)
        # ============================================
        # Create buttons for letters Q through P (top row of keyboard)

        Letter_ButtonQ = QPushButton("Q")
        Letter_ButtonQ.clicked.connect(self.Display_Letter)

        Letter_ButtonW = QPushButton("W")
        Letter_ButtonW.clicked.connect(self.Display_Letter)

        Letter_ButtonE = QPushButton("E")
        Letter_ButtonE.clicked.connect(self.Display_Letter)

        Letter_ButtonR = QPushButton("R")
        Letter_ButtonR.clicked.connect(self.Display_Letter)

        Letter_ButtonT = QPushButton("T")
        Letter_ButtonT.clicked.connect(self.Display_Letter)

        Letter_ButtonY = QPushButton("Y")
        Letter_ButtonY.clicked.connect(self.Display_Letter)

        Letter_ButtonU = QPushButton("U")
        Letter_ButtonU.clicked.connect(self.Display_Letter)

        Letter_ButtonI = QPushButton("I")
        Letter_ButtonI.clicked.connect(self.Display_Letter)

        Letter_ButtonO = QPushButton("O")
        Letter_ButtonO.clicked.connect(self.Display_Letter)

        Letter_ButtonP = QPushButton("P")
        Letter_ButtonP.clicked.connect(self.Display_Letter)

        # Delete button (removes last character)
        Letter_Button_Delete = QPushButton("⌫ Delete")
        Letter_Button_Delete.clicked.connect(self.Delete_Button)

        # ============================================
        # KEYBOARD BUTTONS - ROW 2 (ASDFGHJKL)
        # ============================================
        # Create buttons for letters A through L (middle row of keyboard)

        Letter_ButtonA = QPushButton("A")
        Letter_ButtonA.clicked.connect(self.Display_Letter)

        Letter_ButtonS = QPushButton("S")
        Letter_ButtonS.clicked.connect(self.Display_Letter)

        Letter_ButtonD = QPushButton("D")
        Letter_ButtonD.clicked.connect(self.Display_Letter)

        Letter_ButtonF = QPushButton("F")
        Letter_ButtonF.clicked.connect(self.Display_Letter)

        Letter_ButtonG = QPushButton("G")
        Letter_ButtonG.clicked.connect(self.Display_Letter)

        Letter_ButtonH = QPushButton("H")
        Letter_ButtonH.clicked.connect(self.Display_Letter)

        Letter_ButtonJ = QPushButton("J")
        Letter_ButtonJ.clicked.connect(self.Display_Letter)

        Letter_ButtonK = QPushButton("K")
        Letter_ButtonK.clicked.connect(self.Display_Letter)

        Letter_ButtonL = QPushButton("L")
        Letter_ButtonL.clicked.connect(self.Display_Letter)

        # Serves no purpose atm, will be used for the TTS in future
        Letter_Button_Enter = QPushButton("⏎ Enter") 
        Letter_Button_Enter.clicked.connect(self.Display_Letter)

        # ============================================
        # KEYBOARD BUTTONS - ROW 3 (ZXCVBNM)
        # ============================================
        # Create buttons for letters Z through M (bottom row of keyboard)

        Letter_ButtonZ = QPushButton("Z")
        Letter_ButtonZ.clicked.connect(self.Display_Letter)

        Letter_ButtonX = QPushButton("X")
        Letter_ButtonX.clicked.connect(self.Display_Letter)

        Letter_ButtonC = QPushButton("C")
        Letter_ButtonC.clicked.connect(self.Display_Letter)

        Letter_ButtonV = QPushButton("V")
        Letter_ButtonV.clicked.connect(self.Display_Letter)

        Letter_ButtonB = QPushButton("B")
        Letter_ButtonB.clicked.connect(self.Display_Letter)

        Letter_ButtonN = QPushButton("N")
        Letter_ButtonN.clicked.connect(self.Display_Letter)

        Letter_ButtonM = QPushButton("M")
        Letter_ButtonM.clicked.connect(self.Display_Letter)
        

        # Space bar button (adds space between words)
        Space_Bar = QPushButton("⎵ Space")
        Space_Bar.clicked.connect(self.Space_Bar)

        # ============================================
        # Emergency BUTTONS 
        # ============================================
        # Create pre-defined sentences in a button


        #insert your EM_Buttons here 

        # HORIZONTAL LAYOUT: Text display area
        # Contains the "Your Text:" label and the text display label
        H_layout = QHBoxLayout()
        H_layout.setSpacing(20)
        H_layout.addWidget(Display_Label)
        H_layout.addWidget(self.text_holder_label)
       
        
        # KEYBOARD ROW 1 LAYOUT: Q W E R T Y U I O P + Delete
        Keyboard_Layout_ROW1 = QHBoxLayout()
        Keyboard_Layout_ROW1.addWidget(Letter_ButtonQ)
        Keyboard_Layout_ROW1.addWidget(Letter_ButtonW)
        Keyboard_Layout_ROW1.addWidget(Letter_ButtonE)
        Keyboard_Layout_ROW1.addWidget(Letter_ButtonR)
        Keyboard_Layout_ROW1.addWidget(Letter_ButtonT)
        Keyboard_Layout_ROW1.addWidget(Letter_ButtonY)
        Keyboard_Layout_ROW1.addWidget(Letter_ButtonU)
        Keyboard_Layout_ROW1.addWidget(Letter_ButtonI)
        Keyboard_Layout_ROW1.addWidget(Letter_ButtonO)
        Keyboard_Layout_ROW1.addWidget(Letter_ButtonP)
        Keyboard_Layout_ROW1.addWidget(Letter_Button_Delete)
        
        # KEYBOARD ROW 2 LAYOUT: A S D F G H J K L + Enter
        Keyboard_Layout_ROW2 = QHBoxLayout()
        Keyboard_Layout_ROW2.addWidget(Letter_ButtonA)
        Keyboard_Layout_ROW2.addWidget(Letter_ButtonS)
        Keyboard_Layout_ROW2.addWidget(Letter_ButtonD)
        Keyboard_Layout_ROW2.addWidget(Letter_ButtonF)
        Keyboard_Layout_ROW2.addWidget(Letter_ButtonG)
        Keyboard_Layout_ROW2.addWidget(Letter_ButtonH)
        Keyboard_Layout_ROW2.addWidget(Letter_ButtonJ)
        Keyboard_Layout_ROW2.addWidget(Letter_ButtonK)
        Keyboard_Layout_ROW2.addWidget(Letter_ButtonL)
        Keyboard_Layout_ROW2.addWidget(Letter_Button_Enter)
        
        # KEYBOARD ROW 3 LAYOUT: Z X C V B N M
        Keyboard_Layout_ROW3 = QHBoxLayout()
        Keyboard_Layout_ROW3.addWidget(Letter_ButtonZ)  
        Keyboard_Layout_ROW3.addWidget(Letter_ButtonX)
        Keyboard_Layout_ROW3.addWidget(Letter_ButtonC)
        Keyboard_Layout_ROW3.addWidget(Letter_ButtonV)
        Keyboard_Layout_ROW3.addWidget(Letter_ButtonB)
        Keyboard_Layout_ROW3.addWidget(Letter_ButtonN)
        Keyboard_Layout_ROW3.addWidget(Letter_ButtonM)
        
        # KEYBOARD ROW 4 LAYOUT: Space bar
        Keyboard_Layout_ROW4 = QHBoxLayout()
        Keyboard_Layout_ROW4.addWidget(Space_Bar)
        
        # MAIN VERTICAL LAYOUT: Stack all layouts vertically
        V_layout = QVBoxLayout()
        V_layout.addLayout(H_layout)           # Text display area
        V_layout.addLayout(Keyboard_Layout_ROW1)  # Keyboard row 1
        V_layout.addLayout(Keyboard_Layout_ROW2)  # Keyboard row 2
        V_layout.addLayout(Keyboard_Layout_ROW3)  # Keyboard row 3
        V_layout.addLayout(Keyboard_Layout_ROW4)  # Keyboard row 4 (space bar)
        
        # Set the main layout for the widget
        self.setLayout(V_layout)
    
    # ============================================
    # BUTTON HANDLER FUNCTIONS
    # ============================================
    
    def Display_Letter(self):
        """
        Handles letter button clicks.
        Gets the clicked button's text and appends it to the current text.

        """
        button = self.sender()  # Get the button that was clicked
        
        if button:  # Safety check 
            Letter_pressed = button.text()  # Get the letter/text from the button
            
            self.current_text += Letter_pressed
            
            # Update the display to show the complete typed text
            self.text_holder_label.setText(f"{self.current_text}")
    
    def Space_Bar(self):
        """
        Handles space bar button click.
        Adds a space character to the current text.
    
        """
        # Add a space character to separate words
        self.current_text += " "
        
        # Update the display
        self.text_holder_label.setText(f"{self.current_text}")
    
    def Delete_Button(self):
        """
        Handles delete button click.
        Removes the last character from the current text.

        """
        # Remove the last character (if there is any text)
        if self.current_text:  # Only delete if there's text to delete
            self.current_text = self.current_text[:-1]
            
            # Update the display
            self.text_holder_label.setText(f"{self.current_text}")

    def Clear_Whole_Text(self):
        """
        Handles Clear button click.
        Removes the all character from the current text.

        """
        
        if self.current_text:  # Only Clear if there's text to delete
            self.current_text = ""
            
            # Update the display
            self.text_holder_label.setText(f"{self.current_text}")





