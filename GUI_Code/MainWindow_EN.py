"""
EyeTalk - Virtual Keyboard Widget
This widget creates a virtual keyboard interface for typing text using buttons.
Each button press appends characters to form words and sentences.
"""

from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap, QGuiApplication
from TTS_Engine import TTS 

class Widget_EN(QWidget):
    """
    Main widget class that implements a virtual keyboard.
    Allows users to type text by clicking letter buttons.
    """
    
    Language_Change_req = Signal() # Signal Handler 

    def __init__(self):
        """Initialize the widget and set up the virtual keyboard interface."""
        super().__init__()  # Initialize parent QWidget class
        
        # ============================================
        # DYNAMIC SIZING BASED ON SCREEN RESOLUTION
        # ============================================
        screen = QGuiApplication.primaryScreen()
        size = screen.size()
        screen_w, screen_h = size.width(), size.height()
        
        # Reference: your 10" 1024x600 touchscreen
        ref_w, ref_h = 600, 400
        
        # Compute scale as average of width and height ratios (like head_tracking_gui)
        scale_w = screen_w / ref_w
        scale_h = screen_h / ref_h
        self.scale = (scale_w + scale_h) / 2.0
        self.scale = max(0.7, min(self.scale, 1.8))  # clamp to avoid extremes
        
        # Sizes derived from scale
        font_size = int(16 * self.scale)
        padding = int(5 * self.scale)
        delete_font = int(14 * self.scale)
        enter_font = int(14 * self.scale)
        space_font = int(14 * self.scale)
        space_padding = int(5 * self.scale)
        
        # Emergency buttons
        em_font = int(18 * self.scale)         
        em_padding = int(10 * self.scale)      
        em_height = int(19 * self.scale)        
        em_min_width = int(100 * self.scale)    
        
        logo_width = int(400 * self.scale)
        logo_height = int(100 * self.scale)
        layout_spacing = int(5 * self.scale)
        layout_margin = int(10 * self.scale)
        label_font = int(15 * self.scale)
        text_holder_font = int(20 * self.scale)
        text_holder_padding = int(30 * self.scale)
        
        # ============================================
        # WINDOW SETUP
        # ============================================
        self.current_text = ""  # Variable to store the typed text (starts empty)

        # ============================================
        # TEXT DISPLAY AREA
        # ============================================

        #Calling the text-to-speech class 
        self.tts = TTS()

        #Creating a Label to hold Thebes's Logo 
        Thebes_Logo = QPixmap(r"M:\University\Level 4\Shit_Project\EyeTalk\Documentation\Eye_Talk_Logo.png")
        Thebes_Label = QLabel(self)
        Scaled = Thebes_Logo.scaled(logo_width, logo_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        Thebes_Label.setPixmap(Scaled)
        Thebes_Label.setAlignment(Qt.AlignCenter)  # <--- CENTRE THE LOGO

        # Label to show "Your Text:" prompt
        Display_Label = QLabel("Your Text")
        Display_Label.setObjectName("display_label")
        Display_Label.setStyleSheet(f"""
            QLabel#display_label {{
                color: #FF4500;
                qproperty-alignment: AlignCenter;
                font-size: {label_font}px;
                max-height: {label_font}px;
            }}
        """)

        # Label that displays the typed text (updates as user types)
        self.text_holder_label = QLabel("")
        self.text_holder_label.setObjectName("text_holder")
        self.text_holder_label.setStyleSheet(f"""
            QLabel#text_holder {{
                padding: {text_holder_padding}px;
                font-size: {text_holder_font}px;
                border-radius: 5px;
                background-color: #75706f;
            }}
        """)
    
        # ============================================
        # KEYBOARD BUTTONS - ROW 1 (QWERTYUIOP)
        # ============================================
        # Create buttons for letters Q through P (top row of keyboard)

        Letter_ButtonQ = QPushButton("Q")
        Letter_ButtonQ.setObjectName("letter")
        Letter_ButtonQ.clicked.connect(self.Display_Letter)

        Letter_ButtonW = QPushButton("W")
        Letter_ButtonW.setObjectName("letter")
        Letter_ButtonW.clicked.connect(self.Display_Letter)

        Letter_ButtonE = QPushButton("E")
        Letter_ButtonE.setObjectName("letter")
        Letter_ButtonE.clicked.connect(self.Display_Letter)

        Letter_ButtonR = QPushButton("R")
        Letter_ButtonR.setObjectName("letter")
        Letter_ButtonR.clicked.connect(self.Display_Letter)

        Letter_ButtonT = QPushButton("T")
        Letter_ButtonT.setObjectName("letter")
        Letter_ButtonT.clicked.connect(self.Display_Letter)

        Letter_ButtonY = QPushButton("Y")
        Letter_ButtonY.setObjectName("letter")
        Letter_ButtonY.clicked.connect(self.Display_Letter)

        Letter_ButtonU = QPushButton("U")
        Letter_ButtonU.setObjectName("letter")
        Letter_ButtonU.clicked.connect(self.Display_Letter)

        Letter_ButtonI = QPushButton("I")
        Letter_ButtonI.setObjectName("letter")
        Letter_ButtonI.clicked.connect(self.Display_Letter)

        Letter_ButtonO = QPushButton("O")
        Letter_ButtonO.setObjectName("letter")
        Letter_ButtonO.clicked.connect(self.Display_Letter)

        Letter_ButtonP = QPushButton("P")
        Letter_ButtonP.setObjectName("letter")
        Letter_ButtonP.clicked.connect(self.Display_Letter)

        # Delete button (removes last character)
        Letter_Button_Delete = QPushButton("⌫ Delete")
        Letter_Button_Delete.setObjectName("delete")
        Letter_Button_Delete.clicked.connect(self.Delete_Button)

        # ============================================
        # KEYBOARD BUTTONS - ROW 2 (ASDFGHJKL)
        # ============================================
        # Create buttons for letters A through L (middle row of keyboard)

        Letter_ButtonA = QPushButton("A")
        Letter_ButtonA.setObjectName("letter")
        Letter_ButtonA.clicked.connect(self.Display_Letter)

        Letter_ButtonS = QPushButton("S")
        Letter_ButtonS.setObjectName("letter")
        Letter_ButtonS.clicked.connect(self.Display_Letter)

        Letter_ButtonD = QPushButton("D")
        Letter_ButtonD.setObjectName("letter")
        Letter_ButtonD.clicked.connect(self.Display_Letter)

        Letter_ButtonF = QPushButton("F")
        Letter_ButtonF.setObjectName("letter")
        Letter_ButtonF.clicked.connect(self.Display_Letter)

        Letter_ButtonG = QPushButton("G")
        Letter_ButtonG.setObjectName("letter")
        Letter_ButtonG.clicked.connect(self.Display_Letter)

        Letter_ButtonH = QPushButton("H")
        Letter_ButtonH.setObjectName("letter")
        Letter_ButtonH.clicked.connect(self.Display_Letter)

        Letter_ButtonJ = QPushButton("J")
        Letter_ButtonJ.setObjectName("letter")
        Letter_ButtonJ.clicked.connect(self.Display_Letter)

        Letter_ButtonK = QPushButton("K")
        Letter_ButtonK.setObjectName("letter")
        Letter_ButtonK.clicked.connect(self.Display_Letter)

        Letter_ButtonL = QPushButton("L")
        Letter_ButtonL.setObjectName("letter")
        Letter_ButtonL.clicked.connect(self.Display_Letter)

        # Enter button (adds new line)
        Letter_Button_Enter = QPushButton("⏎ Enter")
        Letter_Button_Enter.setObjectName("enter")
        Letter_Button_Enter.clicked.connect(self.Text_To_Speech)

        # ============================================
        # KEYBOARD BUTTONS - ROW 3 (ZXCVBNM)
        # ============================================
        # Create buttons for letters Z through M (bottom row of keyboard)

        Letter_ButtonZ = QPushButton("Z")
        Letter_ButtonZ.setObjectName("letter")
        Letter_ButtonZ.clicked.connect(self.Display_Letter)

        Letter_ButtonX = QPushButton("X")
        Letter_ButtonX.setObjectName("letter")
        Letter_ButtonX.clicked.connect(self.Display_Letter)

        Letter_ButtonC = QPushButton("C")
        Letter_ButtonC.setObjectName("letter")
        Letter_ButtonC.clicked.connect(self.Display_Letter)

        Letter_ButtonV = QPushButton("V")
        Letter_ButtonV.setObjectName("letter")
        Letter_ButtonV.clicked.connect(self.Display_Letter)

        Letter_ButtonB = QPushButton("B")
        Letter_ButtonB.setObjectName("letter")
        Letter_ButtonB.clicked.connect(self.Display_Letter)

        Letter_ButtonN = QPushButton("N")
        Letter_ButtonN.setObjectName("letter")
        Letter_ButtonN.clicked.connect(self.Display_Letter)

        Letter_ButtonM = QPushButton("M")
        Letter_ButtonM.setObjectName("letter")
        Letter_ButtonM.clicked.connect(self.Display_Letter)

        # Space bar button (adds space between words)
        Space_Bar = QPushButton("⎵ Space")
        Space_Bar.setObjectName("space")
        Space_Bar.clicked.connect(self.Space_Bar)

        # ============================================
        # Emergency BUTTONS 
        # ============================================
        # Create pre-defined sentences in a button
    
        EM1_Button = QPushButton("I Feel Pain!!")
        EM1_Button.setObjectName("emergency_red")
        EM1_Button.clicked.connect(self.Display_Letter)  
        
        EM2_Button = QPushButton("I Need Water")
        EM2_Button.setObjectName("emergency_blue")
        EM2_Button.clicked.connect(self.Display_Letter)
        
        EM3_Button = QPushButton("I Want Doctor")
        EM3_Button.setObjectName("emergency_blue")
        EM3_Button.clicked.connect(self.Display_Letter)
        
        EM4_Button = QPushButton("I Want To Eat")
        EM4_Button.setObjectName("emergency_blue")
        EM4_Button.clicked.connect(self.Display_Letter)
        
        EM5_Button = QPushButton("Clear The screen")
        EM5_Button.setObjectName("emergency_blue")
        EM5_Button.clicked.connect(self.Clear_Whole_Text)

        EM6_Button = QPushButton("EN <-> AR")
        EM6_Button.setObjectName("emergency_blue")
        EM6_Button.clicked.connect(self.Change_Lang)
        
        # ============================================
        # LAYOUT SETUP - ORGANIZING THE INTERFACE
        # ============================================

        #Creating a Vertical Layout For the Emergencey Buttons
        Vertical_EM_Button = QVBoxLayout()
        Vertical_EM_Button.setSpacing(layout_spacing)
        Vertical_EM_Button.setContentsMargins(layout_margin, layout_margin, layout_margin, layout_margin)
        Vertical_EM_Button.addWidget(EM1_Button)
        Vertical_EM_Button.addWidget(EM2_Button)
        Vertical_EM_Button.addWidget(EM3_Button)
        Vertical_EM_Button.addWidget(EM4_Button)
        Vertical_EM_Button.addWidget(EM5_Button)
        Vertical_EM_Button.addWidget(EM6_Button)

        Vertical_TextArea = QVBoxLayout()
        Vertical_TextArea.addWidget(Thebes_Label, stretch=2)
        Vertical_TextArea.addWidget(Display_Label, stretch=0)
        Vertical_TextArea.addWidget(self.text_holder_label, stretch=2)

        # HORIZONTAL LAYOUT: Text display area
        H_layout = QHBoxLayout()
        H_layout.setSpacing(layout_spacing)
        H_layout.addLayout(Vertical_EM_Button)
        H_layout.addLayout(Vertical_TextArea)
        # Give MORE space to the EM column
        H_layout.setStretch(0, 2)
        H_layout.setStretch(1, 3)
    
        # KEYBOARD ROW 1 LAYOUT: Q W E R T Y U I O P + Delete
        Keyboard_Layout_ROW1 = QHBoxLayout()
        Keyboard_Layout_ROW1.setSpacing(layout_spacing)
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
        Keyboard_Layout_ROW2.setSpacing(layout_spacing)
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
        Keyboard_Layout_ROW3.setSpacing(layout_spacing)
        Keyboard_Layout_ROW3.addWidget(Letter_ButtonZ)  
        Keyboard_Layout_ROW3.addWidget(Letter_ButtonX)
        Keyboard_Layout_ROW3.addWidget(Letter_ButtonC)
        Keyboard_Layout_ROW3.addWidget(Letter_ButtonV)
        Keyboard_Layout_ROW3.addWidget(Letter_ButtonB)
        Keyboard_Layout_ROW3.addWidget(Letter_ButtonN)
        Keyboard_Layout_ROW3.addWidget(Letter_ButtonM)
        
        # KEYBOARD ROW 4 LAYOUT: Space bar
        Keyboard_Layout_ROW4 = QHBoxLayout()
        Keyboard_Layout_ROW4.setSpacing(layout_spacing)
        Keyboard_Layout_ROW4.addWidget(Space_Bar)
        
        # MAIN VERTICAL LAYOUT: Stack all layouts vertically
        V_layout = QVBoxLayout()
        V_layout.setSpacing(layout_spacing)
        V_layout.setContentsMargins(layout_margin, layout_margin, layout_margin, layout_margin)
        V_layout.addLayout(H_layout)           # Text display area
        V_layout.addLayout(Keyboard_Layout_ROW1)  # Keyboard row 1
        V_layout.addLayout(Keyboard_Layout_ROW2)  # Keyboard row 2
        V_layout.addLayout(Keyboard_Layout_ROW3)  # Keyboard row 3
        V_layout.addLayout(Keyboard_Layout_ROW4)  # Keyboard row 4 (space bar)
        
        # Set the main layout for the widget
        self.setLayout(V_layout)
        
        # ============================================
        # APPLY GLOBAL STYLESHEET WITH DYNAMIC SIZES
        # ============================================
        self.setStyleSheet(f"""
            QPushButton#letter {{
                color: black;
                background-color: #f8f9fa;
                padding: {padding}px;
                border: 2px solid #0c0c0d;
                border-radius: 5px;
                font-weight: bold;
                font-size: {font_size}px;
            }}
            QPushButton#letter:hover {{
                background-color: #e9ecef;
            }}
            QPushButton#delete {{
                color: white;
                background-color: #dc3545;
                padding: {padding}px;
                border: 2px solid #0c0c0d;
                border-radius: 5px;
                font-weight: bold;
                font-size: {delete_font}px;
            }}
            QPushButton#delete:hover {{
                background-color: #c82333;
            }}
            QPushButton#enter {{
                color: white;
                background-color: #28a745;
                padding: {padding}px;
                border: 2px solid #0c0c0d;
                border-radius: 5px;
                font-weight: bold;
                font-size: {enter_font}px;
            }}
            QPushButton#enter:hover {{
                background-color: #218838;
            }}
            QPushButton#space {{
                color: white;
                background-color: #007bff;
                padding: {space_padding}px;
                border: 2px solid #0056b3;
                border-radius: 5px;
                font-weight: bold;
                font-size: {space_font}px;
            }}
            QPushButton#space:hover {{
                background-color: #0069d9;
            }}
            QPushButton#emergency_red {{
                color: white;
                background-color: #dc3545;
                padding: {em_padding}px;
                border: 2px solid #bd2130;
                border-radius: 10px;
                font-weight: bold;
                font-size: {em_font}px;
                min-height: {em_height}px;
                min-width: {em_min_width}px;
                qproperty-alignment: AlignCenter;
            }}
            QPushButton#emergency_blue {{
                background-color: #357bdc;
                color: white;
                padding: {em_padding}px;
                border: 2px solid #0056b3;
                border-radius: 10px;
                font-weight: bold;
                font-size: {em_font}px;
                min-height: {em_height}px;
                min-width: {em_min_width}px;
                qproperty-alignment: AlignCenter;
            }}
        """)
    
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
        # Add a space character to separate words
        self.current_text += " "
        
        # Update the display
        self.text_holder_label.setText(f"{self.current_text}")
    
    def Delete_Button(self):
        # Remove the last character (if there is any text)
        if self.current_text:  # Only delete if there's text to delete
            self.current_text = self.current_text[:-1]
            
            # Update the display
            self.text_holder_label.setText(f"{self.current_text}")

    def Clear_Whole_Text(self):
        
        if self.current_text:  # Only Clear if there's text to delete
            self.current_text = ""
            
            # Update the display
            self.text_holder_label.setText(f"{self.current_text}")

    def Text_To_Speech(self):
     # Get the typed text
        text_to_speak = self.current_text.strip()  # Your text
        if text_to_speak:
            # SPEAK IT
         self.tts.speak(text_to_speak)

    def Change_Lang(self):
     #Sends signals to change Language 
            self.Language_Change_req.emit()