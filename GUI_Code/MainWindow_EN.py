"""
EyeTalk - Virtual Keyboard Widget
This widget creates a virtual keyboard interface for typing text using buttons.
Each button press appends characters to form words and sentences.
"""

from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap, QGuiApplication, QResizeEvent
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
        # BASE SIZES (designed for 1024x600)
        # ============================================
        self.ref_w, self.ref_h = 1024, 600
        
        # Base values (at 1024x600)
        self.base_font_size = 16
        self.base_padding = 8
        self.base_delete_font = 14
        self.base_enter_font = 14
        self.base_space_font = 14
        self.base_space_padding = 12
        
        # Emergency buttons 
        self.base_em_font = 18
        self.base_em_padding = 8          
        self.base_em_height = 45
        self.base_em_min_width = 100
        
        self.base_logo_width = 500
        self.base_logo_height = 400
        self.base_layout_spacing = 6
        self.base_layout_margin = 15
        self.base_label_font = 15
        
        # Text box 
        self.base_text_holder_font = 18   
        self.base_text_holder_padding = 10 
        
        # Current scale (will be computed in resizeEvent)
        self.scale = 1.0
        
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
        self.Thebes_Logo = QPixmap(r"M:\University\Level 4\Shit_Project\EyeTalk\Documentation\Eye_Talk_Logo.png")
        self.Thebes_Label = QLabel(self)
        self.Thebes_Label.setAlignment(Qt.AlignCenter)  # centre the logo

        # Label to show "Your Text:" prompt
        self.Display_Label = QLabel("Your Text")
        self.Display_Label.setObjectName("display_label")

        # Label that displays the typed text (updates as user types)
        self.text_holder_label = QLabel("")
        self.text_holder_label.setObjectName("text_holder")
        self.text_holder_label.setWordWrap(False)       # force single line
        self.text_holder_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    
        # ============================================
        # KEYBOARD BUTTONS - ROW 1 (QWERTYUIOP)
        # ============================================
        # Create buttons for letters Q through P (top row of keyboard)

        self.Letter_ButtonQ = QPushButton("Q")
        self.Letter_ButtonQ.setObjectName("letter")
        self.Letter_ButtonQ.clicked.connect(self.Display_Letter)

        self.Letter_ButtonW = QPushButton("W")
        self.Letter_ButtonW.setObjectName("letter")
        self.Letter_ButtonW.clicked.connect(self.Display_Letter)

        self.Letter_ButtonE = QPushButton("E")
        self.Letter_ButtonE.setObjectName("letter")
        self.Letter_ButtonE.clicked.connect(self.Display_Letter)

        self.Letter_ButtonR = QPushButton("R")
        self.Letter_ButtonR.setObjectName("letter")
        self.Letter_ButtonR.clicked.connect(self.Display_Letter)

        self.Letter_ButtonT = QPushButton("T")
        self.Letter_ButtonT.setObjectName("letter")
        self.Letter_ButtonT.clicked.connect(self.Display_Letter)

        self.Letter_ButtonY = QPushButton("Y")
        self.Letter_ButtonY.setObjectName("letter")
        self.Letter_ButtonY.clicked.connect(self.Display_Letter)

        self.Letter_ButtonU = QPushButton("U")
        self.Letter_ButtonU.setObjectName("letter")
        self.Letter_ButtonU.clicked.connect(self.Display_Letter)

        self.Letter_ButtonI = QPushButton("I")
        self.Letter_ButtonI.setObjectName("letter")
        self.Letter_ButtonI.clicked.connect(self.Display_Letter)

        self.Letter_ButtonO = QPushButton("O")
        self.Letter_ButtonO.setObjectName("letter")
        self.Letter_ButtonO.clicked.connect(self.Display_Letter)

        self.Letter_ButtonP = QPushButton("P")
        self.Letter_ButtonP.setObjectName("letter")
        self.Letter_ButtonP.clicked.connect(self.Display_Letter)

        # Delete button (removes last character)
        self.Letter_Button_Delete = QPushButton("⌫ Delete")
        self.Letter_Button_Delete.setObjectName("delete")
        self.Letter_Button_Delete.clicked.connect(self.Delete_Button)

        # ============================================
        # KEYBOARD BUTTONS - ROW 2 (ASDFGHJKL)
        # ============================================
        # Create buttons for letters A through L (middle row of keyboard)

        self.Letter_ButtonA = QPushButton("A")
        self.Letter_ButtonA.setObjectName("letter")
        self.Letter_ButtonA.clicked.connect(self.Display_Letter)

        self.Letter_ButtonS = QPushButton("S")
        self.Letter_ButtonS.setObjectName("letter")
        self.Letter_ButtonS.clicked.connect(self.Display_Letter)

        self.Letter_ButtonD = QPushButton("D")
        self.Letter_ButtonD.setObjectName("letter")
        self.Letter_ButtonD.clicked.connect(self.Display_Letter)

        self.Letter_ButtonF = QPushButton("F")
        self.Letter_ButtonF.setObjectName("letter")
        self.Letter_ButtonF.clicked.connect(self.Display_Letter)

        self.Letter_ButtonG = QPushButton("G")
        self.Letter_ButtonG.setObjectName("letter")
        self.Letter_ButtonG.clicked.connect(self.Display_Letter)

        self.Letter_ButtonH = QPushButton("H")
        self.Letter_ButtonH.setObjectName("letter")
        self.Letter_ButtonH.clicked.connect(self.Display_Letter)

        self.Letter_ButtonJ = QPushButton("J")
        self.Letter_ButtonJ.setObjectName("letter")
        self.Letter_ButtonJ.clicked.connect(self.Display_Letter)

        self.Letter_ButtonK = QPushButton("K")
        self.Letter_ButtonK.setObjectName("letter")
        self.Letter_ButtonK.clicked.connect(self.Display_Letter)

        self.Letter_ButtonL = QPushButton("L")
        self.Letter_ButtonL.setObjectName("letter")
        self.Letter_ButtonL.clicked.connect(self.Display_Letter)

        # Enter button (adds new line)
        self.Letter_Button_Enter = QPushButton("⏎ Enter")
        self.Letter_Button_Enter.setObjectName("enter")
        self.Letter_Button_Enter.clicked.connect(self.Text_To_Speech)

        # ============================================
        # KEYBOARD BUTTONS - ROW 3 (ZXCVBNM)
        # ============================================
        # Create buttons for letters Z through M (bottom row of keyboard)

        self.Letter_ButtonZ = QPushButton("Z")
        self.Letter_ButtonZ.setObjectName("letter")
        self.Letter_ButtonZ.clicked.connect(self.Display_Letter)

        self.Letter_ButtonX = QPushButton("X")
        self.Letter_ButtonX.setObjectName("letter")
        self.Letter_ButtonX.clicked.connect(self.Display_Letter)

        self.Letter_ButtonC = QPushButton("C")
        self.Letter_ButtonC.setObjectName("letter")
        self.Letter_ButtonC.clicked.connect(self.Display_Letter)

        self.Letter_ButtonV = QPushButton("V")
        self.Letter_ButtonV.setObjectName("letter")
        self.Letter_ButtonV.clicked.connect(self.Display_Letter)

        self.Letter_ButtonB = QPushButton("B")
        self.Letter_ButtonB.setObjectName("letter")
        self.Letter_ButtonB.clicked.connect(self.Display_Letter)

        self.Letter_ButtonN = QPushButton("N")
        self.Letter_ButtonN.setObjectName("letter")
        self.Letter_ButtonN.clicked.connect(self.Display_Letter)

        self.Letter_ButtonM = QPushButton("M")
        self.Letter_ButtonM.setObjectName("letter")
        self.Letter_ButtonM.clicked.connect(self.Display_Letter)

        # Space bar button (adds space between words)
        self.space_button = QPushButton("⎵ Space")
        self.space_button.setObjectName("space")
        self.space_button.clicked.connect(self.space_pressed)

        # ============================================
        # Emergency BUTTONS 
        # ============================================
        # Create pre-defined sentences in a button
    
        self.EM1_Button = QPushButton("I Feel Pain!!")
        self.EM1_Button.setObjectName("emergency_red")
        self.EM1_Button.clicked.connect(self.Display_Letter)  
        
        self.EM2_Button = QPushButton("I Need Water")
        self.EM2_Button.setObjectName("emergency_blue")
        self.EM2_Button.clicked.connect(self.Display_Letter)
        
        self.EM3_Button = QPushButton("I Want Doctor")
        self.EM3_Button.setObjectName("emergency_blue")
        self.EM3_Button.clicked.connect(self.Display_Letter)
        
        self.EM4_Button = QPushButton("I Want To Eat")
        self.EM4_Button.setObjectName("emergency_blue")
        self.EM4_Button.clicked.connect(self.Display_Letter)
        
        self.EM5_Button = QPushButton("Clear The screen")
        self.EM5_Button.setObjectName("emergency_blue")
        self.EM5_Button.clicked.connect(self.Clear_Whole_Text)

        self.EM6_Button = QPushButton("EN <-> AR")
        self.EM6_Button.setObjectName("emergency_blue")
        self.EM6_Button.clicked.connect(self.Change_Lang)
        
        # ============================================
        # LAYOUT SETUP - ORGANIZING THE INTERFACE
        # ============================================

        #Creating a Vertical Layout For the Emergencey Buttons
        self.Vertical_EM_Button = QVBoxLayout()
        self.Vertical_EM_Button.addWidget(self.EM1_Button)
        self.Vertical_EM_Button.addWidget(self.EM2_Button)
        self.Vertical_EM_Button.addWidget(self.EM3_Button)
        self.Vertical_EM_Button.addWidget(self.EM4_Button)
        self.Vertical_EM_Button.addWidget(self.EM5_Button)
        self.Vertical_EM_Button.addWidget(self.EM6_Button)

        self.Vertical_TextArea = QVBoxLayout()
        self.Vertical_TextArea.addWidget(self.Thebes_Label, stretch=2)    # reduced stretch
        self.Vertical_TextArea.addWidget(self.Display_Label, stretch=0)
        self.Vertical_TextArea.addWidget(self.text_holder_label, stretch=1.5)

        # HORIZONTAL LAYOUT: Text display area
        self.H_layout = QHBoxLayout()
        self.H_layout.addLayout(self.Vertical_EM_Button)
        self.H_layout.addLayout(self.Vertical_TextArea)
        self.H_layout.setStretch(0, 2)
        self.H_layout.setStretch(1, 3)
    
        # KEYBOARD ROW 1 LAYOUT: Q W E R T Y U I O P + Delete
        self.Keyboard_Layout_ROW1 = QHBoxLayout()
        self.Keyboard_Layout_ROW1.addWidget(self.Letter_ButtonQ)
        self.Keyboard_Layout_ROW1.addWidget(self.Letter_ButtonW)
        self.Keyboard_Layout_ROW1.addWidget(self.Letter_ButtonE)
        self.Keyboard_Layout_ROW1.addWidget(self.Letter_ButtonR)
        self.Keyboard_Layout_ROW1.addWidget(self.Letter_ButtonT)
        self.Keyboard_Layout_ROW1.addWidget(self.Letter_ButtonY)
        self.Keyboard_Layout_ROW1.addWidget(self.Letter_ButtonU)
        self.Keyboard_Layout_ROW1.addWidget(self.Letter_ButtonI)
        self.Keyboard_Layout_ROW1.addWidget(self.Letter_ButtonO)
        self.Keyboard_Layout_ROW1.addWidget(self.Letter_ButtonP)
        self.Keyboard_Layout_ROW1.addWidget(self.Letter_Button_Delete)
        
        # KEYBOARD ROW 2 LAYOUT: A S D F G H J K L + Enter
        self.Keyboard_Layout_ROW2 = QHBoxLayout()
        self.Keyboard_Layout_ROW2.addWidget(self.Letter_ButtonA)
        self.Keyboard_Layout_ROW2.addWidget(self.Letter_ButtonS)
        self.Keyboard_Layout_ROW2.addWidget(self.Letter_ButtonD)
        self.Keyboard_Layout_ROW2.addWidget(self.Letter_ButtonF)
        self.Keyboard_Layout_ROW2.addWidget(self.Letter_ButtonG)
        self.Keyboard_Layout_ROW2.addWidget(self.Letter_ButtonH)
        self.Keyboard_Layout_ROW2.addWidget(self.Letter_ButtonJ)
        self.Keyboard_Layout_ROW2.addWidget(self.Letter_ButtonK)
        self.Keyboard_Layout_ROW2.addWidget(self.Letter_ButtonL)
        self.Keyboard_Layout_ROW2.addWidget(self.Letter_Button_Enter)
        
        # KEYBOARD ROW 3 LAYOUT: Z X C V B N M
        self.Keyboard_Layout_ROW3 = QHBoxLayout()
        self.Keyboard_Layout_ROW3.addWidget(self.Letter_ButtonZ)  
        self.Keyboard_Layout_ROW3.addWidget(self.Letter_ButtonX)
        self.Keyboard_Layout_ROW3.addWidget(self.Letter_ButtonC)
        self.Keyboard_Layout_ROW3.addWidget(self.Letter_ButtonV)
        self.Keyboard_Layout_ROW3.addWidget(self.Letter_ButtonB)
        self.Keyboard_Layout_ROW3.addWidget(self.Letter_ButtonN)
        self.Keyboard_Layout_ROW3.addWidget(self.Letter_ButtonM)
        
        # KEYBOARD ROW 4 LAYOUT: Space bar
        self.Keyboard_Layout_ROW4 = QHBoxLayout()
        self.Keyboard_Layout_ROW4.addWidget(self.space_button)
        
        # MAIN VERTICAL LAYOUT: Stack all layouts vertically
        self.V_layout = QVBoxLayout()
        self.V_layout.addLayout(self.H_layout)           # Text display area
        self.V_layout.addLayout(self.Keyboard_Layout_ROW1)  # Keyboard row 1
        self.V_layout.addLayout(self.Keyboard_Layout_ROW2)  # Keyboard row 2
        self.V_layout.addLayout(self.Keyboard_Layout_ROW3)  # Keyboard row 3
        self.V_layout.addLayout(self.Keyboard_Layout_ROW4)  # Keyboard row 4 (space bar)
        
        # Set the main layout for the widget
        self.setLayout(self.V_layout)
        
        # ============================================
        # INITIAL SCALING & STYLESHEET
        # ============================================
        self.update_sizes()

    def update_sizes(self):
        """Recalculate all sizes based on current widget dimensions and reapply stylesheet."""
        # Get current widget size
        w = self.width()
        h = self.height()
        if w == 0 or h == 0:
            return
        
        # Compute new scale (fits both width and height)
        scale_w = w / self.ref_w
        scale_h = h / self.ref_h
        new_scale = min(scale_w, scale_h)
        new_scale = max(0.5, min(new_scale, 1.5))  # clamp
        
        # Only update if scale changed significantly to avoid flicker
        if abs(new_scale - self.scale) < 0.01:
            return
        
        self.scale = new_scale
        
        # Compute scaled values
        font_size = int(self.base_font_size * self.scale)
        padding = int(self.base_padding * self.scale)
        delete_font = int(self.base_delete_font * self.scale)
        enter_font = int(self.base_enter_font * self.scale)
        space_font = int(self.base_space_font * self.scale)
        space_padding = int(self.base_space_padding * self.scale)
        em_font = int(self.base_em_font * self.scale)
        em_padding = int(self.base_em_padding * self.scale)
        em_height = int(self.base_em_height * self.scale)
        em_min_width = int(self.base_em_min_width * self.scale)
        logo_width = int(self.base_logo_width * self.scale)
        logo_height = int(self.base_logo_height * self.scale)
        layout_spacing = int(self.base_layout_spacing * self.scale)
        layout_margin = int(self.base_layout_margin * self.scale)
        label_font = int(self.base_label_font * self.scale)
        text_holder_font = int(self.base_text_holder_font * self.scale)
        text_holder_padding = int(self.base_text_holder_padding * self.scale)
        
        # Update logo
        scaled_pixmap = self.Thebes_Logo.scaled(logo_width, logo_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.Thebes_Label.setPixmap(scaled_pixmap)
        
        # Set maximum height for text holder to keep it single‑line
        self.text_holder_label.setMaximumHeight(int(text_holder_font * 1.8))
        
        # Update layout margins and spacings
        self.V_layout.setSpacing(layout_spacing)
        self.V_layout.setContentsMargins(layout_margin, layout_margin, layout_margin, layout_margin)
        self.H_layout.setSpacing(layout_spacing)
        self.Vertical_EM_Button.setSpacing(layout_spacing)
        self.Vertical_EM_Button.setContentsMargins(layout_margin, layout_margin, layout_margin, layout_margin)
        self.Vertical_TextArea.setSpacing(layout_spacing)
        self.Keyboard_Layout_ROW1.setSpacing(layout_spacing)
        self.Keyboard_Layout_ROW2.setSpacing(layout_spacing)
        self.Keyboard_Layout_ROW3.setSpacing(layout_spacing)
        self.Keyboard_Layout_ROW4.setSpacing(layout_spacing)
        
        # Update stylesheet
        self.setStyleSheet(f"""
            QLabel#display_label {{
                color: #FF4500;
                qproperty-alignment: AlignCenter;
                font-size: {label_font}px;
                max-height: {label_font}px;
            }}
            QLabel#text_holder {{
                padding: {text_holder_padding}px;
                font-size: {text_holder_font}px;
                border-radius: 5px;
                background-color: #75706f;
            }}
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
    
    def resizeEvent(self, event: QResizeEvent):
        """Handle window resize to update sizes dynamically."""
        super().resizeEvent(event)
        self.update_sizes()
    
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
    
    def space_pressed(self):
        """Add a space character to the current text."""
        self.current_text += " "
        self.text_holder_label.setText(self.current_text)
    
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