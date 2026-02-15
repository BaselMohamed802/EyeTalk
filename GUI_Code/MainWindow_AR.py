"""
EyeTalk - Virtual Keyboard Widget
This widget creates a virtual keyboard interface for typing text using buttons.
Each button press appends characters to form words and sentences.
"""

from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from TTS_Engine import TTS 

class Widget_AR(QWidget):
    """
    Main widget class that implements a virtual keyboard.
    Allows users to type text by clicking letter buttons.

    """
    def __init__(self):
        """Initialize the widget and set up the virtual keyboard interface."""
        super().__init__()  # Initialize parent QWidget class
        
        # ============================================
        # WINDOW SETUP
        # ============================================
        self.setWindowTitle("EyeTalk")  # Set window title
        self.current_text = ""  # Variable to store the typed text (starts empty)
        
        # ============================================
        # TEXT DISPLAY AREA
        # ============================================
        
        self.tts = TTS()
        # Label to show "Your Text:" prompt
        Display_Label = QLabel("النص:")
        Display_Label.setStyleSheet("""
        QLabel {
            padding: 15px;
            font-size: 20px;
            min-width: 200px;
            min-height: 100px;
            border-radius: 8px;
        }
    """)

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

        Letter_ButtonQ = QPushButton("ض")
        Letter_ButtonQ.clicked.connect(self.Display_Letter)
        Letter_ButtonQ.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_ButtonW = QPushButton("ص")
        Letter_ButtonW.clicked.connect(self.Display_Letter)
        Letter_ButtonW.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_ButtonE = QPushButton("ث")
        Letter_ButtonE.clicked.connect(self.Display_Letter)
        Letter_ButtonE.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_ButtonR = QPushButton("ق")
        Letter_ButtonR.clicked.connect(self.Display_Letter)
        Letter_ButtonR.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_ButtonT = QPushButton("ف")
        Letter_ButtonT.clicked.connect(self.Display_Letter)
        Letter_ButtonT.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_ButtonY = QPushButton("غ")
        Letter_ButtonY.clicked.connect(self.Display_Letter)
        Letter_ButtonY.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_ButtonU = QPushButton("ع")
        Letter_ButtonU.clicked.connect(self.Display_Letter)
        Letter_ButtonU.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_ButtonI = QPushButton("ه")
        Letter_ButtonI.clicked.connect(self.Display_Letter)
        Letter_ButtonI.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_ButtonO = QPushButton("خ")
        Letter_ButtonO.clicked.connect(self.Display_Letter)
        Letter_ButtonO.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_ButtonP = QPushButton("ح")
        Letter_ButtonP.clicked.connect(self.Display_Letter)
        Letter_ButtonP.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_Button11 = QPushButton("ج")
        Letter_Button11.clicked.connect(self.Display_Letter)
        Letter_Button11.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_Button12 = QPushButton("د")
        Letter_Button12.clicked.connect(self.Display_Letter)
        Letter_Button12.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        # Delete button (removes last character)
        Letter_Button_Delete = QPushButton("⌫ حذف")
        Letter_Button_Delete.clicked.connect(self.Delete_Button)
        Letter_Button_Delete.setStyleSheet("""
            QPushButton {
                color: white;
                background-color: #dc3545;
                padding: 20px;
                border: 2px solid #bd2130;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)

        # ============================================
        # KEYBOARD BUTTONS - ROW 2 (ASDFGHJKL)
        # ============================================
        # Create buttons for letters A through L (middle row of keyboard)

        Letter_ButtonA = QPushButton("ش")
        Letter_ButtonA.clicked.connect(self.Display_Letter)
        Letter_ButtonA.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_ButtonS = QPushButton("س")
        Letter_ButtonS.clicked.connect(self.Display_Letter)
        Letter_ButtonS.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_ButtonD = QPushButton("ي")
        Letter_ButtonD.clicked.connect(self.Display_Letter)
        Letter_ButtonD.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_ButtonF = QPushButton("ب")
        Letter_ButtonF.clicked.connect(self.Display_Letter)
        Letter_ButtonF.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_ButtonG = QPushButton("ل")
        Letter_ButtonG.clicked.connect(self.Display_Letter)
        Letter_ButtonG.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_ButtonH = QPushButton("أ")
        Letter_ButtonH.clicked.connect(self.Display_Letter)
        Letter_ButtonH.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_ButtonJ = QPushButton("ت")
        Letter_ButtonJ.clicked.connect(self.Display_Letter)
        Letter_ButtonJ.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_ButtonK = QPushButton("ن")
        Letter_ButtonK.clicked.connect(self.Display_Letter)
        Letter_ButtonK.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_ButtonL = QPushButton("م")
        Letter_ButtonL.clicked.connect(self.Display_Letter)
        Letter_ButtonL.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_Button20 = QPushButton("ك")
        Letter_Button20.clicked.connect(self.Display_Letter)
        Letter_Button20.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)
        Letter_Button21 = QPushButton("ط")
        Letter_Button21.clicked.connect(self.Display_Letter)
        Letter_Button21.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)
        
        # Enter button (adds new line)
        Letter_Button_Enter = QPushButton("⏎ تحدث")
        Letter_Button_Enter.clicked.connect(self.Text_To_Speech)
        Letter_Button_Enter.setStyleSheet("""
            QPushButton {
                color: white;
                background-color: #28a745;
                padding: 20px;
                border: 2px solid #1e7e34;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)

        # ============================================
        # KEYBOARD BUTTONS - ROW 3 (ZXCVBNM)
        # ============================================
        # Create buttons for letters Z through M (bottom row of keyboard)

        Letter_ButtonZ = QPushButton("ئ")
        Letter_ButtonZ.clicked.connect(self.Display_Letter)
        Letter_ButtonZ.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_ButtonX = QPushButton("ء")
        Letter_ButtonX.clicked.connect(self.Display_Letter)
        Letter_ButtonX.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_ButtonC = QPushButton("ؤ")
        Letter_ButtonC.clicked.connect(self.Display_Letter)
        Letter_ButtonC.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_ButtonV = QPushButton("ر")
        Letter_ButtonV.clicked.connect(self.Display_Letter)
        Letter_ButtonV.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_ButtonB = QPushButton("لا")
        Letter_ButtonB.clicked.connect(self.Display_Letter)
        Letter_ButtonB.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_ButtonN = QPushButton("ى")
        Letter_ButtonN.clicked.connect(self.Display_Letter)
        Letter_ButtonN.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_ButtonM = QPushButton("ة")
        Letter_ButtonM.clicked.connect(self.Display_Letter)
        Letter_ButtonM.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_Button30 = QPushButton("و")
        Letter_Button30.clicked.connect(self.Display_Letter)
        Letter_Button30.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_Button31 = QPushButton("ز")
        Letter_Button31.clicked.connect(self.Display_Letter)
        Letter_Button31.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)

        Letter_Button32 = QPushButton("ظ")
        Letter_Button32.clicked.connect(self.Display_Letter)
        Letter_Button32.setStyleSheet("""
            QPushButton {
                color: black;
                background-color: #f8f9fa;
                padding: 20px;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
        """)
        # Space bar button (adds space between words)
        Space_Bar = QPushButton("⎵ مسافة")
        Space_Bar.clicked.connect(self.Space_Bar)
        Space_Bar.setStyleSheet("""
            QPushButton {
                color: white;
                background-color: #007bff;
                padding: 20px 20px;
                border: 2px solid #0056b3;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0069d9;
            }
        """)
        # ============================================
        # Emergency BUTTONS 
        # ============================================
        # Create pre-defined sentences in a button
    
        EM1_Button = QPushButton("أشعر بألم!!")
        EM1_Button.clicked.connect(self.Display_Letter)  
        EM1_Button.setStyleSheet("color: white; background-color: #dc3545; padding: 10px; border: 2px solid #bd2130; border-radius: 10px; font-weight: bold; font-size: 14px;}")
        
        EM2_Button = QPushButton("أريد قليل من الماء")
        EM2_Button.clicked.connect(self.Display_Letter)
        EM2_Button.setStyleSheet("background-color: #357bdc; color: white; font-weight: bold;font-size: 14px; border-radius: 10px; padding: 10px; text-align: center;")
        
        EM3_Button = QPushButton("أريد الطبيب")
        EM3_Button.clicked.connect(self.Display_Letter)
        EM3_Button.setStyleSheet("background-color: #357bdc; color: white; font-weight: bold;font-size: 14px; border-radius: 10px; padding: 10px; text-align: center;")
        
        EM4_Button = QPushButton("اتركني في حالي")
        EM4_Button.clicked.connect(self.Display_Letter)
        EM4_Button.setStyleSheet("background-color: #357bdc; color: white; font-weight: bold;font-size: 14px; border-radius: 10px; padding: 10px; text-align: center;")
        
        EM5_Button = QPushButton("مسح الشاشة")
        EM5_Button.clicked.connect(self.Clear_Whole_Text)
        EM5_Button.setStyleSheet("background-color: #357bdc; color: white; font-weight: bold;font-size: 14px; border-radius: 10px; padding: 10px; text-align: center;")
        
        # ============================================
        # LAYOUT SETUP - ORGANIZING THE INTERFACE
        # ============================================

        #Creating a Vertical Layout For the Emergencey Buttons
        Vertical_EM_Button = QVBoxLayout()
        Vertical_EM_Button.setSpacing(10)  # Space between buttons (15px)
        Vertical_EM_Button.setContentsMargins(30, 30, 30, 30)  # Padding around the group
        Vertical_EM_Button.addWidget(EM1_Button) #buttons alignments
        Vertical_EM_Button.addWidget(EM2_Button)
        Vertical_EM_Button.addWidget(EM3_Button)
        Vertical_EM_Button.addWidget(EM4_Button)
        Vertical_EM_Button.addWidget(EM5_Button)
        
        # HORIZONTAL LAYOUT: Text display area
        # Contains the "Your Text:" label and the text display label
        H_layout = QHBoxLayout()
        H_layout.setSpacing(20)
        H_layout.addLayout(Vertical_EM_Button)
        H_layout.addWidget(self.text_holder_label)        
        H_layout.addWidget(Display_Label)

       
        
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
        Keyboard_Layout_ROW1.addWidget(Letter_Button11)
        Keyboard_Layout_ROW1.addWidget(Letter_Button12)
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
        Keyboard_Layout_ROW2.addWidget(Letter_Button20)
        Keyboard_Layout_ROW2.addWidget(Letter_Button21)
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
        Keyboard_Layout_ROW3.addWidget(Letter_Button30)
        Keyboard_Layout_ROW3.addWidget(Letter_Button31)
        Keyboard_Layout_ROW3.addWidget(Letter_Button32)
        
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
    
    def Text_To_Speech(self):
        text_to_speak = self.current_text.strip()
        if text_to_speak:
            self.tts.speak(text_to_speak)