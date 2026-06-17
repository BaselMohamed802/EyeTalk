"""
Filename: head_tracking_gui.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 2/2/2026

Description:
    PySide6 GUI configuration interface for Head Tracking Mouse Control System.
    Fully responsive layout optimized for 1024x600 Raspberry Pi touchscreens.
    Integrated Clicking Method selections and strict absolute pathing.
"""

import sys
import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Fix for the Qt DPI warning on Windows/Linux
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCALE_FACTOR"] = "1"

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QLabel, QCheckBox, QComboBox,
    QPushButton, QSpinBox, QDoubleSpinBox, QMessageBox, QFileDialog,
    QScrollArea, QFrame, QSplitter, QTextEdit, QGridLayout,
    QButtonGroup, QSizePolicy
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


@dataclass
class ConfigDescription:
    """Data class for configuration parameter descriptions."""
    name: str
    description: str
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    step: float = 1.0
    unit: str = ""


class ConfigDefinitions:
    """Central repository of all configuration descriptions."""
    
    TRACKING_DESCRIPTIONS = {
        "yaw_range": ConfigDescription(
            name="Yaw Range (degrees)",
            description="How far left/right you need to turn to reach screen edges",
            min_val=5, max_val=45, step=1, unit="°"
        ),
        "pitch_range": ConfigDescription(
            name="Pitch Range (degrees)",
            description="How far up/down you need to tilt to reach screen edges",
            min_val=5, max_val=30, step=1, unit="°"
        ),
        "filter_length": ConfigDescription(
            name="Position Buffer Size",
            description="Number of frames to average for smoothing head position",
            min_val=1, max_val=20, step=1
        ),
        "invert_x": ConfigDescription(
            name="Invert Horizontal",
            description="Reverse left/right movement direction"
        ),
        "invert_y": ConfigDescription(
            name="Invert Vertical",
            description="Reverse up/down movement direction"
        ),
        "min_detection_confidence": ConfigDescription(
            name="Detection Confidence",
            description="Minimum confidence to detect a face (0.0-1.0)",
            min_val=0.1, max_val=0.9, step=0.05
        ),
        "min_tracking_confidence": ConfigDescription(
            name="Tracking Confidence",
            description="Minimum confidence to track face landmarks (0.0-1.0)",
            min_val=0.1, max_val=0.9, step=0.05
        ),
    }
    
    CONTROL_DESCRIPTIONS = {
        "movement_threshold_px": ConfigDescription(
            name="Movement Threshold (px)",
            description="Minimum pixel movement required to update cursor",
            min_val=0.1, max_val=5.0, step=0.1, unit="px"
        ),
        "pixel_deadzone_radius": ConfigDescription(
            name="Deadzone Radius (px)",
            description="Area where small movements are ignored",
            min_val=0.0, max_val=10.0, step=0.5, unit="px"
        ),
        "start_mouse_enabled": ConfigDescription(
            name="Start with Mouse Enabled",
            description="Whether mouse control is active when tracking starts"
        ),
    }
    
    RELATIVE_DESCRIPTIONS = {
        "max_speed_px_per_s": ConfigDescription(
            name="Maximum Speed",
            description="Maximum cursor speed in relative mode",
            min_val=500, max_val=3000, step=50, unit="px/s"
        ),
        "deadzone_deg": ConfigDescription(
            name="Deadzone (degrees)",
            description="Head movement ignored around center",
            min_val=0.0, max_val=5.0, step=0.1, unit="°"
        ),
        "expo": ConfigDescription(
            name="Exponential Factor",
            description="Makes small movements precise, large movements fast",
            min_val=1.0, max_val=2.5, step=0.1
        ),
    }
    
    FILTER_DESCRIPTIONS = {
        "type": ConfigDescription(
            name="Filter Type",
            description="EMA: Simple smoothing | Kalman: Predictive filtering",
        ),
        "alpha": ConfigDescription(
            name="EMA Smoothing Factor",
            description="How much weight new measurements have (EMA only)",
            min_val=0.1, max_val=0.9, step=0.05
        ),
        "process_noise": ConfigDescription(
            name="Process Noise",
            description="How much we trust our motion model (Kalman)",
            min_val=0.0001, max_val=0.01, step=0.0005
        ),
        "measurement_noise": ConfigDescription(
            name="Measurement Noise",
            description="How much we trust new measurements (Kalman)",
            min_val=0.01, max_val=0.5, step=0.01
        ),
        "dt": ConfigDescription(
            name="Time Step",
            description="Time step for Kalman filter (seconds)",
            min_val=0.1, max_val=2.0, step=0.1, unit="s"
        ),
    }
    
    PROFILE_DESCRIPTIONS = {
        "FAST": "Low smoothing, high responsiveness - best for fast navigation",
        "BALANCED": "Good balance of smoothness and responsiveness - default",
        "SMOOTH": "High smoothing, very stable - best for precise reading",
    }


class ConfigControlFactory:
    """Factory class to create appropriate Qt controls for different parameter types."""
    
    @staticmethod
    def create_control(parent, param_name: str, value: Any, param_type: str,
                      description: ConfigDescription, callback):
        
        if param_type == "bool" or isinstance(value, bool):
            return ConfigControlFactory._create_checkbox(parent, param_name, value, description, callback)
        elif param_type == "choice" or param_name in ["type"]:
            return ConfigControlFactory._create_combobox(parent, param_name, value, description, callback)
        elif isinstance(value, (int, float)):
            if isinstance(value, int) and param_name not in ["max_speed_px_per_s"]:
                return ConfigControlFactory._create_spinbox(parent, param_name, value, description, callback, is_float=False)
            else:
                return ConfigControlFactory._create_spinbox(parent, param_name, value, description, callback, is_float=True)
        else:
            container = QWidget()
            layout = QVBoxLayout(container)
            label = QLabel(f"{description.name}: {str(value)}")
            label.setStyleSheet("background-color: #f0f0f0; padding: 5px; border-radius: 3px;")
            layout.addWidget(label)
            return container
    
    @staticmethod
    def _create_checkbox(parent, param_name: str, value: bool, description: ConfigDescription, callback):
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(0, 5, 0, 10)

        checkbox = QCheckBox(description.name)
        checkbox.setChecked(value)
        checkbox.stateChanged.connect(lambda state: callback(param_name, bool(state)))
        main_layout.addWidget(checkbox)
        
        desc_label = QLabel(description.description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666666; font-size: 11px; margin-left: 20px;")
        main_layout.addWidget(desc_label)
        
        return container
    
    @staticmethod
    def _create_combobox(parent, param_name: str, value: str, description: ConfigDescription, callback):
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(0, 5, 0, 10)
        
        h_layout = QHBoxLayout()
        label = QLabel(description.name)
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        h_layout.addWidget(label)
        
        combo = QComboBox()
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        if param_name == "type":
            choices = ["kalman", "ema"]
        else:
            choices = [str(value)]
        
        for choice in choices:
            combo.addItem(choice.capitalize(), choice)
        
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)
        
        desc_label = QLabel(description.description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666666; font-size: 11px;")
        
        combo.currentIndexChanged.connect(lambda: callback(param_name, combo.currentData() or combo.currentText().lower()))
        
        h_layout.addWidget(combo)
        main_layout.addLayout(h_layout)
        main_layout.addWidget(desc_label)
        
        return container
    
    @staticmethod
    def _create_spinbox(parent, param_name: str, value: float, description: ConfigDescription,
                       callback, is_float: bool = True):
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(0, 5, 0, 10)
        
        h_layout = QHBoxLayout()
        label = QLabel(description.name)
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        h_layout.addWidget(label)
        
        if is_float:
            spin = QDoubleSpinBox()
            spin.setDecimals(3 if value < 0.01 else 2)
            spin.setSingleStep(description.step)
        else:
            spin = QSpinBox()
            spin.setSingleStep(int(description.step))
            
        spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        if description.min_val is not None:
            spin.setMinimum(description.min_val)
        if description.max_val is not None:
            spin.setMaximum(description.max_val)
        
        spin.setValue(value)
        spin.valueChanged.connect(lambda val: callback(param_name, val))
        
        if description.unit:
            spin.setSuffix(f" {description.unit}")
        
        h_layout.addWidget(spin)
        main_layout.addLayout(h_layout)
        
        desc_label = QLabel(description.description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666666; font-size: 11px;")
        main_layout.addWidget(desc_label)
        
        return container


class ProfileCard(QFrame):
    """Card-style widget for displaying profile information."""
    def __init__(self, profile_name: str, description: str, is_active: bool = False):
        super().__init__()
        self.profile_name = profile_name
        self.setFrameStyle(QFrame.Box)
        self.setLineWidth(2)
        
        if is_active:
            self.setStyleSheet("QFrame { background-color: #e3f2fd; border-color: #2196f3; border-radius: 5px; }")
        else:
            self.setStyleSheet("QFrame { background-color: #f5f5f5; border-color: #cccccc; border-radius: 5px; }")
        
        layout = QVBoxLayout()
        name_label = QLabel(profile_name)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        name_label.setFont(font)
        layout.addWidget(name_label)
        
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666666;")
        layout.addWidget(desc_label)
        
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)


class ConfigSectionWidget(QWidget):
    """Widget for a configuration section with collapsible groups."""
    def __init__(self, title: str, descriptions: Dict, config_dict: Dict, callback):
        super().__init__()
        self.descriptions = descriptions
        self.config_dict = config_dict
        self.callback = callback
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        title_label = QLabel(title)
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #1976d2; margin-top: 5px;")
        layout.addWidget(title_label)
        
        self._create_groups(layout)
        self.setLayout(layout)
    
    def _create_groups(self, parent_layout):
        if "yaw_range" in self.config_dict:
            self._create_basic_group("Basic Tracking", ["yaw_range", "pitch_range", "filter_length"], parent_layout)
            self._create_basic_group("Inversion", ["invert_x", "invert_y"], parent_layout)
            self._create_basic_group("Confidence", ["min_detection_confidence", "min_tracking_confidence"], parent_layout)
        elif "max_speed_px_per_s" in self.config_dict:
            self._create_basic_group("Relative Mode Settings", ["max_speed_px_per_s", "deadzone_deg", "expo"], parent_layout)
        elif "alpha" in self.config_dict or "process_noise" in self.config_dict:
            self._create_filter_group(parent_layout)
        else:
            self._create_basic_group("Settings", list(self.config_dict.keys()), parent_layout)
    
    def _create_basic_group(self, group_name: str, param_names: list, parent_layout):
        group = QGroupBox(group_name)
        layout = QVBoxLayout()
        
        for param_name in param_names:
            if param_name in self.config_dict:
                value = self.config_dict[param_name]
                desc = self.descriptions.get(param_name, ConfigDescription(
                        name=param_name.replace('_', ' ').title(),
                        description=f"Configure {param_name.replace('_', ' ')} setting"
                    ))
                
                if param_name in ["invert_x", "invert_y", "start_mouse_enabled"]:
                    param_type = "bool"
                elif param_name in ["type"]:
                    param_type = "choice"
                else:
                    param_type = "float" if isinstance(value, float) else "int"
                
                control = ConfigControlFactory.create_control(self, param_name, value, param_type, desc, self.callback)
                layout.addWidget(control)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def _create_filter_group(self, parent_layout):
        group = QGroupBox("Filter Configuration")
        layout = QVBoxLayout()
        
        type_desc = self.descriptions.get("type", ConfigDescription(name="Filter Type", description="Choose filtering algorithm"))
        type_control = ConfigControlFactory.create_control(self, "type", self.config_dict.get("type", "kalman"), "choice", type_desc, self.callback)
        layout.addWidget(type_control)
        
        current_type = self.config_dict.get("type", "kalman")
        if current_type == "ema" and "alpha" in self.config_dict:
            alpha_desc = self.descriptions.get("alpha", ConfigDescription(name="EMA Alpha", description="Smoothing factor (0-1)", min_val=0.1, max_val=0.9, step=0.05))
            layout.addWidget(ConfigControlFactory.create_control(self, "alpha", self.config_dict["alpha"], "float", alpha_desc, self.callback))
        elif current_type == "kalman":
            for param in ["process_noise", "measurement_noise", "dt"]:
                if param in self.config_dict:
                    desc = self.descriptions.get(param, ConfigDescription(name=param.replace('_', ' ').title(), description=f"Kalman filter {param.replace('_', ' ')}"))
                    layout.addWidget(ConfigControlFactory.create_control(self, param, self.config_dict[param], "float", desc, self.callback))
        
        group.setLayout(layout)
        parent_layout.addWidget(group)


class ConfigGUI(QMainWindow):
    """Main configuration window for Head Tracking System."""
    
    def __init__(self):
        super().__init__()
        
        # --- PATHING FIX: Force GUI to target its exact directory ---
        self.config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        
        self.config = self.load_config()
        self.profiles = self.config.get("profiles", {})
        self.current_profile = self.profiles.get("active", "BALANCED") if self.profiles else "BALANCED"
        self.modified = False
        
        self.setWindowTitle("Head Tracking Config")
        self.setMinimumSize(700, 400)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #fafafa; }
            QLabel { color: #333333; }
            QPushButton {
                background-color: #2196f3; color: white; border: none;
                padding: 8px 12px; border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background-color: #1976d2; }
            QPushButton:pressed { background-color: #0d47a1; }
            QPushButton.secondary { background-color: #f5f5f5; color: #333333; border: 1px solid #cccccc; }
            QPushButton.secondary:hover { background-color: #e0e0e0; }
            QTabWidget::pane { border: 1px solid #cccccc; border-radius: 5px; padding: 5px; background-color: white; }
            QTabBar::tab {
                background-color: #f0f0f0; border: 1px solid #cccccc; border-bottom: none;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
                padding: 6px 12px; margin-right: 2px;
            }
            QTabBar::tab:selected { background-color: white; border-bottom: 2px solid #2196f3; }
            QGroupBox {
                font-weight: bold; border: 1px solid #cccccc; border-radius: 5px;
                margin-top: 15px; padding-top: 15px; padding-bottom: 10px; padding-left: 10px; padding-right: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; top: 0px; padding: 0 5px; }
            QScrollArea { border: none; background-color: transparent; }
        """)
        
        self.setup_ui()
        self.statusBar().showMessage("Ready to configure head tracking settings")
    
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        main_layout.setSpacing(5)
        
        title = QLabel("EyeTalk Configuration")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #1976d2; margin: 5px;")
        main_layout.addWidget(title)
        
        splitter = QSplitter(Qt.Horizontal)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self.create_general_tab(), "General")
        self.tab_widget.addTab(self.create_tracking_tab(), "Tracking")
        self.tab_widget.addTab(self.create_control_tab(), "Control")
        self.tab_widget.addTab(self.create_filter_tab(), "Filter")
        self.tab_widget.addTab(self.create_profiles_tab(), "Profiles")
        
        left_layout.addWidget(self.tab_widget)
        
        right_widget = self.create_info_panel()
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        
        main_layout.addWidget(splitter)
        
        button_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save Config")
        save_btn.clicked.connect(self.save_config_dialog)
        
        load_btn = QPushButton("Load Config")
        load_btn.clicked.connect(self.load_config_dialog)
        load_btn.setProperty("class", "secondary")
        
        reset_btn = QPushButton("Reset Defaults")
        reset_btn.clicked.connect(self.reset_to_defaults)
        reset_btn.setProperty("class", "secondary")
        
        start_btn = QPushButton("START TRACKING")
        start_btn.clicked.connect(self.start_tracking)
        start_font = QFont()
        start_font.setPointSize(12)
        start_font.setBold(True)
        start_btn.setFont(start_font)
        start_btn.setStyleSheet("QPushButton { background-color: #4caf50; color: white; border-radius: 4px; padding: 12px; } QPushButton:hover { background-color: #388e3c; }")
        
        button_layout.addWidget(save_btn)
        button_layout.addWidget(load_btn)
        button_layout.addWidget(reset_btn)
        button_layout.addWidget(start_btn, stretch=1)
        
        main_layout.addLayout(button_layout)
        central_widget.setLayout(main_layout)
    
    def create_general_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout()
        
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QGridLayout()
        camera_layout.addWidget(QLabel("Camera ID:"), 0, 0)
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 10)
        self.camera_spin.setValue(self.config.get("camera", {}).get("id", 0))
        camera_layout.addWidget(self.camera_spin, 0, 1)
        
        camera_layout.addWidget(QLabel("Resolution:"), 1, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "1280x720", "1920x1080"])
        self.resolution_combo.setCurrentText(self.config.get("camera", {}).get("resolution", "640x480"))
        camera_layout.addWidget(self.resolution_combo, 1, 1)
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        cal_group = QGroupBox("Calibration")
        cal_layout = QGridLayout()
        cal_layout.addWidget(QLabel("Samples:"), 0, 0)
        self.cal_samples_spin = QSpinBox()
        self.cal_samples_spin.setRange(10, 100)
        self.cal_samples_spin.setValue(self.config.get("calibration", {}).get("samples", 30))
        cal_layout.addWidget(self.cal_samples_spin, 0, 1)
        
        cal_layout.addWidget(QLabel("Auto-save:"), 1, 0)
        self.cal_autosave_check = QCheckBox()
        self.cal_autosave_check.setChecked(self.config.get("calibration", {}).get("autosave", True))
        cal_layout.addWidget(self.cal_autosave_check, 1, 1)
        cal_group.setLayout(cal_layout)
        layout.addWidget(cal_group)
        
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout()
        self.show_landmarks_check = QCheckBox("Show facial landmarks")
        self.show_landmarks_check.setChecked(self.config.get("display", {}).get("show_landmarks", True))
        display_layout.addWidget(self.show_landmarks_check)
        
        self.show_cube_check = QCheckBox("Show head cube")
        self.show_cube_check.setChecked(self.config.get("display", {}).get("show_cube", True))
        display_layout.addWidget(self.show_cube_check)
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_tracking_tab(self) -> QWidget:
        widget = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        
        tracking_config = self.config.get("tracking", {})
        layout.addWidget(ConfigSectionWidget("Tracking Parameters", ConfigDefinitions.TRACKING_DESCRIPTIONS, tracking_config, self.update_tracking_config))
        
        scroll.setWidget(content)
        main_layout = QVBoxLayout(widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
        return widget
    
    def create_control_tab(self) -> QWidget:
        widget = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        
        # --- EXPLICIT CONTROL MODE GROUP ---
        mode_group = QGroupBox("Tracking Control Mode")
        mode_layout = QVBoxLayout()
        self.absolute_radio = QCheckBox("Absolute Mode (Direct 1:1 Mapping)")
        self.relative_radio = QCheckBox("Relative Mode (Joystick-like Movement)")
        
        self.mode_btn_group = QButtonGroup()
        self.mode_btn_group.addButton(self.absolute_radio)
        self.mode_btn_group.addButton(self.relative_radio)
        
        if self.config.get("control", {}).get("default_mode", "absolute") == "absolute":
            self.absolute_radio.setChecked(True)
        else:
            self.relative_radio.setChecked(True)
        
        self.absolute_radio.toggled.connect(lambda c: self.update_mode("absolute" if c else "relative"))
        mode_layout.addWidget(self.absolute_radio)
        mode_layout.addWidget(self.relative_radio)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # --- EXPLICIT CLICKING METHOD GROUP ---
        clicking_group = QGroupBox("Automated Clicking Method")
        clicking_layout = QVBoxLayout()
        
        self.clicking_combo = QComboBox()
        self.clicking_combo.addItem("None (Standard Mouse Click)", "none")
        self.clicking_combo.addItem("Eye Blink (Double Blink)", "blink")
        self.clicking_combo.addItem("Dwell Time (Hold Still)", "dwell")
        
        current_click = self.config.get("control", {}).get("clicking_method", "none")
        idx = self.clicking_combo.findData(current_click)
        if idx >= 0:
            self.clicking_combo.setCurrentIndex(idx)
            
        click_desc_label = QLabel()
        click_desc_label.setWordWrap(True)
        click_desc_label.setStyleSheet("color: #666666; font-size: 11px; margin-top: 5px;")
        
        def update_click_desc():
            val = self.clicking_combo.currentData()
            self.update_control_config("clicking_method", val)
            if val == "none":
                click_desc_label.setText("Standard physical mouse clicks. No automatic clicking from camera.")
            elif val == "blink":
                click_desc_label.setText("Click by deliberately blinking your eyes (uses Eye Aspect Ratio).")
            elif val == "dwell":
                click_desc_label.setText("Click by holding the cursor perfectly still within a small area for 2 seconds.")
                
        self.clicking_combo.currentIndexChanged.connect(update_click_desc)
        update_click_desc() # initialize text
        
        clicking_layout.addWidget(self.clicking_combo)
        clicking_layout.addWidget(click_desc_label)
        clicking_group.setLayout(clicking_layout)
        layout.addWidget(clicking_group)

        # --- GENERAL CONTROL THRESHOLDS ---
        control_config = self.config.get("control", {})
        flat_control = {k: v for k, v in control_config.items() 
                        if not isinstance(v, dict) and k not in ["default_mode", "clicking_method"]}
        
        if flat_control:
            layout.addWidget(ConfigSectionWidget("General Thresholds", ConfigDefinitions.CONTROL_DESCRIPTIONS, flat_control, self.update_control_config))
        
        relative_config = control_config.get("relative", {})
        if relative_config:
            layout.addWidget(ConfigSectionWidget("Relative Mode Settings", ConfigDefinitions.RELATIVE_DESCRIPTIONS, relative_config, self.update_relative_config))
        
        scroll.setWidget(content)
        main_layout = QVBoxLayout(widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
        return widget
    
    def create_filter_tab(self) -> QWidget:
        widget = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        
        filter_config = self.config.get("filter", {})
        layout.addWidget(ConfigSectionWidget("Noise Filtering", ConfigDefinitions.FILTER_DESCRIPTIONS, filter_config, self.update_filter_config))
        
        smoothing_group = QGroupBox("Processing Rate")
        smoothing_layout = QHBoxLayout()
        smoothing_layout.addWidget(QLabel("Interval (s):"))
        self.process_interval_spin = QDoubleSpinBox()
        self.process_interval_spin.setRange(0.001, 0.1)
        self.process_interval_spin.setSingleStep(0.001)
        self.process_interval_spin.setValue(self.config.get("smoothing", {}).get("process_interval", 0.01))
        self.process_interval_spin.valueChanged.connect(lambda v: self.update_nested_config("smoothing", "process_interval", v))
        smoothing_layout.addWidget(self.process_interval_spin)
        smoothing_layout.addStretch()
        smoothing_group.setLayout(smoothing_layout)
        layout.addWidget(smoothing_group)
        
        scroll.setWidget(content)
        main_layout = QVBoxLayout(widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
        return widget
    
    def create_profiles_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        desc = QLabel("Select a profile to quickly switch between configurations.")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        for profile in ["FAST", "BALANCED", "SMOOTH"]:
            card = ProfileCard(profile, ConfigDefinitions.PROFILE_DESCRIPTIONS.get(profile, ""), (profile == self.current_profile))
            card.mousePressEvent = lambda e, p=profile: self.select_profile(p)
            card.setCursor(Qt.PointingHandCursor)
            layout.addWidget(card)
        
        apply_btn = QPushButton("Apply Selected Profile Settings")
        apply_btn.clicked.connect(self.apply_profile_settings)
        layout.addWidget(apply_btn)
        layout.addStretch()
        
        return widget
    
    def create_info_panel(self) -> QWidget:
        widget = QWidget()
        widget.setStyleSheet("QWidget { background-color: #f5f5f5; border-left: 1px solid #cccccc; }")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        title = QLabel("Information")
        title_font = QFont()
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setStyleSheet("QTextEdit { background-color: white; border: 1px solid #ccc; border-radius: 4px; }")
        self.info_text.setHtml("<p>Use the tabs to navigate settings.</p><p>When ready, click <b>START TRACKING</b>.</p>")
        layout.addWidget(self.info_text)
        
        return widget
    
    def select_profile(self, profile: str):
        self.current_profile = profile
        idx = self.tab_widget.currentIndex()
        self.tab_widget.removeTab(4)
        self.tab_widget.insertTab(4, self.create_profiles_tab(), "Profiles")
        self.tab_widget.setCurrentIndex(idx)
    
    def apply_profile_settings(self):
        if self.current_profile in self.profiles:
            profile_config = self.profiles.get(self.current_profile, {})
            for section, values in profile_config.items():
                if section not in self.config:
                    self.config[section] = {}
                if isinstance(values, dict):
                    for key, value in values.items():
                        self.config[section][key] = value
            
            if "profiles" not in self.config:
                self.config["profiles"] = {}
            self.config["profiles"]["active"] = self.current_profile
            
            self.modified = True
            QMessageBox.information(self, "Applied", f"{self.current_profile} applied.")
    
    def update_tracking_config(self, param: str, value):
        self.config.setdefault("tracking", {})[param] = value
        self.modified = True
    
    def update_control_config(self, param: str, value):
        self.config.setdefault("control", {})[param] = value
        self.modified = True
    
    def update_relative_config(self, param: str, value):
        self.config.setdefault("control", {}).setdefault("relative", {})[param] = value
        self.modified = True
    
    def update_filter_config(self, param: str, value):
        self.config.setdefault("filter", {})[param] = value
        self.modified = True
    
    def update_nested_config(self, section: str, param: str, value):
        self.config.setdefault(section, {})[param] = value
        self.modified = True
    
    def update_mode(self, mode: str):
        self.config.setdefault("control", {})["default_mode"] = mode
        self.modified = True
    
    def load_config(self) -> dict:
        # PATH FIX: Only look in the script's exact directory
        if os.path.exists(self.config_file_path):
            try:
                with open(self.config_file_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Error loading config: {e}")
        return self.get_default_config()
    
    def get_default_config(self) -> dict:
        return {
            "camera": {"id": 0, "resolution": "640x480"},
            "tracking": {"yaw_range": 20, "pitch_range": 10, "filter_length": 8, "invert_x": True, "invert_y": True, "min_detection_confidence": 0.5, "min_tracking_confidence": 0.5},
            "smoothing": {"process_interval": 0.01},
            "control": {
                "default_mode": "absolute", 
                "clicking_method": "none", 
                "start_mouse_enabled": True, 
                "movement_threshold_px": 0.5, 
                "pixel_deadzone_radius": 2.0, 
                "relative": {"max_speed_px_per_s": 1800.0, "deadzone_deg": 1.5, "expo": 1.6}
            },
            "filter": {"type": "kalman", "ema": {"alpha": 0.55}, "kalman": {"dt": 1.0, "process_noise": 0.002, "measurement_noise": 0.08}},
            "profiles": {"active": "BALANCED", "FAST": {}, "BALANCED": {}, "SMOOTH": {}},
            "calibration": {"file": "head_calibration.json", "autosave": True, "samples": 30},
            "mouse": {"update_interval": 0.01},
            "display": {"show_landmarks": True, "show_cube": True, "show_gaze_ray": True},
            "hotkeys": {"toggle_mouse": "7", "calibrate": "c", "cycle_profile": "p", "toggle_mode": "m", "quit": "q"}
        }
    
    def save_config(self, filepath: str) -> bool:
        try:
            self.config.setdefault("camera", {})["id"] = self.camera_spin.value()
            self.config["camera"]["resolution"] = self.resolution_combo.currentText()
            self.config.setdefault("calibration", {})["samples"] = self.cal_samples_spin.value()
            self.config["calibration"]["autosave"] = self.cal_autosave_check.isChecked()
            self.config.setdefault("display", {})["show_landmarks"] = self.show_landmarks_check.isChecked()
            self.config["display"]["show_cube"] = self.show_cube_check.isChecked()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            self.modified = False
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save configuration: {e}")
            return False
    
    def save_config_dialog(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Configuration", self.config_file_path, "YAML files (*.yaml)")
        if filepath:
            self.save_config(filepath)
    
    def load_config_dialog(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "YAML files (*.yaml)")
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
                self.modified = False
                QMessageBox.information(self, "Success", "Configuration loaded. Restart window to see changes.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {e}")
    
    def reset_to_defaults(self):
        if QMessageBox.question(self, "Reset", "Reset all settings to defaults?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.config = self.get_default_config()
            self.modified = True
            QMessageBox.information(self, "Success", "Settings reset. Restart window to see changes.")
    
    def start_tracking(self):
        # PATH FIX: Use absolute predefined path instead of relative string
        self.save_config(self.config_file_path)
        self.statusBar().showMessage("Starting head tracking...")
        self.hide()

        try:
            import subprocess
            main_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "head_tracking_control.py")

            if os.path.exists(main_script):
                env = os.environ.copy()
                env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
                env["PYTHONIOENCODING"] = "utf-8"

                if sys.platform == "win32":
                    cmd = f'cmd /c chcp 65001 > nul && python "{main_script}"'
                    subprocess.Popen(cmd, cwd=os.path.dirname(main_script), env=env, shell=True)
                else:
                    subprocess.Popen([sys.executable, main_script], cwd=os.path.dirname(main_script), env=env, text=True)

                self.close()   
            else:
                QMessageBox.critical(self, "Error", f"Could not find head_tracking_control.py.")
                self.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start tracking: {e}")
            self.show()


def main():
    app = QApplication(sys.argv)
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)
    window = ConfigGUI()
    window.showMaximized()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()