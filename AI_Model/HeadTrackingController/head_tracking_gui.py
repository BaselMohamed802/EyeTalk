"""
Filename: head_tracking_config_gui.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 2/2/2026

Description:
    PySide6 GUI configuration interface for Head Tracking Mouse Control System.
    Allows users to tune all parameters before starting the tracking session.
"""

import sys
import os
import json
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

# Fix for the Qt DPI warning on Windows
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCALE_FACTOR"] = "1"

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QLabel, QSlider, QCheckBox, QComboBox,
    QPushButton, QSpinBox, QDoubleSpinBox, QMessageBox, QFileDialog,
    QScrollArea, QFrame, QSplitter, QTextEdit, QGridLayout, QStatusBar,
    QButtonGroup
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QFont, QPalette, QColor, QIcon


@dataclass
class ConfigDescription:
    """Data class for configuration parameter descriptions."""
    name: str
    description: str
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    step: float = 1.0
    unit: str = ""
    effect_increase: str = ""
    effect_decrease: str = ""


class ConfigDefinitions:
    """Central repository of all configuration descriptions."""
    
    # Tracking section descriptions
    TRACKING_DESCRIPTIONS = {
        "yaw_range": ConfigDescription(
            name="Yaw Range (degrees)",
            description="How far left/right you need to turn to reach screen edges",
            min_val=5, max_val=45, step=1, unit="°",
            effect_increase="Less head movement needed to reach edges (more sensitive)",
            effect_decrease="More head movement needed to reach edges (more precise)"
        ),
        "pitch_range": ConfigDescription(
            name="Pitch Range (degrees)",
            description="How far up/down you need to tilt to reach screen edges",
            min_val=5, max_val=30, step=1, unit="°",
            effect_increase="Less head tilt needed (more sensitive)",
            effect_decrease="More head tilt needed (more precise)"
        ),
        "filter_length": ConfigDescription(
            name="Position Buffer Size",
            description="Number of frames to average for smoothing head position",
            min_val=1, max_val=20, step=1,
            effect_increase="Smoother but more laggy movement",
            effect_decrease="More responsive but potentially jittery"
        ),
        "invert_x": ConfigDescription(
            name="Invert Horizontal",
            description="Reverse left/right movement direction",
            effect_increase="Turns inversion ON",
            effect_decrease="Turns inversion OFF"
        ),
        "invert_y": ConfigDescription(
            name="Invert Vertical",
            description="Reverse up/down movement direction",
            effect_increase="Turns inversion ON",
            effect_decrease="Turns inversion OFF"
        ),
        "min_detection_confidence": ConfigDescription(
            name="Detection Confidence",
            description="Minimum confidence to detect a face (0.0-1.0)",
            min_val=0.1, max_val=0.9, step=0.05,
            effect_increase="More reliable detection but may miss faces",
            effect_decrease="Detects faces more easily but may have false positives"
        ),
        "min_tracking_confidence": ConfigDescription(
            name="Tracking Confidence",
            description="Minimum confidence to track face landmarks (0.0-1.0)",
            min_val=0.1, max_val=0.9, step=0.05,
            effect_increase="More stable tracking but may lose tracking",
            effect_decrease="Keeps tracking longer but may be less accurate"
        ),
    }
    
    # Control section descriptions
    CONTROL_DESCRIPTIONS = {
        "default_mode": ConfigDescription(
            name="Control Mode",
            description="Absolute: cursor at head position | Relative: head acts like joystick",
        ),
        "movement_threshold_px": ConfigDescription(
            name="Movement Threshold (pixels)",
            description="Minimum pixel movement required to update cursor",
            min_val=0.1, max_val=5.0, step=0.1, unit="px",
            effect_increase="Less jitter but cursor feels less responsive",
            effect_decrease="More responsive but may be jittery"
        ),
        "pixel_deadzone_radius": ConfigDescription(
            name="Deadzone Radius (pixels)",
            description="Area where small movements are ignored",
            min_val=0.0, max_val=10.0, step=0.5, unit="px",
            effect_increase="More stable center but harder to make small movements",
            effect_decrease="More precise but may drift when still"
        ),
        "start_mouse_enabled": ConfigDescription(
            name="Start with Mouse Enabled",
            description="Whether mouse control is active when tracking starts",
            effect_increase="Mouse control starts enabled",
            effect_decrease="Mouse control starts disabled"
        ),
    }
    
    # Relative mode descriptions
    RELATIVE_DESCRIPTIONS = {
        "max_speed_px_per_s": ConfigDescription(
            name="Maximum Speed",
            description="Maximum cursor speed in relative mode",
            min_val=500, max_val=3000, step=50, unit="px/s",
            effect_increase="Faster cursor movement",
            effect_decrease="Slower, more controlled movement"
        ),
        "deadzone_deg": ConfigDescription(
            name="Deadzone (degrees)",
            description="Head movement ignored around center",
            min_val=0.0, max_val=5.0, step=0.1, unit="°",
            effect_increase="More stable center but requires larger movements to start",
            effect_decrease="More responsive but may drift"
        ),
        "expo": ConfigDescription(
            name="Exponential Factor",
            description="Makes small movements precise, large movements fast",
            min_val=1.0, max_val=2.5, step=0.1,
            effect_increase="More dramatic speed curve (fast/slow difference)",
            effect_decrease="More linear response"
        ),
    }
    
    # Filter descriptions
    FILTER_DESCRIPTIONS = {
        "type": ConfigDescription(
            name="Filter Type",
            description="EMA: Simple smoothing | Kalman: Predictive filtering",
        ),
        "alpha": ConfigDescription(
            name="EMA Smoothing Factor",
            description="How much weight new measurements have (EMA only)",
            min_val=0.1, max_val=0.9, step=0.05,
            effect_increase="More responsive but less smooth",
            effect_decrease="Smoother but more lag"
        ),
        "process_noise": ConfigDescription(
            name="Process Noise",
            description="How much we trust our motion model (Kalman)",
            min_val=0.0001, max_val=0.01, step=0.0005,
            effect_increase="Filter adapts faster to movement",
            effect_decrease="More stable but slower to respond"
        ),
        "measurement_noise": ConfigDescription(
            name="Measurement Noise",
            description="How much we trust new measurements (Kalman)",
            min_val=0.01, max_val=0.5, step=0.01,
            effect_increase="More smoothing, less responsive",
            effect_decrease="More responsive, less smoothing"
        ),
        "dt": ConfigDescription(
            name="Time Step",
            description="Time step for Kalman filter (seconds)",
            min_val=0.1, max_val=2.0, step=0.1, unit="s",
            effect_increase="Slower filter response",
            effect_decrease="Faster filter response"
        ),
    }
    
    # Profile descriptions
    PROFILE_DESCRIPTIONS = {
        "FAST": "Low smoothing, high responsiveness - best for gaming/fast navigation",
        "BALANCED": "Good balance of smoothness and responsiveness - default for most users",
        "SMOOTH": "High smoothing, very stable - best for precise work/reading",
    }


class ConfigControlFactory:
    """Factory class to create appropriate Qt controls for different parameter types."""
    
    @staticmethod
    def create_control(parent, param_name: str, value: Any, param_type: str,
                      description: ConfigDescription, callback):
        """Create the appropriate control widget based on parameter type."""
        
        if param_type == "bool" or isinstance(value, bool):
            return ConfigControlFactory._create_checkbox(parent, param_name, value, description, callback)
        elif param_type == "choice" or param_name in ["default_mode", "type"]:
            return ConfigControlFactory._create_combobox(parent, param_name, value, description, callback)
        elif isinstance(value, (int, float)):
            if isinstance(value, int) and param_name not in ["max_speed_px_per_s"]:
                return ConfigControlFactory._create_spinbox(parent, param_name, value, description, callback, is_float=False)
            else:
                return ConfigControlFactory._create_spinbox(parent, param_name, value, description, callback, is_float=True)
        else:
            # Fallback for unhandled types
            main_layout = QVBoxLayout()
            main_layout.setContentsMargins(0, 0, 0, 8)
            
            label = QLabel(f"{description.name}: {str(value)}")
            label.setStyleSheet("background-color: #f0f0f0; padding: 5px; border-radius: 3px;")
            main_layout.addWidget(label)
            
            desc_label = QLabel(description.description)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("""
                QLabel {
                    color: #555555;
                    font-size: 13px;
                    padding: 6px 8px;
                    background-color: #f9f9f9;
                    border-left: 3px solid #2196f3;
                    border-radius: 3px;
                }
            """)
            main_layout.addWidget(desc_label)
            
            container = QWidget()
            container.setLayout(main_layout)
            return container
    
    @staticmethod
    def _create_checkbox(parent, param_name: str, value: bool, description: ConfigDescription, callback):
        """Create a checkbox with full-width inline description below it."""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 8)
        main_layout.setSpacing(4)
        
        # Checkbox
        checkbox = QCheckBox(description.name)
        checkbox.setChecked(value)
        checkbox.stateChanged.connect(lambda state: callback(param_name, bool(state)))
        main_layout.addWidget(checkbox)
        
        # Build description text
        desc_lines = [description.description]
        if description.effect_increase and description.effect_decrease:
            desc_lines.append(f"✓ ON: {description.effect_increase}")
            desc_lines.append(f"☐ OFF: {description.effect_decrease}")
        
        desc_text = "\n".join(desc_lines)
        
        # Full-width description label
        desc_label = QLabel(desc_text)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("""
            QLabel {
                color: #555555;
                font-size: 13px;
                padding: 6px 8px;
                background-color: #f9f9f9;
                border-left: 3px solid #2196f3;
                border-radius: 3px;
                margin-top: 2px;
            }
        """)
        main_layout.addWidget(desc_label)
        
        return checkbox
    
    @staticmethod
    def _create_combobox(parent, param_name: str, value: str, description: ConfigDescription, callback):
        """Create a combobox with full-width inline description below it."""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 8)
        main_layout.setSpacing(4)
        
        # Create horizontal layout for label and combo
        top_layout = QHBoxLayout()
        
        combo = QComboBox()
        
        # Define choices based on parameter
        if param_name == "default_mode":
            choices = ["absolute", "relative"]
            choice_descriptions = {
                "absolute": "Cursor follows head position directly (1:1 mapping)",
                "relative": "Head movement controls cursor speed (like a joystick)"
            }
        elif param_name == "type":
            choices = ["kalman", "ema"]
            choice_descriptions = {
                "kalman": "Predictive filter - better for smooth, predictable movement",
                "ema": "Simple exponential smoothing - faster response but less smooth"
            }
        else:
            choices = [str(value)]
            choice_descriptions = {}
        
        for choice in choices:
            combo.addItem(choice.capitalize(), choice)
        
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)
        
        combo.currentIndexChanged.connect(
            lambda: callback(param_name, combo.currentData())
        )
        
        label = QLabel(description.name)
        label.setMinimumWidth(150)
        top_layout.addWidget(label)
        top_layout.addWidget(combo)
        top_layout.addStretch()
        
        main_layout.addLayout(top_layout)
        
        # Full-width description
        desc_lines = [description.description]
        
        # Add choice-specific description if available
        current_choice = combo.currentData()
        if current_choice in choice_descriptions:
            desc_lines.append(f"Current selection ({current_choice}): {choice_descriptions[current_choice]}")
        
        desc_text = "\n".join(desc_lines)
        
        desc_label = QLabel(desc_text)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("""
            QLabel {
                color: #555555;
                font-size: 13px;
                padding: 6px 8px;
                background-color: #f9f9f9;
                border-left: 3px solid #2196f3;
                border-radius: 3px;
                margin-top: 2px;
            }
        """)
        main_layout.addWidget(desc_label)
        
        # Update description when combo changes
        def update_description():
            current = combo.currentData()
            new_desc_lines = [description.description]
            if current in choice_descriptions:
                new_desc_lines.append(f"Current selection ({current}): {choice_descriptions[current]}")
            desc_label.setText("\n".join(new_desc_lines))
        
        combo.currentIndexChanged.connect(update_description)
        
        container = QWidget()
        container.setLayout(main_layout)
        return container
    
    @staticmethod
    def _create_spinbox(parent, param_name: str, value: float, description: ConfigDescription,
                       callback, is_float: bool = True):
        """Create a spinbox with full-width inline description below it."""
        # Create main vertical layout for the entire control
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 8)
        main_layout.setSpacing(4)
        
        # Create horizontal layout for label and spinbox
        top_layout = QHBoxLayout()
        
        # Label
        label = QLabel(description.name)
        label.setMinimumWidth(150)
        top_layout.addWidget(label)
        
        # Spinbox
        if is_float:
            spin = QDoubleSpinBox()
            spin.setDecimals(3 if value < 0.01 else 2)
            spin.setSingleStep(description.step)
        else:
            spin = QSpinBox()
            spin.setSingleStep(int(description.step))
        
        if description.min_val is not None:
            spin.setMinimum(description.min_val)
        if description.max_val is not None:
            spin.setMaximum(description.max_val)
        
        spin.setValue(value)
        spin.valueChanged.connect(lambda val: callback(param_name, val))
        
        # Add unit if present
        if description.unit:
            spin.setSuffix(f" {description.unit}")
        
        top_layout.addWidget(spin)
        top_layout.addStretch()
        
        main_layout.addLayout(top_layout)
        
        # Build full-width description text
        desc_lines = [description.description]
        
        if description.effect_increase:
            desc_lines.append(f"↑ INCREASE: {description.effect_increase}")
        if description.effect_decrease:
            desc_lines.append(f"↓ DECREASE: {description.effect_decrease}")
        
        if description.min_val is not None and description.max_val is not None:
            desc_lines.append(f" Range: {description.min_val} - {description.max_val} {description.unit}")
        
        desc_text = "\n".join(desc_lines)
        
        # Full-width description label with visual styling
        desc_label = QLabel(desc_text)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("""
            QLabel {
                color: #555555;
                font-size: 13px;
                padding: 8px 10px;
                background-color: #f9f9f9;
                border-left: 3px solid #2196f3;
                border-radius: 4px;
                margin-top: 4px;
            }
        """)
        main_layout.addWidget(desc_label)
        
        # Update description when value changes (to show current value)
        def update_description_with_value(current_value):
            new_desc_lines = [description.description]
            
            if description.effect_increase:
                new_desc_lines.append(f"↑ INCREASE: {description.effect_increase}")
            if description.effect_decrease:
                new_desc_lines.append(f"↓ DECREASE: {description.effect_decrease}")
            
            if description.min_val is not None and description.max_val is not None:
                new_desc_lines.append(f" Range: {description.min_val} - {description.max_val} {description.unit}")
            
            new_desc_lines.append(f" Current value: {current_value} {description.unit}")
            
            desc_label.setText("\n".join(new_desc_lines))
        
        # Connect to update description when value changes
        spin.valueChanged.connect(update_description_with_value)
        
        # Initial update
        update_description_with_value(value)
        
        container = QWidget()
        container.setLayout(main_layout)
        return container

class ProfileCard(QFrame):
    """Card-style widget for displaying profile information."""
    
    def __init__(self, profile_name: str, description: str, is_active: bool = False):
        super().__init__()
        self.profile_name = profile_name
        self.setFrameStyle(QFrame.Box)
        self.setLineWidth(2)
        
        # Set colors based on active state
        if is_active:
            self.setStyleSheet("""
                QFrame {
                    background-color: #e3f2fd;
                    border-color: #2196f3;
                    border-radius: 5px;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame {
                    background-color: #f5f5f5;
                    border-color: #cccccc;
                    border-radius: 5px;
                }
            """)
        
        layout = QVBoxLayout()
        
        # Profile name
        name_label = QLabel(profile_name)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        name_label.setFont(font)
        layout.addWidget(name_label)
        
        # Description
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666666;")
        layout.addWidget(desc_label)
        
        self.setLayout(layout)
        self.setMinimumHeight(80)
        self.setMaximumHeight(100)


class ConfigSectionWidget(QWidget):
    """Widget for a configuration section with collapsible groups."""
    
    def __init__(self, title: str, descriptions: Dict, config_dict: Dict, callback):
        super().__init__()
        self.descriptions = descriptions
        self.config_dict = config_dict
        self.callback = callback
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Section title
        title_label = QLabel(title)
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #1976d2; margin-top: 10px;")
        layout.addWidget(title_label)
        
        # Create groups for logical organization
        self._create_groups(layout)
        
        self.setLayout(layout)
    
    def _create_groups(self, parent_layout):
        """Create organized groups within the section."""
        
        # For tracking section, create subgroups
        if "yaw_range" in self.config_dict:
            self._create_basic_group("Basic Tracking", 
                ["yaw_range", "pitch_range", "filter_length"], parent_layout)
            self._create_basic_group("Inversion", 
                ["invert_x", "invert_y"], parent_layout)
            self._create_basic_group("Confidence", 
                ["min_detection_confidence", "min_tracking_confidence"], parent_layout)
        
        elif "max_speed_px_per_s" in self.config_dict:
            self._create_basic_group("Relative Mode Settings", 
                ["max_speed_px_per_s", "deadzone_deg", "expo"], parent_layout)
        
        elif "alpha" in self.config_dict or "process_noise" in self.config_dict:
            self._create_filter_group(parent_layout)
        
        else:
            # Generic group for other parameters
            self._create_basic_group("Settings", list(self.config_dict.keys()), parent_layout)
    
    def _create_basic_group(self, group_name: str, param_names: list, parent_layout):
        """Create a basic group box with parameters."""
        group = QGroupBox(group_name)
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        layout = QVBoxLayout()
        
        for param_name in param_names:
            if param_name in self.config_dict:
                value = self.config_dict[param_name]
                # Get description or create a default one
                desc = self.descriptions.get(param_name)
                if desc is None:
                    # Create a default description for parameters not in our definitions
                    desc = ConfigDescription(
                        name=param_name.replace('_', ' ').title(),
                        description=f"Configure {param_name.replace('_', ' ')} setting"
                    )
                
                # Determine parameter type
                if param_name in ["invert_x", "invert_y", "start_mouse_enabled"]:
                    param_type = "bool"
                elif param_name in ["default_mode", "type"]:
                    param_type = "choice"
                else:
                    param_type = "float" if isinstance(value, float) else "int"
                
                control = ConfigControlFactory.create_control(
                    self, param_name, value, param_type, desc, self.callback
                )
                layout.addWidget(control)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)
    
    def _create_filter_group(self, parent_layout):
        """Create specialized group for filter settings."""
        group = QGroupBox("Filter Configuration")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Filter type selector
        type_desc = self.descriptions.get("type", 
            ConfigDescription(name="Filter Type", description="Choose filtering algorithm"))
        type_control = ConfigControlFactory.create_control(
            self, "type", self.config_dict.get("type", "kalman"), 
            "choice", type_desc, self.callback
        )
        layout.addWidget(type_control)
        
        # Dynamic filter parameters based on type
        current_type = self.config_dict.get("type", "kalman")
        
        if current_type == "ema":
            if "alpha" in self.config_dict:
                alpha_desc = self.descriptions.get("alpha", 
                    ConfigDescription(name="EMA Alpha", description="Smoothing factor (0-1)", 
                                    min_val=0.1, max_val=0.9, step=0.05))
                alpha_control = ConfigControlFactory.create_control(
                    self, "alpha", self.config_dict["alpha"], "float", alpha_desc, self.callback
                )
                layout.addWidget(alpha_control)
        
        elif current_type == "kalman":
            kalman_params = ["process_noise", "measurement_noise", "dt"]
            for param in kalman_params:
                if param in self.config_dict:
                    desc = self.descriptions.get(param)
                    if desc is None:
                        desc = ConfigDescription(
                            name=param.replace('_', ' ').title(),
                            description=f"Kalman filter {param.replace('_', ' ')} parameter"
                        )
                    control = ConfigControlFactory.create_control(
                        self, param, self.config_dict[param], "float", desc, self.callback
                    )
                    layout.addWidget(control)
        
        group.setLayout(layout)
        parent_layout.addWidget(group)


class ConfigGUI(QMainWindow):
    """Main configuration window for Head Tracking System."""
    
    def __init__(self):
        super().__init__()
        self.config = self.load_config()
        self.profiles = self.config.get("profiles", {})
        self.current_profile = self.profiles.get("active", "BALANCED") if self.profiles else "BALANCED"
        self.modified = False
        
        self.setWindowTitle("Head Tracking Mouse Control - Configuration")
        self.setMinimumSize(1000, 700)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #fafafa;
            }
            QLabel {
                color: #333333;
            }
            QPushButton {
                background-color: #2196f3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976d2;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QPushButton.secondary {
                background-color: #f5f5f5;
                color: #333333;
                border: 1px solid #cccccc;
            }
            QPushButton.secondary:hover {
                background-color: #e0e0e0;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 10px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #2196f3;
            }
            QTabBar::tab:hover {
                background-color: #e3f2fd;
            }
            QGroupBox {
                font-weight: bold;
                margin-top: 10px;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QStatusBar {
                background-color: #f0f0f0;
                color: #666666;
            }
        """)
        
        self.setup_ui()
        self.statusBar().showMessage("Ready to configure head tracking settings")
    
    def setup_ui(self):
        """Setup the main user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        
        # Title
        title = QLabel("Head Tracking Mouse Control - Advanced Configuration")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #1976d2; margin: 10px;")
        main_layout.addWidget(title)
        
        # Description
        desc = QLabel(
            "Fine-tune your head tracking experience. Hover over any control for detailed "
            "descriptions of what each setting does. Changes take effect when you start tracking."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666666; margin-bottom: 10px;")
        desc.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(desc)
        
        # Create splitter for main content and info panel
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - Tabbed configuration
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self.create_general_tab(), "General")
        self.tab_widget.addTab(self.create_tracking_tab(), "Tracking")
        self.tab_widget.addTab(self.create_control_tab(), "Control")
        self.tab_widget.addTab(self.create_filter_tab(), "Filter")
        self.tab_widget.addTab(self.create_profiles_tab(), "Profiles")
        
        left_layout.addWidget(self.tab_widget)
        left_widget.setLayout(left_layout)
        
        # Right side - Info panel
        right_widget = self.create_info_panel()
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([700, 300])
        
        main_layout.addWidget(splitter)
        
        # Button bar
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # Save/Load buttons
        save_btn = QPushButton("Save Configuration")
        save_btn.clicked.connect(self.save_config_dialog)
        save_btn.setMinimumWidth(150)
        button_layout.addWidget(save_btn)
        
        load_btn = QPushButton("Load Configuration")
        load_btn.clicked.connect(self.load_config_dialog)
        load_btn.setMinimumWidth(150)
        load_btn.setProperty("class", "secondary")
        button_layout.addWidget(load_btn)
        
        # Reset button
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_to_defaults)
        reset_btn.setMinimumWidth(150)
        reset_btn.setProperty("class", "secondary")
        button_layout.addWidget(reset_btn)
        
        # Start button (big and prominent)
        start_btn = QPushButton("START TRACKING")
        start_btn.clicked.connect(self.start_tracking)
        start_btn.setMinimumWidth(200)
        start_btn.setMinimumHeight(50)
        start_font = QFont()
        start_font.setPointSize(14)
        start_font.setBold(True)
        start_btn.setFont(start_font)
        start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #388e3c;
            }
        """)
        button_layout.addWidget(start_btn)
        
        button_layout.addStretch()
        main_layout.addLayout(button_layout)
        
        central_widget.setLayout(main_layout)
    
    def create_general_tab(self) -> QWidget:
        """Create the general settings tab."""
        widget = QWidget()
        
        # Create a scroll area to handle content that might be too tall
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        content = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(20)  # Consistent spacing between groups
        layout.setContentsMargins(15, 15, 15, 15)  # Consistent margins
        
        # Camera selection
        camera_group = QGroupBox("Camera Settings")
        camera_group.setStyleSheet("""
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                top: 0px;
                padding: 0 5px 0 5px;
            }
        """)
        camera_layout = QGridLayout()
        camera_layout.setSpacing(12)
        camera_layout.setContentsMargins(10, 15, 10, 15)
        
        camera_layout.addWidget(QLabel("Camera ID:"), 0, 0)
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 10)
        self.camera_spin.setValue(0)
        self.camera_spin.setMinimumWidth(100)
        camera_layout.addWidget(self.camera_spin, 0, 1)
        
        camera_layout.addWidget(QLabel("Resolution:"), 1, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "1280x720", "1920x1080"])
        self.resolution_combo.setCurrentText("640x480")
        self.resolution_combo.setMinimumWidth(120)
        camera_layout.addWidget(self.resolution_combo, 1, 1)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        # Calibration settings
        cal_group = QGroupBox("Calibration")
        cal_group.setStyleSheet("""
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                top: 0px;
                padding: 0 5px 0 5px;
            }
        """)
        cal_layout = QGridLayout()
        cal_layout.setSpacing(8)
        cal_layout.setContentsMargins(10, 15, 10, 15)

        cal_layout.addWidget(QLabel("Samples:"), 0, 0)
        self.cal_samples_spin = QSpinBox()
        self.cal_samples_spin.setRange(10, 100)
        cal_samples = self.config.get("calibration", {}).get("samples", 30)
        self.cal_samples_spin.setValue(cal_samples)
        self.cal_samples_spin.setMinimumWidth(100)
        cal_layout.addWidget(self.cal_samples_spin, 0, 1)

        cal_layout.addWidget(QLabel("Auto-save:"), 1, 0)
        self.cal_autosave_check = QCheckBox()
        cal_autosave = self.config.get("calibration", {}).get("autosave", True)
        self.cal_autosave_check.setChecked(cal_autosave)
        cal_layout.addWidget(self.cal_autosave_check, 1, 1)

        cal_group.setLayout(cal_layout)
        layout.addWidget(cal_group)
        
        # ----- Clicking Options -------------------------------------------------
        click_group = QGroupBox("Clicking Options")
        click_group.setStyleSheet("""
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                top: 0px;
                padding: 0 5px 0 5px;
            }
        """)
        click_layout = QVBoxLayout()
        click_layout.setSpacing(10)
        click_layout.setContentsMargins(10, 15, 10, 15)

        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Click method:"))
        self.click_method_combo = QComboBox()
        self.click_method_combo.addItem("Disabled (manual click)", "disabled")
        self.click_method_combo.addItem("Eye Blink", "blink")
        self.click_method_combo.addItem("Dwell Click", "dwell")   # NEW
        click_method = self.config.get("clicking", {}).get("method", "disabled")
        idx = self.click_method_combo.findData(click_method)
        if idx >= 0:
            self.click_method_combo.setCurrentIndex(idx)
        self.click_method_combo.currentIndexChanged.connect(self.on_click_method_changed)
        method_layout.addWidget(self.click_method_combo)
        method_layout.addStretch()
        click_layout.addLayout(method_layout)

        # Blink parameters container (shown only when blink is selected)
        self.blink_params_widget = QWidget()
        blink_params_layout = QGridLayout()
        blink_params_layout.setSpacing(8)

        blink_cfg = self.config.get("clicking", {}).get("blink", {})

        self.ear_threshold_spin = QDoubleSpinBox()
        self.ear_threshold_spin.setRange(0.10, 0.35)
        self.ear_threshold_spin.setSingleStep(0.01)
        self.ear_threshold_spin.setDecimals(3)
        self.ear_threshold_spin.setValue(blink_cfg.get("ear_threshold", 0.22))
        self.ear_threshold_spin.setToolTip("EAR value below which eye is considered closed. Lower = harder blink needed.")
        self.ear_threshold_spin.valueChanged.connect(
            lambda v: self.update_clicking_blink_param("ear_threshold", v))
        blink_params_layout.addWidget(QLabel("EAR threshold:"), 0, 0)
        blink_params_layout.addWidget(self.ear_threshold_spin, 0, 1)

        self.consec_frames_spin = QSpinBox()
        self.consec_frames_spin.setRange(1, 6)
        self.consec_frames_spin.setValue(blink_cfg.get("consec_frames", 2))
        self.consec_frames_spin.setToolTip("How many consecutive closed-eye frames confirm a blink.")
        self.consec_frames_spin.valueChanged.connect(
            lambda v: self.update_clicking_blink_param("consec_frames", v))
        blink_params_layout.addWidget(QLabel("Consecutive frames:"), 1, 0)
        blink_params_layout.addWidget(self.consec_frames_spin, 1, 1)

        self.cooldown_spin = QDoubleSpinBox()
        self.cooldown_spin.setRange(0.3, 2.5)
        self.cooldown_spin.setSingleStep(0.1)
        self.cooldown_spin.setDecimals(2)
        self.cooldown_spin.setValue(blink_cfg.get("cooldown_sec", 1.0))
        self.cooldown_spin.setToolTip("Minimum seconds between two click actions.")
        self.cooldown_spin.valueChanged.connect(
            lambda v: self.update_clicking_blink_param("cooldown_sec", v))
        blink_params_layout.addWidget(QLabel("Cooldown (sec):"), 2, 0)
        blink_params_layout.addWidget(self.cooldown_spin, 2, 1)

        self.both_eyes_check = QCheckBox("Require both eyes to blink")
        self.both_eyes_check.setChecked(blink_cfg.get("use_both_eyes", False))
        self.both_eyes_check.setToolTip("If checked, both eyes must blink simultaneously to trigger a click.")
        self.both_eyes_check.stateChanged.connect(
            lambda state: self.update_clicking_blink_param("use_both_eyes", bool(state)))
        blink_params_layout.addWidget(self.both_eyes_check, 3, 0, 1, 2)

        self.blink_params_widget.setLayout(blink_params_layout)
        click_layout.addWidget(self.blink_params_widget)

        # Dwell parameters container (shown only when dwell is selected)
        self.dwell_params_widget = QWidget()
        dwell_params_layout = QGridLayout()
        dwell_params_layout.setSpacing(8)

        dwell_cfg = self.config.get("clicking", {}).get("dwell", {})

        self.dwell_time_spin = QDoubleSpinBox()
        self.dwell_time_spin.setRange(0.5, 5.0)
        self.dwell_time_spin.setSingleStep(0.1)
        self.dwell_time_spin.setDecimals(1)
        self.dwell_time_spin.setValue(dwell_cfg.get("dwell_time_sec", 2.0))
        self.dwell_time_spin.setToolTip("Seconds the cursor must stay still to trigger a click.")
        self.dwell_time_spin.valueChanged.connect(
            lambda v: self.update_clicking_dwell_param("dwell_time_sec", v))
        dwell_params_layout.addWidget(QLabel("Dwell time (sec):"), 0, 0)
        dwell_params_layout.addWidget(self.dwell_time_spin, 0, 1)

        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(5.0, 100.0)
        self.tolerance_spin.setSingleStep(5.0)
        self.tolerance_spin.setDecimals(1)
        self.tolerance_spin.setValue(dwell_cfg.get("tolerance_px", 30.0))
        self.tolerance_spin.setToolTip("Maximum pixel movement allowed while counting as 'still'.")
        self.tolerance_spin.valueChanged.connect(
            lambda v: self.update_clicking_dwell_param("tolerance_px", v))
        dwell_params_layout.addWidget(QLabel("Tolerance (px):"), 1, 0)
        dwell_params_layout.addWidget(self.tolerance_spin, 1, 1)

        self.dwell_params_widget.setLayout(dwell_params_layout)
        click_layout.addWidget(self.dwell_params_widget)

        # Initially show/hide the correct params widget based on current selection
        self.on_click_method_changed()

        click_group.setLayout(click_layout)
        layout.addWidget(click_group)
        # -----------------------------------------------------------------------
        
        # Display settings
        display_group = QGroupBox("Display Options")
        display_group.setStyleSheet("""
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                top: 0px;
                padding: 0 5px 0 5px;
            }
        """)
        display_layout = QVBoxLayout()
        display_layout.setSpacing(10)
        display_layout.setContentsMargins(10, 15, 10, 15)
        
        self.show_landmarks_check = QCheckBox("Show facial landmarks")
        show_landmarks = self.config.get("display", {}).get("show_landmarks", True)
        self.show_landmarks_check.setChecked(show_landmarks)
        display_layout.addWidget(self.show_landmarks_check)
        
        self.show_cube_check = QCheckBox("Show head cube")
        show_cube = self.config.get("display", {}).get("show_cube", True)
        self.show_cube_check.setChecked(show_cube)
        display_layout.addWidget(self.show_cube_check)
        
        self.show_gaze_check = QCheckBox("Show gaze ray")
        show_gaze = self.config.get("display", {}).get("show_gaze_ray", True)
        self.show_gaze_check.setChecked(show_gaze)
        display_layout.addWidget(self.show_gaze_check)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # Hotkey display
        hotkey_group = QGroupBox("Hotkeys (OpenCV window must be focused)")
        hotkey_group.setStyleSheet("""
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                top: 0px;
                padding: 0 5px 0 5px;
            }
        """)
        hotkey_layout = QGridLayout()
        hotkey_layout.setSpacing(10)
        hotkey_layout.setVerticalSpacing(8)
        hotkey_layout.setContentsMargins(10, 15, 10, 15)
        
        hotkeys = self.config.get("hotkeys", {})
        row = 0
        for key, value in hotkeys.items():
            key_text = key.replace('_', ' ').title()
            key_label = QLabel(f"{key_text}:")
            key_label.setMinimumWidth(120)
            hotkey_layout.addWidget(key_label, row, 0)
            
            value_label = QLabel(f"'{value}'")
            value_label.setStyleSheet("""
                QLabel {
                    font-family: monospace;
                    font-weight: bold;
                    color: #2196f3;
                    background-color: #f5f5f5;
                    padding: 4px 8px;
                    border-radius: 4px;
                }
            """)
            hotkey_layout.addWidget(value_label, row, 1)
            row += 1
        
        hotkey_group.setLayout(hotkey_layout)
        layout.addWidget(hotkey_group)
        
        # Add stretch at the end to push everything up
        layout.addStretch()
        
        content.setLayout(layout)
        scroll.setWidget(content)
        
        # Create main layout for the tab
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
        widget.setLayout(main_layout)
        
        return widget
    
    def create_tracking_tab(self) -> QWidget:
        """Create the tracking settings tab."""
        widget = QWidget()
        
        # Create scroll area for many controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        content = QWidget()
        layout = QVBoxLayout()
        
        # Tracking section
        tracking_config = self.config.get("tracking", {})
        tracking_widget = ConfigSectionWidget(
            "Tracking Parameters",
            ConfigDefinitions.TRACKING_DESCRIPTIONS,
            tracking_config,
            self.update_tracking_config
        )
        layout.addWidget(tracking_widget)
        
        content.setLayout(layout)
        scroll.setWidget(content)
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        widget.setLayout(main_layout)
        
        return widget
    
    def create_control_tab(self) -> QWidget:
        """Create the control settings tab."""
        widget = QWidget()
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        content = QWidget()
        layout = QVBoxLayout()
        
        # Mode selection
        mode_group = QGroupBox("Control Mode")
        mode_group.setStyleSheet("""
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                top: 0px;
                padding: 0 5px 0 5px;
            }
        """)
        mode_layout = QVBoxLayout()
        
        self.absolute_radio = QCheckBox("Absolute Mode")
        self.relative_radio = QCheckBox("Relative Mode")
        
        # Create a button group
        self.mode_btn_group = QButtonGroup()
        self.mode_btn_group.addButton(self.absolute_radio)
        self.mode_btn_group.addButton(self.relative_radio)
        
        current_mode = self.config.get("control", {}).get("default_mode", "absolute")
        if current_mode == "absolute":
            self.absolute_radio.setChecked(True)
        else:
            self.relative_radio.setChecked(True)
        
        self.absolute_radio.toggled.connect(lambda checked: self.update_mode("absolute" if checked else "relative"))
        
        mode_layout.addWidget(QLabel("Absolute: Cursor follows head position directly"))
        mode_layout.addWidget(QLabel("Relative: Head movement controls cursor speed (like a joystick)"))
        mode_layout.addWidget(self.absolute_radio)
        mode_layout.addWidget(self.relative_radio)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # General control settings
        control_config = self.config.get("control", {})
        # Filter out nested dicts and non-scalar values
        flat_control = {k: v for k, v in control_config.items() if not isinstance(v, dict)}
        control_widget = ConfigSectionWidget(
            "General Control",
            ConfigDefinitions.CONTROL_DESCRIPTIONS,
            flat_control,
            self.update_control_config
        )
        layout.addWidget(control_widget)
        
        # Relative mode settings
        relative_config = control_config.get("relative", {})
        if relative_config:
            relative_widget = ConfigSectionWidget(
                "Relative Mode Settings",
                ConfigDefinitions.RELATIVE_DESCRIPTIONS,
                relative_config,
                self.update_relative_config
            )
            layout.addWidget(relative_widget)
        
        content.setLayout(layout)
        scroll.setWidget(content)
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        widget.setLayout(main_layout)
        
        return widget
    
    def create_filter_tab(self) -> QWidget:
        """Create the filter settings tab."""
        widget = QWidget()
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        content = QWidget()
        layout = QVBoxLayout()
        
        # Filter section
        filter_config = self.config.get("filter", {})
        filter_widget = ConfigSectionWidget(
            "Noise Filtering",
            ConfigDefinitions.FILTER_DESCRIPTIONS,
            filter_config,
            self.update_filter_config
        )
        layout.addWidget(filter_widget)
        
        # Smoothing interval
        smoothing_group = QGroupBox("Processing Rate")
        smoothing_layout = QHBoxLayout()
        
        smoothing_layout.addWidget(QLabel("Process interval (seconds):"))
        self.process_interval_spin = QDoubleSpinBox()
        self.process_interval_spin.setRange(0.001, 0.1)
        self.process_interval_spin.setSingleStep(0.001)
        self.process_interval_spin.setDecimals(3)
        process_interval = self.config.get("smoothing", {}).get("process_interval", 0.01)
        self.process_interval_spin.setValue(process_interval)
        self.process_interval_spin.setSuffix(" s")
        self.process_interval_spin.valueChanged.connect(
            lambda v: self.update_nested_config("smoothing", "process_interval", v)
        )
        smoothing_layout.addWidget(self.process_interval_spin)
        smoothing_layout.addStretch()
        
        smoothing_group.setLayout(smoothing_layout)
        layout.addWidget(smoothing_group)
        
        content.setLayout(layout)
        scroll.setWidget(content)
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        widget.setLayout(main_layout)
        
        return widget
    
    def create_profiles_tab(self) -> QWidget:
        """Create the profiles tab with preset configurations."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Description
        desc = QLabel(
            "Profiles provide pre-configured settings for different use cases. "
            "Select a profile to quickly switch between configurations."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666666; margin: 10px;")
        layout.addWidget(desc)
        
        # Profile cards
        profiles_layout = QVBoxLayout()
        
        for profile in ["FAST", "BALANCED", "SMOOTH"]:
            is_active = (profile == self.current_profile)
            card = ProfileCard(
                profile,
                ConfigDefinitions.PROFILE_DESCRIPTIONS.get(profile, "No description available"),
                is_active
            )
            
            # Make card clickable
            card.mousePressEvent = lambda e, p=profile: self.select_profile(p)
            card.setCursor(Qt.PointingHandCursor)
            
            profiles_layout.addWidget(card)
        
        layout.addLayout(profiles_layout)
        
        # Current profile indicator
        current_label = QLabel(f"Current Profile: {self.current_profile}")
        current_font = QFont()
        current_font.setBold(True)
        current_label.setFont(current_font)
        current_label.setAlignment(Qt.AlignCenter)
        current_label.setStyleSheet("color: #2196f3; margin: 10px;")
        layout.addWidget(current_label)
        
        # Apply profile button
        apply_btn = QPushButton("Apply Selected Profile Settings")
        apply_btn.clicked.connect(self.apply_profile_settings)
        apply_btn.setMinimumHeight(40)
        layout.addWidget(apply_btn)
        
        layout.addStretch()
        widget.setLayout(layout)
        
        return widget
    
    def create_info_panel(self) -> QWidget:
        """Create the information panel with parameter descriptions."""
        widget = QWidget()
        widget.setMaximumWidth(350)
        widget.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                border-left: 1px solid #cccccc;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Parameter Information")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Info text area
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        
        # Initial info
        self.info_text.setHtml("""
            <h3>Welcome to Configuration</h3>
            <p>Hover over any control to see detailed information about what it does.</p>
            <p>This panel will display comprehensive descriptions of each parameter including:</p>
            <ul>
                <li>What the parameter controls</li>
                <li>Recommended ranges</li>
                <li>Effects of increasing/decreasing</li>
            </ul>
            <p>Use the tabs to navigate between different configuration sections.</p>
            <p>When you're ready, click the green START TRACKING button to begin.</p>
        """)
        
        layout.addWidget(self.info_text)
        
        # Tips section
        tips_group = QGroupBox("Quick Tips")
        tips_group.setStyleSheet("""
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                top: 0px;
                padding: 0 5px 0 5px;
            }
        """)
        tips_layout = QVBoxLayout()
        
        tips = [
            "• Start with BALANCED profile",
            "• Calibrate before each session",
            "• Adjust deadzone if cursor drifts",
            "• Use RELATIVE mode for large screens",
            "• Higher smoothing = more lag",
            "• Press 'C' during tracking to recalibrate"
        ]
        
        for tip in tips:
            tips_layout.addWidget(QLabel(tip))
        
        tips_group.setLayout(tips_layout)
        layout.addWidget(tips_group)
        
        widget.setLayout(layout)
        return widget
    
    def select_profile(self, profile: str):
        """Select a profile (highlight it)."""
        self.current_profile = profile
        
        # Refresh profiles tab to update highlighting
        current_index = self.tab_widget.currentIndex()
        self.tab_widget.removeTab(4)
        self.tab_widget.insertTab(4, self.create_profiles_tab(), "Profiles")
        self.tab_widget.setCurrentIndex(current_index)
    
    def apply_profile_settings(self):
        """Apply the selected profile's settings to the current config."""
        if self.current_profile in self.profiles:
            profile_config = self.profiles.get(self.current_profile, {})
            
            # Update config with profile settings
            for section, values in profile_config.items():
                if section not in self.config:
                    self.config[section] = {}
                if isinstance(values, dict):
                    for key, value in values.items():
                        self.config[section][key] = value
            
            # Update active profile
            if "profiles" not in self.config:
                self.config["profiles"] = {}
            self.config["profiles"]["active"] = self.current_profile
            
            self.modified = True
            self.statusBar().showMessage(f"Applied {self.current_profile} profile settings", 3000)
            QMessageBox.information(self, "Profile Applied", 
                f"{self.current_profile} profile settings have been applied. Some changes may require restarting the configuration window to see updated values.")
    
    def update_tracking_config(self, param: str, value):
        """Update a tracking parameter."""
        if "tracking" not in self.config:
            self.config["tracking"] = {}
        self.config["tracking"][param] = value
        self.modified = True
    
    def update_control_config(self, param: str, value):
        """Update a control parameter."""
        if "control" not in self.config:
            self.config["control"] = {}
        self.config["control"][param] = value
        self.modified = True
    
    def update_relative_config(self, param: str, value):
        """Update a relative mode parameter."""
        if "control" not in self.config:
            self.config["control"] = {}
        if "relative" not in self.config["control"]:
            self.config["control"]["relative"] = {}
        self.config["control"]["relative"][param] = value
        self.modified = True
    
    def update_filter_config(self, param: str, value):
        """Update a filter parameter."""
        if "filter" not in self.config:
            self.config["filter"] = {}
        self.config["filter"][param] = value
        self.modified = True
    
    def update_nested_config(self, section: str, param: str, value):
        """Update a nested configuration value."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][param] = value
        self.modified = True
    
    def update_mode(self, mode: str):
        """Update the control mode."""
        if "control" not in self.config:
            self.config["control"] = {}
        self.config["control"]["default_mode"] = mode
        self.modified = True

    # ----- Clicking config methods -----
    def on_click_method_changed(self):
        """Show/hide the appropriate parameters widget."""
        method = self.click_method_combo.currentData()
        self.blink_params_widget.setVisible(method == "blink")
        self.dwell_params_widget.setVisible(method == "dwell")
        self.update_clicking_method(method)

    def update_clicking_method(self, method: str):
        if "clicking" not in self.config:
            self.config["clicking"] = {}
        self.config["clicking"]["method"] = method
        self.modified = True

    def update_clicking_blink_param(self, param: str, value):
        if "clicking" not in self.config:
            self.config["clicking"] = {}
        if "blink" not in self.config["clicking"]:
            self.config["clicking"]["blink"] = {}
        self.config["clicking"]["blink"][param] = value
        self.modified = True

    def update_clicking_dwell_param(self, param: str, value):
        if "clicking" not in self.config:
            self.config["clicking"] = {}
        if "dwell" not in self.config["clicking"]:
            self.config["clicking"]["dwell"] = {}
        self.config["clicking"]["dwell"][param] = value
        self.modified = True
    # -----------------------------------
    
    def load_config(self) -> dict:
        """Load configuration from config.yaml."""
        try:
            # Look for config in various locations
            possible_paths = [
                "config.yaml",
                os.path.join(os.path.dirname(__file__), "config.yaml"),
                os.path.join(os.getcwd(), "config.yaml")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        return yaml.safe_load(f) or {}
            
            # Return default if not found
            return self.get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> dict:
        """Return default configuration."""
        return {
            "tracking": {
                "yaw_range": 20,
                "pitch_range": 10,
                "filter_length": 8,
                "invert_x": True,
                "invert_y": True,
                "min_detection_confidence": 0.5,
                "min_tracking_confidence": 0.5,
            },
            "smoothing": {"process_interval": 0.01},
            "control": {
                "default_mode": "absolute",
                "start_mouse_enabled": True,
                "movement_threshold_px": 0.5,
                "pixel_deadzone_radius": 2.0,
                "relative": {
                    "max_speed_px_per_s": 1800.0,
                    "deadzone_deg": 1.5,
                    "expo": 1.6,
                },
            },
            "filter": {
                "type": "kalman",
                "ema": {"alpha": 0.55},
                "kalman": {"dt": 1.0, "process_noise": 0.002, "measurement_noise": 0.08},
            },
            "profiles": {
                "active": "BALANCED",
                "FAST": {},
                "BALANCED": {},
                "SMOOTH": {}
            },
            "calibration": {"file": "head_calibration.json", "autosave": True, "samples": 30},
            "mouse": {"update_interval": 0.01},
            "display": {"show_landmarks": True, "show_cube": True, "show_gaze_ray": True},
            "hotkeys": {
                "toggle_mouse": "7",
                "calibrate": "c",
                "cycle_profile": "p",
                "toggle_mode": "m",
                "quit": "q"
            },
            "clicking": {
                "method": "disabled",
                "blink": {
                    "ear_threshold": 0.22,
                    "consec_frames": 2,
                    "cooldown_sec": 1.0,
                    "use_both_eyes": False,
                },
                "dwell": {
                    "dwell_time_sec": 2.0,
                    "tolerance_px": 30.0,
                }
            }
        }
    
    def save_config(self, filepath: str) -> bool:
        """Save current configuration to file."""
        try:
            # Update config with GUI values from general tab
            if "calibration" not in self.config:
                self.config["calibration"] = {}
            self.config["calibration"]["samples"] = self.cal_samples_spin.value()
            self.config["calibration"]["autosave"] = self.cal_autosave_check.isChecked()
            
            if "display" not in self.config:
                self.config["display"] = {}
            self.config["display"]["show_landmarks"] = self.show_landmarks_check.isChecked()
            self.config["display"]["show_cube"] = self.show_cube_check.isChecked()
            self.config["display"]["show_gaze_ray"] = self.show_gaze_check.isChecked()
            
            if "smoothing" not in self.config:
                self.config["smoothing"] = {}
            self.config["smoothing"]["process_interval"] = self.process_interval_spin.value()

            # Clicking settings
            if "clicking" not in self.config:
                self.config["clicking"] = {}
            self.config["clicking"]["method"] = self.click_method_combo.currentData()
            self.config["clicking"]["blink"] = {
                "ear_threshold": self.ear_threshold_spin.value(),
                "consec_frames": self.consec_frames_spin.value(),
                "cooldown_sec": self.cooldown_spin.value(),
                "use_both_eyes": self.both_eyes_check.isChecked(),
            }
            self.config["clicking"]["dwell"] = {
                "dwell_time_sec": self.dwell_time_spin.value(),
                "tolerance_px": self.tolerance_spin.value(),
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            
            self.modified = False
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save configuration: {e}")
            return False
    
    def save_config_dialog(self):
        """Open save file dialog."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "config.yaml", "YAML files (*.yaml)"
        )
        if filepath:
            if self.save_config(filepath):
                self.statusBar().showMessage(f"Configuration saved to {filepath}", 3000)
    
    def load_config_dialog(self):
        """Open load file dialog."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", "YAML files (*.yaml)"
        )
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
                self.modified = False
                self.statusBar().showMessage(f"Configuration loaded from {filepath}", 3000)
                
                # Refresh UI
                QMessageBox.information(self, "Success", 
                    "Configuration loaded. Please restart the configuration window to see all changes.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load configuration: {e}")
    
    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        reply = QMessageBox.question(
            self, "Reset to Defaults",
            "Are you sure you want to reset all settings to default values?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.config = self.get_default_config()
            self.modified = True
            self.statusBar().showMessage("Configuration reset to defaults", 3000)
            
            # Refresh UI
            QMessageBox.information(self, "Success", 
                "Settings reset to defaults. Please restart the configuration window to see all changes.")
    
    def start_tracking(self):
        """Start the head tracking system with current configuration."""
        # Save current config
        if self.modified:
            reply = QMessageBox.question(
                self, "Save Changes",
                "You have unsaved changes. Do you want to save them before starting?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            if reply == QMessageBox.Cancel:
                return
            elif reply == QMessageBox.Save:
                if not self.save_config("config.yaml"):
                    return

        self.statusBar().showMessage("Starting head tracking...")
        self.hide()

        try:
            import subprocess
            import sys
            import os

            script_dir = os.path.dirname(__file__)
            main_script = os.path.join(script_dir, "head_tracking_control.py")

            if not os.path.exists(main_script):
                alt_paths = [
                    os.path.join(os.getcwd(), "head_tracking_control.py"),
                    os.path.join(os.path.dirname(script_dir), "head_tracking_control.py")
                ]
                for path in alt_paths:
                    if os.path.exists(path):
                        main_script = path
                        break

            if os.path.exists(main_script):
                # Prepare environment with protobuf fallback and UTF-8 output
                env = os.environ.copy()
                env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
                env["PYTHONIOENCODING"] = "utf-8"

                # On Windows, also try to set console to UTF-8 (optional)
                if sys.platform == "win32":
                    # Use "start" to open a new console that supports UTF-8
                    cmd = f'cmd /c chcp 65001 > nul && python "{main_script}"'
                    process = subprocess.Popen(
                        cmd,
                        cwd=os.path.dirname(main_script),
                        env=env,
                        shell=True
                    )
                else:
                    process = subprocess.Popen(
                        [sys.executable, main_script],
                        cwd=os.path.dirname(main_script),
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )

                self.statusBar().showMessage("Tracking launched")
                self.close()   # Close the GUI completely
            else:
                QMessageBox.critical(self, "Error", f"Could not find head_tracking_control.py")
                self.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start tracking: {e}")
            self.show()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    
    # Set application-wide font
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)
    
    # Create and show the configuration window
    window = ConfigGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()