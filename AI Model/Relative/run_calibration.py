"""
Filename: enhanced_calibration.py
Creator/Author: Basel Mohamed Mostafa Sayed + ChatGPT
Date: 1/27/2026

Purpose:
    Enhanced calibration that captures:
    1. Eye movement range (min/max X and Y)
    2. Neutral gaze position
    3. Head position reference
    4. Simple linear mapping parameters
    
    Features:
    - Variable sampling time window
    - Real-time visualization
    - Progress indicators
    - Automatic parameter calculation
"""

import os
import sys
import time
import numpy as np
import pickle
import cv2
from datetime import datetime

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

# Import existing modules
try:
    from Calibration.calibration_ui import CalibrationWindow
    from VisionModule.camera import IrisCamera
    from VisionModule.face_utils import FaceMeshDetector
except ImportError:
    # Try alternative import path
    sys.path.insert(0, os.path.join(ROOT_DIR, 'AI Model'))
    from Calibration.calibration_ui import CalibrationWindow
    from VisionModule.camera import IrisCamera
    from VisionModule.face_utils import FaceMeshDetector


class EnhancedCalibrationData:
    """Stores enhanced calibration data for relative gaze mapping."""
    
    def __init__(self):
        # Eye position ranges
        self.left_eye_x = []
        self.left_eye_y = []
        self.right_eye_x = []
        self.right_eye_y = []
        
        # Average eye positions (for relative mapping)
        self.avg_eye_x = []
        self.avg_eye_y = []
        
        # Corresponding screen positions
        self.screen_x = []
        self.screen_y = []
        
        # Face center positions (for head stabilization)
        self.face_center_x = []
        self.face_center_y = []
        
        # Timestamps for each sample
        self.timestamps = []
        
        # Calculated parameters
        self.eye_x_range = None  # (min, max)
        self.eye_y_range = None  # (min, max)
        self.neutral_position = None  # (x, y) when looking at center
        self.screen_mapping = None  # Scaling factors
        
        # Calibration metadata
        self.metadata = {
            'creation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'num_points': 0,
            'total_samples': 0,
            'avg_samples_per_point': 0
        }
        
    def add_sample(self, screen_pos, iris_data, face_center=None):
        """Add a calibration sample with timestamp."""
        screen_x, screen_y = screen_pos
        
        if iris_data and iris_data.get('left') and iris_data.get('right'):
            left_x, left_y = iris_data['left']
            right_x, right_y = iris_data['right']
            avg_x = (left_x + right_x) / 2.0
            avg_y = (left_y + right_y) / 2.0
            
            self.left_eye_x.append(left_x)
            self.left_eye_y.append(left_y)
            self.right_eye_x.append(right_x)
            self.right_eye_y.append(right_y)
            self.avg_eye_x.append(avg_x)
            self.avg_eye_y.append(avg_y)
            self.screen_x.append(screen_x)
            self.screen_y.append(screen_y)
            self.timestamps.append(time.time())
            
            if face_center:
                self.face_center_x.append(face_center[0])
                self.face_center_y.append(face_center[1])
            else:
                self.face_center_x.append(None)
                self.face_center_y.append(None)
            
            return True
        return False
    
    def calculate_parameters(self):
        """Calculate mapping parameters from collected data."""
        if not self.avg_eye_x:
            print("[EnhancedCalibration] No data to calculate parameters")
            return False
        
        # Update metadata
        self.metadata['total_samples'] = len(self.avg_eye_x)
        
        # Convert to numpy arrays
        avg_eye_x = np.array(self.avg_eye_x)
        avg_eye_y = np.array(self.avg_eye_y)
        screen_x = np.array(self.screen_x)
        screen_y = np.array(self.screen_y)
        
        # Filter out any NaN values
        valid_mask = ~np.isnan(avg_eye_x) & ~np.isnan(avg_eye_y) & ~np.isnan(screen_x) & ~np.isnan(screen_y)
        if not np.any(valid_mask):
            print("[EnhancedCalibration] No valid data after filtering")
            return False
        
        avg_eye_x = avg_eye_x[valid_mask]
        avg_eye_y = avg_eye_y[valid_mask]
        screen_x = screen_x[valid_mask]
        screen_y = screen_y[valid_mask]
        
        # Calculate eye movement range
        self.eye_x_range = (np.min(avg_eye_x), np.max(avg_eye_x))
        self.eye_y_range = (np.min(avg_eye_y), np.max(avg_eye_y))
        
        # Find neutral position (when looking at screen center)
        # We'll use the sample closest to screen center (0.5, 0.5)
        center_distances = np.sqrt((screen_x - 0.5)**2 + (screen_y - 0.5)**2)
        center_idx = np.argmin(center_distances)
        self.neutral_position = (avg_eye_x[center_idx], avg_eye_y[center_idx])
        
        # Calculate simple linear mapping
        # For X: screen_x = a * eye_x + b
        # For Y: screen_y = c * eye_y + d
        try:
            A_x = np.vstack([avg_eye_x, np.ones(len(avg_eye_x))]).T
            A_y = np.vstack([avg_eye_y, np.ones(len(avg_eye_y))]).T
            
            # Solve using least squares
            coeff_x, residuals_x, _, _ = np.linalg.lstsq(A_x, screen_x, rcond=None)
            coeff_y, residuals_y, _, _ = np.linalg.lstsq(A_y, screen_y, rcond=None)
            
            # Calculate R² scores
            ss_res_x = residuals_x[0] if len(residuals_x) > 0 else 0
            ss_tot_x = np.sum((screen_x - np.mean(screen_x)) ** 2)
            r2_x = 1 - (ss_res_x / ss_tot_x) if ss_tot_x > 0 else 0
            
            ss_res_y = residuals_y[0] if len(residuals_y) > 0 else 0
            ss_tot_y = np.sum((screen_y - np.mean(screen_y)) ** 2)
            r2_y = 1 - (ss_res_y / ss_tot_y) if ss_tot_y > 0 else 0
            
            self.screen_mapping = {
                'x_slope': float(coeff_x[0]),
                'x_intercept': float(coeff_x[1]),
                'y_slope': float(coeff_y[0]),
                'y_intercept': float(coeff_y[1]),
                'r2_x': float(r2_x),
                'r2_y': float(r2_y),
                'rmse_x': float(np.sqrt(ss_res_x / len(screen_x)) if len(screen_x) > 0 else 0),
                'rmse_y': float(np.sqrt(ss_res_y / len(screen_y)) if len(screen_y) > 0 else 0)
            }
            
            print(f"[EnhancedCalibration] Linear mapping calculated:")
            print(f"  X: R² = {r2_x:.4f}, RMSE = {self.screen_mapping['rmse_x']:.4f}")
            print(f"  Y: R² = {r2_y:.4f}, RMSE = {self.screen_mapping['rmse_y']:.4f}")
            
        except Exception as e:
            print(f"[EnhancedCalibration] Error calculating linear mapping: {e}")
            # Fallback to simple scaling
            eye_x_range = self.eye_x_range[1] - self.eye_x_range[0]
            eye_y_range = self.eye_y_range[1] - self.eye_y_range[0]
            
            if eye_x_range > 0 and eye_y_range > 0:
                self.screen_mapping = {
                    'x_slope': 1.0 / eye_x_range,
                    'x_intercept': -self.eye_x_range[0] / eye_x_range,
                    'y_slope': 1.0 / eye_y_range,
                    'y_intercept': -self.eye_y_range[0] / eye_y_range,
                    'r2_x': 0,
                    'r2_y': 0,
                    'rmse_x': 0,
                    'rmse_y': 0
                }
        
        return True
    
    def get_eye_movement_range(self):
        """Get the range of eye movement in pixels."""
        if self.eye_x_range and self.eye_y_range:
            x_range = self.eye_x_range[1] - self.eye_x_range[0]
            y_range = self.eye_y_range[1] - self.eye_y_range[0]
            return x_range, y_range
        return None, None
    
    def save(self, filename="enhanced_calibration.pkl"):
        """Save calibration data to file."""
        # Update metadata before saving
        self.metadata['num_points'] = len(set(zip(self.screen_x, self.screen_y)))
        if self.metadata['num_points'] > 0:
            self.metadata['avg_samples_per_point'] = self.metadata['total_samples'] / self.metadata['num_points']
        
        data = {
            'left_eye_x': self.left_eye_x,
            'left_eye_y': self.left_eye_y,
            'right_eye_x': self.right_eye_x,
            'right_eye_y': self.right_eye_y,
            'avg_eye_x': self.avg_eye_x,
            'avg_eye_y': self.avg_eye_y,
            'screen_x': self.screen_x,
            'screen_y': self.screen_y,
            'face_center_x': self.face_center_x,
            'face_center_y': self.face_center_y,
            'timestamps': self.timestamps,
            'eye_x_range': self.eye_x_range,
            'eye_y_range': self.eye_y_range,
            'neutral_position': self.neutral_position,
            'screen_mapping': self.screen_mapping,
            'metadata': self.metadata,
            'timestamp': time.time()
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"[EnhancedCalibration] Saved to {filename}")
        return True
    
    def load(self, filename="enhanced_calibration.pkl"):
        """Load calibration data from file."""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            self.left_eye_x = data.get('left_eye_x', [])
            self.left_eye_y = data.get('left_eye_y', [])
            self.right_eye_x = data.get('right_eye_x', [])
            self.right_eye_y = data.get('right_eye_y', [])
            self.avg_eye_x = data.get('avg_eye_x', [])
            self.avg_eye_y = data.get('avg_eye_y', [])
            self.screen_x = data.get('screen_x', [])
            self.screen_y = data.get('screen_y', [])
            self.face_center_x = data.get('face_center_x', [])
            self.face_center_y = data.get('face_center_y', [])
            self.timestamps = data.get('timestamps', [])
            self.eye_x_range = data.get('eye_x_range')
            self.eye_y_range = data.get('eye_y_range')
            self.neutral_position = data.get('neutral_position')
            self.screen_mapping = data.get('screen_mapping')
            self.metadata = data.get('metadata', {})
            
            print(f"[EnhancedCalibration] Loaded from {filename}")
            return True
            
        except Exception as e:
            print(f"[EnhancedCalibration] Error loading: {e}")
            return False
    
    def visualize(self, save_path="enhanced_calibration_visualization.png"):
        """Create visualization plots of the calibration data."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            
            if not self.avg_eye_x or not self.screen_x:
                print("[EnhancedCalibration] No data to visualize")
                return False
            
            # Create figure with multiple subplots
            fig = plt.figure(figsize=(16, 10))
            fig.suptitle('Enhanced Calibration Data Visualization', fontsize=16, fontweight='bold')
            
            # Create grid layout
            gs = GridSpec(3, 3, figure=fig)
            
            # Plot 1: Screen positions (actual calibration points)
            ax1 = fig.add_subplot(gs[0, 0])
            unique_points = set(zip(self.screen_x, self.screen_y))
            for sx, sy in unique_points:
                ax1.scatter(sx, sy, c='blue', s=100, alpha=0.7, edgecolors='black')
            ax1.set_xlabel('Screen X (normalized)')
            ax1.set_ylabel('Screen Y (normalized)')
            ax1.set_title('Calibration Points on Screen')
            ax1.set_xlim(-0.05, 1.05)
            ax1.set_ylim(-0.05, 1.05)
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal', adjustable='box')
            
            # Plot 2: Eye positions distribution
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.scatter(self.avg_eye_x, self.avg_eye_y, c='red', s=30, alpha=0.6)
            if self.eye_x_range and self.eye_y_range:
                x_min, x_max = self.eye_x_range
                y_min, y_max = self.eye_y_range
                ax2.add_patch(plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                          fill=False, edgecolor='green', linewidth=2,
                                          linestyle='--', label='Eye Range'))
            if self.neutral_position:
                ax2.scatter(self.neutral_position[0], self.neutral_position[1],
                          c='green', s=200, marker='*', edgecolors='black',
                          label='Neutral Position')
            ax2.set_xlabel('Eye X (pixels)')
            ax2.set_ylabel('Eye Y (pixels)')
            ax2.set_title('Eye Positions Distribution')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Plot 3: Screen X vs Eye X correlation
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.scatter(self.avg_eye_x, self.screen_x, c='green', s=30, alpha=0.6)
            if self.screen_mapping:
                # Plot regression line
                x_vals = np.array([min(self.avg_eye_x), max(self.avg_eye_x)])
                y_vals = self.screen_mapping['x_slope'] * x_vals + self.screen_mapping['x_intercept']
                ax3.plot(x_vals, y_vals, 'r-', linewidth=2, label=f"R² = {self.screen_mapping['r2_x']:.3f}")
                ax3.legend()
            ax3.set_xlabel('Eye X (pixels)')
            ax3.set_ylabel('Screen X (normalized)')
            ax3.set_title('Eye X → Screen X Mapping')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Screen Y vs Eye Y correlation
            ax4 = fig.add_subplot(gs[1, 0])
            ax4.scatter(self.avg_eye_y, self.screen_y, c='purple', s=30, alpha=0.6)
            if self.screen_mapping:
                # Plot regression line
                y_vals = np.array([min(self.avg_eye_y), max(self.avg_eye_y)])
                x_vals = self.screen_mapping['y_slope'] * y_vals + self.screen_mapping['y_intercept']
                ax4.plot(y_vals, x_vals, 'r-', linewidth=2, label=f"R² = {self.screen_mapping['r2_y']:.3f}")
                ax4.legend()
            ax4.set_xlabel('Eye Y (pixels)')
            ax4.set_ylabel('Screen Y (normalized)')
            ax4.set_title('Eye Y → Screen Y Mapping')
            ax4.grid(True, alpha=0.3)
            
            # Plot 5: Histogram of eye X positions
            ax5 = fig.add_subplot(gs[1, 1])
            ax5.hist(self.avg_eye_x, bins=20, color='blue', alpha=0.7, edgecolor='black')
            ax5.set_xlabel('Eye X (pixels)')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Distribution of Eye X Positions')
            ax5.grid(True, alpha=0.3)
            
            # Plot 6: Histogram of eye Y positions
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.hist(self.avg_eye_y, bins=20, color='red', alpha=0.7, edgecolor='black')
            ax6.set_xlabel('Eye Y (pixels)')
            ax6.set_ylabel('Frequency')
            ax6.set_title('Distribution of Eye Y Positions')
            ax6.grid(True, alpha=0.3)
            
            # Plot 7: Time series of eye X movement
            ax7 = fig.add_subplot(gs[2, 0])
            if self.timestamps:
                rel_times = np.array(self.timestamps) - min(self.timestamps)
                ax7.plot(rel_times, self.avg_eye_x, 'b-', alpha=0.7, label='Eye X')
                ax7.set_xlabel('Time (seconds)')
                ax7.set_ylabel('Eye X (pixels)')
                ax7.set_title('Eye X Movement Over Time')
                ax7.grid(True, alpha=0.3)
                ax7.legend()
            
            # Plot 8: Time series of eye Y movement
            ax8 = fig.add_subplot(gs[2, 1])
            if self.timestamps:
                rel_times = np.array(self.timestamps) - min(self.timestamps)
                ax8.plot(rel_times, self.avg_eye_y, 'r-', alpha=0.7, label='Eye Y')
                ax8.set_xlabel('Time (seconds)')
                ax8.set_ylabel('Eye Y (pixels)')
                ax8.set_title('Eye Y Movement Over Time')
                ax8.grid(True, alpha=0.3)
                ax8.legend()
            
            # Plot 9: Information text
            ax9 = fig.add_subplot(gs[2, 2])
            ax9.axis('off')
            
            info_text = f"Calibration Summary:\n"
            info_text += f"Created: {self.metadata.get('creation_time', 'N/A')}\n"
            info_text += f"Total Samples: {self.metadata.get('total_samples', 0)}\n"
            info_text += f"Calibration Points: {self.metadata.get('num_points', 0)}\n"
            info_text += f"Avg Samples/Point: {self.metadata.get('avg_samples_per_point', 0):.1f}\n\n"
            
            if self.eye_x_range and self.eye_y_range:
                x_range = self.eye_x_range[1] - self.eye_x_range[0]
                y_range = self.eye_y_range[1] - self.eye_y_range[0]
                info_text += f"Eye Movement Range:\n"
                info_text += f"  X: {x_range:.1f} pixels\n"
                info_text += f"  Y: {y_range:.1f} pixels\n\n"
            
            if self.neutral_position:
                info_text += f"Neutral Position:\n"
                info_text += f"  X: {self.neutral_position[0]:.1f}\n"
                info_text += f"  Y: {self.neutral_position[1]:.1f}\n\n"
            
            if self.screen_mapping:
                info_text += f"Mapping Quality:\n"
                info_text += f"  X-axis R²: {self.screen_mapping.get('r2_x', 0):.4f}\n"
                info_text += f"  Y-axis R²: {self.screen_mapping.get('r2_y', 0):.4f}\n"
                info_text += f"  X RMSE: {self.screen_mapping.get('rmse_x', 0):.4f}\n"
                info_text += f"  Y RMSE: {self.screen_mapping.get('rmse_y', 0):.4f}"
            
            ax9.text(0.05, 0.95, info_text, transform=ax9.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"[EnhancedCalibration] Visualization saved to {save_path}")
            return True
            
        except ImportError:
            print("[EnhancedCalibration] matplotlib not available for visualization")
            print("Install with: pip install matplotlib")
            return False
        except Exception as e:
            print(f"[EnhancedCalibration] Error creating visualization: {e}")
            return False


def get_face_center(landmarks, img_width, img_height):
    """
    Estimate face center from landmarks.
    Returns (center_x, center_y) or None if not available.
    """
    if not landmarks or len(landmarks) == 0:
        return None
    
    try:
        # Use key facial landmarks for better center estimation
        # Nose tip is usually a good reference
        nose_tip_indices = [1, 2, 4, 5, 19, 94, 164, 197]  # Various nose points
        
        valid_points = []
        for idx in nose_tip_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                x = int(lm.x * img_width)
                y = int(lm.y * img_height)
                valid_points.append((x, y))
        
        if valid_points:
            center_x = np.mean([p[0] for p in valid_points])
            center_y = np.mean([p[1] for p in valid_points])
            return (center_x, center_y)
        
        # Fallback: average of all landmarks
        all_x = [int(lm.x * img_width) for lm in landmarks]
        all_y = [int(lm.y * img_height) for lm in landmarks]
        return (np.mean(all_x), np.mean(all_y))
        
    except Exception as e:
        print(f"[FaceCenter] Error estimating face center: {e}")
        return None


def run_enhanced_calibration(
    output_path: str = "enhanced_calibration.pkl",
    camera_id: int = 1,
    num_points: int = 9,
    settle_duration: float = 1.0,
    sample_duration: float = 2.5,  # Variable sampling window!
    min_samples_per_point: int = 10,
    enable_visualization: bool = True
):
    """
    Run enhanced calibration with variable sampling time window.
    
    Args:
        output_path: Path to save calibration data
        camera_id: Camera ID
        num_points: Number of calibration points
        settle_duration: Time to wait before sampling (seconds)
        sample_duration: Time to sample at each point (seconds) - VARIABLE!
        min_samples_per_point: Minimum samples to collect per point
        enable_visualization: Generate visualization plots
    """
    app = QApplication(sys.argv)
    
    # Initialize UI
    window = CalibrationWindow(num_points=num_points)
    window.show()
    
    calibration_points = window.calibration_points
    if not calibration_points or len(calibration_points) < 4:
        raise RuntimeError("Not enough calibration points generated.")
    
    # Initialize camera and detector
    camera = IrisCamera(cam_id=camera_id)
    detector = FaceMeshDetector()
    
    # Initialize enhanced calibration data
    calib_data = EnhancedCalibrationData()
    
    # Setup timer
    timer = QTimer()
    timer.setInterval(33)  # ~30 Hz for smoother updates
    
    current_point = 0
    samples_collected = 0
    state = "SETTLING"  # SETTLING, SAMPLING, COMPLETE
    state_start_time = 0
    calibration_complete = False
    
    # Statistics
    total_samples = 0
    successful_samples = 0
    failed_samples = 0
    
    print("=" * 70)
    print("ENHANCED CALIBRATION - Variable Sampling Window")
    print("=" * 70)
    print(f"Points: {num_points}")
    print(f"Settle duration: {settle_duration}s")
    print(f"Sample duration: {sample_duration}s (adjustable!)")
    print(f"Min samples per point: {min_samples_per_point}")
    print("=" * 70)
    print("Look at each dot as it appears.")
    print("The dot will pulse during sampling.")
    print("Press ESC to cancel at any time.")
    print("=" * 70)
    
    def update_progress():
        """Update progress display."""
        nonlocal samples_collected, successful_samples, failed_samples
        
        if state == "SETTLING":
            elapsed = time.time() - state_start_time
            remaining = max(0, settle_duration - elapsed)
            print(f"\r[Calibration] Point {current_point + 1}/{num_points}: "
                  f"Settling... {remaining:.1f}s remaining", end="")
        
        elif state == "SAMPLING":
            elapsed = time.time() - state_start_time
            remaining = max(0, sample_duration - elapsed)
            progress = (elapsed / sample_duration) * 100
            
            print(f"\r[Calibration] Point {current_point + 1}/{num_points}: "
                  f"Sampling... {remaining:.1f}s remaining "
                  f"({progress:.0f}%) | "
                  f"Samples: {samples_collected}", end="")
    
    def finish_calibration():
        nonlocal calibration_complete
        
        print("\n\n" + "=" * 70)
        print("CALIBRATION COMPLETE")
        print("=" * 70)
        
        # Calculate parameters
        print("[Calibration] Calculating parameters...")
        success = calib_data.calculate_parameters()
        
        if success:
            # Show results
            x_range, y_range = calib_data.get_eye_movement_range()
            print(f"[Calibration] Eye movement range:")
            print(f"  X: {x_range:.1f} pixels "
                  f"({calib_data.eye_x_range[0]:.1f} to {calib_data.eye_x_range[1]:.1f})")
            print(f"  Y: {y_range:.1f} pixels "
                  f"({calib_data.eye_y_range[0]:.1f} to {calib_data.eye_y_range[1]:.1f})")
            
            if calib_data.neutral_position:
                print(f"[Calibration] Neutral position: "
                      f"({calib_data.neutral_position[0]:.1f}, {calib_data.neutral_position[1]:.1f})")
            
            if calib_data.screen_mapping:
                print(f"[Calibration] Linear mapping quality:")
                print(f"  X-axis: R² = {calib_data.screen_mapping['r2_x']:.4f}, "
                      f"RMSE = {calib_data.screen_mapping['rmse_x']:.4f}")
                print(f"  Y-axis: R² = {calib_data.screen_mapping['r2_y']:.4f}, "
                      f"RMSE = {calib_data.screen_mapping['rmse_y']:.4f}")
            
            # Save data
            calib_data.save(output_path)
            print(f"[Calibration] Data saved to {output_path}")
            
            # Generate visualization
            if enable_visualization:
                print("[Calibration] Generating visualization...")
                vis_path = output_path.replace('.pkl', '_visualization.png')
                calib_data.visualize(vis_path)
        
        # Show overall statistics
        print(f"\n[Calibration] Overall statistics:")
        print(f"  Total samples collected: {total_samples}")
        print(f"  Successful samples: {successful_samples}")
        print(f"  Failed samples: {failed_samples}")
        print(f"  Success rate: {successful_samples/max(1, total_samples)*100:.1f}%")
        
        print("\n" + "=" * 70)
        print("Calibration complete! Ready for gaze cursor control.")
        print("=" * 70)
        
        # Cleanup
        camera.stop_recording()
        detector.release()
        window.close()
        
        calibration_complete = True
    
    def tick():
        nonlocal current_point, samples_collected, state, state_start_time
        nonlocal total_samples, successful_samples, failed_samples
        
        # Check if window closed
        if not window.isVisible() or calibration_complete:
            timer.stop()
            if not calibration_complete:
                finish_calibration()
            return
        
        # If calibration complete
        if current_point >= num_points:
            timer.stop()
            finish_calibration()
            return
        
        # Get current calibration point
        screen_pos = calibration_points[current_point]
        
        # State machine
        if state == "SETTLING":
            # Show point and start settling
            window.show_calibration_point(current_point)
            window.central_widget.enable_pulse_animation(False)  # No pulse during settling
            
            if time.time() - state_start_time >= settle_duration:
                # Start sampling
                state = "SAMPLING"
                state_start_time = time.time()
                samples_collected = 0
                print(f"\n[Calibration] Starting sampling for point {current_point + 1}")
        
        elif state == "SAMPLING":
            # Enable pulse animation during sampling
            window.central_widget.enable_pulse_animation(True)
            
            # Check if sampling time is up
            if time.time() - state_start_time >= sample_duration:
                # Also check if we have minimum samples
                if samples_collected >= min_samples_per_point:
                    # Move to next point
                    print(f"\n[Calibration] Point {current_point + 1} complete "
                          f"({samples_collected} samples)")
                    current_point += 1
                    state = "SETTLING"
                    state_start_time = time.time()
                    window.central_widget.enable_pulse_animation(False)
                else:
                    # Continue sampling until we have minimum samples
                    print(f"\n[Calibration] Need more samples for point {current_point + 1} "
                          f"({samples_collected}/{min_samples_per_point})")
            
            # Continue sampling
            frame = camera.get_frame()
            if frame is None:
                failed_samples += 1
                total_samples += 1
                return
            
            # Get face mesh landmarks
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detector.results = detector.face_mesh.process(img_rgb)
            
            iris_data = None
            face_center = None
            
            if detector.results.multi_face_landmarks:
                # Get iris centers
                iris_data = detector.get_iris_centers(frame)
                
                # Get face landmarks for center estimation
                img_height, img_width, _ = frame.shape
                for face_landmarks in detector.results.multi_face_landmarks:
                    # Get face center
                    face_center = get_face_center(
                        face_landmarks.landmark, 
                        img_width, 
                        img_height
                    )
                    break  # Only first face
            
            # Add sample if we have iris data
            if iris_data and iris_data.get('left') and iris_data.get('right'):
                success = calib_data.add_sample(screen_pos, iris_data, face_center)
                
                if success:
                    samples_collected += 1
                    successful_samples += 1
                    total_samples += 1
                    
                    # Optional: Show iris centers on frame for debugging
                    # detector.draw_iris_centers(frame, iris_data, with_text=True)
                    # cv2.imshow("Calibration Preview", frame)
                    # cv2.waitKey(1)
                else:
                    failed_samples += 1
                    total_samples += 1
            else:
                failed_samples += 1
                total_samples += 1
        
        # Update progress display
        update_progress()
    
    timer.timeout.connect(tick)
    
    # Start with first point
    state = "SETTLING"
    state_start_time = time.time()
    
    # Start timer
    timer.start()
    
    # Run application
    sys.exit(app.exec())


def test_and_visualize(calib_path="enhanced_calibration.pkl"):
    """Test and visualize calibration data."""
    calib_data = EnhancedCalibrationData()
    
    if not calib_data.load(calib_path):
        print("Failed to load calibration data")
        return
    
    print("\n" + "=" * 70)
    print("CALIBRATION DATA ANALYSIS")
    print("=" * 70)
    
    # Basic stats
    print(f"Creation time: {calib_data.metadata.get('creation_time', 'N/A')}")
    print(f"Total samples: {calib_data.metadata.get('total_samples', 0)}")
    print(f"Calibration points: {calib_data.metadata.get('num_points', 0)}")
    print(f"Avg samples per point: {calib_data.metadata.get('avg_samples_per_point', 0):.1f}")
    
    if calib_data.eye_x_range and calib_data.eye_y_range:
        x_min, x_max = calib_data.eye_x_range
        y_min, y_max = calib_data.eye_y_range
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        print(f"\nEye movement range:")
        print(f"  X: {x_min:.1f} to {x_max:.1f} (Δ={x_range:.1f} px)")
        print(f"  Y: {y_min:.1f} to {y_max:.1f} (Δ={y_range:.1f} px)")
    
    if calib_data.neutral_position:
        print(f"\nNeutral position: {calib_data.neutral_position}")
    
    if calib_data.screen_mapping:
        print(f"\nLinear mapping parameters:")
        print(f"  X: screen = {calib_data.screen_mapping['x_slope']:.6f} * eye_x + {calib_data.screen_mapping['x_intercept']:.6f}")
        print(f"  Y: screen = {calib_data.screen_mapping['y_slope']:.6f} * eye_y + {calib_data.screen_mapping['y_intercept']:.6f}")
        print(f"\nMapping quality:")
        print(f"  X-axis R²: {calib_data.screen_mapping.get('r2_x', 0):.4f}")
        print(f"  Y-axis R²: {calib_data.screen_mapping.get('r2_y', 0):.4f}")
        print(f"  X RMSE: {calib_data.screen_mapping.get('rmse_x', 0):.4f}")
        print(f"  Y RMSE: {calib_data.screen_mapping.get('rmse_y', 0):.4f}")
    
    # Generate visualization
    print("\nGenerating visualization...")
    vis_path = calib_path.replace('.pkl', '_visualization.png')
    calib_data.visualize(vis_path)
    
    print("=" * 70)


if __name__ == "__main__":
    # Run enhanced calibration with variable sampling window
    run_enhanced_calibration(
        output_path="enhanced_calibration.pkl",
        camera_id=1,           # Based on your setup
        num_points=9,          # 3x3 grid
        settle_duration=1.0,   # 1 second to settle
        sample_duration=2.5,   # 2.5 seconds sampling window - ADJUST THIS!
        min_samples_per_point=15,  # Minimum samples required
        enable_visualization=True  # Generate visualization
    )
    
    # Uncomment to test and visualize existing calibration data
    # test_and_visualize("enhanced_calibration.pkl")