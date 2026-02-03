"""
Filename: calibration_regression.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 1/26/2026

Description:
    Calibration regression model.
    
    Responsibilities:
    1. Accept calibration pairs from CalibrationSampler
    2. Train linear/polynomial regression model
    3. Validate input data quality
    4. Predict screen coordinates from iris coordinates
    5. Provide error metrics
    
    Input (from CalibrationSampler):
    [
        ((screen_x, screen_y), (eye_x, eye_y)),
        ((screen_x, screen_y), (eye_x, eye_y)),
        ...
    ]
    
    Training data format:
    X = [[eye_x, eye_y], ...]  # Features
    y = [[screen_x, screen_y], ...]  # Targets
    
    Output (prediction):
    Given: eye_x, eye_y
    Returns: screen_x_pred, screen_y_pred
"""

# Import libraries
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import json
import pickle

# Import sklearn components
try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. Using simple linear regression.")
    print("Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False


@dataclass
class CalibrationPair:
    """Represents a single calibration pair."""
    screen_x: float  # Normalized screen X (0-1)
    screen_y: float  # Normalized screen Y (0-1)
    eye_x: float     # Iris X coordinate (pixels)
    eye_y: float     # Iris Y coordinate (pixels)
    
    def to_training_format(self) -> Tuple[List[float], List[float]]:
        """Convert to (features, target) format."""
        features = [self.eye_x, self.eye_y]
        target = [self.screen_x, self.screen_y]
        return features, target


@dataclass
class RegressionMetrics:
    """Metrics for regression model evaluation."""
    mse_x: float = 0.0
    mse_y: float = 0.0
    mae_x: float = 0.0
    mae_y: float = 0.0
    r2_x: float = 0.0
    r2_y: float = 0.0
    max_error_x: float = 0.0
    max_error_y: float = 0.0
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            "Regression Metrics:",
            f"  X-axis:",
            f"    MSE: {self.mse_x:.4f}",
            f"    MAE: {self.mae_x:.4f}",
            f"    R²:  {self.r2_x:.4f}",
            f"    Max Error: {self.max_error_x:.4f}",
            f"  Y-axis:",
            f"    MSE: {self.mse_y:.4f}",
            f"    MAE: {self.mae_y:.4f}",
            f"    R²:  {self.r2_y:.4f}",
            f"    Max Error: {self.max_error_y:.4f}",
        ]
        return "\n".join(lines)


class CalibrationRegression:
    """
    Regression model for mapping iris coordinates to screen coordinates.
    
    This is the final step in the calibration pipeline.
    """
    
    def __init__(self, 
                 model_type: str = "linear",
                 polynomial_degree: int = 2,
                 regularization: float = 0.0):
        """
        Initialize regression model.
        
        Args:
            model_type: "linear", "polynomial", or "ridge"
            polynomial_degree: Degree for polynomial features (if using polynomial)
            regularization: Regularization strength (if using ridge regression)
        """
        self.model_type = model_type
        self.polynomial_degree = polynomial_degree
        self.regularization = regularization
        
        # Data storage
        self.calibration_pairs: List[CalibrationPair] = []
        self.is_trained = False
        
        # Models (one for X, one for Y)
        self.model_x = None
        self.model_y = None
        
        # Training metrics
        self.training_metrics = None
        
        print(f"Calibration regression initialized")
        print(f"Model type: {model_type}")
        if model_type == "polynomial":
            print(f"Polynomial degree: {polynomial_degree}")
    
    def add_calibration_pair(self, 
                            screen_pos: Tuple[float, float], 
                            eye_pos: Tuple[float, float]
                            ) -> bool:
        """
        Add a single calibration pair.
        
        Args:
            screen_pos: (screen_x, screen_y) normalized coordinates (0-1)
            eye_pos: (eye_x, eye_y) iris coordinates (pixels)
            
        Returns:
            True if pair was added, False if invalid
        """
        screen_x, screen_y = screen_pos
        eye_x, eye_y = eye_pos
        
        # Basic validation
        if not self._validate_pair(screen_x, screen_y, eye_x, eye_y):
            print(f"[Warning] Invalid calibration pair skipped: "
                  f"Screen({screen_x:.3f}, {screen_y:.3f}) → "
                  f"Eye({eye_x:.1f}, {eye_y:.1f})")
            return False
        
        # Create and store pair
        pair = CalibrationPair(
            screen_x=screen_x,
            screen_y=screen_y,
            eye_x=eye_x,
            eye_y=eye_y
        )
        
        self.calibration_pairs.append(pair)
        print(f"[Regression] Added pair {len(self.calibration_pairs)}: "
              f"Screen({screen_x:.3f}, {screen_y:.3f}) → "
              f"Eye({eye_x:.1f}, {eye_y:.1f})")
        
        return True
    
    def add_calibration_pairs(self, pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]]):
        """
        Add multiple calibration pairs at once.
        
        Args:
            pairs: List of ((screen_x, screen_y), (eye_x, eye_y)) tuples
        """
        added_count = 0
        for i, (screen_pos, eye_pos) in enumerate(pairs):
            if self.add_calibration_pair(screen_pos, eye_pos):
                added_count += 1
        
        print(f"[Regression] Added {added_count} calibration pairs "
              f"(total: {len(self.calibration_pairs)})")
    
    def _validate_pair(self, 
                      screen_x: float, 
                      screen_y: float, 
                      eye_x: float, 
                      eye_y: float) -> bool:
        """
        Validate a calibration pair.
        
        Returns:
            True if pair is valid, False otherwise
        """
        # Check for NaN or infinity
        if (np.isnan(screen_x) or np.isnan(screen_y) or 
            np.isnan(eye_x) or np.isnan(eye_y) or
            np.isinf(screen_x) or np.isinf(screen_y) or
            np.isinf(eye_x) or np.isinf(eye_y)):
            return False
        
        # Screen coordinates should be normalized (0-1)
        if not (0.0 <= screen_x <= 1.0):
            return False
        if not (0.0 <= screen_y <= 1.0):
            return False
        
        # Eye coordinates should be reasonable (positive, within camera frame)
        # Assuming camera resolution up to 1920x1080
        if eye_x < 0 or eye_y < 0:
            return False
        if eye_x > 2000 or eye_y > 2000:  # Reasonable upper bound
            return False
        
        return True
    
    def validate_calibration_data(self) -> Dict[str, Any]:
        """
        Validate the collected calibration data.
        
        Returns:
            Dictionary with validation results and recommendations
        """
        num_pairs = len(self.calibration_pairs)
        
        if num_pairs < 4:
            return {
                'valid': False,
                'reason': f"Insufficient calibration points: {num_pairs} (minimum 4 required)",
                'recommendation': "Collect more calibration points",
                'num_points': num_pairs,
                'coverage_score': 0.0
            }
        
        # Check distribution of points
        screen_x_values = [p.screen_x for p in self.calibration_pairs]
        screen_y_values = [p.screen_y for p in self.calibration_pairs]
        
        # Calculate coverage (how well points cover the screen)
        x_range = max(screen_x_values) - min(screen_x_values)
        y_range = max(screen_y_values) - min(screen_y_values)
        coverage_score = (x_range + y_range) / 2.0  # 0-1 scale
        
        # Check for duplicates or very close points
        unique_points = set((round(p.screen_x, 2), round(p.screen_y, 2)) 
                          for p in self.calibration_pairs)
        uniqueness_ratio = len(unique_points) / num_pairs
        
        recommendations = []
        if coverage_score < 0.5:
            recommendations.append("Calibration points are clustered. Try to cover more of the screen.")
        if uniqueness_ratio < 0.8:
            recommendations.append("Many duplicate points detected. Ensure points are distributed.")
        if num_pairs < 9:
            recommendations.append(f"Consider collecting {9 - num_pairs} more points for better accuracy.")
        
        return {
            'valid': True,
            'num_points': num_pairs,
            'coverage_score': coverage_score,
            'uniqueness_ratio': uniqueness_ratio,
            'x_range': x_range,
            'y_range': y_range,
            'recommendations': recommendations
        }
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from calibration pairs.
        
        Returns:
            X: Features array (n_samples, 2) - [eye_x, eye_y]
            y: Targets array (n_samples, 2) - [screen_x, screen_y]
        """
        if not self.calibration_pairs:
            raise ValueError("No calibration data available")
        
        X = []
        y = []
        
        for pair in self.calibration_pairs:
            features, target = pair.to_training_format()
            X.append(features)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def train(self) -> bool:
        """
        Train the regression model.
        
        Returns:
            True if training successful, False otherwise
        """
        # Validate data
        validation = self.validate_calibration_data()
        if not validation['valid']:
            print(f"[Regression] Training failed: {validation['reason']}")
            return False
        
        print(f"\n[Regression] Training model with {len(self.calibration_pairs)} points")
        print(f"[Regression] Data validation: {validation}")
        
        # Prepare training data
        X, y = self.prepare_training_data()
        
        # Split into X and Y components
        y_x = y[:, 0]  # Screen X coordinates
        y_y = y[:, 1]  # Screen Y coordinates
        
        try:
            if SKLEARN_AVAILABLE:
                # Create models based on selected type
                if self.model_type == "linear":
                    self.model_x = LinearRegression()
                    self.model_y = LinearRegression()

                elif self.model_type == "polynomial" and len(self.calibration_pairs) < 9:
                    print("[Regression] Polynomial regression needs at least 9 points")
                    return False
                    
                elif self.model_type == "polynomial":
                    # Create polynomial feature pipeline
                    self.model_x = make_pipeline(
                        PolynomialFeatures(degree=self.polynomial_degree),
                        LinearRegression()
                    )
                    self.model_y = make_pipeline(
                        PolynomialFeatures(degree=self.polynomial_degree),
                        LinearRegression()
                    )
                    
                elif self.model_type == "ridge":
                    self.model_x = Ridge(alpha=self.regularization)
                    self.model_y = Ridge(alpha=self.regularization)
                    
                else:
                    print(f"[Warning] Unknown model type '{self.model_type}', using linear")
                    self.model_x = LinearRegression()
                    self.model_y = LinearRegression()
                
                # Train X model
                print(f"[Regression] Training X-coordinate model...")
                self.model_x.fit(X, y_x)
                
                # Train Y model
                print(f"[Regression] Training Y-coordinate model...")
                self.model_y.fit(X, y_y)
                
            else:
                # Fallback: Simple linear regression using normal equations
                print("[Regression] Using simple linear regression (sklearn not available)")
                self.model_x = self._train_simple_linear(X, y_x)
                self.model_y = self._train_simple_linear(X, y_y)
            
            # Calculate training metrics
            self.is_trained = True
            self.training_metrics = self.calculate_metrics(X, y)

            print("\n" + "="*50)
            print("[Regression] TRAINING COMPLETE")
            print("="*50)
            print(f"Model type: {self.model_type}")
            print(f"Training points: {len(self.calibration_pairs)}")
            print(f"\nMetrics:")
            print(self.training_metrics)
            print("="*50)
            
            return True
            
        except Exception as e:
            print(f"[Regression] Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _train_simple_linear(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Simple linear regression using normal equations.
        Used as fallback when sklearn is not available.
        
        Returns:
            Dictionary with model coefficients
        """
        # Add bias term
        X_with_bias = np.hstack([X, np.ones((X.shape[0], 1))])
        
        # Solve normal equations: (X^T X)^-1 X^T y
        try:
            coeff = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            # If singular matrix, use pseudo-inverse
            coeff = np.linalg.pinv(X_with_bias) @ y
        
        # Flatten coefficients
        coeff = np.asarray(coeff).reshape(-1)

        return {
            'coeff': coeff,
            'intercept': coeff[-1],
            'weights': coeff[:-1]
        }
    
    def predict(self, eye_x: float, eye_y: float) -> Tuple[float, float]:
        """
        Predict screen coordinates from iris coordinates.
        
        Args:
            eye_x: Iris X coordinate
            eye_y: Iris Y coordinate
            
        Returns:
            (predicted_screen_x, predicted_screen_y) normalized (0-1)
            
        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() first.")
        
        # Validate input
        if np.isnan(eye_x) or np.isnan(eye_y) or np.isinf(eye_x) or np.isinf(eye_y):
            return 0.5, 0.5  # Return center as fallback
        
        # Prepare input
        X_input = np.array([[eye_x, eye_y]])
        
        try:
            if SKLEARN_AVAILABLE and not isinstance(self.model_x, dict):
                # Use sklearn model
                screen_x = float(self.model_x.predict(X_input)[0])
                screen_y = float(self.model_y.predict(X_input)[0])
            else:
                # Use simple linear model
                if isinstance(self.model_x, dict):
                    # Simple linear model
                    coeff_x = self.model_x['coeff']
                    coeff_y = self.model_y['coeff']
                    
                    # Add bias term
                    X_with_bias = np.array([[eye_x, eye_y, 1.0]])
                    
                    screen_x = float(X_with_bias @ coeff_x)
                    screen_y = float(X_with_bias @ coeff_y)
                else:
                    raise ValueError("Model not properly initialized")
            
            # Clamp to valid range
            screen_x = np.clip(screen_x, 0.0, 1.0)
            screen_y = np.clip(screen_y, 0.0, 1.0)
            
            return screen_x, screen_y
            
        except Exception as e:
            print(f"[Regression] Prediction error: {e}")
            return 0.5, 0.5  # Return center as fallback
    
    def calculate_metrics(self, X: np.ndarray = None, y: np.ndarray = None) -> RegressionMetrics:
        """
        Calculate regression metrics on training or test data.
        
        Args:
            X: Features array (if None, use training data)
            y: Targets array (if None, use training data)
            
        Returns:
            RegressionMetrics object with calculated metrics
        """
        if not self.is_trained:
            return RegressionMetrics()
        
        # Use provided data or training data
        if X is None or y is None:
            X, y = self.prepare_training_data()
        
        # Split targets
        y_x = y[:, 0]
        y_y = y[:, 1]
        
        # Get predictions
        predictions_x = []
        predictions_y = []
        
        for i in range(len(X)):
            X_input = X[i:i+1]  # shape (1, 2)

            if SKLEARN_AVAILABLE and not isinstance(self.model_x, dict):
                raw_x, raw_y = self._predict_raw_sklearn(X_input)
                pred_x = float(raw_x[0])
                pred_y = float(raw_y[0])
            else:
                if not isinstance(self.model_x, dict):
                    raise RuntimeError("Fallback model expected but not found")

                coeff_x = self.model_x['coeff']
                coeff_y = self.model_y['coeff']
                X_with_bias = np.array([X[i, 0], X[i, 1], 1.0])

                pred_x = float(X_with_bias @ coeff_x)
                pred_y = float(X_with_bias @ coeff_y)

            predictions_x.append(pred_x)
            predictions_y.append(pred_y)

        
        predictions_x = np.array(predictions_x)
        predictions_y = np.array(predictions_y)
        
        # Calculate metrics for X
        errors_x = predictions_x - y_x
        mse_x = np.mean(errors_x ** 2)
        mae_x = np.mean(np.abs(errors_x))
        max_error_x = np.max(np.abs(errors_x))
        
        # Calculate metrics for Y
        errors_y = predictions_y - y_y
        mse_y = np.mean(errors_y ** 2)
        mae_y = np.mean(np.abs(errors_y))
        max_error_y = np.max(np.abs(errors_y))
        
        # Calculate R² score
        var_x = np.var(y_x)
        var_y = np.var(y_y)
        r2_x = 1 - (mse_x / var_x) if var_x > 0 else 0
        r2_y = 1 - (mse_y / var_y) if var_y > 0 else 0
        
        return RegressionMetrics(
            mse_x=mse_x,
            mse_y=mse_y,
            mae_x=mae_x,
            mae_y=mae_y,
            r2_x=r2_x,
            r2_y=r2_y,
            max_error_x=max_error_x,
            max_error_y=max_error_y
        )
    
    def save_model(self, filename: str = "calibration_model.pkl"):
        """
        Save the trained model to a file.
        
        Args:
            filename: Path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        try:
            model_data = {
                'model_type': self.model_type,
                'polynomial_degree': self.polynomial_degree,
                'regularization': self.regularization,
                'calibration_pairs': self.calibration_pairs,
                'is_trained': self.is_trained,
                'training_metrics': self.training_metrics
            }
            
            model_data['model_x'] = self.model_x
            model_data['model_y'] = self.model_y

            
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"[Regression] Model saved to {filename}")
            return True
            
        except Exception as e:
            print(f"[Regression] Failed to save model: {e}")
            return False
    
    def load_model(self, filename: str = "calibration_model.pkl") -> bool:
        """
        Load a trained model from a file.
        
        Args:
            filename: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore model state
            self.model_type = model_data.get('model_type', 'linear')
            self.polynomial_degree = model_data.get('polynomial_degree', 2)
            self.regularization = model_data.get('regularization', 0.0)
            self.calibration_pairs = model_data.get('calibration_pairs', [])
            self.is_trained = model_data.get('is_trained', False)
            self.training_metrics = model_data.get('training_metrics')
            
            # Restore sklearn models if available
            if 'model_x' in model_data and 'model_y' in model_data:
                self.model_x = model_data['model_x']
                self.model_y = model_data['model_y']
            
            print(f"[Regression] Model loaded from {filename}")
            print(f"[Regression] {len(self.calibration_pairs)} calibration pairs loaded")

            if self.is_trained and (self.model_x is None or self.model_y is None):
                raise RuntimeError("Loaded model marked as trained but models are missing")
            return True
            
        except Exception as e:
            print(f"[Regression] Failed to load model: {e}")
            return False
        


    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'num_calibration_points': len(self.calibration_pairs),
            'training_metrics': self.training_metrics.__dict__ if self.training_metrics else None,
            'validation': self.validate_calibration_data()
        }
    
    def visualize_calibration_data(self, show_plot: bool = True):
        """
        Visualize calibration data and model fit.
        
        Args:
            show_plot: Whether to display the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.calibration_pairs:
                print("[Regression] No calibration data to visualize")
                return
            
            # Extract data
            screen_x = [p.screen_x for p in self.calibration_pairs]
            screen_y = [p.screen_y for p in self.calibration_pairs]
            eye_x = [p.eye_x for p in self.calibration_pairs]
            eye_y = [p.eye_y for p in self.calibration_pairs]
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Calibration Data Visualization', fontsize=16)
            
            # Plot 1: Screen positions (actual calibration points)
            ax1 = axes[0, 0]
            ax1.scatter(screen_x, screen_y, c='blue', s=100, alpha=0.7)
            ax1.set_xlabel('Screen X (normalized)')
            ax1.set_ylabel('Screen Y (normalized)')
            ax1.set_title('Calibration Points on Screen')
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Eye positions
            ax2 = axes[0, 1]
            ax2.scatter(eye_x, eye_y, c='red', s=50, alpha=0.7)
            ax2.set_xlabel('Eye X (pixels)')
            ax2.set_ylabel('Eye Y (pixels)')
            ax2.set_title('Corresponding Eye Positions')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Screen X vs Eye X
            ax3 = axes[1, 0]
            ax3.scatter(eye_x, screen_x, c='green', s=50, alpha=0.7)
            ax3.set_xlabel('Eye X (pixels)')
            ax3.set_ylabel('Screen X (normalized)')
            ax3.set_title('Eye X → Screen X Mapping')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Screen Y vs Eye Y
            ax4 = axes[1, 1]
            ax4.scatter(eye_y, screen_y, c='purple', s=50, alpha=0.7)
            ax4.set_xlabel('Eye Y (pixels)')
            ax4.set_ylabel('Screen Y (normalized)')
            ax4.set_title('Eye Y → Screen Y Mapping')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if show_plot:
                plt.show()
            else:
                # Save to file
                plt.savefig('calibration_visualization.png', dpi=150)
                print("[Regression] Visualization saved to calibration_visualization.png")
            
            plt.close()
            
        except ImportError:
            print("[Regression] matplotlib not available for visualization")
            print("Install with: pip install matplotlib")

    def _predict_raw_sklearn(self, X_input):
        """
        Predict raw screen coordinates from iris coordinates using trained regression model.
        
        Args:
            X_input: Iris coordinates (n_samples, 2) - [eye_x, eye_y]
        
        Returns:
            (screen_x_pred, screen_y_pred) normalized (0-1)
        
        Raises:
            RuntimeError: If model is a simple linear model
        """
        if isinstance(self.model_x, dict):
            raise RuntimeError("_predict_raw_sklearn() not supported for simple regression")
        return self.model_x.predict(X_input), self.model_y.predict(X_input)