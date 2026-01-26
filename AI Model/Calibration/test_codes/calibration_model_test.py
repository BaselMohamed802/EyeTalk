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

def test_basic_regression():
    """Test basic regression with simulated data."""
    print("\n" + "="*60)
    print("TESTING BASIC REGRESSION")
    print("="*60)
    
    # Create regression model
    regression = CalibrationRegression(model_type="linear")
    
    # Simulate calibration pairs
    # In reality, these would come from CalibrationSampler
    simulated_pairs = [
        ((0.1, 0.1), (100, 100)),   # Top-left
        ((0.5, 0.1), (320, 100)),   # Top-center
        ((0.9, 0.1), (540, 100)),   # Top-right
        ((0.1, 0.5), (100, 240)),   # Middle-left
        ((0.5, 0.5), (320, 240)),   # Center
        ((0.9, 0.5), (540, 240)),   # Middle-right
        ((0.1, 0.9), (100, 380)),   # Bottom-left
        ((0.5, 0.9), (320, 380)),   # Bottom-center
        ((0.9, 0.9), (540, 380)),   # Bottom-right
    ]
    
    # Add pairs
    regression.add_calibration_pairs(simulated_pairs)
    
    # Validate data
    validation = regression.validate_calibration_data()
    print(f"\nData validation:")
    for key, value in validation.items():
        print(f"  {key}: {value}")
    
    # Train model
    success = regression.train()
    
    if success:
        # Test predictions
        print("\n" + "="*30)
        print("TESTING PREDICTIONS")
        print("="*30)
        
        test_points = [
            (100, 100),    # Should predict ~(0.1, 0.1)
            (320, 240),    # Should predict ~(0.5, 0.5)
            (540, 380),    # Should predict ~(0.9, 0.9)
            (220, 170),    # Intermediate point
            (400, 300),    # Another intermediate
        ]
        
        for eye_x, eye_y in test_points:
            pred_x, pred_y = regression.predict(eye_x, eye_y)
            print(f"Eye({eye_x}, {eye_y}) → Screen({pred_x:.3f}, {pred_y:.3f})")
        
        # Get model info
        print("\n" + "="*30)
        print("MODEL INFORMATION")
        print("="*30)
        info = regression.get_model_info()
        for key, value in info.items():
            if key != 'training_metrics':
                print(f"{key}: {value}")
        
        # Test saving/loading
        print("\n" + "="*30)
        print("TESTING SAVE/LOAD")
        print("="*30)
        
        # Save model
        regression.save_model("test_model.pkl")
        
        # Create new model and load
        new_regression = CalibrationRegression()
        new_regression.load_model("test_model.pkl")
        
        # Test prediction with loaded model
        pred_x, pred_y = new_regression.predict(320, 240)
        print(f"Loaded model prediction for (320, 240): ({pred_x:.3f}, {pred_y:.3f})")
    
    print("\n" + "="*60)
    print("BASIC REGRESSION TEST COMPLETE")
    print("="*60)


def test_polynomial_regression():
    """Test polynomial regression."""
    print("\n" + "="*60)
    print("TESTING POLYNOMIAL REGRESSION")
    print("="*60)
    
    if not SKLEARN_AVAILABLE:
        print("sklearn not available, skipping polynomial test")
        return
    
    # Create polynomial regression model
    regression = CalibrationRegression(
        model_type="polynomial",
        polynomial_degree=2
    )
    
    # Simulate non-linear relationship
    simulated_pairs = []
    for i in range(9):
        screen_x = 0.1 + 0.8 * (i % 3) / 2
        screen_y = 0.1 + 0.8 * (i // 3) / 2
        
        # Simulate non-linear relationship
        eye_x = 100 + 400 * screen_x + 50 * screen_x ** 2
        eye_y = 100 + 300 * screen_y + 40 * screen_y ** 2
        
        # Add some noise
        eye_x += np.random.normal(0, 5)
        eye_y += np.random.normal(0, 5)
        
        simulated_pairs.append(((screen_x, screen_y), (eye_x, eye_y)))
    
    # Add pairs
    regression.add_calibration_pairs(simulated_pairs)
    
    # Train
    success = regression.train()
    
    if success:
        # Test predictions
        print("\nPolynomial model predictions:")
        for screen_x in [0.2, 0.5, 0.8]:
            for screen_y in [0.2, 0.5, 0.8]:
                # Calculate corresponding eye position (inverse of simulation)
                eye_x = 100 + 400 * screen_x + 50 * screen_x ** 2
                eye_y = 100 + 300 * screen_y + 40 * screen_y ** 2
                
                pred_x, pred_y = regression.predict(eye_x, eye_y)
                error_x = abs(pred_x - screen_x)
                error_y = abs(pred_y - screen_y)
                
                print(f"True: ({screen_x:.2f}, {screen_y:.2f}) → "
                      f"Pred: ({pred_x:.3f}, {pred_y:.3f}) | "
                      f"Error: ({error_x:.3f}, {error_y:.3f})")
    
    print("\n" + "="*60)
    print("POLYNOMIAL REGRESSION TEST COMPLETE")
    print("="*60)


def test_error_handling():
    """Test error cases and edge cases."""
    print("\n" + "="*60)
    print("TESTING ERROR HANDLING")
    print("="*60)
    
    regression = CalibrationRegression()
    
    # Test 1: Empty model prediction
    try:
        regression.predict(100, 100)
    except ValueError as e:
        print(f"✓ Expected error for untrained model: {e}")
    
    # Test 2: Invalid calibration pairs
    print("\nTesting invalid pairs:")
    invalid_pairs = [
        ((1.5, 0.5), (100, 100)),   # Screen X > 1
        ((-0.1, 0.5), (100, 100)),  # Screen X < 0
        ((0.5, 0.5), (-10, 100)),   # Eye X < 0
        ((0.5, 0.5), (100, 3000)),  # Eye Y too large
        ((float('nan'), 0.5), (100, 100)),  # NaN in screen
        ((0.5, 0.5), (float('inf'), 100)),  # Inf in eye
    ]
    
    for pair in invalid_pairs:
        added = regression.add_calibration_pair(pair[0], pair[1])
        print(f"  Pair {pair}: {'REJECTED' if not added else 'ACCEPTED'}")
    
    # Test 3: Insufficient points
    print("\nTesting with insufficient points:")
    regression_small = CalibrationRegression()
    regression_small.add_calibration_pair((0.1, 0.1), (100, 100))
    regression_small.add_calibration_pair((0.9, 0.1), (500, 100))
    regression_small.add_calibration_pair((0.1, 0.9), (100, 400))
    
    success = regression_small.train()
    print(f"  Training with 3 points: {'FAILED' if not success else 'SUCCESS'}")
    
    print("\n" + "="*60)
    print("ERROR HANDLING TEST COMPLETE")
    print("="*60)


def main():
    """Main test menu."""
    print("\n" + "="*60)
    print("CALIBRATION REGRESSION - STEP 4")
    print("="*60)
    
    print("\nResponsibilities:")
    print("1. Accept calibration pairs from CalibrationSampler")
    print("2. Train linear/polynomial regression model")
    print("3. Validate input data quality")
    print("4. Predict screen coordinates from iris coordinates")
    print("5. Provide error metrics")
    
    print("\nInput format:")
    print("X = [[eye_x, eye_y], ...]  # Features")
    print("y = [[screen_x, screen_y], ...]  # Targets")
    
    print("\nOutput (prediction):")
    print("Given: eye_x, eye_y")
    print("Returns: screen_x_pred, screen_y_pred")
    
    print("\nNO cursor movement yet!")
    
    while True:
        print("\n" + "="*60)
        print("\nChoose test mode:")
        print("1. Test basic linear regression")
        print("2. Test polynomial regression (requires sklearn)")
        print("3. Test error handling")
        print("4. Visualize calibration data (requires matplotlib)")
        print("5. Exit")
        
        try:
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == "1":
                test_basic_regression()
            elif choice == "2":
                test_polynomial_regression()
            elif choice == "3":
                test_error_handling()
            elif choice == "4":
                try:
                    regression = CalibrationRegression()
                    # Add some test data
                    test_pairs = [
                        ((0.1, 0.1), (100, 100)),
                        ((0.5, 0.5), (320, 240)),
                        ((0.9, 0.9), (540, 380)),
                    ]
                    regression.add_calibration_pairs(test_pairs)
                    regression.visualize_calibration_data(show_plot=True)
                except Exception as e:
                    print(f"Visualization failed: {e}")
            elif choice == "5":
                print("\nExiting...")
                break
            else:
                print("Invalid choice. Please enter 1-5.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break


if __name__ == "__main__":
    import numpy as np  # For test simulations
    main()