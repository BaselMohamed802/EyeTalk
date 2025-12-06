# Filters Package Documentation

This package contains smoothing filters for stabilizing iris coordinate data. It includes a unified filter manager and two filter implementations: Exponential Moving Average (EMA) and Kalman Filter.

## 📁 Package Structure

```
Filters/
├── filterManager.py    # Main filter orchestration
├── EMAFilter.py        # Exponential Moving Average filter
└── kalmanFilter.py     # Kalman Filter implementation
```

---

# Module: filterManager.py

**Creator/Author:** Basel Mohamed Mostafa Sayed  
**Date:** 12/6/2025

## Description

Filter manager for applying different smoothing filters to iris coordinates. Provides a unified interface for switching between Kalman and EMA filters.

## Class: FilterManager

### Constructor

```python
FilterManager(filter_type="ema", **kwargs)
```

**Parameters:**
- `filter_type` (str): Type of filter to use (`"kalman"` or `"ema"`)
- `**kwargs`: Additional filter parameters:
  - For EMA: `alpha` (default 0.35)
  - For Kalman: `dt`, `process_noise`, `measurement_noise`

**Example:**
```python
# EMA filter with custom alpha
ema_manager = FilterManager("ema", alpha=0.4)

# Kalman filter with custom parameters
kalman_manager = FilterManager("kalman", dt=0.5, process_noise=1e-4)
```

---

### Method: `apply_filter(x, y)`

Apply selected filter to input coordinates.

```python
filtered_x, filtered_y = manager.apply_filter(x, y)
```

**Parameters:**
- `x` (float): x-coordinate
- `y` (float): y-coordinate

**Returns:**
- `tuple`: Filtered (x, y) coordinates

**Example:**
```python
# Apply filter to raw coordinates
raw_x, raw_y = 320, 240
smooth_x, smooth_y = manager.apply_filter(raw_x, raw_y)
```

---

### Method: `reset()`

Reset the current filter to its initial state.

```python
manager.reset()
```

**Use Case:**
- Switching between different users/cameras
- Recalibrating the filter
- Resetting after tracking loss

---

### Method: `switch_filter(new_filter_type, **kwargs)`

Switch to a different filter type.

```python
manager.switch_filter("kalman", dt=1.0, process_noise=1e-3)
```

**Parameters:**
- `new_filter_type` (str): New filter type (`"kalman"` or `"ema"`)
- `**kwargs`: New filter parameters

**Example:**
```python
# Switch from EMA to Kalman filter
manager.switch_filter("kalman", dt=0.5, measurement_noise=0.1)
```

---

# Module: EMAFilter.py

**Creator/Author:** Basel Mohamed Mostafa Sayed  
**Date:** 12/6/2025

## Description

Implementation of an Exponential Moving Average (EMA) filter for smoothing iris coordinate movement. Simple, computationally efficient, and effective for basic smoothing needs.

## Class: EMAFilter

### Constructor

```python
EMAFilter(alpha=0.35)
```

**Parameters:**
- `alpha` (float, optional): Smoothing factor (0 ≤ alpha ≤ 1). Defaults to 0.35.
  - Lower alpha = more smoothing, slower response
  - Higher alpha = less smoothing, faster response

**Throws:**
- `ValueError`: If alpha is not between 0 and 1

**Example:**
```python
# Create EMA filters with different smoothing levels
strong_smoothing = EMAFilter(alpha=0.1)    # Very smooth, slow response
medium_smoothing = EMAFilter(alpha=0.35)   # Balanced (default)
light_smoothing = EMAFilter(alpha=0.7)     # Less smooth, fast response
```

---

### Method: `apply_ema(x, y)`

Apply EMA filter to input coordinates.

```python
filtered_x, filtered_y = ema_filter.apply_ema(x, y)
```

**Formula:**
```
filtered_value = alpha * current_value + (1 - alpha) * previous_value
```

**Parameters:**
- `x` (float): Current x-coordinate
- `y` (float): Current y-coordinate

**Returns:**
- `tuple`: Filtered (x, y) coordinates

**Behavior:**
- On first call: Initializes filter with input values
- Subsequent calls: Applies EMA smoothing

---

### Method: `reset()`

Reset the EMA filter to its initial state.

```python
ema_filter.reset()
```

**Resets:**
- `initialized` flag to `False`
- Previous x and y values to 0

**Use Case:**
- When switching users
- After tracking interruption
- Before new calibration session

---

# Module: kalmanFilter.py

**Creator/Author:** Basel Mohamed Mostafa Sayed  
**Date:** 12/6/2025

## Description

Implementation of a Kalman Filter for 2D position and velocity tracking. Uses a constant velocity model to predict and update iris coordinates, providing optimal estimation under Gaussian noise assumptions.

## Class: KalmanFilter

### Constructor

```python
KalmanFilter(dt=1.0, process_noise=1e-3, measurement_noise=1e-1)
```

**Parameters:**
- `dt` (float): Time step between measurements (default 1.0 for discrete frames)
- `process_noise` (float): Process noise covariance (uncertainty in system dynamics)
- `measurement_noise` (float): Measurement noise covariance (sensor noise)

**State Representation:**
- State vector: `[x, y, vx, vy]` (position and velocity)
- Only `[x, y]` are directly measured

---

### Method: `apply_kalman(x, y)`

Apply Kalman filter to new measurement.

```python
filtered_x, filtered_y = kalman_filter.apply_kalman(x, y)
```

**Process:**
1. **Prediction Step:** Predict next state based on current state and dynamics
2. **Update Step:** Incorporate new measurement to refine prediction

**Parameters:**
- `x` (float): Measured x-coordinate
- `y` (float): Measured y-coordinate

**Returns:**
- `tuple`: Filtered (x, y) coordinates

**Initialization:**
- First call: Initializes filter with measurement and high uncertainty
- Subsequent calls: Applies full Kalman filter cycle

---

### Method: `reset()`

Reset Kalman filter to initial state.

```python
kalman_filter.reset()
```

**Resets:**
- State vector to `[0, 0, 0, 0]`
- Covariance matrix to identity
- `initialized` flag to `False`

---

# 📊 Filter Comparison

| Feature | EMA Filter | Kalman Filter |
|---------|------------|---------------|
| **Complexity** | Simple | Complex |
| **Computational Cost** | Low | Medium-High |
| **Memory Usage** | Minimal (2 values) | Moderate (state matrix) |
| **Model** | Statistical smoothing | Dynamic system model |
| **Parameters** | 1 (alpha) | 3+ (dt, noises) |
| **Best For** | Simple smoothing, low latency | Optimal estimation, noisy environments |
| **Response Speed** | Configurable via alpha | Adaptive (gain-based) |

---

# 🚀 Example Usage

## Basic Usage with FilterManager

```python
from filterManager import FilterManager

# Create filter manager with EMA
manager = FilterManager("ema", alpha=0.3)

# Simulate coordinate stream
coordinates = [(100, 200), (102, 198), (105, 196), (108, 195)]

for x, y in coordinates:
    smooth_x, smooth_y = manager.apply_filter(x, y)
    print(f"Raw: ({x}, {y}) → Smooth: ({smooth_x:.1f}, {smooth_y:.1f})")
```

## Switching Filters Dynamically

```python
from filterManager import FilterManager

manager = FilterManager("ema", alpha=0.35)

# Use EMA initially
for i in range(50):
    smooth_x, smooth_y = manager.apply_filter(raw_x, raw_y)

# Switch to Kalman for better performance
manager.switch_filter("kalman", 
                      dt=0.033,  # 30 FPS
                      process_noise=1e-4,
                      measurement_noise=1e-2)

# Continue with Kalman
for i in range(100):
    smooth_x, smooth_y = manager.apply_filter(raw_x, raw_y)
```

## Direct Filter Usage

```python
from EMAFilter import EMAFilter
from kalmanFilter import KalmanFilter

# EMA Filter
ema = EMAFilter(alpha=0.25)
ema_x, ema_y = ema.apply_ema(x, y)

# Kalman Filter
kalman = KalmanFilter(dt=0.5, measurement_noise=0.05)
kalman_x, kalman_y = kalman.apply_kalman(x, y)
```

---

# 🎯 Use Cases

## 1. **Real-time Iris Tracking**
```python
# In your main tracking loop
while tracking:
    # Get raw iris centers from face_utils
    centers = face_mesh.get_iris_centers(frame)
    
    if centers['left']:
        raw_x, raw_y = centers['left']
        smooth_x, smooth_y = filter_manager.apply_filter(raw_x, raw_y)
        # Use smoothed coordinates for cursor control
```

## 2. **Adaptive Filter Selection**
```python
def select_filter_based_on_fps(fps):
    if fps > 30:
        # High FPS - use light smoothing
        return FilterManager("ema", alpha=0.5)
    else:
        # Low FPS - use Kalman for better prediction
        return FilterManager("kalman", dt=1/fps, process_noise=1e-3)
```

## 3. **Reset on User Change**
```python
def on_user_change():
    filter_manager.reset()  # Reset filter state for new user
    # Recalibrate or continue tracking
```

---

# 🔧 Parameter Tuning Guide

## EMA Filter (`alpha`)
- **0.1-0.2**: Very smooth, good for reducing jitter but slow response
- **0.3-0.4**: Balanced (default range)
- **0.5-0.7**: Responsive, preserves detail but less smoothing
- **>0.7**: Minimal smoothing, almost raw data

## Kalman Filter Parameters
- **`dt`**: Should match frame interval (1/FPS)
- **`process_noise`**: 
  - Lower = trust model more (smoother)
  - Higher = adapt faster to changes
- **`measurement_noise`**:
  - Lower = trust measurements more
  - Higher = trust model predictions more

## Recommended Starting Points:
```python
# For 30 FPS webcam
kalman = KalmanFilter(dt=0.033, process_noise=1e-4, measurement_noise=1e-2)

# For 60 FPS high-speed camera
kalman = KalmanFilter(dt=0.0167, process_noise=5e-5, measurement_noise=5e-3)

# EMA for general use
ema = EMAFilter(alpha=0.35)
```

---

# 📝 Notes

1. **Initialization**: Both filters initialize on first measurement
2. **Thread Safety**: Not thread-safe by default
3. **Performance**: EMA is ~10x faster than Kalman for simple cases
4. **Memory**: Both filters maintain minimal state between frames
5. **Integration**: Designed to work with `face_utils.py` iris coordinates

The filters are optimized for real-time eye tracking applications where smooth cursor movement is essential.