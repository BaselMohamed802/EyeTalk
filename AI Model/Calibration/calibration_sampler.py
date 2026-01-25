"""
Filename: calibration_sampler.py
Creator/Author: Basel Mohamed Mostafa Sayed
Date: 1/25/2026

Description:
    Calibration data sampler.
    
    Responsibilities:
    1. Wait for sampling_allowed signal from Timing Controller
    2. Collect iris center samples for fixed duration/count
    3. Average samples to get stable eye position
    4. Emit (screen_x, screen_y) and (eye_x_avg, eye_y_avg) pairs
    5. Signal Timing Controller to advance to next point
    
    Inputs:
    - sampling_allowed (bool) from CalibrationTimingController
    - Continuous iris center coordinates from MediaPipe pipeline
    
    Output (per calibration point):
    - (screen_x_norm, screen_y_norm)  <-- known target position
    - (eye_x_avg, eye_y_avg)          <-- computed average eye position
"""

# Import necessary libraries.
import time
import numpy as np
from typing import List, Tuple, Optional
from collections import deque


class CalibrationSample:
    """Represents a single sample with timestamp and iris data."""
    def __init__(self, timestamp: float, left_iris: Tuple[float, float], 
                 right_iris: Tuple[float, float]):
        self.timestamp = timestamp
        self.left_x, self.left_y = left_iris
        self.right_x, self.right_y = right_iris
    
    def get_left_coords(self) -> Tuple[float, float]:
        return (self.left_x, self.left_y)
    
    def get_right_coords(self) -> Tuple[float, float]:
        return (self.right_x, self.right_y)
    
    def get_average_coords(self) -> Tuple[float, float]:
        """Average of both eyes' coordinates."""
        avg_x = (self.left_x + self.right_x) / 2.0
        avg_y = (self.left_y + self.right_y) / 2.0
        return (avg_x, avg_y)


class CalibrationDataPoint:
    """Represents complete calibration data for one screen position."""
    def __init__(self, screen_x_norm: float, screen_y_norm: float, 
                 point_index: int):
        self.screen_x_norm = screen_x_norm
        self.screen_y_norm = screen_y_norm
        self.point_index = point_index
        self.samples: List[CalibrationSample] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
    def add_sample(self, sample: CalibrationSample):
        """Add a sample to this calibration point."""
        if not self.samples:
            self.start_time = sample.timestamp
        self.samples.append(sample)
        self.end_time = sample.timestamp
    
    def get_sample_count(self) -> int:
        return len(self.samples)
    
    def get_duration(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    def calculate_averages(self) -> dict:
        """
        Calculate average eye positions from collected samples.
        
        Returns:
            dict with averaged coordinates for left, right, and combined eyes
        """
        if not self.samples:
            return {}
        
        left_x_values = [s.left_x for s in self.samples]
        left_y_values = [s.left_y for s in self.samples]
        right_x_values = [s.right_x for s in self.samples]
        right_y_values = [s.right_y for s in self.samples]
        
        return {
            'left_eye': (np.mean(left_x_values), np.mean(left_y_values)),
            'right_eye': (np.mean(right_x_values), np.mean(right_y_values)),
            'average_eye': (
                (np.mean(left_x_values) + np.mean(right_x_values)) / 2,
                (np.mean(left_y_values) + np.mean(right_y_values)) / 2
            ),
            'sample_count': len(self.samples),
            'duration_seconds': self.get_duration()
        }
    
    def get_calibration_pair(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Get the calibration pair: (screen_position, averaged_eye_position)
        
        Returns:
            Tuple of ((screen_x, screen_y), (eye_x_avg, eye_y_avg))
        """
        averages = self.calculate_averages()
        if 'average_eye' in averages:
            screen_pos = (self.screen_x_norm, self.screen_y_norm)
            eye_pos = averages['average_eye']
            return (screen_pos, eye_pos)
        return None


class CalibrationSampler:
    """
    Collects and averages iris samples during calibration.
    
    Works with CalibrationTimingController to:
    1. Wait for sampling_allowed signal
    2. Collect samples for specified duration/count
    3. Compute averages
    4. Emit calibration data pairs
    """
    
    def __init__(self, 
                 sampling_duration: float = 2.0,
                 min_samples: int = 10,
                 max_samples: int = 100):
        """
        Initialize calibration sampler.
        
        Args:
            sampling_duration: How long to sample at each point (seconds)
            min_samples: Minimum number of samples required
            max_samples: Maximum number of samples to collect
        """
        self.sampling_duration = sampling_duration
        self.min_samples = min_samples
        self.max_samples = max_samples
        
        # State
        self.is_sampling = False
        self.sampling_start_time = 0
        self.current_data_point: Optional[CalibrationDataPoint] = None
        self.calibration_data: List[CalibrationDataPoint] = []
        
        # Statistics
        self.total_samples_collected = 0
        self.samples_rejected = 0
        
        # Callbacks
        self.on_sampling_started = None
        self.on_sample_collected = None
        self.on_sampling_completed = None
        self.on_calibration_pair_ready = None
        self.on_advance_requested = None
        
        # Filter for smoothing (optional)
        self.use_filtering = True
        self.filter_window_size = 5
        self.left_eye_buffer = deque(maxlen=self.filter_window_size)
        self.right_eye_buffer = deque(maxlen=self.filter_window_size)
        
        print(f"Calibration sampler initialized")
        print(f"Sampling duration: {sampling_duration}s")
        print(f"Sample range: {min_samples}-{max_samples}")
    
    def start_sampling_for_point(self, point_index: int, 
                                screen_x_norm: float, 
                                screen_y_norm: float):
        """
        Start sampling for a specific calibration point.
        
        Args:
            point_index: Index of current calibration point
            screen_x_norm: Normalized screen X coordinate (0-1)
            screen_y_norm: Normalized screen Y coordinate (0-1)
        """
        if self.is_sampling:
            print("[Warning] Already sampling, ignoring start request")
            return
        
        # Create new data point
        self.current_data_point = CalibrationDataPoint(
            screen_x_norm, screen_y_norm, point_index
        )
        
        # Reset state
        self.is_sampling = True
        self.sampling_start_time = time.time()
        self.left_eye_buffer.clear()
        self.right_eye_buffer.clear()
        
        # Notify
        print(f"\n[Sampler] Started sampling for point {point_index + 1}")
        print(f"[Sampler] Target: ({screen_x_norm:.3f}, {screen_y_norm:.3f})")
        print(f"[Sampler] Will sample for {self.sampling_duration}s or {self.min_samples} samples")
        
        if self.on_sampling_started:
            self.on_sampling_started(point_index, screen_x_norm, screen_y_norm)
    
    def add_iris_sample(self, left_iris: Tuple[float, float], 
                       right_iris: Tuple[float, float]) -> bool:
        """
        Add an iris sample if currently sampling.
        
        Args:
            left_iris: (x, y) coordinates of left iris
            right_iris: (x, y) coordinates of right iris
            
        Returns:
            True if sample was added, False otherwise
        """
        if not self.is_sampling or not self.current_data_point:
            return False

        # Check if sample is valid        
        if not self._valid_coords(left_iris) or not self._valid_coords(right_iris):
            self.samples_rejected += 1
            return False

        # Apply simple moving average filter if enabled
        if self.use_filtering:
            filtered_left = self._apply_filter(left_iris, self.left_eye_buffer)
            filtered_right = self._apply_filter(right_iris, self.right_eye_buffer)
        else:
            filtered_left, filtered_right = left_iris, right_iris
        
        # Create sample (No Filter enabled)
        sample = CalibrationSample(
            timestamp=time.time(),
            left_iris=left_iris,
            right_iris=right_iris
        )
        
        # Add to current data point
        self.current_data_point.add_sample(sample)
        self.total_samples_collected += 1
        
        # Check if we should stop sampling
        self._check_sampling_completion()
        
        # Notify about sample
        if self.on_sample_collected:
            self.on_sample_collected(
                self.current_data_point.point_index,
                self.current_data_point.get_sample_count(),
                filtered_left, filtered_right
            )
        return True
    
    def _apply_filter(self, iris_coords: Tuple[float, float], 
                     buffer: deque) -> Tuple[float, float]:
        """Apply simple moving average filter."""
        x, y = iris_coords
        
        # Add to buffer
        buffer.append((x, y))
        
        # Calculate average
        if buffer:
            avg_x = sum(c[0] for c in buffer) / len(buffer)
            avg_y = sum(c[1] for c in buffer) / len(buffer)
            return (avg_x, avg_y)
        
        return iris_coords
    
    def _check_sampling_completion(self):
        """Check if sampling should stop based on duration or count."""
        if not self.is_sampling or not self.current_data_point:
            return
        
        current_time = time.time()
        elapsed = current_time - self.sampling_start_time
        sample_count = self.current_data_point.get_sample_count()
        
        # Stop if we've collected enough samples OR enough time has passed
        stop_condition = (
            (sample_count >= self.min_samples and elapsed >= self.sampling_duration) or
            (sample_count >= self.max_samples) or
            (elapsed >= self.sampling_duration)
        )
        
        if stop_condition:
            self._stop_sampling()
    
    def _stop_sampling(self):
        """Stop sampling at current point and process data."""
        if not self.is_sampling or not self.current_data_point:
            return
        
        self.is_sampling = False
        
        # Calculate averages
        point = self.current_data_point
        averages = point.calculate_averages()
        
        print(f"\n[Sampler] Sampling complete for point {point.point_index + 1}")
        print(f"[Sampler] Collected {point.get_sample_count()} samples")
        print(f"[Sampler] Duration: {point.get_duration():.2f}s")
        
        if averages:
            print(f"[Sampler] Left eye avg: ({averages['left_eye'][0]:.1f}, {averages['left_eye'][1]:.1f})")
            print(f"[Sampler] Right eye avg: ({averages['right_eye'][0]:.1f}, {averages['right_eye'][1]:.1f})")
            print(f"[Sampler] Combined avg: ({averages['average_eye'][0]:.1f}, {averages['average_eye'][1]:.1f})")
        
        # Store calibration data
        self.calibration_data.append(point)
        
        # Get calibration pair
        calibration_pair = point.get_calibration_pair()
        
        # Notify that sampling is complete
        if self.on_sampling_completed:
            self.on_sampling_completed(
                point.point_index,
                point.get_sample_count(),
                averages.get('average_eye', (0, 0))
            )
        
        # Emit calibration pair
        if calibration_pair and self.on_calibration_pair_ready:
            screen_pos, eye_pos = calibration_pair
            self.on_calibration_pair_ready(
                point.point_index,
                screen_pos,
                eye_pos
            )
        
        # Request advance to next point
        if self.on_advance_requested:
            print("[Sampler] Requesting advance to next point...")
            self.on_advance_requested()
    
    def update(self):
        """Update sampler state (check for timeout)."""
        if self.is_sampling:
            self._check_sampling_completion()
    
    def get_current_stats(self) -> dict:
        """Get current sampling statistics."""
        if not self.current_data_point:
            return {}
        
        elapsed = time.time() - self.sampling_start_time if self.is_sampling else 0
        
        return {
            'is_sampling': self.is_sampling,
            'point_index': self.current_data_point.point_index if self.current_data_point else -1,
            'samples_collected': self.current_data_point.get_sample_count() if self.current_data_point else 0,
            'elapsed_time': elapsed,
            'time_remaining': max(0, self.sampling_duration - elapsed) if self.is_sampling else 0,
            'sampling_progress': min(1.0, elapsed / self.sampling_duration) if self.is_sampling else 0,
        }
    
    def get_calibration_data(self) -> List[CalibrationDataPoint]:
        """Get all collected calibration data."""
        return self.calibration_data
    
    def get_calibration_pairs(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Get all calibration pairs (screen_pos, eye_pos)."""
        pairs = []
        for data_point in self.calibration_data:
            pair = data_point.get_calibration_pair()
            if pair:
                pairs.append(pair)
        return pairs
    
    def reset(self):
        """Reset sampler for new calibration session."""
        self.is_sampling = False
        self.current_data_point = None
        self.calibration_data = []
        self.total_samples_collected = 0
        self.samples_rejected = 0
        print("[Sampler] Reset for new calibration session")

    def _valid_coords(self, coords: Tuple[float, float]) -> bool:
        """
        Check if coordinates are valid (not NaN or inf).
        
        Args:
            coords: Tuple of (x, y) coordinates
        Returns:
            True if coordinates are valid, False otherwise
        """
        x, y = coords
        return np.isfinite(x) and np.isfinite(y)

# ============================================================================
# INTEGRATION WITH TIMING CONTROLLER
# ============================================================================

class CalibrationCoordinator:
    """
    Coordinates between Timing Controller and Sampler.
    This is the glue that connects Step 2 and Step 3.
    """
    
    def __init__(self, timing_controller, sampler):
        """
        Initialize coordinator.
        
        Args:
            timing_controller: CalibrationTimingController instance
            sampler: CalibrationSampler instance
        """
        self.timing_controller = timing_controller
        self.sampler = sampler
        
        # Connect signals
        self._connect_signals()
        
        print("\n" + "="*50)
        print("CALIBRATION COORDINATOR INITIALIZED")
        print("="*50)
        print("Timing Controller ↔ Coordinator ↔ Sampler")
    
    def _connect_signals(self):
        """Connect callbacks between timing controller and sampler."""
        
        # When timing controller starts a point
        def on_point_started(point_index, x_norm, y_norm):
            print(f"\n[Coordinator] Point {point_index + 1} started")
            print(f"[Coordinator] Waiting for sampling signal...")
        
        # When timing controller allows sampling
        def on_sampling_start(point_index):
            print(f"\n[Coordinator] Sampling allowed for point {point_index + 1}")
            
            # Get point info
            point_info = self.timing_controller.get_current_point_info()
            if point_info:
                # Start sampling for this point
                self.sampler.start_sampling_for_point(
                    point_index,
                    point_info['x_norm'],
                    point_info['y_norm']
                )
        
        # When sampler completes sampling
        def on_sampling_completed(point_index, sample_count, avg_eye_pos):
            print(f"\n[Coordinator] Sampler completed point {point_index + 1}")
            print(f"[Coordinator] Collected {sample_count} samples")
            print(f"[Coordinator] Average eye: ({avg_eye_pos[0]:.1f}, {avg_eye_pos[1]:.1f})")
        
        # When sampler requests advance
        def on_advance_requested():
            print(f"\n[Coordinator] Sampler requests advance to next point")
            # Tell timing controller to advance
            self.timing_controller.advance_to_next_point()
        
        # Connect timing controller callbacks
        self.timing_controller.on_point_started = on_point_started
        self.timing_controller.on_sampling_start = on_sampling_start
        
        # Connect sampler callbacks
        self.sampler.on_sampling_completed = on_sampling_completed
        self.sampler.on_advance_requested = on_advance_requested
    
    def update(self):
        """Update both timing controller and sampler."""
        self.timing_controller.update()
        self.sampler.update()
    
    def add_iris_sample(self, left_iris, right_iris):
        """
        Add iris sample to sampler.
        
        Args:
            left_iris: (x, y) coordinates of left iris
            right_iris: (x, y) coordinates of right iris
        """
        return self.sampler.add_iris_sample(left_iris, right_iris)
    
    def start_calibration(self):
        """Start the calibration process."""
        print("\n[Coordinator] Starting calibration...")
        self.timing_controller.start_calibration()
    
    def get_calibration_data(self):
        """Get collected calibration data."""
        return self.sampler.get_calibration_data()