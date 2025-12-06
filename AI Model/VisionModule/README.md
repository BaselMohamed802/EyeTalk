```markdown
# Module 1 — camera.py

This module abstracts OpenCV camera access into a simple, safe, reusable class.

## Class: IrisCamera

A wrapper around OpenCV's VideoCapture that provides:

1. Automatic resolution setup
2. Safe resource cleanup
3. Frame capture
4. FPS measurement

### Constructor

```python
camera = IrisCamera(cam_id, cam_width=640, cam_height=480)
```

**Parameters:**

1. `cam_id` — Camera index (usually 0)
2. `cam_width` — Desired capture width
3. `cam_height` — Desired capture height

**What it does:**

- Opens the camera
- Sets its resolution
- Stores camera state internally

---

#### Method: `get_frame()`

```python
frame = camera.get_frame()
```

Returns the latest frame from the webcam.

**Returns:**

1. A `numpy.ndarray` (the image)
2. `None` if frame cannot be read

**Notes:**

- Automatically used inside loops.
- No need to manually check the capture state.

---

#### Method: `get_frame_rate()`

```python
fps = camera.get_frame_rate()
```

Returns an estimated FPS based on time between consecutive calls.

**Useful for:**

- Debugging
- Performance tuning

---

#### Method: `get_resolution()`

```python
w, h = camera.get_resolution()
```

Returns the camera resolution you configured.

---

#### Method: `is_opened()`

```python
camera.is_opened()
```

Checks if the webcam was successfully opened.

---

#### Method: `stop_recording()`

Releases the camera from memory.

Automatically called when using:

```python
with IrisCamera(0) as camera:
    ...
```

Because of the class’s `__enter__` and `__exit__` context manager support.

---

# 👁️ Module 2 — face_utils.py (FaceMeshDetector)

This module handles iris tracking using MediaPipe Face Mesh.

It extracts:

- 468 face landmarks
- 4 iris landmarks per eye
- (x, y) center of each iris
- Optional drawing utilities for debugging

## Class: FaceMeshDetector

**Purpose:**

A high-level wrapper around MediaPipe FaceMesh for:

- Face mesh detection
- Iris landmark extraction
- Iris center calculation
- Debug drawing

It greatly simplifies interacting with MediaPipe.

### Constructor

```python
detector = FaceMeshDetector(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

**Key parameter:**

- `refine_landmarks=True`
  - ✔ Required for iris detection
  - ✔ Enables the special iris landmarks

---

#### Method: `draw_face_mesh()`

```python
img, landmarks = detector.draw_face_mesh(
    img,
    draw_face=True,
    draw_iris=False,
    draw_tesselation=False,
    draw_contours=False
)
```

Draws one or more visual debugging overlays:

- Face mesh
- Facial contours
- Iris outline (optional)

**Returns:**

- Processed image with drawings
- Landmarks list (if needed for debugging)

**When to use:**

- For testing
- For visualization
- For debugging

Not used in the final Phase 2/3 cursor system.

---

#### Method: `get_iris_landmarks()`

```python
data = detector.get_iris_landmarks(img)
```

Returns the raw iris landmark coordinates for both eyes.

**Output format:**

```python
{
    'left':  [(idx, x, y), ...],
    'right': [(idx, x, y), ...]
}
```

Where:

- `idx` — landmark ID
- `(x, y)` — pixel coordinates

**Notes:**

- Only 4 iris-specific landmarks are used per eye
- Used internally by `get_iris_centers()`

---

#### Method: `get_iris_centers()`

```python
centers = detector.get_iris_centers(img)
```

Calculates the center point of the iris for each eye via averaging.

**Returns:**

```python
{
    'left':  (cx, cy),
    'right': (cx, cy)
}
```

**Use Case:**

This is the primary output required for:

- Gaze estimation
- Calibration
- Cursor control

This is the official end-output of Phase 1.

---

#### Method: `draw_iris_centers()`

```python
detector.draw_iris_centers(img, centers, with_text=False)
```

Draws two small dots representing detected iris centers.

**Parameters:**

- `img` — input image
- `centers` — dictionary produced by `get_iris_centers()`
- `with_text` — optionally label left/right

**When to use:**

- Debugging
- Visual verification that tracking is correct

---

#### Method: `release()`

Releases MediaPipe internal resources.

Always call once when shutting down your program.

---

# 📌 Example Workflow (From example.py)

```python
from camera import IrisCamera
from face_utils import FaceMeshDetector
import cv2

face_mesh = FaceMeshDetector(refine_landmarks=True)

with IrisCamera(0) as camera:
    while True:
        frame = camera.get_frame()

        # Get iris centers
        centers = face_mesh.get_iris_centers(frame)
        
        # Draw centers
        face_mesh.draw_iris_centers(frame, centers)

        cv2.imshow("Iris Tracking", frame)
        if cv2.waitKey(1) == ord('q'):
            break

face_mesh.release()
```

This snippet:

1. Grabs webcam frames
2. Detects left/right iris positions
3. Draws debug dots
4. Shows live tracking

Exactly Phase 1 functionality.
```