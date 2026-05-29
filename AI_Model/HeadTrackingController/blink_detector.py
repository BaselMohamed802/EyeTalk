"""
blink_detector.py
-----------------
Blink-to-click module using Eye Aspect Ratio (EAR) with MediaPipe FaceMesh.
Drop this file next to your other project files and import it.

Usage:
    from blink_detector import BlinkDetector

    detector = BlinkDetector()
    clicked = detector.process(face_landmarks)   # call every frame
    if clicked:
        # do your click action here
        pyautogui.click()
"""

import time
import pyautogui
pyautogui.FAILSAFE = False

# ---------------------------------------------------------------------------
# MediaPipe FaceMesh landmark indices for the eyes
# ---------------------------------------------------------------------------
# Each eye needs 6 points: 2 horizontal (corners) + 4 vertical (lids)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]   # [left, top1, top2, right, bot1, bot2]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]   # same layout


def _ear(landmarks, eye_indices, img_w, img_h):
    """
    Calculate Eye Aspect Ratio for one eye.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Open eye  → ~0.25–0.30
    Closed eye → < EAR_THRESHOLD (default 0.20)
    """
    def pt(idx):
        lm = landmarks[idx]
        return (lm.x * img_w, lm.y * img_h)

    p1, p2, p3, p4, p5, p6 = [pt(i) for i in eye_indices]

    def dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    vertical   = dist(p2, p6) + dist(p3, p5)
    horizontal = dist(p1, p4)

    if horizontal == 0:
        return 0.0
    return vertical / (2.0 * horizontal)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BlinkDetector:
    """
    Detects a deliberate blink and fires a single click event.

    Parameters
    ----------
    ear_threshold : float
        EAR value below which the eye is considered closed (default 0.20).
        Lower  → needs a harder blink to trigger.
        Higher → more sensitive, may false-trigger.

    consec_frames : int
        How many consecutive frames the eye must be below threshold
        before a blink is confirmed (default 2).
        Raise this if you get accidental clicks from normal blinking.

    cooldown_sec : float
        Minimum seconds between two clicks (default 1.0).
        Prevents rapid double-clicks from a slow eye opening.

    use_both_eyes : bool
        If True  → BOTH eyes must blink to trigger (less accidental).
        If False → either eye blinking triggers (more accessible).
        Default: False (easier for users with one functional eye).
    """

    def __init__(
        self,
        ear_threshold  = 0.22,
        consec_frames  = 2,
        cooldown_sec   = 1.0,
        use_both_eyes  = False,
    ):
        self.ear_threshold  = ear_threshold
        self.consec_frames  = consec_frames
        self.cooldown_sec   = cooldown_sec
        self.use_both_eyes  = use_both_eyes

        self._frame_count   = 0          # consecutive closed-eye frames
        self._last_click_t  = 0.0        # timestamp of last click
        self._blink_done    = False      # prevents re-firing while eye stays closed

    # ------------------------------------------------------------------
    def process(self, face_landmarks, img_w: int, img_h: int) -> bool:
        """
        Call this every frame with the landmark list from MediaPipe.

        Parameters
        ----------
        face_landmarks : mediapipe landmark list
            result.multi_face_landmarks[0].landmark
        img_w, img_h : int
            Width and height of the frame (needed to convert normalised coords).

        Returns
        -------
        bool
            True exactly once per confirmed blink (i.e. the frame to click).
            False every other frame.
        """
        lm = face_landmarks

        left_ear  = _ear(lm, LEFT_EYE,  img_w, img_h)
        right_ear = _ear(lm, RIGHT_EYE, img_w, img_h)

        # Decide if the eye condition is met
        if self.use_both_eyes:
            eye_closed = (left_ear < self.ear_threshold and
                          right_ear < self.ear_threshold)
        else:
            eye_closed = (left_ear < self.ear_threshold or
                          right_ear < self.ear_threshold)

        # Count consecutive closed frames
        if eye_closed:
            self._frame_count += 1
        else:
            # Eye just opened → reset
            self._frame_count = 0
            self._blink_done  = False

        # Fire click when threshold is reached (once per blink)
        if (self._frame_count >= self.consec_frames
                and not self._blink_done):

            now = time.time()
            if now - self._last_click_t >= self.cooldown_sec:
                self._last_click_t = now
                self._blink_done   = True   # don't re-fire until eye opens again
                return True                 # ← CLICK!

        return False

    # ------------------------------------------------------------------
    def get_ear(self, face_landmarks, img_w: int, img_h: int):
        """
        Helper: returns (left_ear, right_ear) floats.
        Useful if you want to show a live EAR value on screen for debugging.
        """
        lm = face_landmarks
        return (
            _ear(lm, LEFT_EYE,  img_w, img_h),
            _ear(lm, RIGHT_EYE, img_w, img_h),
        )
    # at the very bottom of blink_detector.py
if __name__ == "__main__":
    import cv2
    import mediapipe as mp

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(max_num_faces=1)
    detector = BlinkDetector()

    cap = cv2.VideoCapture(0)
    print("Running blink test — blink to see CLICK printed. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0].landmark
            clicked = detector.process(face, w, h)

            # show live EAR values on screen
            left_ear, right_ear = detector.get_ear(face, w, h)
            cv2.putText(frame, f"L: {left_ear:.2f}  R: {right_ear:.2f}",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if clicked:
                print("CLICK!")
                pyautogui.click()

                cv2.putText(frame, "BLINK DETECTED",
                            (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Blink Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()