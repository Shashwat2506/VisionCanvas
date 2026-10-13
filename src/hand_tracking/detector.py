import cv2
import time
import threading
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from src.config import MODEL_PATH, MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE


class HandDetector:
    """MediaPipe hand landmark detector running in VIDEO mode.

    Can be used synchronously or driven from a background thread via
    ``detect()``.  Results are stored thread-safely so the main loop
    can read them without blocking.
    """

    def __init__(self):
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.frame_timestamp_ms = 0

        # Thread-safe latest result
        self._lock = threading.Lock()
        self._latest_result = None
        self._latest_frame = None  # the frame that produced the result

    def detect(self, frame):
        """Run inference on *frame* and store result thread-safely.

        Returns the result directly as well, for convenience when
        called synchronously.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        current_ts = int(time.time() * 1000)
        if current_ts <= self.frame_timestamp_ms:
            current_ts = self.frame_timestamp_ms + 1
        self.frame_timestamp_ms = current_ts

        result = self.landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)

        with self._lock:
            self._latest_result = result
            self._latest_frame = frame

        return result

    def get_latest(self):
        """Return (result, frame) from the most recent inference."""
        with self._lock:
            return self._latest_result, self._latest_frame