import cv2
import threading
from src.config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_BUFFER_SIZE


class Webcam:
    """Threaded webcam capture that always provides the most recent frame.

    A background thread continuously reads from the camera so that
    ``get_frame()`` never blocks on I/O and never returns a stale,
    buffered frame.
    """

    def __init__(self, camera_index: int = CAMERA_INDEX, fallback_on_error: bool = True):
        self.camera_index = camera_index
        self.is_connected = False
        self._lock = threading.Lock()
        self._latest_frame = None
        self._running = False
        self._thread = None

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            if not fallback_on_error:
                raise RuntimeError(f"Could not open webcam at index {camera_index}.")
            self.is_connected = False
            self._latest_frame = self._generate_offline_frame()
            return

        self.is_connected = True
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)

        self._running = True

        # Read first frame synchronously so we have something immediately
        success, frame = self.cap.read()
        if success:
            self._latest_frame = frame
        else:
            self._latest_frame = self._generate_offline_frame()

        # Start background capture thread
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _generate_offline_frame(self):
        """Generate a clean placeholder frame when camera is unavailable."""
        import numpy as np
        frame = np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), (20, 22, 32), dtype=np.uint8)
        # Draw placeholder text and icon
        cv2.putText(frame, "CAMERA UNAVAILABLE", (CAMERA_WIDTH // 2 - 140, CAMERA_HEIGHT // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 125, 150), 2, cv2.LINE_AA)
        cv2.putText(frame, "Please check webcam connection / permissions",
                    (CAMERA_WIDTH // 2 - 180, CAMERA_HEIGHT // 2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 85, 110), 1, cv2.LINE_AA)
        return frame


    def _capture_loop(self):
        """Continuously grab the latest frame from the camera."""
        while self._running:
            success, frame = self.cap.read()
            if success:
                with self._lock:
                    self._latest_frame = frame

    def get_frame(self):
        """Return the most recent frame (never blocks on camera I/O).

        NOTE: The frame is NOT flipped here. main.py handles the flip
        so there is no double-flip.
        """
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def release(self):
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()