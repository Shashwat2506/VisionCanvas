import cv2
import threading
from src.config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_BUFFER_SIZE


class Webcam:
    """Threaded webcam capture that always provides the most recent frame.

    A background thread continuously reads from the camera so that
    ``get_frame()`` never blocks on I/O and never returns a stale,
    buffered frame.
    """

    def __init__(self, camera_index: int = CAMERA_INDEX):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        # Minimize internal buffer to avoid processing stale frames
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)

        # Threaded capture state
        self._lock = threading.Lock()
        self._latest_frame = None
        self._running = True

        # Read first frame synchronously so we have something immediately
        success, frame = self.cap.read()
        if success:
            self._latest_frame = frame

        # Start background capture thread
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

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
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.cap.release()