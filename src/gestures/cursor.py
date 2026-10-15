import time
from src.utils.one_euro_filter import OneEuroFilter
from src.config import (
    SMOOTHING_ENABLED,
    ONE_EURO_MIN_CUTOFF,
    ONE_EURO_BETA,
    ONE_EURO_DCUTOFF,
    MAX_CURSOR_JUMP,
)


class VirtualCursor:
    """Converts hand landmark positions to smooth screen coordinates.

    Uses a pair of 1€ filters (one per axis) tuned for low-latency
    jitter-free hand tracking. Handles tracking re-acquisition gracefully
    without freezing or jumping.
    """

    def __init__(self):
        self.filter_x = None
        self.filter_y = None
        self.last_x = 0
        self.last_y = 0
        self.last_time = 0.0
        self.glitch_frames = 0

    def reset(self):
        """Reset filter state when hand tracking is lost."""
        self.filter_x = None
        self.filter_y = None
        self.glitch_frames = 0

    def update(self, landmark, frame_width: int, frame_height: int):
        raw_x = landmark.x * frame_width
        raw_y = landmark.y * frame_height

        t = time.time()
        dt = t - self.last_time if self.last_time > 0 else 0.0
        self.last_time = t

        # If tracking was absent for more than 0.25s, re-initialise immediately
        if self.filter_x is None or dt > 0.25:
            self.filter_x = OneEuroFilter(
                t, raw_x,
                mincutoff=ONE_EURO_MIN_CUTOFF,
                beta=ONE_EURO_BETA,
                dcutoff=ONE_EURO_DCUTOFF,
            )
            self.filter_y = OneEuroFilter(
                t, raw_y,
                mincutoff=ONE_EURO_MIN_CUTOFF,
                beta=ONE_EURO_BETA,
                dcutoff=ONE_EURO_DCUTOFF,
            )
            self.last_x = int(raw_x)
            self.last_y = int(raw_y)
            self.glitch_frames = 0
            return self.last_x, self.last_y

        # Check for large abrupt jump
        dx = abs(raw_x - self.last_x)
        dy = abs(raw_y - self.last_y)
        if (dx + dy) > MAX_CURSOR_JUMP:
            self.glitch_frames += 1
            # If large distance persists for > 3 frames, accept it as deliberate fast movement
            if self.glitch_frames < 4:
                return self.last_x, self.last_y
        else:
            self.glitch_frames = 0

        if SMOOTHING_ENABLED:
            sx = self.filter_x(t, raw_x)
            sy = self.filter_y(t, raw_y)
        else:
            sx, sy = raw_x, raw_y

        self.last_x = int(round(sx))
        self.last_y = int(round(sy))
        return self.last_x, self.last_y