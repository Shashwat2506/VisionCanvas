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
    hand-tracking.  Large jumps (likely tracking glitches) are rejected.
    """

    def __init__(self):
        self.filter_x = None
        self.filter_y = None
        self.last_x = 0
        self.last_y = 0

    def update(self, landmark, frame_width: int, frame_height: int):
        raw_x = landmark.x * frame_width
        raw_y = landmark.y * frame_height

        t = time.time()

        if self.filter_x is None:
            # First frame — initialise filters with sane defaults
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
            return self.last_x, self.last_y

        # Reject large jumps (tracking glitches)
        dx = abs(raw_x - self.last_x)
        dy = abs(raw_y - self.last_y)
        if (dx + dy) > MAX_CURSOR_JUMP:
            return self.last_x, self.last_y

        if SMOOTHING_ENABLED:
            sx = self.filter_x(t, raw_x)
            sy = self.filter_y(t, raw_y)
        else:
            sx, sy = raw_x, raw_y

        self.last_x = int(sx)
        self.last_y = int(sy)
        return self.last_x, self.last_y