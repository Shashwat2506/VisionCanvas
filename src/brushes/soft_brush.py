import cv2
import numpy as np
from src.brushes.base_brush import BaseBrush


class SoftBrush(BaseBrush):
    """Soft-edged brush using localised Gaussian blur.

    Instead of allocating/blurring a full-canvas overlay, we only
    process a small region around the stroke segment.
    """

    def stroke(self, canvas, p1, p2):
        margin = self.thickness + 20  # extra room for the blur kernel
        h, w = canvas.shape[:2]

        # Bounding box of the stroke segment + margin
        x_min = max(0, min(p1[0], p2[0]) - margin)
        y_min = max(0, min(p1[1], p2[1]) - margin)
        x_max = min(w, max(p1[0], p2[0]) + margin)
        y_max = min(h, max(p1[1], p2[1]) + margin)

        rw = x_max - x_min
        rh = y_max - y_min
        if rw <= 0 or rh <= 0:
            return

        # Draw on a small local overlay
        local = np.zeros((rh, rw, 3), dtype=np.uint8)
        local_p1 = (p1[0] - x_min, p1[1] - y_min)
        local_p2 = (p2[0] - x_min, p2[1] - y_min)
        cv2.line(local, local_p1, local_p2, self.color, self.thickness, lineType=cv2.LINE_AA)
        local = cv2.GaussianBlur(local, (15, 15), 0)

        # Additive blend into the canvas region
        roi = canvas[y_min:y_max, x_min:x_max]
        cv2.add(roi, local, dst=roi)