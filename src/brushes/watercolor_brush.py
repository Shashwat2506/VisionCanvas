import cv2
import random
import numpy as np
from src.brushes.base_brush import BaseBrush


class WatercolorBrush(BaseBrush):
    """Watercolor brush with organic wash and translucent layering.

    Uses smooth rounded joints and a layered wash effect that mimics
    water-thinned pigment on paper.
    """

    def stroke(self, canvas, p1, p2):
        jitter_thickness = max(1, self.thickness + random.randint(-1, 1))
        margin = jitter_thickness + 6
        h, w = canvas.shape[:2]

        x_min = max(0, min(p1[0], p2[0]) - margin)
        y_min = max(0, min(p1[1], p2[1]) - margin)
        x_max = min(w, max(p1[0], p2[0]) + margin)
        y_max = min(h, max(p1[1], p2[1]) + margin)

        rw = x_max - x_min
        rh = y_max - y_min
        if rw <= 0 or rh <= 0:
            return

        local_p1 = (p1[0] - x_min, p1[1] - y_min)
        local_p2 = (p2[0] - x_min, p2[1] - y_min)

        roi = canvas[y_min:y_max, x_min:x_max]
        overlay = roi.copy()
        cv2.line(overlay, local_p1, local_p2, self.color, jitter_thickness, lineType=cv2.LINE_AA)
        r = max(1, jitter_thickness // 2)
        cv2.circle(overlay, local_p1, r, self.color, -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, local_p2, r, self.color, -1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.40, roi, 0.60, 0, dst=roi)