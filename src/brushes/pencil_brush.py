import cv2
import numpy as np
from src.brushes.base_brush import BaseBrush


class PencilBrush(BaseBrush):
    """Semi-transparent pencil brush with natural graphite feel.

    Draws a fine, partially opaque line with round joints that mimics
    sketching with a fine-tipped graphite pencil.
    """

    def stroke(self, canvas, p1, p2):
        pen_w = max(1, int(self.thickness * 0.8))
        margin = pen_w + 4
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
        cv2.line(overlay, local_p1, local_p2, self.color, pen_w, lineType=cv2.LINE_AA)
        r = max(1, pen_w // 2)
        cv2.circle(overlay, local_p1, r, self.color, -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, local_p2, r, self.color, -1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.55, roi, 0.45, 0, dst=roi)