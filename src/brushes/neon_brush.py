import cv2
import numpy as np
from src.brushes.base_brush import BaseBrush


class NeonBrush(BaseBrush):
    """Neon glow brush using localised multi-layer rendering.

    Instead of copying the ENTIRE canvas 3 times per stroke point,
    we render the glow layers onto a small local region and blend once.
    """

    def stroke(self, canvas, p1, p2):
        max_extra = self.thickness + 12  # widest glow layer
        margin = max_extra + 10
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

        layers = [
            (self.thickness + 12, 0.2),
            (self.thickness + 6, 0.4),
            (self.thickness, 1.0),
        ]

        for width, opacity in layers:
            overlay = roi.copy()
            cv2.line(overlay, local_p1, local_p2, self.color, width, lineType=cv2.LINE_AA)
            cv2.addWeighted(overlay, opacity, roi, 1 - opacity, 0, dst=roi)