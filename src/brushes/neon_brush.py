import cv2
import numpy as np
from src.brushes.base_brush import BaseBrush


class NeonBrush(BaseBrush):
    """Neon brush with a tight luminous glow and sharp legible core.

    Maintains a narrow, precise center stroke while applying a subtle
    proportional glow halo that never overwhelms the drawing.
    """

    def stroke(self, canvas, p1, p2):
        glow_extra = max(2, min(8, self.thickness + 2))
        margin = self.thickness + glow_extra + 6
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

        # Two glow layers + core stroke
        glow_w2 = self.thickness + glow_extra
        glow_w1 = self.thickness + max(1, glow_extra // 2)

        layers = [
            (glow_w2, self.color, 0.25),
            (glow_w1, self.color, 0.45),
            (self.thickness, self.color, 1.0),
        ]

        for width, color, opacity in layers:
            overlay = roi.copy()
            cv2.line(overlay, local_p1, local_p2, color, width, lineType=cv2.LINE_AA)
            r = max(1, width // 2)
            cv2.circle(overlay, local_p1, r, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(overlay, local_p2, r, color, -1, lineType=cv2.LINE_AA)
            cv2.addWeighted(overlay, opacity, roi, 1.0 - opacity, 0, dst=roi)

        # Highlight center with ultra-fine core dot/line for neon wire look
        if self.thickness >= 3:
            wire_w = max(1, self.thickness - 2)
            wire_col = (
                min(255, self.color[0] + 120),
                min(255, self.color[1] + 120),
                min(255, self.color[2] + 120),
            )
            overlay = roi.copy()
            cv2.line(overlay, local_p1, local_p2, wire_col, wire_w, lineType=cv2.LINE_AA)
            r = max(1, wire_w // 2)
            cv2.circle(overlay, local_p1, r, wire_col, -1, lineType=cv2.LINE_AA)
            cv2.circle(overlay, local_p2, r, wire_col, -1, lineType=cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.6, roi, 0.4, 0, dst=roi)