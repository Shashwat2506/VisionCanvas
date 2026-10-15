import cv2
import numpy as np
from src.brushes.base_brush import BaseBrush


class SoftBrush(BaseBrush):
    """Soft-edged brush with proportional feathered falloff.

    Keeps the core stroke crisp and narrow while providing a subtle,
    beautiful soft edge that respects the selected brush thickness.
    """

    def stroke(self, canvas, p1, p2):
        # Blur kernel size proportional to thickness
        k_size = max(3, (self.thickness // 2) * 2 + 1)
        if k_size % 2 == 0:
            k_size += 1
        margin = self.thickness + k_size + 4
        h, w = canvas.shape[:2]

        x_min = max(0, min(p1[0], p2[0]) - margin)
        y_min = max(0, min(p1[1], p2[1]) - margin)
        x_max = min(w, max(p1[0], p2[0]) + margin)
        y_max = min(h, max(p1[1], p2[1]) + margin)

        rw = x_max - x_min
        rh = y_max - y_min
        if rw <= 0 or rh <= 0:
            return

        # Draw stroke mask on single-channel grayscale
        mask = np.zeros((rh, rw), dtype=np.uint8)
        local_p1 = (p1[0] - x_min, p1[1] - y_min)
        local_p2 = (p2[0] - x_min, p2[1] - y_min)
        cv2.line(mask, local_p1, local_p2, 255, self.thickness, lineType=cv2.LINE_AA)
        r = max(1, self.thickness // 2)
        cv2.circle(mask, local_p1, r, 255, -1, lineType=cv2.LINE_AA)
        cv2.circle(mask, local_p2, r, 255, -1, lineType=cv2.LINE_AA)

        # Soft feathering
        mask_soft = cv2.GaussianBlur(mask, (k_size, k_size), 0)

        # Alpha blend into ROI without blown-out additive over-exposure
        roi = canvas[y_min:y_max, x_min:x_max]
        alpha = (mask_soft.astype(np.float32) / 255.0)[..., np.newaxis]
        color_arr = np.array(self.color, dtype=np.float32)
        blended = (alpha * color_arr + (1.0 - alpha) * roi.astype(np.float32)).clip(0, 255).astype(np.uint8)
        roi[:] = blended