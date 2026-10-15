import cv2
from src.brushes.base_brush import BaseBrush

class HardBrush(BaseBrush):
    def stroke(self, canvas, p1, p2):
        cv2.line(canvas, p1, p2, self.color, self.thickness, lineType=cv2.LINE_AA)
        r = max(1, self.thickness // 2)
        cv2.circle(canvas, p1, r, self.color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p2, r, self.color, -1, lineType=cv2.LINE_AA)