import cv2
from src.brushes.base_brush import BaseBrush

class HardBrush(BaseBrush):
    def stroke(self, canvas, p1, p2):
        cv2.line(canvas, p1, p2, self.color, self.thickness, lineType=cv2.LINE_AA)