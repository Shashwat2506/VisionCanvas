from abc import ABC, abstractmethod

class BaseBrush(ABC):
    def __init__(self, color=(255, 0, 255), thickness=8):
        self.color = color
        self.thickness = thickness

    @abstractmethod
    def stroke(self, canvas, p1, p2):
        pass