import math
import cv2


class DrawingCanvas:
    def __init__(self, width, height, brush):
        import numpy as np
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.brush = brush
        self.prev_point = None
        self.undo_stack = [self.canvas.copy()]
        self.redo_stack = []
        self.is_drawing = False

    def draw(self, point):
        if self.prev_point is None:
            if not self.is_drawing:
                self.undo_stack.append(self.canvas.copy())
                # Limit stack size to prevent memory issues
                if len(self.undo_stack) > 20:
                    self.undo_stack.pop(0)
                self.redo_stack.clear()
                self.is_drawing = True

        if self.prev_point is not None:
            dx = point[0] - self.prev_point[0]
            dy = point[1] - self.prev_point[1]
            dist = math.hypot(dx, dy)
            max_dist = 5.0
            if dist > max_dist:
                steps = int(dist / max_dist)
                for i in range(1, steps + 1):
                    t = i / steps
                    interp_x = int(self.prev_point[0] + t * dx)
                    interp_y = int(self.prev_point[1] + t * dy)
                    curr_p = (interp_x, interp_y)
                    t_prev = (i - 1) / steps
                    prev_interp_x = int(self.prev_point[0] + t_prev * dx)
                    prev_interp_y = int(self.prev_point[1] + t_prev * dy)
                    prev_p = (prev_interp_x, prev_interp_y)
                    self.brush.stroke(self.canvas, prev_p, curr_p)
            else:
                self.brush.stroke(self.canvas, self.prev_point, point)
        self.prev_point = point

    def reset_stroke(self):
        self.prev_point = None
        self.is_drawing = False

    def undo(self):
        if len(self.undo_stack) > 1:
            self.redo_stack.append(self.canvas.copy())
            self.canvas = self.undo_stack.pop().copy()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.canvas.copy())
            self.canvas = self.redo_stack.pop().copy()

    def set_brush(self, brush):
        self.brush = brush

    def clear(self):
        """Clear the canvas and push current state to undo stack."""
        import numpy as np
        self.undo_stack.append(self.canvas.copy())
        if len(self.undo_stack) > 20:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.canvas = np.zeros_like(self.canvas)
        self.prev_point = None
        self.is_drawing = False