import math
import cv2
import numpy as np
from src.config import MAX_INTERPOLATION_DISTANCE, MAX_CURSOR_JUMP


class DrawingCanvas:
    """Manages the drawing surface, stroke interpolation, and undo/redo history.

    Maintains stroke point telemetry for intelligent shape beautification
    and handwriting recognition, while ensuring fine, continuous strokes.
    """

    def __init__(self, width: int, height: int, brush):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.brush = brush
        self.prev_point = None
        self.undo_stack = [self.canvas.copy()]
        self.redo_stack = []
        self.is_drawing = False
        self.current_stroke_points = []
        self.stroke_start_canvas = None

    def draw(self, point):
        """Add a point to the current stroke with anti-aliased interpolation."""
        if point is None:
            return

        if self.prev_point is None:
            if not self.is_drawing:
                self.stroke_start_canvas = self.canvas.copy()
                self.undo_stack.append(self.stroke_start_canvas.copy())
                if len(self.undo_stack) > 25:
                    self.undo_stack.pop(0)
                self.redo_stack.clear()
                self.is_drawing = True
                self.current_stroke_points = [point]

                # Draw initial contact dot so short touches register cleanly
                r = max(1, self.brush.thickness // 2)
                cv2.circle(self.canvas, point, r, self.brush.color, -1, lineType=cv2.LINE_AA)
        else:
            dx = point[0] - self.prev_point[0]
            dy = point[1] - self.prev_point[1]
            dist = math.hypot(dx, dy)

            # Ignore tiny sub-pixel noise jitter
            if dist < 1.0:
                return

            # Reject sudden tracking glitch jumps to prevent long connecting lines
            if dist > MAX_CURSOR_JUMP:
                self.prev_point = point
                self.current_stroke_points.append(point)
                return

            max_dist = MAX_INTERPOLATION_DISTANCE
            if dist > max_dist:
                steps = max(1, int(dist / max_dist))
                for i in range(1, steps + 1):
                    t = i / steps
                    t_prev = (i - 1) / steps
                    p_prev = (
                        int(round(self.prev_point[0] + t_prev * dx)),
                        int(round(self.prev_point[1] + t_prev * dy)),
                    )
                    p_curr = (
                        int(round(self.prev_point[0] + t * dx)),
                        int(round(self.prev_point[1] + t * dy)),
                    )
                    self.brush.stroke(self.canvas, p_prev, p_curr)
            else:
                self.brush.stroke(self.canvas, self.prev_point, point)

            self.current_stroke_points.append(point)

        self.prev_point = point

    def reset_stroke(self):
        """End the current stroke and return points for shape/text recognition."""
        pts = list(self.current_stroke_points)
        start_canvas = self.stroke_start_canvas
        self.prev_point = None
        self.is_drawing = False
        self.current_stroke_points = []
        self.stroke_start_canvas = None
        return pts, start_canvas

    def replace_canvas(self, new_canvas):
        """Replace current canvas state (e.g. with a beautified shape)."""
        if new_canvas is not None and new_canvas.shape == self.canvas.shape:
            self.canvas = new_canvas.copy()

    def commit_operation(self, new_canvas, base_canvas=None):
        """Commit an external operation (like recognized text) as a single undoable step.

        If base_canvas is provided, rolls back any intermediate strokes from undo_stack
        to base_canvas, then records base_canvas as the undo state and updates current canvas.
        """
        if new_canvas is None or new_canvas.shape != self.canvas.shape:
            return

        if base_canvas is not None:
            found_idx = None
            for idx in range(len(self.undo_stack) - 1, -1, -1):
                if np.array_equal(self.undo_stack[idx], base_canvas):
                    found_idx = idx
                    break
            if found_idx is not None:
                self.undo_stack = self.undo_stack[:found_idx + 1]
            else:
                self.undo_stack.append(base_canvas.copy())
        else:
            self.undo_stack.append(self.canvas.copy())

        if len(self.undo_stack) > 25:
            self.undo_stack.pop(0)

        self.redo_stack.clear()
        self.canvas = new_canvas.copy()
        self.prev_point = None
        self.is_drawing = False
        self.current_stroke_points = []
        self.stroke_start_canvas = None

    def revert_to(self, base_canvas):
        """Discard recent temporary strokes and revert cleanly to base_canvas."""
        if base_canvas is None or base_canvas.shape != self.canvas.shape:
            return

        found_idx = None
        for idx in range(len(self.undo_stack) - 1, -1, -1):
            if np.array_equal(self.undo_stack[idx], base_canvas):
                found_idx = idx
                break

        if found_idx is not None:
            self.undo_stack = self.undo_stack[:found_idx + 1]

        self.canvas = base_canvas.copy()
        self.redo_stack.clear()
        self.prev_point = None
        self.is_drawing = False
        self.current_stroke_points = []
        self.stroke_start_canvas = None

    def undo(self):
        """Revert the last stroke or operation."""
        if len(self.undo_stack) > 1:
            self.redo_stack.append(self.canvas.copy())
            self.canvas = self.undo_stack.pop().copy()
            self.prev_point = None
            self.is_drawing = False
            self.current_stroke_points = []

    def redo(self):
        """Restore the previously undone stroke or operation."""
        if self.redo_stack:
            self.undo_stack.append(self.canvas.copy())
            self.canvas = self.redo_stack.pop().copy()
            self.prev_point = None
            self.is_drawing = False
            self.current_stroke_points = []

    def set_brush(self, brush):
        """Change the active drawing brush."""
        self.brush = brush

    def clear(self):
        """Clear the canvas and record in undo history."""
        self.undo_stack.append(self.canvas.copy())
        if len(self.undo_stack) > 25:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.canvas = np.zeros_like(self.canvas)
        self.prev_point = None
        self.is_drawing = False
        self.current_stroke_points = []
        self.stroke_start_canvas = None