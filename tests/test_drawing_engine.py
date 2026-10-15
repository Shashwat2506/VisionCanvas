import unittest
import numpy as np
import cv2
import math

from src.config import BRUSH_DEFAULT_SIZE, SMART_SHAPES_DEFAULT, BRUSH_SIZES
from src.canvas.drawing_canvas import DrawingCanvas
from src.canvas.shape_recognizer import ShapeRecognizer
from src.canvas.text_recognizer import TextRecognizer
from src.ui.modern_ui import ModernUI
from src.brushes.hard_brush import HardBrush
from src.brushes.soft_brush import SoftBrush
from src.brushes.neon_brush import NeonBrush
from src.brushes.pencil_brush import PencilBrush
from src.brushes.watercolor_brush import WatercolorBrush
from src.gestures.cursor import VirtualCursor


class TestVisionCanvasEngine(unittest.TestCase):

    def setUp(self):
        self.w, self.h = 640, 480
        self.brush = HardBrush((255, 0, 255), thickness=4)
        self.canvas = DrawingCanvas(self.w, self.h, self.brush)
        self.shape_rec = ShapeRecognizer()
        self.text_rec = TextRecognizer()

    def test_brush_thickness(self):
        """Verify brush sizes and preset rendering."""
        self.assertEqual(BRUSH_DEFAULT_SIZE, 4)
        self.assertIn(4, BRUSH_SIZES)
        
        # Test neon brush does not blow out
        neon = NeonBrush((255, 255, 0), thickness=4)
        test_img = np.zeros((100, 200, 3), dtype=np.uint8)
        neon.stroke(test_img, (20, 50), (180, 50))
        non_zero = np.count_nonzero(test_img)
        self.assertGreater(non_zero, 0)
        # Verify vertical stroke profile is tight (should not exceed 18px total glow width)
        y_indices = np.where(test_img[:, 100, 0] > 0)[0]
        if len(y_indices) > 0:
            stroke_h = y_indices.max() - y_indices.min()
            self.assertLessEqual(stroke_h, 18, f"Neon stroke height {stroke_h} exceeds 18px")


    def test_straight_line_recognition(self):
        """Verify rough freehand line is beautified to straight line."""
        pts = [(100 + i * 5, 100 + int(math.sin(i) * 2)) for i in range(35)]
        base = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        beautified, name = self.shape_rec.recognize_and_beautify(pts, base, (0, 255, 0), 4)
        self.assertEqual(name, "Straight Line")
        self.assertIsNotNone(beautified)

    def test_circle_recognition(self):
        """Verify rough freehand circle is beautified."""
        pts = [(int(200 + 50 * math.cos(th)), int(200 + 50 * math.sin(th)))
               for th in np.linspace(0, 2 * math.pi, 28)]
        base = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        beautified, name = self.shape_rec.recognize_and_beautify(pts, base, (0, 255, 255), 4)
        self.assertEqual(name, "Circle")
        self.assertIsNotNone(beautified)

    def test_rectangle_recognition(self):
        """Verify rough rectangle is beautified."""
        pts = ([(100 + i * 6, 100) for i in range(20)] +
               [(220, 100 + i * 5) for i in range(16)] +
               [(220 - i * 6, 180) for i in range(20)] +
               [(100, 180 - i * 5) for i in range(16)])
        base = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        beautified, name = self.shape_rec.recognize_and_beautify(pts, base, (255, 100, 50), 4)
        self.assertEqual(name, "Rectangle")
        self.assertIsNotNone(beautified)

    def test_triangle_recognition(self):
        """Verify rough triangle is beautified."""
        pts = ([(int(100 + t * 50), int(250 - t * 80)) for t in np.linspace(0, 1, 12)] +
               [(int(150 + t * 50), int(170 + t * 80)) for t in np.linspace(0, 1, 12)] +
               [(int(200 - t * 100), 250) for t in np.linspace(0, 1, 15)])
        base = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        beautified, name = self.shape_rec.recognize_and_beautify(pts, base, (180, 80, 255), 4)
        self.assertEqual(name, "Triangle")
        self.assertIsNotNone(beautified)

    def test_curve_preservation(self):
        """Verify non-geometric freehand curve is NOT converted to shape."""
        # Wavy S-curve
        pts = [(100 + i * 4, int(200 + 40 * math.sin(i * 0.2))) for i in range(40)]
        base = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        beautified, name = self.shape_rec.recognize_and_beautify(pts, base, (255, 255, 255), 4)
        self.assertIsNone(beautified, "Curved stroke was improperly converted to a shape")

    def test_text_recognition_and_rendering(self):
        """Verify character recognition and digital rendering."""
        o_stroke = [(int(100 + 35 * math.cos(th)), int(100 + 45 * math.sin(th)))
                    for th in np.linspace(0, 2 * math.pi, 24)]
        text, conf, bbox = self.text_rec.recognize_strokes([o_stroke])
        self.assertEqual(text, "O")
        self.assertGreater(conf, 0.70)

        # Render digital font onto canvas
        base = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        rendered = self.text_rec.render_text_to_canvas(base, "AI", (50, 50, 60, 40), (0, 255, 255))
        self.assertGreater(np.count_nonzero(rendered), 0)

    def test_undo_redo_history(self):
        """Verify undo/redo stack consistency."""
        self.canvas.draw((50, 50))
        self.canvas.draw((60, 60))
        pts, _ = self.canvas.reset_stroke()
        self.assertEqual(len(pts), 2)
        self.assertEqual(len(self.canvas.undo_stack), 2)

        # Undo
        self.canvas.undo()
        self.assertEqual(len(self.canvas.undo_stack), 1)
        self.assertEqual(np.count_nonzero(self.canvas.canvas), 0)

        # Redo
        self.canvas.redo()
        self.assertEqual(len(self.canvas.undo_stack), 2)
        self.assertGreater(np.count_nonzero(self.canvas.canvas), 0)

    def test_ui_interaction_and_focus_mode(self):
        """Verify UI workspace expansion, focus mode, and coordinate mapping."""
        ui = ModernUI(1280, 720, 640, 480)
        self.assertFalse(ui.focus_mode)
        # Coordinate mapping inside workspace
        cx, cy = ui.ui_to_canvas(ui.video_x + 10, ui.video_y + 10)
        self.assertIsNotNone(cx)
        self.assertIsNotNone(cy)

        # Focus mode toggle
        ui.toggle_focus_mode()
        self.assertTrue(ui.focus_mode)
        self.assertEqual(ui.left_w, 0)
        self.assertEqual(ui.right_w, 0)
        self.assertGreaterEqual(ui.disp_h, 630)


if __name__ == "__main__":
    unittest.main()
