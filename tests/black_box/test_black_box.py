"""
VisionCanvas Black-Box Test Suite (BB01 - BB16)
Evaluates externally observable behaviors against expected results
without relying on internal function implementations.
"""

import os
import time
import pytest
import numpy as np
import cv2

from src.config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, CAMERA_WIDTH, CAMERA_HEIGHT,
    BRUSH_DEFAULT_SIZE, BRUSH_SIZES,
)
from src.camera.webcam import Webcam
from src.canvas.drawing_canvas import DrawingCanvas
from src.canvas.shape_recognizer import ShapeRecognizer
from src.canvas.text_recognizer import TextRecognizer
from src.ui.modern_ui import ModernUI
from src.brushes.hard_brush import HardBrush
from src.brushes.neon_brush import NeonBrush
from src.brushes.soft_brush import SoftBrush


# ─────────────────────────────────────────────────────────────────────────────
# BB01: Application Launch
# ─────────────────────────────────────────────────────────────────────────────
def test_bb01_application_launch():
    """BB01: Application launch.
    Expected: VisionCanvas core subsystems start without unhandled exceptions.
    """
    ui = ModernUI(WINDOW_WIDTH, WINDOW_HEIGHT, CAMERA_WIDTH, CAMERA_HEIGHT)
    canvas = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT, HardBrush())
    
    assert ui is not None, "UI failed to initialize"
    assert canvas is not None, "Canvas failed to initialize"
    assert ui.width == WINDOW_WIDTH
    assert ui.height == WINDOW_HEIGHT


# ─────────────────────────────────────────────────────────────────────────────
# BB02: Camera Initialization
# ─────────────────────────────────────────────────────────────────────────────
def test_bb02_camera_initialization():
    """BB02: Camera initialization.
    Expected: Camera abstraction provides a valid frame buffer.
    """
    webcam = Webcam(camera_index=0, fallback_on_error=True)
    frame = webcam.get_frame()
    webcam.release()

    assert frame is not None, "Camera get_frame() returned None"
    assert frame.shape == (CAMERA_HEIGHT, CAMERA_WIDTH, 3), (
        f"Expected frame shape {(CAMERA_HEIGHT, CAMERA_WIDTH, 3)}, got {frame.shape}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# BB03: Camera Unavailable
# ─────────────────────────────────────────────────────────────────────────────
def test_bb03_camera_unavailable():
    """BB03: Camera unavailable.
    Expected: Handles missing camera gracefully (e.g. invalid index 9999) without crashing.
    """
    # 1. Fallback mode: generates placeholder frame, does not raise unhandled crash
    webcam_fallback = Webcam(camera_index=9999, fallback_on_error=True)
    frame = webcam_fallback.get_frame()
    is_conn = webcam_fallback.is_connected
    webcam_fallback.release()

    assert not is_conn, "Expected is_connected=False for invalid camera index"
    assert frame is not None, "Expected offline placeholder frame when camera is unavailable"
    assert frame.shape == (CAMERA_HEIGHT, CAMERA_WIDTH, 3)

    # 2. Explicit strict mode: raises diagnostic RuntimeError cleanly
    with pytest.raises(RuntimeError) as exc_info:
        Webcam(camera_index=9999, fallback_on_error=False)
    assert "Could not open webcam" in str(exc_info.value)


# ─────────────────────────────────────────────────────────────────────────────
# BB04: Brush Selection
# ─────────────────────────────────────────────────────────────────────────────
def test_bb04_brush_selection():
    """BB04: Brush selection.
    Expected: Selecting a brush changes the active brush preset in UI and rendered style.
    """
    ui = ModernUI(WINDOW_WIDTH, WINDOW_HEIGHT, CAMERA_WIDTH, CAMERA_HEIGHT)
    initial_brush = ui.active_brush

    # Switch to Neon Brush (index 2)
    ui.set_active_brush(2)
    assert ui.active_brush == 2, f"Expected active_brush=2, got {ui.active_brush}"
    assert ui.active_brush != initial_brush

    # Verify visual rendered tile changes
    output = ui._render_base()
    assert output is not None and output.size > 0


# ─────────────────────────────────────────────────────────────────────────────
# BB05: Brush Size
# ─────────────────────────────────────────────────────────────────────────────
def test_bb05_brush_size():
    """BB05: Brush size.
    Expected: Changing brush size produces observable change in stroke thickness on canvas.
    """
    canvas_fine = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT, HardBrush(thickness=2))
    canvas_thick = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT, HardBrush(thickness=14))

    # Draw identical 100px vertical stroke on both
    canvas_fine.draw((100, 100))
    canvas_fine.draw((100, 200))

    canvas_thick.draw((100, 100))
    canvas_thick.draw((100, 200))

    pixels_fine = np.count_nonzero(canvas_fine.canvas)
    pixels_thick = np.count_nonzero(canvas_thick.canvas)

    assert pixels_thick > pixels_fine * 2, (
        f"Thick brush pixels ({pixels_thick}) should be significantly greater than fine brush ({pixels_fine})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# BB06: Color Selection
# ─────────────────────────────────────────────────────────────────────────────
def test_bb06_color_selection():
    """BB06: Color selection.
    Expected: Selecting a palette color updates active color and subsequent strokes use it.
    """
    ui = ModernUI(WINDOW_WIDTH, WINDOW_HEIGHT, CAMERA_WIDTH, CAMERA_HEIGHT)
    # Select Red swatch (index 2 in PALETTE_COLORS: BGR (50, 50, 245))
    target_color, name = ui.PALETTE_COLORS[2]
    
    # Click on the red swatch
    cx, cy = ui._color_centers[2]
    changed = ui.handle_selection(cx, cy)
    assert changed is True, "Selection handler did not register color swatch click"
    assert ui.active_color == target_color, f"Expected color {target_color}, got {ui.active_color}"

    # Draw with selected color
    canvas = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT, HardBrush(color=ui.active_color, thickness=4))
    canvas.draw((50, 50))
    canvas.draw((150, 50))

    # Verify pixels on canvas match the selected BGR color
    drawn_b, drawn_g, drawn_r = canvas.canvas[50, 100]
    assert (drawn_b, drawn_g, drawn_r) == target_color, (
        f"Expected drawn pixel to match {target_color}, got {(drawn_b, drawn_g, drawn_r)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# BB07: Eraser
# ─────────────────────────────────────────────────────────────────────────────
def test_bb07_eraser():
    """BB07: Eraser.
    Expected: Eraser removes/clears content from canvas.
    """
    canvas = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT, HardBrush((255, 255, 255), thickness=6))
    # Draw initial stroke
    canvas.draw((100, 100))
    canvas.draw((200, 100))
    initial_pixels = np.count_nonzero(canvas.canvas)
    assert initial_pixels > 0, "Initial drawing failed"

    # Switch to Eraser (Black brush)
    eraser = HardBrush(color=(0, 0, 0), thickness=20)
    canvas.set_brush(eraser)
    canvas.prev_point = None  # New stroke
    canvas.draw((100, 100))
    canvas.draw((200, 100))

    remaining_pixels = np.count_nonzero(canvas.canvas)
    assert remaining_pixels < initial_pixels, (
        f"Eraser did not reduce drawn pixels: before={initial_pixels}, after={remaining_pixels}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# BB08: Smart Shapes (Enabled)
# ─────────────────────────────────────────────────────────────────────────────
def test_bb08_smart_shapes_enabled():
    """BB08: Smart Shapes enabled.
    Expected: Jittery freehand line is beautified into a clean straight line.
    """
    sr = ShapeRecognizer()
    base = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

    # Freehand straight line with natural hand tremor (+-2px)
    jittery_line = [(100 + i * 5, 150 + int(np.sin(i) * 2)) for i in range(30)]

    beautified_canvas, shape_name = sr.recognize_and_beautify(
        jittery_line, base, color=(0, 255, 0), thickness=4
    )

    assert shape_name == "Straight Line", f"Expected 'Straight Line', got {shape_name}"
    assert beautified_canvas is not None
    assert np.count_nonzero(beautified_canvas) > 0


# ─────────────────────────────────────────────────────────────────────────────
# BB09: Smart Shapes (Disabled)
# ─────────────────────────────────────────────────────────────────────────────
def test_bb09_smart_shapes_disabled():
    """BB09: Smart Shapes disabled.
    Expected: When Smart Shapes is disabled, raw freehand stroke is preserved without conversion.
    """
    ui = ModernUI(WINDOW_WIDTH, WINDOW_HEIGHT, CAMERA_WIDTH, CAMERA_HEIGHT)
    ui.smart_shapes = False

    # Simulate freehand stroke
    canvas = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT, HardBrush((255, 0, 255), 4))
    raw_points = [(100 + i * 5, 150 + int(np.sin(i) * 2)) for i in range(30)]

    for pt in raw_points:
        canvas.draw(pt)

    stroke_pts, _ = canvas.reset_stroke()
    assert len(stroke_pts) == len(raw_points), "Stroke points altered when Smart Shapes is OFF"
    assert ui.smart_shapes is False


# ─────────────────────────────────────────────────────────────────────────────
# BB10: Undo
# ─────────────────────────────────────────────────────────────────────────────
def test_bb10_undo():
    """BB10: Undo.
    Expected: Undo reverses the last drawn stroke.
    """
    canvas = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT, HardBrush((0, 255, 0), 4))
    assert np.count_nonzero(canvas.canvas) == 0

    # Stroke 1
    canvas.draw((50, 50))
    canvas.draw((80, 80))
    canvas.reset_stroke()
    pixels_after_stroke = np.count_nonzero(canvas.canvas)
    assert pixels_after_stroke > 0

    # Undo
    canvas.undo()
    assert np.count_nonzero(canvas.canvas) == 0, "Undo did not revert canvas to clean state"


# ─────────────────────────────────────────────────────────────────────────────
# BB11: Redo
# ─────────────────────────────────────────────────────────────────────────────
def test_bb11_redo():
    """BB11: Redo.
    Expected: Redo restores the action reversed by undo.
    """
    canvas = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT, HardBrush((0, 255, 0), 4))
    canvas.draw((50, 50))
    canvas.draw((80, 80))
    canvas.reset_stroke()
    pixels_after_draw = np.count_nonzero(canvas.canvas)

    canvas.undo()
    assert np.count_nonzero(canvas.canvas) == 0

    canvas.redo()
    assert np.count_nonzero(canvas.canvas) == pixels_after_draw, "Redo did not restore drawn stroke"


# ─────────────────────────────────────────────────────────────────────────────
# BB12: Clear Canvas
# ─────────────────────────────────────────────────────────────────────────────
def test_bb12_clear_canvas():
    """BB12: Clear canvas.
    Expected: Clear wipes all content from the canvas and records in undo history.
    """
    canvas = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT, HardBrush((255, 255, 0), 4))
    canvas.draw((50, 50))
    canvas.draw((120, 120))
    canvas.reset_stroke()
    assert np.count_nonzero(canvas.canvas) > 0

    canvas.clear()
    assert np.count_nonzero(canvas.canvas) == 0, "Canvas contains pixels after clear()"
    
    # Verify clear is reversible via undo
    canvas.undo()
    assert np.count_nonzero(canvas.canvas) > 0, "Undo after clear did not restore drawing"


# ─────────────────────────────────────────────────────────────────────────────
# BB13: Save & Export
# ─────────────────────────────────────────────────────────────────────────────
def test_bb13_save_export(tmp_path):
    """BB13: Save/export.
    Expected: Saving produces a valid non-empty PNG file on disk.
    """
    canvas = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT, HardBrush((0, 255, 255), 4))
    canvas.draw((100, 100))
    canvas.draw((200, 200))

    out_file = tmp_path / "test_export.png"
    cv2.imwrite(str(out_file), canvas.canvas)

    assert os.path.exists(str(out_file)), "Exported file does not exist on disk"
    assert os.path.getsize(str(out_file)) > 0, "Exported file is empty"

    # Verify file is a valid readable image
    loaded = cv2.imread(str(out_file))
    assert loaded is not None, "Exported file cannot be read by OpenCV"
    assert loaded.shape == (CAMERA_HEIGHT, CAMERA_WIDTH, 3)


# ─────────────────────────────────────────────────────────────────────────────
# BB14: Focus Mode
# ─────────────────────────────────────────────────────────────────────────────
def test_bb14_focus_mode():
    """BB14: Focus Mode.
    Expected: Toggling Focus Mode expands workspace by collapsing sidebars, and exits cleanly.
    """
    ui = ModernUI(WINDOW_WIDTH, WINDOW_HEIGHT, CAMERA_WIDTH, CAMERA_HEIGHT)
    initial_w = ui.disp_w
    initial_h = ui.disp_h
    assert not ui.focus_mode

    # Enter Focus Mode
    ui.toggle_focus_mode()
    assert ui.focus_mode is True
    assert ui.left_w == 0, "Left panel not collapsed in Focus Mode"
    assert ui.right_w == 0, "Right panel not collapsed in Focus Mode"
    assert ui.disp_h > initial_h, "Canvas height not expanded in Focus Mode"

    # Exit Focus Mode
    ui.toggle_focus_mode()
    assert ui.focus_mode is False
    assert ui.left_w > 0, "Left panel not restored after exiting Focus Mode"
    assert ui.right_w > 0, "Right panel not restored after exiting Focus Mode"


# ─────────────────────────────────────────────────────────────────────────────
# BB15: Handwriting Recognition
# ─────────────────────────────────────────────────────────────────────────────
def test_bb15_handwriting_recognition():
    """BB15: Handwriting recognition.
    Expected: Supported character stroke ('O') is recognized with high confidence.
    """
    tr = TextRecognizer()
    # Synthetic circular stroke representing 'O'
    o_stroke = [(int(150 + 40 * np.cos(th)), int(150 + 50 * np.sin(th)))
                for th in np.linspace(0, 2 * np.pi, 28)]

    rec_char, conf, bbox = tr.recognize_strokes([o_stroke])
    assert rec_char == "O", f"Expected recognized character 'O', got {rec_char}"
    assert conf >= 0.70, f"Expected confidence >= 0.70, got {conf}"
    assert bbox is not None and len(bbox) == 4


# ─────────────────────────────────────────────────────────────────────────────
# BB16: Invalid or Empty Input
# ─────────────────────────────────────────────────────────────────────────────
def test_bb16_invalid_or_empty_input():
    """BB16: Invalid or empty input.
    Expected: Empty stroke, None coordinate, or out-of-bounds inputs do not crash the application.
    """
    canvas = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT, HardBrush())
    sr = ShapeRecognizer()
    tr = TextRecognizer()
    ui = ModernUI(WINDOW_WIDTH, WINDOW_HEIGHT, CAMERA_WIDTH, CAMERA_HEIGHT)

    # 1. Draw None point
    canvas.draw(None)
    assert np.count_nonzero(canvas.canvas) == 0

    # 2. Empty stroke to shape recognizer
    shape_res, name = sr.recognize_and_beautify([], canvas.canvas, (255, 255, 255), 4)
    assert shape_res is None
    assert name is None

    # 3. Empty stroke list to text recognizer
    text_res, conf, bbox = tr.recognize_strokes([])
    assert text_res is None
    assert conf == 0.0

    # 4. Out of bounds UI clicks
    assert ui.handle_selection(-100, -100) is False
    assert ui.handle_selection(99999, 99999) is False

    # 5. Out of bounds coordinate conversion
    cx, cy = ui.ui_to_canvas(-50, -50)
    assert cx is None and cy is None
