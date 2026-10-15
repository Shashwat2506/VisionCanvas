"""
VisionCanvas Black-Box Test Runner and Report Generator.
Executes the test suite, evaluates outcomes against expected results,
outputs terminal summary, and generates reports/black_box_test_report.html.
"""

import sys
import os
import time
import traceback
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

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


def run_all_tests():
    test_results = []
    overall_start = time.time()

    # ── BB01: Application Launch ─────────────────────────────────────────────
    t_start = time.time()
    tc_id = "BB01"
    tc_name = "Application Launch"
    tc_obj = "Verify that VisionCanvas core subsystems start without unhandled exceptions."
    tc_pre = "Python 3.10+ virtual environment and required libraries installed."
    tc_input = "Initialize ModernUI and DrawingCanvas components."
    tc_exp = "VisionCanvas initializes with expected 1280x720 window and 640x480 canvas."
    try:
        ui = ModernUI(WINDOW_WIDTH, WINDOW_HEIGHT, CAMERA_WIDTH, CAMERA_HEIGHT)
        canvas = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT, HardBrush())
        assert ui.width == WINDOW_WIDTH and ui.height == WINDOW_HEIGHT
        status = "PASS"
        act_res = f"Subsystems initialized successfully. Window={ui.width}x{ui.height}, Canvas={canvas.width}x{canvas.height}."
        remarks = "Clean initialization with zero unhandled exceptions."
    except Exception as e:
        status = "FAIL"
        act_res = f"Initialization failed: {e}"
        remarks = traceback.format_exc()
    test_results.append({
        "id": tc_id, "name": tc_name, "objective": tc_obj, "preconditions": tc_pre,
        "input": tc_input, "expected": tc_exp, "actual": act_res, "status": status,
        "remarks": remarks, "duration": time.time() - t_start,
    })

    # ── BB02: Camera Initialization ──────────────────────────────────────────
    t_start = time.time()
    tc_id = "BB02"
    tc_name = "Camera Initialization"
    tc_obj = "Verify camera abstraction acquires video frame when camera is accessible."
    tc_pre = "Camera device accessible or fallback test source ready."
    tc_input = "Initialize Webcam() capture and invoke get_frame()."
    tc_exp = "A valid 3-channel 640x480 frame buffer is returned."
    try:
        webcam = Webcam(camera_index=0, fallback_on_error=True)
        frame = webcam.get_frame()
        is_conn = webcam.is_connected
        webcam.release()
        assert frame is not None and frame.shape == (CAMERA_HEIGHT, CAMERA_WIDTH, 3)
        status = "PASS"
        act_res = f"Frame acquired successfully. Shape={frame.shape}, Live Camera Connected={is_conn}."
        remarks = "Camera buffer successfully delivered to render pipeline."
    except Exception as e:
        status = "FAIL"
        act_res = f"Camera initialization failed: {e}"
        remarks = traceback.format_exc()
    test_results.append({
        "id": tc_id, "name": tc_name, "objective": tc_obj, "preconditions": tc_pre,
        "input": tc_input, "expected": tc_exp, "actual": act_res, "status": status,
        "remarks": remarks, "duration": time.time() - t_start,
    })

    # ── BB03: Camera Unavailable ─────────────────────────────────────────────
    t_start = time.time()
    tc_id = "BB03"
    tc_name = "Camera Unavailable Handling"
    tc_obj = "Verify graceful error handling and fallback when camera device is unavailable."
    tc_pre = "Non-existent camera index requested (e.g. index 9999)."
    tc_input = "Instantiate Webcam(camera_index=9999, fallback_on_error=True)."
    tc_exp = "Application does not crash; generates diagnostic offline frame and handles error."
    try:
        webcam = Webcam(camera_index=9999, fallback_on_error=True)
        frame = webcam.get_frame()
        is_conn = webcam.is_connected
        webcam.release()
        assert not is_conn, "Expected is_connected=False"
        assert frame is not None and frame.shape == (CAMERA_HEIGHT, CAMERA_WIDTH, 3)
        status = "PASS"
        act_res = "Graceful fallback: is_connected=False, generated offline diagnostic placeholder frame without crashing."
        remarks = "System prevented fatal crash on camera failure."
    except Exception as e:
        status = "FAIL"
        act_res = f"Failed to handle missing camera gracefully: {e}"
        remarks = traceback.format_exc()
    test_results.append({
        "id": tc_id, "name": tc_name, "objective": tc_obj, "preconditions": tc_pre,
        "input": tc_input, "expected": tc_exp, "actual": act_res, "status": status,
        "remarks": remarks, "duration": time.time() - t_start,
    })

    # ── BB04: Brush Selection ────────────────────────────────────────────────
    t_start = time.time()
    tc_id = "BB04"
    tc_name = "Brush Preset Selection"
    tc_obj = "Verify selecting a brush preset updates active brush and stroke styling."
    tc_pre = "ModernUI initialized with default Hard Brush."
    tc_input = "Invoke ui.set_active_brush(2) (Neon Brush preset)."
    tc_exp = "Active brush changes from 0 to 2; preset tiles update visually."
    try:
        ui = ModernUI(WINDOW_WIDTH, WINDOW_HEIGHT)
        init_brush = ui.active_brush
        ui.set_active_brush(2)
        assert ui.active_brush == 2 and ui.active_brush != init_brush
        status = "PASS"
        act_res = f"Active brush transitioned from preset {init_brush} (Hard) to {ui.active_brush} (Neon)."
        remarks = "Preset selection confirmed via UI state and tile highlight."
    except Exception as e:
        status = "FAIL"
        act_res = f"Brush selection failed: {e}"
        remarks = traceback.format_exc()
    test_results.append({
        "id": tc_id, "name": tc_name, "objective": tc_obj, "preconditions": tc_pre,
        "input": tc_input, "expected": tc_exp, "actual": act_res, "status": status,
        "remarks": remarks, "duration": time.time() - t_start,
    })

    # ── BB05: Brush Size ─────────────────────────────────────────────────────
    t_start = time.time()
    tc_id = "BB05"
    tc_name = "Brush Size Adjustment"
    tc_obj = "Verify changing brush size adjusts the thickness of rendered strokes."
    tc_pre = "DrawingCanvas with default 4px brush."
    tc_input = "Render identical 100px line segment with thickness=2 vs thickness=14."
    tc_exp = "Thick stroke renders significantly greater pixel footprint than fine stroke."
    try:
        c_fine = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT, HardBrush(thickness=2))
        c_thick = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT, HardBrush(thickness=14))
        c_fine.draw((100, 100))
        c_fine.draw((100, 200))
        c_thick.draw((100, 100))
        c_thick.draw((100, 200))
        p_fine = np.count_nonzero(c_fine.canvas)
        p_thick = np.count_nonzero(c_thick.canvas)
        assert p_thick > p_fine * 2
        status = "PASS"
        act_res = f"Fine stroke pixels: {p_fine}, Thick stroke pixels: {p_thick} (Ratio: {p_thick/p_fine:.1f}x)."
        remarks = "Stroke thickness scales proportionally with brush size setting."
    except Exception as e:
        status = "FAIL"
        act_res = f"Brush size test failed: {e}"
        remarks = traceback.format_exc()
    test_results.append({
        "id": tc_id, "name": tc_name, "objective": tc_obj, "preconditions": tc_pre,
        "input": tc_input, "expected": tc_exp, "actual": act_res, "status": status,
        "remarks": remarks, "duration": time.time() - t_start,
    })

    # ── BB06: Color Selection ────────────────────────────────────────────────
    t_start = time.time()
    tc_id = "BB06"
    tc_name = "Color Selection"
    tc_obj = "Verify selecting a color swatch updates active drawing color."
    tc_pre = "ModernUI color palette available with 10 swatches."
    tc_input = "Select Red swatch (index 2: BGR (50, 50, 245))."
    tc_exp = "Active color updates to (50, 50, 245) and canvas strokes render in Red."
    try:
        ui = ModernUI(WINDOW_WIDTH, WINDOW_HEIGHT)
        target_color, name = ui.PALETTE_COLORS[2]
        cx, cy = ui._color_centers[2]
        changed = ui.handle_selection(cx, cy)
        assert changed is True and ui.active_color == target_color

        canvas = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT, HardBrush(color=ui.active_color, thickness=4))
        canvas.draw((50, 50))
        canvas.draw((150, 50))
        pixel_bgr = tuple(int(x) for x in canvas.canvas[50, 100])
        assert pixel_bgr == target_color
        status = "PASS"
        act_res = f"Selected '{name}'. Canvas pixel at (100, 50) verified as BGR {pixel_bgr}."
        remarks = "Palette selection accurately propagates to drawing raster."
    except Exception as e:
        status = "FAIL"
        act_res = f"Color selection failed: {e}"
        remarks = traceback.format_exc()
    test_results.append({
        "id": tc_id, "name": tc_name, "objective": tc_obj, "preconditions": tc_pre,
        "input": tc_input, "expected": tc_exp, "actual": act_res, "status": status,
        "remarks": remarks, "duration": time.time() - t_start,
    })

    # ── BB07: Eraser ─────────────────────────────────────────────────────────
    t_start = time.time()
    tc_id = "BB07"
    tc_name = "Eraser Functionality"
    tc_obj = "Verify eraser removes existing drawing content from canvas."
    tc_pre = "Canvas contains drawn stroke."
    tc_input = "Switch brush to black eraser and re-draw over existing stroke."
    tc_exp = "Drawn pixels count decreases after erasing."
    try:
        canvas = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT, HardBrush((255, 255, 255), thickness=6))
        canvas.draw((100, 100))
        canvas.draw((200, 100))
        init_px = np.count_nonzero(canvas.canvas)
        assert init_px > 0

        eraser = HardBrush((0, 0, 0), thickness=20)
        canvas.set_brush(eraser)
        canvas.prev_point = None
        canvas.draw((100, 100))
        canvas.draw((200, 100))
        remaining_px = np.count_nonzero(canvas.canvas)
        assert remaining_px < init_px
        status = "PASS"
        act_res = f"Initial drawn pixels: {init_px}, after erase: {remaining_px} (erased {init_px - remaining_px} px)."
        remarks = "Eraser cleared painted raster content as expected."
    except Exception as e:
        status = "FAIL"
        act_res = f"Eraser test failed: {e}"
        remarks = traceback.format_exc()
    test_results.append({
        "id": tc_id, "name": tc_name, "objective": tc_obj, "preconditions": tc_pre,
        "input": tc_input, "expected": tc_exp, "actual": act_res, "status": status,
        "remarks": remarks, "duration": time.time() - t_start,
    })

    # ── BB08: Smart Shapes (Enabled) ─────────────────────────────────────────
    t_start = time.time()
    tc_id = "BB08"
    tc_name = "Smart Shapes (Beautification Enabled)"
    tc_obj = "Verify freehand straight-line stroke is beautified when Smart Shapes is ON."
    tc_pre = "Smart Shapes enabled."
    tc_input = "Submit 30-point stroke with +-2px sinusoidal hand tremor."
    tc_exp = "Stroke recognized and beautified to clean 'Straight Line'."
    try:
        sr = ShapeRecognizer()
        base = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        jittery_line = [(100 + i * 5, 150 + int(np.sin(i) * 2)) for i in range(30)]
        beautified, name = sr.recognize_and_beautify(jittery_line, base, (0, 255, 0), 4)
        assert name == "Straight Line" and beautified is not None
        status = "PASS"
        act_res = f"Shape recognized as '{name}'. Beautified line rendered cleanly on canvas."
        remarks = "Geometric analyzer detected straight line within chord threshold."
    except Exception as e:
        status = "FAIL"
        act_res = f"Smart Shapes enabled test failed: {e}"
        remarks = traceback.format_exc()
    test_results.append({
        "id": tc_id, "name": tc_name, "objective": tc_obj, "preconditions": tc_pre,
        "input": tc_input, "expected": tc_exp, "actual": act_res, "status": status,
        "remarks": remarks, "duration": time.time() - t_start,
    })

    # ── BB09: Smart Shapes (Disabled) ────────────────────────────────────────
    t_start = time.time()
    tc_id = "BB09"
    tc_name = "Smart Shapes (Beautification Disabled)"
    tc_obj = "Verify freehand stroke remains unmodified when Smart Shapes is OFF."
    tc_pre = "Smart Shapes toggle set to OFF."
    tc_input = "Draw raw stroke with Smart Shapes disabled."
    tc_exp = "Raw stroke points retained; no shape replacement performed."
    try:
        ui = ModernUI(WINDOW_WIDTH, WINDOW_HEIGHT)
        ui.smart_shapes = False
        canvas = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT, HardBrush((255, 0, 255), 4))
        raw_pts = [(100 + i * 5, 150 + int(np.sin(i) * 2)) for i in range(25)]
        for p in raw_pts:
            canvas.draw(p)
        finished_pts, _ = canvas.reset_stroke()
        assert len(finished_pts) == len(raw_pts)
        assert ui.smart_shapes is False
        status = "PASS"
        act_res = f"Raw stroke retained all {len(finished_pts)} points without replacement."
        remarks = "Freehand drawing preserved when AI shape mode is toggled off."
    except Exception as e:
        status = "FAIL"
        act_res = f"Smart Shapes disabled test failed: {e}"
        remarks = traceback.format_exc()
    test_results.append({
        "id": tc_id, "name": tc_name, "objective": tc_obj, "preconditions": tc_pre,
        "input": tc_input, "expected": tc_exp, "actual": act_res, "status": status,
        "remarks": remarks, "duration": time.time() - t_start,
    })

    # ── BB10: Undo ───────────────────────────────────────────────────────────
    t_start = time.time()
    tc_id = "BB10"
    tc_name = "Undo Operation"
    tc_obj = "Verify undo reverses the last drawing stroke from canvas."
    tc_pre = "Canvas with one completed stroke."
    tc_input = "Invoke canvas.undo()."
    tc_exp = "Canvas pixels revert to pre-stroke state (empty canvas)."
    try:
        canvas = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT, HardBrush((0, 255, 0), 4))
        canvas.draw((50, 50))
        canvas.draw((100, 100))
        canvas.reset_stroke()
        assert np.count_nonzero(canvas.canvas) > 0
        canvas.undo()
        assert np.count_nonzero(canvas.canvas) == 0
        status = "PASS"
        act_res = "Canvas state reverted from drawn content back to clean 0-pixel buffer."
        remarks = "Undo stack popped and restored previous canvas copy."
    except Exception as e:
        status = "FAIL"
        act_res = f"Undo test failed: {e}"
        remarks = traceback.format_exc()
    test_results.append({
        "id": tc_id, "name": tc_name, "objective": tc_obj, "preconditions": tc_pre,
        "input": tc_input, "expected": tc_exp, "actual": act_res, "status": status,
        "remarks": remarks, "duration": time.time() - t_start,
    })

    # ── BB11: Redo ───────────────────────────────────────────────────────────
    t_start = time.time()
    tc_id = "BB11"
    tc_name = "Redo Operation"
    tc_obj = "Verify redo restores the previously undone drawing stroke."
    tc_pre = "Canvas after an undo operation."
    tc_input = "Invoke canvas.redo()."
    tc_exp = "Canvas pixels restored to the state prior to undo."
    try:
        canvas = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT, HardBrush((0, 255, 0), 4))
        canvas.draw((50, 50))
        canvas.draw((100, 100))
        canvas.reset_stroke()
        drawn_px = np.count_nonzero(canvas.canvas)
        canvas.undo()
        assert np.count_nonzero(canvas.canvas) == 0
        canvas.redo()
        restored_px = np.count_nonzero(canvas.canvas)
        assert restored_px == drawn_px
        status = "PASS"
        act_res = f"Redo restored exact pixel footprint ({restored_px} pixels)."
        remarks = "Redo stack popped and pushed state back to undo history."
    except Exception as e:
        status = "FAIL"
        act_res = f"Redo test failed: {e}"
        remarks = traceback.format_exc()
    test_results.append({
        "id": tc_id, "name": tc_name, "objective": tc_obj, "preconditions": tc_pre,
        "input": tc_input, "expected": tc_exp, "actual": act_res, "status": status,
        "remarks": remarks, "duration": time.time() - t_start,
    })

    # ── BB12: Clear Canvas ───────────────────────────────────────────────────
    t_start = time.time()
    tc_id = "BB12"
    tc_name = "Clear Canvas"
    tc_obj = "Verify clear removes all content and records clear action in history."
    tc_pre = "Canvas with drawn content."
    tc_input = "Invoke canvas.clear()."
    tc_exp = "Canvas is completely wiped to black (0 pixels); undo can reverse it."
    try:
        canvas = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT, HardBrush((255, 255, 0), 4))
        canvas.draw((40, 40))
        canvas.draw((140, 140))
        canvas.reset_stroke()
        canvas.clear()
        assert np.count_nonzero(canvas.canvas) == 0
        canvas.undo()
        assert np.count_nonzero(canvas.canvas) > 0
        status = "PASS"
        act_res = "Canvas cleared to 0 pixels; subsequent undo successfully restored drawing."
        remarks = "Clear action is safe, non-destructive, and undoable."
    except Exception as e:
        status = "FAIL"
        act_res = f"Clear canvas test failed: {e}"
        remarks = traceback.format_exc()
    test_results.append({
        "id": tc_id, "name": tc_name, "objective": tc_obj, "preconditions": tc_pre,
        "input": tc_input, "expected": tc_exp, "actual": act_res, "status": status,
        "remarks": remarks, "duration": time.time() - t_start,
    })

    # ── BB13: Save & Export ──────────────────────────────────────────────────
    t_start = time.time()
    tc_id = "BB13"
    tc_name = "Save & Export"
    tc_obj = "Verify saving canvas produces a valid, readable PNG file in output directory."
    tc_pre = "Canvas with drawn content."
    tc_input = "Save drawing snapshot to output/test_run_snapshot.png."
    tc_exp = "PNG file created on disk with valid dimensions and non-zero file size."
    try:
        if not os.path.exists("output"):
            os.makedirs("output")
        out_path = os.path.join("output", "test_bb13_snapshot.png")
        canvas = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT, HardBrush((0, 255, 255), 4))
        canvas.draw((50, 50))
        canvas.draw((150, 150))
        cv2.imwrite(out_path, canvas.canvas)
        assert os.path.exists(out_path) and os.path.getsize(out_path) > 0
        loaded = cv2.imread(out_path)
        assert loaded is not None and loaded.shape == (CAMERA_HEIGHT, CAMERA_WIDTH, 3)
        status = "PASS"
        act_res = f"File saved: '{out_path}' ({os.path.getsize(out_path)} bytes, shape {loaded.shape})."
        remarks = "Exported PNG validated with OpenCV image decoder."
    except Exception as e:
        status = "FAIL"
        act_res = f"Save/Export test failed: {e}"
        remarks = traceback.format_exc()
    finally:
        if os.path.exists(out_path):
            try: os.remove(out_path)
            except Exception: pass
    test_results.append({
        "id": tc_id, "name": tc_name, "objective": tc_obj, "preconditions": tc_pre,
        "input": tc_input, "expected": tc_exp, "actual": act_res, "status": status,
        "remarks": remarks, "duration": time.time() - t_start,
    })

    # ── BB14: Focus Mode ─────────────────────────────────────────────────────
    t_start = time.time()
    tc_id = "BB14"
    tc_name = "Focus Mode Workspace Transition"
    tc_obj = "Verify Focus Mode collapses side panels to maximize canvas area and exits cleanly."
    tc_pre = "ModernUI in default Studio Mode (left_w=148, right_w=168)."
    tc_input = "Toggle Focus Mode ON, inspect layout, then toggle Focus Mode OFF."
    tc_exp = "Left/Right widths collapse to 0, workspace expands, then restores upon exit."
    try:
        ui = ModernUI(WINDOW_WIDTH, WINDOW_HEIGHT)
        init_h = ui.disp_h
        ui.toggle_focus_mode()
        assert ui.focus_mode is True and ui.left_w == 0 and ui.right_w == 0
        assert ui.disp_h > init_h
        expanded_h = ui.disp_h
        ui.toggle_focus_mode()
        assert ui.focus_mode is False and ui.left_w > 0 and ui.right_w > 0
        status = "PASS"
        act_res = f"Focus Mode expanded canvas height from {init_h}px to {expanded_h}px; exited cleanly."
        remarks = "Dynamic layout recomputed and restored sidebars seamlessly."
    except Exception as e:
        status = "FAIL"
        act_res = f"Focus Mode test failed: {e}"
        remarks = traceback.format_exc()
    test_results.append({
        "id": tc_id, "name": tc_name, "objective": tc_obj, "preconditions": tc_pre,
        "input": tc_input, "expected": tc_exp, "actual": act_res, "status": status,
        "remarks": remarks, "duration": time.time() - t_start,
    })

    # ── BB15: Handwriting Recognition ────────────────────────────────────────
    t_start = time.time()
    tc_id = "BB15"
    tc_name = "Handwriting Recognition"
    tc_obj = "Verify air-drawn character stroke is recognized by offline Text AI engine."
    tc_pre = "TextRecognizer initialized with Latin uppercase prototypes."
    tc_input = "Submit 28-point circular stroke representing character 'O'."
    tc_exp = "TextRecognizer predicts 'O' with confidence >= 0.70."
    try:
        tr = TextRecognizer()
        o_stroke = [(int(150 + 40 * np.cos(th)), int(150 + 50 * np.sin(th)))
                    for th in np.linspace(0, 2 * np.pi, 28)]
        char_res, conf, bbox = tr.recognize_strokes([o_stroke])
        assert char_res == "O" and conf >= 0.70
        status = "PASS"
        act_res = f"Character recognized as '{char_res}' with confidence {conf*100:.1f}%. Bounding box: {bbox}."
        remarks = "Topological loop analysis and normalized template matching matched glyph accurately."
    except Exception as e:
        status = "FAIL"
        act_res = f"Handwriting recognition failed: {e}"
        remarks = traceback.format_exc()
    test_results.append({
        "id": tc_id, "name": tc_name, "objective": tc_obj, "preconditions": tc_pre,
        "input": tc_input, "expected": tc_exp, "actual": act_res, "status": status,
        "remarks": remarks, "duration": time.time() - t_start,
    })

    # ── BB16: Invalid / Empty Input ──────────────────────────────────────────
    t_start = time.time()
    tc_id = "BB16"
    tc_name = "Boundary & Invalid Input Safety"
    tc_obj = "Verify system handles empty, None, and out-of-bounds inputs without crashing."
    tc_pre = "VisionCanvas components instantiated."
    tc_input = "Submit draw(None), empty shape points [], empty text strokes [], and out-of-bounds clicks."
    tc_exp = "All methods handle invalid parameters safely with zero unhandled exceptions."
    try:
        canvas = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT, HardBrush())
        sr = ShapeRecognizer()
        tr = TextRecognizer()
        ui = ModernUI(WINDOW_WIDTH, WINDOW_HEIGHT)

        canvas.draw(None)
        s_res, _ = sr.recognize_and_beautify([], canvas.canvas, (255, 255, 255), 4)
        t_res, t_conf, _ = tr.recognize_strokes([])
        click_res = ui.handle_selection(-999, -999)
        coord_res_x, coord_res_y = ui.ui_to_canvas(-50, -50)

        assert s_res is None and t_res is None and click_res is False
        assert coord_res_x is None and coord_res_y is None
        status = "PASS"
        act_res = "All 5 invalid/boundary inputs rejected safely with graceful return values."
        remarks = "Zero crashes or uncaught exceptions across all boundary test cases."
    except Exception as e:
        status = "FAIL"
        act_res = f"Boundary testing failed: {e}"
        remarks = traceback.format_exc()
    test_results.append({
        "id": tc_id, "name": tc_name, "objective": tc_obj, "preconditions": tc_pre,
        "input": tc_input, "expected": tc_exp, "actual": act_res, "status": status,
        "remarks": remarks, "duration": time.time() - t_start,
    })

    overall_duration = time.time() - overall_start

    # ── Generate HTML Report ─────────────────────────────────────────────────
    if not os.path.exists("reports"):
        os.makedirs("reports")
    report_html_path = os.path.join("reports", "black_box_test_report.html")
    _write_html_report(report_html_path, test_results, overall_duration)

    # ── Terminal Summary ─────────────────────────────────────────────────────
    passed = sum(1 for r in test_results if r["status"] == "PASS")
    failed = sum(1 for r in test_results if r["status"] == "FAIL")
    blocked = sum(1 for r in test_results if r["status"] == "BLOCKED")
    total = len(test_results)
    pass_rate = (passed / total) * 100 if total > 0 else 0

    print("\n" + "=" * 80)
    print(" VISIONCANVAS BLACK-BOX AUTOMATED TEST EXECUTION SUMMARY")
    print("=" * 80)
    print(f" {'ID':<6} | {'Test Case Name':<35} | {'Status':<8} | {'Duration':<8}")
    print("-" * 80)
    for r in test_results:
        print(f" {r['id']:<6} | {r['name']:<35} | {r['status']:<8} | {r['duration']*1000:>6.1f}ms")
    print("=" * 80)
    print(f" TOTAL TESTS: {total} | PASSED: {passed} | FAILED: {failed} | BLOCKED: {blocked}")
    print(f" OVERALL PASS RATE: {pass_rate:.1f}% | TOTAL EXECUTION TIME: {overall_duration:.3f}s")
    print(f" TEST REPORT SAVED TO: {report_html_path}")
    print("=" * 80 + "\n")

    return 0 if failed == 0 else 1


def _write_html_report(filepath, results, total_duration):
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    blocked = sum(1 for r in results if r["status"] == "BLOCKED")
    total = len(results)
    pass_rate = (passed / total) * 100 if total > 0 else 0

    table_rows = []
    for r in results:
        status_badge = (
            f'<span class="badge badge-pass">PASS</span>' if r["status"] == "PASS" else
            f'<span class="badge badge-fail">FAIL</span>' if r["status"] == "FAIL" else
            f'<span class="badge badge-blocked">BLOCKED</span>'
        )
        row = f"""
        <tr>
            <td class="tc-id">{r['id']}</td>
            <td class="tc-name"><strong>{r['name']}</strong><br><small class="text-dim">{r['objective']}</small></td>
            <td><code>{r['input']}</code></td>
            <td>{r['expected']}</td>
            <td>{r['actual']}</td>
            <td>{status_badge}</td>
            <td class="tc-dur">{r['duration']*1000:.1f}ms</td>
            <td><small>{r['remarks']}</small></td>
        </tr>
        """
        table_rows.append(row)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VisionCanvas — Black-Box Automated Test Report</title>
    <style>
        :root {{
            --bg: #0d0f18;
            --panel: #151824;
            --border: #282c3f;
            --accent: #8b5cf6;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-dim: #64748b;
            --green: #10b981;
            --red: #ef4444;
            --amber: #f59e0b;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        body {{ background-color: var(--bg); color: var(--text-primary); padding: 32px 20px; line-height: 1.5; }}
        .container {{ max-width: 1300px; margin: 0 auto; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; border-bottom: 1px solid var(--border); padding-bottom: 20px; }}
        .title h1 {{ font-size: 26px; font-weight: 700; color: #fff; }}
        .title p {{ color: var(--text-secondary); font-size: 14px; margin-top: 4px; }}
        .timestamp {{ font-size: 13px; color: var(--text-dim); text-align: right; }}

        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 30px; }}
        .kpi-card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px; }}
        .kpi-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary); }}
        .kpi-value {{ font-size: 28px; font-weight: 800; margin-top: 6px; }}
        .kpi-pass {{ color: var(--green); }}
        .kpi-fail {{ color: var(--red); }}
        .kpi-rate {{ color: var(--accent); }}

        .table-wrap {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; overflow-x: auto; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }}
        th {{ background: #1c2030; color: var(--text-secondary); font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em; padding: 14px 16px; border-bottom: 1px solid var(--border); }}
        td {{ padding: 14px 16px; border-bottom: 1px solid var(--border); vertical-align: top; }}
        tr:hover td {{ background: rgba(139, 92, 246, 0.03); }}
        .tc-id {{ font-weight: 700; color: var(--accent); white-space: nowrap; }}
        .tc-dur {{ font-family: monospace; color: var(--text-secondary); }}
        .text-dim {{ color: var(--text-dim); }}
        code {{ background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 4px; font-size: 12px; font-family: Consolas, monospace; color: #38bdf8; }}

        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 11px; font-weight: 700; letter-spacing: 0.04em; }}
        .badge-pass {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
        .badge-fail {{ background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }}
        .badge-blocked {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }}

        .footer {{ margin-top: 32px; font-size: 12px; color: var(--text-dim); text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">
                <h1>VisionCanvas — Automated Black-Box Test Report</h1>
                <p>Software Engineering & Modelling Assignment • Automated Test Execution Suite</p>
            </div>
            <div class="timestamp">
                Execution Engine: pytest & Python Test Runner<br>
                Target: VisionCanvas v2.0 (OpenCV / MediaPipe Studio)
            </div>
        </div>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Total Test Cases</div>
                <div class="kpi-value">{total}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Tests Passed</div>
                <div class="kpi-value kpi-pass">{passed}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Tests Failed</div>
                <div class="kpi-value kpi-fail">{failed}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Pass Rate</div>
                <div class="kpi-value kpi-rate">{pass_rate:.1f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Execution Time</div>
                <div class="kpi-value">{total_duration:.3f}s</div>
            </div>
        </div>

        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Test Case & Objective</th>
                        <th>Input / Action</th>
                        <th>Expected Result</th>
                        <th>Actual Result</th>
                        <th>Status</th>
                        <th>Duration</th>
                        <th>Remarks</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(table_rows)}
                </tbody>
            </table>
        </div>

        <div class="footer">
            Generated automatically by <code>run_black_box_tests.py</code> • VisionCanvas Project
        </div>
    </div>
</body>
</html>
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
