import cv2
import time
import os
import threading
import numpy as np
from src.config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, CAMERA_WIDTH, CAMERA_HEIGHT,
    GESTURE_DEBOUNCE_TIME, DEBUG_PERFORMANCE, BRUSH_DEFAULT_SIZE,
)
from src.camera.webcam import Webcam
from src.hand_tracking.detector import HandDetector
from src.hand_tracking.finger_detector import FingerDetector
from src.gestures.gesture_recognizer import GestureRecognizer
from src.gestures.cursor import VirtualCursor
from src.canvas.drawing_canvas import DrawingCanvas
from src.canvas.shape_recognizer import ShapeRecognizer
from src.canvas.text_recognizer import TextRecognizer
from src.ui.modern_ui import ModernUI
from src.brushes.hard_brush import HardBrush
from src.brushes.soft_brush import SoftBrush
from src.brushes.neon_brush import NeonBrush
from src.brushes.pencil_brush import PencilBrush
from src.brushes.watercolor_brush import WatercolorBrush


# Brush classes indexed to match UI brush preset order
BRUSH_CLASSES = [HardBrush, SoftBrush, NeonBrush, PencilBrush, WatercolorBrush]


def main():
    # ── Initialise Core Components ──────────────────────────────
    webcam = Webcam(fallback_on_error=True)
    detector = HandDetector()
    finger_detector = FingerDetector()
    gesture_recognizer = GestureRecognizer()
    cursor = VirtualCursor()
    shape_recognizer = ShapeRecognizer()
    text_recognizer = TextRecognizer()

    ui = ModernUI(
        window_width=WINDOW_WIDTH,
        window_height=WINDOW_HEIGHT,
        video_width=CAMERA_WIDTH,
        video_height=CAMERA_HEIGHT,
    )

    if not webcam.is_connected:
        ui.show_toast("Notice: Camera offline (Canvas Studio active)", duration=5.0)


    # Brush & Tool State
    brush_size = ui.brush_size
    active_brush_idx = 0
    brush = HardBrush(color=ui.active_color, thickness=brush_size)
    eraser_brush = HardBrush(color=(0, 0, 0), thickness=max(6, brush_size * 2))

    canvas = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT, brush)
    last_action_time = 0.0

    # Gesture confirmation to prevent accidental dots
    draw_frame_count = 0
    was_drawing = False

    # Text recognition state
    text_strokes = []
    text_start_canvas = None
    last_text_stroke_time = 0.0

    cv2.namedWindow("VisionCanvas", cv2.WINDOW_NORMAL)

    # ── Background inference thread ─────────────────────────────
    _inference_running = [True]

    def _inference_loop():
        while _inference_running[0]:
            frame = webcam.get_frame()
            if frame is None:
                time.sleep(0.001)
                continue
            frame = cv2.flip(frame, 1)
            detector.detect(frame)

    inference_thread = threading.Thread(target=_inference_loop, daemon=True)
    inference_thread.start()

    # ── Helpers ─────────────────────────────────────────────────

    def _rebuild_brush():
        """Recreate the active brush with current color and size."""
        nonlocal brush, eraser_brush
        cls = BRUSH_CLASSES[active_brush_idx] if active_brush_idx < len(BRUSH_CLASSES) else HardBrush
        brush = cls(color=ui.active_color, thickness=brush_size)
        eraser_brush = HardBrush(color=(0, 0, 0), thickness=max(8, brush_size * 2))
        if ui.active_tool in (0, 2, 3):  # Brush, Shapes, Text
            canvas.set_brush(brush)
        elif ui.active_tool == 1:       # Eraser
            canvas.set_brush(eraser_brush)

    def _set_tool(idx):
        """Switch active tool."""
        ui.set_active_tool(idx)
        _rebuild_brush()

    def _set_brush_idx(idx):
        """Switch brush preset."""
        nonlocal active_brush_idx
        active_brush_idx = idx
        ui.set_active_brush(idx)
        _rebuild_brush()

    def _change_brush_size(delta):
        """Adjust brush size by delta."""
        nonlocal brush_size
        brush_size = max(1, min(40, brush_size + delta))
        ui.set_brush_size(brush_size)
        _rebuild_brush()

    def _save_canvas_snapshot():
        """Export drawing canvas to output directory."""
        if not os.path.exists("output"):
            os.makedirs("output")
        timestamp = int(time.time())
        filename = f"output/canvas_{timestamp}.png"
        cv2.imwrite(filename, canvas.canvas)
        ui.show_toast(f"Saved: {filename}")

    def _finish_stroke_and_recognize():
        """Handle stroke termination and smart beautification."""
        nonlocal text_strokes, text_start_canvas, last_text_stroke_time
        pts, start_canvas = canvas.reset_stroke()

        if not pts or len(pts) < 4:
            return

        # Text AI Mode Handling
        if ui.active_mode == 3 or ui.active_tool == 3:
            if not text_strokes:
                text_start_canvas = start_canvas.copy() if start_canvas is not None else canvas.canvas.copy()
            text_strokes.append(pts)
            last_text_stroke_time = time.time()
            ui.show_toast(f"✍️ Recorded stroke {len(text_strokes)} — pause or hold ✌️ to recognize", duration=1.5)
            return

        # Smart Shapes Beautification
        if ui.smart_shapes and start_canvas is not None and len(pts) >= 6:
            beautified, shape_name = shape_recognizer.recognize_and_beautify(
                pts, start_canvas, ui.active_color, brush_size
            )
            if beautified is not None:
                canvas.replace_canvas(beautified)
                ui.show_toast(f"✨ Beautified: {shape_name}")

    def _trigger_text_recognition():
        """Perform handwriting recognition on collected strokes."""
        nonlocal text_strokes, text_start_canvas
        if text_strokes and ui.text_review is None and not was_drawing:
            rec_text, conf, bbox = text_recognizer.recognize_strokes(text_strokes)
            if rec_text and bbox is not None:
                ui.text_review = {
                    "text": rec_text,
                    "conf": conf,
                    "bbox": bbox,
                    "start_canvas": text_start_canvas.copy() if text_start_canvas is not None else canvas.canvas.copy()
                }
                ui.show_toast(f"Recognized: '{rec_text}' ({int(conf * 100)}%) — Review above")
            else:
                ui.show_toast("Recognition low confidence — write again")
                _handle_text_retry()

    def _handle_text_accept():
        """Accept recognized text and render clean digital font on canvas."""
        nonlocal text_strokes, text_start_canvas
        if ui.text_review is not None:
            rec_text = ui.text_review.get("text", "")
            bbox = ui.text_review.get("bbox", None)
            base_img = ui.text_review.get("start_canvas", canvas.canvas)
            if rec_text and bbox:
                font_sz = None if ui.text_auto_size else ui.text_font_size
                rendered = text_recognizer.render_text_to_canvas(
                    base_img,
                    rec_text,
                    bbox,
                    ui.active_color,
                    font_family=ui.get_active_font_family(),
                    font_size=font_sz,
                    bold=ui.text_bold,
                    italic=ui.text_italic,
                )
                canvas.commit_operation(rendered, base_canvas=base_img)
                ui.show_toast(f"✨ Typeset: '{rec_text}' ({ui.get_active_font_family()})")
        text_strokes = []
        text_start_canvas = None
        ui.text_review = None

    def _handle_text_retry():
        """Clear text strokes and retry writing."""
        nonlocal text_strokes, text_start_canvas
        if ui.text_review is not None:
            base_img = ui.text_review.get("start_canvas", None)
            if base_img is not None:
                canvas.revert_to(base_img)
        elif text_start_canvas is not None:
            canvas.revert_to(text_start_canvas)
        text_strokes = []
        text_start_canvas = None
        ui.text_review = None
        ui.show_toast("↺ Ready — write again!")

    def _handle_text_dismiss():
        """Cancel text conversion and cleanly restore previous canvas state."""
        nonlocal text_strokes, text_start_canvas
        if ui.text_review is not None:
            base_img = ui.text_review.get("start_canvas", None)
            if base_img is not None:
                canvas.revert_to(base_img)
        elif text_start_canvas is not None:
            canvas.revert_to(text_start_canvas)
        text_strokes = []
        text_start_canvas = None
        ui.text_review = None
        ui.show_toast("Text conversion cancelled")

    # ── Performance counters ────────────────────────────────────
    frame_count = 0
    fps_timer = time.time()
    render_fps = 0.0

    # ── Main render loop ────────────────────────────────────────
    while True:
        loop_start = time.time()

        # Get latest inference result (non-blocking)
        result, inf_frame = detector.get_latest()

        if inf_frame is not None:
            frame = inf_frame
        else:
            frame = webcam.get_frame()
            if frame is None:
                break
            frame = cv2.flip(frame, 1)

        # ── Process Hand Tracking Result ────────────────────────
        cursor_x, cursor_y = 0, 0
        gesture = "NONE"
        landmarks = None

        if result is not None and result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            fingers = finger_detector.get_finger_states(landmarks)
            gesture = gesture_recognizer.recognize(fingers)

            index_tip = landmarks[8]

            # ── Dual-Mode Coordinate Mapping ────────────────────
            # 1. UI Interaction Mapping: reaches all panels smoothly across full window
            # 2. Drawing Mapping: 1:1 physical alignment with webcam & canvas
            is_select_mode = (gesture == "SELECT" or ui.active_mode == 1)

            if is_select_mode:
                # Comfortable range (10% - 90%) expanded to full window dimensions
                norm_x = np.clip((index_tip.x - 0.10) / 0.80, 0.0, 1.0)
                norm_y = np.clip((index_tip.y - 0.10) / 0.80, 0.0, 1.0)
                screen_x = norm_x * ui.width
                screen_y = norm_y * ui.height
            else:
                # 1:1 direct mapping to canvas position in workspace
                screen_x = ui.video_x + np.clip(index_tip.x, 0.0, 1.0) * ui.disp_w
                screen_y = ui.video_y + np.clip(index_tip.y, 0.0, 1.0) * ui.disp_h

            class MappedLandmark:
                pass
            mapped = MappedLandmark()
            mapped.x = screen_x / ui.width
            mapped.y = screen_y / ui.height

            cx, cy = cursor.update(mapped, ui.width, ui.height)
            cursor_x, cursor_y = cx, cy

            # Check if cursor is over UI panel
            over_ui = ui.is_over_ui(cursor_x, cursor_y)

            # ── Gesture Dispatches ──────────────────────────────
            if gesture == "SELECT":
                draw_frame_count = 0
                if was_drawing:
                    _finish_stroke_and_recognize()
                    was_drawing = False

                # If strokes were drawn in Text AI mode, trigger recognition on SELECT gesture
                if text_strokes and ui.text_review is None:
                    _trigger_text_recognition()

                sel_res = ui.handle_selection(cursor_x, cursor_y)
                if sel_res == "TEXT_ACCEPT":
                    _handle_text_accept()
                elif sel_res == "TEXT_RETRY":
                    _handle_text_retry()
                elif sel_res in ("TEXT_CANCEL", "TEXT_DISMISS"):
                    _handle_text_dismiss()
                elif sel_res == "CLEAR":
                    canvas.clear()
                    ui.show_toast("Canvas Cleared")
                elif sel_res == "UNDO":
                    canvas.undo()
                elif sel_res == "REDO":
                    canvas.redo()
                elif sel_res == "SAVE":
                    _save_canvas_snapshot()
                elif sel_res == "EXPORT":
                    _save_canvas_snapshot()
                elif sel_res is True:
                    _rebuild_brush()

            elif gesture == "DRAW":
                if over_ui:
                    # Accidental drawing prevented when over UI controls
                    draw_frame_count = 0
                    if was_drawing:
                        _finish_stroke_and_recognize()
                        was_drawing = False
                else:
                    draw_frame_count += 1
                    # Require 2 consecutive frames of DRAW gesture to prevent noise dots
                    if draw_frame_count >= 2:
                        canv_x, canv_y = ui.ui_to_canvas(cursor_x, cursor_y)
                        if canv_x is not None and canv_y is not None:
                            canvas.draw((canv_x, canv_y))
                            was_drawing = True
            else:
                draw_frame_count = 0
                if was_drawing:
                    _finish_stroke_and_recognize()
                    was_drawing = False

            # Action gestures (Debounced)
            current_time = time.time()
            if current_time - last_action_time > GESTURE_DEBOUNCE_TIME:
                if gesture == "UNDO":
                    canvas.undo()
                    last_action_time = current_time
                    ui.show_toast("Undo")
                elif gesture == "REDO":
                    canvas.redo()
                    last_action_time = current_time
                    ui.show_toast("Redo")
                elif gesture == "SAVE":
                    _save_canvas_snapshot()
                    last_action_time = current_time
        else:
            # Tracking lost
            cursor.reset()
            draw_frame_count = 0
            if was_drawing:
                _finish_stroke_and_recognize()
                was_drawing = False

        # ── Text Recognition Auto-Trigger on Pause ──────────────
        if (text_strokes and ui.text_review is None and
                time.time() - last_text_stroke_time > 1.8 and not was_drawing):
            _trigger_text_recognition()

        # ── Render Frame ────────────────────────────────────────
        output = ui.render(
            frame, canvas.canvas, cursor_x, cursor_y, gesture, landmarks=landmarks
        )

        # ── Performance overlay ─────────────────────────────────
        if DEBUG_PERFORMANCE:
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                render_fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.time()
            loop_ms = (time.time() - loop_start) * 1000
            cv2.putText(
                output,
                f"FPS: {render_fps:.1f} | Latency: {loop_ms:.1f}ms",
                (14, output.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1, cv2.LINE_AA
            )

        cv2.imshow("VisionCanvas", output)

        # ── Key Handling ────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # q or Esc
            if ui.text_review is not None:
                _handle_text_dismiss()
            else:
                break
        elif key in (13, 10):  # Enter: Accept text recognition or trigger recognition
            if ui.text_review is not None:
                _handle_text_accept()
            elif text_strokes:
                _trigger_text_recognition()
        elif key == 32:  # Space: Trigger recognition on pending strokes
            if text_strokes and ui.text_review is None:
                _trigger_text_recognition()
        elif key == ord('r'):
            if ui.text_review is not None:
                _handle_text_retry()
            else:
                canvas.redo()
        # Tool switching & typography shortcuts
        elif key == ord('b'):
            if ui.active_mode == 3 or ui.active_tool == 3:
                ui.toggle_text_bold()
            else:
                _set_tool(0)  # Brush
        elif key == ord('i'):
            if ui.active_mode == 3 or ui.active_tool == 3:
                ui.toggle_text_italic()
        elif key == ord('e'):
            _set_tool(1)  # Eraser
        elif key == ord('s'):
            ui.toggle_smart_shapes()
        elif key == ord('f'):
            ui.toggle_focus_mode()
        elif key == ord('t'):
            _set_tool(3)  # Text AI
        # Brush presets (1-5)
        elif ord('1') <= key <= ord('5'):
            _set_brush_idx(key - ord('1'))
        # Brush size
        elif key == ord('['):
            _change_brush_size(-2)
        elif key == ord(']'):
            _change_brush_size(2)
        # Canvas operations
        elif key == ord('c'):
            canvas.clear()
            ui.show_toast("Canvas Cleared")
        elif key == ord('z'):
            canvas.undo()
        elif key == ord('y'):
            canvas.redo()
        elif key == ord('x'):
            _save_canvas_snapshot()

    # ── Cleanup ─────────────────────────────────────────────────
    _inference_running[0] = False
    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()