import cv2
import time
import os
import threading
from src.config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, CAMERA_WIDTH, CAMERA_HEIGHT,
    GESTURE_DEBOUNCE_TIME, DEBUG_PERFORMANCE,
)
from src.camera.webcam import Webcam
from src.hand_tracking.detector import HandDetector
from src.hand_tracking.finger_detector import FingerDetector
from src.gestures.gesture_recognizer import GestureRecognizer
from src.gestures.cursor import VirtualCursor
from src.canvas.drawing_canvas import DrawingCanvas
from src.ui.modern_ui import ModernUI
from src.brushes.hard_brush import HardBrush
from src.brushes.soft_brush import SoftBrush
from src.brushes.neon_brush import NeonBrush
from src.brushes.pencil_brush import PencilBrush
from src.brushes.watercolor_brush import WatercolorBrush


# Brush classes indexed to match UI brush preset order
BRUSH_CLASSES = [HardBrush, SoftBrush, NeonBrush, PencilBrush, WatercolorBrush]


def main():
    # ── Initialise components ───────────────────────────────────
    webcam = Webcam()
    detector = HandDetector()
    finger_detector = FingerDetector()
    gesture_recognizer = GestureRecognizer()
    cursor = VirtualCursor()
    ui = ModernUI(
        window_width=WINDOW_WIDTH,
        window_height=WINDOW_HEIGHT,
        video_width=CAMERA_WIDTH,
        video_height=CAMERA_HEIGHT,
    )

    # Brush state
    brush_size = ui.brush_size
    active_brush_idx = 0
    brush = HardBrush(color=ui.active_color, thickness=brush_size)
    eraser_brush = HardBrush(color=(0, 0, 0), thickness=brush_size * 2)

    canvas = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT, brush)
    last_action_time = 0.0

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
        eraser_brush = HardBrush(color=(0, 0, 0), thickness=max(10, brush_size * 2))
        # Only set canvas brush if we're in Brush mode
        if ui.active_tool == 0:
            canvas.set_brush(brush)
        elif ui.active_tool == 1:
            canvas.set_brush(eraser_brush)

    def _set_tool(idx):
        """Switch active tool."""
        ui.set_active_tool(idx)
        if idx == 0:
            canvas.set_brush(brush)
        elif idx == 1:
            canvas.set_brush(eraser_brush)

    def _set_brush_idx(idx):
        """Switch brush preset."""
        nonlocal active_brush_idx
        active_brush_idx = idx
        ui.set_active_brush(idx)
        _rebuild_brush()

    def _change_brush_size(delta):
        """Adjust brush size by delta."""
        nonlocal brush_size
        brush_size = max(1, min(100, brush_size + delta))
        ui.set_brush_size(brush_size)
        _rebuild_brush()

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

        h, w = frame.shape[:2]

        # ── Process hand tracking result ────────────────────────
        cursor_x, cursor_y = 0, 0
        gesture = "NONE"
        landmarks = None

        if result is not None and result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            fingers = finger_detector.get_finger_states(landmarks)
            gesture = gesture_recognizer.recognize(fingers)

            index_tip = landmarks[8]
            
            # Map x-coordinate
            x = index_tip.x
            margin_left, margin_right = 0.10, 0.90
            if x < margin_left:
                screen_x = (x / margin_left) * ui.video_x
            elif x > margin_right:
                screen_x = ui.video_x + ui.disp_w + ((x - margin_right) / (1.0 - margin_right)) * (ui.width - (ui.video_x + ui.disp_w))
            else:
                # 1:1 mapping in the center
                screen_x = ui.video_x + ((x - margin_left) / (margin_right - margin_left)) * ui.disp_w
                
            # Map y-coordinate
            y = index_tip.y
            margin_top, margin_bottom = 0.10, 0.90
            if y < margin_top:
                screen_y = (y / margin_top) * ui.video_y
            elif y > margin_bottom:
                screen_y = ui.video_y + ui.disp_h + ((y - margin_bottom) / (1.0 - margin_bottom)) * (ui.height - (ui.video_y + ui.disp_h))
            else:
                # 1:1 mapping in the center
                screen_y = ui.video_y + ((y - margin_top) / (margin_bottom - margin_top)) * ui.disp_h

            class MappedLandmark: pass
            mapped = MappedLandmark()
            mapped.x = screen_x / ui.width
            mapped.y = screen_y / ui.height

            cx, cy = cursor.update(mapped, ui.width, ui.height)
            cursor_x, cursor_y = cx, cy

            if gesture == "SELECT":
                if ui.handle_selection(cursor_x, cursor_y):
                    _rebuild_brush()
                if ui.handle_brush_size_selection(cursor_x, cursor_y):
                    _rebuild_brush()
                canvas.reset_stroke()
            elif gesture == "DRAW":
                # Only draw/erase when Brush or Eraser tool is active
                if ui.active_tool in (0, 1):
                    cx, cy = ui.ui_to_canvas(cursor_x, cursor_y)
                    if cx is not None and cy is not None:
                        canvas.draw((cx, cy))
                    else:
                        canvas.reset_stroke()
                else:
                    canvas.reset_stroke()
            else:
                canvas.reset_stroke()

            current_time = time.time()
            if current_time - last_action_time > GESTURE_DEBOUNCE_TIME:
                if gesture == "UNDO":
                    canvas.undo()
                    last_action_time = current_time
                elif gesture == "REDO":
                    canvas.redo()
                    last_action_time = current_time
                elif gesture == "SAVE":
                    if not os.path.exists("output"):
                        os.makedirs("output")
                    timestamp = int(time.time())
                    cv2.imwrite(f"output/canvas_{timestamp}.png", canvas.canvas)
                    last_action_time = current_time
        else:
            canvas.reset_stroke()

        # ── Render ──────────────────────────────────────────────
        output = ui.render(frame, canvas.canvas, cursor_x, cursor_y, gesture,
                           landmarks=landmarks)

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
                f"Render FPS: {render_fps:.1f}  Loop: {loop_ms:.1f}ms",
                (10, output.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

        cv2.imshow("VisionCanvas", output)

        # ── Key handling ────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # Tool switching
        elif key == ord('b'):
            _set_tool(0)  # Brush
        elif key == ord('e'):
            _set_tool(1)  # Eraser
        # Brush presets (1-5)
        elif ord('1') <= key <= ord('5'):
            _set_brush_idx(key - ord('1'))
        # Brush size
        elif key == ord('['):
            _change_brush_size(-4)
        elif key == ord(']'):
            _change_brush_size(4)
        # Canvas operations
        elif key == ord('c'):
            canvas.clear()
        elif key == ord('z'):
            canvas.undo()
        elif key == ord('y'):
            canvas.redo()
        elif key == ord('s'):
            if not os.path.exists("output"):
                os.makedirs("output")
            timestamp = int(time.time())
            cv2.imwrite(f"output/canvas_{timestamp}.png", canvas.canvas)

    # ── Cleanup ─────────────────────────────────────────────────
    _inference_running[0] = False
    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()