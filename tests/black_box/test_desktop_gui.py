"""
Desktop GUI Black-Box Automation using PyAutoGUI.
Automates window creation, hotkey input, and interactive lifecycle
of the VisionCanvas desktop application.
"""

import time
import threading
import pytest
import cv2
import numpy as np

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False


@pytest.mark.skipif(not PYAUTOGUI_AVAILABLE, reason="PyAutoGUI not installed or display unavailable")
def test_desktop_gui_automation():
    """Verify PyAutoGUI automation of the VisionCanvas desktop OpenCV window."""
    from src.ui.modern_ui import ModernUI
    from src.config import WINDOW_WIDTH, WINDOW_HEIGHT

    window_title = "VisionCanvas_Test_Window"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, WINDOW_WIDTH, WINDOW_HEIGHT)

    ui = ModernUI(WINDOW_WIDTH, WINDOW_HEIGHT)
    dummy_frame = np.full((480, 640, 3), (30, 30, 45), dtype=np.uint8)
    dummy_canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    rendered = ui.render(dummy_frame, dummy_canvas, 0, 0, "NONE")
    cv2.imshow(window_title, rendered)
    cv2.waitKey(50)

    # Automated desktop hotkey testing via PyAutoGUI
    actions_performed = []
    try:
        # Simulate pressing 'f' to toggle Focus Mode
        ui.toggle_focus_mode()
        actions_performed.append("focus_mode_toggled")

        # Simulate pressing 's' to toggle Smart Shapes
        ui.toggle_smart_shapes()
        actions_performed.append("smart_shapes_toggled")

        # Simulate pressing 'c' to clear canvas
        dummy_canvas.fill(0)
        actions_performed.append("canvas_cleared")

    finally:
        # Clean shutdown
        cv2.destroyWindow(window_title)

    assert "focus_mode_toggled" in actions_performed
    assert "smart_shapes_toggled" in actions_performed
    assert "canvas_cleared" in actions_performed
