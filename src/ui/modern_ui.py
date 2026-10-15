import cv2
import numpy as np
import math
import time
from PIL import Image, ImageDraw, ImageFont
from src.canvas.font_manager import FontManager
from src.config import (
    DRAW_HAND_LANDMARKS,
    LANDMARK_COLOR, LANDMARK_CONNECTION_COLOR,
    LANDMARK_DOT_RADIUS, LANDMARK_LINE_THICKNESS,
    BRUSH_DEFAULT_SIZE, BRUSH_SIZES, SMART_SHAPES_DEFAULT,
)


class ModernUI:
    """Cohesive, premium creative studio UI inspired by Apple Freeform & Microsoft Whiteboard.

    Features:
    - Significantly expanded drawing workspace (>45% larger)
    - Dedicated Focus / Canvas Mode for maximum drawing area
    - High-contrast, precise dual-ring drawing cursor
    - Full-window UI interaction pointer
    - Accurate color wheel, SV picker, and curated palette
    - Smart Shapes toggle and Text AI recognition review controls
    - Non-intrusive collapsible gesture guide
    """

    # ── Color Palette ───────────────────────────────────────────
    BG = (12, 13, 20)                 # Deep charcoal / navy base
    PANEL = (19, 21, 32)              # Studio dark panel
    PANEL_BORDER = (38, 42, 62)       # Crisp subtle panel border
    PANEL_HOVER = (30, 34, 52)        # Interactive hover
    ACCENT = (140, 80, 255)           # Electric violet primary
    ACCENT_GLOW = (180, 130, 255)     # Glowing active violet
    ACCENT_DIM = (65, 40, 125)        # Subdued violet pill
    NEON_CYAN = (0, 225, 255)         # High-contrast cyan pointer
    LIVE_GREEN = (40, 215, 110)       # Status active green
    AMBER = (245, 166, 35)            # Idle / warning amber
    TEXT_PRIMARY = (240, 240, 250)    # Crisp white text
    TEXT_SECONDARY = (145, 150, 175)  # Muted secondary text
    TEXT_DIM = (85, 90, 115)          # Micro label text

    # Curated drawing palette (BGR)
    PALETTE_COLORS = [
        ((255, 255, 255), "White"),
        ((145, 150, 165), "Silver"),
        ((50, 50, 245), "Red"),
        ((30, 140, 255), "Orange"),
        ((0, 225, 255), "Yellow"),
        ((75, 215, 75), "Green"),
        ((255, 220, 0), "Cyan"),
        ((255, 110, 45), "Blue"),
        ((180, 80, 255), "Purple"),
        ((225, 60, 235), "Pink"),
    ]

    TOOLS = [
        ("Brush", "B"),
        ("Eraser", "E"),
        ("Smart Shapes", "S"),
        ("Text AI", "T"),
        ("Clear", "C"),
    ]

    MODES = ["Draw", "Select", "Erase", "Text AI"]

    BRUSH_NAMES = ["Hard", "Soft", "Neon", "Pencil", "Water"]

    PRESET_SIZES = BRUSH_SIZES  # [2, 4, 8, 14, 22]

    # MediaPipe hand skeleton connections
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
        (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (0, 17),                                 # Palm base
    ]

    def __init__(self, window_width: int = 1280, window_height: int = 720,
                 video_width: int = 640, video_height: int = 480):
        self.width = window_width
        self.height = window_height
        self.video_width = video_width
        self.video_height = video_height

        # Application state
        self.active_tool = 0
        self.active_mode = 0
        self.active_color_idx = 8  # Violet Purple default
        self.active_color = self.PALETTE_COLORS[self.active_color_idx][0]
        self.active_brush = 0
        self.brush_size = BRUSH_DEFAULT_SIZE  # 4px
        self.smart_shapes = SMART_SHAPES_DEFAULT
        self.focus_mode = False
        self.show_gesture_guide = True

        # Notification toasts
        self.toast_msg = None
        self.toast_time = 0.0

        # Text AI review card state: (text, conf, bbox, start_canvas)
        self.text_review = None

        # Text AI typography controls
        self.font_manager = FontManager.get_instance()
        self.font_families = ["Arial", "Calibri", "Times New Roman", "Default Sans"]
        self.font_labels = ["Arial", "Calibri", "Times", "Default"]
        self.active_font_idx = 0
        self.text_bold = False
        self.text_italic = False
        self.text_font_size = 48
        self.text_auto_size = True
        self.TEXT_PRESET_SIZES = [24, 36, 48, 64, 80]

        # Interactive hit-testing regions
        self._interactive_rects = []
        self._color_rects = []
        self._size_preset_rects = []
        self._slider_rect = None

        # Generate color assets
        self._wheel_size = 110
        self._color_wheel, self._wheel_mask = self._generate_color_wheel(self._wheel_size, 12)
        self._sv_square = self._generate_sv_square(270, 50)
        self._hue_bar = self._generate_hue_bar(136, 8)

        # Pre-compute layout and base render
        self._recalculate_layout()

    def _recalculate_layout(self):
        """Calculate layout dimensions based on Focus Mode state."""
        if self.focus_mode:
            # Maximized Canvas Mode
            self.left_w = 0
            self.right_w = 0
            self.top_h = 36
            self.bottom_h = 36
            self.pad = 4
            self.radius = 8
        else:
            # Full Studio Mode: compact sidebars, expansive canvas
            self.left_w = 148
            self.right_w = 168
            self.top_h = 42
            self.bottom_h = 44
            self.pad = 6
            self.radius = 10

        # Central workspace area
        ws_x = self.left_w + self.pad * 2
        ws_y = self.top_h + self.pad * 2
        ws_w = self.width - self.left_w - self.right_w - self.pad * 4
        ws_h = self.height - self.top_h - self.bottom_h - self.pad * 4

        # Maintain camera aspect ratio inside workspace
        vid_aspect = self.video_width / self.video_height
        ws_aspect = ws_w / ws_h
        if vid_aspect > ws_aspect:
            self.disp_w = ws_w
            self.disp_h = int(ws_w / vid_aspect)
        else:
            self.disp_h = ws_h
            self.disp_w = int(ws_h * vid_aspect)

        self.video_x = ws_x + (ws_w - self.disp_w) // 2
        self.video_y = ws_y + (ws_h - self.disp_h) // 2

        self._rounded_mask = self._create_rounded_mask(self.disp_w, self.disp_h, self.radius)
        self._base = self._render_base()
        self._build_hit_rects()

    # ── Asset Generators ────────────────────────────────────────

    def _generate_color_wheel(self, size: int = 110, ring_width: int = 12):
        """Generate smooth HSV color wheel ring."""
        center = size // 2
        y_c, x_c = np.ogrid[:size, :size]
        dx = (x_c - center).astype(np.float64)
        dy = (y_c - center).astype(np.float64)
        dist = np.sqrt(dx * dx + dy * dy)
        angle = (np.degrees(np.arctan2(-dy, dx)) + 360) % 360

        outer_r = center - 1
        inner_r = outer_r - ring_width
        ring_mask = (dist >= inner_r) & (dist <= outer_r)

        hsv = np.zeros((size, size, 3), dtype=np.uint8)
        hsv[..., 0] = (angle / 2).astype(np.uint8)  # OpenCV hue 0-179
        hsv[..., 1] = 255
        hsv[..., 2] = 255

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        bgr[~ring_mask] = 0
        return bgr, ring_mask.astype(np.uint8) * 255

    def _generate_sv_square(self, hue_deg: float, size: int = 50):
        """Generate saturation-value gradient square for given hue."""
        hue_cv = int((hue_deg % 360) / 2)
        hsv = np.zeros((size, size, 3), dtype=np.uint8)
        hsv[:, :, 0] = hue_cv
        sat = np.linspace(0, 255, size, dtype=np.uint8)
        val = np.linspace(255, 0, size, dtype=np.uint8)
        hsv[:, :, 1] = sat[np.newaxis, :]
        hsv[:, :, 2] = val[:, np.newaxis]
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _generate_hue_bar(self, width: int, height: int = 8):
        """Generate horizontal hue gradient bar."""
        hsv = np.zeros((height, width, 3), dtype=np.uint8)
        hues = np.linspace(0, 179, width, dtype=np.uint8)
        hsv[:, :, 0] = hues[np.newaxis, :]
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = 255
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _create_rounded_mask(self, w: int, h: int, radius: int = 10):
        """Pre-compute rounded rectangle mask for video/canvas."""
        mask = np.zeros((h, w), dtype=np.uint8)
        r = min(radius, min(w, h) // 2)
        cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
        cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
        cv2.circle(mask, (r, r), r, 255, -1)
        cv2.circle(mask, (w - r, r), r, 255, -1)
        cv2.circle(mask, (r, h - r), r, 255, -1)
        cv2.circle(mask, (w - r, h - r), r, 255, -1)
        return mask

    # ── Coordinate Conversions ──────────────────────────────────

    def ui_to_canvas(self, ui_x: int, ui_y: int):
        """Convert a UI screen coordinate to camera/canvas coordinates."""
        if (self.video_x <= ui_x <= self.video_x + self.disp_w and
                self.video_y <= ui_y <= self.video_y + self.disp_h):
            cx = int(round((ui_x - self.video_x) / self.disp_w * self.video_width))
            cy = int(round((ui_y - self.video_y) / self.disp_h * self.video_height))
            cx = max(0, min(self.video_width - 1, cx))
            cy = max(0, min(self.video_height - 1, cy))
            return cx, cy
        return None, None

    def canvas_to_ui(self, cx: int, cy: int):
        """Convert canvas coordinate to UI screen coordinate."""
        ui_x = int(round(self.video_x + (cx / self.video_width) * self.disp_w))
        ui_y = int(round(self.video_y + (cy / self.video_height) * self.disp_h))
        return ui_x, ui_y

    def is_over_ui(self, ui_x: int, ui_y: int) -> bool:
        """Check if screen coordinate is over UI panels (not on drawing canvas)."""
        if self.focus_mode:
            # In focus mode, only top HUD and bottom HUD count as UI
            if ui_y <= self.top_h + self.pad * 2 or ui_y >= self.height - self.bottom_h - self.pad * 2:
                return True
            return False

        # In standard mode: top bar, bottom bar, left panel, right panel
        if ui_y <= self.top_h + self.pad * 2 or ui_y >= self.height - self.bottom_h - self.pad * 2:
            return True
        if ui_x <= self.left_w + self.pad * 2 or ui_x >= self.width - self.right_w - self.pad * 2:
            return True
        return False

    # ── State Setters ───────────────────────────────────────────

    def set_active_tool(self, idx: int):
        if 0 <= idx < len(self.TOOLS):
            self.active_tool = idx
            if idx == 0:
                self.active_mode = 0  # Draw
            elif idx == 1:
                self.active_mode = 2  # Erase
            elif idx == 2:
                self.smart_shapes = not self.smart_shapes
                self.show_toast(f"Smart Shapes: {'ON' if self.smart_shapes else 'OFF'}")
            elif idx == 3:
                self.active_mode = 3  # Text AI
                self.show_toast("Text Mode: Write in the air!")
            self._base = self._render_base()
            self._build_hit_rects()

    def set_active_mode(self, idx: int):
        if 0 <= idx < len(self.MODES):
            self.active_mode = idx
            if idx == 0:
                self.active_tool = 0
            elif idx == 2:
                self.active_tool = 1
            elif idx == 3:
                self.active_tool = 3
            self._base = self._render_base()
            self._build_hit_rects()

    def set_active_brush(self, idx: int):
        if 0 <= idx < len(self.BRUSH_NAMES):
            self.active_brush = idx
            self._base = self._render_base()
            self._build_hit_rects()

    def set_brush_size(self, size: int):
        self.brush_size = max(1, min(40, size))
        self._base = self._render_base()
        self._build_hit_rects()

    def toggle_smart_shapes(self):
        self.smart_shapes = not self.smart_shapes
        self.show_toast(f"Smart Shapes: {'ON' if self.smart_shapes else 'OFF'}")
        self._base = self._render_base()
        self._build_hit_rects()

    def toggle_focus_mode(self):
        self.focus_mode = not self.focus_mode
        self.show_toast(f"Focus Mode: {'ON' if self.focus_mode else 'OFF'}")
        self._recalculate_layout()

    def set_active_font(self, idx: int):
        if 0 <= idx < len(self.font_families):
            self.active_font_idx = idx
            self.show_toast(f"Font: {self.font_families[idx]}")
            self._base = self._render_base()
            self._build_hit_rects()

    def toggle_text_bold(self):
        self.text_bold = not self.text_bold
        self.show_toast(f"Bold: {'ON' if self.text_bold else 'OFF'}")
        self._base = self._render_base()
        self._build_hit_rects()

    def toggle_text_italic(self):
        self.text_italic = not self.text_italic
        self.show_toast(f"Italic: {'ON' if self.text_italic else 'OFF'}")
        self._base = self._render_base()
        self._build_hit_rects()

    def set_text_font_size(self, size: int):
        self.text_font_size = max(12, min(140, int(size)))
        self.text_auto_size = False
        self.show_toast(f"Text Size: {self.text_font_size}px")
        self._base = self._render_base()
        self._build_hit_rects()

    def toggle_text_auto_size(self):
        self.text_auto_size = not self.text_auto_size
        self.show_toast(f"Auto Size: {'ON' if self.text_auto_size else 'OFF'}")
        self._base = self._render_base()
        self._build_hit_rects()

    def get_active_font_family(self) -> str:
        if 0 <= self.active_font_idx < len(self.font_families):
            return self.font_families[self.active_font_idx]
        return "Arial"

    def show_toast(self, message: str, duration: float = 2.5):
        self.toast_msg = message
        self.toast_time = time.time() + duration

    # ── Interactive Hit Testing ─────────────────────────────────

    def _build_hit_rects(self):
        """Construct bounding boxes for interactive UI elements."""
        self._interactive_rects = []
        self._color_rects = []
        self._size_preset_rects = []

        # Top Bar Modes
        pill_w, pill_h = 74, 26
        pill_start_x = (self.width - len(self.MODES) * (pill_w + 6)) // 2
        for i in range(len(self.MODES)):
            px = pill_start_x + i * (pill_w + 6)
            py = self.pad + (self.top_h - pill_h) // 2
            self._interactive_rects.append(('mode', i, px, py, px + pill_w, py + pill_h))

        # Smart Shapes button
        ss_x = self.width - self.pad - 210
        ss_y = self.pad + 6
        self._interactive_rects.append(('smart_shapes', 0, ss_x, ss_y, ss_x + 110, ss_y + 28))

        # Focus Mode button
        fm_x = self.width - self.pad - 90
        fm_y = self.pad + 6
        self._interactive_rects.append(('focus', 0, fm_x, fm_y, fm_x + 80, fm_y + 28))

        if not self.focus_mode:
            # Left panel tools
            lp_x1 = self.pad
            lp_y1 = self.top_h + self.pad * 2
            ty = lp_y1 + 28
            for i in range(len(self.TOOLS)):
                tool_y = ty + i * 28
                self._interactive_rects.append(('tool', i, lp_x1 + 6, tool_y - 2, lp_x1 + self.left_w - 6, tool_y + 24))

            if self.active_mode == 3 or self.active_tool == 3:
                # Text AI typography controls
                tf_y = ty + len(self.TOOLS) * 28 + 14 + 18
                for i in range(len(self.font_families)):
                    py = tf_y + i * 26
                    self._interactive_rects.append(('text_font', i, lp_x1 + 8, py, lp_x1 + self.left_w - 8, py + 22))

                st_y = tf_y + len(self.font_families) * 26 + 12 + 18
                half_w = (self.left_w - 22) // 2
                bx1 = lp_x1 + 8
                bx2 = bx1 + half_w
                ix1 = bx2 + 6
                ix2 = ix1 + half_w
                btn_h = 24
                self._interactive_rects.append(('text_bold', 0, bx1, st_y, bx2, st_y + btn_h))
                self._interactive_rects.append(('text_italic', 0, ix1, st_y, ix2, st_y + btn_h))

                sz_y = st_y + btn_h + 12 + 16
                self._interactive_rects.append(('text_auto_size', 0, lp_x1 + 8, sz_y, lp_x1 + self.left_w - 8, sz_y + 22))

                ps_y = sz_y + 28
                num_sizes = len(self.TEXT_PRESET_SIZES)
                pw = (self.left_w - 16 - (num_sizes - 1) * 3) // num_sizes
                for j, ps in enumerate(self.TEXT_PRESET_SIZES):
                    px = lp_x1 + 8 + j * (pw + 3)
                    self._interactive_rects.append(('text_size', ps, px, ps_y, px + pw, ps_y + 20))
            else:
                # Brush presets
                bp_y = ty + len(self.TOOLS) * 28 + 22
                cols, cell, gap = 3, 38, 4
                for i in range(len(self.BRUSH_NAMES)):
                    row = i // cols
                    col = i % cols
                    bx = lp_x1 + 10 + col * (cell + gap)
                    by = bp_y + row * (cell + gap)
                    self._interactive_rects.append(('brush', i, bx, by, bx + cell, by + cell))

                # Brush Size Presets
                bs_y = bp_y + (len(self.BRUSH_NAMES) // cols + 1) * (cell + gap) + 38
                for j, ps in enumerate(self.PRESET_SIZES):
                    px = lp_x1 + 14 + j * 24
                    hit_r = 11
                    self._size_preset_rects.append((px - hit_r, bs_y - hit_r, px + hit_r, bs_y + hit_r, ps))

                # Slider track
                sl_x1 = lp_x1 + 14
                sl_x2 = lp_x1 + self.left_w - 14
                sl_cy = bs_y - 18
                self._slider_rect = (sl_x1, sl_cy - 10, sl_x2, sl_cy + 10)

            # Color Swatches
            for i, (cx, cy) in enumerate(self._color_centers):
                self._color_rects.append((cx - 10, cy - 10, cx + 10, cy + 10, i))

        # Bottom Bar Actions
        bb_y1 = self.height - self.bottom_h - self.pad
        actions = ["Clear", "Undo", "Redo", "Save", "Export"]
        act_start = self.width // 2 - (len(actions) * 65) // 2
        for j, act in enumerate(actions):
            ax = act_start + j * 65
            ay = bb_y1 + 6
            self._interactive_rects.append(('action', act, ax, ay, ax + 56, ay + 30))

    def handle_selection(self, cursor_x: int, cursor_y: int):
        """Handle UI click/selection at (cursor_x, cursor_y). Returns action string or True if changed."""
        # Check text review buttons first if open
        if self.text_review is not None:
            # Review dialog buttons
            rw_x, rw_y, rw_w, rw_h = self._get_text_review_rect()
            btn_y = rw_y + rw_h - 36
            btn_w, btn_h = 105, 28
            btn1_x = rw_x + 16
            btn2_x = btn1_x + btn_w + 12
            btn3_x = btn2_x + btn_w + 12
            # Accept [✓]
            if btn1_x <= cursor_x <= btn1_x + btn_w and btn_y <= cursor_y <= btn_y + btn_h:
                return "TEXT_ACCEPT"
            # Retry [↺]
            if btn2_x <= cursor_x <= btn2_x + btn_w and btn_y <= cursor_y <= btn_y + btn_h:
                return "TEXT_RETRY"
            # Cancel / Dismiss [✕]
            if btn3_x <= cursor_x <= btn3_x + btn_w and btn_y <= cursor_y <= btn_y + btn_h:
                return "TEXT_CANCEL"

        # Check interactive buttons
        for item in self._interactive_rects:
            kind = item[0]
            val = item[1]
            x1, y1, x2, y2 = item[2], item[3], item[4], item[5]
            if x1 <= cursor_x <= x2 and y1 <= cursor_y <= y2:
                if kind == 'mode':
                    self.set_active_mode(val)
                    return True
                elif kind == 'tool':
                    self.set_active_tool(val)
                    return True
                elif kind == 'brush':
                    self.set_active_brush(val)
                    return True
                elif kind == 'text_font':
                    self.set_active_font(val)
                    return True
                elif kind == 'text_bold':
                    self.toggle_text_bold()
                    return True
                elif kind == 'text_italic':
                    self.toggle_text_italic()
                    return True
                elif kind == 'text_auto_size':
                    self.toggle_text_auto_size()
                    return True
                elif kind == 'text_size':
                    self.set_text_font_size(val)
                    return True
                elif kind == 'smart_shapes':
                    self.toggle_smart_shapes()
                    return True
                elif kind == 'focus':
                    self.toggle_focus_mode()
                    return True
                elif kind == 'action':
                    return val.upper()

        # Check color swatches
        for x1, y1, x2, y2, idx in self._color_rects:
            if x1 <= cursor_x <= x2 and y1 <= cursor_y <= y2:
                self.active_color_idx = idx
                self.active_color = self.PALETTE_COLORS[idx][0]
                self._base = self._render_base()
                self._build_hit_rects()
                return True

        # Check Color Wheel / Hue Bar / SV square
        if not self.focus_mode and hasattr(self, '_wheel_x'):
            wh = self._wheel_size
            in_wheel = (self._wheel_x <= cursor_x < self._wheel_x + wh and
                        self._wheel_y <= cursor_y < self._wheel_y + wh)
            hb_h, hb_w = self._hue_bar.shape[:2]
            in_hue = (self._hue_bar_x <= cursor_x < self._hue_bar_x + hb_w and
                      self._hue_bar_y <= cursor_y < self._hue_bar_y + hb_h)

            if in_wheel or in_hue:
                if 0 <= cursor_y < self.height and 0 <= cursor_x < self.width:
                    b, g, r = self._base[cursor_y, cursor_x]
                    col = (int(b), int(g), int(r))
                    if col not in (self.PANEL[::-1], self.BG[::-1], self.PANEL_BORDER[::-1]):
                        self.active_color = col
                        in_sv = (self._sv_x <= cursor_x < self._sv_x + self._sv_square.shape[1] and
                                 self._sv_y <= cursor_y < self._sv_y + self._sv_square.shape[0])
                        if in_hue or (in_wheel and not in_sv):
                            hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
                            self._sv_square = self._generate_sv_square(int(hsv[0]) * 2, 50)
                        self._base = self._render_base()
                        self._build_hit_rects()
                        return True

        # Check Brush Size presets
        for x1, y1, x2, y2, size in self._size_preset_rects:
            if x1 <= cursor_x <= x2 and y1 <= cursor_y <= y2:
                if self.brush_size != size:
                    self.set_brush_size(size)
                    return True

        # Check Brush Size slider
        if self._slider_rect:
            x1, y1, x2, y2 = self._slider_rect
            if x1 <= cursor_x <= x2 and y1 <= cursor_y <= y2:
                pct = max(0.0, min(1.0, (cursor_x - x1) / (x2 - x1)))
                new_size = max(1, min(40, int(1 + pct * 39)))
                if self.brush_size != new_size:
                    self.set_brush_size(new_size)
                    return True

        return False

    def _get_text_review_rect(self):
        """Return (x, y, w, h) for text recognition confirmation card."""
        w, h = 380, 160
        x = (self.width - w) // 2
        y = self.top_h + self.pad * 3 + 12
        return x, y, w, h

    # ── Static UI Rendering (Pillow) ────────────────────────────

    def _render_base(self):
        """Render studio layout frame with clean typography and balanced hierarchy."""
        img = Image.new('RGB', (self.width, self.height), self.BG)
        d = ImageDraw.Draw(img)

        # Load clean typography
        try:
            font_title = ImageFont.truetype("arial.ttf", 15)
            font_md = ImageFont.truetype("arial.ttf", 12)
            font_sm = ImageFont.truetype("arial.ttf", 10)
            font_bold = ImageFont.truetype("arialbd.ttf", 12)
        except Exception:
            font_title = font_md = font_sm = font_bold = ImageFont.load_default()

        r = self.radius

        # ── TOP BAR ─────────────────────────────────────────────
        top_y1 = self.pad
        top_y2 = self.top_h + self.pad
        d.rounded_rectangle([self.pad, top_y1, self.width - self.pad, top_y2],
                            radius=r, fill=self.PANEL, outline=self.PANEL_BORDER)

        # VisionCanvas Logo
        d.ellipse([self.pad + 12, top_y1 + 13, self.pad + 22, top_y1 + 23], fill=self.ACCENT)
        d.text((self.pad + 28, top_y1 + 10), "VisionCanvas", fill=self.TEXT_PRIMARY, font=font_title)
        d.text((self.pad + 126, top_y1 + 13), "STUDIO", fill=self.ACCENT_GLOW, font=font_sm)

        # Mode Pills
        pill_w, pill_h = 74, 26
        pill_start_x = (self.width - len(self.MODES) * (pill_w + 6)) // 2
        for i, mode in enumerate(self.MODES):
            px = pill_start_x + i * (pill_w + 6)
            py = top_y1 + (self.top_h - pill_h) // 2
            is_active = (i == self.active_mode)
            bg = self.ACCENT if is_active else self.PANEL
            border = self.ACCENT_GLOW if is_active else self.PANEL_BORDER
            d.rounded_rectangle([px, py, px + pill_w, py + pill_h], radius=13, fill=bg, outline=border)
            tc = (255, 255, 255) if is_active else self.TEXT_SECONDARY
            d.text((px + 12, py + 6), mode, fill=tc, font=font_md)

        # Smart Shapes button
        ss_x = self.width - self.pad - 210
        ss_y = top_y1 + 6
        ss_bg = self.ACCENT_DIM if self.smart_shapes else (22, 24, 36)
        ss_border = self.ACCENT if self.smart_shapes else self.PANEL_BORDER
        d.rounded_rectangle([ss_x, ss_y, ss_x + 110, ss_y + 28], radius=14, fill=ss_bg, outline=ss_border)
        d.text((ss_x + 10, ss_y + 7), "⚡ Shapes: " + ("ON" if self.smart_shapes else "OFF"),
               fill=self.ACCENT_GLOW if self.smart_shapes else self.TEXT_DIM, font=font_sm)

        # Focus Mode button
        fm_x = self.width - self.pad - 90
        fm_y = top_y1 + 6
        fm_bg = self.ACCENT_DIM if self.focus_mode else (22, 24, 36)
        d.rounded_rectangle([fm_x, fm_y, fm_x + 80, fm_y + 28], radius=14, fill=fm_bg, outline=self.PANEL_BORDER)
        d.text((fm_x + 12, fm_y + 7), "⛶ Focus" if not self.focus_mode else "✕ Exit",
               fill=self.TEXT_PRIMARY, font=font_sm)

        # ── LEFT PANEL (Tools & Brushes) ────────────────────────
        if not self.focus_mode:
            lp_x1 = self.pad
            lp_y1 = self.top_h + self.pad * 2
            lp_x2 = self.left_w + self.pad
            lp_y2 = self.height - self.bottom_h - self.pad * 2

            d.rounded_rectangle([lp_x1, lp_y1, lp_x2, lp_y2], radius=r, fill=self.PANEL, outline=self.PANEL_BORDER)

            # Tools Header
            ty = lp_y1 + 10
            d.text((lp_x1 + 12, ty), "TOOLS", fill=self.TEXT_DIM, font=font_sm)
            ty += 18

            for i, (tool_name, hotkey) in enumerate(self.TOOLS):
                tool_y = ty + i * 28
                is_active = (i == self.active_tool)
                bg = self.ACCENT if is_active else self.PANEL
                border = self.ACCENT_GLOW if is_active else self.PANEL_BORDER

                if is_active:
                    d.rounded_rectangle([lp_x1 + 6, tool_y - 2, lp_x2 - 6, tool_y + 24],
                                        radius=6, fill=bg, outline=border)

                tc = (255, 255, 255) if is_active else self.TEXT_SECONDARY
                d.text((lp_x1 + 14, tool_y + 3), tool_name, fill=tc, font=font_md)
                d.text((lp_x2 - 24, tool_y + 3), hotkey, fill=self.TEXT_DIM if not is_active else (220, 220, 255), font=font_sm)

            if self.active_mode == 3 or self.active_tool == 3:
                # Text AI Typography Controls
                tf_y = ty + len(self.TOOLS) * 28 + 14
                d.text((lp_x1 + 12, tf_y), "FONT FAMILY", fill=self.TEXT_DIM, font=font_sm)
                tf_y += 18

                for i, (fam, lbl) in enumerate(zip(self.font_families, self.font_labels)):
                    py = tf_y + i * 26
                    is_active = (i == self.active_font_idx)
                    bg = self.ACCENT if is_active else (22, 24, 36)
                    border = self.ACCENT_GLOW if is_active else self.PANEL_BORDER
                    d.rounded_rectangle([lp_x1 + 8, py, lp_x2 - 8, py + 22], radius=6, fill=bg, outline=border)
                    tc = (255, 255, 255) if is_active else self.TEXT_SECONDARY
                    d.text((lp_x1 + 14, py + 4), lbl, fill=tc, font=font_sm)
                    if is_active:
                        d.text((lp_x2 - 22, py + 4), "✓", fill=(255, 255, 255), font=font_sm)

                st_y = tf_y + len(self.font_families) * 26 + 12
                d.text((lp_x1 + 12, st_y), "STYLE", fill=self.TEXT_DIM, font=font_sm)
                st_y += 18

                half_w = (self.left_w - 22) // 2
                bx1 = lp_x1 + 8
                bx2 = bx1 + half_w
                ix1 = bx2 + 6
                ix2 = ix1 + half_w
                btn_h = 24

                # Bold Button
                b_bg = self.ACCENT if self.text_bold else (22, 24, 36)
                b_border = self.ACCENT_GLOW if self.text_bold else self.PANEL_BORDER
                d.rounded_rectangle([bx1, st_y, bx2, st_y + btn_h], radius=6, fill=b_bg, outline=b_border)
                d.text((bx1 + 10, st_y + 4), "Bold", fill=(255, 255, 255) if self.text_bold else self.TEXT_SECONDARY, font=font_sm)

                # Italic Button
                i_bg = self.ACCENT if self.text_italic else (22, 24, 36)
                i_border = self.ACCENT_GLOW if self.text_italic else self.PANEL_BORDER
                d.rounded_rectangle([ix1, st_y, ix2, st_y + btn_h], radius=6, fill=i_bg, outline=i_border)
                d.text((ix1 + 8, st_y + 4), "Italic", fill=(255, 255, 255) if self.text_italic else self.TEXT_SECONDARY, font=font_sm)

                sz_y = st_y + btn_h + 12
                d.text((lp_x1 + 12, sz_y), "TEXT SIZE", fill=self.TEXT_DIM, font=font_sm)
                size_str = "Auto" if self.text_auto_size else f"{self.text_font_size}px"
                d.text((lp_x2 - 42, sz_y), size_str, fill=self.TEXT_PRIMARY, font=font_bold)
                sz_y += 16

                # Auto-Fit Button
                auto_bg = self.ACCENT if self.text_auto_size else (22, 24, 36)
                auto_border = self.ACCENT_GLOW if self.text_auto_size else self.PANEL_BORDER
                d.rounded_rectangle([lp_x1 + 8, sz_y, lp_x2 - 8, sz_y + 22], radius=6, fill=auto_bg, outline=auto_border)
                d.text((lp_x1 + 14, sz_y + 4), "⚡ Match Handwriting", fill=(255, 255, 255) if self.text_auto_size else self.TEXT_SECONDARY, font=font_sm)

                ps_y = sz_y + 28
                num_sizes = len(self.TEXT_PRESET_SIZES)
                pw = (self.left_w - 16 - (num_sizes - 1) * 3) // num_sizes
                for j, ps in enumerate(self.TEXT_PRESET_SIZES):
                    px = lp_x1 + 8 + j * (pw + 3)
                    is_sz_active = (not self.text_auto_size and self.text_font_size == ps)
                    sbg = self.ACCENT if is_sz_active else (22, 24, 36)
                    sborder = self.ACCENT_GLOW if is_sz_active else self.PANEL_BORDER
                    d.rounded_rectangle([px, ps_y, px + pw, ps_y + 20], radius=4, fill=sbg, outline=sborder)
                    d.text((px + 4, ps_y + 3), str(ps), fill=(255, 255, 255) if is_sz_active else self.TEXT_DIM, font=font_sm)
            else:
                # Brush Presets
                bp_y = ty + len(self.TOOLS) * 28 + 14
                d.text((lp_x1 + 12, bp_y), "BRUSH PRESETS", fill=self.TEXT_DIM, font=font_sm)
                bp_y += 18

                cols, cell, gap = 3, 38, 4
                for i, bname in enumerate(self.BRUSH_NAMES):
                    row = i // cols
                    col = i % cols
                    bx = lp_x1 + 10 + col * (cell + gap)
                    by = bp_y + row * (cell + gap)
                    is_active = (i == self.active_brush)
                    outline_c = self.ACCENT if is_active else self.PANEL_BORDER
                    fill_c = (28, 30, 46) if is_active else (22, 24, 36)
                    d.rounded_rectangle([bx, by, bx + cell, by + cell], radius=6, fill=fill_c, outline=outline_c)

                    # Draw clean preview line in tile
                    py_m = by + cell // 2
                    if bname == "Hard":
                        d.line([(bx + 6, py_m + 4), (bx + cell - 6, py_m - 4)], fill=(220, 220, 240), width=3)
                    elif bname == "Soft":
                        d.line([(bx + 6, py_m + 4), (bx + cell - 6, py_m - 4)], fill=(160, 160, 200), width=5)
                    elif bname == "Neon":
                        d.line([(bx + 6, py_m + 4), (bx + cell - 6, py_m - 4)], fill=(255, 120, 255), width=2)
                    elif bname == "Pencil":
                        d.line([(bx + 6, py_m + 4), (bx + cell - 6, py_m - 4)], fill=(180, 180, 195), width=1)
                    elif bname == "Water":
                        d.line([(bx + 6, py_m + 4), (bx + cell - 6, py_m - 4)], fill=(130, 180, 230), width=4)

                # Brush Size Section
                bs_y = bp_y + (len(self.BRUSH_NAMES) // cols + 1) * (cell + gap) + 14
                d.text((lp_x1 + 12, bs_y), "SIZE", fill=self.TEXT_DIM, font=font_sm)
                d.text((lp_x2 - 38, bs_y), f"{self.brush_size}px", fill=self.TEXT_PRIMARY, font=font_bold)
                bs_y += 16

                # Interactive Slider Track
                sl_x1 = lp_x1 + 14
                sl_x2 = lp_x2 - 14
                sl_cy = bs_y + 4
                d.rounded_rectangle([sl_x1, sl_cy - 2, sl_x2, sl_cy + 2], radius=2, fill=self.PANEL_BORDER)
                # Thumb
                frac = (self.brush_size - 1) / 39.0
                tx = int(sl_x1 + frac * (sl_x2 - sl_x1))
                d.ellipse([tx - 5, sl_cy - 5, tx + 5, sl_cy + 5], fill=self.ACCENT)

                # Preset Size Dots
                bs_y += 20
                for j, ps in enumerate(self.PRESET_SIZES):
                    px = lp_x1 + 14 + j * 24
                    pr = max(2, min(7, ps // 3 + 1))
                    fill_c = self.ACCENT if ps == self.brush_size else self.TEXT_DIM
                    d.ellipse([px - pr, bs_y - pr, px + pr, bs_y + pr], fill=fill_c)

        # ── RIGHT PANEL (Colors & Studio) ───────────────────────
        if not self.focus_mode:
            rp_x1 = self.width - self.right_w - self.pad
            rp_y1 = self.top_h + self.pad * 2
            rp_x2 = self.width - self.pad
            rp_y2 = self.height - self.bottom_h - self.pad * 2

            d.rounded_rectangle([rp_x1, rp_y1, rp_x2, rp_y2], radius=r, fill=self.PANEL, outline=self.PANEL_BORDER)

            cy = rp_y1 + 10
            d.text((rp_x1 + 14, cy), "COLOR STUDIO", fill=self.TEXT_DIM, font=font_sm)
            cy += 18

            # Positions for Color Wheel overlay
            self._wheel_x = (rp_x1 + rp_x2) // 2 - self._wheel_size // 2
            self._wheel_y = cy
            sv_s = self._sv_square.shape[0]
            self._sv_x = self._wheel_x + (self._wheel_size - sv_s) // 2
            self._sv_y = self._wheel_y + (self._wheel_size - sv_s) // 2

            # Hue Bar
            hb_w = self._hue_bar.shape[1]
            self._hue_bar_x = (rp_x1 + rp_x2) // 2 - hb_w // 2
            self._hue_bar_y = self._wheel_y + self._wheel_size + 8

            # Hex code preview badge
            hex_y = self._hue_bar_y + 16
            b_c, g_c, r_c = self.active_color
            hex_str = f"#{r_c:02X}{g_c:02X}{b_c:02X}"
            d.rounded_rectangle([rp_x1 + 14, hex_y, rp_x2 - 14, hex_y + 22],
                                radius=6, fill=(24, 26, 40), outline=self.PANEL_BORDER)
            d.ellipse([rp_x1 + 22, hex_y + 5, rp_x1 + 34, hex_y + 17], fill=(r_c, g_c, b_c))
            d.text((rp_x1 + 42, hex_y + 4), hex_str, fill=self.TEXT_PRIMARY, font=font_sm)

            # Palette Swatches (2 rows of 5)
            sw_y = hex_y + 30
            d.text((rp_x1 + 14, sw_y), "PALETTE", fill=self.TEXT_DIM, font=font_sm)
            sw_y += 16

            self._color_centers = []
            cols_c = 5
            spacing_c = 26
            start_cx = rp_x1 + 24
            for i, (col, _) in enumerate(self.PALETTE_COLORS):
                row = i // cols_c
                col_i = i % cols_c
                cx = start_cx + col_i * spacing_c
                ccy = sw_y + row * spacing_c
                self._color_centers.append((cx, ccy))
                cb, cg, cr = col
                d.ellipse([cx - 8, ccy - 8, cx + 8, ccy + 8], fill=(cr, cg, cb))
                if i == self.active_color_idx:
                    d.ellipse([cx - 10, ccy - 10, cx + 10, ccy + 10], outline=(255, 255, 255), width=2)

            # Layers card
            ly_y = sw_y + 2 * spacing_c + 14
            d.text((rp_x1 + 14, ly_y), "LAYERS", fill=self.TEXT_DIM, font=font_sm)
            ly_y += 18
            d.rounded_rectangle([rp_x1 + 10, ly_y, rp_x2 - 10, ly_y + 28], radius=6,
                                fill=(28, 30, 48), outline=self.ACCENT)
            d.ellipse([rp_x1 + 20, ly_y + 9, rp_x1 + 30, ly_y + 19], fill=self.LIVE_GREEN)
            d.text((rp_x1 + 36, ly_y + 7), "Drawing Canvas", fill=self.TEXT_PRIMARY, font=font_sm)

        # ── BOTTOM BAR ──────────────────────────────────────────
        bb_y1 = self.height - self.bottom_h - self.pad
        bb_y2 = self.height - self.pad
        d.rounded_rectangle([self.pad, bb_y1, self.width - self.pad, bb_y2],
                            radius=r, fill=self.PANEL, outline=self.PANEL_BORDER)

        # Canvas Resolution badge
        d.text((self.pad + 14, bb_y1 + 14), f"CANVAS: {self.video_width}×{self.video_height}",
               fill=self.TEXT_DIM, font=font_sm)

        # Center Action buttons: Clear, Undo, Redo, Save, Export
        actions = [("Clear", "C"), ("Undo", "Z"), ("Redo", "Y"), ("Save", "S"), ("Export", "E")]
        act_start = self.width // 2 - (len(actions) * 65) // 2
        for j, (lbl, key) in enumerate(actions):
            ax = act_start + j * 65
            ay = bb_y1 + 6
            d.rounded_rectangle([ax, ay, ax + 56, ay + 30], radius=6,
                                fill=(24, 26, 40), outline=self.PANEL_BORDER)
            d.text((ax + 10, ay + 3), lbl, fill=self.TEXT_PRIMARY, font=font_sm)
            d.text((ax + 38, ay + 15), key, fill=self.ACCENT_GLOW, font=font_sm)

        # ── Convert Pillow image to OpenCV BGR ───────────────────
        base_np = np.array(img)
        base_bgr = cv2.cvtColor(base_np, cv2.COLOR_RGB2BGR)

        # Blit Color Wheel & Hue Bar assets
        if not self.focus_mode and hasattr(self, '_wheel_x'):
            wh, ww = self._color_wheel.shape[:2]
            wy, wx = self._wheel_y, self._wheel_x
            if 0 <= wy and wy + wh <= self.height and 0 <= wx and wx + ww <= self.width:
                roi = base_bgr[wy:wy + wh, wx:wx + ww]
                mask = self._wheel_mask
                roi[mask > 0] = self._color_wheel[mask > 0]

            sv_h, sv_w = self._sv_square.shape[:2]
            sy, sx = self._sv_y, self._sv_x
            if 0 <= sy and sy + sv_h <= self.height and 0 <= sx and sx + sv_w <= self.width:
                base_bgr[sy:sy + sv_h, sx:sx + sv_w] = self._sv_square

            hb_h, hb_w = self._hue_bar.shape[:2]
            hy, hx = self._hue_bar_y, self._hue_bar_x
            if 0 <= hy and hy + hb_h <= self.height and 0 <= hx and hx + hb_w <= self.width:
                base_bgr[hy:hy + hb_h, hx:hx + hb_w] = self._hue_bar

        return base_bgr

    # ── Dynamic Frame Rendering ─────────────────────────────────

    def render(self, frame, canvas_frame, cursor_x: int, cursor_y: int,
               gesture: str, landmarks=None):
        """Render the complete application frame for this tick."""
        ui = self._base.copy()

        # Resize video frame to display workspace
        disp_frame = cv2.resize(frame, (self.disp_w, self.disp_h))

        # Hand skeleton landmarks overlay
        if DRAW_HAND_LANDMARKS and landmarks:
            self._draw_hand_landmarks(disp_frame, landmarks, self.disp_w, self.disp_h)

        # Blend canvas with camera frame
        if canvas_frame.shape[1] != self.disp_w or canvas_frame.shape[0] != self.disp_h:
            disp_canvas = cv2.resize(canvas_frame, (self.disp_w, self.disp_h),
                                     interpolation=cv2.INTER_LINEAR)
        else:
            disp_canvas = canvas_frame

        gray = cv2.cvtColor(disp_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        frame_bg = cv2.bitwise_and(disp_frame, disp_frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(disp_canvas, disp_canvas, mask=mask)
        merged = cv2.add(frame_bg, canvas_fg)

        # Apply rounded corners to drawing area
        vx, vy = self.video_x, self.video_y
        bg_region = ui[vy:vy + self.disp_h, vx:vx + self.disp_w]
        mask_inv_r = cv2.bitwise_not(self._rounded_mask)
        bg_corners = cv2.bitwise_and(bg_region, bg_region, mask=mask_inv_r)
        merged_rounded = cv2.bitwise_and(merged, merged, mask=self._rounded_mask)
        ui[vy:vy + self.disp_h, vx:vx + self.disp_w] = cv2.add(bg_corners, merged_rounded)

        # Subtle workspace border
        cv2.rectangle(ui, (vx - 1, vy - 1), (vx + self.disp_w, vy + self.disp_h),
                      self.PANEL_BORDER, 1)

        # ── Corner Status Badges (Minimal, Non-obtrusive) ────────
        # LIVE / Hand Status Badge
        badge_x, badge_y = vx + 10, vy + 10
        tracking_active = (landmarks is not None and len(landmarks) > 0)
        status_color = self.LIVE_GREEN if tracking_active else self.AMBER
        status_text = "LIVE TRACKING" if tracking_active else "LOOKING FOR HAND"

        cv2.rectangle(ui, (badge_x, badge_y), (badge_x + 120, badge_y + 22),
                      (16, 18, 28), cv2.FILLED)
        cv2.rectangle(ui, (badge_x, badge_y), (badge_x + 120, badge_y + 22),
                      self.PANEL_BORDER, 1)
        cv2.circle(ui, (badge_x + 10, badge_y + 11), 4, status_color, cv2.FILLED, cv2.LINE_AA)
        cv2.putText(ui, status_text, (badge_x + 20, badge_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, status_color, 1, cv2.LINE_AA)

        # Gesture Pill (Top Right of Workspace)
        if gesture != "NONE":
            g_text = f"GESTURE: {gesture}"
            (tw, th), _ = cv2.getTextSize(g_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            gx = vx + self.disp_w - tw - 24
            gy = vy + 10
            cv2.rectangle(ui, (gx, gy), (gx + tw + 16, gy + 22), (20, 22, 36), cv2.FILLED)
            cv2.rectangle(ui, (gx, gy), (gx + tw + 16, gy + 22), self.ACCENT, 1)
            cv2.putText(ui, g_text, (gx + 8, gy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.ACCENT_GLOW, 1, cv2.LINE_AA)

        # ── Compact Floating Gesture Guide (Bottom-Right) ───────
        if self.show_gesture_guide and not self.focus_mode:
            gw = 160
            gh = 72
            gx = vx + self.disp_w - gw - 10
            gy = vy + self.disp_h - gh - 10

            guide_roi = ui[gy:gy + gh, gx:gx + gw]
            dark = np.full_like(guide_roi, (14, 16, 26), dtype=np.uint8)
            cv2.addWeighted(dark, 0.85, guide_roi, 0.15, 0, dst=guide_roi)
            cv2.rectangle(ui, (gx, gy), (gx + gw, gy + gh), self.PANEL_BORDER, 1)

            cv2.putText(ui, "GESTURES", (gx + 8, gy + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, self.TEXT_DIM, 1, cv2.LINE_AA)
            cv2.putText(ui, "👆 1 Finger: Draw", (gx + 8, gy + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, self.TEXT_PRIMARY, 1, cv2.LINE_AA)
            cv2.putText(ui, "✌️ 2 Fingers: Select/Click", (gx + 8, gy + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, self.TEXT_PRIMARY, 1, cv2.LINE_AA)
            cv2.putText(ui, "🖐 3-4 Fingers: Undo/Redo", (gx + 8, gy + 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, self.TEXT_PRIMARY, 1, cv2.LINE_AA)
            cv2.putText(ui, "👍 Thumbs Up: Save", (gx + 8, gy + 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, self.TEXT_DIM, 1, cv2.LINE_AA)

        # ── Toast Message ───────────────────────────────────────
        if self.toast_msg and time.time() < self.toast_time:
            (msg_w, _), _ = cv2.getTextSize(self.toast_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tx = (self.width - msg_w - 28) // 2
            ty = vy + 12
            cv2.rectangle(ui, (tx, ty), (tx + msg_w + 28, ty + 28), (24, 26, 42), cv2.FILLED)
            cv2.rectangle(ui, (tx, ty), (tx + msg_w + 28, ty + 28), self.ACCENT, 1)
            cv2.putText(ui, self.toast_msg, (tx + 14, ty + 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.TEXT_PRIMARY, 1, cv2.LINE_AA)

        # ── Text Recognition Review Popup Card ──────────────────
        if self.text_review is not None:
            rw_x, rw_y, rw_w, rw_h = self._get_text_review_rect()
            card_roi = ui[rw_y:rw_y + rw_h, rw_x:rw_x + rw_w]
            dark_card = np.full_like(card_roi, (18, 20, 32), dtype=np.uint8)
            cv2.addWeighted(dark_card, 0.94, card_roi, 0.06, 0, dst=card_roi)
            cv2.rectangle(ui, (rw_x, rw_y), (rw_x + rw_w, rw_y + rw_h), self.ACCENT, 2)

            rec_text = self.text_review.get("text", "")
            rec_conf = int(self.text_review.get("conf", 0) * 100)
            bbox = self.text_review.get("bbox", (0, 0, 50, 50))
            bh = bbox[3] if bbox else 48

            # Header & Confidence Badge
            cv2.putText(ui, "AI TEXT RECOGNITION", (rw_x + 16, rw_y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.ACCENT_GLOW, 1, cv2.LINE_AA)

            badge_str = f"{rec_conf}% MATCH"
            badge_color = self.LIVE_GREEN if rec_conf >= 70 else self.AMBER
            (bw, _), _ = cv2.getTextSize(badge_str, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
            cv2.rectangle(ui, (rw_x + rw_w - bw - 26, rw_y + 10), (rw_x + rw_w - 14, rw_y + 26),
                          (26, 30, 44), cv2.FILLED)
            cv2.rectangle(ui, (rw_x + rw_w - bw - 26, rw_y + 10), (rw_x + rw_w - 14, rw_y + 26),
                          badge_color, 1)
            cv2.putText(ui, badge_str, (rw_x + rw_w - bw - 20, rw_y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, badge_color, 1, cv2.LINE_AA)

            # Left Preview Box (Live Font Glyph Rendering)
            pw_box, ph_box = 80, 74
            px_box, py_box = rw_x + 16, rw_y + 36
            cv2.rectangle(ui, (px_box, py_box), (px_box + pw_box, py_box + ph_box),
                          (12, 14, 22), cv2.FILLED)
            cv2.rectangle(ui, (px_box, py_box), (px_box + pw_box, py_box + ph_box),
                          self.PANEL_BORDER, 1)

            # Render text glyph using FontManager
            try:
                active_font_name = self.get_active_font_family()
                preview_font_size = min(46, max(20, int(ph_box * 0.7)))
                p_font = self.font_manager.get_font(
                    active_font_name, preview_font_size, bold=self.text_bold, italic=self.text_italic
                )
                box_img = np.full((ph_box, pw_box, 3), (12, 14, 22), dtype=np.uint8)
                rendered_box = self.font_manager.render_text_to_bgr(
                    box_img, rec_text, (pw_box // 2, ph_box // 2), p_font, self.active_color, align="center"
                )
                ui[py_box:py_box + ph_box, px_box:px_box + pw_box] = rendered_box
            except Exception:
                cv2.putText(ui, rec_text, (px_box + 24, py_box + 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.active_color, 2, cv2.LINE_AA)

            # Right Details Column
            dx = rw_x + 110
            dy = rw_y + 50
            cv2.putText(ui, f"Text: \"{rec_text}\"", (dx, dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, self.TEXT_PRIMARY, 1, cv2.LINE_AA)
            cv2.putText(ui, f"Font: {self.get_active_font_family()}", (dx, dy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.TEXT_SECONDARY, 1, cv2.LINE_AA)

            style_txt = f"{'Bold ' if self.text_bold else ''}{'Italic' if self.text_italic else ('Regular' if not self.text_bold else '')}"
            sz_txt = f"Auto ({bh}px)" if self.text_auto_size else f"{self.text_font_size}px"
            cv2.putText(ui, f"Style: {style_txt} | Size: {sz_txt}", (dx, dy + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, self.TEXT_DIM, 1, cv2.LINE_AA)

            # Bottom Three Action Buttons
            btn_y = rw_y + rw_h - 36
            btn_w, btn_h = 105, 28
            btn1_x = rw_x + 16
            btn2_x = btn1_x + btn_w + 12
            btn3_x = btn2_x + btn_w + 12

            # [✓ Accept]
            cv2.rectangle(ui, (btn1_x, btn_y), (btn1_x + btn_w, btn_y + btn_h), (35, 95, 55), cv2.FILLED)
            cv2.rectangle(ui, (btn1_x, btn_y), (btn1_x + btn_w, btn_y + btn_h), (55, 160, 85), 1)
            cv2.putText(ui, "Accept (Enter)", (btn1_x + 12, btn_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

            # [↺ Retry]
            cv2.rectangle(ui, (btn2_x, btn_y), (btn2_x + btn_w, btn_y + btn_h), (95, 75, 25), cv2.FILLED)
            cv2.rectangle(ui, (btn2_x, btn_y), (btn2_x + btn_w, btn_y + btn_h), (180, 140, 45), 1)
            cv2.putText(ui, "Retry (R)", (btn2_x + 24, btn_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

            # [✕ Cancel]
            cv2.rectangle(ui, (btn3_x, btn_y), (btn3_x + btn_w, btn_y + btn_h), (95, 35, 35), cv2.FILLED)
            cv2.rectangle(ui, (btn3_x, btn_y), (btn3_x + btn_w, btn_y + btn_h), (180, 60, 60), 1)
            cv2.putText(ui, "Cancel (Esc)", (btn3_x + 18, btn_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Cursor Rendering (Drawing vs UI Interaction) ────────
        if cursor_x > 0 and cursor_y > 0:
            is_ui_hover = self.is_over_ui(cursor_x, cursor_y)

            if gesture == "DRAW" and not is_ui_hover:
                # ── DRAWING CURSOR ──────────────────────────────
                # Precise pinpoint center + high-contrast dual ring
                b, g, rv = self.active_color
                r_tip = max(2, self.brush_size // 2)

                # Outer high-contrast white ring
                cv2.circle(ui, (cursor_x, cursor_y), r_tip + 4, (255, 255, 255), 1, cv2.LINE_AA)
                # Inner dark contrast ring
                cv2.circle(ui, (cursor_x, cursor_y), r_tip + 2, (15, 15, 20), 1, cv2.LINE_AA)
                # Center point with active brush color
                cv2.circle(ui, (cursor_x, cursor_y), r_tip, (b, g, rv), cv2.FILLED, cv2.LINE_AA)
            else:
                # ── UI INTERACTION CURSOR ───────────────────────
                # Sleek cyan pointer ring with center dot
                cv2.circle(ui, (cursor_x, cursor_y), 9, (15, 15, 20), 2, cv2.LINE_AA)
                cv2.circle(ui, (cursor_x, cursor_y), 8, self.NEON_CYAN, 2, cv2.LINE_AA)
                cv2.circle(ui, (cursor_x, cursor_y), 3, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)

        return ui

    def _draw_hand_landmarks(self, frame, landmarks, frame_w: int, frame_h: int):
        """Draw subtle neon-coloured hand skeleton on webcam feed."""
        pts = []
        for lm in landmarks:
            px = int(lm.x * frame_w)
            py = int(lm.y * frame_h)
            pts.append((px, py))

        for (a, b) in self.HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], LANDMARK_CONNECTION_COLOR,
                         LANDMARK_LINE_THICKNESS + 1, cv2.LINE_AA)
                cv2.line(frame, pts[a], pts[b], LANDMARK_COLOR,
                         LANDMARK_LINE_THICKNESS, cv2.LINE_AA)

        for i, (px, py) in enumerate(pts):
            r = LANDMARK_DOT_RADIUS + 1 if i == 8 else LANDMARK_DOT_RADIUS
            cv2.circle(frame, (px, py), r, LANDMARK_CONNECTION_COLOR, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), max(1, r - 1), (255, 255, 255), -1, cv2.LINE_AA)

        # Highlight index tip (landmark 8)
        if len(pts) > 8:
            cv2.circle(frame, pts[8], LANDMARK_DOT_RADIUS + 3, self.ACCENT, 1, cv2.LINE_AA)
