import cv2
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
from src.config import (
    DRAW_HAND_LANDMARKS,
    LANDMARK_COLOR, LANDMARK_CONNECTION_COLOR,
    LANDMARK_DOT_RADIUS, LANDMARK_LINE_THICKNESS,
)


class ModernUI:
    """Premium dark glassmorphic UI matching the Screenshot 5 target.

    Layout:
    ┌─────────────────────────────────────────────────────────┐
    │  TOP NAV: Logo | Draw Select Erase Move | Settings Pro  │
    ├────────┬────────────────────────────────┬───────────────┤
    │  LEFT  │                                │    RIGHT      │
    │  TOOLS │     CAMERA / DRAWING           │    COLORS     │
    │  BRUSH │     WORKSPACE                  │    LAYERS     │
    │  PRES. │                                │               │
    │  SIZE  │     GESTURE GUIDE              │               │
    ├────────┴────────────────────────────────┴───────────────┤
    │  BOTTOM BAR: Canvas | Clear Undo Redo | Track | Save    │
    └─────────────────────────────────────────────────────────┘
    """

    # ── Colour constants ────────────────────────────────────────
    BG = (10, 10, 18)
    PANEL = (18, 18, 30)
    PANEL_BORDER = (35, 35, 55)
    ACCENT = (140, 80, 255)
    ACCENT_GLOW = (180, 120, 255)
    ACCENT_DIM = (80, 50, 160)
    NEON_PINK = (255, 50, 150)
    NEON_CYAN = (0, 220, 255)
    TEXT_PRIMARY = (230, 230, 240)
    TEXT_SECONDARY = (130, 130, 150)
    TEXT_DIM = (80, 80, 100)
    LIVE_GREEN = (50, 220, 100)

    # Drawing colour palette (BGR for OpenCV)
    PALETTE_COLORS = [
        ((180, 80, 255), "Purple"),
        ((255, 100, 50), "Blue"),
        ((50, 220, 100), "Green"),
        ((0, 220, 255), "Cyan"),
        ((50, 150, 255), "Orange"),
        ((255, 50, 150), "Pink"),
        ((0, 0, 255), "Red"),
        ((0, 255, 255), "Yellow"),
        ((255, 255, 255), "White"),
        ((128, 128, 128), "Grey"),
    ]

    TOOLS = [
        "Brush",
        "Eraser",
        "Fill",
        "Shape",
        "Text (AI)",
        "Zoom",
        "Hand",
    ]

    MODES = ["Draw", "Select", "Erase", "Move"]

    GESTURE_GUIDE = [
        ("Draw", "Index Finger"),
        ("Select / Move", "Index + Middle"),
        ("Undo", "3 Fingers"),
        ("Redo", "4 Fingers"),
        ("Save", "Thumbs Up"),
    ]

    BRUSH_NAMES = ["Hard", "Soft", "Neon", "Pencil", "Water", "Custom"]

    # MediaPipe hand skeleton connections
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
        (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (0, 17),                                 # Palm base
    ]

    def __init__(self, window_width=1280, window_height=720,
                 video_width=640, video_height=480):
        self.width = window_width
        self.height = window_height
        self.video_width = video_width
        self.video_height = video_height

        # Layout constants
        self.left_w = 180
        self.right_w = 200
        self.top_h = 52
        self.bottom_h = 60
        self.pad = 10
        self.radius = 12

        # Workspace area
        ws_x = self.left_w + self.pad * 2
        ws_y = self.top_h + self.pad * 2
        ws_w = self.width - self.left_w - self.right_w - self.pad * 6
        ws_h = self.height - self.top_h - self.bottom_h - self.pad * 6

        # Fit video into workspace maintaining aspect ratio
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

        # State
        self.active_tool = 0
        self.active_mode = 0
        self.active_color_idx = 0
        self.active_color = self.PALETTE_COLORS[0][0]
        self.active_brush = 0
        self.brush_size = 12
        self.tracking_active = True

        # Pre-generate assets
        self._color_wheel, self._wheel_mask = self._generate_color_wheel(130, 14)
        self._sv_square = self._generate_sv_square(270, 56)
        self._hue_bar = self._generate_hue_bar(168, 10)
        self._rounded_mask = self._create_rounded_mask(self.disp_w, self.disp_h, 14)

        # Positions populated by _render_base
        self._color_centers = []

        # Pre-render static parts
        self._base = self._render_base()
        self._build_color_rects()

    # ── Asset generators ────────────────────────────────────────

    def _generate_color_wheel(self, size: int = 130, ring_width: int = 14):
        """Generate smooth HSV color wheel ring using vectorised numpy."""
        center = size // 2
        y_c, x_c = np.ogrid[:size, :size]
        dx = (x_c - center).astype(np.float64)
        dy = (y_c - center).astype(np.float64)
        dist = np.sqrt(dx * dx + dy * dy)
        angle = (np.degrees(np.arctan2(-dy, dx)) + 360) % 360

        outer_r = center - 2
        inner_r = outer_r - ring_width
        ring_mask = (dist >= inner_r) & (dist <= outer_r)

        hsv = np.zeros((size, size, 3), dtype=np.uint8)
        hsv[..., 0] = (angle / 2).astype(np.uint8)  # OpenCV hue 0-179
        hsv[..., 1] = 255
        hsv[..., 2] = 255

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        bgr[~ring_mask] = 0

        return bgr, ring_mask.astype(np.uint8) * 255

    def _generate_sv_square(self, hue_deg: float, size: int = 56):
        """Generate saturation-value gradient square for a given hue."""
        hue_cv = int((hue_deg % 360) / 2)
        hsv = np.zeros((size, size, 3), dtype=np.uint8)
        hsv[:, :, 0] = hue_cv
        sat = np.linspace(0, 255, size, dtype=np.uint8)
        val = np.linspace(255, 0, size, dtype=np.uint8)
        hsv[:, :, 1] = sat[np.newaxis, :]
        hsv[:, :, 2] = val[:, np.newaxis]
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _generate_hue_bar(self, width: int, height: int = 10):
        """Generate horizontal hue gradient bar."""
        hsv = np.zeros((height, width, 3), dtype=np.uint8)
        hues = np.linspace(0, 179, width, dtype=np.uint8)
        hsv[:, :, 0] = hues[np.newaxis, :]
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = 255
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _create_rounded_mask(self, w: int, h: int, radius: int = 14):
        """Pre-compute rounded rectangle mask for camera feed."""
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (radius, 0), (w - radius, h), 255, -1)
        cv2.rectangle(mask, (0, radius), (w, h - radius), 255, -1)
        cv2.circle(mask, (radius, radius), radius, 255, -1)
        cv2.circle(mask, (w - radius, radius), radius, 255, -1)
        cv2.circle(mask, (radius, h - radius), radius, 255, -1)
        cv2.circle(mask, (w - radius, h - radius), radius, 255, -1)
        return mask

    # ── Colour hit-testing ──────────────────────────────────────

    def _build_color_rects(self):
        """Build click regions from positions stored during _render_base."""
        self._color_rects = []
        for (cx, cy) in self._color_centers:
            self._color_rects.append((cx - 12, cy - 12, cx + 12, cy + 12))

    def handle_selection(self, cursor_x, cursor_y):
        """Check if cursor is over a colour button or wheel. Returns True if selected."""
        for i, (x1, y1, x2, y2) in enumerate(self._color_rects):
            if x1 <= cursor_x <= x2 and y1 <= cursor_y <= y2:
                self.active_color_idx = i
                self.active_color = self.PALETTE_COLORS[i][0]
                return True

        wh = self._color_wheel.shape[0]
        hb_h, hb_w = self._hue_bar.shape[:2]
        
        in_wheel = (self._wheel_x <= cursor_x < self._wheel_x + wh and 
                    self._wheel_y <= cursor_y < self._wheel_y + wh)
        in_hue = (self._hue_bar_x <= cursor_x < self._hue_bar_x + hb_w and 
                  self._hue_bar_y <= cursor_y < self._hue_bar_y + hb_h)
                  
        if in_wheel or in_hue:
            if 0 <= cursor_y < self.height and 0 <= cursor_x < self.width:
                b, g, r = self._base[cursor_y, cursor_x]
                new_color = (int(b), int(g), int(r))
                if new_color != self.PANEL[::-1] and new_color != self.BG[::-1] and new_color != self.PANEL_BORDER[::-1]:
                    if new_color != self.active_color:
                        self.active_color = new_color
                        in_sv = (self._sv_x <= cursor_x < self._sv_x + self._sv_square.shape[1] and 
                                 self._sv_y <= cursor_y < self._sv_y + self._sv_square.shape[0])
                        if in_hue or (in_wheel and not in_sv):
                            hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
                            self._sv_square = self._generate_sv_square(int(hsv[0]) * 2, 56)
                        self._base = self._render_base()
                        return True
        return False

    def handle_brush_size_selection(self, cursor_x, cursor_y):
        """Check if cursor is over the brush size slider or presets. Returns True if changed."""
        changed = False
        
        # Check presets
        if hasattr(self, '_size_preset_rects'):
            for x1, y1, x2, y2, size in self._size_preset_rects:
                if x1 <= cursor_x <= x2 and y1 <= cursor_y <= y2:
                    if self.brush_size != size:
                        self.set_brush_size(size)
                        changed = True
                        break
                        
        # Check slider
        if not changed and hasattr(self, '_slider_rect'):
            x1, y1, x2, y2 = self._slider_rect
            if x1 <= cursor_x <= x2 and y1 <= cursor_y <= y2:
                pct = (cursor_x - x1) / (x2 - x1)
                pct = max(0.0, min(1.0, pct))
                new_size = int(pct * 100)
                new_size = max(1, new_size)
                if self.brush_size != new_size:
                    self.set_brush_size(new_size)
                    changed = True
                    
        return changed

    def ui_to_canvas(self, ui_x, ui_y):
        """Convert a UI screen coordinate to camera/canvas coordinate."""
        if self.video_x <= ui_x <= self.video_x + self.disp_w and self.video_y <= ui_y <= self.video_y + self.disp_h:
            cx = int((ui_x - self.video_x) / self.disp_w * self.video_width)
            cy = int((ui_y - self.video_y) / self.disp_h * self.video_height)
            return cx, cy
        return None, None

    # ── State changes (re-render base on change) ────────────────

    def set_active_tool(self, idx: int):
        if 0 <= idx < len(self.TOOLS):
            self.active_tool = idx
            self._base = self._render_base()
            self._build_color_rects()

    def set_active_brush(self, idx: int):
        if 0 <= idx < len(self.BRUSH_NAMES):
            self.active_brush = idx
            self._base = self._render_base()
            self._build_color_rects()

    def set_brush_size(self, size: int):
        self.brush_size = max(1, min(100, size))
        self._base = self._render_base()
        self._build_color_rects()

    # ── Base UI (static panels, rendered via Pillow) ────────────

    def _render_base(self):
        img = Image.new('RGB', (self.width, self.height), self.BG)
        d = ImageDraw.Draw(img)

        # Fonts
        try:
            font_lg = ImageFont.truetype("arial.ttf", 16)
            font_md = ImageFont.truetype("arial.ttf", 13)
            font_sm = ImageFont.truetype("arial.ttf", 11)
            font_brand = ImageFont.truetype("arial.ttf", 18)
            font_icon = ImageFont.truetype("arial.ttf", 14)
        except (IOError, OSError):
            font_lg = ImageFont.load_default()
            font_md = font_sm = font_icon = font_lg
            font_brand = font_lg

        r = self.radius

        # ── Top Navigation Bar ──────────────────────────────────
        d.rounded_rectangle(
            [self.pad, self.pad, self.width - self.pad, self.top_h + self.pad],
            radius=r, fill=self.PANEL, outline=self.PANEL_BORDER,
        )
        d.text((self.pad + 16, self.pad + 8), "VisionCanvas", fill=self.ACCENT, font=font_brand)
        d.text((self.pad + 16, self.pad + 30), "Gesture Studio", fill=self.TEXT_DIM, font=font_sm)

        # Mode pills
        pill_total_w = len(self.MODES) * 90
        pill_start_x = (self.width - pill_total_w) // 2
        for i, mode in enumerate(self.MODES):
            px = pill_start_x + i * 90
            py = self.pad + 10
            pw, ph = 80, 32
            if i == self.active_mode:
                d.rounded_rectangle([px, py, px + pw, py + ph], radius=16, fill=self.ACCENT)
                d.text((px + 16, py + 8), mode, fill=(255, 255, 255), font=font_md)
            else:
                d.text((px + 16, py + 8), mode, fill=self.TEXT_SECONDARY, font=font_md)

        # Right controls
        right_x = self.width - self.pad - 200
        d.rounded_rectangle([right_x + 60, self.pad + 10, right_x + 160, self.pad + 38],
                            radius=14, fill=self.ACCENT_DIM)
        d.text((right_x + 72, self.pad + 15), "Pro Mode", fill=self.ACCENT_GLOW, font=font_sm)

        # ── Left Panel ──────────────────────────────────────────
        lp_x1 = self.pad
        lp_y1 = self.top_h + self.pad * 2
        lp_x2 = self.left_w + self.pad
        lp_y2 = self.height - self.bottom_h - self.pad * 2

        d.rounded_rectangle([lp_x1, lp_y1, lp_x2, lp_y2],
                            radius=r, fill=self.PANEL, outline=self.PANEL_BORDER)

        # Tools
        ty = lp_y1 + 12
        d.text((lp_x1 + 16, ty), "TOOLS", fill=self.TEXT_DIM, font=font_sm)
        ty += 22

        for i, tool_name in enumerate(self.TOOLS):
            tool_y = ty + i * 32
            
            icon_color = (255, 255, 255) if i == self.active_tool else self.TEXT_DIM
            text_color = (255, 255, 255) if i == self.active_tool else self.TEXT_SECONDARY
            
            if i == self.active_tool:
                d.rounded_rectangle([lp_x1 + 8, tool_y - 2, lp_x2 - 8, tool_y + 24],
                                    radius=8, fill=self.ACCENT)
                                    
            d.text((lp_x1 + 36, tool_y + 3), tool_name, fill=text_color, font=font_md)
            
            ix, iy = lp_x1 + 16, tool_y + 4
            if tool_name == "Brush":
                d.line([(ix+2, iy+10), (ix+10, iy+2)], fill=icon_color, width=2)
                d.point((ix+1, iy+11), fill=icon_color)
            elif tool_name == "Eraser":
                d.rectangle([(ix, iy+4), (ix+12, iy+10)], outline=icon_color, width=1)
                d.line([(ix+8, iy+4), (ix+8, iy+10)], fill=icon_color, width=1)
            elif tool_name == "Fill":
                d.polygon([(ix, iy+10), (ix+4, iy+4), (ix+12, iy+8), (ix+8, iy+14)], outline=icon_color)
                d.line([(ix+6, iy+12), (ix+6, iy+14)], fill=icon_color, width=1)
            elif tool_name == "Shape":
                d.ellipse([(ix, iy+2), (ix+10, iy+12)], outline=icon_color, width=1)
                d.rectangle([(ix+4, iy+6), (ix+14, iy+16)], outline=icon_color, width=1)
            elif tool_name == "Text (AI)":
                d.line([(ix+2, iy+2), (ix+12, iy+2)], fill=icon_color, width=2)
                d.line([(ix+7, iy+2), (ix+7, iy+12)], fill=icon_color, width=2)
                d.line([(ix+5, iy+12), (ix+9, iy+12)], fill=icon_color, width=1)
            elif tool_name == "Zoom":
                d.ellipse([(ix, iy), (ix+8, iy+8)], outline=icon_color, width=1)
                d.line([(ix+7, iy+7), (ix+12, iy+12)], fill=icon_color, width=2)
            elif tool_name == "Hand":
                d.rectangle([(ix+4, iy+4), (ix+10, iy+12)], outline=icon_color, width=1)
                d.line([(ix+4, iy+6), (ix+2, iy+6), (ix+2, iy+12), (ix+4, iy+12)], fill=icon_color, width=1)

        # Brush Presets
        bp_y = ty + len(self.TOOLS) * 32 + 12
        d.text((lp_x1 + 16, bp_y), "BRUSH PRESETS", fill=self.TEXT_DIM, font=font_sm)
        bp_y += 18

        cols = 3
        cell = 42
        gap = 5
        brush_preview_styles = [
            ((200, 200, 220), 3, "solid"),     # Hard
            ((150, 150, 180), 5, "solid"),     # Soft
            ((255, 120, 255), 2, "solid"),     # Neon
            ((180, 180, 200), 1, "solid"),     # Pencil
            ((130, 160, 200), 4, "wavy"),      # Water
            ((100, 100, 120), 2, "dotted"),    # Custom
        ]
        for i, bname in enumerate(self.BRUSH_NAMES):
            row = i // cols
            col = i % cols
            bx = lp_x1 + 12 + col * (cell + gap)
            by = bp_y + row * (cell + gap)
            outline_c = self.ACCENT if i == self.active_brush else self.PANEL_BORDER
            d.rounded_rectangle([bx, by, bx + cell, by + cell], radius=8,
                                fill=(22, 22, 36), outline=outline_c)
            # Draw stroke preview
            if i < len(brush_preview_styles):
                color, width, style = brush_preview_styles[i]
                py1 = by + cell // 2
                if style == "wavy":
                    pts = []
                    for sx in range(bx + 8, bx + cell - 8, 3):
                        sy = py1 + int(4 * math.sin((sx - bx) * 0.5))
                        pts.append((sx, sy))
                    if len(pts) >= 2:
                        d.line(pts, fill=color, width=width)
                elif style == "dotted":
                    for sx in range(bx + 10, bx + cell - 8, 6):
                        d.ellipse([sx - 1, py1 - 1, sx + 1, py1 + 1], fill=color)
                else:
                    d.line([(bx + 8, py1 + 4), (bx + cell - 8, py1 - 4)],
                           fill=color, width=width)

        # Brush Size
        bs_y = bp_y + (len(self.BRUSH_NAMES) // cols + 1) * (cell + gap) + 8
        d.text((lp_x1 + 16, bs_y), "BRUSH SIZE", fill=self.TEXT_DIM, font=font_sm)
        d.text((lp_x2 - 44, bs_y), f"{self.brush_size}px", fill=self.TEXT_PRIMARY, font=font_sm)
        bs_y += 18

        # Slider track
        sl_x1 = lp_x1 + 16
        sl_x2 = lp_x2 - 16
        sl_cy = bs_y + 6
        self._slider_rect = (sl_x1, sl_cy - 12, sl_x2, sl_cy + 12)
        d.rounded_rectangle([sl_x1, sl_cy - 2, sl_x2, sl_cy + 2], radius=2, fill=self.PANEL_BORDER)
        # Slider thumb
        t_frac = min(self.brush_size / 100.0, 1.0)
        thumb_x = int(sl_x1 + t_frac * (sl_x2 - sl_x1))
        d.ellipse([thumb_x - 6, sl_cy - 6, thumb_x + 6, sl_cy + 6], fill=self.ACCENT)

        # Size preset dots
        bs_y += 20
        preset_sizes = [4, 12, 16, 28, 40, 56]
        self._size_preset_rects = []
        for j, ps in enumerate(preset_sizes):
            px = lp_x1 + 14 + j * 26
            pr = max(3, ps // 6)
            
            # Save hit box for gesture selection
            hit_r = max(12, pr + 6)
            self._size_preset_rects.append((px - hit_r, bs_y + 6 - hit_r, px + hit_r, bs_y + 6 + hit_r, ps))
            
            fill_c = self.ACCENT if ps == self.brush_size else self.TEXT_DIM
            d.ellipse([px - pr, bs_y + 6 - pr, px + pr, bs_y + 6 + pr], fill=fill_c)

        # ── Right Panel ─────────────────────────────────────────
        rp_x1 = self.width - self.right_w - self.pad
        rp_y1 = self.top_h + self.pad * 2
        rp_x2 = self.width - self.pad
        rp_y2 = self.height - self.bottom_h - self.pad * 2

        d.rounded_rectangle([rp_x1, rp_y1, rp_x2, rp_y2],
                            radius=r, fill=self.PANEL, outline=self.PANEL_BORDER)

        # Colors header
        cy = rp_y1 + 12
        d.text((rp_x1 + 16, cy), "COLORS", fill=self.TEXT_DIM, font=font_sm)
        cy += 20

        # ── Positions for color wheel overlay (placed on numpy later) ──
        wheel_size = self._color_wheel.shape[0]  # 130
        self._wheel_x = (rp_x1 + rp_x2) // 2 - wheel_size // 2
        self._wheel_y = cy + 5

        # SV square inside wheel
        sv_size = self._sv_square.shape[0]  # 56
        self._sv_x = self._wheel_x + (wheel_size - sv_size) // 2
        self._sv_y = self._wheel_y + (wheel_size - sv_size) // 2

        # Hue bar below wheel
        hue_bar_w = self._hue_bar.shape[1]
        self._hue_bar_x = rp_x1 + 16
        self._hue_bar_y = self._wheel_y + wheel_size + 10

        # Hex value
        hex_y = self._hue_bar_y + 18
        b, g, rv = self.active_color
        hex_str = f"#{rv:02X}{g:02X}{b:02X}"
        d.rounded_rectangle([rp_x1 + 16, hex_y, rp_x2 - 16, hex_y + 18],
                            radius=6, fill=(22, 22, 36), outline=self.PANEL_BORDER)
        d.text((rp_x1 + 26, hex_y + 2), hex_str, fill=self.TEXT_PRIMARY, font=font_sm)

        # ── Recent Colors (store positions for hit testing) ─────
        rc_y = hex_y + 28
        d.text((rp_x1 + 16, rc_y), "RECENT COLORS", fill=self.TEXT_DIM, font=font_sm)
        rc_y += 18

        self._color_centers = []
        color_cols = 5
        color_spacing = 28
        for i, (col, _) in enumerate(self.PALETTE_COLORS):
            row = i // color_cols
            coli = i % color_cols
            cx = rp_x1 + 30 + coli * color_spacing
            ccy = rc_y + row * color_spacing
            self._color_centers.append((cx, ccy))
            b_c, g_c, r_c = col
            d.ellipse([cx - 9, ccy - 9, cx + 9, ccy + 9], fill=(r_c, g_c, b_c))
            if i == self.active_color_idx:
                d.ellipse([cx - 11, ccy - 11, cx + 11, ccy + 11],
                          outline=(255, 255, 255), width=2)

        # ── Layers ──────────────────────────────────────────────
        ly_y = rc_y + (len(self.PALETTE_COLORS) // color_cols + 1) * color_spacing + 8
        d.text((rp_x1 + 16, ly_y), "LAYERS", fill=self.TEXT_DIM, font=font_sm)
        d.text((rp_x2 - 24, ly_y), "+", fill=self.ACCENT, font=font_lg)
        ly_y += 20

        layer_names = ["Layer 1", "Background"]
        for i, lname in enumerate(layer_names):
            ly = ly_y + i * 26
            bg = (25, 25, 40) if i == 0 else self.PANEL
            d.rounded_rectangle([rp_x1 + 8, ly, rp_x2 - 8, ly + 22], radius=6, fill=bg)
            d.text((rp_x1 + 32, ly + 4), lname, fill=self.TEXT_PRIMARY if i == 0 else self.TEXT_DIM, font=font_sm)

        # ── Bottom Bar ──────────────────────────────────────────
        bb_y1 = self.height - self.bottom_h - self.pad
        bb_y2 = self.height - self.pad
        d.rounded_rectangle([self.pad, bb_y1, self.width - self.pad, bb_y2],
                            radius=r, fill=self.PANEL, outline=self.PANEL_BORDER)

        # Canvas size
        d.text((self.pad + 16, bb_y1 + 10), "CANVAS", fill=self.TEXT_DIM, font=font_sm)
        d.rounded_rectangle([self.pad + 70, bb_y1 + 8, self.pad + 170, bb_y1 + 28],
                            radius=6, fill=(22, 22, 36), outline=self.PANEL_BORDER)
        d.text((self.pad + 78, bb_y1 + 11), f"{self.video_width} x {self.video_height}",
               fill=self.TEXT_PRIMARY, font=font_sm)

        # Action buttons
        actions = [("Clear", "C"), ("Undo", "U"), ("Redo", "R")]
        act_start = self.width // 2 - 150
        for j, (label, key) in enumerate(actions):
            ax = act_start + j * 75
            ay = bb_y1 + 6
            d.rounded_rectangle([ax, ay, ax + 60, ay + 36], radius=10,
                                fill=(22, 22, 36), outline=self.PANEL_BORDER)
            d.text((ax + 22, ay + 4), key, fill=self.TEXT_PRIMARY, font=font_lg)
            d.text((ax + 10, ay + 22), label, fill=self.TEXT_DIM, font=font_sm)

        # Tracking indicator
        tk_x = self.width // 2 + 90
        tk_cy = (bb_y1 + bb_y2) // 2
        d.ellipse([tk_x - 18, tk_cy - 18, tk_x + 18, tk_cy + 18],
                  fill=self.ACCENT_DIM, outline=self.ACCENT, width=2)
        d.text((tk_x - 16, tk_cy + 20), "Tracking", fill=self.ACCENT, font=font_sm)

        # Save / Export
        for j, label in enumerate(["Save", "Export"]):
            ax = tk_x + 50 + j * 75
            ay = bb_y1 + 6
            d.rounded_rectangle([ax, ay, ax + 60, ay + 36], radius=10,
                                fill=(22, 22, 36), outline=self.PANEL_BORDER)
            d.text((ax + 10, ay + 4), label[0], fill=self.TEXT_PRIMARY, font=font_lg)
            d.text((ax + 8, ay + 22), label, fill=self.TEXT_DIM, font=font_sm)

        # Zoom
        zoom_x = self.width - self.pad - 190
        d.text((zoom_x, bb_y1 + 14), "ZOOM", fill=self.TEXT_DIM, font=font_sm)
        d.rounded_rectangle([zoom_x + 50, bb_y1 + 14, zoom_x + 120, bb_y1 + 26],
                            radius=4, fill=self.PANEL_BORDER)
        # Zoom thumb
        d.ellipse([zoom_x + 82, bb_y1 + 12, zoom_x + 92, bb_y1 + 28], fill=self.ACCENT)
        d.text((zoom_x + 130, bb_y1 + 14), "100%", fill=self.TEXT_PRIMARY, font=font_sm)

        # ── Convert to OpenCV BGR and overlay generated assets ──
        base_np = np.array(img)
        base_bgr = cv2.cvtColor(base_np, cv2.COLOR_RGB2BGR)

        # Overlay color wheel
        wh, ww = self._color_wheel.shape[:2]
        wy, wx = self._wheel_y, self._wheel_x
        if 0 <= wy and wy + wh <= self.height and 0 <= wx and wx + ww <= self.width:
            roi = base_bgr[wy:wy + wh, wx:wx + ww]
            mask = self._wheel_mask
            roi[mask > 0] = self._color_wheel[mask > 0]

        # Overlay SV square
        sv_h, sv_w = self._sv_square.shape[:2]
        sy, sx = self._sv_y, self._sv_x
        if 0 <= sy and sy + sv_h <= self.height and 0 <= sx and sx + sv_w <= self.width:
            base_bgr[sy:sy + sv_h, sx:sx + sv_w] = self._sv_square

        # Overlay hue bar
        hb_h, hb_w = self._hue_bar.shape[:2]
        hy, hx = self._hue_bar_y, self._hue_bar_x
        if 0 <= hy and hy + hb_h <= self.height and 0 <= hx and hx + hb_w <= self.width:
            base_bgr[hy:hy + hb_h, hx:hx + hb_w] = self._hue_bar

        return base_bgr

    # ── Hand landmark drawing ───────────────────────────────────

    def _draw_hand_landmarks(self, frame, landmarks, frame_w, frame_h):
        """Draw neon-coloured hand skeleton on a camera frame."""
        if not DRAW_HAND_LANDMARKS or not landmarks:
            return

        # Convert normalised landmarks to pixel coords
        pts = []
        for lm in landmarks:
            px = int(lm.x * frame_w)
            py = int(lm.y * frame_h)
            pts.append((px, py))

        # Draw connections (glow line)
        for (a, b) in self.HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b],
                         LANDMARK_CONNECTION_COLOR,
                         LANDMARK_LINE_THICKNESS + 2, cv2.LINE_AA)
                cv2.line(frame, pts[a], pts[b],
                         LANDMARK_COLOR,
                         LANDMARK_LINE_THICKNESS, cv2.LINE_AA)

        # Draw landmark dots
        for i, (px, py) in enumerate(pts):
            r = LANDMARK_DOT_RADIUS + 1 if i == 8 else LANDMARK_DOT_RADIUS
            cv2.circle(frame, (px, py), r + 2, LANDMARK_CONNECTION_COLOR, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), r, (255, 255, 255), -1, cv2.LINE_AA)

        # Special highlight on index fingertip (landmark 8)
        if len(pts) > 8:
            cv2.circle(frame, pts[8], LANDMARK_DOT_RADIUS + 5,
                       self.ACCENT, 2, cv2.LINE_AA)

    # ── Dynamic rendering (per frame) ───────────────────────────

    def render(self, frame, canvas_frame, cursor_x, cursor_y, gesture,
              landmarks=None):
        ui = self._base.copy()

        # Resize frames to display size
        disp_frame = cv2.resize(frame, (self.disp_w, self.disp_h))

        # Draw hand landmarks on the display frame
        if landmarks:
            self._draw_hand_landmarks(disp_frame, landmarks,
                                      self.disp_w, self.disp_h)

        # Resize canvas and merge with camera
        if (canvas_frame.shape[1] != self.disp_w or
                canvas_frame.shape[0] != self.disp_h):
            disp_canvas = cv2.resize(canvas_frame, (self.disp_w, self.disp_h))
        else:
            disp_canvas = canvas_frame

        gray = cv2.cvtColor(disp_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        frame_bg = cv2.bitwise_and(disp_frame, disp_frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(disp_canvas, disp_canvas, mask=mask)
        merged = cv2.add(frame_bg, canvas_fg)

        # Apply rounded corners
        vx, vy = self.video_x, self.video_y
        bg_region = ui[vy:vy + self.disp_h, vx:vx + self.disp_w]
        mask_inv_r = cv2.bitwise_not(self._rounded_mask)
        bg_corners = cv2.bitwise_and(bg_region, bg_region, mask=mask_inv_r)
        merged_rounded = cv2.bitwise_and(merged, merged, mask=self._rounded_mask)
        combined = cv2.add(bg_corners, merged_rounded)
        ui[vy:vy + self.disp_h, vx:vx + self.disp_w] = combined

        # Thin border around camera region
        cv2.rectangle(ui, (vx - 1, vy - 1),
                      (vx + self.disp_w, vy + self.disp_h),
                      self.PANEL_BORDER, 1)

        # ── LIVE badge ──────────────────────────────────────────
        badge_x, badge_y = vx + 10, vy + 10
        cv2.rectangle(ui, (badge_x, badge_y), (badge_x + 58, badge_y + 22),
                      (15, 40, 15), cv2.FILLED)
        cv2.rectangle(ui, (badge_x, badge_y), (badge_x + 58, badge_y + 22),
                      (30, 80, 30), 1)
        cv2.circle(ui, (badge_x + 12, badge_y + 11), 4,
                   self.LIVE_GREEN, cv2.FILLED, cv2.LINE_AA)
        cv2.putText(ui, "LIVE", (badge_x + 22, badge_y + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.LIVE_GREEN, 1, cv2.LINE_AA)

        # Hand tracking status
        if gesture != "NONE":
            cv2.putText(ui, "Hand Tracking Active", (badge_x + 66, badge_y + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.TEXT_SECONDARY, 1, cv2.LINE_AA)

        # ── Gesture Guide overlay ───────────────────────────────
        guide_h = 48
        guide_y = vy + self.disp_h - guide_h - 6
        guide_x = vx + 8
        guide_w = self.disp_w - 16

        # Semi-transparent background
        if guide_y >= 0 and guide_y + guide_h <= self.height:
            ov_region = ui[guide_y:guide_y + guide_h, guide_x:guide_x + guide_w]
            dark = np.full_like(ov_region, (12, 12, 20), dtype=np.uint8)
            cv2.addWeighted(dark, 0.78, ov_region, 0.22, 0, dst=ov_region)

            cv2.putText(ui, "GESTURE GUIDE", (guide_x + 8, guide_y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, self.TEXT_DIM, 1, cv2.LINE_AA)

            item_w = guide_w // len(self.GESTURE_GUIDE)
            for i, (gname, gdesc) in enumerate(self.GESTURE_GUIDE):
                ix = guide_x + 8 + i * item_w
                # Highlight active gesture
                is_active = (
                    gesture == gname.upper().split("/")[0].strip().split(" ")[0]
                    or (gesture == "SELECT" and "Select" in gname)
                    or (gesture == "SAVE" and "Save" in gname)
                )
                col = self.ACCENT_GLOW if is_active else self.TEXT_PRIMARY
                cv2.putText(ui, gname, (ix, guide_y + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)
                cv2.putText(ui, gdesc, (ix, guide_y + 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, self.TEXT_DIM, 1, cv2.LINE_AA)

        # ── Cursor ──────────────────────────────────────────────
        if cursor_x > 0 and cursor_y > 0:
            if gesture == "DRAW":
                b, g, rv = self.active_color
                cv2.circle(ui, (cursor_x, cursor_y), 14,
                           (b // 3, g // 3, rv // 3), 2, cv2.LINE_AA)
                cv2.circle(ui, (cursor_x, cursor_y), 6,
                           self.active_color, cv2.FILLED, cv2.LINE_AA)
            else:
                cv2.circle(ui, (cursor_x, cursor_y), 10,
                           self.NEON_CYAN, 2, cv2.LINE_AA)
                cv2.circle(ui, (cursor_x, cursor_y), 3,
                           (255, 255, 255), cv2.FILLED, cv2.LINE_AA)

        # ── Active gesture status on top bar ────────────────────
        cv2.putText(ui, f"Active: {gesture}", (self.left_w + 30, self.top_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.NEON_CYAN, 1, cv2.LINE_AA)

        return ui
