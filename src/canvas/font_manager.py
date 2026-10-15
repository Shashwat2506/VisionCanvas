import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class FontManager:
    """Cross-platform font discovery, styling, caching, and anti-aliased rendering engine.

    Supports Arial, Calibri, Times New Roman, and Clean Default Sans fonts with
    Bold and Italic variants. Automatically locates installed system fonts
    across Windows, macOS, and Linux, with zero-crash graceful fallbacks.
    """

    FONT_FAMILIES = ["Arial", "Calibri", "Times New Roman", "Default Sans"]

    # Candidate file names in order of preference for each platform
    _CANDIDATES = {
        "Arial": {
            "regular": ["arial.ttf", "Arial.ttf", "LiberationSans-Regular.ttf", "DejaVuSans.ttf", "FreeSans.ttf"],
            "bold": ["arialbd.ttf", "Arial Bold.ttf", "LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf", "FreeSansBold.ttf"],
            "italic": ["ariali.ttf", "Arial Italic.ttf", "LiberationSans-Italic.ttf", "DejaVuSans-Oblique.ttf", "FreeSansOblique.ttf"],
        },
        "Calibri": {
            "regular": ["calibri.ttf", "Calibri.ttf", "Carlito-Regular.ttf", "arial.ttf", "LiberationSans-Regular.ttf"],
            "bold": ["calibrib.ttf", "Calibri Bold.ttf", "Carlito-Bold.ttf", "arialbd.ttf", "LiberationSans-Bold.ttf"],
            "italic": ["calibrii.ttf", "Calibri Italic.ttf", "Carlito-Italic.ttf", "ariali.ttf", "LiberationSans-Italic.ttf"],
        },
        "Times New Roman": {
            "regular": ["times.ttf", "Times New Roman.ttf", "LiberationSerif-Regular.ttf", "DejaVuSerif.ttf", "FreeSerif.ttf"],
            "bold": ["timesbd.ttf", "Times New Roman Bold.ttf", "LiberationSerif-Bold.ttf", "DejaVuSerif-Bold.ttf", "FreeSerifBold.ttf"],
            "italic": ["timesi.ttf", "Times New Roman Italic.ttf", "LiberationSerif-Italic.ttf", "DejaVuSerif-Italic.ttf", "FreeSerifItalic.ttf"],
        },
        "Default Sans": {
            "regular": ["segoeui.ttf", "Segoe UI.ttf", "arial.ttf", "DejaVuSans.ttf", "Ubuntu-R.ttf"],
            "bold": ["segoeuib.ttf", "Segoe UI Bold.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "Ubuntu-B.ttf"],
            "italic": ["segoeuii.ttf", "Segoe UI Italic.ttf", "ariali.ttf", "DejaVuSans-Oblique.ttf", "Ubuntu-RI.ttf"],
        },
    }

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._font_cache = {}
        self._resolved_paths = {}
        self._font_search_dirs = self._detect_font_dirs()
        self._pre_resolve_fonts()

    def _detect_font_dirs(self):
        dirs = []
        if sys.platform == "win32":
            windir = os.environ.get("WINDIR", r"C:\Windows")
            dirs.append(os.path.join(windir, "Fonts"))
            user_fonts = os.path.expanduser(r"~\AppData\Local\Microsoft\Windows\Fonts")
            if os.path.exists(user_fonts):
                dirs.append(user_fonts)
        elif sys.platform == "darwin":
            dirs.extend([
                "/System/Library/Fonts",
                "/Library/Fonts",
                os.path.expanduser("~/Library/Fonts"),
            ])
        else:
            dirs.extend([
                "/usr/share/fonts",
                "/usr/local/share/fonts",
                os.path.expanduser("~/.fonts"),
                os.path.expanduser("~/.local/share/fonts"),
            ])
        return [d for d in dirs if os.path.exists(d)]

    def _find_font_file(self, candidate_names):
        """Search system directories and PIL search path for the first available font."""
        for name in candidate_names:
            # Check system font directories
            for d in self._font_search_dirs:
                full_path = os.path.join(d, name)
                if os.path.exists(full_path):
                    return full_path
            # Check by raw filename (PIL font loader checks system registry on Windows)
            try:
                ImageFont.truetype(name, 12)
                return name
            except Exception:
                continue
        return None

    def _pre_resolve_fonts(self):
        """Map each family and style to an available file path or font spec."""
        for family, styles in self._CANDIDATES.items():
            self._resolved_paths[family] = {}
            for style, candidates in styles.items():
                found = self._find_font_file(candidates)
                self._resolved_paths[family][style] = found

    def get_available_families(self):
        return list(self.FONT_FAMILIES)

    def get_resolved_font_name(self, family: str) -> str:
        """Return human-readable resolved font name."""
        res = self._resolved_paths.get(family, {}).get("regular")
        if res:
            base = os.path.basename(res).replace(".ttf", "").replace(".otf", "")
            return base.replace("-Regular", "")
        return "Default"

    def get_font(self, family: str = "Arial", size: int = 48,
                 bold: bool = False, italic: bool = False) -> ImageFont.ImageFont:
        """Load and cache a TrueType font matching family, size, and styling."""
        size = max(8, min(240, int(size)))
        if family not in self._resolved_paths:
            family = "Arial"

        style_key = "regular"
        if bold and italic:
            # Check if italic or bold is resolved
            style_key = "bold" if self._resolved_paths[family].get("bold") else "italic"
        elif bold:
            style_key = "bold"
        elif italic:
            style_key = "italic"

        cache_key = (family, size, bold, italic)
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        font_source = self._resolved_paths.get(family, {}).get(style_key)
        if not font_source:
            # Fall back to regular style
            font_source = self._resolved_paths.get(family, {}).get("regular")

        font = None
        if font_source:
            try:
                font = ImageFont.truetype(font_source, size)
            except Exception:
                pass

        if font is None:
            # Try Arial or default truetype
            for fallback in ["arial.ttf", "segoeui.ttf", "times.ttf"]:
                try:
                    font = ImageFont.truetype(fallback, size)
                    break
                except Exception:
                    continue

        if font is None:
            font = ImageFont.load_default()

        self._font_cache[cache_key] = font
        return font

    def measure_text(self, text: str, font: ImageFont.ImageFont):
        """Return (width, height, left_offset, top_offset) for the given text."""
        dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        d = ImageDraw.Draw(dummy)
        bbox = d.textbbox((0, 0), text, font=font)
        left, top, right, bottom = bbox
        w = max(1, right - left)
        h = max(1, bottom - top)
        return w, h, left, top

    def render_text_to_bgr(self, base_canvas: np.ndarray, text: str,
                           anchor_xy: tuple, font: ImageFont.ImageFont,
                           color_bgr: tuple, align: str = "center") -> np.ndarray:
        """Render anti-aliased font glyphs with transparent background onto canvas."""
        canvas_h, canvas_w = base_canvas.shape[:2]
        tw, th, left_off, top_off = self.measure_text(text, font)

        ax, ay = anchor_xy
        if align == "center":
            pos_x = int(round(ax - tw / 2.0 - left_off))
            pos_y = int(round(ay - th / 2.0 - top_off))
        else:  # "topleft"
            pos_x = int(round(ax - left_off))
            pos_y = int(round(ay - top_off))

        # Clamp position gracefully within canvas
        pos_x = max(2, min(canvas_w - tw - 4, pos_x))
        pos_y = max(2, min(canvas_h - th - 4, pos_y))

        # Create transparent PIL canvas
        overlay = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # BGR to RGB
        b, g, r = color_bgr[:3]
        draw.text((pos_x, pos_y), text, fill=(int(r), int(g), int(b), 255), font=font)

        # Alpha composite cleanly onto base_canvas
        overlay_np = np.array(overlay)
        alpha = overlay_np[..., 3].astype(np.float32) / 255.0

        if not np.any(alpha > 0):
            return base_canvas.copy()

        out_canvas = base_canvas.copy().astype(np.float32)
        rgb_layer = overlay_np[..., :3]
        bgr_layer = rgb_layer[..., ::-1].astype(np.float32)

        for c in range(3):
            out_canvas[..., c] = out_canvas[..., c] * (1.0 - alpha) + bgr_layer[..., c] * alpha

        return np.clip(out_canvas, 0, 255).astype(np.uint8)
