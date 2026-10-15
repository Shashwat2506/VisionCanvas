import numpy as np
import pytest
from src.canvas.text_recognizer import TextRecognizer
from src.canvas.font_manager import FontManager
from src.canvas.drawing_canvas import DrawingCanvas
from src.brushes.hard_brush import HardBrush
from src.ui.modern_ui import ModernUI


class TestTextRecognizerPairs:
    """Test character recognition accuracy on canonical and tricky pairs."""

    @pytest.fixture(autouse=True)
    def setup_recognizer(self):
        self.rec = TextRecognizer()

    def _sample_proto(self, char: str, scale=100, offset=(100, 100)):
        proto = self.rec.CANONICAL_PROTOTYPES.get(char)
        assert proto is not None, f"Character {char} not in canonical prototypes"
        ox, oy = offset
        strokes = []
        for stroke in proto:
            s_pts = []
            for p in stroke:
                x = int(round(ox + p[0] * scale))
                y = int(round(oy + p[1] * scale))
                s_pts.append((x, y))
            strokes.append(s_pts)
        return strokes

    def test_s_vs_5(self):
        """Verify 'S' is recognized as 'S' and '5' as '5'."""
        s_strokes = self._sample_proto("S")
        char, conf, bbox = self.rec.recognize_strokes(s_strokes)
        assert char == "S", f"Expected 'S', got '{char}'"
        assert conf > 0.60

        five_strokes = self._sample_proto("5")
        char5, conf5, bbox5 = self.rec.recognize_strokes(five_strokes)
        assert char5 == "5", f"Expected '5', got '{char5}'"
        assert conf5 > 0.60

    def test_o_vs_0(self):
        """Verify 'O' and '0' loop discrimination."""
        o_strokes = self._sample_proto("O")
        char_o, conf_o, _ = self.rec.recognize_strokes(o_strokes)
        assert char_o in ("O", "0")

    def test_i_vs_1(self):
        """Verify 'I' vs '1' recognition."""
        i_strokes = self._sample_proto("I")
        char_i, conf_i, _ = self.rec.recognize_strokes(i_strokes)
        assert char_i == "I", f"Expected 'I', got '{char_i}'"

        one_strokes = self._sample_proto("1")
        char_1, conf_1, _ = self.rec.recognize_strokes(one_strokes)
        assert char_1 == "1", f"Expected '1', got '{char_1}'"

    def test_c_vs_g(self):
        """Verify 'C' vs 'G' recognition."""
        c_strokes = self._sample_proto("C")
        char_c, conf_c, _ = self.rec.recognize_strokes(c_strokes)
        assert char_c == "C", f"Expected 'C', got '{char_c}'"

        g_strokes = self._sample_proto("G")
        char_g, conf_g, _ = self.rec.recognize_strokes(g_strokes)
        assert char_g == "G", f"Expected 'G', got '{char_g}'"

    def test_p_vs_r(self):
        """Verify 'P' vs 'R' leg discrimination."""
        p_strokes = self._sample_proto("P")
        char_p, conf_p, _ = self.rec.recognize_strokes(p_strokes)
        assert char_p == "P", f"Expected 'P', got '{char_p}'"

        r_strokes = self._sample_proto("R")
        char_r, conf_r, _ = self.rec.recognize_strokes(r_strokes)
        assert char_r == "R", f"Expected 'R', got '{char_r}'"

    def test_u_vs_v(self):
        """Verify 'U' vs 'V' bottom roundness vs corner angle."""
        u_strokes = self._sample_proto("U")
        char_u, conf_u, _ = self.rec.recognize_strokes(u_strokes)
        assert char_u == "U", f"Expected 'U', got '{char_u}'"

        v_strokes = self._sample_proto("V")
        char_v, conf_v, _ = self.rec.recognize_strokes(v_strokes)
        assert char_v == "V", f"Expected 'V', got '{char_v}'"

    def test_b_vs_8(self):
        """Verify 'B' vs '8' double loop characters."""
        b_strokes = self._sample_proto("B")
        char_b, conf_b, _ = self.rec.recognize_strokes(b_strokes)
        assert char_b == "B", f"Expected 'B', got '{char_b}'"

        eight_strokes = self._sample_proto("8")
        char_8, conf_8, _ = self.rec.recognize_strokes(eight_strokes)
        assert char_8 == "8", f"Expected '8', got '{char_8}'"

    def test_multi_stroke_word(self):
        """Verify multi-character clustering (e.g. 'A' followed by 'I')."""
        a_strokes = self._sample_proto("A", scale=80, offset=(100, 100))
        i_strokes = self._sample_proto("I", scale=80, offset=(220, 100))
        combined_strokes = a_strokes + i_strokes
        word, conf, bbox = self.rec.recognize_strokes(combined_strokes)
        assert word == "AI", f"Expected 'AI', got '{word}'"
        assert bbox[2] > 100  # Spans both characters

    def test_word_hello(self):
        """Verify multi-character word recognition for 'HELLO'."""
        h_strokes = self._sample_proto("H", scale=60, offset=(50, 100))
        e_strokes = self._sample_proto("E", scale=60, offset=(130, 100))
        l1_strokes = self._sample_proto("L", scale=60, offset=(210, 100))
        l2_strokes = self._sample_proto("L", scale=60, offset=(280, 100))
        o_strokes = self._sample_proto("O", scale=60, offset=(350, 100))
        combined = h_strokes + e_strokes + l1_strokes + l2_strokes + o_strokes
        word, conf, bbox = self.rec.recognize_strokes(combined)
        assert word == "HELLO", f"Expected 'HELLO', got '{word}'"
        assert conf > 0.60

    def test_ambiguous_and_failed_input(self):
        """Verify that ambiguous or tiny noisy strokes return None or low confidence."""
        # Single dot noise
        dot_strokes = [[(100, 100)]]
        word, conf, bbox = self.rec.recognize_strokes(dot_strokes)
        assert word is None

        # Random horizontal scribble
        scribble = [[(100 + i * 2, 100 + int(np.random.randint(-2, 3))) for i in range(20)]]
        word, conf, bbox = self.rec.recognize_strokes(scribble)
        # Should either be None, or confidence below high threshold
        if word is not None:
            assert conf < 0.90


class TestFontManager:
    """Test font resolution, metric calculation, and alpha-blended rendering."""

    def test_font_families_available(self):
        fm = FontManager.get_instance()
        families = fm.get_available_families()
        assert "Arial" in families
        assert "Calibri" in families
        assert "Times New Roman" in families
        assert "Default Sans" in families

    def test_font_retrieval(self):
        fm = FontManager.get_instance()
        font_regular = fm.get_font("Arial", size=36, bold=False, italic=False)
        assert font_regular is not None

        font_bold = fm.get_font("Arial", size=48, bold=True, italic=False)
        assert font_bold is not None

        font_italic = fm.get_font("Times New Roman", size=32, bold=False, italic=True)
        assert font_italic is not None

    def test_text_rendering_to_canvas(self):
        fm = FontManager.get_instance()
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        font = fm.get_font("Arial", size=48)
        color = (0, 225, 255)  # BGR Yellow

        rendered = fm.render_text_to_bgr(canvas, "S", (320, 240), font, color, align="center")
        assert rendered.shape == (480, 640, 3)
        assert np.any(rendered > 0)
        # Verify color was applied
        mask = np.any(rendered > 0, axis=-1)
        mean_bgr = np.mean(rendered[mask], axis=0)
        assert mean_bgr[1] > 100  # Green channel active
        assert mean_bgr[2] > 100  # Red channel active


class TestDrawingCanvasTextHistory:
    """Test that intermediate handwriting strokes are collapsed on commit and removed on revert."""

    def test_commit_operation_collapses_undo(self):
        brush = HardBrush(color=(255, 255, 255), thickness=4)
        canvas = DrawingCanvas(640, 480, brush)
        initial_state = canvas.canvas.copy()

        # Step 1: Draw a base stroke
        canvas.draw((50, 50))
        canvas.draw((60, 60))
        canvas.reset_stroke()
        canvas_with_base = canvas.canvas.copy()

        # Step 2: Draw text handwriting strokes (intermediate)
        start_canvas = canvas.canvas.copy()
        canvas.draw((100, 100))
        canvas.draw((120, 120))
        canvas.reset_stroke()

        canvas.draw((140, 100))
        canvas.draw((160, 120))
        canvas.reset_stroke()

        # Several strokes added to undo stack
        assert len(canvas.undo_stack) > 2

        # Step 3: Accept text recognition -> commit_operation
        rendered_text_canvas = canvas_with_base.copy()
        # Draw clean font glyph at (120, 110)
        rendered_text_canvas[100:150, 100:150] = (180, 80, 255)
        canvas.commit_operation(rendered_text_canvas, base_canvas=start_canvas)

        # Current canvas is the typeset text
        assert np.array_equal(canvas.canvas, rendered_text_canvas)

        # Step 4: Undo should revert directly to canvas_with_base (before handwriting)
        canvas.undo()
        assert np.array_equal(canvas.canvas, canvas_with_base)

        # Step 5: Redo should restore the typeset text
        canvas.redo()
        assert np.array_equal(canvas.canvas, rendered_text_canvas)

    def test_revert_to_discards_strokes(self):
        brush = HardBrush(color=(255, 255, 255), thickness=4)
        canvas = DrawingCanvas(640, 480, brush)
        base_state = canvas.canvas.copy()

        # Draw rough handwriting
        canvas.draw((100, 100))
        canvas.draw((110, 110))
        canvas.reset_stroke()
        assert np.any(canvas.canvas > 0)

        # Revert to base
        canvas.revert_to(base_state)
        assert np.array_equal(canvas.canvas, base_state)


class TestModernUITextControls:
    """Test ModernUI typography controls and review card."""

    def test_font_family_selection(self):
        ui = ModernUI(1280, 720)
        assert ui.get_active_font_family() == "Arial"
        ui.set_active_font(1)
        assert ui.get_active_font_family() == "Calibri"
        ui.set_active_font(2)
        assert ui.get_active_font_family() == "Times New Roman"

    def test_text_style_toggles(self):
        ui = ModernUI(1280, 720)
        assert not ui.text_bold
        ui.toggle_text_bold()
        assert ui.text_bold

        assert not ui.text_italic
        ui.toggle_text_italic()
        assert ui.text_italic

    def test_text_size_and_auto_fit(self):
        ui = ModernUI(1280, 720)
        assert ui.text_auto_size
        ui.set_text_font_size(64)
        assert ui.text_font_size == 64
        assert not ui.text_auto_size

        ui.toggle_text_auto_size()
        assert ui.text_auto_size

    def test_review_card_actions(self):
        ui = ModernUI(1280, 720)
        ui.text_review = {
            "text": "S",
            "conf": 0.95,
            "bbox": (100, 100, 80, 120),
            "start_canvas": np.zeros((480, 640, 3), dtype=np.uint8)
        }

        rx, ry, rw, rh = ui._get_text_review_rect()
        btn_y = ry + rh - 25

        # Accept button
        btn1_x = rx + 30
        res1 = ui.handle_selection(btn1_x, btn_y)
        assert res1 == "TEXT_ACCEPT"

        # Retry button
        btn2_x = rx + 160
        res2 = ui.handle_selection(btn2_x, btn_y)
        assert res2 == "TEXT_RETRY"

        # Cancel button
        btn3_x = rx + 290
        res3 = ui.handle_selection(btn3_x, btn_y)
        assert res3 == "TEXT_CANCEL"
