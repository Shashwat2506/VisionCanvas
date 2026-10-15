import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.canvas.font_manager import FontManager


class TextRecognizer:
    """Offline, high-accuracy handwriting-to-text recognition engine.

    Recognizes air-drawn uppercase letters (A-Z), numbers (0-9), and words
    using bidirectional point-cloud matching (BINN), curvature and corner analysis,
    topological loop analysis, and targeted geometric disambiguation.

    Effectively distinguishes tricky pairs:
    - S and 5
    - O and 0
    - I and 1
    - C and G
    - P and R
    - U and V
    - B and 8

    100% offline, privacy-safe, with zero external network or paid API requirements.
    """

    CHAR_SET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    # Canonical handwritten stroke prototypes for A-Z and 0-9
    CANONICAL_PROTOTYPES = {
        # --- Letters ---
        "A": [[(0.15, 0.9), (0.5, 0.1), (0.85, 0.9)], [(0.3, 0.58), (0.7, 0.58)]],
        "B": [[(0.2, 0.1), (0.2, 0.9)], [(0.2, 0.1), (0.6, 0.1), (0.8, 0.3), (0.6, 0.5), (0.2, 0.5)], [(0.2, 0.5), (0.65, 0.5), (0.85, 0.7), (0.65, 0.9), (0.2, 0.9)]],
        "C": [[(0.85, 0.2), (0.5, 0.1), (0.15, 0.3), (0.15, 0.7), (0.5, 0.9), (0.85, 0.8)]],
        "D": [[(0.2, 0.1), (0.2, 0.9)], [(0.2, 0.1), (0.6, 0.1), (0.85, 0.35), (0.85, 0.65), (0.6, 0.9), (0.2, 0.9)]],
        "E": [[(0.85, 0.1), (0.2, 0.1), (0.2, 0.9), (0.85, 0.9)], [(0.2, 0.5), (0.7, 0.5)]],
        "F": [[(0.85, 0.1), (0.2, 0.1), (0.2, 0.9)], [(0.2, 0.5), (0.7, 0.5)]],
        "G": [[(0.85, 0.2), (0.5, 0.1), (0.15, 0.3), (0.15, 0.7), (0.5, 0.9), (0.85, 0.8), (0.85, 0.5), (0.5, 0.5)]],
        "H": [[(0.2, 0.1), (0.2, 0.9)], [(0.8, 0.1), (0.8, 0.9)], [(0.2, 0.5), (0.8, 0.5)]],
        "I": [[(0.5, 0.1), (0.5, 0.5), (0.5, 0.9)]],
        "J": [[(0.7, 0.1), (0.7, 0.75), (0.5, 0.9), (0.3, 0.8), (0.2, 0.6)]],
        "K": [[(0.2, 0.1), (0.2, 0.9)], [(0.8, 0.1), (0.2, 0.55), (0.85, 0.9)]],
        "L": [[(0.2, 0.1), (0.2, 0.9), (0.85, 0.9)]],
        "M": [[(0.15, 0.9), (0.15, 0.1), (0.5, 0.65), (0.85, 0.1), (0.85, 0.9)]],
        "N": [[(0.2, 0.9), (0.2, 0.1), (0.8, 0.9), (0.8, 0.1)]],
        "O": [[(0.5 + 0.35 * math.cos(t), 0.5 + 0.4 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 24)]],
        "P": [[(0.2, 0.1), (0.2, 0.9)], [(0.2, 0.1), (0.65, 0.1), (0.85, 0.3), (0.65, 0.5), (0.2, 0.5)]],
        "Q": [[(0.5 + 0.35 * math.cos(t), 0.5 + 0.4 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 24)], [(0.55, 0.65), (0.85, 0.95)]],
        "R": [[(0.2, 0.1), (0.2, 0.9)], [(0.2, 0.1), (0.65, 0.1), (0.85, 0.3), (0.65, 0.5), (0.2, 0.5)], [(0.45, 0.5), (0.85, 0.9)]],
        "S": [[(0.85, 0.15), (0.55, 0.1), (0.2, 0.2), (0.2, 0.35), (0.4, 0.48), (0.7, 0.55), (0.85, 0.68), (0.75, 0.85), (0.45, 0.9), (0.15, 0.82)]],
        "T": [[(0.15, 0.12), (0.85, 0.12)], [(0.5, 0.12), (0.5, 0.9)]],
        "U": [[(0.2, 0.1), (0.2, 0.7), (0.35, 0.9), (0.5, 0.93), (0.65, 0.9), (0.8, 0.7), (0.8, 0.1)]],
        "V": [[(0.2, 0.1), (0.5, 0.9), (0.8, 0.1)]],
        "W": [[(0.15, 0.1), (0.35, 0.9), (0.5, 0.45), (0.65, 0.9), (0.85, 0.1)]],
        "X": [[(0.2, 0.1), (0.8, 0.9)], [(0.8, 0.1), (0.2, 0.9)]],
        "Y": [[(0.2, 0.1), (0.5, 0.5), (0.8, 0.1)], [(0.5, 0.5), (0.5, 0.9)]],
        "Z": [[(0.2, 0.15), (0.8, 0.15), (0.2, 0.85), (0.8, 0.85)]],
        # --- Digits ---
        "0": [[(0.5 + 0.35 * math.cos(t), 0.5 + 0.4 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 24)]],
        "1": [[(0.35, 0.25), (0.5, 0.1), (0.5, 0.9)]],
        "2": [[(0.2, 0.3), (0.35, 0.12), (0.65, 0.12), (0.8, 0.3), (0.2, 0.88), (0.85, 0.88)]],
        "3": [[(0.2, 0.15), (0.8, 0.15), (0.45, 0.48), (0.8, 0.6), (0.65, 0.88), (0.2, 0.85)]],
        "4": [[(0.7, 0.88), (0.7, 0.1), (0.15, 0.65), (0.85, 0.65)]],
        "5": [[(0.85, 0.1), (0.2, 0.1), (0.2, 0.48), (0.65, 0.48), (0.85, 0.68), (0.65, 0.9), (0.3, 0.9), (0.15, 0.8)]],
        "6": [[(0.75, 0.15), (0.35, 0.4), (0.2, 0.65), (0.4, 0.9), (0.75, 0.85), (0.8, 0.65), (0.5, 0.5), (0.25, 0.65)]],
        "7": [[(0.15, 0.12), (0.85, 0.12), (0.45, 0.9)]],
        "8": [[(0.5 + 0.25 * math.sin(2 * t), 0.5 - 0.38 * math.cos(t)) for t in np.linspace(0, 2 * math.pi, 36)]],
        "9": [[(0.75, 0.35), (0.5, 0.5), (0.2, 0.35), (0.25, 0.15), (0.65, 0.12), (0.8, 0.35), (0.7, 0.75), (0.35, 0.9)]],
    }

    def __init__(self):
        self.font_manager = FontManager()
        self._canonical_clouds = {}
        for ch, strokes in self.CANONICAL_PROTOTYPES.items():
            self._canonical_clouds[ch] = self._resample_all_strokes(strokes, n=36)

    def _resample_all_strokes(self, strokes, n: int = 36) -> np.ndarray:
        """Resample a list of strokes to N normalized equidistant points in [-0.5, 0.5] box."""
        lens = []
        for s in strokes:
            s_arr = np.array(s, dtype=np.float32)
            if len(s_arr) > 1:
                diffs = s_arr[1:] - s_arr[:-1]
                lens.append(float(np.sqrt((diffs ** 2).sum(axis=1)).sum()))
            else:
                lens.append(0.0)

        total_len = sum(lens)
        all_resampled = []

        if total_len < 1e-4:
            for s in strokes:
                all_resampled.extend(s)
            if not all_resampled:
                return np.zeros((n, 2), dtype=np.float32)
            pts = np.array(all_resampled, dtype=np.float32)
            return np.repeat(pts[:1], n, axis=0)

        for i, s in enumerate(strokes):
            s_arr = np.array(s, dtype=np.float32)
            if len(s_arr) == 0:
                continue
            n_pts = max(2, int(round((lens[i] / max(1e-4, total_len)) * n)))
            if len(s_arr) == 1:
                all_resampled.extend([s_arr[0]] * n_pts)
                continue

            diffs = s_arr[1:] - s_arr[:-1]
            seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
            cum_lens = np.zeros(len(s_arr), dtype=np.float32)
            cum_lens[1:] = np.cumsum(seg_lens)
            total_seg = cum_lens[-1]
            if total_seg < 1e-4:
                all_resampled.extend([s_arr[0]] * n_pts)
                continue

            target = np.linspace(0, total_seg, n_pts)
            res = np.zeros((n_pts, 2), dtype=np.float32)
            res[:, 0] = np.interp(target, cum_lens, s_arr[:, 0])
            res[:, 1] = np.interp(target, cum_lens, s_arr[:, 1])
            all_resampled.extend(res)

        pts = np.array(all_resampled, dtype=np.float32)
        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)
        w = max(1e-4, float(max_xy[0] - min_xy[0]))
        h = max(1e-4, float(max_xy[1] - min_xy[1]))
        norm_pts = np.zeros_like(pts)
        norm_pts[:, 0] = (pts[:, 0] - min_xy[0]) / w - 0.5
        norm_pts[:, 1] = (pts[:, 1] - min_xy[1]) / h - 0.5
        return norm_pts.astype(np.float32)

    def _cloud_distance(self, cloud1: np.ndarray, cloud2: np.ndarray) -> float:
        """Bidirectional Nearest-Neighbor Point-Cloud distance."""
        dists1 = np.min(np.linalg.norm(cloud1[:, None, :] - cloud2[None, :, :], axis=2), axis=1)
        dists2 = np.min(np.linalg.norm(cloud2[:, None, :] - cloud1[None, :, :], axis=2), axis=1)
        return float((np.mean(dists1) + np.mean(dists2)) / 2.0)

    def recognize_strokes(self, stroke_list):
        """Recognize a list of air-drawn strokes into text.

        Returns: (recognized_text, confidence, (bbox_x, bbox_y, bbox_w, bbox_h))
        """
        if not stroke_list:
            return None, 0.0, None

        valid_strokes = [s for s in stroke_list if len(s) >= 2]
        if not valid_strokes:
            valid_strokes = [s for s in stroke_list if len(s) >= 1]
            if not valid_strokes:
                return None, 0.0, None

        all_pts = []
        for s in valid_strokes:
            all_pts.extend(s)

        if len(all_pts) < 2:
            return None, 0.0, None

        pts_arr = np.array(all_pts, dtype=np.int32)
        x_min, y_min = int(pts_arr[:, 0].min()), int(pts_arr[:, 1].min())
        x_max, y_max = int(pts_arr[:, 0].max()), int(pts_arr[:, 1].max())
        bw = max(1, x_max - x_min)
        bh = max(1, y_max - y_min)
        overall_bbox = (x_min, y_min, bw, bh)

        # ── Cluster strokes into individual characters (Word Segmentation) ──
        clusters = self._cluster_strokes(valid_strokes, bh)

        if len(clusters) > 1 and (bw / max(1.0, float(bh))) > 1.20:
            word_chars = []
            word_confs = []
            for cluster in clusters:
                ch_pred, conf = self._classify_single_character(cluster["strokes"])
                if ch_pred:
                    word_chars.append(ch_pred)
                    word_confs.append(conf)

            if word_chars:
                recognized = "".join(word_chars)
                avg_conf = float(sum(word_confs) / len(word_confs))
                return recognized, avg_conf, overall_bbox

        # Single character recognition
        pred_char, conf = self._classify_single_character(valid_strokes)
        if pred_char:
            return pred_char, conf, overall_bbox

        return None, 0.0, None

    def _cluster_strokes(self, strokes, char_height):
        """Group strokes belonging to the same character by horizontal bounding proximity."""
        stroke_infos = []
        for s in strokes:
            s_arr = np.array(s, dtype=np.int32)
            min_x, max_x = int(s_arr[:, 0].min()), int(s_arr[:, 0].max())
            min_y, max_y = int(s_arr[:, 1].min()), int(s_arr[:, 1].max())
            stroke_infos.append({
                "stroke": s,
                "bbox": (min_x, min_y, max_x, max_y),
            })

        stroke_infos.sort(key=lambda item: (item["bbox"][0] + item["bbox"][2]) / 2.0)
        threshold = max(12, int(char_height * 0.25))
        clusters = []

        for info in stroke_infos:
            bx1, by1, bx2, by2 = info["bbox"]
            matched_cluster = None
            for c in clusters:
                cx1, cy1, cx2, cy2 = c["bbox"]
                if not (bx2 < cx1 - threshold or bx1 > cx2 + threshold):
                    matched_cluster = c
                    break

            if matched_cluster is not None:
                matched_cluster["strokes"].append(info["stroke"])
                cx1, cy1, cx2, cy2 = matched_cluster["bbox"]
                matched_cluster["bbox"] = (min(cx1, bx1), min(cy1, by1), max(cx2, bx2), max(cy2, by2))
            else:
                clusters.append({
                    "strokes": [info["stroke"]],
                    "bbox": (bx1, by1, bx2, by2),
                })

        clusters.sort(key=lambda c: c["bbox"][0])
        return clusters

    def _classify_single_character(self, strokes):
        """Classify a list of strokes belonging to one character."""
        all_pts = []
        for s in strokes:
            all_pts.extend(s)

        if len(all_pts) < 2:
            return None, 0.0

        pts_np = np.array(all_pts, dtype=np.float32)
        x_min, y_min = float(pts_np[:, 0].min()), float(pts_np[:, 1].min())
        x_max, y_max = float(pts_np[:, 0].max()), float(pts_np[:, 1].max())
        w = max(1.0, x_max - x_min)
        h = max(1.0, y_max - y_min)
        aspect = w / h

        # ── 1. Topological Analysis (Holes & Loops) ──
        pad = 6
        rw = int(w + pad * 2)
        rh = int(h + pad * 2)
        char_canvas = np.zeros((rh, rw), dtype=np.uint8)

        for stroke in strokes:
            if len(stroke) < 2:
                continue
            for i in range(len(stroke) - 1):
                p1 = (int(stroke[i][0] - x_min + pad), int(stroke[i][1] - y_min + pad))
                p2 = (int(stroke[i + 1][0] - x_min + pad), int(stroke[i + 1][1] - y_min + pad))
                cv2.line(char_canvas, p1, p2, 255, 3, lineType=cv2.LINE_AA)

        contours, hierarchy = cv2.findContours(char_canvas, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        num_holes = 0
        if hierarchy is not None:
            for h_info in hierarchy[0]:
                if h_info[3] != -1:  # Enclosed hole
                    num_holes += 1

        # ── 2. Primary Stroke Geometry & Turning Angles ──
        primary_stroke = max(strokes, key=lambda s: len(s))
        s_pts = np.array(primary_stroke, dtype=np.float32)

        s_diffs = s_pts[1:] - s_pts[:-1]
        angles = np.arctan2(s_diffs[:, 1], s_diffs[:, 0])
        angle_diffs = np.diff(angles)
        angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi

        # Sharp corners (turning angle > 65 degrees)
        sharp_corners = np.where(np.abs(angle_diffs) > 1.15)[0]
        num_sharp_corners = len(sharp_corners)

        # Inflections (curvature changes sign)
        sig_turns = angle_diffs[np.abs(angle_diffs) > 0.15]
        inflections = 0
        if len(sig_turns) >= 2:
            signs = np.sign(sig_turns)
            inflections = int(np.count_nonzero(signs[:-1] != signs[1:]))

        # Normalized start and end points of primary stroke
        p_start_norm = ((s_pts[0][0] - x_min) / w, (s_pts[0][1] - y_min) / h)
        p_end_norm = ((s_pts[-1][0] - x_min) / w, (s_pts[-1][1] - y_min) / h)

        # ── 3. Bidirectional Point-Cloud Matching ──
        q_cloud = self._resample_all_strokes(strokes, n=36)

        # Detect sharp corners using stride on normalized resampled curve
        stride = 3
        corner_indices = []
        for i in range(stride, len(q_cloud) - stride):
            v_in = q_cloud[i] - q_cloud[i - stride]
            v_out = q_cloud[i + stride] - q_cloud[i]
            n_in = float(np.linalg.norm(v_in))
            n_out = float(np.linalg.norm(v_out))
            if n_in < 1e-4 or n_out < 1e-4:
                continue
            cos_a = np.dot(v_in, v_out) / (n_in * n_out)
            ang_deg = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
            if ang_deg > 65.0:
                corner_indices.append((i, ang_deg))

        # Distinct corners (clusters separated by at least 2 points)
        distinct_corners = 0
        last_c_idx = -99
        for c_idx, _ in corner_indices:
            if c_idx - last_c_idx > 2:
                distinct_corners += 1
            last_c_idx = c_idx

        scores = {}
        for ch, c_cloud in self._canonical_clouds.items():
            dist = self._cloud_distance(q_cloud, c_cloud)
            score = math.exp(-dist * 16.0)

            # Topological weighting
            if num_holes >= 2:
                if ch in ("B", "8"):
                    score += 0.30
                else:
                    score -= 0.30
            elif num_holes == 1:
                if ch in ("A", "D", "O", "P", "Q", "R", "0", "4", "6", "9"):
                    score += 0.20
                elif ch in ("I", "1", "L", "T", "C", "S", "M", "N", "V", "W", "E", "F", "H"):
                    score -= 0.25
            elif num_holes == 0:
                if ch in ("O", "0", "B", "8", "D"):
                    score -= 0.35
                elif ch in ("S", "C", "U", "V", "I", "L", "Z", "M", "N", "W"):
                    score += 0.05

            # Aspect ratio weighting
            if aspect < 0.40 and ch in ("I", "1", "L", "J"):
                score += 0.15
            elif aspect > 1.25 and ch in ("W", "M"):
                score += 0.15

            scores[ch] = max(0.001, score)

        # ── 4. Dedicated Confusing Pair Tie-Breakers ──
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top1, top2 = ranked[0][0], ranked[1][0] if len(ranked) > 1 else ""
        diff_top = abs(scores.get(top1, 0) - scores.get(top2, 0))

        # Tie-break 'S' vs '5' (only if genuine tie < 0.02)
        if {top1, top2} == {"S", "5"} and diff_top < 0.02:
            if distinct_corners <= 1:
                scores["S"] = scores.get("5", 0) + 0.05
            else:
                scores["5"] = scores.get("S", 0) + 0.05

        # Tie-break 'O' vs '0' (both loops, differentiate by aspect ratio)
        if {top1, top2} == {"O", "0"} and num_holes >= 1:
            if aspect >= 0.65:
                scores["O"] = max(scores.get("O", 0), scores.get("0", 0) + 0.08)
            else:
                scores["0"] = max(scores.get("0", 0), scores.get("O", 0) + 0.08)

        # Tie-break 'I' vs '1' (only if close tie < 0.03)
        if {top1, top2} == {"I", "1"} and diff_top < 0.03:
            has_top_flag = (p_start_norm[0] < 0.42 and p_start_norm[1] < 0.35 and distinct_corners >= 1)
            if has_top_flag:
                scores["1"] = scores.get("I", 0) + 0.05
            else:
                scores["I"] = scores.get("1", 0) + 0.05

        # Tie-break 'C' vs 'G' (only if close tie < 0.03)
        if {top1, top2} == {"C", "G"} and diff_top < 0.03:
            n_tail = max(2, int(len(s_pts) * 0.25))
            tail = s_pts[-n_tail:]
            dx_tail = float(tail[-1][0] - tail[0][0])
            if dx_tail < -0.12 * w and p_end_norm[0] < 0.80:
                scores["G"] = scores.get("C", 0) + 0.05
            else:
                scores["C"] = scores.get("G", 0) + 0.05

        # Tie-break 'P' vs 'R' (only if close tie < 0.03)
        if {top1, top2} == {"P", "R"} and diff_top < 0.03:
            has_leg = int(np.count_nonzero((q_cloud[:, 0] > 0.05) & (q_cloud[:, 1] > 0.15))) >= 2
            if has_leg:
                scores["R"] = scores.get("P", 0) + 0.05
            else:
                scores["P"] = scores.get("R", 0) + 0.05

        # Tie-break 'U' vs 'V' (only if close tie < 0.03)
        if {top1, top2} == {"U", "V"} and diff_top < 0.03:
            idx_bot = int(np.argmax(q_cloud[:, 1]))
            turn_angle = 0.0
            if stride <= idx_bot <= len(q_cloud) - stride - 1:
                v_in = q_cloud[idx_bot] - q_cloud[idx_bot - stride]
                v_out = q_cloud[idx_bot + stride] - q_cloud[idx_bot]
                n_in = float(np.linalg.norm(v_in))
                n_out = float(np.linalg.norm(v_out))
                if n_in > 1e-4 and n_out > 1e-4:
                    cos_ang = np.dot(v_in, v_out) / (n_in * n_out)
                    turn_angle = float(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))

            if turn_angle > 85.0:
                scores["V"] = scores.get("U", 0) + 0.05
            else:
                scores["U"] = scores.get("V", 0) + 0.05

        # Tie-break 'B' vs '8' (only if close tie < 0.03)
        if {top1, top2} == {"B", "8"} and diff_top < 0.03:
            if len(strokes) >= 2:
                scores["B"] = scores.get("8", 0) + 0.05
            else:
                scores["8"] = scores.get("B", 0) + 0.05

        best_char = max(scores, key=scores.get)
        confidence = float(min(0.99, max(0.40, scores[best_char])))
        return best_char, confidence

    def render_text_to_canvas(self, canvas, text, bbox, color,
                              font_family="Arial", font_size=None,
                              bold=False, italic=False):
        """Render clean, anti-aliased font text onto canvas at the handwriting bounding box."""
        if not text or bbox is None:
            return canvas.copy()

        bx, by, bw, bh = bbox

        if font_size is None or font_size <= 0:
            font_size = max(24, min(140, int(round(bh * 1.05))))

        font = self.font_manager.get_font(font_family, font_size, bold=bold, italic=italic)

        cx = int(round(bx + bw / 2.0))
        cy = int(round(by + bh / 2.0))

        rendered = self.font_manager.render_text_to_bgr(
            canvas, text, (cx, cy), font, color, align="center"
        )
        return rendered
