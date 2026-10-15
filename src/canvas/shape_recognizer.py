import math
import cv2
import numpy as np


class ShapeRecognizer:
    """Intelligent geometric stroke beautification engine.

    Analyzes finished drawing strokes using geometric heuristics and
    contour analysis. Converts rough freehand lines, rectangles, circles,
    ellipses, and triangles into crisp vector shapes without external models.
    """

    def __init__(self, confidence_threshold: float = 0.75):
        self.confidence_threshold = confidence_threshold

    def recognize_and_beautify(self, points, base_canvas, color, thickness):
        """Analyze stroke points and render beautified shape if confident.

        Returns (beautified_canvas, shape_name) or (None, None).
        """
        if len(points) < 6:
            return None, None

        pts = np.array(points, dtype=np.int32)
        n = len(pts)

        # Total arc length
        diffs = pts[1:] - pts[:-1]
        segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        arc_length = float(segment_lengths.sum())

        if arc_length < 20.0:
            return None, None

        p_start = pts[0]
        p_end = pts[-1]
        chord_len = float(math.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1]))
        endpoint_dist_ratio = chord_len / arc_length

        # ── 1. Straight Line Check ─────────────────────────────────
        if chord_len > 25.0 and endpoint_dist_ratio > 0.90:
            # Calculate maximum perpendicular deviation from line segment
            vx = p_end[0] - p_start[0]
            vy = p_end[1] - p_start[1]
            line_len = chord_len

            # Perpendicular distance formula: |vx*(y0 - y) - vy*(x0 - x)| / line_len
            dxs = pts[:, 0] - p_start[0]
            dys = pts[:, 1] - p_start[1]
            perp_dists = np.abs(vx * dys - vy * dxs) / line_len
            max_dev = float(perp_dists.max())
            mean_dev = float(perp_dists.mean())

            if max_dev < min(24.0, 0.15 * line_len) and mean_dev < min(10.0, 0.06 * line_len):
                # High confidence straight line!
                canvas = base_canvas.copy()
                p1 = (int(p_start[0]), int(p_start[1]))
                p2 = (int(p_end[0]), int(p_end[1]))
                cv2.line(canvas, p1, p2, color, thickness, lineType=cv2.LINE_AA)
                r = max(1, thickness // 2)
                cv2.circle(canvas, p1, r, color, -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, p2, r, color, -1, lineType=cv2.LINE_AA)
                return canvas, "Straight Line"

        # ── Closed / Semi-Closed Shapes ────────────────────────────
        is_closed = (chord_len / arc_length) < 0.35 or chord_len < 40.0

        if is_closed and n >= 12:
            hull = cv2.convexHull(pts)
            hull_peri = float(cv2.arcLength(hull, True))
            hull_area = float(cv2.contourArea(hull))

            if hull_peri > 30.0 and hull_area > 80.0:
                # ── 2. Circle / Ellipse Check ──────────────────────
                circularity = (4.0 * math.pi * hull_area) / (hull_peri * hull_peri)

                # Min enclosing circle
                (mcx, mcy), m_radius = cv2.minEnclosingCircle(pts)
                m_radius = float(m_radius)
                circle_area = math.pi * (m_radius ** 2)

                # Bounding box
                x, y, w, h = cv2.boundingRect(pts)
                aspect = float(w) / max(1.0, float(h))

                if circularity > 0.70 and (hull_area / circle_area) > 0.65:
                    if 0.78 <= aspect <= 1.28:
                        # Perfect Circle
                        canvas = base_canvas.copy()
                        center = (int(round(mcx)), int(round(mcy)))
                        rad = max(2, int(round(m_radius * 0.95)))
                        cv2.circle(canvas, center, rad, color, thickness, lineType=cv2.LINE_AA)
                        return canvas, "Circle"
                    elif 0.35 <= aspect <= 2.8 and len(pts) >= 5:
                        # Smooth Ellipse
                        try:
                            ellipse = cv2.fitEllipse(pts)
                            canvas = base_canvas.copy()
                            cv2.ellipse(canvas, ellipse, color, thickness, lineType=cv2.LINE_AA)
                            return canvas, "Ellipse"
                        except Exception:
                            pass

                # ── 3. Polygon Simplification (Triangle / Rectangle)
                # Approximate polygon with adjustable epsilon
                for eps_factor in [0.035, 0.045, 0.025]:
                    epsilon = eps_factor * hull_peri
                    approx = cv2.approxPolyDP(hull, epsilon, True)

                    # Triangle check
                    if len(approx) == 3:
                        canvas = base_canvas.copy()
                        cv2.polylines(canvas, [approx], isClosed=True, color=color,
                                      thickness=thickness, lineType=cv2.LINE_AA)
                        for pt in approx:
                            cv2.circle(canvas, tuple(pt[0]), max(1, thickness // 2), color, -1, lineType=cv2.LINE_AA)
                        return canvas, "Triangle"

                    # Rectangle / Quadrilateral check
                    if len(approx) == 4:
                        # Check angles between consecutive edges
                        p_poly = approx.reshape(-1, 2)
                        angles = []
                        for i in range(4):
                            v1 = p_poly[(i + 1) % 4] - p_poly[i]
                            v2 = p_poly[(i - 1) % 4] - p_poly[i]
                            cos_th = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                            angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_th))))
                            angles.append(angle)

                        # If angles are reasonably rectangular (70-110 deg)
                        if all(70 <= a <= 112 for a in angles):
                            canvas = base_canvas.copy()
                            # Use axis-aligned or rotated rect
                            rect = cv2.minAreaRect(pts)
                            box = cv2.boxPoints(rect)
                            box = np.int32(box)
                            cv2.polylines(canvas, [box], isClosed=True, color=color,
                                          thickness=thickness, lineType=cv2.LINE_AA)
                            for pt in box:
                                cv2.circle(canvas, tuple(pt), max(1, thickness // 2), color, -1, lineType=cv2.LINE_AA)
                            return canvas, "Rectangle"

        # Not confident enough — keep natural stroke
        return None, None
