"""Procedural vector-stroke synthesizer for training and benchmarking DrawingIA."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image, ImageDraw

from ...domain.entities import CLASS_IDS
from ..preprocessing.image_pipeline import CenterOfMassPreprocessor


class ProceduralStrokeGenerator:
    """Synthesizes diverse, realistic 28x28 hand-drawn stroke samples for all 15 DrawingIA classes."""

    def __init__(self, seed: int | None = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.preprocessor = CenterOfMassPreprocessor()

    def _jitter(self, pt: tuple[float, float], scale: float = 2.2) -> tuple[float, float]:
        return (
            pt[0] + float(self.rng.normal(0.0, scale)),
            pt[1] + float(self.rng.normal(0.0, scale)),
        )

    def _draw_polyline(
        self,
        draw: ImageDraw.ImageDraw,
        points: list[tuple[float, float]],
        width: int,
        jitter_scale: float = 1.8,
        closed: bool = False,
    ) -> None:
        pts = [self._jitter(p, jitter_scale) for p in points]
        if closed and pts:
            pts.append(pts[0])
        if len(pts) >= 2:
            draw.line(pts, fill=255, width=width, joint="curve")

    def render_sample(self, class_id: str, jitter: bool = True) -> np.ndarray:
        """Render a single 64x64 anti-aliased stroke image and normalize it to 28x28."""
        size = 64
        img = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(img)
        w = int(self.rng.integers(4, 7)) if jitter else 5
        j = 2.0 if jitter else 0.0

        if class_id == "0":
            rx = float(self.rng.uniform(14, 19)) if jitter else 16.0
            ry = float(self.rng.uniform(20, 24)) if jitter else 22.0
            pts = [
                (32 + rx * math.cos(t), 32 + ry * math.sin(t))
                for t in np.linspace(0, 2 * math.pi, 20, endpoint=False)
            ]
            self._draw_polyline(draw, pts, w, j, closed=True)

        elif class_id == "1":
            slant = float(self.rng.uniform(-4, 4)) if jitter else 0.0
            self._draw_polyline(draw, [(32 + slant, 10), (32 - slant, 54)], w, j)
            if not jitter or self.rng.random() < 0.65:
                self._draw_polyline(draw, [(24 + slant, 17), (32 + slant, 10)], max(3, w - 1), j * 0.5)
            if not jitter or self.rng.random() < 0.55:
                self._draw_polyline(draw, [(22, 54), (42, 54)], max(3, w - 1), j * 0.5)

        elif class_id == "2":
            pts = [
                (16, 20),
                (24, 11),
                (38, 11),
                (47, 19),
                (44, 29),
                (30, 41),
                (16, 53),
                (48, 53),
            ]
            self._draw_polyline(draw, pts, w, j)

        elif class_id == "3":
            pts = [
                (17, 14),
                (34, 10),
                (46, 18),
                (42, 29),
                (27, 32),
                (44, 36),
                (47, 47),
                (34, 54),
                (16, 50),
            ]
            self._draw_polyline(draw, pts, w, j)

        elif class_id == "4":
            self._draw_polyline(draw, [(38, 12), (14, 38), (49, 38)], w, j)
            self._draw_polyline(draw, [(38, 12), (38, 54)], w, j)

        elif class_id == "5":
            pts = [
                (45, 12),
                (19, 12),
                (17, 30),
                (35, 28),
                (47, 36),
                (45, 49),
                (31, 54),
                (16, 49),
            ]
            self._draw_polyline(draw, pts, w, j)

        elif class_id == "6":
            pts = [
                (42, 11),
                (25, 18),
                (16, 33),
                (18, 48),
                (32, 54),
                (46, 47),
                (45, 35),
                (31, 31),
                (17, 36),
            ]
            self._draw_polyline(draw, pts, w, j)

        elif class_id == "7":
            self._draw_polyline(draw, [(14, 13), (49, 13), (24, 54)], w, j)
            if not jitter or self.rng.random() < 0.45:
                self._draw_polyline(draw, [(22, 33), (42, 33)], max(3, w - 1), j * 0.5)

        elif class_id == "8":
            top_loop = [
                (32 + 13 * math.cos(t), 21 + 10 * math.sin(t))
                for t in np.linspace(0, 2 * math.pi, 16, endpoint=False)
            ]
            bot_loop = [
                (32 + 15 * math.cos(t), 42 + 11 * math.sin(t))
                for t in np.linspace(0, 2 * math.pi, 16, endpoint=False)
            ]
            self._draw_polyline(draw, top_loop, w, j * 0.8, closed=True)
            self._draw_polyline(draw, bot_loop, w, j * 0.8, closed=True)

        elif class_id == "9":
            pts = [
                (45, 26),
                (31, 32),
                (17, 25),
                (19, 13),
                (33, 10),
                (46, 17),
                (45, 34),
                (36, 48),
                (21, 54),
            ]
            self._draw_polyline(draw, pts, w, j)

        elif class_id == "triangle":
            pts = [(32, 10), (53, 52), (11, 52)]
            self._draw_polyline(draw, pts, w, j, closed=True)

        elif class_id == "star":
            outer_r = 24.0
            inner_r = 10.0
            pts = []
            for i in range(10):
                angle = -math.pi / 2 + i * (math.pi / 5)
                r = outer_r if i % 2 == 0 else inner_r
                pts.append((32 + r * math.cos(angle), 34 + r * math.sin(angle)))
            self._draw_polyline(draw, pts, max(3, w - 1), j * 0.7, closed=True)

        elif class_id == "heart":
            pts = []
            for t in np.linspace(0, 2 * math.pi, 28, endpoint=False):
                x = 16 * (math.sin(t) ** 3)
                y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
                pts.append((32 + x * 1.35, 30 + y * 1.35))
            self._draw_polyline(draw, pts, w, j * 0.6, closed=True)

        elif class_id == "lightning":
            pts = [(38, 8), (18, 33), (34, 33), (22, 56)]
            self._draw_polyline(draw, pts, w + 1, j * 0.9, closed=False)

        elif class_id == "smile":
            # Two eyes + curved mouth arc
            draw.ellipse((19, 16, 26, 23), fill=255)
            draw.ellipse((38, 16, 45, 23), fill=255)
            arc_pts = [
                (32 + 18 * math.cos(t), 32 + 16 * math.sin(t))
                for t in np.linspace(0.2 * math.pi, 0.8 * math.pi, 14)
            ]
            self._draw_polyline(draw, arc_pts, w, j * 0.7, closed=False)

        if jitter:
            angle = float(self.rng.uniform(-14.0, 14.0))
            img = img.rotate(angle, resample=Image.Resampling.BILINEAR)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        return self.preprocessor.process(arr)

    def build_dataset(self, samples_per_class: int = 120) -> tuple[np.ndarray, np.ndarray]:
        """Generate (X, y) arrays of shape (N, 28, 28) and (N,) across all 15 classes."""
        total = len(CLASS_IDS) * int(samples_per_class)
        X = np.zeros((total, 28, 28), dtype=np.float32)
        y = np.zeros((total,), dtype=np.int64)
        idx = 0
        for class_idx, cid in enumerate(CLASS_IDS):
            # Include 1 canonical unjittered prototype per class plus jittered variations
            X[idx] = self.render_sample(cid, jitter=False)
            y[idx] = class_idx
            idx += 1
            for _ in range(int(samples_per_class) - 1):
                X[idx] = self.render_sample(cid, jitter=True)
                y[idx] = class_idx
                idx += 1
        perm = self.rng.permutation(total)
        return X[perm], y[perm]
