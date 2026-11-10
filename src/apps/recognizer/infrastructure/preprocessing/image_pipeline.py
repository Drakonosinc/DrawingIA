"""Center-of-mass bounding-box image preprocessing pipeline for hand-drawn strokes."""

from __future__ import annotations

import base64
import io
import numpy as np
from PIL import Image

from ...domain.interfaces import IImagePreprocessor


class CenterOfMassPreprocessor(IImagePreprocessor):
    """Normalizes arbitrary canvas strokes into centered 28x28 float32 tensors in [0.0, 1.0].

    Steps:
    1. Decode Base64 PNG (handling both alpha-channel strokes and dark-background canvases) or 2D array.
    2. Crop tight bounding box around non-zero ink pixels.
    3. Resize preserving aspect ratio so the longest side is 20 pixels (anti-aliased).
    4. Paste into a 28x28 frame and shift so the ink center-of-mass sits at (13.5, 13.5).
    """

    def __init__(self, target_size: int = 28, inner_box: int = 20, threshold: float = 0.08) -> None:
        self.target_size = int(target_size)
        self.inner_box = int(inner_box)
        self.threshold = float(threshold)

    def _decode_to_grayscale(self, source: str | np.ndarray) -> np.ndarray:
        if isinstance(source, np.ndarray):
            arr = np.asarray(source, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr.mean(axis=-1)
            if arr.max() > 1.0:
                arr = arr / 255.0
            return np.clip(arr, 0.0, 1.0)

        if not isinstance(source, str) or not source.strip():
            return np.zeros((self.target_size, self.target_size), dtype=np.float32)

        encoded = source.split(",", 1)[1] if "," in source else source
        raw_bytes = base64.b64decode(encoded)
        with Image.open(io.BytesIO(raw_bytes)) as img:
            rgba = img.convert("RGBA")
            arr_rgba = np.asarray(rgba, dtype=np.float32) / 255.0

        rgb_luminance = (
            0.299 * arr_rgba[:, :, 0]
            + 0.587 * arr_rgba[:, :, 1]
            + 0.114 * arr_rgba[:, :, 2]
        )
        alpha = arr_rgba[:, :, 3]

        # If background is transparent, alpha channel directly represents stroke intensity
        if alpha.min() < 0.1 and alpha.max() > 0.2:
            ink = alpha * np.maximum(rgb_luminance, 0.5)
        else:
            # Determine whether background is dark (ink is bright) or light (ink is dark)
            border_pixels = np.concatenate(
                [rgb_luminance[0, :], rgb_luminance[-1, :], rgb_luminance[:, 0], rgb_luminance[:, -1]]
            )
            bg_median = float(np.median(border_pixels))
            ink = rgb_luminance if bg_median < 0.5 else (1.0 - rgb_luminance)

        if ink.max() > 0.0:
            ink = ink / ink.max()
        return np.clip(ink.astype(np.float32), 0.0, 1.0)

    def process(self, source: str | np.ndarray) -> np.ndarray:
        ink = self._decode_to_grayscale(source)
        mask = ink > self.threshold
        if not np.any(mask):
            return np.zeros((self.target_size, self.target_size), dtype=np.float32)

        ys, xs = np.where(mask)
        y_min, y_max = int(ys.min()), int(ys.max()) + 1
        x_min, x_max = int(xs.min()), int(xs.max()) + 1
        cropped = ink[y_min:y_max, x_min:x_max]

        h, w = cropped.shape
        scale = self.inner_box / float(max(h, w, 1))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        pil_crop = Image.fromarray(np.uint8(np.clip(cropped * 255.0, 0, 255)), mode="L")
        pil_resized = pil_crop.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        resized = np.asarray(pil_resized, dtype=np.float32) / 255.0

        canvas = np.zeros((self.target_size, self.target_size), dtype=np.float32)
        top = (self.target_size - new_h) // 2
        left = (self.target_size - new_w) // 2
        canvas[top : top + new_h, left : left + new_w] = resized

        # Shift by center of mass so the stroke's centroid sits at the center of the 28x28 grid
        total_mass = float(canvas.sum())
        if total_mass > 1e-5:
            grid_y, grid_x = np.indices(canvas.shape, dtype=np.float32)
            cy = float((grid_y * canvas).sum() / total_mass)
            cx = float((grid_x * canvas).sum() / total_mass)
            center = (self.target_size - 1) / 2.0
            dy = int(np.clip(round(center - cy), -4, 4))
            dx = int(np.clip(round(center - cx), -4, 4))
            canvas = self._shift_2d(canvas, dy, dx)

        max_val = float(canvas.max())
        if max_val > 1e-5:
            canvas = canvas / max_val
        return np.clip(canvas, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _shift_2d(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
        h, w = arr.shape
        out = np.zeros_like(arr)
        src_y0 = max(0, -dy)
        src_y1 = min(h, h - dy)
        dst_y0 = max(0, dy)
        dst_y1 = min(h, h + dy)

        src_x0 = max(0, -dx)
        src_x1 = min(w, w - dx)
        dst_x0 = max(0, dx)
        dst_x1 = min(w, w + dx)

        if src_y1 > src_y0 and src_x1 > src_x0:
            out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
        return out
