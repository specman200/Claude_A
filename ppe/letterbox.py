"""Aspect-preserving resize and the exact inverse mapping for boxes.

Capture resolution and inference size can be changed freely: frames are scaled
by a single factor and padded, never squashed, so object geometry — and with it
detection accuracy — is preserved. Every box is mapped back through the same
factor, which is why overlays land on the pixel they were found in.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PAD_VALUE = 114  # the grey Ultralytics trains and validates with


@dataclass(frozen=True, slots=True)
class Letterbox:
    """The transform that took a ``src_w x src_h`` frame into a square canvas."""

    gain: float
    pad_x: float
    pad_y: float
    src_w: int
    src_h: int

    def to_source(self, boxes: np.ndarray) -> np.ndarray:
        """Map xyxy boxes from canvas space back to source-frame pixels."""
        if boxes.size == 0:
            return boxes.reshape(0, 4).astype(np.float32)
        out = boxes.astype(np.float32, copy=True)
        # Strided slices are views, so every step below is in-place.
        xs, ys = out[:, 0::2], out[:, 1::2]
        xs -= self.pad_x
        xs /= self.gain
        ys -= self.pad_y
        ys /= self.gain
        np.clip(xs, 0, self.src_w, out=xs)
        np.clip(ys, 0, self.src_h, out=ys)
        return out


def letterbox(frame: np.ndarray, size: int, scaleup: bool = True) -> tuple[np.ndarray, Letterbox]:
    """Resize ``frame`` into a ``size x size`` centred canvas, keeping aspect."""
    h, w = frame.shape[:2]
    gain = min(size / h, size / w)
    if not scaleup:
        gain = min(gain, 1.0)

    new_w, new_h = round(w * gain), round(h * gain)
    pad_x, pad_y = (size - new_w) / 2, (size - new_h) / 2

    if (new_w, new_h) != (w, h):
        # INTER_AREA is the right filter for shrinking: it averages instead of
        # point-sampling, so small objects survive the downscale.
        interp = cv2.INTER_AREA if gain < 1 else cv2.INTER_LINEAR
        frame = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    canvas[top : top + new_h, left : left + new_w] = frame
    # Report the pad actually applied, so the inverse is exact.
    return canvas, Letterbox(gain, float(left), float(top), w, h)


def fit(src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[float, float, float]:
    """Aspect-fit ``src`` inside ``dst``; returns ``(scale, offset_x, offset_y)``.

    The UI uses this to place both the pixmap and its boxes with one transform,
    so they cannot drift apart when the window is resized.
    """
    if src_w <= 0 or src_h <= 0:
        return 1.0, 0.0, 0.0
    scale = min(dst_w / src_w, dst_h / src_h)
    return scale, (dst_w - src_w * scale) / 2, (dst_h - src_h * scale) / 2
