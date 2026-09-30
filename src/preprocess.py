"""Photometric normalisation, shadow masking and resampling.

All functions take and return 2-D float32 arrays in [0, 1] unless stated.
"""
from __future__ import annotations

import cv2
import numpy as np


def to_unit(img: np.ndarray, p_low: float = 0.5, p_high: float = 99.5,
            valid: np.ndarray | None = None) -> np.ndarray:
    """Percentile stretch to [0, 1] (robust to hot pixels and nodata)."""
    img = np.asarray(img, np.float32)
    vals = img[valid] if valid is not None else img[np.isfinite(img)]
    lo, hi = np.percentile(vals, [p_low, p_high])
    out = (img - lo) / max(hi - lo, 1e-6)
    return np.clip(np.nan_to_num(out, nan=0.0), 0, 1).astype(np.float32)


def clahe(img: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation."""
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    out = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(u8)
    return out.astype(np.float32) / 255.0


def shadow_mask(img: np.ndarray, dark_pct: float = 8.0, abs_thresh: float = 0.04,
                edge_px: int = 3) -> np.ndarray:
    """Boolean mask of pixels that are in or next to shadow.

    A pixel is shadow if it is darker than the given percentile or an absolute
    threshold. The mask is dilated by `edge_px` because shadow BOUNDARIES are
    what moves with the sun and produces false matches.
    """
    t = max(float(np.percentile(img, dark_pct)), abs_thresh)
    m = (img <= t).astype(np.uint8)
    if edge_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * edge_px + 1, 2 * edge_px + 1))
        m = cv2.dilate(m, k)
    return m.astype(bool)


def resize(img: np.ndarray, scale: float) -> np.ndarray:
    """Resize by `scale` with area averaging when shrinking (avoids aliasing)."""
    if abs(scale - 1.0) < 1e-6:
        return img
    h, w = img.shape
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, size, interpolation=interp)


def prepare(img: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply the configured normalisation. Returns (image for matching, shadow mask)."""
    out = to_unit(img)
    s = cfg.get("shadow", {})
    if s.get("enabled", True):
        smask = shadow_mask(out, s.get("dark_pct", 8.0), s.get("abs_thresh", 0.04), s.get("edge_px", 3))
    else:
        smask = np.zeros(out.shape, bool)
    if cfg.get("clahe", {}).get("enabled", True):
        c = cfg.get("clahe", {})
        out = clahe(out, c.get("clip", 2.0), c.get("tile", 8))
    return out, smask
