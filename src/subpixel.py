"""Sub-pixel refinement of matches by upsampled phase correlation.

For each inlier (p in source, q in reference) and a coarse transform T:
  1. sample a small source patch centred on p,
  2. sample the reference at T(p + offsets), i.e. the reference resampled into
     the source's local pixel grid (so scale/rotation are already removed),
  3. phase-correlate the two patches -> residual shift s (source pixels),
  4. refined reference point q' = T(p - s).
Source points stay fixed; only reference locations move. That keeps the
source-side distribution unchanged and works for any transform kind.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.registration import phase_cross_correlation

from .geometry import Transform


def _sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return cv2.remap(img, xs.astype(np.float32), ys.astype(np.float32),
                     interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def refine_matches(src: np.ndarray, ref: np.ndarray, pts_src: np.ndarray, T: Transform,
                   patch: int = 32, upsample: int = 20, max_shift: float = 2.0,
                   use_gradient: bool = False, normalization: str | None = None,
                   inner_iters: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return (refined reference points (N, 2), ok mask (N,)).

    Points whose refinement fails, lands outside the image, or moves more than
    `max_shift` source pixels are left at T(p) and flagged ok=False.

    normalization=None (windowed cross-correlation) is the default: on a known
    0.37/-0.21 px shift it is off by <= 0.05 px, while windowed phase correlation
    under-estimates by ~40 % in one pass (needs 3+ `inner_iters` to converge).
    `inner_iters` re-samples the reference at the updated position and re-estimates.
    """
    half = patch / 2.0
    off = np.arange(patch, dtype=np.float64) - half + 0.5
    ox, oy = np.meshgrid(off, off)
    win = np.outer(np.hanning(patch), np.hanning(patch))
    h_s, w_s = src.shape
    h_r, w_r = ref.shape
    out = T.apply(pts_src)
    ok = np.zeros(len(pts_src), bool)

    def prep(a: np.ndarray) -> np.ndarray:
        if use_gradient:
            gx = cv2.Sobel(a, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(a, cv2.CV_32F, 0, 1)
            a = np.hypot(gx, gy)
        a = a - a.mean()
        return a * win

    for i, (px, py) in enumerate(pts_src):
        if not (half <= px < w_s - half and half <= py < h_s - half):
            continue
        sx, sy = px + ox, py + oy
        a = prep(_sample(src, sx, sy))
        if a.std() < 1e-6:
            continue
        acc = np.zeros(2)
        good = True
        for _ in range(max(1, inner_iters)):
            rq = T.apply(np.stack([(sx - acc[0]).ravel(), (sy - acc[1]).ravel()], 1))
            rx, ry = rq[:, 0].reshape(ox.shape), rq[:, 1].reshape(ox.shape)
            if rx.min() < 0 or ry.min() < 0 or rx.max() > w_r - 1 or ry.max() > h_r - 1:
                good = False
                break
            b = prep(_sample(ref, rx, ry))
            if b.std() < 1e-6:
                good = False
                break
            try:
                shift, _, _ = phase_cross_correlation(a, b, upsample_factor=upsample,
                                                      normalization=normalization)
            except Exception:
                good = False
                break
            acc += np.array([shift[1], shift[0]])  # (row, col) -> (x, y)
            if np.hypot(*acc) > max_shift:
                good = False
                break
        if not good:
            continue
        s = acc
        out[i] = T.apply(np.array([[px - s[0], py - s[1]]]))[0]
        ok[i] = True
    return out, ok
