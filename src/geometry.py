"""Geometric transform models: fitting, robust fitting and application.

Coordinate convention used everywhere in this project:
    points are (x, y) in PIXEL coordinates, x = column, y = row, origin at the
    centre of the top-left pixel. Arrays of points have shape (N, 2).

A transform object maps SOURCE pixel coords -> REFERENCE pixel coords.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Transform:
    """A source->reference pixel transform.

    kind: "similarity" | "affine" | "homography" | "poly2" | "poly3"
    params: 3x3 matrix for linear kinds; (n_terms, 2) coefficient array for polynomials.
    norm: (cx, cy, s) used to normalise coordinates for polynomials (numerical stability).
    """

    kind: str
    params: np.ndarray
    norm: tuple[float, float, float] = field(default=(0.0, 0.0, 1.0))

    def apply(self, pts: np.ndarray) -> np.ndarray:
        """Map (N, 2) source points to reference points."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if self.kind in ("similarity", "affine", "homography"):
            h = np.hstack([pts, np.ones((len(pts), 1))]) @ self.params.T
            return h[:, :2] / h[:, 2:3]
        order = 2 if self.kind == "poly2" else 3
        return _poly_terms(_normalise(pts, self.norm), order) @ self.params

    def to_dict(self) -> dict:
        """JSON-serialisable form."""
        return {"kind": self.kind, "params": self.params.tolist(), "norm": list(self.norm)}


def _normalise(pts: np.ndarray, norm: tuple[float, float, float]) -> np.ndarray:
    cx, cy, s = norm
    return (pts - np.array([cx, cy])) / s


def _poly_terms(pts: np.ndarray, order: int) -> np.ndarray:
    """Design matrix of all monomials x^i y^j with i + j <= order."""
    x, y = pts[:, 0], pts[:, 1]
    cols = [x**i * y**j for d in range(order + 1) for i in range(d + 1) for j in [d - i]]
    return np.stack(cols, axis=1)


MIN_POINTS = {"similarity": 2, "affine": 3, "homography": 4, "poly2": 6, "poly3": 10}


def fit_transform(src: np.ndarray, ref: np.ndarray, kind: str) -> Transform:
    """Least-squares fit of a transform mapping src -> ref (no outlier rejection)."""
    src = np.asarray(src, np.float64)
    ref = np.asarray(ref, np.float64)
    if len(src) < MIN_POINTS[kind]:
        raise ValueError(f"{kind} needs >= {MIN_POINTS[kind]} points, got {len(src)}")
    if kind == "homography":
        H, _ = cv2.findHomography(src, ref, 0)
        if H is None:
            raise RuntimeError("homography fit failed")
        return Transform("homography", H)
    if kind == "affine":
        A = np.hstack([src, np.ones((len(src), 1))])
        coef, *_ = np.linalg.lstsq(A, ref, rcond=None)
        return Transform("affine", np.vstack([coef.T, [0, 0, 1]]))
    if kind == "similarity":
        M, _ = cv2.estimateAffinePartial2D(src, ref, method=cv2.LMEDS)
        if M is None:
            raise RuntimeError("similarity fit failed")
        return Transform("similarity", np.vstack([M, [0, 0, 1]]))
    order = 2 if kind == "poly2" else 3
    cx, cy = src.mean(axis=0)
    s = max(float(np.abs(src - [cx, cy]).max()), 1.0)
    norm = (float(cx), float(cy), s)
    coef, *_ = np.linalg.lstsq(_poly_terms(_normalise(src, norm), order), ref, rcond=None)
    return Transform(kind, coef, norm)


def robust_fit(
    src: np.ndarray,
    ref: np.ndarray,
    kind: str,
    thresh_px: float = 3.0,
    method: str = "magsac",
    max_iters: int = 10000,
    confidence: float = 0.9999,
) -> tuple[Transform, np.ndarray]:
    """Robustly fit a transform and return (transform, inlier mask).

    MAGSAC++ (OpenCV USAC) finds inliers with a homography/affine model; the final
    model of the requested kind is then refit by least squares on those inliers.
    For polynomial kinds the robust stage uses a homography, then an extra
    residual check with the polynomial removes remaining outliers.
    """
    src = np.asarray(src, np.float32)
    ref = np.asarray(ref, np.float32)
    n = len(src)
    if n < 4:
        return None, np.zeros(n, bool)
    flag = {"magsac": cv2.USAC_MAGSAC, "ransac": cv2.RANSAC, "usac": cv2.USAC_DEFAULT}[method]
    if kind in ("affine", "similarity"):
        fn = cv2.estimateAffine2D if kind == "affine" else cv2.estimateAffinePartial2D
        M, mask = fn(src, ref, method=flag, ransacReprojThreshold=thresh_px,
                     maxIters=max_iters, confidence=confidence)
    else:
        M, mask = cv2.findHomography(src, ref, flag, thresh_px, maxIters=max_iters,
                                     confidence=confidence)
    if M is None or mask is None:
        return None, np.zeros(n, bool)
    inl = mask.ravel().astype(bool)
    if inl.sum() < MIN_POINTS[kind]:
        return None, inl
    T = fit_transform(src[inl], ref[inl], kind)
    # one refinement pass: re-evaluate residuals with the final model
    res = residuals(T, src, ref)
    inl = res < thresh_px
    if inl.sum() >= MIN_POINTS[kind]:
        T = fit_transform(src[inl], ref[inl], kind)
    return T, inl


def residuals(T: Transform, src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Euclidean residual ||T(src) - ref|| per point, in reference pixels."""
    return np.linalg.norm(T.apply(src) - np.asarray(ref, np.float64), axis=1)


def compose_scale(T: Transform, s: float) -> Transform:
    """Return T' with T'(p) = T(s * p). Used when the source was resized by s before matching."""
    S = np.diag([s, s, 1.0])
    if T.kind in ("similarity", "affine", "homography"):
        return Transform(T.kind, T.params @ S)
    cx, cy, n = T.norm
    return Transform(T.kind, T.params, (cx / s, cy / s, n / s))
