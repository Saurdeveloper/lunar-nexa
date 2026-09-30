"""Fast sanity tests: python -m pytest -q"""
import numpy as np

from src.geometry import Transform, fit_transform, residuals, robust_fit
from src.metrics import distribution_score, split_checkpoints
from src.subpixel import refine_matches


def _H():
    return np.array([[0.5, 0.05, 10.0], [-0.04, 0.52, 20.0], [1e-5, -2e-5, 1.0]])


def test_fit_roundtrip_all_models():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 500, (200, 2))
    T = Transform("homography", _H())
    q = T.apply(p)
    for kind, tol in [("homography", 1e-4), ("poly3", 0.05), ("poly2", 0.5)]:
        assert residuals(fit_transform(p, q, kind), p, q).max() < tol, kind


def test_robust_fit_rejects_outliers():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 500, (300, 2))
    q = Transform("homography", _H()).apply(p)
    q[:90] += rng.uniform(-40, 40, (90, 2))
    T, inl = robust_fit(p, q, "homography", 2.0)
    assert inl[90:].mean() > 0.95 and inl[:90].mean() < 0.1


def test_subpixel_recovers_known_shift():
    """Reference = source shifted by (0.37, -0.21) px; refinement must find it."""
    import cv2
    rng = np.random.default_rng(2)
    src = cv2.GaussianBlur(rng.random((256, 256)).astype(np.float32), (0, 0), 1.5)
    dx, dy = 0.37, -0.21
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    ref = cv2.warpAffine(src, M, (256, 256), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    pts = rng.uniform(40, 216, (30, 2))
    T0 = Transform("homography", np.eye(3))  # coarse guess: identity (true is a pure shift)
    q, ok = refine_matches(src, ref, pts, T0, patch=32, upsample=50)
    err = q - (pts + [dx, dy])
    assert ok.all() and np.abs(err).max() < 0.1  # cubic-interpolation limit on random texture


def test_checkpoint_split_and_distribution():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 800, (100, 2))
    hold = split_checkpoints(100, 0.25, 0)
    assert 20 <= hold.sum() <= 30
    assert distribution_score(p, (800, 800), 8) > 0.7
