"""Evaluation metrics: checkpoint RMSE, fit RMSE, inlier stats, distribution.

Honest-accuracy rule: the checkpoint RMSE is measured on inliers that were NOT
used to fit the transform.
"""
from __future__ import annotations

import numpy as np

from .geometry import Transform, fit_transform, residuals


def rmse(err: np.ndarray) -> float:
    """Root-mean-square of per-point Euclidean errors."""
    return float(np.sqrt(np.mean(np.square(err)))) if len(err) else float("nan")


def grid_cells(pts: np.ndarray, shape: tuple[int, int], grid: int) -> np.ndarray:
    """Cell index (row-major) of each (x, y) point on a grid x grid layout."""
    h, w = shape
    cx = np.clip((pts[:, 0] * grid / w).astype(int), 0, grid - 1)
    cy = np.clip((pts[:, 1] * grid / h).astype(int), 0, grid - 1)
    return cy * grid + cx


def distribution_score(pts: np.ndarray, shape: tuple[int, int], grid: int = 8) -> float:
    """Share of grid cells holding at least one point (1.0 = every cell covered)."""
    if len(pts) == 0:
        return 0.0
    return len(np.unique(grid_cells(pts, shape, grid))) / float(grid * grid)


def spatial_entropy(pts: np.ndarray, shape: tuple[int, int], grid: int = 8) -> float:
    """Normalised Shannon entropy of point counts per cell (1.0 = perfectly even)."""
    if len(pts) == 0:
        return 0.0
    counts = np.bincount(grid_cells(pts, shape, grid), minlength=grid * grid).astype(float)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(grid * grid))


def grid_filter(pts: np.ndarray, scores: np.ndarray, shape: tuple[int, int],
                grid: int = 8, per_cell: int = 10) -> np.ndarray:
    """Indices of the best `per_cell` points (by score) in each grid cell."""
    if len(pts) == 0:
        return np.zeros(0, int)
    cells = grid_cells(pts, shape, grid)
    keep = []
    for c in np.unique(cells):
        idx = np.where(cells == c)[0]
        keep.extend(idx[np.argsort(-scores[idx])[:per_cell]])
    return np.array(sorted(keep), int)


def split_checkpoints(n: int, frac: float, seed: int, cells: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask of held-out checkpoints; stratified by grid cell when `cells` is given."""
    rng = np.random.default_rng(seed)
    hold = np.zeros(n, bool)
    if cells is None:
        hold[rng.choice(n, int(round(frac * n)), replace=False)] = True
        return hold
    for c in np.unique(cells):
        idx = np.where(cells == c)[0]
        k = int(round(frac * len(idx)))
        if k:
            hold[rng.choice(idx, k, replace=False)] = True
    # sparse cells round down to 0; top up at random so the total reaches frac * n (min 3)
    target = max(3, int(round(frac * n))) if n >= 12 else int(round(frac * n))
    missing = target - int(hold.sum())
    if missing > 0:
        free = np.where(~hold)[0]
        hold[rng.choice(free, min(missing, len(free)), replace=False)] = True
    return hold


def evaluate(pts_src: np.ndarray, pts_ref: np.ndarray, kind: str, src_shape: tuple[int, int],
             n_tentative: int, holdout: float = 0.25, seed: int = 42, grid: int = 8,
             px_size_m: float | None = None) -> tuple[dict, Transform, np.ndarray]:
    """Fit on (1 - holdout) of the inliers, test on the rest, and report all metrics.

    Returns (metrics dict, transform fitted on ALL inliers for the final product,
    boolean checkpoint mask).
    """
    n = len(pts_src)
    cells = grid_cells(pts_src, src_shape, grid)
    hold = split_checkpoints(n, holdout, seed, cells)
    T_fit = fit_transform(pts_src[~hold], pts_ref[~hold], kind)
    cp_err = residuals(T_fit, pts_src[hold], pts_ref[hold])
    fit_err = residuals(T_fit, pts_src[~hold], pts_ref[~hold])
    T_all = fit_transform(pts_src, pts_ref, kind)
    m = {
        "n_tentative": int(n_tentative),
        "n_inliers": int(n),
        "inlier_ratio": float(n / max(n_tentative, 1)),
        "n_checkpoints": int(hold.sum()),
        "checkpoint_rmse_px": rmse(cp_err),
        "checkpoint_median_px": float(np.median(cp_err)) if len(cp_err) else float("nan"),
        "checkpoint_p90_px": float(np.percentile(cp_err, 90)) if len(cp_err) else float("nan"),
        "subpixel_rate": float(np.mean(cp_err < 1.0)) if len(cp_err) else float("nan"),
        "fit_rmse_px": rmse(fit_err),
        "distribution_score": distribution_score(pts_src, src_shape, grid),
        "spatial_entropy": spatial_entropy(pts_src, src_shape, grid),
    }
    if px_size_m:
        m["checkpoint_rmse_m"] = m["checkpoint_rmse_px"] * px_size_m
    return m, T_all, hold


def ground_truth_error(T_est: Transform, T_true: Transform, shape: tuple[int, int],
                       step: int = 16) -> dict:
    """Exact error of an estimated transform vs the known one, on a dense source grid."""
    h, w = shape
    xs, ys = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
    p = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64)
    e = np.linalg.norm(T_est.apply(p) - T_true.apply(p), axis=1)
    return {"gt_rmse_px": rmse(e), "gt_median_px": float(np.median(e)),
            "gt_p90_px": float(np.percentile(e, 90)), "gt_max_px": float(e.max())}
