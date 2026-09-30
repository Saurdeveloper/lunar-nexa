"""End-to-end registration of a source (moving) image to a reference (fixed) image.

Stages (each controlled by the config):
  1. scale normalisation   resize the source by the known resolution ratio (metadata)
  2. photometric prep      percentile stretch, CLAHE, shadow mask
  3. matching              SIFT / RootSIFT / AKAZE / ORB / SuperPoint+LightGlue
  4. outlier rejection     MAGSAC++ (OpenCV USAC)
  5. uniform distribution  keep the best k inliers per grid cell
  6. sub-pixel refinement  upsampled phase correlation on locally resampled patches
  7. final fit + metrics   checkpoints held out, never used for fitting
  8. warp                  source resampled onto the reference grid

Point coordinates: (x, y) pixels, x = column, y = row.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from . import matching, metrics, preprocess, subpixel
from .geometry import Transform, fit_transform, residuals, robust_fit


def _scaled_to_orig(pts: np.ndarray, s: float) -> np.ndarray:
    """Undo cv2.resize by factor s (pixel-centre convention)."""
    return (pts + 0.5) / s - 0.5


def inverse_transform(T: Transform, src_shape: tuple[int, int]) -> Transform:
    """Reference -> source transform (exact for linear kinds, fitted for polynomials)."""
    if T.kind in ("similarity", "affine", "homography"):
        return Transform(T.kind, np.linalg.inv(T.params))
    h, w = src_shape
    xs, ys = np.meshgrid(np.linspace(0, w - 1, 40), np.linspace(0, h - 1, 40))
    p = np.stack([xs.ravel(), ys.ravel()], 1)
    return fit_transform(T.apply(p), p, T.kind)


def warp_to_reference(src: np.ndarray, T: Transform, ref_shape: tuple[int, int]) -> np.ndarray:
    """Resample the source onto the reference pixel grid; outside = NaN."""
    h, w = ref_shape
    Tinv = inverse_transform(T, src.shape)
    if T.kind in ("similarity", "affine", "homography"):
        out = cv2.warpPerspective(src.astype(np.float32), T.params, (w, h),
                                  flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=np.nan)
        return out
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    p = Tinv.apply(np.stack([xs.ravel(), ys.ravel()], 1))
    mx = p[:, 0].reshape(h, w).astype(np.float32)
    my = p[:, 1].reshape(h, w).astype(np.float32)
    return cv2.remap(src.astype(np.float32), mx, my, cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)


def register(src_img: np.ndarray, ref_img: np.ndarray, cfg: dict,
             scale_hint: float | None = None, extra_src_mask: np.ndarray | None = None,
             extra_ref_mask: np.ndarray | None = None) -> dict:
    """Register `src_img` to `ref_img`.

    scale_hint: source pixel size / reference pixel size (e.g. 1 m / 4 m = 0.25).
                When given (normally from product labels) the source is resized by
                it before matching, which removes most of the scale difference.
    extra_src_mask / extra_ref_mask: optional boolean masks (True = detection allowed), e.g.
                predicted cast-shadow edges from a DTM; combined with the darkness mask.
    Returns a dict with transform, matches, metrics, timings and the warped image.
    """
    t0 = time.perf_counter()
    timing = {}
    s = float(scale_hint) if (scale_hint and cfg.get("use_scale_hint", True)) else 1.0

    # 1-2. scale + photometric preparation
    src_prep, src_shadow = preprocess.prepare(src_img, cfg.get("preprocess", {}))
    ref_prep, ref_shadow = preprocess.prepare(ref_img, cfg.get("preprocess", {}))
    src_s = preprocess.resize(src_prep, s)
    src_shadow_s = preprocess.resize(src_shadow.astype(np.float32), s) > 0.5
    timing["preprocess_s"] = time.perf_counter() - t0

    # 3. matching (detection disallowed in shadows when configured)
    t = time.perf_counter()
    mcfg = cfg.get("matching", {})
    use_masks = cfg.get("preprocess", {}).get("shadow", {}).get("mask_matches", True)
    sm = ~src_shadow_s if use_masks else np.ones(src_s.shape, bool)
    rm = ~ref_shadow if use_masks else np.ones(ref_prep.shape, bool)
    if extra_src_mask is not None:
        sm &= preprocess.resize(extra_src_mask.astype(np.float32), s) > 0.5
    if extra_ref_mask is not None:
        rm &= extra_ref_mask
    ms, mr, sc = matching.match(src_s, ref_prep, mcfg, sm, rm)
    n_tent = len(ms)
    timing["matching_s"] = time.perf_counter() - t

    # 4. robust fit (scaled-source coords)
    t = time.perf_counter()
    rcfg = cfg.get("robust", {})
    T_s, inl = robust_fit(ms, mr, rcfg.get("model", "homography"),
                          rcfg.get("thresh_px", 3.0), rcfg.get("method", "magsac"))
    timing["robust_s"] = time.perf_counter() - t
    result = {"n_tentative": n_tent, "timing": timing, "scale_used": s}
    if T_s is None or inl.sum() < 8:
        result.update(ok=False, reason=f"too few inliers ({int(inl.sum())} of {n_tent})")
        return result
    ms, mr, sc = ms[inl], mr[inl], sc[inl]

    # 5. uniform distribution
    dcfg = cfg.get("distribution", {})
    if dcfg.get("enabled", True):
        keep = metrics.grid_filter(ms, sc, src_s.shape, dcfg.get("grid", 8), dcfg.get("per_cell", 12))
        ms, mr, sc = ms[keep], mr[keep], sc[keep]

    # 6. sub-pixel refinement (at matching scale, where patches have equal pixel size)
    t = time.perf_counter()
    scfg = cfg.get("subpixel", {})
    refined_ok = np.zeros(len(ms), bool)
    if scfg.get("enabled", True):
        Tg = fit_transform(ms, mr, rcfg.get("model", "homography"))
        for _ in range(scfg.get("iterations", 2)):
            mr_new, refined_ok = subpixel.refine_matches(
                src_s, ref_prep, ms, Tg, scfg.get("patch", 32), scfg.get("upsample", 20),
                scfg.get("max_shift", 2.0), scfg.get("use_gradient", False),
                scfg.get("normalization"), scfg.get("inner_iters", 1))
            mr = np.where(refined_ok[:, None], mr_new, mr)
            Tg = fit_transform(ms, mr, rcfg.get("model", "homography"))
        if scfg.get("drop_unrefined", True) and refined_ok.sum() >= 8:
            ms, mr, sc, refined_ok = ms[refined_ok], mr[refined_ok], sc[refined_ok], refined_ok[refined_ok]
    timing["subpixel_s"] = time.perf_counter() - t

    # back to original source pixel coordinates
    ps = _scaled_to_orig(ms, s)

    # 7. final model + honest metrics
    kind = cfg.get("final_model", "homography")
    # final residual screen against a model of the final kind
    T_tmp = fit_transform(ps, mr, kind)
    res = residuals(T_tmp, ps, mr)
    good = res < cfg.get("final_thresh_px", 1.5)
    if good.sum() >= 8:
        ps, mr, sc, refined_ok = ps[good], mr[good], sc[good], refined_ok[good]
    ecfg = cfg.get("evaluation", {})
    m, T, hold = metrics.evaluate(ps, mr, kind, src_img.shape, n_tent,
                                  ecfg.get("holdout", 0.25), cfg.get("seed", 42),
                                  ecfg.get("grid", 8), ecfg.get("ref_px_size_m"))
    m["n_refined"] = int(refined_ok.sum())
    q = cfg.get("quality", {})
    m["quality_pass"] = bool(m["n_inliers"] >= q.get("min_inliers", 20)
                             and m["checkpoint_rmse_px"] <= q.get("max_checkpoint_rmse_px", 1.0))

    # 8. warp
    t = time.perf_counter()
    warped = warp_to_reference(src_img, T, ref_img.shape) if cfg.get("warp", True) else None
    timing["warp_s"] = time.perf_counter() - t
    timing["total_s"] = time.perf_counter() - t0

    result.update(ok=True, transform=T, metrics=m, pts_src=ps, pts_ref=mr, scores=sc,
                  checkpoint=hold, residual=residuals(T, ps, mr), refined=refined_ok,
                  warped=warped)
    return result
