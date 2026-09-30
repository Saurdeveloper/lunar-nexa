"""Feature detection and matching.

Returns tentative correspondences as (pts_src, pts_ref, score) where points are
(x, y) pixel coordinates of the images passed in, and score is higher = better.
"""
from __future__ import annotations

import cv2
import numpy as np


def _to_u8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _make_detector(method: str, cfg: dict):
    if method == "sift":
        return cv2.SIFT_create(nfeatures=0, contrastThreshold=cfg.get("contrast_threshold", 0.01),
                               edgeThreshold=cfg.get("edge_threshold", 10)), cv2.NORM_L2
    if method == "rootsift":
        return cv2.SIFT_create(nfeatures=0, contrastThreshold=cfg.get("contrast_threshold", 0.01)), cv2.NORM_L2
    if method == "akaze":
        return cv2.AKAZE_create(threshold=cfg.get("akaze_threshold", 0.0005)), cv2.NORM_HAMMING
    if method == "orb":
        return cv2.ORB_create(nfeatures=cfg.get("max_features", 8000), fastThreshold=5), cv2.NORM_HAMMING
    raise ValueError(f"unknown classical method {method}")


def _grid_keep(kps: list, desc: np.ndarray, shape: tuple[int, int], grid: int, per_cell: int):
    """Keep the strongest `per_cell` keypoints in each grid cell, so every region gets candidates."""
    if len(kps) == 0:
        return kps, desc
    h, w = shape
    pts = np.array([k.pt for k in kps])
    resp = np.array([k.response for k in kps])
    cell = (np.minimum((pts[:, 1] * grid / h).astype(int), grid - 1) * grid
            + np.minimum((pts[:, 0] * grid / w).astype(int), grid - 1))
    keep = []
    for c in np.unique(cell):
        idx = np.where(cell == c)[0]
        keep.extend(idx[np.argsort(-resp[idx])[:per_cell]])
    keep = np.array(sorted(keep))
    return [kps[i] for i in keep], desc[keep]


def detect(img: np.ndarray, method: str, cfg: dict, mask: np.ndarray | None = None):
    """Detect keypoints and descriptors. `mask` = True where detection is ALLOWED."""
    det, norm = _make_detector(method, cfg)
    m = None if mask is None else mask.astype(np.uint8) * 255
    kps, desc = det.detectAndCompute(_to_u8(img), m)
    if desc is None:
        return [], None, norm
    if method == "rootsift":
        desc = desc / (np.abs(desc).sum(axis=1, keepdims=True) + 1e-7)
        desc = np.sqrt(desc).astype(np.float32)
    g = cfg.get("detect_grid", 16)
    kps, desc = _grid_keep(list(kps), desc, img.shape, g, cfg.get("per_cell", 40))
    return kps, desc, norm


def match_classical(src: np.ndarray, ref: np.ndarray, method: str, cfg: dict,
                    src_mask: np.ndarray | None = None, ref_mask: np.ndarray | None = None):
    """Detect + describe + ratio-test + mutual-check matching."""
    k1, d1, norm = detect(src, method, cfg, src_mask)
    k2, d2, _ = detect(ref, method, cfg, ref_mask)
    empty = (np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0))
    if d1 is None or d2 is None or len(k1) < 2 or len(k2) < 2:
        return empty
    bf = cv2.BFMatcher(norm)
    fwd = bf.knnMatch(d1, d2, k=2)
    bwd = {m[0].queryIdx: m[0].trainIdx for m in bf.knnMatch(d2, d1, k=1) if m}
    ratio = cfg.get("ratio", 0.85)
    ps, pr, sc = [], [], []
    for pair in fwd:
        if len(pair) < 2:
            continue
        a, b = pair
        r = a.distance / max(b.distance, 1e-9)
        if r < ratio and (not cfg.get("mutual", True) or bwd.get(a.trainIdx) == a.queryIdx):
            ps.append(k1[a.queryIdx].pt)
            pr.append(k2[a.trainIdx].pt)
            sc.append(1.0 - r)
    if not ps:
        return empty
    return np.array(ps), np.array(pr), np.array(sc)


def match_lightglue(src: np.ndarray, ref: np.ndarray, cfg: dict,
                    src_mask: np.ndarray | None = None, ref_mask: np.ndarray | None = None):
    """SuperPoint + LightGlue (needs `torch` and `lightglue`). Uses GPU if available."""
    import torch
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ext = SuperPoint(max_num_keypoints=cfg.get("max_features", 4096)).eval().to(dev)
    lg = LightGlue(features="superpoint").eval().to(dev)
    t = lambda im: torch.from_numpy(np.ascontiguousarray(im, np.float32))[None, None].to(dev)
    with torch.inference_mode():
        f0, f1 = ext.extract(t(src)), ext.extract(t(ref))
        out = lg({"image0": f0, "image1": f1})
    f0, f1, out = [rbd(x) for x in (f0, f1, out)]
    m = out["matches"].cpu().numpy()
    ps = f0["keypoints"].cpu().numpy()[m[:, 0]]
    pr = f1["keypoints"].cpu().numpy()[m[:, 1]]
    sc = out["scores"].cpu().numpy()
    keep = np.ones(len(ps), bool)
    for pts, msk in ((ps, src_mask), (pr, ref_mask)):
        if msk is not None:
            xi = np.clip(pts[:, 0].round().astype(int), 0, msk.shape[1] - 1)
            yi = np.clip(pts[:, 1].round().astype(int), 0, msk.shape[0] - 1)
            keep &= msk[yi, xi]
    return ps[keep].astype(np.float64), pr[keep].astype(np.float64), sc[keep].astype(np.float64)


def match(src: np.ndarray, ref: np.ndarray, cfg: dict,
          src_mask: np.ndarray | None = None, ref_mask: np.ndarray | None = None):
    """Dispatch to the configured matcher."""
    method = cfg.get("method", "sift")
    if method == "lightglue":
        return match_lightglue(src, ref, cfg, src_mask, ref_mask)
    return match_classical(src, ref, method, cfg, src_mask, ref_mask)
