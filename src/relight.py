"""Sun-matched reference rendering (novelty 1) + shadow-aware filtering (novelty 2).

Instead of matching the Chandrayaan-2 image against a reference photo taken under a
different sun, we match it against the reference DTM rendered under the Chandrayaan-2
image's OWN sun azimuth/elevation (from its label). Because the DTM is co-registered with
the reference orthophoto (same LROC product), the matches give the source->reference
transform directly.

The DTM (3 m) is coarser than the images, so matching runs on a working grid of
`work_px` reference pixels (default: the DTM spacing); points are converted back to the
reference grid for the final fit, evaluation and warp.

Usage:
    python -m src.relight --pair data/processed/p01b_fix --config configs/relight.yaml --out results/p01b_relight
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import metrics
from .geometry import residuals
from .pipeline import register, warp_to_reference
from .synthetic import cast_shadows, render
from .viz import flicker_gif, quicklook


def block_mean(a: np.ndarray, k: int) -> np.ndarray:
    """Area-average by an integer factor k (NaN-aware)."""
    if k == 1:
        return a.astype(np.float32)
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    b = a[:h, :w].reshape(h // k, k, w // k, k)
    import warnings
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(b, axis=(1, 3)).astype(np.float32)


def sun_render(dtm: np.ndarray, dx: float, az: float, el: float) -> tuple[np.ndarray, np.ndarray]:
    """Lunar-Lambert render of the DTM with cast shadows; returns (image, cast-shadow mask)."""
    fill = np.where(np.isfinite(dtm), dtm, np.nanmean(dtm)).astype(np.float32)
    el = max(el, 0.5)
    img = render(fill, dx, az, el, shadows=True)
    sh = cast_shadows(fill, dx, az, el)
    img[~np.isfinite(dtm)] = np.nan
    return img, sh


def to_fine(p: np.ndarray, k: int) -> np.ndarray:
    """Working-grid pixel centre -> reference-grid pixel coordinate."""
    return p * k + (k - 1) / 2.0


def calibrate_dtm(pair: Path, cfg: dict) -> dict:
    """Measure the DTM-vs-reference-orthophoto offset once per site/product.

    Renders the DTM under the REFERENCE image's own sun (estimated by a grid search on
    render/reference correlation, since LROC ortho labels omit it), matches the reference
    to that render and returns the median offset (render minus ortho) in metres. The same
    product's DTM and ortho should agree; any residual here would otherwise bias every
    DTM-based registration.
    """
    meta = json.loads((pair / "meta.json").read_text())
    ref = cv2.imread(str(pair / "reference.tif"), cv2.IMREAD_UNCHANGED)
    dtm = cv2.imread(str(pair / "dtm_ref.tif"), cv2.IMREAD_UNCHANGED)
    dx = float(meta["reference"]["res_m"])
    k = int(cfg.get("relight", {}).get("work_px") or max(1, round(3.0 / dx)))
    R, Dk = block_mean(ref, k), block_mean(dtm, k)
    fill = np.where(np.isfinite(Dk), Dk, np.nanmean(Dk)).astype(np.float32)

    def corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1])

    coarse = max((corr(render(fill, dx * k, az, el), R), az, el)
                 for az in range(0, 360, 15) for el in (5, 10, 20, 35))
    fine = max((corr(render(fill, dx * k, az, el), R), az, el)
               for az in range(int(coarse[1]) - 12, int(coarse[1]) + 13, 3)
               for el in np.arange(max(2, coarse[2] - 5), coarse[2] + 6, 1.0))
    rend, _ = sun_render(Dk, dx * k, fine[1], float(fine[2]))
    c = copy.deepcopy(cfg)
    c["warp"] = False
    r = register(R, rend, c)
    if not r["ok"]:
        return {"ok": False, "reason": r["reason"]}
    d = (to_fine(r["pts_ref"], k) - to_fine(r["pts_src"], k)) * dx
    cal = {"ok": True, "ref_sun_est": [float(fine[1]), float(fine[2])], "render_corr": fine[0],
           "dtm_minus_ortho_offset_m": np.median(d, 0).tolist(), "std_m": d.std(0).tolist(), "n": int(len(d))}
    (pair / "dtm_calibration.json").write_text(json.dumps(cal, indent=2))
    return cal


def run(pair: Path, cfg: dict, out: Path) -> dict:
    meta = json.loads((pair / "meta.json").read_text())
    src = cv2.imread(str(pair / "source_mapped.tif"), cv2.IMREAD_UNCHANGED)
    ref = cv2.imread(str(pair / "reference.tif"), cv2.IMREAD_UNCHANGED)
    dtm = cv2.imread(str(pair / "dtm_ref.tif"), cv2.IMREAD_UNCHANGED)
    dx = float(meta["reference"]["res_m"])
    rc = cfg.get("relight", {})
    k = int(rc.get("work_px", max(1, round(3.0 / dx))))
    az, el = meta["source"]["sun_az_deg"], meta["source"]["sun_el_deg"]

    src_k = block_mean(src, k)
    rend_k, shadow_k = sun_render(block_mean(dtm, k), dx * k, az, el)

    # novelty 2: shadow-aware filtering -> forbid detection on/near DTM-predicted cast-shadow
    # edges, where small relief missing from the DTM makes image and render disagree
    src_mask = ref_mask = None
    if rc.get("shadow_filter", False):
        e = int(rc.get("shadow_edge_px", 2))
        edge = cv2.morphologyEx(shadow_k.astype(np.uint8), cv2.MORPH_GRADIENT,
                                np.ones((3, 3), np.uint8)).astype(bool)
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((2 * e + 1, 2 * e + 1), np.uint8)).astype(bool)
        src_mask = ref_mask = ~edge

    c = copy.deepcopy(cfg)
    c["warp"] = False
    r = register(src_k, rend_k, c, scale_hint=None, extra_src_mask=src_mask, extra_ref_mask=ref_mask)
    result = {"pair": pair.name, "work_px": k, "sun": [az, el], "ok": r["ok"], "n_tentative": r["n_tentative"]}
    if not r["ok"]:
        result["reason"] = r["reason"]
        return result

    ps, pr = to_fine(r["pts_src"], k), to_fine(r["pts_ref"], k)
    calf = pair / "dtm_calibration.json"
    offset_m = [0.0, 0.0]
    if rc.get("use_calibration", True) and calf.exists():
        offset_m = json.loads(calf.read_text())["dtm_minus_ortho_offset_m"]
        pr = pr - np.asarray(offset_m) / dx              # render position -> ortho position
    result["dtm_offset_applied_m"] = offset_m
    kind = cfg.get("final_model", "poly3")
    ecfg = cfg.get("evaluation", {})
    m, T, hold = metrics.evaluate(ps, pr, kind, src.shape, r["n_tentative"], ecfg.get("holdout", 0.25),
                                  cfg.get("seed", 42), ecfg.get("grid", 8), dx)
    m["quality_pass"] = bool(m["n_inliers"] >= cfg.get("quality", {}).get("min_inliers", 20)
                             and m["checkpoint_rmse_px"] <= k * cfg.get("quality", {}).get("max_checkpoint_rmse_px", 1.0))
    warped = warp_to_reference(src, T, ref.shape)
    out.mkdir(parents=True, exist_ok=True)
    res = {"pts_src": ps, "pts_ref": pr, "checkpoint": hold, "transform": T, "metrics": m,
           "residual": residuals(T, ps, pr)}
    quicklook(src, ref, warped, res, out / "quicklook.png", f"{pair.name}: matched to DTM rendered at the source sun")
    flicker_gif(ref, warped, out / "flicker.gif")
    cv2.imwrite(str(out / "registered.tif"), warped.astype(np.float32))
    cv2.imwrite(str(out / "render_at_source_sun.png"),
                cv2.normalize(np.nan_to_num(rend_k), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    np.savetxt(out / "matches.csv", np.column_stack([ps, pr, hold]), delimiter=",", fmt="%.4f",
               header="src_x,src_y,ref_x,ref_y,checkpoint", comments="")
    (out / "transform.json").write_text(json.dumps(T.to_dict(), indent=2))
    result.update(metrics=m, units=f"reference pixels ({dx} m); matching on {k}x{k} blocks")
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=float))
    result["_T"] = T
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--config", default="configs/relight.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calibrate", action="store_true", help="measure the DTM-vs-ortho offset first")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.calibrate:
        print("calibration:", json.dumps(calibrate_dtm(Path(a.pair), cfg), default=float))
    r = run(Path(a.pair), cfg, Path(a.out))
    r.pop("_T", None)
    print(json.dumps(r, indent=2, default=float))


if __name__ == "__main__":
    main()
