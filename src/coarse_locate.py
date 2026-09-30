"""Illumination-matched coarse localisation (novelty 1, sun-matched DTM rendering).

The Chandrayaan-2 image, already put on the reference map grid from its own geometry,
may still be off by hundreds of metres. We render the NAC DTM under the SOURCE image's
sun azimuth/elevation (read from its label), over a search window much larger than the
image, and find the image inside that render by normalised cross-correlation. Because
render and image share the same lighting, shadows and shading agree, which plain
image-to-image matching under different sun angles cannot rely on.

Usage:
    python -m src.coarse_locate --pair data/processed/p01b --dtm .../NAC_DTM_VIKRAMSITE1.TIF \
        --grid-img .../_3M.IMG --ref-img .../_100CM.IMG --search-m 2000
Writes meta["coarse_locate"] with the offset (reference pixels and metres) and its reliability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .dtm import StripDTM
from .prep_pair import read_pds3_label
from .synthetic import render


def ref_to_grid(c, r, ref_lab, grid_lab):
    s_r, s_g = float(ref_lab["MAP_SCALE"]), float(grid_lab["MAP_SCALE"])
    x = (np.asarray(c, float) + 0.5 - float(ref_lab["SAMPLE_PROJECTION_OFFSET"])) * s_r
    y = (float(ref_lab["LINE_PROJECTION_OFFSET"]) - np.asarray(r, float) - 0.5) * s_r
    return x / s_g + float(grid_lab["SAMPLE_PROJECTION_OFFSET"]) - 0.5, float(grid_lab["LINE_PROJECTION_OFFSET"]) - y / s_g - 0.5


def zs(a):
    a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1.0)
    return (a - a.mean()) / (a.std() + 1e-9)


def locate(pair: Path, dtm_path: Path, grid_img: Path, ref_img: Path, search_m: float,
           search_y_m: float | None = None, shadows: bool = True) -> dict:
    meta = json.loads((pair / "meta.json").read_text())
    src = cv2.imread(str(pair / "source_mapped.tif"), cv2.IMREAD_UNCHANGED)
    rl, gl = read_pds3_label(ref_img), read_pds3_label(grid_img)
    s_r, s_g = float(rl["MAP_SCALE"]), float(gl["MAP_SCALE"])
    k = int(round(s_g / s_r))                     # e.g. 3 for 1 m -> 3 m
    ys, xs = np.where(np.isfinite(src))
    pad = 30 * k
    y0, y1, x0, x1 = ys.min() + pad, ys.max() - pad, xs.min() + pad, xs.max() - pad
    y1, x1 = y0 + (y1 - y0) // k * k, x0 + (x1 - x0) // k * k
    tpl = src[y0:y1, x0:x1]
    tpl = np.where(np.isfinite(tpl), tpl, np.nanmean(tpl)).astype(np.float32)
    if k > 1:
        tpl = tpl.reshape(tpl.shape[0] // k, k, tpl.shape[1] // k, k).mean((1, 3))
    # where the template SHOULD be on the DTM grid (from metadata), top-left block centre
    r = meta["reference"]
    gc0, gr0 = ref_to_grid(r["crop_col0"] + x0 + (k - 1) / 2, r["crop_row0"] + y0 + (k - 1) / 2, rl, gl)
    S = int(round(search_m / s_g))                     # across-track (x) half-range
    Sy = int(round((search_y_m or search_m) / s_g))     # along-track (y) half-range
    dtm = StripDTM(dtm_path)
    ca, ra = int(round(gc0)) - S, int(round(gr0)) - Sy
    cb, rb = ca + tpl.shape[1] + 2 * S, ra + tpl.shape[0] + 2 * Sy
    ca_, ra_, cb_, rb_ = max(ca, 0), max(ra, 0), min(cb, dtm.w), min(rb, dtm.h)
    dem = np.full((rb - ra, cb - ca), np.nan, np.float32)
    dem[ra_ - ra:rb_ - ra, ca_ - ca:cb_ - ca] = dtm[slice(ra_, rb_), slice(ca_, cb_)]
    dem[dem < -1e30] = np.nan
    dem = np.where(np.isfinite(dem), dem, np.nanmean(dem)).astype(np.float32)
    az, el = meta["source"]["sun_az_deg"], max(meta["source"]["sun_el_deg"], 1.0)
    rend = render(dem, s_g, az, el, shadows=shadows)
    ncc = cv2.matchTemplate(zs(rend), zs(tpl), cv2.TM_CCOEFF_NORMED)
    _, peak, _, (px, py) = cv2.minMaxLoc(ncc)
    # second peak outside a +-10 px exclusion zone -> distinctiveness
    n2 = ncc.copy()
    n2[max(0, py - 25):py + 26, max(0, px - 25):px + 26] = -1
    peak2 = float(n2.max())
    _, _, _, (qx, qy) = cv2.minMaxLoc(n2)
    # sub-pixel peak (parabola)
    def sub(a, i, n):
        if 0 < i < n - 1:
            d = a[i - 1] - 2 * a[i] + a[i + 1]
            return i + 0.5 * (a[i - 1] - a[i + 1]) / d if d != 0 else i
        return i
    fx, fy = sub(ncc[py, :], px, ncc.shape[1]), sub(ncc[:, px], py, ncc.shape[0])
    dgx, dgy = float(fx - S), float(fy - Sy)                  # grid (3 m) pixels: true minus metadata position
    # in reference pixels: the image content sits at (metadata + d); to fix, shift source by +d
    out = {"sun_used": [float(az), float(el)], "grid_scale_m": s_g, "search_m": [search_m, search_y_m or search_m],
           "offset_ref_px": [dgx * k, dgy * k], "offset_m": [dgx * s_g, dgy * s_g],
           "ncc_peak": float(peak), "ncc_second": peak2, "peak_ratio": float(peak / max(peak2, 1e-6)),
           "ncc_at_metadata_pos": float(ncc[Sy, S]),
           "second_offset_m": [float((qx - S) * s_g), float((qy - Sy) * s_g)]}
    meta["coarse_locate"] = out
    (pair / "meta.json").write_text(json.dumps(meta, indent=2))
    ql = np.hstack([cv2.normalize(rend[int(round(fy)):int(round(fy)) + tpl.shape[0], int(round(fx)):int(round(fx)) + tpl.shape[1]], None, 0, 255, cv2.NORM_MINMAX),
                    cv2.normalize(tpl, None, 0, 255, cv2.NORM_MINMAX)]).astype(np.uint8)
    cv2.imwrite(str(pair / "coarse_locate.png"), ql)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--dtm", required=True)
    ap.add_argument("--grid-img", required=True)
    ap.add_argument("--ref-img", required=True)
    ap.add_argument("--search-m", type=float, default=2000.0)
    ap.add_argument("--search-y-m", type=float, default=None, help="larger along-track range (y)")
    ap.add_argument("--no-shadows", action="store_true", help="skip cast shadows (saves memory on big searches)")
    a = ap.parse_args()
    print(json.dumps(locate(Path(a.pair), Path(a.dtm), Path(a.grid_img), Path(a.ref_img), a.search_m, a.search_y_m, not a.no_shadows), indent=1))


if __name__ == "__main__":
    main()
