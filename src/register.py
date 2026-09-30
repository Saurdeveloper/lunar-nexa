"""Command-line entry point: register one source image to one reference image.

Usage:
    python -m src.register --src data/processed/ohrc.tif --ref data/processed/nac.tif \
        --src-res 0.25 --ref-res 0.5 --config configs/baseline.yaml --out results/pair01

    # no data yet? make a synthetic pair first:
    python -m src.make_demo_pair --out data/synthetic
    python -m src.register --src data/synthetic/source.tif --ref data/synthetic/reference.tif \
        --src-res 1 --ref-res 2 --out results/demo

Outputs in --out:
    registered.tif     source resampled onto the reference grid (GeoTIFF with the reference CRS
                       and transform when rasterio is installed and the reference is georeferenced)
    matches.csv        src_x, src_y, ref_x, ref_y, [map_x, map_y, lon, lat], score, inlier,
                       checkpoint, residual_px, refined
    metrics.json       checkpoint/fit RMSE, inlier count and ratio, sub-pixel rate, coverage, timings
    transform.json     fitted source->reference pixel transform
    quicklook.png      matches, residual vectors and checkerboard overlay
    flicker.gif        reference vs registered source
    config_used.yaml   exact config of this run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .io_utils import pixel_to_map, read_image, write_image
from .pipeline import register
from .viz import flicker_gif, quicklook


def _lonlat(georef: dict | None, xy: np.ndarray | None) -> np.ndarray | None:
    """Map coords -> lunar lon/lat (degrees) if pyproj is available."""
    if georef is None or xy is None:
        return None
    try:
        from pyproj import CRS, Transformer
        crs = CRS.from_user_input(georef["crs"].to_wkt())
        tr = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
        lon, lat = tr.transform(xy[:, 0], xy[:, 1])
        return np.stack([lon, lat], 1)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source (moving) image, e.g. Chandrayaan-2 OHRC/TMC-2/IIRS")
    ap.add_argument("--ref", required=True, help="reference (fixed) image, e.g. LRO NAC / SELENE")
    ap.add_argument("--src-res", type=float, help="source pixel size (m); from the product label")
    ap.add_argument("--ref-res", type=float, help="reference pixel size (m); from the product label")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--out", default="results/run")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src, src_geo = read_image(a.src)
    ref, ref_geo = read_image(a.ref)
    if ref_geo and not a.ref_res:
        a.ref_res = abs(ref_geo["res"][0])
    if src_geo and not a.src_res:
        a.src_res = abs(src_geo["res"][0])
    hint = a.src_res / a.ref_res if (a.src_res and a.ref_res) else None
    if a.ref_res:
        cfg.setdefault("evaluation", {})["ref_px_size_m"] = a.ref_res

    r = register(src, ref, cfg, scale_hint=hint)
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg))
    if not r["ok"]:
        (out / "metrics.json").write_text(json.dumps({"ok": False, "reason": r["reason"],
                                                      "n_tentative": r["n_tentative"]}, indent=2))
        raise SystemExit(f"registration failed: {r['reason']}")

    reg_path = write_image(out / "registered.tif", r["warped"], ref_geo)
    df = pd.DataFrame({"src_x": r["pts_src"][:, 0], "src_y": r["pts_src"][:, 1],
                       "ref_x": r["pts_ref"][:, 0], "ref_y": r["pts_ref"][:, 1]})
    mxy = pixel_to_map(ref_geo, r["pts_ref"])
    if mxy is not None:
        df["map_x"], df["map_y"] = mxy[:, 0], mxy[:, 1]
        ll = _lonlat(ref_geo, mxy)
        if ll is not None:
            df["lon"], df["lat"] = ll[:, 0], ll[:, 1]
    df["score"] = r["scores"]
    df["inlier"] = True
    df["checkpoint"] = r["checkpoint"]
    df["residual_px"] = r["residual"]
    df["refined"] = r["refined"]
    df.to_csv(out / "matches.csv", index=False, float_format="%.4f")

    m = {"ok": True, **r["metrics"], "scale_hint": hint, "timing_s": r["timing"],
         "registered": str(reg_path.name), "georeferenced": bool(ref_geo)}
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    (out / "transform.json").write_text(json.dumps(r["transform"].to_dict(), indent=2))
    quicklook(src, ref, r["warped"], r, out / "quicklook.png", Path(a.src).stem)
    flicker_gif(ref, r["warped"], out / "flicker.gif")
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
