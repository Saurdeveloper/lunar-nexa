"""Cut an LROC NAC DTM (32-bit GeoTIFF, uncompressed strips) onto a reference crop grid.

The NAC_DTM_<SITE>.TIF shares its pixel grid with the <SITE>_*_3M orthophoto (same size and
bounds), so DTM pixel (col,row) == 3 m ortho pixel (sample,line). Any other ortho of the same
site (e.g. 1 m) is related to it through the PDS projection offsets in the labels.

Usage (normally called from prep_pair.py --dtm/--dtm-grid-img):
    python -m src.dtm --dtm NAC_DTM_VIKRAMSITE1.TIF --grid-img ..._3M.IMG --ref-img ..._100CM.IMG \
        --meta data/processed/p01a/meta.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .prep_pair import read_pds3_label

Image.MAX_IMAGE_PIXELS = None


class StripDTM:
    """Row-addressable view of an uncompressed float32 GeoTIFF with one row per strip.

    Strips may be stored in any order in the file (LROC DTMs are), so rows are fetched
    through the StripOffsets table instead of assuming a contiguous block.
    """

    def __init__(self, path: Path):
        im = Image.open(path)
        t = im.tag_v2
        if t.get(259, 1) != 1 or t.get(278, 1) != 1 or 322 in t:
            raise ValueError("need uncompressed, 1-row strips: gdal_translate -co COMPRESS=NONE -co BLOCKYSIZE=1")
        self.w, self.h = im.size
        self.offs = np.asarray(t[273], np.int64)
        self.raw = np.memmap(path, dtype=np.uint8, mode="r")
        self.shape = (self.h, self.w)

    def __getitem__(self, idx):
        rows, cols = idx
        r = range(*rows.indices(self.h))
        out = np.empty((len(r), self.w), np.float32)
        for i, row in enumerate(r):
            o = int(self.offs[row])
            out[i] = np.frombuffer(self.raw[o:o + 4 * self.w].tobytes(), "<f4")
        return out[:, cols]


def dtm_memmap(path: Path) -> "StripDTM":
    return StripDTM(path)


def dtm_on_ref_grid(dtm: np.ndarray, grid_lab: dict, ref_lab: dict, c0: int, r0: int, w: int, h: int,
                    nodata_below: float = -1e30) -> np.ndarray:
    """DTM heights (m) resampled (bilinear) onto the reference crop [r0:r0+h, c0:c0+w]."""
    s_ref, s_g = float(ref_lab["MAP_SCALE"]), float(grid_lab["MAP_SCALE"])
    so_r, lo_r = float(ref_lab["SAMPLE_PROJECTION_OFFSET"]), float(ref_lab["LINE_PROJECTION_OFFSET"])
    so_g, lo_g = float(grid_lab["SAMPLE_PROJECTION_OFFSET"]), float(grid_lab["LINE_PROJECTION_OFFSET"])
    cols = np.arange(c0, c0 + w, dtype=np.float64)
    rows = np.arange(r0, r0 + h, dtype=np.float64)
    x = (cols + 0.5 - so_r) * s_ref                    # same convention as prep_pair.NacGrid
    y = (lo_r - rows - 0.5) * s_ref
    gc = x / s_g + so_g - 0.5
    gr = lo_g - y / s_g - 0.5
    ca, cb = int(np.floor(gc.min())) - 2, int(np.ceil(gc.max())) + 3
    ra, rb = int(np.floor(gr.min())) - 2, int(np.ceil(gr.max())) + 3
    H, W = dtm.shape
    win = np.full((rb - ra, cb - ca), np.nan, np.float32)
    rs, re_, cs, ce = max(ra, 0), min(rb, H), max(ca, 0), min(cb, W)
    sub = np.asarray(dtm[rs:re_, cs:ce], np.float32)
    sub[sub < nodata_below] = np.nan
    win[rs - ra:re_ - ra, cs - ca:ce - ca] = sub
    mx, my = np.meshgrid((gc - ca).astype(np.float32), (gr - ra).astype(np.float32))
    return cv2.remap(win, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtm", required=True)
    ap.add_argument("--grid-img", required=True, help="the 3 m ortho .IMG that shares the DTM grid")
    ap.add_argument("--ref-img", required=True, help="the ortho .IMG used as reference in prep_pair")
    ap.add_argument("--meta", required=True, help="meta.json written by prep_pair")
    a = ap.parse_args()
    meta = json.loads(Path(a.meta).read_text())
    r = meta["reference"]
    dem = dtm_on_ref_grid(dtm_memmap(Path(a.dtm)), read_pds3_label(Path(a.grid_img)),
                          read_pds3_label(Path(a.ref_img)), r["crop_col0"], r["crop_row0"], r["crop_w"], r["crop_h"])
    out = Path(a.meta).parent / "dtm_ref.tif"
    cv2.imwrite(str(out), dem.astype(np.float32))
    meta["dtm"] = {"file": Path(a.dtm).name, "grid_img": Path(a.grid_img).name,
                   "valid_frac": float(np.isfinite(dem).mean()),
                   "min_m": float(np.nanmin(dem)), "max_m": float(np.nanmax(dem))}
    Path(a.meta).write_text(json.dumps(meta, indent=2))
    print(out, meta["dtm"])


if __name__ == "__main__":
    main()
