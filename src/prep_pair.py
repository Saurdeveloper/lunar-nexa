"""Phase 3: cut a site out of a Chandrayaan-2 product and an LROC NAC orthophoto and
put the Chandrayaan-2 image on the NAC map grid (coarse alignment from metadata only).

Works with numpy + OpenCV + pandas only (no GDAL needed).

Inputs
    --ch2-zip   Chandrayaan-2 calibrated product zip (OHRC or TMC-2) as downloaded from ISSDC
    --nac-img   LROC NAC orthophoto .IMG (PDS3, attached label, polar stereographic or equirectangular)
    --lat/--lon site centre (deg, planetocentric, east-positive)
    --half-m    half-size of the cut-out on the ground (m)

Outputs in --out
    source_raw.tif        Chandrayaan-2 cut-out in its own (raw, unprojected) pixel grid
    source_mapped.tif     same cut-out resampled onto the reference grid (coarse, metadata-only)
    reference.tif         NAC cut-out (I/F reflectance, float32, NaN = no data)
    *.tfw / *.prj         world file + projection so QGIS/GDAL place the images on the Moon
    meta.json             sun angles, resolutions, crop offsets, projection, raw->ref polynomial
    quicklook.png         side-by-side + checkerboard for a visual check

Coordinates: (x, y) = (sample, line) = (column, row), 0-based, pixel centres.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

R_MOON = 1737400.0


# --------------------------------------------------------------------------- NAC (PDS3)
def read_pds3_label(path: Path) -> dict:
    """Parse the attached PDS3 label of an LROC RDR .IMG into a flat dict (last value wins)."""
    head = open(path, "rb").read(200_000).decode("latin-1", "ignore")
    head = head.split("\nEND\r\n")[0].split("\nEND\n")[0]
    out = {}
    for m in re.finditer(r"^\s*([A-Z0-9_^:]+)\s*=\s*(.+?)\s*$", head, re.M):
        v = m.group(2).split("<")[0].strip().strip('"')
        try:
            v = float(v)
        except ValueError:
            pass
        out[m.group(1)] = v
    return out


class NacGrid:
    """Pixel <-> lat/lon for an LROC NAC RDR orthophoto (polar stereographic or equirectangular)."""

    def __init__(self, lab: dict):
        self.lab = lab
        self.proj = str(lab["MAP_PROJECTION_TYPE"]).upper()
        self.scale = float(lab["MAP_SCALE"])
        self.loff = float(lab["LINE_PROJECTION_OFFSET"])
        self.soff = float(lab["SAMPLE_PROJECTION_OFFSET"])
        self.clat = np.radians(float(lab["CENTER_LATITUDE"]))
        self.clon = np.radians(float(lab["CENTER_LONGITUDE"]))
        self.R = float(lab.get("A_AXIS_RADIUS", 1737.4)) * 1000.0

    def latlon_to_xy_m(self, lat, lon):
        lat, lon = np.radians(lat), np.radians(lon)
        if "POLAR" in self.proj:
            k = 1 + np.sin(abs(self.clat))                  # true scale at CENTER_LATITUDE
            if self.clat < 0:                                # south polar
                rho = self.R * k * np.tan(np.pi / 4 + lat / 2)
                return rho * np.sin(lon - self.clon), rho * np.cos(lon - self.clon)
            rho = self.R * k * np.tan(np.pi / 4 - lat / 2)
            return rho * np.sin(lon - self.clon), -rho * np.cos(lon - self.clon)
        # equirectangular
        return self.R * (lon - self.clon) * np.cos(self.clat), self.R * lat

    def latlon_to_pix(self, lat, lon):
        """0-based (sample, line)."""
        x, y = self.latlon_to_xy_m(lat, lon)
        return x / self.scale + self.soff - 0.5, self.loff - y / self.scale - 0.5

    def prj_wkt(self) -> str:
        geog = ('GEOGCS["Moon 2000",DATUM["D_Moon_2000",SPHEROID["Moon_2000_IAU_IAG",%.1f,0]],'
                'PRIMEM["Greenwich",0],UNIT["Decimal_Degree",0.0174532925199433]]' % self.R)
        if "POLAR" in self.proj:
            return ('PROJCS["Moon_PolarStereographic",%s,PROJECTION["Polar_Stereographic"],'
                    'PARAMETER["latitude_of_origin",%.6f],PARAMETER["central_meridian",%.6f],'
                    'PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],'
                    'UNIT["Meter",1]]' % (geog, np.degrees(self.clat), np.degrees(self.clon)))
        return ('PROJCS["Moon_Equirectangular",%s,PROJECTION["Equirectangular"],'
                'PARAMETER["standard_parallel_1",%.6f],PARAMETER["central_meridian",%.6f],'
                'PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1]]'
                % (geog, np.degrees(self.clat), np.degrees(self.clon)))

    def world_file(self, col0: int, row0: int) -> str:
        """ESRI world file for a crop starting at (col0, row0). Map x,y of the top-left pixel centre."""
        s = self.scale
        x = (col0 - self.soff + 0.5) * s
        y = (self.loff - row0 - 0.5) * s
        if "POLAR" in self.proj and self.clat < 0:
            # our south-polar formulas use +y away from the pole; GDAL's polar stereographic uses
            # y = -rho cos(dlon) for the south pole, so flip the sign of y
            return f"{s}\n0\n0\n{s}\n{x}\n{-y}\n"
        return f"{s}\n0\n0\n{-s}\n{x}\n{y}\n"


def read_nac_window(path: Path, lab: dict, c0: int, r0: int, w: int, h: int) -> np.ndarray:
    """Read a window of a NAC RDR .IMG as I/F float32 (NaN = null) without loading the file."""
    rb, nrec = int(lab["RECORD_BYTES"]), int(lab["LABEL_RECORDS"])
    lines, samples = int(lab["LINES"]), int(lab["LINE_SAMPLES"])
    bits = int(lab["SAMPLE_BITS"])
    dt = {16: "<u2", 32: "<f4", 8: "u1"}[bits]
    mm = np.memmap(path, dtype=dt, mode="r", offset=rb * nrec, shape=(lines, samples))
    r0c, c0c = max(0, r0), max(0, c0)
    r1c, c1c = min(lines, r0 + h), min(samples, c0 + w)
    out = np.full((h, w), np.nan, np.float32)
    raw = np.asarray(mm[r0c:r1c, c0c:c1c]).astype(np.float32)
    val = raw * float(lab.get("SCALING_FACTOR", 1.0)) + float(lab.get("OFFSET", 0.0))
    val[raw == float(lab.get("CORE_NULL", 0))] = np.nan
    out[r0c - r0:r1c - r0, c0c - c0:c1c - c0] = val
    return out


# ----------------------------------------------------------------------- Chandrayaan-2 zip
def ch2_members(zpath: Path) -> dict:
    z = zipfile.ZipFile(zpath)
    names = z.namelist()
    pick = lambda pat: next(n for n in names if re.search(pat, n))
    return {"zip": z, "img": pick(r"data/.*_d_img_.*\.img$"), "xml": pick(r"data/.*_d_img_.*\.xml$"),
            "grd": pick(r"geometry/.*\.csv$")}


def ch2_label(xml: str) -> dict:
    get = lambda tag: float(re.search(rf"<isda:{tag}[^>]*>([-\d.eE+]+)<", xml).group(1))
    elems = [int(x) for x in re.findall(r"<elements>(\d+)</elements>", xml)]
    dtype = re.search(r"<data_type>(\w+)</data_type>", xml).group(1)
    return {"lines": elems[0], "samples": elems[1],
            "dtype": {"UnsignedByte": "u1", "UnsignedLSB2": "<u2", "IEEE754LSBSingle": "<f4"}[dtype],
            "res_m": get("pixel_resolution"), "sun_az_deg": get("sun_azimuth"),
            "sun_el_deg": get("sun_elevation"), "altitude_km": get("spacecraft_altitude"),
            "start_time": re.search(r"<start_date_time>([^<]+)<", xml).group(1)}


def extract_img(m: dict, dest: Path) -> Path:
    """Extract the (large) image member once; reuse afterwards."""
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / Path(m["img"]).name
    info = m["zip"].getinfo(m["img"])
    if not out.exists() or out.stat().st_size != info.file_size:
        with m["zip"].open(m["img"]) as src, open(out, "wb") as dst:
            while chunk := src.read(64 << 20):
                dst.write(chunk)
    return out


def poly_terms(x, y, order=3):
    return np.stack([x**i * y**(d - i) for d in range(order + 1) for i in range(d + 1)], 1)


def fit_poly(src: np.ndarray, dst: np.ndarray, order=3):
    c = src.mean(0)
    s = np.abs(src - c).max()
    A = poly_terms(*((src - c) / s).T, order)
    coef, *_ = np.linalg.lstsq(A, dst, rcond=None)
    f = lambda p: poly_terms(*((np.atleast_2d(p) - c) / s).T, order) @ coef
    return f, {"center": c.tolist(), "scale": float(s), "order": order, "coef": coef.tolist()}


# -------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ch2-zip", required=True)
    ap.add_argument("--nac-img", required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--half-m", type=float, default=1000.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin-px", type=int, default=100, help="extra reference pixels around the footprint")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- Chandrayaan-2 product
    m = ch2_members(Path(a.ch2_zip))
    lab = ch2_label(m["zip"].read(m["xml"]).decode())
    g = pd.read_csv(io.BytesIO(m["zip"].read(m["grd"])))
    # site -> raw pixel via a local polynomial on the geometry grid
    d = np.hypot(g.Latitude - a.lat, (g.Longitude - a.lon) * np.cos(np.radians(a.lat)))
    near = g[d < max(3 * a.half_m / 30300.0, 0.05)]
    to_raw, _ = fit_poly(near[["Longitude", "Latitude"]].to_numpy(), near[["Pixel", "Scan"]].to_numpy(), 2)
    cx, cy = to_raw(np.array([a.lon, a.lat]))[0]
    half = int(round(a.half_m / lab["res_m"]))
    x0 = int(max(0, min(lab["samples"] - 2 * half, round(cx) - half)))
    y0 = int(max(0, min(lab["lines"] - 2 * half, round(cy) - half)))
    w = min(2 * half, lab["samples"])
    h = min(2 * half, lab["lines"])
    img_path = extract_img(m, Path(a.ch2_zip).parent / "extracted")
    mm = np.memmap(img_path, dtype=lab["dtype"], mode="r", shape=(lab["lines"], lab["samples"]))
    src_raw = np.asarray(mm[y0:y0 + h, x0:x0 + w]).astype(np.float32)
    src_raw[src_raw == 0] = np.nan

    # --- reference grid: raw pixel -> lat/lon (grid) -> NAC pixel
    nlab = read_pds3_label(Path(a.nac_img))
    nac = NacGrid(nlab)
    sel = g[(g.Pixel >= x0 - 300) & (g.Pixel <= x0 + w + 300) & (g.Scan >= y0 - 300) & (g.Scan <= y0 + h + 300)]
    raw_xy = sel[["Pixel", "Scan"]].to_numpy(float) - [x0, y0]          # crop coords
    ns, nl = nac.latlon_to_pix(sel.Latitude.to_numpy(), sel.Longitude.to_numpy())
    nac_xy = np.stack([ns, nl], 1)
    fwd, fwd_meta = fit_poly(raw_xy, nac_xy, 3)
    fit_res = np.linalg.norm(fwd(raw_xy) - nac_xy, axis=1)
    corners = fwd(np.array([[0, 0], [w, 0], [w, h], [0, h]], float))
    c0 = int(np.floor(corners[:, 0].min())) - a.margin_px
    r0 = int(np.floor(corners[:, 1].min())) - a.margin_px
    rw = int(np.ceil(corners[:, 0].max())) + a.margin_px - c0
    rh = int(np.ceil(corners[:, 1].max())) + a.margin_px - r0
    ref = read_nac_window(Path(a.nac_img), nlab, c0, r0, rw, rh)

    # --- resample the raw cut-out onto the reference crop grid (coarse, metadata only)
    inv, inv_meta = fit_poly(nac_xy - [c0, r0], raw_xy, 3)
    gx, gy = np.meshgrid(np.arange(rw, dtype=float), np.arange(rh, dtype=float))
    p = inv(np.stack([gx.ravel(), gy.ravel()], 1))
    mx = p[:, 0].reshape(rh, rw).astype(np.float32)
    my = p[:, 1].reshape(rh, rw).astype(np.float32)
    ratio = nac.scale / lab["res_m"]
    blur = cv2.GaussianBlur(np.nan_to_num(src_raw), (0, 0), 0.5 * ratio) if ratio > 1.5 else np.nan_to_num(src_raw)
    mapped = cv2.remap(blur, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    valid = cv2.remap(np.isfinite(src_raw).astype(np.float32), mx, my, cv2.INTER_NEAREST,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mapped[valid < 0.5] = np.nan

    # --- write outputs
    cv2.imwrite(str(out / "source_raw.tif"), src_raw)
    cv2.imwrite(str(out / "source_mapped.tif"), mapped.astype(np.float32))
    cv2.imwrite(str(out / "reference.tif"), ref)
    for name in ("source_mapped", "reference"):
        (out / f"{name}.tfw").write_text(nac.world_file(c0, r0))
        (out / f"{name}.prj").write_text(nac.prj_wkt())
    site_ref = np.array(nac.latlon_to_pix(a.lat, a.lon)) - [c0, r0]
    meta = {
        "site": {"lat": a.lat, "lon": a.lon, "ref_px": site_ref.tolist(), "raw_px": [cx - x0, cy - y0]},
        "source": {"zip": Path(a.ch2_zip).name, **lab, "crop_x0": x0, "crop_y0": y0, "crop_w": w, "crop_h": h},
        "reference": {"img": Path(a.nac_img).name, "product_id": nlab.get("PRODUCT_ID"),
                      "start_time": nlab.get("START_TIME"), "res_m": nac.scale, "projection": nac.proj,
                      "crop_col0": c0, "crop_row0": r0, "crop_w": rw, "crop_h": rh},
        "raw_to_ref_poly3": fwd_meta, "ref_to_raw_poly3": inv_meta,
        "geometry_grid_fit_px": {"rms": float(np.sqrt((fit_res**2).mean())), "max": float(fit_res.max()),
                                 "n": int(len(fit_res))},
        "scale_ratio_ref_over_src": ratio,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    # --- quicklook: reference | coarse-mapped source | checkerboard
    def u8(a):
        v = a[np.isfinite(a)]
        lo, hi = np.percentile(v, [1, 99]) if v.size else (0, 1)
        return (np.clip((np.nan_to_num(a, nan=lo) - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
    A, B = u8(ref), u8(mapped)
    t = max(rw, rh) // 8
    yy, xx = np.mgrid[0:rh, 0:rw]
    chk = np.where(((yy // t) + (xx // t)) % 2 == 0, A, B)
    ql = np.hstack([A, B, chk])
    s = min(1.0, 2400 / ql.shape[1])
    cv2.imwrite(str(out / "quicklook.png"), cv2.resize(ql, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    print(json.dumps({k: meta[k] for k in ("site", "geometry_grid_fit_px", "scale_ratio_ref_over_src")}, indent=1))
    print("reference crop", rw, "x", rh, "| source crop", w, "x", h)


if __name__ == "__main__":
    main()
