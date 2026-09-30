"""Image input/output with georeferencing when rasterio is available.

`read_image` returns (array float32 2-D, georef dict or None).
georef = {"crs": ..., "transform": rasterio Affine, "res": (xres, yres)}.
Without rasterio, plain images (PNG/TIFF) are read with OpenCV and georef is None.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

try:
    import rasterio
    HAVE_RASTERIO = True
except ImportError:  # the pipeline still runs on plain images
    HAVE_RASTERIO = False


def read_image(path: str | Path, band: int = 1) -> tuple[np.ndarray, dict | None]:
    """Read one band as float32. Nodata becomes NaN when declared in the file."""
    path = Path(path)
    if HAVE_RASTERIO:
        try:
            with rasterio.open(path) as src:
                arr = src.read(band, masked=True).astype(np.float32).filled(np.nan)
                geo = {"crs": src.crs, "transform": src.transform, "res": src.res}
                if src.crs is None:
                    geo = None
                return arr, geo
        except rasterio.errors.RasterioIOError:
            pass
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"cannot read {path}")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return arr.astype(np.float32), None


def write_image(path: str | Path, arr: np.ndarray, georef: dict | None = None) -> Path:
    """Write a float32 image. GeoTIFF with CRS + transform when georef and rasterio exist;
    otherwise a float TIFF plus a JSON sidecar noting it is not georeferenced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr, np.float32)
    if HAVE_RASTERIO:
        profile = {"driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1], "count": 1,
                   "dtype": "float32", "compress": "deflate", "nodata": np.nan}
        if georef:
            profile.update(crs=georef["crs"], transform=georef["transform"])
        with rasterio.open(path.with_suffix(".tif"), "w", **profile) as dst:
            dst.write(arr, 1)
        return path.with_suffix(".tif")
    cv2.imwrite(str(path.with_suffix(".tif")), arr)
    path.with_suffix(".json").write_text(json.dumps({"georeferenced": False,
                                                     "note": "rasterio not installed"}))
    return path.with_suffix(".tif")


def pixel_to_map(georef: dict | None, pts: np.ndarray) -> np.ndarray | None:
    """Pixel (x, y) -> map coordinates using the raster's affine transform (pixel centres)."""
    if not georef:
        return None
    a = georef["transform"]
    x, y = pts[:, 0] + 0.5, pts[:, 1] + 0.5
    return np.stack([a.a * x + a.b * y + a.c, a.d * x + a.e * y + a.f], 1)


def save_u8(path: str | Path, img: np.ndarray) -> None:
    """Save a [0, 1] float image as 8-bit PNG (quick looks)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.clip(np.nan_to_num(img), 0, 1) * 255).astype(np.uint8))
