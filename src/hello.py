"""Phase 1 exit check: load an image with rasterio and report the GPU.

Usage:
    python -m src.hello                  # uses a synthetic GeoTIFF
    python -m src.hello path/to/image    # any GDAL-readable file (GeoTIFF, PDS3 .IMG, PDS4 .xml)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def make_synthetic_geotiff(path: Path, size: int = 256) -> Path:
    """Write a small random GeoTIFF so the check works before any data is downloaded."""
    rng = np.random.default_rng(42)
    arr = (rng.random((size, size)) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype=arr.dtype, transform=from_origin(0.0, 0.0, 1.0, 1.0),
    ) as dst:
        dst.write(arr, 1)
    return path


def describe(path: Path) -> None:
    """Print size, dtype, CRS, pixel size and basic stats of band 1."""
    with rasterio.open(path) as src:
        band = src.read(1, masked=True)
        print(f"file      : {path}")
        print(f"driver    : {src.driver}")
        print(f"size (WxH): {src.width} x {src.height}, bands={src.count}, dtype={src.dtypes[0]}")
        print(f"crs       : {src.crs}")
        print(f"pixel size: {src.res}")
        print(f"band1     : min={band.min()}, max={band.max()}, mean={band.mean():.2f}")


def check_gpu() -> None:
    """Report whether PyTorch can see the CUDA GPU."""
    try:
        import torch
    except ImportError:
        print("torch     : not installed")
        return
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"torch     : {torch.__version__}, GPU={props.name}, VRAM={props.total_memory / 1e9:.1f} GB")
    else:
        print(f"torch     : {torch.__version__}, CUDA NOT available (CPU only)")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else make_synthetic_geotiff(Path("results/_hello.tif"))
    describe(path)
    check_gpu()
    import cv2
    print(f"opencv    : {cv2.__version__} (SIFT available: {hasattr(cv2, 'SIFT_create')})")
    print("Phase 1 check passed.")


if __name__ == "__main__":
    main()
