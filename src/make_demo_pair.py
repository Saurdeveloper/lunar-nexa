"""Write a synthetic source/reference pair (plus its true transform) to disk.

Usage:
    python -m src.make_demo_pair --out data/synthetic --src-az 120 --src-el 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io_utils import write_image
from .synthetic import PairSpec, make_dem, make_pair


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--ref-az", type=float, default=90.0)
    ap.add_argument("--ref-el", type=float, default=30.0)
    ap.add_argument("--src-az", type=float, default=110.0)
    ap.add_argument("--src-el", type=float, default=25.0)
    ap.add_argument("--rotation", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out)
    dem = make_dem(3072, 1.0, seed=a.seed)
    spec = PairSpec(ref_sun=(a.ref_az, a.ref_el), src_sun=(a.src_az, a.src_el), ref_px=2.0,
                    src_px=1.0, rotation_deg=a.rotation, perspective=2e-5, src_size=1024,
                    src_gamma=0.8, seed=a.seed)
    src, ref, T_true, info = make_pair(dem, 1.0, spec)
    write_image(out / "source.tif", src)
    write_image(out / "reference.tif", ref)
    (out / "truth.json").write_text(json.dumps({"T_true_src_to_ref": T_true.to_dict(),
                                                "src_res_m": 1.0, "ref_res_m": 2.0,
                                                "src_sun": spec.src_sun, "ref_sun": spec.ref_sun,
                                                **info}, indent=2))
    print(f"wrote {out}/source.tif, reference.tif, truth.json")


if __name__ == "__main__":
    main()
