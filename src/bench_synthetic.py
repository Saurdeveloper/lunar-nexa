"""Synthetic benchmark: accuracy vs sun-angle difference and scale ratio, with exact truth.

Usage:
    python -m src.bench_synthetic --config configs/baseline.yaml --out results/synth_baseline
    python -m src.bench_synthetic --methods sift akaze --quick

Writes runs.csv (one row per pair x method), summary.md, and plots.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from .metrics import ground_truth_error  # noqa: E402
from .pipeline import register  # noqa: E402
from .synthetic import PairSpec, make_dem, make_pair  # noqa: E402
from .viz import quicklook  # noqa: E402

REF_SUN = (90.0, 30.0)


def build_cases(quick: bool) -> list[dict]:
    """List of sweep cases. Each case: sweep name, x value, PairSpec kwargs, use_hint."""
    offs = [(0, 0), (-500, 400), (450, -350)] if not quick else [(0, 0)]
    cases = []
    for k, off in enumerate(offs):
        base = dict(ref_sun=REF_SUN, ref_px=4.0, src_px=2.0, rotation_deg=10.0 + 7 * k,
                    perspective=2e-5, offset=off, src_gamma=0.8, seed=k)
        for daz in ([0, 15, 30, 45, 60, 90, 120, 180] if not quick else [0, 30, 90]):
            cases.append(dict(sweep="sun_azimuth", x=daz, hint=True,
                              spec={**base, "src_sun": (REF_SUN[0] + daz, REF_SUN[1])}))
        for el in ([5, 10, 15, 20, 30, 45, 60] if not quick else [10, 30]):
            cases.append(dict(sweep="sun_elevation", x=el, hint=True,
                              spec={**base, "src_sun": (REF_SUN[0], float(el))}))
        for ratio in ([1, 2, 4, 8] if not quick else [2, 4]):
            for hint in (True, False):
                cases.append(dict(sweep="scale_hint" if hint else "scale_nohint", x=ratio, hint=hint,
                                  spec={**base, "src_sun": REF_SUN, "src_px": 4.0 / ratio}))
    return cases


def run(cfg: dict, methods: list[str], out: Path, quick: bool) -> pd.DataFrame:
    """Run all cases for all methods and return the results table."""
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dem = make_dem(3072, 1.0, seed=cfg.get("seed", 42))
    renders: dict = {}
    rows = []
    cases = build_cases(quick)
    for i, case in enumerate(cases):
        spec = PairSpec(**case["spec"])
        src, ref, T_true, info = make_pair(dem, 1.0, spec, renders=renders)
        daz = (spec.src_sun[0] - spec.ref_sun[0] + 180) % 360 - 180
        for method in methods:
            c = copy.deepcopy(cfg)
            c.setdefault("matching", {})["method"] = method
            r = register(src, ref, c, scale_hint=spec.src_px / spec.ref_px if case["hint"] else None)
            row = {"sweep": case["sweep"], "x": case["x"], "method": method, "seed": spec.seed,
                   "sun_az_diff": abs(daz), "sun_el_src": spec.src_sun[1],
                   "scale_ratio": info["scale_ratio"], "ok": r["ok"],
                   "n_tentative": r["n_tentative"]}
            if r["ok"]:
                row.update(r["metrics"])
                row.update(ground_truth_error(r["transform"], T_true, src.shape))
                row["runtime_s"] = r["timing"]["total_s"]
                if i == 0 or (case["sweep"] == "sun_azimuth" and case["x"] in (30, 45) and spec.seed == 0):
                    quicklook(src, ref, r["warped"], r, out / f"quicklook_{method}_{case['sweep']}_{case['x']}.png",
                              f"{method} | sun Δaz={abs(daz):.0f}° | scale x{info['scale_ratio']:.0f}")
            rows.append(row)
        print(f"[{i + 1}/{len(cases)}] {case['sweep']}={case['x']} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df["success"] = df["ok"] & (df.get("gt_rmse_px", np.nan) < 1.0)
    df.to_csv(out / "runs.csv", index=False)
    return df


def summarise(df: pd.DataFrame, out: Path) -> str:
    """Write summary.md and plots; return the markdown."""
    lines = ["# Synthetic benchmark summary", ""]
    g = df.groupby(["method"]).agg(
        pairs=("ok", "size"), success_rate=("success", "mean"),
        median_gt_rmse_px=("gt_rmse_px", "median"), median_cp_rmse_px=("checkpoint_rmse_px", "median"),
        median_inliers=("n_inliers", "median"), median_coverage=("distribution_score", "median"),
        median_runtime_s=("runtime_s", "median")).round(3)
    lines += ["## Overall (all sweeps)", "", g.to_markdown(), ""]
    for sweep in df["sweep"].unique():
        d = df[df["sweep"] == sweep]
        p = d.pivot_table(index="x", columns="method", values="success", aggfunc="mean").round(2)
        e = d[d["ok"]].pivot_table(index="x", columns="method", values="gt_rmse_px", aggfunc="median").round(3)
        lines += [f"## {sweep}: success rate (GT RMSE < 1 px)", "", p.to_markdown(), "",
                  f"## {sweep}: median GT RMSE (px, successful runs)", "", e.to_markdown(), ""]
    md = "\n".join(lines)
    (out / "summary.md").write_text(md)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, sweep, xl in zip(axs, ["sun_azimuth", "sun_elevation", "scale_hint"],
                             ["Sun azimuth difference (deg)", "Source sun elevation (deg); ref = 30",
                              "Scale ratio (ref px / src px)"]):
        d = df[df["sweep"] == sweep]
        for method, dm in d.groupby("method"):
            s = dm.groupby("x")["success"].mean()
            ax.plot(s.index, s.values * 100, marker="o", label=method)
        if sweep == "scale_hint":
            d2 = df[df["sweep"] == "scale_nohint"]
            for method, dm in d2.groupby("method"):
                s = dm.groupby("x")["success"].mean()
                ax.plot(s.index, s.values * 100, marker="x", ls="--", label=f"{method} (no metadata scale)")
            ax.set_xscale("log", base=2)
        ax.set_xlabel(xl)
        ax.set_ylabel("Success rate (%)")
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Baseline robustness on synthetic lunar pairs (exact ground truth)")
    fig.tight_layout()
    fig.savefig(out / "success_vs_sun_and_scale.png", dpi=120)
    plt.close(fig)
    return md


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--out", default="results/synth_baseline")
    ap.add_argument("--methods", nargs="+", default=["sift", "rootsift", "akaze"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out = Path(a.out)
    df = run(cfg, a.methods, out, a.quick)
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg))
    print(summarise(df, out))
    (out / "meta.json").write_text(json.dumps({"methods": a.methods, "quick": a.quick,
                                               "n_rows": len(df)}, indent=2))


if __name__ == "__main__":
    main()
