"""Quick-look figures: match vectors, checkerboard overlay, flicker GIF."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .preprocess import to_unit  # noqa: E402


def checkerboard(a: np.ndarray, b: np.ndarray, tiles: int = 8) -> np.ndarray:
    """Alternate tiles of a and b (same shape); NaNs in b show a."""
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((yy * tiles // h) + (xx * tiles // w)) % 2 == 0
    out = np.where(mask, a, b)
    return np.where(np.isnan(out), a, out)


def quicklook(src: np.ndarray, ref: np.ndarray, warped: np.ndarray | None, res: dict,
              path: str | Path, title: str = "") -> None:
    """Three panels: source with matches, reference with residual vectors, checkerboard."""
    ps, pr, hold = res["pts_src"], res["pts_ref"], res["checkpoint"]
    resid = res["transform"].apply(ps) - pr
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.4))
    ax[0].imshow(to_unit(src), cmap="gray")
    ax[0].scatter(ps[~hold, 0], ps[~hold, 1], s=6, c="#2a9df4", label="fit points")
    ax[0].scatter(ps[hold, 0], ps[hold, 1], s=10, c="#f4a62a", marker="s", label="checkpoints")
    ax[0].set_title("Source + matches")
    ax[0].legend(loc="lower right", fontsize=8)
    # crop reference-side panels to the source footprint (+10 % margin)
    h, w = src.shape
    corners = res["transform"].apply(np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float))
    (x0, y0), (x1, y1) = corners.min(0), corners.max(0)
    mx, my = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
    x0, y0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
    x1, y1 = int(min(ref.shape[1], x1 + mx)), int(min(ref.shape[0], y1 + my))
    ref_u = to_unit(ref)
    ax[1].imshow(ref_u[y0:y1, x0:x1], cmap="gray")
    scale = 100.0
    ax[1].quiver(pr[:, 0] - x0, pr[:, 1] - y0, resid[:, 0] * scale, resid[:, 1] * scale,
                 color="#e8384f", angles="xy", scale_units="xy", scale=1, width=0.004)
    ax[1].set_title(f"Reference + residual vectors (x{scale:.0f})")
    if warped is not None:
        wa = to_unit(warped, valid=np.isfinite(warped))
        wa[~np.isfinite(warped)] = np.nan
        ax[2].imshow(checkerboard(ref_u, wa)[y0:y1, x0:x1], cmap="gray")
        ax[2].set_title("Checkerboard: reference / registered source")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    m = res["metrics"]
    fig.suptitle(f"{title}  |  inliers {m['n_inliers']}  checkpoint RMSE {m['checkpoint_rmse_px']:.3f} px"
                 f"  sub-px rate {m['subpixel_rate']:.0%}  coverage {m['distribution_score']:.0%}")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def flicker_gif(ref: np.ndarray, warped: np.ndarray, path: str | Path, size: int = 512) -> None:
    """Two-frame GIF alternating reference and registered source."""
    import cv2
    import imageio.v2 as imageio

    def u8(a):
        a = to_unit(a, valid=np.isfinite(a))
        a = np.nan_to_num(a)
        return cv2.resize((a * 255).astype(np.uint8), (size, size), interpolation=cv2.INTER_AREA)

    imageio.mimsave(path, [u8(ref), u8(warped)], duration=0.6, loop=0)
