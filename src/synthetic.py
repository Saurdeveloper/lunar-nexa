"""Synthetic lunar test bench with exact ground truth.

Builds a crater-covered DEM, renders it under any sun azimuth/elevation with a
Lunar-Lambert reflectance model and cast shadows, and produces a
(source, reference) pair related by a KNOWN transform. Because the truth is
known, registration error can be measured exactly, and accuracy can be swept
against sun-angle difference and scale ratio.

Conventions:
    DEM heights in metres on a "fine" grid with pixel size `dx` metres.
    Image x = column (east, right), y = row (south, down).
    Sun azimuth in degrees clockwise from north (north = up = -y).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import Transform


# ----------------------------------------------------------------------------- DEM
def fractal_surface(n: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian random surface with power spectrum ~ f^-beta, zero mean, unit std."""
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.rfftfreq(n)[None, :]
    f = np.sqrt(fx**2 + fy**2)
    f[0, 0] = 1.0
    amp = f ** (-beta / 2.0)
    amp[0, 0] = 0.0
    phase = rng.uniform(0, 2 * np.pi, amp.shape)
    z = np.fft.irfft2(amp * np.exp(1j * phase), s=(n, n))
    return ((z - z.mean()) / z.std()).astype(np.float32)


def add_craters(dem: np.ndarray, dx: float, rng: np.random.Generator, n_craters: int,
                d_min_m: float, d_max_m: float, slope: float = 2.0) -> np.ndarray:
    """Add simple bowl craters with raised rims; diameters follow N(>D) ~ D^-slope.

    Profile (r = distance / radius): inside  h = depth*(r^2 - 1) + rim
                                     outside h = rim * r^-3   (ejecta blanket)
    with depth = 0.2 D and rim = 0.04 D (typical fresh simple craters).
    """
    n = dem.shape[0]
    u = rng.uniform(size=n_craters)
    a, b = d_min_m ** -slope, d_max_m ** -slope
    diam = (a - u * (a - b)) ** (-1.0 / slope)
    order = np.argsort(-diam)  # big first so small ones overprint them
    for D in diam[order]:
        R = D / 2.0 / dx
        cx, cy = rng.uniform(0, n, 2)
        ext = int(np.ceil(3 * R)) + 2
        x0, x1 = max(0, int(cx) - ext), min(n, int(cx) + ext)
        y0, y1 = max(0, int(cy) - ext), min(n, int(cy) + ext)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        r = np.hypot(xx - cx, yy - cy) / R
        depth, rim = 0.2 * D, 0.04 * D * rng.uniform(0.6, 1.2)
        prof = np.where(r < 1, depth * (r**2 - 1) + rim, rim * np.maximum(r, 1) ** -3)
        prof[r > 3] = 0
        # degrade older craters a bit: blend toward existing surface
        fresh = rng.uniform(0.5, 1.0)
        dem[y0:y1, x0:x1] += fresh * prof.astype(np.float32)
    return dem


def make_dem(n: int = 3072, dx: float = 1.0, seed: int = 0) -> np.ndarray:
    """Crater-covered terrain in metres, sized n x n at dx m/px."""
    rng = np.random.default_rng(seed)
    dem = fractal_surface(n, beta=3.6, rng=rng) * (0.006 * n * dx)      # long-wavelength relief
    dem += fractal_surface(n, beta=2.8, rng=rng) * (0.08 * dx)          # regolith roughness
    area_km2 = (n * dx / 1000.0) ** 2
    dem = add_craters(dem, dx, rng, n_craters=int(2500 * area_km2 / 9.4 + 200),
                      d_min_m=6 * dx, d_max_m=0.35 * n * dx)
    return dem


# ----------------------------------------------------------------------- rendering
def sun_vector(az_deg: float, el_deg: float) -> np.ndarray:
    """Unit vector pointing TO the sun in (x=east, y=south, z=up)."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(el) * np.sin(az), -np.cos(el) * np.cos(az), np.sin(el)])


def cast_shadows(dem: np.ndarray, dx: float, az_deg: float, el_deg: float) -> np.ndarray:
    """Boolean mask of cast shadows, via rotate -> running max -> rotate back (O(N))."""
    s = sun_vector(az_deg, el_deg)
    theta = np.arctan2(s[1], s[0])            # direction toward the sun in image coords
    phi = np.pi - theta                        # rotate so the sun sits at -x (left)
    c, si = np.cos(phi), np.sin(phi)
    h, w = dem.shape
    diag = int(np.ceil(np.hypot(h, w))) + 2
    ctr_old = np.array([(w - 1) / 2, (h - 1) / 2])
    ctr_new = np.array([(diag - 1) / 2, (diag - 1) / 2])
    Rm = np.array([[c, -si], [si, c]])
    M = np.hstack([Rm, (ctr_new - Rm @ ctr_old)[:, None]])
    low = float(dem.min()) - 1e4
    rot = cv2.warpAffine(dem, M, (diag, diag), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=low)
    tan_el = np.tan(np.radians(el_deg))
    a = rot + np.arange(diag, dtype=np.float32)[None, :] * (dx * tan_el)
    run = np.maximum.accumulate(a, axis=1)
    shadow_rot = np.zeros_like(a, dtype=np.float32)
    shadow_rot[:, 1:] = (run[:, :-1] > a[:, 1:] + 1e-3).astype(np.float32)
    back = cv2.warpAffine(shadow_rot, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    return back > 0.5


def render(dem: np.ndarray, dx: float, az_deg: float, el_deg: float,
           albedo: np.ndarray | None = None, L: float = 0.6, shadows: bool = True,
           ambient: float = 0.01) -> np.ndarray:
    """Lunar-Lambert radiance for nadir viewing: A*[2L cos i/(cos i + cos e) + (1-L) cos i]."""
    gy, gx = np.gradient(dem.astype(np.float64), dx)
    nrm = np.sqrt(gx**2 + gy**2 + 1.0)
    nx, ny, nz = -gx / nrm, -gy / nrm, 1.0 / nrm
    s = sun_vector(az_deg, el_deg)
    cos_i = np.clip(nx * s[0] + ny * s[1] + nz * s[2], 0, None)
    cos_e = np.clip(nz, 1e-3, None)
    img = 2 * L * cos_i / (cos_i + cos_e + 1e-9) + (1 - L) * cos_i
    if shadows:
        img[cast_shadows(dem, dx, az_deg, el_deg)] = 0.0
    if albedo is not None:
        img = img * albedo
    return (img + ambient).astype(np.float32)


# ------------------------------------------------------------------------- pairing
@dataclass
class PairSpec:
    """Parameters of one synthetic source/reference pair."""
    ref_sun: tuple[float, float] = (90.0, 30.0)   # (azimuth, elevation) deg
    src_sun: tuple[float, float] = (90.0, 30.0)
    ref_px: float = 4.0        # reference pixel size in fine-grid pixels
    src_px: float = 2.0        # source pixel size in fine-grid pixels
    rotation_deg: float = 0.0  # source rotation relative to reference
    perspective: float = 0.0   # small projective term (1e-5 .. 1e-4 is realistic)
    src_size: int = 768
    offset: tuple[float, float] = (0.0, 0.0)  # source centre offset in fine px
    noise: float = 0.01
    src_gamma: float = 1.0     # sensor response difference
    seed: int = 0


def _gauss(img: np.ndarray, sigma: float) -> np.ndarray:
    return img if sigma < 0.3 else cv2.GaussianBlur(img, (0, 0), sigma)


def make_pair(dem: np.ndarray, dx: float, spec: PairSpec,
              albedo: np.ndarray | None = None,
              renders: dict | None = None) -> tuple[np.ndarray, np.ndarray, Transform, dict]:
    """Create (source, reference, T_true source->reference, info).

    `renders` is an optional cache {(az, el): fine_render} to avoid re-rendering.
    """
    rng = np.random.default_rng(spec.seed)
    renders = {} if renders is None else renders

    def get(sun):
        if sun not in renders:
            renders[sun] = render(dem, dx, *sun, albedo=albedo)
        return renders[sun]

    n = dem.shape[0]
    # reference: area-average the fine render by ref_px
    r = int(round(spec.ref_px))
    fine_ref = get(spec.ref_sun)
    m = (n // r) * r
    ref = fine_ref[:m, :m].reshape(m // r, r, m // r, r).mean(axis=(1, 3))
    # A_ref: ref pixel q -> fine coords  = r*q + (r-1)/2
    A_ref = np.array([[r, 0, (r - 1) / 2], [0, r, (r - 1) / 2], [0, 0, 1.0]])
    # G: source pixel p -> fine coords (scale, rotation, perspective, offset)
    c = (spec.src_size - 1) / 2
    th = np.radians(spec.rotation_deg)
    S = spec.src_px
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) * S
    ctr = np.array([n / 2 + spec.offset[0], n / 2 + spec.offset[1]])
    G = np.eye(3)
    G[:2, :2] = Rm
    G[:2, 2] = ctr - Rm @ np.array([c, c])
    P = np.eye(3)
    P[2, 0] = spec.perspective
    P[2, 1] = -0.5 * spec.perspective
    Tc = np.array([[1, 0, -c], [0, 1, -c], [0, 0, 1]])
    G = G @ np.linalg.inv(Tc) @ P @ Tc          # perspective about the source centre
    ys, xs = np.mgrid[0:spec.src_size, 0:spec.src_size].astype(np.float64)
    hp = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    f = G @ hp
    fx = (f[0] / f[2]).reshape(xs.shape).astype(np.float32)
    fy = (f[1] / f[2]).reshape(xs.shape).astype(np.float32)
    fine_src = _gauss(get(spec.src_sun), 0.5 * S)  # anti-alias to the source resolution
    src = cv2.remap(fine_src, fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # sensor differences and noise
    src = np.clip(src / max(src.max(), 1e-6), 0, 1) ** spec.src_gamma
    ref = np.clip(ref / max(ref.max(), 1e-6), 0, 1)
    src = np.clip(src + rng.normal(0, spec.noise, src.shape), 0, 1).astype(np.float32)
    ref = np.clip(ref + rng.normal(0, spec.noise, ref.shape), 0, 1).astype(np.float32)
    T_true = Transform("homography", np.linalg.inv(A_ref) @ G)
    T_true.params = T_true.params / T_true.params[2, 2]
    inside = (fx.min() >= 0 and fy.min() >= 0 and fx.max() < m and fy.max() < m)
    info = {"scale_ratio": spec.ref_px / spec.src_px, "inside": bool(inside)}
    return src, ref.astype(np.float32), T_true, info
