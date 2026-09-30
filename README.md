# lunar-nexa: SIH26166 lunar image correspondence

Registers Chandrayaan-2 optical images (OHRC, TMC-2, IIRS) to lunar reference images (LRO NAC/WAC, SELENE) with sub-pixel accuracy and uniformly distributed match points. Outputs a registered GeoTIFF, a matches CSV and a metrics JSON.

## Layout

| Path | Holds |
| --- | --- |
| `src/pipeline.py` | End-to-end registration: scale normalisation → CLAHE + shadow mask → matching → MAGSAC++ → grid distribution → sub-pixel refinement → final fit → warp |
| `src/register.py` | CLI for one source/reference pair |
| `src/matching.py` | SIFT, RootSIFT, AKAZE, ORB, SuperPoint + LightGlue |
| `src/geometry.py` | Similarity, affine, homography, 2nd/3rd-order polynomial; robust fitting |
| `src/subpixel.py` | Sub-pixel refinement by upsampled cross-correlation on locally resampled patches |
| `src/metrics.py` | Checkpoint RMSE (held-out), fit RMSE, inlier stats, sub-pixel rate, coverage |
| `src/synthetic.py` | Synthetic crater terrain, Lunar-Lambert rendering with cast shadows, known-transform pairs |
| `src/bench_synthetic.py` | Sweeps accuracy against sun-angle difference and scale ratio |
| `src/labels.py` | Pulls sun angles, resolution and corners from PDS4/PDS3 labels |
| `src/make_demo_pair.py` | Writes a synthetic pair to disk for demos |
| `configs/baseline.yaml` | Every pipeline parameter |
| `data/raw/`, `data/processed/` | Products (not in git); `data/manifest.csv` lists pairs |
| `results/` | One folder per run |
| `tests/test_core.py` | Fast sanity tests |

Point coordinates are always `(x, y)` pixels with x = column and y = row. A transform maps source pixels to reference pixels.

## Setup (Arch Linux, RTX 4050)

Use Python 3.11. The existing `.venv` is on Python 3.14, which several geospatial and PyTorch wheels don't fully support yet.

```bash
# once: install uv (pacman -S uv) — it fetches Python 3.11 for you
rm -rf .venv
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
uv pip install "lightglue @ git+https://github.com/cvg/LightGlue.git"
python -m src.hello          # prints GPU and "Phase 1 check passed"
python -m pytest -q tests    # 4 tests
```

## Run

```bash
# a synthetic pair with known truth
python -m src.make_demo_pair --out data/synthetic
python -m src.register --src data/synthetic/source.tif --ref data/synthetic/reference.tif \
    --src-res 1 --ref-res 2 --out results/demo

# a real pair (both map-projected to the same CRS; resolutions from the labels)
python -m src.register --src data/processed/pair01/ohrc.tif --ref data/processed/pair01/nac.tif \
    --config configs/baseline.yaml --out results/pair01

# robustness sweep vs sun angle and scale (about 10 min on CPU)
python -m src.bench_synthetic --methods sift rootsift akaze --out results/synth_baseline

# read a product label
python -m src.labels data/raw/pair01/*.xml
```

Each `register` run writes `registered.tif`, `matches.csv`, `metrics.json`, `transform.json`, `quicklook.png`, `flicker.gif` and `config_used.yaml`.
