# PROGRESS: SIH26166 lunar image correspondence

**Last updated:** Wed 30 Sep 2026
**Owner of all tasks:** Aranyak Dutta (Claude does the build work)
**Compute:** RTX 4050 laptop GPU, 6 GB VRAM
**SIH stage:** Idea submission. Deadline flexible (portal date not yet confirmed).
**Repo:** `~/Projects/lunar-nexa`

## Where we are

- **Phase 1 (Setup):** code and configs done. Exit check still pending on the laptop: rebuild `.venv` on Python 3.11 per README, run `python -m src.hello`.
- **Phase 2 (Data):** blocked. The session's network allowlist blocks ISSDC, LROC, JAXA, PyPI and GitHub. Waiting for the official dataset, either as files in `data/raw/` or a link plus allowlist change.
- **Phase 4 (Baseline):** built ahead of data. The full pipeline and CLI run on synthetic lunar pairs with exact ground truth. Phase 4 exit (real pairs) is still pending.

## Plan

| Dates (2026) | Phase | Tasks | Owner | Exit check |
| --- | --- | --- | --- | --- |
| 30 Sep – 2 Oct | 1. Setup | Python 3.11 env, CUDA torch, rasterio; first git commit | Aranyak Dutta | `src.hello` passes on the laptop |
| on data arrival | 2. Data | 1–3 pairs (one large sun-angle difference, one polar, one TMC-2); parse labels with `src.labels`; footprints; fill manifest | Aranyak Dutta | Manifest complete with sun-angle diffs and scale ratios |
| +2 days | 3. Coarse alignment | Map-project to a common CRS (ISIS `cam2map` or GDAL), crop overlap | Aranyak Dutta | Pair overlaps within tens of px |
| +3 days | 4. Baseline on real pairs | `src.register` on each pair; SIFT vs LightGlue (GPU) | Aranyak Dutta | GeoTIFF + matches CSV + metrics JSON per pair |
| ~18 Oct | Idea deck | Problem, architecture, novelties, synthetic + real results | Aranyak Dutta | Deck reviewed twice |
| after submission | 5–8 | Novelties with ablations, packaging, demo, finale deck | Aranyak Dutta | See project instructions |

## Done on 30 Sep

- Repo skeleton, pinned requirements, README with setup and run commands.
- `src/pipeline.py`: metadata scale normalisation → percentile stretch + CLAHE + shadow mask → SIFT/RootSIFT/AKAZE/ORB/SuperPoint+LightGlue → MAGSAC++ → 8×8 grid distribution filter → sub-pixel refinement (windowed cross-correlation, upsampled ×20) → final fit on held-out split → warp.
- `src/register.py` CLI writes `registered.tif` (GeoTIFF when rasterio + georeferenced input), `matches.csv`, `metrics.json`, `transform.json`, `quicklook.png`, `flicker.gif`.
- `src/synthetic.py`: crater DEM (power-law size distribution), Lunar-Lambert rendering with cast shadows, known-homography source/reference pairs.
- `src/bench_synthetic.py`: sweeps sun azimuth difference, sun elevation and scale ratio.
- `src/labels.py`: extracts sun angles, resolution and corners from PDS4/PDS3 labels.
- 4 unit tests pass (transform fitting, outlier rejection, known sub-pixel shift, checkpoint split).
- Quality gate: a result is flagged if it has fewer than 20 inliers or checkpoint RMSE above 1 px. On the benchmark it caught the one silent failure; it also flagged 14 correct but weak runs.

## Results: synthetic baseline (69 pairs per method, exact ground truth)

Success = true transform error (GT RMSE) below 1 px. The reference is lit from azimuth 90°, elevation 30°. Scale ratio is 2 unless swept. Rotation is 10–24° with a small perspective term.

| Condition | SIFT | RootSIFT | AKAZE |
| --- | --- | --- | --- |
| Sun azimuth diff 0–30° | 100% (GT RMSE ≤ 0.08 px) | 100% | 100% |
| Sun azimuth diff ≥ 45° | **0%** | **0%** | **0%** |
| Sun elevation 5–60° (same azimuth) | 100% (0.17 px at 5°) | 100% | 100% |
| Scale ×8 with metadata scale | 100% (0.06 px) | 100% | 33% |
| Scale ×8 without metadata scale | 67% | 100% | 0% |
| Median checkpoint RMSE (successful runs) | 0.049 px | 0.049 px | 0.049 px |
| Median grid coverage (8×8) | 99% | 100% | 97% |
| Median runtime (CPU, 768 px source) | 1.5 s | 1.5 s | 0.9 s |

**Takeaway:** sub-pixel accuracy and uniform coverage are solved once matching works. Metadata-based scale normalisation handles scale. **The baseline breaks when the sun azimuth changes by 45° or more.** That is exactly ISRO's illumination challenge, and it is where the novelties should go.

Figures: `docs/figures/success_vs_sun_and_scale.png`, `docs/figures/demo_quicklook.png`. Full table: `results/synth_baseline/summary.md`.

## Decisions

| Date | Decision | Why |
| --- | --- | --- |
| 30 Sep | Python 3.11 via `uv`, not the 3.14 venv | 3.14 wheel support for rasterio/torch/kornia is patchy |
| 30 Sep | Aranyak Dutta owns every task | Requested by team |
| 30 Sep | Build and benchmark on synthetic pairs before real data | Data download blocked; synthetic truth also gives exact error, needed for the sun/scale plots anyway |
| 30 Sep | Sub-pixel refinement uses windowed cross-correlation, not phase correlation | Unit test: phase correlation under-estimated a known 0.37 px shift by ~40%; cross-correlation error ≤ 0.05 px |
| 30 Sep | Source points stay fixed during refinement; only reference points move | Keeps the source-side distribution intact and works for any transform model |
| 30 Sep | Checkpoints are 25% of inliers, stratified by 8×8 cell, never used to fit | Honest accuracy; the fit RMSE is reported beside it |

## Open questions

1. Official dataset: files or link from Aranyak.
2. Network allowlist: allow PyPI, PyTorch, GitHub, ISSDC, LROC, JAXA for this project, or keep providing files manually.
3. Exact idea-submission deadline on the SIH portal.
4. Which illumination novelty to build first (see next tasks).

## Next three tasks

1. **Aranyak Dutta:** put 1–3 official pairs (images + labels) in `data/raw/pairNN/`, or share the link.
2. **Aranyak Dutta:** rebuild `.venv` on Python 3.11 and run `python -m src.hello` and `python -m pytest -q tests`.
3. **Claude:** once data lands, run Phases 2–4 on the real pairs, add the SuperPoint + LightGlue comparison on the GPU, then prototype the chosen illumination fix, measured on the same sweep.
