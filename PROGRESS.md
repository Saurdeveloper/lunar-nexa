# PROGRESS: SIH26166 lunar image correspondence

**Last updated:** Wed 30 Sep 2026
**Owner of all tasks:** Aranyak Dutta (Claude does the build work)
**Compute:** RTX 4050 laptop GPU, 6 GB VRAM
**SIH stage:** Idea submission. Deadline flexible (portal date not yet confirmed).
**Repo:** `~/Projects/lunar-nexa`

## Where we are

- **Phase 1 (Setup): PASSED.** Python 3.11 venv (uv), numpy 1.26.4, one OpenCV (4.10 contrib headless), torch 2.5.1 + CUDA; `src.hello` and 4/4 tests pass; first commit "baseline pipe".
- **Phase 2 (Data): PASSED for pairs p01a, p01b, p03.** Downloaded by Aranyak from chmapbrowse + LROC Data Portal (see `docs/DATA_CHECKLIST.md`); zips tested; manifest filled. Missing: DTM `.TIF` files for both sites, Apollo 11 OHRC strip.
- **Phase 3 (Coarse alignment): PASSED for p01a** (metadata-only offset ≈ 23 m / 23 NAC px). `src/prep_pair.py` cuts the site from the Ch-2 zip and NAC .IMG and maps the Ch-2 image onto the NAC grid using the product's geometry grid (grid fit RMS 0.12 px). p01b and p03 coarse offsets not yet verified (phase correlation fails under their lighting).
- **Phase 4 (Baseline): PASSED on p01a, FAILS on p01b and p03.** See real results below.

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

## Results: real pairs (Chandrayaan-3 landing site, reference NAC M1442997156 ortho, 2023-07-03)

| Pair | Source | Sun az / el | Best matcher | Inliers | Checkpoint RMSE | Sub-px rate | Coverage | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p01a | OHRC 2024-04-25, 0.26 m | 304° / 10.8° | AKAZE + poly3 | 509 (53%) | **0.55 px (≈0.55 m)** | 94% | 98% | pass |
| p01a | same | same | SIFT + poly3 | 152 (39%) | 0.45 px | 97% | 80% | pass |
| p01b | OHRC 2023-10-03, 0.22 m | 277° / **2.2°** | all three | ≤6 | – | – | – | **fail** |
| p03 | TMC-2 2023-01-30, 4.72 m | **53°** / 17.2° | all three | ≤6 | – | – | – | **fail** |

On p01a, homography gives 0.74–0.85 px vs poly3 0.45–0.55 px: terrain and along-track (N–S) residual pattern, so a local model / orthorectification with the DTM is the next accuracy gain. Full table: `results/real_baseline/real_results.md`.

**Real data confirms the synthetic finding:** the baseline breaks under strong illumination change (2° sun; ~109° azimuth difference). Open check: whether p03 also has a larger metadata offset (TMC-2 geolocation) — needs an illumination-robust coarse step to tell.

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
| 30 Sep | Coarse alignment from each Ch-2 product's geometry grid (lat/lon every 100 px) mapped into NAC polar-stereographic pixels with a 3rd-order polynomial | Uses only product metadata, no GDAL/ISIS needed; 0.12 px fit, ~23 m absolute offset on p01a |
| 30 Sep | Final model poly3 for OHRC over this terrain | 0.45–0.55 px vs 0.74–0.85 px for homography on held-out points |
| 30 Sep | Checkpoints are 25% of inliers, stratified by 8×8 cell, never used to fit | Honest accuracy; the fit RMSE is reported beside it |

## Open questions

1. DTM files: `NAC_DTM_VIKRAMSITE1.TIF` (and `NAC_DTM_APOLLO11.TIF`) still to download; needed for the sun-angle fix and orthorectification.
2. NAC reference sun angles: not in the ortho label; get from the NAC EDR labels (M1442997156, M1443025251).
3. Exact idea-submission deadline on the SIH portal.
4. Which illumination fix to build first (A: DTM relighting + shadow filtering, recommended; B: learned dense matcher).

## Next three tasks

1. **Aranyak Dutta:** download `NAC_DTM_VIKRAMSITE1.TIF` into `data/raw/pair01_ch3site/`; decide A or B.
2. **Claude:** run SuperPoint + LightGlue on p01a/p01b/p03 on the RTX 4050 (needs a run on the laptop venv) as the learned baseline row.
3. **Claude:** build the chosen illumination fix and re-run p01b/p03 + the synthetic sun sweep; add rows to the ablation table.
