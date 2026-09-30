# Data download checklist (SIH26166)

Keep every download as-is (zips stay zipped). Tick each box when the file is in its folder.

## Pair 1: Chandrayaan-3 landing site (69.37°S, 32.32°E), MUST HAVE
Folder: `data/raw/pair01_ch3site/`

From chmapbrowse.issdc.gov.in (OHRC, Calibrated; box: lat -69.45 to -69.30, lon 32.10 to 32.55)
- [ ] `ch2_ohr_ncp_20240425T1209509264_d_img_d18`: main source image
- [ ] `ch2_ohr_ncp_20231003T1756206908_d_img_d18`: same place, different sun
- [ ] (optional) newest `ch2_ohr` entry on the last results page, if dated after 2024-04-25
- [ ] (optional) `ch2_ohr_ncp_20230823T1450475804_d_img_n18`: landing day, for change detection

From the LROC Data Portal (search `NAC_DTM_VIKRAMSITE1`)
- [ ] Orthophoto M1442997156, 1.00 m/px
- [ ] Orthophoto M1443025251, 1.00 m/px
- [ ] Digital Terrain Model, 32-bit GeoTIFF (3 m)
- [ ] README

## Pair 3: TMC-2 at the same site, MUST HAVE
Folder: `data/raw/pair03_tmc2/`

From chmapbrowse (TMC 2, Calibrated, same box)
- [ ] `ch2_tmc_ncn_20230130T1900132182_d_img_d32`: nadir camera (check the preview isn't mostly black)

From the LROC Data Portal (same `NAC_DTM_VIKRAMSITE1` page)
- [ ] Orthophoto M1442997156, 3.00 m/px: reference closer to TMC-2's ~5 m scale

(Optional) chmapbrowse, same box, TMC 2 with **Derived Ortho** and **Derived DTM** ticked
- [ ] one TMC-2 Derived Ortho, if listed
- [ ] one TMC-2 Derived DTM, if listed

## Pair 2: Apollo 11 site (0.67°N, 23.47°E), NICE TO HAVE, do last
Folder: `data/raw/pair02_apollo11/`

From chmapbrowse (OHRC, Calibrated; box: lat 0.60 to 0.75, lon 23.40 to 23.55, both positive)
- [ ] one `ch2_ohr_...` strip (preview shows surface detail)

From the LROC Data Portal (search `NAC_DTM_APOLLO11`)
- [ ] Orthophoto M150361817, 0.50 m/px
- [ ] Digital Terrain Model, 32-bit GeoTIFF
- [ ] README

## Don't download
- Anything starting with `ch1_` (Chandrayaan-1, 2009)
- `ncf` / `nca` TMC files (fore/aft views)
- Raw products, IIRS, SAR, TMC 1, HYSI, SELENE (for now)
- LROC posters / global mosaics from the "Curated Downloads" page
