"""Read the fields the pipeline needs from PDS4 (.xml) and PDS3 (.LBL / attached .IMG) labels.

Field names differ between missions and product versions, so this searches by
keyword rather than by exact path, and returns everything it recognises. Always
eyeball the output for a new product type before trusting it.

Caution: azimuth conventions differ. LRO NAC's SUB_SOLAR_AZIMUTH is measured in the
image frame (clockwise from the 3 o'clock direction), not from north. Convert
both products to "degrees clockwise from north in the map-projected image"
before computing sun-angle differences.

Usage:
    python -m src.labels data/raw/pair01/*.xml data/raw/pair01/*.IMG
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# canonical name -> keyword patterns (case-insensitive, matched against tag / keyword names)
FIELDS = {
    "sun_azimuth_deg": [r"sun_?azimuth", r"solar_?azimuth", r"sub_solar_azimuth"],
    "sun_elevation_deg": [r"sun_?elevation", r"solar_?elevation"],
    "incidence_deg": [r"incidence_?angle"],
    "emission_deg": [r"emission_?angle"],
    "phase_deg": [r"phase_?angle"],
    "pixel_res_m": [r"pixel_?resolution", r"map_?resolution", r"ground_?sampling", r"resolution"],
    "altitude_km": [r"spacecraft_?altitude", r"altitude"],
    "start_time": [r"start_?date_?time", r"start_?time"],
    "lines": [r"^lines$", r"image_?lines"],
    "samples": [r"^line_?samples$", r"^samples$"],
}
CORNER = re.compile(r"(upper|lower)_?(left|right).*(lat|lon)", re.I)


def _match(name: str) -> str | None:
    for canon, pats in FIELDS.items():
        if any(re.search(p, name, re.I) for p in pats):
            return canon
    return None


def _num(v: str):
    v = v.strip().strip('"').split("<")[0].strip()
    try:
        return float(v.split()[0])
    except (ValueError, IndexError):
        return v


def parse_pds4(path: Path) -> dict:
    """Collect recognised fields from a PDS4 XML label (namespace-agnostic)."""
    from lxml import etree
    root = etree.parse(str(path)).getroot()
    out: dict = {"label_format": "PDS4", "raw": {}}
    for el in root.iter():
        if not isinstance(el.tag, str) or el.text is None or not el.text.strip():
            continue
        tag = etree.QName(el).localname
        canon = _match(tag)
        if canon and canon not in out:
            out[canon] = _num(el.text)
            out["raw"][canon] = tag
        if CORNER.search(tag):
            out.setdefault("corners", {})[tag] = _num(el.text)
        if tag.lower() in ("file_name",) and "file_name" not in out:
            out["file_name"] = el.text.strip()
    return out


def parse_pds3(path: Path) -> dict:
    """Collect recognised KEYWORD = value pairs from a PDS3 label (detached or attached)."""
    text = path.read_bytes()[:200_000].decode("latin-1", errors="ignore")
    text = text.split("\nEND\r")[0].split("\nEND\n")[0]
    out: dict = {"label_format": "PDS3", "raw": {}}
    for m in re.finditer(r"^\s*([A-Z0-9_:^]+)\s*=\s*(.+?)\s*$", text, re.M):
        key, val = m.group(1), m.group(2)
        canon = _match(key)
        if canon and canon not in out:
            out[canon] = _num(val)
            out["raw"][canon] = key
        if CORNER.search(key):
            out.setdefault("corners", {})[key] = _num(val)
        if key in ("MAP_PROJECTION_TYPE", "TARGET_NAME", "PRODUCT_ID", "INSTRUMENT_ID"):
            out[key.lower()] = str(_num(val)).strip('"')
    # incidence -> elevation when elevation is not given directly
    return out


def parse_label(path: str | Path) -> dict:
    """Dispatch on file type; adds sun elevation from incidence when missing."""
    path = Path(path)
    d = parse_pds4(path) if path.suffix.lower() == ".xml" else parse_pds3(path)
    if "sun_elevation_deg" not in d and isinstance(d.get("incidence_deg"), float):
        d["sun_elevation_deg"] = 90.0 - d["incidence_deg"]
        d["raw"]["sun_elevation_deg"] = "90 - incidence"
    d["path"] = str(path)
    return d


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(parse_label(p), indent=2, default=str))
