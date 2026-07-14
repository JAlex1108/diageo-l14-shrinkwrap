#!/usr/bin/env python3
"""Extract pair composites by the back (leftmost) strip's Phase-A filtered tilt.

Phase naming follows capture time: Phase A is the EARLIER frame (classify_phase1),
Phase B the LATER one (measurements.json, phase-3 measure view). This extraction
uses Phase A.

The Phase-A filtered dataset is the one plotted in
``hist_phaseA_classify_phase1_filtered.png``: angles measured on the
classify_phase1 frames (``measurements_phase1.json``), kept only when a frame
passes the geometry filter (exactly 3 strips, centres within Y_TOL px, longest
strip <= SIZE_RATIO x shortest), and joined to a hand class (1/2/3) via
``classify_phase1.csv``.

"Back (leftmost) strip" is index 0 of ``angles_deg`` (spatial left->right is
[back, middle, front]).

For every qualifying frame across all classes, this copies the pre-built pair
composite (classify overlay + measure tilt, labelled) from ``compare_by_class``
into two folders:
  * phaseA_earlier_back_leftmost_lt21deg/   angle < 21 deg
  * phaseA_earlier_back_leftmost_gt24deg/   angle > 24 deg

Output filenames are prefixed with the angle so a plain sort is angle-ordered.

Usage:
    python extract_backstrip_pairs.py            # copy the composites
    python extract_backstrip_pairs.py --dry      # report counts only, copy nothing
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent / "output"
PHASE1_JSON = BASE / "measurements_phase1.json"
CLASSIFY_CSV = BASE / "classify_phase1.csv"
COMPARE_BY_CLASS = BASE / "compare_by_class"

OUT_LT21 = BASE / "phaseA_earlier_back_leftmost_lt21deg"
OUT_GT24 = BASE / "phaseA_earlier_back_leftmost_gt24deg"

# Same geometry filter as measure_phase1_tilt.is_valid (the _filtered histogram).
Y_TOL = 10.0
SIZE_RATIO = 1.5

# Back = leftmost = first entry of the spatial left->right angles_deg list.
BACK_IDX = 0
LOW_THRESH = 21.0   # angle < 21 deg  -> lt21 folder
HIGH_THRESH = 24.0  # angle > 24 deg  -> gt24 folder

CLASS_NAMES = {1: "Class 1", 2: "Class 2", 3: "Class 3"}
ID_RE = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_C\d+)")


def is_valid(record: dict) -> bool:
    """True if the frame passes the same geometry gate as the filtered histogram."""
    if record.get("n_strips") != 3:
        return False
    ys = record.get("centers_y", [])
    lengths = record.get("lengths", [])
    if len(ys) != 3 or len(lengths) != 3:
        return False
    if max(ys) - min(ys) > Y_TOL:
        return False
    if min(lengths) <= 0 or max(lengths) / min(lengths) > SIZE_RATIO:
        return False
    return True


def load_labels() -> dict[str, int]:
    """Map classify_phase1 image basename -> hand class label."""
    labels: dict[str, int] = {}
    with open(CLASSIFY_CSV, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                labels[row["video_name"].strip()] = int(row["label"])
            except (ValueError, KeyError):
                continue
    return labels


def index_composites() -> dict[str, Path]:
    """Map id (clip_Cxxx) -> pair composite path under compare_by_class (first wins)."""
    out: dict[str, Path] = {}
    for p in COMPARE_BY_CLASS.rglob("*.png"):
        m = ID_RE.search(p.name)
        if m and m.group(1) not in out:
            out[m.group(1)] = p
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry", action="store_true", help="report counts only, copy nothing")
    args = ap.parse_args()

    records = json.loads(PHASE1_JSON.read_text())
    labels = load_labels()
    comp_idx = index_composites()

    buckets = {"lt21": [], "gt24": []}
    n_valid = n_classed = 0
    misses: list[str] = []

    for record in records:
        if not is_valid(record):
            continue
        n_valid += 1
        basename = record["image"]
        cls = labels.get(basename)
        if cls not in CLASS_NAMES:
            continue
        n_classed += 1

        back_angle = float(record["angles_deg"][BACK_IDX])
        if back_angle < LOW_THRESH:
            bucket = "lt21"
        elif back_angle > HIGH_THRESH:
            bucket = "gt24"
        else:
            continue

        m = ID_RE.search(basename)
        if not m:
            misses.append(f"{basename} (no id in name)")
            continue
        cid = m.group(1)
        comp = comp_idx.get(cid)
        if comp is None:
            misses.append(f"{cid} (no pair composite)")
            continue
        buckets[bucket].append((back_angle, cls, cid, comp))

    print(f"Phase-A filtered (valid geometry)  : {n_valid}")
    print(f"  of which joined to a class (1/2/3): {n_classed}")
    print(f"  back strip < {LOW_THRESH:.0f} deg : {len(buckets['lt21'])}")
    print(f"  back strip > {HIGH_THRESH:.0f} deg : {len(buckets['gt24'])}")
    if misses:
        print(f"  no composite for {len(misses)} qualifying frame(s); e.g. {misses[:5]}")

    if args.dry:
        print("\n(dry run - nothing copied)")
        return

    for bucket, out_dir in (("lt21", OUT_LT21), ("gt24", OUT_GT24)):
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for back_angle, cls, cid, comp in sorted(buckets[bucket]):
            dest = out_dir / f"{back_angle:05.2f}deg_class{cls}_{cid}.png"
            shutil.copy2(comp, dest)
        print(f"  {len(buckets[bucket]):4d} pair(s) -> {out_dir}")


if __name__ == "__main__":
    main()
