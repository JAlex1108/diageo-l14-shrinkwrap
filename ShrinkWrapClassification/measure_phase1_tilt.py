#!/usr/bin/env python3
"""Phase A: measure bottle tilt directly on the classify_phase1 frames.

Phase naming follows capture time: Phase A is the EARLIER frame (this classify_phase1
view), Phase B the LATER one (``measurements.json``, the phase-3 measure view).

Reuses the newly exported phase-1 pipeline (``measure_tilt_phase1.process_frame``,
which carries the phase-1 ``Bottles`` ROI + tuned HSV/contour config) to find the
3 label strips, fits a ``cv2.minAreaRect`` to each, and records its tilt angle in
the same spatial left->right order as Phase B. For each frame it writes a
validation overlay (rotated rects + angle labels) so detection can be eyeballed,
dumps the angles to ``measurements_phase1.json``, then renders the Phase-A
position/class histogram by reusing ``tilt_histograms.plot_position_histograms``.

Usage:
    python measure_phase1_tilt.py [--limit N] [--no-hist]
    python measure_phase1_tilt.py --from-json   # re-render histograms from the saved
                                                # measurements_phase1.json (no re-measure)
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

import tilt_histograms as th
from measure_tilt import _centroid_x, draw_debug_overlay, line_angle_deg
from measure_tilt_phase1 import process_frame

OUT = Path(__file__).parent / "output"
CLASSIFY_DIR = OUT / "classify_phase1"
OVERLAY_DIR = OUT / "classify_phase1_overlays"
PHASE1_JSON = OUT / "measurements_phase1.json"
PHASE_A_OUT = OUT / "hist_phaseA_classify_phase1.png"
PHASE_A_FILTERED_OUT = OUT / "hist_phaseA_classify_phase1_filtered.png"

# A frame's 3 strips are a valid measurement only if they sit at the same conveyor
# height (centres within Y_TOL px) and are roughly the same size (longest strip no
# more than SIZE_RATIO x the shortest). Rejects noise contours and partial strips.
Y_TOL = 10.0
SIZE_RATIO = 1.5


def is_valid(record: Dict, y_tol: float = Y_TOL, size_ratio=SIZE_RATIO) -> Tuple[bool, str]:
    """Return (valid, reason). Valid = exactly 3 strips, aligned in y, similar size.

    ``size_ratio=None`` disables the size gate (the phase-3 view has legitimately
    uneven strip lengths, so size is not a validity signal there).
    """
    if record["n_strips"] != 3:
        return False, "n_strips"
    ys = record["centers_y"]
    lengths = record["lengths"]
    if max(ys) - min(ys) > y_tol:
        return False, "y_spread"
    if size_ratio is not None and (min(lengths) <= 0 or max(lengths) / min(lengths) > size_ratio):
        return False, "size_ratio"
    return True, "ok"


def measure_image(image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
    """Detect label strips in one phase-1 frame and measure each strip's tilt.

    Returns ``(mask, measurements)`` with measurements ordered left->right, each
    carrying id, center, length, width, angle_deg and the rotated-rect box.
    """
    _, mask, contours = process_frame(image.copy())
    contours = sorted(contours, key=_centroid_x)

    measurements: List[Dict] = []
    for index, contour in enumerate(contours):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        (cx, cy), (w, h), _ = rect
        measurements.append({
            "id": index + 1,
            "center": (float(cx), float(cy)),
            "length": float(max(w, h)),
            "width": float(min(w, h)),
            "angle_deg": line_angle_deg(box),
            "box": box,
        })
    return mask, measurements


def measure_all(limit: int = 0, write_overlays: bool = True) -> List[Dict]:
    """Measure every classify_phase1 image, recording per-strip geometry.

    Stores angle, centre-y, length and width per strip (spatial left->right) so the
    geometry filter is reproducible from the JSON. ``write_overlays`` re-renders the
    validation overlays; skip it to re-measure quickly when overlays already exist.
    """
    if write_overlays:
        OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(CLASSIFY_DIR.glob("*.png"))
    if limit:
        images = images[:limit]

    records: List[Dict] = []
    n_three = 0
    for i, path in enumerate(images, 1):
        image = cv2.imread(str(path))
        if image is None:
            print(f"  skip (unreadable): {path.name}")
            continue

        _, measurements = measure_image(image)
        if write_overlays:
            overlay = draw_debug_overlay(image, measurements)
            cv2.imwrite(str(OVERLAY_DIR / f"{path.stem}_overlay.png"), overlay)

        if len(measurements) == 3:
            n_three += 1
        records.append({
            "image": path.name,
            "n_strips": len(measurements),
            # all arrays in spatial left->right order, same convention as Phase B
            "angles_deg": [round(m["angle_deg"], 2) for m in measurements],
            "centers_y": [round(m["center"][1], 1) for m in measurements],
            "lengths": [round(m["length"], 1) for m in measurements],
            "widths": [round(m["width"], 1) for m in measurements],
        })
        if i % 200 == 0:
            print(f"  {i}/{len(images)} processed ({n_three} clean 3-strip so far)")

    PHASE1_JSON.write_text(json.dumps(records, indent=2))
    print(f"\nMeasured {len(records)} images -> {n_three} with exactly 3 strips "
          f"({len(records) - n_three} off-count, excluded from histogram)")
    print(f"Angles -> {PHASE1_JSON}")
    print(f"Overlays -> {OVERLAY_DIR}")
    return records


def render_histogram(records: List[Dict], filtered: bool = False) -> None:
    """Join phase-B angles to hand labels and render the 3-axis figure.

    With ``filtered=True``, only frames passing :func:`is_valid` are plotted and the
    output gets a ``_filtered`` tag, leaving the unfiltered figure untouched.
    """
    if filtered:
        verdicts = [(r, *is_valid(r)) for r in records]
        kept = [r for r, ok, _ in verdicts if ok]
        reasons = [reason for _, ok, reason in verdicts if not ok]
        dropped = {key: reasons.count(key) for key in ("n_strips", "y_spread", "size_ratio")}
        print(f"Filter funnel: {len(records)} total -> {len(kept)} valid  "
              f"(dropped at first failure: n_strips!=3={dropped['n_strips']}, "
              f"y_spread>{Y_TOL:.0f}px={dropped['y_spread']}, "
              f"size_ratio>{SIZE_RATIO}={dropped['size_ratio']})")
        records = kept
        out_path = PHASE_A_FILTERED_OUT
        title = (f"Phase A (filtered) — tilt by position & class  "
                 f"[3 strips, Δy≤{Y_TOL:.0f}px, len ratio≤{SIZE_RATIO}]  (classify_phase1, earlier frame)")
    else:
        out_path = PHASE_A_OUT
        title = "Phase A — tilt angle by bottle position & class (measured on classify_phase1 frames, earlier frame)"

    labels = th.load_labels(th.CLASSIFY_CSV)
    by_position, stats = th.build_dataset(records, labels, image_field="image")
    print(
        f"  histogram: {stats['matched']} matched / {stats['records']} records "
        f"({stats['unlabelled']} unlabelled, {stats['wrong_strip_count']} not-3-strips)"
    )

    every = [a for per_cls in by_position.values() for vals in per_cls.values() for a in vals]
    if not every:
        print("No angles to plot.")
        return
    lo, hi = float(np.percentile(every, 0.5)), float(np.percentile(every, 99.5))
    pad = 0.05 * (hi - lo)
    xlim = (lo - pad, hi + pad)
    print(f"  angle 0.5/50/99.5 pct: "
          f"{np.percentile(every, [0.5, 50, 99.5]).round(1)}  -> xlim {tuple(round(v,1) for v in xlim)}")

    th.plot_position_histograms(by_position, title, out_path, xlim=xlim)
    print(f"Saved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure tilt on classify_phase1 frames.")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N images (0 = all)")
    parser.add_argument("--no-hist", action="store_true", help="Skip the histogram step")
    parser.add_argument("--no-overlays", action="store_true",
                        help="Skip re-rendering validation overlays (re-measure geometry only)")
    parser.add_argument("--filtered-only", action="store_true",
                        help="Render only the filtered figure (leave the unfiltered one as-is)")
    parser.add_argument("--from-json", action="store_true",
                        help="Re-render histograms from the saved measurements_phase1.json (no re-measure)")
    args = parser.parse_args()

    if args.from_json:
        records = json.loads(PHASE1_JSON.read_text())
        print(f"Loaded {len(records)} records from {PHASE1_JSON.name} (no re-measure)")
    else:
        records = measure_all(limit=args.limit, write_overlays=not args.no_overlays)
    if not args.no_hist:
        if not args.filtered_only:
            render_histogram(records, filtered=False)
        render_histogram(records, filtered=True)


if __name__ == "__main__":
    main()
