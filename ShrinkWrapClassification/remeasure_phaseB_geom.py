#!/usr/bin/env python3
"""Re-measure the Phase-B (phase-3) frames to recover per-strip geometry.

Phase naming follows capture time: Phase A is the EARLIER frame (classify_phase1),
Phase B the LATER one (this phase-3 ``measurements.json`` measure view).

``measurements.json`` only stored angles, so the geometry filter (Δy, size) could
not be applied to Phase B. This re-decodes *only* each record's ``measure_frame``
from its source video, reproduces the exact rotation the pipeline used
(``cv2.rotate(rgb[:, :, ::-1], ROTATE_180)``), re-runs ``measure_tilt`` to get each
strip's centre/length/width, and **verifies the re-measured angles match the stored
ones** (proof the right frame was decoded). It touches nothing else — the
classify_phase1 images and their hand labels are left exactly as they are.

Outputs:
    measurements_phaseB_geom.json          enriched records (angles + centres_y/lengths/widths)
    hist_phaseB_measurements_filtered.png  filtered Phase-B figure (unfiltered kept)

Usage:
    python remeasure_phaseB_geom.py [--limit-clips N]
    python remeasure_phaseB_geom.py --from-geom   # re-render from saved geom JSON (no decord)
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

import measure_tilt
import measure_phase1_tilt as mp
import tilt_histograms as th

# decord is imported lazily inside remeasure() — only the video re-decode needs it,
# so --from-geom re-rendering works in environments without decord installed.

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
SRC = HERE / "source_videos"
MEASUREMENTS_JSON = OUT / "measurements.json"
GEOM_JSON = OUT / "measurements_phaseB_geom.json"
PHASE_B_FILTERED_OUT = OUT / "hist_phaseB_measurements_filtered.png"

ANGLE_MATCH_TOL = 0.06   # stored angles are 2 dp; allow rounding slack on the check

# Phase-3 strips are legitimately uneven in size (the right strip is often half-length
# by perspective), so the phase-1 size gate is dropped here and the y gate is loosened
# to catch only genuinely misaligned strips (y-spread jumps from ~13px at p75 to ~38px
# at p90 -> 20px sits in the gap). Tunable.
PHASE_B_Y_TOL = 20.0
PHASE_B_SIZE_RATIO = None   # None disables the size gate


def clip_tag(stem: str) -> str:
    m = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", stem)
    return m.group(0) if m else stem


def video_map() -> Dict[str, Path]:
    return {clip_tag(p.stem): p for p in SRC.glob("*.ts")}


def remeasure(limit_clips: int = 0) -> List[Dict]:
    """Re-decode each record's measure frame, re-measure, and verify vs stored angles."""
    from decord import VideoReader, cpu  # lazy: only the re-decode path needs decord

    records = json.loads(MEASUREMENTS_JSON.read_text())
    vids = video_map()

    by_clip: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_clip[r["clip"]].append(r)
    clips = sorted(by_clip)
    if limit_clips:
        clips = clips[:limit_clips]

    enriched: List[Dict] = []
    n_checked = n_match = n_badcount = 0
    mismatches: List[str] = []

    for ci, clip in enumerate(clips, 1):
        recs = by_clip[clip]
        vpath = vids.get(clip)
        if vpath is None:
            print(f"  [{ci}/{len(clips)}] {clip}: NO source video, skipped {len(recs)} records")
            continue

        vr = VideoReader(str(vpath), ctx=cpu(0))
        frames_needed = sorted({r["frame"] for r in recs})
        batch = vr.get_batch(frames_needed).asnumpy()  # RGB
        # reproduce phase_pipeline.process_video step 3 exactly: RGB->BGR, rotate 180
        frame_img = {idx: cv2.rotate(batch[i][:, :, ::-1].copy(), cv2.ROTATE_180)
                     for i, idx in enumerate(frames_needed)}

        for r in recs:
            _, meas = measure_tilt.measure_tilt(frame_img[r["frame"]])
            angles = [round(m["angle_deg"], 2) for m in meas]
            stored = r["angles_deg"]

            n_checked += 1
            if len(meas) == 3 and len(stored) == 3 and \
                    all(abs(a - b) <= ANGLE_MATCH_TOL for a, b in zip(angles, stored)):
                n_match += 1
            else:
                if len(meas) != 3:
                    n_badcount += 1
                if len(mismatches) < 8:
                    mismatches.append(f"{r['id']}: stored={stored} remeasured={angles} (n={len(meas)})")

            enriched.append({
                "id": r["id"],
                "clip": clip,
                "classify_image": r["classify_image"],
                "n_strips": len(meas),
                "angles_deg": angles,
                "centers_y": [round(m["center"][1], 1) for m in meas],
                "lengths": [round(m["length"], 1) for m in meas],
                "widths": [round(m["width"], 1) for m in meas],
            })
        del batch, frame_img, vr
        print(f"  [{ci}/{len(clips)}] {clip}: {len(recs)} records re-measured")

    GEOM_JSON.write_text(json.dumps(enriched, indent=2))
    print(f"\nVerification: {n_match}/{n_checked} re-measured angles match stored "
          f"(within ±{ANGLE_MATCH_TOL}°); {n_badcount} re-measured to !=3 strips")
    if mismatches:
        print("  sample mismatches:")
        for s in mismatches:
            print(f"    {s}")
    print(f"Enriched geometry -> {GEOM_JSON}")
    return enriched


def _funnel(enriched: List[Dict], y_tol: float, size_ratio) -> Dict:
    verdicts = [(r, *mp.is_valid(r, y_tol=y_tol, size_ratio=size_ratio)) for r in enriched]
    kept = [r for r, ok, _ in verdicts if ok]
    reasons = [reason for _, ok, reason in verdicts if not ok]
    dropped = {key: reasons.count(key) for key in ("n_strips", "y_spread", "size_ratio")}
    return {"kept": kept, "dropped": dropped}


def render_filtered(enriched: List[Dict]) -> None:
    """Apply the phase-3 geometry filter and render the filtered Phase-B figure.

    Also prints what the user's original phase-1 thresholds would have kept, for
    comparison, so the deviation is explicit.
    """
    strict = _funnel(enriched, mp.Y_TOL, mp.SIZE_RATIO)
    print(f"  [phase-1 (Phase A) thresholds dy<={mp.Y_TOL:.0f}, ratio<={mp.SIZE_RATIO}] would keep "
          f"{len(strict['kept'])}/{len(enriched)}  (drop y={strict['dropped']['y_spread']}, "
          f"size={strict['dropped']['size_ratio']})")

    f = _funnel(enriched, PHASE_B_Y_TOL, PHASE_B_SIZE_RATIO)
    kept = f["kept"]
    size_label = "off" if PHASE_B_SIZE_RATIO is None else f"<={PHASE_B_SIZE_RATIO}"
    print(f"  [phase-3 thresholds dy<={PHASE_B_Y_TOL:.0f}, size {size_label}] keep "
          f"{len(kept)}/{len(enriched)}  (drop n_strips={f['dropped']['n_strips']}, "
          f"y={f['dropped']['y_spread']})")

    labels = th.load_labels(th.CLASSIFY_CSV)
    by_position, stats = th.build_dataset(kept, labels, image_field="classify_image")
    print(f"  histogram: {stats['matched']} matched / {stats['records']} records "
          f"({stats['unlabelled']} unlabelled)")

    th.plot_position_histograms(
        by_position,
        (f"Phase B (filtered) — tilt by position & class  "
         f"[3 strips, Δy≤{PHASE_B_Y_TOL:.0f}px, size {size_label}]  (measurements.json, phase-3 measure, later frame)"),
        PHASE_B_FILTERED_OUT,
        xlim=th.XLIM,
    )
    print(f"Saved -> {PHASE_B_FILTERED_OUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-measure Phase-B (phase-3) geometry and filter.")
    parser.add_argument("--limit-clips", type=int, default=0, help="Only process first N clips (0 = all)")
    parser.add_argument("--from-geom", action="store_true",
                        help="Skip re-decoding; render from the saved measurements_phaseB_geom.json")
    args = parser.parse_args()
    if args.from_geom:
        enriched = json.loads(GEOM_JSON.read_text())
        print(f"Loaded {len(enriched)} records from {GEOM_JSON.name}")
    else:
        enriched = remeasure(limit_clips=args.limit_clips)
    render_filtered(enriched)


if __name__ == "__main__":
    main()
