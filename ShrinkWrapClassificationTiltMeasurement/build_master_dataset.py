#!/usr/bin/env python3
"""Consolidate the four scattered phase-pipeline JSONs + the hand labels into ONE
master dataset: per cycle, the before (classify-frame) and after (measure-frame)
measurements, the class, the source video, the cycle number, angles, and per-strip
xy positions / lengths / widths.

Sources unified (one row per cycle, keyed by id = <clip>_C<cycle>):
  classification.json            spine: video, cycle, classify+measure frames & images
  measurements.json              AFTER  angles + min_angle (phase-3 measure frame, Phase B)
  measurements_phaseB_geom.json  AFTER  geometry (centres_y, lengths, widths; Phase B)
  measurements_phase1.json       BEFORE geometry+angles (phase-1 classify frame, Phase A)
  classify_phase1.csv            class label (1=looped, 2=smooth, 3=other)

The original pipeline serialised only centres_y, dropping each strip's x and box.
This script optionally recovers the true x:
  --before-xy  re-measure the on-disk classify_phase1 PNGs (cv2 only)        -> before center_x + box
  --after-xy   re-decode each measure frame from its source .ts (needs decord) -> after  center_x + box
Both recoveries verify the re-measured angles match the stored ones (proof the
right pixels were measured) and warn loudly on any mismatch -- nothing is silently
substituted.

Output:
  master_dataset.json   nested, one object per cycle (before/after each a strip list)
  master_dataset.csv    flat, one row per cycle (per-strip arrays ';'-joined)

Usage (full xy needs the decord venv that the pipeline uses):
  24h_env/Scripts/python.exe build_master_dataset.py --before-xy --after-xy
  python build_master_dataset.py            # join only, centres_y from the JSONs
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "phase_pipeline_out"
SRC = HERE / "source_videos"

CLASSIFICATION_JSON = OUT / "classification.json"
MEASUREMENTS_JSON = OUT / "measurements.json"
PHASEB_GEOM_JSON = OUT / "measurements_phaseB_geom.json"
PHASE1_JSON = OUT / "measurements_phase1.json"
CLASSIFY_CSV = OUT / "classify_phase1.csv"

MASTER_JSON = OUT / "master_dataset.json"
MASTER_CSV = OUT / "master_dataset.csv"

CLASS_NAMES = {0: "unlabelled", 1: "looped", 2: "smooth", 3: "other"}
ANGLE_TOL = 0.06  # stored angles are 2 dp; allow rounding slack on the verify check


def clip_tag(stem: str) -> str:
    m = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", stem)
    return m.group(0) if m else stem


def box_to_list(box) -> list[list[float]]:
    return [[round(float(x), 1), round(float(y), 1)] for x, y in box]


def strip_from_measure(m: dict) -> dict:
    """A re-measured strip dict -> our uniform strip record (has true xy + box)."""
    return {
        "pos": m["id"],
        "angle_deg": round(float(m["angle_deg"]), 2),
        "center_x": round(float(m["center"][0]), 1),
        "center_y": round(float(m["center"][1]), 1),
        "length": round(float(m["length"]), 1),
        "width": round(float(m["width"]), 1),
        "box": box_to_list(m["box"]),
    }


def strips_from_geom(angles, centers_y, lengths, widths) -> list[dict]:
    """Build strip records from the geometry JSON arrays (centre_y only, no x/box)."""
    out = []
    for i in range(len(angles)):
        out.append({
            "pos": i + 1,
            "angle_deg": angles[i],
            "center_x": None,
            "center_y": centers_y[i] if i < len(centers_y) else None,
            "length": lengths[i] if i < len(lengths) else None,
            "width": widths[i] if i < len(widths) else None,
            "box": None,
        })
    return out


def angles_match(a: list[float], b: list[float]) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= ANGLE_TOL for x, y in zip(a, b))


# --------------------------------------------------------------------------- #
#  x-recovery passes
# --------------------------------------------------------------------------- #
def recover_before_xy(classify_bases: set[str]) -> dict[str, list[dict]]:
    """Re-measure each classify_phase1 PNG -> {classify_basename: [strip,...]}."""
    import cv2
    from measure_phase1_tilt import measure_image

    classify_dir = OUT / "classify_phase1"
    result: dict[str, list[dict]] = {}
    miss = []
    for i, base in enumerate(sorted(classify_bases), 1):
        path = classify_dir / base
        img = cv2.imread(str(path))
        if img is None:
            miss.append(base)
            continue
        _, meas = measure_image(img)
        result[base] = [strip_from_measure(m) for m in meas]
        if i % 400 == 0:
            print(f"    before-xy {i}/{len(classify_bases)}")
    if miss:
        print(f"  ! before-xy: {len(miss)} classify PNGs unreadable/missing (first: {miss[:3]})")
    return result


def recover_after_xy(after_by_clip: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Re-decode each record's measure frame from its source .ts -> {id: [strip,...]}."""
    import cv2
    from decord import VideoReader, cpu
    import measure_tilt

    vids = {clip_tag(p.stem): p for p in SRC.glob("*.ts")}
    result: dict[str, list[dict]] = {}
    no_video = []
    clips = sorted(after_by_clip)
    for ci, clip in enumerate(clips, 1):
        recs = after_by_clip[clip]
        vpath = vids.get(clip)
        if vpath is None:
            no_video.append(clip)
            continue
        vr = VideoReader(str(vpath), ctx=cpu(0))
        frames = sorted({r["frame"] for r in recs})
        batch = vr.get_batch(frames).asnumpy()  # RGB
        img_by_frame = {idx: cv2.rotate(batch[i][:, :, ::-1].copy(), cv2.ROTATE_180)
                        for i, idx in enumerate(frames)}
        for r in recs:
            _, meas = measure_tilt.measure_tilt(img_by_frame[r["frame"]])
            result[r["id"]] = [strip_from_measure(m) for m in meas]
        del batch, img_by_frame, vr
        print(f"    after-xy [{ci}/{len(clips)}] {clip}: {len(recs)} frames")
    if no_video:
        print(f"  ! after-xy: {len(no_video)} clips have no source video (first: {no_video[:3]})")
    return result


# --------------------------------------------------------------------------- #
#  build
# --------------------------------------------------------------------------- #
def build(before_xy: bool, after_xy: bool) -> list[dict]:
    spine = json.loads(CLASSIFICATION_JSON.read_text())
    after_ang = {e["id"]: e for e in json.loads(MEASUREMENTS_JSON.read_text())}
    after_geom = {e["id"]: e for e in json.loads(PHASEB_GEOM_JSON.read_text())}
    before_geom = {e["image"]: e for e in json.loads(PHASE1_JSON.read_text())}

    label_by_base: dict[str, int] = {}
    with CLASSIFY_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                label_by_base[row["video_name"].strip()] = int(row["label"])
            except (ValueError, KeyError):
                pass

    video_file = {clip_tag(p.stem): p.name for p in SRC.glob("*.ts")}

    # optional x-recovery
    before_meas: dict[str, list[dict]] = {}
    after_meas: dict[str, list[dict]] = {}
    if before_xy:
        print("  recovering before (classify-frame) xy by re-measuring PNGs ...")
        before_meas = recover_before_xy({Path(e["classify_image"]).name for e in spine})
    if after_xy:
        print("  recovering after (measure-frame) xy by re-decoding videos ...")
        by_clip = defaultdict(list)
        for e in after_ang.values():
            by_clip[e["clip"]].append(e)
        after_meas = recover_after_xy(by_clip)

    records = []
    n_before_mismatch = n_after_mismatch = 0

    for e in sorted(spine, key=lambda r: r["id"]):
        cid = e["id"]
        classify_base = Path(e["classify_image"]).name
        label = label_by_base.get(classify_base, 0)
        aa = after_ang.get(cid, {})

        # ---- before (phase-1 classify frame) ----
        bg = before_geom.get(classify_base, {})
        if classify_base in before_meas:
            before_strips = before_meas[classify_base]
            if bg.get("angles_deg") and not angles_match(
                    [s["angle_deg"] for s in before_strips], bg["angles_deg"]):
                n_before_mismatch += 1
        elif bg:
            before_strips = strips_from_geom(
                bg["angles_deg"], bg["centers_y"], bg["lengths"], bg["widths"])
        else:
            before_strips = []
        before_angles = [s["angle_deg"] for s in before_strips]

        # ---- after (phase-3 measure frame) ----
        ag = after_geom.get(cid, {})
        if cid in after_meas:
            after_strips = after_meas[cid]
            if aa.get("angles_deg") and not angles_match(
                    [s["angle_deg"] for s in after_strips], aa["angles_deg"]):
                n_after_mismatch += 1
        elif ag:
            after_strips = strips_from_geom(
                ag["angles_deg"], ag["centers_y"], ag["lengths"], ag["widths"])
        elif aa.get("angles_deg"):
            after_strips = strips_from_geom(aa["angles_deg"], [], [], [])
        else:
            after_strips = []
        after_angles = [s["angle_deg"] for s in after_strips]

        records.append({
            "id": cid,
            "video": e["clip"],
            "source_video": video_file.get(e["clip"]),
            "cycle": e["cycle"],
            "class": label,
            "class_name": CLASS_NAMES.get(label, str(label)),
            "before": {
                "stage": "phase1_classify",
                "frame": e.get("frame"),
                "image": e["classify_image"],
                "n_strips": len(before_strips),
                "min_angle": round(min(before_angles), 2) if before_angles else None,
                "angles_deg": before_angles,
                "strips": before_strips,
            },
            "after": {
                "stage": "phase3_measure",
                "frame": aa.get("frame"),
                "image": aa.get("measure_image") or e.get("measure_image"),
                "n_strips": aa.get("n_strips", len(after_strips)),
                "min_angle": aa.get("min_angle"),
                "angles_deg": after_angles,
                "strips": after_strips,
            },
        })

    if before_xy and n_before_mismatch:
        print(f"  ! before-xy: {n_before_mismatch} cycles' re-measured angles differ from phase1 JSON")
    if after_xy and n_after_mismatch:
        print(f"  ! after-xy: {n_after_mismatch} cycles' re-measured angles differ from measurements.json")
    return records


def write_csv(records: list[dict]) -> None:
    def arr(strips, key):
        return ";".join("" if s[key] is None else f"{s[key]}" for s in strips)

    cols = [
        "id", "video", "source_video", "cycle", "class", "class_name",
        "before_frame", "before_n_strips", "before_min_angle",
        "before_angles", "before_centers_x", "before_centers_y", "before_lengths", "before_widths",
        "after_frame", "after_n_strips", "after_min_angle",
        "after_angles", "after_centers_x", "after_centers_y", "after_lengths", "after_widths",
        "classify_image", "measure_image",
    ]
    with MASTER_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in records:
            b, a = r["before"], r["after"]
            w.writerow([
                r["id"], r["video"], r["source_video"], r["cycle"], r["class"], r["class_name"],
                b["frame"], b["n_strips"], b["min_angle"],
                arr(b["strips"], "angle_deg"), arr(b["strips"], "center_x"),
                arr(b["strips"], "center_y"), arr(b["strips"], "length"), arr(b["strips"], "width"),
                a["frame"], a["n_strips"], a["min_angle"],
                arr(a["strips"], "angle_deg"), arr(a["strips"], "center_x"),
                arr(a["strips"], "center_y"), arr(a["strips"], "length"), arr(a["strips"], "width"),
                b["image"], a["image"],
            ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before-xy", action="store_true",
                    help="recover before center_x + box by re-measuring classify PNGs (cv2)")
    ap.add_argument("--after-xy", action="store_true",
                    help="recover after center_x + box by re-decoding source videos (decord)")
    args = ap.parse_args()

    print("Building master dataset ...")
    records = build(before_xy=args.before_xy, after_xy=args.after_xy)

    MASTER_JSON.write_text(json.dumps(records, indent=2))
    write_csv(records)

    have_bx = sum(1 for r in records if r["before"]["strips"] and r["before"]["strips"][0]["center_x"] is not None)
    have_ax = sum(1 for r in records if r["after"]["strips"] and r["after"]["strips"][0]["center_x"] is not None)
    by_class: dict[str, int] = defaultdict(int)
    for r in records:
        by_class[r["class_name"]] += 1
    print(f"\n{len(records)} cycles -> {MASTER_JSON.name} + {MASTER_CSV.name}")
    print(f"  per class           : {dict(sorted(by_class.items()))}")
    print(f"  before strips w/ x   : {have_bx}/{len(records)}")
    print(f"  after  strips w/ x   : {have_ax}/{len(records)}")


if __name__ == "__main__":
    main()
