#!/usr/bin/env python3
"""Phase-aware angle + classification pipeline -> ONE global, traceable dataset.

Processes every .ts in source_videos/ and writes a single global dataset under phase_pipeline_out/:

  measure/<minAngle>_<clip>_C<cycle>_f<frame>.png   P{MEASURE_PHASE} angle overlay (rects + angle text)
  classify_phase<p>/<clip>_C<cycle>_P<p>_f<frame>.png   context frame for looped/smooth/other
  grids/<clip>.png                                   per-clip phase grid (QC)
  measurements.{json,csv}   one row per cycle: angle data + measure_image + classify_image
  classification.{json,csv} one row per cycle: classify_image + measure_image + classification slot

Every row carries id = <clip>_C<cycle>, the clip timestamp, cycle, frame, and BOTH image paths, so the
two files cross-reference each other and every image traces back to its source clip / cycle / frame.
Image paths are relative to phase_pipeline_out/, so the dataset is portable.

Resumable (skips clips already in measurements.json) and stops once TARGET_IMAGES is reached.
Run with the 24H_Insights venv (fCWT + decord):
    24h_env/Scripts/python.exe phase_pipeline.py
"""
import csv
import json
import re
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, r"C:/Users/jkind/Documents/McLaren/24H_Insights")
sys.path.insert(0, str(HERE))
from VideoModule.video_io import read_video
from VideoModule.phase_detection import compute_energy_signal, detect_dynamic_phases
from VideoModule.preprocessing.frame_sampling import phase_grid_frame_indices
from VideoModule.plotting.phase_plots import plot_phase_alignment
from decord import VideoReader, cpu
import measure_tilt

# --- config ---
REF = HERE / "ref.png"
NUM_PHASES = 30
SHIFT_PHASES = 2          # new cycle starts 2 phases earlier => OLD P28 becomes NEW P0
MEASURE_PHASE = 3         # NEW P3 -> angle measurement (ROI + hue/colour + contour filtering)
CONTEXT_PHASES = [1]      # NEW P1 -> classification (looped / smooth / other)
GRID_PHASES = [0, 1, 2, 3, 4, 5]   # phases shown on the per-clip grid
GRID_CYCLES = 8
TARGET_IMAGES = 2000      # stop once the dataset reaches this many measure rows
KEEP_ONLY_3_STRIPS = True # drop any cycle whose measurement didn't find all 3 label strips
OUT = HERE / "phase_pipeline_out"
MEASURE_DIR = OUT / "measure"
GRID_DIR = OUT / "grids"


def classify_dir(phase: int) -> Path:
    return OUT / f"classify_phase{phase}"


def clip_tag(stem: str) -> str:
    """The clip's YYYY-MM-DD_HH-MM-SS timestamp — the traceable clip id used everywhere."""
    m = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", stem)
    return m.group(0) if m else stem


def measure_name(min_angle: float, tag: str, cycle: int, frame: int) -> str:
    return f"{min_angle:06.2f}_{tag}_C{cycle:03d}_f{frame}.png"


def context_name(tag: str, cycle: int, phase: int, frame: int) -> str:
    return f"{tag}_C{cycle:03d}_P{phase}_f{frame}.png"


def make_record(tag, cycle, m_phase, m_frame, n_strips, angles, contexts):
    """Build one traceable record AND its canonical (relative) image paths.

    contexts: list of (phase, frame). The returned image paths are where the pipeline writes (or the
    migration moves) the images, so records and files can never drift.
    """
    min_angle = min(angles) if angles else 99.99
    ctx = [{"phase": p, "frame": f,
            "image": f"classify_phase{p}/{context_name(tag, cycle, p, f)}"} for (p, f) in contexts]
    return {
        "id": f"{tag}_C{cycle:03d}",
        "clip": tag,
        "cycle": cycle,
        "measure": {"phase": m_phase, "frame": m_frame, "n_strips": n_strips,
                    "min_angle": round(min_angle, 2), "angles_deg": [round(a, 2) for a in angles],
                    "image": f"measure/{measure_name(min_angle, tag, cycle, m_frame)}"},
        "context": ctx,
        "classify_image": ctx[0]["image"] if ctx else None,
        "classification": None,
    }


def shift_cycles(cycles, num_phases, shift_phases):
    out = []
    for cs, ce in cycles:
        s = round(shift_phases / num_phases * (ce - cs))
        if cs - s >= 0:
            out.append((cs - s, ce - s))
    return out


def write_global(records, out_dir: Path) -> None:
    """(Re)write the global measurements/classification files from all records."""
    meas = sorted(records, key=lambda r: r["measure"]["min_angle"])   # worst (lowest angle) first
    clf = sorted(records, key=lambda r: r["id"])                       # chronological by clip+cycle

    (out_dir / "measurements.json").write_text(json.dumps([{
        "id": r["id"], "clip": r["clip"], "cycle": r["cycle"],
        "phase": r["measure"]["phase"], "frame": r["measure"]["frame"],
        "n_strips": r["measure"]["n_strips"], "min_angle": r["measure"]["min_angle"],
        "angles_deg": r["measure"]["angles_deg"],
        "measure_image": r["measure"]["image"], "classify_image": r["classify_image"],
    } for r in meas], indent=2))

    (out_dir / "classification.json").write_text(json.dumps([{
        "id": r["id"], "clip": r["clip"], "cycle": r["cycle"],
        "phase": r["context"][0]["phase"] if r["context"] else None,
        "frame": r["context"][0]["frame"] if r["context"] else None,
        "classify_image": r["classify_image"], "measure_image": r["measure"]["image"],
        "classification": r["classification"],
    } for r in clf], indent=2))

    with open(out_dir / "measurements.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "clip", "cycle", "phase", "frame", "n_strips", "min_angle",
                    "angles_deg", "measure_image", "classify_image"])
        for r in meas:
            w.writerow([r["id"], r["clip"], r["cycle"], r["measure"]["phase"], r["measure"]["frame"],
                        r["measure"]["n_strips"], f'{r["measure"]["min_angle"]:.2f}',
                        ";".join(f"{a:.2f}" for a in r["measure"]["angles_deg"]),
                        r["measure"]["image"], r["classify_image"]])

    with open(out_dir / "classification.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "clip", "cycle", "phase", "frame", "classify_image",
                    "measure_image", "classification"])
        for r in clf:
            ctx = r["context"][0] if r["context"] else {"phase": "", "frame": ""}
            w.writerow([r["id"], r["clip"], r["cycle"], ctx["phase"], ctx["frame"],
                        r["classify_image"], r["measure"]["image"], r["classification"] or ""])


def process_video(video: Path) -> list:
    """Phase-aware pass over one clip; writes images to the global folders, returns its records."""
    tag = clip_tag(video.stem)

    # 1. phase awareness (downscaled; ref rotated 180 to match the upside-down camera)
    frames_small, fps = read_video(str(video), resize=(480, 250))
    ref = cv2.rotate(cv2.imread(str(REF)), cv2.ROTATE_180)
    energy, _ = compute_energy_signal(frames_small, energy_method="ncc_ref", reference_frame=ref)
    dyn = detect_dynamic_phases(energy, fps, min_hz=0.2, max_hz=1.0, min_region_seconds=4.0,
                                min_freq_change_hz=0.15, use_fcwt=True, gate=True)
    shifted = shift_cycles(dyn.cycles, NUM_PHASES, SHIFT_PHASES)
    grid = phase_grid_frame_indices(shifted, NUM_PHASES, n_frames=len(frames_small))
    print(f"  backend={dyn.method}  cycles={len(grid)}")

    # 2. per-clip phase grid (upright display) -> grids/<clip>.png
    frames_disp = [cv2.rotate(f, cv2.ROTATE_180) for f in frames_small]
    plot_phase_alignment(frames_disp, shifted, num_phases=NUM_PHASES,
                         num_cycles=min(GRID_CYCLES, len(shifted)), only_phases=GRID_PHASES,
                         title=f"{tag}\nP{MEASURE_PHASE}=measure, P{CONTEXT_PHASES}=classify",
                         save_path=str(GRID_DIR / f"{tag}.png"))

    # 3. re-decode wanted frames at full res (decord random access), rotate upright
    vr = VideoReader(str(video), ctx=cpu(0))
    wanted_by_cycle = [{"cycle": k, "measure_frame": int(row[MEASURE_PHASE]),
                        "context": [(p, int(row[p])) for p in CONTEXT_PHASES]}
                       for k, row in enumerate(grid)]
    wanted = sorted({w["measure_frame"] for w in wanted_by_cycle}
                    | {f for w in wanted_by_cycle for (_, f) in w["context"]})
    batch = vr.get_batch(wanted).asnumpy()
    full = {idx: cv2.rotate(batch[i][:, :, ::-1].copy(), cv2.ROTATE_180)
            for i, idx in enumerate(wanted)}

    # 4. measure + dump context, writing images to the global folders at the record's canonical paths
    records = []
    skipped = 0
    for w in wanted_by_cycle:
        _, measurements = measure_tilt.measure_tilt(full[w["measure_frame"]])
        if KEEP_ONLY_3_STRIPS and len(measurements) != 3:   # keep only fully-measured cycles
            skipped += 1
            continue
        overlay = measure_tilt.draw_debug_overlay(full[w["measure_frame"]], measurements)
        angles = [m["angle_deg"] for m in measurements]
        rec = make_record(tag, w["cycle"], MEASURE_PHASE, w["measure_frame"],
                          len(measurements), angles, w["context"])
        cv2.imwrite(str(OUT / rec["measure"]["image"]), overlay)
        for (p, frame), c in zip(w["context"], rec["context"]):
            cv2.imwrite(str(OUT / c["image"]), full[frame])
        records.append(rec)

    print(f"  {len(records)} cycles kept (3 strips); {skipped} dropped (<3 strips)")
    return records


def _ensure_dirs() -> None:
    for d in (OUT, MEASURE_DIR, GRID_DIR, *[classify_dir(p) for p in CONTEXT_PHASES]):
        d.mkdir(parents=True, exist_ok=True)


def load_existing_records():
    """Re-read the global files into records + the set of clip tags already in the dataset (resume)."""
    mj = OUT / "measurements.json"
    if not mj.exists():
        return [], set()
    prev = json.loads(mj.read_text())
    clf_prev = {c["id"]: c for c in json.loads((OUT / "classification.json").read_text())}
    records, done = [], set()
    for row in prev:
        done.add(row["clip"])
        cid = clf_prev.get(row["id"], {})
        records.append({
            "id": row["id"], "clip": row["clip"], "cycle": row["cycle"],
            "measure": {"phase": row["phase"], "frame": row["frame"], "n_strips": row["n_strips"],
                        "min_angle": row["min_angle"], "angles_deg": row["angles_deg"],
                        "image": row["measure_image"]},
            "context": [{"phase": cid.get("phase"), "frame": cid.get("frame"),
                         "image": row["classify_image"]}] if row["classify_image"] else [],
            "classify_image": row["classify_image"], "classification": cid.get("classification"),
        })
    return records, done


def main() -> None:
    _ensure_dirs()
    records, done = load_existing_records()
    videos = sorted((HERE / "source_videos").glob("*.ts"))
    print(f"target {TARGET_IMAGES}; {len(videos)} clips available; {len(done)} already in dataset")
    for v in videos:
        if len(records) >= TARGET_IMAGES:
            break
        if clip_tag(v.stem) in done:
            continue
        print(f"\n=== {v.name} ===")
        try:
            records.extend(process_video(v))
        except Exception as e:
            print(f"  FAILED {v.name}: {type(e).__name__}: {e}")
            continue
        write_global(records, OUT)
        print(f"  dataset now {len(records)}/{TARGET_IMAGES} measure rows")

    write_global(records, OUT)
    n3 = sum(1 for r in records if r["measure"]["n_strips"] == 3)
    print(f"\nGLOBAL dataset: {len(records)} measure rows ({n3} with 3 strips) -> "
          f"measurements.* / classification.* in {OUT.name}/")


if __name__ == "__main__":
    main()
