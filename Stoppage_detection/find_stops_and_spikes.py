"""Post-process finished gap-run days: rank likely STOPPAGES and extract STRONG anomaly frames.

Uses only artefacts already on disk (coverage logs, run summaries, per-clip scores.npz) plus
targeted S3 downloads for the frames actually exported. No pipeline re-run.

Stoppage candidates, two signatures, ranked together:
  * mid-file: a thrown-away clip whose activity spans END well before the file does — the
    scene went still mid-clip and stayed still (spans parsed from the skip reason).
  * between-file: a working-hours recording gap (this camera records only while the scene
    moves, so minutes of silence = the line stood still).
  A dead tail flowing INTO a recording gap ranks highest (the stop persisted).

Strong anomaly frames: every clip's scores.npz frame signal is scanned for frames above
--sigma. In the pooled tree the signal is already per-window z (sigma units); in raw
(un-normalised) trees each clip is standardised against itself first. The top spikes are
exported as jpgs (frame +- neighbours) with the sigma value stamped on.

Stop VERIFICATION (--verify-stops N) downloads the top-N candidate clips and runs the
tested stop-detection brick over each: clip verdict (GENUINE = stop on film / BOUNDARY =
stop at the recording cut / ALREADY = line was already down) plus every flat-line or
amplitude-drop interval found inside the clip. Results -> curated_stops.csv.

Usage:
    python find_stops_and_spikes.py                      # scan default trees, report
    python find_stops_and_spikes.py --export-frames 6    # also export top spike frames
    python find_stops_and_spikes.py --verify-stops 25    # verify top candidates (S3)
"""
import argparse
import csv
import json
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import boto3
import cv2
import numpy as np
from botocore.exceptions import BotoCoreError, ClientError

# Lives in Diageo_ShrinkWrap/Stoppage_detection; runs against the 24H_Insights library
# (dev repo). The diageo run trees sit under output/ next to this script.
REPO = Path(r"c:\Users\jkind\Documents\McLaren\24H_Insights")
ANALYSIS = Path(__file__).resolve().parent / "output"   # .../Diageo_ShrinkWrap/Stoppage_detection/output
sys.path.insert(0, str(REPO))

from VideoModule.anomaly_detection.stop_detection import (  # noqa: E402
    classify_stop_verdict,
    detect_stops,
    video_motion_trace,
)

BUCKET = "diageo-prod-global-dashcam-mc-nuc-video"
PREFIX = "cortexvpu-01a-005-41884872/"          # the side-view camera these trees were run on
PROFILE = "522196013725_DashcamGlbDiageoProdDataContrib"

Z_ROOTS = [ANALYSIS / "out_normed_diageo" / "pooled" / "runs"]   # scores.npz already in sigma units
RAW_ROOTS = [ANALYSIS / "out_diageo" / "runs"]                    # raw dissimilarity -> per-clip z

SPANS_RE = re.compile(r"active_spans=\[(.*?)\]")
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")
WORK_HOURS = (6, 22)


def parse_spans(reason: str) -> list[tuple[int, int]]:
    m = SPANS_RE.search(reason)
    if not m or not m.group(1).strip():
        return []
    return [(int(a), int(b)) for a, b in re.findall(r"\((\d+),\s*(\d+)\)", m.group(1))]


def clip_hour(clip_or_ts: str) -> int:
    m = TS_RE.search(clip_or_ts)
    return int(m.group(1)[11:13]) if m else -1


# --- stoppage candidates ------------------------------------------------------------------------

def find_stop_candidates(day_dir: Path) -> list[dict]:
    """Rank-ready stop candidates for one completed day (needs its coverage log)."""
    log = day_dir / "anomaly_coverage_log.csv"
    if not log.exists():
        return []
    rows = list(csv.DictReader(log.open(encoding="utf-8")))

    n_frames_by_clip: dict[str, tuple[int, float]] = {}
    summary = day_dir / "anomaly_run_summary.json"
    if summary.exists():
        for c in json.loads(summary.read_text(encoding="utf-8"))["clips"]:
            n_frames_by_clip[c["clip_id"]] = (int(c["n_frames"]), float(c["fps"]) or 60.0)

    out = []
    for i, r in enumerate(rows):
        following_gap = 0.0
        if i + 1 < len(rows) and rows[i + 1]["data_present"] == "n":
            following_gap = float(rows[i + 1]["duration_s"])

        # mid-file: activity died before the file ended
        if r["data_present"] == "y" and "active_spans" in r["reason"]:
            spans = parse_spans(r["reason"])
            if spans:
                n_frames, fps = n_frames_by_clip.get(
                    r["clip_id"], (int(float(r["duration_s"]) * 60.0), 60.0))
                dead_tail_s = max(0.0, (n_frames - spans[-1][1]) / fps)
                if dead_tail_s >= 3.0:
                    out.append({
                        "kind": "mid-file stop", "when": r["start"], "clip": r["clip_id"],
                        "dead_tail_s": round(dead_tail_s, 1),
                        "following_gap_s": round(following_gap, 1),
                        "score": dead_tail_s + min(following_gap, 600.0),
                        "detail": f"motion ends at frame {spans[-1][1]}/{n_frames}",
                    })

        # between-file: a working-hours recording gap (skip the overnight/edge monsters)
        if (r["data_present"] == "n" and 90.0 <= float(r["duration_s"]) <= 4 * 3600
                and WORK_HOURS[0] <= clip_hour(r["start"]) < WORK_HOURS[1]):
            prev_clip = rows[i - 1]["clip_id"] if i > 0 else ""
            out.append({
                "kind": "recording gap", "when": r["start"], "clip": prev_clip,
                "dead_tail_s": 0.0, "following_gap_s": round(float(r["duration_s"]), 1),
                "score": float(r["duration_s"]),
                "detail": f"camera silent {float(r['duration_s'])/60:.1f} min (last clip before)",
            })
    return out


def verify_stop_candidates(stops: list[dict], out_dir: Path, top_n: int) -> None:
    """Download the top-N candidate clips and verify each with the stop-detection brick.

    Writes curated_stops.csv: per clip the 07-06-rules verdict plus every flat-line /
    amplitude-drop interval detect_stops finds inside the clip.
    """
    # a clip can appear twice (mid-file + gap-before): keep the best gap info per clip
    cands: dict[str, dict] = {}
    for s in stops:
        if not s["clip"]:
            continue
        c = cands.setdefault(s["clip"], {"clip": s["clip"], "when": s["when"], "gap_s": 0.0})
        c["gap_s"] = max(c["gap_s"], float(s["following_gap_s"]))
        c["when"] = min(c["when"], s["when"])
    chosen = sorted(cands.values(), key=lambda c: -c["gap_s"])[:top_n]
    print(f"\n=== verifying {len(chosen)} candidate clip(s) with the stop-detection brick ===")

    s3 = boto3.Session(profile_name=PROFILE).client("s3")
    rows = []
    for c in chosen:
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / f"{c['clip']}.ts"
            try:
                s3.download_file(BUCKET, f"{PREFIX}{c['clip']}.ts", str(local))
                trace = video_motion_trace(local)
            except (BotoCoreError, ClientError, OSError, ValueError) as exc:
                print(f"  [skip] {c['clip'][-26:]}: {type(exc).__name__}: {exc}")
                continue
        if trace.signal.size == 0:
            print(f"  [skip] {c['clip'][-26:]}: single-frame clip, no motion signal")
            continue
        v = classify_stop_verdict(trace.signal, trace.fps)
        found = detect_stops(trace.signal, trace.fps)
        rows.append({
            **c, "verdict": v.verdict, "n_frames": trace.n_frames, "fps": trace.fps,
            "death_frame": v.death_frame, "dead_tail_s": round(v.dead_tail_s, 1),
            "active_early_frac": round(v.active_early_frac, 2),
            "stops": "; ".join(f"{s.kind} {s.start_s:.1f}-{s.end_s:.1f}s" for s in found),
        })
        print(f"  {c['when']}  {v.verdict:12s} tail={v.dead_tail_s:5.1f}s "
              f"gap={c['gap_s']:6.0f}s  stops=[{rows[-1]['stops']}]  {c['clip'][-26:]}")

    if rows:
        with (out_dir / "curated_stops.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: (r["verdict"] != "GENUINE",
                                                    r["verdict"] != "BOUNDARY", -r["gap_s"])))
        print(f"  -> {out_dir / 'curated_stops.csv'}")


# --- strong anomaly spikes ----------------------------------------------------------------------

def find_spikes(runs_root: Path, sigma: float, already_z: bool) -> list[dict]:
    """Frames above sigma in every clip's scores.npz under one runs tree."""
    spikes = []
    for npz_path in runs_root.glob("*/*/scores.npz"):
        d = np.load(npz_path)
        sig = d["frame_signal"].astype(float)
        if not sig.size:
            continue
        if not already_z:
            sd = float(sig.std())
            if sd == 0:
                continue
            sig = (sig - sig.mean()) / sd
        fps = float(d["fps"]) if "fps" in d.files else 60.0
        over = np.flatnonzero(sig > sigma)
        if not over.size:
            continue
        # group consecutive over-threshold frames into one spike each
        breaks = np.flatnonzero(np.diff(over) > 1)
        for grp in np.split(over, breaks + 1):
            peak = int(grp[np.argmax(sig[grp])])
            spikes.append({
                "clip": npz_path.parent.name, "day": npz_path.parent.parent.name,
                "frame": peak, "t_s": round(peak / fps, 1),
                "sigma": round(float(sig[peak]), 2),
                "n_frames_over": int(grp.size), "tree": runs_root.parent.name,
            })
    return spikes


def export_spike_frames(spikes: list[dict], out_dir: Path, scale: float = 0.5) -> None:
    """Download each spike's clip once and save the peak frame +- 2 neighbours as jpgs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    s3 = boto3.Session(profile_name=PROFILE).client("s3")
    by_clip: dict[str, list[dict]] = {}
    for s in spikes:
        by_clip.setdefault(s["clip"], []).append(s)
    for clip_id, clip_spikes in by_clip.items():
        wanted: dict[int, dict] = {}
        for s in clip_spikes:
            for k in (-2, 0, 2):
                wanted.setdefault(max(0, s["frame"] + k), s)
        print(f"[frames] {clip_id}: {len(clip_spikes)} spike(s) -> {len(wanted)} frame(s)")
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / f"{clip_id}.ts"
            s3.download_file(BUCKET, f"{PREFIX}{clip_id}.ts", str(local))
            cap = cv2.VideoCapture(str(local))
            idx = -1
            while wanted:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                if idx not in wanted:
                    continue
                s = wanted.pop(idx)
                if scale != 1.0:
                    frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA)
                label = f"{s['sigma']:.1f} sigma  frame {idx}  ({s['t_s']}s)"
                cv2.putText(frame, label, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 8)
                cv2.putText(frame, label, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                path = out_dir / f"spike_{s['sigma']:.1f}sig_{clip_id}_f{idx:05d}.jpg"
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cap.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sigma", default=5.0, type=float, help="spike threshold in sigma units")
    ap.add_argument("--top", default=20, type=int, help="rows to print per table")
    ap.add_argument("--export-frames", default=0, type=int, metavar="N",
                    help="download + export jpgs for the top N spike clips (0 = report only)")
    ap.add_argument("--verify-stops", default=0, type=int, metavar="N",
                    help="download the top N stop-candidate clips and verify them with the "
                         "stop-detection brick (0 = report only)")
    ap.add_argument("--out", default=ANALYSIS / "out_postprocess", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # --- stoppages, over every completed day in every tree ---
    stops = []
    for root in Z_ROOTS + RAW_ROOTS:
        if root.exists():
            for day_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                stops.extend(find_stop_candidates(day_dir))
    stops.sort(key=lambda s: -s["score"])
    print(f"\n=== stoppage candidates: {len(stops)} (top {args.top}) ===")
    for s in stops[:args.top]:
        print(f"  {s['when']}  {s['kind']:14s} dead_tail={s['dead_tail_s']:6.1f}s "
              f"gap_after={s['following_gap_s']:7.1f}s  {s['detail']}")
    if stops:
        with (args.out / "stop_candidates.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(stops[0].keys()))
            w.writeheader()
            w.writerows(stops)
        print(f"  -> full list: {args.out / 'stop_candidates.csv'}")

    if args.verify_stops and stops:
        verify_stop_candidates(stops, args.out, args.verify_stops)

    # --- sigma spikes across all scores.npz ---
    spikes = []
    for root in Z_ROOTS:
        if root.exists():
            spikes.extend(find_spikes(root, args.sigma, already_z=True))
    for root in RAW_ROOTS:
        if root.exists():
            spikes.extend(find_spikes(root, args.sigma, already_z=False))
    spikes.sort(key=lambda s: -s["sigma"])
    print(f"\n=== frames > {args.sigma:.0f} sigma: {len(spikes)} spike(s) (top {args.top}) ===")
    for s in spikes[:args.top]:
        print(f"  {s['sigma']:6.2f} sigma  {s['day']}  {s['clip'][-26:]}  frame {s['frame']} "
              f"({s['t_s']}s, {s['n_frames_over']} frames over)  [{s['tree']}]")
    if spikes:
        with (args.out / "sigma_spikes.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(spikes[0].keys()))
            w.writeheader()
            w.writerows(spikes)
        print(f"  -> full list: {args.out / 'sigma_spikes.csv'}")

    if args.export_frames and spikes:
        top_clips = []
        for s in spikes:                        # top N distinct clips, keeping all their spikes
            if s["clip"] not in top_clips:
                top_clips.append(s["clip"])
            if len(top_clips) >= args.export_frames:
                break
        chosen = [s for s in spikes if s["clip"] in top_clips]
        export_spike_frames(chosen, args.out / "spike_frames")
        print(f"\nspike frames -> {args.out / 'spike_frames'}")


if __name__ == "__main__":
    main()
