"""Export evenly-distributed full-res frame samples of the bottle-visible phases (side-view Diageo).

What it does, in order:
  1. Reads the 2026-07-06 pooled coverage log and keeps only clips the pipeline verified as
     genuinely cycling (valid_cycles == "y") — a cycling machine means bottles on the line.
  2. Evenly subsamples N clips across the day, downloads each from S3 (cached on disk).
  3. Pass 1 (500x500 decode, same as the pipeline): runs the pipeline's own phase awareness
     (ncc_ref anchor -> cycles -> [cycle][phase] index grid) and collects one candidate frame
     per (cycle, phase) for phases 0..4 — the preset's bottle-in-view window (phase_range).
     Each candidate keeps a small grayscale ROI crop for the similarity check.
  4. Builds a per-phase MEDIAN template from all candidates' crops, then scores every candidate
     against its phase template with NCC (TM_CCOEFF_NORMED). Frames without bottles (empty belt,
     odd machine state) score low and are dropped by --min-ncc.
  5. Per phase: splits the day into <target> time buckets and keeps the highest-NCC candidate
     in each bucket -> even temporal spread AND high similarity. Shortfall is backfilled from
     the remaining highest-NCC candidates (never below the threshold).
  6. Pass 2: re-decodes ONLY the selected frames at native resolution (sequential frame-index
     walk — ffmpeg -ss seeking is unreliable on these re-encoded .ts) and writes JPEG q95 into
     <out>/phase1..phase5, plus a manifest.csv.

Usage:
    python export_phase_samples.py                       # 1000 frames -> Diageo_ShrinkWrap
    python export_phase_samples.py --total 1000 --clips 150 --min-ncc 0.70
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import boto3
import cv2
import numpy as np

REPO = Path(r"c:\Users\jkind\Documents\McLaren\24H_Insights")
sys.path.insert(0, str(REPO))

from VideoModule.image_similarity.roi import crop_to_roi                      # noqa: E402
from VideoModule.io.read import read_video                                    # noqa: E402
from VideoModule.phase_detection.phase_awareness import run_phase_awareness   # noqa: E402
from VideoModule.pipelines.anomaly_detection.src.batch_run import KNOWN_S3_SOURCES  # noqa: E402
from VideoModule.pipelines.anomaly_detection.src.scoring_grid import _load_reference_anchor  # noqa: E402
from VideoModule.pipelines.anomaly_detection.video_anomaly_detection_pipeline import (  # noqa: E402
    CORTEX_SIDEVIEW_CONFIG, NOTEBOOK_ROI)

HERE = Path(__file__).resolve().parent
# The pooled-run coverage log lives in this repo's stoppage analysis (moved from 24H_Insights
# on 2026-07-13) — it is the list of clips verified as cycling.
COVERAGE_LOG = (HERE.parent / "Stoppage_detection" / "output" / "out_normed_diageo"
                / "pooled" / "runs" / "2026-07-06" / "anomaly_coverage_log.csv")
# Keep every complete cycle (no 5% edge trim) — the same setting the pooled run used when it
# verified these clips as cycling, so we recover the same ~3 cycles per short clip.
CFG = CORTEX_SIDEVIEW_CONFIG.replace(cycle_extraction_config={"edge_trim_fraction": 0.0})
OUT_ROOT = HERE / "output"
CROP_SIZE = (256, 160)          # (w, h) of the grayscale ROI crop used for the NCC check
JPEG_QUALITY = 95


def list_source_clips(n_clips: int) -> list[str]:
    """Evenly-spaced clip ids across the day, from the pooled run's verified-cycling rows."""
    rows = [r for r in csv.DictReader(COVERAGE_LOG.open(encoding="utf-8"))
            if r["valid_cycles"] == "y" and r["clip_id"]]
    rows.sort(key=lambda r: r["start"])
    if len(rows) <= n_clips:
        return [r["clip_id"] for r in rows]
    idx = np.linspace(0, len(rows) - 1, n_clips).round().astype(int)
    return [rows[i]["clip_id"] for i in sorted(set(idx.tolist()))]


def fetch_clip(s3, src: dict, clip_id: str, cache_dir: Path) -> Path:
    path = cache_dir / f"{clip_id}.ts"
    if not path.exists() or path.stat().st_size == 0:
        s3.download_file(src["bucket"], f"{src['prefix']}{clip_id}.ts", str(path))
    return path


def candidate_crop(frame_500: np.ndarray) -> np.ndarray:
    """Grayscale ROI crop at a fixed size — the unit the NCC similarity check compares."""
    roi = crop_to_roi(frame_500, NOTEBOOK_ROI)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, CROP_SIZE, interpolation=cv2.INTER_AREA)


def collect_candidates(clip_path: Path, clip_id: str, anchor: np.ndarray,
                       phases: list[int]) -> list[dict]:
    """Pass-1 unit: decode at 500x500, run the pipeline's phase awareness, one candidate per
    (cycle, phase) cell in the bottle-visible phase columns."""
    cfg = CFG
    frames, fps = read_video(str(clip_path), resize=cfg.load_resize)
    if len(frames) < 3:
        return []
    pa = run_phase_awareness(frames, fps, cfg.phase, reference_frame=anchor)
    if len(pa.dyn.cycles) < 2:
        return []
    out = []
    for c, row in enumerate(pa.idx_grid):
        for p in phases:
            if p >= len(row):
                continue
            fi = int(row[p])
            out.append({
                "clip_id": clip_id, "clip_path": clip_path, "frame_idx": fi,
                "cycle": c, "phase": p, "t_in_clip_s": fi / fps if fps > 0 else 0.0,
                "crop": candidate_crop(frames[fi]),
            })
    return out


def score_against_phase_template(cands: list[dict]) -> None:
    """Masked ZNCC per candidate, written into c["ncc"].

    Plain whole-crop NCC fails here: ~80% of the ROI is static machinery, so an EMPTY-BELT
    frame still scores ~0.94 against the phase template (seen on the 11:39 idle-cycling clip).
    Instead: template = per-pixel median over all candidates; mask = the top-quartile
    most-VARIABLE pixels across candidates (the bottle/film region — machinery is constant);
    score = zero-mean NCC over the masked pixels only. A frame without bottles disagrees with
    the template exactly where the mask looks, and drops out of the distribution.
    """
    stack = np.stack([c["crop"] for c in cands]).astype(np.float32)   # (N, H, W)
    template = np.median(stack, axis=0)
    variability = stack.std(axis=0)
    mask = variability >= np.percentile(variability, 75)
    t = template[mask]
    t = t - t.mean()
    t_norm = float(np.linalg.norm(t))
    for c, crop in zip(cands, stack):
        v = crop[mask]
        v = v - v.mean()
        v_norm = float(np.linalg.norm(v))
        c["ncc"] = float(v @ t / (v_norm * t_norm)) if v_norm > 0 and t_norm > 0 else 0.0


def otsu_threshold(vals: np.ndarray) -> float:
    """Otsu's two-cluster split on 1D scores — lands in the valley between the empty-belt
    cluster and the genuine-frame mass. Returns -inf when there is no meaningful low cluster
    (lower class < 3% of candidates), so it never carves up a clean single-cluster phase."""
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-6:
        return float("-inf")
    hist, edges = np.histogram(vals, bins=256, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2
    w_lo = np.cumsum(hist)
    w_hi = w_lo[-1] - w_lo
    sum_lo = np.cumsum(hist * centers)
    mu_lo = np.divide(sum_lo, w_lo, out=np.zeros_like(sum_lo), where=w_lo > 0)
    mu_hi = np.divide(sum_lo[-1] - sum_lo, w_hi, out=np.zeros_like(sum_lo), where=w_hi > 0)
    between = w_lo * w_hi * (mu_lo - mu_hi) ** 2
    k = int(np.argmax(between))
    thr = float(centers[k])
    if (vals < thr).mean() < 0.03:
        return float("-inf")
    return thr


def select_even(cands: list[dict], target: int, min_ncc: float) -> list[dict]:
    """Per phase: best-NCC candidate in each of <target> time buckets, backfilled by NCC rank."""
    passing = [c for c in cands if c["ncc"] >= min_ncc]
    passing.sort(key=lambda c: (c["clip_id"], c["frame_idx"]))   # clip ids sort chronologically
    if len(passing) <= target:
        return passing
    chosen: list[dict] = []
    for bucket in np.array_split(np.arange(len(passing)), target):
        if len(bucket) == 0:
            continue
        best = max((passing[i] for i in bucket), key=lambda c: c["ncc"])
        chosen.append(best)
    left = [c for c in passing if c not in chosen]
    left.sort(key=lambda c: c["ncc"], reverse=True)
    chosen.extend(left[:max(0, target - len(chosen))])
    return chosen


def export_full_res(selected: list[dict], out_root: Path) -> int:
    """Pass 2: per clip, one sequential native-res decode; save only the selected frame indices."""
    by_clip: dict[Path, list[dict]] = defaultdict(list)
    for s in selected:
        by_clip[s["clip_path"]].append(s)
    n_saved = 0
    for k, (clip_path, items) in enumerate(sorted(by_clip.items()), 1):
        wanted = {s["frame_idx"]: [] for s in items}
        for s in items:
            wanted[s["frame_idx"]].append(s)
        cap = cv2.VideoCapture(str(clip_path))
        fi, remaining = 0, set(wanted)
        while remaining:
            ok, frame = cap.read()
            if not ok:
                break
            if fi in remaining:
                for s in wanted[fi]:
                    folder = out_root / f"phase{s['phase']}"
                    folder.mkdir(parents=True, exist_ok=True)
                    name = (f"{s['clip_id']}_c{s['cycle']:02d}_f{s['frame_idx']:04d}"
                            f"_ncc{s['ncc']:.3f}.jpg")
                    cv2.imwrite(str(folder / name), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    s["file"] = f"phase{s['phase']}/{name}"
                    n_saved += 1
                remaining.discard(fi)
            fi += 1
        cap.release()
        if remaining:
            print(f"  [warn] {clip_path.name}: {len(remaining)} selected frame(s) beyond decode end")
        if k % 20 == 0:
            print(f"[pass2] {k}/{len(by_clip)} clips re-decoded, {n_saved} frames saved")
    return n_saved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total", type=int, default=1000)
    ap.add_argument("--clips", type=int, default=150,
                    help="cycling clips to sample evenly across the day")
    ap.add_argument("--min-ncc", type=float, default=0.50,
                    help="drop candidates whose masked NCC vs their phase template is below this")
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    ap.add_argument("--from-cache", action="store_true",
                    help="skip pass 1: reuse <out>/_pass1_cache.npz (fast threshold iteration)")
    args = ap.parse_args()

    cfg = CORTEX_SIDEVIEW_CONFIG
    a0, a1 = cfg.phase_range                       # the preset's bottle-in-view window (0, 4)
    phases = list(range(a0, a1 + 1))
    per_phase = args.total // len(phases)
    anchor = _load_reference_anchor(cfg.reference_image)

    clip_ids = list_source_clips(args.clips)
    print(f"[plan] {len(clip_ids)} clips evenly across 2026-07-06, phases {phases} "
          f"(exported as phase{a0}..phase{a1} — anchor-locked bin numbering), "
          f"{per_phase}/phase, min NCC {args.min_ncc}")

    src = KNOWN_S3_SOURCES["diageo-cortex-41884872"]
    s3 = boto3.Session(profile_name=src["profile"]).client("s3")
    cache_dir = args.out / "_ts_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    pass1_cache = args.out / "_pass1_cache.npz"
    if args.from_cache:
        z = np.load(pass1_cache, allow_pickle=False)
        candidates = [
            {"clip_id": str(cid), "clip_path": cache_dir / f"{cid}.ts", "frame_idx": int(fi),
             "cycle": int(cy), "phase": int(ph), "t_in_clip_s": float(t), "crop": crop}
            for cid, fi, cy, ph, t, crop in zip(
                z["clip_id"], z["frame_idx"], z["cycle"], z["phase"], z["t_in_clip_s"], z["crop"])]
        print(f"[pass1] reused {len(candidates)} candidates from {pass1_cache}")
    else:
        candidates = []
        n_no_cycles = 0
        for k, clip_id in enumerate(clip_ids, 1):
            try:
                path = fetch_clip(s3, src, clip_id, cache_dir)
                got = collect_candidates(path, clip_id, anchor, phases)
            except Exception as e:   # noqa: BLE001 — one bad clip must not kill the sweep
                print(f"  [warn] {clip_id}: {type(e).__name__}: {e}")
                continue
            if not got:
                n_no_cycles += 1
            candidates.extend(got)
            if k % 15 == 0:
                print(f"[pass1] {k}/{len(clip_ids)} clips, {len(candidates)} candidates "
                      f"({n_no_cycles} without usable cycles)")
        if candidates:
            np.savez_compressed(
                pass1_cache,
                clip_id=np.array([c["clip_id"] for c in candidates]),
                frame_idx=np.array([c["frame_idx"] for c in candidates]),
                cycle=np.array([c["cycle"] for c in candidates]),
                phase=np.array([c["phase"] for c in candidates]),
                t_in_clip_s=np.array([c["t_in_clip_s"] for c in candidates]),
                crop=np.stack([c["crop"] for c in candidates]))
            print(f"[pass1] cached to {pass1_cache} (re-run with --from-cache to re-score fast)")

    if not candidates:
        raise SystemExit("no candidates found — nothing cycled?")

    by_phase: dict[int, list[dict]] = defaultdict(list)
    for c in candidates:
        by_phase[c["phase"]].append(c)

    # Per-phase template + variance-masked NCC, then a per-phase floor: Otsu's split between
    # the empty-belt cluster and the genuine mass (never below the --min-ncc absolute floor).
    selected: list[dict] = []
    for p, cands in sorted(by_phase.items()):
        score_against_phase_template(cands)
        vals = np.array([c["ncc"] for c in cands])
        floor = max(args.min_ncc, otsu_threshold(vals))
        pcts = np.percentile(vals, [1, 10, 50, 90])
        print(f"[phase {p} -> folder phase{p}] {len(cands)} candidates, masked NCC "
              f"p1/p10/p50/p90 = {pcts[0]:.3f}/{pcts[1]:.3f}/{pcts[2]:.3f}/{pcts[3]:.3f}; "
              f"floor {floor:.3f} keeps {(vals >= floor).sum()}")
        picked = select_even(cands, per_phase, floor)
        if len(picked) < per_phase:
            print(f"  [short] phase {p}: only {len(picked)}/{per_phase} pass the NCC floor")
        selected.extend(picked)

    print(f"[select] {len(selected)} frames chosen — re-decoding at native resolution")
    for c in selected:
        c.pop("crop", None)                      # free before pass 2
    n_saved = export_full_res(selected, args.out)

    manifest = args.out / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "file", "folder", "phase", "clip_id", "cycle", "frame_idx",
            "t_in_clip_s", "ncc_vs_phase_template"])
        w.writeheader()
        for s in sorted(selected, key=lambda s: (s["phase"], s["clip_id"], s["frame_idx"])):
            w.writerow({
                "file": s.get("file", ""), "folder": f"phase{s['phase']}",
                "phase": s["phase"], "clip_id": s["clip_id"],
                "cycle": s["cycle"], "frame_idx": s["frame_idx"],
                "t_in_clip_s": round(s["t_in_clip_s"], 3),
                "ncc_vs_phase_template": round(s["ncc"], 4)})
    # Traceability dictionary: image filename -> exact source moment (video, cycle, phase, frame).
    trace = {}
    for s in selected:
        if not s.get("file"):
            continue
        name = s["file"].split("/")[-1]
        trace[name] = {
            "video": f"{s['clip_id']}.ts",
            "cycle": s["cycle"],
            "phase": s["phase"],
            "frame_number": s["frame_idx"],
            "t_in_clip_s": round(s["t_in_clip_s"], 3),
            "ncc_vs_phase_template": round(s["ncc"], 4),
        }
    (args.out / "manifest.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(f"[done] {n_saved} full-res frames in {args.out} (+ manifest.csv / manifest.json). "
          f"Clip cache kept in {cache_dir} — delete it to reclaim disk.")


if __name__ == "__main__":
    main()
