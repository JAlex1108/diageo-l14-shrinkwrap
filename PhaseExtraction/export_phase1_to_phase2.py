"""Per cycle: 5 evenly spaced full-res frames spanning phase 1 -> phase 2 (inclusive) — the
lap-bar rising moment. Reuses the export run's pass-1 cache (no re-download / re-phase):
each (clip, cycle) already has its phase-1 and phase-2 frame indices and ROI crops. A cycle
is kept only when BOTH endpoints pass their phase's bottle filter (variance-masked NCC vs
the phase template, per-phase Otsu floor — same filter as the main export). Output goes to
<out>/phase1_to_phase2/ with its own manifest pair."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

SP = Path(__file__).parent
sys.path.insert(0, str(SP))
from export_phase_samples import (   # noqa: E402
    JPEG_QUALITY, OUT_ROOT, otsu_threshold, score_against_phase_template)

N_BETWEEN = 5
PHASE_A, PHASE_B = 1, 2
MIN_NCC_ABS = 0.5

out_root = OUT_ROOT
sub = out_root / "phase1_to_phase2"
sub.mkdir(parents=True, exist_ok=True)
cache_dir = out_root / "_ts_cache"

z = np.load(out_root / "_pass1_cache.npz", allow_pickle=False)

# Score phases 1 and 2 exactly like the main export, then keep each phase's floor.
floors: dict[int, float] = {}
scores: dict[tuple[str, int, int], float] = {}     # (clip_id, cycle, phase) -> ncc
endpoint: dict[tuple[str, int, int], dict] = {}    # ... -> {"frame_idx", "t"}
for p in (PHASE_A, PHASE_B):
    sel = np.where(z["phase"] == p)[0]
    cands = [{"crop": z["crop"][i]} for i in sel]
    score_against_phase_template(cands)
    vals = np.array([c["ncc"] for c in cands])
    floors[p] = max(MIN_NCC_ABS, otsu_threshold(vals))
    for i, c in zip(sel, cands):
        key = (str(z["clip_id"][i]), int(z["cycle"][i]), p)
        scores[key] = c["ncc"]
        endpoint[key] = {"frame_idx": int(z["frame_idx"][i]),
                         "t": float(z["t_in_clip_s"][i])}
    print(f"phase {p}: floor {floors[p]:.3f}")

# Valid cycles: both endpoints exist and pass their floor.
cycles = sorted({(cid, cy) for (cid, cy, _p) in scores})
plan: dict[str, list[dict]] = defaultdict(list)    # clip_id -> planned frames
n_cycles_kept = 0
for cid, cy in cycles:
    ka, kb = (cid, cy, PHASE_A), (cid, cy, PHASE_B)
    if ka not in scores or kb not in scores:
        continue
    if scores[ka] < floors[PHASE_A] or scores[kb] < floors[PHASE_B]:
        continue
    n_cycles_kept += 1
    fa, fb = endpoint[ka]["frame_idx"], endpoint[kb]["frame_idx"]
    ta, tb = endpoint[ka]["t"], endpoint[kb]["t"]
    idxs = np.linspace(fa, fb, N_BETWEEN).round().astype(int)
    for k, fi in enumerate(idxs):
        frac = k / (N_BETWEEN - 1)
        plan[cid].append({
            "clip_id": cid, "cycle": cy, "k": k, "frame_idx": int(fi),
            "t_in_clip_s": round(ta + frac * (tb - ta), 3),
            "ncc_phase1": round(scores[ka], 4), "ncc_phase2": round(scores[kb], 4)})
print(f"{n_cycles_kept} cycles pass the bottle filter -> "
      f"{sum(len(v) for v in plan.values())} frames from {len(plan)} clips")

trace: dict[str, dict] = {}
n_saved = 0
for j, (cid, items) in enumerate(sorted(plan.items()), 1):
    clip_path = cache_dir / f"{cid}.ts"
    if not clip_path.exists():
        print(f"  [warn] {cid}: not in _ts_cache, skipped ({len(items)} frames)")
        continue
    wanted = defaultdict(list)
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
                name = f"{cid}_c{s['cycle']:02d}_k{s['k']}_f{s['frame_idx']:04d}.jpg"
                cv2.imwrite(str(sub / name), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                trace[name] = {
                    "video": f"{cid}.ts", "cycle": s["cycle"],
                    "position_k": s["k"], "n_between": N_BETWEEN,
                    "span": f"phase{PHASE_A}->phase{PHASE_B}",
                    "frame_number": s["frame_idx"], "t_in_clip_s": s["t_in_clip_s"],
                    "ncc_phase1": s["ncc_phase1"], "ncc_phase2": s["ncc_phase2"]}
                n_saved += 1
            remaining.discard(fi)
        fi += 1
    cap.release()
    if j % 20 == 0:
        print(f"[export] {j}/{len(plan)} clips, {n_saved} frames")

(sub / "manifest.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
with (sub / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["file", "video", "cycle", "position_k", "frame_number",
                                       "t_in_clip_s", "ncc_phase1", "ncc_phase2"])
    w.writeheader()
    for name in sorted(trace):
        e = trace[name]
        w.writerow({"file": name, "video": e["video"], "cycle": e["cycle"],
                    "position_k": e["position_k"], "frame_number": e["frame_number"],
                    "t_in_clip_s": e["t_in_clip_s"], "ncc_phase1": e["ncc_phase1"],
                    "ncc_phase2": e["ncc_phase2"]})
print(f"[done] {n_saved} frames in {sub} (+ manifest.csv / manifest.json)")
