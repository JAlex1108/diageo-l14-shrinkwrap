#!/usr/bin/env python3
"""Phase-aware frame extraction for the Diageo shrink-wrapper.

Runs the VideoModule phase-awareness detector on a (long) source video, locates the
phase whose appearance matches a reference image (the measurement phase), and dumps
two folders of full-resolution frames:

  * ``phase_measure/`` - one representative frame per cycle AT the measurement phase
  * ``phase_before/``  - one representative frame per cycle at ``measurement_phase - before_offset``

Each cycle is split into ``--num-phases`` (N) equal phase positions (the module's
``num_phases`` convention). The measurement phase index is found by NCC-matching every
frame against the reference inside an ROI, taking the per-cycle best match, and circular-
averaging those phases across all cycles (robust to a single bad cycle).

Memory: the source is too large to hold full-res, so a downscaled 128x128 ROI stack is
streamed for the signal; only the chosen frames are re-decoded at full res for the dump.

Usage:
    python extract_phases.py --video <source.ts> --ref ref.png --out-dir phase_output
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# VideoModule lives one level up from 24H_Insights on disk.
sys.path.insert(0, r"C:/Users/jkind/Documents/McLaren/24H_Insights")
from VideoModule.phase_detection import detect_dynamic_phases  # noqa: E402
from VideoModule.preprocessing.frame_sampling import phase_grid_frame_indices  # noqa: E402

# Bottle ROI from StraightStrip.json (x, y, w, h), in full-res coordinates.
DEFAULT_ROI = (746, 527, 418, 331)
EMB_DIM = 128  # ROI is downscaled to this square for the NCC signal


def _prep_roi(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop the ROI, grayscale, resize to EMB_DIM, and z-normalise for NCC (dot = corr)."""
    x, y, w, h = roi
    crop = frame[y:y + h, x:x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(gray, (EMB_DIM, EMB_DIM)).astype(np.float32)
    g -= g.mean()
    std = g.std()
    return g / (std + 1e-6)


def stream_roi_stack(video: str, roi: Tuple[int, int, int, int]) -> Tuple[np.ndarray, float]:
    """One sequential decode pass: build the normalised 128x128 ROI stack for every frame.

    Returns (stack (N, 128, 128) float32, fps). ~0.46 GB for a 7k-frame clip.
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    stack: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        stack.append(_prep_roi(frame, roi))
    cap.release()
    return np.asarray(stack, dtype=np.float32), float(fps)


def circular_mean_phase(phases: np.ndarray) -> float:
    """Circular mean of phases in [0, 1). Returns a value in [0, 1)."""
    ang = 2.0 * np.pi * np.asarray(phases, dtype=float)
    mean = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    return float((mean / (2.0 * np.pi)) % 1.0)


def locate_measurement_phase(
    ref_sig: np.ndarray,
    phase_per_frame: np.ndarray,
    cycle_idx_per_frame: np.ndarray,
    cycles: List[Tuple[int, int]],
) -> Tuple[float, List[int]]:
    """Phase (0-1) where the clip best matches the reference, robust across cycles.

    For each cycle, take the frame with the highest reference NCC; that frame's phase is the
    measurement phase for that cycle. Circular-average across cycles. Returns
    (measurement_phase, per_cycle_best_frame_indices).
    """
    per_cycle_best: List[int] = []
    per_cycle_phase: List[float] = []
    for (c_start, c_end) in cycles:
        seg = ref_sig[c_start:c_end]
        if len(seg) == 0:
            continue
        best = c_start + int(np.argmax(seg))
        per_cycle_best.append(best)
        per_cycle_phase.append(float(phase_per_frame[best]))
    if not per_cycle_phase:
        raise RuntimeError("no cycles available to locate the measurement phase")
    return circular_mean_phase(np.asarray(per_cycle_phase)), per_cycle_best


def dump_frames(video: str, wanted: dict[int, Tuple[str, int]], out_root: Path) -> List[dict]:
    """Re-decode the source once and write the wanted frames at full res.

    wanted maps absolute_frame_index -> (folder_name, cycle_index). Returns manifest rows.
    """
    folders = {name for name, _ in wanted.values()}
    for name in folders:
        (out_root / name).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video)
    rows: List[dict] = []
    idx = 0
    remaining = dict(wanted)
    while remaining:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in remaining:
            folder, cyc = remaining.pop(idx)
            fname = f"cycle{cyc:03d}_frame{idx:05d}.png"
            path = out_root / folder / fname
            cv2.imwrite(str(path), frame)
            rows.append({"folder": folder, "cycle": cyc, "frame": idx, "path": str(path)})
        idx += 1
    cap.release()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="source video (.ts/.mp4)")
    ap.add_argument("--ref", required=True, help="reference image of the measurement phase")
    ap.add_argument("--out-dir", default="phase_output", help="output directory")
    ap.add_argument("--num-phases", type=int, default=10, help="N: phases per cycle")
    ap.add_argument("--before-offset", type=int, default=2, help="phases before the measurement phase")
    ap.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"), default=list(DEFAULT_ROI))
    ap.add_argument("--min-hz", type=float, default=0.2, help="cycle frequency band, lower")
    ap.add_argument("--max-hz", type=float, default=1.0, help="cycle frequency band, upper")
    ap.add_argument("--use-fcwt", action="store_true", default=True)
    ap.add_argument("--no-fcwt", dest="use_fcwt", action="store_false")
    # This camera's NCC signal is a clean oscillation whose amplitude grows ~2x across the clip
    # (lighting/scene drift), which trips the default amplitude-stationarity gate ([0.5, 2.0]).
    # The cycle is genuine (high prominence, negative half-period autocorrelation), so widen the
    # amplitude bounds rather than reject the region.
    ap.add_argument("--amp-ratio-min", type=float, default=0.2)
    ap.add_argument("--amp-ratio-max", type=float, default=5.0)
    # The amplitude GATE (active_spans) is for footage with idle/dead stretches. This conveyor
    # oscillates continuously, and the growing amplitude makes the gate collapse the clip to a
    # tiny window — so it is OFF by default here. Re-enable with --gate for gappy footage.
    ap.add_argument("--gate", action="store_true", default=False,
                    help="enable the activity gate (off by default for continuous footage)")
    args = ap.parse_args()

    roi = tuple(args.roi)
    N = args.num_phases
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    ref = cv2.imread(args.ref)
    if ref is None:
        raise FileNotFoundError(f"could not read ref: {args.ref}")
    ref_vec = _prep_roi(ref, roi)

    print(f"[1/5] streaming ROI stack from {Path(args.video).name} ...", flush=True)
    stack, fps = stream_roi_stack(args.video, roi)
    n = len(stack)
    print(f"      {n} frames @ {fps:.1f} fps, stack {stack.nbytes/1e9:.2f} GB", flush=True)

    # NCC signals: self-similarity to frame 0 (drives cycle detection) and to the reference.
    flat = stack.reshape(n, -1)
    energy = (flat @ flat[0]) / (EMB_DIM * EMB_DIM)
    ref_sig = (flat @ ref_vec.reshape(-1)) / (EMB_DIM * EMB_DIM)

    print(f"[2/5] detecting phases ({args.min_hz}-{args.max_hz} Hz) ...", flush=True)
    validation_cfg = {"amp_ratio_min": args.amp_ratio_min, "amp_ratio_max": args.amp_ratio_max}
    dyn = detect_dynamic_phases(energy, fps, min_hz=args.min_hz, max_hz=args.max_hz,
                                use_fcwt=args.use_fcwt, gate=args.gate,
                                fft_validation_config=validation_cfg)
    cycles = list(dyn.cycles)
    print(f"      backend={dyn.method} cycles={len(cycles)} "
          f"region_freqs={[round(r.dominant_freq_hz, 3) for r in dyn.regions]}", flush=True)
    if not cycles:
        raise RuntimeError("no cycles detected; widen the band or lower min_cycles_per_region")

    print("[3/5] locating measurement phase from reference NCC ...", flush=True)
    meas_phase, per_cycle_best = locate_measurement_phase(
        ref_sig, dyn.phase_per_frame, dyn.cycle_idx_per_frame, cycles)
    m = int(np.clip(meas_phase * N, 0, N - 1))
    before_col = m - args.before_offset
    print(f"      measurement phase={meas_phase:.3f} -> bin {m}/{N}; "
          f"before bin = {before_col % N} (offset {args.before_offset})", flush=True)

    # Phase grid: [cycle][phase] -> absolute frame index.
    grid = phase_grid_frame_indices(cycles, N, n_frames=n)

    # Build the wanted-frame set. The measurement frame is grid[k][m]. The "before" frame is
    # grid[k][m-off] when in range, else it wraps to the LATE phase of the PREVIOUS cycle
    # (temporally still `off` phases earlier), skipped for the first cycle.
    wanted: dict[int, Tuple[str, int]] = {}
    for k in range(len(grid)):
        wanted[grid[k][m]] = ("phase_measure", k)
        if before_col >= 0:
            wanted[grid[k][before_col]] = ("phase_before", k)
        elif k > 0:
            wanted[grid[k - 1][before_col + N]] = ("phase_before", k)

    print(f"[4/5] dumping {len(wanted)} frames at full res ...", flush=True)
    rows = dump_frames(args.video, wanted, out_root)

    # Manifest + a compact diagnostic image (energy, cycles, ref_sig, chosen phases).
    with open(out_root / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["folder", "cycle", "frame", "path"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["folder"], r["cycle"])))

    _save_diagnostic(out_root, energy, ref_sig, dyn, cycles, m, before_col, N, fps, per_cycle_best)

    n_meas = sum(1 for r in rows if r["folder"] == "phase_measure")
    n_before = sum(1 for r in rows if r["folder"] == "phase_before")
    print(f"[5/5] done. phase_measure: {n_meas} frames, phase_before: {n_before} frames")
    print(f"      -> {out_root.resolve()}")


def _save_diagnostic(out_root, energy, ref_sig, dyn, cycles, m, before_col, N, fps, per_cycle_best):
    """Render a phase-detection diagnostic plot to out_root/phase_diagnostic.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    t = np.arange(len(energy)) / fps
    fig, ax = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
    ax[0].plot(t, energy, lw=0.8, label="NCC self-sim (energy)")
    for (cs, ce) in cycles:
        ax[0].axvline(cs / fps, color="g", alpha=0.25, lw=0.6)
    ax[0].set_ylabel("energy"); ax[0].legend(loc="upper right")
    ax[0].set_title(f"{len(cycles)} cycles | measurement bin {m}/{N} | before bin {before_col % N}")

    ax[1].plot(t, ref_sig, lw=0.8, color="purple", label="NCC vs reference")
    ax[1].scatter(np.array(per_cycle_best) / fps, ref_sig[per_cycle_best],
                  s=12, color="red", zorder=3, label="per-cycle best match")
    ax[1].set_ylabel("ref NCC"); ax[1].set_xlabel("time (s)"); ax[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_root / "phase_diagnostic.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
