#!/usr/bin/env python3
"""Phase awareness for a source video, using the VideoModule's OWN plotters.

read_video -> compute_energy_signal -> detect_dynamic_phases (fCWT), then renders the module's
standard figures (no hand-rolled matplotlib):
  * plot_dynamic_phase_overview  -> phase_overview.png  (energy+boundaries, regions, phase, cycle lengths)
  * plot_phase_alignment         -> phase_grid.png      (frames at each phase across cycles)

Run with the 24H_Insights venv so fCWT is available:
    24h_env/Scripts/python.exe energy_cycle.py --video source_videos/<name>.ts
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, r"C:/Users/jkind/Documents/McLaren/24H_Insights")
from VideoModule.video_io import read_video  # noqa: E402
from VideoModule.phase_detection import compute_energy_signal, detect_dynamic_phases  # noqa: E402
from VideoModule.plotting.dynamic_phase_plots import plot_dynamic_phase_overview  # noqa: E402
from VideoModule.plotting.phase_plots import plot_phase_alignment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--energy", default="ncc", choices=["ncc", "ncc_ref", "velocity", "spatial", "ncc_edges"])
    ap.add_argument("--ref", default=None, help="reference frame (required for --energy ncc_ref)")
    ap.add_argument("--rotate-ref-180", action="store_true",
                    help="rotate the reference 180 deg before matching (ref.png is inverted vs the source frames)")
    ap.add_argument("--rotate-180", action="store_true",
                    help="rotate every source frame 180 deg to ref/measurement orientation (camera is "
                         "mounted upside down). Same energy as --rotate-ref-180 but the grid shows upright frames.")
    ap.add_argument("--min-hz", type=float, default=0.2)
    ap.add_argument("--max-hz", type=float, default=1.0)
    # Region params scaled to a low-frequency camera (module defaults target 5-25 Hz labellers).
    ap.add_argument("--min-region-seconds", type=float, default=4.0)
    ap.add_argument("--min-freq-change-hz", type=float, default=0.15)
    ap.add_argument("--num-phases", type=int, default=10, help="phase columns in the phase grid")
    ap.add_argument("--grid-cycles", type=int, default=10, help="cycles shown in the phase grid")
    ap.add_argument("--resize", type=int, nargs=2, default=[480, 250], metavar=("W", "H"))
    ap.add_argument("--roi", type=int, nargs=4, default=None, metavar=("X", "Y", "W", "H"),
                    help="crop the phase grid to this full-res ROI (legible bottles per phase)")
    ap.add_argument("--grid-brighten", type=float, default=1.0, help="brightness multiplier for grid cells")
    ap.add_argument("--overview-out", default="phase_overview.png")
    ap.add_argument("--grid-out", default="phase_grid.png")
    args = ap.parse_args()

    import cv2
    print(f"read_video({Path(args.video).name}, resize={tuple(args.resize)}) ...", flush=True)
    frames, fps = read_video(args.video, resize=tuple(args.resize))
    if args.rotate_180:
        frames = [cv2.rotate(f, cv2.ROTATE_180) for f in frames]
        print("  rotated frames 180 deg -> measurement orientation", flush=True)
    print(f"  {len(frames)} frames @ {fps:.1f} fps ({len(frames)/fps:.1f} s)", flush=True)

    kw = {}
    if args.energy == "ncc_ref":
        if not args.ref:
            raise SystemExit("--energy ncc_ref requires --ref")
        ref_img = cv2.imread(args.ref)
        if args.rotate_ref_180:
            ref_img = cv2.rotate(ref_img, cv2.ROTATE_180)
        kw["reference_frame"] = ref_img
    energy, used = compute_energy_signal(frames, energy_method=args.energy, **kw)

    dyn = detect_dynamic_phases(
        energy, fps, min_hz=args.min_hz, max_hz=args.max_hz,
        min_region_seconds=args.min_region_seconds, min_freq_change_hz=args.min_freq_change_hz,
        use_fcwt=True, gate=True,
    )
    print(f"  backend={dyn.method}  energy={used}  regions={len(dyn.regions)}  cycles={len(dyn.cycles)}")
    print(f"  active_spans={dyn.metadata.get('active_spans')}")
    for i, r in enumerate(dyn.regions):
        print(f"    region {i}: {r.start_idx}-{r.end_idx} ({(r.end_idx-r.start_idx)/fps:.1f}s) "
              f"f={r.dominant_freq_hz:.3f}Hz cycle_len={r.cycle_length_frames:.0f}f valid={r.is_valid}")

    # Standard module figures.
    plot_dynamic_phase_overview(dyn, output_path=Path(args.overview_out))
    print(f"  overview -> {Path(args.overview_out).resolve()}")

    if len(dyn.cycles) >= 2:
        grid_frames = frames
        roi_note = "full frame"
        if args.roi:
            # Scale the full-res ROI to the decode size and crop every frame for the grid.
            sx, sy = args.resize[0] / 1920.0, args.resize[1] / 1000.0
            x, y, w, h = args.roi
            rx, ry, rw, rh = int(x * sx), int(y * sy), int(w * sx), int(h * sy)
            grid_frames = [f[ry:ry + rh, rx:rx + rw] for f in frames]
            roi_note = f"ROI {tuple(args.roi)}"
        if args.grid_brighten != 1.0:
            import numpy as np
            grid_frames = [np.clip(f.astype(np.float32) * args.grid_brighten, 0, 255).astype(np.uint8)
                           for f in grid_frames]
        plot_phase_alignment(
            grid_frames, dyn.cycles,
            num_phases=args.num_phases, num_cycles=min(args.grid_cycles, len(dyn.cycles)),
            title=f"Phase grid ({used}, {dyn.method}, {roi_note}) - {Path(args.video).stem}",
            save_path=args.grid_out,
        )
        print(f"  phase grid -> {Path(args.grid_out).resolve()}")
    else:
        print("  phase grid skipped (need >=2 cycles)")


if __name__ == "__main__":
    main()
