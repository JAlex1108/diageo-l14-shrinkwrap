#!/usr/bin/env python3
"""Phase grid: VideoModule phase awareness on the source video, ref anchored as phase 0.

Full frame (no ROI), 20 phases. Each cell is titled C{cycle}P{phase}.
Run with the 24H_Insights venv so fCWT is used:
    24h_env/Scripts/python.exe phase_grid.py
"""
import sys
import cv2

sys.path.insert(0, r"C:/Users/jkind/Documents/McLaren/24H_Insights")
from VideoModule.video_io import read_video
from VideoModule.phase_detection import compute_energy_signal, detect_dynamic_phases
from VideoModule.plotting.phase_plots import plot_phase_alignment

VIDEO = "source_videos/cortexvpu-01a-005-41884872_2026-06-20_10-18-18_947201.ts"
REF = "ref.png"
NUM_PHASES = 30
NUM_CYCLES = 8

# Load frames (downscaled for the signal; full frame). ref.png is upside down vs the camera, so
# rotate the ref 180 deg before matching -> phase 0 anchors to the reference configuration.
frames, fps = read_video(VIDEO, resize=(480, 250))
ref = cv2.rotate(cv2.imread(REF), cv2.ROTATE_180)

energy, _ = compute_energy_signal(frames, energy_method="ncc_ref", reference_frame=ref)
dyn = detect_dynamic_phases(energy, fps, min_hz=0.2, max_hz=1.0,
                            min_region_seconds=4.0, min_freq_change_hz=0.15,
                            use_fcwt=True, gate=True)
print(f"backend={dyn.method}  cycles={len(dyn.cycles)}")

plot_phase_alignment(frames, dyn.cycles, num_phases=NUM_PHASES, num_cycles=NUM_CYCLES,
                     title=f"Phase grid ({NUM_PHASES} phases, ref = phase 0)", save_path="phase_grid.png")
print("saved phase_grid.png")
