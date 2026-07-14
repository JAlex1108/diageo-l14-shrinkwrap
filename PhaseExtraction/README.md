# Phase extraction — sampling frames by machine-cycle position

Tools for pulling frames out of the side-view shrink-wrapper videos (camera 41884872) at a
chosen point in the machine cycle. Built 2026-07-13 in the 24H_Insights session that produced
`output/` (formerly the repo-root `exported_samples/`); they run against the 24H_Insights
library (hardcoded path inside each script) but are Diageo-specific, so they live here.
The list of clips worth sampling comes from the stoppage analysis's pooled coverage log
(`../Stoppage_detection/output/out_normed_diageo/pooled/runs/2026-07-06/`).

## The anchor images (essential)

Phase 0 is defined by an anchor image: every video frame is compared against it (normalised
cross-correlation), and the moment that best matches the anchor is phase 0 of each cycle.

- `phase_anchor_41884872_bottles_entering.png` — the CURRENT anchor: a real frame of the
  bottle pack entering the camera view (frame 182 of clip 2026-07-06_09-54-22). Phase 0 =
  bottles entering; phases 0-4 then cover entering -> centred -> film wrapping.
- `phase_anchor_41884872_original_backup.png` — the OLD anchor it replaced (bottles near the
  exit). Kept so earlier results (anything phase-numbered before 2026-07-13, including
  `../AnomalyDetection/phase_anchor_41884872.png`, which is this same old image) can be
  interpreted.

The live copy the 24H pipeline actually reads sits at the 24H_Insights repo root as
`phase_anchor_41884872.png` (currently = the bottles-entering image). If you change the
anchor, change it there and keep a named copy here.

## Scripts

- `export_phase_samples.py` — the main exporter. Samples verified-cycling clips evenly across
  a day (S3, clips cached in `output/_ts_cache`), assigns each frame a phase with
  the anchor, and exports N full-resolution frames per phase, evenly spread over the day.
  Frames without bottles (line cycling empty) are dropped by a similarity check: each frame is
  scored against its phase's median appearance using only the pixels that vary between cycles
  (the bottle/film region — whole-frame scores are blinded by the static machinery), with an
  automatic per-phase pass floor. Produced `output/phase0..phase4` (1000 frames,
  200/phase) + `manifest.json` / `manifest.csv` (image name -> source video, cycle, phase,
  frame number, time in clip, similarity score).
- `export_phase1_to_phase2.py` — per cycle, 5 evenly spaced frames spanning phase 1 -> phase 2
  (the lap bar rising). Produced `output/phase1_to_phase2` (2,240 frames, 448
  cycles) with its own manifests. Reuses the main exporter's cached pass-1 data
  (`output/_pass1_cache.npz`), so it needs no re-download.
- `cycle_montage.py` — 12 labelled frames across one machine cycle; how the bottles-entering
  anchor frame was chosen.
- `score_boundary_montage.py` — candidate crops ranked by similarity score per phase; how the
  empty-belt pass floors were verified by eye.

The lap-bar measurement work that consumes `phase1_to_phase2` lives in
`../LapbarMeasurements`.
