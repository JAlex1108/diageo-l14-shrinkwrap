# Stoppage detection — finding when and why the line stops

Analysis of side-view shrink-wrapper footage (camera 41884872) for line stops: where the
machine's movement dies out, whether that happened on film or at a recording cut, and what
the footage looked like just before. Moved out of the 24H_Insights repo on 2026-07-13 (the
dev happened there for ease); the scripts still run against the 24H_Insights library by a
hardcoded path. The reusable, tested stop-detection algorithm itself (flat-line and
amplitude-drop detection on a motion signal) is NOT here — it is library code:
`24H_Insights/VideoModule/anomaly_detection/stop_detection.py` (task 239, with 23 tests).

## Scripts

- `export_frames_before_gaps.py` — runs the phase-aware pipeline over an S3 date range and
  snapshots every good->bad transition: a composite image (frames before/after + the motion
  energy trace with the cause marked) and a short review video per transition. Resumable per
  calendar day. Produced the `output/out_diageo` (raw) and `output/out_normed_diageo`
  (pooled/normalised) run trees.
- `find_stops_and_spikes.py` — post-processes finished run trees (no pipeline re-run): ranks
  likely stoppages (mid-file motion death vs recording gaps), exports strong anomaly-score
  spike frames, and with `--verify-stops N` downloads the top candidates and runs the tested
  stop-detection brick over them (verdict per clip: GENUINE / BOUNDARY / ALREADY). Produced
  `output/out_postprocess`.

## Data trees (all under `output/`)

- `output/out_diageo/`, `output/out_normed_diageo/` — pipeline run outputs for
  2026-07-03..07-06 (coverage logs listing every clip: footage present? did it cycle? flagged
  anomalous?). The pooled 2026-07-06 log is also what `../PhaseExtraction` uses as its list
  of verified-cycling clips.
- `output/out_postprocess/` — stop curation results: `curated_stops_0706/` (the 19 verified
  07-06 stops + verdicts, incl. `replay_via_brick.csv`, the task-239 receipt),
  `sigma_spikes.csv`, spike frames, saved transition review videos, known-anomaly traces.
- `output/out_original_1019/` — deep-dive on the confirmed fallen-pack clip (2026-06-20 10:18/10:19):
  the 29-second mid-clip stop that the coverage log could not see, which motivated building
  the stop-detection brick. The clip itself is kept as the brick's test ground truth at
  `24H_Insights/datasets/video/diageo/side_view/`.
