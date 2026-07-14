# Diageo L14 Shrink-Wrapper Analysis

Analyses of the L14 shrink-wrap machine at Diageo — the machine that pulls glass bottles
into packs of three and wraps them in film. Two data sources feed everything here:

- **Video** from the side-view camera on the machine (device `cortexvpu-01a-005-41884872`),
  pulled from S3 or from local `.ts` clips.
- **PLC signals** read live off the machine's Siemens S7-300 controller (counters, fault
  bits, speed setpoints).

Each analysis lives in its own top-level folder. The convention is the same everywhere:
**scripts and their fixed inputs (reference images, config JSONs, symbol tables) sit at the
top of the folder; everything a script generates goes into that folder's `output/`
subfolder.** All `output/` folders are gitignored — their contents can be regenerated.

## Layout

```
Diageo_ShrinkWrap/
├── AnomalyDetection/          # score footage for "does this look unusual?"
├── LapbarMeasurements/        # measure bottle positions vs the lap-bar height
├── PhaseExtraction/           # sample frames at a chosen point in the machine cycle
├── PLC_data/                  # log live signals off the machine's PLC
├── ShrinkWrapClassification/  # per-bottle tilt measurement + wrap-quality classes
├── Stoppage_detection/        # find when and why the line stopped
├── docs/tasks/                # per-task engineering records (open/closed, with receipts)
├── diageo_env/                # local Python 3.12 virtual environment (gitignored)
└── requirements.txt           # python-snap7 + the VideoModule library + pins
```

How the analyses feed each other:

```
S3 footage ──> Stoppage_detection (pipeline runs) ──> coverage log of clips that cycled
                                                            │
                              PhaseExtraction  <────────────┘
                              (frames sampled per cycle phase)
                                       │
                              LapbarMeasurements
                              (bottle + lap-bar geometry per frame)

S3 / local clips ──> AnomalyDetection (scored directly)
local clips      ──> ShrinkWrapClassification (its own phase pipeline + tilt dataset)
PLC (live)       ──> PLC_data (rolling CSV log)
```

## The folders

### PhaseExtraction

The machine works in a repeating cycle (bottles enter → get centred → film wraps). These
scripts pull individual frames out of the videos at a chosen point in that cycle. The cycle
position of every frame is found by comparing it against an **anchor image** — a saved frame
of the bottles entering the view — using normalised cross-correlation (a similarity score
between two images); the best match in each cycle is "phase 0" and the rest of the cycle is
split into numbered phases from there. Frames where the line is cycling empty (no bottles)
are dropped by scoring each frame against its phase's typical appearance.

- `export_phase_samples.py` — main exporter: `output/phase0..phase4` (200 frames per phase)
  plus manifests mapping every image back to its source clip, cycle and frame number.
- `export_phase1_to_phase2.py` — 5 frames per cycle spanning the lap-bar-rising moment:
  `output/phase1_to_phase2` (2,240 frames), consumed by LapbarMeasurements.
- `cycle_montage.py` / `score_boundary_montage.py` — the by-eye checks used to pick the
  anchor frame and the empty-belt score floors.

The list of clips worth sampling comes from Stoppage_detection's coverage log (only clips
verified as actually cycling).

### LapbarMeasurements

Measures where the bottles are versus where the lap bar (the bar that folds the film) is,
frame by frame, over the phase 1 → 2 window exported by PhaseExtraction.

- `extract_measurements.py` — applies a colour-and-contour recipe (`Lapbar+Bottles.json`,
  built in the GUIOpenCV tool) to every frame: it must find exactly 3 bottle shapes with
  matching size/angle/spacing, plus the film's top-left corner. Frames that don't produce a
  clean triplet are rejected. Writes a measurements JSON, per-frame debug overlays, and a
  scatter of middle-bottle x vs lap-bar height into `output/`.
- `analyze_measurements.py` — fits a line through that scatter and studies the spread: does
  measurement noise change with bottle position, and does it drift over the session?

### AnomalyDetection

A press-play launcher for the anomaly-scoring pipeline that lives in the 24H_Insights
library. The pipeline locks onto the machine cycle (same anchor-image idea as
PhaseExtraction), then scores each cycle against a cohort of other cycles — "does this
cycle look like its neighbours?" — and exports per-clip diagnostic plots into `output/`.

- `run_anomaly_detection.py` — pick S3 date range or local folder at the top of the file and
  run. All the actual logic is library code; nothing is re-implemented here.
- `phase_anchor_41884872.png` — the anchor image the config points at (kept tracked in git
  because the script needs it).

### ShrinkWrapClassification

The largest analysis: builds a labelled dataset of wrapped packs from raw clips in
`source_videos/`, measures how much each of the three bottles is tilted, and sorts packs
into wrap-quality classes. `failure_mode_examples/` holds reference clips of known failure
types (crinkled wrap, fallen bottle, tilted bottle) from two products.

- `phase_pipeline.py` / `sample_and_append.py` / `build_master_dataset.py` — process every
  clip in `source_videos/` into one global dataset under `output/` (per-cycle frames +
  `measurements.json`); resumable, skips clips already present.
- `measure_tilt.py`, `measure_phase1_tilt.py`, `remeasure_phaseB_geom.py` — per-bottle tilt
  angles at different cycle phases, measured from bottle contours.
- `tilt_histograms.py`, `sort_overlays_by_class.py`, `compose_pairs.py`,
  `extract_backstrip_pairs.py` — analysis and review artefacts: tilt distributions split by
  bottle position and class, overlays sorted into class folders, before/after image pairs.
- `ref.png` and `StraightStrip.json` — the phase anchor and region-of-interest config for
  this folder's own pipeline.

### Stoppage_detection

Finds line stops in the footage: moments where the machine's movement dies out, whether
that happened on film or between recordings (this camera only records while the scene
moves, so minutes of silence usually mean the line stood still).

- `export_frames_before_gaps.py` — runs the pipeline over an S3 date range and snapshots
  every good→bad transition: a composite image of the frames around it plus the motion
  trace, and a short review video. Produced `output/out_diageo` and
  `output/out_normed_diageo`, whose coverage logs also drive PhaseExtraction.
- `find_stops_and_spikes.py` — post-processes those run trees without re-running anything:
  ranks likely stoppages, exports the strongest anomaly frames, and can download the top
  candidates to verify each stop with the tested stop-detection code in the library
  (verdict per clip: stop on film / stop at the recording cut / line already down).

See this folder's own README for what each `output/out_*` tree contains.

### PLC_data

Logs live signals straight off the machine's PLC over Ethernet, and keeps the machine
builder's signal reference (the symbol table) alongside.

- `plc_comms.py` — the logger: reads the exported symbol table CSV to learn every named
  signal's memory address and type, connects to the PLC, bulk-reads each memory area once
  per second, and appends a CSV row only when a value changed. Rolling storage in `output/`
  with a size cap. Captured sessions `plc_00000167.csv`… live there.
- `plc_watch.py` — diagnostic: watches the PLC's data blocks for a while and reports which
  values actually move, to find signals the logger isn't capturing yet.
- `L14_shrinkwrapper_symbol_table_EN.csv` (+ source PDF), `L14_subset.csv`,
  `header_map.csv` — the signal reference, a hand-picked shortlist, and friendly column
  renames.

This folder's README walks through the logger's decision rules and memory-area layout in
detail.

## Setup

Everything runs from the local `diageo_env` virtual environment (Python 3.12):

```powershell
diageo_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

Two external dependencies to know about:

- **24H_Insights** — the VideoModule CV library. It is installed from git by
  `requirements.txt`, but several scripts here (PhaseExtraction, Stoppage_detection) instead
  import it straight from a local clone at `C:\Users\jkind\Documents\McLaren\24H_Insights`
  via a hardcoded path at the top of each script. If that clone lives elsewhere, edit the
  `REPO = ...` line.
- **AWS access** — S3 runs need the SSO profile named in each script
  (`522196013725_DashcamGlbDiageoProdDataContrib`).

## Task tracking

`docs/tasks/` holds one file per engineering task (open / closed, with verification
receipts). See `docs/tasks/README.md` for the convention.
