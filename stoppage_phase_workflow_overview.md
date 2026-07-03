# Stoppage And Phase-Aware Workflow Overview

This note explains what the current scripts and notebooks do in:

- `measurements/`
- `stop_clip_anomaly_detection/`
- `stoppage_detection/`
- `video_coverage_log/`

It also explains how they connect, and which parts are older exploratory work versus the path that looks like it is replacing them.

## Short Version

The workflows split into two layers:

- `stoppage_detection/` is the raw stop-finding layer. It looks at conveyor motion in a fixed ROI, finds likely stoppages, and saves short context clips in `stop_clips/`.
- `measurements/`, `video_coverage_log/`, and `stop_clip_anomaly_detection/` are the newer phase-aware analysis layer. They all use a normal `reference_clip` as the baseline for expected machine behaviour.

In practice:

- Use `stoppage_detection/` to extract candidate stop events from raw camera footage.
- Use `video_coverage_log/` to answer whether footage exists and whether it contains valid cyclic machine motion.
- Use `stop_clip_anomaly_detection/` to rank or classify extracted stop clips by how abnormal they look relative to normal entry-phase behaviour.
- Use `measurements/` when the goal is geometric measurement on aligned stopped frames, not stop detection.

## How The Four Areas Relate

The current dependency flow is:

1. `stoppage_detection/motion_detect.py` reads raw `.ts` clips and creates `stoppage_detection/stop_clips/`.
2. `measurements/reference_clip/` provides the normal running baseline used by the phase-aware workflows.
3. `video_coverage_log/` uses that reference baseline to decide whether new footage has normal cyclic motion, gaps, or camera-off periods.
4. `stop_clip_anomaly_detection/` uses that same reference baseline plus `stoppage_detection/stop_clips/` to score stopped clips for abnormal entry behaviour.
5. `measurements/` can also use `stoppage_detection/stop_clips/` when exporting measurement frames from stoppage clips instead of from the reference clip.

So the shared pattern is:

- `stoppage_detection/` produces stop clips.
- `measurements/reference_clip/` produces the normal-phase baseline.
- The other folders consume one or both of those assets.

## `stoppage_detection/`

This is the oldest and most operationally direct workflow. It does raw motion-based stop extraction.

### Main files

- `motion_detect.py`
- `detect_anomalies.py`
- `roi_coords.py`
- `classify_stop_clips.py`
- `classify_stop_clips_v2.py`
- `stoppage_analysis.ipynb`
- `stoppage_event_detection.ipynb`

### What it does

`motion_detect.py` is the main entrypoint. It:

- lists raw camera videos from S3
- filters them by filename timestamp
- downloads each clip locally
- scores motion inside one fixed conveyor ROI
- detects low-motion windows as stoppages
- writes one summary row per processed video
- exports a short `before_stop.mp4` context clip for detected stops

`detect_anomalies.py` is the helper module behind that pipeline. The important point is that it contains both:

- the current ROI stop detector used by `motion_detect.py` via `detect_motion_stop_in_roi(...)`
- a large amount of older generic motion, mask, and colour-detection code kept from earlier experiments

`roi_coords.py` is a small manual utility to pick the conveyor ROI on a sample frame.

`stoppage_analysis.ipynb` compares extracted stops against the Harford audit extracts. It is the validation notebook for "did our stop extraction line up with the external stop record?"

`stoppage_event_detection.ipynb` looks like an older exploratory notebook for stoppage-event work rather than the current scripted pipeline.

`classify_stop_clips.py` tries to auto-label stop clips with simple contour heuristics such as:

- `fallen_on_entry`
- `fell_while_visible`
- `possible_multiple_bottle_jam`
- `unknown_stop`

`classify_stop_clips_v2.py` is not really a better automatic classifier. It is more of a review-prep tool: it builds a contact-sheet style review image from the first, middle, and last frame of each stop clip and writes a manifest for manual labelling.

### Role in the wider workflow

This folder is still the producer of `stop_clips/`, which downstream work depends on. That means it is not obsolete overall.

What is becoming older inside this folder is the style of analysis:

- raw contour heuristics
- ad hoc classification helpers
- exploratory mask/colour functions

The stop extraction step itself is still foundational.

## `video_coverage_log/`

This is a newer phase-aware logging workflow. Its job is not to detect a stoppage inside a clip, but to answer whether video coverage exists and whether the clip contains valid cyclic machine motion.

### Main file

- `phase_aware_motion_cv_log.ipynb`

### What it does

The notebook:

- uses `measurements/reference_clip/` to establish the expected cycle profile
- runs `VideoModule` phase awareness on each local or S3 video
- records whether data is present
- records clip start and end times based on filename timestamps
- marks whether valid cycles were found
- splits outputs into typical motion, stoppages, missing-video rows, and camera-off-after-stop gaps

This makes it a coverage and quality screen for footage, not a stop extractor.

### Role in the wider workflow

This folder sits upstream of detailed analysis. It helps answer:

- do we have clips for this time window?
- are they normal running clips?
- are they non-cyclic / stoppage clips?
- did the camera likely stop after the line stopped?

### What it appears to supersede

It appears to supersede using raw stop-detection outputs to answer coverage questions.

If the question is:

- "Is there footage?"
- "Is the footage cyclic and usable?"
- "Are there gaps or camera-off periods?"

then `video_coverage_log/` is the newer tool to use, not `stoppage_detection/`.

## `stop_clip_anomaly_detection/`

This is a newer phase-aware stop-clip analysis workflow. It works on clips that have already been extracted as stops.

### Main file

- `phase_aware_stoppage_ncc_anomaly_detection.ipynb`

### What it does

The notebook:

- uses `measurements/reference_clip/` as the normal machine baseline
- loads clips from `stoppage_detection/stop_clips/`
- detects or aligns expected phases from the reference clip
- compares stop-clip entry-phase frames against the normal reference appearance
- scores anomaly likelihood with NCC / phase-grid comparisons
- exports anomalous frames, frame scores, and summary tables

This is much more phase-aware than the contour heuristics in `classify_stop_clips.py`.

### Role in the wider workflow

This folder is the review and prioritisation layer for stop clips. It does not replace stop extraction. It assumes the stop clips already exist.

### What it appears to supersede

It appears to supersede `stoppage_detection/classify_stop_clips.py` as the main way to classify or rank stop clips.

Reason:

- `classify_stop_clips.py` uses simple contour shape heuristics
- `stop_clip_anomaly_detection/` uses the shared normal-phase baseline and explicit phase-aware comparisons

`classify_stop_clips_v2.py` still has a use as a lightweight manual review helper, but not as the main analytical classifier.

## `measurements/`

This is the phase-aware geometric measurement workflow.

### Main file

- `phase_aware_lapbar_bottle_position_measurement.ipynb`

### What it does

The notebook:

- establishes repeatable machine phases from a normal `reference_clip`
- aligns selected phase frames to a reference image
- applies Pitwall configs for lap bar, bottle, and datum regions
- exports candidate phase frames and review overlays
- writes measurement manifests and measurement CSVs
- supports batch output for the Chapter 8 workflow

It can export from either:

- `reference_clip`
- `stoppage_detection/stop_clips/`

So this notebook can analyse stopped clips, but its purpose is measurement, not stop detection.

### Role in the wider workflow

This folder provides two things to the rest of the repo:

- the `reference_clip/` baseline used by the phase-aware workflows
- the actual bottle/lap-bar measurement outputs

### What it appears to supersede

Within this four-folder set, `measurements/` does not directly supersede `stoppage_detection/` or `video_coverage_log/`.

Instead, it replaces a more manual frame-picking and measurement process with a phase-aware aligned measurement workflow.

Its main relationship to the others is:

- upstream baseline provider for `video_coverage_log/` and `stop_clip_anomaly_detection/`
- downstream consumer of `stoppage_detection/stop_clips/` when the goal is to measure stopped-state geometry

## Recommended Current Mental Model

If the question is "which workflow should I use?", the practical answer is:

- Use `stoppage_detection/motion_detect.py` to create candidate stop clips from raw footage.
- Use `video_coverage_log/phase_aware_motion_cv_log.ipynb` to understand footage availability and whether clips contain usable cyclic motion.
- Use `stop_clip_anomaly_detection/phase_aware_stoppage_ncc_anomaly_detection.ipynb` to investigate which stop clips look abnormal relative to normal entry behaviour.
- Use `measurements/phase_aware_lapbar_bottle_position_measurement.ipynb` when you need numeric bottle/lap-bar/datum measurements on aligned phase frames.

## What Is Being Superseded

This is the clearest reading of the current state of the repo.

### Still current

- `stoppage_detection/motion_detect.py` as the raw stop-clip producer
- `video_coverage_log/phase_aware_motion_cv_log.ipynb` as the coverage and cyclic-motion screen
- `stop_clip_anomaly_detection/phase_aware_stoppage_ncc_anomaly_detection.ipynb` as the richer stop-clip anomaly workflow
- `measurements/phase_aware_lapbar_bottle_position_measurement.ipynb` as the measurement workflow

### Partly legacy or being replaced

- older helper functions in `stoppage_detection/detect_anomalies.py` that are unrelated to `detect_motion_stop_in_roi(...)`
- the inactive old mask-based block at the bottom of `stoppage_detection/motion_detect.py`
- `stoppage_detection/classify_stop_clips.py` as the main stop-clip classifier
- `stoppage_detection/stoppage_event_detection.ipynb` as the main path for stoppage work

### Likely replacement direction

- stoppage_detection = find and validate stops -> video_coverage_log
- stop_clip_anomaly_detection = inspect / classify abnormal motion patterns around those stops -> future CNN classifier
- measurements = replace manual stopped-frame measurement selection with a phase-aware aligned measurement workflow
- video_coverage_log = replace older ad hoc coverage and stop-validation checks with a phase-aware coverage / cyclic-motion screen

### Inline summary

To be superseded:

- stoppage_detection = find and validate stops -> video_coverage_log
- stop_clip_anomaly_detection = inspect / classify abnormal motion patterns around those stops -> future CNN classifier
- measurements = phase-aware aligned lap-bar / bottle / datum measurement workflow -> current measurement path, not obviously queued to be replaced by another folder here
- video_coverage_log = phase-aware coverage, cyclic-motion, and stoppage screening workflow -> likely the replacement for older stop-validation / coverage-check work

## One-Sentence Summary

`stoppage_detection/` finds stops, `video_coverage_log/` tells you whether footage is present and cyclic, `stop_clip_anomaly_detection/` tells you which extracted stops look abnormal, and `measurements/` turns selected aligned frames into numeric bottle/lap-bar measurements.







