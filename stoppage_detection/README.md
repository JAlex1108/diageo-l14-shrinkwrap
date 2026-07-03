# Stoppage Detection

ROI-based conveyor stoppage detection and stop-event review workflow.

## What It Does

- Downloads or processes camera clips for a configured time window.
- Scores motion inside a fixed conveyor ROI.
- Detects low-motion windows as stoppages.
- Saves pre-stop context clips for review.
- Compares detected stops against Harford audit extracts.
- Supports simple manual classification of stop clips.

## Main Scripts And Notebooks

- `motion_detect.py` - main stop detection pipeline.
- `detect_anomalies.py` - motion/ROI detection functions used by the pipeline.
- `roi_coords.py` - helper for manually selecting ROI coordinates.
- `classify_stop_clips.py` and `classify_stop_clips_v2.py` - stop clip review/classification helpers.
- `stoppage_analysis.ipynb` - compares extracted stoppages with Harford records.
- `stoppage_event_detection.ipynb` - exploratory stoppage event workflow.

## Key Inputs

- `data/` - Harford audit extracts.
- ROI tuple in `motion_detect.py`.
- S3 camera videos configured in `motion_detect.py`.

## Key Outputs

- `processed_motion.csv` - per-video stop detection summaries.
- `processed_videos.csv` - processed video tracking.
- `output/` - frame-level motion outputs.
- `stop_clips/` - saved context clips around detected stops.
- `stoppage_analysis_output/` - comparison tables and plots.
