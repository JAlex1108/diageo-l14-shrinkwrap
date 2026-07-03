# Stop Clip Anomaly Detection

Phase-aware anomaly analysis workflow for extracted stop clips.

## What It Does

- Uses reference phase timing from normal machine motion.
- Compares stop clips against selected entry-phase reference frames.
- Scores anomaly likelihood with NCC-based phase-aware comparisons.
- Exports anomalous frames and classification summaries for review.

## Main Notebook

- `phase_aware_stoppage_ncc_anomaly_detection.ipynb`

## Key Inputs

- `../measurements/reference_clip/` - reference clip used for normal phase timing.
- `../stoppage_detection/stop_clips/` - stop clips to classify or inspect.

## Key Outputs

- `phase_aware_entry_ncc_output/` - anomaly scores, exported frames, and plots.
- `process_flagged/` - working area for flagged/candidate videos.
