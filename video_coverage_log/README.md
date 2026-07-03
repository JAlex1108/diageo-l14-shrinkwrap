# Video Coverage Log

Phase-aware CV logging workflow for checking whether video data exists and whether it contains valid cyclic machine motion.

## What It Does

- Establishes the expected phase/cycle profile from `../measurements/reference_clip/` before batch processing.
- Probes local or S3 video files and builds a `data_present/start/end/valid_cycles` log using the filename timestamp as the video start time.
- Uses `VideoModule` phase awareness to compare each processed clip with the reference cycle profile.
- Marks gaps after a video ends as `camera_off_after_motion_stop` when they exceed the configured grace period.
- Exports stoppage/non-cyclic subsets, inferred camera-off gaps, timeline charts, and summary CSVs.

## Main Notebook

- `phase_aware_motion_cv_log.ipynb`

## Key Inputs

- `../measurements/reference_clip/` - reference baseline and default local workflow test input.
- Local cyclic-motion folder, optional - only use this when the clips should contain normal phase cycles.
- Diageo S3 source configured in the notebook: `s3://diageo-prod-global-dashcam-mc-nuc-video/cortexvpu-01a-005-41884872/`.
- Optional `expected_start_time` and `expected_end_time` settings in the notebook to create missing-video rows for local runs.

## S3 Behaviour

- Lists clips by filename timestamp in the requested S3 time range.
- Downloads clips in small chunks using `VideoModule.parallel_io.s3_clip_source`.
- Appends one CV-log row per clip and records completed filenames in `s3_processed_videos.csv`.
- Deletes downloaded videos after processing unless `s3_keep_downloads=True`.

## Key Outputs

- `phase_aware_motion_cv_log_output/reference_clip_test/` - reference baseline CSV and any context clips.
- `phase_aware_motion_cv_log_output/` - local CV log CSVs, summary, timeline chart, stoppage context clips, and inferred camera-off gaps.
- `phase_aware_motion_cv_log_output/s3_run/` - S3 CV log, processed tracking CSV, summary, timeline chart, and S3 context clips.

