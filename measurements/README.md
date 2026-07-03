# Measurements

Phase-aware lap bar and bottle position measurement workflow.

## What It Does

- Detects repeatable machine phases from video clips.
- Aligns selected phase frames to a reference image.
- Applies Pitwall mask configs for lap bar, bottle, and datum detection.
- Measures datum-to-lapbar and datum-to-bottle positions.
- Exports review overlays and batch measurement CSVs.

## Main Notebook

- `phase_aware_lapbar_bottle_position_measurement.ipynb`

## Key Inputs

- `reference_clip/` - reference video and anchor image for phase alignment.
- `pitwall_config/` - mask configs for lap bar, bottles, and datum.
- `../stoppage_detection/stop_clips/` - stop clips used when exporting measurement frames.

## Key Outputs

- `phase_aware_position_output/config/` - saved ROI config.
- `phase_aware_position_output/frames/` - exported phase frames.
- `phase_aware_position_output/masks/` - generated Pitwall masks.
- `phase_aware_position_output/measurements/` - measurement CSVs.
- `phase_aware_position_output/overlays/` - visual review overlays.
- `phase_aware_position_output/batch/` - Chapter 8 batch outputs.
