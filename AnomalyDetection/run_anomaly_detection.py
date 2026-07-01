"""Diageo L14 shrink-wrapper — anomaly detection launcher.

This is just the VideoModule pipeline's own press-play entry point, driven from this repo:
pick a ``RunProfile`` (folder OR S3 + config + out dir + clip cap) and call ``run_profile`` —
exactly the pattern the library uses in its own module. All the work (ncc_ref phase-lock,
top-k cohort scoring, per-phase normalisation, pooled cohorts, and the full per-clip diagnostic
exports) lives in the maintained ``CORTEX_SIDEVIEW_CONFIG`` preset and ``run_profile``; nothing
is re-implemented here.

Run it:

    python AnomalyDetection/run_anomaly_detection.py

Then edit the two knobs below: ``RUN`` (S3 vs local) and, in ``CONFIG``, the pooled/exports lines.
"""
from pathlib import Path

from VideoModule.pipelines.anomaly_detection import run_profile, RunProfile, PooledCohortConfig
# The tuned preset + the S3 source (bucket / prefix / AWS SSO profile) for this camera are
# module-level, not re-exported from the package __init__.
from VideoModule.pipelines.anomaly_detection.video_anomaly_detection_pipeline import (
    CORTEX_SIDEVIEW_CONFIG,
    CORTEX_SIDEVIEW,
)

HERE = Path(__file__).resolve().parent
ANCHOR = HERE / "phase_anchor_41884872.png"
OUT = HERE / "anomaly_out"
LOCAL_FOLDER = HERE.parent / "ShrinkWrapClassificationTiltMeasurement" / "source_videos"

# The maintained preset, with the ONLY override the pip install forces: CORTEX_SIDEVIEW_CONFIG
# bakes the anchor path into the 24H repo root, which doesn't exist inside the installed package —
# repoint reference_image at the local copy. The other two edits are ordinary config knobs:
#   - pooled_cohort: windowed cross-video cohort (delete the line for plain self-cohort),
#   - export_*: turn the per-clip diagnostic plots on (the preset disables them for bulk scans).
CONFIG = CORTEX_SIDEVIEW_CONFIG.replace(
    reference_image=str(ANCHOR),
    pooled_cohort=PooledCohortConfig(target_population=150),
    export_overview=True, export_phase_grid=True, export_phase_overview=True,
)

# Two ready RunProfiles — pick one with RUN below (mirrors the library's own RUN block).
S3_RUN = RunProfile(
    config=CONFIG, s3=CORTEX_SIDEVIEW,
    start="2026-06-20_05-58-00", end="2026-06-20_07-00-00",   # "YYYY-MM-DD[_HH-MM-SS]", end None = up to now
    out=str(OUT), max_clips=3,                               # max_clips=None = every clip in range
)
LOCAL_RUN = RunProfile(
    config=CONFIG, folder=str(LOCAL_FOLDER),
    out=str(OUT), max_clips=3,
)

RUN: RunProfile = S3_RUN     # <- switch to LOCAL_RUN to scan the on-disk folder instead

if __name__ == "__main__":
    raise SystemExit(run_profile(RUN))
