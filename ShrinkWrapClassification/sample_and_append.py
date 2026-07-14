#!/usr/bin/env python3
"""Sample one clip per 10 minutes from Saturday 20th (05:30+), process, and append to the global
dataset until phase_pipeline.TARGET_IMAGES. Incremental: download one sampled clip, process it
(3-strip only via the pipeline), append, rewrite the global files, then move to the next bucket.
Resumable via the dataset (skips clips already present). Source clips are kept in source_videos/.

Run with the 24H_Insights venv:
    24h_env/Scripts/python.exe sample_and_append.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, r"C:/Users/jkind/Documents/McLaren/24H_Insights")
sys.path.insert(0, str(HERE))
import phase_pipeline as pp
from VideoModule.parallel_io.s3_clip_source import list_clips_in_range, parse_time_bound
from VideoModule.parallel_io.async_downloader import AsyncS3Downloader, DownloadTask

BUCKET = "diageo-prod-global-dashcam-mc-nuc-video"
PREFIX = "cortexvpu-01a-005-41884872/"
PROFILE = "522196013725_DashcamGlbDiageoProdDataContrib"
SRC = HERE / "source_videos"
INTERVAL_MIN = 10
START = parse_time_bound("2026-06-20_05-30-00")     # Saturday early morning
END = parse_time_bound("2026-06-20", end=True)      # rest of Saturday


def main() -> None:
    pp._ensure_dirs()
    SRC.mkdir(exist_ok=True)
    records, done = pp.load_existing_records()
    print(f"start: {len(records)} records in dataset; target {pp.TARGET_IMAGES}", flush=True)

    refs = list_clips_in_range(START, END, bucket=BUCKET, prefix=PREFIX, profile=PROFILE)
    picked, seen = [], set()
    for r in sorted(refs, key=lambda x: x.timestamp):
        b = int((r.timestamp - START).total_seconds() // (INTERVAL_MIN * 60))
        if b not in seen:
            seen.add(b)
            picked.append(r)
    print(f"{len(refs)} Saturday clips -> {len(picked)} sampled (1 per {INTERVAL_MIN} min)", flush=True)

    dl = AsyncS3Downloader(max_workers=1, show_progress=False, aws_profile=PROFILE)
    for r in picked:
        if len(records) >= pp.TARGET_IMAGES:
            print("reached target.", flush=True)
            break
        if pp.clip_tag(r.stem) in done:
            continue
        dest = SRC / f"{r.stem}.ts"
        if not (dest.exists() and dest.stat().st_size > 0):
            res = dl.download_batch([DownloadTask(bucket=r.bucket, key=r.key, local_path=dest)])
            if not res or not res[0].success:
                print(f"{r.timestamp}: download failed", flush=True)
                continue
        try:
            recs = pp.process_video(dest)
            records.extend(recs)
            done.add(pp.clip_tag(r.stem))
            pp.write_global(records, pp.OUT)
            print(f"{r.timestamp}: +{len(recs)} -> {len(records)}/{pp.TARGET_IMAGES}", flush=True)
        except Exception as e:
            print(f"{r.timestamp}: FAILED {type(e).__name__}: {e}", flush=True)

    print(f"\nDONE: {len(records)} records in global dataset", flush=True)


if __name__ == "__main__":
    main()
