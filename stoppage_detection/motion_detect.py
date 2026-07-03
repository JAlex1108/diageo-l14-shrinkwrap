import csv
from datetime import datetime
from pathlib import Path
import re
import subprocess

import boto3
import pandas as pd

import detect_anomalies as da


# S3 source for raw camera videos.
AWS_PROFILE = "DashcamGlbDiageoProdDataContrib-522196013725"
BUCKET = "diageo-prod-global-dashcam-mc-nuc-video"
PREFIX = "cortexvpu-01a-005-41884872/"
VIDEO_EXTENSIONS = [".ts"]

# Conveyor region to score, as (x, y, width, height).
ROI = (567, 332, 247, 243)

# Optional filename timestamp range. Use None to disable either side.
START_TIME = "2026-06-19 01:30:00"
END_TIME = "2026-06-20 23:59:00"

# Motion scoring settings.
EVERY_XTH_FRAME = 5
MIN_AREA = 50  # Ignore tiny moving blobs smaller than this many pixels.
MAX_AREA = 1000000  # Ignore huge blobs bigger than this many pixels.
TAIL_SECONDS = 1
MIN_STOP_SECONDS = 1
STOP_MOTION_RATIO = 0.1  # Stopped if motion appears in <=10% of sampled stop-window frames.

# Runtime controls.
MAX_VIDEOS_TO_PROCESS = None  # Keep small while tuning. Set to None for all videos
UPLOAD_TO_S3 = False

SAVE_STOP_CONTEXT_CLIPS = True
STOP_CONTEXT_SECONDS = 5

# Set this to a single S3 filename when tuning ROI/thresholds on one clip.
# Test mode keeps the frame CSV and does not update processed tracking files.
TEST_VIDEO_FILENAME = None # TEST_VIDEO_FILENAME = "cortexvpu-01a-005-41884872_2026-06-20_09-28-02_923245.ts" -- stop in this video.


def main():
    # Keep generated files next to this script so the paths are stable
    # regardless of the terminal working directory.
    script_dir = Path(__file__).resolve().parent
    videos_dir = script_dir / "videos_for_processing"
    output_dir = script_dir / "output"
    stop_clips_dir = script_dir / "stop_clips"
    processed_motion_csv_path = script_dir / "processed_motion.csv"
    processed_videos_csv_path = script_dir / "processed_videos.csv"

    videos_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    stop_clips_dir.mkdir(parents=True, exist_ok=True)

    start_time = pd.to_datetime(START_TIME) if START_TIME else None
    end_time = pd.to_datetime(END_TIME) if END_TIME else None
    filename_timestamp_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d+)")
    test_mode = TEST_VIDEO_FILENAME is not None

    # Use the configured AWS profile so this works with SSO credentials.
    session = boto3.Session(profile_name=AWS_PROFILE) if AWS_PROFILE else boto3.Session()
    s3 = session.client("s3")

    candidate_videos = []
    raw_video_count = 0
    timestamp_match_count = 0

    if test_mode:
        # Test mode bypasses listing/date/processed filters and runs one named clip.
        candidate_videos.append({
            "filename": TEST_VIDEO_FILENAME,
            "timestamp": pd.NaT,
        })
        raw_video_count = 1
        timestamp_match_count = 1
    else:
        # List raw videos directly under the camera prefix and apply the date filter.
        #
        # The same S3 prefix can also contain generated outputs, such as
        # processed_motion_frame_data/*.csv. We only want original camera clips here.
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
            for s3_object in page.get("Contents", []):
                s3_key = s3_object["Key"]
                filename = Path(s3_key).name

                # Skip anything in nested output folders under the same prefix.
                # Example kept:  cortexvpu-.../clip.ts
                # Example skipped: cortexvpu-.../processed_motion_frame_data/file.csv
                if s3_key != f"{PREFIX}{filename}":
                    continue

                # Only raw video files should enter the processing queue.
                if Path(s3_key).suffix.lower() not in VIDEO_EXTENSIONS:
                    continue
                raw_video_count += 1

                # The clip timestamp is encoded in the filename, not taken from
                # LastModified. LastModified can change if files are copied or re-uploaded.
                timestamp_match = filename_timestamp_pattern.search(filename)
                if not timestamp_match:
                    continue
                timestamp_match_count += 1

                video_timestamp = pd.to_datetime(
                    timestamp_match.group(1),
                    format="%Y-%m-%d_%H-%M-%S_%f",
                )
                if start_time is not None and video_timestamp < start_time:
                    continue
                if end_time is not None and video_timestamp > end_time:
                    continue

                # Store the filename plus parsed timestamp so we can process in time order.
                # This also makes START_TIME / END_TIME behavior easy to inspect.
                candidate_videos.append({
                    "filename": filename,
                    "timestamp": video_timestamp,
                })

    video_files = pd.DataFrame(candidate_videos, columns=["filename", "timestamp"])
    video_files = video_files.sort_values("timestamp").reset_index(drop=True)

    # Skip videos already completed in earlier runs.
    #
    # This prevents the script from reprocessing the same clip every time it runs.
    # If you want to rerun a clip, remove its row from processed_videos.csv.
    if processed_videos_csv_path.exists():
        processed_videos_df = pd.read_csv(processed_videos_csv_path)
        processed_filenames = set(processed_videos_df["filename"].dropna().astype(str))
    else:
        processed_filenames = set()

    if test_mode:
        videos_to_process = video_files.copy()
    else:
        videos_to_process = video_files[~video_files["filename"].isin(processed_filenames)].copy()

    if MAX_VIDEOS_TO_PROCESS is not None and not test_mode:
        # Keep tuning runs short. Set MAX_VIDEOS_TO_PROCESS to None for a full backfill.
        videos_to_process = videos_to_process.head(MAX_VIDEOS_TO_PROCESS)

    processed_videos_df = pd.DataFrame(
        sorted(processed_filenames),
        columns=["filename"],
    )

    print("Run settings")
    print(f"  Bucket: s3://{BUCKET}/{PREFIX}")
    print(f"  Date range: {start_time} to {end_time}")
    print(f"  ROI: {ROI}")
    print(f"  Test video: {TEST_VIDEO_FILENAME}")
    print(f"  Max videos this run: {MAX_VIDEOS_TO_PROCESS}")
    print(f"  Upload to S3: {UPLOAD_TO_S3}")
    print("Video counts")
    print(f"  Raw videos before date filter: {raw_video_count}")
    print(f"  Videos with parsed timestamps: {timestamp_match_count}")
    print(f"  Raw videos in date range: {len(video_files)}")
    print(f"  Already processed: {len(processed_filenames)}")
    print(f"  Videos to process this run: {len(videos_to_process)}")
    if not video_files.empty:
        print(f"  First video in date range: {video_files['timestamp'].min()}")
        print(f"  Last video in date range: {video_files['timestamp'].max()}")

    # Process each remaining video independently so failures and cleanup are isolated.
    for video_index, video_row in enumerate(videos_to_process.itertuples(index=False), start=1):
        filename = video_row.filename
        local_video_path = videos_dir / filename
        frame_output_csv_path = output_dir / f"{Path(filename).stem}_motion_detection_output.csv"

        print("----------------------------------------------------")
        print(f"Processing video {video_index}/{len(videos_to_process)}")
        print(f"  Filename: {filename}")
        print(f"  Downloading to: {local_video_path}")

        s3.download_file(BUCKET, f"{PREFIX}{filename}", str(local_video_path))

        try:
            # Score only the conveyor ROI and summarize motion in the final tail window.
            #
            # The detector writes a per-frame CSV, then returns a compact summary.
            # The key summary field is tail_motion_ratio:
            #   0.0 means no sampled tail frames had motion.
            #   1.0 means every sampled tail frame had motion.
            # stop_detected is 1 when a low-motion run is found anywhere in the clip.
            print("  Running ROI motion-stop detection")
            _, summary = da.detect_motion_stop_in_roi(
                video_file_path=str(local_video_path),
                roi=ROI,
                output_file_path=str(frame_output_csv_path),
                every_xth_frame=EVERY_XTH_FRAME,
                min_area=MIN_AREA,
                max_area=MAX_AREA,
                tail_seconds=TAIL_SECONDS,
                min_stop_seconds=MIN_STOP_SECONDS,
                stop_motion_ratio=STOP_MOTION_RATIO,
                show_frame=False,
            )

            # These values are the main signal for whether the conveyor stopped.
            print("  Detection summary")
            print(f"    Frames scored: {summary['frames_scored']}")
            print(f"    Tail frames scored: {summary['tail_frames_scored']}")
            print(f"    Tail motion ratio: {summary['tail_motion_ratio']}")
            print(f"    Tail avg contours: {summary['tail_avg_contours']}")
            print(f"    Min stop seconds: {summary['min_stop_seconds']}")
            print(f"    Stop start seconds: {summary['stop_start_seconds']}")
            print(f"    Stop end seconds: {summary['stop_end_seconds']}")
            print(f"    Stop duration seconds: {summary['stop_duration_seconds']}")
            print(f"    Stop motion ratio: {summary['stop_motion_ratio']}")
            print(f"    Stop detected: {summary['stop_detected']}")

            stop_context_clip_path = None
            if SAVE_STOP_CONTEXT_CLIPS and summary["stop_detected"] == 1:
                # Save the few seconds before the stop is confirmed so the clip
                # shows what was happening immediately before motion stopped.
                stop_confirmed_seconds = (
                    summary["stop_start_seconds"] + summary["min_stop_seconds"]
                    if summary["stop_start_seconds"] is not None
                    else None
                )
                if stop_confirmed_seconds is not None:
                    clip_start_seconds = max(0, stop_confirmed_seconds - STOP_CONTEXT_SECONDS)
                    clip_duration_seconds = stop_confirmed_seconds - clip_start_seconds
                    stop_context_clip_path = stop_clips_dir / f"{Path(filename).stem}_before_stop.mp4"

                    ffmpeg_cmd = [
                        "ffmpeg",
                        "-y",
                        "-ss", str(clip_start_seconds),
                        "-i", str(local_video_path),
                        "-t", str(clip_duration_seconds),
                        "-c:v", "libx264",
                        "-c:a", "aac",
                        str(stop_context_clip_path),
                    ]
                    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    print(f"    Saved stop context clip: {stop_context_clip_path}")

            summary_row = {
                "video_name": summary["video_name"],
                "roi_x": summary["roi_x"],
                "roi_y": summary["roi_y"],
                "roi_w": summary["roi_w"],
                "roi_h": summary["roi_h"],
                "frames_scored": summary["frames_scored"],
                "tail_frames_scored": summary["tail_frames_scored"],
                "tail_motion_ratio": summary["tail_motion_ratio"],
                "tail_avg_contours": summary["tail_avg_contours"],
                "tail_motion_frames": summary["tail_motion_frames"],
                "min_stop_seconds": summary["min_stop_seconds"],
                "stop_start_seconds": summary["stop_start_seconds"],
                "stop_end_seconds": summary["stop_end_seconds"],
                "stop_duration_seconds": summary["stop_duration_seconds"],
                "stop_motion_ratio": summary["stop_motion_ratio"],
                "stop_detected": summary["stop_detected"],
                "stop_context_clip_path": str(stop_context_clip_path) if stop_context_clip_path else None,
                "processed_at": datetime.now().isoformat(timespec="seconds"),
            }

            if test_mode:
                print("  Test mode: not updating processed CSVs")
            else:
                # Append one compact row per video to the summary CSV.
                should_write_header = (
                    not processed_motion_csv_path.exists()
                    or processed_motion_csv_path.stat().st_size == 0
                )
                with open(processed_motion_csv_path, "a", newline="") as summary_file:
                    writer = csv.DictWriter(summary_file, fieldnames=summary_row.keys())
                    if should_write_header:
                        writer.writeheader()
                    writer.writerow(summary_row)
                print(f"  Appended summary to: {processed_motion_csv_path}")

                # Mark the video complete after the summary has been written.
                # This order avoids losing a result if the script stops mid-video.
                processed_videos_df = pd.concat(
                    [processed_videos_df, pd.DataFrame([{"filename": filename}])],
                    ignore_index=True,
                ).drop_duplicates()
                processed_videos_df.to_csv(processed_videos_csv_path, index=False)
                print(f"  Marked as processed in: {processed_videos_csv_path}")

            # Upload detailed per-frame output only when explicitly enabled.
            if UPLOAD_TO_S3 and not test_mode:
                upload_key = f"{PREFIX}processed_motion_frame_data/{frame_output_csv_path.name}"
                s3.upload_file(str(frame_output_csv_path), BUCKET, upload_key)
                print(f"  Uploaded frame data to: s3://{BUCKET}/{upload_key}")
            elif test_mode:
                print("  Test mode: S3 upload disabled")
            else:
                print("  S3 upload disabled")

        finally:
            # Always remove large local files even if detection or upload fails.
            if local_video_path.exists():
                local_video_path.unlink()
                print(f"  Deleted local video: {local_video_path}")

            if frame_output_csv_path.exists() and not test_mode:
                frame_output_csv_path.unlink()
                print(f"  Deleted local frame CSV: {frame_output_csv_path}")
            elif frame_output_csv_path.exists():
                print(f"  Test mode: kept local frame CSV: {frame_output_csv_path}")

        print("----------------------------------------------------")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Old mask-based workflow kept for reference. This is intentionally inactive.
# ---------------------------------------------------------------------------
# import cv2
# import pandas as pd
# import numpy as np
# import boto3
# from pathlib import Path
# import csv
# from datetime import datetime
#
# import detect_anomalies as da
#
#
# profile = "DashcamGlbDiageoProdDataContrib-522196013725"
# bucket = "diageo-prod-global-dashcam-mc-nuc-video"
# prefix = "cortexvpu-01a-005-41884872/"
# extensions = [".ts"]
# local_dir = ""
# processed_motion_csv_path = Path("processed_motion.csv")
# processed_videos_csv_path = Path("processed_videos.csv")
#
# # Define the conveyor ROI as (x, y, w, h).
# ROI = (567, 332, 247, 243)
#
# s3_session = boto3.Session(profile_name=profile) if profile else boto3.Session()
# s3 = s3_session.client("s3")
# paginator = s3.get_paginator("list_objects_v2")
#
# videos_for_processing_dir = Path("videos_for_processing")
# videos_for_processing_dir.mkdir(parents=True, exist_ok=True)
# Path("output").mkdir(parents=True, exist_ok=True)
# Path("masks").mkdir(parents=True, exist_ok=True)
#
# # Get video filenames from the s3 bucket.
# video_files = []
# for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
#     for obj in page.get("Contents", []):
#         key = obj["Key"]
#         suffix = Path(key).suffix.lower()
#         if suffix in extensions:
#             video_files.append({"filename": Path(key).name})
#             local_path = Path(local_dir) / Path(key).name
#
# video_files = pd.DataFrame(video_files)
#
# if processed_videos_csv_path.exists():
#     already_processed_videos = pd.read_csv(processed_videos_csv_path)
#     already_processed_videos = set(already_processed_videos["filename"].dropna().astype(str))
# else:
#     already_processed_videos = set()
#
# video_files = set(video_files["filename"].dropna().astype(str))
# videos_to_process = video_files - already_processed_videos
# videos_to_process = pd.DataFrame(videos_to_process, columns=["filename"])
# videos_to_process = videos_to_process.sort_values("filename")
#
# already_processed_videos = pd.DataFrame(list(already_processed_videos), columns=["filename"])
#
# start_date = 0
# last_process_motion_upload_date = None
#
# for vid in videos_to_process["filename"]:
#     print("----------------------------------------------------")
#     print(vid)
#     now = datetime.now()
#     today = now.date()
#
#     filename_df = pd.DataFrame([vid], columns=["filename"])
#
#     s3.download_file(bucket, f"{prefix}{vid}", str(videos_for_processing_dir / vid))
#     vid_path = str(videos_for_processing_dir / vid)
#     current_date = vid.split("_")[1]
#
#     cap = cv2.VideoCapture(vid_path)
#     if not cap.isOpened():
#         print("Error: could not open video")
#         continue
#
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     cap.release()
#
#     da.is_line_running_motion(
#         vid_path,
#         80,
#         10000000,
#         "motion.csv",
#         every_xth_frame=5,
#         line_running_thresh=8,
#         show_frame=False,
#     )
#
#     fmf = da.find_mask_frames("motion.csv", count=200)
#     print(fmf)
#
#     if fmf == "line_not_runnung":
#         already_processed_videos = pd.concat([already_processed_videos, filename_df], ignore_index=True).drop_duplicates()
#         already_processed_videos.to_csv(processed_videos_csv_path, index=False)
#
#         if Path(vid_path).exists():
#             Path(vid_path).unlink()
#             print(f"{vid} deleted")
#         else:
#             print(f"{vid} not found for deleting")
#         continue
#
#     if start_date != current_date:
#         start_date = current_date
#         print("making roi mask")
#
#         roi_mask = 255 * np.ones((frame_height, frame_width), dtype="uint8")
#         x, y, w, h = ROI
#         roi_mask[:] = 0
#         roi_mask[y:y + h, x:x + w] = 255
#         cv2.imwrite("masks/motion_mask.png", roi_mask)
#
#     print("detecting motion")
#     da.motion_detection(
#         vid_path,
#         "masks/motion_mask.png",
#         "output",
#         250,
#         1000,
#         line_running_indicator=fmf,
#         line_running_time_series="motion.csv",
#         every_xth_frame=5,
#         show_frame=False,
#         output_video=False,
#         resize=False,
#     )
#
#     already_processed_videos = pd.concat([already_processed_videos, filename_df], ignore_index=True).drop_duplicates()
#     already_processed_videos.to_csv(processed_videos_csv_path, index=False)
#
#     old_file = Path("output/motion_detection_output.csv")
#     new_file = Path(f"output/{Path(vid).stem}_motion_detection_output.csv")
#     old_file.rename(new_file)
#
#     print(new_file)
#
#     motion_detected = pd.read_csv(new_file)
#     motion_detected = motion_detected.iloc[1:]
#
#     total_contours = sum(motion_detected["num_contours"])
#     contours_per_row = total_contours / len(motion_detected) if len(motion_detected) else 0
#     print(total_contours)
#
#     processed_motion = {
#         "video_name": vid,
#         "total_contours": total_contours,
#         "contours_per_row": contours_per_row,
#         "date": current_date,
#         "roi_x": ROI[0],
#         "roi_y": ROI[1],
#         "roi_w": ROI[2],
#         "roi_h": ROI[3],
#     }
#
#     write_header = (
#         not processed_motion_csv_path.exists()
#         or processed_motion_csv_path.stat().st_size == 0
#     )
#
#     if 6 <= total_contours < 200:
#         with open(processed_motion_csv_path, "a", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=processed_motion.keys())
#             if write_header:
#                 writer.writeheader()
#             writer.writerow(processed_motion)
#
#     if Path(vid_path).exists():
#         Path(vid_path).unlink()
#         print(f"{vid} deleted")
#     else:
#         print(f"{vid} not found for deleting")
#
#     upload_name = f"{prefix}processed_motion_frame_data/{new_file.name}"
#     s3.upload_file(str(new_file), bucket, upload_name)
#
#     if new_file.exists():
#         new_file.unlink()
#         print(f"{new_file.name} deleted")
#     else:
#         print(f"{new_file.name} not found for deleting")
#
#     if last_process_motion_upload_date != today:
#         motion_upload_name = f"{prefix}processed_motion_results/processed_motion_{today}.csv"
#         if processed_motion_csv_path.exists():
#             s3.upload_file(str(processed_motion_csv_path), bucket, motion_upload_name)
#             print(f"Uploaded processed_motion.csv to s3://{bucket}/{motion_upload_name}")
#             last_process_motion_upload_date = today
#         else:
#             print(f"File not found: {processed_motion_csv_path}")
#
#     print("----------------------------------------------------")
