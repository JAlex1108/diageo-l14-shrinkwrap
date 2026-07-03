from pathlib import Path

import cv2
import pandas as pd


STOP_CLIPS_DIR = Path(__file__).resolve().parent / "stop_clips"
REVIEW_DIR = Path(__file__).resolve().parent / "stop_clip_review"
OUTPUT_CSV = Path(__file__).resolve().parent / "stop_clip_review_manifest.csv"

VIDEO_EXTENSIONS = [".mp4", ".ts", ".mov", ".mkv", ".avi"]

# Keep this conservative. Small contours are usually noise.
MIN_CONTOUR_AREA = 150


def read_review_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        cap.release()
        return []

    frame_numbers = [
        0,
        max(0, frame_count // 2),
        max(0, frame_count - 1),
    ]

    frames = []
    for frame_number in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()
        if ok:
            frames.append((frame_number, frame))

    cap.release()
    return frames


def draw_contours(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = frame.copy()
    kept_contours = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        kept_contours += 1
        x, y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 0, 255), 2)

    return annotated, kept_contours


def main():
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    clip_paths = [
        path for path in sorted(STOP_CLIPS_DIR.iterdir())
        if path.suffix.lower() in VIDEO_EXTENSIONS
    ]

    rows = []
    for clip_index, clip_path in enumerate(clip_paths, start=1):
        print("----------------------------------------------------")
        print(f"Reviewing clip {clip_index}/{len(clip_paths)}")
        print(f"  Clip: {clip_path.name}")

        review_frames = read_review_frames(clip_path)
        if not review_frames:
            print("  Could not read frames")
            rows.append({
                "clip_name": clip_path.name,
                "review_image": None,
                "status": "could_not_read",
                "manual_label": "",
                "notes": "",
            })
            continue

        annotated_frames = []
        contour_counts = []
        for frame_number, frame in review_frames:
            annotated, contour_count = draw_contours(frame)
            contour_counts.append(contour_count)

            label = f"frame {frame_number}, contours {contour_count}"
            cv2.putText(
                annotated,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            annotated_frames.append(annotated)

        # Put first, middle, and final annotated frames into one review image.
        target_height = 360
        resized_frames = []
        for frame in annotated_frames:
            height, width = frame.shape[:2]
            target_width = int(width * (target_height / height))
            resized_frames.append(cv2.resize(frame, (target_width, target_height)))

        review_image = cv2.hconcat(resized_frames)
        review_image_path = REVIEW_DIR / f"{clip_path.stem}_review.jpg"
        cv2.imwrite(str(review_image_path), review_image)

        rows.append({
            "clip_name": clip_path.name,
            "review_image": str(review_image_path),
            "status": "review_ready",
            "first_frame_contours": contour_counts[0] if len(contour_counts) > 0 else None,
            "middle_frame_contours": contour_counts[1] if len(contour_counts) > 1 else None,
            "final_frame_contours": contour_counts[2] if len(contour_counts) > 2 else None,
            "manual_label": "",
            "notes": "",
        })
        print(f"  Wrote review image: {review_image_path}")

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_CSV, index=False)
    print("----------------------------------------------------")
    print(f"Wrote review manifest: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
