from pathlib import Path

import cv2
import pandas as pd


STOP_CLIPS_DIR = Path(__file__).resolve().parent / "stop_clips"
OUTPUT_CSV = Path(__file__).resolve().parent / "stop_clip_classifications.csv"

VIDEO_EXTENSIONS = [".mp4", ".ts", ".mov", ".mkv", ".avi"]

# Contours smaller than this are usually camera noise or compression artifacts.
MIN_CONTOUR_AREA = 150

# Classifier rules. Keep these simple until you have reviewed enough examples.
ENTRY_EDGE_MARGIN_RATIO = 0.20
FALLEN_ASPECT_RATIO = 1.20
ANGLE_CHANGE_DEGREES = 30
AREA_GROWTH_RATIO = 2.0


def main():
    clip_paths = [
        path for path in sorted(STOP_CLIPS_DIR.iterdir())
        if path.suffix.lower() in VIDEO_EXTENSIONS
    ]

    results = []
    for clip_index, clip_path in enumerate(clip_paths, start=1):
        print("----------------------------------------------------")
        print(f"Classifying clip {clip_index}/{len(clip_paths)}")
        print(f"  Clip: {clip_path.name}")

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            print("  Could not open clip")
            results.append({
                "clip_name": clip_path.name,
                "classification": "could_not_open",
            })
            continue

        frame_features = []
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100,
            varThreshold=5,
            detectShadows=False,
        )

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            foreground_mask = background_subtractor.apply(blurred)
            _, threshold = cv2.threshold(foreground_mask, 128, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            largest_contour = None
            largest_area = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= MIN_CONTOUR_AREA and area > largest_area:
                    largest_contour = contour
                    largest_area = area

            if largest_contour is None:
                frame_features.append({
                    "frame": frame_number,
                    "has_object": 0,
                    "area": 0,
                    "centroid_x": None,
                    "aspect_ratio": None,
                    "angle": None,
                })
                continue

            x, y, width, height = cv2.boundingRect(largest_contour)
            moments = cv2.moments(largest_contour)
            centroid_x = (moments["m10"] / moments["m00"]) if moments["m00"] else x + width / 2
            aspect_ratio = width / height if height else None
            angle = cv2.minAreaRect(largest_contour)[2]

            frame_features.append({
                "frame": frame_number,
                "has_object": 1,
                "area": largest_area,
                "centroid_x": centroid_x,
                "aspect_ratio": aspect_ratio,
                "angle": angle,
            })

        cap.release()

        features_df = pd.DataFrame(frame_features)
        if features_df.empty or features_df["has_object"].sum() == 0:
            classification = "unknown_stop"
            reason = "no_large_contours"
        else:
            object_frames = features_df[features_df["has_object"] == 1].copy()
            first_object = object_frames.iloc[0]
            early_frames = object_frames.head(max(1, len(object_frames) // 3))
            late_frames = object_frames.tail(max(1, len(object_frames) // 3))

            early_aspect_ratio = early_frames["aspect_ratio"].median()
            late_aspect_ratio = late_frames["aspect_ratio"].median()
            early_angle = early_frames["angle"].median()
            late_angle = late_frames["angle"].median()
            early_area = early_frames["area"].median()
            late_area = late_frames["area"].median()

            entry_edge_x = frame_width * ENTRY_EDGE_MARGIN_RATIO
            first_object_near_entry = first_object["centroid_x"] is not None and first_object["centroid_x"] <= entry_edge_x
            first_object_already_fallen = (
                first_object["aspect_ratio"] is not None
                and first_object["aspect_ratio"] >= FALLEN_ASPECT_RATIO
            )
            aspect_ratio_increased = (
                pd.notna(early_aspect_ratio)
                and pd.notna(late_aspect_ratio)
                and late_aspect_ratio >= FALLEN_ASPECT_RATIO
                and late_aspect_ratio > early_aspect_ratio
            )
            angle_changed = (
                pd.notna(early_angle)
                and pd.notna(late_angle)
                and abs(late_angle - early_angle) >= ANGLE_CHANGE_DEGREES
            )
            area_grew = early_area > 0 and late_area / early_area >= AREA_GROWTH_RATIO

            if first_object_near_entry and first_object_already_fallen:
                classification = "fallen_on_entry"
                reason = "first_large_object_near_entry_and_wide"
            elif aspect_ratio_increased or angle_changed:
                classification = "fell_while_visible"
                reason = "shape_or_angle_changed_before_stop"
            elif area_grew:
                classification = "possible_multiple_bottle_jam"
                reason = "large_area_growth"
            else:
                classification = "unknown_stop"
                reason = "no_clear_shape_change"

        result = {
            "clip_name": clip_path.name,
            "classification": classification,
            "reason": reason,
            "frame_count": frame_count,
            "object_frames": int(features_df["has_object"].sum()) if not features_df.empty else 0,
        }
        results.append(result)
        print(f"  Classification: {classification}")
        print(f"  Reason: {reason}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print("----------------------------------------------------")
    print(f"Wrote classifications to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
