#!/usr/bin/env python3
"""
Auto-generated processing script from qt-pitwall.
Generated: 2026-06-25 11:51:55

Usage:
    python measure_tilt_phase1.py <input_path> [--output <output_dir>]

Input can be:
    - Image file (jpg, png, bmp, tiff)
    - Video file (mp4, avi, mov, mkv)
    - Folder of images

Dependencies:
    - Python 3.8+
    - OpenCV (cv2)
    - NumPy
"""
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np


# === CONFIGURATION (auto-generated from qt-pitwall settings) ===
ROI_CONFIG = {
    'enabled': True,
    'active_rois': [
        {
            'id': 1,
            'name': 'Bottles',
            'enabled': True,
            'mode': 'include',
            'shape': 'rectangle',
            'pipeline': {'use_global': True},
            'x': 596,
            'y': 654,
            'width': 507,
            'height': 117
        }
    ],
    'next_id': 2,
    'mode': 'include',
    'x': 0,
    'y': 0,
    'width': 0,
    'height': 0
}

PROCESSOR_CONFIGS = {
    'hue_detection': {
        'enabled': True,
        'target_hue': 0,
        'hue_range': 10,
        'saturation_min': 0,
        'saturation_max': 149,
        'value_min': 57,
        'value_max': 143,
        'hue_value': 11,
        'hue_tolerance': 31,
        'saturation_value': 61,
        'saturation_tolerance': 88,
        'value_value': 100,
        'value_tolerance': 43,
        'hue_min': 0,
        'hue_max': 42
    },
    'contour_filtering': {
        'enabled': True,
        'min_area': 250,
        'max_area': 210281,
        'min_perimeter': 200,
        'max_perimeter': 10000,
        'min_convexity': 0.0,
        'max_convexity': 1.0,
        'min_circularity': 0.0,
        'max_circularity': 1.0,
        'min_aspect_ratio': 1.5
    },
}

PROCESSOR_ORDER = ['hue_detection', 'contour_filtering']

# === PROCESSOR FUNCTIONS ===
def apply_hue_detection(image: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Apply HSV-based color detection."""
    cfg = PROCESSOR_CONFIGS.get("hue_detection", {})

    # Get parameters (support both old and new naming)
    hue_value = cfg.get("hue_value", cfg.get("target_hue", 0))
    hue_tolerance = cfg.get("hue_tolerance", cfg.get("hue_range", 10))

    # Saturation
    if "saturation_value" in cfg:
        sat_value = cfg["saturation_value"]
        sat_tolerance = cfg.get("saturation_tolerance", 128)
        sat_min = max(0, sat_value - sat_tolerance)
        sat_max = min(255, sat_value + sat_tolerance)
    else:
        sat_min = cfg.get("saturation_min", 0)
        sat_max = cfg.get("saturation_max", 255)

    # Value
    if "value_value" in cfg:
        val_value = cfg["value_value"]
        val_tolerance = cfg.get("value_tolerance", 128)
        val_min = max(0, val_value - val_tolerance)
        val_max = min(255, val_value + val_tolerance)
    else:
        val_min = cfg.get("value_min", 0)
        val_max = cfg.get("value_max", 255)

    # Calculate hue range
    hue_min = max(0, hue_value - hue_tolerance)
    hue_max = min(180, hue_value + hue_tolerance)

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create mask based on HSV range
    if hue_min < 0:
        # Wrap around for hues near 0
        lower1 = np.array([0, sat_min, val_min])
        upper1 = np.array([hue_max, sat_max, val_max])
        lower2 = np.array([180 + hue_min, sat_min, val_min])
        upper2 = np.array([180, sat_max, val_max])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        hue_mask = cv2.bitwise_or(mask1, mask2)
    elif hue_max > 180:
        # Wrap around for hues near 180
        lower1 = np.array([hue_min, sat_min, val_min])
        upper1 = np.array([180, sat_max, val_max])
        lower2 = np.array([0, sat_min, val_min])
        upper2 = np.array([hue_max - 180, sat_max, val_max])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        hue_mask = cv2.bitwise_or(mask1, mask2)
    else:
        lower = np.array([hue_min, sat_min, val_min])
        upper = np.array([hue_max, sat_max, val_max])
        hue_mask = cv2.inRange(hsv, lower, upper)

    # Combine with existing mask
    if mask is not None:
        hue_mask = cv2.bitwise_and(hue_mask, mask)

    return image, hue_mask

def apply_contour_filtering(image: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Filter contours based on geometric properties."""
    cfg = PROCESSOR_CONFIGS.get("contour_filtering", {})
    min_area = cfg.get("min_area", 100)
    max_area = cfg.get("max_area", 100000)
    min_perimeter = cfg.get("min_perimeter", 10)
    max_perimeter = cfg.get("max_perimeter", 10000)
    min_convexity = cfg.get("min_convexity", 0.0)
    max_convexity = cfg.get("max_convexity", 1.0)
    min_circularity = cfg.get("min_circularity", 0.0)
    max_circularity = cfg.get("max_circularity", 1.0)
    min_aspect_ratio = cfg.get("min_aspect_ratio", 0.0)
    max_aspect_ratio = cfg.get("max_aspect_ratio", None)

    if mask is None:
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

    # Find contours (RETR_LIST to match qt-pitwall ContourFilter exactly)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Create output mask
    filtered_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter < min_perimeter or perimeter > max_perimeter:
            continue

        # Convexity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        if convexity < min_convexity or convexity > max_convexity:
            continue

        # Circularity
        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        if circularity < min_circularity or circularity > max_circularity:
            continue

        # Aspect ratio (width / height from bounding rect)
        _, _, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < min_aspect_ratio:
            continue
        if max_aspect_ratio is not None and aspect_ratio > max_aspect_ratio:
            continue

        cv2.drawContours(filtered_mask, [contour], -1, 255, -1)

    return image, filtered_mask

# === ROI FUNCTIONS ===
def create_roi_mask(shape: Tuple[int, int]) -> np.ndarray:
    """Create ROI mask based on configuration.

    Args:
        shape: (height, width) of the target mask

    Returns:
        Binary mask (255 = process, 0 = ignore)
    """
    roi_config = ROI_CONFIG

    if not roi_config or not roi_config.get("enabled", False):
        return np.ones(shape, dtype=np.uint8) * 255

    # Multi-ROI mode
    if "active_rois" in roi_config and isinstance(roi_config["active_rois"], list):
        return _create_multi_roi_mask(shape, roi_config["active_rois"])

    # Single ROI mode (legacy)
    roi_shape = roi_config.get("shape", "rectangle")

    if roi_shape == "freehand":
        # Freehand masks stored as base64
        mask_data = roi_config.get("mask_data")
        if mask_data:
            import base64
            mask_bytes = base64.b64decode(mask_data.encode('utf-8'))
            h = roi_config.get("mask_height", shape[0])
            w = roi_config.get("mask_width", shape[1])
            restored_mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape((h, w))
            mode = roi_config.get("mode", "include")
            if mode == "include":
                return restored_mask.copy()
            else:
                return cv2.bitwise_not(restored_mask)
    else:
        # Rectangle ROI
        x = roi_config.get("x", 0) or 0
        y = roi_config.get("y", 0) or 0
        w = roi_config.get("width", 0) or 0
        h = roi_config.get("height", 0) or 0

        if w > 0 and h > 0:
            mode = roi_config.get("mode", "include")
            if mode == "include":
                mask = np.zeros(shape, dtype=np.uint8)
                mask[y:y+h, x:x+w] = 255
            else:
                mask = np.ones(shape, dtype=np.uint8) * 255
                mask[y:y+h, x:x+w] = 0
            return mask

    return np.ones(shape, dtype=np.uint8) * 255


def _create_multi_roi_mask(shape: Tuple[int, int], rois: List[Dict]) -> np.ndarray:
    """Create combined mask from multiple ROIs.

    Args:
        shape: (height, width) of the target mask
        rois: List of ROI dictionaries from config

    Returns:
        Combined binary mask
    """
    # Separate include and exclude ROIs
    include_rois = [r for r in rois if r.get("enabled", True) and r.get("mode") == "include"]
    exclude_rois = [r for r in rois if r.get("enabled", True) and r.get("mode") == "exclude"]

    # Start with zeros (all excluded)
    if include_rois:
        combined_mask = np.zeros(shape, dtype=np.uint8)
    else:
        # No include ROIs, start with all included
        combined_mask = np.ones(shape, dtype=np.uint8) * 255

    # Apply include ROIs
    for roi in include_rois:
        roi_mask = _create_single_roi_mask(shape, roi)
        combined_mask = cv2.bitwise_or(combined_mask, roi_mask)

    # Apply exclude ROIs (subtract from combined)
    for roi in exclude_rois:
        roi_mask = _create_single_roi_mask(shape, roi)
        combined_mask = cv2.bitwise_and(combined_mask, cv2.bitwise_not(roi_mask))

    return combined_mask


def _create_single_roi_mask(shape: Tuple[int, int], roi: Dict) -> np.ndarray:
    """Create mask for a single ROI.

    Args:
        shape: (height, width) of the target mask
        roi: ROI dictionary from config

    Returns:
        Binary mask for this ROI
    """
    roi_shape = roi.get("shape", "rectangle")
    mask = np.zeros(shape, dtype=np.uint8)

    if roi_shape == "rectangle":
        x = roi.get("x", 0) or 0
        y = roi.get("y", 0) or 0
        w = roi.get("width", 0) or 0
        h = roi.get("height", 0) or 0
        if w > 0 and h > 0:
            mask[y:y+h, x:x+w] = 255

    elif roi_shape == "freehand":
        mask_data = roi.get("mask_data")
        if mask_data:
            import base64
            mask_bytes = base64.b64decode(mask_data.encode('utf-8'))
            h = roi.get("mask_height", shape[0])
            w = roi.get("mask_width", shape[1])
            mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape((h, w))

    return mask

# === MAIN PROCESSING PIPELINE ===
def process_frame(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Process a single frame through the pipeline.

    Args:
        image: Input BGR image

    Returns:
        Tuple of (result_image, mask, contours)
    """
    # Create initial ROI mask
    mask = create_roi_mask(image.shape[:2])

    # Run processors in order
    image, mask = apply_hue_detection(image, mask)
    image, mask = apply_contour_filtering(image, mask)

    # Find final contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on result
    result = image.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    return result, mask, list(contours)

# === SOURCE HANDLING ===
def process_image(path: Path, output_dir: Path) -> Dict:
    """Process a single image file.

    Args:
        path: Path to input image
        output_dir: Directory to save results

    Returns:
        Dictionary with processing results
    """
    image = cv2.imread(str(path))
    if image is None:
        print(f"Error: Could not read image: {path}")
        return {"success": False, "error": f"Could not read image: {path}"}

    result, mask, contours = process_frame(image)

    # Save outputs
    result_path = output_dir / f"{path.stem}_result.png"
    mask_path = output_dir / f"{path.stem}_mask.png"

    cv2.imwrite(str(result_path), result)
    cv2.imwrite(str(mask_path), mask)

    print(f"Processed: {path.name} -> {result_path.name} ({len(contours)} objects)")

    return {
        "success": True,
        "input": str(path),
        "result": str(result_path),
        "mask": str(mask_path),
        "object_count": len(contours)
    }


def process_video(path: Path, output_dir: Path) -> Dict:
    """Process a video file frame by frame.

    Args:
        path: Path to input video
        output_dir: Directory to save results

    Returns:
        Dictionary with processing results
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {path}")
        return {"success": False, "error": f"Could not open video: {path}"}

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output video
    result_path = output_dir / f"{path.stem}_result.mp4"
    mask_path = output_dir / f"{path.stem}_mask.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    result_writer = cv2.VideoWriter(str(result_path), fourcc, fps, (width, height))
    mask_writer = cv2.VideoWriter(str(mask_path), fourcc, fps, (width, height), isColor=False)

    frame_num = 0
    total_objects = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, mask, contours = process_frame(frame)
        result_writer.write(result)
        mask_writer.write(mask)

        total_objects += len(contours)
        frame_num += 1

        if frame_num % 100 == 0:
            print(f"Processed {frame_num}/{total_frames} frames...")

    cap.release()
    result_writer.release()
    mask_writer.release()

    avg_objects = total_objects / frame_num if frame_num > 0 else 0
    print(f"Processed: {path.name} -> {result_path.name} ({frame_num} frames, avg {avg_objects:.1f} objects/frame)")

    return {
        "success": True,
        "input": str(path),
        "result": str(result_path),
        "mask": str(mask_path),
        "frames": frame_num,
        "avg_object_count": avg_objects
    }


def process_folder(path: Path, output_dir: Path) -> Dict:
    """Process all images in a folder.

    Args:
        path: Path to input folder
        output_dir: Directory to save results

    Returns:
        Dictionary with processing results
    """
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    images = [f for f in path.iterdir() if f.suffix.lower() in extensions]
    images.sort()

    print(f"Found {len(images)} images in folder")

    results = []
    for img_path in images:
        result = process_image(img_path, output_dir)
        results.append(result)

    successful = sum(1 for r in results if r.get("success"))
    total_objects = sum(r.get("object_count", 0) for r in results if r.get("success"))
    avg_objects = total_objects / successful if successful > 0 else 0

    print(f"\nFolder processing complete: {successful}/{len(images)} images, avg {avg_objects:.1f} objects/image")

    return {
        "success": True,
        "total_images": len(images),
        "successful": successful,
        "total_objects": total_objects,
        "avg_object_count": avg_objects,
        "results": results
    }

# === ENTRY POINT ===
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process images/videos using the exported qt-pitwall pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_path", help="Path to image, video, or folder")
    parser.add_argument("--output", "-o", help="Output directory (default: ./output)")

    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output) if args.output else Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine input type and process
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    if input_path.is_dir():
        result = process_folder(input_path, output_dir)
    elif input_path.suffix.lower() in video_extensions:
        result = process_video(input_path, output_dir)
    elif input_path.suffix.lower() in image_extensions:
        result = process_image(input_path, output_dir)
    else:
        print(f"Error: Unsupported file type: {input_path.suffix}")
        sys.exit(1)

    if not result.get("success"):
        print(f"Error: {result.get('error', 'Unknown error')}")
        sys.exit(1)

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
