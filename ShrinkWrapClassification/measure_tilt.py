#!/usr/bin/env python3
"""Tilt measurement for Diageo shrink-wrap bottles.

Reuses the exported qt-pitwall pipeline (``process_pipeline_diageo_tilt.process_frame``)
to obtain the cleaned mask of the 3 Johnnie Walker label strips, fits a
``cv2.minAreaRect`` to each strip, and reports its tilt angle. Tilt is the signed
angle of the strip's long axis from horizontal, in degrees, with positive meaning
the strip rises toward the right of the frame. An upright bottle gives ~0 deg.

A debug overlay is written showing each rotated rectangle and its angle as text.

Usage:
    python measure_tilt.py <image_path> [--output <dir>]
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from process_pipeline_diageo_tilt import process_frame

# Drawing constants
BOX_COLOR = (0, 255, 0)        # green rotated rect
CENTER_COLOR = (0, 165, 255)   # orange centroid
TEXT_COLOR = (0, 255, 255)     # yellow angle label
OUTLINE_COLOR = (0, 0, 0)      # black text outline
PANEL_BG = (40, 40, 40)


def line_angle_deg(box: np.ndarray) -> float:
    """Signed tilt of a minAreaRect's long axis from horizontal.

    Args:
        box: 4x2 array of the rotated-rectangle corner points (cv2.boxPoints).

    Returns:
        Angle in degrees in the range (-90, 90]. 0 = horizontal,
        positive = long axis rises toward the right (screen up-right).
    """
    edges = [box[(i + 1) % 4] - box[i] for i in range(4)]
    lengths = [float(np.hypot(e[0], e[1])) for e in edges]
    long_edge = edges[int(np.argmax(lengths))]
    dx, dy = float(long_edge[0]), float(long_edge[1])
    if dx < 0:  # orient the vector rightward so the sign is unambiguous
        dx, dy = -dx, -dy
    # Image y grows downward; negate dy so "rises to the right" reads positive.
    return float(np.degrees(np.arctan2(-dy, dx)))


def _centroid_x(contour: np.ndarray) -> float:
    """X of a contour's centroid (falls back to bounding-box x)."""
    moments = cv2.moments(contour)
    if moments["m00"]:
        return moments["m10"] / moments["m00"]
    return float(cv2.boundingRect(contour)[0])


def measure_tilt(image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
    """Run the pipeline and measure the tilt of each detected label strip.

    Args:
        image: Input BGR frame.

    Returns:
        Tuple of (mask, measurements). ``measurements`` is a list of dicts ordered
        left-to-right, each with id, center, length, width, angle_deg, and box.
    """
    _, mask, contours = process_frame(image.copy())
    contours = sorted(contours, key=_centroid_x)

    measurements: List[Dict] = []
    for index, contour in enumerate(contours):
        rect = cv2.minAreaRect(contour)  # ((cx, cy), (w, h), angle)
        box = cv2.boxPoints(rect)
        (rect_cx, rect_cy), (rect_w, rect_h), _ = rect
        measurements.append({
            "id": index + 1,
            "center": (float(rect_cx), float(rect_cy)),
            "length": float(max(rect_w, rect_h)),
            "width": float(min(rect_w, rect_h)),
            "angle_deg": line_angle_deg(box),
            "box": box,
        })
    return mask, measurements


def _put_label(image: np.ndarray, text: str, org: Tuple[int, int],
               scale: float = 0.6, thickness: int = 2) -> None:
    """Draw outlined text so it stays legible on a dark frame."""
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                OUTLINE_COLOR, thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                TEXT_COLOR, thickness, cv2.LINE_AA)


def draw_debug_overlay(image: np.ndarray, measurements: List[Dict]) -> np.ndarray:
    """Draw rotated rects, centroids, and angle labels onto a copy of the frame."""
    overlay = image.copy()

    for meas in measurements:
        box = meas["box"].astype(np.int32)
        cv2.drawContours(overlay, [box], -1, BOX_COLOR, 2, cv2.LINE_AA)

        center = (int(round(meas["center"][0])), int(round(meas["center"][1])))
        cv2.circle(overlay, center, 4, CENTER_COLOR, -1, cv2.LINE_AA)

        label = f"L{meas['id']}: {meas['angle_deg']:+.1f} deg"
        label_org = (center[0] - 40, center[1] - 18)
        _put_label(overlay, label, label_org)

    # Summary panel (top-left): per-strip angle + mean
    angles = [m["angle_deg"] for m in measurements]
    lines = [f"Bottles: {len(measurements)}"]
    lines += [f"L{m['id']} tilt: {m['angle_deg']:+.2f} deg" for m in measurements]
    if angles:
        lines.append(f"mean   : {float(np.mean(angles)):+.2f} deg")

    pad, line_h = 10, 26
    panel_w = 260
    panel_h = pad * 2 + line_h * len(lines)
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), PANEL_BG, -1)
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), BOX_COLOR, 1)
    for row, text in enumerate(lines):
        org = (20, 10 + pad + line_h * (row + 1) - 8)
        cv2.putText(overlay, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    TEXT_COLOR, 1, cv2.LINE_AA)

    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure bottle tilt from label strips.")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--output", "-o", help="Output directory (default: ./output)")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: input does not exist: {image_path}")
        sys.exit(1)

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: could not read image: {image_path}")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)

    mask, measurements = measure_tilt(image)
    overlay = draw_debug_overlay(image, measurements)

    overlay_path = output_dir / f"{image_path.stem}_tilt_overlay.png"
    mask_path = output_dir / f"{image_path.stem}_tilt_mask.png"
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(mask_path), mask)

    print(f"Measured {len(measurements)} strip(s) in {image_path.name}:")
    for meas in measurements:
        print(f"  L{meas['id']}: tilt={meas['angle_deg']:+.2f} deg "
              f"length={meas['length']:.0f}px center=({meas['center'][0]:.0f},"
              f"{meas['center'][1]:.0f})")
    if measurements:
        mean_angle = float(np.mean([m["angle_deg"] for m in measurements]))
        print(f"  mean tilt: {mean_angle:+.2f} deg")
    print(f"Overlay -> {overlay_path}")


if __name__ == "__main__":
    main()
