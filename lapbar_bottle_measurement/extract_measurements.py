"""
Lapbar + Bottles measurement extraction.

Applies the GUIOpenCV pipeline config (Lapbar+Bottles.json) to a folder of frames:
  - "bottles" ROI: expects exactly 3 bottle contours with matching minAreaRect
    angle / area / length / width, similar centre y, and even x spacing.
    Searches all contour triplets for one meeting the tolerances; frame is
    rejected if none does.
  - "Lap" ROI: expects one large contour (the blue film); extracts the
    topmost-leftmost corner of it.

Outputs a debug overlay per frame and a single JSON of measurements.

All paths and tolerances are in the CONFIG block below.
"""

import itertools
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# CONFIG — edit paths / tolerances here
# ============================================================================
CONFIG_JSON = Path(r"C:\Users\jkind\Documents\McLaren\Diageo_ShrinkWrap\lapbar_bottle_measurement\Lapbar+Bottles.json")
FRAMES_DIR = Path(r"C:\Users\jkind\Documents\McLaren\Diageo_ShrinkWrap\exported_samples\phase1_to_phase2")
OUTPUT_DIR = Path(r"C:\Users\jkind\Documents\McLaren\Diageo_ShrinkWrap\lapbar_bottle_measurement\debug_phase1_to_phase2")
OUTPUT_JSON = Path(r"C:\Users\jkind\Documents\McLaren\Diageo_ShrinkWrap\lapbar_bottle_measurement\measurements_phase1_to_phase2.json")
SCATTER_PNG = Path(r"C:\Users\jkind\Documents\McLaren\Diageo_ShrinkWrap\lapbar_bottle_measurement\scatter_b2x_vs_lapY_phase1_to_phase2.png")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SAVE_DEBUG_FOR = "passes"  # "passes" | "all"

BOTTLES_ROI_NAME = "bottles"
LAP_ROI_NAME = "Lap"

# Triplet-validation tolerances for the bottle contours
TRIPLET_TOLERANCES = {
    "angle_tol_deg": 15.0,       # max spread of long-axis orientation across the 3 rects
    "area_rel_tol": 0.35,        # max (max-min)/mean relative spread of rect areas
    "length_rel_tol": 0.25,      # max relative spread of rect long side
    "width_rel_tol": 0.25,       # max relative spread of rect short side
    "y_tol_px": 30.0,            # max spread of centre y (pixels, full-frame coords)
    "spacing_rel_tol": 0.30,     # |gap12 - gap23| / mean(gap) must be below this
    "min_spacing_px": 20.0,      # gaps must be at least this many px (rejects split blobs)
}

# The GUI tool never implemented min_aspect_ratio, so it is off by default.
# Set to True to also enforce config's min_aspect_ratio on minAreaRect long/short.
APPLY_MIN_ASPECT_RATIO = False
# ============================================================================


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def hue_mask(image_bgr, params):
    """HSV inRange matching GUIOpenCV HueDetector semantics (hue_min/hue_max)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lo_h, hi_h = params["hue_min"], params["hue_max"]
    s_lo, s_hi = params["saturation_min"], params["saturation_max"]
    v_lo, v_hi = params["value_min"], params["value_max"]

    def in_range(h1, h2):
        return cv2.inRange(hsv, np.array([h1, s_lo, v_lo]), np.array([h2, s_hi, v_hi]))

    if lo_h < 0:
        return cv2.bitwise_or(in_range(0, hi_h), in_range(180 + lo_h, 180))
    if hi_h > 180:
        return cv2.bitwise_or(in_range(lo_h, 180), in_range(0, hi_h - 180))
    return in_range(lo_h, hi_h)


def contour_filter_mask(mask, params):
    """Filter mask contours on area/perimeter/convexity/circularity (tool semantics).

    A max of 0 in the config means "no limit"."""
    min_area = params.get("min_area", 0)
    max_area = params.get("max_area", 0) or float("inf")
    min_per = params.get("min_perimeter", 0)
    max_per = params.get("max_perimeter", 0) or float("inf")
    min_cvx = params.get("min_convexity", 0.0)
    max_cvx = params.get("max_convexity", 1.0)
    min_circ = params.get("min_circularity", 0.0)
    max_circ = params.get("max_circularity", 1.0)
    min_ar = params.get("min_aspect_ratio", 0.0) if APPLY_MIN_ASPECT_RATIO else 0.0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        perimeter = cv2.arcLength(c, True)
        if not (min_per <= perimeter <= max_per):
            continue
        hull_area = cv2.contourArea(cv2.convexHull(c))
        convexity = area / hull_area if hull_area > 0 else 0
        if not (min_cvx <= convexity <= max_cvx):
            continue
        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        if not (min_circ <= circularity <= max_circ):
            continue
        if min_ar > 0:
            (_, _), (rw, rh), _ = cv2.minAreaRect(c)
            short, long_ = min(rw, rh), max(rw, rh)
            if short <= 0 or long_ / short < min_ar:
                continue
        cv2.drawContours(out, [c], -1, 255, -1)
    return out


def morphology_mask(mask, params):
    k = max(1, int(params.get("kernel_size", 3)))
    kernel = np.ones((k, k), np.uint8)
    op = params.get("operation", "dilate")
    ops = {
        "dilate": lambda m: cv2.dilate(m, kernel),
        "erode": lambda m: cv2.erode(m, kernel),
        "open": lambda m: cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel),
        "close": lambda m: cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel),
    }
    if op not in ops:
        raise ValueError(f"Unsupported morphology operation: {op}")
    return ops[op](mask)


def denoise_image(image, params):
    k = max(1, int(params.get("kernel_size", 3)))
    filter_type = params.get("filter_type", "median")
    if filter_type == "median":
        k = k if k % 2 == 1 else k + 1
        return cv2.medianBlur(image, k)
    if filter_type == "gaussian":
        k = k if k % 2 == 1 else k + 1
        return cv2.GaussianBlur(image, (k, k), params.get("sigma", 0))
    raise ValueError(f"Unsupported denoising filter: {filter_type}")


def run_roi_pipeline(frame, roi_cfg):
    """Run one ROI's processor pipeline (processors ordered by their 'index').

    Returns (contours, roi_rect) with contour coordinates in full-frame space."""
    x, y = roi_cfg["x"], roi_cfg["y"]
    w, h = roi_cfg["width"], roi_cfg["height"]
    fh, fw = frame.shape[:2]
    x2, y2 = min(x + w, fw), min(y + h, fh)
    x, y = max(x, 0), max(y, 0)
    if x2 <= x or y2 <= y:
        return [], (x, y, 0, 0)

    crop = frame[y:y2, x:x2]
    pipeline = roi_cfg["pipeline"]

    steps = sorted(
        (
            (name, cfg)
            for name, cfg in pipeline.items()
            if isinstance(cfg, dict) and cfg.get("enabled", False)
        ),
        key=lambda item: item[1].get("index", 99),
    )

    image = crop
    mask = np.full(crop.shape[:2], 255, np.uint8)
    for name, cfg in steps:
        if name == "denoising":
            image = denoise_image(image, cfg)
        elif name == "hue_detection":
            mask = cv2.bitwise_and(hue_mask(image, cfg), mask)
        elif name == "morphology":
            mask = morphology_mask(mask, cfg)
        elif name == "contour_filtering":
            mask = contour_filter_mask(mask, cfg)
        # edge_detection / shape_extraction disabled in this config; skip silently
        # only if disabled — enabled-but-unsupported should be loud:
        elif name in ("edge_detection", "shape_extraction"):
            raise NotImplementedError(f"Processor '{name}' enabled in config but not supported here")

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c + np.array([x, y]) for c in contours], (x, y, x2 - x, y2 - y)


def rect_descriptor(contour):
    """minAreaRect summarised as centre, long/short side, long-axis angle (0-180), area."""
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
    if rw >= rh:
        long_, short = rw, rh
        axis_angle = angle % 180.0
    else:
        long_, short = rh, rw
        axis_angle = (angle + 90.0) % 180.0
    return {
        "cx": float(cx),
        "cy": float(cy),
        "length": float(long_),
        "width": float(short),
        "angle": float(axis_angle),
        "area": float(long_ * short),
        "contour_area": float(cv2.contourArea(contour)),
        "box": cv2.boxPoints(((cx, cy), (rw, rh), angle)).astype(int).tolist(),
    }


def angular_spread(angles):
    """Max pairwise difference of orientations on a 0-180 circle."""
    diffs = [
        min(abs(a - b), 180.0 - abs(a - b))
        for a, b in itertools.combinations(angles, 2)
    ]
    return max(diffs) if diffs else 0.0


def rel_spread(values):
    mean = sum(values) / len(values)
    return (max(values) - min(values)) / mean if mean > 0 else float("inf")


def find_bottle_triplet(rects, tol):
    """Search all 3-combinations of candidate rects for one meeting the
    similarity requirements. Returns (best_triplet_sorted_by_x, reason)."""
    if len(rects) < 3:
        return None, f"only {len(rects)} candidate contour(s), need 3"

    best, best_score = None, float("inf")
    reasons = []
    for combo in itertools.combinations(rects, 3):
        trio = sorted(combo, key=lambda r: r["cx"])
        angle_spread = angular_spread([r["angle"] for r in trio])
        if angle_spread > tol["angle_tol_deg"]:
            reasons.append(f"angle spread {angle_spread:.1f}deg")
            continue
        area_spread = rel_spread([r["area"] for r in trio])
        if area_spread > tol["area_rel_tol"]:
            reasons.append(f"area spread {area_spread:.2f}")
            continue
        length_spread = rel_spread([r["length"] for r in trio])
        if length_spread > tol["length_rel_tol"]:
            reasons.append(f"length spread {length_spread:.2f}")
            continue
        width_spread = rel_spread([r["width"] for r in trio])
        if width_spread > tol["width_rel_tol"]:
            reasons.append(f"width spread {width_spread:.2f}")
            continue
        y_spread = max(r["cy"] for r in trio) - min(r["cy"] for r in trio)
        if y_spread > tol["y_tol_px"]:
            reasons.append(f"y spread {y_spread:.0f}px")
            continue
        gap12 = trio[1]["cx"] - trio[0]["cx"]
        gap23 = trio[2]["cx"] - trio[1]["cx"]
        if min(gap12, gap23) < tol["min_spacing_px"]:
            reasons.append(f"spacing too small ({gap12:.0f}/{gap23:.0f}px)")
            continue
        mean_gap = (gap12 + gap23) / 2.0
        spacing_dev = abs(gap12 - gap23) / mean_gap
        if spacing_dev > tol["spacing_rel_tol"]:
            reasons.append(f"uneven spacing {spacing_dev:.2f}")
            continue

        score = (
            angle_spread / tol["angle_tol_deg"]
            + area_spread / tol["area_rel_tol"]
            + length_spread / tol["length_rel_tol"]
            + width_spread / tol["width_rel_tol"]
            + y_spread / tol["y_tol_px"]
            + spacing_dev / tol["spacing_rel_tol"]
        )
        if score < best_score:
            best, best_score = trio, score

    if best is None:
        detail = "; ".join(reasons[:4]) if reasons else "no combinations"
        return None, f"no valid triplet among {len(rects)} candidates ({detail})"
    return best, None


def lap_top_left_corner(contour):
    """Topmost-leftmost corner: contour point minimising x + y."""
    pts = contour.reshape(-1, 2)
    idx = int(np.argmin(pts[:, 0] + pts[:, 1]))
    return int(pts[idx, 0]), int(pts[idx, 1])


def draw_overlay(frame, bottles_roi, lap_roi, bottle_rects, triplet, lap_contour,
                 lap_corner, status_lines):
    vis = frame.copy()
    for (rx, ry, rw, rh), color in ((bottles_roi, (0, 200, 255)), (lap_roi, (255, 200, 0))):
        cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), color, 1)

    for r in bottle_rects:
        cv2.polylines(vis, [np.array(r["box"])], True, (128, 128, 128), 1)

    if triplet:
        for i, r in enumerate(triplet):
            cv2.polylines(vis, [np.array(r["box"])], True, (0, 255, 0), 2)
            centre = (int(r["cx"]), int(r["cy"]))
            cv2.circle(vis, centre, 4, (0, 0, 255), -1)
            cv2.putText(vis, f"B{i + 1}", (centre[0] - 12, centre[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if lap_contour is not None:
        cv2.drawContours(vis, [lap_contour], -1, (255, 0, 0), 2)
    if lap_corner is not None:
        cv2.circle(vis, lap_corner, 8, (0, 0, 255), 2)
        cv2.putText(vis, f"lap TL {lap_corner}", (lap_corner[0] + 12, lap_corner[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for i, line in enumerate(status_lines):
        colour = (0, 255, 0) if line.startswith("OK") else (0, 0, 255)
        cv2.putText(vis, line, (10, 30 + 28 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    return vis


def get_roi(config, name):
    for roi in config["roi"]["active_rois"]:
        if roi["name"] == name and roi.get("enabled", False):
            return roi
    raise KeyError(f"ROI '{name}' not found or disabled in config")


def process_frame(frame, bottles_cfg, lap_cfg):
    """Returns (record_dict, overlay_image)."""
    bottle_contours, bottles_rect = run_roi_pipeline(frame, bottles_cfg)
    lap_contours, lap_rect = run_roi_pipeline(frame, lap_cfg)

    bottle_rects = [rect_descriptor(c) for c in bottle_contours]
    triplet, bottle_reason = find_bottle_triplet(bottle_rects, TRIPLET_TOLERANCES)

    lap_contour, lap_corner, lap_reason = None, None, None
    if lap_contours:
        lap_contour = max(lap_contours, key=cv2.contourArea)
        lap_corner = lap_top_left_corner(lap_contour)
    else:
        lap_reason = "no lap contour passed the filter"

    valid = triplet is not None and lap_corner is not None
    record = {
        "valid": valid,
        "bottles": None,
        "lap_top_left": None,
        "rejection_reasons": [],
    }
    if triplet:
        record["bottles"] = [
            {
                "center_x": round(r["cx"], 1),
                "center_y": round(r["cy"], 1),
                "angle_deg": round(r["angle"], 1),
                "length": round(r["length"], 1),
                "width": round(r["width"], 1),
                "rect_area": round(r["area"], 1),
            }
            for r in triplet
        ]
    else:
        record["rejection_reasons"].append(f"bottles: {bottle_reason}")
    if lap_corner:
        record["lap_top_left"] = {
            "x": lap_corner[0],
            "y": lap_corner[1],
            "contour_area": round(float(cv2.contourArea(lap_contour)), 1),
        }
    else:
        record["rejection_reasons"].append(f"lap: {lap_reason}")

    status = ["OK" if valid else "REJECTED"]
    status += record["rejection_reasons"]
    overlay = draw_overlay(frame, bottles_rect, lap_rect, bottle_rects, triplet,
                           lap_contour, lap_corner, status)
    return record, overlay


def plot_b2_vs_lap(results, out_path):
    """Scatter of middle-bottle (B2) centre x vs lapbar top-left corner y,
    over all valid frames. Image y grows downward, so the y axis is inverted
    to read 'up = higher in frame'."""
    xs, ys = [], []
    for record in results.values():
        if record.get("valid"):
            xs.append(record["bottles"][1]["center_x"])
            ys.append(record["lap_top_left"]["y"])
    if not xs:
        print("No valid frames - scatter not generated")
        return

    surface, ink, ink2, series = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6"
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=surface)
    ax.set_facecolor(surface)
    ax.scatter(xs, ys, s=36, color=series, alpha=0.65, edgecolors="none")
    ax.invert_yaxis()
    ax.set_xlabel("B2 (middle bottle) centre x [px]", color=ink)
    ax.set_ylabel("Lapbar top-left corner y [px]\n(axis inverted: up = higher in frame)",
                  color=ink)
    ax.set_title(f"Middle bottle x vs lapbar height - {len(xs)} valid frames",
                 color=ink)
    ax.grid(True, color="#e4e3e0", linewidth=0.7)
    ax.tick_params(colors=ink2)
    for spine in ax.spines.values():
        spine.set_color("#e4e3e0")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=surface)
    plt.close(fig)
    print(f"Scatter plot: {out_path}")


def main():
    config = load_config(CONFIG_JSON)
    bottles_cfg = get_roi(config, BOTTLES_ROI_NAME)
    lap_cfg = get_roi(config, LAP_ROI_NAME)

    frames = sorted(
        p for p in FRAMES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not frames:
        raise FileNotFoundError(f"No images found in {FRAMES_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUTPUT_DIR.glob("*_debug.jpg"):
        stale.unlink()

    results = {}
    n_valid = 0
    for i, path in enumerate(frames, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            results[path.name] = {"valid": False,
                                  "rejection_reasons": ["failed to read image"]}
            continue
        record, overlay = process_frame(frame, bottles_cfg, lap_cfg)
        results[path.name] = record
        n_valid += record["valid"]
        if SAVE_DEBUG_FOR == "all" or record["valid"]:
            cv2.imwrite(str(OUTPUT_DIR / f"{path.stem}_debug.jpg"), overlay)
        if i % 25 == 0 or i == len(frames):
            print(f"  {i}/{len(frames)} frames processed ({n_valid} valid)")

    summary = {
        "frames_dir": str(FRAMES_DIR),
        "config": str(CONFIG_JSON),
        "total_frames": len(frames),
        "valid_frames": n_valid,
        "tolerances": TRIPLET_TOLERANCES,
        "frames": results,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{n_valid}/{len(frames)} frames valid")
    print(f"Measurements: {OUTPUT_JSON}")
    print(f"Debug overlays ({SAVE_DEBUG_FOR}): {OUTPUT_DIR}")

    plot_b2_vs_lap(results, SCATTER_PNG)


if __name__ == "__main__":
    main()
