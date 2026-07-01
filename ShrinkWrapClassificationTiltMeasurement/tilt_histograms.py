#!/usr/bin/env python3
"""Position-split tilt-angle histograms, overlaid by hand class.

One axis per bottle position (Front / Middle / Back); on each axis the hand
classes (1/2/3) are drawn as semi-transparent histograms so their tilt
distributions can be compared directly.

Phase naming follows capture time: Phase A is the EARLIER frame, Phase B the LATER.
Phase B reads angles straight from ``measurements.json`` (the phase-3 measure view,
the later-in-cycle frame). Phase A is produced by ``measure_phase1_tilt.py`` from the
earlier classify_phase1 frame and reuses ``plot_position_histograms`` here.

Strip order: ``angles_deg`` is spatial left->right ``[leftmost, middle, rightmost]``.
User mapping: front = rightmost, middle = middle, back = leftmost.

Usage:
    python tilt_histograms.py            # builds Phase B figure (measurements.json, later frame)
"""
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "phase_pipeline_out"
MEASUREMENTS_JSON = OUT / "measurements.json"
CLASSIFY_CSV = OUT / "classify_phase1.csv"
PHASE_B_OUT = OUT / "hist_phaseB_measurements.png"

# (display name, index into the spatial-left->right angles_deg list)
# Laid out left->right as Back | Middle | Front to match the physical strip order.
POSITIONS: Sequence[Tuple[str, int]] = (
    ("Back  (leftmost strip)", 0),
    ("Middle", 1),
    ("Front  (rightmost strip)", 2),
)

# Hand classes present in classify_phase1.csv.
CLASS_NAMES: Dict[int, str] = {1: "Class 1", 2: "Class 2", 3: "Class 3"}
CLASS_COLORS: Dict[int, str] = {1: "#1f77b4", 2: "#d62728", 3: "#2ca02c"}

# Higher angle = strip's right end rises = bottle rotated anticlockwise; lower = clockwise.
ANGLE_NOTE = ("Tilt convention:  lower angle → bottle rotated clockwise (CW)    |    "
              "higher angle → rotated anticlockwise (CCW)")

DENSITY = True       # normalise each class (counts are imbalanced 1350/364/283)
ALPHA = 0.35         # well transparent -> all three classes show through each other
BIN_WIDTH = 0.15     # narrow bins -> distribution looks near-continuous
XLIM = (15.0, 25.0)  # view window; density still computed over the full range


def load_labels(csv_path: Path) -> Dict[str, int]:
    """Map phase-1 image basename -> hand class label."""
    labels: Dict[str, int] = {}
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            labels[row["video_name"]] = int(row["label"])
    return labels


def build_dataset(
    measurements: List[Dict],
    labels: Dict[str, int],
    image_field: str = "classify_image",
) -> Tuple[Dict[int, Dict[int, List[float]]], Dict[str, int]]:
    """Group angles by position index, then by class, joining on the hand label.

    ``image_field`` names the record key holding the phase-1 image (a path or a
    bare basename — both reduce to a basename for the join).

    Returns ``(by_position, stats)`` where ``by_position[pos_index][class]`` is the
    list of angles, and ``stats`` reports join coverage.
    """
    by_position: Dict[int, Dict[int, List[float]]] = {
        idx: {cls: [] for cls in CLASS_NAMES} for _, idx in POSITIONS
    }
    matched = unlabelled = wrong_strip_count = 0

    for record in measurements:
        angles = record.get("angles_deg", [])
        if len(angles) != 3:
            wrong_strip_count += 1
            continue
        basename = Path(record[image_field]).name
        cls = labels.get(basename)
        if cls is None:
            unlabelled += 1
            continue
        if cls not in CLASS_NAMES:
            continue
        matched += 1
        for _, idx in POSITIONS:
            by_position[idx][cls].append(float(angles[idx]))

    stats = {
        "records": len(measurements),
        "matched": matched,
        "unlabelled": unlabelled,
        "wrong_strip_count": wrong_strip_count,
    }
    return by_position, stats


def _common_bins(
    by_position: Dict[int, Dict[int, List[float]]], bin_width: float
) -> np.ndarray:
    """Shared bin edges across every axis and class, padded slightly."""
    every = [a for per_cls in by_position.values() for vals in per_cls.values() for a in vals]
    lo = np.floor(float(np.min(every)) / bin_width) * bin_width
    hi = np.ceil(float(np.max(every)) / bin_width) * bin_width
    return np.arange(lo, hi + bin_width, bin_width)


def plot_position_histograms(
    by_position: Dict[int, Dict[int, List[float]]],
    title: str,
    out_path: Path,
    xlim: Tuple[float, float] = XLIM,
    bin_width: float = BIN_WIDTH,
) -> None:
    """Render the 3-axis (Back/Middle/Front) overlaid-class figure.

    ``xlim`` clips the view (density is still computed over the full range);
    pass ``None`` to autoscale. ``bin_width`` is degrees per bin.
    """
    bins = _common_bins(by_position, bin_width)
    curve_x = np.linspace(bins[0], bins[-1], 400)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharex=True, sharey=True)

    for ax, (pos_name, idx) in zip(axes, POSITIONS):
        for cls, name in CLASS_NAMES.items():
            angles = by_position[idx][cls]
            if not angles:
                continue
            ax.hist(
                angles,
                bins=bins,
                density=DENSITY,
                alpha=ALPHA,
                color=CLASS_COLORS[cls],
                edgecolor=CLASS_COLORS[cls],
                linewidth=0.6,
                label=f"{name}  (n={len(angles)}, μ={np.mean(angles):+.1f}°)",
            )
            # Overlay a simple Gaussian fit (mean/std of the class) on the density
            # histogram so each class's spread is easy to compare at a glance.
            mu = float(np.mean(angles))
            sigma = float(np.std(angles, ddof=1)) if len(angles) > 1 else 0.0
            if sigma > 0:
                pdf = np.exp(-0.5 * ((curve_x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
                ax.plot(curve_x, pdf, color=CLASS_COLORS[cls], linewidth=1.6, alpha=0.95)
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_title(pos_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Tilt angle (deg)")
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Density" if DENSITY else "Count")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.015, ANGLE_NOTE, ha="center", fontsize=10, style="italic", color="0.25")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    labels = load_labels(CLASSIFY_CSV)
    measurements = json.loads(MEASUREMENTS_JSON.read_text())
    by_position, stats = build_dataset(measurements, labels, image_field="classify_image")

    print(
        f"Phase B: {stats['matched']} matched / {stats['records']} records "
        f"({stats['unlabelled']} unlabelled, {stats['wrong_strip_count']} not-3-strips)"
    )
    for pos_name, idx in POSITIONS:
        per_cls = "  ".join(
            f"{CLASS_NAMES[c]} n={len(by_position[idx][c])}" for c in CLASS_NAMES
        )
        print(f"  {pos_name:26s} {per_cls}")

    plot_position_histograms(
        by_position,
        "Phase B — tilt angle by bottle position & class (measurements.json, phase-3 measure view, later frame)",
        PHASE_B_OUT,
    )
    print(f"Saved -> {PHASE_B_OUT}")


if __name__ == "__main__":
    main()
