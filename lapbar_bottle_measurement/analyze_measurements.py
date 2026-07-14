"""
Regression + variance analysis of the lapbar/bottle measurements.

Reads the measurements JSON produced by extract_measurements.py and:
  1. Fits a linear regression of lapbar top-left y on middle-bottle (B2) centre x.
  2. Models how the residual spread changes with bottle position (heteroscedasticity):
     residual standard deviation per equal-count bin of B2 x.
  3. Looks for variance drift over time: each frame gets an absolute capture time
     from the clip timestamp in its filename plus frame_number / FPS, then a
     rolling residual std is traced over the session.

All paths and knobs are in the CONFIG block below.
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ============================================================================
# CONFIG — edit paths / knobs here
# ============================================================================
MEASUREMENTS_JSON = Path(r"C:\Users\jkind\Documents\McLaren\Diageo_ShrinkWrap\lapbar_bottle_measurement\measurements_phase1_to_phase2.json")
OUTPUT_PNG = Path(r"C:\Users\jkind\Documents\McLaren\Diageo_ShrinkWrap\lapbar_bottle_measurement\analysis_phase1_to_phase2.png")

FPS = 30.0                 # source video frame rate, used for frame->seconds offset
N_POSITION_BINS = 8        # equal-count bins of B2 x for the variance-vs-position model
ROLLING_WINDOW = 50        # frames per window for the rolling residual std over time

# Filename pattern: ..._YYYY-MM-DD_HH-MM-SS_ffffff_c<cam>[_k<n>]_f<frame>[._]...
TIMESTAMP_RE = re.compile(
    r"_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_(\d{6})_c(\d+)(?:_k\d+)?_f(\d+)[._]"
)
# ============================================================================

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3e0"
SERIES_1, SERIES_2 = "#2a78d6", "#1baf7a"   # data points, model lines


def parse_capture_time(filename):
    """Absolute capture time = clip timestamp + frame_number / FPS."""
    m = TIMESTAMP_RE.search(filename)
    if not m:
        raise ValueError(f"Filename does not match timestamp pattern: {filename}")
    date_s, time_s, micro_s, _cam, frame_s = m.groups()
    clip_start = datetime.strptime(f"{date_s} {time_s}", "%Y-%m-%d %H-%M-%S")
    clip_start += timedelta(microseconds=int(micro_s))
    return clip_start + timedelta(seconds=int(frame_s) / FPS)


def load_valid_records(path):
    with open(path, "r") as f:
        data = json.load(f)
    rows = []
    for name, rec in data["frames"].items():
        if not rec.get("valid"):
            continue
        rows.append({
            "time": parse_capture_time(name),
            "b2_x": rec["bottles"][1]["center_x"],
            "lap_y": rec["lap_top_left"]["y"],
        })
    rows.sort(key=lambda r: r["time"])
    return rows


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.tick_params(colors=INK2)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def plot_regression(ax, b2_x, lap_y, fit):
    ax.scatter(b2_x, lap_y, s=24, color=SERIES_1, alpha=0.45, edgecolors="none")
    xs = np.linspace(b2_x.min(), b2_x.max(), 100)
    ax.plot(xs, fit.slope * xs + fit.intercept, color=SERIES_2, linewidth=2,
            label=(f"y = {fit.slope:.2f}x + {fit.intercept:.0f}\n"
                   f"R² = {fit.rvalue ** 2:.3f}, p = {fit.pvalue:.1e}"))
    ax.invert_yaxis()
    ax.set_xlabel("B2 centre x [px]", color=INK)
    ax.set_ylabel("Lapbar top-left y [px] (inverted: up = higher)", color=INK)
    ax.set_title(f"Linear regression (n = {len(b2_x)})", color=INK)
    ax.legend(loc="best", frameon=False, labelcolor=INK)


def variance_by_position(ax, b2_x, residuals, n_bins):
    """Residual std per equal-count bin of B2 x."""
    edges = np.quantile(b2_x, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    centres, stds, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (b2_x >= lo) & (b2_x < hi)
        if sel.sum() < 3:
            continue
        centres.append((lo + hi) / 2)
        stds.append(residuals[sel].std(ddof=1))
        counts.append(int(sel.sum()))

    ax.plot(centres, stds, color=SERIES_2, linewidth=2, zorder=2)
    ax.scatter(centres, stds, s=48, color=SERIES_1, zorder=3)
    for cx, sd, n in zip(centres, stds, counts):
        ax.annotate(f"n={n}", (cx, sd), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=INK2)
    ax.set_xlabel("B2 centre x [px] (equal-count bins)", color=INK)
    ax.set_ylabel("Residual std [px]", color=INK)
    ax.set_title("Variance vs bottle position", color=INK)
    return centres, stds, counts


def variance_over_time(ax, times, residuals, window):
    """Residuals over the session + rolling std."""
    ax.scatter(times, residuals, s=14, color=SERIES_1, alpha=0.35,
               edgecolors="none", label="residual")
    if len(residuals) >= window:
        kernel = np.ones(window) / window
        mean = np.convolve(residuals, kernel, mode="valid")
        sq_mean = np.convolve(residuals ** 2, kernel, mode="valid")
        rolling_std = np.sqrt(np.maximum(sq_mean - mean ** 2, 0))
        centre_times = times[window - 1:]
        ax.plot(centre_times, rolling_std, color=SERIES_2, linewidth=2,
                label=f"rolling std ({window} frames)")
    ax.axhline(0, color=GRID, linewidth=1)
    ax.set_xlabel("Capture time", color=INK)
    ax.set_ylabel("Residual [px]", color=INK)
    ax.set_title("Residuals & variance drift over time", color=INK)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(loc="best", frameon=False, labelcolor=INK)


def main():
    rows = load_valid_records(MEASUREMENTS_JSON)
    if len(rows) < 10:
        raise ValueError(f"Only {len(rows)} valid records - not enough to analyse")

    b2_x = np.array([r["b2_x"] for r in rows])
    lap_y = np.array([r["lap_y"] for r in rows], dtype=float)
    times = np.array([r["time"] for r in rows])

    fit = stats.linregress(b2_x, lap_y)
    residuals = lap_y - (fit.slope * b2_x + fit.intercept)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), facecolor=SURFACE)
    for ax in axes:
        style_axis(ax)
    plot_regression(axes[0], b2_x, lap_y, fit)
    centres, stds, counts = variance_by_position(axes[1], b2_x, residuals,
                                                 N_POSITION_BINS)
    variance_over_time(axes[2], times, residuals, ROLLING_WINDOW)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    span = times[-1] - times[0]
    print(f"n = {len(rows)} valid frames, {times[0]} -> {times[-1]} ({span})")
    print(f"fit: lap_y = {fit.slope:.3f} * b2_x + {fit.intercept:.1f}")
    print(f"     R² = {fit.rvalue ** 2:.3f}, p = {fit.pvalue:.2e}, "
          f"slope SE = {fit.stderr:.3f}")
    print(f"residual std overall = {residuals.std(ddof=1):.2f} px")
    print("residual std by B2-x bin:")
    for cx, sd, n in zip(centres, stds, counts):
        print(f"  x~{cx:7.1f}: {sd:5.2f} px (n={n})")
    print(f"Figure: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
