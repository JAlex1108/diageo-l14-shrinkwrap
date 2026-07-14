"""Diageo: phase-aware run over an S3 time range, then snapshot each good->bad data transition.

Three steps:
  1. Run the phase-aware anomaly pipeline over the range (``run_known_s3_source``), split
     into CALENDAR-DAY sub-runs under <out>/runs/<date>/, each writing its own
     ``anomaly_coverage_log.csv``. A day whose log exists is skipped — so a killed sweep
     resumes at the first unfinished day. Day windows butt-join at midnight and their rows
     are concatenated, so a bad region crossing midnight is still found whole.
  2. Find BAD REGIONS: consecutive log rows where data is absent OR present but thrown away
     (``valid_cycles == "n"`` — the clip didn't score cyclically). Keep regions longer than
     ``--min-gap`` seconds that directly follow a good (cycling) clip. Transitions export
     AS THE RUN PROGRESSES: a streaming callback fires per finished clip (per window flush in
     pooled mode), so a crash mid-day only loses the unexported tail. Per-clip pipeline
     artefacts are OFF by default for speed (--keep-exports restores them).
  3. For each region, save ONE image (JPEG, frames downscaled by --scale, default half
     resolution) spanning the transition: row 1 = 50..1 frames before it (end of the last
     good clip), row 2 = +1..+200 after it (start of the first thrown-away clip; absent when
     the recording just stopped). Underneath, the phase energy signal (per-frame
     NCC vs first frame — the pipeline's default phase method) across the good clip AND into
     the bad region, with the transition marked and the CAUSE distinguished everywhere: red
     trace/shading = footage that exists but was thrown away by phase awareness (no valid
     cycles, --max-bad-clips clips plotted); grey hatched span = recording stopped (no frames
     exist). The header gives the seconds of each. Clips flagged anomalous are shaded orange
     with their name written in the span, and the run summary's EXACT anomalous frame ranges
     are drawn as gold bands. Each transition also gets a review video in transition_clips/
     (100 frames before the transition + 300 after, same stem as the image; the bad side
     carries a red border) so an interesting still can be watched immediately. Decodes sequentially — frame-index seeking
     is unreliable on these re-encoded .ts files.

Usage (from anywhere — repo root is derived from this file's location):
    python export_frames_before_gaps.py 2026-07-05 2026-07-06 --min-gap 30
Pooled second pass over the same range (side-view recipe: shared ncc_ref anchor + ROI,
rolling cross-video window, short clips scored instead of skipped; outputs under
<out>/pooled/ so the self-cohort tree is untouched):
    python export_frames_before_gaps.py 2026-07-05 2026-07-06 --out output/out_diageo --pooled
"""
import argparse
import csv
import json
import sys
import tempfile
import traceback
from collections import deque
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path

import boto3
import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

# Lives in Diageo_ShrinkWrap/Stoppage_detection; runs against the 24H_Insights library
# (dev repo). Output trees default under output/ next to this script.
REPO = Path(r"c:\Users\jkind\Documents\McLaren\24H_Insights")
ANALYSIS = Path(__file__).resolve().parent / "output"   # .../Diageo_ShrinkWrap/Stoppage_detection/output
sys.path.insert(0, str(REPO))
from VideoModule.data_classes.anomaly.cfg import (   # noqa: E402
    PhaseAwareAnomalyConfig, PooledCohortConfig)
from VideoModule.parallel_io.s3_clip_source import parse_time_bound   # noqa: E402
from VideoModule.pipelines.anomaly_detection.src.anomaly_export import (   # noqa: E402
    _build_coverage_rows)
from VideoModule.pipelines.anomaly_detection.src.batch_run import (   # noqa: E402
    KNOWN_S3_SOURCES, run_known_s3_source)
from VideoModule.pipelines.anomaly_detection.video_anomaly_detection_pipeline import (   # noqa: E402
    CORTEX_SIDEVIEW_CONFIG, CORTEX_SIDEVIEW_REFERENCE_IMAGE)

OFFSETS_BEFORE = (50, 40, 30, 20, 10, 1)     # frames before the transition (good clip tail)
OFFSETS_AFTER = (1, 40, 80, 120, 160, 200)   # frames after it (first bad clip head, if footage)
GRID_COLS = 6                                # 6 before + 6 after -> 2 rows
VIDEO_BEFORE = 100                           # review video: frames kept before the transition
VIDEO_AFTER = 300                            # ...and after it (from the first bad clip)


TS_FMT = "%Y-%m-%d_%H-%M-%S"


def day_windows(start_dt: datetime, end_dt: datetime) -> list[tuple[datetime, datetime]]:
    """Split [start, end] into calendar-day sub-windows, butt-joined at midnight.

    Butt-joined boundaries mean the merged coverage rows stay contiguous, so a bad region
    crossing midnight is still found as ONE region.
    """
    windows = []
    cur = start_dt
    while cur < end_dt:
        nxt = min(end_dt, datetime.combine(cur.date() + timedelta(days=1), dt_time.min))
        windows.append((cur, nxt))
        cur = nxt
    return windows or [(start_dt, end_dt)]


# Per-clip artefacts this workflow never opens (it builds its own composites + review videos).
# Turning them off skips the biggest per-anomalous-clip cost: the export stage measured 4.6s of
# a 9.7s clip — including a full-resolution re-decode for ROI stills — vs 1.1s of compute.
# export_scores stays ON (cheap npz; lets plots be regenerated later).
QUIET_EXPORTS = dict(export_clips=False, export_frames=False, export_overview=False,
                     export_phase_overview=False, export_phase_grid=False,
                     export_worst_frames_fallback=False)


def build_config(source: str, pooled: bool, pool_population: int,
                 keep_exports: bool) -> PhaseAwareAnomalyConfig:
    """The run config for one mode. Both modes keep every complete cycle (no 5% edge trim).

    Self-cohort: the same bare config ``run_known_s3_source`` would build. Pooled: the
    repo's pre-wired side-view recipe (shared ncc_ref phase anchor + ROI) with a rolling
    cross-video window of ``pool_population`` cycles, each clip scored leave-one-video-out;
    short clips are scored against the window instead of skipped. Per-clip artefact exports
    are OFF in both modes unless ``keep_exports`` (see QUIET_EXPORTS).
    """
    quiet = {} if keep_exports else QUIET_EXPORTS
    if not pooled:
        src = KNOWN_S3_SOURCES[source]
        return PhaseAwareAnomalyConfig(
            min_hz=src["min_hz"], max_hz=src["max_hz"],
            min_freq_change_hz=min(0.3, src["min_hz"]),
            cycle_extraction_config={"edge_trim_fraction": 0.0},
            **quiet,
        )
    if source != "diageo-cortex-41884872":
        raise SystemExit(
            f"--pooled is wired for the side-view camera only (its shared phase anchor + "
            f"ROI live in CORTEX_SIDEVIEW_CONFIG); got --source {source}")
    if not Path(CORTEX_SIDEVIEW_REFERENCE_IMAGE).exists():
        raise SystemExit(f"pooled run needs the phase-0 anchor image: "
                         f"{CORTEX_SIDEVIEW_REFERENCE_IMAGE}")
    return CORTEX_SIDEVIEW_CONFIG.replace(
        pooled_cohort=PooledCohortConfig(target_population=pool_population),
        cycle_extraction_config={"edge_trim_fraction": 0.0},
        # Pinned: per-phase z with the baseline fit over the WINDOW's ~pool_population cycles
        # (task 237), not each clip's own — a clip uniformly worse than its window keeps that
        # shift through the z instead of having its own elevated mean subtracted away.
        normalise=True, normalise_scope="per_window",
        **quiet,
    )


def run_day_if_needed(source: str, win_start: datetime, win_end: datetime, day_dir: Path,
                      max_clips: int | None, config: PhaseAwareAnomalyConfig,
                      on_result=None) -> list[dict]:
    """Run one day's pipeline sub-window unless its coverage log exists; return its rows.

    This is the resume unit: a swept range re-runs only the days without a completed log.
    A day with NO clips in range gets a synthetic all-absent log (the pipeline writes
    nothing for an empty range), so it is not re-listed on every resume. ``on_result``
    streams each finished clip result out mid-run (the incremental exporter).
    """
    log_path = day_dir / "anomaly_coverage_log.csv"
    if log_path.exists():
        print(f"[reuse] {log_path.parent.name}: coverage log exists — skipping "
              f"(delete the day folder to force a re-run; reuse ignores config changes)")
    else:
        run_known_s3_source(source, win_start, win_end, day_dir, config=config,
                            max_clips=max_clips, on_result=on_result)
        if not log_path.exists():
            secs = (win_end - win_start).total_seconds()
            print(f"[empty] no clips in {log_path.parent.name} — recording the whole "
                  f"sub-window as absent")
            with log_path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=[
                    "data_present", "start", "end", "valid_cycles", "clip_id", "status",
                    "reason", "n_cycles", "is_anomalous", "duration_s"])
                w.writeheader()
                w.writerow({"data_present": "n", "start": win_start.strftime(TS_FMT),
                            "end": win_end.strftime(TS_FMT), "valid_cycles": "n",
                            "clip_id": "", "status": "", "reason": "no clips in S3 range",
                            "n_cycles": 0, "is_anomalous": False, "duration_s": round(secs, 3)})
    return list(csv.DictReader(log_path.open(encoding="utf-8")))


def find_transitions(rows: list[dict], min_gap_s: float, *, verbose: bool = True) -> list[dict]:
    """Good->bad transitions: maximal bad runs (absent OR non-cycling) >= min_gap_s.

    ``rows`` = coverage rows in chronological order — one day's log or several days'
    concatenated (their windows butt-join, so a run crossing midnight merges naturally).
    Each transition carries the last GOOD clip before the run (the build-up footage), the
    bad clips inside the run that DO have footage (thrown away for lacking valid cycles),
    and the split of the run's duration into the two causes: footage thrown away by phase
    awareness (``nocyc_s``) vs no recording at all (``absent_s``). ``reaches_end`` marks a
    run touching the last row — it may still grow when the next day's rows arrive, so the
    incremental exporter defers it. ``verbose=False`` silences the skip prints (used by the
    per-day incremental passes so they don't repeat every day).
    """
    is_bad = [r["data_present"] == "n" or r["valid_cycles"] == "n" for r in rows]

    transitions = []
    i = 0
    while i < len(rows):
        if not is_bad[i]:
            i += 1
            continue
        j = i                                       # [i, j) = one maximal bad run
        while j < len(rows) and is_bad[j]:
            j += 1
        region = rows[i:j]
        absent_s = sum(float(r["duration_s"]) for r in region if r["data_present"] == "n")
        nocyc_s = sum(float(r["duration_s"]) for r in region if r["data_present"] == "y")
        duration = absent_s + nocyc_s
        if duration >= min_gap_s:
            if i == 0 or not rows[i - 1]["clip_id"]:
                if verbose:
                    print(f"[bad region] {rows[i]['start']} ({duration:.0f}s): "
                          f"no good clip before it, skipping")
            else:
                resume = rows[j] if j < len(rows) and rows[j]["clip_id"] else None
                transitions.append({
                    "start": rows[i]["start"],
                    "duration_s": duration,
                    "absent_s": absent_s,
                    "nocyc_s": nocyc_s,
                    "n_rows": j - i,
                    "good_clip": rows[i - 1]["clip_id"],
                    "good_anom": rows[i - 1]["is_anomalous"] == "True",
                    "bad_clips": [(r["clip_id"], r["is_anomalous"] == "True")
                                  for r in region if r["clip_id"]],
                    "reaches_end": j == len(rows),
                    # the clip the line RESUMED on (first good row after the region) — lets the
                    # plot show whether/how it came back
                    "resume_clip": resume["clip_id"] if resume else "",
                    "resume_start": resume["start"] if resume else "",
                })
        i = j
    return transitions


def scan_clip(video_path: Path, scale: float, max_frames: int | None = None,
              ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, float]:
    """One streaming decode: (head frames, tail frames, energy signal, fps), scaled by ``scale``.

    Head = the first max(OFFSETS_AFTER, VIDEO_AFTER) frames, tail = the last
    max(OFFSETS_BEFORE, VIDEO_BEFORE) — kept already-downscaled so a clip never costs
    full-resolution RAM. The signal is per-frame NCC vs the clip's OWN first frame on a
    128x128 grayscale resize, mapped to [0, 1] — the same computation as the pipeline's
    default phase method (``VideoModule.phase_detection.shared_functions.compute_ncc_signal``),
    done streaming.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    head: list[np.ndarray] = []
    tail: deque[np.ndarray] = deque(maxlen=max(*OFFSETS_BEFORE, VIDEO_BEFORE))
    n_head = max(*OFFSETS_AFTER, VIDEO_AFTER)
    signal: list[float] = []
    ref_gray = None
    while True:
        if max_frames is not None and len(signal) >= max_frames:
            break                                # truncated decode (e.g. the resume sample)
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.cvtColor(cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2GRAY)
        if ref_gray is None:
            ref_gray = small
            signal.append(1.0)
        else:
            raw = float(cv2.matchTemplate(small, ref_gray, cv2.TM_CCOEFF_NORMED)[0, 0])
            signal.append(max(0.0, min(1.0, (raw + 1.0) / 2.0)))
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if len(head) < n_head:
            head.append(frame)
        tail.append(frame)
    cap.release()
    return head, list(tail), np.asarray(signal), fps


def load_anomaly_regions(out_dir: Path) -> dict[str, list[tuple[int, int]]]:
    """clip_id -> exact anomalous frame ranges, from the run's anomaly_run_summary.json."""
    summary_path = out_dir / "anomaly_run_summary.json"
    if not summary_path.exists():
        print(f"[warn] {summary_path} not found — anomaly frame ranges won't be highlighted")
        return {}
    clips = json.loads(summary_path.read_text(encoding="utf-8"))["clips"]
    return {c["clip_id"]: [tuple(r) for r in c.get("anomaly_regions", [])] for c in clips}


def write_transition_video(out_path: Path, before: list[np.ndarray],
                           after: list[np.ndarray], fps: float) -> None:
    """Concatenate VIDEO_BEFORE + VIDEO_AFTER frames into a review clip; bad side red-bordered."""
    frames = list(before) + list(after)
    if not frames:
        return
    if fps <= 0:
        print(f"  [warn] source fps unknown, writing {out_path.name} at 25 fps")
        fps = 25.0
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for k, frame in enumerate(frames):
        f = frame.copy()
        if k >= len(before):                  # after the transition -> mark it unmissably
            cv2.rectangle(f, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
            cv2.putText(f, "BAD", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 5)
        writer.write(f)
    writer.release()


def pick_frames(frames: list[np.ndarray], offsets: tuple[int, ...],
                from_end: bool) -> list[tuple[int, np.ndarray]]:
    """[(offset, frame)] at ``offsets`` counted from the end (tail) or start (head).

    Clamps when the clip is shorter than the largest offset and drops the duplicates
    that clamping creates.
    """
    picked: list[tuple[int, np.ndarray]] = []
    for off in offsets:
        if not frames:
            break
        i = max(-off, -len(frames)) if from_end else min(off - 1, len(frames) - 1)
        if picked and picked[-1][0] == (-i if from_end else i + 1):
            continue
        picked.append((-i if from_end else i + 1, frames[i]))
    return picked


def signal_panel(good_sig: np.ndarray, bad_sigs: list[np.ndarray], absent_s: float,
                 clip_labels: list[tuple[str, bool]], anom_regions: list[tuple[int, int]],
                 marks_before: list[int], marks_after: list[int],
                 width_px: int, n_bad_unplotted: int,
                 resume_sig: np.ndarray | None = None, resume_label: str = "") -> np.ndarray:
    """Big, obvious energy plot across the transition, with the CAUSE of the bad data shown.

    Blue = last good clip. Red = footage inside the bad region (exists but was thrown away
    by phase awareness: no valid cycles), one trace per bad clip, dotted separators. Grey
    hatched span = recording stopped (no frames exist, so no signal can be drawn there).
    ``clip_labels`` = (clip name, pipeline-flagged-anomalous) per plotted clip, good clip
    first — each span is named in place; anomaly-flagged spans are shaded orange.
    ``anom_regions`` = the good clip's EXACT anomalous frame ranges (from the run summary),
    drawn as gold bands with their frame numbers.
    """
    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, 900 / dpi), dpi=dpi)
    n_good = len(good_sig)
    ax.plot(np.arange(n_good), good_sig, lw=4, color="tab:blue", label="last good clip")

    spans = [(0, n_good)]                     # (start, end) per plotted clip, good first
    x = n_good
    for k, sig in enumerate(bad_sigs):
        ax.plot(np.arange(x, x + len(sig)), sig, lw=4, color="tab:red",
                label="thrown away by phase awareness (footage, no valid cycles)" if k == 0 else None)
        spans.append((x, x + len(sig)))
        x += len(sig)
        if k < len(bad_sigs) - 1:
            ax.axvline(x, color="darkred", ls=":", lw=2)
    if bad_sigs:
        ax.axvspan(n_good, x, color="red", alpha=0.08)

    anom_labelled = False
    for (lo, hi), (name, is_anom) in zip(spans, clip_labels):
        short = name.split("_", 1)[1] if "_" in name else name   # drop the device prefix
        if is_anom:
            ax.axvspan(lo, hi, color="orange", alpha=0.25,
                       label=None if anom_labelled else "pipeline flagged ANOMALY in this clip")
            anom_labelled = True
        ax.text((lo + hi) / 2, 0.035, ("ANOMALY  " if is_anom else "") + short,
                fontsize=22, fontweight="bold" if is_anom else "normal",
                color="darkorange" if is_anom else "dimgray",
                ha="center", va="bottom", transform=ax.get_xaxis_transform())
    for k, (s, e) in enumerate(anom_regions):     # exact anomalous frames within the good clip
        ax.axvspan(s, min(e, n_good), color="gold", alpha=0.6,
                   label="exact anomaly region (frames)" if k == 0 else None)
        ax.text((s + min(e, n_good)) / 2, 0.90, f"frames {s}-{e}", fontsize=20,
                fontweight="bold", color="darkgoldenrod", ha="center", va="bottom",
                transform=ax.get_xaxis_transform())
    if absent_s > 0:
        # No frames exist here, so there is nothing to plot — a fixed-width hatched span
        # stands in for the stopped recording (its real length is in the label/header).
        span = max(200, int(0.15 * max(x, 1)))
        ax.axvspan(x, x + span, color="gray", alpha=0.35, hatch="//",
                   label=f"recording stopped — no frames ({absent_s:.0f}s)")
        x += span
    if n_bad_unplotted:
        ax.text(0.995, 0.03, f"(+{n_bad_unplotted} more thrown-away clip(s) not plotted)",
                fontsize=24, ha="right", va="bottom", transform=ax.transAxes, color="darkred")

    if resume_sig is not None and len(resume_sig):
        # Truncated sample of the clip the line RESUMED on. The x-axis jumps here (the bad
        # region sits between), so a break marker + the wall-clock timestamp make that explicit.
        gap_px = max(60, int(0.02 * x))
        x += gap_px
        ax.axvline(x - gap_px / 2, color="black", ls=(0, (2, 4)), lw=3)
        ax.plot(np.arange(x, x + len(resume_sig)), resume_sig, lw=4, color="tab:green",
                label="resumed (truncated sample)")
        ax.text(x, 1.02, f"  {resume_label}", fontsize=26, fontweight="bold", ha="left",
                va="bottom", color="darkgreen", transform=ax.get_xaxis_transform())
        x += len(resume_sig)

    ax.axvline(n_good - 0.5, color="black", lw=6)
    ax.text(n_good, 1.02, "  TRANSITION -> bad data", fontsize=30, fontweight="bold",
            ha="left", va="bottom", color="black", transform=ax.get_xaxis_transform())
    for off in marks_before:
        ax.axvline(n_good - off, color="red", ls="--", lw=2.5, alpha=0.9)
    for off in marks_after:
        ax.axvline(n_good + off - 1, color="darkred", ls="--", lw=2.5, alpha=0.9)
    ax.set_xlim(0, x)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("frame index", fontsize=30)
    ax.set_ylabel("phase energy (NCC)", fontsize=30)
    ax.tick_params(labelsize=24)
    ax.legend(fontsize=26, loc="lower left")
    ax.set_title("phase energy up to and into the bad region — red dashes = exported frames",
                 fontsize=34, fontweight="bold", loc="left", pad=45)
    ax.grid(alpha=0.4, lw=1.5)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    panel = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    if panel.shape[1] != width_px:            # guard against dpi rounding
        panel = cv2.resize(panel, (width_px, panel.shape[0]))
    return panel


def _label(tile: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    out = tile.copy()
    cv2.putText(out, text, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 10)
    cv2.putText(out, text, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4)
    return out


def compose(picked_before: list[tuple[int, np.ndarray]],
            picked_after: list[tuple[int, np.ndarray]],
            good_sig: np.ndarray, bad_sigs: list[np.ndarray],
            clip_labels: list[tuple[str, bool]], anom_regions: list[tuple[int, int]],
            t: dict, n_bad_unplotted: int,
            resume_sig: np.ndarray | None = None, resume_label: str = "") -> np.ndarray:
    """Header + transition frame grid (50 before / 200 after) + energy plot, one image.

    Row 1 = before the transition (end of the last good clip), row 2 = after it (start of
    the first thrown-away clip; missing entirely when the recording just stopped). The plot
    names every plotted clip in its span and shades pipeline-flagged anomaly clips orange.
    """
    tiles = [_label(f, f"{n} before transition") for n, f in picked_before]
    tiles += [_label(f, f"+{n} after (bad)", (0, 0, 255)) for n, f in picked_after]
    blank = np.zeros_like(tiles[0])                 # pad the last row if tiles % GRID_COLS != 0
    tiles += [blank] * (-len(tiles) % GRID_COLS)
    grid = np.vstack([np.hstack(tiles[i:i + GRID_COLS])
                      for i in range(0, len(tiles), GRID_COLS)])
    header = np.full((80, grid.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header,
                f"bad region starts {t['start']}  ({t['duration_s']:.0f}s total: "
                f"{t['nocyc_s']:.0f}s thrown away by phase awareness, "
                f"{t['absent_s']:.0f}s recording stopped)",
                (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    plot = signal_panel(good_sig, bad_sigs, t["absent_s"], clip_labels, anom_regions,
                        [n for n, _f in picked_before], [n for n, _f in picked_after],
                        grid.shape[1], n_bad_unplotted,
                        resume_sig=resume_sig, resume_label=resume_label)
    return np.vstack([header, grid, plot])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("start", help="YYYY-MM-DD or YYYY-MM-DD_HH-MM-SS (inclusive)")
    ap.add_argument("end", help="YYYY-MM-DD or YYYY-MM-DD_HH-MM-SS (inclusive)")
    ap.add_argument("--source", default="diageo-cortex-41884872", choices=sorted(KNOWN_S3_SOURCES),
                    help="default = the diageo SIDE-VIEW camera (41891555 is the top-down one)")
    ap.add_argument("--out", default=None, type=Path,
                    help="default: out_gap_frames_<source>, so cameras never share a coverage log")
    ap.add_argument("--min-gap", default=30.0, type=float,
                    help="minimum bad-region length, seconds")
    ap.add_argument("--max-clips", default=None, type=int, help="cap the pipeline run for a dry run")
    ap.add_argument("--scale", default=0.5, type=float,
                    help="downscale factor for the exported frames (0.5 = half resolution)")
    ap.add_argument("--max-bad-clips", default=1, type=int,
                    help="thrown-away clips to download per region for the after-frames + red trace")
    ap.add_argument("--pooled", action="store_true",
                    help="cross-video cohort pooling (side-view recipe: ncc_ref anchor + ROI); "
                         "outputs go under <out>/pooled/ so self-cohort runs are never mixed")
    ap.add_argument("--pool-population", default=150, type=int,
                    help="pooled mode: cycles per rolling cohort window")
    ap.add_argument("--keep-exports", action="store_true",
                    help="keep the pipeline's per-clip artefacts (anomaly clips/stills/plots); "
                         "OFF by default — this workflow builds its own review media, and the "
                         "export stage costs more than compute on anomalous clips")
    ap.add_argument("--resume-sample", default=15.0, type=float, metavar="SECONDS",
                    help="plot this many seconds of the clip the line RESUMED on after the bad "
                         "region (green segment, truncated, timestamped); 0 disables")
    ap.add_argument("--force", action="store_true",
                    help="re-export transitions even when their jpg+mp4 already exist "
                         "(regenerate outputs with the current composite format)")
    args = ap.parse_args()
    if args.out is None:
        args.out = ANALYSIS / f"out_gap_frames_{args.source}"
    root = args.out / "pooled" if args.pooled else args.out
    config = build_config(args.source, args.pooled, args.pool_population, args.keep_exports)

    src = KNOWN_S3_SOURCES[args.source]
    if (root / "anomaly_coverage_log.csv").exists():
        print(f"[note] {root / 'anomaly_coverage_log.csv'} is from the old single-run "
              f"layout and is IGNORED — runs now live under {root / 'runs'}/<date>/")

    start_dt = parse_time_bound(args.start)
    end_dt = parse_time_bound(args.end, end=True)
    windows = day_windows(start_dt, end_dt)
    mode = (f"pooled cohort ({args.pool_population} cycles/window)" if args.pooled
            else "self-cohort")
    print(f"[plan] {len(windows)} day sub-run(s), {mode} -> {root}; completed days are "
          f"reused (resume unit)")
    s3 = boto3.Session(profile_name=src["profile"]).client("s3")
    frames_dir = root / "gap_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = root / "transition_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    device = src["prefix"].rstrip("/")           # e.g. cortexvpu-01a-005-41884872

    def fetch(tmp: str, clip_id: str) -> Path:
        clip_path = Path(tmp) / f"{clip_id}.ts"
        s3.download_file(src["bucket"], f"{src['prefix']}{clip_id}.ts", str(clip_path))
        return clip_path

    def already_exported(t: dict) -> bool:
        """Both artefacts must exist — a crash between the jpg and mp4 writes re-exports both."""
        stem = f"transition_{t['start']}__{t['good_clip']}"
        return (frames_dir / f"{stem}.jpg").exists() and (clips_dir / f"{stem}.mp4").exists()

    def export_transition(t: dict) -> None:
        print(f"[transition] {t['start']} ({t['duration_s']:.0f}s bad: {t['nocyc_s']:.0f}s "
              f"thrown away, {t['absent_s']:.0f}s no recording; {t['n_rows']} interval(s)) "
              f"— last good clip {t['good_clip']}")
        with tempfile.TemporaryDirectory() as tmp:
            _head, tail, good_sig, good_fps = scan_clip(fetch(tmp, t["good_clip"]), args.scale)
            picked_before = pick_frames(tail, OFFSETS_BEFORE, from_end=True)
            picked_after, bad_sigs, bad_head = [], [], []
            clip_labels = [(t["good_clip"], t["good_anom"])]
            for k, (bad_id, bad_anom) in enumerate(t["bad_clips"][:args.max_bad_clips]):
                head, _tail, sig, _fps = scan_clip(fetch(tmp, bad_id), args.scale)
                bad_sigs.append(sig)
                clip_labels.append((bad_id, bad_anom))
                if k == 0:
                    picked_after = pick_frames(head, OFFSETS_AFTER, from_end=False)
                    bad_head = head[:VIDEO_AFTER]
            resume_sig, resume_label = None, ""
            if args.resume_sample > 0 and t.get("resume_clip"):
                n_sample = int(args.resume_sample * (good_fps or 60.0))
                _h, _t2, resume_sig, _fps2 = scan_clip(fetch(tmp, t["resume_clip"]),
                                                       args.scale, max_frames=n_sample)
                resume_label = (f"RESUMES {t['resume_start']} "
                                f"(+{t['duration_s'] / 60.0:.1f} min)")
        if not picked_before:
            print(f"  could not decode any frame from {t['good_clip']}")
            return
        n_bad_unplotted = max(0, len(t["bad_clips"]) - args.max_bad_clips)
        stem = f"transition_{t['start']}__{t['good_clip']}"
        out_path = frames_dir / f"{stem}.jpg"
        cv2.imwrite(str(out_path),
                    compose(picked_before, picked_after, good_sig, bad_sigs, clip_labels,
                            anomaly_regions.get(t["good_clip"], []), t, n_bad_unplotted,
                            resume_sig=resume_sig, resume_label=resume_label),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  saved {out_path}")
        video_path = clips_dir / f"{stem}.mp4"
        write_transition_video(video_path, tail[-VIDEO_BEFORE:], bad_head, good_fps)
        print(f"  saved {video_path}")

    rows: list[dict] = []
    anomaly_regions: dict[str, list[tuple[int, int]]] = {}
    days_with_summary = 0
    n_transitions = 0

    def make_on_result():
        """Mid-day streaming exporter: fires per finished clip (per window flush when pooled).

        Rebuilds partial coverage rows from the day's in-memory results (window=None, so only
        inter-clip gaps — nothing premature at the edges), appends them to the completed days'
        rows, and exports any newly-completed transition immediately. A crash mid-day then
        loses only the unexported tail. An export failure is printed in full and never aborts
        the pipeline run (batch boundary — one bad export must not cost a 4k-clip day).
        """
        day_results: list = []

        def on_result(res) -> None:
            day_results.append(res)
            if getattr(res, "anomaly_regions", ()):     # gold bands without waiting for the summary
                anomaly_regions[res.clip_id] = [tuple(r) for r in res.anomaly_regions]
            partial, _bad_ts = _build_coverage_rows(day_results, config, None)
            partial = [dict(r, is_anomalous=str(r["is_anomalous"])) for r in partial]
            found = find_transitions(rows + partial, args.min_gap, verbose=False)
            ready = [t for t in found if not t["reaches_end"]
                     and (args.force or not already_exported(t))]
            for t in ready:
                try:
                    export_transition(t)
                except Exception:   # noqa: BLE001 — loud fail-soft at the run boundary
                    print(f"[export] FAILED for transition {t['start']} — pipeline run continues")
                    traceback.print_exc()

        return on_result

    for i, (win_start, win_end) in enumerate(windows):
        day_dir = root / "runs" / win_start.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        rows.extend(run_day_if_needed(args.source, win_start, win_end, day_dir,
                                      args.max_clips, config, on_result=make_on_result()))
        if (day_dir / "anomaly_run_summary.json").exists():
            anomaly_regions.update(load_anomaly_regions(day_dir))
            days_with_summary += 1

        # Incremental export: transitions are recomputed over ALL rows so far and exported as
        # soon as their bad region is complete. A region touching the end of the data may still
        # grow when the next day arrives, so it waits for the final pass; already-exported
        # composites (jpg on disk) are skipped, which also makes this resume-safe.
        is_final = i == len(windows) - 1
        found = find_transitions(rows, args.min_gap, verbose=is_final)
        wrong = [t["good_clip"] for t in found if not t["good_clip"].startswith(device)]
        if wrong:
            raise RuntimeError(
                f"coverage logs under {root / 'runs'} hold clips from a different camera than "
                f"--source {args.source} (expected ids starting {device!r}, e.g. got "
                f"{wrong[0]!r}). Delete those runs or point --out elsewhere.")
        n_transitions = len(found)
        ready = [t for t in found
                 if (is_final or not t["reaches_end"])
                 and (args.force or not already_exported(t))]
        if ready:
            print(f"[export] {len(ready)} new transition(s) ready after {day_dir.name}")
            for t in ready:
                export_transition(t)

    print(f"[regions] anomaly frame ranges loaded for {days_with_summary}/{len(windows)} "
          f"day run(s) (empty days have none)")
    print(f"[transitions] {n_transitions} total > {args.min_gap:.0f}s; "
          f"done: composites in {frames_dir}, review clips in {clips_dir}")


if __name__ == "__main__":
    main()
