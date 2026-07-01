#!/usr/bin/env python3
"""Sort the classify_phase1 validation overlays into per-class sub-folders.

Reads the hand labels from classify_phase1.csv and moves each ``*_overlay.png``
in ``classify_phase1_overlays/`` into a sub-folder named for its class:

    1_looped / 2_smooth / 3_other        (and 0_unlabelled for images with no label)

Only files sitting directly in the overlays folder are touched, so it is safe to
re-run (already-sorted files in sub-folders are left alone). Use ``--copy`` to
duplicate instead of move.

Usage:
    python sort_overlays_by_class.py [--copy]
"""
import argparse
import shutil
from pathlib import Path
from typing import Dict

import tilt_histograms as th

OVERLAY_DIR = th.OUT / "classify_phase1_overlays"
OVERLAY_SUFFIX = "_overlay.png"

# label value -> sub-folder name
CLASS_DIRS: Dict[int, str] = {1: "1_looped", 2: "2_smooth", 3: "3_other"}
UNLABELLED_DIR = "0_unlabelled"


def overlay_to_video_name(overlay_path: Path) -> str:
    """`<stem>_overlay.png` -> `<stem>.png`, the key used in classify_phase1.csv."""
    return overlay_path.name[: -len(OVERLAY_SUFFIX)] + ".png"


def sort_overlays(copy: bool = False) -> None:
    if not OVERLAY_DIR.is_dir():
        raise FileNotFoundError(f"Overlay folder not found: {OVERLAY_DIR}")

    labels = th.load_labels(th.CLASSIFY_CSV)
    for name in (*CLASS_DIRS.values(), UNLABELLED_DIR):
        (OVERLAY_DIR / name).mkdir(exist_ok=True)

    overlays = sorted(p for p in OVERLAY_DIR.glob(f"*{OVERLAY_SUFFIX}") if p.is_file())
    counts = {name: 0 for name in (*CLASS_DIRS.values(), UNLABELLED_DIR)}

    for overlay in overlays:
        label = labels.get(overlay_to_video_name(overlay))
        subdir = CLASS_DIRS.get(label, UNLABELLED_DIR)
        dest = OVERLAY_DIR / subdir / overlay.name
        if copy:
            shutil.copy2(overlay, dest)
        else:
            shutil.move(str(overlay), str(dest))
        counts[subdir] += 1

    verb = "Copied" if copy else "Moved"
    print(f"{verb} {len(overlays)} overlays into {OVERLAY_DIR.name}/:")
    for name in (*CLASS_DIRS.values(), UNLABELLED_DIR):
        print(f"  {name:14s} {counts[name]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sort phase-1 overlays into per-class folders.")
    parser.add_argument("--copy", action="store_true", help="Copy instead of move")
    args = parser.parse_args()
    sort_overlays(copy=args.copy)


if __name__ == "__main__":
    main()
