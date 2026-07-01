"""Build one before/after composite per cycle: the phase-1 classify overlay beside
its tilt-measure image, with the video, cycle number and classification labelled
across the top. Output is grouped into per-class folders.

Unified data sources (the point of this script -- one join, three files):
  * classification.json   id -> (classify_image, measure_image)   [the pairing]
  * classify_phase1.csv   classify_basename -> label (1/2/3)       [the class]
  * measurements.json     id -> min_angle, angles_deg, n_strips    [the tilt data]

Pixels come from the two by-class trees:
  * classify_phase1_overlays/<classdir>/<...>_overlay.png   (LEFT:  classify / phase 1)
  * measure_by_class/<labeldir>/<...>.png                   (RIGHT: measure / tilt)

Files are indexed by id (clip_Cxxx) parsed from the filename, so the differing
folder names (1 vs 1_looped, unlabeled vs 0_unlabelled) don't matter.

Default: one composite per pair into compare_by_class/<class>/, named so a plain
alphabetical sort = worst-tilt-first. Use --sheets for paginated contact sheets.

Usage:
    python compose_pairs.py            # one image per pair (default)
    python compose_pairs.py --width 1280
    python compose_pairs.py --dry      # join stats only
    python compose_pairs.py --sheets   # paginated contact sheets instead
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent / "phase_pipeline_out"
CLASSIFICATION_JSON = BASE / "classification.json"
MEASUREMENTS_JSON = BASE / "measurements.json"
CLASSIFY_CSV = BASE / "classify_phase1.csv"
MEASURE_BY_CLASS = BASE / "measure_by_class"
OVERLAY_BY_CLASS = BASE / "classify_phase1_overlays"
OUT_DIR = BASE / "compare_by_class"

# label int -> canonical class name (matches sort_overlays_by_class.py)
CLASS_NAMES: dict[int, str] = {0: "0_unlabelled", 1: "1_looped", 2: "2_smooth", 3: "3_other"}

ID_RE = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_C\d+)")

SRC_RATIO = 1000 / 1920  # source frames are 1920x1000

# per-pair header colours
BG = (22, 22, 26)
FG = (240, 240, 240)
CLASSIFY_TAG = (120, 200, 255)
MEASURE_TAG = (255, 200, 120)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("arialbd.ttf", "arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


@dataclass(frozen=True)
class Pair:
    id: str
    clip: str
    cycle: int
    label: int
    overlay: Path
    measure: Path
    min_angle: float | None
    angles_deg: tuple[float, ...]
    n_strips: int | None

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.label, str(self.label))

    @property
    def sort_key(self) -> float:
        return self.min_angle if self.min_angle is not None else 1e9

    def header(self) -> str:
        bits = [self.clip, f"cycle {self.cycle}", f"class {self.label} ({self.class_name})"]
        if self.min_angle is not None:
            bits.append(f"min tilt {self.min_angle:+.1f}°")
        if self.n_strips is not None:
            bits.append(f"{self.n_strips} strips")
        return "    ".join(bits)


def index_by_id(root: Path) -> dict[str, Path]:
    """Map id (clip_Cxxx) -> image path for every png under root (first wins)."""
    out: dict[str, Path] = {}
    dupes = 0
    for p in root.rglob("*.png"):
        m = ID_RE.search(p.name)
        if not m:
            continue
        if m.group(1) in out:
            dupes += 1
            continue
        out[m.group(1)] = p
    if dupes:
        print(f"  ! {root.name}: {dupes} duplicate ids ignored (kept first)")
    return out


def build_pairs() -> list[Pair]:
    classification = json.loads(CLASSIFICATION_JSON.read_text())
    measurements = json.loads(MEASUREMENTS_JSON.read_text())
    meas_by_id = {e["id"]: e for e in measurements}

    label_by_classify_base: dict[str, int] = {}
    with CLASSIFY_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                label_by_classify_base[row["video_name"].strip()] = int(row["label"])
            except (ValueError, KeyError):
                continue  # blank/non-int label -> falls through to unlabelled

    measure_idx = index_by_id(MEASURE_BY_CLASS)
    overlay_idx = index_by_id(OVERLAY_BY_CLASS)

    pairs: list[Pair] = []
    missing_overlay: list[str] = []
    missing_measure: list[str] = []

    for e in classification:
        cid = e["id"]
        overlay = overlay_idx.get(cid)
        measure = measure_idx.get(cid)
        if overlay is None:
            missing_overlay.append(cid)
            continue
        if measure is None:
            missing_measure.append(cid)
            continue

        label = label_by_classify_base.get(Path(e["classify_image"]).name, 0)
        meas = meas_by_id.get(cid, {})
        pairs.append(
            Pair(
                id=cid,
                clip=e.get("clip", cid.rsplit("_C", 1)[0]),
                cycle=int(e.get("cycle", -1)),
                label=label,
                overlay=overlay,
                measure=measure,
                min_angle=meas.get("min_angle"),
                angles_deg=tuple(meas.get("angles_deg") or []),
                n_strips=meas.get("n_strips"),
            )
        )

    counts: dict[str, int] = {}
    for p in pairs:
        counts[p.class_name] = counts.get(p.class_name, 0) + 1

    print(f"classification.json entries : {len(classification)}")
    print(f"measure_by_class indexed    : {len(measure_idx)}")
    print(f"overlays indexed            : {len(overlay_idx)}")
    print(f"paired (both present)       : {len(pairs)}")
    print(f"per-class pairs             : {dict(sorted(counts.items()))}")
    print(f"missing overlay / measure   : {len(missing_overlay)} / {len(missing_measure)}")
    if missing_measure[:5]:
        print(f"  e.g. no measure img : {missing_measure[:5]}")
    if missing_overlay[:5]:
        print(f"  e.g. no overlay img : {missing_overlay[:5]}")
    return pairs


def render_per_pair(pairs: list[Pair], src_w: int) -> None:
    src_h = round(src_w * SRC_RATIO)
    header_h = max(40, src_w // 22)
    header_font = load_font(max(18, header_h // 2))
    tag_font = load_font(max(14, src_w // 60))

    by_class: dict[int, list[Pair]] = {}
    for p in pairs:
        by_class.setdefault(p.label, []).append(p)

    total = 0
    for label in sorted(by_class):
        cdir = OUT_DIR / CLASS_NAMES.get(label, str(label))
        cdir.mkdir(parents=True, exist_ok=True)
        ranked = sorted(by_class[label], key=lambda p: (p.sort_key, p.cycle))
        for rank, p in enumerate(ranked, start=1):
            comp = Image.new("RGB", (src_w * 2, header_h + src_h), BG)
            for i, path in enumerate((p.overlay, p.measure)):
                with Image.open(path) as im:
                    im = im.convert("RGB").resize((src_w, src_h), Image.LANCZOS)
                    comp.paste(im, (i * src_w, header_h))
            d = ImageDraw.Draw(comp)
            d.text((10, (header_h - header_font.size) // 2), p.header(), font=header_font, fill=FG)
            d.text((8, header_h + 6), "classify (phase 1)", font=tag_font, fill=CLASSIFY_TAG)
            d.text((src_w + 8, header_h + 6), "measure (tilt)", font=tag_font, fill=MEASURE_TAG)
            comp.save(cdir / f"{rank:04d}__{p.clip}_C{p.cycle:03d}.png")
            total += 1
        print(f"  {CLASS_NAMES.get(label, label):14s} {len(ranked):4d} -> {cdir}")
    print(f"\n{total} per-pair composites in {OUT_DIR}")


# ---- optional contact-sheet mode (--sheets) ----
SHEET = dict(THUMB_W=300, GAP=6, CELL_PAD=8, CAPTION_H=22, COLS=3,
             ROWS=14, MARGIN=16, TITLE_H=44)


def render_sheets(pairs: list[Pair]) -> None:
    s = SHEET
    font, title_font = load_font(13), load_font(22)
    thumb_h = round(s["THUMB_W"] * SRC_RATIO)
    cell_w = s["THUMB_W"] * 2 + s["GAP"] + s["CELL_PAD"]
    cell_h = s["CAPTION_H"] + thumb_h + s["CELL_PAD"]
    per_page = s["COLS"] * s["ROWS"]
    page_w = s["MARGIN"] * 2 + s["COLS"] * cell_w
    page_h = s["MARGIN"] * 2 + s["TITLE_H"] + s["ROWS"] * cell_h

    by_class: dict[int, list[Pair]] = {}
    for p in pairs:
        by_class.setdefault(p.label, []).append(p)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for label in sorted(by_class):
        ranked = sorted(by_class[label], key=lambda p: (p.sort_key, p.cycle))
        name = CLASS_NAMES.get(label, str(label))
        n_pages = (len(ranked) + per_page - 1) // per_page
        for pg in range(n_pages):
            chunk = ranked[pg * per_page: (pg + 1) * per_page]
            page = Image.new("RGB", (page_w, page_h), BG)
            d = ImageDraw.Draw(page)
            d.text((s["MARGIN"], s["MARGIN"]),
                   f"Class {name}  -  {len(ranked)} pairs  -  page {pg + 1}/{n_pages}",
                   font=title_font, fill=FG)
            for i, p in enumerate(chunk):
                r, c = divmod(i, s["COLS"])
                x = s["MARGIN"] + c * cell_w
                y = s["MARGIN"] + s["TITLE_H"] + r * cell_h
                cell = Image.new("RGB", (s["THUMB_W"] * 2 + s["GAP"], s["CAPTION_H"] + thumb_h), BG)
                cd = ImageDraw.Draw(cell)
                cap = f"C{p.cycle:03d} {p.clip}" + (f" {p.min_angle:+.1f}°" if p.min_angle is not None else "")
                cd.text((2, 3), cap, font=font, fill=FG)
                for j, path in enumerate((p.overlay, p.measure)):
                    with Image.open(path) as im:
                        im = im.convert("RGB").resize((s["THUMB_W"], thumb_h), Image.LANCZOS)
                        cell.paste(im, (j * (s["THUMB_W"] + s["GAP"]), s["CAPTION_H"]))
                page.paste(cell, (x, y))
            page.save(OUT_DIR / f"{name}_p{pg + 1:02d}.png")
        print(f"  {name:14s} {len(ranked):4d} pairs -> {n_pages} page(s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry", action="store_true", help="join stats only, render nothing")
    ap.add_argument("--sheets", action="store_true", help="paginated contact sheets instead of per-pair")
    ap.add_argument("--width", type=int, default=960, help="per-image width in a pair composite (default 960)")
    args = ap.parse_args()

    pairs = build_pairs()
    if args.dry:
        print("\n(dry run - no images written)")
        return
    if not pairs:
        sys.exit("No pairs built - nothing to render.")

    if args.sheets:
        print("\nRendering contact sheets:")
        render_sheets(pairs)
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\nRendering one composite per pair (each image {args.width * 2}px wide):")
        render_per_pair(pairs, args.width)


if __name__ == "__main__":
    main()
