#!/usr/bin/env python3
"""One-off: migrate existing per-clip output/<clip>/ outputs into the global dataset.

Moves every image into the pooled global folders with traceable names and writes the global
measurements/classification files, rewriting all paths via the same helpers the pipeline uses so
nothing dangles. Removes the per-clip folders afterward. Safe to delete this script once run.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase_pipeline as pp

OUT = pp.OUT


def main() -> None:
    pp._ensure_dirs()
    clip_dirs = sorted(d for d in OUT.iterdir() if d.is_dir() and (d / "pairs.json").exists())
    print(f"migrating {len(clip_dirs)} per-clip folders -> global dataset")

    records, moved = [], 0
    for cd in clip_dirs:
        tag = pp.clip_tag(cd.name)
        for m in json.loads((cd / "pairs.json").read_text()):
            contexts = [(c["phase"], c["frame"]) for c in m["context"]]
            rec = pp.make_record(tag, m["cycle"], m["measure"]["phase"], m["measure"]["frame"],
                                 m["measure"]["n_strips"], m["measure"]["angles_deg"], contexts)
            shutil.move(str(cd / m["measure"]["image"]), str(OUT / rec["measure"]["image"]))
            moved += 1
            for c, rc in zip(m["context"], rec["context"]):
                shutil.move(str(cd / c["image"]), str(OUT / rc["image"]))
                moved += 1
            records.append(rec)
        grid = cd / "phase_grid_kept.png"
        if grid.exists():
            shutil.move(str(grid), str(pp.GRID_DIR / f"{tag}.png"))

    pp.write_global(records, OUT)
    print(f"  moved {moved} images; wrote {len(records)} records")

    for cd in clip_dirs:
        shutil.rmtree(cd)
    old_csv = OUT / "all_measurements_by_angle.csv"
    if old_csv.exists():
        old_csv.unlink()
    print("  removed per-clip folders + superseded all_measurements_by_angle.csv")


if __name__ == "__main__":
    main()
