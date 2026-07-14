"""Montage of 12 frames across one machine cycle (original anchor phasing) to pick a new
phase-0 anchor frame: the moment the bottles ENTER the camera view."""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(r"c:\Users\jkind\Documents\McLaren\24H_Insights")
sys.path.insert(0, str(REPO))

from VideoModule.io.read import read_video
from VideoModule.phase_detection.phase_awareness import run_phase_awareness
from VideoModule.pipelines.anomaly_detection.src.scoring_grid import _load_reference_anchor
from VideoModule.pipelines.anomaly_detection.video_anomaly_detection_pipeline import (
    CORTEX_SIDEVIEW_CONFIG)

SP = Path(__file__).parent
CLIP = SP / "smoke_out" / "_ts_cache" / "cortexvpu-01a-005-41884872_2026-07-06_09-54-22_316667.ts"
N_TILES = 12

cfg = CORTEX_SIDEVIEW_CONFIG.replace(cycle_extraction_config={"edge_trim_fraction": 0.0})
anchor = _load_reference_anchor(cfg.reference_image)
frames, fps = read_video(str(CLIP), resize=cfg.load_resize)
pa = run_phase_awareness(frames, fps, cfg.phase, reference_frame=anchor)
cycles = pa.dyn.cycles
print(f"fps {fps}, {len(frames)} frames, cycles: {cycles}")

c0, c1 = int(cycles[1][0]), int(cycles[1][1])          # second cycle (clear of clip edges)
idxs = np.linspace(c0, c1 - 1, N_TILES).round().astype(int)

# Full-res tiles for readability: sequential decode, grab the needed indices.
cap = cv2.VideoCapture(str(CLIP))
grab = {int(i) for i in idxs}
full = {}
fi = 0
while grab:
    ok, fr = cap.read()
    if not ok:
        break
    if fi in grab:
        full[fi] = fr
        grab.discard(fi)
    fi += 1
cap.release()

tiles = []
for i in idxs:
    t = cv2.resize(full[int(i)], (640, 333))
    cv2.putText(t, f"frame {i}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
    cv2.putText(t, f"frame {i}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    tiles.append(t)
rows = [np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(3)]
out = SP / "cycle_montage.jpg"
cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"saved {out}")
