"""Per-phase montage of candidate crops ranked by masked-NCC score, to eyeball where the
empty-belt cluster ends and genuine bottle frames begin."""
import sys
from pathlib import Path

import cv2
import numpy as np

SP = Path(__file__).parent
sys.path.insert(0, str(SP))
from export_phase_samples import score_against_phase_template   # noqa: E402

CACHE = SP / "output" / "_pass1_cache.npz"
z = np.load(CACHE, allow_pickle=False)
N_TILES = 18

rows_all = []
for p in range(5):
    sel = z["phase"] == p
    cands = [{"crop": c} for c in z["crop"][sel]]
    score_against_phase_template(cands)
    scores = np.array([c["ncc"] for c in cands])
    order = np.argsort(scores)
    picks = np.linspace(0, len(order) - 1, N_TILES).round().astype(int)
    tiles = []
    for r in picks:
        i = order[r]
        t = cv2.cvtColor(cands[i]["crop"], cv2.COLOR_GRAY2BGR)
        cv2.putText(t, f"{scores[i]:.2f}", (4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(t, f"{scores[i]:.2f}", (4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(t)
    row = np.hstack(tiles)
    label = np.full((36, row.shape[1], 3), 255, np.uint8)
    cv2.putText(label, f"phase {p} — candidates ranked worst->best (evenly spaced)", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    rows_all.extend([label, row])

out = SP / "score_boundary.jpg"
cv2.imwrite(str(out), np.vstack(rows_all), [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"saved {out}")
