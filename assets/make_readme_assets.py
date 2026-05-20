"""Render the three sample images embedded in ``README.md``.

The ``Data/`` folder is not checked into git; this script regenerates the
visualisations from the local copy so that ``assets/`` always reflects the
current preprocessing / depth-estimation output.

Edit ``SAMPLE`` below to pick a different frame, then re-run:

    python assets/make_readme_assets.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# --- sample identifier (left image stem, no extension) ----------------------
SESSION = "2018-10-11-16-03-19"
SAMPLE  = "2018-10-11-16-03-19_2018-10-11-16-05-51-366"
# Picked from evaluation-report.json: high valid-pixel count and abs_rel ≈ 0.026.

# --- visualisation parameters -----------------------------------------------
DEPTH_SCALE = 256          # KITTI uint16 convention: depth_m = pixel / 256
CLIP_MAX_M  = 60.0         # depth range used to colourmap both maps
COLORMAP    = cv2.COLORMAP_MAGMA


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def colorize_depth(depth_m: np.ndarray, vmax: float = CLIP_MAX_M) -> np.ndarray:
    """Map a metric depth image to a BGR colour image; invalid pixels stay black."""
    valid = depth_m > 0
    norm = np.zeros_like(depth_m, dtype=np.uint8)
    if valid.any():
        clipped = np.clip(depth_m[valid], 0, vmax) / vmax
        norm[valid] = (clipped * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, COLORMAP)
    color[~valid] = (0, 0, 0)
    return color


def main() -> None:
    base = root() / "Data"
    out  = root() / "assets"
    out.mkdir(parents=True, exist_ok=True)

    rgb_src  = base / "left-image-half-size"  / "left-image-half-size"  / f"{SAMPLE}.jpg"
    gt_src   = base / "depth-map-half-size"   / "depth-map-half-size"   / SESSION / f"{SAMPLE}.png"
    pred_src = base / "predicted-depth-half-size" / SESSION / f"{SAMPLE}.png"

    for p in (rgb_src, gt_src, pred_src):
        if not p.exists():
            raise SystemExit(f"Missing source: {p}\n"
                             f"Run `python pipeline.py` to populate predicted depths.")

    rgb  = cv2.imread(str(rgb_src), cv2.IMREAD_COLOR)
    gt   = cv2.imread(str(gt_src),  cv2.IMREAD_UNCHANGED).astype(np.float32) / DEPTH_SCALE
    pred = cv2.imread(str(pred_src), cv2.IMREAD_UNCHANGED).astype(np.float32) / DEPTH_SCALE

    cv2.imwrite(str(out / "sample_rgb.jpg"), rgb)
    cv2.imwrite(str(out / "sample_gt_depth.png"),   colorize_depth(gt))
    cv2.imwrite(str(out / "sample_pred_depth.png"), colorize_depth(pred))

    print(f"Sample: {SAMPLE}")
    print(f"Colourmap: MAGMA, depth range 0..{CLIP_MAX_M:.0f} m, invalid pixels black.")
    print(f"Wrote: {out}/sample_rgb.jpg, sample_gt_depth.png, sample_pred_depth.png")


if __name__ == "__main__":
    main()
