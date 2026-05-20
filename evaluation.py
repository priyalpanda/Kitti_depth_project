import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda iterable, **kwargs: iterable

from depth_estimation import session_of
from pre_process import run as preprocess_run


# ---- depth metrics ----------------------------------------------------------

def depth_metrics(pred: np.ndarray, gt: np.ndarray,
                  min_depth: float = 1.0, max_depth: float = 80.0) -> Dict[str, float]:
    """Standard monocular/stereo depth metrics on a single (pred, gt) pair.

    Uses only pixels where both pred and gt fall in [min_depth, max_depth].
    Returns a dict with abs_rel, sq_rel, rmse, rmse_log, log10, d1/d2/d3, n_valid.
    """
    mask = (gt > min_depth) & (gt < max_depth) & (pred > min_depth) & (pred < max_depth)
    n = int(mask.sum())
    if n == 0:
        return {k: float("nan") for k in
                ("abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "d1", "d2", "d3")} | {"n_valid": 0}
    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)
    diff = p - g
    rel = np.maximum(p / g, g / p)
    return {
        "abs_rel":  float(np.mean(np.abs(diff) / g)),
        "sq_rel":   float(np.mean(diff * diff / g)),
        "rmse":     float(np.sqrt(np.mean(diff * diff))),
        "rmse_log": float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))),
        "log10":    float(np.mean(np.abs(np.log10(p) - np.log10(g)))),
        "d1":       float(np.mean(rel < 1.25)),
        "d2":       float(np.mean(rel < 1.25 ** 2)),
        "d3":       float(np.mean(rel < 1.25 ** 3)),
        "n_valid":  n,
    }


# ---- epipolar / rectification quality --------------------------------------

def epipolar_metrics(left: np.ndarray, right: np.ndarray,
                     n_features: int = 1500, ratio: float = 0.75) -> Dict[str, float]:
    """Rectification quality via ORB matches between a stereo pair.

    For a well-rectified pair, corresponding points have the same y-coordinate
    (horizontal epipolar lines). We report:
      - mean/median absolute vertical disparity (|dy|) in pixels
      - fraction of inliers under 1px / 2px / 3px vertical error
      - Sampson distance to the canonical rectified F = [[0,0,0],[0,0,-1],[0,1,0]]
      - n_matches actually used
    """
    if left.ndim == 3:  left  = cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY)
    if right.ndim == 3: right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=10)
    kpL, dL = orb.detectAndCompute(left,  None)
    kpR, dR = orb.detectAndCompute(right, None)
    nan = {k: float("nan") for k in ("mean_dy", "median_dy", "p1", "p2", "p3", "sampson_rect")}
    if dL is None or dR is None or len(kpL) < 8 or len(kpR) < 8:
        return nan | {"n_matches": 0}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(dL, dR, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
    if len(good) < 8:
        return nan | {"n_matches": len(good)}

    pl = np.array([kpL[m.queryIdx].pt for m in good], dtype=np.float64)
    pr = np.array([kpR[m.trainIdx].pt for m in good], dtype=np.float64)
    dy = np.abs(pl[:, 1] - pr[:, 1])
    # Sampson distance under the canonical rectified F: F x_r = (0, x_r[0]*1, -x_r[1])^T-ish.
    # For rectified pairs the algebraic residual is simply y_l - y_r.
    return {
        "mean_dy":      float(np.mean(dy)),
        "median_dy":    float(np.median(dy)),
        "p1":           float(np.mean(dy < 1.0)),
        "p2":           float(np.mean(dy < 2.0)),
        "p3":           float(np.mean(dy < 3.0)),
        "sampson_rect": float(np.mean(dy * dy) ** 0.5),
        "n_matches":    int(len(good)),
    }


# ---- per-pair driver --------------------------------------------------------

def load_depth_png(path: Path, scale: int = 256) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    return raw.astype(np.float32) / float(scale)


def _evaluate_one(args) -> Optional[Dict[str, float]]:
    lp, rp, pred_path, gt_path, scale, with_epi = args
    if not pred_path.exists() or not gt_path.exists():
        return None
    pred = load_depth_png(pred_path, scale=scale)
    gt   = load_depth_png(gt_path,   scale=scale)
    if pred.shape != gt.shape:
        return None
    out = {"name": lp.name}
    out.update(depth_metrics(pred, gt))
    if with_epi:
        left  = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)
        if left is not None and right is not None:
            out.update({f"epi_{k}": v for k, v in epipolar_metrics(left, right).items()})
    return out


def aggregate(results: List[Dict[str, float]]) -> Dict[str, float]:
    """Weighted mean of depth metrics by n_valid; simple mean of epipolar metrics."""
    depth_keys = ("abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "d1", "d2", "d3")
    epi_keys   = ("epi_mean_dy", "epi_median_dy", "epi_p1", "epi_p2", "epi_p3", "epi_sampson_rect")
    agg: Dict[str, float] = {}
    weights = np.array([r.get("n_valid", 0) for r in results], dtype=np.float64)
    total_w = float(weights.sum()) or 1.0
    for k in depth_keys:
        vals = np.array([r.get(k, np.nan) for r in results], dtype=np.float64)
        m = np.isfinite(vals)
        agg[k] = float(np.nansum(vals[m] * weights[m]) / max(weights[m].sum(), 1.0))
    for k in epi_keys:
        vals = np.array([r.get(k, np.nan) for r in results], dtype=np.float64)
        m = np.isfinite(vals)
        agg[k] = float(np.mean(vals[m])) if m.any() else float("nan")
    agg["n_pairs"]     = int(len(results))
    agg["total_valid"] = int(total_w)
    return agg


def run(pairs: List[Tuple[Path, Path]], pred_dir: Path, gt_dir: Path,
        parallel: bool = False, workers: Optional[int] = None,
        scale: int = 256, with_epipolar: bool = True) -> Tuple[List[Dict], Dict]:
    items = []
    for lp, rp in pairs:
        sess = session_of(lp.name)
        stem = Path(lp.name).stem
        items.append((lp, rp, pred_dir / sess / f"{stem}.png", gt_dir / sess / f"{stem}.png",
                      scale, with_epipolar))

    if parallel:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(tqdm(ex.map(_evaluate_one, items, chunksize=4),
                                total=len(items), desc="Eval (par)", unit="pair"))
    else:
        results = [_evaluate_one(it) for it in tqdm(items, desc="Eval (seq)", unit="pair")]
    results = [r for r in results if r is not None]
    return results, aggregate(results)


def parse_args() -> argparse.Namespace:
    base = Path(__file__).parent / "Data"
    p = argparse.ArgumentParser(description="Evaluate predicted depth maps vs KITTI ground truth.")
    p.add_argument("--left-dir",  type=Path, default=base / "left-image-half-size"  / "left-image-half-size")
    p.add_argument("--right-dir", type=Path, default=base / "right-image-half-size" / "right-image-half-size")
    p.add_argument("--pred-dir",  type=Path, default=base / "predicted-depth-half-size")
    p.add_argument("--gt-dir",    type=Path, default=base / "depth-map-half-size" / "depth-map-half-size")
    p.add_argument("--report",    type=Path, default=base / "evaluation-report.json")
    p.add_argument("--no-epipolar", action="store_true")
    p.add_argument("--parallel", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def print_summary(agg: Dict[str, float]) -> None:
    print("\n--- Depth metrics (lower is better, δ higher is better) ---")
    for k in ("abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "d1", "d2", "d3"):
        print(f"  {k:<10s}: {agg[k]:.4f}")
    print("\n--- Epipolar / rectification quality ---")
    for k in ("epi_mean_dy", "epi_median_dy", "epi_p1", "epi_p2", "epi_p3", "epi_sampson_rect"):
        v = agg.get(k, float("nan"))
        print(f"  {k:<18s}: {v:.4f}")
    print(f"\n  pairs evaluated   : {agg['n_pairs']}")
    print(f"  valid depth pixels: {agg['total_valid']}")


def main() -> None:
    a = parse_args()
    pairs = preprocess_run(a.left_dir, a.right_dir, parallel=a.parallel, workers=a.workers)
    if a.limit:
        pairs = pairs[: a.limit]
    per_pair, agg = run(pairs, a.pred_dir, a.gt_dir,
                        parallel=a.parallel, workers=a.workers,
                        with_epipolar=not a.no_epipolar)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps({"aggregate": agg, "per_pair": per_pair}, indent=2))
    print_summary(agg)
    print(f"\nReport written to {a.report}")


if __name__ == "__main__":
    main()
