import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda iterable, **kwargs: iterable

from pre_process import run as preprocess_run

SESSION_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_")


def session_of(name: str) -> str:
    m = SESSION_RE.match(Path(name).name)
    if not m:
        raise ValueError(f"Cannot infer session from {name!r}")
    return m.group(1)


def parse_calibration(calib_path: Path) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for line in calib_path.read_text().splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        try:
            arr = np.fromstring(val, sep=" ", dtype=np.float64)
        except ValueError:
            continue
        if arr.size:
            out[key.strip()] = arr
    return out


def stereo_geometry(calib: Dict[str, np.ndarray]) -> Tuple[float, float, float, float, float]:
    """Return (fx, fy, cx, cy, baseline_m) for the rectified left/right cameras.

    The physical baseline comes from ``T_103`` (translation of the right camera in the
    left's frame). The projection-matrix translation ``P_rect_103[0,3]`` is not used
    here because in this dataset's half-size calibration it does not equal ``-fx * B``.
    """
    pl = calib["P_rect_101"].reshape(3, 4)
    fx, fy, cx, cy = pl[0, 0], pl[1, 1], pl[0, 2], pl[1, 2]
    if "T_103" in calib:
        baseline = float(abs(calib["T_103"][0]))
    else:
        pr = calib["P_rect_103"].reshape(3, 4)
        baseline = float(abs(pr[0, 3] / pr[0, 0]))
    return float(fx), float(fy), float(cx), float(cy), baseline


def load_calibrations(calib_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    return {p.stem: parse_calibration(p) for p in sorted(calib_dir.glob("*.txt"))}


# ---- stereo matching algorithm registry ------------------------------------

# Every algorithm: ``f(left_gray, right_gray, **kwargs) -> float32 disparity (NaN for invalid)``.
StereoAlgorithm = Callable[..., np.ndarray]
ALGORITHMS: Dict[str, StereoAlgorithm] = {}


def register_algorithm(name: str) -> Callable[[StereoAlgorithm], StereoAlgorithm]:
    """Decorator: register a stereo algorithm under ``name`` so it can be selected via CLI/API."""
    def deco(fn: StereoAlgorithm) -> StereoAlgorithm:
        ALGORITHMS[name] = fn
        return fn
    return deco


def _to_disparity(raw_int16: np.ndarray) -> np.ndarray:
    """OpenCV matchers emit int16 disparity at scale 16; normalise to float32 px, NaN for invalid."""
    out = raw_int16.astype(np.float32) / 16.0
    out[out <= 0] = np.nan
    return out


@register_algorithm("sgbm")
def _sgbm(left: np.ndarray, right: np.ndarray,
          num_disparities: int = 192, block_size: int = 5, **_) -> np.ndarray:
    """Semi-Global Block Matching (default). Best balance of quality and speed on KITTI-scale data."""
    ch = 1
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * ch * block_size * block_size,
        P2=32 * ch * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return _to_disparity(matcher.compute(left, right))


@register_algorithm("bm")
def _bm(left: np.ndarray, right: np.ndarray,
        num_disparities: int = 128, block_size: int = 15, **_) -> np.ndarray:
    """Classic block matching. Faster than SGBM but noisier; block_size is forced odd (BM requirement)."""
    bs = block_size if block_size % 2 == 1 else block_size + 1
    matcher = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=bs)
    matcher.setUniquenessRatio(10)
    matcher.setSpeckleWindowSize(100)
    matcher.setSpeckleRange(32)
    matcher.setTextureThreshold(10)
    return _to_disparity(matcher.compute(left, right))


def compute_disparity(left: np.ndarray, right: np.ndarray,
                      algorithm: str = "sgbm", **algo_kwargs) -> np.ndarray:
    """Dispatch to the named stereo algorithm. Returns float32 disparity (NaN for invalid)."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algorithm!r}. Available: {sorted(ALGORITHMS)}")
    if left.ndim == 3:
        left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    if right.ndim == 3:
        right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    return ALGORITHMS[algorithm](left, right, **algo_kwargs)


def disparity_to_depth(disp: np.ndarray, fx: float, baseline: float,
                       z_min: float = 0.5, z_max: float = 200.0) -> np.ndarray:
    """Convert disparity (pixels) to depth (meters). Invalid -> 0."""
    depth = np.zeros_like(disp, dtype=np.float32)
    valid = np.isfinite(disp) & (disp > 0)
    depth[valid] = (fx * baseline) / disp[valid]
    depth[(depth < z_min) | (depth > z_max)] = 0.0
    return depth


def save_depth_png(depth_m: np.ndarray, path: Path, scale: int = 256) -> None:
    """Save depth (meters) as 16-bit PNG using KITTI convention: pixel = depth * scale."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(depth_m * scale, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    cv2.imwrite(str(path), arr)


# ---- per-pair worker (top-level for ProcessPoolExecutor pickling) ----------

def _process_one(args) -> Tuple[str, bool]:
    lp, rp, out_dir, calib_geom, algorithm, algo_kwargs, scale = args
    fx, _, _, _, baseline = calib_geom
    left = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)
    if left is None or right is None or left.shape != right.shape:
        return (lp.name, False)
    disp = compute_disparity(left, right, algorithm=algorithm, **algo_kwargs)
    depth = disparity_to_depth(disp, fx, baseline)
    out_path = Path(out_dir) / session_of(lp.name) / (Path(lp.name).stem + ".png")
    save_depth_png(depth, out_path, scale=scale)
    return (lp.name, True)


def run(pairs: List[Tuple[Path, Path]], calib_dir: Path, output_dir: Path,
        parallel: bool = False, workers: Optional[int] = None,
        algorithm: str = "sgbm", algo_kwargs: Optional[Dict] = None,
        scale: int = 256) -> int:
    """Estimate depth maps for each validated pair. Returns count saved."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algorithm!r}. Available: {sorted(ALGORITHMS)}")
    algo_kwargs = algo_kwargs or {}
    calibs = load_calibrations(calib_dir)
    items = []
    for lp, rp in pairs:
        sess = session_of(lp.name)
        if sess not in calibs:
            continue
        items.append((lp, rp, output_dir, stereo_geometry(calibs[sess]),
                      algorithm, algo_kwargs, scale))

    suffix = "par" if parallel else "seq"
    if parallel:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(tqdm(ex.map(_process_one, items, chunksize=4),
                                total=len(items), desc=f"Depth [{algorithm}, {suffix}]", unit="pair"))
    else:
        results = [_process_one(it) for it in tqdm(items, desc=f"Depth [{algorithm}, {suffix}]", unit="pair")]
    return sum(int(ok) for _, ok in results)


def parse_args() -> argparse.Namespace:
    base = Path(__file__).parent / "Data"
    p = argparse.ArgumentParser(description="Estimate depth maps via a registered stereo matching algorithm.")
    p.add_argument("--left-dir",  type=Path, default=base / "left-image-half-size"  / "left-image-half-size")
    p.add_argument("--right-dir", type=Path, default=base / "right-image-half-size" / "right-image-half-size")
    p.add_argument("--calib-dir", type=Path, default=base / "half-image-calib" / "half-image-calib")
    p.add_argument("--output-dir", type=Path, default=base / "predicted-depth-half-size")
    p.add_argument("--algorithm", choices=sorted(ALGORITHMS), default="sgbm",
                   help=f"Stereo matcher to use. Registered: {sorted(ALGORITHMS)}.")
    p.add_argument("--num-disparities", type=int, default=192)
    p.add_argument("--block-size", type=int, default=5)
    p.add_argument("--parallel", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="Process only the first N pairs (debug).")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    pairs = preprocess_run(a.left_dir, a.right_dir, parallel=a.parallel, workers=a.workers)
    if a.limit:
        pairs = pairs[: a.limit]
    n = run(pairs, a.calib_dir, a.output_dir,
            parallel=a.parallel, workers=a.workers,
            algorithm=a.algorithm,
            algo_kwargs=dict(num_disparities=a.num_disparities, block_size=a.block_size))
    print(f"Saved {n} depth maps ({a.algorithm}) to {a.output_dir}")


if __name__ == "__main__":
    main()
