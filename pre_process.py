import argparse
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda iterable, **kwargs: iterable

VALID_EXT = ".jpg .jpeg .png".split()


def timestamp_key(name: str) -> str:
    stem = Path(name).stem
    return stem.split("_", 1)[1] if "_" in stem else ""


def match_timestamps(left_dir: Path, right_dir: Path) -> List[Tuple[str, str]]:
    left = {timestamp_key(p.name): p.name for p in sorted(left_dir.iterdir()) if p.suffix.lower() in VALID_EXT}
    right = {timestamp_key(p.name): p.name for p in sorted(right_dir.iterdir()) if p.suffix.lower() in VALID_EXT}
    return [(left[k], right[k]) for k in sorted(left.keys() & right.keys()) if k]


def remove_duplicates(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen_l, seen_r, result = set(), set(), []
    for l, r in pairs:
        if l not in seen_l and r not in seen_r:
            seen_l.add(l); seen_r.add(r); result.append((l, r))
    return result


def _validate_one(args: Tuple[Path, Path]) -> Optional[Tuple[Path, Path]]:
    lp, rp = args
    li = cv2.imread(str(lp))
    ri = cv2.imread(str(rp))
    if li is None or ri is None or li.shape[:2] != ri.shape[:2]:
        return None
    return (lp, rp)


def validate_pairs(pairs: List[Tuple[str, str]], left_dir: Path, right_dir: Path,
                   parallel: bool = False, workers: Optional[int] = None) -> List[Tuple[Path, Path]]:
    items = [(left_dir / l, right_dir / r) for l, r in pairs]
    if parallel:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(tqdm(ex.map(_validate_one, items, chunksize=16),
                                total=len(items), desc="Validating", unit="pair"))
    else:
        results = [_validate_one(it) for it in tqdm(items, desc="Validating", unit="pair")]
    return [r for r in results if r is not None]


def _save_one(args: Tuple[Path, Path, Path, bool]) -> None:
    lp, rp, out_dir, gray = args
    li = cv2.imread(str(lp))
    ri = cv2.imread(str(rp))
    if li is None or ri is None:
        return
    if gray:
        li = cv2.cvtColor(li, cv2.COLOR_BGR2GRAY)
        ri = cv2.cvtColor(ri, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(out_dir / "left" / lp.name), li)
    cv2.imwrite(str(out_dir / "right" / rp.name), ri)


def save_pairs(pairs: List[Tuple[Path, Path]], output_dir: Path,
               gray: bool = True, parallel: bool = False, workers: Optional[int] = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "left").mkdir(exist_ok=True)
    (output_dir / "right").mkdir(exist_ok=True)
    items = [(lp, rp, output_dir, gray) for lp, rp in pairs]
    if parallel:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            list(tqdm(ex.map(_save_one, items, chunksize=16),
                      total=len(items), desc="Saving", unit="pair"))
    else:
        for it in tqdm(items, desc="Saving", unit="pair"):
            _save_one(it)


def run(left_dir: Path, right_dir: Path, output_dir: Optional[Path] = None,
        save: bool = False, gray: bool = True, parallel: bool = False,
        workers: Optional[int] = None) -> List[Tuple[Path, Path]]:
    pairs = remove_duplicates(match_timestamps(left_dir, right_dir))
    validated = validate_pairs(pairs, left_dir, right_dir, parallel=parallel, workers=workers)
    if save:
        assert output_dir is not None, "output_dir required when save=True"
        save_pairs(validated, output_dir, gray=gray, parallel=parallel, workers=workers)
    return validated


def parse_args() -> argparse.Namespace:
    base = Path(__file__).parent / "Data"
    p = argparse.ArgumentParser(description="Validate and (optionally) save matched stereo pairs.")
    p.add_argument("--left-dir",  type=Path, default=base / "left-image-half-size"  / "left-image-half-size")
    p.add_argument("--right-dir", type=Path, default=base / "right-image-half-size" / "right-image-half-size")
    p.add_argument("--output-dir", type=Path, default=base / "pre-processed-images")
    p.add_argument("--save", action="store_true", help="Persist validated pairs to disk (off by default).")
    p.add_argument("--color", action="store_true", help="Save as color instead of grayscale.")
    p.add_argument("--parallel", action="store_true", help="Use multiprocessing.")
    p.add_argument("--workers", type=int, default=None)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    pairs = run(a.left_dir, a.right_dir, a.output_dir,
                save=a.save, gray=not a.color, parallel=a.parallel, workers=a.workers)
    print(f"Validated pairs: {len(pairs)}{' (saved to ' + str(a.output_dir) + ')' if a.save else ''}")


if __name__ == "__main__":
    main()
