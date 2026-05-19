import cv2
from pathlib import Path
from typing import List, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda iterable, **kwargs: iterable

VALID_EXT = ".jpg .jpeg .png".split()


def timestamp_key(name: str) -> str:
    stem = Path(name).stem
    return stem.split("_", 1)[1] if "_" in stem else ""


def ensure_output_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "left").mkdir(exist_ok=True)
    (output_dir / "right").mkdir(exist_ok=True)


def match_timestamps(left_dir: Path, right_dir: Path) -> List[Tuple[str, str]]:
    left = {timestamp_key(p.name): p.name for p in sorted(left_dir.iterdir()) if p.suffix.lower() in VALID_EXT}
    right = {timestamp_key(p.name): p.name for p in sorted(right_dir.iterdir()) if p.suffix.lower() in VALID_EXT}
    return [(left[k], right[k]) for k in sorted(left.keys() & right.keys()) if k]


def remove_duplicates(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen_l = set()
    seen_r = set()
    result = []
    for l, r in pairs:
        if l not in seen_l and r not in seen_r:
            seen_l.add(l)
            seen_r.add(r)
            result.append((l, r))
    return result


def validate_and_save_pairs(pairs: List[Tuple[str, str]], left_dir: Path, right_dir: Path, output_dir: Path) -> None:
    ensure_output_dirs(output_dir)
    saved = 0
    for l, r in tqdm(pairs, desc="Processing image pairs", unit="pair"):
        lp = left_dir / l
        rp = right_dir / r
        li = cv2.imread(str(lp))
        ri = cv2.imread(str(rp))

        #Check if images are loaded and have the same dimensions
        if li is None or ri is None:
            continue
        if li.shape[:2] != ri.shape[:2]:
            continue
        
        cv2.imwrite(str(output_dir / "left" / lp.name), cv2.cvtColor(li, cv2.COLOR_BGR2GRAY))
        cv2.imwrite(str(output_dir / "right" / rp.name), cv2.cvtColor(ri, cv2.COLOR_BGR2GRAY))
        saved += 1


def main() -> None:
    base = Path(__file__).parent / "Data"
    left_dir = base / "left-image-half-size"
    right_dir = base / "right-image-half-size"
    output_dir = base / "pre-processed-images"

    pairs = match_timestamps(left_dir, right_dir)
    pairs = remove_duplicates(pairs)
    validate_and_save_pairs(pairs, left_dir, right_dir, output_dir)


if __name__ == "__main__":
    main()
