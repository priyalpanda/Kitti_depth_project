"""End-to-end orchestrator: pre-process -> depth estimation -> evaluation.

Run all stages or any subset. ``--mode parallel`` parallelises *within* each stage
(each pair is processed by an independent worker); ``--mode sequential`` keeps the
classic single-process loop. Stages always run in dependency order.
"""
import argparse
import json
import time
from pathlib import Path
from typing import List

import pre_process
import depth_estimation
import evaluation


STAGE_ORDER = ["pre", "depth", "eval"]


def parse_stages(spec: str) -> List[str]:
    if spec.lower() == "all":
        return list(STAGE_ORDER)
    chosen = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [s for s in chosen if s not in STAGE_ORDER]
    if bad:
        raise SystemExit(f"Unknown stage(s): {bad}. Choose from {STAGE_ORDER} or 'all'.")
    return [s for s in STAGE_ORDER if s in chosen]


def parse_args() -> argparse.Namespace:
    base = Path(__file__).parent / "Data"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stages", default="all", help="Comma list from {pre,depth,eval} or 'all'.")
    p.add_argument("--mode", choices=("sequential", "parallel"), default="sequential")
    p.add_argument("--workers", type=int, default=None)

    p.add_argument("--left-dir",  type=Path, default=base / "left-image-half-size"  / "left-image-half-size")
    p.add_argument("--right-dir", type=Path, default=base / "right-image-half-size" / "right-image-half-size")
    p.add_argument("--calib-dir", type=Path, default=base / "half-image-calib" / "half-image-calib")
    p.add_argument("--gt-dir",    type=Path, default=base / "depth-map-half-size" / "depth-map-half-size")
    p.add_argument("--pred-dir",  type=Path, default=base / "predicted-depth-half-size")
    p.add_argument("--preproc-dir", type=Path, default=base / "pre-processed-images")
    p.add_argument("--report",    type=Path, default=base / "evaluation-report.json")

    p.add_argument("--save-preproc", action="store_true", help="Persist pre-processed (grayscale) pairs to disk.")
    p.add_argument("--algorithm", choices=sorted(depth_estimation.ALGORITHMS), default="sgbm",
                   help=f"Stereo matcher to use. Registered: {sorted(depth_estimation.ALGORITHMS)}.")
    p.add_argument("--num-disparities", type=int, default=192)
    p.add_argument("--block-size", type=int, default=5)
    p.add_argument("--no-epipolar", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N pairs (debug).")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    stages = parse_stages(a.stages)
    parallel = a.mode == "parallel"
    t0 = time.perf_counter()
    timings = {}

    print(f"[pipeline] stages={stages}  mode={a.mode}  workers={a.workers or 'default'}")

    t = time.perf_counter()
    pairs = pre_process.run(a.left_dir, a.right_dir, a.preproc_dir,
                            save=a.save_preproc and "pre" in stages,
                            parallel=parallel, workers=a.workers)
    if a.limit:
        pairs = pairs[: a.limit]
    timings["pre"] = time.perf_counter() - t
    print(f"[pre]   validated {len(pairs)} pairs in {timings['pre']:.1f}s"
          + (f"  (saved to {a.preproc_dir})" if a.save_preproc and 'pre' in stages else ""))

    if "depth" in stages:
        t = time.perf_counter()
        n = depth_estimation.run(pairs, a.calib_dir, a.pred_dir,
                                 parallel=parallel, workers=a.workers,
                                 algorithm=a.algorithm,
                                 algo_kwargs=dict(num_disparities=a.num_disparities,
                                                  block_size=a.block_size))
        timings["depth"] = time.perf_counter() - t
        print(f"[depth] {a.algorithm}: saved {n} maps in {timings['depth']:.1f}s -> {a.pred_dir}")

    if "eval" in stages:
        t = time.perf_counter()
        per_pair, agg = evaluation.run(pairs, a.pred_dir, a.gt_dir,
                                       parallel=parallel, workers=a.workers,
                                       with_epipolar=not a.no_epipolar)
        timings["eval"] = time.perf_counter() - t
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps({"aggregate": agg, "timings": timings,
                                        "config": {k: str(v) for k, v in vars(a).items()},
                                        "per_pair": per_pair}, indent=2))
        evaluation.print_summary(agg)
        print(f"\n[eval]  report: {a.report}  ({timings['eval']:.1f}s)")

    print(f"\n[pipeline] done in {time.perf_counter() - t0:.1f}s  per-stage: "
          + ", ".join(f"{k}={v:.1f}s" for k, v in timings.items()))


if __name__ == "__main__":
    main()
