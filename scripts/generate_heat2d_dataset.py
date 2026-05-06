"""Generate the 2D heat equation dataset and split into train / val / test Parquet files.

Reads generation configuration from configs/heat2d.yaml by default, or from
a path supplied via --config (e.g. configs/heat2d_smoke.yaml for a quick demo).

Output (dataset/):
  heat2d_train.parquet   — 70 % of cases
  heat2d_val.parquet     — 15 % of cases
  heat2d_test.parquet    — 15 % of cases

Split is by case_id — all 10 time steps of a case stay in the same split.

Usage (from repo root):
    python scripts/generate_heat2d_dataset.py                              # full run
    python scripts/generate_heat2d_dataset.py --config configs/heat2d_smoke.yaml  # 50-case demo
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neural_operators.data.heat2d import (
    NX, NY, N_TIMES, T_START, T_END,
    solve_case, sample_params, save_split,
)

OUTPUT_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 2D heat equation dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "heat2d.yaml",
        help="Path to YAML config (default: configs/heat2d.yaml)",
    )
    cfg_path = parser.parse_args().config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ds  = cfg["dataset"]
    N_CASES   = int(ds["n_cases"])
    SEED      = int(ds["seed"])
    N_WORKERS = ds.get("n_workers") or os.cpu_count()
    SPLIT     = tuple(ds["split"])

    print("=" * 60)
    print("  2D Heat Equation — Dataset Generator (vectorized)")
    print("=" * 60)
    print(f"  Cases      : {N_CASES:,}")
    print(f"  Time steps : {N_TIMES}  ({T_START}s → {T_END}s, log-spaced)")
    print(f"  Grid       : {NX}×{NY}")
    print(f"  Records    : {N_CASES * N_TIMES:,}")
    print(f"  Workers    : {N_WORKERS}")
    print(f"  Split      : train {SPLIT[0]:.0%} / val {SPLIT[1]:.0%} / test {SPLIT[2]:.0%}")
    print("=" * 60)

    t_bm = time.time()
    solve_case(sample_params(1, 0)[0])
    bm  = time.time() - t_bm
    est = N_CASES * bm / N_WORKERS
    print(f"  Benchmark  : {bm:.3f}s/case → estimated {est:.0f}s ({est/60:.1f} min)\n")

    args_list = sample_params(N_CASES, SEED)
    all_rows  = [None] * N_CASES
    done = 0
    t0   = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(solve_case, a): a[0] for a in args_list}
        for fut in as_completed(futures):
            cid = futures[fut]
            all_rows[cid] = fut.result()
            done += 1
            if done % 500 == 0 or done == N_CASES:
                elapsed = time.time() - t0
                rate    = done / elapsed
                eta     = (N_CASES - done) / rate if rate > 0 else 0
                print(f"  [{done:>5}/{N_CASES}]  {elapsed:6.1f}s  {rate:.1f} cases/s  ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nGeneration complete: {N_CASES} cases in {elapsed:.1f}s ({elapsed/N_CASES:.3f}s/case)\n")

    flat_rows = [row for case_rows in all_rows for row in case_rows]

    rng      = np.random.default_rng(SEED)
    ids      = np.arange(N_CASES); rng.shuffle(ids)
    n_train  = int(round(SPLIT[0] * N_CASES))
    n_val    = int(round(SPLIT[1] * N_CASES))
    n_test   = N_CASES - n_train - n_val
    train_ids = set(ids[:n_train].tolist())
    val_ids   = set(ids[n_train : n_train + n_val].tolist())

    train_rows = [r for r in flat_rows if r["case_id"] in train_ids]
    val_rows   = [r for r in flat_rows if r["case_id"] in val_ids]
    test_rows  = [r for r in flat_rows if r["case_id"] not in train_ids and r["case_id"] not in val_ids]

    print(f"Split (cases):  train={n_train}  val={n_val}  test={n_test}")
    print(f"Split (rows):   train={len(train_rows)}  val={len(val_rows)}  test={len(test_rows)}\n")

    save_split(train_rows, OUTPUT_DIR / "heat2d_train.parquet", "train", n_train)
    save_split(val_rows,   OUTPUT_DIR / "heat2d_val.parquet",   "val",   n_val)
    save_split(test_rows,  OUTPUT_DIR / "heat2d_test.parquet",  "test",  n_test)

    total_mb = sum(
        (OUTPUT_DIR / f"heat2d_{s}.parquet").stat().st_size
        for s in ("train", "val", "test")
    ) / 1e6
    print(f"\nTotal size: {total_mb:.1f} MB  |  {N_CASES * N_TIMES:,} records")
    print("Done.")


if __name__ == "__main__":
    main()