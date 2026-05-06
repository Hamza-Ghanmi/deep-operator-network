"""Build train / val / test Parquet splits from per-case VTK files.

Reads the .vtu files produced by generate_lame_sphere_fields.py from
dataset/lame_sphere_cases/ and assembles the three Parquet dataset splits:

    dataset/lame_sphere_train.parquet  — 70 % of cases  (default)
    dataset/lame_sphere_val.parquet    — 15 % of cases
    dataset/lame_sphere_test.parquet   — 15 % of cases

The train/val/test assignment is deterministic: case_ids are shuffled with
the same seed used during field generation so the splits are stable across
re-runs, even if VTK files were written in a different order.

VTK files are read concurrently with a ProcessPoolExecutor (CPU-bound:
zlib decompression inside VTK bypasses the GIL only with separate processes).
Rows are flushed to Parquet in batches to bound peak memory.

Usage (from repo root):
    python scripts/generate_lame_sphere_dataset.py
    python scripts/generate_lame_sphere_dataset.py --workers 4 --batch-size 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyvista as pv
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neural_operators.data.lame_sphere import SCHEMA, rows_to_table

OUTPUT_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR.mkdir(exist_ok=True)

# Default: 8 processes, 20 rows per batch.
# VTU reading is CPU-bound (zlib decompression); ProcessPoolExecutor bypasses the GIL.
# Each worker process: ~300 MB (pyvista imports + in-flight VTU).
# Pending buffers in main process: splits × batch × 6.5 MB.
# At defaults (8 workers, batch 20): ~2.4 GB + 3×20×6.5 = ~2.8 GB peak.
# Use --workers 2 --batch-size 5 on low-RAM machines (<8 GB).
DEFAULT_WORKERS    = 8
DEFAULT_BATCH_SIZE = 20


def _read_vtk_case(path: Path) -> dict:
    """Read one .vtu file; returns a row dict compatible with rows_to_table."""
    mesh = pv.read(str(path))
    return {
        "case_id":     int(mesh.field_data["case_id"][0]),
        "p_i":         float(mesh.field_data["p_i"][0]),
        "p_e":         float(mesh.field_data["p_e"][0]),
        "E":           float(mesh.field_data["E"][0]),
        "nu":          float(mesh.field_data["nu"][0]),
        "sigma_vm":    mesh.point_data["sigma_vm"].astype(np.float32),
        "u_r":         mesh.point_data["u_r"].astype(np.float32),
        "sigma_r":     mesh.point_data["sigma_r"].astype(np.float32),
        "sigma_theta": mesh.point_data["sigma_theta"].astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lame_sphere Parquet splits from VTU files.")
    parser.add_argument("--workers",    type=int, default=None,
                        help=f"Number of reader processes (default: {DEFAULT_WORKERS}; use 2 on low-RAM machines)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"Rows buffered per split before flushing (default: {DEFAULT_BATCH_SIZE})")
    args = parser.parse_args()

    cfg_path = PROJECT_ROOT / "configs" / "lame_sphere.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ds        = cfg["dataset"]
    SEED      = int(ds["seed"])
    SPLIT     = tuple(ds["split"])
    # CLI flag > yaml config > DEFAULT_WORKERS (cap at logical CPU count)
    N_WORKERS = min(args.workers or ds.get("n_workers") or DEFAULT_WORKERS, os.cpu_count() or 8)
    BATCH_SIZE = args.batch_size or DEFAULT_BATCH_SIZE
    CASES_DIR = PROJECT_ROOT / ds.get("cases_dir", "dataset/lame_sphere_cases")

    vtk_files = sorted(CASES_DIR.glob("case_*.vtu"))
    N_CASES   = len(vtk_files)

    if N_CASES == 0:
        print(f"No .vtu files found in {CASES_DIR}")
        print("Run generate_lame_sphere_fields.py first.")
        sys.exit(1)

    # Deterministic split: shuffle case_ids by rank (same seed as field generation)
    all_ids = np.array([int(p.stem.split("_")[1]) for p in vtk_files], dtype=np.int64)
    rng     = np.random.default_rng(SEED)
    order   = rng.permutation(len(all_ids))

    n_train = int(round(SPLIT[0] * N_CASES))
    n_val   = int(round(SPLIT[1] * N_CASES))
    n_test  = N_CASES - n_train - n_val

    split_of: dict[int, int] = {}
    for rank, idx in enumerate(order):
        cid = int(all_ids[idx])
        if rank < n_train:
            split_of[cid] = 0
        elif rank < n_train + n_val:
            split_of[cid] = 1
        else:
            split_of[cid] = 2

    split_names = ("train", "val", "test")
    split_ns    = (n_train, n_val, n_test)
    paths       = [OUTPUT_DIR / f"lame_sphere_{s}.parquet" for s in split_names]
    writers     = [pq.ParquetWriter(p, SCHEMA) for p in paths]
    pending     = [[], [], []]
    n_written   = [0, 0, 0]

    peak_mb = N_WORKERS * 300 + 3 * BATCH_SIZE * 6.5
    print("=" * 60)
    print("  Lamé Hollow Sphere — Dataset Builder")
    print("=" * 60)
    print(f"  Source     : {CASES_DIR.name}/  ({N_CASES:,} .vtu files)")
    print(f"  Split      : train {SPLIT[0]:.0%} / val {SPLIT[1]:.0%} / test {SPLIT[2]:.0%}")
    print(f"               {n_train} / {n_val} / {n_test} cases")
    print(f"  Workers    : {N_WORKERS}  (processes)   batch-size: {BATCH_SIZE}")
    print(f"  Peak RAM   : ~{peak_mb:.0f} MB  (use --workers 2 --batch-size 5 to minimise)")
    print("=" * 60)

    def _flush(si: int) -> None:
        if not pending[si]:
            return
        writers[si].write_table(rows_to_table(pending[si]))
        n_written[si] += len(pending[si])
        pending[si].clear()

    done = 0
    t0   = time.time()
    # Submit files in chunks so completed Future results don't accumulate.
    # Each result dict holds ~6.6 MB; submitting all N_CASES at once would
    # buffer N_CASES × 6.6 MB ≈ 33 GB for 5 000 cases.
    submit_chunk = N_WORKERS * 4   # ≤ submit_chunk × 6.6 MB live at any time

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for chunk_start in range(0, N_CASES, submit_chunk):
            chunk = vtk_files[chunk_start : chunk_start + submit_chunk]
            futs  = {pool.submit(_read_vtk_case, p): p for p in chunk}
            for fut in as_completed(futs):
                row = fut.result()
                si  = split_of[row["case_id"]]
                pending[si].append(row)
                if len(pending[si]) >= BATCH_SIZE:
                    _flush(si)
                done += 1
                if done % 200 == 0 or done == N_CASES:
                    elapsed = time.time() - t0
                    rate    = done / elapsed
                    eta     = (N_CASES - done) / rate if rate > 0 else 0.0
                    print(f"  [{done:>5}/{N_CASES}]  {elapsed:5.0f}s  "
                          f"{rate:.1f} files/s  ETA {eta:.0f}s")

    for si in range(3):
        _flush(si)
        writers[si].close()

    elapsed = time.time() - t0
    print(f"\nDataset built in {elapsed:.1f}s\n")
    for si, (name, path, n_c) in enumerate(zip(split_names, paths, split_ns)):
        mb = path.stat().st_size / 1e6
        print(f"  Saved {name:5s}: {n_written[si]:>6} rows  ({n_c} cases) "
              f"→ {path.name}  [{mb:.1f} MB]")
    print("\nDone.")


if __name__ == "__main__":
    main()