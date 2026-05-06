"""Generate per-case VTK files for the Lamé hollow-sphere problem.

For each sampled parameter set (p_i, p_e, E, nu), solves the Lamé analytical
stress field and writes a self-contained binary .vtu file that bundles the
sphere FEM mesh geometry with four point-data arrays:

    sigma_vm    — von Mises stress [Pa]
    u_r         — radial displacement [m]
    sigma_r     — radial stress [Pa]
    sigma_theta — hoop stress [Pa]

The four physics parameters plus case_id are stored as VTK field-data scalars
so the file is fully self-contained and can be opened in ParaView.

Each worker loads the base mesh once (via ProcessPoolExecutor initializer),
then computes and writes VTK files without sending large arrays over IPC.

Output:
    dataset/lame_sphere_cases/case_{case_id:05d}.vtu   (one per case)

Usage (from repo root):
    python scripts/generate_lame_sphere_fields.py
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyvista as pv
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neural_operators.data.lame_sphere import A, B, N_PTS, sample_params, solve_case

_VTK_BASE_PATH = str(PROJECT_ROOT / "data" / "sphere-FEMMeshGmsh.vtk")

# Per-worker globals set by the initializer — never cross process boundaries.
_BASE_MESH:   pv.UnstructuredGrid | None = None
_CASES_DIR:   str | None                 = None


def _worker_init(vtk_base_path: str, cases_dir: str) -> None:
    global _BASE_MESH, _CASES_DIR
    _BASE_MESH = pv.read(vtk_base_path)
    _BASE_MESH.points *= 1e-3  # mm → m, once per worker
    _CASES_DIR = cases_dir


def _solve_and_write(args: tuple) -> int:
    """Compute Lamé fields for one case and save a .vtu file. Returns case_id."""
    case_id, p_i, p_e, E, nu = args
    row = solve_case(args)[0]

    mesh = _BASE_MESH.copy()

    # Point-data arrays: one value per mesh vertex
    mesh.point_data["sigma_vm"]    = row["sigma_vm"]     # Pa
    mesh.point_data["u_r"]         = row["u_r"]          # m
    mesh.point_data["sigma_r"]     = row["sigma_r"]      # Pa
    mesh.point_data["sigma_theta"] = row["sigma_theta"]  # Pa

    # Field-data scalars: one value per dataset (the loading parameters)
    mesh.field_data["case_id"] = np.array([case_id], dtype=np.int32)
    mesh.field_data["p_i"]     = np.array([p_i])   # Pa
    mesh.field_data["p_e"]     = np.array([p_e])   # Pa
    mesh.field_data["E"]       = np.array([E])      # Pa
    mesh.field_data["nu"]      = np.array([nu])

    out = Path(_CASES_DIR) / f"case_{case_id:05d}.vtu"
    mesh.save(str(out), binary=True)
    return case_id


def main() -> None:
    cfg_path = PROJECT_ROOT / "configs" / "lame_sphere.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ds        = cfg["dataset"]
    N_CASES   = int(ds["n_cases"])
    SEED      = int(ds["seed"])
    N_WORKERS = ds.get("n_workers") or os.cpu_count()
    CASES_DIR = PROJECT_ROOT / ds.get("cases_dir", "dataset/lame_sphere_cases")
    CASES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Lamé Hollow Sphere — Field Generator")
    print("=" * 60)
    print(f"  Cases      : {N_CASES:,}")
    print(f"  Geometry   : a={A} m  b={B} m")
    print(f"  Mesh pts   : {N_PTS:,}  (sphere-FEMMeshGmsh.vtk)")
    print(f"  Workers    : {N_WORKERS}  (each loads base mesh once)")
    print(f"  Output     : {CASES_DIR}")
    print("=" * 60)

    # Estimate from a single case before committing
    t_bm = time.time()
    solve_case(sample_params(1, 0)[0])
    bm_ms = (time.time() - t_bm) * 1000
    est_s = N_CASES * (bm_ms / 1000) / N_WORKERS
    print(f"  Benchmark  : {bm_ms:.1f} ms/case → ~{est_s:.0f}s wall ({est_s/60:.1f} min)\n")

    args_list = sample_params(N_CASES, SEED)
    done = 0
    t0   = time.time()

    with ProcessPoolExecutor(
        max_workers=N_WORKERS,
        initializer=_worker_init,
        initargs=(_VTK_BASE_PATH, str(CASES_DIR)),
    ) as pool:
        futs = {pool.submit(_solve_and_write, a): a[0] for a in args_list}
        for fut in as_completed(futs):
            fut.result()   # propagate any worker exception immediately
            done += 1
            if done % 100 == 0 or done == N_CASES:
                elapsed = time.time() - t0
                rate    = done / elapsed
                eta     = (N_CASES - done) / rate if rate > 0 else 0.0
                print(f"  [{done:>5}/{N_CASES}]  {elapsed:5.0f}s  "
                      f"{rate:.1f} cases/s  ETA {eta:.0f}s")

    elapsed  = time.time() - t0
    total_mb = sum(p.stat().st_size for p in CASES_DIR.glob("case_*.vtu")) / 1e6
    print(f"\nDone — {N_CASES:,} VTK files written in {elapsed:.1f}s")
    print(f"Total size : {total_mb:.0f} MB  ({total_mb/N_CASES:.1f} MB/case)")
    print(f"Output dir : {CASES_DIR}")


if __name__ == "__main__":
    main()