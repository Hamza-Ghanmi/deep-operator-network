"""Lamé thick-walled hollow sphere — analytical solver and dataset utilities.

Computes exact closed-form stress and displacement fields for a hollow sphere
under combined internal and external pressure (Lamé, 1852).

Fixed geometry: inner radius a = 0.2 m, outer radius b = 0.5 m.
Variable parameters per case: (p_i, p_e, E, nu).

Query points are the FEM mesh vertices from data/sphere-FEMMeshGmsh.vtk
(VTK legacy ASCII, DATASET UNSTRUCTURED_GRID), loaded once at import time and
inherited by ProcessPoolExecutor fork workers at zero serialisation cost.
Do not import this module inside a tight loop.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyvista as pv

from neural_operators.domain.ports.physics_solver import IPhysicsSolver

__all__ = [
    "A", "B", "N_PTS",
    "PI_RANGE", "PE_RANGE", "E_RANGE", "NU_RANGE",
    "QUERY_XYZ",
    "load_sphere_mesh",
    "sample_params",
    "solve_case",
    "rows_to_table",
    "save_split",
    "SCHEMA",
    "LameSolver",
]

# ── Geometry constants (fixed across all cases) ───────────────────────────────

A: float = 0.2    # inner radius [m]
B: float = 0.5    # outer radius [m]

# ── Parameter ranges ──────────────────────────────────────────────────────────

PI_RANGE: tuple[float, float] = (1.0e6,  20.0e6)    # internal pressure [Pa]
PE_RANGE: tuple[float, float] = (0.0,     5.0e6)    # external pressure [Pa]
E_RANGE:  tuple[float, float] = (50.0e9, 250.0e9)   # Young's modulus [Pa]
NU_RANGE: tuple[float, float] = (0.20,    0.45)     # Poisson's ratio [-]

# ── Load FEM mesh query points from the sphere VTK (ASCII legacy format) ──────
# The VTK stores coordinates in millimetres; divide by 1000 to get metres.
# Inherited by fork workers at zero cost (Linux default fork start-method).

_VTK_PATH = Path(__file__).parent.parent.parent.parent.parent / "data" / "sphere-FEMMeshGmsh.vtk"


def _load_vtk_vertices(path: Path) -> np.ndarray:
    """Return all mesh vertices from a VTK file via PyVista, shape (N, 3) in metres.

    Coordinates in the file are in millimetres and are converted to metres.
    """
    mesh = pv.read(str(path))
    return (mesh.points * 1e-3).astype(np.float32)  # mm → m


QUERY_XYZ: np.ndarray = _load_vtk_vertices(_VTK_PATH)  # (N_PTS, 3) float32, metres
N_PTS: int = len(QUERY_XYZ)


def load_sphere_mesh() -> pv.UnstructuredGrid:
    """Return the FEM sphere mesh as a PyVista UnstructuredGrid (coordinates in metres)."""
    mesh = pv.UnstructuredGrid(str(_VTK_PATH))
    mesh.points *= 1e-3  # mm → m in-place
    return mesh

# Radial distances in float64 — r³ appears in denominators, precision matters.
# Clamped to [A, B]: float32 surface mesh vertices can sit slightly outside the
# shell bounds due to truncation; clamping avoids evaluating the Lamé solution
# outside its valid domain while keeping QUERY_XYZ coordinates exact for the trunk net.
_QR: np.ndarray = np.clip(
    np.linalg.norm(QUERY_XYZ, axis=1).astype(np.float64), A, B
)  # (N_PTS,)

# Precomputed geometry factors (fixed)
_A3:    float = A ** 3
_B3:    float = B ** 3
_DENOM: float = _B3 - _A3


# ── Parameter sampling ────────────────────────────────────────────────────────

def sample_params(n_cases: int, seed: int) -> list[tuple]:  # type: ignore[type-arg]
    """Sample random physics parameters for n_cases dataset entries.

    Returns a list of (case_id, p_i, p_e, E, nu) tuples, all values in SI units.
    p_e is sampled conditional on p_e < p_i to ensure a non-trivial deviatoric term.
    E is sampled log-uniformly over E_RANGE.
    """
    rng = np.random.default_rng(seed)

    p_i = rng.uniform(PI_RANGE[0], PI_RANGE[1], size=n_cases)

    # p_e conditional: Uniform(0, min(PE_RANGE[1], p_i - margin))
    _margin = 0.1e6  # 0.1 MPa minimum gap between p_i and p_e
    p_e_upper = np.minimum(PE_RANGE[1], p_i - _margin)
    p_e = rng.uniform(0.0, 1.0, size=n_cases) * p_e_upper

    E = np.exp(
        rng.uniform(np.log(E_RANGE[0]), np.log(E_RANGE[1]), size=n_cases)
    )
    nu = rng.uniform(NU_RANGE[0], NU_RANGE[1], size=n_cases)

    return [(i, p_i[i], p_e[i], E[i], nu[i]) for i in range(n_cases)]


# ── Analytical solver ─────────────────────────────────────────────────────────

def solve_case(args: tuple) -> list[dict]:  # type: ignore[type-arg]
    """Evaluate the Lamé analytical solution at all N_PTS query points.

    Args:
        args: (case_id, p_i, p_e, E, nu) — a single tuple from sample_params.

    Returns:
        A list containing one dict with keys:
            case_id, p_i, p_e, E, nu,
            sigma_vm, u_r, sigma_r, sigma_theta  (each a flat list of N_PTS float32 values)
    """
    case_id, p_i, p_e, E, nu = args

    # Lamé combination coefficients (scalars)
    C1 = (p_i * _A3 - p_e * _B3) / _DENOM   # hydrostatic term
    C2 = (p_i - p_e) * _A3 * _B3 / _DENOM   # deviatoric term

    r   = _QR          # (N_PTS,) float64
    r3  = r ** 3
    r2  = r ** 2

    # Radial displacement [m]  — (N_PTS,)
    u_r = (1.0 / E) * (C1 * (1.0 - 2.0 * nu) * r + C2 * (1.0 + nu) / (2.0 * r2))

    # Radial and hoop stresses [Pa]  — (N_PTS,)
    sigma_r     = C1 - C2 / r3
    sigma_theta = C1 + C2 / (2.0 * r3)

    # Von Mises stress [Pa]:  σ_vm = |σ_r − σ_θ| = |C2| · 3 / (2r³)
    sigma_vm = np.abs(sigma_r - sigma_theta)

    return [
        {
            "case_id":     int(case_id),
            "p_i":         float(p_i),
            "p_e":         float(p_e),
            "E":           float(E),
            "nu":          float(nu),
            "sigma_vm":    sigma_vm.astype(np.float32),
            "u_r":         u_r.astype(np.float32),
            "sigma_r":     sigma_r.astype(np.float32),
            "sigma_theta": sigma_theta.astype(np.float32),
        }
    ]


# ── Parquet schema and helpers ────────────────────────────────────────────────

SCHEMA = pa.schema(
    [
        pa.field("case_id",     pa.int32()),
        pa.field("p_i",         pa.float32()),
        pa.field("p_e",         pa.float32()),
        pa.field("E",           pa.float32()),
        pa.field("nu",          pa.float32()),
        pa.field("sigma_vm",    pa.list_(pa.float32())),
        pa.field("u_r",         pa.list_(pa.float32())),
        pa.field("sigma_r",     pa.list_(pa.float32())),
        pa.field("sigma_theta", pa.list_(pa.float32())),
    ]
)

_FIELD_KEYS = ("case_id", "p_i", "p_e", "E", "nu",
               "sigma_vm", "u_r", "sigma_r", "sigma_theta")


def rows_to_table(rows: list[dict], extra_meta: dict | None = None) -> pa.Table:  # type: ignore[type-arg]
    """Convert row dicts from solve_case into a PyArrow table."""
    arrays = {k: [r[k] for r in rows] for k in _FIELD_KEYS}
    pa_arrays = {
        "case_id":     pa.array(arrays["case_id"],     type=pa.int32()),
        "p_i":         pa.array(arrays["p_i"],         type=pa.float32()),
        "p_e":         pa.array(arrays["p_e"],         type=pa.float32()),
        "E":           pa.array(arrays["E"],           type=pa.float32()),
        "nu":          pa.array(arrays["nu"],          type=pa.float32()),
        "sigma_vm":    pa.array(arrays["sigma_vm"],    type=pa.list_(pa.float32())),
        "u_r":         pa.array(arrays["u_r"],         type=pa.list_(pa.float32())),
        "sigma_r":     pa.array(arrays["sigma_r"],     type=pa.list_(pa.float32())),
        "sigma_theta": pa.array(arrays["sigma_theta"], type=pa.list_(pa.float32())),
    }
    meta: dict[bytes, bytes] = {
        b"inner_radius": str(A).encode(),
        b"outer_radius": str(B).encode(),
        b"N_PTS":        str(N_PTS).encode(),
        b"query_source": b"sphere-FEMMeshGmsh.vtk",
    }
    if extra_meta:
        meta.update({k.encode(): v.encode() for k, v in extra_meta.items()})
    table = pa.table(pa_arrays, schema=SCHEMA)
    return table.replace_schema_metadata(meta)


def save_split(
    rows: list[dict],  # type: ignore[type-arg]
    path: Path,
    split_name: str,
    n_cases_split: int,
) -> None:
    """Write rows to a Snappy-compressed Parquet file and print a summary line."""
    table = rows_to_table(rows, {"split": split_name, "n_cases": str(n_cases_split)})
    pq.write_table(table, path, compression="snappy")
    mb = path.stat().st_size / 1e6
    print(f"  Saved {split_name:5s}: {len(rows):>6} rows  ({n_cases_split} cases) → {path}  [{mb:.1f} MB]")


# ── Hexagonal adapter ─────────────────────────────────────────────────────────

class LameSolver(IPhysicsSolver[tuple, list[dict]]):  # type: ignore[type-arg]
    """IPhysicsSolver implementation using the exact Lamé analytical solution.

    Uses ``ProcessPoolExecutor`` with ``fork`` workers for CPU-bound parallelism.
    Module-level QUERY_XYZ and precomputed geometry factors are inherited by
    fork workers at zero serialisation cost.
    """

    def sample_params(self, n_cases: int, seed: int) -> list[tuple]:  # type: ignore[type-arg]
        return sample_params(n_cases, seed)

    def batch_solve(self, params_list: list[tuple], n_workers: int | None = None) -> list[list[dict]]:  # type: ignore[type-arg]
        """Solve all cases in parallel and return results in input order."""
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            return list(pool.map(solve_case, params_list))
