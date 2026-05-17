"""Analytical 2-D heat equation solver and dataset utilities.

Solves the transient heat equation on a unit square [0,1]² with Dirichlet BCs
via superposition of a steady-state Fourier sine series and a double-sine
transient series.

Module-level precomputed arrays (sin/sinh bases) are computed once at import
time (~50 ms).  On Linux the default ``fork`` start-method means
``ProcessPoolExecutor`` workers inherit these arrays at zero serialisation cost.
Do not import this module inside a tight loop.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from neural_operators.domain.ports.physics_solver import IPhysicsSolver

__all__ = [
    "NX", "NY", "Lx", "Ly",
    "N_TIMES", "T_START", "T_END",
    "T_BC_RANGE", "T_INIT_RANGE", "ALPHA_RANGE",
    "solve_case",
    "sample_params",
    "analytical_field",
    "rows_to_table",
    "save_split",
    "SCHEMA",
    "FourierHeat2DSolver",
]

# ── Physics / grid constants ──────────────────────────────────────────────────

NX: int = 32
NY: int = 32
Lx: float = 1.0
Ly: float = 1.0

N_STEADY: int = 40     # odd Fourier terms for steady-state (k = 1, 3, …, 2N-1)
M_TRANS: int  = 20     # double-sine transient modes (m-direction)
N_TRANS: int  = 20     # double-sine transient modes (n-direction)
FINE: int     = 256    # quadrature grid for Cmn coefficients

N_TIMES: int  = 10
T_START: float = 0.05
T_END: float   = 30.0

T_BC_RANGE:   tuple[float, float] = (0.0,   200.0)
T_INIT_RANGE: tuple[float, float] = (0.0,   100.0)
ALPHA_RANGE:  tuple[float, float] = (0.005, 0.05)

# ── Module-level precomputed arrays ──────────────────────────────────────────

_x_out  = np.linspace(0, Lx, NX,   dtype=np.float64)
_y_out  = np.linspace(0, Ly, NY,   dtype=np.float64)
_x_fine = np.linspace(0, Lx, FINE, dtype=np.float64)
_y_fine = np.linspace(0, Ly, FINE, dtype=np.float64)

_k_arr = np.arange(1, 2 * N_STEADY, 2, dtype=np.float64)   # (K,) odd terms
_c_arr = 4.0 / (_k_arr * np.pi)                              # (K,)

_m_arr = np.arange(1, M_TRANS + 1, dtype=np.float64)        # (M,)
_n_arr = np.arange(1, N_TRANS + 1, dtype=np.float64)        # (N,)

# Transient bases on output grid
_SIN_MX_OUT = np.sin(np.outer(_m_arr * np.pi / Lx, _x_out))  # (M, NX)
_SIN_NY_OUT = np.sin(np.outer(_n_arr * np.pi / Ly, _y_out))  # (N, NY)
_LAMBDA_SQ  = (_m_arr[:, None] * np.pi / Lx) ** 2 + (_n_arr[None, :] * np.pi / Ly) ** 2  # (M, N)

# Transient bases on fine grid
_SIN_MX_F = np.sin(np.outer(_m_arr * np.pi / Lx, _x_fine))   # (M, FINE)
_SIN_NY_F = np.sin(np.outer(_n_arr * np.pi / Ly, _y_fine))   # (N, FINE)

# Quadrature weights (trapezoidal)
_wx = np.ones(FINE)
_wx[0] = _wx[-1] = 0.5
_wx *= Lx / (FINE - 1)
_wy = np.ones(FINE)
_wy[0] = _wy[-1] = 0.5
_wy *= Ly / (FINE - 1)

# Default time grid (module-level so fork workers see it without recomputing)
_t_values = np.logspace(np.log10(T_START), np.log10(T_END), N_TIMES)


def _precompute_steady_bases(
    x_grid: np.ndarray, y_grid: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Return (sin_kx_x, sinh_B, sinh_T, sin_ky_y, sinh_L, sinh_R) on the given grid."""
    kx  = _k_arr * np.pi / Lx
    ky  = _k_arr * np.pi / Ly
    aLy = kx * Ly
    aLx = ky * Lx

    sin_kx_x = np.sin(np.outer(kx, x_grid))  # (K, Nx)
    sin_ky_y = np.sin(np.outer(ky, y_grid))  # (K, Ny)

    def _sr(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        large = den > 15.0
        with np.errstate(over="ignore", invalid="ignore"):
            r = np.sinh(num) / np.where(large, 1.0, np.sinh(np.where(large, 1.0, den)))
        return np.where(large, np.exp(num - den), r)

    sinh_B = _sr(kx[:, None] * (Ly - y_grid[None, :]), aLy[:, None] * np.ones(len(y_grid)))
    sinh_T = _sr(kx[:, None] * y_grid[None, :],        aLy[:, None] * np.ones(len(y_grid)))
    sinh_L = _sr(ky[:, None] * (Lx - x_grid[None, :]), aLx[:, None] * np.ones(len(x_grid)))
    sinh_R = _sr(ky[:, None] * x_grid[None, :],        aLx[:, None] * np.ones(len(x_grid)))

    return sin_kx_x, sinh_B, sinh_T, sin_ky_y, sinh_L, sinh_R


# Steady-state bases on output and fine grids
(
    _SKX_OUT, _SB_OUT, _ST_OUT, _SKY_OUT, _SL_OUT, _SR_OUT,
) = _precompute_steady_bases(_x_out, _y_out)

(
    _SKX_F, _SB_F, _ST_F, _SKY_F, _SL_F, _SR_F,
) = _precompute_steady_bases(_x_fine, _y_fine)


def _steady_vec(
    TL: float, TR: float, TB: float, TT: float,
    skx: np.ndarray, sb: np.ndarray, st: np.ndarray,
    sky: np.ndarray, sl: np.ndarray, sr: np.ndarray,
) -> np.ndarray:
    """Steady-state field (Nx, Ny) via vectorised matrix multiplications."""
    u = np.zeros((skx.shape[1], sb.shape[1]))
    if TB:
        u += TB * (_c_arr[:, None] * skx).T @ sb
    if TT:
        u += TT * (_c_arr[:, None] * skx).T @ st
    if TL:
        u += TL * sl.T @ (_c_arr[:, None] * sky)
    if TR:
        u += TR * sr.T @ (_c_arr[:, None] * sky)
    return u


def _solve_heat2d(
    T_L: float, T_R: float, T_B: float, T_T: float,
    T_init: float, alpha: float,
    t_values: np.ndarray,
) -> np.ndarray:
    """Core solver. Returns (len(t_values), NX, NY) temperature field in °C."""
    tL, tR, tB, tT = T_L - T_init, T_R - T_init, T_B - T_init, T_T - T_init

    # Steady state on fine grid → Fourier coefficients C_mn
    us_f = _steady_vec(tL, tR, tB, tT, _SKX_F, _SB_F, _ST_F, _SKY_F, _SL_F, _SR_F)
    fn   = (-us_f) @ (_SIN_NY_F * _wy).T               # (FINE, N)
    C    = _SIN_MX_F @ (fn * _wx[:, None]) * (4.0 / (Lx * Ly))  # (M, N)

    # Steady state on output grid
    us_out = _steady_vec(tL, tR, tB, tT, _SKX_OUT, _SB_OUT, _ST_OUT, _SKY_OUT, _SL_OUT, _SR_OUT)

    fields = np.empty((len(t_values), NX, NY), dtype=np.float64)
    for i, t in enumerate(t_values):
        v = _SIN_MX_OUT.T @ (C * np.exp(-alpha * _LAMBDA_SQ * t)) @ _SIN_NY_OUT
        fields[i] = T_init + us_out + v
    return fields


# ── Public API ────────────────────────────────────────────────────────────────

def analytical_field(
    t_values: np.ndarray,
    T_L: float, T_R: float,
    T_B: float, T_T: float,
    T_init: float,
    alpha: float,
) -> np.ndarray:
    """Analytical 2-D heat solution on the NX×NY output grid.

    Returns array of shape ``(len(t_values), NX, NY)`` in °C.
    Gibbs oscillations are expected at domain corners where adjacent BCs conflict.
    """
    return _solve_heat2d(T_L, T_R, T_B, T_T, T_init, alpha, t_values)


def solve_case(args: tuple) -> list[dict]:  # type: ignore[type-arg]
    """Solve one case and return Parquet-ready row dicts.

    ``args`` must be ``(case_id, T_L, T_R, T_B, T_T, T_init, alpha)``.
    Uses the module-level ``_t_values`` time grid.
    Compatible with ``concurrent.futures.ProcessPoolExecutor``.
    """
    case_id, T_L, T_R, T_B, T_T, T_init, alpha = args
    fields = _solve_heat2d(T_L, T_R, T_B, T_T, T_init, alpha, _t_values)
    return [
        {
            "case_id": int(case_id),
            "t":       float(t),
            "alpha":   float(alpha),
            "T_L":     float(T_L),
            "T_R":     float(T_R),
            "T_B":     float(T_B),
            "T_T":     float(T_T),
            "T_init":  float(T_init),
            "T_field": fields[i].astype(np.float32).ravel().tolist(),
        }
        for i, t in enumerate(_t_values)
    ]


def sample_params(n_cases: int, seed: int) -> list[tuple]:  # type: ignore[type-arg]
    """Sample random physical parameters for ``n_cases`` cases.

    Returns a list of ``(case_id, T_L, T_R, T_B, T_T, T_init, alpha)`` tuples
    suitable for passing to ``solve_case`` via a process pool.
    """
    rng     = np.random.default_rng(seed)
    BCs     = rng.uniform(*T_BC_RANGE,   size=(n_cases, 4))
    T_inits = rng.uniform(*T_INIT_RANGE, size=n_cases)
    alphas  = np.exp(
        rng.uniform(np.log(ALPHA_RANGE[0]), np.log(ALPHA_RANGE[1]), size=n_cases)
    )
    return [
        (i, BCs[i, 0], BCs[i, 1], BCs[i, 2], BCs[i, 3], T_inits[i], alphas[i])
        for i in range(n_cases)
    ]


# ── Parquet schema and helpers ────────────────────────────────────────────────

SCHEMA = pa.schema([
    pa.field("case_id", pa.int32()),
    pa.field("t",       pa.float32()),
    pa.field("alpha",   pa.float32()),
    pa.field("T_L",     pa.float32()),
    pa.field("T_R",     pa.float32()),
    pa.field("T_B",     pa.float32()),
    pa.field("T_T",     pa.float32()),
    pa.field("T_init",  pa.float32()),
    pa.field("T_field", pa.list_(pa.float32())),
])


def rows_to_table(rows: list[dict], extra_meta: dict | None = None) -> pa.Table:  # type: ignore[type-arg]
    """Convert row dicts from ``solve_case`` into a PyArrow table."""
    keys = ("case_id", "t", "alpha", "T_L", "T_R", "T_B", "T_T", "T_init", "T_field")
    arrays = {k: [r[k] for r in rows] for k in keys}
    pa_arrays = {
        "case_id": pa.array(arrays["case_id"], type=pa.int32()),
        "t":       pa.array(arrays["t"],       type=pa.float32()),
        "alpha":   pa.array(arrays["alpha"],   type=pa.float32()),
        "T_L":     pa.array(arrays["T_L"],     type=pa.float32()),
        "T_R":     pa.array(arrays["T_R"],     type=pa.float32()),
        "T_B":     pa.array(arrays["T_B"],     type=pa.float32()),
        "T_T":     pa.array(arrays["T_T"],     type=pa.float32()),
        "T_init":  pa.array(arrays["T_init"],  type=pa.float32()),
        "T_field": pa.array(arrays["T_field"], type=pa.list_(pa.float32())),
    }
    meta: dict[bytes, bytes] = {
        b"grid_NX":  str(NX).encode(),
        b"grid_NY":  str(NY).encode(),
        b"Lx":       b"1.0",
        b"Ly":       b"1.0",
        b"N_times":  str(N_TIMES).encode(),
        b"t_start":  str(T_START).encode(),
        b"t_end":    str(T_END).encode(),
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

class FourierHeat2DSolver(IPhysicsSolver[tuple, list[dict]]):  # type: ignore[type-arg]
    """IPhysicsSolver implementation using the Fourier analytical 2-D heat solver.

    Uses ``ProcessPoolExecutor`` with ``fork`` workers for CPU-bound parallelism.
    Module-level precomputed arrays are inherited by workers at zero cost.
    """

    def sample_params(self, n_cases: int, seed: int) -> list[tuple]:  # type: ignore[type-arg]
        return sample_params(n_cases, seed)

    def batch_solve(self, params_list: list[tuple], n_workers: int | None = None) -> list[list[dict]]:  # type: ignore[type-arg]
        """Solve all cases in parallel and return results in input order."""
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            return list(pool.map(solve_case, params_list))
