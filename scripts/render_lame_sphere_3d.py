"""Render 3-D PyVista visualisations for the trained Lamé-sphere DeepONet.

Loads the saved checkpoint (outputs/don_lame_sphere.pt), runs full-mesh inference
on three test cases, and saves two PNG files to outputs/.
No training data loading required — only the 4 scalar params per test case.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyvista as pv
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

from neural_operators.adapters.physics.lame_sphere_solver import QUERY_XYZ, A, B, load_sphere_mesh
from neural_operators.models import DeepONet3D

DATA_DIR = REPO_ROOT / "dataset"
OUT_DIR  = REPO_ROOT / "outputs"
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load checkpoint ───────────────────────────────────────────────────────────
ckpt    = torch.load(OUT_DIR / "don_lame_sphere.pt", map_location=device, weights_only=False)
cfg     = ckpt["cfg"]
don_cfg = cfg["deeponet"]
norm    = cfg["normalisation"]

model = DeepONet3D(
    param_dim    = don_cfg["param_dim"],
    p            = don_cfg["p"],
    width        = don_cfg["width"],
    depth        = don_cfg["depth"],
    trunk_in_dim = don_cfg.get("trunk_in_dim", 3),
).to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print(f"Checkpoint loaded  — best val MSE = {ckpt['val_loss']:.4e}")

# ── Full-mesh trunk (r normalised to [0, 1]) ──────────────────────────────────
_r_full  = np.linalg.norm(QUERY_XYZ, axis=1, keepdims=True).astype(np.float32)
TRUNK_R_FULL = torch.from_numpy((_r_full - A) / (B - A)).to(device)   # (409230, 1)
print(f"Full mesh: {len(QUERY_XYZ):,} points")

# ── Load test params (scalar columns only — fast, no sigma_vm loading) ────────
pf = pq.ParquetFile(DATA_DIR / "lame_sphere_test.parquet")
param_cols = ["case_id", "p_i", "p_e", "E", "nu"]
test_df = next(pf.iter_batches(batch_size=10_000, columns=param_cols)).to_pydict()

n_test  = len(test_df["case_id"])
pi_min, pi_max = cfg["normalisation"]["pi_min"], cfg["normalisation"]["pi_max"]
pe_max         = cfg["normalisation"]["pe_max"]
E_log_min      = cfg["normalisation"]["E_log_min"]
E_log_max      = cfg["normalisation"]["E_log_max"]
nu_min, nu_max = cfg["normalisation"]["nu_min"], cfg["normalisation"]["nu_max"]

p_i = np.array(test_df["p_i"], dtype=np.float32)
p_e = np.array(test_df["p_e"], dtype=np.float32)
E   = np.array(test_df["E"],   dtype=np.float64)
nu  = np.array(test_df["nu"],  dtype=np.float32)

params_norm = np.stack([
    (p_i - pi_min) / (pi_max - pi_min),
    p_e / pe_max,
    ((np.log(E) - E_log_min) / (E_log_max - E_log_min)).astype(np.float32),
    (nu - nu_min) / (nu_max - nu_min),
], axis=1).astype(np.float32)

vis_ids = [0, n_test // 2, n_test - 1]
print(f"Visualising cases: {vis_ids}")

def render_case(case_idx: int, out_stem: str) -> None:
    row_pi = float(test_df["p_i"][case_idx])
    row_pe = float(test_df["p_e"][case_idx])

    p_b = torch.from_numpy(params_norm[case_idx : case_idx + 1]).to(device)
    with torch.no_grad():
        pred_vm = model(p_b, TRUNK_R_FULL).cpu().numpy()[0] * norm["sigma_vm_scale"]

    _r_all   = np.clip(np.linalg.norm(QUERY_XYZ, axis=1), A, B).astype(np.float64)
    _C2      = (row_pi - row_pe) * A**3 * B**3 / (B**3 - A**3)
    true_vm  = (np.abs(_C2) * 3.0 / (2.0 * _r_all**3)).astype(np.float32)

    mesh = load_sphere_mesh()
    mesh["sigma_vm_pred_MPa"] = pred_vm / 1e6
    mesh["sigma_vm_true_MPa"] = true_vm / 1e6
    mesh["abs_err_MPa"]       = np.abs(pred_vm - true_vm) / 1e6

    clip = mesh.clip(normal="y", origin=(0, 0, 0))

    # — Cross-section ——————————————————————————————————————————————————————
    path_clip = str(OUT_DIR / f"{out_stem}_crosssection.png")
    pl = pv.Plotter(shape=(1, 3), window_size=[1500, 500])
    pl.subplot(0, 0)
    pl.add_mesh(clip, scalars="sigma_vm_pred_MPa",
                scalar_bar_args={"title": "σ_vm pred [MPa]", "vertical": True})
    pl.add_title(f"Predicted — case {case_idx}", font_size=10)
    pl.view_isometric()
    pl.subplot(0, 1)
    pl.add_mesh(clip.copy(), scalars="sigma_vm_true_MPa",
                scalar_bar_args={"title": "σ_vm true [MPa]", "vertical": True})
    pl.add_title("Ground-truth", font_size=10)
    pl.view_isometric()
    pl.subplot(0, 2)
    pl.add_mesh(clip.copy(), scalars="abs_err_MPa",
                scalar_bar_args={"title": "|error| [MPa]", "vertical": True})
    pl.add_title("Abs. error", font_size=10)
    pl.view_isometric()
    pl.show()
    pl.screenshot(path_clip)
    pl.close()
    print(f"  Saved {path_clip}")

    # — Outer surface ——————————————————————————————————————————————————————
    surface = mesh.extract_surface(algorithm="dataset_surface")
    _smin = float(surface["sigma_vm_pred_MPa"].min())
    _smax = float(surface["sigma_vm_pred_MPa"].max())

    path_outer = str(OUT_DIR / f"{out_stem}_outer.png")
    pl2 = pv.Plotter(shape=(1, 2), window_size=[1000, 500])
    pl2.subplot(0, 0)
    pl2.add_mesh(surface, scalars="sigma_vm_pred_MPa",
                 clim=[_smin, _smax],
                 scalar_bar_args={"title": "σ_vm pred [MPa]", "vertical": True})
    pl2.add_title("Outer surface — predicted", font_size=10)
    pl2.view_isometric()
    pl2.subplot(0, 1)
    pl2.add_mesh(surface.copy(), scalars="abs_err_MPa",
                 scalar_bar_args={"title": "|error| [MPa]", "vertical": True})
    pl2.add_title("Outer surface — abs. error", font_size=10)
    pl2.view_isometric()
    pl2.show()
    pl2.screenshot(path_outer)
    pl2.close()
    print(f"  Saved {path_outer}")


for case_idx in vis_ids:
    row_pi = float(test_df["p_i"][case_idx])
    row_pe = float(test_df["p_e"][case_idx])
    print(f"\nCase {case_idx}: p_i={row_pi/1e6:.1f} MPa, p_e={row_pe/1e6:.1f} MPa")
    render_case(case_idx, f"lame_sphere_3d_case{case_idx:03d}")

print("\nDone.")