"""Export the trained Lamé sphere DeepONet for inference-hub.

Loads the checkpoint produced by notebooks/lame_sphere_train.ipynb, wraps
the model in LameSpherePredictor (applies normalisation + fixed radial grid),
and exports to outputs/don_lame_sphere_scripted.pt via torch.export.

Usage (from repo root):
    python scripts/export_lame_sphere.py
    python scripts/export_lame_sphere.py \\
        --checkpoint outputs/don_lame_sphere.pt \\
        --output     outputs/don_lame_sphere_scripted.pt

Loading in inference-hub:
    POST /api/v1/models/lame_sphere/load
        {"path": "/path/to/don_lame_sphere_scripted.pt", "framework": "pytorch"}

    POST /api/v1/models/lame_sphere/infer
        {"inputs": [[p_i, p_e]]}
        # e.g. {"inputs": [[10e6, 2e6]]}  (p_i=10 MPa, p_e=2 MPa)
        # Returns (batch, 200) σ_vm in Pa on a fixed radial grid [0.2 m, 0.5 m]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neural_operators.adapters.exporters import TorchExportExporter
from neural_operators.adapters.physics.lame_sphere_solver import A, B
from neural_operators.models import DeepONet3D, LameSpherePredictor
from neural_operators.use_cases import ExportModelUseCase


def _load_state_dict(checkpoint_path: Path) -> dict:  # type: ignore[type-arg]
    """Load state dict from a checkpoint (handles plain dict or nested format)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]  # type: ignore[return-value]
    return ckpt  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Lamé sphere DeepONet to TorchScript")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "outputs" / "don_lame_sphere.pt")
    parser.add_argument("--config",     type=Path, default=PROJECT_ROOT / "configs" / "lame_sphere.yaml")
    parser.add_argument("--output",     type=Path, default=PROJECT_ROOT / "outputs" / "don_lame_sphere_scripted.pt")
    parser.add_argument("--n-eval-pts", type=int,  default=200, help="Radial evaluation points")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Train the model first: run notebooks/lame_sphere_train.ipynb")
        sys.exit(1)

    cfg = yaml.safe_load(args.config.read_text())
    norm = cfg["normalisation"]
    don_cfg = cfg["deeponet"]

    # Reconstruct model architecture (param_dim=1 for Δp only; trunk_in_dim=1 for r)
    base_model = DeepONet3D(
        param_dim=don_cfg["param_dim"],
        p=don_cfg["p"],
        width=don_cfg["width"],
        depth=don_cfg["depth"],
        trunk_in_dim=don_cfg["trunk_in_dim"],
    )
    state_dict = _load_state_dict(args.checkpoint)
    base_model.load_state_dict(state_dict)
    print(f"  Loaded checkpoint: {args.checkpoint}")

    # Wrap with normalisation and fixed radial evaluation grid
    predictor = LameSpherePredictor(
        model=base_model,
        dp_min=norm["dp_min"],
        dp_max=norm["dp_max"],
        log_sigma_ref=norm["log_sigma_ref"],
        r_min=A,
        r_max=B,
        n_eval_pts=args.n_eval_pts,
    )

    # Export
    sample = torch.tensor([[10.0e6, 2.0e6]])  # p_i=10 MPa, p_e=2 MPa
    use_case = ExportModelUseCase(TorchExportExporter())
    out_path = use_case.execute(predictor, args.output, trace_input=sample)
    print(f"  Done: {out_path}")

    # Smoke-test the exported model
    ep = torch.export.load(str(out_path))
    out = ep.module()(sample)
    assert out.shape == (1, args.n_eval_pts), f"Unexpected output shape: {out.shape}"
    print(f"  Smoke test passed: output shape {tuple(out.shape)}")


if __name__ == "__main__":
    main()
