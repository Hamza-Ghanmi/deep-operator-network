"""Export the trained Heat2D DeepONet to a TorchScript file for inference-hub.

Loads the checkpoint produced by notebooks/heat2d_train_compare.ipynb, wraps
the model in Heat2DPredictor (applies normalisation + fixed 32×32 output grid),
and exports to outputs/don_heat2d_scripted.pt via torch.jit.script.

Usage (from repo root):
    python scripts/export_heat2d.py
    python scripts/export_heat2d.py --checkpoint outputs/don_heat2d.pt \\
                                    --output    outputs/don_heat2d_scripted.pt

Loading in inference-hub:
    POST /api/v1/models/heat2d/load
        {"path": "/path/to/don_heat2d_scripted.pt", "framework": "pytorch"}

    POST /api/v1/models/heat2d/infer
        {"inputs": [[T_L, T_R, T_B, T_T, T_init, alpha, t]]}
        # e.g. {"inputs": [[0.0, 100.0, 0.0, 0.0, 20.0, 0.02, 5.0]]}
        # Returns (batch, 1024) flattened 32×32 temperature field in °C
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neural_operators.adapters.exporters import TorchScriptExporter
from neural_operators.models import DeepONet2D, Heat2DPredictor
from neural_operators.use_cases import ExportModelUseCase


def _load_state_dict(checkpoint_path: Path) -> dict:  # type: ignore[type-arg]
    """Load state dict from a checkpoint.

    The notebook saves a plain state dict; the training script saves a dict
    with a ``model_state_dict`` key.  Both formats are handled.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]  # type: ignore[return-value]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]  # type: ignore[return-value]
    return ckpt  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Heat2D DeepONet to TorchScript")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "outputs" / "don_heat2d.pt")
    parser.add_argument("--config",     type=Path, default=PROJECT_ROOT / "configs" / "heat2d.yaml")
    parser.add_argument("--output",     type=Path, default=PROJECT_ROOT / "outputs" / "don_heat2d_scripted.pt")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Train the model first: run notebooks/heat2d_train_compare.ipynb")
        sys.exit(1)

    cfg = yaml.safe_load(args.config.read_text())
    norm = cfg["normalisation"]
    don_cfg = cfg["deeponet"]

    # Reconstruct model architecture
    base_model = DeepONet2D(
        param_dim=don_cfg["param_dim"],
        p=don_cfg["p"],
        width=don_cfg["width"],
        depth=don_cfg["depth"],
    )
    state_dict = _load_state_dict(args.checkpoint)
    base_model.load_state_dict(state_dict)
    print(f"  Loaded checkpoint: {args.checkpoint}")

    # Wrap with normalisation and fixed output grid
    predictor = Heat2DPredictor(
        model=base_model,
        t_scale=norm["t_scale"],
        alpha_min=norm["alpha_min"],
        alpha_max=norm["alpha_max"],
        t_max=norm["t_max"],
    )

    # Export
    use_case = ExportModelUseCase(TorchScriptExporter())
    out_path = use_case.execute(predictor, args.output)
    print(f"  Done: {out_path}")

    # Smoke-test the exported model
    scripted = torch.jit.load(str(out_path), map_location="cpu")
    sample = torch.tensor([[0.0, 100.0, 0.0, 0.0, 20.0, 0.02, 5.0]])
    out = scripted(sample)
    assert out.shape == (1, 32 * 32), f"Unexpected output shape: {out.shape}"
    print(f"  Smoke test passed: output shape {tuple(out.shape)}")


if __name__ == "__main__":
    main()
