"""Thin inference wrappers for serving trained models via inference-hub.

Each wrapper:
- Accepts a single normalized input tensor (``inference-hub``'s
  ``PyTorchBackend`` converts ``request.inputs`` to one ``float32`` tensor).
- Stores normalization constants as non-parameter buffers so they are saved
  and restored with the TorchScript file — callers need no external config.
- Evaluates the underlying DeepONet on a fixed evaluation grid stored as a
  buffer, so the output shape is deterministic.
- Is scriptable via ``torch.jit.script``.

Normalisation conventions (matching training in the notebooks):

  Heat2DPredictor:
    input  (batch, 7): [T_L, T_R, T_B, T_T, T_init, alpha, t]  — raw SI units
    output (batch, 1024): temperature field in °C on a fixed 32×32 grid

  LameSpherePredictor:
    input  (batch, 2): [p_i, p_e]  — raw pressures in Pa
    output (batch, N_eval): σ_vm in Pa on a fixed radial evaluation grid
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neural_operators.models.deeponet import DeepONet2D, DeepONet3D

__all__ = ["Heat2DPredictor", "LameSpherePredictor"]

# ── 2-D Heat Equation ─────────────────────────────────────────────────────────

class Heat2DPredictor(nn.Module):
    """Inference wrapper for the Heat2D DeepONet.

    Wraps ``DeepONet2D`` with built-in normalisation and a fixed 32×32 output
    grid.  Accepts raw physical parameters and returns the temperature field
    in °C.

    Normalisation (from configs/heat2d.yaml):
        T_scale     = 200.0   (all temperatures divided by this)
        alpha_min   = 0.005
        alpha_max   = 0.05
        t_max       = 30.0    (seconds)
        t_scale     = 200.0   (output temperature scale)
    """

    def __init__(
        self,
        model: DeepONet2D,
        t_scale: float = 200.0,
        alpha_min: float = 0.005,
        alpha_max: float = 0.05,
        t_max: float = 30.0,
        nx: int = 32,
        ny: int = 32,
    ) -> None:
        super().__init__()
        self.model = model

        # Scalar normalisation constants stored as 0-dim buffers
        self.register_buffer("t_scale",   torch.tensor(t_scale))
        self.register_buffer("alpha_min", torch.tensor(alpha_min))
        self.register_buffer("alpha_max", torch.tensor(alpha_max))
        self.register_buffer("t_max",     torch.tensor(t_max))

        # Fixed output grid: all (x, y) pairs in [0,1]² for an nx×ny mesh
        x = torch.linspace(0.0, 1.0, nx)
        y = torch.linspace(0.0, 1.0, ny)
        # meshgrid with indexing='ij' → shape (nx, ny) for each; flatten to (nx*ny, 2)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        xy = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # (nx*ny, 2)
        self.register_buffer("xy_grid", xy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, 7) — [T_L, T_R, T_B, T_T, T_init, alpha, t] in raw SI units
        returns (batch, nx*ny) — temperature field in °C
        """
        t_scale:   torch.Tensor = self.t_scale    # type: ignore[assignment]
        alpha_min: torch.Tensor = self.alpha_min  # type: ignore[assignment]
        alpha_max: torch.Tensor = self.alpha_max  # type: ignore[assignment]
        t_max:     torch.Tensor = self.t_max      # type: ignore[assignment]
        xy_grid:   torch.Tensor = self.xy_grid    # type: ignore[assignment]

        # Normalise branch inputs to [0, 1] range
        T_norm   = x[:, :5] / t_scale
        a_norm   = (x[:, 5:6] - alpha_min) / (alpha_max - alpha_min)
        t_norm   = x[:, 6:7] / t_max
        params   = torch.cat([T_norm, a_norm, t_norm], dim=1)  # (batch, 7)

        # Evaluate on fixed grid; output is normalised → scale back to °C
        out_norm = self.model(params, xy_grid)   # (batch, nx*ny)
        return out_norm * t_scale


# ── Lamé Hollow Sphere ────────────────────────────────────────────────────────

class LameSpherePredictor(nn.Module):
    """Inference wrapper for the Lamé sphere DeepONet.

    Wraps ``DeepONet3D`` (param_dim=1, trunk_in_dim=1) with built-in
    normalisation and a fixed radial evaluation grid.

    The model predicts log(σ_vm / log_sigma_ref); this wrapper converts back
    to σ_vm in Pa.

    Normalisation (from configs/lame_sphere.yaml):
        dp_min        = 0.1e+6   Pa
        dp_max        = 20.0e+6  Pa
        log_sigma_ref = 1.0e+6   Pa
        r_min (= A)   = 0.2      m
        r_max (= B)   = 0.5      m
    """

    def __init__(
        self,
        model: DeepONet3D,
        dp_min: float = 0.1e6,
        dp_max: float = 20.0e6,
        log_sigma_ref: float = 1.0e6,
        r_min: float = 0.2,
        r_max: float = 0.5,
        n_eval_pts: int = 200,
    ) -> None:
        super().__init__()
        self.model = model

        self.register_buffer("dp_min",        torch.tensor(dp_min))
        self.register_buffer("dp_max",        torch.tensor(dp_max))
        self.register_buffer("log_sigma_ref", torch.tensor(log_sigma_ref))

        # Fixed radial evaluation grid from inner to outer radius
        r = torch.linspace(r_min, r_max, n_eval_pts).unsqueeze(1)  # (n_eval_pts, 1)
        self.register_buffer("r_grid",   r)
        self.register_buffer("r_min",    torch.tensor(r_min))
        self.register_buffer("r_max",    torch.tensor(r_max))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, 2) — [p_i, p_e] in Pa (raw, un-normalised)
        returns (batch, n_eval_pts) — σ_vm in Pa on the fixed radial grid
        """
        dp_min:        torch.Tensor = self.dp_min         # type: ignore[assignment]
        dp_max:        torch.Tensor = self.dp_max         # type: ignore[assignment]
        log_sigma_ref: torch.Tensor = self.log_sigma_ref  # type: ignore[assignment]
        r_grid:        torch.Tensor = self.r_grid         # type: ignore[assignment]
        r_min:         torch.Tensor = self.r_min          # type: ignore[assignment]
        r_max:         torch.Tensor = self.r_max          # type: ignore[assignment]

        # Compute Δp and normalise to [0, 1]
        dp      = x[:, 0:1] - x[:, 1:2]
        dp_norm = (dp - dp_min) / (dp_max - dp_min)  # (batch, 1)

        # Normalise radial query points to [0, 1]
        r_norm = (r_grid - r_min) / (r_max - r_min)   # (n_eval_pts, 1)

        # Model output is log(σ_vm / log_sigma_ref)
        log_out = self.model(dp_norm, r_norm)         # (batch, n_eval_pts)
        return torch.exp(log_out) * log_sigma_ref
