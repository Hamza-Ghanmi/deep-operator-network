"""DeepONet model definitions for 1-D, 2-D and 3-D operator learning."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["mlp", "DeepONet", "DeepONet2D", "DeepONet3D"]


def mlp(
    in_dim: int,
    hidden_dim: int,
    n_layers: int,
    out_dim: int | None = None,
    dropout: float = 0.0,
) -> nn.Sequential:
    """Build a fully-connected MLP with Tanh activations.

    Args:
        in_dim:     Input dimension.
        hidden_dim: Width of every hidden layer.
        n_layers:   Total number of Linear layers (including the final one).
        out_dim:    Output dimension. Defaults to ``hidden_dim`` when None.
        dropout:    Dropout probability applied after hidden activations (0 = disabled).
    """
    _out = out_dim if out_dim is not None else hidden_dim
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, _out))
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    """1-D DeepONet: G(a)(x) ≈ branch(a) · trunk(x) + bias.

    Args:
        branch_net: Encodes the input function a at sensor points.
        trunk_net:  Encodes the query coordinate x.
    """

    def __init__(self, branch_net: nn.Module, trunk_net: nn.Module) -> None:
        super().__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        a : (batch, n_sensors) — branch input (discretised input function)
        x : (batch, n_dim)     — trunk input (query coordinate)
        returns (batch, 1)
        """
        return (self.branch_net(a) * self.trunk_net(x)).sum(dim=-1, keepdim=True) + self.bias


class DeepONet2D(nn.Module):
    """DeepONet for 2-D heat: G(params)(x,y) ≈ branch(params) · trunk(x,y) + bias.

    Branch: (batch, param_dim) → (batch, p)
    Trunk:  (n_pts, 2)         → (n_pts, p)  — shared across the batch
    Output: einsum 'bp,np→bn'  → (batch, n_pts)

    Args:
        param_dim: Number of scalar input parameters (default 7 for the heat problem).
        p:         Latent dimension shared by branch and trunk outputs.
        width:     MLP hidden width for both branch and trunk.
        depth:     Number of Linear layers in each sub-network.
    """

    def __init__(
        self,
        param_dim: int = 7,
        p: int = 128,
        width: int = 256,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.branch = mlp(param_dim, width, depth, out_dim=p)
        self.trunk  = mlp(2,         width, depth, out_dim=p)
        self.bias   = nn.Parameter(torch.zeros(1))

    def forward(self, params: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """
        params : (batch, param_dim) — normalised physical parameters
        xy     : (n_pts, 2)         — (x, y) query coordinates in [0, 1]²
        returns  (batch, n_pts)
        """
        b = self.branch(params)                                        # (B, p)
        t = torch.tanh(self.trunk(xy))                                 # (N, p) — tanh keeps trunk orthogonal
        return torch.einsum("bp,np->bn", b, t) + self.bias            # (B, N)


class DeepONet3D(nn.Module):
    """DeepONet for 3-D scalar fields: G(params)(q) ≈ branch(params) · trunk(q) + bias.

    Branch: (batch, param_dim) → (batch, p)
    Trunk:  (n_pts, trunk_in_dim) → (n_pts, p)  — shared across the batch
    Output: einsum 'bp,np→bn'     → (batch, n_pts)

    Args:
        param_dim:    Number of scalar input parameters (default 4 for the Lamé sphere).
        p:            Latent dimension shared by branch and trunk outputs.
        width:        MLP hidden width for both branch and trunk.
        depth:        Number of Linear layers in each sub-network.
        trunk_in_dim: Dimension of the trunk query coordinate.
                      Use 3 for raw (x,y,z) or 1 for the radial distance r = ‖xyz‖
                      (preferred for radially-symmetric problems like Lamé sphere).
    """

    def __init__(
        self,
        param_dim: int = 4,
        p: int = 128,
        width: int = 256,
        depth: int = 3,
        trunk_in_dim: int = 3,
    ) -> None:
        super().__init__()
        self.branch = mlp(param_dim,    width, depth, out_dim=p)
        self.trunk  = mlp(trunk_in_dim, width, depth, out_dim=p)
        self.bias   = nn.Parameter(torch.zeros(1))

    def forward(self, params: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        params : (batch, param_dim)    — normalised physical parameters
        q      : (n_pts, trunk_in_dim) — query coordinates (xyz or r)
        returns  (batch, n_pts)
        """
        b = self.branch(params)                              # (B, p)
        t = torch.tanh(self.trunk(q))                        # (N, p)
        return torch.einsum("bp,np->bn", b, t) + self.bias  # (B, N)