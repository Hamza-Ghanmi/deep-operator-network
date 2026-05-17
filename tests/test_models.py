"""Smoke tests for neural operator forward passes (CPU only)."""

from __future__ import annotations

import pytest
import torch

from neural_operators.models import DeepONet, DeepONet2D, mlp


@pytest.mark.unit
def test_deeponet_forward():
    branch_net = mlp(100, 128, 4)
    trunk_net = mlp(1, 128, 4)
    model = DeepONet(branch_net, trunk_net)

    out = model(torch.randn(8, 100), torch.randn(8, 1))
    assert out.shape == (8, 1)


@pytest.mark.unit
def test_deeponet2d_forward():
    model = DeepONet2D(param_dim=7, p=128, width=256, depth=3)

    out = model(torch.randn(4, 7), torch.randn(1024, 2))
    assert out.shape == (4, 1024)


@pytest.mark.unit
def test_fno_forward():
    from neuralop.models import FNO

    model = FNO(n_modes=(4, 4), hidden_channels=8, n_layers=2, in_channels=5, out_channels=1)
    out = model(torch.randn(2, 5, 32, 32))
    assert out.shape == (2, 1, 32, 32)
