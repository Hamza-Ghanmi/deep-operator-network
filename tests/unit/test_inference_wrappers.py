"""Smoke tests for inference wrapper forward passes (CPU only, no checkpoints)."""

from __future__ import annotations

import pytest
import torch

from neural_operators.models import DeepONet2D, DeepONet3D, Heat2DPredictor, LameSpherePredictor


@pytest.mark.unit
def test_heat2d_predictor_output_shape():
    base = DeepONet2D(param_dim=7, p=64, width=64, depth=2)
    predictor = Heat2DPredictor(base)

    batch = torch.tensor([[0.0, 100.0, 0.0, 0.0, 20.0, 0.02, 5.0]])
    out = predictor(batch)
    assert out.shape == (1, 32 * 32)


@pytest.mark.unit
def test_heat2d_predictor_batch():
    base = DeepONet2D(param_dim=7, p=64, width=64, depth=2)
    predictor = Heat2DPredictor(base)

    batch = torch.randn(4, 7)
    # Raw inputs need to be in valid physical ranges for the test to not assert,
    # but shape correctness is sufficient here.
    out = predictor(batch)
    assert out.shape == (4, 32 * 32)


@pytest.mark.unit
def test_lame_sphere_predictor_output_shape():
    base = DeepONet3D(param_dim=1, p=64, width=64, depth=2, trunk_in_dim=1)
    predictor = LameSpherePredictor(base, n_eval_pts=50)

    batch = torch.tensor([[10.0e6, 2.0e6]])  # p_i=10 MPa, p_e=2 MPa
    out = predictor(batch)
    assert out.shape == (1, 50)


@pytest.mark.unit
def test_lame_sphere_predictor_batch():
    base = DeepONet3D(param_dim=1, p=64, width=64, depth=2, trunk_in_dim=1)
    predictor = LameSpherePredictor(base, n_eval_pts=50)

    batch = torch.zeros(3, 2)
    batch[:, 0] = 10.0e6
    batch[:, 1] = 2.0e6
    out = predictor(batch)
    assert out.shape == (3, 50)


@pytest.mark.unit
def test_heat2d_predictor_is_scriptable():
    base = DeepONet2D(param_dim=7, p=32, width=32, depth=1)
    predictor = Heat2DPredictor(base)
    scripted = torch.jit.script(predictor)
    batch = torch.tensor([[0.0, 100.0, 0.0, 0.0, 20.0, 0.02, 5.0]])
    out = scripted(batch)
    assert out.shape == (1, 32 * 32)


@pytest.mark.unit
def test_lame_sphere_predictor_is_scriptable():
    base = DeepONet3D(param_dim=1, p=32, width=32, depth=1, trunk_in_dim=1)
    predictor = LameSpherePredictor(base, n_eval_pts=20)
    scripted = torch.jit.script(predictor)
    batch = torch.tensor([[10.0e6, 2.0e6]])
    out = scripted(batch)
    assert out.shape == (1, 20)
