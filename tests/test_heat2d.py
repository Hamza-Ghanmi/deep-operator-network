"""Smoke tests for the heat2d data module."""

from __future__ import annotations

import numpy as np

from neural_operators.data.heat2d import analytical_field, sample_params, solve_case


def test_solve_case_shape():
    rows = solve_case((0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.02))
    assert len(rows) == 10
    assert all(len(r["T_field"]) == 32 * 32 for r in rows)
    assert all(r["case_id"] == 0 for r in rows)


def test_analytical_field_uniform_bc():
    # All BCs = T_init → zero transient, field stays flat at T everywhere
    T = 75.0
    t_values = np.array([0.5, 5.0, 30.0])
    field = analytical_field(t_values, T, T, T, T, T, 0.02)
    assert field.shape == (3, 32, 32)
    np.testing.assert_allclose(field, T, atol=0.01)


def test_sample_params_ranges():
    params = sample_params(20, seed=0)
    assert len(params) == 20
    for case_id, T_L, T_R, T_B, T_T, T_init, alpha in params:
        assert 0.005 <= alpha <= 0.05
        assert 0.0 <= T_L <= 200.0
        assert 0.0 <= T_init <= 100.0
