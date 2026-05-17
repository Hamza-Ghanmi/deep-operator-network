"""Unit tests for domain entities — no PyTorch, no external data files."""

from __future__ import annotations

import pytest

from neural_operators.domain.entities import (
    Heat2DField,
    Heat2DParams,
    LameSphereField,
    LameSphereParams,
)


@pytest.mark.unit
class TestHeat2DParams:
    def test_construction(self):
        p = Heat2DParams(case_id=0, T_L=0.0, T_R=100.0, T_B=50.0, T_T=25.0, T_init=20.0, alpha=0.02)
        assert p.case_id == 0
        assert p.alpha == 0.02

    def test_frozen(self):
        p = Heat2DParams(case_id=0, T_L=0.0, T_R=100.0, T_B=50.0, T_T=25.0, T_init=20.0, alpha=0.02)
        with pytest.raises(Exception):
            p.T_L = 99.0  # type: ignore[misc]

    def test_equality(self):
        p1 = Heat2DParams(0, 0.0, 100.0, 50.0, 25.0, 20.0, 0.02)
        p2 = Heat2DParams(0, 0.0, 100.0, 50.0, 25.0, 20.0, 0.02)
        assert p1 == p2


@pytest.mark.unit
class TestHeat2DField:
    def test_construction_with_defaults(self):
        f = Heat2DField(case_id=0, t=5.0, alpha=0.02,
                        T_L=0.0, T_R=100.0, T_B=50.0, T_T=25.0, T_init=20.0)
        assert f.T_field == []

    def test_construction_with_field(self):
        vals = [1.0] * 1024
        f = Heat2DField(case_id=1, t=1.0, alpha=0.01,
                        T_L=0.0, T_R=0.0, T_B=0.0, T_T=0.0, T_init=0.0, T_field=vals)
        assert len(f.T_field) == 1024


@pytest.mark.unit
class TestLameSphereParams:
    def test_construction(self):
        p = LameSphereParams(case_id=7, p_i=10e6, p_e=2e6, E=200e9, nu=0.3)
        assert p.p_i == 10e6

    def test_frozen(self):
        p = LameSphereParams(case_id=0, p_i=10e6, p_e=2e6, E=200e9, nu=0.3)
        with pytest.raises(Exception):
            p.p_i = 5e6  # type: ignore[misc]


@pytest.mark.unit
class TestLameSphereField:
    def test_construction_with_defaults(self):
        f = LameSphereField(case_id=0, p_i=10e6, p_e=2e6, E=200e9, nu=0.3)
        assert f.sigma_vm == []

    def test_construction_with_fields(self):
        n = 100
        f = LameSphereField(
            case_id=0, p_i=10e6, p_e=2e6, E=200e9, nu=0.3,
            sigma_vm=[1.0] * n, u_r=[0.0] * n,
            sigma_r=[0.5] * n, sigma_theta=[0.3] * n,
        )
        assert len(f.sigma_vm) == n
