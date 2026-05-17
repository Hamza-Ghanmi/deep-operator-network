"""Domain entities for the Lamé thick-walled hollow sphere problem."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["LameSphereParams", "LameSphereField"]


@dataclass(frozen=True)
class LameSphereParams:
    """Physical parameters for one Lamé sphere case."""

    case_id: int
    p_i: float
    p_e: float
    E: float
    nu: float


@dataclass(frozen=True)
class LameSphereField:
    """Stress and displacement fields at all FEM mesh query points."""

    case_id: int
    p_i: float
    p_e: float
    E: float
    nu: float
    sigma_vm: list[float] = field(default_factory=list)
    u_r: list[float] = field(default_factory=list)
    sigma_r: list[float] = field(default_factory=list)
    sigma_theta: list[float] = field(default_factory=list)
