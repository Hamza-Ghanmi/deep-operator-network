"""Domain entities for the 2-D heat equation problem."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Heat2DParams", "Heat2DField"]


@dataclass(frozen=True)
class Heat2DParams:
    """Physical parameters for one 2-D heat equation case."""

    case_id: int
    T_L: float
    T_R: float
    T_B: float
    T_T: float
    T_init: float
    alpha: float


@dataclass(frozen=True)
class Heat2DField:
    """One time-snapshot of the 2-D temperature field (NX × NY flattened)."""

    case_id: int
    t: float
    alpha: float
    T_L: float
    T_R: float
    T_B: float
    T_T: float
    T_init: float
    T_field: list[float] = field(default_factory=list)
