"""Port: physics simulation solver interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

__all__ = ["IPhysicsSolver"]

P = TypeVar("P")
R = TypeVar("R")


class IPhysicsSolver(ABC, Generic[P, R]):
    """Adapter contract for generating physics simulation datasets.

    Type parameters:
        P: parameter type (e.g., a tuple of floats or a domain entity)
        R: one solved-case result (e.g., list[dict] for Parquet rows)
    """

    @abstractmethod
    def sample_params(self, n_cases: int, seed: int) -> list[P]:
        """Draw ``n_cases`` random parameter sets deterministically from ``seed``."""

    @abstractmethod
    def batch_solve(self, params_list: list[P]) -> list[R]:
        """Solve all parameter sets and return results in the same order."""
