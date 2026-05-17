"""Use case: generate a physics simulation dataset."""

from __future__ import annotations

from neural_operators.domain.ports.physics_solver import IPhysicsSolver

__all__ = ["GenerateDatasetUseCase"]


class GenerateDatasetUseCase:
    """Orchestrate physics simulation dataset generation.

    Delegates sampling and solving to an ``IPhysicsSolver`` adapter;
    the use case contains no physics knowledge of its own.
    """

    def __init__(self, solver: IPhysicsSolver) -> None:  # type: ignore[type-arg]
        self._solver = solver

    def execute(
        self,
        n_cases: int,
        seed: int,
        n_workers: int | None = None,
    ) -> list[list[dict]]:  # type: ignore[type-arg]
        """Sample parameters and solve all cases.

        Args:
            n_cases:   Number of independent physics cases to generate.
            seed:      RNG seed for reproducible parameter sampling.
            n_workers: Passed to the solver's parallel executor.
                       ``None`` defaults to ``os.cpu_count()``.

        Returns:
            List of per-case result lists (as returned by the solver).
        """
        params = self._solver.sample_params(n_cases, seed)
        return self._solver.batch_solve(params, n_workers)  # type: ignore[call-arg]
