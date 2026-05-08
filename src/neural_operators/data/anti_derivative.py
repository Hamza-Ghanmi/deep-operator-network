"""Data loader for the 1-D anti-derivative dataset (from NVIDIA Modulus Sym)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["load_anti_derivative"]


def load_anti_derivative(data_path: str | Path) -> dict[str, np.ndarray]:
    """Load ``anti_derivative.npy`` and return a dict of NumPy arrays.

    Keys: ``a_train``, ``x_train``, ``u_train``, ``a_test``, ``x_test``, ``u_test``.

    The file is available from NGC:
    https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/physicsnemo_sym_examples_supplemental_materials
    """
    raw: dict = np.load(Path(data_path), allow_pickle=True).item()
    return raw