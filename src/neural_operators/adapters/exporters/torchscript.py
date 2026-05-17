"""torch.export model exporter — implements IModelExporter for inference-hub."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.export import Dim

from neural_operators.domain.ports.model_exporter import IModelExporter

__all__ = ["TorchExportExporter"]


class TorchExportExporter(IModelExporter):
    """Serialise a model via ``torch.export`` for inference-hub.

    Uses ``torch.export.export()`` with a dynamic batch dimension, then writes
    the ``ExportedProgram`` with ``torch.export.save()``.  ``inference-hub``
    loads with ``torch.export.load(...).module()``.

    ``trace_input`` must be a representative ``(1, n_features)`` float32 tensor
    that exercises the full forward path; it is used as both the export example
    and the template for the dynamic batch shape.
    """

    def export(
        self,
        model: Any,
        path: Path,
        trace_input: Any | None = None,
    ) -> None:
        """Save ``model`` as a ``torch.export`` file at ``path``.

        Args:
            model:       ``nn.Module`` to serialise.
            path:        Destination ``.pt`` file (created or overwritten).
            trace_input: Required — a ``(1, n_features)`` float32 example input.
        """
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be nn.Module, got {type(model)}")
        if trace_input is None:
            raise ValueError("trace_input is required for torch.export")

        model.eval()

        batch = Dim("batch", min=1)
        exported = torch.export.export(
            model,
            args=(trace_input,),
            dynamic_shapes={"x": {0: batch}},
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.export.save(exported, path)
        print(f"  Exported → {path}  ({path.stat().st_size / 1e6:.1f} MB)")