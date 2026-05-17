"""TorchScript model exporter — implements IModelExporter for inference-hub."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from neural_operators.domain.ports.model_exporter import IModelExporter

__all__ = ["TorchScriptExporter"]


class TorchScriptExporter(IModelExporter):
    """Serialise a model to TorchScript via ``torch.jit.script`` or ``torch.jit.trace``.

    ``inference-hub``'s ``PyTorchBackend`` loads the result with
    ``torch.jit.load(path)``.

    Strategy:
    - ``torch.jit.script`` is attempted first (graph captures all control flow).
    - If the model is not scriptable (e.g. third-party layers), callers must
      supply ``trace_input``; ``torch.jit.trace`` is used instead.
      Traced models only generalise to the shapes present in ``trace_input``.
    """

    def export(
        self,
        model: Any,
        path: Path,
        trace_input: Any | None = None,
    ) -> None:
        """Save ``model`` as a TorchScript file at ``path``.

        Args:
            model:       ``nn.Module`` to serialise.
            path:        Destination ``.pt`` file (created or overwritten).
            trace_input: Required when ``torch.jit.script`` fails.
                         Pass the example input(s) for ``torch.jit.trace``.
        """
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be nn.Module, got {type(model)}")

        model.eval()
        scripted: torch.ScriptModule

        if trace_input is None:
            scripted = torch.jit.script(model)
        else:
            scripted = torch.jit.trace(model, trace_input)

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.save(scripted, path)
        print(f"  Exported → {path}  ({path.stat().st_size / 1e6:.1f} MB)")
