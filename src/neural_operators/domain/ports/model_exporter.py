"""Port: model serialisation interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

__all__ = ["IModelExporter"]


class IModelExporter(ABC):
    """Adapter contract for exporting trained models to a loadable format.

    ``inference-hub``'s ``PyTorchBackend`` expects TorchScript files produced
    by ``torch.jit.save``.  Implementations are free to use ``torch.jit.script``
    or ``torch.jit.trace``; the choice is documented per-adapter.
    """

    @abstractmethod
    def export(
        self,
        model: Any,
        path: Path,
        trace_input: Any | None = None,
    ) -> None:
        """Serialise ``model`` to ``path``.

        Args:
            model:       The trained model to export.
            path:        Destination file path (created or overwritten).
            trace_input: Example input tensor(s) required by ``torch.jit.trace``.
                         May be ``None`` when using ``torch.jit.script``.
        """
