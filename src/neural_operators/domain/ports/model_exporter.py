"""Port: model serialisation interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

__all__ = ["IModelExporter"]


class IModelExporter(ABC):
    """Adapter contract for exporting trained models to a loadable format.

    ``inference-hub``'s ``PyTorchBackend`` loads the result with
    ``torch.export.load(path).module()``.  Implementations document their
    serialisation format.
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
            trace_input: Representative example input tensor required by
                         ``torch.export``-based implementations.
        """
