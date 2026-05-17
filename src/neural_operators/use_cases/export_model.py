"""Use case: export a trained model to a TorchScript file for inference-hub."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from neural_operators.domain.ports.model_exporter import IModelExporter

__all__ = ["ExportModelUseCase"]


class ExportModelUseCase:
    """Orchestrate model serialisation.

    Delegates the actual serialisation format to an ``IModelExporter`` adapter.
    Validates that the output file was created so callers receive a clear error
    rather than a silent no-op.
    """

    def __init__(self, exporter: IModelExporter) -> None:
        self._exporter = exporter

    def execute(
        self,
        model: Any,
        path: Path,
        trace_input: Any | None = None,
    ) -> Path:
        """Export ``model`` and return the resolved output path.

        Args:
            model:       Trained ``nn.Module`` to serialise.
            path:        Destination file (passed to the exporter).
            trace_input: Example input for trace-based exporters; ``None``
                         when the exporter uses ``torch.jit.script``.

        Returns:
            The absolute path to the exported file.

        Raises:
            FileNotFoundError: If the exporter did not create the output file.
        """
        self._exporter.export(model, path, trace_input)
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"Export reported success but {resolved} was not created."
            )
        return resolved
