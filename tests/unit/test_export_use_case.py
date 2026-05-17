"""Unit tests for ExportModelUseCase — no real .pt files needed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch.nn as nn

from neural_operators.domain.ports.model_exporter import IModelExporter
from neural_operators.use_cases.export_model import ExportModelUseCase


class FakeExporter(IModelExporter):
    """In-memory stub: records calls and creates an empty file to satisfy validation."""

    def __init__(self) -> None:
        self.called = False
        self.last_model: Any = None
        self.last_path: Path | None = None

    def export(self, model: Any, path: Path, trace_input: Any | None = None) -> None:
        self.called = True
        self.last_model = model
        self.last_path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


class FakeModel(nn.Module):
    def forward(self, x: Any) -> Any:
        return x


@pytest.mark.unit
class TestExportModelUseCase:
    def test_delegates_to_exporter(self, tmp_path: Path):
        exporter = FakeExporter()
        use_case = ExportModelUseCase(exporter)
        model = FakeModel()
        out = use_case.execute(model, tmp_path / "model.pt")

        assert exporter.called
        assert exporter.last_model is model
        assert out.exists()

    def test_returns_resolved_path(self, tmp_path: Path):
        use_case = ExportModelUseCase(FakeExporter())
        result = use_case.execute(FakeModel(), tmp_path / "model.pt")
        assert result.is_absolute()

    def test_raises_if_file_not_created(self, tmp_path: Path):
        class BrokenExporter(IModelExporter):
            def export(self, model: Any, path: Path, trace_input: Any | None = None) -> None:
                pass  # does NOT create the file

        use_case = ExportModelUseCase(BrokenExporter())
        with pytest.raises(FileNotFoundError):
            use_case.execute(FakeModel(), tmp_path / "missing.pt")

    def test_trace_input_forwarded(self, tmp_path: Path):
        exporter = FakeExporter()
        use_case = ExportModelUseCase(exporter)
        sentinel = object()
        use_case.execute(FakeModel(), tmp_path / "m.pt", trace_input=sentinel)
        # FakeExporter ignores trace_input but the call must not raise
        assert exporter.called
