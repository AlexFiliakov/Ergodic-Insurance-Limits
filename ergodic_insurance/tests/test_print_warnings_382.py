"""Tests for issue #382: replace print() warnings with logging/warnings module.

Verifies that:
- Custom warning classes exist and have the correct hierarchy.
- Modules that previously used print() for warnings now use logging.warning().
- No bare print("Warning: ...") calls remain in the package source.
"""

import logging
from unittest.mock import MagicMock
import warnings

import pytest

from ergodic_insurance._warnings import (
    ConfigurationWarning,
    DataQualityWarning,
    ErgodicInsuranceWarning,
    ExportWarning,
)

# ---------------------------------------------------------------------------
# Warning class hierarchy
# ---------------------------------------------------------------------------


class TestWarningClassHierarchy:
    """Custom warning classes have the expected inheritance chain."""

    def test_base_is_user_warning(self):
        assert issubclass(ErgodicInsuranceWarning, UserWarning)

    def test_configuration_warning(self):
        assert issubclass(ConfigurationWarning, ErgodicInsuranceWarning)
        assert issubclass(ConfigurationWarning, UserWarning)

    def test_data_quality_warning(self):
        assert issubclass(DataQualityWarning, ErgodicInsuranceWarning)
        assert issubclass(DataQualityWarning, UserWarning)

    def test_export_warning(self):
        assert issubclass(ExportWarning, ErgodicInsuranceWarning)
        assert issubclass(ExportWarning, UserWarning)

    def test_can_be_raised_and_caught(self):
        with pytest.warns(ConfigurationWarning):
            warnings.warn("test", ConfigurationWarning)

    def test_filterable(self):
        """Users can suppress warnings by category."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", category=ConfigurationWarning)
            warnings.warn("should be suppressed", ConfigurationWarning)
            warnings.warn("should appear", DataQualityWarning)
            assert len(w) == 1
            assert issubclass(w[0].category, DataQualityWarning)


# ---------------------------------------------------------------------------
# Package-level lazy imports
# ---------------------------------------------------------------------------


class TestPackageLevelImports:
    """Warning classes are accessible from the top-level package."""

    def test_import_from_package(self):
        import ergodic_insurance  # pylint: disable=reimported

        for name in (
            "ConfigurationWarning",
            "DataQualityWarning",
            "ErgodicInsuranceWarning",
            "ExportWarning",
        ):
            cls = getattr(ergodic_insurance, name)
            assert issubclass(cls, UserWarning)

    def test_in_all(self):
        import ergodic_insurance

        for name in (
            "ErgodicInsuranceWarning",
            "ConfigurationWarning",
            "DataQualityWarning",
            "ExportWarning",
        ):
            assert name in ergodic_insurance.__all__


# ---------------------------------------------------------------------------
# sensitivity.py — logging instead of print
# ---------------------------------------------------------------------------


class TestSensitivityLogging:
    """sensitivity.py uses logger.warning() instead of print()."""

    def test_analyze_one_way_logs_on_error(self, caplog):
        """create_tornado_diagram logs when a parameter fails analysis."""
        from ergodic_insurance.sensitivity import SensitivityAnalyzer

        # Build a minimal mock optimizer whose run always raises
        mock_optimizer = MagicMock()
        mock_optimizer.optimize.side_effect = RuntimeError("boom")

        analyzer = SensitivityAnalyzer.__new__(SensitivityAnalyzer)
        analyzer.base_config = MagicMock()
        analyzer.optimizer = mock_optimizer
        analyzer.results_cache = {}

        with caplog.at_level(logging.WARNING, logger="ergodic_insurance.sensitivity"):
            # analyze_parameter will raise; tornado diagram skips it
            df = analyzer.create_tornado_diagram(
                parameters=["bad_param"],
                metric="optimal_roe",
            )

        assert any("bad_param" in r.message for r in caplog.records)
        assert all(r.levelno == logging.WARNING for r in caplog.records if "bad_param" in r.message)

    def test_analyze_parameter_group_logs_on_error(self, caplog):
        """analyze_parameter_group logs when a parameter fails analysis."""
        from ergodic_insurance.sensitivity import SensitivityAnalyzer

        mock_optimizer = MagicMock()
        mock_optimizer.optimize.side_effect = RuntimeError("boom")

        analyzer = SensitivityAnalyzer.__new__(SensitivityAnalyzer)
        analyzer.base_config = MagicMock()
        analyzer.optimizer = mock_optimizer
        analyzer.results_cache = {}

        with caplog.at_level(logging.WARNING, logger="ergodic_insurance.sensitivity"):
            results = analyzer.analyze_parameter_group({"bad_param": (0, 1)}, n_points=3)

        assert any("bad_param" in r.message for r in caplog.records)
        assert results == {}


# ---------------------------------------------------------------------------
# visualization/export.py — logging instead of print
# ---------------------------------------------------------------------------


class TestExportLogging:
    """visualization/export.py uses logger.warning() instead of print()."""

    def test_save_figure_logs_on_kaleido_missing(self, caplog, tmp_path):
        """save_figure logs when plotly static export fails."""
        import plotly.graph_objects as go

        from ergodic_insurance.visualization.export import save_figure

        fig = go.Figure()
        fig.write_image = MagicMock(side_effect=ValueError("kaleido not installed"))

        with caplog.at_level(logging.WARNING, logger="ergodic_insurance.visualization.export"):
            result = save_figure(fig, str(tmp_path / "test"), formats=["png"])

        assert any("kaleido" in r.message for r in caplog.records)
        assert all(r.levelno == logging.WARNING for r in caplog.records if "kaleido" in r.message)

    def test_save_for_presentation_logs_on_kaleido_missing(self, caplog, tmp_path):
        """save_for_presentation logs when plotly static export fails."""
        import plotly.graph_objects as go

        from ergodic_insurance.visualization.export import save_for_presentation

        fig = go.Figure()
        fig.write_image = MagicMock(side_effect=ValueError("kaleido not installed"))

        with caplog.at_level(logging.WARNING, logger="ergodic_insurance.visualization.export"):
            save_for_presentation(fig, str(tmp_path / "pub"))

        assert any("kaleido" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# No remaining print("Warning: …") in package source
# ---------------------------------------------------------------------------


class TestNoBareWarningPrints:
    """Ensure no print('Warning: ...') calls remain in package source."""

    def test_no_print_warnings_in_source(self):
        """Scan all .py source files for print('Warning: ...')."""
        from pathlib import Path
        import re

        pkg_root = Path(__file__).resolve().parent.parent
        pattern = re.compile(r"""print\(f?['"]Warning:""")

        violations = []
        for py_file in pkg_root.rglob("*.py"):
            # Skip tests and notebooks
            rel = py_file.relative_to(pkg_root)
            if "tests" in rel.parts or "notebooks" in rel.parts:
                continue
            text = py_file.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    violations.append(f"{rel}:{i}: {line.strip()}")

        assert violations == [], "Found bare print() warnings in package source:\n" + "\n".join(
            violations
        )
