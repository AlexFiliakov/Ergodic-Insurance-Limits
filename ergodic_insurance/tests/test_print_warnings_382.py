"""Tests for issue #382 and #1310: replace print() with logging/warnings module.

Verifies that:
- Custom warning classes exist and have the correct hierarchy.
- Modules that previously used print() for warnings now use logging.warning().
- No bare print("Warning: ...") calls remain in the package source.
- Production modules (progress_monitor, batch_processor, trajectory_storage,
  bootstrap_analysis) use logging instead of print() (#1310).
"""

import ast
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch
import warnings

import numpy as np
import pytest

from ergodic_insurance._warnings import (
    ConfigurationWarning,
    DataQualityWarning,
    ErgodicInsuranceDeprecationWarning,
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

    def test_deprecation_warning(self):
        assert issubclass(ErgodicInsuranceDeprecationWarning, DeprecationWarning)
        assert not issubclass(ErgodicInsuranceDeprecationWarning, ErgodicInsuranceWarning)

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

        cls = getattr(ergodic_insurance, "ErgodicInsuranceDeprecationWarning")
        assert issubclass(cls, DeprecationWarning)

    def test_importable_but_not_in_all(self):
        """Warning classes are importable via __getattr__ but not in __all__.

        Since #477, only essential user-facing names are in __all__.
        Warning classes remain accessible via lazy imports for backward compat.
        """
        import ergodic_insurance

        for name in (
            "ErgodicInsuranceWarning",
            "ConfigurationWarning",
            "DataQualityWarning",
            "ExportWarning",
        ):
            # Still importable via __getattr__
            cls = getattr(ergodic_insurance, name)
            assert issubclass(cls, UserWarning)
            # But not in the trimmed __all__
            assert name not in ergodic_insurance.__all__

        # ErgodicInsuranceDeprecationWarning is a DeprecationWarning, not UserWarning
        cls = getattr(ergodic_insurance, "ErgodicInsuranceDeprecationWarning")
        assert issubclass(cls, DeprecationWarning)
        assert "ErgodicInsuranceDeprecationWarning" not in ergodic_insurance.__all__


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
# progress_monitor.py — logging instead of print (#1310)
# ---------------------------------------------------------------------------


class TestProgressMonitorLogging:
    """progress_monitor.py uses logging instead of print()."""

    def test_module_has_logger(self):
        """Module defines a module-level logger."""
        from ergodic_insurance import progress_monitor

        assert hasattr(progress_monitor, "logger")
        assert isinstance(progress_monitor.logger, logging.Logger)

    def test_convergence_message_logged(self, caplog):
        """Convergence achievement is logged, not printed."""
        from ergodic_insurance.progress_monitor import ProgressMonitor

        monitor = ProgressMonitor(total_iterations=100_000, show_console=True)
        with caplog.at_level(logging.INFO, logger="ergodic_insurance.progress_monitor"):
            monitor._print_convergence_message(50_000, 1.05)

        assert any("Convergence achieved" in r.message for r in caplog.records)
        assert any("[OK]" in r.message for r in caplog.records)

    def test_finalize_logs_summary(self, caplog):
        """finalize() logs summary via logger, not print()."""
        import time

        from ergodic_insurance.progress_monitor import ProgressMonitor

        monitor = ProgressMonitor(total_iterations=1000, show_console=True)
        monitor.current_iteration = 1000
        # Ensure non-zero elapsed time to avoid division by zero
        monitor.start_time = time.time() - 1.0

        with caplog.at_level(logging.INFO, logger="ergodic_insurance.progress_monitor"):
            monitor.finalize()

        assert any("Simulation Complete" in r.message for r in caplog.records)

    def test_finalize_uses_ascii_markers(self, caplog):
        """finalize() uses ASCII [OK]/[FAIL] instead of Unicode emoji."""
        import time

        from ergodic_insurance.progress_monitor import ProgressMonitor

        monitor = ProgressMonitor(total_iterations=1000, show_console=True)
        monitor.current_iteration = 1000
        monitor.converged = True
        monitor.converged_at = 500
        # Ensure non-zero elapsed time to avoid division by zero
        monitor.start_time = time.time() - 1.0

        with caplog.at_level(logging.INFO, logger="ergodic_insurance.progress_monitor"):
            monitor.finalize()

        log_text = " ".join(r.message for r in caplog.records)
        assert "[OK]" in log_text
        # No Unicode checkmark or cross
        assert "\u2713" not in log_text
        assert "\u2717" not in log_text

    def test_console_update_uses_debug(self, caplog):
        """_update_console logs at DEBUG level."""
        import time

        from ergodic_insurance.progress_monitor import ProgressMonitor

        monitor = ProgressMonitor(total_iterations=1000, show_console=True)
        monitor.current_iteration = 500

        with caplog.at_level(logging.DEBUG, logger="ergodic_insurance.progress_monitor"):
            monitor._update_console(time.time())

        assert any(r.levelno == logging.DEBUG for r in caplog.records)

    def test_no_print_in_source(self):
        """progress_monitor.py has zero print() calls in executable code."""
        from ergodic_insurance import progress_monitor

        source_path = Path(progress_monitor.__file__)
        _assert_no_print_calls(source_path)


# ---------------------------------------------------------------------------
# batch_processor.py — logging instead of print (#1310)
# ---------------------------------------------------------------------------


class TestBatchProcessorLogging:
    """batch_processor.py uses logging instead of print()."""

    def test_module_has_logger(self):
        """Module defines a module-level logger."""
        from ergodic_insurance import batch_processor

        assert hasattr(batch_processor, "logger")
        assert isinstance(batch_processor.logger, logging.Logger)

    def test_all_completed_logs_info(self, caplog):
        """'All scenarios already completed' is logged, not printed."""
        from ergodic_insurance.batch_processor import BatchProcessor

        processor = BatchProcessor.__new__(BatchProcessor)
        processor.completed_scenarios = {"s1"}
        processor.failed_scenarios = set()
        processor.batch_results = []
        processor.checkpoint_dir = Path("/tmp/test_checkpoints")
        processor.use_parallel = False
        processor.progress_bar = False

        mock_scenario = MagicMock()
        mock_scenario.scenario_id = "s1"
        mock_scenario.priority = 1

        with (
            patch.object(processor, "_aggregate_results", return_value=MagicMock()),
            patch.object(processor, "_load_checkpoint"),
            caplog.at_level(logging.INFO, logger="ergodic_insurance.batch_processor"),
        ):
            processor.process_batch([mock_scenario], resume_from_checkpoint=False)

        assert any("All scenarios already completed" in r.message for r in caplog.records)

    def test_no_print_in_source(self):
        """batch_processor.py has zero print() calls in executable code."""
        from ergodic_insurance import batch_processor

        source_path = Path(batch_processor.__file__)
        _assert_no_print_calls(source_path)


# ---------------------------------------------------------------------------
# trajectory_storage.py — logging instead of print (#1310)
# ---------------------------------------------------------------------------


class TestTrajectoryStorageLogging:
    """trajectory_storage.py uses logging instead of print()."""

    def test_module_has_logger(self):
        """Module defines a module-level logger."""
        from ergodic_insurance import trajectory_storage

        assert hasattr(trajectory_storage, "logger")
        assert isinstance(trajectory_storage.logger, logging.Logger)

    def test_export_csv_logs_info(self, caplog, tmp_path):
        """export_summaries_csv logs export count via logger."""
        from ergodic_insurance.trajectory_storage import (
            SimulationSummary,
            StorageConfig,
            TrajectoryStorage,
        )

        config = StorageConfig(storage_dir=str(tmp_path / "storage"), backend="memmap")
        storage = TrajectoryStorage(config)

        # Add a summary directly
        storage._summaries[0] = SimulationSummary(
            sim_id=0,
            final_assets=100.0,
            total_losses=10.0,
            total_recoveries=5.0,
            mean_annual_loss=1.0,
            max_annual_loss=3.0,
            min_annual_loss=0.1,
            growth_rate=0.05,
            ruin_occurred=False,
        )

        output_csv = str(tmp_path / "summaries.csv")
        with caplog.at_level(logging.INFO, logger="ergodic_insurance.trajectory_storage"):
            storage.export_summaries_csv(output_csv)

        assert any("Exported" in r.message for r in caplog.records)

    def test_no_print_in_source(self):
        """trajectory_storage.py has zero print() calls in executable code."""
        from ergodic_insurance import trajectory_storage

        source_path = Path(trajectory_storage.__file__)
        _assert_no_print_calls(source_path)


# ---------------------------------------------------------------------------
# bootstrap_analysis.py — logging instead of print (#1310)
# ---------------------------------------------------------------------------


class TestBootstrapAnalysisLogging:
    """bootstrap_analysis.py uses logging instead of print()."""

    def test_module_has_logger(self):
        """Module defines a module-level logger."""
        from ergodic_insurance import bootstrap_analysis

        assert hasattr(bootstrap_analysis, "logger")
        assert isinstance(bootstrap_analysis.logger, logging.Logger)

    def test_parallel_fallback_logs_warning(self, caplog):
        """Parallel bootstrap fallback logs a warning, not a print."""
        from ergodic_insurance.bootstrap_analysis import BootstrapAnalyzer

        analyzer = BootstrapAnalyzer(n_bootstrap=100, seed=42, show_progress=False)

        # Make _parallel_bootstrap always fail
        with (
            patch.object(
                analyzer,
                "_parallel_bootstrap",
                side_effect=RuntimeError("pool error"),
            ),
            caplog.at_level(logging.WARNING, logger="ergodic_insurance.bootstrap_analysis"),
        ):
            result = analyzer.confidence_interval(
                np.random.default_rng(42).normal(0, 1, 100),
                np.mean,
                parallel=True,
            )

        assert any("Parallel bootstrap failed" in r.message for r in caplog.records)
        assert any(
            r.levelno == logging.WARNING
            for r in caplog.records
            if "Parallel bootstrap failed" in r.message
        )

    def test_no_print_in_source(self):
        """bootstrap_analysis.py has zero print() calls in executable code."""
        from ergodic_insurance import bootstrap_analysis

        source_path = Path(bootstrap_analysis.__file__)
        _assert_no_print_calls(source_path)


# ---------------------------------------------------------------------------
# No remaining print("Warning: …") in package source
# ---------------------------------------------------------------------------


def _assert_no_print_calls(source_path: Path) -> None:
    """Assert that a Python source file has zero print() calls in executable code.

    Uses AST parsing to distinguish real print() calls from those in
    docstrings and comments.
    """
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id == "print":
                violations.append(f"  line {node.lineno}: print() call")

    assert violations == [], f"Found print() calls in {source_path.name}:\n" + "\n".join(violations)


class TestNoBareWarningPrints:
    """Ensure no print('Warning: ...') calls remain in package source."""

    def test_no_print_warnings_in_source(self):
        """Scan all .py source files for print('Warning: ...')."""
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

    def test_no_print_in_issue_1310_modules(self):
        """Modules listed in issue #1310 have zero print() calls."""
        from ergodic_insurance import (
            batch_processor,
            bootstrap_analysis,
            progress_monitor,
            sensitivity,
            trajectory_storage,
        )

        for module in (
            progress_monitor,
            batch_processor,
            trajectory_storage,
            sensitivity,
            bootstrap_analysis,
        ):
            assert module.__file__ is not None
            source_path = Path(module.__file__)
            _assert_no_print_calls(source_path)
