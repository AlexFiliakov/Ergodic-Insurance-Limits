"""Tests for visualization namespace grouping (issue #1317).

Validates that:
- Submodule-level imports work (visualization.executive.plot_loss_distribution)
- FigureFactory has domain-specific convenience methods
- __all__ is organized by category and consistent with actual exports
- Backward compatibility is maintained for all existing flat imports
"""

import importlib
import sys
import types

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")  # non-interactive backend for CI


# ---------------------------------------------------------------------------
# Namespace submodule imports
# ---------------------------------------------------------------------------


class TestNamespaceSubmodules:
    """Test that namespace submodule aliases are importable."""

    def test_import_executive_namespace(self):
        from ergodic_insurance.visualization import executive

        assert isinstance(executive, types.ModuleType)
        assert hasattr(executive, "plot_loss_distribution")
        assert hasattr(executive, "plot_return_period_curve")
        assert hasattr(executive, "plot_insurance_layers")
        assert hasattr(executive, "plot_roe_ruin_frontier")
        assert hasattr(executive, "plot_ruin_cliff")

    def test_import_technical_namespace(self):
        from ergodic_insurance.visualization import technical

        assert isinstance(technical, types.ModuleType)
        assert hasattr(technical, "plot_convergence_diagnostics")
        assert hasattr(technical, "plot_ergodic_divergence")
        assert hasattr(technical, "plot_pareto_frontier_2d")
        assert hasattr(technical, "plot_path_dependent_wealth")

    def test_import_interactive_namespace(self):
        from ergodic_insurance.visualization import interactive

        assert isinstance(interactive, types.ModuleType)
        assert hasattr(interactive, "create_interactive_dashboard")
        assert hasattr(interactive, "create_risk_dashboard")
        assert hasattr(interactive, "create_time_series_dashboard")
        assert hasattr(interactive, "create_correlation_heatmap")

    def test_import_batch_namespace(self):
        from ergodic_insurance.visualization import batch

        assert isinstance(batch, types.ModuleType)
        assert hasattr(batch, "plot_scenario_comparison")
        assert hasattr(batch, "plot_sensitivity_heatmap")
        assert hasattr(batch, "plot_parameter_sweep_3d")

    def test_import_annotations_namespace(self):
        from ergodic_insurance.visualization import annotations

        assert isinstance(annotations, types.ModuleType)
        assert hasattr(annotations, "add_value_labels")
        assert hasattr(annotations, "add_footnote")
        assert hasattr(annotations, "add_callout")

    def test_import_export_namespace(self):
        from ergodic_insurance.visualization import export

        assert isinstance(export, types.ModuleType)
        assert hasattr(export, "save_figure")
        assert hasattr(export, "save_for_publication")
        assert hasattr(export, "batch_export")

    def test_import_core_namespace(self):
        from ergodic_insurance.visualization import core

        assert isinstance(core, types.ModuleType)
        assert hasattr(core, "WSJ_COLORS")
        assert hasattr(core, "COLOR_SEQUENCE")

    def test_sys_modules_alias_executive(self):
        """Verify that 'import ergodic_insurance.visualization.executive' resolves."""
        from ergodic_insurance.visualization import executive as exec_mod

        assert hasattr(exec_mod, "plot_loss_distribution")

    def test_sys_modules_alias_technical(self):
        from ergodic_insurance.visualization import technical as tech_mod

        assert hasattr(tech_mod, "plot_convergence_diagnostics")

    def test_sys_modules_alias_interactive(self):
        from ergodic_insurance.visualization import interactive as inter_mod

        assert hasattr(inter_mod, "create_interactive_dashboard")

    def test_sys_modules_alias_batch(self):
        from ergodic_insurance.visualization import batch as batch_mod

        assert hasattr(batch_mod, "plot_scenario_comparison")

    def test_from_submodule_import_function(self):
        """Test that ``from ergodic_insurance.visualization.executive import func`` works."""
        from ergodic_insurance.visualization.executive import plot_loss_distribution

        assert callable(plot_loss_distribution)

    def test_from_submodule_import_technical_function(self):
        from ergodic_insurance.visualization.technical import plot_convergence_diagnostics

        assert callable(plot_convergence_diagnostics)

    def test_from_submodule_import_interactive_function(self):
        from ergodic_insurance.visualization.interactive import create_interactive_dashboard

        assert callable(create_interactive_dashboard)

    def test_from_submodule_import_batch_function(self):
        from ergodic_insurance.visualization.batch import plot_scenario_comparison

        assert callable(plot_scenario_comparison)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Ensure all existing import paths still work."""

    def test_flat_import_executive_functions(self):
        from ergodic_insurance.visualization import (
            plot_insurance_layers,
            plot_loss_distribution,
            plot_return_period_curve,
            plot_roe_ruin_frontier,
            plot_ruin_cliff,
        )

        assert callable(plot_loss_distribution)
        assert callable(plot_return_period_curve)
        assert callable(plot_insurance_layers)
        assert callable(plot_roe_ruin_frontier)
        assert callable(plot_ruin_cliff)

    def test_flat_import_technical_functions(self):
        from ergodic_insurance.visualization import (
            create_interactive_pareto_frontier,
            plot_convergence_diagnostics,
            plot_ergodic_divergence,
            plot_pareto_frontier_2d,
            plot_path_dependent_wealth,
            plot_trace_plots,
        )

        assert callable(plot_convergence_diagnostics)
        assert callable(plot_ergodic_divergence)
        assert callable(plot_pareto_frontier_2d)
        assert callable(plot_path_dependent_wealth)
        assert callable(plot_trace_plots)
        assert callable(create_interactive_pareto_frontier)

    def test_flat_import_interactive_functions(self):
        from ergodic_insurance.visualization import (
            create_correlation_heatmap,
            create_interactive_dashboard,
            create_risk_dashboard,
            create_time_series_dashboard,
        )

        assert callable(create_interactive_dashboard)
        assert callable(create_time_series_dashboard)
        assert callable(create_correlation_heatmap)
        assert callable(create_risk_dashboard)

    def test_flat_import_batch_functions(self):
        from ergodic_insurance.visualization import (
            plot_parallel_scenarios,
            plot_parameter_sweep_3d,
            plot_scenario_comparison,
            plot_scenario_convergence,
            plot_sensitivity_heatmap,
        )

        assert callable(plot_scenario_comparison)
        assert callable(plot_sensitivity_heatmap)
        assert callable(plot_parameter_sweep_3d)
        assert callable(plot_scenario_convergence)
        assert callable(plot_parallel_scenarios)

    def test_flat_import_annotation_functions(self):
        from ergodic_insurance.visualization import (
            add_benchmark_line,
            add_callout,
            add_data_source,
            add_footnote,
            add_shaded_region,
            add_trend_annotation,
            add_value_labels,
        )

        assert callable(add_value_labels)
        assert callable(add_trend_annotation)
        assert callable(add_callout)
        assert callable(add_benchmark_line)
        assert callable(add_shaded_region)
        assert callable(add_data_source)
        assert callable(add_footnote)

    def test_flat_import_export_functions(self):
        from ergodic_insurance.visualization import (
            batch_export,
            save_figure,
            save_for_presentation,
            save_for_publication,
            save_for_web,
        )

        assert callable(save_figure)
        assert callable(save_for_publication)
        assert callable(save_for_presentation)
        assert callable(save_for_web)
        assert callable(batch_export)

    def test_flat_import_factory_and_style(self):
        from ergodic_insurance.visualization import FigureFactory, StyleManager, Theme

        assert FigureFactory is not None
        assert StyleManager is not None
        assert Theme is not None

    def test_flat_import_core_constants(self):
        from ergodic_insurance.visualization import (
            COLOR_SEQUENCE,
            WSJ_COLORS,
            WSJFormatter,
            format_currency,
            format_percentage,
            set_wsj_style,
        )

        assert isinstance(WSJ_COLORS, dict)
        assert isinstance(COLOR_SEQUENCE, list)
        assert callable(set_wsj_style)
        assert callable(format_currency)
        assert callable(format_percentage)

    def test_original_submodule_paths_still_work(self):
        """The original _plots-suffixed modules must remain importable."""
        from ergodic_insurance.visualization import batch_plots  # noqa: F401
        from ergodic_insurance.visualization import executive_plots  # noqa: F401
        from ergodic_insurance.visualization import interactive_plots  # noqa: F401
        from ergodic_insurance.visualization import technical_plots  # noqa: F401


# ---------------------------------------------------------------------------
# __all__ consistency
# ---------------------------------------------------------------------------


class TestAllConsistency:
    """Verify __all__ is comprehensive and consistent."""

    def test_all_contains_namespace_modules(self):
        import ergodic_insurance.visualization as viz

        for name in (
            "executive",
            "technical",
            "interactive",
            "batch",
            "annotations",
            "export",
            "core",
        ):
            assert name in viz.__all__, f"{name!r} missing from __all__"

    def test_all_contains_factory_and_style(self):
        import ergodic_insurance.visualization as viz

        for name in ("FigureFactory", "StyleManager", "Theme"):
            assert name in viz.__all__, f"{name!r} missing from __all__"

    def test_all_items_are_importable(self):
        import ergodic_insurance.visualization as viz

        for name in viz.__all__:
            assert hasattr(viz, name), f"{name!r} in __all__ but not importable"

    def test_submodule_all_matches_init_imports(self):
        """Each submodule's __all__ should be a subset of the top-level imports."""
        import ergodic_insurance.visualization as viz
        from ergodic_insurance.visualization import executive_plots

        for fn_name in executive_plots.__all__:
            assert hasattr(viz, fn_name), (
                f"executive_plots.__all__ has {fn_name!r} "
                "but it is not re-exported by visualization"
            )


# ---------------------------------------------------------------------------
# FigureFactory domain methods
# ---------------------------------------------------------------------------


class TestFigureFactoryDomainMethods:
    """Test FigureFactory has convenience methods for common plot types."""

    def test_has_executive_methods(self):
        from ergodic_insurance.visualization import FigureFactory

        factory = FigureFactory()
        assert callable(getattr(factory, "loss_distribution", None))
        assert callable(getattr(factory, "return_period_curve", None))
        assert callable(getattr(factory, "insurance_layers", None))
        assert callable(getattr(factory, "roe_ruin_frontier", None))
        assert callable(getattr(factory, "ruin_cliff", None))

    def test_has_technical_methods(self):
        from ergodic_insurance.visualization import FigureFactory

        factory = FigureFactory()
        assert callable(getattr(factory, "convergence_diagnostics", None))
        assert callable(getattr(factory, "ergodic_divergence", None))
        assert callable(getattr(factory, "path_dependent_wealth", None))
        assert callable(getattr(factory, "correlation_structure", None))
        assert callable(getattr(factory, "premium_decomposition", None))

    def test_has_interactive_methods(self):
        from ergodic_insurance.visualization import FigureFactory

        factory = FigureFactory()
        assert callable(getattr(factory, "interactive_dashboard", None))
        assert callable(getattr(factory, "risk_dashboard", None))

    def test_loss_distribution_delegates(self):
        """Verify loss_distribution calls through to the real function."""
        from ergodic_insurance.visualization import FigureFactory

        factory = FigureFactory()
        rng = np.random.default_rng(42)
        losses = rng.lognormal(10, 2, 500)
        fig = factory.loss_distribution(losses, title="Test Distribution")
        assert fig is not None
        plt.close(fig)

    def test_ruin_cliff_delegates(self):
        """Verify ruin_cliff calls through to the real function."""
        from ergodic_insurance.visualization import FigureFactory

        factory = FigureFactory()
        fig = factory.ruin_cliff(n_points=5, show_3d_effect=False, show_inset=False)
        assert fig is not None
        plt.close(fig)

    def test_convergence_diagnostics_delegates(self):
        """Verify convergence_diagnostics calls through."""
        from ergodic_insurance.visualization import FigureFactory

        factory = FigureFactory()
        stats = {
            "r_hat_history": [1.5, 1.3, 1.15, 1.05],
            "iterations": [100, 200, 300, 400],
            "ess_history": [500, 800, 1200, 1500],
        }
        fig = factory.convergence_diagnostics(stats)
        assert fig is not None
        plt.close(fig)
