"""Unified visualization infrastructure for ergodic insurance analysis.

This package provides a comprehensive visualization toolkit with:
- Professional Wall Street Journal styling
- Executive-level and technical visualizations
- Interactive dashboards
- Export utilities for various formats
"""

# Annotation utilities
from .annotations import (
    add_benchmark_line,
    add_callout,
    add_data_source,
    add_footnote,
    add_shaded_region,
    add_trend_annotation,
    add_value_labels,
)

# Batch processing visualizations
from .batch_plots import (
    plot_parallel_scenarios,
    plot_parameter_sweep_3d,
    plot_scenario_comparison,
    plot_scenario_convergence,
    plot_sensitivity_heatmap,
)

# Core utilities and constants
from .core import (
    COLOR_SEQUENCE,
    WSJ_COLORS,
    WSJFormatter,
    format_currency,
    format_percentage,
    set_wsj_style,
)

# Executive visualization functions
from .executive_plots import (
    plot_insurance_layers,
    plot_loss_distribution,
    plot_return_period_curve,
    plot_roe_ruin_frontier,
    plot_ruin_cliff,
)

# Export utilities
from .export import (
    batch_export,
    save_figure,
    save_for_presentation,
    save_for_publication,
    save_for_web,
)

# Figure factory and style management
from .figure_factory import FigureFactory

# Interactive visualization functions
from .interactive_plots import (
    create_correlation_heatmap,
    create_interactive_dashboard,
    create_risk_dashboard,
    create_time_series_dashboard,
)
from .style_manager import StyleManager, Theme

# Technical visualization functions
from .technical_plots import (
    create_interactive_pareto_frontier,
    plot_capital_efficiency_frontier_3d,
    plot_convergence_diagnostics,
    plot_correlation_structure,
    plot_enhanced_convergence_diagnostics,
    plot_ergodic_divergence,
    plot_loss_distribution_validation,
    plot_monte_carlo_convergence,
    plot_pareto_frontier_2d,
    plot_pareto_frontier_3d,
    plot_path_dependent_wealth,
    plot_premium_decomposition,
    plot_trace_plots,
)

__all__ = [
    # Core
    "WSJ_COLORS",
    "COLOR_SEQUENCE",
    "set_wsj_style",
    "format_currency",
    "format_percentage",
    "WSJFormatter",
    # Executive plots
    "plot_loss_distribution",
    "plot_return_period_curve",
    "plot_insurance_layers",
    "plot_roe_ruin_frontier",
    "plot_ruin_cliff",
    # Technical plots
    "plot_convergence_diagnostics",
    "plot_enhanced_convergence_diagnostics",
    "plot_ergodic_divergence",
    "plot_trace_plots",
    "plot_loss_distribution_validation",
    "plot_monte_carlo_convergence",
    "plot_pareto_frontier_2d",
    "plot_pareto_frontier_3d",
    "plot_path_dependent_wealth",
    "create_interactive_pareto_frontier",
    "plot_correlation_structure",
    "plot_premium_decomposition",
    "plot_capital_efficiency_frontier_3d",
    # Interactive plots
    "create_interactive_dashboard",
    "create_time_series_dashboard",
    "create_correlation_heatmap",
    "create_risk_dashboard",
    # Batch plots
    "plot_scenario_comparison",
    "plot_sensitivity_heatmap",
    "plot_parameter_sweep_3d",
    "plot_scenario_convergence",
    "plot_parallel_scenarios",
    # Annotations
    "add_value_labels",
    "add_trend_annotation",
    "add_callout",
    "add_benchmark_line",
    "add_shaded_region",
    "add_data_source",
    "add_footnote",
    # Export
    "save_figure",
    "save_for_publication",
    "save_for_presentation",
    "save_for_web",
    "batch_export",
    # Factory and style
    "FigureFactory",
    "StyleManager",
    "Theme",
]

# Package metadata
__version__ = "2.0.0"
__author__ = "Ergodic Insurance Team"
__description__ = "Professional visualization infrastructure for insurance analytics"
