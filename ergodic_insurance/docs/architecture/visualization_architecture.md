# Visualization Module Architecture

This document describes the comprehensive visualization module architecture, supporting both executive and technical visualization needs with Wall Street Journal (WSJ) professional styling.

## Module Overview

```{mermaid}
graph TB
    %% Input Layer
    subgraph Input["Data Sources"]
        SIM_DATA["Simulation Results"]
        RISK_DATA["Risk Metrics"]
        OPT_DATA["Optimization Results"]
        COMP_DATA["Comparison Data"]
    end

    %% Core Components
    subgraph Core["Core Visualization Engine"]
        VIZ_CORE["core.py<br/>WSJ_COLORS, WSJFormatter,<br/>format_currency, format_percentage"]
        FIG_FACTORY["figure_factory.py<br/>FigureFactory Class"]
        STYLE_MGR["style_manager.py<br/>StyleManager, Theme,<br/>ColorPalette, FontConfig"]
        ANNOTATIONS["annotations.py<br/>SmartAnnotationPlacer,<br/>Annotation Functions"]
    end

    %% Visualization Types
    subgraph VizTypes["Visualization Types"]
        EXEC_PLOTS["executive_plots.py<br/>Executive Views"]
        TECH_PLOTS["technical_plots.py<br/>Technical Analysis"]
        BATCH_PLOTS["batch_plots.py<br/>Batch Processing"]
        INTER_PLOTS["interactive_plots.py<br/>Interactive Charts (Plotly)"]
        TOWER_PLOT["improved_tower_plot.py<br/>Insurance Tower"]
    end

    %% Export Layer
    subgraph Export["Export & Integration"]
        EXPORT["export.py<br/>Export Functions"]
        FORMATS["Output Formats<br/>PNG / SVG / HTML / PDF"]
        EMBED["Report Integration<br/>Web / Publication / Presentation"]
    end

    %% Infrastructure & Legacy
    subgraph InfraLegacy["Infrastructure & Legacy"]
        VIZ_INFRA["visualization_infra/<br/>FigureFactory, StyleManager<br/>(Infrastructure Support)"]
        VIZ_LEGACY["visualization_legacy.py<br/>Backward Compatibility Facade"]
        SENS_VIZ["sensitivity_visualization.py<br/>Tornado / Sweep / Matrix Plots"]
    end

    %% Data Flow
    Input --> Core
    Core --> VizTypes
    VizTypes --> Export
    InfraLegacy -.-> Core

    SIM_DATA --> VIZ_CORE
    RISK_DATA --> VIZ_CORE
    OPT_DATA --> VIZ_CORE
    COMP_DATA --> VIZ_CORE

    VIZ_CORE --> FIG_FACTORY
    VIZ_CORE --> ANNOTATIONS
    STYLE_MGR --> FIG_FACTORY
    FIG_FACTORY --> ANNOTATIONS

    FIG_FACTORY --> EXEC_PLOTS
    FIG_FACTORY --> TECH_PLOTS
    FIG_FACTORY --> BATCH_PLOTS
    VIZ_CORE --> INTER_PLOTS
    VIZ_CORE --> TOWER_PLOT

    EXEC_PLOTS --> EXPORT
    TECH_PLOTS --> EXPORT
    BATCH_PLOTS --> EXPORT
    INTER_PLOTS --> EXPORT
    TOWER_PLOT --> EXPORT

    EXPORT --> FORMATS
    EXPORT --> EMBED

    VIZ_LEGACY -.->|delegates to| VizTypes
    SENS_VIZ -.-> VIZ_CORE

    %% Styling
    classDef input fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef core fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef viz fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef export fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef legacy fill:#ffebee,stroke:#c62828,stroke-width:2px

    class SIM_DATA,RISK_DATA,OPT_DATA,COMP_DATA input
    class VIZ_CORE,FIG_FACTORY,STYLE_MGR,ANNOTATIONS core
    class EXEC_PLOTS,TECH_PLOTS,BATCH_PLOTS,INTER_PLOTS,TOWER_PLOT viz
    class EXPORT,FORMATS,EMBED export
    class VIZ_INFRA,VIZ_LEGACY,SENS_VIZ legacy
```

## File Layout

### Visualization Submodule (`visualization/`)

| File | Purpose |
|------|---------|
| `core.py` | WSJ color palette, `WSJFormatter` class, `format_currency()`, `format_percentage()`, `set_wsj_style()` |
| `figure_factory.py` | `FigureFactory` class for creating standardized, themed plots |
| `style_manager.py` | `StyleManager` class, `Theme` enum, `ColorPalette`, `FontConfig`, `FigureConfig`, `GridConfig` dataclasses |
| `annotations.py` | `SmartAnnotationPlacer` class, `AnnotationBox` dataclass, annotation utility functions |
| `executive_plots.py` | Executive-level plot functions (loss distributions, frontiers, dashboards) |
| `technical_plots.py` | Technical analysis plot functions (convergence, Pareto, correlation, MC paths) |
| `interactive_plots.py` | Plotly-based interactive dashboard functions |
| `batch_plots.py` | Batch scenario comparison and sensitivity visualization functions |
| `improved_tower_plot.py` | Insurance tower structure visualization |
| `export.py` | Multi-format export utility functions |

### Visualization Infrastructure (`visualization_infra/`)

| File | Purpose |
|------|---------|
| `figure_factory.py` | Infrastructure-level `FigureFactory` (mirrors `visualization/figure_factory.py`) |
| `style_manager.py` | Infrastructure-level `StyleManager` and `Theme` (mirrors `visualization/style_manager.py`) |

### Legacy and Standalone Modules

| File | Purpose |
|------|---------|
| `visualization_legacy.py` | Backward-compatible facade, delegates to the new modular `visualization/` package |
| `sensitivity_visualization.py` | Standalone sensitivity analysis plots (tornado diagrams, sweep charts, matrices) |

## Class Architecture

```{mermaid}
classDiagram
    %% Core Classes
    class FigureFactory {
        -style_manager: StyleManager
        -auto_apply: bool
        +create_figure(size_type, orientation, dpi_type, title) Tuple~Figure, Axes~
        +create_subplots(rows, cols, size_type, ...) Tuple~Figure, Axes~
        +create_line_plot(x_data, y_data, ...) Tuple~Figure, Axes~
        +create_bar_plot(categories, values, ...) Tuple~Figure, Axes~
        +create_scatter_plot(x_data, y_data, ...) Tuple~Figure, Axes~
        +create_histogram(data, ...) Tuple~Figure, Axes~
        +create_heatmap(data, ...) Tuple~Figure, Axes~
        +create_box_plot(data, ...) Tuple~Figure, Axes~
        +format_axis_currency(ax, axis) None
        +format_axis_percentage(ax, axis) None
        +add_annotations(ax, x, y, text, arrow) None
        +save_figure(fig, filename, output_type) None
    }

    class StyleManager {
        -current_theme: Theme
        -themes: Dict~Theme, Dict~
        +set_theme(theme) None
        +get_theme_config(theme) Dict
        +get_colors() ColorPalette
        +get_fonts() FontConfig
        +get_figure_config() FigureConfig
        +get_grid_config() GridConfig
        +update_colors(updates) None
        +update_fonts(updates) None
        +apply_style() None
        +get_figure_size(size_type, orientation) Tuple
        +get_dpi(output_type) int
        +load_config(config_path) None
        +save_config(config_path) None
        +create_style_sheet() Dict
        +inherit_from(parent_theme, modifications) Theme
    }

    class Theme {
        <<enumeration>>
        DEFAULT
        COLORBLIND
        PRESENTATION
        MINIMAL
        PRINT
    }

    class ColorPalette {
        <<dataclass>>
        +primary: str
        +secondary: str
        +accent: str
        +warning: str
        +success: str
        +neutral: str
        +background: str
        +text: str
        +grid: str
        +series: List~str~
    }

    class FontConfig {
        <<dataclass>>
        +family: str
        +size_base: int
        +size_title: int
        +size_label: int
        +size_tick: int
        +size_legend: int
        +weight_normal: str
        +weight_bold: str
    }

    class FigureConfig {
        <<dataclass>>
        +size_small: Tuple
        +size_medium: Tuple
        +size_large: Tuple
        +size_blog: Tuple
        +size_technical: Tuple
        +size_presentation: Tuple
        +dpi_screen: int
        +dpi_web: int
        +dpi_print: int
    }

    class GridConfig {
        <<dataclass>>
        +show_grid: bool
        +grid_alpha: float
        +grid_linewidth: float
        +spine_top: bool
        +spine_right: bool
        +spine_bottom: bool
        +spine_left: bool
        +spine_linewidth: float
        +tick_major_width: float
        +tick_minor_width: float
    }

    class WSJFormatter {
        +currency_formatter(x, pos)$ str
        +currency(x, decimals)$ str
        +percentage_formatter(x, pos)$ str
        +percentage(x, decimals)$ str
        +number(x, decimals)$ str
        +millions_formatter(x, pos)$ str
    }

    class SmartAnnotationPlacer {
        -ax: Axes
        -placed_annotations: List~AnnotationBox~
        -candidate_positions: List~Tuple~
        -used_colors: Set~str~
        +find_best_position(target_point, text, priority) Tuple
        +add_smart_callout(text, target_point, ...) None
        +add_smart_annotations(annotations, fontsize) None
    }

    class AnnotationBox {
        <<dataclass>>
        +text: str
        +position: Tuple
        +width: float
        +height: float
        +priority: int
        +get_bounds() Tuple
        +overlaps(other, margin) bool
    }

    %% Module-level function groups (documented as interfaces)
    class ExecutivePlots {
        <<module: executive_plots.py>>
        +plot_loss_distribution(losses, ...) Figure
        +plot_return_period_curve(losses, ...) Figure
        +plot_insurance_layers(layers, ...) Figure
        +plot_roe_ruin_frontier(results, ...) Figure
        +plot_ruin_cliff(results, ...) Figure
        +plot_simulation_architecture(data, ...) Figure
        +plot_sample_paths(results, ...) Figure
        +plot_optimal_coverage_heatmap(data, ...) Figure
        +plot_sensitivity_tornado(data, ...) Figure
        +plot_robustness_heatmap(data, ...) Figure
        +plot_premium_multiplier(data, ...) Figure
        +plot_breakeven_timeline(data, ...) Figure
    }

    class TechnicalPlots {
        <<module: technical_plots.py>>
        +plot_convergence_diagnostics(stats, ...) Figure
        +plot_enhanced_convergence_diagnostics(stats, ...) Figure
        +plot_ergodic_divergence(data, ...) Figure
        +plot_trace_plots(data, ...) Figure
        +plot_loss_distribution_validation(data, ...) Figure
        +plot_monte_carlo_convergence(data, ...) Figure
        +plot_pareto_frontier_2d(results, ...) Figure
        +plot_pareto_frontier_3d(results, ...) Figure
        +create_interactive_pareto_frontier(results, ...) Figure
        +plot_path_dependent_wealth(data, ...) Figure
        +plot_correlation_structure(data, ...) Figure
        +plot_premium_decomposition(data, ...) Figure
        +plot_capital_efficiency_frontier_3d(data, ...) Figure
    }

    class InteractivePlots {
        <<module: interactive_plots.py>>
        +create_interactive_dashboard(results, ...) go.Figure
        +create_time_series_dashboard(data, ...) go.Figure
        +create_correlation_heatmap(data, ...) go.Figure
        +create_risk_dashboard(risk_metrics, ...) go.Figure
    }

    class BatchPlotter {
        <<module: batch_plots.py>>
        +plot_scenario_comparison(results, ...) Figure
        +plot_sensitivity_heatmap(results, ...) Figure
        +plot_parameter_sweep_3d(data, ...) Figure
        +plot_scenario_convergence(data, ...) Figure
        +plot_parallel_scenarios(data, ...) Figure
    }

    class ImprovedTowerPlot {
        <<module: improved_tower_plot.py>>
        +plot_insurance_tower(layers, ...) Figure
    }

    class ExportManager {
        <<module: export.py>>
        +save_figure(fig, filename, dpi, formats) List~str~
        +save_for_publication(fig, filename, width, height, dpi) str
        +save_for_presentation(fig, filename, width, height) str
        +save_for_web(fig, filename, optimize) Dict~str, str~
        +batch_export(figures, output_dir, formats, dpi) Dict
    }

    %% Relationships
    FigureFactory --> StyleManager : uses
    FigureFactory ..> WSJFormatter : references
    StyleManager --> Theme : manages
    StyleManager --> ColorPalette : configures
    StyleManager --> FontConfig : configures
    StyleManager --> FigureConfig : configures
    StyleManager --> GridConfig : configures

    SmartAnnotationPlacer --> AnnotationBox : places

    ExecutivePlots ..> WSJFormatter : uses
    ExecutivePlots ..> FigureFactory : optional
    TechnicalPlots ..> WSJFormatter : uses
    InteractivePlots ..> WSJFormatter : uses
    BatchPlotter ..> WSJFormatter : uses

    ExecutivePlots ..> ExportManager : exports
    TechnicalPlots ..> ExportManager : exports
    InteractivePlots ..> ExportManager : exports
    BatchPlotter ..> ExportManager : exports
    ImprovedTowerPlot ..> ExportManager : exports
```

## Visualization Pipeline

```{mermaid}
sequenceDiagram
    participant User
    participant Data
    participant Core as core.py (WSJ Style)
    participant Style as StyleManager
    participant Factory as FigureFactory
    participant Plot as Plot Module
    participant Annot as SmartAnnotationPlacer
    participant Export as export.py

    User->>Data: Request visualization
    Data->>Core: set_wsj_style()

    alt Using FigureFactory (recommended)
        User->>Factory: Create factory with Theme
        Factory->>Style: Initialize StyleManager
        Style->>Style: Load theme (ColorPalette, FontConfig, GridConfig)
        Style-->>Factory: Configured manager
        Factory->>Factory: apply_style() via auto_apply

        User->>Factory: create_figure() or create_line_plot() etc.
        Factory->>Style: get_figure_size(), get_dpi()
        Factory->>Style: get_colors(), get_grid_config()
        Factory-->>User: Styled (Figure, Axes)
    else Using Plot Module Functions
        User->>Plot: Call plot function (e.g. plot_loss_distribution)
        Plot->>Core: set_wsj_style()
        Plot->>Plot: Create matplotlib/plotly figure
        Plot-->>User: Completed Figure
    end

    opt Smart Annotations
        User->>Annot: Create SmartAnnotationPlacer(ax)
        Annot->>Annot: Generate candidate positions
        User->>Annot: add_smart_callout() / add_smart_annotations()
        Annot->>Annot: find_best_position() with overlap detection
        Annot-->>User: Annotated figure
    end

    opt Export Required
        User->>Export: save_figure() / save_for_publication() / save_for_web()
        Export->>Export: Determine format (PNG/SVG/HTML/PDF)
        Export-->>User: Exported file(s)
    end
```

## Visualization Types

### Executive Visualizations (`executive_plots.py`)

Functions designed for C-suite reporting and board presentations, using matplotlib with WSJ styling:

- **`plot_loss_distribution()`** -- Two-panel loss distribution histogram with Q-Q plot and VaR/TVaR lines
- **`plot_return_period_curve()`** -- Return period exceedance curves
- **`plot_insurance_layers()`** -- Insurance program layer structure diagram
- **`plot_roe_ruin_frontier()`** -- ROE vs ruin probability efficient frontier
- **`plot_ruin_cliff()`** -- Ruin probability cliff analysis
- **`plot_simulation_architecture()`** -- Simulation architecture overview
- **`plot_sample_paths()`** -- Monte Carlo sample path visualization
- **`plot_optimal_coverage_heatmap()`** -- Optimal coverage parameter heatmap
- **`plot_sensitivity_tornado()`** -- Sensitivity tornado diagram
- **`plot_robustness_heatmap()`** -- Robustness analysis heatmap
- **`plot_premium_multiplier()`** -- Premium multiplier analysis
- **`plot_breakeven_timeline()`** -- Insurance breakeven timeline

### Technical Visualizations (`technical_plots.py`)

Detailed analytical views for actuaries and risk engineers, using matplotlib and plotly:

- **`plot_convergence_diagnostics()`** -- R-hat, ESS, autocorrelation, and MC standard error panels
- **`plot_enhanced_convergence_diagnostics()`** -- Extended convergence analysis
- **`plot_ergodic_divergence()`** -- Ensemble vs time-average divergence visualization
- **`plot_trace_plots()`** -- MCMC-style trace plots
- **`plot_loss_distribution_validation()`** -- Distribution fit validation
- **`plot_monte_carlo_convergence()`** -- MC convergence analysis
- **`plot_pareto_frontier_2d()`** / **`plot_pareto_frontier_3d()`** -- Multi-objective Pareto frontiers
- **`create_interactive_pareto_frontier()`** -- Interactive Plotly Pareto frontier
- **`plot_path_dependent_wealth()`** -- Path-dependent wealth evolution
- **`plot_correlation_structure()`** -- Correlation matrix analysis
- **`plot_premium_decomposition()`** -- Stacked premium component breakdown
- **`plot_capital_efficiency_frontier_3d()`** -- 3D capital efficiency surface

### Interactive Visualizations (`interactive_plots.py`)

Plotly-based interactive dashboards for exploratory analysis:

- **`create_interactive_dashboard()`** -- Multi-panel MC simulation dashboard with growth rates, loss exceedance, convergence, and risk metrics
- **`create_time_series_dashboard()`** -- Time series with range slider, moving average, and forecast bands
- **`create_correlation_heatmap()`** -- Interactive correlation heatmap with hover values
- **`create_risk_dashboard()`** -- 6-panel risk analytics dashboard (VaR distribution, expected shortfall, risk contribution pie, stress tests, VaR breaches, risk trends)

### Batch Visualizations (`batch_plots.py`)

Scenario comparison and parameter sensitivity functions:

- **`plot_scenario_comparison()`** -- Multi-metric scenario comparison bar charts
- **`plot_sensitivity_heatmap()`** -- Two-parameter sensitivity heatmap
- **`plot_parameter_sweep_3d()`** -- 3D surface plot for parameter sweeps
- **`plot_scenario_convergence()`** -- Cross-scenario convergence comparison
- **`plot_parallel_scenarios()`** -- Parallel coordinate scenario visualization

### Insurance Tower (`improved_tower_plot.py`)

- **`plot_insurance_tower()`** -- Stacked bar tower diagram with smart annotations, log-scale support, layer details (attachment, limit, premium, expected loss, rate on line), and optional summary statistics

## Style Management

```{mermaid}
graph LR
    %% Theme Components
    subgraph Themes["Theme Enum"]
        DEFAULT["DEFAULT<br/>(Corporate WSJ)"]
        COLORBLIND["COLORBLIND<br/>(Accessible)"]
        PRES["PRESENTATION<br/>(Large Fonts)"]
        MINIMAL["MINIMAL<br/>(Greyscale)"]
        PRINT["PRINT<br/>(High DPI)"]
    end

    %% Configuration Dataclasses
    subgraph Config["Configuration Dataclasses"]
        COLORS["ColorPalette<br/>primary, secondary, accent,<br/>warning, success, neutral,<br/>background, text, grid, series"]
        FONTS["FontConfig<br/>family, size_base, size_title,<br/>size_label, size_tick, size_legend"]
        FIGCFG["FigureConfig<br/>size_small/medium/large/blog/<br/>technical/presentation,<br/>dpi_screen/web/print"]
        GRIDCFG["GridConfig<br/>show_grid, grid_alpha, spine_*,<br/>tick_major/minor_width"]
    end

    %% Application
    subgraph Application["Applied To"]
        RCPARAMS["matplotlib rcParams"]
        FIGURES["Figure Sizing"]
        AXES["Axis Formatting"]
        EXPORTS["Export DPI"]
    end

    Themes --> Config
    Config --> Application

    DEFAULT --> COLORS
    COLORBLIND --> COLORS
    PRES --> FONTS
    MINIMAL --> GRIDCFG
    PRINT --> FIGCFG

    COLORS --> RCPARAMS
    FONTS --> RCPARAMS
    GRIDCFG --> AXES
    FIGCFG --> FIGURES
    FIGCFG --> EXPORTS

    %% Styling
    classDef theme fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef config fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef apply fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px

    class DEFAULT,COLORBLIND,PRES,MINIMAL,PRINT theme
    class COLORS,FONTS,FIGCFG,GRIDCFG config
    class RCPARAMS,FIGURES,AXES,EXPORTS apply
```

### Theme Details

| Theme | Primary Use | Key Characteristics |
|-------|------------|---------------------|
| **DEFAULT** | Corporate reporting | WSJ blue palette, Arial fonts, clean grid |
| **COLORBLIND** | Accessible reports | Distinct hue palette safe for color vision deficiencies |
| **PRESENTATION** | Slide decks | Larger fonts (14-18pt), bolder colors, bigger figures |
| **MINIMAL** | Technical papers | Greyscale palette, no grid, Helvetica fonts |
| **PRINT** | Publication output | High DPI (300-600), high contrast, thicker lines |

### YAML Configuration Support

Themes can be loaded from or saved to YAML files via `StyleManager.load_config()` and `StyleManager.save_config()`:

```yaml
# visualization/config.yaml
themes:
  default:
    colors:
      primary: "#0080C7"
      secondary: "#003F5C"
      accent: "#FF9800"
      warning: "#D32F2F"
      success: "#4CAF50"
    fonts:
      family: "Arial"
      size_base: 11
      size_title: 14
    figure:
      dpi_screen: 100
      dpi_web: 150
      dpi_print: 300
    grid:
      show_grid: true
      grid_alpha: 0.3
      spine_top: false
      spine_right: false

  presentation:
    fonts:
      size_base: 14
      size_title: 18
      size_label: 16
    figure:
      size_medium: [10, 7.5]
      size_large: [14, 10]

  print:
    figure:
      dpi_screen: 300
      dpi_web: 300
      dpi_print: 600
    grid:
      grid_linewidth: 0.3
      spine_linewidth: 1.0
```

## Smart Annotations

The annotation system in `annotations.py` provides both simple utility functions and an intelligent placement engine.

### Utility Functions

| Function | Purpose |
|----------|---------|
| `add_value_labels()` | Add formatted values on top of bar chart bars |
| `add_trend_annotation()` | Add trend arrow with percentage change |
| `add_callout()` | Add callout with arrow pointing to a data point |
| `add_benchmark_line()` | Add horizontal reference line with label |
| `add_shaded_region()` | Add shaded vertical band for highlighting periods |
| `add_data_source()` | Add data source attribution at figure bottom |
| `add_footnote()` | Add explanatory footnote text |
| `auto_annotate_peaks_valleys()` | Automatically detect and annotate peaks/valleys using `scipy.signal.find_peaks` |
| `create_leader_line()` | Draw leader lines (straight, curved, or elbow style) |

### SmartAnnotationPlacer

The `SmartAnnotationPlacer` class provides overlap-aware annotation placement:

```python
from ergodic_insurance.visualization.annotations import SmartAnnotationPlacer

placer = SmartAnnotationPlacer(ax)

# Add a single callout with smart placement
placer.add_smart_callout(
    text="Peak value",
    target_point=(x, y),
    priority=80,
    preferred_quadrant="NE"
)

# Add multiple annotations with automatic overlap resolution
annotations = [
    {"text": "Peak: $1.2M", "point": (5, 1.2e6), "priority": 90, "color": "green"},
    {"text": "Valley: $0.3M", "point": (12, 3e5), "priority": 70, "color": "red"},
]
placer.add_smart_annotations(annotations, fontsize=9)
```

Key features:
- Generates candidate positions in a grid with safe margins
- Scores positions based on overlap penalty, distance, edge proximity, and readability
- Supports preferred quadrant hints (`NE`, `NW`, `SE`, `SW`)
- Tracks used colors to avoid conflicts with plot line colors
- Priority-based layering (higher priority annotations are placed first and rendered on top)

## Export Utilities

The `export.py` module provides format-specific export functions for both matplotlib and Plotly figures:

| Function | Formats | Purpose |
|----------|---------|---------|
| `save_figure()` | PNG, JPG, SVG, PDF, EPS, HTML | General-purpose multi-format export |
| `save_for_publication()` | PDF + PNG | 600 DPI, exact dimensions, metadata embedding |
| `save_for_presentation()` | PNG (or HTML fallback) | 1920x1080 with transparent background |
| `save_for_web()` | PNG (thumb + full) + SVG or HTML | Multi-resolution web-optimized output |
| `batch_export()` | Configurable | Export a dictionary of named figures to a directory |

## Performance Optimization

### Caching Strategy
```python
@lru_cache(maxsize=128)
def create_figure(data_hash, plot_type, style):
    """Cache frequently used figures"""
    return FigureFactory.create(data_hash, plot_type, style)
```

### Batch Processing
```python
def batch_generate(data_list, plot_configs):
    """Generate multiple plots efficiently"""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for data, config in zip(data_list, plot_configs):
            future = executor.submit(create_figure, data, config)
            futures.append(future)

        results = [f.result() for f in futures]
    return results
```

## Integration Examples

### Using FigureFactory with StyleManager
```python
from ergodic_insurance.visualization import FigureFactory, StyleManager, Theme

# Create factory with presentation theme
factory = FigureFactory(theme=Theme.PRESENTATION)

# Create a styled line plot
fig, ax = factory.create_line_plot(
    x_data=years,
    y_data={"Insured": growth_insured, "Uninsured": growth_uninsured},
    title="Growth Rate Comparison",
    x_label="Year",
    y_label="Cumulative Growth",
)
factory.format_axis_percentage(ax, axis="y")
factory.save_figure(fig, "growth_comparison.png", output_type="web")
```

### Using Executive Plot Functions
```python
from ergodic_insurance.visualization import (
    plot_loss_distribution,
    plot_roe_ruin_frontier,
    plot_ruin_cliff,
)

# Create executive-level visualizations
fig1 = plot_loss_distribution(losses, title="Annual Loss Distribution", show_metrics=True)
fig2 = plot_roe_ruin_frontier(optimization_results, title="Risk-Return Frontier")
fig3 = plot_ruin_cliff(simulation_results, title="Ruin Probability Analysis")
```

### Using Interactive Dashboards
```python
from ergodic_insurance.visualization import (
    create_interactive_dashboard,
    create_risk_dashboard,
)

# Create Plotly interactive dashboard
fig = create_interactive_dashboard(simulation_results, title="MC Simulation Dashboard")
fig.show()  # Opens in browser

# Create risk analytics dashboard
risk_fig = create_risk_dashboard(risk_metrics, title="Risk Analytics")
```

### Using Smart Annotations
```python
from ergodic_insurance.visualization.annotations import (
    auto_annotate_peaks_valleys,
    add_benchmark_line,
    add_data_source,
)

fig, ax = plt.subplots()
ax.plot(x_data, y_data)

# Auto-detect and annotate peaks and valleys
placer = auto_annotate_peaks_valleys(ax, x_data, y_data, n_peaks=3, n_valleys=2)

# Add reference line and attribution
add_benchmark_line(ax, industry_average, "Industry Average", color="gray")
add_data_source(fig, "Source: Company Financial Reports, 2024")
```

### Multi-Format Export
```python
from ergodic_insurance.visualization import save_figure, save_for_web, batch_export

# Save in multiple formats
save_figure(fig, "analysis_chart", formats=["png", "pdf", "svg"])

# Web-optimized export (thumbnail + full + SVG/HTML)
web_files = save_for_web(fig, "web_chart", optimize=True)

# Batch export all figures
figures = {"loss_dist": fig1, "frontier": fig2, "ruin": fig3}
batch_export(figures, "output/reports/", formats=["png", "pdf"])
```

## Infrastructure and Legacy

### `visualization_infra/` Directory

The `visualization_infra/` directory provides infrastructure-level copies of `FigureFactory` and `StyleManager` with identical APIs. These are used as standalone infrastructure support modules that can be imported independently of the main visualization package.

### `visualization_legacy.py` Facade

The legacy module acts as a backward-compatible facade. It re-exports functions from the new modular `visualization/` package so that existing code using the old import path continues to work:

```python
# Old import path (still works via facade)
from ergodic_insurance.visualization_legacy import plot_loss_distribution

# New recommended import path
from ergodic_insurance.visualization import plot_loss_distribution
```

### `sensitivity_visualization.py`

A standalone module for sensitivity analysis plots, providing:

- `plot_tornado_diagram()` -- Tornado sensitivity diagram
- `plot_two_way_sensitivity()` -- Two-way sensitivity heatmap
- `plot_parameter_sweep()` -- Parameter sweep line plots
- `create_sensitivity_report()` -- Multi-panel sensitivity report
- `plot_sensitivity_matrix()` -- Sensitivity correlation matrix

## Future Enhancements

1. **AI-Powered Insights**
   - Automatic insight detection
   - Natural language descriptions
   - Anomaly highlighting

2. **Advanced Interactivity**
   - Real-time data updates
   - Collaborative annotations
   - VR/AR visualizations

3. **Performance Improvements**
   - GPU acceleration
   - Streaming visualizations
   - Progressive rendering

4. **Enhanced Export Options**
   - Video generation
   - Interactive PDFs
   - Web dashboards
