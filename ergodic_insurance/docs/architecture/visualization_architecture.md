# Visualization Module Architecture

This document describes the comprehensive visualization module architecture, supporting both executive and technical visualization needs.

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
        VIZ_CORE["core.py<br/>Base Functions"]
        FIG_FACTORY["figure_factory.py<br/>Figure Creation"]
        STYLE_MGR["style_manager.py<br/>Styling Engine"]
        ANNOTATIONS["annotations.py<br/>Smart Annotations"]
    end

    %% Visualization Types
    subgraph VizTypes["Visualization Types"]
        EXEC_PLOTS["executive_plots.py<br/>Executive Views"]
        TECH_PLOTS["technical_plots.py<br/>Technical Analysis"]
        BATCH_PLOTS["batch_plots.py<br/>Batch Processing"]
        INTER_PLOTS["interactive_plots.py<br/>Interactive Charts"]
        TOWER_PLOT["improved_tower_plot.py<br/>Insurance Tower"]
    end

    %% Export Layer
    subgraph Export["Export & Integration"]
        EXPORT["export.py<br/>Export Manager"]
        FORMATS["Output Formats<br/>PNG/SVG/HTML/PDF"]
        EMBED["Report Integration<br/>Excel/PowerPoint"]
    end

    %% Legacy Support
    subgraph Legacy["Legacy Support"]
        VIZ_LEGACY["visualization_legacy.py<br/>Backward Compatibility"]
        SENS_VIZ["sensitivity_visualization.py<br/>Sensitivity Plots"]
    end

    %% Data Flow
    Input --> Core
    Core --> VizTypes
    VizTypes --> Export
    Legacy --> Core

    SIM_DATA --> VIZ_CORE
    RISK_DATA --> VIZ_CORE
    OPT_DATA --> VIZ_CORE
    COMP_DATA --> VIZ_CORE

    VIZ_CORE --> FIG_FACTORY
    FIG_FACTORY --> STYLE_MGR
    STYLE_MGR --> ANNOTATIONS

    FIG_FACTORY --> EXEC_PLOTS
    FIG_FACTORY --> TECH_PLOTS
    FIG_FACTORY --> BATCH_PLOTS
    FIG_FACTORY --> INTER_PLOTS
    FIG_FACTORY --> TOWER_PLOT

    EXEC_PLOTS --> EXPORT
    TECH_PLOTS --> EXPORT
    BATCH_PLOTS --> EXPORT
    INTER_PLOTS --> EXPORT

    EXPORT --> FORMATS
    EXPORT --> EMBED

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
    class VIZ_LEGACY,SENS_VIZ legacy
```

## Class Architecture

```{mermaid}
classDiagram
    %% Core Classes
    class FigureFactory {
        -style_manager: StyleManager
        -annotation_engine: AnnotationEngine
        +create_figure(data, plot_type) Figure
        +apply_theme(fig, theme) Figure
        +add_annotations(fig, annotations) Figure
        +export_figure(fig, format) bytes
    }

    class StyleManager {
        -themes: Dict[str, Theme]
        -color_palettes: Dict[str, Palette]
        -font_settings: FontConfig
        +apply_style(fig, style_name) Figure
        +create_custom_theme(settings) Theme
        +get_color_sequence(n_colors) List[str]
        +format_axis(axis, settings) Axis
    }

    class AnnotationEngine {
        -smart_placement: bool
        -avoid_overlap: bool
        +add_text_annotation(fig, text, position) Annotation
        +add_arrow_annotation(fig, start, end) Annotation
        +highlight_region(fig, x_range, y_range) Annotation
        +auto_annotate_peaks(fig, data) List[Annotation]
    }

    %% Plot Type Classes
    class ExecutivePlots {
        -factory: FigureFactory
        +create_roe_evolution(data) Figure
        +create_insurance_impact(data) Figure
        +create_growth_comparison(data) Figure
        +create_executive_dashboard(data) Dashboard
    }

    class TechnicalPlots {
        -factory: FigureFactory
        +create_convergence_plot(data) Figure
        +create_distribution_plot(data) Figure
        +create_correlation_matrix(data) Figure
        +create_sensitivity_heatmap(data) Figure
    }

    class InteractivePlots {
        -factory: FigureFactory
        +create_interactive_timeline(data) Figure
        +create_3d_surface(data) Figure
        +create_animated_evolution(data) Animation
        +add_hover_data(fig, data) Figure
    }

    class BatchPlotter {
        -factory: FigureFactory
        -parallel: bool
        +generate_all_plots(data, config) List[Figure]
        +save_plots_batch(figures, directory) None
        +create_plot_index(figures) HTML
    }

    class ImprovedTowerPlot {
        -factory: FigureFactory
        +create_tower_diagram(layers) Figure
        +add_retention_lines(fig, retentions) Figure
        +highlight_optimal_structure(fig, optimal) Figure
        +animate_claim_flow(fig, claims) Animation
    }

    %% Export Classes
    class ExportManager {
        -supported_formats: List[str]
        -quality_settings: Dict
        +export_to_png(fig, dpi) bytes
        +export_to_svg(fig) str
        +export_to_html(fig, interactive) str
        +export_to_pdf(figures) bytes
        +embed_in_excel(fig, workbook) None
    }

    %% Relationships
    FigureFactory --> StyleManager : uses
    FigureFactory --> AnnotationEngine : uses

    ExecutivePlots --> FigureFactory : uses
    TechnicalPlots --> FigureFactory : uses
    InteractivePlots --> FigureFactory : uses
    BatchPlotter --> FigureFactory : uses
    ImprovedTowerPlot --> FigureFactory : uses

    ExecutivePlots --> ExportManager : exports
    TechnicalPlots --> ExportManager : exports
    InteractivePlots --> ExportManager : exports
    BatchPlotter --> ExportManager : exports
```

## Visualization Pipeline

```{mermaid}
sequenceDiagram
    participant User
    participant Data
    participant Factory as FigureFactory
    participant Style as StyleManager
    participant Plot as PlotType
    participant Export as ExportManager

    User->>Data: Request visualization
    Data->>Factory: Provide data

    Factory->>Plot: Create base plot
    Plot-->>Factory: Raw figure

    Factory->>Style: Apply styling
    Style->>Style: Load theme
    Style->>Style: Apply colors
    Style->>Style: Format axes
    Style-->>Factory: Styled figure

    Factory->>Factory: Add annotations
    Factory->>Factory: Optimize layout

    Factory-->>User: Display figure

    opt Export Required
        User->>Export: Export request
        Export->>Export: Convert format
        Export-->>User: Exported file
    end
```

## Visualization Types

### Executive Visualizations
```python
# High-level business metrics
- ROE Evolution Chart
- Insurance Impact Analysis
- Growth Rate Comparison
- Risk-Return Frontier
- Executive Dashboard
```

### Technical Visualizations
```python
# Detailed analytical views
- Convergence Diagnostics
- Distribution Analysis
- Correlation Matrices
- Sensitivity Heatmaps
- Monte Carlo Paths
```

### Interactive Visualizations
```python
# User-interactive charts
- Timeline Sliders
- 3D Surface Plots
- Animated Evolutions
- Hover Information
- Drill-down Charts
```

## Style Management

```{mermaid}
graph LR
    %% Theme Components
    subgraph Themes["Theme System"]
        CORP["Corporate Theme"]
        TECH["Technical Theme"]
        PRINT["Print Theme"]
        PRES["Presentation Theme"]
        CUSTOM["Custom Themes"]
    end

    %% Style Elements
    subgraph Elements["Style Elements"]
        COLORS["Color Palettes"]
        FONTS["Font Settings"]
        LAYOUT["Layout Rules"]
        ANNOT["Annotation Styles"]
    end

    %% Application
    subgraph Application["Application"]
        FIGS["Figures"]
        AXES["Axes"]
        LEGENDS["Legends"]
        TITLES["Titles"]
    end

    Themes --> Elements
    Elements --> Application

    CORP --> COLORS
    TECH --> FONTS
    PRINT --> LAYOUT
    PRES --> ANNOT

    COLORS --> FIGS
    FONTS --> TITLES
    LAYOUT --> AXES
    ANNOT --> LEGENDS

    %% Styling
    classDef theme fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef element fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef apply fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px

    class CORP,TECH,PRINT,PRES,CUSTOM theme
    class COLORS,FONTS,LAYOUT,ANNOT element
    class FIGS,AXES,LEGENDS,TITLES apply
```

## Smart Annotations

```python
class SmartAnnotation:
    """Intelligent annotation placement"""

    def auto_annotate(self, figure, data):
        # Detect key points
        peaks = self.find_peaks(data)
        valleys = self.find_valleys(data)
        inflections = self.find_inflections(data)

        # Place annotations avoiding overlap
        for peak in peaks:
            position = self.find_clear_space(figure, peak)
            self.add_annotation(figure, peak, position)

        # Add trend lines
        trend = self.calculate_trend(data)
        self.add_trend_annotation(figure, trend)

        # Highlight significant regions
        significant = self.find_significant_regions(data)
        for region in significant:
            self.highlight_region(figure, region)
```

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

### With Reporting Module
```python
from visualization import ExecutivePlots
from reporting import ReportBuilder

# Create visualizations for report
plots = ExecutivePlots()
roe_chart = plots.create_roe_evolution(data)
impact_chart = plots.create_insurance_impact(data)

# Embed in report
report = ReportBuilder()
report.add_figure(roe_chart, "Figure 1: ROE Evolution")
report.add_figure(impact_chart, "Figure 2: Insurance Impact")
```

### With Excel Export
```python
from visualization import ExportManager
import openpyxl

# Export to Excel
export_mgr = ExportManager()
wb = openpyxl.Workbook()
ws = wb.active

# Embed chart in worksheet
export_mgr.embed_in_excel(figure, ws, position="B2")
wb.save("analysis_report.xlsx")
```

## Configuration

```yaml
# visualization/config.yaml
visualization:
  default_theme: corporate

  themes:
    corporate:
      colors:
        primary: "#1565C0"
        secondary: "#FFA726"
        accent: "#4CAF50"
      fonts:
        title: "Arial Bold"
        label: "Arial"
        size: 12

    technical:
      colors:
        primary: "#333333"
        grid: "#CCCCCC"
      fonts:
        family: "Helvetica"
        size: 10

  export:
    dpi: 300
    format: "png"
    quality: "high"

  performance:
    cache_enabled: true
    parallel_batch: true
    max_workers: 4
```

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
