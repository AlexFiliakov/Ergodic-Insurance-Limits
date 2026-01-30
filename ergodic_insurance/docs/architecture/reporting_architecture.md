# Reporting Module Architecture

This document describes the architecture of the comprehensive reporting module, which provides both executive and technical reporting capabilities, Excel-based financial statement generation, result aggregation, and summary statistics computation.

## Module Overview

```{mermaid}
graph TB
    %% Entry Points
    subgraph Entry["Entry Points"]
        USER["User Request"]
        CONFIG["ReportConfig<br/>(Pydantic model)"]
        DATA["Simulation Results"]
    end

    %% Core Components
    subgraph Core["Core Reporting Engine"]
        BUILDER["ReportBuilder (ABC)<br/>Orchestrates report generation"]
        VALIDATOR["ReportValidator<br/>Validates inputs and outputs"]
        CACHE["CacheManager<br/>HDF5/Parquet caching"]
    end

    %% Report Types
    subgraph Reports["Report Types"]
        EXEC_REPORT["ExecutiveReport<br/>High-level insights"]
        TECH_REPORT["TechnicalReport<br/>Detailed analysis"]
        SCENARIO_COMP["ScenarioComparator<br/>Multi-scenario analysis"]
    end

    %% Supporting Components
    subgraph Support["Supporting Components"]
        TABLE_GEN["TableGenerator<br/>Creates formatted tables"]
        INSIGHT_EXT["InsightExtractor<br/>Extracts key insights"]
        FORMATTERS["Formatters<br/>NumberFormatter, ColorCoder,<br/>TableFormatter"]
    end

    %% External Reporting Components
    subgraph External["Extended Reporting"]
        EXCEL_REP["ExcelReporter<br/>Excel workbook generation"]
        RESULT_AGG["ResultAggregator<br/>Monte Carlo aggregation"]
        FIN_STMT["FinancialStatementGenerator<br/>Balance sheet, income,<br/>cash flow statements"]
        SUMMARY_STATS["SummaryStatistics<br/>Statistical summaries"]
    end

    %% Output Formats
    subgraph Output["Output Formats"]
        EXCEL["Excel Reports"]
        PDF["PDF Documents"]
        HTML["HTML Reports"]
        JSON["JSON Data"]
        MARKDOWN["Markdown"]
        CSV["CSV / Parquet"]
    end

    %% Data Flow
    USER --> CONFIG
    CONFIG --> BUILDER
    DATA --> VALIDATOR
    VALIDATOR --> BUILDER

    BUILDER --> CACHE
    CACHE --> BUILDER

    BUILDER --> EXEC_REPORT
    BUILDER --> TECH_REPORT

    SCENARIO_COMP --> TABLE_GEN

    EXEC_REPORT --> INSIGHT_EXT
    EXEC_REPORT --> TABLE_GEN
    TECH_REPORT --> TABLE_GEN

    INSIGHT_EXT --> FORMATTERS
    TABLE_GEN --> FORMATTERS

    FORMATTERS --> EXCEL
    FORMATTERS --> PDF
    FORMATTERS --> HTML
    FORMATTERS --> JSON
    FORMATTERS --> MARKDOWN

    DATA --> EXCEL_REP
    EXCEL_REP --> FIN_STMT
    EXCEL_REP --> EXCEL

    DATA --> RESULT_AGG
    RESULT_AGG --> CSV
    RESULT_AGG --> JSON

    DATA --> SUMMARY_STATS
    SUMMARY_STATS --> MARKDOWN
    SUMMARY_STATS --> HTML

    %% Styling
    classDef entry fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef core fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef report fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef support fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef external fill:#fce4ec,stroke:#c62828,stroke-width:2px
    classDef output fill:#ffebee,stroke:#c62828,stroke-width:2px

    class USER,CONFIG,DATA entry
    class BUILDER,VALIDATOR,CACHE core
    class EXEC_REPORT,TECH_REPORT,SCENARIO_COMP report
    class TABLE_GEN,INSIGHT_EXT,FORMATTERS support
    class EXCEL_REP,RESULT_AGG,FIN_STMT,SUMMARY_STATS external
    class EXCEL,PDF,HTML,JSON,MARKDOWN,CSV output
```

## Class Structure

```{mermaid}
classDiagram
    %% Configuration Classes (Pydantic Models)
    class ReportConfig {
        +metadata: ReportMetadata
        +style: ReportStyle
        +sections: List~SectionConfig~
        +template: str
        +output_formats: List~str~
        +output_dir: Path
        +cache_dir: Path
        +debug: bool
        +to_yaml(path) str
        +from_yaml(path) ReportConfig
    }

    class ReportMetadata {
        +title: str
        +subtitle: Optional~str~
        +authors: List~str~
        +date: datetime
        +version: str
        +organization: str
        +confidentiality: str
        +keywords: List~str~
        +abstract: Optional~str~
    }

    class ReportStyle {
        +font_family: str
        +font_size: int
        +line_spacing: float
        +margins: Dict~str, float~
        +page_size: str
        +orientation: str
        +header_footer: bool
        +page_numbers: bool
        +color_scheme: str
    }

    class SectionConfig {
        +title: str
        +level: int
        +content: Optional~str~
        +figures: List~FigureConfig~
        +tables: List~TableConfig~
        +subsections: List~SectionConfig~
        +page_break: bool
    }

    class FigureConfig {
        +name: str
        +caption: str
        +source: Union~str, Path~
        +width: float
        +height: float
        +dpi: int
        +position: str
        +cache_key: Optional~str~
    }

    class TableConfig {
        +name: str
        +caption: str
        +data_source: Union~str, Path~
        +format: str
        +columns: Optional~List~str~~
        +index: bool
        +precision: int
        +style: Dict
    }

    %% Core Abstract Builder
    class ReportBuilder {
        <<abstract>>
        #config: ReportConfig
        #cache_manager: CacheManager
        #table_generator: TableGenerator
        #content: List~str~
        #figures: List~Dict~
        #tables: List~Dict~
        +generate()* Path
        +build_section(section) str
        +compile_report() str
        +save(output_format) Path
        -_load_content(content_ref) str
        -_embed_figure(fig_config) str
        -_generate_figure(fig_config) Path
        -_embed_table(table_config) str
        -_load_table_data(data_source) DataFrame
        -_generate_header() str
        -_generate_footer() str
    }

    %% Cache Manager
    class CacheManager {
        -config: CacheConfig
        -stats: CacheStats
        -backend: BaseStorageBackend
        -_cache_index: Dict~str, CacheKey~
        +cache_simulation_paths(params, paths, metadata) str
        +load_simulation_paths(params, memory_map) Optional~ndarray~
        +cache_processed_results(params, results, result_type) str
        +load_processed_results(params, result_type) Optional~DataFrame~
        +cache_figure(params, figure, figure_name, figure_type) str
        +invalidate_cache(params) void
        +clear_cache(confirm) void
        +get_cache_stats() CacheStats
        +warm_cache(scenarios, compute_func, result_type) int
        +validate_cache() Dict
    }

    class CacheConfig {
        +cache_dir: Path
        +max_cache_size_gb: float
        +ttl_hours: Optional~int~
        +compression: Optional~str~
        +compression_level: int
        +enable_memory_mapping: bool
        +backend: StorageBackend
        +backend_config: Dict
    }

    class CacheStats {
        +total_size_bytes: int
        +n_entries: int
        +n_hits: int
        +n_misses: int
        +hit_rate: float
        +avg_load_time_ms: float
        +avg_save_time_ms: float
        +oldest_entry: Optional~datetime~
        +newest_entry: Optional~datetime~
        +update_hit_rate() void
    }

    class CacheKey {
        +hash_key: str
        +params: Dict
        +timestamp: datetime
        +size_bytes: int
        +access_count: int
        +last_accessed: datetime
        +to_dict() Dict
        +from_dict(data) CacheKey
    }

    %% Validator
    class ReportValidator {
        -config: ReportConfig
        -errors: List~str~
        -warnings: List~str~
        -info: List~str~
        +validate() Tuple~bool, List, List~
        -_validate_structure() void
        -_validate_references() void
        -_validate_data_sources() void
        -_validate_formatting() void
        -_validate_completeness() void
        -_validate_quality() void
    }

    %% Concrete Report Types
    class ExecutiveReport {
        -results: Dict~str, Any~
        -style_manager: StyleManager
        -key_metrics: Dict~str, Any~
        +generate() Path
        +generate_roe_frontier(fig_config) Figure
        +generate_performance_table() DataFrame
        +generate_decision_matrix() DataFrame
        +generate_convergence_plot(fig_config) Figure
        +generate_convergence_table() DataFrame
        -_extract_key_metrics() Dict
        -_generate_abstract() str
        -_generate_key_findings() str
        -_generate_recommendations() str
    }

    class TechnicalReport {
        -results: Dict~str, Any~
        -parameters: Dict~str, Any~
        -validation_metrics: Dict~str, Any~
        +generate() Path
        +generate_parameter_sensitivity_plot(fig_config) Figure
        +generate_qq_plot(fig_config) Figure
        +generate_model_parameters_table() DataFrame
        +generate_correlation_matrix_plot(fig_config) Figure
        -_compute_validation_metrics() Dict
        -_generate_ergodic_methodology() str
        -_generate_simulation_methodology() str
        -_generate_validation_summary() str
    }

    %% Scenario Comparator
    class ScenarioComparator {
        -baseline_scenario: Optional~str~
        -comparison_data: Optional~ScenarioComparison~
        +compare_scenarios(results, baseline, metrics, parameters) ScenarioComparison
        +create_comparison_grid(metrics, figsize, show_diff) Figure
        +create_parameter_diff_table(scenario, threshold) DataFrame
        +export_comparison_report(output_path, include_plots) Dict
        -_extract_metrics(results, metrics) Dict
        -_extract_parameters(results, parameters) Dict
        -_compute_diffs(param_data, baseline) Dict
        -_perform_statistical_tests(metric_data) Dict
    }

    class ScenarioComparison {
        +scenarios: List~str~
        +metrics: Dict~str, Dict~str, float~~
        +parameters: Dict~str, Dict~str, Any~~
        +statistics: Dict~str, Any~
        +diffs: Dict~str, Dict~str, Any~~
        +rankings: Dict~str, List~Tuple~~
        +get_metric_df(metric) DataFrame
        +get_top_performers(metric, n, ascending) List
    }

    %% Supporting Classes
    class TableGenerator {
        -default_format: str
        -precision: int
        -max_width: int
        -number_formatter: NumberFormatter
        -color_coder: ColorCoder
        -table_formatter: TableFormatter
        +generate(data, caption, columns, index, output_format, precision, style) str
        +generate_summary_statistics(df, metrics, output_format) str
        +generate_comparison_table(data, caption, output_format) str
        +generate_decision_matrix(alternatives, criteria, scores, weights) str
        +generate_optimal_limits_by_size(company_sizes, optimal_limits) str
        +generate_quick_reference_matrix(characteristics, recommendations) str
        +generate_parameter_grid(parameters, scenarios) str
        +generate_loss_distribution_params(loss_types, distribution_params) str
        +generate_insurance_pricing_grid(layers, pricing_params) str
        +generate_statistical_validation(metrics) str
        +generate_comprehensive_results(results, ranking_metric) str
        +generate_walk_forward_validation(validation_results) str
        +export_to_file(df, file_path, output_format) void
    }

    class InsightExtractor {
        -insights: List~Insight~
        -data: Optional~Any~
        +extract_insights(data, focus_metrics, threshold_importance) List~Insight~
        +generate_executive_summary(max_points, focus_positive) str
        +generate_technical_notes() List~str~
        +export_insights(output_path, output_format) str
        -_extract_performance_insights(data, focus_metrics) void
        -_extract_trend_insights(data, focus_metrics) void
        -_extract_outlier_insights(data, focus_metrics) void
        -_extract_threshold_insights(data, focus_metrics) void
        -_extract_correlation_insights(data, focus_metrics) void
    }

    class Insight {
        +category: str
        +importance: float
        +title: str
        +description: str
        +data: Dict
        +metrics: List~str~
        +confidence: float
        +to_bullet_point() str
        +to_executive_summary() str
    }

    class NumberFormatter {
        +currency_symbol: str
        +decimal_places: int
        +thousands_separator: str
        +decimal_separator: str
        +format_currency(value, decimals, abbreviate) str
        +format_percentage(value, decimals, multiply_by_100) str
        +format_number(value, decimals, scientific, abbreviate) str
        +format_ratio(value, decimals) str
    }

    class ColorCoder {
        +output_format: str
        +color_scheme: Dict
        +traffic_light(value, thresholds, text) str
        +heatmap(value, min_val, max_val, text) str
        +threshold_color(value, threshold, above_color, below_color, text) str
    }

    class TableFormatter {
        +number_formatter: NumberFormatter
        +color_coder: ColorCoder
        +output_format: str
        +format_dataframe(df, column_formats, row_colors, alternating_rows) DataFrame
        +add_totals_row(df, columns, label, operation) DataFrame
        +add_footnotes(table_str, footnotes, output_format) str
    }

    %% Extended Reporting Classes
    class ExcelReporter {
        -config: ExcelReportConfig
        -workbook: Optional~Any~
        -formats: Dict~str, Any~
        -engine: str
        +generate_trajectory_report(manufacturer, output_file, title) Path
        +generate_monte_carlo_report(results, output_file, title) Path
        -_select_engine() void
        -_generate_with_xlsxwriter(generator, output_path, title) void
        -_generate_with_openpyxl(generator, output_path, title) void
        -_generate_with_pandas(generator, output_path) void
    }

    class ResultAggregator {
        -config: AggregationConfig
        -custom_functions: Dict~str, Callable~
        -_cache: Dict~str, Any~
        +aggregate(data) Dict~str, Any~
        -_calculate_moments(data) Dict
        -_fit_distributions(data) Dict
    }

    class FinancialStatementGenerator {
        -manufacturer_data: Dict
        -config: FinancialStatementConfig
        -metrics_history: List~Dict~
        -years_available: int
        -ledger: Optional~Ledger~
        +generate_balance_sheet(year) DataFrame
        +generate_income_statement(year) DataFrame
        +generate_cash_flow_statement(year) DataFrame
        +generate_reconciliation_report(year) DataFrame
    }

    class SummaryStatistics {
        -confidence_level: float
        -bootstrap_iterations: int
        -_rng: Generator
        +calculate_summary(data, weights) StatisticalSummary
        -_calculate_basic_stats(data, weights) Dict
        -_fit_distributions(data) Dict
        -_calculate_confidence_intervals(data) Dict
        -_perform_hypothesis_tests(data) Dict
        -_calculate_extreme_values(data) Dict
    }

    %% Relationships
    ReportConfig --> ReportMetadata : contains
    ReportConfig --> ReportStyle : contains
    ReportConfig --> SectionConfig : contains
    SectionConfig --> FigureConfig : contains
    SectionConfig --> TableConfig : contains
    SectionConfig --> SectionConfig : subsections

    ReportBuilder --> ReportConfig : uses
    ReportBuilder --> CacheManager : uses
    ReportBuilder --> TableGenerator : uses

    CacheManager --> CacheConfig : uses
    CacheManager --> CacheStats : tracks
    CacheManager --> CacheKey : indexes

    ReportValidator --> ReportConfig : validates

    ExecutiveReport --|> ReportBuilder : extends
    TechnicalReport --|> ReportBuilder : extends

    ScenarioComparator --> ScenarioComparison : creates

    TableGenerator --> NumberFormatter : uses
    TableGenerator --> ColorCoder : uses
    TableGenerator --> TableFormatter : uses

    TableFormatter --> NumberFormatter : uses
    TableFormatter --> ColorCoder : uses

    InsightExtractor --> Insight : creates

    ExcelReporter --> FinancialStatementGenerator : uses
```

## Data Flow Sequence

```{mermaid}
sequenceDiagram
    participant User
    participant Config as ReportConfig
    participant Builder as ReportBuilder
    participant Cache as CacheManager
    participant Valid as ReportValidator
    participant Report as Report Type
    participant Support as Support Components
    participant Output

    User->>Config: Define report requirements
    Config->>Builder: Initialize with config

    Builder->>Cache: Check for cached report
    alt Cached report exists
        Cache-->>Builder: Return cached report
        Builder-->>User: Return report
    else No cached report
        Builder->>Valid: Validate input data
        Valid-->>Builder: Validation result

        alt Data valid
            Builder->>Report: Generate report
            Report->>Support: Process data
            Support-->>Report: Formatted content
            Report-->>Builder: Complete report

            Builder->>Cache: Cache report
            Builder->>Output: Export to format
            Output-->>User: Final report
        else Data invalid
            Builder-->>User: Error message
        end
    end
```

## File Layout

### Reporting Submodule (`reporting/`)

| File | Primary Class(es) | Purpose |
|------|-------------------|---------|
| `config.py` | `ReportConfig`, `ReportMetadata`, `ReportStyle`, `SectionConfig`, `FigureConfig`, `TableConfig` | Pydantic configuration models for report structure |
| `report_builder.py` | `ReportBuilder` (ABC) | Abstract base class for report compilation, section building, figure/table embedding, and multi-format output (Markdown, HTML, PDF) |
| `executive_report.py` | `ExecutiveReport` | Generates concise executive summaries with key metrics, ROE-Ruin frontiers, decision matrices, and recommendations |
| `technical_report.py` | `TechnicalReport` | Generates detailed technical appendices with methodology, convergence diagnostics, statistical validation, and parameter sensitivity |
| `scenario_comparator.py` | `ScenarioComparator`, `ScenarioComparison` | Multi-scenario comparison framework with statistical tests, parameter diff tracking, and visualization grids |
| `table_generator.py` | `TableGenerator`, `create_performance_table`, `create_parameter_table`, `create_sensitivity_table` | Formatted table creation supporting Markdown, HTML, LaTeX, grid, CSV, and Excel formats; includes specialized insurance and executive table methods |
| `insight_extractor.py` | `InsightExtractor`, `Insight` | Automated insight extraction from simulation data: performance, trends, outliers, thresholds, and correlations with natural language generation |
| `formatters.py` | `NumberFormatter`, `ColorCoder`, `TableFormatter`, `format_for_export` | Number, currency, percentage, and ratio formatting; traffic light and heatmap color coding; multi-format export utilities |
| `cache_manager.py` | `CacheManager`, `CacheConfig`, `CacheStats`, `CacheKey`, `StorageBackend`, `LocalStorageBackend` | High-performance caching with HDF5 for simulation paths, Parquet for processed results, hash-based invalidation, LRU eviction, and memory-mapped loading |
| `validator.py` | `ReportValidator`, `validate_results_data`, `validate_parameters` | Comprehensive validation of report structure, references, data sources, formatting, completeness, and quality |

### Top-Level Reporting Modules

| File | Primary Class(es) | Purpose |
|------|-------------------|---------|
| `excel_reporter.py` | `ExcelReporter`, `ExcelReportConfig` | Professional Excel workbook generation with XlsxWriter, openpyxl, or pandas backends; creates financial statement sheets, metrics dashboards, reconciliation reports, and pivot data |
| `result_aggregator.py` | `ResultAggregator`, `TimeSeriesAggregator`, `PercentileTracker`, `HierarchicalAggregator`, `ResultExporter` | Monte Carlo result aggregation with percentiles, moments, distribution fitting, time-series analysis, streaming quantile tracking (t-digest), and multi-format export |
| `financial_statements.py` | `FinancialStatementGenerator`, `CashFlowStatement`, `MonteCarloStatementAggregator` | Balance sheet, income statement, and cash flow generation from simulation data; supports both indirect and direct (ledger-based) methods; Monte Carlo aggregation across trajectories |
| `summary_statistics.py` | `SummaryStatistics`, `TDigest`, `QuantileCalculator`, `DistributionFitter`, `SummaryReportGenerator` | Comprehensive statistical analysis with distribution fitting, bootstrap confidence intervals, hypothesis testing, extreme value statistics, streaming quantile estimation, and formatted report output |

## Key Features

### 1. **Report Building Pipeline**
- Abstract `ReportBuilder` base class with template method pattern
- `ExecutiveReport` and `TechnicalReport` as concrete implementations
- Jinja2 template rendering for content sections
- Automatic figure generation and caching
- Multi-format output: Markdown, HTML (via markdown2), PDF (via WeasyPrint)

### 2. **Caching System**
- HDF5 storage for large simulation path arrays (10,000 x 1,000)
- Parquet storage for processed tabular results
- SHA256 hash-based cache keys from parameter dictionaries
- Configurable TTL expiration and LRU eviction
- Memory-mapped loading for large files
- Cache integrity validation and orphan detection
- Cache warming for common scenarios

### 3. **Validation Framework**
- Report structure validation (metadata, sections, hierarchy)
- Figure and table reference validation (duplicates, undefined, unused)
- Data source existence checks
- Formatting parameter validation (dimensions, DPI, font sizes, margins)
- Completeness checks against report templates
- Quality checks on captions, abstracts, and content balance

### 4. **Table Generation**
- Multiple output formats: Markdown, HTML, LaTeX, grid, CSV, Excel
- Specialized table types:
  - Optimal insurance limits by company size
  - Quick reference decision matrices
  - Parameter grids across scenarios
  - Loss distribution parameters
  - Insurance pricing grids
  - Statistical validation metrics
  - Walk-forward validation results
  - Comprehensive optimization results with ranking
- Integration with `NumberFormatter` and `ColorCoder` for professional formatting

### 5. **Insight Extraction**
- Automated detection of: best/worst performers, trends, outliers, threshold violations, correlations
- Template-based natural language generation
- Importance scoring and confidence levels
- Executive summary generation (simplified language)
- Technical notes generation
- Export to Markdown, JSON, or CSV

### 6. **Excel Reporting**
- Multiple engine support: XlsxWriter (preferred), openpyxl, pandas fallback
- Financial statement sheets: Balance Sheet, Income Statement, Cash Flow
- Reconciliation reports with pass/fail status formatting
- Metrics dashboard with conditional formatting
- Pivot-ready data sheets for custom analysis
- Monte Carlo aggregated reports

### 7. **Result Aggregation**
- Basic statistics: mean, std, min, max, percentiles
- Statistical moments: variance, skewness, kurtosis, coefficient of variation
- Distribution fitting: normal, lognormal, gamma, exponential
- Time-series aggregation: period-wise, cumulative, rolling window, growth rates
- Streaming quantile estimation using t-digest algorithm
- Hierarchical aggregation across multiple levels (scenario/year/simulation)
- Export to CSV, JSON, HDF5

### 8. **Summary Statistics**
- Comprehensive statistical summary (basic stats, distributions, confidence intervals, hypothesis tests, extreme values)
- Bootstrap confidence intervals for mean, median, and standard deviation
- Normality tests (Shapiro-Wilk, Jarque-Bera)
- Distribution fitting with AIC model selection (normal, lognormal, gamma, exponential, Weibull, beta, Pareto)
- T-digest streaming quantile estimation for large datasets
- Formatted report generation (Markdown, HTML, LaTeX)

### 9. **Multiple Output Formats**
- **Markdown**: Primary format for report content
- **HTML**: Web viewing with styled templates
- **PDF**: Professional layout via WeasyPrint
- **Excel**: Financial statements and dashboards via XlsxWriter/openpyxl
- **JSON**: Programmatic access and insight export
- **CSV/Parquet**: Tabular data export

### 10. **Performance Optimization**
- HDF5 with gzip/lzf compression for simulation arrays
- Parquet with Snappy compression for tabular data
- Memory-mapped file loading for large datasets
- Hash-based cache invalidation (no unnecessary recomputation)
- LRU eviction when cache exceeds size limits
- T-digest streaming algorithm for bounded-memory quantile estimation
- Chunked processing for large Monte Carlo datasets

## Integration Points

### With Simulation Engine
```python
from ergodic_insurance.reporting import ExecutiveReport, ReportConfig
from ergodic_insurance.reporting.config import create_executive_config

# Generate executive report from results
config = create_executive_config()
report = ExecutiveReport(
    results={'roe': 0.18, 'ruin_probability': 0.01, 'growth_rate': 0.07},
    config=config,
)
report_path = report.generate()
```

### With Excel Financial Statements
```python
from ergodic_insurance.excel_reporter import ExcelReporter, ExcelReportConfig
from ergodic_insurance.manufacturer import WidgetManufacturer

config = ExcelReportConfig(
    output_path=Path("./reports"),
    include_balance_sheet=True,
    include_income_statement=True,
    include_cash_flow=True,
    include_metrics_dashboard=True,
)
reporter = ExcelReporter(config)
output_file = reporter.generate_trajectory_report(manufacturer, "statements.xlsx")
```

### With Result Aggregation
```python
from ergodic_insurance.result_aggregator import ResultAggregator, AggregationConfig

config = AggregationConfig(
    percentiles=[1, 5, 10, 25, 50, 75, 90, 95, 99],
    calculate_moments=True,
    calculate_distribution_fit=True,
)
aggregator = ResultAggregator(config)
summary = aggregator.aggregate(simulation_results)
```

### With Summary Statistics
```python
from ergodic_insurance.summary_statistics import SummaryStatistics

stats_calc = SummaryStatistics(confidence_level=0.95, bootstrap_iterations=1000)
summary = stats_calc.calculate_summary(data)
# Returns StatisticalSummary with basic_stats, distribution_params,
# confidence_intervals, hypothesis_tests, extreme_values
```

### With Scenario Comparison
```python
from ergodic_insurance.reporting import ScenarioComparator

comparator = ScenarioComparator()
comparison = comparator.compare_scenarios(
    results={'base': base_results, 'optimized': optimized_results},
    baseline='base',
)
# Generate comparison grid visualization
fig = comparator.create_comparison_grid()
# Export full comparison report
outputs = comparator.export_comparison_report("comparison_output")
```

### With Caching
```python
from ergodic_insurance.reporting import CacheManager
from ergodic_insurance.reporting.cache_manager import CacheConfig

cache = CacheManager(CacheConfig(cache_dir=Path("./cache"), max_cache_size_gb=10.0))
# Cache simulation paths (HDF5)
key = cache.cache_simulation_paths(params={'n_sims': 10000}, paths=paths_array)
# Load from cache (memory-mapped)
cached_paths = cache.load_simulation_paths(params={'n_sims': 10000})
```

### With Configuration System
```python
from ergodic_insurance.reporting.config import ReportConfig

# Load from YAML
config = ReportConfig.from_yaml("reporting/technical.yaml")
# Export to YAML
config.to_yaml(Path("reporting/config_export.yaml"))
```

## Design Patterns

### 1. **Builder Pattern**
- `ReportBuilder` constructs complex reports step by step via `build_section()` and `compile_report()`
- Sections, figures, and tables are composed incrementally
- `ExcelReporter` builds workbooks sheet-by-sheet

### 2. **Strategy Pattern**
- Different report types (`ExecutiveReport`, `TechnicalReport`) implement different generation strategies through the abstract `generate()` method
- `TableGenerator` supports multiple output format strategies (Markdown, HTML, LaTeX, etc.)
- `ExcelReporter` selects engine strategy (XlsxWriter, openpyxl, pandas)

### 3. **Decorator Pattern**
- `NumberFormatter` decorates raw numeric values with currency, percentage, or ratio formatting
- `ColorCoder` decorates values with traffic light, heatmap, or threshold-based coloring
- `TableFormatter` composes `NumberFormatter` and `ColorCoder` for full table styling

### 4. **Cache-Aside Pattern**
- `CacheManager` checks cache before expensive computation
- Hash-based keys ensure parameter changes trigger recomputation
- TTL and LRU eviction policies maintain cache health
- Support for multiple storage formats: HDF5 (arrays), Parquet (tables), pickle (figures)

### 5. **Template Method Pattern**
- `ReportBuilder` defines the report generation skeleton (`compile_report()`, `build_section()`, `save()`)
- Subclasses implement specific steps: `generate()`, content generation methods
- `SectionConfig` allows flexible section composition within the template

### 6. **Abstract Factory Pattern**
- `create_executive_config()` and `create_technical_config()` factory functions produce pre-configured `ReportConfig` instances
- `BaseStorageBackend` defines the interface; `LocalStorageBackend` provides local filesystem implementation

## Error Handling

```python
# Validation errors
from ergodic_insurance.reporting.validator import validate_results_data, validate_parameters

is_valid, errors = validate_results_data(results)
# Checks: required keys (roe, ruin_probability, trajectories),
#          data types, value ranges

is_valid, errors = validate_parameters(params)
# Checks: required groups (financial, insurance, simulation),
#          positive values, valid ranges

# Report validation
from ergodic_insurance.reporting.validator import ReportValidator

validator = ReportValidator(config)
is_valid, errors, warnings = validator.validate()
# Checks: structure, references, data sources, formatting,
#          completeness, quality
```

## Configuration Example

```yaml
# Report configuration via Pydantic models
reporting:
  metadata:
    title: "Insurance Optimization Analysis"
    subtitle: "Executive Summary"
    authors: ["Analytics Team"]
    organization: "Ergodic Insurance Analytics"
    confidentiality: "Internal"
    keywords: ["insurance", "optimization", "risk", "ergodic"]

  style:
    font_family: "Arial"
    font_size: 11
    line_spacing: 1.5
    margins:
      top: 1.0
      bottom: 1.0
      left: 1.0
      right: 1.0
    page_size: "Letter"
    orientation: "portrait"

  template: "executive"
  output_formats: ["pdf", "html", "markdown"]
  output_dir: "reports"
  cache_dir: "reports/cache"

# Cache configuration
cache:
  cache_dir: "./cache"
  max_cache_size_gb: 10.0
  ttl_hours: 24
  compression: "gzip"
  compression_level: 4
  enable_memory_mapping: true
  backend: "local"

# Excel report configuration
excel:
  output_path: "./reports"
  include_balance_sheet: true
  include_income_statement: true
  include_cash_flow: true
  include_reconciliation: true
  include_metrics_dashboard: true
  include_pivot_data: true
  engine: "auto"
  currency_format: "$#,##0"
```

## Future Enhancements

1. **Cloud Storage Backends**
   - S3, Azure Blob, and GCS backends for CacheManager (interface defined via `StorageBackend` enum)

2. **Real-time Report Generation**
   - Stream processing for live data
   - WebSocket support for updates

3. **AI-Powered Insights**
   - Natural language generation for findings
   - Anomaly detection and alerting

4. **Collaborative Features**
   - Report sharing and commenting
   - Version control for reports

5. **Advanced Visualization**
   - Interactive dashboards
   - 3D visualizations for complex data

6. **Automated Scheduling**
   - Scheduled report generation
   - Email distribution system
