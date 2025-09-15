# Reporting Module Architecture

This document describes the architecture of the comprehensive reporting module, which provides both executive and technical reporting capabilities.

## Module Overview

```{mermaid}
graph TB
    %% Entry Points
    subgraph Entry["Entry Points"]
        USER["User Request"]
        CONFIG["Report Configuration"]
        DATA["Simulation Results"]
    end

    %% Core Components
    subgraph Core["Core Reporting Engine"]
        BUILDER["ReportBuilder<br/>Orchestrates report generation"]
        VALIDATOR["Validator<br/>Validates inputs and outputs"]
        CACHE["CacheManager<br/>Manages report caching"]
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
        FORMATTERS["Formatters<br/>Format data for output"]
    end

    %% Output Formats
    subgraph Output["Output Formats"]
        EXCEL["Excel Reports"]
        PDF["PDF Documents"]
        HTML["HTML Reports"]
        JSON["JSON Data"]
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
    BUILDER --> SCENARIO_COMP

    EXEC_REPORT --> INSIGHT_EXT
    TECH_REPORT --> TABLE_GEN
    SCENARIO_COMP --> TABLE_GEN

    INSIGHT_EXT --> FORMATTERS
    TABLE_GEN --> FORMATTERS

    FORMATTERS --> EXCEL
    FORMATTERS --> PDF
    FORMATTERS --> HTML
    FORMATTERS --> JSON

    %% Styling
    classDef entry fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef core fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef report fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef support fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef output fill:#ffebee,stroke:#c62828,stroke-width:2px

    class USER,CONFIG,DATA entry
    class BUILDER,VALIDATOR,CACHE core
    class EXEC_REPORT,TECH_REPORT,SCENARIO_COMP report
    class TABLE_GEN,INSIGHT_EXT,FORMATTERS support
    class EXCEL,PDF,HTML,JSON output
```

## Class Structure

```{mermaid}
classDiagram
    %% Base Classes
    class ReportConfig {
        +report_type: str
        +output_formats: List[str]
        +include_sections: List[str]
        +style_settings: dict
        +validate() bool
    }

    class ReportBuilder {
        -config: ReportConfig
        -cache_manager: CacheManager
        -validator: Validator
        +build_report(data, config) Report
        +export_report(report, format) str
        -validate_inputs(data) bool
        -apply_formatting(report) Report
    }

    class CacheManager {
        -cache_dir: Path
        -max_age: int
        -cache_size_limit: int
        +get_cached_report(key) Optional[Report]
        +cache_report(key, report) bool
        +clear_cache() bool
        +get_cache_stats() dict
    }

    class Validator {
        +validate_data(data) ValidationResult
        +validate_config(config) ValidationResult
        +validate_report(report) ValidationResult
        -check_completeness(data) bool
        -check_consistency(data) bool
    }

    %% Report Types
    class ExecutiveReport {
        -insight_extractor: InsightExtractor
        +generate_summary() dict
        +create_key_metrics() dict
        +build_recommendations() List[str]
        +format_for_executives() dict
    }

    class TechnicalReport {
        -table_generator: TableGenerator
        +generate_detailed_analysis() dict
        +create_statistical_tables() List[Table]
        +build_convergence_analysis() dict
        +include_technical_appendix() dict
    }

    class ScenarioComparator {
        -scenarios: List[Scenario]
        +compare_scenarios() ComparisonResult
        +generate_comparison_tables() List[Table]
        +identify_best_scenario() Scenario
        +create_sensitivity_matrix() ndarray
    }

    %% Supporting Classes
    class TableGenerator {
        -formatters: Formatters
        +create_table(data, columns) Table
        +apply_styling(table) Table
        +export_to_format(table, format) str
        +create_pivot_table(data) Table
    }

    class InsightExtractor {
        +extract_key_findings(data) List[Finding]
        +identify_trends(data) List[Trend]
        +generate_recommendations(findings) List[str]
        +calculate_business_impact(data) dict
    }

    class Formatters {
        +format_number(value, precision) str
        +format_percentage(value) str
        +format_currency(value) str
        +format_date(date) str
        +apply_conditional_formatting(data) styled_data
    }

    %% Relationships
    ReportBuilder --> ReportConfig : uses
    ReportBuilder --> CacheManager : uses
    ReportBuilder --> Validator : uses

    ExecutiveReport --> InsightExtractor : uses
    TechnicalReport --> TableGenerator : uses
    ScenarioComparator --> TableGenerator : uses

    TableGenerator --> Formatters : uses
    InsightExtractor --> Formatters : uses

    ReportBuilder --> ExecutiveReport : creates
    ReportBuilder --> TechnicalReport : creates
    ReportBuilder --> ScenarioComparator : creates
```

## Data Flow Sequence

```{mermaid}
sequenceDiagram
    participant User
    participant Config
    participant Builder as ReportBuilder
    participant Cache as CacheManager
    participant Valid as Validator
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

## Key Features

### 1. **Caching System**
- Intelligent caching of generated reports
- Configurable cache expiration
- Cache invalidation on data changes

### 2. **Validation Framework**
- Input data validation
- Configuration validation
- Output report validation
- Consistency checks

### 3. **Flexible Report Types**
- Executive summaries with key insights
- Technical deep-dives with detailed analysis
- Scenario comparisons with recommendations

### 4. **Multiple Output Formats**
- Excel with formatted worksheets
- PDF with professional layout
- HTML for web viewing
- JSON for programmatic access

### 5. **Performance Optimization**
- Lazy loading of data
- Incremental report generation
- Parallel processing for large datasets

## Integration Points

### With Simulation Engine
```python
# Example integration
from reporting import ReportBuilder
from simulation import SimulationResults

results = SimulationResults.load("simulation_output.pkl")
config = ReportConfig(report_type="executive", output_formats=["excel", "pdf"])
builder = ReportBuilder(config)
report = builder.build_report(results)
```

### With Visualization Module
```python
# Embedding visualizations in reports
from visualization import FigureFactory
from reporting import TechnicalReport

fig = FigureFactory.create_roe_chart(data)
report = TechnicalReport()
report.add_figure(fig, caption="ROE Evolution")
```

### With Configuration System
```python
# Using configuration profiles for reports
from config_manager import ConfigManager
from reporting import ReportBuilder

config_mgr = ConfigManager()
report_config = config_mgr.load_profile("reporting/technical")
builder = ReportBuilder(report_config)
```

## Design Patterns

### 1. **Builder Pattern**
- ReportBuilder constructs complex reports step by step
- Allows for flexible report composition

### 2. **Strategy Pattern**
- Different report types implement different strategies
- Easy to add new report types

### 3. **Decorator Pattern**
- Formatters decorate raw data with styling
- Chainable formatting operations

### 4. **Cache-Aside Pattern**
- Check cache before generating expensive reports
- Update cache after generation

### 5. **Template Method Pattern**
- Base report class defines template
- Subclasses implement specific steps

## Error Handling

```python
class ReportGenerationError(Exception):
    """Raised when report generation fails"""
    pass

class ValidationError(Exception):
    """Raised when validation fails"""
    pass

class CacheError(Exception):
    """Raised when cache operations fail"""
    pass
```

## Configuration Example

```yaml
# reporting/config.yaml
reporting:
  cache:
    enabled: true
    max_age_hours: 24
    size_limit_mb: 500

  executive_report:
    sections:
      - summary
      - key_metrics
      - recommendations
    style:
      theme: professional
      color_scheme: corporate

  technical_report:
    sections:
      - detailed_analysis
      - statistical_tables
      - convergence_plots
      - appendix
    include_raw_data: false

  output:
    excel:
      template: "templates/report_template.xlsx"
      include_charts: true
    pdf:
      page_size: A4
      orientation: portrait
```

## Future Enhancements

1. **Real-time Report Generation**
   - Stream processing for live data
   - WebSocket support for updates

2. **AI-Powered Insights**
   - Natural language generation for findings
   - Anomaly detection and alerting

3. **Collaborative Features**
   - Report sharing and commenting
   - Version control for reports

4. **Advanced Visualization**
   - Interactive dashboards
   - 3D visualizations for complex data

5. **Automated Scheduling**
   - Scheduled report generation
   - Email distribution system
