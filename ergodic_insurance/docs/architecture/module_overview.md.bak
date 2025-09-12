# Module Overview and Dependencies

This diagram shows the detailed module structure and dependencies within the Ergodic Insurance framework.

```mermaid
graph LR
    %% Configuration Layer
    subgraph Config["Configuration Management"]
        CONFIG_BASE["config.py<br/>Base Configuration"]
        CONFIG_V2["config_v2.py<br/>Enhanced Config"]
        CONFIG_MGR["config_manager.py<br/>Config Manager"]
        CONFIG_LOADER["config_loader.py<br/>Config Loader"]
        CONFIG_COMPAT["config_compat.py<br/>Compatibility Layer"]
        CONFIG_MIG["config_migrator.py<br/>Migration Tools"]
    end

    %% Core Business Logic
    subgraph Business["Business Logic"]
        MANUFACTURER["manufacturer.py<br/>Widget Manufacturer"]
        INSURANCE["insurance.py<br/>Insurance Policy"]
        INS_PROGRAM["insurance_program.py<br/>Insurance Program"]
        INS_PRICING["insurance_pricing.py<br/>Pricing Models"]
        CLAIM_GEN["claim_generator.py<br/>Claim Events"]
        CLAIM_DEV["claim_development.py<br/>Claim Development"]
    end

    %% Simulation Engine
    subgraph Simulation["Simulation Core"]
        SIM_CORE["simulation.py<br/>Main Engine"]
        MONTE_CARLO["monte_carlo.py<br/>Monte Carlo"]
        STOCHASTIC["stochastic_processes.py<br/>Stochastic Models"]
        LOSS_DIST["loss_distributions.py<br/>Loss Distributions"]
    end

    %% Analysis Tools
    subgraph Analysis["Analysis & Optimization"]
        ERGODIC_ANALYZER["ergodic_analyzer.py<br/>Ergodic Analysis"]
        BUSINESS_OPT["business_optimizer.py<br/>Optimization"]
        DECISION_ENGINE["decision_engine.py<br/>Decision Making"]
        OPTIMIZATION["optimization.py<br/>Optimization Algos"]
        HJB_SOLVER["hjb_solver.py<br/>HJB Equations"]
        OPTIMAL_CTRL["optimal_control.py<br/>Control Theory"]
    end

    %% Validation & Testing
    subgraph Validation["Validation"]
        ACCURACY_VAL["accuracy_validator.py<br/>Accuracy Checks"]
        STRATEGY_BACK["strategy_backtester.py<br/>Backtesting"]
        WALK_FORWARD["walk_forward_validator.py<br/>Walk-Forward"]
        VALIDATION_METRICS["validation_metrics.py<br/>Metrics"]
        STATISTICAL_TESTS["statistical_tests.py<br/>Statistical Tests"]
    end

    %% Risk Analysis
    subgraph Risk["Risk Analysis"]
        RISK_METRICS["risk_metrics.py<br/>Risk Metrics"]
        RUIN_PROB["ruin_probability.py<br/>Ruin Analysis"]
        SENSITIVITY["sensitivity.py<br/>Sensitivity Analysis"]
        PARETO["pareto_frontier.py<br/>Pareto Analysis"]
        BOOTSTRAP["bootstrap_analysis.py<br/>Bootstrap Methods"]
    end

    %% Performance & Infrastructure
    subgraph Infrastructure["Infrastructure"]
        BATCH_PROC["batch_processor.py<br/>Batch Processing"]
        PARALLEL_EXEC["parallel_executor.py<br/>Parallelization"]
        PERF_OPT["performance_optimizer.py<br/>Performance"]
        TRAJ_STORAGE["trajectory_storage.py<br/>Data Storage"]
        PROGRESS_MON["progress_monitor.py<br/>Progress Tracking"]
        PARAM_SWEEP["parameter_sweep.py<br/>Parameter Sweeps"]
    end

    %% Reporting & Visualization
    subgraph Reporting["Reporting & Visualization"]
        VIZ_LEGACY["visualization_legacy.py<br/>Legacy Plots"]
        EXCEL_REPORT["excel_reporter.py<br/>Excel Reports"]
        SUMMARY_STATS["summary_statistics.py<br/>Statistics"]
        RESULT_AGG["result_aggregator.py<br/>Aggregation"]
        FINANCIAL_STMT["financial_statements.py<br/>Statements"]
    end

    %% Visualization Submodule
    subgraph VizModule["visualization/"]
        VIZ_CORE["core.py<br/>Core Functions"]
        VIZ_EXEC["executive_plots.py<br/>Executive Views"]
        VIZ_TECH["technical_plots.py<br/>Technical Views"]
        VIZ_ANNOT["annotations.py<br/>Annotations"]
        VIZ_STYLE["style_manager.py<br/>Styling"]
        VIZ_FACTORY["figure_factory.py<br/>Figure Factory"]
        VIZ_EXPORT["export.py<br/>Export Tools"]
    end

    %% Advanced Features
    subgraph Advanced["Advanced Features"]
        CONVERGENCE["convergence.py<br/>Convergence"]
        CONV_ADV["convergence_advanced.py<br/>Advanced Conv."]
        CONV_PLOTS["convergence_plots.py<br/>Conv. Plots"]
        ADAPTIVE_STOP["adaptive_stopping.py<br/>Adaptive Stopping"]
        SCENARIO_MGR["scenario_manager.py<br/>Scenarios"]
        BENCHMARKING["benchmarking.py<br/>Benchmarks"]
    end

    %% Key Dependencies
    CONFIG_BASE --> MANUFACTURER
    CONFIG_V2 --> MANUFACTURER
    CONFIG_MGR --> CONFIG_LOADER

    MANUFACTURER --> SIM_CORE
    INSURANCE --> INS_PROGRAM
    INS_PRICING --> INS_PROGRAM
    CLAIM_GEN --> SIM_CORE

    SIM_CORE --> MONTE_CARLO
    STOCHASTIC --> MONTE_CARLO
    LOSS_DIST --> CLAIM_GEN

    MONTE_CARLO --> ERGODIC_ANALYZER
    ERGODIC_ANALYZER --> BUSINESS_OPT
    BUSINESS_OPT --> DECISION_ENGINE

    MONTE_CARLO --> ACCURACY_VAL
    STRATEGY_BACK --> WALK_FORWARD

    ERGODIC_ANALYZER --> RISK_METRICS
    RISK_METRICS --> RUIN_PROB
    SENSITIVITY --> PARETO

    BATCH_PROC --> PARALLEL_EXEC
    PARALLEL_EXEC --> MONTE_CARLO

    RESULT_AGG --> SUMMARY_STATS
    SUMMARY_STATS --> EXCEL_REPORT
    FINANCIAL_STMT --> EXCEL_REPORT

    VIZ_CORE --> VIZ_FACTORY
    VIZ_STYLE --> VIZ_EXEC
    VIZ_STYLE --> VIZ_TECH
    VIZ_FACTORY --> VIZ_EXPORT

    %% Styling
    classDef config fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef business fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef simulation fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef analysis fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef validation fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef risk fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef infra fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    classDef reporting fill:#f1f8e9,stroke:#558b2f,stroke-width:2px
    classDef viz fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef advanced fill:#fafafa,stroke:#424242,stroke-width:2px

    class CONFIG_BASE,CONFIG_V2,CONFIG_MGR,CONFIG_LOADER,CONFIG_COMPAT,CONFIG_MIG config
    class MANUFACTURER,INSURANCE,INS_PROGRAM,INS_PRICING,CLAIM_GEN,CLAIM_DEV business
    class SIM_CORE,MONTE_CARLO,STOCHASTIC,LOSS_DIST simulation
    class ERGODIC_ANALYZER,BUSINESS_OPT,DECISION_ENGINE,OPTIMIZATION,HJB_SOLVER,OPTIMAL_CTRL analysis
    class ACCURACY_VAL,STRATEGY_BACK,WALK_FORWARD,VALIDATION_METRICS,STATISTICAL_TESTS validation
    class RISK_METRICS,RUIN_PROB,SENSITIVITY,PARETO,BOOTSTRAP risk
    class BATCH_PROC,PARALLEL_EXEC,PERF_OPT,TRAJ_STORAGE,PROGRESS_MON,PARAM_SWEEP infra
    class VIZ_LEGACY,EXCEL_REPORT,SUMMARY_STATS,RESULT_AGG,FINANCIAL_STMT reporting
    class VIZ_CORE,VIZ_EXEC,VIZ_TECH,VIZ_ANNOT,VIZ_STYLE,VIZ_FACTORY,VIZ_EXPORT viz
    class CONVERGENCE,CONV_ADV,CONV_PLOTS,ADAPTIVE_STOP,SCENARIO_MGR,BENCHMARKING advanced
```

## Module Categories

### Configuration Management
Handles all configuration aspects including loading, validation, migration, and compatibility between different configuration versions.

### Business Logic
Core business domain models including the manufacturer, insurance policies, pricing, and claim processing.

### Simulation Core
The main simulation engine that orchestrates time evolution, Monte Carlo runs, and stochastic processes.

### Analysis & Optimization
Advanced analytical tools including ergodic analysis, business optimization, and decision-making engines.

### Validation
Comprehensive validation framework for ensuring accuracy and robustness of simulations.

### Risk Analysis
Specialized risk assessment tools including ruin probability, sensitivity analysis, and bootstrap methods.

### Infrastructure
High-performance computing infrastructure for parallel processing, caching, and data management.

### Reporting & Visualization
Output generation including Excel reports, visualizations, and statistical summaries.

### Advanced Features
Sophisticated features for convergence monitoring, adaptive stopping, and benchmarking.
