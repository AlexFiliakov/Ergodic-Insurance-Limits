# Module Overview and Dependencies

This diagram shows the detailed module structure and dependencies within the Ergodic Insurance framework.

```{mermaid}
graph LR
       %% Configuration Layer
       subgraph Config["Configuration Management"]
           CONFIG_BASE["config.py<br/>Base Configuration"]
           CONFIG_V2["config.py<br/>Config Models"]
           CONFIG_MGR["config_manager.py<br/>Config Manager"]
           CONFIG_LOADER["config_loader.py<br/>Config Loader"]
           CONFIG_MIG["config_migrator.py<br/>Migration Tools"]
       end

       %% Core Business Logic
       subgraph Business["Business Logic"]
           MANUFACTURER["manufacturer.py<br/>ClaimLiability, TaxHandler,<br/>WidgetManufacturer"]
           INSURANCE["insurance.py<br/>Insurance Policy"]
           INS_PROGRAM["insurance_program.py<br/>Insurance Program"]
           INS_PRICING["insurance_pricing.py<br/>Pricing Models"]
           CLAIM_DEV["claim_development.py<br/>Claim Development"]
           EXPOSURE["exposure_base.py<br/>Exposure Models &amp;<br/>FinancialStateProvider Protocol"]
           LEDGER["ledger.py<br/>Double-Entry Ledger"]
           ACCRUAL["accrual_manager.py<br/>Accrual Accounting"]
           INS_ACCT["insurance_accounting.py<br/>Insurance Accounting"]
           DECIMAL_UTILS["decimal_utils.py<br/>Decimal Precision"]
           TRENDS["trends.py<br/>Trend Analysis"]
       end

       %% Simulation Engine
       subgraph Simulation["Simulation Core"]
           SIM_CORE["simulation.py<br/>Main Engine"]
           MONTE_CARLO["monte_carlo.py<br/>Monte Carlo"]
           MONTE_WORKER["monte_carlo_worker.py<br/>MC Worker"]
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
           SENS_VIZ["sensitivity_visualization.py<br/>Sensitivity Viz"]
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
           VIZ_BATCH["batch_plots.py<br/>Batch Plotting"]
           VIZ_INTERACT["interactive_plots.py<br/>Interactive Plots"]
           VIZ_TOWER["improved_tower_plot.py<br/>Tower Plots"]
       end

       %% Reporting Submodule
       subgraph ReportModule["reporting/"]
           REP_BUILDER["report_builder.py<br/>Report Builder"]
           REP_EXEC["executive_report.py<br/>Executive Reports"]
           REP_TECH["technical_report.py<br/>Technical Reports"]
           REP_SCENARIO["scenario_comparator.py<br/>Scenario Compare"]
           REP_TABLE["table_generator.py<br/>Table Generator"]
           REP_INSIGHT["insight_extractor.py<br/>Insights"]
           REP_FORMAT["formatters.py<br/>Formatters"]
           REP_CACHE["cache_manager.py<br/>Cache Manager"]
           REP_VALID["validator.py<br/>Report Validator"]
           REP_CONFIG["config.py<br/>Report Config"]
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

       %% Configuration dependencies
       CONFIG_BASE --> MANUFACTURER
       CONFIG_V2 --> CONFIG_MGR
       CONFIG_MGR --> CONFIG_LOADER
       CONFIG_LOADER --> CONFIG_MGR

       %% Business Logic: Decimal utilities feed into accounting modules
       DECIMAL_UTILS --> LEDGER
       DECIMAL_UTILS --> ACCRUAL
       DECIMAL_UTILS --> INS_ACCT
       DECIMAL_UTILS --> MANUFACTURER

       %% Business Logic: Accounting modules feed into manufacturer
       LEDGER --> MANUFACTURER
       ACCRUAL --> MANUFACTURER
       INS_ACCT --> MANUFACTURER

       %% Business Logic: Insurance and exposure relationships
       INSURANCE --> INS_PROGRAM
       INS_PRICING --> INS_PROGRAM
       EXPOSURE --> MANUFACTURER

       %% Simulation dependencies
       MANUFACTURER --> SIM_CORE
       INS_PROGRAM --> SIM_CORE
       LOSS_DIST --> SIM_CORE
       SIM_CORE --> MONTE_CARLO
       MONTE_CARLO --> MONTE_WORKER
       STOCHASTIC --> MONTE_CARLO

       %% Analysis dependencies
       MONTE_CARLO --> ERGODIC_ANALYZER
       ERGODIC_ANALYZER --> BUSINESS_OPT
       BUSINESS_OPT --> DECISION_ENGINE

       %% Validation dependencies
       MONTE_CARLO --> ACCURACY_VAL
       STRATEGY_BACK --> WALK_FORWARD

       %% Risk dependencies
       ERGODIC_ANALYZER --> RISK_METRICS
       RISK_METRICS --> RUIN_PROB
       SENSITIVITY --> PARETO
       SENSITIVITY --> SENS_VIZ

       %% Infrastructure dependencies
       BATCH_PROC --> PARALLEL_EXEC
       PARALLEL_EXEC --> MONTE_CARLO

       %% Reporting dependencies
       RESULT_AGG --> SUMMARY_STATS
       SUMMARY_STATS --> EXCEL_REPORT
       FINANCIAL_STMT --> EXCEL_REPORT

       %% Visualization dependencies
       VIZ_CORE --> VIZ_FACTORY
       VIZ_STYLE --> VIZ_EXEC
       VIZ_STYLE --> VIZ_TECH
       VIZ_FACTORY --> VIZ_EXPORT
       VIZ_BATCH --> VIZ_CORE
       VIZ_INTERACT --> VIZ_CORE
       VIZ_TOWER --> VIZ_STYLE

       %% Reporting module dependencies
       REP_BUILDER --> REP_EXEC
       REP_BUILDER --> REP_TECH
       REP_SCENARIO --> REP_TABLE
       REP_INSIGHT --> REP_EXEC
       REP_FORMAT --> REP_TABLE
       REP_CACHE --> REP_BUILDER
       REP_VALID --> REP_BUILDER

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
       classDef report fill:#fff8e1,stroke:#f9a825,stroke-width:2px
       classDef advanced fill:#fafafa,stroke:#424242,stroke-width:2px

       class CONFIG_BASE,CONFIG_V2,CONFIG_MGR,CONFIG_LOADER,CONFIG_MIG config
       class MANUFACTURER,INSURANCE,INS_PROGRAM,INS_PRICING,CLAIM_DEV,EXPOSURE,LEDGER,ACCRUAL,INS_ACCT,DECIMAL_UTILS,TRENDS business
       class SIM_CORE,MONTE_CARLO,MONTE_WORKER,STOCHASTIC,LOSS_DIST simulation
       class ERGODIC_ANALYZER,BUSINESS_OPT,DECISION_ENGINE,OPTIMIZATION,HJB_SOLVER,OPTIMAL_CTRL analysis
       class ACCURACY_VAL,STRATEGY_BACK,WALK_FORWARD,VALIDATION_METRICS,STATISTICAL_TESTS validation
       class RISK_METRICS,RUIN_PROB,SENSITIVITY,SENS_VIZ,PARETO,BOOTSTRAP risk
       class BATCH_PROC,PARALLEL_EXEC,PERF_OPT,TRAJ_STORAGE,PROGRESS_MON,PARAM_SWEEP infra
       class VIZ_LEGACY,EXCEL_REPORT,SUMMARY_STATS,RESULT_AGG,FINANCIAL_STMT reporting
       class VIZ_CORE,VIZ_EXEC,VIZ_TECH,VIZ_ANNOT,VIZ_STYLE,VIZ_FACTORY,VIZ_EXPORT,VIZ_BATCH,VIZ_INTERACT,VIZ_TOWER viz
       class REP_BUILDER,REP_EXEC,REP_TECH,REP_SCENARIO,REP_TABLE,REP_INSIGHT,REP_FORMAT,REP_CACHE,REP_VALID,REP_CONFIG report
       class CONVERGENCE,CONV_ADV,CONV_PLOTS,ADAPTIVE_STOP,SCENARIO_MGR,BENCHMARKING advanced
```

## Module Categories

### Configuration Management
Handles all configuration aspects including loading, validation, migration, and compatibility between different configuration versions.

### Business Logic
Core business domain models including the manufacturer, insurance policies, pricing, claim processing, and financial accounting infrastructure.

- **ledger.py** - Double-entry financial ledger implementing event-sourced transaction tracking. Provides `AccountType`, `AccountName`, `EntryType`, and `TransactionType` enums along with `LedgerEntry` and `Ledger` classes for GAAP-compliant accounting with full audit trails.
- **accrual_manager.py** - Accrual accounting management following GAAP timing principles. Contains `AccrualType` and `PaymentSchedule` enums, plus `AccrualItem` and `AccrualManager` classes for tracking timing differences between cash movements and accounting recognition.
- **insurance_accounting.py** - Insurance premium accounting with prepaid asset tracking and systematic amortization. Provides `InsuranceRecovery` and `InsuranceAccounting` classes for claim recovery receivables and premium expense management.
- **decimal_utils.py** - Decimal precision utilities for financial calculations. Provides `to_decimal`, `quantize_currency`, and related helpers along with constants (`ZERO`, `ONE`, `PENNY`) to prevent floating-point accumulation errors in iterative simulations.
- **trends.py** - Trend analysis for insurance claim frequency and severity adjustments. Implements a hierarchy of trend classes (`Trend`, `NoTrend`, `LinearTrend`, `RandomWalkTrend`, `MeanRevertingTrend`, `RegimeSwitchingTrend`, `ScenarioTrend`) that apply multiplicative adjustments over time.
- **manufacturer.py** - Core financial model containing `ClaimLiability` (actuarial claim payment tracking), `TaxHandler` (tax computation logic), and `WidgetManufacturer` (main business simulation class). `WidgetManufacturer` integrates with `Ledger`, `AccrualManager`, and `InsuranceAccounting` for full double-entry financial modeling.
- **exposure_base.py** - Exposure models and the `FinancialStateProvider` protocol. The protocol defines the interface for providing real-time financial state to exposure bases; `WidgetManufacturer` implements this protocol.

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
