# High-Level System Context Diagram

## Overview
This diagram shows the overall architecture of the Ergodic Insurance Limits system, illustrating how different components interact to provide insurance optimization through ergodic theory.

```mermaid
flowchart TB
    subgraph External["External Systems & I/O"]
        YAML["/Configuration Files<br/>(YAML/JSON)"/]
        CSV["/Data Export<br/>(CSV/Excel/Parquet)"/]
        CHECKPOINT["/Checkpoint Storage<br/>(Persistence)"/]
        JUPYTER["/Jupyter Notebooks<br/>(Interactive Analysis)"/]
        SPHINX["/Documentation<br/>(Sphinx API Docs)"/]
    end

    subgraph ConfigLayer["Configuration Management v2.0"]
        CONFIG_MGR["Config Manager<br/>(3-Tier System)"]
        CONFIG_V2["ConfigV2<br/>(Pydantic Models)"]
        CONFIG_COMPAT["Legacy Adapter<br/>(Backward Compat)"]
        CONFIG_MIGRATOR["Config Migrator<br/>(Version Upgrade)"]
    end

    subgraph FinancialCore["Financial Modeling Core"]
        MANUFACTURER["Widget Manufacturer<br/>(Balance Sheet Model)"]
        CLAIM_GEN["Claim Generator<br/>(Loss Events)"]
        CLAIM_DEV["Claim Development<br/>(Payment Patterns)"]
        STOCHASTIC["Stochastic Processes<br/>(GBM, OU, Lognormal)"]
    end

    subgraph InsuranceLayer["Insurance & Risk Management"]
        INSURANCE["Insurance Policy<br/>(Basic Coverage)"]
        INSURANCE_PROG["Insurance Program<br/>(Multi-Layer)"]
        LOSS_DIST["Loss Distributions<br/>(Frequency/Severity)"]
        RUIN_PROB["Ruin Probability<br/>(Survival Analysis)"]
    end

    subgraph OptimizationEngine["Optimization & Control"]
        OPTIMIZER["Business Optimizer<br/>(Strategy Search)"]
        DECISION["Decision Engine<br/>(Multi-Objective)"]
        PARETO["Pareto Frontier<br/>(Trade-off Analysis)"]
        HJB["HJB Solver<br/>(Dynamic Programming)"]
        OPTIMAL_CTRL["Optimal Control<br/>(Feedback Strategies)"]
    end

    subgraph SimulationFramework["Simulation & Execution"]
        SIMULATION["Main Simulation<br/>(Orchestrator)"]
        MONTE_CARLO["Monte Carlo Engine<br/>(Ensemble Runs)"]
        PARALLEL_EXEC["Parallel Executor<br/>(CPU Optimization)"]
        BATCH_PROC["Batch Processor<br/>(Scenario Management)"]
        TRAJECTORY["Trajectory Storage<br/>(Memory Efficient)"]
    end

    subgraph AnalyticsLayer["Analytics & Metrics"]
        ERGODIC["Ergodic Analyzer<br/>(Time vs Ensemble)"]
        RISK_METRICS["Risk Metrics<br/>(VaR, CVaR, TVaR)"]
        CONVERGENCE["Convergence Tools<br/>(Diagnostics)"]
        CONVERGENCE_ADV["Advanced Convergence<br/>(Multi-Metric)"]
        CONVERGENCE_PLOT["Convergence Plots<br/>(Visualization)"]
        BOOTSTRAP["Bootstrap Analysis<br/>(Confidence Intervals)"]
        STATS_TESTS["Statistical Tests<br/>(Hypothesis Testing)"]
        SENSITIVITY["Sensitivity Analysis<br/>(Parameter Impact)"]
        PARAMETER_SWEEP["Parameter Sweep<br/>(Grid Search)"]
    end

    subgraph ResultsProcessing["Results & Reporting"]
        AGGREGATOR["Result Aggregator<br/>(Hierarchical)"]
        SUMMARY_STATS["Summary Statistics<br/>(Distribution Analysis)"]
        SCENARIO_MGR["Scenario Manager<br/>(Parameter Grids)"]
        PROGRESS_MON["Progress Monitor<br/>(Real-time Tracking)"]
        VISUALIZATION["Visualization<br/>(WSJ-Style Charts)"]
        SENSITIVITY_VIZ["Sensitivity Viz<br/>(Tornado Charts)"]
        EXCEL_REPORTER["Excel Reporter<br/>(Business Reports)"]
        FINANCIAL_STMT["Financial Statements<br/>(P&L, Balance Sheet)"]
    end

    subgraph ValidationFramework["Validation & Testing"]
        WALK_FORWARD["Walk-Forward Validator<br/>(Out-of-Sample)"]
        STRATEGY_BACKTEST["Strategy Backtester<br/>(Historical Validation)"]
        VALIDATION_METRICS["Validation Metrics<br/>(Performance KPIs)"]
        ACCURACY_VAL["Accuracy Validator<br/>(Numerical Precision)"]
        PERFORMANCE_OPT["Performance Optimizer<br/>(Speed Tuning)"]
        BENCHMARKING["Benchmarking Suite<br/>(Performance Tests)"]
        ADAPTIVE_STOP["Adaptive Stopping<br/>(Early Termination)"]
    end

    subgraph InsuranceAdvanced["Advanced Insurance Features"]
        INSURANCE_PRICING["Insurance Pricing<br/>(Premium Models)"]
    end

    %% Configuration Flow
    YAML -->|Load| CONFIG_MGR
    CONFIG_MGR --> CONFIG_V2
    CONFIG_V2 --> CONFIG_COMPAT
    CONFIG_MIGRATOR -->|Upgrade| CONFIG_V2

    %% Initialization Flow
    CONFIG_V2 -->|Initialize| MANUFACTURER
    CONFIG_V2 -->|Configure| INSURANCE_PROG
    CONFIG_V2 -->|Setup| STOCHASTIC
    CONFIG_V2 -->|Define| LOSS_DIST

    %% Simulation Flow
    MANUFACTURER -->|State| SIMULATION
    CLAIM_GEN -->|Events| MANUFACTURER
    CLAIM_DEV -->|Cash Flows| MANUFACTURER
    STOCHASTIC -->|Shocks| MANUFACTURER
    INSURANCE_PROG -->|Coverage| MANUFACTURER

    %% Parallel Execution
    SIMULATION -->|Orchestrate| MONTE_CARLO
    MONTE_CARLO -->|Distribute| PARALLEL_EXEC
    PARALLEL_EXEC -->|Execute| BATCH_PROC
    BATCH_PROC -->|Store| TRAJECTORY
    TRAJECTORY -->|Checkpoint| CHECKPOINT

    %% Analytics Flow
    TRAJECTORY -->|Time Series| ERGODIC
    TRAJECTORY -->|Paths| RISK_METRICS
    RISK_METRICS -->|Metrics| CONVERGENCE
    CONVERGENCE -->|Advanced| CONVERGENCE_ADV
    CONVERGENCE_ADV -->|Visualize| CONVERGENCE_PLOT
    CONVERGENCE -->|Tests| STATS_TESTS
    STATS_TESTS -->|Bootstrap| BOOTSTRAP
    RISK_METRICS -->|Parameters| SENSITIVITY
    SENSITIVITY -->|Grid| PARAMETER_SWEEP

    %% Optimization Flow
    RISK_METRICS -->|Objectives| OPTIMIZER
    ERGODIC -->|Growth Rates| OPTIMIZER
    OPTIMIZER -->|Search| DECISION
    DECISION -->|Evaluate| PARETO
    HJB -->|Value Function| OPTIMAL_CTRL
    OPTIMAL_CTRL -->|Strategy| DECISION

    %% Results Flow
    BATCH_PROC -->|Results| AGGREGATOR
    AGGREGATOR -->|Statistics| SUMMARY_STATS
    SUMMARY_STATS -->|Reports| SCENARIO_MGR
    SCENARIO_MGR -->|Monitor| PROGRESS_MON
    SUMMARY_STATS -->|Visualize| VISUALIZATION
    SENSITIVITY -->|Charts| SENSITIVITY_VIZ
    SUMMARY_STATS -->|Business| EXCEL_REPORTER
    MANUFACTURER -->|Financials| FINANCIAL_STMT

    %% Validation Flow
    TRAJECTORY -->|Historical| WALK_FORWARD
    WALK_FORWARD -->|Backtest| STRATEGY_BACKTEST
    STRATEGY_BACKTEST -->|Metrics| VALIDATION_METRICS
    VALIDATION_METRICS -->|Accuracy| ACCURACY_VAL
    MONTE_CARLO -->|Performance| PERFORMANCE_OPT
    PERFORMANCE_OPT -->|Benchmark| BENCHMARKING
    CONVERGENCE -->|Early Stop| ADAPTIVE_STOP

    %% Insurance Advanced Flow
    LOSS_DIST -->|Pricing| INSURANCE_PRICING
    INSURANCE_PRICING -->|Premium| INSURANCE_PROG

    %% Output Flow
    VISUALIZATION -->|Export| CSV
    AGGREGATOR -->|Save| CHECKPOINT
    VISUALIZATION -->|Interactive| JUPYTER
    CONFIG_V2 -->|Document| SPHINX
    EXCEL_REPORTER -->|Export| CSV
    FINANCIAL_STMT -->|Export| CSV

    style ConfigLayer fill:#e3f2fd
    style FinancialCore fill:#e8f5e9
    style InsuranceLayer fill:#fff3e0
    style InsuranceAdvanced fill:#ffe0b2
    style OptimizationEngine fill:#fce4ec
    style SimulationFramework fill:#f3e5f5
    style AnalyticsLayer fill:#e0f2f1
    style ResultsProcessing fill:#fff9c4
    style ValidationFramework fill:#e8eaf6
    style External fill:#efebe9
```

## Component Descriptions

### External Systems & I/O
- **Configuration Files**: YAML and JSON configuration files for scenarios and parameters
- **Data Export**: Results export to CSV, Excel, and Parquet formats
- **Checkpoint Storage**: Persistence layer for simulation checkpoints and recovery
- **Jupyter Notebooks**: Interactive analysis and demonstration environment
- **Documentation**: Sphinx-generated API documentation

### Configuration Management v2.0
- **Config Manager**: 3-tier configuration system (profiles, modules, presets)
- **ConfigV2**: Enhanced Pydantic v2 models with validation
- **Legacy Adapter**: Backward compatibility with v1 configurations
- **Config Migrator**: Tool for upgrading legacy configurations

### Financial Modeling Core
- **Widget Manufacturer**: Balance sheet evolution and financial state modeling
- **Claim Generator**: Loss event generation with configurable distributions
- **Claim Development**: Payment pattern modeling over time
- **Stochastic Processes**: GBM, Ornstein-Uhlenbeck, and lognormal volatility

### Insurance & Risk Management
- **Insurance Policy**: Basic insurance coverage implementation
- **Insurance Program**: Multi-layer insurance structures with reinstatements
- **Loss Distributions**: Frequency and severity modeling (Poisson, lognormal, Pareto)
- **Ruin Probability**: Survival analysis and bankruptcy risk assessment

### Optimization & Control
- **Business Optimizer**: Strategy search and optimization algorithms
- **Decision Engine**: Multi-objective decision support system
- **Pareto Frontier**: Trade-off analysis between competing objectives
- **HJB Solver**: Hamilton-Jacobi-Bellman equation solver for dynamic programming
- **Optimal Control**: Feedback control strategies for adaptive decisions

### Simulation Framework
- **Main Simulation**: Central orchestrator for simulation execution
- **Monte Carlo Engine**: Ensemble simulation runner
- **Parallel Executor**: CPU-optimized parallel processing
- **Batch Processor**: Scenario batch management with checkpointing
- **Trajectory Storage**: Memory-efficient storage of simulation paths

### Analytics & Metrics
- **Ergodic Analyzer**: Time average vs ensemble average comparison
- **Risk Metrics**: Comprehensive risk measures (VaR, CVaR, TVaR, Sharpe, etc.)
- **Convergence Tools**: Statistical convergence diagnostics
- **Advanced Convergence**: Multi-metric convergence analysis with sophisticated tests
- **Convergence Plots**: Visualization of convergence diagnostics
- **Bootstrap Analysis**: Confidence interval estimation
- **Statistical Tests**: Hypothesis testing framework
- **Sensitivity Analysis**: Parameter impact assessment and tornado diagrams
- **Parameter Sweep**: Grid search across parameter spaces

### Results Processing
- **Result Aggregator**: Hierarchical aggregation of simulation results
- **Summary Statistics**: Distribution analysis and statistical summaries
- **Scenario Manager**: Parameter grid management and scenario generation
- **Progress Monitor**: Real-time simulation progress tracking
- **Visualization**: WSJ-style professional charting
- **Sensitivity Visualization**: Tornado charts and sensitivity heat maps
- **Excel Reporter**: Automated business reports in Excel format
- **Financial Statements**: P&L and balance sheet generation

### Validation & Testing Framework
- **Walk-Forward Validator**: Out-of-sample validation for strategy robustness
- **Strategy Backtester**: Historical validation of insurance strategies
- **Validation Metrics**: Comprehensive performance KPIs and metrics
- **Accuracy Validator**: Numerical precision and correctness testing
- **Performance Optimizer**: Speed tuning and computational optimization
- **Benchmarking Suite**: Performance testing and regression detection
- **Adaptive Stopping**: Early termination based on convergence criteria

### Advanced Insurance Features
- **Insurance Pricing**: Sophisticated premium modeling and rate calculation

(data-flow-patterns)=
(data-flow-patterns)=
## Data Flow Patterns

1. **Configuration Pipeline**: External configs → Manager → V2 models → Component initialization
2. **Simulation Pipeline**: Financial model + Stochastic + Insurance → Simulation engine → Parallel execution
3. **Analytics Pipeline**: Raw trajectories → Metrics calculation → Statistical analysis → Optimization
4. **Results Pipeline**: Aggregation → Summary stats → Visualization → Export
5. **Validation Pipeline**: Historical data → Walk-forward → Backtesting → Performance metrics
6. **Sensitivity Pipeline**: Base scenario → Parameter variations → Impact analysis → Visualization
7. **Reporting Pipeline**: Results → Financial statements → Excel reports → Business insights

## Key Architectural Decisions

1. **Modular Design**: Each component has a single responsibility with clear interfaces
2. **Configuration-Driven**: All parameters externalized through Pydantic models
3. **Parallel Processing**: CPU-optimized execution for large-scale simulations
4. **Memory Efficiency**: Trajectory storage and batch processing for long simulations
5. **Extensibility**: Plugin-style architecture for new loss distributions and processes
6. **Testing**: 100% test coverage with comprehensive unit and integration tests
