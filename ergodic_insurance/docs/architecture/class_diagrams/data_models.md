# Data Models and Analysis Classes

This document shows the data structures and analysis models used throughout the system. The diagrams are split into focused sections for readability.

## Ergodic Analysis

The ergodic analysis subsystem implements Ole Peters' ergodic economics framework, comparing time-average versus ensemble-average growth rates to demonstrate how insurance transforms business growth dynamics.

```{mermaid}
classDiagram
    class ErgodicAnalyzer {
        -convergence_threshold: float
        +calculate_time_average_growth(trajectories) dict
        +calculate_ensemble_average(trajectories) dict
        +compare_scenarios(insured, uninsured, metric) dict
        +check_convergence(values, window_size) tuple
        +analyze_simulation_batch(results, label) dict
        +integrate_loss_ergodic_analysis(loss_data, insurance, manufacturer) ErgodicAnalysisResults
        +validate_insurance_ergodic_impact(...) ValidationResults
        +significance_test(insured_growth, uninsured_growth) dict
    }

    class ErgodicData {
        <<dataclass>>
        +time_series: ndarray
        +values: ndarray
        +metadata: dict
        +validate() bool
    }

    class ErgodicAnalysisResults {
        <<dataclass>>
        +time_average_growth: float
        +ensemble_average_growth: float
        +survival_rate: float
        +ergodic_divergence: float
        +insurance_impact: dict
        +validation_passed: bool
        +metadata: dict
    }

    class ValidationResults {
        <<dataclass>>
        +is_valid: bool
        +checks: dict
        +warnings: list
        +summary: str
    }

    ErgodicAnalyzer --> ErgodicData : accepts
    ErgodicAnalyzer --> ErgodicAnalysisResults : produces
    ErgodicAnalyzer --> ValidationResults : validates with
    ErgodicAnalysisResults --> ErgodicData : derived from
```

**ErgodicAnalyzer** is the core analysis engine. It accepts trajectories as `ErgodicData` or `SimulationResults`, calculates time-average and ensemble-average growth rates, performs convergence checks, and runs integrated loss-ergodic analysis. The `compare_scenarios()` method is the primary entry point for comparing insured versus uninsured outcomes.

**ErgodicData** is a lightweight dataclass holding time series arrays and metadata. It validates array length consistency before analysis.

**ErgodicAnalysisResults** captures the complete output of an integrated analysis, including growth rates, survival statistics, insurance impact metrics, and validation status.

## Business Optimization

The optimization subsystem uses ergodic metrics to find insurance strategies that maximize real business outcomes such as ROE, growth rate, and survival probability.

```{mermaid}
classDiagram
    class BusinessOptimizer {
        -manufacturer: WidgetManufacturer
        -loss_distribution: LossDistribution
        -decision_engine: InsuranceDecisionEngine
        -ergodic_analyzer: ErgodicAnalyzer
        -optimizer_config: BusinessOptimizerConfig
        +maximize_roe_with_insurance(constraints, time_horizon) OptimalStrategy
        +minimize_bankruptcy_risk(growth_targets, budget) OptimalStrategy
        +optimize_capital_efficiency(constraints) OptimalStrategy
        +optimize_business_outcomes(objectives, constraints) BusinessOptimizationResult
    }

    class OptimalStrategy {
        <<dataclass>>
        +coverage_limit: float
        +deductible: float
        +premium_rate: float
        +expected_roe: float
        +bankruptcy_risk: float
        +growth_rate: float
        +capital_efficiency: float
        +recommendations: list~str~
        +to_dict() dict
    }

    class BusinessObjective {
        <<dataclass>>
        +name: str
        +weight: float
        +target_value: float
        +optimization_direction: OptimizationDirection
        +constraint_type: str
        +constraint_value: float
    }

    class BusinessConstraints {
        <<dataclass>>
        +max_risk_tolerance: float
        +min_roe_threshold: float
        +max_leverage_ratio: float
        +min_liquidity_ratio: float
        +max_premium_budget: float
        +min_coverage_ratio: float
        +regulatory_requirements: dict
    }

    class BusinessOptimizationResult {
        <<dataclass>>
        +optimal_strategy: OptimalStrategy
        +objective_values: dict
        +constraint_satisfaction: dict
        +convergence_info: dict
        +sensitivity_analysis: dict
        +is_feasible() bool
    }

    BusinessOptimizer --> OptimalStrategy : finds
    BusinessOptimizer --> BusinessObjective : uses
    BusinessOptimizer --> BusinessConstraints : respects
    BusinessOptimizer --> BusinessOptimizationResult : produces
    BusinessOptimizationResult --> OptimalStrategy : contains
```

**BusinessOptimizer** provides multiple optimization methods: `maximize_roe_with_insurance()` for ROE-focused optimization, `minimize_bankruptcy_risk()` for safety-first strategies, `optimize_capital_efficiency()` for capital allocation, and `optimize_business_outcomes()` for multi-objective optimization using `BusinessObjective` definitions.

**OptimalStrategy** is the output dataclass capturing the recommended insurance parameters (coverage limit, deductible, premium rate) along with expected business outcomes and actionable recommendations.

## Risk Analysis

Risk metrics and ruin probability analysis provide the quantitative foundation for evaluating tail risk and insurance value.

```{mermaid}
classDiagram
    class RiskMetrics {
        -losses: ndarray
        -weights: ndarray
        -rng: Generator
        +var(confidence, method, bootstrap_ci) float
        +tvar(confidence) float
        +expected_shortfall(confidence) float
        +pml(return_period) float
        +maximum_drawdown() float
        +economic_capital(confidence) float
        +tail_index(threshold) float
        +risk_adjusted_metrics() dict
        +coherence_test() dict
        +summary_statistics() dict
        +plot_distribution()
    }

    class RiskMetricsResult {
        <<dataclass>>
        +metric_name: str
        +value: float
        +confidence_level: float
        +confidence_interval: tuple
        +metadata: dict
    }

    class RuinProbabilityAnalyzer {
        -manufacturer: WidgetManufacturer
        -loss_generator: ManufacturingLossGenerator
        -insurance_program: InsuranceProgram
        -config: SimulationConfig
        +analyze_ruin_probability(config) RuinProbabilityResults
    }

    class RuinProbabilityResults {
        <<dataclass>>
        +time_horizons: ndarray
        +ruin_probabilities: ndarray
        +confidence_intervals: ndarray
        +bankruptcy_causes: dict
        +survival_curves: ndarray
        +execution_time: float
        +n_simulations: int
        +convergence_achieved: bool
        +mid_year_ruin_count: int
        +ruin_month_distribution: dict
        +summary() str
    }

    class RuinProbabilityConfig {
        <<dataclass>>
        +time_horizons: list~int~
        +n_simulations: int
        +min_assets_threshold: float
        +min_equity_threshold: float
        +early_stopping: bool
        +parallel: bool
        +n_workers: int
        +seed: int
        +n_bootstrap: int
    }

    RiskMetrics --> RiskMetricsResult : returns
    RuinProbabilityAnalyzer --> RuinProbabilityResults : produces
    RuinProbabilityAnalyzer --> RuinProbabilityConfig : configured by
```

**RiskMetrics** is initialized with a loss array and provides VaR, TVaR (CVaR), Expected Shortfall, PML, Maximum Drawdown, and other tail-risk measures. It supports both empirical and parametric methods with optional bootstrap confidence intervals.

**RuinProbabilityAnalyzer** runs Monte Carlo ruin analysis across multiple time horizons, with support for parallel execution, bootstrap confidence intervals, and mid-year ruin tracking.

## Convergence Diagnostics

Convergence analysis ensures Monte Carlo simulations have run long enough to produce reliable results.

```{mermaid}
classDiagram
    class ConvergenceDiagnostics {
        -r_hat_threshold: float
        -min_ess: int
        -relative_mcse_threshold: float
        +calculate_r_hat(chains) float
        +calculate_ess(chain, max_lag) float
        +calculate_batch_ess(chains, method) float
        +calculate_ess_per_second(chain, time) float
        +calculate_mcse(chain, ess) float
        +check_convergence(chains, metric_names) dict
        +geweke_test(chain) tuple
        +heidelberger_welch_test(chain, alpha) dict
    }

    class ConvergenceStats {
        <<dataclass>>
        +r_hat: float
        +ess: float
        +mcse: float
        +converged: bool
        +n_iterations: int
        +autocorrelation: float
    }

    ConvergenceDiagnostics --> ConvergenceStats : produces
```

**ConvergenceDiagnostics** implements Gelman-Rubin R-hat, Effective Sample Size (ESS), Monte Carlo Standard Error (MCSE), Geweke test, and Heidelberger-Welch stationarity test. The `check_convergence()` method returns a `ConvergenceStats` dataclass for each metric being tracked.

## Loss Modeling

The loss modeling subsystem uses a composite pattern to combine attritional, large, and catastrophic loss generators into a unified manufacturing risk model.

```{mermaid}
classDiagram
    class LossDistribution {
        <<abstract>>
        #rng: Generator
        +generate_severity(n_samples)* ndarray
        +expected_value()* float
        +reset_seed(seed) void
    }

    class LognormalLoss {
        +mean: float
        +cv: float
        +mu: float
        +sigma: float
        +generate_severity(n_samples) ndarray
        +expected_value() float
    }

    class ParetoLoss {
        +alpha: float
        +xm: float
        +generate_severity(n_samples) ndarray
        +expected_value() float
    }

    class GeneralizedParetoLoss {
        +severity_shape: float
        +severity_scale: float
        +generate_severity(n_samples) ndarray
        +expected_value() float
    }

    class LossEvent {
        <<dataclass>>
        +amount: float
        +time: float
        +loss_type: str
        +description: str
    }

    class LossData {
        <<dataclass>>
        +timestamps: ndarray
        +loss_amounts: ndarray
        +loss_types: list~str~
        +claim_ids: list~str~
        +development_factors: ndarray
        +metadata: dict
        +validate() bool
        +to_ergodic_format() ErgodicData
        +apply_insurance(program) LossData
        +from_loss_events(events)$ LossData
        +to_loss_events() list~LossEvent~
        +get_annual_aggregates(years) dict
        +calculate_statistics() dict
    }

    LossDistribution <|-- LognormalLoss
    LossDistribution <|-- ParetoLoss
    LossDistribution <|-- GeneralizedParetoLoss
    LossData --> LossEvent : converts to/from
```

**LossDistribution** is the abstract base class defining the interface for severity distributions. The three concrete implementations (Lognormal, Pareto, Generalized Pareto) cover the full spectrum from attritional to extreme tail modeling.

**LossEvent** is a lightweight dataclass representing a single loss occurrence with timing, amount, and type classification. **LossData** is the unified data container for cross-module compatibility, providing conversion to ergodic format and insurance application methods.

## Loss Generation (Composite Pattern)

The manufacturing loss generator uses the Composite pattern to combine multiple loss layer generators, each with independent frequency and severity models.

```{mermaid}
classDiagram
    class ManufacturingLossGenerator {
        +attritional: AttritionalLossGenerator
        +large: LargeLossGenerator
        +catastrophic: CatastrophicLossGenerator
        +gpd_generator: GeneralizedParetoLoss
        +threshold_value: float
        +exposure: ExposureBase
        +generate_losses(duration, revenue) tuple
        +reseed(seed) void
        +create_simple(frequency, severity_mean, severity_std, seed)$ ManufacturingLossGenerator
        +validate_distributions(n_simulations) dict
    }

    class AttritionalLossGenerator {
        +frequency_generator: FrequencyGenerator
        +severity_distribution: LognormalLoss
        +loss_type: str
        +generate_losses(duration, revenue) list~LossEvent~
        +reseed(seed) void
    }

    class LargeLossGenerator {
        +frequency_generator: FrequencyGenerator
        +severity_distribution: LognormalLoss
        +loss_type: str
        +generate_losses(duration, revenue) list~LossEvent~
        +reseed(seed) void
    }

    class CatastrophicLossGenerator {
        +frequency_generator: FrequencyGenerator
        +severity_distribution: ParetoLoss
        +loss_type: str
        +generate_losses(duration, revenue) list~LossEvent~
        +reseed(seed) void
    }

    class FrequencyGenerator {
        +base_frequency: float
        +revenue_scaling_exponent: float
        +reference_revenue: float
        -rng: Generator
        +reseed(seed) void
        +get_scaled_frequency(revenue) float
        +generate_event_times(duration, revenue) ndarray
    }

    ManufacturingLossGenerator *-- AttritionalLossGenerator : composes
    ManufacturingLossGenerator *-- LargeLossGenerator : composes
    ManufacturingLossGenerator *-- CatastrophicLossGenerator : composes
    ManufacturingLossGenerator o-- GeneralizedParetoLoss : optional extreme
    AttritionalLossGenerator --> FrequencyGenerator : uses
    LargeLossGenerator --> FrequencyGenerator : uses
    CatastrophicLossGenerator --> FrequencyGenerator : uses
    AttritionalLossGenerator --> LognormalLoss : severity
    LargeLossGenerator --> LognormalLoss : severity
    CatastrophicLossGenerator --> ParetoLoss : severity
```

**ManufacturingLossGenerator** is the composite orchestrator that combines three loss layers (attritional, large, catastrophic) with optional GPD extreme value transformation. The `create_simple()` class method provides a migration-friendly factory for basic use cases. Each sub-generator pairs a `FrequencyGenerator` (Poisson process with revenue scaling) with a `LossDistribution` for severities.

## Sensitivity Analysis

Sensitivity tools analyze how parameter changes affect optimization outcomes, with built-in caching for computational efficiency.

```{mermaid}
classDiagram
    class SensitivityAnalyzer {
        -base_config: dict
        -optimizer: Any
        -results_cache: dict
        -cache_dir: Path
        +analyze_parameter(param_name, param_range, n_points) SensitivityResult
        +create_tornado_diagram(parameters, metric) dict
        +analyze_parameter_group(params, metric) dict
    }

    class SensitivityResult {
        <<dataclass>>
        +parameter: str
        +baseline_value: float
        +variations: ndarray
        +metrics: dict
        +parameter_path: str
        +units: str
        +calculate_impact(metric) float
        +get_metric_bounds(metric) tuple
        +to_dataframe() DataFrame
    }

    class TwoWaySensitivityResult {
        <<dataclass>>
        +parameter1: str
        +parameter2: str
        +values1: ndarray
        +values2: ndarray
        +metric_grid: ndarray
        +metric_name: str
        +find_optimal_region(target, tolerance) ndarray
        +to_dataframe() DataFrame
    }

    SensitivityAnalyzer --> SensitivityResult : produces
    SensitivityAnalyzer --> TwoWaySensitivityResult : produces
```

**SensitivityAnalyzer** provides one-way parameter analysis, tornado diagram generation, and parameter group analysis. It uses MD5-based caching to avoid redundant optimizer runs. Results are captured as `SensitivityResult` (one-way) or `TwoWaySensitivityResult` (two-way interaction) dataclasses with built-in DataFrame conversion.

## Financial Statements

The financial statement subsystem generates GAAP-compliant Balance Sheet, Income Statement, and Cash Flow Statement from simulation data, with support for both indirect and direct (ledger-based) cash flow methods.

```{mermaid}
classDiagram
    class FinancialStatementGenerator {
        -manufacturer: WidgetManufacturer
        -manufacturer_data: dict
        -config: FinancialStatementConfig
        -metrics_history: list
        -years_available: int
        -ledger: Ledger
        +generate_balance_sheet(year) DataFrame
        +generate_income_statement(year) DataFrame
        +generate_cash_flow_statement(year) DataFrame
        +generate_reconciliation_report(year) DataFrame
    }

    class CashFlowStatement {
        -metrics_history: list
        -config: Any
        -ledger: Ledger
        +generate_statement(year, period, method) DataFrame
    }

    class FinancialStatementConfig {
        <<dataclass>>
        +currency_symbol: str
        +decimal_places: int
        +include_yoy_change: bool
        +include_percentages: bool
        +fiscal_year_end: int
        +consolidate_monthly: bool
        +current_claims_ratio: float
    }

    FinancialStatementGenerator --> CashFlowStatement : delegates to
    FinancialStatementGenerator --> FinancialStatementConfig : configured by
    FinancialStatementGenerator ..> WidgetManufacturer : reads from
```

**FinancialStatementGenerator** is the primary entry point, accepting a `WidgetManufacturer` (or raw data dictionary) and generating formatted DataFrames for each financial statement. It supports ledger-based direct method cash flow when a `Ledger` is available. The `generate_reconciliation_report()` method validates the accounting equation and solvency checks.

**CashFlowStatement** handles the three-section cash flow statement (Operating, Investing, Financing) with both indirect and direct method support.

## Data Flow Sequence

```{mermaid}
sequenceDiagram
    participant LG as ManufacturingLossGenerator
    participant Sim as Simulation
    participant EA as ErgodicAnalyzer
    participant BO as BusinessOptimizer
    participant SA as SensitivityAnalyzer
    participant RM as RiskMetrics
    participant FS as FinancialStatementGenerator

    LG->>Sim: Generate losses (attritional + large + catastrophic)
    Sim->>EA: Trajectory data (insured & uninsured)
    EA->>EA: Calculate time-average growth
    EA->>EA: Calculate ensemble-average growth
    EA->>RM: Loss data for tail risk
    RM-->>EA: VaR, TVaR, drawdown metrics
    EA-->>BO: Ergodic metrics & analysis results

    BO->>BO: Define objectives & constraints
    BO->>SA: Request parameter sensitivity
    SA->>SA: Parameter sweep with caching
    SA-->>BO: SensitivityResult
    BO->>BO: Find optimal strategy via scipy.optimize
    BO-->>BO: OptimalStrategy

    BO->>FS: Generate financial statements
    FS->>FS: Build balance sheet
    FS->>FS: Build income statement
    FS->>FS: Build cash flow statement
    FS-->>BO: Formatted DataFrames
```

## Key Design Patterns

### 1. **Composite Pattern**
- `ManufacturingLossGenerator` composes `AttritionalLossGenerator`, `LargeLossGenerator`, and `CatastrophicLossGenerator` into a unified interface
- Each sub-generator independently pairs a `FrequencyGenerator` with a `LossDistribution`

### 2. **Template Method (Abstract Base Class)**
- `LossDistribution` (ABC) defines the interface with `generate_severity()` and `expected_value()` as abstract methods
- `LognormalLoss`, `ParetoLoss`, and `GeneralizedParetoLoss` implement distribution-specific behavior

### 3. **Dataclass Data Transfer Objects**
- `ErgodicData`, `ErgodicAnalysisResults`, `OptimalStrategy`, `LossEvent`, `LossData`, `ConvergenceStats`, `RuinProbabilityResults`, `SensitivityResult` all use `@dataclass` for clean data transfer between modules

### 4. **Factory Method**
- `ManufacturingLossGenerator.create_simple()` provides a simplified factory for migration from legacy `ClaimGenerator`
- `LossData.from_loss_events()` constructs data from a list of `LossEvent` objects

### 5. **Strategy Pattern**
- `BusinessOptimizer` supports multiple optimization strategies: ROE maximization, bankruptcy risk minimization, capital efficiency optimization, and multi-objective optimization
- Each strategy uses different objective functions with `scipy.optimize`

### 6. **Caching**
- `SensitivityAnalyzer` uses MD5-based in-memory and persistent disk caching to avoid redundant optimization runs during parameter sweeps
