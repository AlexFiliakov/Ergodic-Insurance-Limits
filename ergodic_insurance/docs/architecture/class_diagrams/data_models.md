# Data Models and Analysis Classes

This diagram shows the data structures and analysis models used throughout the system.

.. mermaid diagram (pre-rendered as SVG)
.. raw:: html

   <div class="mermaid-diagram">
   <img src="/_static/mermaid/data_models_diagram_0_f2a01e09.svg" alt="Diagram 1" style="max-width: 100%; height: auto;">
   </div>

.. code-block:: text
   :class: mermaid-source

   classDiagram
       %% Ergodic Analysis Models
       class ErgodicAnalyzer {
           -convergence_threshold: float
           -min_paths: int
           -confidence_level: float
           +analyze_trajectory(trajectory: ndarray) ErgodicData
           +compare_scenarios(insured: List, uninsured: List) dict
           +calculate_ergodic_metrics(data: ErgodicData) dict
           +validate_insurance_impact() ValidationResults
           +plot_ergodic_comparison()
       }

       class ErgodicData {
           +trajectory: ndarray
           +time_points: ndarray
           +time_average: float
           +ensemble_average: float
           +growth_rate: float
           +volatility: float
           +survival_rate: float
           +calculate_time_average() float
           +calculate_ensemble_average() float
           +calculate_ergodic_divergence() float
       }

       class ErgodicAnalysisResults {
           +insured_metrics: dict
           +uninsured_metrics: dict
           +ergodic_advantage: dict
           +confidence_intervals: dict
           +convergence_metrics: ConvergenceMetrics
           +to_dataframe() DataFrame
           +plot_results()
       }

       %% Optimization Models
       class BusinessOptimizer {
           -objective: BusinessObjective
           -constraints: BusinessConstraints
           -algorithm: OptimizationAlgorithm
           +optimize(initial_guess: dict) OptimalStrategy
           +run_pareto_analysis() ParetoFrontier
           +sensitivity_analysis() SensitivityResult
       }

       class OptimalStrategy {
           +insurance_limit: float
           +retention: float
           +premium_budget: float
           +expected_growth: float
           +risk_metrics: dict
           +implementation_steps: List[str]
       }

       class BusinessObjective {
           +metric: str
           +target_value: float
           +weight: float
           +evaluate(simulation_results: SimulationResults) float
       }

       class BusinessConstraints {
           +min_equity: float
           +max_leverage: float
           +min_liquidity: float
           +max_premium_ratio: float
           +validate(state: dict) bool
       }

       %% Risk Metrics
       class RiskMetrics {
           +value_at_risk: float
           +conditional_value_at_risk: float
           +expected_shortfall: float
           +maximum_drawdown: float
           +sharpe_ratio: float
           +sortino_ratio: float
           +calculate_var(returns: ndarray, confidence: float) float
           +calculate_cvar(returns: ndarray, confidence: float) float
           +calculate_max_drawdown(equity: ndarray) float
       }

       class RuinProbability {
           +threshold: float
           +time_horizon: int
           +probability: float
           +expected_time_to_ruin: float
           +calculate_ruin_prob(trajectories: List) float
           +estimate_recovery_time() float
       }

       %% Convergence and Validation
       class ConvergenceMetrics {
           +mean_estimate: float
           +std_estimate: float
           +confidence_interval: tuple
           +effective_sample_size: int
           +gelman_rubin_stat: float
           +is_converged: bool
           +plot_convergence()
       }

       class ValidationResults {
           +accuracy_metrics: dict
           +statistical_tests: dict
           +edge_cases: List[dict]
           +performance_benchmarks: dict
           +is_valid: bool
           +generate_report() str
       }

       %% Sensitivity Analysis
       class SensitivityAnalyzer {
           -base_params: dict
           -param_ranges: dict
           -n_samples: int
           +run_one_way_analysis(param: str) SensitivityResult
           +run_two_way_analysis(param1: str, param2: str) TwoWaySensitivityResult
           +run_sobol_analysis() SobolIndices
           +plot_tornado_diagram()
       }

       class SensitivityResult {
           +parameter: str
           +values: ndarray
           +outputs: ndarray
           +elasticity: float
           +critical_threshold: float
           +plot()
       }

       %% Financial Statements
       class FinancialStatements {
           +balance_sheet: BalanceSheet
           +income_statement: IncomeStatement
           +cash_flow: CashFlowStatement
           +ratios: FinancialRatios
           +generate_statements(manufacturer: WidgetManufacturer)
           +export_to_excel(path: str)
       }

       class BalanceSheet {
           +assets: dict
           +liabilities: dict
           +equity: dict
           +total_assets: float
           +total_liabilities: float
           +total_equity: float
           +validate_balance() bool
       }

       class IncomeStatement {
           +revenue: float
           +operating_income: float
           +insurance_expense: float
           +tax_expense: float
           +net_income: float
           +ebitda: float
           +calculate_margins() dict
       }

       %% Loss Distributions
       class LossDistribution {
           +distribution_type: str
           +parameters: dict
           +fitted: bool
           +fit(data: ndarray)
           +sample(n: int) ndarray
           +pdf(x: float) float
           +cdf(x: float) float
           +quantile(p: float) float
       }

       class LossData {
           +historical_losses: DataFrame
           +frequency_data: ndarray
           +severity_data: ndarray
           +exposure_base: float
           +clean_data()
           +fit_distributions() dict
           +validate_fit() bool
       }

       %% Relationships
       ErgodicAnalyzer --> ErgodicData : creates
       ErgodicAnalyzer --> ErgodicAnalysisResults : produces
       ErgodicAnalyzer --> ValidationResults : validates with

       BusinessOptimizer --> OptimalStrategy : finds
       BusinessOptimizer --> BusinessObjective : uses
       BusinessOptimizer --> BusinessConstraints : respects

       OptimalStrategy --> RiskMetrics : includes

       SensitivityAnalyzer --> SensitivityResult : produces

       FinancialStatements --> BalanceSheet : contains
       FinancialStatements --> IncomeStatement : contains

       LossDistribution --> LossData : fitted from

       ErgodicAnalysisResults --> ConvergenceMetrics : includes
       ValidationResults --> RiskMetrics : uses

## Data Flow Sequence

.. mermaid diagram (pre-rendered as SVG)
.. raw:: html

   <div class="mermaid-diagram">
   <img src="/_static/mermaid/data_models_diagram_1_5534d5ca.svg" alt="Diagram 2" style="max-width: 100%; height: auto;">
   </div>

.. code-block:: text
   :class: mermaid-source

   sequenceDiagram
       participant Sim as Simulation
       participant EA as ErgodicAnalyzer
       participant BO as BusinessOptimizer
       participant SA as SensitivityAnalyzer
       participant RM as RiskMetrics
       participant FS as FinancialStatements

       Sim->>EA: Trajectory data
       EA->>EA: Calculate time averages
       EA->>EA: Calculate ensemble averages
       EA->>RM: Request risk metrics
       RM-->>EA: VaR, CVaR, Sharpe
       EA-->>BO: Ergodic metrics

       BO->>BO: Define objective
       BO->>BO: Set constraints
       BO->>SA: Request sensitivity
       SA->>SA: Parameter sweep
       SA-->>BO: Sensitivity results
       BO-->>BO: Find optimal strategy

       BO->>FS: Generate statements
       FS->>FS: Build balance sheet
       FS->>FS: Build income statement
       FS-->>BO: Financial reports

## Key Data Patterns

### 1. **Immutable Data Objects**
- Results objects are immutable after creation
- Ensures data integrity through analysis pipeline

### 2. **Lazy Evaluation**
- Metrics calculated on-demand
- Caching of expensive computations

### 3. **Composite Pattern**
- FinancialStatements composed of multiple statement types
- ErgodicAnalysisResults aggregates multiple metric types

### 4. **Template Method**
- Base distribution class with template methods
- Subclasses implement specific distributions

### 5. **Data Transfer Objects (DTO)**
- Result classes act as DTOs between modules
- Clean separation of data and logic
