# Service Layer Architecture

## Overview
This document details the service layer components that provide analysis, decision-making, and visualization capabilities.

## Analysis Services

```mermaid
classDiagram
    class ErgodicAnalyzer {
        -float convergence_threshold
        +__init__(convergence_threshold)
        +calculate_time_average(values) float
        +calculate_ensemble_average(paths) float
        +calculate_growth_rate(trajectory) float
        +compare_averages(simulation_results) Dict
        +analyze_insurance_impact(with_insurance, without_insurance) Dict
        +validate_ergodicity(results) bool
        +plot_ergodic_comparison(results)
    }

    class RiskAnalyzer {
        -Dict risk_parameters
        +__init__(risk_parameters)
        +calculate_var(data, confidence) float
        +calculate_tvar(data, confidence) float
        +calculate_expected_shortfall(data, confidence) float
        +calculate_maximum_drawdown(trajectory) float
        +calculate_ruin_probability(paths, threshold) float
        +calculate_sharpe_ratio(returns, risk_free_rate) float
        +calculate_sortino_ratio(returns, mar) float
        +comprehensive_risk_report(results) RiskMetrics
    }

    class ConvergenceAnalyzer {
        -float tolerance
        -int min_samples
        +__init__(tolerance, min_samples)
        +test_convergence(data) ConvergenceResults
        +calculate_standard_error(data) float
        +gelman_rubin_statistic(chains) float
        +effective_sample_size(data) int
        +plot_convergence_diagnostic(data)
    }

    class SensitivityAnalyzer {
        -Dict baseline_params
        -List~str~ parameters_to_vary
        +__init__(baseline_params, parameters_to_vary)
        +run_sensitivity_analysis(simulation, param_ranges) Dict
        +calculate_elasticity(param, results) float
        +tornado_diagram(sensitivities)
        +heatmap_2d_sensitivity(param1, param2, results)
    }
```

## Decision Engine

```mermaid
classDiagram
    class DecisionEngine {
        -ErgodicAnalyzer ergodic_analyzer
        -RiskAnalyzer risk_analyzer
        -Dict optimization_parameters
        +__init__(ergodic_analyzer, risk_analyzer, optimization_params)
        +optimize_insurance_program(manufacturer, constraints) OptimizationResults
        +evaluate_strategy(strategy, simulations) StrategyEvaluation
        +compare_strategies(strategies) ComparisonResults
        +recommend_action(current_state, market_conditions) Recommendation
    }

    class OptimizationEngine {
        -str optimization_method
        -Dict constraints
        -callable objective_function
        +__init__(method, constraints, objective)
        +optimize() OptimalSolution
        +grid_search(param_grid) GridSearchResults
        +bayesian_optimization(n_iterations) BayesianResults
        +genetic_algorithm(population_size, generations) GAResults
    }

    class StrategyEvaluator {
        +evaluate_fixed_retention(retention, simulations) Dict
        +evaluate_dynamic_retention(retention_function, simulations) Dict
        +evaluate_layer_structure(layers, simulations) Dict
        +rank_strategies(evaluations) List~Tuple~
    }

    class RecommendationEngine {
        -DecisionEngine decision_engine
        -Dict business_rules
        +generate_recommendation(analysis_results) Recommendation
        +explain_recommendation(recommendation) str
        +confidence_score(recommendation) float
    }

    DecisionEngine --> ErgodicAnalyzer: uses
    DecisionEngine --> RiskAnalyzer: uses
    DecisionEngine --> OptimizationEngine: uses
    DecisionEngine --> StrategyEvaluator: uses
    RecommendationEngine --> DecisionEngine: uses
```

## Visualization Services

```mermaid
classDiagram
    class VisualizationService {
        -Dict plot_config
        +__init__(plot_config)
        +plot_trajectory(results, metrics) Figure
        +plot_distribution(data, type) Figure
        +plot_comparison(datasets, labels) Figure
        +create_dashboard(results) Dashboard
    }

    class TrajectoryPlotter {
        +plot_assets_evolution(results)
        +plot_equity_evolution(results)
        +plot_roe_timeline(results)
        +plot_claims_timeline(results)
        +plot_multiple_paths(monte_carlo_results)
    }

    class DistributionPlotter {
        +plot_histogram(data, bins)
        +plot_density(data, kernel)
        +plot_qq(data, distribution)
        +plot_ecdf(data)
        +plot_violin(groups)
    }

    class ComparisonPlotter {
        +plot_ergodic_comparison(time_avg, ensemble_avg)
        +plot_strategy_comparison(strategies)
        +plot_sensitivity_tornado(sensitivities)
        +plot_efficient_frontier(risk, return)
    }

    class InteractiveDashboard {
        -List~Component~ components
        +add_component(component)
        +update_data(new_data)
        +export_html(path)
        +serve(port)
    }

    VisualizationService --> TrajectoryPlotter: uses
    VisualizationService --> DistributionPlotter: uses
    VisualizationService --> ComparisonPlotter: uses
    VisualizationService --> InteractiveDashboard: creates
```

## Integration Layer

```mermaid
sequenceDiagram
    participant User
    participant API as Service API
    participant Engine as Decision Engine
    participant Analyzer as Analysis Services
    participant Sim as Simulation
    participant Viz as Visualization

    User->>API: Request optimization
    API->>Engine: Initialize decision engine

    Engine->>Sim: Run baseline simulation
    Sim-->>Engine: Baseline results

    Engine->>Analyzer: Analyze baseline
    Analyzer-->>Engine: Risk metrics

    loop Optimization iterations
        Engine->>Engine: Generate candidate
        Engine->>Sim: Test candidate
        Sim-->>Engine: Candidate results
        Engine->>Analyzer: Evaluate candidate
        Analyzer-->>Engine: Candidate metrics
    end

    Engine->>Engine: Select optimal
    Engine->>Viz: Prepare visualizations
    Viz-->>Engine: Charts and reports

    Engine-->>API: Optimization results
    API-->>User: Results + visualizations
```

## Service Orchestration

```mermaid
graph TB
    subgraph Input["Input Layer"]
        Config[Configuration]
        Data[Historical Data]
        Constraints[Business Constraints]
    end

    subgraph Services["Service Layer"]
        Decision[Decision Engine]
        Ergodic[Ergodic Analyzer]
        Risk[Risk Analyzer]
        Sensitivity[Sensitivity Analyzer]
        Convergence[Convergence Analyzer]
    end

    subgraph Core["Core Simulation"]
        Simulation[Simulation Engine]
        MonteCarlo[Monte Carlo]
    end

    subgraph Output["Output Layer"]
        Reports[Report Generator]
        Viz[Visualization Service]
        Export[Data Exporter]
    end

    Config --> Decision
    Data --> Risk
    Constraints --> Decision

    Decision --> Simulation
    Decision --> Ergodic
    Decision --> Risk
    Decision --> Sensitivity

    Simulation --> MonteCarlo
    MonteCarlo --> Convergence

    Ergodic --> Reports
    Risk --> Reports
    Sensitivity --> Viz
    Convergence --> Viz

    Reports --> Export
    Viz --> Export

    style Services fill:#e0f2f1
    style Core fill:#fff3e0
    style Output fill:#f3e5f5
    style Input fill:#e8f5e9
```

## Complete Service Architecture

```mermaid
classDiagram
    class ServiceRegistry {
        -Dict~str_Service~ services
        +register(name, service)
        +get(name) Service
        +list_services() List~str~
        +health_check() Dict
    }

    class ServiceBase {
        <<abstract>>
        +str name
        +str version
        +Dict config
        +initialize()
        +shutdown()
        +health_check() bool
    }

    class AnalysisService {
        +analyze(data) Results
        +validate(results) bool
    }

    class ComputeService {
        +compute(inputs) Outputs
        +estimate_runtime(inputs) float
    }

    class DataService {
        +load(source) Data
        +save(data, destination)
        +transform(data) Data
    }

    ServiceBase <|-- AnalysisService: inherits
    ServiceBase <|-- ComputeService: inherits
    ServiceBase <|-- DataService: inherits

    ServiceRegistry --> ServiceBase: manages

    ErgodicAnalyzer --|> AnalysisService: implements
    RiskAnalyzer --|> AnalysisService: implements
    DecisionEngine --|> ComputeService: implements
    VisualizationService --|> DataService: implements
```
