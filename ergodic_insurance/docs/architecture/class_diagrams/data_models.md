# Data Models and Configuration Architecture

## Overview
This document details the data models, configuration structures, and their relationships within the Ergodic Insurance system.

## Configuration Models (Pydantic)

```mermaid
classDiagram
    class BaseModel {
        <<pydantic>>
        +model_validate()
        +model_dump()
        +model_json_schema()
    }

    class Config {
        +ManufacturerConfig manufacturer
        +WorkingCapitalConfig working_capital
        +DebtConfig debt
        +GrowthConfig growth
        +SimulationConfig simulation
        +OutputConfig output
        +LoggingConfig logging
        +validate_config()
        +from_yaml(path) Config
        +to_yaml(path)
    }

    class ManufacturerConfig {
        +float initial_assets
        +float asset_turnover
        +float operating_margin
        +float tax_rate
        +float depreciation_rate
        +float capex_rate
        +float dividend_payout_ratio
        +float min_cash
        +validate_positive_values()
        +validate_ratios()
    }

    class WorkingCapitalConfig {
        +float receivables_days
        +float inventory_days
        +float payables_days
        +float cash_conversion_cycle
    }

    class DebtConfig {
        +float max_debt_to_equity
        +float target_debt_to_equity
        +float interest_rate
        +float min_interest_coverage
        +float emergency_credit_rate
    }

    class SimulationConfig {
        +int years
        +int n_simulations
        +Optional~int~ seed
        +bool use_parallel
        +int n_workers
        +str checkpoint_dir
        +int checkpoint_frequency
        +validate_simulation_params()
    }

    class GrowthConfig {
        +float base_growth_rate
        +float growth_volatility
        +str growth_type
    }

    class OutputConfig {
        +str output_dir
        +bool save_trajectories
        +bool save_checkpoints
        +int plot_frequency
    }

    class LoggingConfig {
        +str level
        +str format
        +bool console
        +bool file
        +str file_path
    }

    Config --|> BaseModel: inherits
    ManufacturerConfig --|> BaseModel: inherits
    WorkingCapitalConfig --|> BaseModel: inherits
    DebtConfig --|> BaseModel: inherits
    GrowthConfig --|> BaseModel: inherits
    SimulationConfig --|> BaseModel: inherits
    OutputConfig --|> BaseModel: inherits
    LoggingConfig --|> BaseModel: inherits

    Config --> ManufacturerConfig: contains
    Config --> WorkingCapitalConfig: contains
    Config --> DebtConfig: contains
    Config --> GrowthConfig: contains
    Config --> SimulationConfig: contains
    Config --> OutputConfig: contains
    Config --> LoggingConfig: contains
```

## Result Data Models

```mermaid
classDiagram
    class SimulationResults {
        +ndarray years
        +ndarray assets
        +ndarray equity
        +ndarray roe
        +ndarray revenue
        +ndarray net_income
        +ndarray claim_counts
        +ndarray claim_amounts
        +Optional~int~ insolvency_year
        +to_dataframe() DataFrame
        +to_dict() Dict
        +to_csv(path)
        +plot_trajectories()
        +summary_stats() Dict
    }

    class MonteCarloResults {
        +List~SimulationResults~ paths
        +Dict ensemble_stats
        +Dict ergodic_stats
        +float convergence_metric
        +to_dataframe() DataFrame
        +aggregate_paths() DataFrame
        +get_percentiles(percentiles) Dict
        +calculate_var(confidence) float
        +calculate_tvar(confidence) float
        +plot_distribution()
    }

    class ErgodicAnalysisResults {
        +float time_average_growth
        +float ensemble_average_growth
        +float survival_rate
        +float ergodic_divergence
        +Dict insurance_impact
        +bool validation_passed
        +Dict metadata
        +to_dict() Dict
        +plot_comparison()
    }

    class RiskMetrics {
        +float var_95
        +float var_99
        +float tvar_95
        +float tvar_99
        +float expected_shortfall
        +float maximum_drawdown
        +float ruin_probability
        +float sharpe_ratio
        +float sortino_ratio
        +to_dict() Dict
        +to_dataframe() DataFrame
    }

    class OptimizationResults {
        +InsuranceProgram optimal_program
        +float optimal_retention
        +Dict~str_float~ optimal_limits
        +float expected_cost
        +float expected_utility
        +RiskMetrics risk_metrics
        +List~Dict~ iteration_history
        +plot_efficient_frontier()
        +sensitivity_analysis() Dict
    }

    MonteCarloResults --> SimulationResults: aggregates
    OptimizationResults --> RiskMetrics: contains
    OptimizationResults --> InsuranceProgram: references
```

## Event and State Models

```mermaid
classDiagram
    class ClaimEvent {
        +float amount
        +int year
        +str claim_type
        +Optional~str~ severity_class
        +Optional~Dict~ metadata
        +to_dict() Dict
        +is_catastrophic() bool
        +is_attritional() bool
    }

    class ManufacturerState {
        +float assets
        +float equity
        +float revenue
        +float operating_income
        +float net_income
        +float collateral
        +float restricted_assets
        +List~ClaimLiability~ outstanding_claims
        +int year
        +to_dict() Dict
        +is_solvent() bool
        +liquidity_ratio() float
    }

    class MarketState {
        +float interest_rate
        +float inflation_rate
        +float gdp_growth
        +float market_volatility
        +Dict~str_float~ sector_indices
        +update(dt) MarketState
    }

    class InsuranceState {
        +Dict~str_float~ layer_utilization
        +float total_recoveries
        +float total_premiums_paid
        +float ytd_losses
        +List~Dict~ claim_history
        +reset_annual()
        +update_claim(claim_event, recovery)
    }
```

## Data Flow Relationships

```mermaid
graph LR
    subgraph Input["Input Data"]
        YAML[YAML Files]
        CSV[CSV Data]
    end

    subgraph Config["Configuration Layer"]
        ConfigLoader[ConfigLoader]
        ConfigModels[Pydantic Models]
    end

    subgraph Processing["Processing"]
        Simulation[Simulation Engine]
        Analysis[Analysis Tools]
    end

    subgraph Output["Output Data"]
        Results[Result Models]
        DataFrames[Pandas DataFrames]
        JSON[JSON Export]
    end

    YAML --> ConfigLoader
    CSV --> ConfigLoader
    ConfigLoader --> ConfigModels
    ConfigModels --> Simulation
    Simulation --> Results
    Results --> Analysis
    Analysis --> Results
    Results --> DataFrames
    Results --> JSON

    style Input fill:#e8f5e9
    style Config fill:#e3f2fd
    style Processing fill:#fff3e0
    style Output fill:#fce4ec
```

## Configuration Loading Pattern

```mermaid
sequenceDiagram
    participant User
    participant Loader as ConfigLoader
    participant YAML as YAML File
    participant Validator as Pydantic
    participant Config as Config Object

    User->>Loader: load_config("parameters/baseline.yaml")
    Loader->>YAML: Read file
    YAML-->>Loader: Raw data
    Loader->>Validator: Parse and validate

    alt Validation Success
        Validator-->>Loader: Valid config dict
        Loader->>Config: Create Config instance
        Config-->>Loader: Config object
        Loader-->>User: Config object
    else Validation Error
        Validator-->>Loader: ValidationError
        Loader-->>User: Error with details
    end

    User->>Config: Access parameters
    Config-->>User: Typed attributes
```

## Data Persistence

```mermaid
classDiagram
    class CheckpointManager {
        -str checkpoint_dir
        -int checkpoint_frequency
        +save_checkpoint(year, state, results)
        +load_checkpoint(checkpoint_id) Tuple
        +list_checkpoints() List
        +cleanup_old_checkpoints(keep_last_n)
    }

    class ResultsExporter {
        +export_to_csv(results, path)
        +export_to_excel(results, path)
        +export_to_json(results, path)
        +export_to_parquet(results, path)
        +create_report(results, template) str
    }

    class DataCache {
        -Dict cache_data
        -int max_size
        -float ttl
        +get(key) Optional~Any~
        +set(key, value)
        +invalidate(key)
        +clear()
    }
```
