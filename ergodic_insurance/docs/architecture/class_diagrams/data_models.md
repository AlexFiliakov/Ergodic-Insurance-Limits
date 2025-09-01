# Data Models and Configuration Classes

(configuration-data-models)=
## Configuration Data Models

```mermaid
classDiagram
    class Config {
        +ManufacturerConfig manufacturer
        +SimulationConfig simulation
        +WorkingCapitalConfig working_capital
        +DebtConfig debt
        +GrowthConfig growth
        +OutputConfig output
        +LoggingConfig logging
        +validate() bool
        +to_yaml(path)
        +from_yaml(path) Config
    }

    class ManufacturerConfig {
        +float initial_assets
        +float asset_turnover_mean
        +float asset_turnover_std
        +float operating_margin_mean
        +float operating_margin_std
        +float tax_rate
        +float dividend_payout_ratio
        +float max_debt_to_equity
        +float interest_rate
    }

    class SimulationConfig {
        +int n_years
        +int n_simulations
        +float time_step
        +int seed
        +bool use_stochastic
        +float volatility
        +bool save_trajectories
        +str checkpoint_dir
    }

    class WorkingCapitalConfig {
        +float base_ratio
        +float safety_stock_days
        +float receivables_days
        +float payables_days
        +float inventory_turnover
        +calculate_requirement(revenue) float
    }

    class DebtConfig {
        +float initial_debt
        +float interest_rate
        +float term_years
        +str repayment_type
        +float covenant_debt_to_equity
        +float covenant_coverage_ratio
    }

    class GrowthConfig {
        +float base_growth_rate
        +float volatility
        +float mean_reversion_speed
        +float long_term_mean
        +str growth_model
    }

    Config --> ManufacturerConfig : contains
    Config --> SimulationConfig : contains
    Config --> WorkingCapitalConfig : contains
    Config --> DebtConfig : contains
    Config --> GrowthConfig : contains
```

## ConfigV2 Enhanced Models

```mermaid
classDiagram
    class ConfigV2 {
        +ProfileMetadata metadata
        +Dict~str,ModuleConfig~ modules
        +PresetConfig preset
        +InsuranceConfig insurance
        +LossDistributionConfig losses
        +validate_consistency() bool
        +merge_with_preset(preset) ConfigV2
        +export_profile() dict
    }

    class ProfileMetadata {
        +str name
        +str version
        +str description
        +datetime created_at
        +datetime updated_at
        +List~str~ tags
        +str author
        +dict compatibility
    }

    class InsuranceLayerConfig {
        +float limit
        +float attachment
        +float premium_rate
        +int reinstatements
        +float reinstatement_cost
        +str coverage_type
        +dict exclusions
    }

    class InsuranceConfig {
        +List~InsuranceLayerConfig~ layers
        +float retention
        +float aggregate_limit
        +str program_type
        +dict optimization_params
        +calculate_total_limit() float
        +optimize_structure() List~InsuranceLayerConfig~
    }

    class LossDistributionConfig {
        +str frequency_distribution
        +dict frequency_params
        +str severity_distribution
        +dict severity_params
        +float correlation
        +List~str~ peril_types
        +dict calibration_data
    }

    class ModuleConfig {
        +str module_type
        +dict parameters
        +bool enabled
        +int priority
        +List~str~ dependencies
    }

    class PresetConfig {
        +str base_preset
        +dict overrides
        +List~str~ included_modules
        +List~str~ excluded_modules
        +merge(other) PresetConfig
    }

    ConfigV2 --> ProfileMetadata : has
    ConfigV2 --> InsuranceConfig : contains
    ConfigV2 --> LossDistributionConfig : contains
    ConfigV2 --> ModuleConfig : contains many
    ConfigV2 --> PresetConfig : uses
    InsuranceConfig --> InsuranceLayerConfig : contains many
```

(result-data-models)=
## Result Data Models

```mermaid
classDiagram
    class SimulationResults {
        +ndarray timestamps
        +ndarray asset_values
        +ndarray revenues
        +ndarray claims
        +ndarray insurance_recoveries
        +dict metrics
        +dict metadata
        +to_dataframe() pd.DataFrame
        +to_dict() dict
        +save(path)
        +load(path) SimulationResults
    }

    class BatchResult {
        +str batch_id
        +ProcessingStatus status
        +List~SimulationResults~ results
        +datetime start_time
        +datetime end_time
        +dict performance_metrics
        +aggregate() AggregatedResults
    }

    class AggregatedResults {
        +Dict~str,ndarray~ percentiles
        +Dict~str,float~ mean_metrics
        +Dict~str,float~ std_metrics
        +Dict~str,float~ risk_metrics
        +pd.DataFrame summary_table
        +export_summary() dict
    }

    class RiskMetricsResult {
        +float var_95
        +float var_99
        +float cvar_95
        +float cvar_99
        +float sharpe_ratio
        +float sortino_ratio
        +float max_drawdown
        +dict tail_statistics
    }

    class BootstrapResult {
        +ndarray samples
        +float mean
        +float std
        +Tuple~float,float~ confidence_interval
        +float bias
        +float standard_error
        +plot_distribution()
    }

    class StatisticalSummary {
        +float mean
        +float median
        +float std
        +float skewness
        +float kurtosis
        +Dict~int,float~ percentiles
        +float min
        +float max
        +to_series() pd.Series
    }

    BatchResult --> SimulationResults : contains many
    BatchResult --> AggregatedResults : produces
    AggregatedResults --> RiskMetricsResult : includes
    AggregatedResults --> StatisticalSummary : uses
    SimulationResults --> StatisticalSummary : generates
```

## Analysis Result Models

```mermaid
classDiagram
    class ErgodicAnalysisResult {
        +float time_average_growth
        +float ensemble_average_growth
        +float ergodic_difference
        +ndarray time_averages
        +ndarray ensemble_averages
        +dict convergence_metrics
        +bool is_ergodic
        +plot_comparison()
    }

    class OptimalStrategy {
        +Dict~str,float~ parameters
        +float expected_return
        +float risk_level
        +InsuranceProgram recommended_insurance
        +dict sensitivity_analysis
        +str reasoning
        +to_recommendation() str
    }

    class DecisionMetrics {
        +float expected_value
        +float downside_risk
        +float upside_potential
        +float information_ratio
        +Dict~str,float~ scenario_outcomes
        +rank_alternatives() List~Tuple~
    }

    class ParetoPoint {
        +List~float~ objectives
        +Dict~str,Any~ parameters
        +int rank
        +float crowding_distance
        +bool is_dominated_by(other) bool
        +distance_to(ideal) float
    }

    class HJBSolution {
        +ndarray value_function
        +ndarray optimal_control
        +ndarray state_grid
        +float convergence_error
        +int iterations
        +plot_value_function()
        +get_control_at_state(state) float
    }

    class ConvergenceStats {
        +List~float~ running_mean
        +List~float~ running_std
        +float gelman_rubin_stat
        +float effective_sample_size
        +bool has_converged
        +int burn_in_period
        +plot_diagnostics()
    }

    OptimalStrategy --> DecisionMetrics : based on
    OptimalStrategy --> InsuranceProgram : recommends
    ParetoPoint --> DecisionMetrics : evaluates
    ErgodicAnalysisResult --> ConvergenceStats : uses
    HJBSolution --> OptimalStrategy : informs
```

(state-and-progress-models)=
## State and Progress Models

```mermaid
classDiagram
    class SimulationState {
        +int current_period
        +float current_assets
        +float current_equity
        +List~ClaimLiability~ pending_claims
        +InsuranceProgramState insurance_state
        +dict custom_state
        +checkpoint() bytes
        +restore(data)
    }

    class InsuranceProgramState {
        +Dict~str,LayerState~ layer_states
        +float ytd_losses
        +float ytd_recoveries
        +float ytd_premiums
        +List~ClaimEvent~ claim_history
        +reset_annual()
        +get_summary() dict
    }

    class LayerState {
        +str layer_id
        +float remaining_limit
        +float remaining_aggregate
        +int reinstatements_used
        +float premium_paid
        +bool is_exhausted
        +update(loss) float
    }

    class ProgressStats {
        +int total_tasks
        +int completed_tasks
        +float progress_percentage
        +datetime start_time
        +datetime estimated_completion
        +float tasks_per_second
        +str current_task
        +update(completed)
        +get_eta() timedelta
    }

    class CheckpointData {
        +str simulation_id
        +SimulationState state
        +int iteration
        +datetime timestamp
        +dict metadata
        +save(path)
        +load(path) CheckpointData
    }

    class ScenarioConfig {
        +str name
        +ScenarioType type
        +Dict~str,ParameterSpec~ parameters
        +List~str~ dependencies
        +int priority
        +generate_variations() List~dict~
        +validate_parameters() bool
    }

    SimulationState --> InsuranceProgramState : contains
    InsuranceProgramState --> LayerState : manages many
    CheckpointData --> SimulationState : stores
    ScenarioConfig --> ParameterSpec : defines
```

## Data Transfer Objects

```mermaid
classDiagram
    class ClaimRequest {
        +float amount
        +datetime occurrence_date
        +str claim_type
        +str policy_number
        +dict supporting_docs
        +validate() bool
    }

    class ClaimResult {
        +ClaimRequest request
        +float gross_loss
        +float deductible
        +float recovery
        +float net_loss
        +Dict~str,float~ layer_recoveries
        +datetime processed_date
    }

    class YearResult {
        +int year
        +float starting_assets
        +float ending_assets
        +float revenue
        +float operating_income
        +float net_income
        +List~ClaimResult~ claims
        +float total_recovery
        +float insurance_premium
    }

    class MonteCarloRequest {
        +int n_simulations
        +int n_periods
        +Config config
        +bool parallel
        +int batch_size
        +str output_format
    }

    class OptimizationRequest {
        +List~str~ objectives
        +Dict~str,Tuple~ constraints
        +str algorithm
        +int max_iterations
        +float tolerance
        +dict initial_guess
    }

    class ReportRequest {
        +str report_type
        +List~str~ metrics
        +str format
        +bool include_charts
        +dict filters
        +str output_path
    }

    ClaimResult --> ClaimRequest : processes
    YearResult --> ClaimResult : contains many
    MonteCarloRequest --> Config : uses
    OptimizationRequest --> OptimalStrategy : produces
    ReportRequest --> AggregatedResults : generates from
```

## Key Data Model Patterns

1. **Configuration Hierarchy**: Nested configuration models with validation at each level
2. **Result Aggregation**: Hierarchical results from individual simulations to batch aggregations
3. **State Management**: Comprehensive state tracking for checkpointing and recovery
4. **Request-Response**: Clean DTOs for API boundaries and service interactions
5. **Immutable Results**: Result objects are designed to be immutable after creation
6. **Serialization**: All data models support serialization to/from standard formats (JSON, YAML, HDF5)
