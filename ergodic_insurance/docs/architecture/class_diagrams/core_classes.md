# Core Classes Diagram

(financial-core-classes)=
## Financial Core Classes

```mermaid
classDiagram
    class WidgetManufacturer {
        -Config config
        -float assets
        -float debt
        -float retained_earnings
        -List~ClaimLiability~ pending_claims
        -float last_revenue
        -float last_operating_income
        +__init__(config)
        +simulate_year(revenue_shock, claims)
        +get_balance_sheet() BalanceSheet
        +get_income_statement() IncomeStatement
        +apply_insurance_recovery(recovery)
        +calculate_working_capital() float
        +calculate_tax(income) float
        +calculate_roe() float
        +reset()
    }

    class ClaimLiability {
        +float amount
        +int year_occurred
        +float paid_to_date
        +List~float~ payment_pattern
        +__init__(amount, year, pattern)
        +get_payment(current_year) float
        +is_settled() bool
    }

    class ClaimEvent {
        +float amount
        +int period
        +str type
        +datetime timestamp
        +__init__(amount, period, type)
        +to_dict() dict
    }

    class ClaimGenerator {
        -Config config
        -np.random.RandomState rng
        -float frequency
        -dict severity_params
        +__init__(config, seed)
        +generate(period) List~ClaimEvent~
        +_sample_frequency() int
        +_sample_severity() float
        +set_seed(seed)
    }

    class ClaimDevelopment {
        -str pattern_type
        -List~float~ factors
        -float tail_factor
        -int max_periods
        +__init__(pattern_type, factors)
        +develop_claim(claim, periods) List~float~
        +get_ultimate_loss(paid) float
        +get_ibnr(claims) float
        +create_payment_schedule(amount) List~float~
    }

    class StochasticProcess {
        <<abstract>>
        #float dt
        #np.random.RandomState rng
        +simulate(t, n_paths) ndarray
        +get_parameters() dict
        +set_seed(seed)
    }

    class GeometricBrownianMotion {
        -float mu
        -float sigma
        -float S0
        +__init__(mu, sigma, S0, dt, seed)
        +simulate(t, n_paths) ndarray
        +_simulate_path(t) ndarray
    }

    class LognormalVolatility {
        -float volatility
        -float mean
        +__init__(volatility, mean, seed)
        +simulate(t, n_paths) ndarray
        +generate_shock() float
    }

    class MeanRevertingProcess {
        -float theta
        -float mu
        -float sigma
        -float X0
        +__init__(theta, mu, sigma, X0, dt, seed)
        +simulate(t, n_paths) ndarray
        +_ou_step(X) float
    }

    WidgetManufacturer --> ClaimLiability : manages
    WidgetManufacturer --> ClaimGenerator : uses
    WidgetManufacturer --> ClaimDevelopment : uses
    ClaimGenerator --> ClaimEvent : creates
    ClaimDevelopment --> ClaimLiability : develops
    StochasticProcess <|-- GeometricBrownianMotion : inherits
    StochasticProcess <|-- LognormalVolatility : inherits
    StochasticProcess <|-- MeanRevertingProcess : inherits
    WidgetManufacturer ..> StochasticProcess : uses for shocks
```

(insurance-classes)=
(insurance-classes)=
## Insurance Classes

```mermaid
classDiagram
    class InsuranceLayer {
        +float limit
        +float attachment
        +float premium_rate
        +float reinstatement_premium
        +int reinstatements
        +__init__(limit, attachment, rate)
        +calculate_recovery(loss) float
        +calculate_premium(exposure) float
        +is_exhausted() bool
        +reset()
    }

    class InsurancePolicy {
        -List~InsuranceLayer~ layers
        -float total_limit
        -float total_premium
        +__init__(layers)
        +calculate_total_recovery(loss) float
        +calculate_total_premium(exposure) float
        +get_layer_recoveries(loss) Dict
        +optimize_structure(constraints) List~InsuranceLayer~
        +reset_all_layers()
    }

    class EnhancedInsuranceLayer {
        +float aggregate_limit
        +float aggregate_attachment
        +ReinstatementType reinstatement_type
        +bool drop_down
        +float participation
        +__init__(limit, attachment, **kwargs)
        +apply_aggregate_limits(loss) float
        +calculate_reinstatement_premium(loss) float
        +handle_drop_down(lower_exhausted) float
    }

    class LayerState {
        +float remaining_limit
        +float remaining_aggregate
        +int reinstatements_used
        +float premium_paid
        +List~float~ loss_history
        +__init__(layer)
        +update(loss) float
        +can_respond(loss) bool
        +get_statistics() dict
    }

    class InsuranceProgram {
        -List~EnhancedInsuranceLayer~ layers
        -Dict~str,LayerState~ layer_states
        -ProgramState program_state
        +__init__(layers)
        +process_claim(loss) ClaimResult
        +optimize_program(objectives) OptimalStructure
        +simulate_year(losses) YearResult
        +get_program_statistics() dict
        +reset()
    }

    class ProgramState {
        +float total_losses
        +float total_recoveries
        +float total_premium
        +List~ClaimResult~ claim_history
        +__init__()
        +update(claim_result)
        +get_loss_ratio() float
        +get_efficiency() float
    }

    InsurancePolicy --> InsuranceLayer : contains
    InsuranceProgram --> EnhancedInsuranceLayer : contains
    InsuranceProgram --> LayerState : tracks
    InsuranceProgram --> ProgramState : maintains
    EnhancedInsuranceLayer --|> InsuranceLayer : extends
    LayerState --> EnhancedInsuranceLayer : references
```

(loss-distribution-classes)=
(loss-distribution-classes)=
## Loss Distribution Classes

```mermaid
classDiagram
    class LossDistribution {
        <<abstract>>
        #dict parameters
        #np.random.RandomState rng
        +sample(size) ndarray
        +pdf(x) ndarray
        +cdf(x) ndarray
        +mean() float
        +variance() float
        +set_seed(seed)
    }

    class LognormalLoss {
        -float mu
        -float sigma
        +__init__(mu, sigma, seed)
        +sample(size) ndarray
        +pdf(x) ndarray
        +cdf(x) ndarray
        +fit_parameters(data) dict
    }

    class ParetoLoss {
        -float alpha
        -float xm
        +__init__(alpha, xm, seed)
        +sample(size) ndarray
        +pdf(x) ndarray
        +cdf(x) ndarray
        +tail_index() float
    }

    class FrequencyGenerator {
        -str distribution_type
        -dict parameters
        -np.random.RandomState rng
        +__init__(dist_type, params, seed)
        +generate(periods) ndarray
        +expected_frequency() float
        +variance() float
    }

    class LossEvent {
        +float amount
        +datetime timestamp
        +str category
        +str description
        +__init__(amount, timestamp, category)
        +is_large_loss() bool
        +is_catastrophic() bool
    }

    class LossData {
        -List~LossEvent~ events
        -pd.DataFrame df
        +__init__(events)
        +add_event(event)
        +get_statistics() dict
        +fit_distribution(dist_type) LossDistribution
        +bootstrap_sample(n) List~LossEvent~
        +to_dataframe() pd.DataFrame
    }

    class ManufacturingLossGenerator {
        -FrequencyGenerator freq_gen
        -LossDistribution severity_dist
        -float correlation
        +__init__(freq_params, sev_params, correlation)
        +generate_annual_losses() List~LossEvent~
        +generate_correlated_losses(base_losses) List~LossEvent~
        +calibrate_to_data(historical_data) dict
    }

    LossDistribution <|-- LognormalLoss : inherits
    LossDistribution <|-- ParetoLoss : inherits
    ManufacturingLossGenerator --> FrequencyGenerator : uses
    ManufacturingLossGenerator --> LossDistribution : uses
    ManufacturingLossGenerator --> LossEvent : creates
    LossData --> LossEvent : contains
    LossData ..> LossDistribution : fits
```

(simulation-engine-classes)=
## Simulation Engine Classes

```mermaid
classDiagram
    class Simulation {
        -Config config
        -WidgetManufacturer manufacturer
        -ClaimGenerator claim_generator
        -InsuranceProgram insurance
        -ErgodicAnalyzer analyzer
        +__init__(config)
        +run(years, seed) SimulationResults
        +run_single_year(year, shock) YearResult
        +apply_insurance(claims) float
        +calculate_metrics() dict
        +reset()
    }

    class SimulationResults {
        +ndarray asset_paths
        +ndarray revenue_paths
        +ndarray claim_paths
        +ndarray growth_rates
        +dict metrics
        +__init__(paths, metrics)
        +get_terminal_wealth() float
        +get_time_average_growth() float
        +get_ensemble_average() float
        +to_dataframe() pd.DataFrame
        +plot_paths()
    }

    class MonteCarloEngine {
        -SimulationConfig config
        -ParallelExecutor executor
        -TrajectoryStorage storage
        +__init__(config)
        +run_ensemble(n_simulations) MonteCarloResults
        +run_parallel_batch(batch_size) List~SimulationResults~
        +aggregate_results(results) dict
        +calculate_confidence_intervals() dict
        +save_checkpoint(path)
    }

    class SimulationConfig {
        +int n_paths
        +int n_periods
        +float dt
        +int batch_size
        +int n_workers
        +bool save_trajectories
        +__init__(**kwargs)
        +validate()
        +to_dict() dict
    }

    class ParallelExecutor {
        -int n_workers
        -ChunkingStrategy strategy
        -SharedMemoryManager mem_manager
        +__init__(n_workers, strategy)
        +execute(tasks, func) List
        +map_reduce(data, map_func, reduce_func) Any
        +optimize_chunks(data_size) List~Tuple~
        +get_performance_metrics() PerformanceMetrics
    }

    class TrajectoryStorage {
        -StorageConfig config
        -h5py.File file
        -Dict cache
        +__init__(config)
        +store_trajectory(path_id, data)
        +retrieve_trajectory(path_id) ndarray
        +store_summary(summary)
        +get_summaries() List~SimulationSummary~
        +cleanup_old_data(days)
    }

    Simulation --> SimulationResults : produces
    Simulation --> WidgetManufacturer : contains
    Simulation --> ClaimGenerator : uses
    Simulation --> InsuranceProgram : uses
    MonteCarloEngine --> SimulationConfig : configured by
    MonteCarloEngine --> ParallelExecutor : uses
    MonteCarloEngine --> TrajectoryStorage : stores to
    MonteCarloEngine --> Simulation : runs multiple
```

## Key Relationships Summary

1. **Financial Core**: `WidgetManufacturer` is the central financial model that integrates claims, stochastic shocks, and insurance
2. **Claims Pipeline**: `ClaimGenerator` → `ClaimEvent` → `ClaimDevelopment` → `ClaimLiability`
3. **Insurance Structure**: `InsuranceProgram` orchestrates multiple `EnhancedInsuranceLayer` with state tracking
4. **Loss Modeling**: `ManufacturingLossGenerator` combines frequency and severity distributions
5. **Simulation**: `MonteCarloEngine` parallelizes multiple `Simulation` runs with result aggregation
6. **Stochastic**: All stochastic processes inherit from abstract `StochasticProcess` base class
