# Core Classes Diagram

This diagram shows the main classes and their relationships in the core simulation engine.
The diagrams are split into focused views for readability: an overview diagram showing
high-level relationships, followed by detailed diagrams for the business model, insurance,
loss generation, and simulation subsystems.

## Overview Diagram

```{mermaid}
classDiagram
    direction TB

    class Simulation {
        +manufacturer: WidgetManufacturer
        +loss_generator: List~ManufacturingLossGenerator~
        +insurance_program: InsuranceProgram
        +time_horizon: int
        +run() SimulationResults
        +step_annual(year, losses) dict
        +run_with_loss_data() SimulationResults
        +run_monte_carlo() dict
        +get_trajectory() DataFrame
        +compare_insurance_strategies() dict
    }

    class MonteCarloEngine {
        +loss_generator: ManufacturingLossGenerator
        +insurance_program: InsuranceProgram
        +manufacturer: WidgetManufacturer
        +config: SimulationConfig
        +run() SimulationResults
        +export_results(results, filepath)
        +compute_bootstrap_confidence_intervals() dict
        +run_with_progress_monitoring() SimulationResults
        +run_with_convergence_monitoring() SimulationResults
        +estimate_ruin_probability() RuinProbabilityResults
    }

    class WidgetManufacturer {
        +config: ManufacturerConfig
        +ledger: Ledger
        +insurance_accounting: InsuranceAccounting
        +accrual_manager: AccrualManager
        +step() MetricsDict
        +calculate_revenue() Decimal
        +process_insurance_claim() tuple
        +check_solvency() bool
        +copy() WidgetManufacturer
        +reset()
    }

    class InsuranceProgram {
        +layers: List~EnhancedInsuranceLayer~
        +deductible: float
        +layer_states: List~LayerState~
        +calculate_annual_premium() float
        +process_claim(amount) dict
        +calculate_ergodic_benefit() dict
        +optimize_layer_structure() OptimalStructure
    }

    class InsurancePolicy {
        <<deprecated>>
        +layers: List~InsuranceLayer~
        +deductible: float
        +process_claim(amount) tuple
        +calculate_premium() float
        +to_enhanced_program() InsuranceProgram
    }

    class ManufacturingLossGenerator {
        +attritional: AttritionalLossGenerator
        +large: LargeLossGenerator
        +catastrophic: CatastrophicLossGenerator
        +generate_losses(duration, revenue) tuple
        +reseed(seed)
        +validate_distributions() dict
    }

    Simulation --> WidgetManufacturer : simulates
    Simulation --> ManufacturingLossGenerator : uses
    Simulation --> InsurancePolicy : uses (deprecated)
    Simulation --> SimulationResults : produces

    MonteCarloEngine --> WidgetManufacturer : copies per path
    MonteCarloEngine --> ManufacturingLossGenerator : uses
    MonteCarloEngine --> InsuranceProgram : uses
    MonteCarloEngine --> MCSimulationResults : produces

    InsurancePolicy --> InsuranceProgram : converts to (deprecated)

    WidgetManufacturer ..|> FinancialStateProvider : implements
```

## Business Model Detail

This diagram shows the internal structure of the manufacturer model, including
the financial ledger, tax handling, claim liabilities, and accounting modules.

```{mermaid}
classDiagram
    class WidgetManufacturer {
        +config: ManufacturerConfig
        +ledger: Ledger
        +insurance_accounting: InsuranceAccounting
        +accrual_manager: AccrualManager
        +stochastic_process: StochasticProcess
        +claim_liabilities: List~ClaimLiability~
        +is_ruined: bool
        +cash: Decimal
        +accounts_receivable: Decimal
        +inventory: Decimal
        +total_assets: Decimal
        +equity: Decimal
        +step() MetricsDict
        +calculate_revenue() Decimal
        +calculate_operating_income() Decimal
        +calculate_net_income() Decimal
        +process_insurance_claim() tuple
        +process_uninsured_claim() tuple
        +record_insurance_premium(amount)
        +record_insurance_loss(amount)
        +check_solvency() bool
        +handle_insolvency()
        +calculate_metrics() MetricsDict
        +copy() WidgetManufacturer
        +reset()
    }

    class ClaimLiability {
        +original_amount: Decimal
        +remaining_amount: Decimal
        +year_incurred: int
        +is_insured: bool
        +development_strategy: ClaimDevelopment
        +payment_schedule: List~float~
        +get_payment(years_since_incurred) Decimal
        +make_payment(amount) Decimal
    }

    class TaxHandler {
        +tax_rate: float
        +accrual_manager: AccrualManager
        +calculate_tax_liability(income) Decimal
        +apply_limited_liability_cap(tax, equity) tuple
        +calculate_and_accrue_tax() tuple
    }

    class ClaimDevelopment {
        +pattern_name: str
        +development_factors: List~float~
        +tail_factor: float
        +calculate_payments(amount, accident_yr, payment_yr) float
        +get_cumulative_paid(years_since_accident) float
        +create_immediate()$ ClaimDevelopment
        +create_medium_tail_5yr()$ ClaimDevelopment
        +create_long_tail_10yr()$ ClaimDevelopment
        +create_very_long_tail_15yr()$ ClaimDevelopment
    }

    class Ledger {
        <<Single Source of Truth>>
        +record_transaction()
        +get_balance(account) Decimal
        +prune_entries(before_date)
    }

    class AccrualManager {
        +record_accrual()
        +process_accrued_payments()
        +get_total_accruals() Decimal
    }

    class InsuranceAccounting {
        +record_premium()
        +record_loss()
        +record_recovery()
    }

    class FinancialStateProvider {
        <<Protocol>>
        +current_revenue: Decimal
        +current_assets: Decimal
        +current_equity: Decimal
        +base_revenue: Decimal
        +base_assets: Decimal
        +base_equity: Decimal
    }

    WidgetManufacturer --> Ledger : owns
    WidgetManufacturer --> AccrualManager : owns
    WidgetManufacturer --> InsuranceAccounting : owns
    WidgetManufacturer --> ClaimLiability : manages 0..*
    WidgetManufacturer --> TaxHandler : uses
    WidgetManufacturer ..|> FinancialStateProvider : implements

    ClaimLiability --> ClaimDevelopment : uses strategy
    TaxHandler --> AccrualManager : records accruals
```

## Insurance Subsystem Detail

This diagram shows the primary insurance path (`InsuranceProgram` / `EnhancedInsuranceLayer` / `LayerState`)
and the deprecated basic path (`InsurancePolicy` / `InsuranceLayer`).

```{mermaid}
classDiagram
    class InsurancePolicy {
        <<deprecated>>
        +layers: List~InsuranceLayer~
        +deductible: float
        +pricing_enabled: bool
        +pricer: InsurancePricer
        +process_claim(amount) tuple
        +calculate_recovery(amount) float
        +calculate_premium() float
        +get_total_coverage() float
        +from_yaml(path)$ InsurancePolicy
        +to_enhanced_program() InsuranceProgram
        +apply_pricing(revenue)
        +create_with_pricing()$ InsurancePolicy
    }

    class InsuranceLayer {
        <<deprecated, dataclass>>
        +attachment_point: float
        +limit: float
        +rate: float
        +calculate_recovery(loss_amount) float
        +calculate_premium() float
    }

    class InsuranceProgram {
        +name: str
        +layers: List~EnhancedInsuranceLayer~
        +deductible: float
        +layer_states: List~LayerState~
        +pricing_enabled: bool
        +pricer: InsurancePricer
        +calculate_annual_premium() float
        +process_claim(amount) dict
        +process_annual_claims(claims) dict
        +reset_annual()
        +get_program_summary() dict
        +get_total_coverage() float
        +calculate_ergodic_benefit(loss_history) dict
        +find_optimal_attachment_points(data) list
        +optimize_layer_widths(points, budget) list
        +optimize_layer_structure(loss_data) OptimalStructure
        +from_yaml(path)$ InsuranceProgram
        +create_standard_manufacturing_program()$ InsuranceProgram
        +apply_pricing(revenue)
        +create_with_pricing()$ InsuranceProgram
        +get_pricing_summary() dict
    }

    class EnhancedInsuranceLayer {
        <<dataclass>>
        +attachment_point: float
        +limit: float
        +base_premium_rate: float
        +reinstatements: int
        +reinstatement_premium: float
        +reinstatement_type: ReinstatementType
        +aggregate_limit: float
        +limit_type: str
        +calculate_base_premium() float
        +calculate_reinstatement_premium() float
        +can_respond(loss_amount) bool
        +calculate_layer_loss(total_loss) float
    }

    class LayerState {
        <<dataclass>>
        +layer: EnhancedInsuranceLayer
        +current_limit: float
        +used_limit: float
        +is_exhausted: bool
        +aggregate_used: float
        +process_claim(amount, timing) tuple
        +reset()
        +get_available_limit() float
        +get_utilization_rate() float
    }

    InsurancePolicy --> InsuranceLayer : contains 1..* (deprecated)
    InsurancePolicy ..> InsuranceProgram : converts to (deprecated)

    InsuranceProgram --> EnhancedInsuranceLayer : contains 1..*
    InsuranceProgram --> LayerState : tracks 1..*

    LayerState --> EnhancedInsuranceLayer : wraps
```

## Loss Generation Subsystem

This diagram shows the composite loss generator pattern and the loss event model.
`ManufacturingLossGenerator` composes three specialized generators for different
severity bands: attritional, large, and catastrophic.

```{mermaid}
classDiagram
    class ManufacturingLossGenerator {
        +attritional: AttritionalLossGenerator
        +large: LargeLossGenerator
        +catastrophic: CatastrophicLossGenerator
        +exposure: ExposureBase
        +gpd_generator: GeneralizedParetoLoss
        +generate_losses(duration, revenue) tuple
        +reseed(seed)
        +create_simple(freq, mean, std)$ ManufacturingLossGenerator
        +validate_distributions() dict
    }

    class AttritionalLossGenerator {
        +frequency: float
        +severity: LognormalLoss
        +generate_losses(duration, revenue) list
        +reseed(seed)
    }

    class LargeLossGenerator {
        +frequency: float
        +severity: LognormalLoss
        +generate_losses(duration, revenue) list
        +reseed(seed)
    }

    class CatastrophicLossGenerator {
        +frequency: float
        +severity: ParetoLoss
        +generate_losses(duration, revenue) list
        +reseed(seed)
    }

    class LossEvent {
        <<dataclass>>
        +amount: float
        +time: float
        +loss_type: str
        +description: str
    }

    class LossDistribution {
        <<abstract>>
        +rng: Generator
        +generate_severity(n)* ndarray
        +expected_value()* float
        +reset_seed(seed)
    }

    class LognormalLoss {
        +mu: float
        +sigma: float
        +mean: float
        +generate_severity(n) ndarray
        +expected_value() float
    }

    class ParetoLoss {
        +alpha: float
        +xm: float
        +generate_severity(n) ndarray
        +expected_value() float
    }

    ManufacturingLossGenerator *-- AttritionalLossGenerator : composes
    ManufacturingLossGenerator *-- LargeLossGenerator : composes
    ManufacturingLossGenerator *-- CatastrophicLossGenerator : composes
    ManufacturingLossGenerator ..> LossEvent : produces

    AttritionalLossGenerator ..> LossEvent : produces
    LargeLossGenerator ..> LossEvent : produces
    CatastrophicLossGenerator ..> LossEvent : produces

    LognormalLoss --|> LossDistribution
    ParetoLoss --|> LossDistribution

    AttritionalLossGenerator --> LognormalLoss : uses
    LargeLossGenerator --> LognormalLoss : uses
    CatastrophicLossGenerator --> ParetoLoss : uses
```

## Simulation and Monte Carlo Detail

This diagram shows the simulation orchestration layer, including both the
single-path `Simulation` class and the multi-path `MonteCarloEngine`.

```{mermaid}
classDiagram
    class Simulation {
        +manufacturer: WidgetManufacturer
        +loss_generator: List~ManufacturingLossGenerator~
        +insurance_program: InsuranceProgram
        +time_horizon: int
        +seed: int
        +run(progress_interval) SimulationResults
        +step_annual(year, losses) dict
        +run_with_loss_data(loss_data) SimulationResults
        +run_monte_carlo(config, policy, n_scenarios)$ dict
        +get_trajectory() DataFrame
        +compare_insurance_strategies(strategies) dict
    }

    class SimulationResults {
        <<dataclass>>
        +years: ndarray
        +assets: ndarray
        +equity: ndarray
        +roe: ndarray
        +revenue: ndarray
        +net_income: ndarray
        +claim_counts: ndarray
        +claim_amounts: ndarray
        +insolvency_year: int
        +to_dataframe() DataFrame
        +calculate_time_weighted_roe() float
        +calculate_rolling_roe(window) ndarray
        +summary_stats() dict
    }

    class MonteCarloEngine {
        +loss_generator: ManufacturingLossGenerator
        +insurance_program: InsuranceProgram
        +manufacturer: WidgetManufacturer
        +config: SimulationConfig
        +convergence_diagnostics: ConvergenceDiagnostics
        +parallel_executor: ParallelExecutor
        +trajectory_storage: TrajectoryStorage
        +run() MCSimulationResults
        -_run_sequential() MCSimulationResults
        -_run_parallel() MCSimulationResults
        -_run_enhanced_parallel() MCSimulationResults
        -_calculate_growth_rates(assets) ndarray
        -_calculate_metrics(results) dict
        -_check_convergence(results) dict
        +export_results(results, filepath)
        +compute_bootstrap_confidence_intervals(results) dict
        +run_with_progress_monitoring() MCSimulationResults
        +run_with_convergence_monitoring() MCSimulationResults
        +estimate_ruin_probability(config) RuinProbabilityResults
    }

    class MCSimulationResults {
        <<dataclass>>
        +final_assets: ndarray
        +annual_losses: ndarray
        +insurance_recoveries: ndarray
        +retained_losses: ndarray
        +growth_rates: ndarray
        +ruin_probability: dict
        +metrics: dict
        +convergence: dict
        +execution_time: float
        +config: SimulationConfig
        +performance_metrics: PerformanceMetrics
        +bootstrap_confidence_intervals: dict
        +summary() str
    }

    class SimulationConfig {
        <<dataclass>>
        +n_simulations: int
        +n_years: int
        +parallel: bool
        +n_workers: int
        +seed: int
        +use_enhanced_parallel: bool
        +insolvency_tolerance: float
    }

    Simulation --> SimulationResults : produces
    MonteCarloEngine --> MCSimulationResults : produces
    MonteCarloEngine --> SimulationConfig : configured by
```

## Class Interactions

```{mermaid}
sequenceDiagram
    participant MC as MonteCarloEngine
    participant S as Simulation
    participant M as WidgetManufacturer
    participant LG as ManufacturingLossGenerator
    participant IP as InsuranceProgram
    participant SR as SimulationResults

    MC->>M: copy() for each path
    MC->>LG: reseed() per path

    rect rgb(240, 240, 255)
    Note over S,SR: Single Simulation Path
    loop Each Year
        S->>LG: generate_losses(duration, revenue)
        LG-->>S: List of LossEvent

        loop Each LossEvent
            S->>IP: process_claim(amount)
            IP-->>S: recovery details dict
            S->>M: record_insurance_loss(retained)
            S->>M: record_insurance_premium(premium)
        end

        S->>M: step(growth_rate)
        M->>M: calculate_revenue()
        M->>M: calculate_operating_income()
        M->>M: calculate_net_income()
        M->>M: check_solvency()
        M-->>S: MetricsDict

        alt Insolvent
            M->>M: handle_insolvency()
            S-->>MC: Early termination
        end
    end

    S->>SR: Compile results
    end

    SR-->>MC: Path results
    MC->>MC: Aggregate all paths
    MC->>MC: Calculate risk metrics
    MC->>MC: Check convergence
```

## Key Design Patterns

### 1. **Strategy Pattern**
- `ClaimLiability` uses `ClaimDevelopment` as a payment strategy
- Insurance structures can use different pricing engines (`InsurancePricer`)
- Loss generators use pluggable severity distributions (`LossDistribution`)

### 2. **Composite Pattern**
- `ManufacturingLossGenerator` composes `AttritionalLossGenerator`, `LargeLossGenerator`, and `CatastrophicLossGenerator`
- `InsuranceProgram` manages multiple `EnhancedInsuranceLayer` instances

### 3. **Protocol (Structural Typing)**
- `FinancialStateProvider` protocol enables exposure-based classes to query live financial state from `WidgetManufacturer` without tight coupling
- Implemented via Python `typing.Protocol` for duck-typed structural subtyping

### 4. **Factory Pattern**
- `ManufacturingLossGenerator.create_simple()` for easy setup
- `InsuranceProgram.create_standard_manufacturing_program()` for standard configurations
- `ClaimDevelopment.create_immediate()`, `create_medium_tail_5yr()`, etc. for preset patterns
- `InsurancePolicy.create_with_pricing()` and `InsuranceProgram.create_with_pricing()` for priced programs

### 5. **Event Sourcing**
- `Ledger` serves as the single source of truth for all balance sheet accounts
- All financial mutations go through ledger transactions
- Balance sheet values are derived from ledger state, not stored independently

### 6. **Observer Pattern**
- `ProgressMonitor` observes Monte Carlo simulation progress
- `ConvergenceDiagnostics` monitors chain convergence during execution

### 7. **Facade Pattern**
- `MonteCarloEngine` provides a simplified interface to complex parallel execution, checkpointing, and aggregation
- `InsuranceProgram` facades complex multi-layer claim allocation with reinstatements
