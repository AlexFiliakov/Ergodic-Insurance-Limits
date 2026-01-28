# Core Classes Diagram

This diagram shows the main classes and their relationships in the core simulation engine.

```{mermaid}
classDiagram
    %% Core Simulation Classes
    class Simulation {
        -config: Config
        -manufacturer: WidgetManufacturer
        -claim_generator: ClaimGenerator
        -insurance_program: InsuranceProgram
        +run() SimulationResults
        +step(year: int) dict
        +process_claims(claims: List[ClaimEvent])
        +check_solvency() bool
    }

    class SimulationResults {
        +years: ndarray
        +assets: ndarray
        +equity: ndarray
        +roe: ndarray
        +revenue: ndarray
        +net_income: ndarray
        +claim_counts: ndarray
        +claim_amounts: ndarray
        +insolvency_year: Optional[int]
        +to_dataframe() DataFrame
        +calculate_time_weighted_roe() float
        +calculate_rolling_roe(window: int) ndarray
    }

    class MonteCarloEngine {
        -config: Config
        -n_simulations: int
        -parallel_executor: ParallelExecutor
        -trajectory_storage: TrajectoryStorage
        +run_simulations(n: int) List[SimulationResults]
        +run_parallel(n_workers: int) List[SimulationResults]
        +aggregate_results(results: List) dict
        +calculate_convergence() ConvergenceMetrics
    }

    %% Business Model Classes
    class WidgetManufacturer {
        -config: ManufacturerConfig
        -assets: float
        -equity: float
        -liabilities: float
        -claim_liabilities: List[ClaimLiability]
        +calculate_revenue() float
        +calculate_operating_income() float
        +process_insurance_claim(amount, deductible, limit) tuple
        +step(growth_rate) dict
        +apply_dividends(retention_ratio: float)
        +update_balance_sheet()
        +is_solvent() bool
    }

    class ClaimLiability {
        +original_amount: Decimal
        +remaining_amount: Decimal
        +year_incurred: int
        +is_insured: bool
        +development_strategy: ClaimDevelopment
        +payment_schedule: List[float]  <<property>>
        +get_payment(years_since_incurred) Decimal
        +make_payment(amount) Decimal
    }

    class ClaimDevelopment {
        +pattern_name: str
        +development_factors: List[float]
        +tail_factor: float
        +calculate_payments(claim_amount, accident_year, payment_year) float
        +get_cumulative_paid(years_since_accident) float
        +create_immediate() ClaimDevelopment
        +create_medium_tail_5yr() ClaimDevelopment
        +create_long_tail_10yr() ClaimDevelopment
        +create_very_long_tail_15yr() ClaimDevelopment
    }

    ClaimLiability --> ClaimDevelopment : uses

    %% Insurance Classes
    class InsurancePolicy {
        +deductible: float
        +limit: float
        +premium: float
        +coverage_type: str
        +calculate_payout(loss: float) float
        +calculate_retained_loss(loss: float) float
    }

    class InsuranceProgram {
        -policies: List[InsurancePolicy]
        -total_premium: float
        -letter_of_credit_rate: float
        +add_policy(policy: InsurancePolicy)
        +calculate_total_premium() float
        +process_claim(amount: float) ClaimResult
        +get_coverage_tower() List[dict]
        +calculate_effective_retention() float
    }

    class ClaimResult {
        +total_loss: float
        +company_payment: float
        +insurance_payment: float
        +by_policy: List[PolicyPayout]
    }

    %% Claim Generation
    class ClaimGenerator {
        -frequency_dist: Distribution
        -severity_dist: Distribution
        -random_state: RandomState
        +generate_claims(year: int) List[ClaimEvent]
        +set_seed(seed: int)
        +calibrate_to_historical(data: LossData)
    }

    class ClaimEvent {
        +year: int
        +amount: float
        +cause: str
        +severity_level: str
        +timestamp: float
    }

    %% Configuration
    class Config {
        +simulation: SimulationConfig
        +manufacturer: ManufacturerConfig
        +insurance: InsuranceConfig
        +monte_carlo: MonteCarloConfig
        +validate()
        +to_dict() dict
        +from_dict(data: dict) Config
    }

    class ManufacturerConfig {
        +initial_assets: float
        +asset_turnover_ratio: float
        +base_operating_margin: float
        +tax_rate: float
        +retention_ratio: float
        +target_leverage: float
    }

    %% Relationships
    Simulation --> WidgetManufacturer : uses
    Simulation --> ClaimGenerator : uses
    Simulation --> InsuranceProgram : uses
    Simulation --> SimulationResults : produces

    MonteCarloEngine --> Simulation : runs multiple
    MonteCarloEngine --> SimulationResults : aggregates

    WidgetManufacturer --> ClaimLiability : manages
    WidgetManufacturer --> InsuranceProgram : interacts with

    InsuranceProgram --> InsurancePolicy : contains
    InsuranceProgram --> ClaimResult : produces

    ClaimGenerator --> ClaimEvent : generates

    Config --> ManufacturerConfig : contains
    Simulation --> Config : configured by
```

## Class Interactions

```{mermaid}
sequenceDiagram
    participant MC as MonteCarloEngine
    participant S as Simulation
    participant M as Manufacturer
    participant CG as ClaimGenerator
    participant IP as InsuranceProgram
    participant SR as SimulationResults

    MC->>S: Initialize simulation
    S->>M: Create manufacturer
    S->>CG: Create claim generator
    S->>IP: Setup insurance program

    loop Each Year
        S->>CG: Generate claims
        CG-->>S: List[ClaimEvent]

        loop Each Claim
            S->>IP: Process claim
            IP->>M: Calculate payout
            M-->>IP: Company/insurance split
            IP-->>S: ClaimResult
        end

        S->>M: Step forward (growth, operations)
        M->>M: Update balance sheet
        M-->>S: Financial metrics

        S->>S: Check solvency
        alt Insolvent
            S-->>MC: Early termination
        end
    end

    S->>SR: Compile results
    SR-->>MC: SimulationResults
    MC->>MC: Aggregate all paths
```

## Key Design Patterns

### 1. **Strategy Pattern**
- Insurance strategies can be swapped dynamically
- Different optimization algorithms can be plugged in

### 2. **Builder Pattern**
- Configuration objects use builder pattern for complex initialization
- SimulationResults built incrementally during simulation

### 3. **Observer Pattern**
- Progress monitoring observes simulation progress
- Convergence monitors observe Monte Carlo iterations

### 4. **Factory Pattern**
- ClaimGenerator acts as factory for ClaimEvents
- FigureFactory creates visualization objects

### 5. **Facade Pattern**
- MonteCarloEngine provides simplified interface to complex parallel execution
- InsuranceProgram facades complex multi-policy interactions
