# Core Classes Architecture

## Overview
This document details the class structures and relationships for the core components of the Ergodic Insurance system.

## Manufacturer and Financial Model

```mermaid
classDiagram
    class WidgetManufacturer {
        -ManufacturerConfig config
        -StochasticProcess stochastic_process
        -float assets
        -float collateral
        -float restricted_assets
        -float equity
        -List~ClaimLiability~ outstanding_claims
        -ClaimDevelopment claim_development
        +__init__(config, stochastic_process)
        +generate_revenue(year) float
        +process_claim(amount, year) Tuple
        +manage_collateral() float
        +pay_outstanding_claims(current_year) float
        +step(year, claims, insurance_recovery) Dict
        +is_solvent() bool
        +unrestricted_assets() float
        +reset()
    }

    class ClaimLiability {
        +float original_amount
        +float remaining_amount
        +int year_incurred
        +List~float~ payment_schedule
        +get_payment(years_since_incurred) float
        +make_payment(amount) float
    }

    class ManufacturerConfig {
        +float initial_assets
        +float asset_turnover_mean
        +float asset_turnover_std
        +float operating_margin_mean
        +float operating_margin_std
        +float tax_rate
        +float working_capital_ratio
        +float collateral_requirement_ratio
        +float min_cash_ratio
        +float max_leverage_ratio
        +validate_constraints()
    }

    class StochasticProcess {
        <<abstract>>
        +float dt
        +int seed
        +RandomState rng
        +generate(t, current_value) float
        +reset()
        +simulate_path(t_start, t_end, initial_value) ndarray
    }

    class GBMProcess {
        +float mu
        +float sigma
        +generate(t, current_value) float
    }

    class MeanRevertingProcess {
        +float theta
        +float mu
        +float sigma
        +generate(t, current_value) float
    }

    WidgetManufacturer --> ManufacturerConfig: uses
    WidgetManufacturer --> StochasticProcess: uses
    WidgetManufacturer --> ClaimLiability: manages
    GBMProcess --|> StochasticProcess: inherits
    MeanRevertingProcess --|> StochasticProcess: inherits
```

## Simulation Engine

```mermaid
classDiagram
    class Simulation {
        -Config config
        -WidgetManufacturer manufacturer
        -ClaimGenerator claim_generator
        -InsurancePolicy insurance_policy
        -InsuranceProgram insurance_program
        -MonteCarloEngine monte_carlo
        +__init__(config, insurance_policy, insurance_program)
        +run(years, n_simulations) SimulationResults
        +run_single(years) SimulationResults
        +_process_year(year, state) Dict
        +_collect_results(trajectory) SimulationResults
        +save_checkpoint(path, year, state)
        +load_checkpoint(path) Tuple
    }

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
        +summary_stats() Dict
    }

    class MonteCarloEngine {
        -Simulation base_simulation
        -int n_workers
        -bool use_parallel
        -Optional~int~ seed
        +__init__(base_simulation, n_workers, use_parallel, seed)
        +run(years, n_simulations) MonteCarloResults
        +_run_batch(batch_params) List~SimulationResults~
        +analyze_convergence(results) ConvergenceMetrics
    }

    class MonteCarloResults {
        +List~SimulationResults~ paths
        +Dict~str_float~ ensemble_stats
        +Dict~str_float~ ergodic_stats
        +float convergence_metric
        +to_dataframe() DataFrame
        +get_percentiles(percentiles) Dict
        +calculate_var(confidence) float
        +calculate_tvar(confidence) float
    }

    Simulation --> WidgetManufacturer: creates
    Simulation --> ClaimGenerator: uses
    Simulation --> InsurancePolicy: uses
    Simulation --> SimulationResults: produces
    MonteCarloEngine --> Simulation: runs multiple
    MonteCarloEngine --> MonteCarloResults: produces
```

## Insurance Framework

```mermaid
classDiagram
    class InsuranceLayer {
        +float attachment_point
        +float limit
        +float rate
        +calculate_recovery(loss_amount) float
        +calculate_premium() float
    }

    class InsurancePolicy {
        -List~InsuranceLayer~ layers
        -float deductible
        +__init__(layers, deductible)
        +process_claim(loss_amount) Tuple
        +total_premium() float
        +total_limit() float
        +layer_details() List~Dict~
    }

    class EnhancedInsuranceLayer {
        +float attachment_point
        +float limit
        +float rate
        +int reinstatements
        +float reinstatement_premium
        +float remaining_limit
        +List~float~ reinstatement_costs
        +process_loss(loss_amount) Tuple
        +reset()
        +get_annual_premium() float
    }

    class InsuranceProgram {
        +List~EnhancedInsuranceLayer~ layers
        +float retention
        +str aggregate_deductible_type
        +float aggregate_deductible
        +float ytd_losses
        +__init__(layers, retention, agg_ded_type, agg_ded)
        +process_loss(loss_amount) Dict
        +get_total_premium() float
        +reset_annual()
        +get_program_summary() Dict
    }

    InsurancePolicy --> InsuranceLayer: contains
    InsuranceProgram --> EnhancedInsuranceLayer: contains
    EnhancedInsuranceLayer --|> InsuranceLayer: extends
```

## Claims and Loss Modeling

```mermaid
classDiagram
    class ClaimGenerator {
        -float frequency
        -float severity_mean
        -float severity_std
        -Optional~int~ seed
        -RandomState rng
        +__init__(frequency, severity_mean, severity_std, seed)
        +generate_annual_claims() List~ClaimEvent~
        +expected_annual_loss() float
        +simulate_year() Tuple
    }

    class ClaimEvent {
        +float amount
        +int year
        +str claim_type
        +Optional~Dict~ metadata
    }

    class ClaimDevelopment {
        +str pattern_type
        +List~float~ development_factors
        +float tail_factor
        +develop_claim(claim) DevelopedClaim
        +get_payment_schedule(claim_amount) List~float~
        +ultimate_loss(initial_loss) float
    }

    class LossDistribution {
        <<abstract>>
        +str name
        +Dict~str_float~ parameters
        +generate_loss() float
        +expected_loss() float
        +var(confidence) float
        +tvar(confidence) float
    }

    class CompoundPoissonLognormal {
        +float frequency
        +float severity_mean
        +float severity_cv
        +generate_annual_losses() List~float~
        +annual_aggregate_var(confidence) float
    }

    ClaimGenerator --> ClaimEvent: generates
    ClaimDevelopment --> ClaimEvent: develops
    CompoundPoissonLognormal --|> LossDistribution: inherits
```

## Bird's-Eye View: Complete System

```mermaid
graph TB
    subgraph Core["Core Domain"]
        Manufacturer[WidgetManufacturer]
        Config[Configuration]
    end

    subgraph Simulation["Simulation Layer"]
        Sim[Simulation Engine]
        MC[Monte Carlo]
    end

    subgraph Insurance["Insurance Domain"]
        Policy[Insurance Policy]
        Program[Insurance Program]
    end

    subgraph Loss["Loss Modeling"]
        Claims[Claim Generator]
        Distributions[Loss Distributions]
    end

    subgraph Analysis["Analysis Layer"]
        Ergodic[Ergodic Analyzer]
        Risk[Risk Metrics]
        Decision[Decision Engine]
    end

    Config --> Manufacturer
    Config --> Sim
    Manufacturer --> Sim
    Claims --> Sim
    Policy --> Sim
    Program --> Sim

    Sim --> MC
    MC --> Ergodic
    MC --> Risk

    Ergodic --> Decision
    Risk --> Decision
    Program --> Decision

    Distributions --> Claims

    style Core fill:#e3f2fd
    style Simulation fill:#f3e5f5
    style Insurance fill:#fff3e0
    style Loss fill:#fce4ec
    style Analysis fill:#e0f2f1
```
