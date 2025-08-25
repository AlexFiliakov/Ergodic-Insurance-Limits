# Module Dependencies and Relationships

## Overview
This diagram illustrates the detailed module dependencies and import relationships within the Ergodic Insurance system.

```mermaid
graph LR
    subgraph Configuration["Configuration Management"]
        config[config.py]
        config_loader[config_loader.py]
    end

    subgraph Financial["Financial Modeling"]
        manufacturer[manufacturer.py]
        simulation[simulation.py]
    end

    subgraph Insurance["Insurance Framework"]
        insurance[insurance.py]
        insurance_program[insurance_program.py]
    end

    subgraph Loss["Loss & Claims"]
        claim_generator[claim_generator.py]
        claim_development[claim_development.py]
        loss_distributions[loss_distributions.py]
    end

    subgraph Stochastic["Stochastic Modeling"]
        stochastic_processes[stochastic_processes.py]
    end

    subgraph Analysis["Analysis & Metrics"]
        ergodic_analyzer[ergodic_analyzer.py]
        monte_carlo[monte_carlo.py]
        risk_metrics[risk_metrics.py]
        convergence[convergence.py]
        decision_engine[decision_engine.py]
    end

    subgraph Visualization["Output & Visualization"]
        visualization[visualization.py]
    end

    %% Configuration Dependencies
    config_loader -->|loads| config

    %% Manufacturer Dependencies
    manufacturer -->|uses| config
    manufacturer -->|uses| stochastic_processes
    manufacturer -->|processes| claim_development

    %% Simulation Dependencies
    simulation -->|creates| manufacturer
    simulation -->|uses| claim_generator
    simulation -->|uses| config
    simulation -->|uses| insurance
    simulation -->|uses| insurance_program
    simulation -->|uses| monte_carlo

    %% Insurance Dependencies
    insurance_program -->|extends| insurance
    insurance_program -->|uses| loss_distributions

    %% Loss Dependencies
    claim_generator -->|uses| config
    loss_distributions -->|generates| claim_generator
    claim_development -->|develops| claim_generator

    %% Analysis Dependencies
    ergodic_analyzer -->|analyzes| simulation
    monte_carlo -->|runs multiple| simulation
    risk_metrics -->|calculates from| simulation
    convergence -->|tests| monte_carlo
    decision_engine -->|uses| ergodic_analyzer
    decision_engine -->|uses| risk_metrics
    decision_engine -->|optimizes| insurance_program

    %% Visualization Dependencies
    visualization -->|plots| ergodic_analyzer
    visualization -->|plots| risk_metrics
    visualization -->|plots| simulation

    style Configuration fill:#e3f2fd
    style Financial fill:#f3e5f5
    style Insurance fill:#fff3e0
    style Loss fill:#fce4ec
    style Stochastic fill:#e8f5e9
    style Analysis fill:#e0f2f1
    style Visualization fill:#f1f8e9
```

## Module Interaction Patterns

### 1. Configuration Flow
```mermaid
sequenceDiagram
    participant User
    participant YAML as YAML Files
    participant Loader as config_loader
    participant Config as config (Pydantic)
    participant Component as System Component

    User->>YAML: Define parameters
    User->>Loader: load_config()
    Loader->>YAML: Read file
    Loader->>Config: Validate & Create
    Config-->>Loader: Config object
    Loader-->>User: Validated config
    User->>Component: Initialize(config)
    Component->>Config: Access parameters
```

### 2. Simulation Flow
```mermaid
sequenceDiagram
    participant Sim as Simulation
    participant Mfr as Manufacturer
    participant Claim as ClaimGenerator
    participant Ins as Insurance
    participant Process as StochasticProcess

    Sim->>Mfr: Initialize
    Sim->>Claim: Initialize
    Sim->>Ins: Initialize

    loop Each Year
        Sim->>Process: Generate shock
        Process-->>Sim: Revenue multiplier
        Sim->>Mfr: Apply revenue shock

        Sim->>Claim: Generate claims
        Claim-->>Sim: Claim events

        Sim->>Ins: Process claims
        Ins-->>Sim: Recoveries

        Sim->>Mfr: Update financials
        Mfr-->>Sim: New state
    end

    Sim-->>Sim: SimulationResults
```

### 3. Ergodic Analysis Flow
```mermaid
sequenceDiagram
    participant MC as MonteCarloEngine
    participant Sim as Simulation
    participant EA as ErgodicAnalyzer
    participant Risk as RiskMetrics
    participant DE as DecisionEngine

    MC->>Sim: Run N simulations
    loop N times
        Sim-->>MC: Path results
    end

    MC->>EA: Analyze paths
    EA->>EA: Calculate time averages
    EA->>EA: Calculate ensemble averages
    EA-->>MC: Ergodic metrics

    MC->>Risk: Calculate risk metrics
    Risk-->>MC: VaR, TVaR, etc.

    MC->>DE: Provide metrics
    DE->>DE: Optimize insurance
    DE-->>MC: Optimal strategy
```

## Key Module Responsibilities

### Core Modules

| Module | Primary Responsibility | Key Classes |
|--------|----------------------|-------------|
| `config.py` | Configuration models with validation | Config, ManufacturerConfig, InsuranceConfig |
| `manufacturer.py` | Financial model and balance sheet | WidgetManufacturer, ClaimLiability |
| `simulation.py` | Time evolution orchestration | Simulation, SimulationResults |
| `insurance.py` | Basic insurance structures | InsuranceLayer, InsurancePolicy |

### Advanced Modules

| Module | Primary Responsibility | Key Classes |
|--------|----------------------|-------------|
| `insurance_program.py` | Complex insurance programs | EnhancedInsuranceLayer, InsuranceProgram |
| `ergodic_analyzer.py` | Ergodic theory calculations | ErgodicAnalyzer, ErgodicAnalysisResults |
| `monte_carlo.py` | Ensemble simulation engine | MonteCarloEngine, MonteCarloResults |
| `decision_engine.py` | Insurance optimization | DecisionEngine, OptimizationResults |

### Support Modules

| Module | Primary Responsibility | Key Classes |
|--------|----------------------|-------------|
| `stochastic_processes.py` | Random process generation | GBMProcess, MeanRevertingProcess |
| `claim_generator.py` | Claim event generation | ClaimGenerator, ClaimEvent |
| `loss_distributions.py` | Loss distribution modeling | LossDistribution, LossData |
| `risk_metrics.py` | Risk metric calculations | RiskMetrics, RiskResults |
