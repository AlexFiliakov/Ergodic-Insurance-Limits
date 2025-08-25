# High-Level System Context Diagram

## Overview
This diagram shows the overall architecture of the Ergodic Insurance Limits system, illustrating how different components interact to provide insurance optimization through ergodic theory.

```mermaid
flowchart TB
    subgraph External["External Dependencies"]
        YAML[/"YAML Config Files<br/>(parameters/*.yaml)"/]
        CSV[/"Results Export<br/>(CSV/Excel)"/]
        HTML[/"Coverage Reports<br/>(HTML)"/]
        JUPYTER[/"Jupyter Notebooks<br/>(Analysis & Demos)"/]
    end

    subgraph Core["Core System"]
        CONFIG["Configuration Layer<br/>(Pydantic Models)"]
        MANUFACTURER["Manufacturer Model<br/>(Financial Simulation)"]
        INSURANCE["Insurance Framework<br/>(Policies & Programs)"]
        STOCHASTIC["Stochastic Processes<br/>(GBM, Mean-Reversion)"]
        LOSS["Loss Modeling<br/>(Distributions & Claims)"]
        ERGODIC["Ergodic Analysis<br/>(Time vs Ensemble)"]
    end

    subgraph Analytics["Analytics & Decision Support"]
        MONTE["Monte Carlo Engine<br/>(Ensemble Simulations)"]
        RISK["Risk Metrics<br/>(VaR, TVaR, Ruin)"]
        DECISION["Decision Engine<br/>(Optimization)"]
        CONVERGENCE["Convergence Analysis<br/>(Statistical Tests)"]
    end

    subgraph Outputs["Visualization & Reporting"]
        VIZ["Visualization Module<br/>(Matplotlib/Seaborn)"]
        REPORTS["Report Generation<br/>(DataFrames & Summaries)"]
    end

    %% Data Flow
    YAML -->|Load Parameters| CONFIG
    CONFIG -->|Initialize| MANUFACTURER
    CONFIG -->|Configure| INSURANCE
    CONFIG -->|Setup| STOCHASTIC
    CONFIG -->|Define| LOSS

    MANUFACTURER -->|Financial State| ERGODIC
    INSURANCE -->|Coverage Impact| MANUFACTURER
    STOCHASTIC -->|Revenue Volatility| MANUFACTURER
    LOSS -->|Claim Events| MANUFACTURER

    MANUFACTURER -->|Simulation Data| MONTE
    MONTE -->|Ensemble Results| RISK
    MONTE -->|Path Data| CONVERGENCE
    RISK -->|Metrics| DECISION
    ERGODIC -->|Growth Rates| DECISION

    DECISION -->|Optimal Strategy| REPORTS
    CONVERGENCE -->|Statistics| REPORTS
    RISK -->|Risk Analysis| VIZ
    ERGODIC -->|Comparisons| VIZ

    VIZ -->|Charts| HTML
    REPORTS -->|Data Export| CSV
    VIZ -->|Interactive| JUPYTER

    style Core fill:#e1f5fe
    style Analytics fill:#fff3e0
    style Outputs fill:#f3e5f5
    style External fill:#e8f5e9
```

## Component Descriptions

### External Dependencies
- **YAML Config Files**: Parameter configurations for different scenarios (baseline, conservative, optimistic, stochastic)
- **Results Export**: Simulation results exported to CSV/Excel for further analysis
- **Coverage Reports**: HTML test coverage reports from pytest
- **Jupyter Notebooks**: Interactive analysis and demonstration notebooks

### Core System
- **Configuration Layer**: Pydantic-based configuration management with validation
- **Manufacturer Model**: Widget manufacturer financial simulation with balance sheet evolution
- **Insurance Framework**: Multi-layer insurance policies and programs with reinstatements
- **Stochastic Processes**: GBM, lognormal volatility, and mean-reversion processes for uncertainty modeling
- **Loss Modeling**: Claim generation with Poisson frequency and lognormal severity
- **Ergodic Analysis**: Core ergodic theory implementation comparing time vs ensemble averages

### Analytics & Decision Support
- **Monte Carlo Engine**: Parallel simulation engine for ensemble analysis
- **Risk Metrics**: Comprehensive risk metrics including VaR, TVaR, and ruin probability
- **Decision Engine**: Insurance optimization and decision-making algorithms
- **Convergence Analysis**: Statistical tests for simulation convergence

### Visualization & Reporting
- **Visualization Module**: Matplotlib and Seaborn-based charting capabilities
- **Report Generation**: Structured output generation with pandas DataFrames

## Data Flow Patterns

1. **Configuration Flow**: YAML → Pydantic Models → Component Initialization
2. **Simulation Flow**: Manufacturer + Stochastic + Loss → Time Evolution → Results
3. **Analysis Flow**: Simulation Results → Risk/Ergodic Analysis → Decision Support
4. **Output Flow**: Analysis Results → Visualization/Reports → External Formats
