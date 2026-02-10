# High-Level System Context Diagram

## Executive Summary

The Ergodic Insurance Limits framework analyzes insurance decisions using time-average (ergodic) theory rather than traditional ensemble averages. This approach reveals that insurance can enhance business growth even when premiums exceed expected losses by 200-500%, transforming insurance from a cost center to a growth enabler.

### Simplified System Architecture

```{mermaid}
flowchart LR
       %% Simplified Executive View
       INPUT[("📊 Market Data<br/>& Configuration")]
       BUSINESS[("🏭 Business<br/>Simulation")]
       ERGODIC[("📈 Ergodic<br/>Analysis")]
       OPTIMIZE[("🎯 Strategy<br/>Optimization")]
       OUTPUT[("📑 Reports &<br/>Insights")]

       INPUT --> BUSINESS
       BUSINESS --> ERGODIC
       ERGODIC --> OPTIMIZE
       OPTIMIZE --> OUTPUT

       %% Styling
       classDef inputStyle fill:#e3f2fd,stroke:#0d47a1,stroke-width:3px,font-size:14px
       classDef processStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:3px,font-size:14px
       classDef outputStyle fill:#e8f5e9,stroke:#1b5e20,stroke-width:3px,font-size:14px

       class INPUT inputStyle
       class BUSINESS,ERGODIC,OPTIMIZE processStyle
       class OUTPUT outputStyle
```

**Key Innovation**: By comparing time-average growth (what one business experiences over time) with ensemble-average growth (statistical average across many businesses), the framework demonstrates that insurance fundamentally transforms the growth dynamics of volatile businesses.

### System Architecture Overview (Detailed)

The actual implementation follows a sophisticated multi-layer architecture:

```{mermaid}
graph TB
       %% Input Layer
       subgraph Inputs["📥 Input Layer"]
           CONF["Configuration<br/>(YAML/JSON)"]
           HIST["Historical Loss Data"]
           PARAMS["Business Parameters"]
       end

       %% Core Simulation
       subgraph Core["⚙️ Core Simulation Engine"]
           MANU["WidgetManufacturer<br/>(Business Model)"]
           LOSSG["ManufacturingLossGenerator<br/>(Loss Events)"]
           INS["InsuranceProgram<br/>(Coverage Tower)"]
           SIM["Simulation Engine<br/>(Time Evolution)"]
       end

       %% Financial Core
       subgraph Financial["💰 Financial Core"]
           LEDGER["Ledger<br/>(Double-Entry Accounting)"]
           ACCRUAL["AccrualManager<br/>(Accrual Timing)"]
           INSACCT["InsuranceAccounting<br/>(Premium Amortization)"]
           TAXH["TaxHandler<br/>(Tax Calculations)"]
           DECUTIL["decimal_utils<br/>(Decimal Precision)"]
       end

       %% Analysis Layer
       subgraph Analysis["📊 Analysis & Optimization"]
           MONTE["Monte Carlo Engine<br/>(10,000+ paths)"]
           ERGODIC["Ergodic Analyzer<br/>(Time vs Ensemble)"]
           OPT["Business Optimizer<br/>(Strategy Selection)"]
           SENS["Sensitivity Analysis<br/>(Parameter Impact)"]
       end

       %% Output Layer
       subgraph Outputs["📤 Output & Insights"]
           EXCEL["Excel Reports<br/>(Detailed Results)"]
           VIZ["Visualizations<br/>(Executive & Technical)"]
           METRICS["Risk Metrics<br/>(VaR, CVaR, Ruin Prob)"]
           STRATEGY["Optimal Strategy<br/>(Limits & Retentions)"]
       end

       %% Data Flow
       Inputs --> Core
       Core --> MONTE
       MONTE --> Analysis
       Analysis --> Outputs

       %% Key Connections
       MANU -.-> INS
       LOSSG -.-> INS
       INS -.-> SIM
       SIM -.-> MONTE
       ERGODIC -.-> OPT
       OPT -.-> SENS

       %% Financial Core Connections
       MANU --> LEDGER
       MANU --> ACCRUAL
       MANU --> INSACCT
       TAXH --> ACCRUAL
       LEDGER --> DECUTIL
       ACCRUAL --> DECUTIL
       INSACCT --> DECUTIL

       classDef inputClass fill:#e3f2fd,stroke:#1565c0
       classDef coreClass fill:#fff3e0,stroke:#ef6c00
       classDef financialClass fill:#fff9c4,stroke:#f9a825
       classDef analysisClass fill:#f3e5f5,stroke:#7b1fa2
       classDef outputClass fill:#e8f5e9,stroke:#2e7d32

       class CONF,HIST,PARAMS inputClass
       class MANU,LOSSG,INS,SIM coreClass
       class LEDGER,ACCRUAL,INSACCT,TAXH,DECUTIL financialClass
       class MONTE,ERGODIC,OPT,SENS analysisClass
       class EXCEL,VIZ,METRICS,STRATEGY outputClass
```

### Reference to System Architecture Diagram

For a visual representation, see: [`assets/system_architecture.png`](../../../assets/system_architecture.png)

The PNG diagram shows the simplified flow, while the detailed architecture above reflects the actual implementation with all major components.

## Detailed System Architecture

This diagram shows the overall architecture of the Ergodic Insurance Limits framework, including the main components, external dependencies, and data flow between major modules.

```{mermaid}
flowchart TB
       %% External Inputs and Configurations
       subgraph External["External Inputs"]
           CONFIG[("Configuration Files<br/>YAML/JSON")]
           MARKET[("Market Data<br/>Loss Distributions")]
           PARAMS[("Business Parameters<br/>Financial Metrics")]
       end

       %% Core System Components
       subgraph Core["Core Simulation Engine"]
           SIM["Simulation<br/>Engine"]
           MANU["Widget<br/>Manufacturer<br/>Model"]
           LOSSG["Manufacturing<br/>Loss Generator"]
           INS["Insurance<br/>Program"]
       end

       %% Financial Accounting Subsystem
       subgraph FinAcct["Financial Accounting Subsystem"]
           LEDGER["Ledger<br/>(Double-Entry)"]
           ACCRUAL["AccrualManager<br/>(GAAP Timing)"]
           INSACCT["InsuranceAccounting<br/>(Premium & Recovery)"]
           TAXH["TaxHandler<br/>(Tax Accruals)"]
           DECUTIL["decimal_utils<br/>(Precision)"]
       end

       %% Insurance Subsystem
       subgraph InsuranceSub["Insurance Subsystem"]
           INSPOL["InsurancePolicy<br/>(Deprecated)"]
           INSLAY["InsuranceLayer<br/>(Deprecated)"]
           INSPROG["InsuranceProgram<br/>(Primary)"]
           ENHLAY["EnhancedInsuranceLayer<br/>(Primary)"]
           PRICER["InsurancePricer<br/>(Market Cycles)"]
       end

       %% Exposure & Trend System
       subgraph ExposureSub["Exposure & Trend System"]
           EXPBASE["ExposureBase<br/>(Dynamic Frequency)"]
           FSPROV["FinancialStateProvider<br/>(Protocol)"]
           TRENDS["trends.py<br/>(Trend Analysis)"]
       end

       %% Analysis and Optimization
       subgraph Analysis["Analysis & Optimization"]
           ERGODIC["Ergodic<br/>Analyzer"]
           OPT["Business<br/>Optimizer"]
           MONTE["Monte Carlo<br/>Engine"]
           SENS["Sensitivity<br/>Analyzer"]
       end

       %% Validation and Testing
       subgraph Validation["Validation & Testing"]
           ACC["Accuracy<br/>Validator"]
           BACK["Strategy<br/>Backtester"]
           WALK["Walk-Forward<br/>Validator"]
           CONV["Convergence<br/>Monitor"]
       end

       %% Processing Infrastructure
       subgraph Infrastructure["Processing Infrastructure"]
           BATCH["Batch<br/>Processor"]
           PARALLEL["Parallel<br/>Executor"]
           CACHE["Smart<br/>Cache"]
           STORAGE["Trajectory<br/>Storage"]
       end

       %% Reporting and Visualization
       subgraph Output["Reporting & Visualization"]
           VIZ["Visualization<br/>Engine"]
           EXCEL["Excel<br/>Reporter"]
           STATS["Summary<br/>Statistics"]
           METRICS["Risk<br/>Metrics"]
       end

       %% Data Flow - Input to Core
       CONFIG --> SIM
       MARKET --> LOSSG
       PARAMS --> MANU

       %% Core orchestration
       SIM --> MANU
       SIM --> LOSSG
       SIM --> INS

       MANU <--> INS
       LOSSG --> INS

       %% Manufacturer to Financial Accounting
       MANU --> LEDGER
       MANU --> ACCRUAL
       MANU --> INSACCT
       TAXH --> ACCRUAL
       LEDGER --> DECUTIL
       ACCRUAL --> DECUTIL
       INSACCT --> DECUTIL

       %% Insurance subsystem relationships
       INSPOL --> INSLAY
       INSPROG --> ENHLAY
       PRICER --> INSPROG
       PRICER --> INSPOL
       INS -.-> INSPROG
       INS -.-> INSPOL

       %% Exposure system
       EXPBASE --> FSPROV
       MANU -.-> FSPROV
       TRENDS --> LOSSG

       %% Core to Analysis
       SIM --> MONTE
       MONTE --> ERGODIC
       MONTE --> OPT

       ERGODIC --> SENS
       OPT --> SENS

       %% Validation
       MONTE --> ACC
       MONTE --> BACK
       BACK --> WALK

       MONTE --> CONV
       CONV --> BATCH

       %% Infrastructure
       BATCH --> PARALLEL
       PARALLEL --> CACHE
       CACHE --> STORAGE

       %% Output
       ERGODIC --> VIZ
       OPT --> VIZ
       SENS --> VIZ

       STORAGE --> STATS
       STATS --> EXCEL
       STATS --> METRICS

       VIZ --> EXCEL

       %% Styling
       classDef external fill:#e1f5fe,stroke:#01579b,stroke-width:2px
       classDef core fill:#fff3e0,stroke:#e65100,stroke-width:2px
       classDef financial fill:#fff9c4,stroke:#f9a825,stroke-width:2px
       classDef insurance fill:#ffe0b2,stroke:#e65100,stroke-width:2px
       classDef exposure fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
       classDef analysis fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
       classDef validation fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
       classDef infra fill:#fce4ec,stroke:#880e4f,stroke-width:2px
       classDef output fill:#e0f2f1,stroke:#004d40,stroke-width:2px

       class CONFIG,MARKET,PARAMS external
       class SIM,MANU,LOSSG,INS core
       class LEDGER,ACCRUAL,INSACCT,TAXH,DECUTIL financial
       class INSPOL,INSLAY,INSPROG,ENHLAY,PRICER insurance
       class EXPBASE,FSPROV,TRENDS exposure
       class ERGODIC,OPT,MONTE,SENS analysis
       class ACC,BACK,WALK,CONV validation
       class BATCH,PARALLEL,CACHE,STORAGE infra
       class VIZ,EXCEL,STATS,METRICS output
```

## System Overview

The Ergodic Insurance Limits framework is designed as a modular, high-performance system for analyzing insurance purchasing decisions through the lens of ergodic theory. The architecture follows these key principles:

### 1. **Separation of Concerns**
- **Core Simulation**: Handles the fundamental business and insurance mechanics
- **Financial Accounting**: Provides double-entry ledger, accrual accounting, insurance accounting, and tax handling -- all using Python's `Decimal` type for precision
- **Insurance Subsystem**: Provides `InsuranceProgram` with `EnhancedInsuranceLayer` for coverage modeling, with market-cycle-aware pricing via `InsurancePricer`. (The legacy `InsurancePolicy`/`InsuranceLayer` classes are deprecated.)
- **Exposure & Trends**: Dynamically adjusts claim frequencies using actual financial state (via the `FinancialStateProvider` protocol) and applies trend multipliers over time
- **Analysis Layer**: Provides ergodic and optimization capabilities
- **Infrastructure**: Manages computational efficiency and data handling
- **Validation**: Ensures accuracy and robustness of results
- **Output**: Delivers insights through visualizations and reports

### 2. **Data Flow Architecture**
- Configuration and market data flow into the simulation engine
- The `WidgetManufacturer` internally uses `Ledger`, `AccrualManager`, `InsuranceAccounting`, and `TaxHandler` for precise financial tracking
- All financial amounts use Python's `Decimal` type (via `decimal_utils`) to prevent floating-point drift across long simulations
- The `Ledger` maintains an O(1) current balance cache with pruning support for performance
- Simulations generate trajectories processed by analysis modules
- Infrastructure layers provide caching and parallelization
- Results flow to visualization and reporting components

### 3. **Key Interactions**
- The **Simulation Engine** orchestrates the time evolution of the business model
- The **Manufacturer Model** interacts with the **Insurance Program** for claim processing and uses the **Ledger** for all balance sheet operations
- **AccrualManager** tracks timing differences between cash movements and accounting recognition (wages, interest, taxes, insurance claims)
- **InsuranceAccounting** handles premium amortization as a prepaid asset and tracks insurance claim recoveries
- **TaxHandler** consolidates tax calculation, accrual, and payment logic, delegating accrual tracking to the **AccrualManager**
- **InsurancePricer** supports market cycles (Soft / Normal / Hard) to generate realistic premiums for insurance programs
- The **Exposure System** uses a `FinancialStateProvider` protocol so that `ExposureBase` subclasses query live financial state from the manufacturer for state-driven claim generation
- **Trend classes** (in `trends.py`) provide multiplicative adjustments to claim frequencies and severities over time, supporting linear, scenario-based, and stochastic trends
- **Monte Carlo Engine** generates multiple scenarios for statistical analysis
- **Ergodic Analyzer** compares time-average vs ensemble-average growth
- **Batch Processor** and **Parallel Executor** enable high-performance computing

### 4. **Financial Accounting Subsystem**

The financial accounting subsystem was introduced to provide GAAP-compliant financial tracking within the simulation. This subsystem is internal to the `WidgetManufacturer` and consists of four tightly integrated components:

```{mermaid}
flowchart LR
       MANU["WidgetManufacturer"] --> LEDGER["Ledger"]
       MANU --> ACCRUAL["AccrualManager"]
       MANU --> INSACCT["InsuranceAccounting"]
       MANU --> TAXH["TaxHandler"]

       TAXH --> ACCRUAL
       LEDGER --> DECUTIL["decimal_utils"]
       ACCRUAL --> DECUTIL
       INSACCT --> DECUTIL

       classDef manuClass fill:#fff3e0,stroke:#e65100,stroke-width:2px
       classDef finClass fill:#fff9c4,stroke:#f9a825,stroke-width:2px
       classDef utilClass fill:#e0f2f1,stroke:#004d40,stroke-width:2px

       class MANU manuClass
       class LEDGER,ACCRUAL,INSACCT,TAXH finClass
       class DECUTIL utilClass
```

- **Ledger**: Event-sourcing double-entry ledger with a typed `AccountName` enum (preventing typo bugs), `AccountType` classification, O(1) balance lookups via an internal cache, and support for pruning old transactions
- **AccrualManager**: Tracks accrual items (wages, interest, taxes, insurance claims, revenue) with configurable payment schedules (immediate, quarterly, annual, custom)
- **InsuranceAccounting**: Manages premium payments as prepaid assets with straight-line monthly amortization, and tracks insurance claim recoveries separately from claim liabilities
- **TaxHandler**: Centralizes tax calculation and accrual management, explicitly designed to avoid circular dependencies in the tax flow; delegates accrual tracking to `AccrualManager`
- **decimal_utils**: Foundation module providing `to_decimal()`, `quantize_currency()`, and standard constants (`ZERO`, `ONE`, `PENNY`) used by all financial modules

### 5. **Insurance Subsystem**

The insurance subsystem provides two complementary paths for modeling coverage:

```{mermaid}
flowchart TB
       subgraph Deprecated["Deprecated"]
           INSPOL["InsurancePolicy"]
           INSLAY["InsuranceLayer"]
           INSPOL --> INSLAY
       end

       subgraph Primary["Primary"]
           INSPROG["InsuranceProgram"]
           ENHLAY["EnhancedInsuranceLayer"]
           INSPROG --> ENHLAY
       end

       PRICER["InsurancePricer<br/>(Soft / Normal / Hard)"]
       PRICER --> INSPROG

       INSPOL -.->|deprecated, use| INSPROG

       classDef deprecatedClass fill:#ffcdd2,stroke:#b71c1c,stroke-width:2px
       classDef primaryClass fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
       classDef pricerClass fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px

       class INSPOL,INSLAY deprecatedClass
       class INSPROG,ENHLAY primaryClass
       class PRICER pricerClass
```

- **Primary Path**: `InsuranceProgram` (in `insurance_program.py`) uses `EnhancedInsuranceLayer` objects for full-featured coverage modeling including reinstatements, aggregate limits, and market-cycle-aware pricing
- **Deprecated**: `InsurancePolicy` (in `insurance.py`) with `InsuranceLayer` is deprecated in favor of `InsuranceProgram`
- **InsurancePricer** (in `insurance_pricing.py`) supports three `MarketCycle` states -- `HARD` (60% loss ratio), `NORMAL` (70%), and `SOFT` (80%)

### 6. **Exposure & Trend System**

The exposure and trend system models how insurance risks evolve dynamically during simulation:

```{mermaid}
flowchart LR
       MANU["WidgetManufacturer<br/>(implements protocol)"] -.-> FSPROV["FinancialStateProvider<br/>(Protocol)"]
       FSPROV --> EXPBASE["ExposureBase<br/>(Dynamic Frequency)"]
       TRENDS["trends.py<br/>(Trend Multipliers)"] --> LOSSG["ManufacturingLossGenerator"]
       EXPBASE --> LOSSG

       classDef coreClass fill:#fff3e0,stroke:#e65100,stroke-width:2px
       classDef protoClass fill:#e1f5fe,stroke:#01579b,stroke-width:2px
       classDef trendClass fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px

       class MANU coreClass
       class FSPROV,EXPBASE protoClass
       class TRENDS,LOSSG trendClass
```

- **FinancialStateProvider**: A `Protocol` (in `exposure_base.py`) defining properties like `current_revenue`, `current_assets`, `current_equity` and their base counterparts. `WidgetManufacturer` implements this protocol.
- **ExposureBase**: Abstract base for exposure classes that query live financial state to compute frequency multipliers (e.g., `RevenueExposure` scales claim frequency based on actual revenue vs. base revenue)
- **trends.py**: Provides a hierarchy of trend classes (`Trend` ABC, `LinearTrend`, `ScenarioTrend`, and stochastic variants) that apply multiplicative adjustments to claim frequencies and severities over time, supporting both annual and sub-annual time steps with optional seeded reproducibility

### 7. **External Dependencies**
The system integrates with:
- NumPy/SciPy for numerical computations
- Pandas for data manipulation
- Matplotlib/Plotly for visualizations
- OpenPyXL for Excel reporting
- Multiprocessing for parallel execution
- Python's `decimal` module for precise financial arithmetic
