# Claim Lifecycle Architecture

This document provides comprehensive diagrams and descriptions of the claim lifecycle
in the Ergodic Insurance framework -- from loss generation through development, insurance
processing, accounting impact, and final payment.

## Overview

A claim in the framework follows a multi-stage lifecycle:

1. **Loss Generation** -- Stochastic processes produce loss events based on frequency and severity distributions, scaled by business exposure metrics.
2. **Claim Development** -- Raw loss events are assigned development patterns that determine how payments emerge over time.
3. **Insurance Processing** -- Each loss is processed through a multi-layer insurance program with deductibles, attachment points, limits, and reinstatements.
4. **Accounting Impact** -- Claim payments and recoveries are recorded through double-entry ledger transactions and accrual management.
5. **Cash Flow Projection** -- Future payment streams are projected, discounted, and reserved.

---

## 1. End-to-End Claim Lifecycle Flowchart

This diagram shows the complete flow from loss generation through final payment,
identifying which classes are responsible for each stage.

```{mermaid}
flowchart TB
    subgraph Generation["Loss Generation"]
        direction TB
        MLG["ManufacturingLossGenerator<br/>(composite)"]
        ALG["AttritionalLossGenerator<br/>freq: 5/yr, severity: $25K"]
        LLG["LargeLossGenerator<br/>freq: 0.3/yr, severity: $2M"]
        CLG["CatastrophicLossGenerator<br/>freq: 0.03/yr, Pareto"]
        FG["FrequencyGenerator<br/>Scaled Poisson Process"]
        LD["LossDistribution<br/>LognormalLoss | ParetoLoss | GPD"]

        MLG --> ALG
        MLG --> LLG
        MLG --> CLG
        ALG --> FG
        LLG --> FG
        CLG --> FG
        FG -->|"event_times"| LE["LossEvent<br/>time, amount, loss_type"]
        ALG --> LD
        LLG --> LD
        CLG --> LD
        LD -->|"severity samples"| LE
    end

    subgraph Development["Claim Development"]
        direction TB
        LE2["LossEvent"]
        CD["ClaimDevelopment<br/>pattern_name, development_factors, tail_factor"]
        CL["Claim<br/>claim_id, accident_year, reported_year,<br/>initial_estimate, payments_made"]
        CC["ClaimCohort<br/>groups by accident_year"]

        LE2 --> CL
        CD --> CL
        CL --> CC
    end

    subgraph Insurance["Insurance Processing"]
        direction TB
        IP["InsuranceProgram<br/>deductible, layers[]"]
        EIL["EnhancedInsuranceLayer<br/>attachment_point, limit, reinstatements"]
        LS["LayerState<br/>current_limit, used_limit, is_exhausted"]

        IP -->|"process_claim()"| DED{"Deductible<br/>Applied?"}
        DED -->|"Yes: company pays"| RET["retained_loss"]
        DED -->|"Excess"| EIL
        EIL --> LS
        LS -->|"payment + reinstatement_premium"| REC["insurance_recovery"]
    end

    subgraph Accounting["Accounting Impact"]
        direction TB
        WM["WidgetManufacturer"]
        LDG["Ledger<br/>double-entry transactions"]
        IA["InsuranceAccounting<br/>recoveries, prepaid_insurance"]
        AM["AccrualManager<br/>payment timing"]
        CLI["ClaimLiability<br/>remaining_amount, payment_schedule"]

        WM -->|"process_insurance_claim()"| LDG
        WM --> IA
        WM --> AM
        WM --> CLI
    end

    subgraph CashFlow["Cash Flow Projection"]
        direction TB
        CFP["CashFlowProjector<br/>discount_rate"]
        PROJ["project_payments()"]
        PV["calculate_present_value()"]
        IBNR["estimate_ibnr()"]
        RES["calculate_total_reserves()"]

        CFP --> PROJ
        CFP --> PV
        CFP --> IBNR
        CFP --> RES
    end

    Generation -->|"List[LossEvent]"| Development
    Development -->|"claim_amount"| Insurance
    Insurance -->|"retained_loss,<br/>insurance_recovery"| Accounting
    Accounting -->|"ClaimCohort"| CashFlow
```

### How it works

1. `ManufacturingLossGenerator` combines three sub-generators (attritional, large, catastrophic), each using a `FrequencyGenerator` for event timing and a `LossDistribution` for severity sampling.
2. Generated `LossEvent` objects are wrapped into `Claim` objects with a `ClaimDevelopment` pattern and grouped into `ClaimCohort` collections by accident year.
3. Each claim amount passes through `InsuranceProgram.process_claim()`, which applies the deductible, then routes the excess through sorted `EnhancedInsuranceLayer` objects tracked by `LayerState`.
4. The `WidgetManufacturer` records retained losses via the `Ledger` (double-entry accounting), creates `ClaimLiability` objects for multi-year payment schedules, and tracks insurance recoveries through `InsuranceAccounting`.
5. The `CashFlowProjector` uses `ClaimCohort` data to project future payment streams, estimate IBNR reserves, and calculate present values.

---

## 2. Insurance Layer Processing Sequence Diagram

This sequence diagram shows the detailed interaction when a claim is processed through
the multi-layer insurance program, including deductible application, layer responses,
and reinstatement mechanics.

```{mermaid}
sequenceDiagram
    participant Caller as Caller<br/>(Simulation)
    participant IP as InsuranceProgram
    participant L1State as LayerState<br/>(Primary)
    participant L1 as EnhancedInsuranceLayer<br/>(Primary: $250K xs $250K)
    participant L2State as LayerState<br/>(1st Excess)
    participant L2 as EnhancedInsuranceLayer<br/>(1st Excess: $20M xs $5M)
    participant L3State as LayerState<br/>(2nd Excess)
    participant L3 as EnhancedInsuranceLayer<br/>(2nd Excess: $25M xs $25M)

    Note over Caller,L3: Example: $12M Claim, $250K Deductible

    Caller->>IP: process_claim(claim_amount=12,000,000)
    activate IP

    Note over IP: Step 1: Apply Deductible
    IP->>IP: deductible_paid = min(claim, deductible)<br/>= $250,000
    IP->>IP: max_recoverable = claim - deductible<br/>= $11,750,000

    Note over IP: Step 2: Iterate Through Layers (sorted by attachment)

    rect rgb(230, 245, 255)
        Note over IP,L1: Primary Layer: $250K attachment, $4.75M limit
        IP->>L1: can_respond(12,000,000)?
        L1-->>IP: true (claim > attachment)
        IP->>L1: calculate_layer_loss(12,000,000)
        L1-->>IP: min(12M - 250K, 4.75M) = $4,750,000
        IP->>L1State: process_claim(4,750,000, timing_factor)
        Note over L1State: per-occurrence limit type:<br/>payment = min(claim, limit)
        L1State-->>IP: payment=$4,750,000, reinstatement_premium=$0
    end

    rect rgb(255, 245, 230)
        Note over IP,L2: 1st Excess Layer: $5M attachment, $20M limit
        IP->>L2: can_respond(12,000,000)?
        L2-->>IP: true (claim > attachment)
        IP->>L2: calculate_layer_loss(12,000,000)
        L2-->>IP: min(12M - 5M, 20M) = $7,000,000
        IP->>L2State: process_claim(7,000,000, timing_factor)
        Note over L2State: Check aggregate limit,<br/>process with reinstatements
        L2State-->>IP: payment=$7,000,000, reinstatement_premium=$0
    end

    rect rgb(245, 255, 230)
        Note over IP,L3: 2nd Excess Layer: $25M attachment, $25M limit
        IP->>L3: can_respond(12,000,000)?
        L3-->>IP: false (claim < attachment)
        Note over IP,L3: Layer not triggered
    end

    Note over IP: Step 3: Calculate Result
    IP->>IP: insurance_recovery = $11,750,000<br/>Guard: min(recovery, max_recoverable)
    IP->>IP: uncovered_loss = $0

    IP-->>Caller: {total_claim: 12M,<br/>deductible_paid: 250K,<br/>insurance_recovery: 11.75M,<br/>uncovered_loss: 0,<br/>layers_triggered: [Primary, 1st Excess]}
    deactivate IP
```

### Reinstatement Example

When a layer's aggregate limit is exhausted, the reinstatement mechanism activates:

```{mermaid}
sequenceDiagram
    participant IP as InsuranceProgram
    participant LS as LayerState<br/>(Aggregate Type)
    participant Layer as EnhancedInsuranceLayer<br/>($5M agg limit, 1 reinstatement)

    Note over IP,Layer: First Claim: $5M layer loss

    IP->>LS: process_claim(5,000,000)
    activate LS
    LS->>LS: available_limit = $5,000,000
    LS->>LS: payment = min(5M, 5M) = $5,000,000
    LS->>LS: aggregate_used = $5M >= aggregate_limit
    Note over LS: Aggregate exhausted!<br/>Reinstatements available: 1
    LS->>LS: reinstatements_used = 1
    LS->>LS: current_limit = $5M (restored)
    LS->>LS: aggregate_used = 0 (reset)
    LS->>Layer: calculate_reinstatement_premium(timing=0.75)
    Layer-->>LS: premium = base_premium * rate * 0.75<br/>(pro-rata for time remaining)
    LS-->>IP: payment=$5M, reinstatement_premium=$X
    deactivate LS

    Note over IP,Layer: Second Claim: $3M layer loss

    IP->>LS: process_claim(3,000,000)
    activate LS
    LS->>LS: available_limit = $5,000,000 (reinstated)
    LS->>LS: payment = $3,000,000
    LS->>LS: aggregate_used = $3M < $5M limit
    Note over LS: Still within reinstated limit
    LS-->>IP: payment=$3M, reinstatement_premium=$0
    deactivate LS

    Note over IP,Layer: Third Claim: $4M layer loss

    IP->>LS: process_claim(4,000,000)
    activate LS
    LS->>LS: available = min($2M remaining, $4M claim)
    LS->>LS: payment = $2,000,000
    LS->>LS: aggregate_used = $5M >= limit
    Note over LS: Exhausted again!<br/>No reinstatements left (1/1 used)
    LS->>LS: is_exhausted = true
    LS-->>IP: payment=$2M, reinstatement_premium=$0
    deactivate LS
```

---

## 3. Claim State Diagram

This state diagram shows the lifecycle states a claim transitions through,
from initial generation to final settlement.

```{mermaid}
stateDiagram-v2
    [*] --> Generated : LossEvent created by<br/>ManufacturingLossGenerator

    Generated --> Reported : Claim registered<br/>(reported_year assigned)
    Generated --> IBNR : Reporting lag<br/>(not yet reported)

    IBNR --> Reported : Late reporting<br/>(IBNR estimate updated)

    Reported --> InsuranceProcessed : InsuranceProgram<br/>process_claim()

    InsuranceProcessed --> Developing : ClaimDevelopment<br/>pattern assigned

    state InsuranceProcessed {
        [*] --> DeductibleApplied
        DeductibleApplied --> LayerAllocation : Excess above<br/>deductible
        LayerAllocation --> RecoveryCalculated : Sum layer<br/>payments
        RecoveryCalculated --> [*]
    }

    state Developing {
        [*] --> PaymentYear1
        PaymentYear1 --> PaymentYear2 : development_factors[0]<br/>applied
        PaymentYear2 --> PaymentYear3 : development_factors[1]<br/>applied
        PaymentYear3 --> FurtherYears : development_factors[2]<br/>applied
        FurtherYears --> TailPayments : Remaining factors<br/>applied
        TailPayments --> [*] : tail_factor or<br/>fully developed
    }

    Developing --> Settled : get_cumulative_paid() >= 1.0<br/>OR remaining_amount <= 0

    Settled --> [*] : ClaimLiability removed<br/>from active tracking

    note right of IBNR
        CashFlowProjector.estimate_ibnr()
        estimates unreported claims
        using chain-ladder method
    end note

    note right of Developing
        ClaimDevelopment patterns:
        - IMMEDIATE: [1.0]
        - MEDIUM_TAIL_5YR: [0.40, 0.25, 0.15, 0.10, 0.10]
        - LONG_TAIL_10YR: [0.10, 0.20, ... 0.02]
        - VERY_LONG_TAIL_15YR: [0.05, 0.10, ... 0.01]
    end note

    note left of Settled
        CashFlowProjector.calculate_total_reserves()
        tracks case_reserves + IBNR
    end note
```

### State Descriptions

| State | Description | Key Class |
|-------|-------------|-----------|
| **Generated** | A `LossEvent` has been produced by one of the loss generators with a time and amount. | `LossEvent` |
| **IBNR** | The loss occurred but has not yet been reported. Estimated via chain-ladder methods. | `CashFlowProjector` |
| **Reported** | The loss is registered as a `Claim` with a `reported_year` and `initial_estimate`. | `Claim` |
| **InsuranceProcessed** | The claim has been routed through `InsuranceProgram.process_claim()`, splitting it into retained loss and insurance recovery. | `InsuranceProgram` |
| **Developing** | Payments are being made according to the `ClaimDevelopment` pattern over multiple years. The `ClaimLiability` tracks remaining amounts. | `ClaimDevelopment`, `ClaimLiability` |
| **Settled** | All payments have been made. Cumulative paid reaches 100% of the claim amount. The liability is removed. | `Claim`, `ClaimLiability` |

---

## 4. Exposure-Based Frequency Scaling Flowchart

This diagram shows how exposure-based frequency scaling works, connecting the
financial state of the business to the frequency of loss events through the
`ExposureBase` hierarchy.

```{mermaid}
flowchart LR
    subgraph BusinessState["Business Financial State"]
        WM["WidgetManufacturer<br/>(implements FinancialStateProvider)"]
        CR["current_revenue"]
        CA["current_assets"]
        CE["current_equity"]
        BR["base_revenue"]
        BA["base_assets"]
        BE["base_equity"]

        WM --> CR
        WM --> CA
        WM --> CE
        WM --> BR
        WM --> BA
        WM --> BE
    end

    subgraph ExposureLayer["Exposure Base Layer"]
        direction TB
        EB["ExposureBase (ABC)"]
        RE["RevenueExposure<br/>current_revenue / base_revenue"]
        AE["AssetExposure<br/>current_assets / base_assets"]
        EE["EquityExposure<br/>current_equity / base_equity"]
        EME["EmployeeExposure<br/>base_employees * (1+rate)^t"]
        PE["ProductionExposure<br/>base_units * growth * seasonality"]
        COMP["CompositeExposure<br/>weighted combination"]

        EB --> RE
        EB --> AE
        EB --> EE
        EB --> EME
        EB --> PE
        EB --> COMP
    end

    subgraph FrequencyScaling["Frequency Scaling"]
        direction TB
        FG["FrequencyGenerator"]
        BF["base_frequency<br/>(e.g., 5.0 events/yr)"]
        RSE["revenue_scaling_exponent<br/>(e.g., 0.5 = sqrt scaling)"]
        SF["get_scaled_frequency(revenue)"]
        PP["Poisson Process<br/>n_events = Poisson(freq * duration)"]
        ET["generate_event_times()<br/>Uniform on [0, duration]"]

        FG --> BF
        FG --> RSE
        BF --> SF
        RSE --> SF
        SF --> PP
        PP --> ET
    end

    subgraph LossOutput["Loss Output"]
        direction TB
        LE["List[LossEvent]<br/>time, amount, loss_type"]
        LD["LossData<br/>timestamps, loss_amounts"]
    end

    CR -->|"get_exposure(time)"| RE
    CA -->|"get_exposure(time)"| AE
    CE -->|"get_exposure(time)"| EE

    RE -->|"get_frequency_multiplier()"| FM["frequency_multiplier<br/>= current / base"]
    AE -->|"get_frequency_multiplier()"| FM
    EE -->|"get_frequency_multiplier()"| FM

    FM -->|"actual_revenue"| SF
    ET -->|"event times"| LE
    LE --> LD
```

### Frequency Scaling Formula

The frequency scaling mechanism works as follows:

1. The `FinancialStateProvider` protocol (implemented by `WidgetManufacturer`) exposes current and base financial metrics.
2. An `ExposureBase` subclass calculates a frequency multiplier from the ratio of current-to-base metrics. For example, `RevenueExposure` computes `current_revenue / base_revenue`.
3. The `FrequencyGenerator` applies a power-law scaling: `scaled_frequency = base_frequency * (revenue / reference_revenue) ^ scaling_exponent`.
4. The scaled frequency parameterizes a Poisson process: `n_events ~ Poisson(scaled_frequency * duration)`.
5. Event times are drawn uniformly over the simulation period and sorted chronologically.

For attritional losses with `revenue_scaling_exponent = 0.5`, doubling revenue increases frequency by a factor of `sqrt(2) ~ 1.41`. This sub-linear scaling reflects that larger companies have more incidents but not proportionally more per unit of revenue.

---

## 5. Accounting Impact Detail

This diagram shows how insurance claim processing flows through the accounting
subsystem, covering ledger entries, accrual management, and claim liability tracking.

```{mermaid}
flowchart TB
    subgraph ClaimInput["Claim Processing Input"]
        CA["claim_amount"]
        IR["insurance_recovery"]
        RL["retained_loss<br/>(company pays)"]
    end

    subgraph Insured["Insured Claim Path<br/>WidgetManufacturer.process_insurance_claim()"]
        direction TB
        IC1["Calculate company_payment<br/>= deductible + excess over limit"]
        IC2["Calculate insurance_payment<br/>= min(claim - deductible, limit)"]
        IC3["Create ClaimLiability<br/>original_amount = company_payment<br/>development_strategy = long_tail_10yr"]
        IC4["Post collateral<br/>restricted_assets += company_payment"]
        IC5["Record in Ledger:<br/>DR Insurance Loss<br/>CR Claim Liabilities"]
        IC6["Record in InsuranceAccounting:<br/>record_claim_recovery(insurance_payment)"]
        IC7["Record in AccrualManager:<br/>record_expense_accrual(INSURANCE_CLAIMS)"]

        IC1 --> IC3
        IC2 --> IC6
        IC3 --> IC4
        IC4 --> IC5
        IC5 --> IC7
    end

    subgraph Uninsured["Uninsured Claim Path<br/>WidgetManufacturer.process_uninsured_claim()"]
        direction TB
        UC1["Full amount borne by company"]
        UC2{"immediate_payment?"}
        UC3["Reduce cash/assets immediately"]
        UC4["Create ClaimLiability<br/>is_insured = false<br/>No collateral required"]
        UC5["Record in Ledger:<br/>DR Insurance Loss<br/>CR Cash (or Claim Liabilities)"]

        UC1 --> UC2
        UC2 -->|"Yes"| UC3
        UC2 -->|"No"| UC4
        UC3 --> UC5
        UC4 --> UC5
    end

    subgraph Payment["Payment Over Time"]
        direction TB
        PY["pay_claim_liabilities()<br/>called each period"]
        CS["ClaimLiability.get_payment(dev_year)"]
        MP["ClaimLiability.make_payment(amount)"]
        RC["Release collateral<br/>restricted_assets -= payment"]
        LE["Ledger Entry:<br/>DR Claim Liabilities<br/>CR Cash"]

        PY --> CS
        CS --> MP
        MP --> RC
        RC --> LE
    end

    subgraph Recovery["Insurance Recovery"]
        direction TB
        IAR["InsuranceAccounting<br/>.receive_recovery_payment()"]
        LE2["Ledger Entry:<br/>DR Cash<br/>CR Insurance Receivables"]

        IAR --> LE2
    end

    CA --> Insured
    CA --> Uninsured
    Insured -->|"ClaimLiability<br/>created"| Payment
    Insured -->|"InsuranceRecovery<br/>receivable"| Recovery
    Uninsured -->|"ClaimLiability<br/>created"| Payment
```

### Key Accounting Entries

| Event | Debit | Credit | Module |
|-------|-------|--------|--------|
| Insured claim recognized | Insurance Loss (Expense) | Claim Liabilities (Liability) | `Ledger` |
| Collateral posted | Restricted Cash (Asset) | Cash (Asset) | `Ledger` |
| Claim payment made | Claim Liabilities (Liability) | Cash (Asset) | `Ledger` |
| Collateral released | Cash (Asset) | Restricted Cash (Asset) | `Ledger` |
| Recovery receivable recorded | Insurance Receivables (Asset) | Insurance Recovery (Revenue) | `InsuranceAccounting` |
| Recovery payment received | Cash (Asset) | Insurance Receivables (Asset) | `InsuranceAccounting` |
| Premium paid | Prepaid Insurance (Asset) | Cash (Asset) | `InsuranceAccounting` |
| Premium amortized monthly | Insurance Expense (Expense) | Prepaid Insurance (Asset) | `InsuranceAccounting` |

---

## 6. Class Relationship Summary

The following diagram summarizes the key classes and their relationships across
all stages of the claim lifecycle.

```{mermaid}
flowchart TB
    subgraph LossGen["Loss Generation Module<br/>loss_distributions.py"]
        LossDistribution["LossDistribution (ABC)"]
        LognormalLoss["LognormalLoss"]
        ParetoLoss["ParetoLoss"]
        GPDLoss["GeneralizedParetoLoss"]
        FreqGen["FrequencyGenerator"]
        AttritionalGen["AttritionalLossGenerator"]
        LargeGen["LargeLossGenerator"]
        CatGen["CatastrophicLossGenerator"]
        MfgGen["ManufacturingLossGenerator"]
        LossEvent2["LossEvent"]
        LossData2["LossData"]

        LossDistribution --> LognormalLoss
        LossDistribution --> ParetoLoss
        LossDistribution --> GPDLoss
        MfgGen -->|"composes"| AttritionalGen
        MfgGen -->|"composes"| LargeGen
        MfgGen -->|"composes"| CatGen
        AttritionalGen -->|"uses"| FreqGen
        AttritionalGen -->|"uses"| LognormalLoss
        LargeGen -->|"uses"| FreqGen
        LargeGen -->|"uses"| LognormalLoss
        CatGen -->|"uses"| FreqGen
        CatGen -->|"uses"| ParetoLoss
        AttritionalGen -->|"produces"| LossEvent2
        LossEvent2 -->|"collected in"| LossData2
    end

    subgraph Exposure["Exposure Module<br/>exposure_base.py"]
        FSP["FinancialStateProvider<br/>(Protocol)"]
        ExposureBase2["ExposureBase (ABC)"]
        RevExp["RevenueExposure"]
        AssetExp["AssetExposure"]
        EqExp["EquityExposure"]

        ExposureBase2 --> RevExp
        ExposureBase2 --> AssetExp
        ExposureBase2 --> EqExp
        FSP -.->|"queried by"| RevExp
        FSP -.->|"queried by"| AssetExp
        FSP -.->|"queried by"| EqExp
    end

    subgraph ClaimDev["Claim Development Module<br/>claim_development.py"]
        ClaimDevelopment2["ClaimDevelopment"]
        Claim2["Claim"]
        ClaimCohort2["ClaimCohort"]
        CashFlowProjector2["CashFlowProjector"]

        ClaimDevelopment2 -->|"defines pattern for"| Claim2
        Claim2 -->|"grouped into"| ClaimCohort2
        ClaimCohort2 -->|"analyzed by"| CashFlowProjector2
    end

    subgraph InsProg["Insurance Program Module<br/>insurance_program.py"]
        InsuranceProgram2["InsuranceProgram"]
        EnhancedLayer["EnhancedInsuranceLayer"]
        LayerState2["LayerState"]

        InsuranceProgram2 -->|"manages"| EnhancedLayer
        EnhancedLayer -->|"tracked by"| LayerState2
    end

    subgraph Acctg["Accounting Modules"]
        Manufacturer["WidgetManufacturer<br/>manufacturer.py"]
        ClaimLiab["ClaimLiability<br/>manufacturer.py"]
        Ledger2["Ledger<br/>ledger.py"]
        InsAcct["InsuranceAccounting<br/>insurance_accounting.py"]
        AccrualMgr["AccrualManager<br/>accrual_manager.py"]

        Manufacturer -->|"creates"| ClaimLiab
        Manufacturer -->|"records in"| Ledger2
        Manufacturer -->|"tracks via"| InsAcct
        Manufacturer -->|"schedules via"| AccrualMgr
    end

    MfgGen -.->|"uses for scaling"| ExposureBase2
    LossEvent2 -.->|"becomes"| Claim2
    LossData2 -.->|"processed by"| InsuranceProgram2
    InsuranceProgram2 -.->|"results feed"| Manufacturer
    Manufacturer -.->|"implements"| FSP
    ClaimDevelopment2 -.->|"used by"| ClaimLiab
```

---

## 7. Development Pattern Reference

The framework provides four standard development patterns, each suited to different
claim types common in manufacturing operations.

| Pattern | Years | Use Case | Factor Distribution |
|---------|-------|----------|---------------------|
| `IMMEDIATE` | 1 | Property/equipment damage | `[1.0]` |
| `MEDIUM_TAIL_5YR` | 5 | Workers compensation | `[0.40, 0.25, 0.15, 0.10, 0.10]` |
| `LONG_TAIL_10YR` | 10 | General liability | `[0.10, 0.20, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.03, 0.02]` |
| `VERY_LONG_TAIL_15YR` | 15 | Product liability | `[0.05, 0.10, 0.15, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02, 0.01, 0.01]` |

All patterns' factors sum to 1.0. The `ClaimDevelopment` class validates this invariant
at construction time (within a tolerance of 0.01).

---

## Key Source Files

| File | Purpose |
|------|---------|
| `loss_distributions.py` | Loss generation: `LossDistribution`, `FrequencyGenerator`, `ManufacturingLossGenerator`, `LossEvent`, `LossData` |
| `exposure_base.py` | Exposure scaling: `FinancialStateProvider`, `ExposureBase`, `RevenueExposure`, `AssetExposure`, `EquityExposure` |
| `claim_development.py` | Claim development: `ClaimDevelopment`, `Claim`, `ClaimCohort`, `CashFlowProjector` |
| `insurance_program.py` | Insurance processing: `InsuranceProgram`, `EnhancedInsuranceLayer`, `LayerState` |
| `manufacturer.py` | Business model and accounting: `WidgetManufacturer`, `ClaimLiability` |
| `ledger.py` | Double-entry accounting: `Ledger`, `AccountName`, `AccountType` |
| `insurance_accounting.py` | Insurance-specific accounting: `InsuranceAccounting`, `InsuranceRecovery` |
| `accrual_manager.py` | Payment timing: `AccrualManager`, `AccrualItem`, `PaymentSchedule` |
