# Ledger & Accounting System

This document provides architectural diagrams for the double-entry accounting subsystem
that underpins all financial operations in the Ergodic Insurance simulation framework.

The accounting system consists of four primary components:

- **Ledger** -- an event-sourcing, double-entry financial ledger with O(1) balance caching
- **AccrualManager** -- GAAP-compliant timing and accrual tracking with FIFO payment matching
- **InsuranceAccounting** -- premium amortization and recovery receivable tracking
- **TaxHandler** -- tax calculation, limited-liability capping, and accrual recording

All financial amounts use Python's `Decimal` type for precision. The `WidgetManufacturer`
class owns instances of each component and routes every financial operation through the
`Ledger` as balanced double-entry transactions.

---

## 1. Class Diagram

The diagram below shows the four accounting classes, their enumerations, and the
ownership relationships maintained by `WidgetManufacturer`.

```{mermaid}
classDiagram
    direction LR

    class AccountType {
        <<enumeration>>
        ASSET
        LIABILITY
        EQUITY
        REVENUE
        EXPENSE
    }

    class AccountName {
        <<enumeration>>
        CASH
        ACCOUNTS_RECEIVABLE
        INVENTORY
        PREPAID_INSURANCE
        INSURANCE_RECEIVABLES
        GROSS_PPE
        ACCUMULATED_DEPRECIATION
        RESTRICTED_CASH
        COLLATERAL
        ACCOUNTS_PAYABLE
        ACCRUED_EXPENSES
        ACCRUED_WAGES
        ACCRUED_TAXES
        ACCRUED_INTEREST
        CLAIM_LIABILITIES
        UNEARNED_REVENUE
        RETAINED_EARNINGS
        COMMON_STOCK
        DIVIDENDS
        REVENUE
        SALES_REVENUE
        INTEREST_INCOME
        INSURANCE_RECOVERY
        COST_OF_GOODS_SOLD
        OPERATING_EXPENSES
        DEPRECIATION_EXPENSE
        INSURANCE_EXPENSE
        INSURANCE_LOSS
        TAX_EXPENSE
        INTEREST_EXPENSE
        COLLATERAL_EXPENSE
        WAGE_EXPENSE
    }

    class EntryType {
        <<enumeration>>
        DEBIT
        CREDIT
    }

    class TransactionType {
        <<enumeration>>
        REVENUE
        COLLECTION
        EXPENSE
        PAYMENT
        WAGE_PAYMENT
        INTEREST_PAYMENT
        INVENTORY_PURCHASE
        INVENTORY_SALE
        INSURANCE_PREMIUM
        INSURANCE_CLAIM
        TAX_ACCRUAL
        TAX_PAYMENT
        DEPRECIATION
        WORKING_CAPITAL
        CAPEX
        ASSET_SALE
        DIVIDEND
        EQUITY_ISSUANCE
        DEBT_ISSUANCE
        DEBT_REPAYMENT
        ADJUSTMENT
        ACCRUAL
        WRITE_OFF
        REVALUATION
        LIQUIDATION
        TRANSFER
    }

    class LedgerEntry {
        <<dataclass>>
        +int date
        +str account
        +Decimal amount
        +EntryType entry_type
        +TransactionType transaction_type
        +str description
        +str reference_id
        +datetime timestamp
        +int month
        +signed_amount() Decimal
    }

    class Ledger {
        +List~LedgerEntry~ entries
        +Dict~str, AccountType~ chart_of_accounts
        -bool _strict_validation
        -Dict~str, Decimal~ _balances
        -Dict~str, Decimal~ _pruned_balances
        -Optional~int~ _prune_cutoff
        +record(entry: LedgerEntry) void
        +record_double_entry(date, debit_account, credit_account, amount, transaction_type, description, month) Tuple
        +get_balance(account, as_of_date) Decimal
        +get_period_change(account, period, month) Decimal
        +get_entries(account, start_date, end_date, transaction_type) List~LedgerEntry~
        +sum_by_transaction_type(transaction_type, period, account, entry_type) Decimal
        +get_cash_flows(period) Dict
        +verify_balance() Tuple~bool, Decimal~
        +get_trial_balance(as_of_date) Dict
        +prune_entries(before_date) int
        +clear() void
    }

    class AccrualType {
        <<enumeration>>
        WAGES
        INTEREST
        TAXES
        INSURANCE_CLAIMS
        REVENUE
        OTHER
    }

    class PaymentSchedule {
        <<enumeration>>
        IMMEDIATE
        QUARTERLY
        ANNUAL
        CUSTOM
    }

    class AccrualItem {
        <<dataclass>>
        +AccrualType item_type
        +Decimal amount
        +int period_incurred
        +PaymentSchedule payment_schedule
        +List~int~ payment_dates
        +List~Decimal~ amounts_paid
        +str description
        +remaining_balance() Decimal
        +is_fully_paid() bool
    }

    class AccrualManager {
        +Dict~AccrualType, List~ accrued_expenses
        +List~AccrualItem~ accrued_revenues
        +int current_period
        +record_expense_accrual(item_type, amount, payment_schedule, payment_dates, description) AccrualItem
        +record_revenue_accrual(amount, collection_dates, description) AccrualItem
        +process_payment(item_type, amount, period) List~Tuple~
        +get_quarterly_tax_schedule(annual_tax) List~Tuple~
        +get_claim_payment_schedule(claim_amount, development_pattern) List~Tuple~
        +get_total_accrued_expenses() Decimal
        +get_total_accrued_revenues() Decimal
        +get_accruals_by_type(item_type) List~AccrualItem~
        +get_payments_due(period) Dict
        +advance_period(periods) void
        +get_balance_sheet_items() Dict
        +clear_fully_paid() void
    }

    class InsuranceRecovery {
        <<dataclass>>
        +Decimal amount
        +str claim_id
        +int year_approved
        +Decimal amount_received
        +outstanding() Decimal
    }

    class InsuranceAccounting {
        <<dataclass>>
        +Decimal prepaid_insurance
        +Decimal monthly_expense
        +Decimal annual_premium
        +int months_in_period
        +int current_month
        +List~InsuranceRecovery~ recoveries
        +pay_annual_premium(premium_amount) Dict
        +record_monthly_expense() Dict
        +record_claim_recovery(recovery_amount, claim_id, year) Dict
        +receive_recovery_payment(amount, claim_id) Dict
        +get_total_receivables() Decimal
        +get_amortization_schedule() List~Dict~
        +reset_for_new_period() void
        +get_summary() Dict
    }

    class TaxHandler {
        <<dataclass>>
        +float tax_rate
        +AccrualManager accrual_manager
        +calculate_tax_liability(income_before_tax) Decimal
        +apply_limited_liability_cap(tax_amount, current_equity) Tuple~Decimal, bool~
        +record_tax_accrual(amount, time_resolution, current_year, current_month, description) void
        +calculate_and_accrue_tax(income_before_tax, current_equity, use_accrual, time_resolution, current_year, current_month) Tuple~Decimal, bool~
    }

    class WidgetManufacturer {
        +Ledger ledger
        +AccrualManager accrual_manager
        +InsuranceAccounting insurance_accounting
        +float tax_rate
        ...
    }

    %% Composition: WidgetManufacturer owns accounting subsystems
    WidgetManufacturer *-- Ledger : owns
    WidgetManufacturer *-- AccrualManager : owns
    WidgetManufacturer *-- InsuranceAccounting : owns
    WidgetManufacturer ..> TaxHandler : creates per tax calc

    %% Ledger internal relationships
    Ledger "1" o-- "*" LedgerEntry : stores
    LedgerEntry --> EntryType : has
    LedgerEntry --> TransactionType : has
    Ledger --> AccountType : maps accounts to
    Ledger --> AccountName : validates against

    %% AccrualManager internal relationships
    AccrualManager "1" o-- "*" AccrualItem : manages
    AccrualItem --> AccrualType : classified by
    AccrualItem --> PaymentSchedule : scheduled by

    %% InsuranceAccounting internal relationships
    InsuranceAccounting "1" o-- "*" InsuranceRecovery : tracks

    %% TaxHandler delegates to AccrualManager
    TaxHandler --> AccrualManager : records accruals via
```

### Key Relationships

- **WidgetManufacturer** owns a single `Ledger`, `AccrualManager`, and `InsuranceAccounting`
  instance throughout the life of a simulation trial.
- **TaxHandler** is a lightweight dataclass instantiated on each tax calculation call. It
  receives a reference to the `AccrualManager` so it can record tax accruals.
- Every financial mutation flows through `Ledger.record_double_entry()` to maintain the
  accounting equation (Assets = Liabilities + Equity).
- `AccrualManager` uses FIFO ordering when `process_payment()` is called to match
  payments against the oldest outstanding accruals first.
- `InsuranceAccounting` tracks the prepaid insurance asset and amortizes it monthly using
  straight-line amortization, absorbing any rounding residual in the final month.

---

## 2. Sequence Diagram -- Insurance Premium Payment Flow

This diagram traces the complete lifecycle of an annual insurance premium payment, from
the initial cash outflow through monthly amortization. It shows how three subsystems
collaborate: the `WidgetManufacturer` orchestrates, `InsuranceAccounting` tracks the
prepaid asset, and `Ledger` records every transaction as balanced double entries.

```{mermaid}
sequenceDiagram
    autonumber
    participant WM as WidgetManufacturer
    participant IA as InsuranceAccounting
    participant L as Ledger

    Note over WM: Start of coverage period

    WM->>IA: pay_annual_premium(premium_amount)
    activate IA
    IA-->>IA: Set prepaid_insurance = premium
    IA-->>IA: Calculate monthly_expense = premium / 12
    IA-->>IA: Reset current_month = 0
    IA-->>WM: {cash_outflow, prepaid_asset, monthly_expense}
    deactivate IA

    WM->>L: record_double_entry(debit=PREPAID_INSURANCE, credit=CASH, amount=premium, type=INSURANCE_PREMIUM)
    activate L
    L-->>L: Create debit LedgerEntry (PREPAID_INSURANCE +)
    L-->>L: Create credit LedgerEntry (CASH -)
    L-->>L: Update _balances cache for both accounts
    L-->>WM: (debit_entry, credit_entry)
    deactivate L

    Note over WM: Each month during coverage period

    loop Month 1 through 12
        WM->>IA: record_monthly_expense()
        activate IA
        IA-->>IA: expense = min(monthly_expense, prepaid_insurance)
        Note right of IA: Final month absorbs rounding residual
        IA-->>IA: prepaid_insurance -= expense
        IA-->>IA: current_month += 1
        IA-->>WM: {insurance_expense, prepaid_reduction, remaining_prepaid}
        deactivate IA

        WM->>L: record_double_entry(debit=INSURANCE_EXPENSE, credit=PREPAID_INSURANCE, amount=expense, type=EXPENSE)
        activate L
        L-->>L: Create debit LedgerEntry (INSURANCE_EXPENSE +)
        L-->>L: Create credit LedgerEntry (PREPAID_INSURANCE -)
        L-->>L: Update _balances cache
        L-->>WM: (debit_entry, credit_entry)
        deactivate L
    end

    Note over WM,L: Prepaid balance is now zero; full premium expensed
```

### Transaction Summary

| Step | Debit Account | Credit Account | Effect |
|------|--------------|----------------|--------|
| Premium payment | PREPAID_INSURANCE | CASH | Cash decreases; prepaid asset created |
| Monthly amortization (x12) | INSURANCE_EXPENSE | PREPAID_INSURANCE | Expense recognized; prepaid asset reduced |

After 12 months, `PREPAID_INSURANCE` returns to zero and the full premium has been
expensed through `INSURANCE_EXPENSE`.

---

## 3. Flowchart -- Double-Entry Transaction Flow

This flowchart shows the internal decision path when `Ledger.record_double_entry()` is
called. It highlights validation, zero-amount short-circuiting, account name resolution,
and the O(1) balance cache update.

```{mermaid}
flowchart TD
    A([record_double_entry called]) --> B{amount == 0?}
    B -- Yes --> C([Return None, None])
    B -- No --> D{amount < 0?}
    D -- Yes --> E([Raise ValueError])
    D -- No --> F{strict_validation?}

    F -- Yes --> G[Resolve debit_account via _resolve_account_name]
    G --> H{Account in CHART_OF_ACCOUNTS?}
    H -- No --> I([Raise ValueError with suggestions])
    H -- Yes --> J[Resolve credit_account via _resolve_account_name]
    J --> K{Account in CHART_OF_ACCOUNTS?}
    K -- No --> I
    K -- Yes --> L[Generate shared reference_id UUID]

    F -- No --> M[Convert AccountName enums to strings]
    M --> L

    L --> N[Create debit LedgerEntry]
    N --> O[Create credit LedgerEntry]
    O --> P[Call record for debit entry]

    P --> Q{Account in chart_of_accounts?}
    Q -- No, strict --> I
    Q -- No, non-strict --> R[Add account as ASSET type]
    R --> S[Append entry to entries list]
    Q -- Yes --> S

    S --> T[Update _balances cache]
    T --> U{Account is ASSET or EXPENSE?}
    U -- Yes --> V[DEBIT adds, CREDIT subtracts]
    U -- No --> W[CREDIT adds, DEBIT subtracts]
    V --> X[Call record for credit entry]
    W --> X

    X --> Y[Same validation and cache update for credit entry]
    Y --> Z([Return debit_entry, credit_entry])

    style A fill:#e1f5fe
    style C fill:#fff9c4
    style E fill:#ffcdd2
    style I fill:#ffcdd2
    style Z fill:#c8e6c9
```

### Balance Cache Behavior

The `_balances` dictionary acts as a running total for each account, maintained
incrementally as entries are recorded. This enables O(1) current-balance queries
through `get_balance(account)` without iterating over entries. Historical queries
(with `as_of_date`) still perform an O(N) scan but benefit from the pruned-balance
snapshot when `prune_entries()` has been called.

| Account Type | Debit Effect | Credit Effect | Normal Balance |
|-------------|-------------|--------------|----------------|
| ASSET | Increase (+) | Decrease (-) | Debit |
| EXPENSE | Increase (+) | Decrease (-) | Debit |
| LIABILITY | Decrease (-) | Increase (+) | Credit |
| EQUITY | Decrease (-) | Increase (+) | Credit |
| REVENUE | Decrease (-) | Increase (+) | Credit |

---

## 4. Integration with WidgetManufacturer

The `WidgetManufacturer` initializes all three accounting subsystems during construction:

```python
self.ledger = Ledger()                          # Double-entry transaction log
self.insurance_accounting = InsuranceAccounting() # Premium amortization tracker
self.accrual_manager = AccrualManager()           # Expense timing manager
```

All financial operations follow the same pattern:

1. The domain method (e.g., `calculate_net_income`, `process_insurance_premium`) computes
   business logic and determines amounts.
2. The relevant accounting subsystem (InsuranceAccounting, AccrualManager, or TaxHandler)
   is invoked to track the domain-specific state (prepaid balances, accrual schedules,
   tax liabilities).
3. The `Ledger` records the corresponding double-entry transaction(s), ensuring the
   accounting equation holds and cash flows are traceable.

This layered approach ensures that:

- The **Ledger** is the single source of truth for all balance sheet and income
  statement values.
- **AccrualManager** handles the timing gap between expense recognition and cash payment
  using FIFO matching.
- **InsuranceAccounting** provides domain-specific logic for prepaid asset amortization
  and recovery receivable tracking.
- **TaxHandler** enforces limited-liability constraints and routes tax accruals through
  the AccrualManager without circular dependencies.
