# Configuration Loading Flow

## Overview

The Ergodic Insurance framework uses a **configuration architecture** with profiles, modules, and presets to manage simulation parameters. This document provides detailed flow diagrams showing how configuration is loaded, resolved, validated, and delivered to consumers throughout the system.

The three tiers are:

| Tier | Component | Purpose | Location |
|------|-----------|---------|----------|
| **1** | Profiles | Complete configuration sets with inheritance | `data/config/profiles/` |
| **2** | Modules | Reusable optional components | `data/config/modules/` |
| **3** | Presets | Quick-apply templates | `data/config/presets/` |

The primary entry point is `ConfigManager.load_profile()`, which orchestrates the entire loading pipeline: file resolution, inheritance, module composition, preset application, runtime overrides, Pydantic validation, and caching.

---

## 1. Configuration Loading Pipeline

This sequence diagram shows the complete lifecycle of a `load_profile()` call, from the caller through `ConfigManager`, into the file system, and back through validation into a fully resolved `Config` object.

```{mermaid}
sequenceDiagram
    participant Caller as Caller
    participant CM as ConfigManager
    participant Cache as LRU Cache
    participant FS as File System<br/>(data/config/)
    participant CFG as Config<br/>(Pydantic)

    Caller->>CM: load_profile(profile_name, use_cache, **overrides)

    Note over CM: Generate cache key from<br/>profile_name + SHA-256(overrides)

    CM->>Cache: Check cache_key
    alt Cache hit (use_cache=True)
        Cache-->>CM: Return cached Config
        CM-->>Caller: Config (cached)
    else Cache miss or use_cache=False
        Note over CM: Resolve profile file path

        CM->>FS: Read profiles/{profile_name}.yaml
        alt Profile not found
            CM->>FS: Read profiles/custom/{profile_name}.yaml
            alt Custom profile not found
                CM-->>Caller: Raise FileNotFoundError
            end
        end
        FS-->>CM: Raw YAML data (dict)

        Note over CM: _load_with_inheritance()

        CM->>CM: Check for "extends" field
        alt Has parent profile
            CM->>FS: Read profiles/{parent_name}.yaml
            FS-->>CM: Parent YAML data
            CM->>CM: _load_with_inheritance(parent)<br/>(recursive)
            CM->>CM: _deep_merge(parent_data, child_data)
        end

        CM->>CFG: Config(**merged_data)
        Note over CFG: Pydantic validation<br/>(type checks, constraints,<br/>cross-field validators)
        CFG-->>CM: Validated Config

        Note over CM: Apply modules from<br/>profile.includes list

        loop For each module in profile.includes
            CM->>FS: Read modules/{module_name}.yaml
            FS-->>CM: Module YAML data
            CM->>CM: _apply_module(config, module_name)
            Note over CM: Deep merge module data<br/>into Config fields
        end

        Note over CM: Apply presets from<br/>profile.presets dict

        loop For each (preset_type, preset_name)
            CM->>FS: Read presets/{preset_type}.yaml
            FS-->>CM: Preset library data
            CM->>CM: _apply_preset(config, type, name)
            Note over CM: config.apply_preset()<br/>merges preset parameters
        end

        Note over CM: Apply runtime overrides

        alt Has **overrides
            CM->>CFG: config.with_overrides(**overrides)
            Note over CFG: Creates new Config<br/>with merged overrides
            CFG-->>CM: New Config
        end

        CM->>CFG: validate_completeness()
        CFG-->>CM: List of issues (warnings)

        CM->>Cache: Store config at cache_key
        CM-->>Caller: Config (validated)
    end
```

**Key observations:**

- The cache key is computed as `"{profile_name}_{sha256(overrides)[:16]}"` to ensure unique keys per parameter combination.
- Inheritance resolution is **recursive** -- a profile can extend another profile which itself extends a third profile, forming an inheritance chain.
- The `_deep_merge()` operation performs recursive dictionary merging, where child values override parent values at every nesting level.
- Module application mutates the `Config` instance in-place using `setattr`, while `with_overrides()` creates a **new** `Config` instance.
- Validation warnings (e.g., "Insurance enabled but no loss distribution configured") are emitted via Python `warnings.warn()` and do not block loading.

---

## 2. Profile Inheritance Resolution

This flowchart shows how a profile request is resolved through the inheritance chain, module overlay, preset application, and runtime override stages.

```{mermaid}
flowchart TD
    START([load_profile called]) --> RESOLVE_PATH

    subgraph Resolution["Phase 1: File Resolution"]
        RESOLVE_PATH[/"Resolve profile file path<br/>profiles/{name}.yaml"/]
        RESOLVE_PATH --> CHECK_EXISTS{Profile file<br/>exists?}
        CHECK_EXISTS -- Yes --> LOAD_YAML["Load YAML with<br/>yaml.safe_load()"]
        CHECK_EXISTS -- No --> CHECK_CUSTOM{Custom profile<br/>in custom/ dir?}
        CHECK_CUSTOM -- Yes --> LOAD_YAML
        CHECK_CUSTOM -- No --> ERROR_404["Raise FileNotFoundError<br/>with available profiles"]
    end

    subgraph Inheritance["Phase 2: Inheritance Chain"]
        LOAD_YAML --> STRIP_ANCHORS["Strip YAML anchors<br/>(keys starting with _)"]
        STRIP_ANCHORS --> HAS_EXTENDS{profile.extends<br/>is set?}
        HAS_EXTENDS -- No --> CREATE_CFG
        HAS_EXTENDS -- Yes --> LOAD_PARENT["Load parent profile<br/>_load_with_inheritance(parent)"]
        LOAD_PARENT --> PARENT_HAS_EXTENDS{Parent also<br/>has extends?}
        PARENT_HAS_EXTENDS -- Yes --> RECURSE["Recurse up<br/>inheritance chain"]
        RECURSE --> MERGE_PARENT
        PARENT_HAS_EXTENDS -- No --> MERGE_PARENT["_deep_merge(parent, child)<br/>Child overrides parent"]
        MERGE_PARENT --> CREATE_CFG["Create Config(**merged_data)<br/>Pydantic validates all fields"]
    end

    subgraph Modules["Phase 3: Module Overlay"]
        CREATE_CFG --> HAS_INCLUDES{profile.includes<br/>is non-empty?}
        HAS_INCLUDES -- No --> CHECK_PRESETS
        HAS_INCLUDES -- Yes --> LOAD_MODULE["Load module YAML<br/>modules/{name}.yaml"]
        LOAD_MODULE --> MODULE_EXISTS{Module file<br/>exists?}
        MODULE_EXISTS -- No --> WARN_MODULE["Warn: module not found"]
        WARN_MODULE --> MORE_MODULES
        MODULE_EXISTS -- Yes --> APPLY_MODULE["For each key in module:<br/>Deep merge into config fields"]
        APPLY_MODULE --> MORE_MODULES{More modules<br/>to apply?}
        MORE_MODULES -- Yes --> LOAD_MODULE
        MORE_MODULES -- No --> CHECK_PRESETS
    end

    subgraph Presets["Phase 4: Preset Application"]
        CHECK_PRESETS{profile.presets<br/>is non-empty?}
        CHECK_PRESETS -- No --> CHECK_OVERRIDES
        CHECK_PRESETS -- Yes --> LOAD_PRESET_LIB["Load preset library<br/>presets/{type}.yaml"]
        LOAD_PRESET_LIB --> PRESET_EXISTS{Preset name found<br/>in library?}
        PRESET_EXISTS -- No --> WARN_PRESET["Warn: preset not found"]
        WARN_PRESET --> MORE_PRESETS
        PRESET_EXISTS -- Yes --> APPLY_PRESET["config.apply_preset()<br/>Merge preset parameters"]
        APPLY_PRESET --> TRACK_PRESET["Track in<br/>applied_presets list"]
        TRACK_PRESET --> MORE_PRESETS{More presets<br/>to apply?}
        MORE_PRESETS -- Yes --> LOAD_PRESET_LIB
        MORE_PRESETS -- No --> CHECK_OVERRIDES
    end

    subgraph Overrides["Phase 5: Runtime Overrides"]
        CHECK_OVERRIDES{Runtime<br/>overrides<br/>provided?}
        CHECK_OVERRIDES -- No --> VALIDATE
        CHECK_OVERRIDES -- Yes --> APPLY_OVERRIDES["config.with_overrides(**kwargs)<br/>Creates NEW Config instance"]
        APPLY_OVERRIDES --> VALIDATE
    end

    subgraph Validation["Phase 6: Final Validation"]
        VALIDATE["validate_completeness()<br/>Check required sections<br/>Check logical consistency"]
        VALIDATE --> HAS_ISSUES{Validation<br/>issues?}
        HAS_ISSUES -- Yes --> EMIT_WARNINGS["Emit warnings via<br/>warnings.warn()"]
        EMIT_WARNINGS --> CACHE_RESULT
        HAS_ISSUES -- No --> CACHE_RESULT
    end

    CACHE_RESULT["Cache result at<br/>computed cache_key"] --> RETURN([Return Config])

    style START fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style RETURN fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style ERROR_404 fill:#ffebee,stroke:#c62828,stroke-width:2px
```

**Inheritance example:**

A custom profile `custom/client_abc.yaml` with `extends: conservative` triggers the following chain:

1. Load `custom/client_abc.yaml`
2. See `extends: conservative`, load `conservative.yaml`
3. See `extends: default`, load `default.yaml` (base -- no parent)
4. Merge: `default` <-- `conservative` overrides <-- `client_abc` overrides
5. Result: client-specific values override conservative values, which override defaults

---

## 3. Migration Path from Legacy System

This flowchart documents how the legacy 12-file configuration system maps to the current architecture, and shows the automated migration tool (`ConfigMigrator`).

```{mermaid}
flowchart TB
    subgraph Legacy["Legacy System (data/parameters/)"]
        B_YAML["baseline.yaml"]
        C_YAML["conservative.yaml"]
        O_YAML["optimistic.yaml"]
        INS_FILES["insurance.yaml<br/>insurance_market.yaml<br/>insurance_structures.yaml<br/>insurance_pricing_scenarios.yaml"]
        LOSS_FILES["losses.yaml<br/>loss_distributions.yaml"]
        STOCH_FILE["stochastic.yaml"]
        BIZ_FILE["business_optimization.yaml"]
    end

    subgraph MigrationTool["ConfigMigrator (Automated)"]
        direction TB
        RUN["run_migration()"]
        CONV_BASE["convert_baseline()<br/>baseline -> default profile"]
        CONV_CONS["convert_conservative()<br/>conservative -> conservative profile<br/>(extends: default)"]
        CONV_OPT["convert_optimistic()<br/>optimistic -> aggressive profile<br/>(extends: default)"]
        EXTRACT["extract_modules()<br/>Merge related YAML files<br/>into single modules"]
        CREATE_PRE["create_presets()<br/>Generate preset libraries"]
        VALIDATE_MIG["validate_migration()<br/>Check all files exist"]
        REPORT["generate_migration_report()"]

        RUN --> CONV_BASE --> CONV_CONS --> CONV_OPT
        CONV_OPT --> EXTRACT --> CREATE_PRE
        CREATE_PRE --> VALIDATE_MIG --> REPORT
    end

    subgraph NewSystem["New 3-Tier System (data/config/)"]
        direction TB
        subgraph Profiles["profiles/"]
            DEFAULT_P["default.yaml"]
            CONSERV_P["conservative.yaml<br/>(extends: default)"]
            AGGRESS_P["aggressive.yaml<br/>(extends: default)"]
            CUSTOM_P["custom/*.yaml"]
        end
        subgraph ModulesDir["modules/"]
            INS_MOD["insurance.yaml"]
            LOSS_MOD["losses.yaml"]
            STOCH_MOD["stochastic.yaml"]
            BIZ_MOD["business.yaml"]
        end
        subgraph PresetsDir["presets/"]
            MARKET_PRE["market_conditions.yaml"]
            LAYER_PRE["layer_structures.yaml"]
            RISK_PRE["risk_scenarios.yaml"]
        end
    end

    B_YAML --> CONV_BASE
    C_YAML --> CONV_CONS
    O_YAML --> CONV_OPT
    INS_FILES --> EXTRACT
    LOSS_FILES --> EXTRACT
    STOCH_FILE --> EXTRACT
    BIZ_FILE --> EXTRACT

    CONV_BASE -.-> DEFAULT_P
    CONV_CONS -.-> CONSERV_P
    CONV_OPT -.-> AGGRESS_P
    EXTRACT -.-> INS_MOD
    EXTRACT -.-> LOSS_MOD
    EXTRACT -.-> STOCH_MOD
    EXTRACT -.-> BIZ_MOD
    CREATE_PRE -.-> MARKET_PRE
    CREATE_PRE -.-> LAYER_PRE
    CREATE_PRE -.-> RISK_PRE

    subgraph Consumers["Consumer Code"]
        OLD_CODE["Legacy Code<br/>ConfigLoader().load('baseline')<br/>(deprecated)"]
        NEW_CODE["New Code<br/>ConfigManager().load_profile('default')"]
    end

    OLD_CODE --> NewSystem
    NEW_CODE --> NewSystem

    style Legacy fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style MigrationTool fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style NewSystem fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style Consumers fill:#fce4ec,stroke:#c62828,stroke-width:2px
```

**Migration steps in detail:**

1. **Profile conversion**: Each legacy scenario file (`baseline.yaml`, `conservative.yaml`, `optimistic.yaml`) is wrapped with `ProfileMetadata` and saved under `profiles/`. The conservative and aggressive profiles declare `extends: default`.

2. **Module extraction**: Related legacy files are merged using `_deep_merge()` into consolidated modules. For example, four insurance-related YAML files are merged into a single `modules/insurance.yaml`.

3. **Preset generation**: The migrator creates preset libraries with predefined parameter combinations (market conditions, layer structures, risk scenarios).

4. **Validation**: `validate_migration()` checks that all expected files exist in the new directory structure.

5. **Runtime compatibility**: The deprecated `ConfigLoader` delegates to `ConfigManager` internally, mapping old config names to new profile names.

---

## 4. Configuration Model Hierarchy

This class diagram shows the `Config` model and all of its sub-models. Required fields are shown with solid borders; optional modules have dashed borders.

```{mermaid}
classDiagram
    class Config {
        +ProfileMetadata profile
        +ManufacturerConfig manufacturer
        +WorkingCapitalConfig working_capital
        +GrowthConfig growth
        +DebtConfig debt
        +SimulationConfig simulation
        +OutputConfig output
        +LoggingConfig logging
        +InsuranceConfig insurance?
        +LossDistributionConfig losses?
        +ExcelReportConfig excel_reporting?
        +WorkingCapitalRatiosConfig working_capital_ratios?
        +ExpenseRatioConfig expense_ratios?
        +DepreciationConfig depreciation?
        +IndustryConfig industry_config?
        +Dict custom_modules
        +List applied_presets
        +Dict overrides
        +from_profile(Path) Config$
        +with_inheritance(Path, Path) Config$
        +apply_module(Path) void
        +apply_preset(str, Dict) void
        +with_overrides(**kwargs) Config
        +validate_completeness() List~str~
    }

    class ProfileMetadata {
        +str name
        +str description
        +str version
        +str extends?
        +List~str~ includes
        +Dict~str,str~ presets
        +str author?
        +datetime created?
        +List~str~ tags
    }

    class ManufacturerConfig {
        +float initial_assets
        +float asset_turnover_ratio
        +float base_operating_margin
        +float tax_rate
        +float retention_ratio
        +float ppe_ratio?
        +float insolvency_tolerance
        +ExpenseRatioConfig expense_ratios?
        +int premium_payment_month
        +str revenue_pattern
        +bool check_intra_period_liquidity
        +from_industry_config(IndustryConfig) ManufacturerConfig$
    }

    class WorkingCapitalConfig {
        +float percent_of_sales
    }

    class GrowthConfig {
        +str type
        +float annual_growth_rate
        +float volatility
    }

    class DebtConfig {
        +float interest_rate
        +float max_leverage_ratio
        +float minimum_cash_balance
    }

    class SimulationConfig {
        +str time_resolution
        +int time_horizon_years
        +int max_horizon_years
        +int random_seed?
        +int fiscal_year_end
    }

    class OutputConfig {
        +str output_directory
        +str file_format
        +int checkpoint_frequency
        +bool detailed_metrics
    }

    class LoggingConfig {
        +bool enabled
        +str level
        +str log_file?
        +bool console_output
        +str format
    }

    class InsuranceConfig {
        +bool enabled
        +List~InsuranceLayerConfig~ layers
        +float deductible
        +float coinsurance
        +int waiting_period_days
        +float claims_handling_cost
    }

    class InsuranceLayerConfig {
        +str name
        +float limit
        +float attachment
        +float base_premium_rate
        +int reinstatements
        +float aggregate_limit?
        +str limit_type
        +float per_occurrence_limit?
    }

    class LossDistributionConfig {
        +str frequency_distribution
        +float frequency_annual
        +str severity_distribution
        +float severity_mean
        +float severity_std
        +float correlation_factor
        +float tail_alpha
    }

    class ExcelReportConfig {
        +bool enabled
        +str output_path
        +bool include_balance_sheet
        +bool include_income_statement
        +bool include_cash_flow
        +str engine
    }

    class ExpenseRatioConfig {
        +float gross_margin_ratio
        +float sga_expense_ratio
        +float manufacturing_depreciation_allocation
        +float admin_depreciation_allocation
        +float direct_materials_ratio
        +float direct_labor_ratio
        +float manufacturing_overhead_ratio
        +float selling_expense_ratio
        +float general_admin_ratio
    }

    class DepreciationConfig {
        +float ppe_useful_life_years
        +int prepaid_insurance_amortization_months
        +float initial_accumulated_depreciation
    }

    class WorkingCapitalRatiosConfig {
        +float days_sales_outstanding
        +float days_inventory_outstanding
        +float days_payable_outstanding
    }

    class ModuleConfig {
        +str module_name
        +str module_version
        +List~str~ dependencies
    }

    class PresetConfig {
        +str preset_name
        +str preset_type
        +str description
        +Dict parameters
    }

    class PresetLibrary {
        +str library_type
        +str description
        +Dict~str,PresetConfig~ presets
        +from_yaml(Path) PresetLibrary$
    }

    Config *-- ProfileMetadata : profile
    Config *-- ManufacturerConfig : manufacturer
    Config *-- WorkingCapitalConfig : working_capital
    Config *-- GrowthConfig : growth
    Config *-- DebtConfig : debt
    Config *-- SimulationConfig : simulation
    Config *-- OutputConfig : output
    Config *-- LoggingConfig : logging
    Config o-- InsuranceConfig : insurance (optional)
    Config o-- LossDistributionConfig : losses (optional)
    Config o-- ExcelReportConfig : excel_reporting (optional)
    Config o-- WorkingCapitalRatiosConfig : working_capital_ratios (optional)
    Config o-- ExpenseRatioConfig : expense_ratios (optional)
    Config o-- DepreciationConfig : depreciation (optional)
    Config o-- ModuleConfig : custom_modules (dict)

    InsuranceConfig *-- InsuranceLayerConfig : layers (list)
    ManufacturerConfig o-- ExpenseRatioConfig : expense_ratios (optional)

    PresetLibrary *-- PresetConfig : presets (dict)
```

**Design notes:**

- **Composition over inheritance**: `Config` uses Pydantic `BaseModel` composition rather than class inheritance. Each sub-model is an independent, validated component.
- **Required vs. optional**: The seven required sub-models (`profile`, `manufacturer`, `working_capital`, `growth`, `debt`, `simulation`, `output`, `logging`) define the minimum viable configuration. Optional modules (`insurance`, `losses`, etc.) are loaded only when declared in the profile's `includes` list.
- **`with_overrides()` immutability**: Calling `with_overrides()` returns a **new** `Config` instance rather than mutating the existing one, enabling safe concurrent usage and cache correctness.
- **`validate_completeness()`** checks for logical consistency beyond Pydantic's type validation, for example verifying that insurance is not enabled without a loss distribution.

---

## Orchestration Summary

The following table summarizes the key classes and their responsibilities:

| Class | Module | Responsibility |
|-------|--------|----------------|
| `ConfigManager` | `config_manager.py` | Main entry point; orchestrates loading, caching, inheritance, modules, presets, and overrides |
| `ConfigLoader` | `config_loader.py` | **Deprecated.** Legacy YAML/JSON file loading; delegates to `ConfigManager` internally |
| `Config` | `config.py` | Pydantic-validated configuration container; holds all sub-models (required and optional) |
| `ConfigMigrator` | `config_migrator.py` | One-time migration tool: converts 12 legacy YAML files to 3-tier directory structure |
| `PresetLibrary` | `config.py` | Collection of `PresetConfig` entries loaded from a single preset YAML file |

---

## File System Layout

```
ergodic_insurance/
    data/
        config/                     # New 3-tier system root
            profiles/
                default.yaml        # Tier 1: Standard baseline
                conservative.yaml   # Tier 1: Conservative (extends: default)
                aggressive.yaml     # Tier 1: Aggressive (extends: default)
                custom/             # Tier 1: User-defined profiles
                    *.yaml
            modules/
                insurance.yaml      # Tier 2: Insurance configuration
                losses.yaml         # Tier 2: Loss distribution configuration
                stochastic.yaml     # Tier 2: Stochastic process configuration
                business.yaml       # Tier 2: Business model configuration
            presets/
                market_conditions.yaml   # Tier 3: Market scenario presets
                layer_structures.yaml    # Tier 3: Insurance layer templates
                risk_scenarios.yaml      # Tier 3: Risk scenario presets
        parameters/                 # Legacy system (deprecated)
            baseline.yaml
            conservative.yaml
            optimistic.yaml
            ...
```

---

## Related Documentation

- [Configuration System Architecture](configuration_v2.md) -- Detailed design specification and usage examples
- [Module Overview](module_overview.md) -- How modules interact across the framework
- [Data Models](class_diagrams/data_models.md) -- Broader data model class diagrams
