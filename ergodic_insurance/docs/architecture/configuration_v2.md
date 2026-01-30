# Configuration System v2.0 Architecture

## Overview

The configuration system has been completely redesigned in v2.0 to provide a modern, flexible, and maintainable approach to managing simulation parameters. It uses Pydantic v2 models for strict validation and type safety, supports a three-tier file architecture (profiles, modules, presets), and maintains full backward compatibility with the legacy 12-file configuration system through dedicated adapter and migration utilities.

**Key source files:**

- `ergodic_insurance/config.py` -- All Pydantic models (`Config`, `ConfigV2`, sub-models, presets, industry configs)
- `ergodic_insurance/config_manager.py` -- `ConfigManager` (main interface for the 3-tier system)
- `ergodic_insurance/config_loader.py` -- `ConfigLoader` (deprecated legacy interface)
- `ergodic_insurance/config_compat.py` -- `LegacyConfigAdapter`, `ConfigTranslator`, `migrate_config_usage()`
- `ergodic_insurance/config_migrator.py` -- `ConfigMigrator` (automated file migration)
- `ergodic_insurance/stochastic_processes.py` -- `StochasticConfig` (stochastic process parameters)
- `ergodic_insurance/reporting/config.py` -- `ReportConfig` and related reporting models

## Architecture Diagram

```{mermaid}
graph TB
    %% Main Components
    subgraph ConfigSystem["Configuration System v2.0"]
        CM["ConfigManager<br/>Main Interface"]
        CV2["ConfigV2<br/>Pydantic Models"]
        CL["ConfigLoader<br/>Legacy (Deprecated)"]
        COMPAT["LegacyConfigAdapter<br/>Backward Compatibility"]
        MIGRATOR["ConfigMigrator<br/>Migration Tool"]
    end

    %% File Structure
    subgraph FileSystem["File System"]
        subgraph Profiles["Profiles/"]
            DEFAULT["default.yaml"]
            CONSERV["conservative.yaml"]
            AGGRESS["aggressive.yaml"]
            CUSTOM["custom/*.yaml"]
        end

        subgraph Modules["Modules/"]
            MOD_INS["insurance.yaml"]
            MOD_LOSS["losses.yaml"]
            MOD_STOCH["stochastic.yaml"]
            MOD_BUS["business.yaml"]
        end

        subgraph Presets["Presets/"]
            PRE_MARKET["market_conditions.yaml"]
            PRE_LAYER["layer_structures.yaml"]
            PRE_RISK["risk_scenarios.yaml"]
        end
    end

    %% Data Models
    subgraph Models["Configuration Models"]
        PROF_META["ProfileMetadata"]
        MANU_CFG["ManufacturerConfig"]
        WC_CFG["WorkingCapitalConfig"]
        INS_CFG["InsuranceConfig"]
        SIM_CFG["SimulationConfig"]
        LOSS_CFG["LossDistributionConfig"]
        GROWTH_CFG["GrowthConfig"]
        DEBT_CFG["DebtConfig"]
        OUTPUT_CFG["OutputConfig"]
        LOG_CFG["LoggingConfig"]
        EXCEL_CFG["ExcelReportConfig"]
        WCR_CFG["WorkingCapitalRatiosConfig"]
        EXP_CFG["ExpenseRatioConfig"]
        DEP_CFG["DepreciationConfig"]
    end

    %% Relationships
    CM --> CV2
    CL --> COMPAT
    COMPAT --> CM
    MIGRATOR --> Profiles
    MIGRATOR --> Modules
    MIGRATOR --> Presets

    CM --> Profiles
    CM --> Modules
    CM --> Presets

    Profiles --> DEFAULT
    Profiles --> CONSERV
    Profiles --> AGGRESS
    Profiles --> CUSTOM

    DEFAULT -.includes.-> Modules
    CONSERV -.includes.-> Modules
    AGGRESS -.includes.-> Modules

    Modules --> MOD_INS
    Modules --> MOD_LOSS
    Modules --> MOD_STOCH
    Modules --> MOD_BUS

    DEFAULT -.applies.-> Presets
    CONSERV -.applies.-> Presets
    AGGRESS -.applies.-> Presets

    CV2 --> Models
    Models --> PROF_META
    Models --> MANU_CFG
    Models --> WC_CFG
    Models --> INS_CFG
    Models --> SIM_CFG
    Models --> LOSS_CFG
    Models --> GROWTH_CFG
    Models --> DEBT_CFG
    Models --> OUTPUT_CFG
    Models --> LOG_CFG
    Models --> EXCEL_CFG
    Models --> WCR_CFG
    Models --> EXP_CFG
    Models --> DEP_CFG

    %% Styling
    classDef manager fill:#e3f2fd,stroke:#1565c0,stroke-width:3px
    classDef deprecated fill:#fce4ec,stroke:#c62828,stroke-width:2px,stroke-dasharray: 5 5
    classDef profiles fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef modules fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef presets fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef models fill:#ffebee,stroke:#c62828,stroke-width:2px

    class CM manager
    class CL,COMPAT deprecated
    class MIGRATOR manager
    class DEFAULT,CONSERV,AGGRESS,CUSTOM profiles
    class MOD_INS,MOD_LOSS,MOD_STOCH,MOD_BUS modules
    class PRE_MARKET,PRE_LAYER,PRE_RISK presets
    class PROF_META,MANU_CFG,WC_CFG,INS_CFG,SIM_CFG,LOSS_CFG,GROWTH_CFG,DEBT_CFG,OUTPUT_CFG,LOG_CFG,EXCEL_CFG,WCR_CFG,EXP_CFG,DEP_CFG models
```

## Three-Tier Configuration Architecture

```{mermaid}
graph LR
    %% Tier 1: Profiles
    subgraph Tier1["Tier 1: Profiles"]
        P1["Complete Configuration Sets"]
        P2["Default Profile"]
        P3["Conservative Profile"]
        P4["Aggressive Profile"]
        P5["Custom Profiles"]
    end

    %% Tier 2: Modules
    subgraph Tier2["Tier 2: Modules"]
        M1["Reusable Components"]
        M2["Insurance Module"]
        M3["Loss Module"]
        M4["Stochastic Module"]
        M5["Business Module"]
    end

    %% Tier 3: Presets
    subgraph Tier3["Tier 3: Presets"]
        PR1["Quick-Apply Templates"]
        PR2["Market Conditions"]
        PR3["Layer Structures"]
        PR4["Risk Scenarios"]
    end

    %% Relationships
    P1 --> P2
    P1 --> P3
    P1 --> P4
    P1 --> P5

    P2 -.includes.-> M1
    P3 -.includes.-> M1
    P4 -.includes.-> M1

    M1 --> M2
    M1 --> M3
    M1 --> M4
    M1 --> M5

    P2 -.applies.-> PR1
    P3 -.applies.-> PR1
    P4 -.applies.-> PR1

    PR1 --> PR2
    PR1 --> PR3
    PR1 --> PR4

    %% Styling
    classDef tier1 fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef tier2 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef tier3 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px

    class P1,P2,P3,P4,P5 tier1
    class M1,M2,M3,M4,M5 tier2
    class PR1,PR2,PR3,PR4 tier3
```

## Config Loading Pipeline

The following sequence diagram shows the full lifecycle of a configuration load request, including cache check, inheritance resolution, module application, preset application, and runtime overrides.

```{mermaid}
sequenceDiagram
    participant User
    participant CM as ConfigManager
    participant Cache as _cache (dict)
    participant FS as File System
    participant CV2 as ConfigV2

    User->>CM: load_profile("conservative", use_cache=True, **overrides)
    CM->>CM: Compute cache_key (SHA-256 of profile + overrides)
    CM->>Cache: Check cache_key
    alt Cache hit
        Cache-->>CM: Return cached ConfigV2
        CM-->>User: Return ConfigV2
    else Cache miss
        CM->>FS: Find profiles/conservative.yaml
        CM->>CM: _load_with_inheritance(profile_path)
        CM->>FS: Read conservative.yaml
        Note over CM: profile.extends = "default"
        CM->>FS: Read profiles/default.yaml (parent)
        CM->>CM: _deep_merge(parent_data, child_data)
        CM->>CV2: ConfigV2(**merged_data)
        CV2-->>CM: Validated config instance

        loop For each module in profile.includes
            CM->>FS: Read modules/{module}.yaml
            CM->>CV2: apply_module(module_data)
        end

        loop For each preset in profile.presets
            CM->>FS: Read presets/{type}.yaml
            CM->>CV2: apply_preset(preset_name, preset_data)
        end

        alt Runtime overrides provided
            CM->>CV2: with_overrides(**overrides)
            CV2-->>CM: New ConfigV2 with overrides
        end

        CM->>CV2: validate_completeness()
        CV2-->>CM: List of issues (warnings if any)

        CM->>Cache: Store result
        CM-->>User: Return ConfigV2
    end
```

## Profile Inheritance Resolution

Profiles support single inheritance through the `extends` field. When a profile extends another, the parent is loaded first (recursively if it also extends a parent), and the child's values are deep-merged on top.

```{mermaid}
graph TD
    subgraph InheritanceChain["Inheritance Resolution"]
        CUSTOM_CLIENT["custom/client_abc.yaml<br/>extends: conservative"]
        CONSERVATIVE["conservative.yaml<br/>extends: default"]
        DEFAULT["default.yaml<br/>base profile"]
    end

    DEFAULT -->|"1. Load base values"| CONSERVATIVE
    CONSERVATIVE -->|"2. Deep merge child overrides"| CUSTOM_CLIENT
    CUSTOM_CLIENT -->|"3. Final merged config"| RESULT["ConfigV2 Instance"]

    classDef base fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef derived fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef result fill:#fff3e0,stroke:#ef6c00,stroke-width:3px

    class DEFAULT base
    class CONSERVATIVE,CUSTOM_CLIENT derived
    class RESULT result
```

**Inheritance rules:**

1. The chain is resolved recursively from child up to the root profile.
2. Parent profiles are loaded first; child values are deep-merged on top.
3. Deep merge recurses into nested dictionaries; non-dict values are replaced entirely.
4. Missing parent profiles emit a warning but do not raise an error.
5. Circular inheritance is prevented by Python's natural recursion limit.

```yaml
# profiles/conservative.yaml
profile:
  name: conservative
  description: "Conservative risk parameters"
  extends: default
  version: "2.0.0"

# Only fields that differ from default need to be specified
manufacturer:
  base_operating_margin: 0.06
growth:
  annual_growth_rate: 0.03
```

## Component Descriptions

### ConfigManager

**Source:** `ergodic_insurance/config_manager.py`

The main interface for the 3-tier configuration system. Handles profile loading with inheritance, module composition, preset application, runtime overrides, caching, and validation.

```python
class ConfigManager:
    """Manages configuration loading with profiles, modules, and presets."""

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize manager.

        Args:
            config_dir: Root config directory. Defaults to
                        ergodic_insurance/data/config.
        """

    # --- Primary API ---

    def load_profile(
        self,
        profile_name: str = "default",
        use_cache: bool = True,
        **overrides
    ) -> ConfigV2:
        """Load a configuration profile with optional overrides."""

    # --- Discovery ---

    def list_profiles(self) -> List[str]:
        """List all available profile names (including custom/)."""

    def list_modules(self) -> List[str]:
        """List all available module names."""

    def list_presets(self) -> Dict[str, List[str]]:
        """List presets grouped by type."""

    def get_profile_metadata(self, profile_name: str) -> Dict[str, Any]:
        """Get profile metadata without loading full config (LRU-cached)."""

    # --- Mutation ---

    def create_profile(
        self, name: str, description: str,
        base_profile: str = "default",
        custom: bool = True,
        **config_params
    ) -> Path:
        """Create and save a new profile YAML file."""

    def with_preset(
        self, config: ConfigV2,
        preset_type: str, preset_name: str
    ) -> ConfigV2:
        """Return a new ConfigV2 with a preset applied."""

    def with_overrides(
        self, config: ConfigV2, **overrides
    ) -> ConfigV2:
        """Return a new ConfigV2 with runtime overrides."""

    # --- Validation ---

    def validate(self, config: ConfigV2) -> List[str]:
        """Validate config for completeness and consistency."""

    # --- Cache ---

    def clear_cache(self) -> None:
        """Clear configuration and preset caches."""
```

**Key Features:**

- **Profile inheritance resolution** -- recursive parent loading with deep merge
- **Module composition** -- selectively include optional configuration modules
- **Preset application** -- apply quick-change templates from preset libraries
- **Runtime overrides** -- override any parameter at load time via `**kwargs`
- **SHA-256 cache keys** -- deterministic caching based on profile name + overrides
- **Validation at load time** -- Pydantic field validation plus business logic checks
- **Profile creation** -- programmatic creation of new profile YAML files

### ConfigV2

**Source:** `ergodic_insurance/config.py`

The main Pydantic BaseModel that holds the entire configuration state. It composes required sub-models for core business parameters and optional sub-models for extended functionality.

```python
class ConfigV2(BaseModel):
    """Enhanced unified configuration model for the 3-tier system."""

    # --- Required sections ---
    profile: ProfileMetadata
    manufacturer: ManufacturerConfig
    working_capital: WorkingCapitalConfig
    growth: GrowthConfig
    debt: DebtConfig
    simulation: SimulationConfig
    output: OutputConfig
    logging: LoggingConfig

    # --- Optional module sections ---
    insurance: Optional[InsuranceConfig] = None
    losses: Optional[LossDistributionConfig] = None
    excel_reporting: Optional[ExcelReportConfig] = None
    working_capital_ratios: Optional[WorkingCapitalRatiosConfig] = None
    expense_ratios: Optional[ExpenseRatioConfig] = None
    depreciation: Optional[DepreciationConfig] = None
    industry_config: Optional[IndustryConfig] = None

    # --- Extensibility ---
    custom_modules: Dict[str, ModuleConfig] = {}
    applied_presets: List[str] = []
    overrides: Dict[str, Any] = {}
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `from_profile(profile_path)` | Class method: load from a single YAML file |
| `with_inheritance(profile_path, config_dir)` | Class method: load with recursive inheritance |
| `apply_module(module_path)` | Apply a module YAML file to this config (in-place) |
| `apply_preset(preset_name, preset_data)` | Apply preset parameters (in-place) |
| `with_overrides(**kwargs)` | Return a new ConfigV2 with overrides (supports `section__field` notation) |
| `validate_completeness()` | Return list of missing or inconsistent items |
| `_deep_merge(base, override)` | Static method: recursive dictionary merge |

### Sub-Model Reference

#### ProfileMetadata

Metadata attached to every configuration profile.

```python
class ProfileMetadata(BaseModel):
    name: str               # Alphanumeric + hyphens/underscores
    description: str
    version: str = "2.0.0"  # Semantic version (validated via regex)
    extends: Optional[str] = None    # Parent profile name
    includes: List[str] = []         # Module names to include
    presets: Dict[str, str] = {}     # {preset_type: preset_name}
    author: Optional[str] = None
    created: Optional[datetime] = None
    tags: List[str] = []             # Discovery tags
```

#### ManufacturerConfig

Core financial parameters for the simulated business entity.

```python
class ManufacturerConfig(BaseModel):
    initial_assets: float          # > 0, starting asset value in dollars
    asset_turnover_ratio: float    # > 0, <= 5
    base_operating_margin: float   # > -1, < 1 (warns if > 0.3 or < 0)
    tax_rate: float                # [0, 1]
    retention_ratio: float         # [0, 1]
    ppe_ratio: Optional[float]     # Auto-set based on margin if None
    insolvency_tolerance: float    # Default $10,000
    expense_ratios: Optional[ExpenseRatioConfig]  # COGS/SG&A breakdown

    # Mid-year liquidity (Issue #279)
    premium_payment_month: int     # 0-11, month of premium payment
    revenue_pattern: Literal["uniform", "seasonal", "back_loaded"]
    check_intra_period_liquidity: bool = True
```

Factory method: `ManufacturerConfig.from_industry_config(industry_config, **kwargs)` creates a config from an `IndustryConfig` instance.

#### InsuranceConfig

Enhanced insurance program configuration with layered structure.

```python
class InsuranceConfig(BaseModel):
    enabled: bool = True
    layers: List[InsuranceLayerConfig] = []   # Validated: no overlaps, sorted by attachment
    deductible: float = 0
    coinsurance: float = 1.0       # (0, 1]
    waiting_period_days: int = 0
    claims_handling_cost: float = 0.05  # [0, 1]
```

Each layer is defined by `InsuranceLayerConfig`:

```python
class InsuranceLayerConfig(BaseModel):
    name: str
    limit: float                # > 0
    attachment: float           # >= 0
    base_premium_rate: float    # (0, 1]
    reinstatements: int = 0
    aggregate_limit: Optional[float] = None
    limit_type: str = "per-occurrence"  # "per-occurrence", "aggregate", "hybrid"
    per_occurrence_limit: Optional[float] = None
```

#### SimulationConfig

Controls simulation execution parameters.

```python
class SimulationConfig(BaseModel):
    time_resolution: Literal["annual", "monthly"] = "annual"
    time_horizon_years: int     # > 0, <= 1000
    max_horizon_years: int = 1000  # [100, 10000]
    random_seed: Optional[int] = None
    fiscal_year_end: int = 12   # [1, 12], month of fiscal year end
```

Cross-field validation ensures `time_horizon_years <= max_horizon_years`.

#### LossDistributionConfig

Configuration for actuarial loss modeling.

```python
class LossDistributionConfig(BaseModel):
    frequency_distribution: str = "poisson"  # poisson | negative_binomial | binomial
    frequency_annual: float       # > 0
    severity_distribution: str = "lognormal"  # lognormal | gamma | pareto | weibull
    severity_mean: float          # > 0
    severity_std: float           # > 0
    correlation_factor: float = 0.0  # [-1, 1]
    tail_alpha: float = 2.0      # > 1
```

#### GrowthConfig

Growth model selection and parameters.

```python
class GrowthConfig(BaseModel):
    type: Literal["deterministic", "stochastic"] = "deterministic"
    annual_growth_rate: float    # [-0.5, 1.0]
    volatility: float = 0.0     # [0, 1], must be > 0 when type="stochastic"
```

#### DebtConfig

Debt financing parameters.

```python
class DebtConfig(BaseModel):
    interest_rate: float          # [0, 0.5]
    max_leverage_ratio: float     # [0, 10]
    minimum_cash_balance: float   # >= 0
```

#### WorkingCapitalConfig

Basic working capital as a fraction of sales.

```python
class WorkingCapitalConfig(BaseModel):
    percent_of_sales: float      # [0, 1], raises ValueError if > 0.5
```

#### WorkingCapitalRatiosConfig (Optional)

Detailed working capital with standard financial ratios.

```python
class WorkingCapitalRatiosConfig(BaseModel):
    days_sales_outstanding: float = 45     # [0, 365]
    days_inventory_outstanding: float = 60 # [0, 365]
    days_payable_outstanding: float = 30   # [0, 365]
    # Warns if cash conversion cycle < 0 or > 180 days
```

#### ExpenseRatioConfig (Optional)

COGS and SG&A breakdown ratios (Issue #255).

```python
class ExpenseRatioConfig(BaseModel):
    gross_margin_ratio: float = 0.15         # (0, 1)
    sga_expense_ratio: float = 0.07          # (0, 1)
    manufacturing_depreciation_allocation: float = 0.7  # [0, 1]
    admin_depreciation_allocation: float = 0.3          # [0, 1]

    # COGS breakdown (must sum to 1.0)
    direct_materials_ratio: float = 0.4
    direct_labor_ratio: float = 0.3
    manufacturing_overhead_ratio: float = 0.3

    # SG&A breakdown (must sum to 1.0)
    selling_expense_ratio: float = 0.4
    general_admin_ratio: float = 0.6

    # Computed properties
    @property
    def cogs_ratio(self) -> float: ...
    @property
    def operating_margin_ratio(self) -> float: ...
```

#### DepreciationConfig (Optional)

Depreciation and amortization tracking.

```python
class DepreciationConfig(BaseModel):
    ppe_useful_life_years: float = 10              # (0, 50]
    prepaid_insurance_amortization_months: int = 12 # (0, 24]
    initial_accumulated_depreciation: float = 0

    @property
    def annual_depreciation_rate(self) -> float: ...
    @property
    def monthly_insurance_amortization_rate(self) -> float: ...
```

#### OutputConfig and LoggingConfig

```python
class OutputConfig(BaseModel):
    output_directory: str = "outputs"
    file_format: Literal["csv", "parquet", "json"] = "csv"
    checkpoint_frequency: int = 0     # 0 = disabled
    detailed_metrics: bool = True

    @property
    def output_path(self) -> Path: ...

class LoggingConfig(BaseModel):
    enabled: bool = True
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: Optional[str] = None
    console_output: bool = True
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

#### ExcelReportConfig (Optional)

```python
class ExcelReportConfig(BaseModel):
    enabled: bool = True
    output_path: str = "./reports"
    include_balance_sheet: bool = True
    include_income_statement: bool = True
    include_cash_flow: bool = True
    include_reconciliation: bool = True
    include_metrics_dashboard: bool = True
    include_pivot_data: bool = True
    engine: str = "auto"  # xlsxwriter | openpyxl | auto | pandas
    currency_format: str = "$#,##0"
    decimal_places: int = 0
    date_format: str = "yyyy-mm-dd"
```

#### ModuleConfig and PresetConfig

Base models for extensibility:

```python
class ModuleConfig(BaseModel):
    module_name: str
    module_version: str = "2.0.0"
    dependencies: List[str] = []
    model_config = {"extra": "allow"}  # Allows additional fields

class PresetConfig(BaseModel):
    preset_name: str
    preset_type: str     # market | layers | risk | optimization | scenario
    description: str
    parameters: Dict[str, Any]
```

### StochasticConfig

**Source:** `ergodic_insurance/stochastic_processes.py`

Standalone Pydantic model for stochastic process parameters, used by all stochastic process implementations (GBM, mean-reverting, jump-diffusion, etc.).

```python
class StochasticConfig(BaseModel):
    """Configuration for stochastic processes."""

    volatility: float    # [0, 2], annual volatility (standard deviation)
    drift: float         # [-1, 1], annual drift rate
    random_seed: Optional[int] = None  # >= 0
    time_step: float = 1.0  # (0, 1], time step in years
```

This model is separate from the main `ConfigV2` hierarchy and is consumed directly by `StochasticProcess` subclasses.

### Config (Legacy)

**Source:** `ergodic_insurance/config.py`

The original configuration model, still used by components that have not yet migrated to `ConfigV2`. It contains only the core required sections (no optional modules or profile metadata).

```python
class Config(BaseModel):
    """Complete configuration for the Ergodic Insurance simulation (legacy)."""

    manufacturer: ManufacturerConfig
    working_capital: WorkingCapitalConfig
    growth: GrowthConfig
    debt: DebtConfig
    simulation: SimulationConfig
    output: OutputConfig
    logging: LoggingConfig
```

**Key methods:** `from_yaml(path)`, `from_dict(data, base_config)`, `override(**kwargs)`, `to_yaml(path)`, `setup_logging()`, `validate_paths()`.

## Class Diagram

```{mermaid}
classDiagram
    class ConfigManager {
        +config_dir: Path
        +profiles_dir: Path
        +modules_dir: Path
        +presets_dir: Path
        -_cache: Dict
        -_preset_libraries: Dict
        +load_profile(profile_name, use_cache, **overrides) ConfigV2
        +list_profiles() List~str~
        +list_modules() List~str~
        +list_presets() Dict
        +get_profile_metadata(profile_name) Dict
        +create_profile(name, description, ...) Path
        +with_preset(config, preset_type, preset_name) ConfigV2
        +with_overrides(config, **overrides) ConfigV2
        +validate(config) List~str~
        +clear_cache() void
        -_load_with_inheritance(profile_path) ConfigV2
        -_apply_module(config, module_name) void
        -_apply_preset(config, preset_type, preset_name) void
        -_deep_merge(base, override) Dict
        -_validate_structure() void
    }

    class ConfigV2 {
        +profile: ProfileMetadata
        +manufacturer: ManufacturerConfig
        +working_capital: WorkingCapitalConfig
        +growth: GrowthConfig
        +debt: DebtConfig
        +simulation: SimulationConfig
        +output: OutputConfig
        +logging: LoggingConfig
        +insurance: Optional~InsuranceConfig~
        +losses: Optional~LossDistributionConfig~
        +excel_reporting: Optional~ExcelReportConfig~
        +working_capital_ratios: Optional~WorkingCapitalRatiosConfig~
        +expense_ratios: Optional~ExpenseRatioConfig~
        +depreciation: Optional~DepreciationConfig~
        +industry_config: Optional~IndustryConfig~
        +custom_modules: Dict
        +applied_presets: List~str~
        +overrides: Dict
        +from_profile(profile_path)$ ConfigV2
        +with_inheritance(profile_path, config_dir)$ ConfigV2
        +apply_module(module_path) void
        +apply_preset(preset_name, preset_data) void
        +with_overrides(**kwargs) ConfigV2
        +validate_completeness() List~str~
    }

    class ProfileMetadata {
        +name: str
        +description: str
        +version: str
        +extends: Optional~str~
        +includes: List~str~
        +presets: Dict
        +author: Optional~str~
        +created: Optional~datetime~
        +tags: List~str~
    }

    class ManufacturerConfig {
        +initial_assets: float
        +asset_turnover_ratio: float
        +base_operating_margin: float
        +tax_rate: float
        +retention_ratio: float
        +ppe_ratio: Optional~float~
        +insolvency_tolerance: float
        +expense_ratios: Optional~ExpenseRatioConfig~
        +premium_payment_month: int
        +revenue_pattern: str
        +check_intra_period_liquidity: bool
        +from_industry_config(industry_config, **kwargs)$ ManufacturerConfig
    }

    class InsuranceConfig {
        +enabled: bool
        +layers: List~InsuranceLayerConfig~
        +deductible: float
        +coinsurance: float
        +waiting_period_days: int
        +claims_handling_cost: float
    }

    class SimulationConfig {
        +time_resolution: str
        +time_horizon_years: int
        +max_horizon_years: int
        +random_seed: Optional~int~
        +fiscal_year_end: int
    }

    class StochasticConfig {
        +volatility: float
        +drift: float
        +random_seed: Optional~int~
        +time_step: float
    }

    class LegacyConfigAdapter {
        +config_manager: ConfigManager
        -_profile_mapping: Dict
        +load(config_name, override_params, **kwargs) Config
        +load_config(config_path, config_name, **overrides) Config
        -_convert_to_legacy(config_v2) Config
        -_load_legacy_direct(config_name, overrides) Config
        -_flatten_dict(d, parent_key) Dict
    }

    class ConfigTranslator {
        +legacy_to_v2(legacy_config)$ Dict
        +v2_to_legacy(config_v2)$ Dict
        +validate_translation(original, translated)$ bool
    }

    class ConfigMigrator {
        +legacy_dir: Path
        +new_dir: Path
        +migration_report: List~str~
        +run_migration() bool
        +convert_baseline() Dict
        +convert_conservative() Dict
        +convert_optimistic() Dict
        +extract_modules() void
        +create_presets() void
        +validate_migration() bool
        +generate_migration_report() str
    }

    class ConfigLoader {
        <<deprecated>>
        +config_dir: Path
        +load(config_name, overrides, **kwargs) Config
        +load_scenario(scenario, overrides, **kwargs) Config
        +compare_configs(config1, config2) Dict
        +validate_config(config) bool
        +load_pricing_scenarios(scenario_file) PricingScenarioConfig
        +list_available_configs() List~str~
        +clear_cache() void
    }

    ConfigManager --> ConfigV2 : creates
    ConfigV2 *-- ProfileMetadata
    ConfigV2 *-- ManufacturerConfig
    ConfigV2 *-- InsuranceConfig
    ConfigV2 *-- SimulationConfig
    LegacyConfigAdapter --> ConfigManager : delegates to
    LegacyConfigAdapter --> Config : returns
    ConfigLoader --> LegacyConfigAdapter : delegates to
    ConfigMigrator ..> ConfigV2 : produces YAML for
```

## Backward Compatibility

### ConfigCompat (LegacyConfigAdapter)

**Source:** `ergodic_insurance/config_compat.py`

Maps the legacy `ConfigLoader` interface to the new `ConfigManager`. This is the primary backward compatibility mechanism.

```python
class LegacyConfigAdapter:
    """Adapter for old ConfigLoader interface."""

    def __init__(self):
        # Internal ConfigManager with correct config directory
        self.config_manager = ConfigManager(config_dir)

        # Legacy name mapping
        self._profile_mapping = {
            "baseline": "default",
            "conservative": "conservative",
            "optimistic": "aggressive",
            "aggressive": "aggressive",
        }

    def load(self, config_name: str, override_params=None, **kwargs) -> Config:
        """Load config using old interface.
        1. Maps legacy name to profile name
        2. Loads via ConfigManager.load_profile()
        3. Converts ConfigV2 -> Config via _convert_to_legacy()
        4. Falls back to direct YAML loading if profile not found
        """
```

**Name mapping:**

| Legacy Name | Profile Name |
|-------------|-------------|
| `baseline` | `default` |
| `conservative` | `conservative` |
| `optimistic` | `aggressive` |
| `aggressive` | `aggressive` |

### ConfigTranslator

**Source:** `ergodic_insurance/config_compat.py`

Static utility methods for converting between `Config` and `ConfigV2` formats:

- `legacy_to_v2(legacy_config)` -- Adds a `ProfileMetadata` wrapper and returns a dict suitable for `ConfigV2(**data)`
- `v2_to_legacy(config_v2)` -- Extracts only the 7 legacy sections (`manufacturer`, `working_capital`, `growth`, `debt`, `simulation`, `output`, `logging`)
- `validate_translation(original, translated)` -- Checks that critical fields (`initial_assets`, `time_horizon_years`, `annual_growth_rate`) match after conversion

### ConfigLoader (Deprecated)

**Source:** `ergodic_insurance/config_loader.py`

The original configuration loader. Now delegates all loading to `LegacyConfigAdapter` internally. Emits a `DeprecationWarning` on first use.

```python
class ConfigLoader:
    """Deprecated. Use ConfigManager instead."""

    def load(self, config_name="baseline", overrides=None, **kwargs) -> Config: ...
    def load_scenario(self, scenario, overrides=None, **kwargs) -> Config: ...
    def compare_configs(self, config1, config2) -> Dict: ...
    def validate_config(self, config) -> bool: ...
    def load_pricing_scenarios(self, scenario_file) -> PricingScenarioConfig: ...
    def list_available_configs(self) -> List[str]: ...
    def clear_cache(self) -> None: ...
```

### Migration Helper

A standalone function `migrate_config_usage(file_path)` in `config_compat.py` automates import and class name replacements in Python source files:

- `ConfigLoader` -> `ConfigManager`
- `ConfigLoader.load(` -> `ConfigManager().load_profile(`
- Creates `.bak` backup before modifying files

## Migration Path

### ConfigMigrator

**Source:** `ergodic_insurance/config_migrator.py`

Automated migration tool that converts the legacy flat YAML files into the 3-tier directory structure.

```python
class ConfigMigrator:
    """Migrates legacy configurations to new system."""

    def run_migration(self) -> bool:
        """Execute full migration pipeline:
        1. convert_baseline()     -> profiles/default.yaml
        2. convert_conservative() -> profiles/conservative.yaml
        3. convert_optimistic()   -> profiles/aggressive.yaml
        4. extract_modules()      -> modules/{insurance,losses,stochastic,business}.yaml
        5. create_presets()       -> presets/{market_conditions,layer_structures,risk_scenarios}.yaml
        6. validate_migration()   -> verify all expected files exist
        """

    def generate_migration_report(self) -> str:
        """Generate formatted migration report with status of each step."""
```

**Module extraction:**

| Legacy Files | New Module |
|-------------|-----------|
| `insurance.yaml`, `insurance_market.yaml`, `insurance_structures.yaml`, `insurance_pricing_scenarios.yaml` | `modules/insurance.yaml` |
| `losses.yaml`, `loss_distributions.yaml` | `modules/losses.yaml` |
| `stochastic.yaml` | `modules/stochastic.yaml` |
| `business_optimization.yaml` | `modules/business.yaml` |

**Preset generation:**

| Preset Library | Presets |
|---------------|---------|
| `market_conditions.yaml` | `stable`, `volatile`, `growth`, `recession` |
| `layer_structures.yaml` | `basic`, `comprehensive`, `catastrophic` |
| `risk_scenarios.yaml` | `low_risk`, `moderate_risk`, `high_risk` |

Run the migrator from the command line:

```bash
python -m ergodic_insurance.config_migrator
```

## Industry Configuration

The system includes industry-specific configuration templates via the `IndustryConfig` dataclass hierarchy:

| Config Class | Industry | Gross Margin | PP&E Ratio | DSO |
|-------------|----------|-------------|-----------|-----|
| `ManufacturingConfig` | Manufacturing | 35% | 50% | 45 days |
| `ServiceConfig` | Services | 60% | 20% | 30 days |
| `RetailConfig` | Retail | 30% | 40% | 5 days |

These feed into `ManufacturerConfig.from_industry_config()` to create appropriately parameterized simulations.

## Profile System

Profiles provide complete configuration sets:

```yaml
# profiles/default.yaml
profile:
  name: default
  description: "Standard baseline configuration"
  version: "2.0.0"

modules:
  - insurance
  - losses

manufacturer:
  initial_assets: 10_000_000
  base_operating_margin: 0.10
  tax_rate: 0.25
```

**Inheritance:**
```yaml
# profiles/conservative.yaml
profile:
  name: conservative
  description: "Conservative risk parameters"
  extends: default
  version: "2.0.0"

manufacturer:
  base_operating_margin: 0.06
growth:
  annual_growth_rate: 0.03
```

## Module System

Modules are optional, reusable components:

```yaml
# modules/insurance.yaml
insurance:
  enabled: true
  layers:
    - name: "Primary"
      attachment: 0
      limit: 5_000_000
      base_premium_rate: 0.015
    - name: "Excess"
      attachment: 5_000_000
      limit: 20_000_000
      base_premium_rate: 0.008
```

**Benefits:**

- Selective inclusion via profile's `includes` list
- Reusability across profiles
- Clean separation of concerns
- Easy testing in isolation

## Preset System

Presets are quick-apply templates:

```yaml
# presets/market_conditions.yaml
stable:
  manufacturer:
    revenue_volatility: 0.10
    cost_volatility: 0.08
  simulation:
    stochastic_revenue: false

volatile:
  manufacturer:
    revenue_volatility: 0.25
    cost_volatility: 0.20
  simulation:
    stochastic_revenue: true
    stochastic_costs: true
```

**Usage:**
```python
config = manager.load_profile(
    "default",
    presets=["hard_market", "high_volatility"]
)
```

## Performance Considerations

### Caching Strategy

The `ConfigManager` uses a dictionary-based cache keyed by a SHA-256 hash of the profile name and serialized overrides:

```python
cache_key = f"{profile_name}_{hashlib.sha256(
    json.dumps(overrides, sort_keys=True, default=str).encode()
).hexdigest()[:16]}"
```

Profile metadata lookups are cached separately with `@lru_cache(maxsize=32)`.

### Optimization Techniques

1. **SHA-256 cache keys**: Deterministic hashing of profile + overrides for fast lookup
2. **Lazy module loading**: Modules loaded only when listed in `profile.includes`
3. **Deep merging**: Efficient recursive dictionary merge for inheritance chains
4. **Preset library caching**: Preset YAML files loaded once and reused
5. **Pydantic v2 validation**: Compiled validators for fast model construction
6. **LRU caching**: `get_profile_metadata()` avoids repeated file I/O

## Best Practices

### Creating Custom Profiles

```yaml
# profiles/custom/client_abc.yaml
profile:
  name: client-abc
  description: "Custom configuration for Client ABC"
  extends: conservative
  version: "2.0.0"
  includes:
    - insurance
    - stochastic
  presets:
    market_conditions: stable

manufacturer:
  initial_assets: 50_000_000
simulation:
  time_horizon_years: 30
```

### Runtime Overrides

```python
# Override specific parameters at runtime
config = manager.load_profile(
    "default",
    manufacturer={"base_operating_margin": 0.12},
    simulation={"random_seed": 42}
)

# Double-underscore notation for nested overrides
config = config.with_overrides(
    manufacturer__initial_assets=20_000_000,
    simulation__time_horizon_years=100
)
```

### Testing Configurations

```python
def test_profile_loads():
    """Test that all profiles load successfully."""
    manager = ConfigManager()

    for profile in manager.list_profiles():
        config = manager.load_profile(profile)
        assert isinstance(config, ConfigV2)
        assert config.manufacturer.initial_assets > 0

def test_validation():
    """Test configuration validation."""
    manager = ConfigManager()
    config = manager.load_profile("default")
    issues = manager.validate(config)
    assert len(issues) == 0
```

### Migrating from Legacy Code

```python
# Before (deprecated)
from ergodic_insurance.config_loader import ConfigLoader
loader = ConfigLoader()
config = loader.load("baseline", overrides={"manufacturer": {"initial_assets": 5e6}})

# After (recommended)
from ergodic_insurance.config_manager import ConfigManager
manager = ConfigManager()
config = manager.load_profile("default", manufacturer={"initial_assets": 5e6})
```

## Security Considerations

1. **Path traversal protection**: Profile names are resolved within the configured directory structure only
2. **YAML safe loading**: All YAML files loaded with `yaml.safe_load()` (no arbitrary code execution)
3. **Pydantic validation**: All inputs validated through Pydantic field constraints and model validators
4. **Access control**: Custom profiles stored in a separate `custom/` subdirectory
5. **YAML anchor filtering**: Private anchors (keys starting with `_`) are stripped before parsing

## Future Enhancements

### Planned Features

1. **Hot reloading**: Automatic reload on file changes during development
2. **Schema versioning**: Automatic migration between configuration schema versions
3. **Remote configs**: Load from S3/HTTP endpoints
4. **Config diffing**: Visual comparison of configurations (partially implemented via `ConfigLoader.compare_configs()`)
5. **A/B testing**: Built-in experiment configuration support

### API Stability

The configuration API is considered stable as of v2.0:

- `ConfigManager.load_profile()` -- **Stable**
- `ConfigV2` models -- **Stable**
- Profile YAML format -- **Stable**
- Module/Preset format -- **Stable**
- `ConfigLoader` -- **Deprecated** (will be removed in v3.0.0)
- `LegacyConfigAdapter` -- **Transitional** (will be removed in v3.0.0)

## Conclusion

The v2.0 configuration system provides a modern, flexible, and maintainable approach to managing complex simulation parameters while maintaining full backward compatibility. The 3-tier architecture enables both simplicity for basic use cases and power for advanced scenarios. The `ConfigManager` serves as the single entry point, with `LegacyConfigAdapter` providing a seamless bridge for existing code, and `ConfigMigrator` offering automated tooling for transitioning legacy YAML files to the new structure.
