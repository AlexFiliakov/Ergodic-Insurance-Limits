# Configuration System v2.0 Architecture

## Overview

The configuration system has been completely redesigned in v2.0 to provide a modern, flexible, and maintainable approach to managing simulation parameters.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     ConfigManager                           │
│                  (Main Interface)                           │
├─────────────────────────────────────────────────────────────┤
│  - load_profile()                                           │
│  - list_profiles()                                          │
│  - validate_config()                                        │
│  - _cache: Dict[str, ConfigV2]                             │
└──────────┬──────────────────────────────┬───────────────────┘
           │                              │
           ▼                              ▼
┌──────────────────────┐      ┌──────────────────────┐
│      Profiles        │      │     ConfigV2         │
│   (YAML Files)       │      │  (Pydantic Models)   │
├──────────────────────┤      ├──────────────────────┤
│ - default.yaml       │      │ - ProfileMetadata    │
│ - conservative.yaml  │      │ - ManufacturerConfig │
│ - aggressive.yaml    │      │ - InsuranceConfig    │
│ - custom/*.yaml      │      │ - SimulationConfig   │
└──────────┬───────────┘      └──────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│                  3-Tier Structure                     │
├──────────────────────────────────────────────────────┤
│  Profiles ──extends──> Base Profile                   │
│     │                                                 │
│     ├──includes──> Modules                          │
│     │              - insurance.yaml                  │
│     │              - losses.yaml                     │
│     │              - stochastic.yaml                 │
│     │              - business.yaml                   │
│     │                                                │
│     └──applies──> Presets                           │
│                    - market_conditions.yaml          │
│                    - layer_structures.yaml           │
│                    - risk_scenarios.yaml             │
└──────────────────────────────────────────────────────┘
```

## Component Descriptions

### ConfigManager

The main interface for configuration management:

```python
class ConfigManager:
    """Manages configuration loading with profiles, modules, and presets."""

    def load_profile(
        self,
        profile_name: str = "default",
        use_cache: bool = True,
        **overrides
    ) -> ConfigV2:
        """Load a configuration profile with optional overrides."""
```

**Key Features:**
- Profile inheritance resolution
- Module composition
- Preset application
- Runtime overrides
- LRU caching for performance
- Validation at load time

### ConfigV2 Models

Enhanced Pydantic v2 models with strict validation:

```python
class ConfigV2(BaseModel):
    """Main configuration model with all settings."""

    profile: ProfileMetadata
    manufacturer: ManufacturerConfig
    working_capital: WorkingCapitalConfig
    growth: GrowthConfig
    debt: DebtConfig
    simulation: SimulationConfig
    insurance: Optional[InsuranceConfig]
    losses: Optional[LossDistributionConfig]
    stochastic: Optional[StochasticConfig]
    business: Optional[BusinessOptimizationConfig]
```

**Enhancements:**
- Type safety with Pydantic v2
- Optional module support
- Nested configuration validation
- Runtime override methods
- JSON schema generation

### Profile System

Profiles provide complete configuration sets:

```yaml
# profiles/default.yaml
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
extends: default
description: "Conservative risk parameters"

overrides:
  manufacturer:
    base_operating_margin: 0.06
  growth:
    annual_growth_rate: 0.03
```

### Module System

Modules are optional, reusable components:

```yaml
# modules/insurance.yaml
insurance:
  use_program: true
  layers:
    - name: "Primary"
      attachment: 0
      limit: 5_000_000
      premium_rate: 0.015
    - name: "Excess"
      attachment: 5_000_000
      limit: 20_000_000
      premium_rate: 0.008
```

**Benefits:**
- Selective inclusion
- Reusability across profiles
- Clean separation of concerns
- Easy testing in isolation

### Preset System

Presets are quick-apply templates:

```yaml
# presets/market_conditions.yaml
presets:
  hard_market:
    insurance:
      layers:
        - premium_rate: 0.025  # Higher rates
        - premium_rate: 0.015

  soft_market:
    insurance:
      layers:
        - premium_rate: 0.010  # Lower rates
        - premium_rate: 0.005
```

**Usage:**
```python
config = manager.load_profile(
    "default",
    presets=["hard_market", "high_volatility"]
)
```

## Migration Path

### Backward Compatibility

The system maintains full backward compatibility through `ConfigCompat`:

```python
class LegacyConfigAdapter:
    """Adapter for old ConfigLoader interface."""

    def load(self, config_name: str, **overrides) -> Config:
        """Load config using old interface."""
        # Maps to new system internally
        profile_name = self._map_legacy_name(config_name)
        config_v2 = self.config_manager.load_profile(profile_name)
        return self._convert_to_legacy(config_v2)
```

### Migration Tool

Automated migration from 12 YAML files to 3-tier:

```python
class ConfigMigrator:
    """Migrates legacy configurations to new system."""

    def run_migration(self) -> bool:
        """Execute full migration."""
        # 1. Convert baseline → default profile
        # 2. Extract modules from multiple files
        # 3. Create preset libraries
        # 4. Validate migrated configs
```

## Performance Considerations

### Caching Strategy

```python
@lru_cache(maxsize=32)
def _load_with_inheritance(self, profile_path: Path) -> dict:
    """Load profile with inheritance chain resolution."""
    # Cached to avoid repeated file I/O
```

### Optimization Techniques

1. **Lazy Loading**: Modules loaded only when needed
2. **Deep Merging**: Efficient nested dictionary merging
3. **Validation Caching**: Pydantic model validation results cached
4. **File Watch**: Optional file system watching for development

## Best Practices

### Creating Custom Profiles

```yaml
# profiles/custom/client_abc.yaml
extends: conservative
description: "Custom configuration for Client ABC"

modules:
  - insurance
  - stochastic

presets:
  - steady_market

overrides:
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
```

## Security Considerations

1. **Path Traversal Protection**: Profile names sanitized
2. **YAML Safe Loading**: Only safe constructors allowed
3. **Validation**: All inputs validated through Pydantic
4. **Access Control**: Custom profiles in separate directory

## Future Enhancements

### Planned Features

1. **Hot Reloading**: Automatic reload on file changes
2. **Schema Versioning**: Automatic migration between versions
3. **Remote Configs**: Load from S3/HTTP endpoints
4. **Config Diffing**: Compare configurations
5. **A/B Testing**: Built-in experiment configuration

### API Stability

The configuration API is considered stable as of v2.0:
- `ConfigManager.load_profile()` - Stable
- `ConfigV2` models - Stable
- Profile YAML format - Stable
- Module/Preset format - Stable

## Conclusion

The v2.0 configuration system provides a modern, flexible, and maintainable approach to managing complex simulation parameters while maintaining full backward compatibility. The 3-tier architecture enables both simplicity for basic use cases and power for advanced scenarios.
