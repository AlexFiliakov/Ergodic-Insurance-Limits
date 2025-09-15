# Configuration System Migration Guide

## Overview
This guide helps you migrate from the legacy 12-file YAML configuration system to the new simplified 3-tier architecture.

## What's New

### 3-Tier Architecture
The new system organizes configuration into three clear layers:

1. **Profiles** (`data/config/profiles/`): Complete configuration sets (default, conservative, aggressive)
2. **Modules** (`data/config/modules/`): Reusable components (insurance, losses, stochastic, business)
3. **Presets** (`data/config/presets/`): Quick-apply templates (market conditions, layer structures, risk scenarios)

### Key Benefits
- **50% fewer files** to manage (12 → 6 core files)
- **Profile inheritance** for easy customization
- **Module composition** for flexible configurations
- **Preset libraries** for common scenarios
- **Full backward compatibility** during transition

## Migration Path

### Phase 1: Immediate Migration (Recommended)
Use the new `ConfigManager` for all new code:

```python
# Old way (deprecated)
from ergodic_insurance.config_loader import ConfigLoader
loader = ConfigLoader()
config = loader.load("baseline")

# New way (recommended)
from ergodic_insurance.config_manager import ConfigManager
manager = ConfigManager()
config = manager.load_profile("default")  # Note: "baseline" → "default"
```

### Phase 2: Gradual Migration
Existing code continues to work with deprecation warnings:

```python
# This still works but shows a deprecation warning
from ergodic_insurance.config_loader import ConfigLoader
loader = ConfigLoader()
config = loader.load("baseline")  # Automatically mapped to new system
```

## Common Migration Scenarios

### Loading a Basic Configuration

**Old:**
```python
from ergodic_insurance.config_loader import ConfigLoader

loader = ConfigLoader()
config = loader.load("baseline")
```

**New:**
```python
from ergodic_insurance.config_manager import ConfigManager

manager = ConfigManager()
config = manager.load_profile("default")
```

### Loading with Overrides

**Old:**
```python
config = loader.load(
    "conservative",
    overrides={"manufacturer": {"base_operating_margin": 0.12}}
)
```

**New:**
```python
config = manager.load_profile(
    "conservative",
    manufacturer={"base_operating_margin": 0.12}
)
```

### Using Presets

**New capability** - apply preset templates:

```python
# Apply a market condition preset
config = manager.load_profile(
    "default",
    presets=["hard_market"]  # Automatically adjusts insurance rates
)

# Apply multiple presets
config = manager.load_profile(
    "default",
    presets=["hard_market", "high_volatility"]
)
```

### Creating Custom Profiles

**Old:** Required editing YAML files directly

**New:** Create custom profiles with inheritance:

```yaml
# data/config/profiles/custom/my_scenario.yaml
extends: default
description: "My custom scenario with high growth"

modules:
  - insurance
  - stochastic

overrides:
  manufacturer:
    annual_growth_rate: 0.15
  insurance:
    use_program: true
```

Load it:
```python
config = manager.load_profile("custom/my_scenario")
```

## Profile Name Mapping

| Old Name | New Name | Notes |
|----------|----------|--------|
| baseline | default | Standard configuration |
| conservative | conservative | Risk-averse parameters |
| optimistic | aggressive | Growth-oriented settings |

## Module System

Modules can be selectively included:

```python
# Load only specific modules
config = manager.load_profile(
    "default",
    modules=["insurance", "stochastic"]  # Exclude others
)
```

## Advanced Features

### Configuration Caching
The new system includes automatic caching:

```python
# First load reads from disk
config1 = manager.load_profile("default")

# Subsequent loads use cache (fast)
config2 = manager.load_profile("default")

# Bypass cache if needed
config3 = manager.load_profile("default", use_cache=False)
```

### Profile Inheritance
Create variations easily:

```python
# Start with conservative base
config = manager.load_profile("conservative")

# Create a variant with just one change
variant = config.with_overrides(
    manufacturer={"base_operating_margin": 0.15}
)
```

## Testing Your Migration

1. **Run existing tests**: All should pass with the compatibility layer
2. **Check for warnings**: Look for deprecation warnings in test output
3. **Validate configurations**: Use the built-in validation

```python
# Validate a configuration
from ergodic_insurance.config_v2 import ConfigV2

config = manager.load_profile("default")
assert isinstance(config, ConfigV2)  # New config type
```

## Troubleshooting

### Common Issues

1. **"ConfigLoader is deprecated" warning**
   - This is expected during migration
   - Update to ConfigManager when convenient

2. **Profile not found**
   - Check profile name mapping (baseline → default)
   - Ensure migration was run: `ConfigMigrator().run_migration()`

3. **Module not loaded**
   - Check modules list in profile
   - Verify module file exists in `data/config/modules/`

### Getting Help

- Check examples in `ergodic_insurance/examples/`
- Review test files for usage patterns
- See API documentation for ConfigManager

## Timeline

- **Phase 1-2** (Current): Foundation and compatibility layer ✅
- **Phase 3** (In Progress): Migrating existing code
- **Phase 4** (Upcoming): Documentation and cleanup
- **Version 3.0**: Legacy system removal (future release)

## Best Practices

1. **Use ConfigManager for new code**: Don't use ConfigLoader in new development
2. **Leverage presets**: Use preset libraries for common scenarios
3. **Create custom profiles**: Don't modify core profiles directly
4. **Cache wisely**: Use caching for performance, disable for testing
5. **Validate early**: Check configurations at startup

## Example: Complete Migration

Here's a complete example showing old vs new approaches:

### Old Approach
```python
from ergodic_insurance.config_loader import ConfigLoader
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.simulation import run_simulation

# Load configuration
loader = ConfigLoader()
config = loader.load("baseline", overrides={
    "manufacturer": {"base_operating_margin": 0.10},
    "simulation": {"time_horizon_years": 100}
})

# Create manufacturer
manufacturer = WidgetManufacturer(config.manufacturer)

# Run simulation
results = run_simulation(manufacturer, config.simulation)
```

### New Approach
```python
from ergodic_insurance.config_manager import ConfigManager
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.simulation import run_simulation

# Load configuration with new system
manager = ConfigManager()
config = manager.load_profile(
    "default",
    presets=["steady_market"],  # Use a preset
    manufacturer={"base_operating_margin": 0.10},
    simulation={"time_horizon_years": 100}
)

# Create manufacturer (same as before)
manufacturer = WidgetManufacturer(config.manufacturer)

# Run simulation (same as before)
results = run_simulation(manufacturer, config.simulation)
```

## Next Steps

1. **Update imports**: Replace ConfigLoader with ConfigManager
2. **Update profile names**: baseline → default, optimistic → aggressive
3. **Leverage new features**: Use presets and inheritance
4. **Remove warnings**: Address deprecation warnings
5. **Contribute**: Create custom profiles for your use cases

---

*This migration guide is part of the configuration system upgrade. For questions or issues, please refer to the project documentation or create an issue in the repository.*
