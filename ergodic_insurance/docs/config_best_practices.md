# Configuration Best Practices

This guide provides best practices for using and extending the configuration system effectively.

## Core Principles

### 1. Use Profiles for Complete Scenarios
Profiles should represent complete, self-contained configurations:

```yaml
# profiles/high_growth.yaml
extends: default
description: "High growth scenario with increased risk tolerance"

overrides:
  manufacturer:
    base_operating_margin: 0.12
    retention_ratio: 0.9
  growth:
    annual_growth_rate: 0.20
    volatility: 0.25
```

### 2. Leverage Inheritance
Don't repeat yourself - extend existing profiles:

```yaml
# profiles/custom/client_abc.yaml
extends: conservative  # Build on conservative base
description: "Custom configuration for Client ABC requirements"

modules:
  - insurance
  - stochastic

overrides:
  manufacturer:
    initial_assets: 50_000_000  # Client-specific asset base
```

### 3. Use Modules for Optional Features
Modules should be self-contained and optional:

```yaml
# Only load what you need
modules:
  - insurance      # Include insurance parameters
  - stochastic     # Include stochastic modeling
  # losses        # Excluded - not needed for this analysis
```

## Configuration Patterns

### Pattern 1: Environment-Specific Profiles

Create profiles for different environments:

```python
# Development
config_dev = manager.load_profile("custom/development")

# Testing
config_test = manager.load_profile("custom/testing")

# Production
config_prod = manager.load_profile("custom/production")
```

### Pattern 2: Scenario Analysis

Use runtime overrides for quick scenario comparisons:

```python
scenarios = {
    "base": {},
    "optimistic": {"growth": {"annual_growth_rate": 0.15}},
    "pessimistic": {"growth": {"annual_growth_rate": 0.05}},
    "high_margin": {"manufacturer": {"base_operating_margin": 0.15}}
}

results = {}
for name, overrides in scenarios.items():
    config = manager.load_profile("default", **overrides)
    results[name] = run_simulation(config)
```

### Pattern 3: Preset Combinations

Combine presets for complex scenarios:

```python
# Hard insurance market with high volatility
config = manager.load_profile(
    "default",
    presets=["hard_market", "high_volatility", "conservative_growth"]
)
```

### Pattern 4: Progressive Overrides

Build configurations incrementally:

```python
# Start with base
config = manager.load_profile("default")

# Add market conditions
config = manager.load_profile("default", presets=["soft_market"])

# Add specific overrides
config = manager.load_profile(
    "default",
    presets=["soft_market"],
    manufacturer={"tax_rate": 0.21}
)
```

## Performance Optimization

### 1. Use Caching for Repeated Loads

```python
# Cache is enabled by default
config1 = manager.load_profile("default")  # Reads from disk
config2 = manager.load_profile("default")  # Uses cache (fast)

# Disable cache when files might change
config3 = manager.load_profile("default", use_cache=False)
```

### 2. Load Minimal Modules

```python
# Only load what you need
config = manager.load_profile(
    "default",
    modules=["insurance"]  # Skip stochastic, losses, etc.
)
```

### 3. Reuse ConfigManager Instances

```python
# Good - reuse manager
manager = ConfigManager()
for profile in profiles:
    config = manager.load_profile(profile)

# Avoid - creating new managers
for profile in profiles:
    manager = ConfigManager()  # Unnecessary overhead
    config = manager.load_profile(profile)
```

## Creating Custom Configurations

### Step 1: Choose a Base Profile

```yaml
# Start with the closest existing profile
extends: conservative  # or default, aggressive
```

### Step 2: Add Description

```yaml
description: "Q4 2024 planning scenario with updated tax rates"
```

### Step 3: Select Modules

```yaml
modules:
  - insurance
  - stochastic
  - business  # Add business optimization
```

### Step 4: Apply Overrides

```yaml
overrides:
  manufacturer:
    tax_rate: 0.21  # Updated tax rate
    base_operating_margin: 0.09  # Adjusted margin
  growth:
    annual_growth_rate: 0.10
```

### Step 5: Save and Use

Save as `profiles/custom/q4_2024_planning.yaml`:

```python
config = manager.load_profile("custom/q4_2024_planning")
```

## Validation Best Practices

### 1. Validate Early

```python
try:
    config = manager.load_profile("custom/new_profile")
except FileNotFoundError:
    print("Profile not found")
except ValidationError as e:
    print(f"Invalid configuration: {e}")
```

### 2. Use Type Hints

```python
from ergodic_insurance.config_v2 import ConfigV2

def run_analysis(config: ConfigV2) -> dict:
    """Run analysis with validated configuration."""
    assert isinstance(config, ConfigV2)
    # ... rest of function
```

### 3. Test Configuration Changes

```python
def test_custom_profile():
    """Test that custom profile loads correctly."""
    manager = ConfigManager()
    config = manager.load_profile("custom/my_profile")

    assert config.manufacturer.base_operating_margin > 0
    assert config.growth.annual_growth_rate >= 0
    assert hasattr(config, 'insurance')
```

## Common Pitfalls to Avoid

### 1. ❌ Modifying Core Profiles
```python
# BAD - Don't modify core profiles
edit_file("profiles/default.yaml")

# GOOD - Create custom profile
create_file("profiles/custom/my_default.yaml")
```

### 2. ❌ Hardcoding Paths
```python
# BAD - Hardcoded path
config = load_yaml("/absolute/path/to/config.yaml")

# GOOD - Use ConfigManager
config = manager.load_profile("profile_name")
```

### 3. ❌ Ignoring Validation Errors
```python
# BAD - Suppressing all exceptions
try:
    config = manager.load_profile("profile")
except:
    config = {}  # Silent failure

# GOOD - Handle specific exceptions
try:
    config = manager.load_profile("profile")
except FileNotFoundError:
    config = manager.load_profile("default")  # Fallback
```

### 4. ❌ Deep Nesting in Overrides
```python
# BAD - Too deeply nested
overrides = {
    "manufacturer": {
        "sub_config": {
            "deep_nested": {
                "very_deep": {
                    "value": 42
                }
            }
        }
    }
}

# GOOD - Flatten when possible or use custom profile
```

## Migration from Legacy System

### Gradual Migration Strategy

1. **Phase 1**: Add new ConfigManager alongside old ConfigLoader
2. **Phase 2**: Update new code to use ConfigManager
3. **Phase 3**: Gradually update existing code
4. **Phase 4**: Remove ConfigLoader when ready

### Mapping Old to New

| Old Pattern | New Pattern | Notes |
|------------|-------------|--------|
| `load("baseline")` | `load_profile("default")` | Name change |
| `load("optimistic")` | `load_profile("aggressive")` | Name change |
| Manual YAML editing | Runtime overrides | More flexible |
| Multiple YAML files | Single profile + modules | Cleaner |

## Testing Configurations

### Unit Tests
```python
def test_profile_loads():
    """Test all profiles load successfully."""
    manager = ConfigManager()
    for profile in manager.list_profiles():
        config = manager.load_profile(profile)
        assert isinstance(config, ConfigV2)
```

### Integration Tests
```python
def test_simulation_with_profile():
    """Test simulation runs with custom profile."""
    manager = ConfigManager()
    config = manager.load_profile("custom/test_profile")

    manufacturer = WidgetManufacturer(config.manufacturer)
    result = run_simulation(manufacturer, config.simulation)

    assert result.final_equity > 0
```

## Summary

The key to effective configuration management is:

1. **Use profiles** for complete scenarios
2. **Leverage inheritance** to avoid duplication
3. **Apply runtime overrides** for flexibility
4. **Use presets** for common patterns
5. **Validate early** and handle errors gracefully
6. **Cache appropriately** for performance
7. **Test configurations** as part of your test suite

Following these practices will help you maintain clean, maintainable, and performant configuration management in your project.
