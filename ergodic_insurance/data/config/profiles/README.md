# Configuration Profiles

This directory contains complete configuration profiles for different scenarios.

## Structure

- **default.yaml** - Standard baseline configuration
- **conservative.yaml** - Risk-averse settings with higher safety margins
- **aggressive.yaml** - Growth-focused settings with higher risk tolerance
- **custom/** - User-defined custom profiles (gitignored)

## Profile Features

### Inheritance
Profiles can extend other profiles using the `extends` field:
```yaml
profile:
  name: my-profile
  extends: default
```

### Module Inclusion
Profiles can include configuration modules:
```yaml
profile:
  includes:
    - insurance
    - losses
```

### Preset Application
Profiles can apply presets to quickly configure settings:
```yaml
profile:
  presets:
    market: volatile
    layers: comprehensive
```

## Usage

```python
from ergodic_insurance.config_manager import ConfigManager

# Load a profile
config = ConfigManager().load_profile("conservative")

# With overrides
config = ConfigManager().load_profile("default",
    manufacturer__starting_assets=20_000_000)
```
