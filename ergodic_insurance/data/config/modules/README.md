# Configuration Modules

This directory contains reusable configuration fragments that can be included in profiles.

## Available Modules

- **insurance.yaml** - Insurance program configuration (layers, premiums, etc.)
- **losses.yaml** - Loss distribution parameters
- **simulation.yaml** - Simulation settings (duration, timesteps, etc.)
- **stochastic.yaml** - Stochastic process parameters
- **manufacturer.yaml** - Manufacturer-specific settings

## Module Structure

Each module contains a subset of the full configuration:

```yaml
# insurance.yaml
insurance:
  layers:
    - name: Primary
      limit: 5_000_000
      attachment: 0
      premium_rate: 0.015
```

## Usage in Profiles

Modules are included via the profile's `includes` field:

```yaml
profile:
  name: my-profile
  includes:
    - insurance
    - losses
```

Modules are merged in order, with later modules overriding earlier ones.
