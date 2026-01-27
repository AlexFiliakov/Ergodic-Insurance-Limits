# Migration Guide: ClaimGenerator → ManufacturingLossGenerator

## Overview

The `ClaimGenerator` class is deprecated as of version 0.2.0 and will be removed in version 1.0.0. This guide helps you migrate to `ManufacturingLossGenerator`, which provides better statistical independence, multiple loss types, and more sophisticated risk modeling.

## Why Migrate?

### Issues with ClaimGenerator

1. **Fragile Dependencies**: Complex try/except blocks create brittle import chains
2. **Single Responsibility Violation**: Mixes simple and enhanced functionality
3. **Weaker Seed Management**: Uses naive seed incrementing instead of SeedSequence
4. **Limited Loss Modeling**: Only supports simple Poisson/Lognormal distributions

### Benefits of ManufacturingLossGenerator

1. **Better Statistical Independence**: Uses numpy.random.SeedSequence for proper stream independence
2. **Multiple Loss Types**: Attritional, large, and catastrophic losses modeled separately
3. **Extreme Value Modeling**: Optional GPD for tail losses
4. **Simplified Architecture**: Single source of truth for loss generation
5. **Better Maintainability**: Cleaner codebase with fewer edge cases

## Quick Migration (Simple Use Cases)

For basic claim generation, use the `create_simple()` factory method:

### Before (ClaimGenerator)

```python
from ergodic_insurance.claim_generator import ClaimGenerator

generator = ClaimGenerator(
    base_frequency=0.1,
    severity_mean=5_000_000,
    severity_std=2_000_000,
    seed=42
)

claims = generator.generate_claims(years=10)
total_loss = sum(c.amount for c in claims)
```

### After (ManufacturingLossGenerator)

```python
from ergodic_insurance.loss_distributions import ManufacturingLossGenerator

generator = ManufacturingLossGenerator.create_simple(
    frequency=0.1,
    severity_mean=5_000_000,
    severity_std=2_000_000,
    seed=42
)

losses, stats = generator.generate_losses(duration=10, revenue=10_000_000)
total_loss = sum(loss.amount for loss in losses)
```

### Key Differences

| Aspect | ClaimGenerator | ManufacturingLossGenerator |
|--------|---------------|---------------------------|
| Return Type | `List[ClaimEvent]` | `Tuple[List[LossEvent], Dict]` |
| Time Parameter | `years` | `duration` |
| Loss Object | `ClaimEvent(year, amount)` | `LossEvent(amount, time, loss_type)` |
| Statistics | Call separate method | Returned with losses |
| Revenue | Not required | Required parameter |

## Advanced Migration

For advanced use cases with exposure scaling, trends, or custom parameters:

### Before (with ExposureBase)

```python
from ergodic_insurance.claim_generator import ClaimGenerator
from ergodic_insurance.exposure_base import RevenueExposure

exposure = RevenueExposure(base_revenue=10_000_000, growth_rate=0.10)

generator = ClaimGenerator(
    base_frequency=0.1,
    severity_mean=5_000_000,
    exposure_base=exposure,
    seed=42
)

claims = generator.generate_year(year=5)
```

### After (with ExposureBase)

```python
from ergodic_insurance.loss_distributions import ManufacturingLossGenerator
from ergodic_insurance.exposure_base import RevenueExposure

exposure = RevenueExposure(base_revenue=10_000_000, growth_rate=0.10)

generator = ManufacturingLossGenerator.create_simple(
    frequency=0.1,
    severity_mean=5_000_000,
    seed=42
)

# Pass exposure during generation
losses, stats = generator.generate_losses(
    duration=1,
    revenue=exposure.get_exposure(time=5),
    time=5
)
```

## Custom Loss Type Configuration

For full control over loss types, use the standard constructor:

```python
from ergodic_insurance.loss_distributions import ManufacturingLossGenerator

generator = ManufacturingLossGenerator(
    attritional_params={
        "frequency_mean": 0.5,
        "severity_mu": 15.0,
        "severity_sigma": 0.8,
    },
    large_params={
        "frequency_mean": 0.05,
        "severity_alpha": 2.5,
        "severity_threshold": 10_000_000,
    },
    catastrophic_params={
        "frequency_mean": 0.01,
        "severity_mu": 17.0,
        "severity_sigma": 1.2,
    },
    seed=42
)

losses, stats = generator.generate_losses(duration=10, revenue=10_000_000)
```

## Statistics and Analysis

### Before (ClaimGenerator)

```python
generator = ClaimGenerator(base_frequency=0.1, severity_mean=5_000_000, seed=42)
claims = generator.generate_claims(years=100)

# Manual statistics
total_loss = sum(c.amount for c in claims)
claim_count = len(claims)
avg_severity = total_loss / claim_count if claim_count > 0 else 0
```

### After (ManufacturingLossGenerator)

```python
generator = ManufacturingLossGenerator.create_simple(
    frequency=0.1, severity_mean=5_000_000, seed=42
)
losses, stats = generator.generate_losses(duration=100, revenue=10_000_000)

# Statistics included in return value
total_loss = stats['total_amount']
claim_count = stats['total_losses']
avg_severity = stats['average_loss']
annual_frequency = stats['annual_frequency']

# Breakdown by loss type
print(f"Attritional: {stats['attritional_count']} events, ${stats['attritional_amount']:,.0f}")
print(f"Large: {stats['large_count']} events, ${stats['large_amount']:,.0f}")
print(f"Catastrophic: {stats['catastrophic_count']} events, ${stats['catastrophic_amount']:,.0f}")
```

## Working with LossEvent vs ClaimEvent

### ClaimEvent Structure

```python
@dataclass
class ClaimEvent:
    year: int        # Integer year (0-indexed)
    amount: float    # Claim amount in dollars
```

### LossEvent Structure

```python
@dataclass
class LossEvent:
    amount: float      # Loss amount in dollars
    time: float        # Fractional time (e.g., 5.3 = midway through year 5)
    loss_type: str     # "attritional", "large", "catastrophic", or "extreme"
```

### Conversion Example

```python
# If you need ClaimEvent format
from ergodic_insurance.claim_generator import ClaimEvent

losses, stats = generator.generate_losses(duration=10, revenue=10_000_000)

# Convert LossEvent to ClaimEvent
claims = [ClaimEvent(year=int(loss.time), amount=loss.amount) for loss in losses]
```

## Testing Considerations

When updating tests:

1. **Update imports**:
   ```python
   # Before
   from ergodic_insurance.claim_generator import ClaimGenerator

   # After
   from ergodic_insurance.loss_distributions import ManufacturingLossGenerator
   ```

2. **Update instantiation**:
   ```python
   # Before
   gen = ClaimGenerator(base_frequency=0.1, severity_mean=5_000_000, seed=42)

   # After
   gen = ManufacturingLossGenerator.create_simple(
       frequency=0.1, severity_mean=5_000_000, seed=42
   )
   ```

3. **Update generation calls**:
   ```python
   # Before
   claims = gen.generate_claims(years=10)

   # After
   losses, stats = gen.generate_losses(duration=10, revenue=10_000_000)
   ```

4. **Update assertions**:
   ```python
   # Before
   assert len(claims) > 0
   assert all(isinstance(c.year, int) for c in claims)

   # After
   assert len(losses) > 0
   assert stats['total_losses'] == len(losses)
   assert all(isinstance(loss.time, float) for loss in losses)
   ```

## Gradual Migration Strategy

For large codebases, consider a gradual migration:

1. **Phase 1**: Update new code to use ManufacturingLossGenerator
2. **Phase 2**: Update tests to use the new generator
3. **Phase 3**: Update documentation and examples
4. **Phase 4**: Refactor existing production code module by module
5. **Phase 5**: Remove ClaimGenerator entirely (version 1.0.0)

## Troubleshooting

### DeprecationWarning appears

This is expected! The warning reminds you to migrate. To suppress temporarily:

```python
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='ergodic_insurance.claim_generator')
```

### Different random numbers with same seed

This is expected due to better seed management. If you need exact reproducibility:
- Use a different seed value
- The new method provides better statistical properties

### Revenue parameter required

ManufacturingLossGenerator requires a `revenue` parameter for exposure-based scaling. Use a reasonable value:

```python
# Use typical company revenue
losses, stats = generator.generate_losses(duration=10, revenue=10_000_000)

# Or use 1.0 if revenue is not relevant
losses, stats = generator.generate_losses(duration=10, revenue=1.0)
```

## Support

For questions or issues during migration:
- Check the API documentation: `help(ManufacturingLossGenerator)`
- Review examples in `ergodic_insurance/notebooks/`
- Open an issue on GitHub with the "migration" label

## Timeline

- **Version 0.2.0**: ClaimGenerator deprecated (deprecation warnings added)
- **Version 0.5.0**: ClaimGenerator marked as legacy (moved to `ergodic_insurance.legacy`)
- **Version 1.0.0**: ClaimGenerator removed entirely

Migrate before version 1.0.0 to avoid breaking changes!
