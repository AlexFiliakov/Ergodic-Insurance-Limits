# Specification: Loss Frequency and Exposure Relationship

## Current State Analysis

### What We Have Now
The current `ClaimGenerator` implementation uses a **fixed frequency parameter** that is independent of the manufacturer's business size or activities:

```python
frequency: float = 0.1  # Expected claims per year (fixed)
```

This means whether the company has $1M or $100M in assets, or whether it generates $500K or $50M in revenue, the expected number of claims remains constant. This is unrealistic and fails to capture the fundamental insurance principle that **exposure drives frequency**.

### The Problem
1. **No Exposure Scaling**: A company with 10x the operations doesn't have 10x the loss exposure in our model
2. **No Business Activity Link**: Claims are disconnected from actual business operations
3. **Unrealistic Growth Dynamics**: As companies grow, their risk doesn't grow proportionally

## Proposed: Exposure-Based Frequency Model

### Core Principle
Loss frequency should be a function of **exposure bases** that reflect the company's operational scale and risk-generating activities.

### Key Exposure Bases for Widget Manufacturing

#### 1. **Sales/Revenue Exposure**
- **Rationale**: More sales = more products shipped = more product liability claims
- **Formula**: `frequency_sales = base_rate_per_million * (revenue / 1_000_000)`
- **Example**:
  - Base rate: 0.02 claims per $1M revenue
  - $10M revenue → 0.2 expected claims/year
  - $50M revenue → 1.0 expected claims/year

#### 2. **Asset Exposure**
- **Rationale**: More assets = more property at risk, more equipment failures
- **Formula**: `frequency_assets = base_rate_per_million * (assets / 1_000_000)`
- **Example**:
  - Base rate: 0.01 claims per $1M assets
  - $10M assets → 0.1 expected claims/year
  - $100M assets → 1.0 expected claims/year

#### 3. **Production Volume Exposure** (Implicit)
- Captured through asset turnover ratio
- Higher turnover = more intensive asset use = higher claim frequency
- Could add multiplier: `frequency_multiplier = 1 + (turnover_ratio - 1) * 0.5`

### Recommended Implementation

```python
class ExposureBasedClaimGenerator:
    """Generate claims based on company exposure metrics."""

    def __init__(self,
                 # Frequency rates per $1M exposure
                 sales_frequency_rate: float = 0.02,      # Claims per $1M revenue
                 asset_frequency_rate: float = 0.01,      # Claims per $1M assets
                 # Severity parameters (can also scale with exposure)
                 severity_pct_of_revenue: float = 0.10,   # Severity as % of revenue
                 severity_pct_of_assets: float = 0.05,    # Severity as % of assets
                 seed: Optional[int] = None):
        self.sales_frequency_rate = sales_frequency_rate
        self.asset_frequency_rate = asset_frequency_rate
        self.severity_pct_revenue = severity_pct_of_revenue
        self.severity_pct_assets = severity_pct_of_assets
        self.rng = np.random.RandomState(seed)

    def generate_claims(self,
                       years: int,
                       annual_revenues: List[float],
                       annual_assets: List[float]) -> List[ClaimEvent]:
        """Generate claims based on exposure metrics."""
        claims = []

        for year in range(years):
            # Calculate frequency based on exposures
            revenue = annual_revenues[year] if year < len(annual_revenues) else annual_revenues[-1]
            assets = annual_assets[year] if year < len(annual_assets) else annual_assets[-1]

            # Combined frequency from multiple exposure bases
            sales_frequency = self.sales_frequency_rate * (revenue / 1_000_000)
            asset_frequency = self.asset_frequency_rate * (assets / 1_000_000)
            total_frequency = sales_frequency + asset_frequency

            # Generate number of claims
            n_claims = self.rng.poisson(total_frequency)

            for _ in range(n_claims):
                # Severity also scales with exposure
                # Mix of revenue-based and asset-based severities
                if self.rng.random() < 0.7:  # 70% are operational losses
                    severity_mean = revenue * self.severity_pct_revenue
                else:  # 30% are property/asset losses
                    severity_mean = assets * self.severity_pct_assets

                # Add variability
                severity = self.rng.lognormal(
                    np.log(severity_mean) - 0.5,  # mu
                    1.0  # sigma for variability
                )

                claims.append(ClaimEvent(year=year, amount=severity))

        return claims
```

### Benefits of Exposure-Based Model

1. **Realistic Scaling**: Large companies have more claims than small companies
2. **Growth Impact**: As companies grow, their risk profile naturally increases
3. **Multiple Risk Sources**: Captures both operational (sales) and property (assets) risks
4. **Dynamic Severity**: Claim sizes scale with company size
5. **Ergodic Properties**: Better reflects time-average behavior as company evolves

### Calibration Guidelines

#### For Widget Manufacturing:
- **Product Liability**: 0.01-0.03 claims per $1M revenue
- **Property Damage**: 0.005-0.015 claims per $1M assets
- **Workers Comp**: Could add employee-based frequency
- **Cyber/Tech**: Could add IT-asset-based frequency

#### Severity Distributions:
- **Attritional**: 1-5% of annual revenue per claim
- **Large**: 10-50% of annual revenue per claim
- **Catastrophic**: 100-500% of annual revenue (rare)

### Integration Points

1. **Manufacturer Class**: Should track and provide exposure metrics
2. **Simulation Class**: Should pass exposure data to claim generator
3. **Configuration**: Exposure rates should be configurable parameters

### Example Usage

```python
# In simulation.py
def step_annual(self, year: int) -> Dict[str, float]:
    # Get current exposures
    metrics = self.manufacturer.calculate_metrics()
    revenue = metrics['revenue']
    assets = metrics['assets']

    # Generate claims based on exposures
    claims = self.claim_generator.generate_claims_for_year(
        year=year,
        revenue=revenue,
        assets=assets
    )

    # Process claims...
```

## Migration Path

### Phase 1: Add Exposure Awareness (Backward Compatible)
- Add optional exposure parameters to existing `ClaimGenerator`
- Default to fixed frequency if exposures not provided

### Phase 2: Implement Full Exposure Model
- Create new `ExposureBasedClaimGenerator` class
- Update `Simulation` to use exposure-based generation
- Maintain old generator for comparison studies

### Phase 3: Calibrate and Validate
- Use industry data to calibrate frequency rates
- Validate against historical loss patterns
- Tune for ergodic properties

## Key Insight for Ergodic Analysis

The exposure-based model is **essential for ergodic analysis** because:

1. **Time Average Reality**: A single company's risk grows over time as it grows
2. **Ensemble Average**: Cross-sectional view shows different sized companies with proportional risks
3. **Convergence**: The exposure model ensures time and ensemble averages converge appropriately
4. **Insurance Optimization**: Premium should scale with exposure, making the ergodic trade-offs realistic

Without exposure-based frequency, we're essentially modeling insurance for a company that never changes its risk profile despite growing or shrinking—which violates the fundamental premise of the ergodic insurance problem.

## Recommended Next Steps

1. **Implement** `ExposureBasedClaimGenerator` as described
2. **Update** `Simulation` class to pass exposure data
3. **Calibrate** using industry loss data for manufacturing
4. **Test** ergodic properties with new model
5. **Document** the exposure bases and calibration methodology

This will make the model significantly more realistic and better suited for demonstrating the ergodic advantages of optimal insurance limits.
