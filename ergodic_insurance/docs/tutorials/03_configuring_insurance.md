# Configuring Insurance Tutorial

This tutorial explains how to configure insurance programs, from simple single-layer coverage to complex multi-layer structures. You'll learn about retentions, limits, premiums, and how to design effective insurance programs.

## Learning Objectives

By the end of this tutorial, you will understand:
- Insurance terminology and structure
- How to configure single-layer insurance
- Multi-layer insurance programs
- Premium calculation methods
- Coverage effectiveness analysis
- Real-world insurance program design

## Insurance Fundamentals

### Key Terms Explained

```python
# Let's illustrate insurance terms with a concrete example
import numpy as np
import matplotlib.pyplot as plt

# Example loss of \$3M
loss_amount = 3_000_000

# Insurance structure
retention = 500_000      # Company pays first \$500K (deductible)
limit = 5_000_000        # Insurance covers up to \$5M
premium_rate = 0.02      # 2% of limit

# Calculate who pays what
company_pays = min(loss_amount, retention)
insurance_pays = min(max(0, loss_amount - retention), limit)
uncovered = max(0, loss_amount - retention - limit)

print(f"Loss Amount: ${loss_amount:,.0f}")
print(f"\nPayment Breakdown:")
print(f"  Company pays (retention): ${company_pays:,.0f}")
print(f"  Insurance pays: ${insurance_pays:,.0f}")
print(f"  Uncovered amount: ${uncovered:,.0f}")
print(f"\nAnnual Premium: ${limit * premium_rate:,.0f}")

# Visualize the structure
fig, ax = plt.subplots(figsize=(10, 6))
layers = [company_pays, insurance_pays, uncovered]
labels = ['Retention\n(Company)', 'Insurance\nCoverage', 'Uncovered']
colors = ['red', 'green', 'orange']
bottom = 0
for layer, label, color in zip(layers, labels, colors):
    if layer > 0:
        ax.bar(0, layer, bottom=bottom, color=color, alpha=0.7, label=label, width=0.5)
        bottom += layer

ax.set_ylabel('Loss Amount ($)')
ax.set_title('Insurance Structure Visualization')
ax.set_xticks([])
ax.legend()
ax.set_ylim([0, loss_amount * 1.1])
plt.grid(True, alpha=0.3, axis='y')
plt.show()
```

## Single-Layer Insurance

### Basic Configuration

```python
from ergodic_insurance.insurance import InsuranceLayer
from ergodic_insurance.manufacturer import Manufacturer
from ergodic_insurance.claim_generator import ClaimGenerator
from ergodic_insurance.simulation import Simulation

# Create manufacturer and claim generator
manufacturer = Manufacturer(
    initial_assets=10_000_000,
    asset_turnover=1.0,
    base_operating_margin=0.08
)

claim_generator = ClaimGenerator(
    frequency=5,
    severity_mu=10.0,
    severity_sigma=1.5
)

# Configure single insurance layer
insurance = InsuranceLayer(
    retention=1_000_000,     # \$1M deductible
    limit=10_000_000,        # \$10M coverage
    premium_rate=0.018       # 1.8% rate
)

# Calculate annual premium
annual_premium = insurance.limit * insurance.premium_rate
print(f"Insurance Configuration:")
print(f"  Retention: ${insurance.retention:,.0f}")
print(f"  Limit: ${insurance.limit:,.0f}")
print(f"  Annual Premium: ${annual_premium:,.0f}")
print(f"  Premium as % of limit: {insurance.premium_rate:.2%}")
```

### Analyzing Coverage Effectiveness

```python
# Generate sample losses to analyze coverage
np.random.seed(42)
sample_losses = []
for _ in range(1000):
    annual_losses = claim_generator.generate_claims(years=1)
    sample_losses.extend(annual_losses)

# Analyze how insurance responds
covered_amounts = []
retained_amounts = []
uncovered_amounts = []

for loss in sample_losses:
    retained = min(loss, insurance.retention)
    covered = min(max(0, loss - insurance.retention), insurance.limit)
    uncovered = max(0, loss - insurance.retention - insurance.limit)

    retained_amounts.append(retained)
    covered_amounts.append(covered)
    uncovered_amounts.append(uncovered)

# Calculate statistics
total_losses = sum(sample_losses)
total_retained = sum(retained_amounts)
total_covered = sum(covered_amounts)
total_uncovered = sum(uncovered_amounts)

print(f"\nCoverage Analysis (1000 loss samples):")
print(f"  Total Losses: ${total_losses:,.0f}")
print(f"  Retained by Company: ${total_retained:,.0f} ({total_retained/total_losses:.1%})")
print(f"  Covered by Insurance: ${total_covered:,.0f} ({total_covered/total_losses:.1%})")
print(f"  Uncovered: ${total_uncovered:,.0f} ({total_uncovered/total_losses:.1%})")

# Loss exceedance curve
sorted_losses = sorted(sample_losses, reverse=True)
exceedance_probs = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)

plt.figure(figsize=(10, 6))
plt.semilogy(exceedance_probs * 100, sorted_losses, label='Loss Amount')
plt.axhline(y=insurance.retention, color='red', linestyle='--', alpha=0.5, label='Retention')
plt.axhline(y=insurance.retention + insurance.limit, color='orange', linestyle='--', alpha=0.5, label='Coverage Limit')
plt.xlabel('Exceedance Probability (%)')
plt.ylabel('Loss Amount ($, log scale)')
plt.title('Loss Exceedance Curve with Insurance Layers')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Multi-Layer Insurance Programs

### Configuring Multiple Layers

```python
from ergodic_insurance.insurance_program import InsuranceProgram

# Create a 3-layer insurance program
insurance_program = InsuranceProgram()

# Layer 1: Primary layer (working layer)
insurance_program.add_layer(
    name="Primary",
    retention=250_000,
    limit=2_000_000,
    premium_rate=0.03  # Higher rate for primary layer
)

# Layer 2: First excess layer
insurance_program.add_layer(
    name="First Excess",
    retention=2_250_000,  # Sits above primary
    limit=5_000_000,
    premium_rate=0.015    # Medium rate
)

# Layer 3: Second excess layer (catastrophic)
insurance_program.add_layer(
    name="Catastrophic",
    retention=7_250_000,  # Sits above first excess
    limit=10_000_000,
    premium_rate=0.008    # Lower rate for high layer
)

# Display program structure
print("Multi-Layer Insurance Program:")
print("-" * 50)
total_premium = 0
for i, layer in enumerate(insurance_program.layers):
    premium = layer.limit * layer.premium_rate
    total_premium += premium
    print(f"Layer {i+1} - {layer.name}:")
    print(f"  Attachment Point: ${layer.retention:,.0f}")
    print(f"  Limit: ${layer.limit:,.0f}")
    print(f"  Exhaustion Point: ${layer.retention + layer.limit:,.0f}")
    print(f"  Premium: ${premium:,.0f}")
    print()

print(f"Total Annual Premium: ${total_premium:,.0f}")
```

### Visualizing Layer Structure

```python
# Visualize the tower structure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Tower visualization
ax1.set_title('Insurance Tower Structure')
bottom = 0
colors = ['red', 'orange', 'yellow', 'green']
for i, layer in enumerate(insurance_program.layers):
    if i == 0:
        # Add retention block
        ax1.bar(0, layer.retention, bottom=0, color='lightgray',
               alpha=0.7, label='Retention', width=0.6)
        bottom = layer.retention

    ax1.bar(0, layer.limit, bottom=bottom, color=colors[i],
           alpha=0.7, label=layer.name, width=0.6)

    # Add text annotations
    mid_point = bottom + layer.limit / 2
    ax1.text(0, mid_point, f'{layer.name}\n${layer.limit/1e6:.1f}M',
            ha='center', va='center', fontweight='bold')

    bottom += layer.limit

ax1.set_ylabel('Coverage Amount ($)')
ax1.set_xlim([-0.5, 0.5])
ax1.set_xticks([])
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3, axis='y')

# Response to different loss sizes
ax2.set_title('Insurance Response by Loss Size')
loss_sizes = np.linspace(0, 20_000_000, 100)
company_pays = []
insurance_pays = []

for loss in loss_sizes:
    company_payment, insurance_payment = insurance_program.calculate_payments(loss)
    company_pays.append(company_payment)
    insurance_pays.append(insurance_payment)

ax2.plot(loss_sizes/1e6, np.array(company_pays)/1e6, label='Company Pays', linewidth=2)
ax2.plot(loss_sizes/1e6, np.array(insurance_pays)/1e6, label='Insurance Pays', linewidth=2)
ax2.plot(loss_sizes/1e6, loss_sizes/1e6, '--', alpha=0.3, label='Total Loss')
ax2.set_xlabel('Loss Size ($M)')
ax2.set_ylabel('Payment ($M)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Premium Calculation Methods

### Rate on Line (ROL)

```python
# Rate on Line: Premium as percentage of limit
def calculate_rol_premium(limit, rol_percentage):
    """Calculate premium using Rate on Line method."""
    return limit * rol_percentage / 100

# Example ROL calculations
layers_rol = [
    {"name": "Primary", "limit": 2_000_000, "rol": 5.0},     # 5% ROL
    {"name": "Excess 1", "limit": 5_000_000, "rol": 2.5},    # 2.5% ROL
    {"name": "Excess 2", "limit": 10_000_000, "rol": 1.0},   # 1% ROL
]

print("Rate on Line Premium Calculation:")
print("-" * 40)
total_rol_premium = 0
for layer in layers_rol:
    premium = calculate_rol_premium(layer["limit"], layer["rol"])
    total_rol_premium += premium
    print(f"{layer['name']}: ${premium:,.0f} ({layer['rol']}% ROL)")
print(f"Total Premium: ${total_rol_premium:,.0f}")
```

### Experience-Based Rating

```python
# Calculate premium based on historical losses
def experience_based_premium(historical_losses, retention, limit, load_factor=1.5):
    """
    Calculate premium based on historical experience.
    load_factor: Multiplier above expected losses (for profit and expenses)
    """
    # Calculate expected losses in the layer
    layer_losses = []
    for loss in historical_losses:
        if loss > retention:
            layer_loss = min(loss - retention, limit)
            layer_losses.append(layer_loss)
        else:
            layer_losses.append(0)

    expected_layer_loss = np.mean(layer_losses)
    premium = expected_layer_loss * load_factor

    return premium, expected_layer_loss

# Generate historical loss data
np.random.seed(42)
historical_years = 10
historical_losses = []
for _ in range(historical_years):
    annual = claim_generator.generate_claims(years=1)
    historical_losses.extend(annual)

# Calculate experience-based premiums
print("\nExperience-Based Premium Calculation:")
print("-" * 50)
for layer in insurance_program.layers:
    premium, expected_loss = experience_based_premium(
        historical_losses,
        layer.retention,
        layer.limit,
        load_factor=1.5
    )
    loss_ratio = expected_loss / premium if premium > 0 else 0
    print(f"{layer.name}:")
    print(f"  Expected Loss: ${expected_loss:,.0f}")
    print(f"  Premium (1.5x load): ${premium:,.0f}")
    print(f"  Implied Loss Ratio: {loss_ratio:.1%}")
```

### Exposure-Based Rating

```python
# Premium based on exposure metrics
def exposure_based_premium(revenue, industry_rate, risk_factors):
    """
    Calculate premium based on revenue exposure and risk factors.
    """
    base_premium = revenue * industry_rate

    # Apply risk factor adjustments
    adjustment_factor = 1.0
    for factor_name, factor_value in risk_factors.items():
        adjustment_factor *= factor_value

    adjusted_premium = base_premium * adjustment_factor
    return base_premium, adjusted_premium

# Example calculation
company_revenue = manufacturer.initial_assets * manufacturer.asset_turnover
industry_base_rate = 0.002  # 0.2% of revenue

risk_factors = {
    "loss_history": 1.1,      # 10% surcharge for poor history
    "risk_controls": 0.95,    # 5% discount for good controls
    "deductible": 0.9,        # 10% discount for higher deductible
    "industry_risk": 1.05     # 5% load for riskier industry
}

base_prem, adjusted_prem = exposure_based_premium(
    company_revenue,
    industry_base_rate,
    risk_factors
)

print("\nExposure-Based Premium Calculation:")
print(f"  Revenue: ${company_revenue:,.0f}")
print(f"  Base Rate: {industry_base_rate:.3%}")
print(f"  Base Premium: ${base_prem:,.0f}")
print(f"  Risk Adjustments: {dict(risk_factors)}")
print(f"  Adjusted Premium: ${adjusted_prem:,.0f}")
print(f"  Effective Rate: {adjusted_prem/company_revenue:.3%}")
```

## Optimizing Coverage Structure

### Finding Optimal Retention

```python
from ergodic_insurance.optimization import optimize_retention

# Test different retention levels
retention_levels = np.linspace(100_000, 3_000_000, 20)
results = []

for retention in retention_levels:
    # Run quick simulation
    sim = Simulation(manufacturer, claim_generator)
    result = sim.run(
        n_years=10,
        retention=retention,
        limit=10_000_000,
        premium_rate=0.02,
        seed=42
    )

    results.append({
        'retention': retention,
        'growth_rate': result.growth_rate,
        'survived': result.survived
    })

# Find optimal retention
optimal_idx = max(range(len(results)),
                 key=lambda i: results[i]['growth_rate'] if results[i]['survived'] else -float('inf'))
optimal_retention = results[optimal_idx]['retention']

print(f"Optimal Retention Analysis:")
print(f"  Optimal Retention: ${optimal_retention:,.0f}")
print(f"  Growth Rate: {results[optimal_idx]['growth_rate']:.2%}")

# Visualize optimization
retentions = [r['retention'] for r in results]
growth_rates = [r['growth_rate'] if r['survived'] else np.nan for r in results]

plt.figure(figsize=(10, 6))
plt.plot(np.array(retentions)/1e6, np.array(growth_rates)*100, 'o-')
plt.axvline(x=optimal_retention/1e6, color='red', linestyle='--', alpha=0.5)
plt.xlabel('Retention ($M)')
plt.ylabel('Growth Rate (%)')
plt.title('Growth Rate vs Retention Level')
plt.grid(True, alpha=0.3)
plt.show()
```

### Layer Optimization

```python
# Optimize multi-layer structure
def optimize_layer_structure(total_limit_budget, n_layers=3):
    """
    Optimize the structure of multiple layers given a total limit budget.
    """
    # Simple heuristic: decreasing portion for higher layers
    portions = [0.5, 0.3, 0.2][:n_layers]

    # Normalize to sum to 1
    portions = [p/sum(portions) for p in portions]

    layers = []
    attachment = 250_000  # Starting retention

    for i, portion in enumerate(portions):
        limit = total_limit_budget * portion
        # Higher layers have lower rates
        rate = 0.03 * (0.6 ** i)  # Each layer 60% of previous rate

        layers.append({
            'attachment': attachment,
            'limit': limit,
            'rate': rate,
            'premium': limit * rate
        })
        attachment += limit

    return layers

# Optimize structure
total_limit = 15_000_000
optimized_layers = optimize_layer_structure(total_limit, n_layers=3)

print("\nOptimized Layer Structure:")
print("-" * 60)
print(f"{'Layer':<10} {'Attachment':<15} {'Limit':<15} {'Rate':<10} {'Premium':<15}")
print("-" * 60)
total_premium = 0
for i, layer in enumerate(optimized_layers):
    total_premium += layer['premium']
    print(f"Layer {i+1:<4} ${layer['attachment']:>13,.0f} ${layer['limit']:>13,.0f} "
          f"{layer['rate']:>8.2%} ${layer['premium']:>13,.0f}")
print("-" * 60)
print(f"{'Total':<40} ${total_premium:>13,.0f}")
```

## Real-World Considerations

### Reinstatement Provisions

```python
class InsuranceLayerWithReinstatement:
    """Insurance layer with reinstatement provisions."""

    def __init__(self, retention, limit, premium_rate, n_reinstatements=1):
        self.retention = retention
        self.limit = limit
        self.premium_rate = premium_rate
        self.n_reinstatements = n_reinstatements
        self.available_limit = limit * (1 + n_reinstatements)
        self.used_limit = 0

    def apply_loss(self, loss_amount):
        """Apply a loss and track limit usage."""
        if loss_amount <= self.retention:
            return 0, loss_amount  # No insurance payment

        insurance_payment = min(
            loss_amount - self.retention,
            self.available_limit - self.used_limit
        )

        self.used_limit += insurance_payment
        company_payment = loss_amount - insurance_payment

        # Check if reinstatement needed
        reinstatements_used = int(self.used_limit / self.limit)

        return insurance_payment, company_payment

# Example with reinstatements
layer_with_reinstatement = InsuranceLayerWithReinstatement(
    retention=500_000,
    limit=2_000_000,
    premium_rate=0.02,
    n_reinstatements=2  # 2 free reinstatements
)

print("Insurance with Reinstatements:")
print(f"  Base Limit: ${layer_with_reinstatement.limit:,.0f}")
print(f"  Reinstatements: {layer_with_reinstatement.n_reinstatements}")
print(f"  Total Available: ${layer_with_reinstatement.available_limit:,.0f}")

# Simulate multiple losses
test_losses = [1_500_000, 2_000_000, 1_000_000, 3_000_000]
for i, loss in enumerate(test_losses):
    ins_pay, co_pay = layer_with_reinstatement.apply_loss(loss)
    print(f"\nLoss {i+1}: ${loss:,.0f}")
    print(f"  Insurance pays: ${ins_pay:,.0f}")
    print(f"  Company pays: ${co_pay:,.0f}")
    print(f"  Remaining limit: ${layer_with_reinstatement.available_limit - layer_with_reinstatement.used_limit:,.0f}")
```

### Aggregate Deductibles

```python
class AggregateDeductible:
    """Insurance with aggregate deductible."""

    def __init__(self, annual_aggregate_deductible, per_occurrence_limit):
        self.aggregate_deductible = annual_aggregate_deductible
        self.per_occurrence_limit = per_occurrence_limit
        self.annual_retained = 0

    def apply_loss(self, loss_amount):
        """Apply loss with aggregate deductible."""
        # Check if aggregate deductible is satisfied
        if self.annual_retained < self.aggregate_deductible:
            # Company retains up to aggregate
            retention = min(
                loss_amount,
                self.aggregate_deductible - self.annual_retained
            )
            self.annual_retained += retention

            # Insurance covers the rest up to per-occurrence limit
            insurance_payment = min(
                loss_amount - retention,
                self.per_occurrence_limit
            )
        else:
            # Aggregate satisfied, insurance covers up to limit
            insurance_payment = min(loss_amount, self.per_occurrence_limit)
            retention = loss_amount - insurance_payment

        return insurance_payment, retention

# Example
agg_deductible = AggregateDeductible(
    annual_aggregate_deductible=1_000_000,
    per_occurrence_limit=5_000_000
)

print("Aggregate Deductible Example:")
print(f"  Annual Aggregate: ${agg_deductible.aggregate_deductible:,.0f}")
print(f"  Per Occurrence Limit: ${agg_deductible.per_occurrence_limit:,.0f}")

# Apply series of losses
losses = [300_000, 400_000, 500_000, 600_000]
for i, loss in enumerate(losses):
    ins_pay, retained = agg_deductible.apply_loss(loss)
    print(f"\nLoss {i+1}: ${loss:,.0f}")
    print(f"  Company retains: ${retained:,.0f}")
    print(f"  Insurance pays: ${ins_pay:,.0f}")
    print(f"  Aggregate used: ${agg_deductible.annual_retained:,.0f}")
```

## Best Practices

### 1. Layer Structure Guidelines

```python
# Recommended layer structure by company size
layer_guidelines = {
    "Small (\$1-10M assets)": [
        {"name": "Primary", "retention": "5-10% of assets", "limit": "20-50% of assets"},
        {"name": "Excess", "retention": "25% of assets", "limit": "50-100% of assets"}
    ],
    "Medium (\$10-50M assets)": [
        {"name": "Primary", "retention": "2-5% of assets", "limit": "10-20% of assets"},
        {"name": "Excess 1", "retention": "15% of assets", "limit": "20-40% of assets"},
        {"name": "Excess 2", "retention": "35% of assets", "limit": "50-100% of assets"}
    ],
    "Large (\$50M+ assets)": [
        {"name": "Primary", "retention": "1-2% of assets", "limit": "5-10% of assets"},
        {"name": "Multiple Excess Layers", "retention": "Staggered", "limit": "Total 200%+ of assets"}
    ]
}

print("Insurance Structure Guidelines:")
for size, guidelines in layer_guidelines.items():
    print(f"\n{size}:")
    for layer in guidelines:
        print(f"  - {layer['name']}: Retention {layer['retention']}, Limit {layer['limit']}")
```

### 2. Premium Benchmarking

```python
# Industry premium benchmarks (as % of revenue)
premium_benchmarks = {
    "Manufacturing": {"low": 0.5, "median": 1.0, "high": 2.0},
    "Construction": {"low": 1.0, "median": 2.5, "high": 5.0},
    "Technology": {"low": 0.3, "median": 0.7, "high": 1.5},
    "Healthcare": {"low": 1.5, "median": 3.0, "high": 6.0},
    "Retail": {"low": 0.4, "median": 0.8, "high": 1.5}
}

# Check if premium is reasonable
revenue = manufacturer.initial_assets * manufacturer.asset_turnover
current_premium = total_premium
premium_as_pct = (current_premium / revenue) * 100

print("\nPremium Benchmarking:")
print(f"  Your Revenue: ${revenue:,.0f}")
print(f"  Your Premium: ${current_premium:,.0f} ({premium_as_pct:.2f}% of revenue)")

industry = "Manufacturing"
benchmark = premium_benchmarks[industry]
print(f"\n{industry} Benchmarks (% of revenue):")
print(f"  Low: {benchmark['low']}%")
print(f"  Median: {benchmark['median']}%")
print(f"  High: {benchmark['high']}%")

if premium_as_pct < benchmark['low']:
    print("  ⚠️ Your premium seems low - check coverage adequacy")
elif premium_as_pct > benchmark['high']:
    print("  ⚠️ Your premium seems high - consider optimization")
else:
    print("  ✅ Your premium is within industry norms")
```

## Next Steps

Now that you understand insurance configuration:

1. **[Optimization Workflow](04_optimization_workflow.md)**: Learn to find optimal insurance parameters
2. **[Analyzing Results](05_analyzing_results.md)**: Interpret metrics and make decisions
3. **[Advanced Scenarios](06_advanced_scenarios.md)**: Complex multi-peril programs

## Summary

You've learned:
- ✅ Insurance terminology and structure
- ✅ Single and multi-layer configuration
- ✅ Premium calculation methods
- ✅ Coverage effectiveness analysis
- ✅ Real-world provisions (reinstatements, aggregates)
- ✅ Industry benchmarks and best practices

You're ready to optimize your insurance program for maximum value!
