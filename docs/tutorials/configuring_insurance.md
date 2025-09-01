---
layout: default
title: Configuring Insurance
---

# Configuring Insurance Programs

Learn how to design and configure single-layer and multi-layer insurance programs for optimal protection.

## Overview

This tutorial covers:
- Basic insurance concepts
- Single-layer configuration
- Multi-layer programs
- Premium calculation
- Optimization strategies

## Insurance Fundamentals

### Key Terms

- **Retention (Deductible)**: Amount you pay before insurance responds
- **Limit**: Maximum amount insurance will pay
- **Attachment Point**: Where a layer begins paying
- **Premium**: Cost of insurance coverage
- **Layer**: Segment of coverage between attachment and limit

### Basic Structure

```python
from ergodic_insurance.src.insurance import Insurance

# Simple insurance layer
insurance = Insurance(
    retention=250_000,      # $250K deductible
    limit=5_000_000,       # $5M coverage
    annual_premium=150_000  # $150K/year cost
)
```

## Single-Layer Insurance

### Configuration

```python
from ergodic_insurance.src.insurance_program import InsuranceLayer

# Define a primary layer
primary_layer = InsuranceLayer(
    name="Primary",
    attachment_point=250_000,  # Starts at $250K
    limit=5_000_000,           # Up to $5M total
    premium_rate=0.03          # 3% of limit
)

# Calculate annual premium
annual_premium = primary_layer.calculate_premium()
print(f"Annual Premium: ${annual_premium:,.0f}")
```

### Coverage Example

For a $2M loss:
- You pay: $250K (retention)
- Insurance pays: $1.75M
- Total covered: $2M

For a $10M loss:
- You pay: $250K + $5M = $5.25M
- Insurance pays: $4.75M (capped at limit)
- Uncovered: $5M

## Multi-Layer Programs

### Why Multiple Layers?

1. **Cost Efficiency**: Higher layers are cheaper per dollar
2. **Capacity**: Access more total coverage
3. **Risk Segmentation**: Different insurers for different risks

### Configuration

```python
from ergodic_insurance.src.insurance_program import InsuranceProgram

# Create multi-layer program
program = InsuranceProgram()

# Add layers (each attaches where previous exhausts)
program.add_layer(
    name="Primary",
    attachment=250_000,
    limit=4_750_000,  # $250K to $5M
    premium_rate=0.030  # 3.0%
)

program.add_layer(
    name="Excess-1",
    attachment=5_000_000,
    limit=20_000_000,  # $5M to $25M
    premium_rate=0.008  # 0.8%
)

program.add_layer(
    name="Excess-2",
    attachment=25_000_000,
    limit=25_000_000,  # $25M to $50M
    premium_rate=0.004  # 0.4%
)

# Calculate total program cost
total_premium = program.calculate_total_premium()
print(f"Total Annual Premium: ${total_premium:,.0f}")
```

### Coverage Calculation

```python
def calculate_insurance_recovery(loss_amount, program):
    """Calculate how much insurance pays."""
    recovery = 0
    remaining_loss = loss_amount

    for layer in program.layers:
        if remaining_loss <= layer.attachment:
            break  # Loss doesn't reach this layer

        # Amount covered by this layer
        layer_loss = min(
            remaining_loss - layer.attachment,
            layer.limit
        )

        recovery += layer_loss
        remaining_loss -= layer_loss

    return recovery

# Examples
small_loss = 500_000
large_loss = 30_000_000

print(f"$500K loss: Insurance pays ${calculate_insurance_recovery(small_loss, program):,.0f}")
print(f"$30M loss: Insurance pays ${calculate_insurance_recovery(large_loss, program):,.0f}")
```

## Premium Calculation Methods

### 1. Rate on Line

Premium as percentage of limit:

```python
def rate_on_line_premium(limit, rate):
    """Calculate premium using rate on line."""
    return limit * rate

# Example: $10M limit at 2% rate
premium = rate_on_line_premium(10_000_000, 0.02)
print(f"Premium: ${premium:,.0f}")
```

### 2. Burning Cost Plus Loading

Based on historical losses:

```python
def burning_cost_premium(historical_losses, years, loading_factor=1.3):
    """Calculate premium based on historical losses."""
    average_annual_loss = sum(historical_losses) / years
    return average_annual_loss * loading_factor

# Example: $5M losses over 10 years
historical = [500_000, 0, 2_000_000, 0, 0,
              1_500_000, 0, 0, 1_000_000, 0]
premium = burning_cost_premium(historical, 10, 1.3)
print(f"Premium: ${premium:,.0f}")
```

### 3. Exposure Rating

Based on revenue or assets:

```python
def exposure_based_premium(revenue, rate_per_thousand):
    """Calculate premium based on exposure."""
    return (revenue / 1000) * rate_per_thousand

# Example: $50M revenue at $3 per $1000
premium = exposure_based_premium(50_000_000, 3)
print(f"Premium: ${premium:,.0f}")
```

## Optimization Strategies

### Finding Optimal Retention

```python
from ergodic_insurance.src.optimization import optimize_retention

def optimize_retention_level(manufacturer, loss_params):
    """Find retention that maximizes growth."""

    retention_options = np.logspace(5, 7, 20)  # $100K to $10M
    best_retention = None
    best_growth = -float('inf')

    for retention in retention_options:
        # Configure insurance with this retention
        insurance = Insurance(
            retention=retention,
            limit=50_000_000 - retention,
            premium_rate=calculate_rate(retention)
        )

        # Run simulation
        results = run_simulation(
            manufacturer,
            loss_params,
            insurance
        )

        # Track best option
        if results.growth_rate > best_growth:
            best_growth = results.growth_rate
            best_retention = retention

    return best_retention, best_growth
```

### Multi-Objective Optimization

Balance growth, risk, and cost:

```python
from ergodic_insurance.src.pareto_frontier import ParetoOptimizer

optimizer = ParetoOptimizer()

# Define objectives
objectives = {
    "maximize_growth": lambda x: x.growth_rate,
    "minimize_risk": lambda x: -x.ruin_probability,
    "minimize_cost": lambda x: -x.premium_ratio
}

# Find Pareto optimal solutions
pareto_solutions = optimizer.optimize(
    insurance_configurations,
    objectives
)

# Select based on preferences
selected = optimizer.select_solution(
    pareto_solutions,
    weights={"growth": 0.5, "risk": 0.3, "cost": 0.2}
)
```

## Model Cases

### Example 1: Manufacturing Company

```python
# Company profile
manufacturer = {
    "assets": 50_000_000,
    "revenue": 75_000_000,
    "margin": 0.10,
    "loss_frequency": 0.8,  # Higher operational risk
    "loss_severity": 3_000_000
}

# Recommended structure
insurance_program = [
    {"name": "Primary", "attach": 500_000, "limit": 4_500_000, "rate": 0.035},
    {"name": "Excess", "attach": 5_000_000, "limit": 20_000_000, "rate": 0.010},
    {"name": "Umbrella", "attach": 25_000_000, "limit": 25_000_000, "rate": 0.005}
]

# Total cost: ~$600K/year (0.8% of revenue)
```

### Example 2: Technology Company

```python
# Company profile
tech_company = {
    "assets": 20_000_000,
    "revenue": 100_000_000,
    "margin": 0.20,
    "loss_frequency": 0.3,  # Lower frequency
    "loss_severity": 10_000_000  # But higher severity (cyber, IP)
}

# Recommended structure
insurance_program = [
    {"name": "Primary", "attach": 1_000_000, "limit": 9_000_000, "rate": 0.025},
    {"name": "Excess", "attach": 10_000_000, "limit": 40_000_000, "rate": 0.007},
    {"name": "Cat", "attach": 50_000_000, "limit": 50_000_000, "rate": 0.003}
]

# Total cost: ~$730K/year (0.73% of revenue)
```

## Advanced Configurations

### Aggregate Deductibles

```python
class AggregateDeductible:
    """Insurance with annual aggregate deductible."""

    def __init__(self, annual_aggregate, per_occurrence, limit):
        self.annual_aggregate = annual_aggregate
        self.per_occurrence = per_occurrence
        self.limit = limit
        self.annual_retention_used = 0

    def apply_to_loss(self, loss_amount):
        """Calculate insurance payment for a loss."""
        # Apply per-occurrence deductible
        if loss_amount <= self.per_occurrence:
            return 0

        # Check aggregate
        if self.annual_retention_used >= self.annual_aggregate:
            # Aggregate met, insurance pays all above per-occurrence
            return min(loss_amount - self.per_occurrence, self.limit)

        # Partial aggregate coverage
        retention_remaining = self.annual_aggregate - self.annual_retention_used
        loss_after_deductible = loss_amount - self.per_occurrence

        self.annual_retention_used += min(loss_after_deductible, retention_remaining)

        return max(0, loss_after_deductible - retention_remaining)
```

### Reinstatements

```python
class ReinstatableLimit:
    """Insurance limit that can be reinstated after use."""

    def __init__(self, limit, reinstatements, reinstatement_premium_pct):
        self.original_limit = limit
        self.current_limit = limit
        self.reinstatements_available = reinstatements
        self.reinstatement_premium_pct = reinstatement_premium_pct
        self.additional_premiums = 0

    def apply_to_loss(self, loss_amount):
        """Apply loss and handle reinstatements."""
        if self.current_limit == 0:
            return 0  # Limit exhausted

        payment = min(loss_amount, self.current_limit)
        self.current_limit -= payment

        # Automatic reinstatement if exhausted
        if self.current_limit == 0 and self.reinstatements_available > 0:
            self.current_limit = self.original_limit
            self.reinstatements_available -= 1
            self.additional_premiums += (
                self.original_limit * self.reinstatement_premium_pct
            )

        return payment
```

## Best Practices

1. **Start with Industry Benchmarks**: Use typical structures for your industry
2. **Consider Correlations**: Account for systemic risks
3. **Stress Test**: Evaluate performance in extreme scenarios
4. **Regular Review**: Reassess annually or with major changes
5. **Document Rationale**: Record why you chose specific structures

## Summary

You now understand:
- How to configure single and multi-layer insurance
- Premium calculation methods
- Optimization strategies
- Real-world application examples

## Next Steps

- [Optimization Workflow](optimization_workflow.md) - Automate finding optimal configurations
- [Analyzing Results](analyzing_results.md) - Interpret optimization outputs

For implementation details, see `ergodic_insurance/src/insurance_program.py`.
