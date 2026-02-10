# Tutorial 3: Configuring Insurance

In the previous tutorials, you learned how to set up a manufacturer, generate losses, and run a basic simulation. Now it is time to do what this framework was built for: configuring insurance programs and measuring their impact on long-term business growth.

We will follow **NovaTech Plastics**, a $10M plastics manufacturer with an 8% operating margin, as they design their insurance protection. NovaTech faces a familiar dilemma: insurance premiums feel expensive relative to expected losses, but a single catastrophic event could cripple the business. Ergodic analysis will show us why the "expensive" premium might actually be a bargain.

## A Quick Insurance Refresher

If you already work with commercial insurance, feel free to skip ahead. For everyone else, here are the four concepts that matter most:

- **Deductible (Self-Insured Retention)**: The dollar amount NovaTech pays out of pocket before any insurance kicks in. Higher deductibles reduce premium but increase retained risk.
- **Attachment Point**: The loss amount at which a specific insurance layer begins to respond. For the first layer, this typically equals the deductible.
- **Limit**: The maximum amount a layer will pay. Once a layer is exhausted, any remaining loss falls to the next layer or back on the company.
- **Premium Rate**: The annual cost of a layer, expressed as a percentage of its limit.

Insurance is structured in **layers** stacked on top of one another, forming a **tower**. Lower layers are more expensive per dollar of coverage because they pay out more frequently. Higher excess layers are cheaper because they only respond to large, rare events.

## Creating a Simple Insurance Program

The `InsuranceProgram` class in `ergodic_insurance.insurance_program` is the building block you will use with both the `Simulation` engine and the `MonteCarloEngine`. Let us start NovaTech with a straightforward single-layer program.

```python
from ergodic_insurance import InsuranceProgram

# NovaTech's first insurance program:
#   - $100K deductible (they retain the first $100K of any loss)
#   - $5M of coverage above the deductible
#   - 2.5% premium rate on the limit
policy = InsuranceProgram.simple(
    deductible=100_000,            # Self-insured retention
    limit=5_000_000,               # Maximum payout: $5M
    rate=0.025                     # Annual premium = 2.5% x $5M = $125K
)

print(f"Annual Premium: ${policy.calculate_annual_premium():,.0f}")
print(f"Total Coverage: ${policy.get_total_coverage():,.0f}")
```

**Expected Output:**
```
Annual Premium: $125,000
Total Coverage: $5,000,000
```

That $125K premium represents about 1.25% of NovaTech's $10M asset base. It might look like a drag on earnings, but wait until we see what happens without it.

## Building a Multi-Layer Tower

Real-world insurance programs rarely consist of a single layer. NovaTech's risk manager wants broader protection, so she builds a three-layer tower:

```python
from ergodic_insurance import InsuranceProgram, EnhancedInsuranceLayer

# Primary layer: $5M xs $250K (covers $250K to $5.25M)
primary = EnhancedInsuranceLayer(
    attachment_point=250_000,
    limit=5_000_000,
    base_premium_rate=0.025     # 2.5% -- highest rate, most frequent claims
)

# First excess layer: $5M xs $5.25M (covers $5.25M to $10.25M)
excess_1 = EnhancedInsuranceLayer(
    attachment_point=5_250_000,
    limit=5_000_000,
    base_premium_rate=0.015     # 1.5% -- mid-layer
)

# Second excess layer: $10M xs $10.25M (covers $10.25M to $20.25M)
excess_2 = EnhancedInsuranceLayer(
    attachment_point=10_250_000,
    limit=10_000_000,
    base_premium_rate=0.008     # 0.8% -- catastrophe layer, lowest rate
)

tower = InsuranceProgram(
    layers=[primary, excess_1, excess_2],
    deductible=250_000
)

# Print the tower summary
print("=== NovaTech Insurance Tower ===")
print(f"Deductible: ${tower.deductible:,.0f}")
for i, layer in enumerate(tower.layers, 1):
    exhaustion = layer.attachment_point + layer.limit
    premium = layer.calculate_base_premium()
    print(
        f"  Layer {i}: ${layer.limit/1e6:.0f}M xs ${layer.attachment_point/1e6:,.2f}M "
        f"| Rate: {layer.base_premium_rate:.1%} | Premium: ${premium:,.0f}"
    )
print(f"Total Coverage: ${tower.get_total_coverage():,.0f}")
print(f"Total Annual Premium: ${tower.calculate_annual_premium():,.0f}")
```

**Expected Output:**
```
=== NovaTech Insurance Tower ===
Deductible: $250,000
  Layer 1: $5M xs $0.25M | Rate: 2.5% | Premium: $125,000
  Layer 2: $5M xs $5.25M | Rate: 1.5% | Premium: $75,000
  Layer 3: $10M xs $10.25M | Rate: 0.8% | Premium: $80,000
Total Coverage: $20,000,000
Total Annual Premium: $280,000
```

Notice how the premium rate decreases as the layers go higher. The primary layer (closest to expected losses) costs the most per dollar of coverage, while the catastrophe layer is the cheapest. This reflects the decreasing probability that losses will reach those heights.

## Processing Claims Through the Program

Before running a full simulation, it helps to understand how a single claim flows through the tower. The `process_claim()` method returns a detailed dictionary:

```python
# Scenario 1: Small loss -- entirely within the deductible
result = tower.process_claim(150_000)
print(f"$150K loss -> Company: ${result['deductible_paid'] + result['uncovered_loss']:,.0f}, Insurance: ${result['insurance_recovery']:,.0f}")

# Scenario 2: Medium loss -- penetrates the primary layer
result = tower.process_claim(3_000_000)
print(f"$3M loss   -> Company: ${result['deductible_paid'] + result['uncovered_loss']:,.0f}, Insurance: ${result['insurance_recovery']:,.0f}")

# Scenario 3: Large loss -- hits two layers
result = tower.process_claim(8_000_000)
print(f"$8M loss   -> Company: ${result['deductible_paid'] + result['uncovered_loss']:,.0f}, Insurance: ${result['insurance_recovery']:,.0f}")

# Scenario 4: Catastrophic loss -- exceeds all coverage
result = tower.process_claim(25_000_000)
print(f"$25M loss  -> Company: ${result['deductible_paid'] + result['uncovered_loss']:,.0f}, Insurance: ${result['insurance_recovery']:,.0f}")
```

**Expected Output:**
```
$150K loss -> Company: $150,000, Insurance: $0
$3M loss   -> Company: $250,000, Insurance: $2,750,000
$8M loss   -> Company: $250,000, Insurance: $7,750,000
$25M loss  -> Company: $4,750,000, Insurance: $20,250,000
```

A few things to notice. Small losses stay entirely with NovaTech. Medium losses hit the primary layer and NovaTech only pays the deductible. The catastrophic $25M loss exceeds the tower, so NovaTech absorbs $4.75M ($250K deductible plus $4.75M above the tower). This is the "uninsured gap" that keeps risk managers up at night.

## Insured vs. Uninsured: The Simulation Comparison

Now for the central question: does insurance actually improve NovaTech's long-term growth? Let us run parallel simulations with and without coverage.

```python
from ergodic_insurance import (
    ManufacturerConfig, WidgetManufacturer, ManufacturingLossGenerator,
    Simulation, InsuranceProgram,
)

# -- NovaTech's financial profile --
novatech_config = ManufacturerConfig(
    initial_assets=10_000_000,
    asset_turnover_ratio=1.0,
    base_operating_margin=0.08,
    tax_rate=0.25,
    retention_ratio=1.0
)

# -- Insurance program: $5M xs $100K --
policy = InsuranceProgram.simple(
    deductible=100_000,
    limit=5_000_000,
    rate=0.025
)

# -- Loss profile: moderate frequency, high severity variability --
loss_gen = ManufacturingLossGenerator.create_simple(
    frequency=0.15,
    severity_mean=1_000_000,
    severity_std=1_500_000,
    seed=42
)

# -- Simulation WITH insurance --
manufacturer_insured = WidgetManufacturer(novatech_config)
sim_insured = Simulation(
    manufacturer=manufacturer_insured,
    loss_generator=loss_gen,
    insurance_program=policy,
    time_horizon=30,
    seed=42
)
results_insured = sim_insured.run()

# -- Simulation WITHOUT insurance (same seed for fair comparison) --
loss_gen_no_ins = ManufacturingLossGenerator.create_simple(
    frequency=0.15,
    severity_mean=1_000_000,
    severity_std=1_500_000,
    seed=42
)
manufacturer_uninsured = WidgetManufacturer(novatech_config)
sim_uninsured = Simulation(
    manufacturer=manufacturer_uninsured,
    loss_generator=loss_gen_no_ins,
    time_horizon=30,
    seed=42
)
results_uninsured = sim_uninsured.run()

# -- Compare outcomes --
insured_growth = results_insured.calculate_time_weighted_roe()
uninsured_growth = results_uninsured.calculate_time_weighted_roe()

print("=== NovaTech: 30-Year Insurance Impact ===")
print(f"{'Metric':<30} {'Insured':>14} {'Uninsured':>14}")
print("-" * 60)
print(f"{'Final Equity':<30} ${results_insured.equity[-1]:>13,.0f} ${results_uninsured.equity[-1]:>13,.0f}")
print(f"{'Time-Weighted ROE':<30} {insured_growth:>13.2%} {uninsured_growth:>13.2%}")
print(f"{'Survived':<30} {'Yes' if results_insured.insolvency_year is None else 'No':>14} {'Yes' if results_uninsured.insolvency_year is None else 'No':>14}")
print(f"{'Annual Premium Paid':<30} ${policy.calculate_annual_premium():>13,.0f} {'$0':>14}")
print(f"{'Growth Improvement':<30} {insured_growth - uninsured_growth:>+13.2%}")
```

The key metric is `Time-Weighted ROE`: this is the ergodic (time-average) growth rate that determines what actually happens to NovaTech over its lifetime. Even though NovaTech pays $125K per year in premiums, insurance reduces the devastating impact of large losses on compounding growth.

> **Why does this work?** Ensemble-average thinking says: "expected loss is $150K, the premium is $125K, so insurance is a fair deal." Ergodic thinking reveals something deeper: without insurance, a single $3M loss destroys equity that would have compounded for decades. The growth you lose to volatility drag far exceeds the premium cost.

## Advanced Features

`InsuranceProgram` and `EnhancedInsuranceLayer` also support reinstatements, aggregate limits, participation rates, and different limit types for more sophisticated modeling.

### Reinstatements

Reinstatements restore layer coverage after a claim erodes the limit. They are common in reinsurance and catastrophe layers:

```python
from ergodic_insurance.insurance_program import (
    InsuranceProgram,
    EnhancedInsuranceLayer,
    ReinstatementType,
)

# Catastrophe layer with 2 reinstatements
cat_layer = EnhancedInsuranceLayer(
    attachment_point=5_000_000,
    limit=5_000_000,
    base_premium_rate=0.02,
    reinstatements=2,                             # Two reinstatements available
    reinstatement_premium=1.0,                    # 100% of base premium per reinstatement
    reinstatement_type=ReinstatementType.PRO_RATA # Premium prorated by time remaining
)

print(f"Base Premium: ${cat_layer.calculate_base_premium():,.0f}")
print(f"Reinstatements: {cat_layer.reinstatements}")
print(f"Max Total Coverage: ${cat_layer.limit * (1 + cat_layer.reinstatements):,.0f}")
```

**Expected Output:**
```
Base Premium: $100,000
Reinstatements: 2
Max Total Coverage: $15,000,000
```

The four reinstatement types are:

| Type | Behavior |
|------|----------|
| `ReinstatementType.NONE` | No reinstatements (layer exhausts permanently) |
| `ReinstatementType.PRO_RATA` | Premium prorated based on time remaining in the policy period |
| `ReinstatementType.FULL` | Full base premium charged regardless of timing |
| `ReinstatementType.FREE` | Coverage restores at no additional cost |

### Aggregate Limits

Aggregate limits cap the total payout from a layer across all claims in a policy year, regardless of individual claim sizes:

```python
aggregate_layer = EnhancedInsuranceLayer(
    attachment_point=100_000,
    limit=5_000_000,
    base_premium_rate=0.015,
    limit_type="aggregate",
    aggregate_limit=10_000_000   # Annual aggregate cap
)
```

### Hybrid Limits

Hybrid layers combine per-occurrence and aggregate limits. Each individual claim is capped, and total annual payouts are also capped:

```python
hybrid_layer = EnhancedInsuranceLayer(
    attachment_point=100_000,
    limit=5_000_000,
    base_premium_rate=0.018,
    limit_type="hybrid",
    per_occurrence_limit=5_000_000,   # No single claim pays more than $5M
    aggregate_limit=15_000_000        # Total annual payouts capped at $15M
)
```

### Participation Rates (Quota Share)

When NovaTech shares risk with a co-insurer, the participation rate controls how much of each covered loss the insurer pays:

```python
quota_share_layer = EnhancedInsuranceLayer(
    attachment_point=100_000,
    limit=5_000_000,
    base_premium_rate=0.016,
    participation_rate=0.80   # Insurer covers 80%, NovaTech retains 20%
)
```

### Building a Complete InsuranceProgram

Here is a realistic multi-layer program for NovaTech using all the advanced features:

```python
program = InsuranceProgram(
    layers=[
        # Primary: per-occurrence, no reinstatements
        EnhancedInsuranceLayer(
            attachment_point=250_000,
            limit=5_000_000,
            base_premium_rate=0.025,
            limit_type="per-occurrence",
        ),
        # First excess: aggregate with 1 reinstatement
        EnhancedInsuranceLayer(
            attachment_point=5_250_000,
            limit=5_000_000,
            base_premium_rate=0.015,
            limit_type="aggregate",
            aggregate_limit=10_000_000,
            reinstatements=1,
            reinstatement_premium=1.0,
            reinstatement_type=ReinstatementType.FULL,
        ),
        # Catastrophe excess: aggregate with 2 free reinstatements
        EnhancedInsuranceLayer(
            attachment_point=10_250_000,
            limit=10_000_000,
            base_premium_rate=0.008,
            limit_type="aggregate",
            aggregate_limit=10_000_000,
            reinstatements=2,
            reinstatement_type=ReinstatementType.FREE,
        ),
    ],
    deductible=250_000,
)

# Print program summary
print(f"Deductible: ${program.deductible:,.0f}")
print(f"Total Coverage: ${program.get_total_coverage():,.0f}")
print(f"Total Annual Premium: ${program.calculate_annual_premium():,.0f}")
```

### Processing Claims Through InsuranceProgram

The `InsuranceProgram.process_claim()` method returns a detailed dictionary rather than a simple tuple:

```python
# Process a $7M claim through the program
result = program.process_claim(7_000_000)

print(f"Total Claim:         ${result['total_claim']:,.0f}")
print(f"Deductible Paid:     ${result['deductible_paid']:,.0f}")
print(f"Insurance Recovery:  ${result['insurance_recovery']:,.0f}")
print(f"Uncovered Loss:      ${result['uncovered_loss']:,.0f}")
print(f"Layers Triggered:    {len(result['layers_triggered'])}")

for layer_info in result['layers_triggered']:
    print(f"  Layer at ${layer_info['attachment']:,.0f}: paid ${layer_info['payment']:,.0f}")
```

You can also process an entire year of claims and get aggregate statistics:

```python
# Reset for a fresh policy year
program.reset_annual()

# Process multiple claims in a single year
annual_claims = [500_000, 2_000_000, 8_000_000]
annual_result = program.process_annual_claims(annual_claims)

print(f"\n=== Annual Summary ===")
print(f"Total Losses:               ${annual_result['total_losses']:,.0f}")
print(f"Total Deductible:           ${annual_result['total_deductible']:,.0f}")
print(f"Total Recovery:             ${annual_result['total_recovery']:,.0f}")
print(f"Base Premium:               ${annual_result['base_premium']:,.0f}")
print(f"Reinstatement Premiums:     ${annual_result['total_reinstatement_premiums']:,.0f}")
print(f"Net Benefit (Recovery - Premium): ${annual_result['net_benefit']:,.0f}")
```

## Premium Loading Analysis

Insurance premiums always exceed expected losses (that is how insurers stay solvent). The **premium loading** measures this markup. Traditional analysis sees loading as pure cost. Ergodic analysis reveals it can be a growth investment.

```python
# NovaTech's loss profile
expected_frequency = 0.15           # 15% chance of a loss each year
expected_severity = 1_000_000       # $1M average severity
expected_annual_loss = expected_frequency * expected_severity
print(f"Expected Annual Loss: ${expected_annual_loss:,.0f}")

# Premium paid for the simple $5M xs $100K program
annual_premium = policy.calculate_annual_premium()
print(f"Annual Premium:       ${annual_premium:,.0f}")

# Loading calculation
loading = (annual_premium / expected_annual_loss) - 1 if expected_annual_loss > 0 else float('inf')
print(f"Premium Loading:      {loading:.0%}")

# The ergodic perspective
print(f"\n--- Ergodic Perspective ---")
print(f"Time-Weighted ROE (Insured):   {insured_growth:.2%}")
print(f"Time-Weighted ROE (Uninsured): {uninsured_growth:.2%}")
print(f"Growth Improvement:            {insured_growth - uninsured_growth:+.2%}")

if insured_growth > uninsured_growth:
    print(
        f"\nDespite a {loading:.0%} loading, insurance IMPROVED time-average growth "
        f"by {(insured_growth - uninsured_growth):.2%} per year."
    )
    print(
        "This is the ergodic advantage: reducing volatility drag on compounding "
        "is worth more than the premium cost."
    )
```

This is the core insight of the framework. From an ensemble-average perspective, NovaTech is paying more in premiums than it expects to receive in claims. From a time-average perspective, insurance is removing the volatility that destroys compound growth, and that growth improvement is worth far more than the loading.

To find the **break-even loading**, the maximum premium at which insurance still improves time-average growth, you can sweep across different premium rates:

```python
import numpy as np

loadings = np.arange(0.5, 5.5, 0.5)  # 50% to 500% loading
results_by_loading = []

for load in loadings:
    # Calculate premium rate that produces this loading
    adjusted_rate = (1 + load) * expected_annual_loss / 5_000_000

    test_policy = InsuranceProgram.simple(
        deductible=100_000,
        limit=5_000_000,
        rate=adjusted_rate
    )

    test_loss_gen = ManufacturingLossGenerator.create_simple(
        frequency=0.15, severity_mean=1_000_000,
        severity_std=1_500_000, seed=42
    )
    test_mfg = WidgetManufacturer(novatech_config)
    test_sim = Simulation(
        manufacturer=test_mfg,
        loss_generator=test_loss_gen,
        insurance_program=test_policy,
        time_horizon=30,
        seed=42
    )
    test_results = test_sim.run()
    growth = test_results.calculate_time_weighted_roe()

    results_by_loading.append({
        'loading': load,
        'premium': test_policy.calculate_annual_premium(),
        'growth': growth,
        'vs_uninsured': growth - uninsured_growth
    })

print(f"\n{'Loading':>10} {'Premium':>12} {'Growth':>10} {'vs Uninsured':>14}")
print("-" * 48)
for r in results_by_loading:
    marker = " <-- break-even" if abs(r['vs_uninsured']) < 0.005 else ""
    print(f"{r['loading']:>9.0%} ${r['premium']:>10,.0f} {r['growth']:>9.2%} {r['vs_uninsured']:>+13.2%}{marker}")
```

This analysis demonstrates a key finding of the framework: optimal insurance premiums can exceed expected losses by 200-500% and still enhance time-average growth. The exact break-even point depends on the business's operating margin, asset size, and loss volatility.

## Exercises

The following exercises build on the NovaTech scenario. Use the same financial profile (`novatech_config`) and loss parameters from this tutorial.

### Exercise 1: Build a Custom Insurance Tower

NovaTech's board wants a 3-layer insurance tower with the following specifications:

- **Retention**: $500K self-insured retention
- **Primary layer**: $4.5M xs $500K at a 3.0% rate
- **First excess**: $5M xs $5M at a 1.5% rate
- **Second excess**: $15M xs $10M at a 0.6% rate

Tasks:
1. Create the `InsuranceProgram` with these three layers.
2. Calculate the total annual premium and total coverage.
3. Process claims of $200K, $4M, $9M, and $22M through the tower. For each claim, print the company payment and insurance recovery.
4. Identify which claim sizes fall entirely within the deductible and which exceed the tower.

### Exercise 2: Survival Rate Comparison

Compare three insurance strategies over a 30-year horizon using 10 different random seeds (seeds 0 through 9):

- **Strategy A: Uninsured** -- no insurance at all
- **Strategy B: Low Retention** -- $100K deductible, $5M limit, 2.5% rate
- **Strategy C: High Retention** -- $1M deductible, $5M limit (attachment at $1M), 1.5% rate

For each strategy and each seed, run a simulation with `frequency=0.2`, `severity_mean=1_000_000`, and `severity_std=1_500_000`. Track:
- Survival count (how many of the 10 seeds survived all 30 years)
- Mean final equity among survivors
- Mean time-weighted ROE

Present your results in a table comparing the three strategies. Which strategy maximizes survival? Which maximizes long-term growth among survivors?

### Exercise 3: Maximum Profitable Loading

Determine the maximum premium loading at which insurance still improves NovaTech's time-average growth.

Using the base policy structure ($5M limit, $100K deductible) and loss parameters (`frequency=0.15`, `severity_mean=1_000_000`, `severity_std=1_500_000`):

1. Run simulations at premium loadings from 0% to 600% in 50% increments (i.e., rates ranging from `expected_loss / limit` up to `7 * expected_loss / limit`).
2. For each loading, calculate the time-weighted ROE over a 30-year horizon (use seed=42).
3. Plot or tabulate the results and identify the break-even loading where insurance growth advantage falls to zero.
4. How does the break-even loading change if you increase `severity_std` to $3,000,000 (higher tail risk)? Explain why.

*Hint: Higher tail risk means larger potential losses, which increases volatility drag on uninsured growth. This should shift the break-even loading higher, as NovaTech can benefit despite even more expensive premiums when the downside is more severe.*

## Next Steps

- [Tutorial 4: Optimization Workflow](04_optimization_workflow.md) -- Use the optimizer to automatically find the best deductible and limit for your business
- [Tutorial 5: Analyzing Results](05_analyzing_results.md) -- Deep dive into ergodic analysis, volatility drag, and DuPont decomposition
- [Tutorial 6: Advanced Scenarios](06_advanced_scenarios.md) -- Monte Carlo simulations, market cycles, and multi-line programs
