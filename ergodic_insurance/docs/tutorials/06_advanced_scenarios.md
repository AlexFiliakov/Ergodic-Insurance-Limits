# Advanced Scenarios Tutorial

This tutorial covers complex, real-world insurance scenarios including multi-peril programs, correlated risks, dynamic strategies, and industry-specific applications. You'll learn to model sophisticated insurance structures and make strategic decisions under uncertainty.

## Learning Objectives

By the end of this tutorial, you will be able to:
- Model multi-peril insurance programs
- Handle correlated risks and dependencies
- Implement dynamic insurance strategies
- Apply the framework to specific industries
- Optimize complex insurance portfolios
- Handle regulatory and capital constraints

## Multi-Peril Insurance Programs

### Modeling Multiple Risk Types

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
from ergodic_insurance.src.manufacturer import Manufacturer
from ergodic_insurance.src.claim_generator import ClaimGenerator
from ergodic_insurance.src.insurance_program import InsuranceProgram
from ergodic_insurance.src.monte_carlo import MonteCarloAnalyzer

# Define multiple perils
class MultiPerilGenerator:
    """Generate losses from multiple correlated perils."""

    def __init__(self, perils_config, correlation_matrix=None):
        self.perils = perils_config
        self.correlation_matrix = correlation_matrix

    def generate_annual_losses(self, seed=None):
        """Generate correlated losses for all perils."""
        if seed:
            np.random.seed(seed)

        annual_losses = {}

        # Generate base losses for each peril
        for peril_name, config in self.perils.items():
            # Frequency (Poisson)
            n_losses = np.random.poisson(config['frequency'])

            # Severity (Lognormal)
            if n_losses > 0:
                severities = np.random.lognormal(
                    config['severity_mu'],
                    config['severity_sigma'],
                    n_losses
                )
                annual_losses[peril_name] = severities
            else:
                annual_losses[peril_name] = np.array([])

        # Apply correlation if specified
        if self.correlation_matrix is not None:
            annual_losses = self._apply_correlation(annual_losses)

        return annual_losses

    def _apply_correlation(self, losses):
        """Apply correlation structure to losses."""
        # Simplified correlation: increase severity when multiple perils hit
        total_perils_hit = sum(1 for l in losses.values() if len(l) > 0)

        if total_perils_hit > 1:
            correlation_factor = 1 + 0.2 * (total_perils_hit - 1)
            for peril in losses:
                losses[peril] = losses[peril] * correlation_factor

        return losses

# Configure multiple perils
perils_config = {
    'property': {
        'frequency': 3,           # 3 property losses per year
        'severity_mu': 10.5,       # Moderate severity
        'severity_sigma': 1.2
    },
    'liability': {
        'frequency': 2,           # 2 liability claims per year
        'severity_mu': 11.0,       # Higher severity
        'severity_sigma': 1.5
    },
    'cyber': {
        'frequency': 1,           # 1 cyber incident per year
        'severity_mu': 11.5,       # Potentially severe
        'severity_sigma': 2.0
    },
    'business_interruption': {
        'frequency': 0.5,         # Once every 2 years
        'severity_mu': 12.0,       # Very severe
        'severity_sigma': 1.8
    }
}

# Create correlation matrix (simplified)
correlation_matrix = np.array([
    [1.0, 0.3, 0.2, 0.5],  # Property
    [0.3, 1.0, 0.1, 0.2],  # Liability
    [0.2, 0.1, 1.0, 0.4],  # Cyber
    [0.5, 0.2, 0.4, 1.0]   # Business Interruption
])

multi_peril_gen = MultiPerilGenerator(perils_config, correlation_matrix)

# Generate sample losses
print("Multi-Peril Loss Generation Example:")
print("-" * 50)
for year in range(3):
    annual_losses = multi_peril_gen.generate_annual_losses(seed=year)
    print(f"\nYear {year + 1}:")
    for peril, losses in annual_losses.items():
        total = np.sum(losses)
        count = len(losses)
        print(f"  {peril}: {count} losses, Total: ${total:,.0f}")
```

### Structuring Multi-Peril Coverage

```python
class MultiPerilInsuranceProgram:
    """Comprehensive multi-peril insurance program."""

    def __init__(self):
        self.peril_specific_layers = {}
        self.umbrella_layers = []
        self.aggregate_covers = {}

    def add_peril_specific_layer(self, peril, retention, limit, premium_rate):
        """Add insurance layer for specific peril."""
        if peril not in self.peril_specific_layers:
            self.peril_specific_layers[peril] = []

        self.peril_specific_layers[peril].append({
            'retention': retention,
            'limit': limit,
            'premium_rate': premium_rate,
            'remaining_limit': limit  # Track exhaustion
        })

    def add_umbrella_layer(self, retention, limit, premium_rate):
        """Add umbrella layer covering all perils."""
        self.umbrella_layers.append({
            'retention': retention,
            'limit': limit,
            'premium_rate': premium_rate,
            'remaining_limit': limit
        })

    def add_aggregate_cover(self, annual_aggregate_retention, annual_limit, premium_rate):
        """Add aggregate stop-loss cover."""
        self.aggregate_covers['stop_loss'] = {
            'retention': annual_aggregate_retention,
            'limit': annual_limit,
            'premium_rate': premium_rate,
            'used': 0
        }

    def apply_losses(self, annual_losses):
        """Apply losses to insurance structure."""
        company_payments = {}
        insurance_payments = {}
        total_company_payment = 0

        # Process each peril
        for peril, losses in annual_losses.items():
            company_payments[peril] = 0
            insurance_payments[peril] = 0

            for loss in losses:
                remaining_loss = loss

                # Apply peril-specific layers
                if peril in self.peril_specific_layers:
                    for layer in self.peril_specific_layers[peril]:
                        if remaining_loss <= layer['retention']:
                            company_payments[peril] += remaining_loss
                            remaining_loss = 0
                            break
                        else:
                            company_payments[peril] += layer['retention']
                            covered = min(remaining_loss - layer['retention'],
                                        layer['remaining_limit'])
                            insurance_payments[peril] += covered
                            layer['remaining_limit'] -= covered
                            remaining_loss -= layer['retention'] + covered

                # Apply umbrella layers
                for umbrella in self.umbrella_layers:
                    if remaining_loss > 0 and umbrella['remaining_limit'] > 0:
                        if remaining_loss <= umbrella['retention']:
                            company_payments[peril] += remaining_loss
                            remaining_loss = 0
                        else:
                            covered = min(remaining_loss - umbrella['retention'],
                                        umbrella['remaining_limit'])
                            insurance_payments[peril] += covered
                            umbrella['remaining_limit'] -= covered
                            remaining_loss -= covered

                # Any remaining loss
                company_payments[peril] += remaining_loss
                total_company_payment += company_payments[peril]

        # Apply aggregate stop-loss
        if 'stop_loss' in self.aggregate_covers:
            stop_loss = self.aggregate_covers['stop_loss']
            if total_company_payment > stop_loss['retention']:
                recovery = min(total_company_payment - stop_loss['retention'],
                             stop_loss['limit'] - stop_loss['used'])
                stop_loss['used'] += recovery
                total_company_payment -= recovery

        return company_payments, insurance_payments, total_company_payment

    def calculate_total_premium(self):
        """Calculate total annual premium."""
        total = 0

        # Peril-specific premiums
        for peril, layers in self.peril_specific_layers.items():
            for layer in layers:
                total += layer['limit'] * layer['premium_rate']

        # Umbrella premiums
        for umbrella in self.umbrella_layers:
            total += umbrella['limit'] * umbrella['premium_rate']

        # Aggregate cover premium
        if 'stop_loss' in self.aggregate_covers:
            sl = self.aggregate_covers['stop_loss']
            total += sl['limit'] * sl['premium_rate']

        return total

# Create comprehensive insurance program
insurance_program = MultiPerilInsuranceProgram()

# Add peril-specific layers
insurance_program.add_peril_specific_layer('property', 100_000, 2_000_000, 0.025)
insurance_program.add_peril_specific_layer('liability', 250_000, 5_000_000, 0.02)
insurance_program.add_peril_specific_layer('cyber', 500_000, 3_000_000, 0.03)
insurance_program.add_peril_specific_layer('business_interruption', 1_000_000, 10_000_000, 0.015)

# Add umbrella coverage
insurance_program.add_umbrella_layer(5_000_000, 20_000_000, 0.01)

# Add aggregate stop-loss
insurance_program.add_aggregate_cover(3_000_000, 15_000_000, 0.008)

# Calculate total premium
total_premium = insurance_program.calculate_total_premium()

print("\nMulti-Peril Insurance Program Structure:")
print("=" * 60)
print("Peril-Specific Layers:")
for peril, layers in insurance_program.peril_specific_layers.items():
    for i, layer in enumerate(layers):
        premium = layer['limit'] * layer['premium_rate']
        print(f"  {peril}: ${layer['retention']:,.0f} x ${layer['limit']:,.0f} @ {layer['premium_rate']:.1%} = ${premium:,.0f}")

print("\nUmbrella Layers:")
for layer in insurance_program.umbrella_layers:
    premium = layer['limit'] * layer['premium_rate']
    print(f"  ${layer['retention']:,.0f} x ${layer['limit']:,.0f} @ {layer['premium_rate']:.1%} = ${premium:,.0f}")

print("\nAggregate Stop-Loss:")
sl = insurance_program.aggregate_covers['stop_loss']
premium = sl['limit'] * sl['premium_rate']
print(f"  ${sl['retention']:,.0f} x ${sl['limit']:,.0f} @ {sl['premium_rate']:.1%} = ${premium:,.0f}")

print(f"\nTotal Annual Premium: ${total_premium:,.0f}")

# Simulate a year with the program
annual_losses = multi_peril_gen.generate_annual_losses(seed=42)
company_pay, insurance_pay, net_company = insurance_program.apply_losses(annual_losses)

print(f"\nLoss Application Example:")
print(f"Total Losses: ${sum(sum(losses) for losses in annual_losses.values()):,.0f}")
print(f"Insurance Pays: ${sum(insurance_pay.values()):,.0f}")
print(f"Company Pays (after all recoveries): ${net_company:,.0f}")
```

## Correlated Risks and Dependencies

### Modeling Risk Correlations

```python
from scipy.stats import multivariate_normal, norm

class CorrelatedRiskModel:
    """Model correlated risks using copulas."""

    def __init__(self, n_risks, correlation_matrix):
        self.n_risks = n_risks
        self.correlation = correlation_matrix

    def generate_correlated_losses(self, n_simulations, marginal_distributions):
        """Generate correlated losses using Gaussian copula."""

        # Generate correlated uniform variables
        mean = np.zeros(self.n_risks)
        mvn = multivariate_normal(mean=mean, cov=self.correlation)
        normal_samples = mvn.rvs(size=n_simulations)

        # Transform to uniform using CDF
        uniform_samples = norm.cdf(normal_samples)

        # Transform to target marginal distributions
        correlated_losses = np.zeros((n_simulations, self.n_risks))

        for i, dist in enumerate(marginal_distributions):
            if dist['type'] == 'lognormal':
                # Inverse transform for lognormal
                quantiles = uniform_samples[:, i]
                correlated_losses[:, i] = np.exp(
                    norm.ppf(quantiles) * dist['sigma'] + dist['mu']
                )
            elif dist['type'] == 'pareto':
                # Inverse transform for Pareto
                quantiles = uniform_samples[:, i]
                correlated_losses[:, i] = dist['scale'] / (1 - quantiles) ** (1/dist['alpha'])

        return correlated_losses

# Define marginal distributions for different risk types
marginal_distributions = [
    {'type': 'lognormal', 'mu': 10, 'sigma': 1.5},     # Operational risk
    {'type': 'lognormal', 'mu': 11, 'sigma': 2.0},     # Market risk
    {'type': 'pareto', 'scale': 50000, 'alpha': 1.5},  # Catastrophic risk
]

# Define correlation structure
risk_correlation = np.array([
    [1.0, 0.4, 0.2],  # Operational
    [0.4, 1.0, 0.3],  # Market
    [0.2, 0.3, 1.0]   # Catastrophic
])

# Generate correlated losses
corr_model = CorrelatedRiskModel(3, risk_correlation)
correlated_losses = corr_model.generate_correlated_losses(1000, marginal_distributions)

# Analyze correlation in generated data
empirical_correlation = np.corrcoef(correlated_losses.T)

print("Risk Correlation Analysis:")
print("-" * 50)
print("Target Correlation Matrix:")
print(risk_correlation)
print("\nEmpirical Correlation (from 1000 simulations):")
print(np.round(empirical_correlation, 2))

# Visualize correlated risks
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

risk_names = ['Operational', 'Market', 'Catastrophic']
pairs = [(0, 1), (0, 2), (1, 2)]

for idx, (i, j) in enumerate(pairs):
    ax = axes[idx]
    ax.scatter(correlated_losses[:, i]/1000, correlated_losses[:, j]/1000,
              alpha=0.5, s=10)
    ax.set_xlabel(f'{risk_names[i]} Loss ($K)')
    ax.set_ylabel(f'{risk_names[j]} Loss ($K)')
    ax.set_title(f'Correlation: {empirical_correlation[i, j]:.2f}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Tail Dependence Modeling

```python
class TailDependenceModel:
    """Model extreme event correlations."""

    def __init__(self, normal_correlation, tail_dependence_factor):
        self.normal_corr = normal_correlation
        self.tail_factor = tail_dependence_factor

    def calculate_dynamic_correlation(self, loss_percentile):
        """Calculate correlation that increases in the tail."""
        # Higher correlation for extreme events
        if loss_percentile > 90:
            tail_adjustment = (loss_percentile - 90) / 10 * self.tail_factor
            adjusted_corr = self.normal_corr + tail_adjustment * (1 - self.normal_corr)
            return min(adjusted_corr, 0.99)
        return self.normal_corr

    def simulate_with_tail_dependence(self, n_sims, severity_params):
        """Simulate losses with tail dependence."""
        losses = []

        for _ in range(n_sims):
            # Determine if this is a tail event
            is_tail_event = np.random.random() < 0.1  # 10% chance

            if is_tail_event:
                # High correlation in tail
                correlation = self.calculate_dynamic_correlation(95)
                # Generate highly correlated severe losses
                base_loss = np.random.lognormal(severity_params['mu'] + 2,
                                               severity_params['sigma'])
                loss1 = base_loss * np.random.uniform(0.8, 1.2)
                loss2 = base_loss * np.random.uniform(0.7, 1.3)
            else:
                # Normal correlation
                correlation = self.normal_corr
                # Generate moderately correlated losses
                loss1 = np.random.lognormal(severity_params['mu'],
                                          severity_params['sigma'])
                if np.random.random() < correlation:
                    loss2 = loss1 * np.random.uniform(0.5, 1.5)
                else:
                    loss2 = np.random.lognormal(severity_params['mu'],
                                              severity_params['sigma'])

            losses.append([loss1, loss2])

        return np.array(losses)

# Create tail dependence model
tail_model = TailDependenceModel(normal_correlation=0.3, tail_dependence_factor=0.5)

# Simulate with tail dependence
severity_params = {'mu': 10, 'sigma': 1.5}
tail_losses = tail_model.simulate_with_tail_dependence(1000, severity_params)

# Analyze tail correlation
threshold_90 = np.percentile(tail_losses[:, 0], 90)
tail_events = tail_losses[tail_losses[:, 0] > threshold_90]
normal_events = tail_losses[tail_losses[:, 0] <= threshold_90]

tail_correlation = np.corrcoef(tail_events.T)[0, 1]
normal_correlation = np.corrcoef(normal_events.T)[0, 1]

print("\nTail Dependence Analysis:")
print("-" * 50)
print(f"Normal Events Correlation: {normal_correlation:.3f}")
print(f"Tail Events Correlation: {tail_correlation:.3f}")
print(f"Correlation Increase in Tail: {(tail_correlation - normal_correlation):.3f}")
```

## Dynamic Insurance Strategies

### Adaptive Coverage Based on Financial Health

```python
class DynamicInsuranceStrategy:
    """Adjust insurance based on company's financial position."""

    def __init__(self, base_retention, base_limit, health_thresholds):
        self.base_retention = base_retention
        self.base_limit = base_limit
        self.thresholds = health_thresholds

    def determine_coverage(self, current_assets, initial_assets, year):
        """Determine optimal coverage based on financial health."""

        # Calculate health ratio
        health_ratio = current_assets / initial_assets

        # Determine financial state
        if health_ratio > self.thresholds['strong']:
            state = 'strong'
            retention_multiplier = 1.5  # Can afford higher retention
            limit_multiplier = 0.8      # Need less coverage

        elif health_ratio > self.thresholds['stable']:
            state = 'stable'
            retention_multiplier = 1.0
            limit_multiplier = 1.0

        elif health_ratio > self.thresholds['weak']:
            state = 'weak'
            retention_multiplier = 0.7  # Need lower retention
            limit_multiplier = 1.2      # Need more coverage

        else:
            state = 'critical'
            retention_multiplier = 0.5  # Minimum retention
            limit_multiplier = 1.5      # Maximum coverage

        # Adjust for time (more conservative early on)
        if year < 5:
            retention_multiplier *= 0.8
            limit_multiplier *= 1.1

        optimal_retention = self.base_retention * retention_multiplier
        optimal_limit = self.base_limit * limit_multiplier

        return {
            'state': state,
            'retention': optimal_retention,
            'limit': optimal_limit,
            'health_ratio': health_ratio
        }

    def simulate_adaptive_strategy(self, manufacturer, claim_generator, n_years=20):
        """Simulate with adaptive insurance strategy."""

        wealth_trajectory = [manufacturer.initial_assets]
        insurance_history = []

        current_assets = manufacturer.initial_assets

        for year in range(n_years):
            # Determine coverage for this year
            coverage = self.determine_coverage(
                current_assets,
                manufacturer.initial_assets,
                year
            )

            insurance_history.append(coverage)

            # Generate operating income
            revenue = current_assets * manufacturer.asset_turnover
            operating_income = revenue * manufacturer.operating_margin
            after_tax_income = operating_income * (1 - manufacturer.tax_rate)

            # Generate and apply losses
            losses = claim_generator.generate_claims(n_years=1)
            total_loss = sum(losses)

            # Apply insurance
            if total_loss <= coverage['retention']:
                net_loss = total_loss
            else:
                net_loss = coverage['retention'] + max(0, total_loss - coverage['retention'] - coverage['limit'])

            # Calculate premium (dynamic based on coverage)
            base_rate = 0.02
            if coverage['state'] == 'critical':
                base_rate = 0.03  # Higher rate for risky companies
            elif coverage['state'] == 'weak':
                base_rate = 0.025

            premium = coverage['limit'] * base_rate

            # Update wealth
            current_assets = current_assets + after_tax_income - net_loss - premium
            wealth_trajectory.append(current_assets)

            # Check for bankruptcy
            if current_assets <= 0:
                break

        return wealth_trajectory, insurance_history

# Create dynamic strategy
health_thresholds = {
    'strong': 1.5,   # 50% above initial
    'stable': 1.0,   # At initial level
    'weak': 0.7      # 30% below initial
}

dynamic_strategy = DynamicInsuranceStrategy(
    base_retention=1_000_000,
    base_limit=10_000_000,
    health_thresholds=health_thresholds
)

# Run simulation
manufacturer = Manufacturer(initial_assets=10_000_000)
claim_generator = ClaimGenerator(frequency=5, severity_mu=10, severity_sigma=1.5)

wealth, insurance = dynamic_strategy.simulate_adaptive_strategy(
    manufacturer, claim_generator, n_years=20
)

# Visualize dynamic strategy
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Wealth trajectory
ax1 = axes[0, 0]
ax1.plot(wealth, 'b-', linewidth=2)
ax1.axhline(y=manufacturer.initial_assets, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Year')
ax1.set_ylabel('Wealth ($)')
ax1.set_title('Wealth Evolution with Dynamic Strategy')
ax1.grid(True, alpha=0.3)

# Insurance parameters over time
ax2 = axes[0, 1]
retentions = [ins['retention'] for ins in insurance]
limits = [ins['limit'] for ins in insurance]
ax2.plot(retentions, 'g-', label='Retention', linewidth=2)
ax2.plot(limits, 'r-', label='Limit', linewidth=2)
ax2.set_xlabel('Year')
ax2.set_ylabel('Amount ($)')
ax2.set_title('Dynamic Insurance Parameters')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Financial state over time
ax3 = axes[1, 0]
states = [ins['state'] for ins in insurance]
state_colors = {'strong': 'green', 'stable': 'blue', 'weak': 'orange', 'critical': 'red'}
for i, state in enumerate(states):
    ax3.bar(i, 1, color=state_colors[state], alpha=0.7)
ax3.set_xlabel('Year')
ax3.set_ylabel('Financial State')
ax3.set_title('Financial Health Status')
ax3.set_yticks([])

# Health ratio
ax4 = axes[1, 1]
health_ratios = [ins['health_ratio'] for ins in insurance]
ax4.plot(health_ratios, 'purple', linewidth=2)
ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Initial Level')
ax4.fill_between(range(len(health_ratios)), 0.7, 1.5, alpha=0.2, color='green', label='Stable Zone')
ax4.set_xlabel('Year')
ax4.set_ylabel('Assets / Initial Assets')
ax4.set_title('Financial Health Ratio')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Dynamic Strategy Summary:")
print("-" * 50)
print(f"Starting Assets: ${manufacturer.initial_assets:,.0f}")
print(f"Final Assets: ${wealth[-1]:,.0f}")
print(f"Years Survived: {len(wealth) - 1}")
print(f"State Transitions: {' → '.join(dict.fromkeys(states))}")
```

## Industry-Specific Applications

### Manufacturing Industry

```python
class ManufacturingRiskModel:
    """Specialized risk model for manufacturing companies."""

    def __init__(self, company_size, industry_subsector):
        self.size = company_size
        self.subsector = industry_subsector
        self.risk_profile = self._determine_risk_profile()

    def _determine_risk_profile(self):
        """Determine risk profile based on industry characteristics."""

        profiles = {
            'automotive': {
                'product_liability': 'high',
                'supply_chain': 'high',
                'equipment': 'medium',
                'cyber': 'medium'
            },
            'pharmaceuticals': {
                'product_liability': 'very_high',
                'regulatory': 'high',
                'intellectual_property': 'high',
                'cyber': 'high'
            },
            'food_processing': {
                'product_liability': 'medium',
                'contamination': 'high',
                'equipment': 'medium',
                'supply_chain': 'medium'
            },
            'electronics': {
                'product_liability': 'medium',
                'intellectual_property': 'high',
                'supply_chain': 'high',
                'cyber': 'very_high'
            }
        }

        return profiles.get(self.subsector, profiles['automotive'])

    def recommend_insurance_structure(self, assets):
        """Recommend insurance structure based on profile."""

        recommendations = []

        # Base recommendations on risk profile
        if self.risk_profile.get('product_liability') in ['high', 'very_high']:
            recommendations.append({
                'coverage': 'Product Liability',
                'retention': assets * 0.01,
                'limit': assets * 2.0,
                'priority': 'Critical'
            })

        if self.risk_profile.get('cyber') in ['high', 'very_high']:
            recommendations.append({
                'coverage': 'Cyber Liability',
                'retention': assets * 0.005,
                'limit': assets * 1.0,
                'priority': 'Critical'
            })

        if self.risk_profile.get('supply_chain') == 'high':
            recommendations.append({
                'coverage': 'Supply Chain Disruption',
                'retention': assets * 0.02,
                'limit': assets * 0.5,
                'priority': 'High'
            })

        # Always recommend general liability and property
        recommendations.extend([
            {
                'coverage': 'General Liability',
                'retention': assets * 0.005,
                'limit': assets * 1.5,
                'priority': 'Critical'
            },
            {
                'coverage': 'Property',
                'retention': assets * 0.01,
                'limit': assets * 1.0,
                'priority': 'Critical'
            }
        ])

        return recommendations

# Example: Pharmaceutical manufacturer
pharma_company = ManufacturingRiskModel(
    company_size='large',
    industry_subsector='pharmaceuticals'
)

company_assets = 50_000_000
recommendations = pharma_company.recommend_insurance_structure(company_assets)

print("Insurance Recommendations for Pharmaceutical Manufacturer:")
print("=" * 70)
print(f"Company Assets: ${company_assets:,.0f}")
print(f"Risk Profile: {pharma_company.risk_profile}")
print("\nRecommended Coverage Structure:")
print("-" * 70)
print(f"{'Coverage Type':<25} {'Retention':<15} {'Limit':<15} {'Priority':<10}")
print("-" * 70)

total_premium_estimate = 0
for rec in recommendations:
    retention = rec['retention']
    limit = rec['limit']
    # Estimate premium based on coverage type and limits
    if 'Liability' in rec['coverage']:
        premium_rate = 0.025
    elif 'Cyber' in rec['coverage']:
        premium_rate = 0.03
    else:
        premium_rate = 0.02

    premium = limit * premium_rate
    total_premium_estimate += premium

    print(f"{rec['coverage']:<25} ${retention:>13,.0f} ${limit:>13,.0f} {rec['priority']:<10}")

print("-" * 70)
print(f"Estimated Total Annual Premium: ${total_premium_estimate:,.0f}")
print(f"Premium as % of Assets: {total_premium_estimate/company_assets:.2%}")
```

### Technology Sector

```python
class TechCompanyRiskModel:
    """Risk model for technology companies."""

    def __init__(self, company_type, revenue, market_cap=None):
        self.type = company_type
        self.revenue = revenue
        self.market_cap = market_cap or revenue * 5
        self.risk_factors = self._identify_risks()

    def _identify_risks(self):
        """Identify key risks for tech companies."""

        base_risks = {
            'cyber': {'frequency': 2.0, 'severity': 12.0},
            'errors_omissions': {'frequency': 3.0, 'severity': 11.0},
            'intellectual_property': {'frequency': 1.0, 'severity': 13.0},
            'key_person': {'frequency': 0.2, 'severity': 14.0}
        }

        # Adjust based on company type
        if self.type == 'saas':
            base_risks['data_breach'] = {'frequency': 1.5, 'severity': 12.5}
            base_risks['service_interruption'] = {'frequency': 4.0, 'severity': 10.0}

        elif self.type == 'fintech':
            base_risks['regulatory'] = {'frequency': 2.0, 'severity': 11.5}
            base_risks['fraud'] = {'frequency': 5.0, 'severity': 10.5}

        elif self.type == 'hardware':
            base_risks['product_liability'] = {'frequency': 2.0, 'severity': 11.0}
            base_risks['supply_chain'] = {'frequency': 3.0, 'severity': 10.5}

        return base_risks

    def calculate_var_metrics(self, confidence=0.95):
        """Calculate Value at Risk for tech company."""

        # Simulate potential losses
        n_simulations = 10000
        annual_losses = []

        for _ in range(n_simulations):
            total_loss = 0
            for risk, params in self.risk_factors.items():
                n_events = np.random.poisson(params['frequency'])
                if n_events > 0:
                    losses = np.random.lognormal(params['severity'], 1.5, n_events)
                    total_loss += np.sum(losses)
            annual_losses.append(total_loss)

        var = np.percentile(annual_losses, (1 - confidence) * 100)
        cvar = np.mean([l for l in annual_losses if l >= var])

        return {
            'var': var,
            'cvar': cvar,
            'expected_loss': np.mean(annual_losses),
            'max_loss': np.max(annual_losses)
        }

# Example: SaaS company
saas_company = TechCompanyRiskModel(
    company_type='saas',
    revenue=100_000_000,
    market_cap=500_000_000
)

var_metrics = saas_company.calculate_var_metrics()

print("\nTechnology Company Risk Analysis:")
print("=" * 60)
print(f"Company Type: SaaS")
print(f"Annual Revenue: ${saas_company.revenue:,.0f}")
print(f"Market Cap: ${saas_company.market_cap:,.0f}")
print("\nRisk Metrics:")
print(f"  Expected Annual Loss: ${var_metrics['expected_loss']:,.0f}")
print(f"  VaR (95%): ${var_metrics['var']:,.0f}")
print(f"  CVaR (95%): ${var_metrics['cvar']:,.0f}")
print(f"  Maximum Simulated Loss: ${var_metrics['max_loss']:,.0f}")
print("\nKey Risk Factors:")
for risk, params in saas_company.risk_factors.items():
    print(f"  {risk}: Frequency={params['frequency']:.1f}/year")
```

## Regulatory and Capital Optimization

### Solvency II / Basel III Compliance

```python
class RegulatoryCapitalOptimizer:
    """Optimize insurance considering regulatory capital requirements."""

    def __init__(self, regulatory_framework='solvency_ii'):
        self.framework = regulatory_framework
        self.capital_requirements = self._load_requirements()

    def _load_requirements(self):
        """Load regulatory capital requirements."""

        if self.framework == 'solvency_ii':
            return {
                'scr_factor': 0.06,  # Solvency Capital Requirement
                'mcr_factor': 0.025,  # Minimum Capital Requirement
                'target_ratio': 1.5   # Target SCR coverage ratio
            }
        elif self.framework == 'basel_iii':
            return {
                'tier1_ratio': 0.06,
                'total_capital_ratio': 0.08,
                'leverage_ratio': 0.03,
                'target_buffer': 1.25
            }

    def calculate_required_capital(self, risk_exposures):
        """Calculate regulatory capital requirements."""

        total_exposure = sum(risk_exposures.values())

        if self.framework == 'solvency_ii':
            scr = total_exposure * self.capital_requirements['scr_factor']
            mcr = total_exposure * self.capital_requirements['mcr_factor']
            target_capital = scr * self.capital_requirements['target_ratio']

            return {
                'scr': scr,
                'mcr': mcr,
                'target': target_capital,
                'minimum_acceptable': scr
            }

    def optimize_insurance_for_capital(self, available_capital, risk_exposures):
        """Optimize insurance to minimize capital requirements."""

        required = self.calculate_required_capital(risk_exposures)
        capital_shortfall = required['minimum_acceptable'] - available_capital

        if capital_shortfall > 0:
            # Need insurance to reduce capital requirements
            reduction_needed = capital_shortfall / self.capital_requirements['scr_factor']

            # Calculate optimal insurance to buy
            recommendations = []
            for risk, exposure in risk_exposures.items():
                if exposure > reduction_needed * 0.2:  # Focus on material risks
                    recommendations.append({
                        'risk': risk,
                        'current_exposure': exposure,
                        'recommended_limit': exposure * 0.8,
                        'capital_benefit': exposure * 0.8 * self.capital_requirements['scr_factor']
                    })

            return recommendations

        return []

# Example regulatory optimization
reg_optimizer = RegulatoryCapitalOptimizer('solvency_ii')

risk_exposures = {
    'market_risk': 50_000_000,
    'credit_risk': 30_000_000,
    'operational_risk': 20_000_000,
    'insurance_risk': 15_000_000
}

available_capital = 8_000_000

capital_req = reg_optimizer.calculate_required_capital(risk_exposures)
insurance_recs = reg_optimizer.optimize_insurance_for_capital(available_capital, risk_exposures)

print("\nRegulatory Capital Analysis:")
print("=" * 60)
print(f"Framework: Solvency II")
print(f"Total Risk Exposure: ${sum(risk_exposures.values()):,.0f}")
print(f"Available Capital: ${available_capital:,.0f}")
print("\nCapital Requirements:")
print(f"  SCR: ${capital_req['scr']:,.0f}")
print(f"  MCR: ${capital_req['mcr']:,.0f}")
print(f"  Target Capital: ${capital_req['target']:,.0f}")
print(f"  Capital Shortfall: ${max(0, capital_req['minimum_acceptable'] - available_capital):,.0f}")

if insurance_recs:
    print("\nInsurance Recommendations to Reduce Capital Requirements:")
    for rec in insurance_recs:
        print(f"  {rec['risk']}: Buy ${rec['recommended_limit']:,.0f} limit")
        print(f"    Capital Benefit: ${rec['capital_benefit']:,.0f}")
```

## Portfolio Optimization

### Optimizing Across Multiple Entities

```python
class InsurancePortfolioOptimizer:
    """Optimize insurance across portfolio of companies."""

    def __init__(self, companies):
        self.companies = companies
        self.correlation_matrix = self._estimate_correlations()

    def _estimate_correlations(self):
        """Estimate risk correlations between companies."""
        n = len(self.companies)
        corr = np.eye(n)

        for i in range(n):
            for j in range(i+1, n):
                # Correlation based on industry and geography
                if self.companies[i]['industry'] == self.companies[j]['industry']:
                    corr[i, j] = corr[j, i] = 0.6
                elif self.companies[i]['region'] == self.companies[j]['region']:
                    corr[i, j] = corr[j, i] = 0.3
                else:
                    corr[i, j] = corr[j, i] = 0.1

        return corr

    def calculate_portfolio_var(self, confidence=0.95):
        """Calculate portfolio-level Value at Risk."""

        # Individual VaRs
        individual_vars = []
        for company in self.companies:
            # Simplified VaR calculation
            var = company['revenue'] * 0.1 * company['risk_factor']
            individual_vars.append(var)

        individual_vars = np.array(individual_vars)

        # Portfolio VaR considering correlation
        portfolio_var = np.sqrt(
            individual_vars @ self.correlation_matrix @ individual_vars.T
        )

        # Diversification benefit
        sum_vars = np.sum(individual_vars)
        diversification_benefit = sum_vars - portfolio_var

        return {
            'portfolio_var': portfolio_var,
            'sum_individual_vars': sum_vars,
            'diversification_benefit': diversification_benefit,
            'diversification_ratio': diversification_benefit / sum_vars
        }

    def optimize_portfolio_insurance(self, total_budget):
        """Optimize insurance allocation across portfolio."""

        from scipy.optimize import minimize

        n = len(self.companies)

        # Objective: minimize portfolio risk
        def objective(allocations):
            # allocations = fraction of budget for each company
            portfolio_risk = 0
            for i, alloc in enumerate(allocations):
                # Risk reduction from insurance
                risk_reduction = min(alloc * total_budget / self.companies[i]['revenue'], 0.5)
                residual_risk = self.companies[i]['risk_factor'] * (1 - risk_reduction)
                portfolio_risk += residual_risk ** 2
            return portfolio_risk

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum to 1
        ]

        # Bounds (0 to 1 for each allocation)
        bounds = [(0, 1) for _ in range(n)]

        # Initial guess (equal allocation)
        x0 = np.ones(n) / n

        # Optimize
        result = minimize(objective, x0, method='SLSQP',
                        bounds=bounds, constraints=constraints)

        optimal_allocations = result.x

        # Calculate insurance for each company
        insurance_allocation = []
        for i, company in enumerate(self.companies):
            insurance_budget = optimal_allocations[i] * total_budget
            insurance_allocation.append({
                'company': company['name'],
                'budget': insurance_budget,
                'coverage': insurance_budget / 0.02,  # Assuming 2% rate
                'risk_reduction': min(insurance_budget / company['revenue'], 0.5)
            })

        return insurance_allocation

# Portfolio of companies
portfolio = [
    {'name': 'TechCo', 'industry': 'technology', 'region': 'US',
     'revenue': 100_000_000, 'risk_factor': 0.3},
    {'name': 'ManuCo', 'industry': 'manufacturing', 'region': 'US',
     'revenue': 150_000_000, 'risk_factor': 0.2},
    {'name': 'FinCo', 'industry': 'finance', 'region': 'EU',
     'revenue': 200_000_000, 'risk_factor': 0.25},
    {'name': 'RetailCo', 'industry': 'retail', 'region': 'Asia',
     'revenue': 80_000_000, 'risk_factor': 0.15}
]

portfolio_optimizer = InsurancePortfolioOptimizer(portfolio)

# Calculate portfolio risk metrics
portfolio_risk = portfolio_optimizer.calculate_portfolio_var()

print("\nPortfolio Risk Analysis:")
print("=" * 60)
print(f"Number of Companies: {len(portfolio)}")
print(f"Total Revenue: ${sum(c['revenue'] for c in portfolio):,.0f}")
print("\nRisk Metrics:")
print(f"  Portfolio VaR: ${portfolio_risk['portfolio_var']:,.0f}")
print(f"  Sum of Individual VaRs: ${portfolio_risk['sum_individual_vars']:,.0f}")
print(f"  Diversification Benefit: ${portfolio_risk['diversification_benefit']:,.0f}")
print(f"  Diversification Ratio: {portfolio_risk['diversification_ratio']:.1%}")

# Optimize insurance allocation
total_insurance_budget = 5_000_000
optimal_insurance = portfolio_optimizer.optimize_portfolio_insurance(total_insurance_budget)

print(f"\nOptimal Insurance Allocation (Budget: ${total_insurance_budget:,.0f}):")
print("-" * 60)
for allocation in optimal_insurance:
    print(f"{allocation['company']:10} Budget: ${allocation['budget']:>10,.0f} "
          f"Coverage: ${allocation['coverage']:>12,.0f} "
          f"Risk Reduction: {allocation['risk_reduction']:>5.1%}")
```

## Summary

This advanced tutorial has covered:

- ✅ Multi-peril insurance programs with complex structures
- ✅ Modeling correlated risks and tail dependencies
- ✅ Dynamic insurance strategies that adapt to financial health
- ✅ Industry-specific risk models and recommendations
- ✅ Regulatory capital optimization
- ✅ Portfolio-level insurance optimization

You now have the tools to handle sophisticated real-world insurance scenarios and make strategic decisions under complex constraints. The framework can be adapted to virtually any industry or risk profile, providing quantitative support for insurance decision-making at both individual company and portfolio levels.

## Next Steps

- Customize the models for your specific industry
- Integrate with real loss data and insurance quotes
- Build monitoring dashboards for dynamic strategies
- Perform stress testing and scenario analysis
- Optimize across multiple objectives and constraints

Remember: The key to successful insurance optimization is balancing growth objectives with survival constraints while considering the unique characteristics of your business and risk environment.
