# Insurance Mathematics

<div style="flex: 1; padding: 15px; border: 2px solid #2196F3; border-radius: 8px; background-color: #E3F2FD;">
    <h3 style="margin-top: 0; color: #1e82d3ff !important;">💰 Why This Matters</h3>
    <p>Insurance mathematics reveal that frequency-severity modeling captures the dual nature of risk (how often losses occur and how severe they are), with heavy-tailed distributions like Pareto essential for modeling catastrophic events where traditional Gaussian assumptions fail to capture the true magnitude of the downside. The layer pricing framework shows why excess-of-loss structures dominate: they efficiently separate attritional losses (predictable, retained) from severity losses (volatile, transferred), optimizing the premium-to-protection tradeoff. Retention optimization through the ergodic lens demonstrates that optimal retention increases with wealth in absolute terms but decreases as a percentage of wealth; i.e., wealthier entities should retain more risk but proportionally less. The compound distribution mathematics proves that aggregate losses have fundamentally different properties than individual claims, explaining why reinsurers price differently than primary insurers. Claims development triangles and chain ladder methods quantify the time value of uncertainty, showing why early reserving decisions compound into material impacts. This framework transforms insurance from a cost center to a growth enabler by quantifying exactly how volatility reduction through strategic risk transfer enhances long-term compound returns, the mathematical foundation for why insurance creates value beyond simple loss indemnification.</p>
</div>

## Table of Contents
1. [Frequency-Severity Models](#frequency-severity-models)
2. [Compound Distributions](#compound-distributions)
3. [Layer Pricing Theory](#layer-pricing-theory)
4. [Retention Optimization](#retention-optimization)
5. [Premium Calculation Principles](#premium-calculation-principles)
6. [Claims Development](#claims-development)
7. [Reinsurance Structures](#reinsurance-structures)
8. [Practical Applications](#practical-applications)
9. [Key Takeaways](#key-takeaways)

(frequency-severity-models)=
## Frequency-Severity Models

### Classical Framework

Insurance losses are modeled as a two-stage process:

1. **Frequency**: Number of claims in a period

2. **Severity**: Size of each claim

Total loss:

$$
S = \sum_{i=1}^{N} X_i
$$

- $N$ = Number of claims (random)
- $X_i$ = Size of $i$-th claim (random)

### Frequency Distributions

#### Poisson Distribution

Most common for claim counts:

$$
P(N = n) = \frac{\lambda^n e^{-\lambda}}{n!}
$$

Properties:

- Mean = Variance = $\lambda$
- Memoryless inter-arrival times
- Suitable for homogeneous risks

*Note: in practice, Over-Dispersed Poisson (ODP), where Variance exceeds the Mean, is preferred because claim estimation introduces uncertainty. For simplicity, we start the implementation with a regular Poisson model.*

#### Negative Binomial

For overdispersed counts (variance > mean) and correlated claims:

$$
P(N = n) = \binom{n + r - 1}{n} p^r (1-p)^n
$$

Properties:
- Mean = $r(1-p)/p$
- Variance = $r(1-p)/p^2$ > Mean
- Captures heterogeneity via mixing

#### Zero-Inflated Models

When many policies have no claims:

$$
P(N = 0) = \pi + (1-\pi)P_0(N = 0)
$$


$$
P(N = n) = (1-\pi)P_0(N = n), \quad n \geq 1
$$

### Severity Distributions

#### Log-Normal

For moderate to large claims:

$$
f(x) = \frac{1}{x\sigma\sqrt{2\pi}} \exp\left[-\frac{(\ln x - \mu)^2}{2\sigma^2}\right]
$$

Properties:

- Right-skewed
- Multiplicative effects
- No upper bound

#### Pareto

For extreme losses (heavy-tailed):

$$
f(x) = \frac{\alpha x_m^\alpha}{x^{\alpha+1}}, \quad x \geq x_m
$$

Properties:

- Power-law tail
- Infinite variance if $\alpha \leq 2$
- Scale-invariant

#### Generalized Pareto (GPD)

For excess losses above threshold:

$$
F(x) = 1 - \left(1 + \xi \frac{x}{\sigma}\right)^{-1/\xi}
$$

- $\xi$ = Shape parameter (tail index)
- $\sigma$ = Scale parameter

### Implementation Example

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


class FrequencySeverityModel:
    """Model insurance losses using frequency-severity approach."""

    def __init__(self, freq_dist, sev_dist):
        self.freq_dist = freq_dist
        self.sev_dist = sev_dist

    def simulate_annual_loss(self, n_sims=10_000):
        """Simulate total annual losses."""
        total_losses = []

        for _ in range(n_sims):
            # Number of claims
            n_claims = self.freq_dist.rvs()

            if n_claims == 0:
                total_losses.append(0)
            else:
                # Individual claim amounts
                claims = self.sev_dist.rvs(size=n_claims)
                total_losses.append(np.sum(claims))

        return np.array(total_losses)

    def calculate_statistics(self, losses):
        """Calculate key statistics."""
        return {
            'mean': np.mean(losses),
            'std': np.std(losses),
            'median': np.median(losses),
            'p95': np.percentile(losses, 95),
            'p99': np.percentile(losses, 99),
            'p99.5': np.percentile(losses, 99.5),
            'max': np.max(losses),
            'prob_zero': np.mean(losses == 0)
        }

    def plot_distribution(self, losses):
        """Visualize loss distribution."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Histogram
        axes[0, 0].hist(losses[losses > 0], bins=50,
                        edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Loss Amount')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Loss Distribution (excluding zeros)')

        # Log-log plot for tail
        sorted_losses = np.sort(losses[losses > 0])
        exceedance_prob = np.arange(len(sorted_losses), 0, -1) / len(losses)
        axes[0, 1].loglog(sorted_losses, exceedance_prob)
        axes[0, 1].set_xlabel('Loss Amount (log scale)')
        axes[0, 1].set_ylabel('Exceedance Probability (log scale)')
        axes[0, 1].set_title('Tail Behavior')

        # Recreate Q-Q plot manually to control colors: data points in default blue, fit line in orange
        (osm, osr), (slope, intercept, r) = stats.probplot(
            np.log(losses[losses > 0]), dist="norm", fit=True)
        axes[1, 0].plot(osm, osr, marker='.', linestyle='none',
                        markersize=4, color='C0', alpha=0.8)
        axes[1, 0].plot(osm, slope * np.asarray(osm) + intercept,
                        color='orange', linestyle='--', linewidth=1.5)

        # Empirical CDF
        axes[1, 1].plot(sorted_losses, np.arange(
            1, len(sorted_losses) + 1) / len(sorted_losses))
        axes[1, 1].set_xlabel('Loss Amount')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Empirical CDF')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# Example: Commercial property insurance
freq_dist = stats.poisson(mu=3)  # 3 claims per year on average
sev_dist = stats.lognorm(s=2, scale=50_000)  # Log-normal severity

model = FrequencySeverityModel(freq_dist, sev_dist)
losses = model.simulate_annual_loss(n_sims=10_000)
statistics = model.calculate_statistics(losses)

model.plot_distribution(losses)

print("Annual Loss Statistics:")
for key, value in statistics.items():
    if key == 'prob_zero':
        print(f"{key}: {value:.1%}")
    else:
        print(f"{key}: ${value:,.0f}")
```

#### Sample Output

![Log-Normal Sample Plots](figures/lognormal_sample.png)

```
Annual Loss Statistics:
mean: $1,089,751
std: $3,579,880
median: $318,641
p95: $4,191,252
p99: $12,699,935
p99.5: $18,849,964
max: $196,094,596
prob_zero: 5.0%
```

(compound-distributions)=
## Compound Distributions

### Definition

The compound distribution of total losses $S = \sum_{i=1}^N X_i$ has:

**Characteristic function**:

$$
\phi_S(t) = G_N(\phi_X(t))
$$

where $G_N$ is the probability generating function of $N$.

### Compound Poisson

When frequency $N \sim \text{Poisson}(\lambda)$ and severities $X_i$ are i.i.d.:

**Mean**: $E[S] = \lambda \cdot E[X]$
**Variance**: $\text{Var}(S) = \lambda \cdot E[X^2]$
**Skewness**: $\text{Skew}(S) = \frac{E[X^3]}{\lambda^{1/2} \cdot E[X^2]^{3/2}}$

### Panjer Recursion

For discrete severities, recursive approximation of $S$ is given by:

$$p_k = \frac{1}{1 - af_0} \sum_{j=1}^k \left(a + \frac{bj}{k}\right) f_j p_{k-j}$$

where:
- $p_k = P(S = k)$
- $f_j = P(X = j)$
- $(a, b)$ depend on frequency distribution

### Fast Fourier Transform Method

FFT provides a computationally efficient approach for determining the aggregate loss distribution for a given frequency-severity model. The method exploits the relationship between the characteristic functions and probability generating functions.

For continuous distributions:

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def compound_distribution_fft(freq_params, sev_params, x_max=1e8, n_points=2**16):
    """Calculate compound distribution using FFT - CORRECTED."""

    dx = x_max / n_points
    x = np.arange(n_points) * dx

    # Discretize severity distribution WITHOUT normalization
    sev_pmf = stats.lognorm.pdf(x[1:], s=sev_params["s"], scale=sev_params["scale"]) * dx

    # Prepend zero for x[0] since individual claims can't be zero
    sev_pmf = np.concatenate([[0], sev_pmf])

    # Check how much probability mass we captured (should be close to 1)
    captured_mass = sev_pmf.sum()
    print(f"Captured severity mass: {captured_mass:.6f}")

    # DON'T normalize - use the natural discretization
    # The small missing mass in the tail is acceptable

    # FFT operations
    sev_cf = np.fft.fft(sev_pmf)
    lambda_param = freq_params["lambda"]
    compound_cf = np.exp(lambda_param * (sev_cf - 1))
    compound_pmf = np.real(np.fft.ifft(compound_cf))

    # The compound_pmf now contains:
    # - At index 0: P(S = 0) (point mass)
    # - At other indices: probability masses for discretized positive values

    prob_zero = compound_pmf[0]

    # Convert to density (but keep point mass separate)
    pdf = compound_pmf / dx
    pdf[0] = 0  # Remove point mass from density

    return x, pdf, prob_zero

# Calculate FFT solution
freq_params={"lambda": 3}
sev_params={"s": 2, "scale": 50_000}
x, pdf, prob_zero = compound_distribution_fft(freq_params=freq_params, sev_params=sev_params)

# Simulate
n_sims = 100_000
freq = stats.poisson(mu=3)
sev = stats.lognorm(s=2, scale=50_000)

sim_losses = np.zeros(n_sims)
for i in range(n_sims):
    n_claims = freq.rvs()
    if n_claims > 0:
        sim_losses[i] = sev.rvs(size=n_claims).sum()

plt.figure(figsize=(10, 6))

# Plot with adjusted bins
x_plot_max = x[999]
bin_edges = np.concatenate([[0, 1], np.linspace(1, x_plot_max, 199)])
plt.hist(sim_losses, bins=bin_edges, density=True,
         alpha=0.45, color='C1', zorder=1, label='Simulated (adjusted bins)')

# Plot continuous part
plt.semilogy(x[1:1000], pdf[1:1000], label='FFT-based Density', linewidth=2)

# Mark the point mass
plt.scatter([0], [pdf[1]], color='red', s=100, zorder=5,
            label=f'Point mass: {prob_zero:.3f}')

plt.xlabel("Total Annual Loss")
plt.ylabel("Probability Density (log scale)")
plt.title("Compound Poisson-Lognormal Distribution")
plt.grid(True, alpha=0.3)
plt.legend(loc='best')
plt.show()

# Better validation: Compare key statistics
sim_mean = np.mean(sim_losses)
sim_std = np.std(sim_losses)

# Theoretical values
theory_mean = freq_params["lambda"] * stats.lognorm.mean(s=sev_params["s"], scale=sev_params["scale"])
theory_var = (freq_params["lambda"] *
              stats.lognorm.var(s=sev_params["s"], scale=sev_params["scale"]) +
              freq_params["lambda"] *
              stats.lognorm.mean(s=sev_params["s"], scale=sev_params["scale"])**2)
theory_std = np.sqrt(theory_var)

print(f"\nValidation Statistics:")
print(f"Mean - Simulated: {sim_mean:,.0f}, Theoretical: {theory_mean:,.0f}")
print(f"Std Dev - Simulated: {sim_std:,.0f}, Theoretical: {theory_std:,.0f}")
print(f"P(Loss = 0) - Simulated: {np.mean(sim_losses == 0):.4f}, Theoretical: {prob_zero:.4f}")
```

#### Sample Output

![Compound Poisson-Lognormal Distribution](figures/compound_poi_lognormal.png)

```
Validation Statistics:
Mean - Simulated: 1,109,983, Theoretical: 1,108,358
Std Dev - Simulated: 4,216,943, Theoretical: 4,728,338
P(Loss = 0) - Simulated: 0.0496, Theoretical: 0.0498
```

(layer-pricing-theory)=
## Layer Pricing Theory

### Excess of Loss Layers  Insurance coverage is structured in layers:

- **Primary**: $\$0$ to $L_1$
- **First Excess**: $L_1$ to $L_2$
- **Second Excess**: $L_2$ to $L_3$, etc.

### Layer Loss Calculation

For layer$[a, b]$, the loss is:

$$ Y_{[a,b]} = \min(X, b) - \min(X, a) = (X \wedge b) - (X \wedge a) $$

Expected layer loss:

$$ E[Y_{[a,b]}] = \int_a^b [1 - F_X(x)] dx $$

### Increased Limits Factors (ILFs)

Ratio of expected loss at different limits:

$$ \text{ILF}(L) = \frac{E[X \wedge L]}{E[X \wedge L_0]} $$

where $L_0$ is the base limit.

### Exposure Curves

Proportion of loss in layer:

$$ \text{G}(r) = \frac{E[X \wedge rM]}{E[X]} $$

where $M$ is the maximum possible loss.

### Layer Pricing Example

For an exploration of Insurance Layers, see this [Jupyter Notebook on Insurance Layer Optimization](../../notebooks/07_insurance_layers.ipynb).

![Layer Pricing](../../../theory/figures/layer_pricing.png)

(retention-optimization)=
## Retention Optimization

### Objective Function

Maximize utility or growth:

$$ \max_R \quad U(W - P(R) - (L \wedge R)) $$

- $R$ = Retention level
- $P(R)$ = Premium function
- $L$ = Random loss
- $W$ = Initial wealth

This function balances lower premium $P(R)$ vs risk exposure $(L \wedge R)$.

### First-Order Condition
For differentiable utility:
$$ P'(R) = E[U'(W - P(R) - (L \wedge R)) \cdot \mathbf{1}_{L > R}] $$

### Ergodic Optimization

Maximize time-average growth:

$$ \max_R \quad E[\ln(W - P(R) - L \wedge R)] $$

### Constraints

1. **Budget constraint**:$P(R) \leq B$
2. **Ruin constraint**:$P(\text{ruin}) \leq \alpha$
3. **Regulatory minimum**: $R \geq R_{\text{min}}$

### Ergodic Optimization Example

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. CORRECTED PREMIUM AND OBJECTIVE FUNCTIONS
# ============================================

def premium_function(R, expected_loss, loading=0.3, expense_ratio=0.05):
    """
    Premium function for retention R.
    Premium decreases as retention increases (you keep more risk).
    """
    # Create loss distribution (lognormal with CV = 0.5)
    loss_dist = stats.lognorm(s=0.5, scale=expected_loss)

    # Calculate E[max(L - R, 0)] using the lognormal properties
    # More efficient calculation using CDF
    def excess_loss(R):
        if R <= 0:
            return expected_loss
        # For lognormal, we can use the partial expectation formula
        mu = np.log(expected_loss) - 0.5 * 0.5**2
        sigma = 0.5

        # Partial expectation E[L * 1{L>R}]
        z = (np.log(R) - mu - sigma**2) / sigma
        partial_exp = expected_loss * np.exp(0.5 * sigma**2) * (1 - stats.norm.cdf(z))

        # E[max(L-R, 0)] = E[L * 1{L>R}] - R * P(L>R)
        prob_exceed = 1 - loss_dist.cdf(R)
        expected_excess = partial_exp - R * prob_exceed

        return expected_excess

    # Expected excess loss
    expected_excess = excess_loss(R)

    # Premium with loading and expenses
    premium = (1 + loading) * expected_excess + expense_ratio * expected_loss

    return premium

def ergodic_objective(R, wealth, loss_dist, loading=0.3):
    """
    Corrected ergodic objective: E[ln(W - P(R) - min(L, R))]
    """
    # Calculate premium for this retention
    premium = premium_function(R, loss_dist.mean(), loading)

    # Check if we can afford this strategy
    if premium >= wealth * 0.9:  # Can't spend 90%+ of wealth on premium
        return -np.inf

    # More samples for better accuracy
    np.random.seed(42)
    n_samples = 50000
    losses = loss_dist.rvs(n_samples)

    # Calculate retained losses
    retained_losses = np.minimum(losses, R)

    # Final wealth for each scenario
    final_wealth = wealth - premium - retained_losses

    # Check for bankruptcy
    if np.any(final_wealth <= 0):
        # Calculate probability-weighted utility including bankruptcy scenarios
        valid = final_wealth > 0
        if not valid.any():
            return -np.inf

        # Penalize strategies that lead to bankruptcy
        bankruptcy_penalty = -100  # Large negative utility for bankruptcy
        utility = np.where(valid, np.log(final_wealth), bankruptcy_penalty)
        return np.mean(utility)

    # Expected log wealth (ergodic growth rate)
    return np.mean(np.log(final_wealth))

def solve_ergodic_retention(wealth, expected_loss=100_000, loading=0.3, cv=0.5):
    """
    Find optimal retention that maximizes ergodic growth rate.
    Uses global optimization to avoid local minima.
    """
    # Create loss distribution
    loss_dist = stats.lognorm(s=cv, scale=expected_loss)

    # Objective function (negative for minimization)
    def neg_objective(R):
        return -ergodic_objective(R, wealth, loss_dist, loading)

    # Bounds: retention between small value and reasonable maximum
    min_retention = expected_loss * 0.001  # Very small retention
    max_retention = min(wealth * 0.3, expected_loss * 3)  # Conservative upper bound

    # Use global optimization to avoid local minima
    from scipy.optimize import differential_evolution

    result = differential_evolution(
        neg_objective,
        bounds=[(min_retention, max_retention)],
        seed=42,
        maxiter=100,
        workers=1
    )

    optimal_retention = result.x[0]
    optimal_growth_rate = -result.fun

    return optimal_retention, optimal_growth_rate

# ============================================
# 2. IMPROVED DYNAMIC PROGRAMMING
# ============================================

def dp_ergodic_retention(wealth_grid, loss_dist, n_periods=10,
                        loading=0.3, discount=0.95):
    """
    Improved DP using continuous optimization at each step instead of grid search
    """
    n_wealth = len(wealth_grid)

    # Store optimal retentions (continuous values, not grid indices)
    optimal_retentions = np.zeros((n_periods, n_wealth))
    V = np.zeros((n_periods + 1, n_wealth))

    # Terminal value
    V[-1, :] = np.log(np.maximum(wealth_grid, 1e-6))

    # Backward induction with continuous optimization
    for t in range(n_periods - 1, -1, -1):
        for i, w in enumerate(wealth_grid):

            # Define objective for this (t, w) pair
            def objective(R):
                # Calculate immediate payoff and continuation value
                premium = premium_function(R, loss_dist.mean(), loading)

                if premium > w * 0.3 or R > w * 0.5:
                    return -np.inf

                # Sample losses
                losses = loss_dist.rvs(2000)
                retained = np.minimum(losses, R)
                next_wealth = w - premium - retained

                valid = next_wealth > 0
                if not valid.any():
                    return -np.inf

                current_util = np.mean(np.log(next_wealth[valid]))

                if t < n_periods - 1:
                    # Interpolate continuation values
                    from scipy.interpolate import interp1d
                    value_func = interp1d(wealth_grid, V[t + 1, :],
                                        kind='cubic', bounds_error=False,
                                        fill_value='extrapolate')
                    continuation = discount * np.mean(value_func(next_wealth[valid]))
                else:
                    continuation = 0

                return current_util + continuation

            # Continuous optimization instead of grid search
            from scipy.optimize import minimize_scalar

            # Smart bounds based on wealth level
            R_min = max(1000, w * 0.001)
            R_max = min(w * 0.3, loss_dist.mean() * 2)

            result = minimize_scalar(
                lambda R: -objective(R),
                bounds=(R_min, R_max),
                method='bounded'
            )

            optimal_retentions[t, i] = result.x
            V[t, i] = -result.fun

    return V, optimal_retentions

# ============================================
# 3. ANALYSIS WITH CORRECTIONS
# ============================================

# Set up parameters
expected_loss = 100_000
cv = 0.5
loading = 0.3
loss_dist = stats.lognorm(s=cv, scale=expected_loss)

# 1. Single-period analysis
print("Running single-period analysis...")
wealth_levels = np.linspace(500_000, 20_000_000, 20)
optimal_retentions = []
growth_rates = []

for wealth in wealth_levels:
    R_opt, g_opt = solve_ergodic_retention(wealth, expected_loss, loading, cv)
    optimal_retentions.append(R_opt)
    growth_rates.append(g_opt)
    print(f"Wealth ${wealth/1e6:.1f}M: Retention ${R_opt/1e3:.1f}K")

optimal_retentions = np.array(optimal_retentions)
growth_rates = np.array(growth_rates)

# 2. Multi-period DP with finer grids
print("\nRunning multi-period DP...")
wealth_grid_dp = np.linspace(500_000, 10_000_000, 15)  # Fewer wealth points for speed

V, optimal_policy = dp_ergodic_retention(wealth_grid_dp, loss_dist,
                                        n_periods=10, loading=loading,
                                        discount=0.95)

# 3. Compare different loadings (CORRECTED EXPECTATION)
print("\nAnalyzing loading effects...")
loadings = [0.1, 0.3, 0.5, 0.7]
retention_by_loading = {}

for load in loadings:
    print(f"Loading {load:.1%}...")
    retentions = []
    for wealth in wealth_levels:
        R_opt, _ = solve_ergodic_retention(wealth, expected_loss, load, cv)
        retentions.append(R_opt)
    retention_by_loading[load] = np.array(retentions)

# ============================================
# 4. PLOTTING WITH CORRECTIONS
# ============================================

fig = plt.figure(figsize=(16, 12))

# Plot 1: Optimal retention vs wealth
ax1 = plt.subplot(2, 3, 1)
ax1.plot(wealth_levels/1e6, optimal_retentions/1e3, 'b-', linewidth=2)
ax1.set_xlabel('Wealth ($M)')
ax1.set_ylabel('Optimal Retention ($K)')
ax1.set_title('Ergodic Optimal Retention\n(Single Period)')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=expected_loss/1e3, color='r', linestyle='--', alpha=0.5, label='Expected Loss')
ax1.legend()

# Plot 2: Retention as percentage of wealth
ax2 = plt.subplot(2, 3, 2)
retention_pct = (optimal_retentions / wealth_levels) * 100
ax2.plot(wealth_levels/1e6, retention_pct, 'g-', linewidth=2)
ax2.set_xlabel('Wealth ($M)')
ax2.set_ylabel('Retention as % of Wealth')
ax2.set_title('Relative Risk Retention\n(Should decrease with wealth)')
ax2.grid(True, alpha=0.3)

# Plot 3: Expected growth rate
ax3 = plt.subplot(2, 3, 3)
ax3.plot(wealth_levels/1e6, growth_rates, 'r-', linewidth=2)
ax3.set_xlabel('Wealth ($M)')
ax3.set_ylabel('Expected Log Growth Rate')
ax3.set_title('Ergodic Growth Rate at Optimum')
ax3.grid(True, alpha=0.3)

# Plot 4: Effect of loading (CORRECTED)
ax4 = plt.subplot(2, 3, 4)
colors = ['blue', 'green', 'orange', 'red']
for (load, retentions), color in zip(retention_by_loading.items(), colors):
    ax4.plot(wealth_levels/1e6, retentions/1e3, linewidth=2,
             label=f'Loading = {load:.0%}', color=color)
ax4.set_xlabel('Wealth ($M)')
ax4.set_ylabel('Optimal Retention ($K)')
ax4.set_title('Impact of Premium Loading\n(Higher loading → Higher retention)')
ax4.grid(True, alpha=0.3)
ax4.legend()

# Verify the relationship
wealth_test = 5_000_000
print(f"\nVerifying loading relationship at ${wealth_test/1e6}M wealth:")
for load in loadings:
    idx = np.argmin(np.abs(wealth_levels - wealth_test))
    ret = retention_by_loading[load][idx]
    print(f"  Loading {load:.0%}: Retention ${ret/1e3:.1f}K")

# Plot 5: Multi-period vs single-period (with finer grid)
ax5 = plt.subplot(2, 3, 5)
ax5.plot(wealth_grid_dp/1e6, optimal_policy[0, :]/1e3, 'b-', linewidth=2,
         label='DP Solution (t=0)', marker='o', markersize=4)

# Compute single-period for comparison
sp_retentions = []
for w in wealth_grid_dp:
    R_opt, _ = solve_ergodic_retention(w, expected_loss, loading, cv)
    sp_retentions.append(R_opt)
ax5.plot(wealth_grid_dp/1e6, np.array(sp_retentions)/1e3, 'r--',
         linewidth=2, label='Single Period', marker='s', markersize=4)
ax5.set_xlabel('Wealth ($M)')
ax5.set_ylabel('Optimal Retention ($K)')
ax5.set_title('DP vs Single-Period\n(Should be similar with fine grid)')
ax5.grid(True, alpha=0.3)
ax5.legend()

# Plot 6: Retention over time (FIXED to show wealth dependence)
ax6 = plt.subplot(2, 3, 6)
time_periods = np.arange(optimal_policy.shape[0])
wealth_indices = [2, 5, 8, 11, 14]  # More spread out indices
colors_time = ['purple', 'blue', 'green', 'orange', 'red']

for idx, color in zip(wealth_indices, colors_time):
    if idx < len(wealth_grid_dp):
        wealth_val = wealth_grid_dp[idx]
        ax6.plot(time_periods, optimal_policy[:, idx]/1e3,
                linewidth=2, label=f'W=${wealth_val/1e6:.1f}M',
                marker='o', color=color, alpha=0.6)

ax6.set_xlabel('Time Period')
ax6.set_ylabel('Optimal Retention ($K)')
ax6.set_title('Retention Policy Over Time\n(Should show wealth dependence)')
ax6.grid(True, alpha=0.3)
ax6.legend()
ax6.set_xticks(time_periods)

plt.suptitle('CORRECTED: Ergodic (Kelly) Insurance Retention Optimization\n' +
             f'Loss: LogNormal(μ=${expected_loss/1e3:.0f}K, CV={cv})',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

#### Sample Output

![Retention Optimization](figures/ergodic_optimization_example.png)

(premium-calculation-principles)=
## Premium Calculation Principles

There are several approaches to determining what premium to charge for insurance.

### Pure Premium

Expected loss costs only:

$$ P_0 = E[L] $$

This is typically the foundation of premium rates.

### Expected Value Principle

Add proportional loading:

$$ P = (1 + \theta) E[L] $$

where $\theta$ is the safety loading.

### Variance Principle

Account for risk:

$$ P = E[L] + \alpha \cdot \text{Var}(L) $$

### Standard Deviation Principle

$$ P = E[L] + \beta \cdot \text{SD}(L) $$

### Exponential Principle

Based on exponential utility:

$$ P = \frac{1}{\alpha} \ln(E[e^{\alpha L}]) $$

### Wang Transform

Distort probability measure:

$$ P = \int_0^\infty g(S_L(x)) dx $$

where $g$ is the distortion function.

### Implementation Comparison

```python
import numpy as np
from scipy import stats
from scipy.integrate import quad, quad_vec
from scipy.special import ndtr
import warnings

class PremiumPrinciples:
    """
    Compare different premium calculation methods for P&C insurance.
    Enhanced with numerical stability and realistic loss distributions.
    """

    def __init__(self, loss_dist, max_loss=None):
        self.loss_dist = loss_dist
        self.mean = loss_dist.mean()
        self.std = loss_dist.std()
        self.var = loss_dist.var()

        # Set practical upper bound for integration (99.99th percentile or specified max)
        self.max_loss = max_loss if max_loss else loss_dist.ppf(0.9999)

        # Store some percentiles for context
        self.percentiles = {
            'p50': loss_dist.ppf(0.50),
            'p75': loss_dist.ppf(0.75),
            'p90': loss_dist.ppf(0.90),
            'p95': loss_dist.ppf(0.95),
            'p99': loss_dist.ppf(0.99),
            'p99.5': loss_dist.ppf(0.995),
        }

    def pure_premium(self):
        """Net premium with no loading."""
        return self.mean

    def expected_value(self, loading=0.3):
        """Expected value principle with proportional loading."""
        return self.mean * (1 + loading)

    def variance_principle(self, alpha=0.00001):
        """Variance principle: μ + α·σ²
        Alpha typically small for P&C (e.g., 0.00001 to 0.0001)"""
        return self.mean + alpha * self.var

    def standard_deviation(self, beta=0.5):
        """Standard deviation principle: μ + β·σ
        Beta typically 0.3 to 0.7 for P&C"""
        return self.mean + beta * self.std

    def exponential_principle(self, alpha=None):
        """
        Exponential/Esscher principle using limited integration range.
        For heavy-tailed distributions, we use a small alpha and bounded integration.
        """
        # Auto-select alpha if not provided (smaller for higher variance)
        if alpha is None:
            cv = self.std / self.mean  # Coefficient of variation
            alpha = min(0.5 / self.std, 0.001)  # Adaptive alpha based on volatility

        try:
            # Use bounded integration to avoid numerical issues
            def integrand(x):
                return np.exp(alpha * x) * self.loss_dist.pdf(x)

            # Integrate over practical range
            mgf, error = quad(integrand, 0, self.max_loss, limit=100)

            if mgf <= 0 or np.isnan(mgf) or np.isinf(mgf):
                # Fallback to approximation using cumulants
                return self.mean + alpha * self.var / 2  # Second-order approximation

            return np.log(mgf) / alpha

        except Exception as e:
            # Fallback to Taylor approximation
            return self.mean + alpha * self.var / 2

    def wang_transform(self, lambda_param=0.25):
        """
        Wang transform with power distortion function.
        g(S(x)) = S(x)^(1/(1+λ)) where S(x) is survival function
        Lambda typically 0.1 to 0.5 for P&C
        """
        try:
            def distorted_survival_density(x):
                # Survival function
                S_x = 1 - self.loss_dist.cdf(x)

                # Avoid numerical issues near boundaries
                if S_x <= 1e-10:
                    return 0

                # Power distortion: g(u) = u^(1/(1+λ))
                exponent = 1 / (1 + lambda_param)
                return S_x ** exponent

            # Integrate the distorted survival function
            premium, error = quad(distorted_survival_density, 0, self.max_loss,
                                 limit=100, epsabs=1e-8)

            return premium

        except Exception as e:
            print(f"Wang Transform calculation error: {e}")
            return np.nan

    def swiss_solvency(self, confidence=0.99):
        """
        Swiss Solvency Test principle (simplified).
        Premium = Mean + Loading based on tail risk
        """
        tvar = self.tail_value_at_risk(confidence)
        loading_factor = 0.06  # Typical SST cost of capital rate
        return self.mean + loading_factor * (tvar - self.mean)

    def tail_value_at_risk(self, confidence=0.99):
        """Calculate TVaR (Conditional Tail Expectation)."""
        var = self.loss_dist.ppf(confidence)

        def tail_expectation(x):
            return x * self.loss_dist.pdf(x)

        tail_integral, _ = quad(tail_expectation, var, self.max_loss, limit=100)
        tail_prob = 1 - confidence

        if tail_prob > 0:
            return tail_integral / tail_prob
        else:
            return var

    def distortion_principle_general(self, distortion_func):
        """
        General distortion principle for any distortion function.
        Premium = ∫[0,∞] g(S(x)) dx where g is the distortion function
        """
        def integrand(x):
            survival = 1 - self.loss_dist.cdf(x)
            return distortion_func(survival)

        premium, _ = quad(integrand, 0, self.max_loss, limit=100)
        return premium

    def compare_all(self, show_stats=True):
        """Compare all premium principles with enhanced output."""

        if show_stats:
            print("=" * 70)
            print("LOSS DISTRIBUTION STATISTICS")
            print("=" * 70)
            print(f"Mean Loss:              ${self.mean:15,.2f}")
            print(f"Standard Deviation:     ${self.std:15,.2f}")
            print(f"Coefficient of Var:     {self.std/self.mean:15.3f}")
            print(f"Skewness:               {self.loss_dist.stats(moments='s'):15.3f}")
            print("\nKey Percentiles:")
            for p_name, p_val in self.percentiles.items():
                print(f"  {p_name:5}:               ${p_val:15,.2f}")
            print(f"\nIntegration bound:      ${self.max_loss:15,.2f}")
            print("\n" + "=" * 70)
            print("PREMIUM CALCULATIONS")
            print("=" * 70)

        results = {
            'Pure Premium': self.pure_premium(),
            'Expected Value (30%)': self.expected_value(0.3),
            'Expected Value (40%)': self.expected_value(0.4),
            'Variance (α=0.00001)': self.variance_principle(0.00001),
            'Std Dev (β=0.5)': self.standard_deviation(0.5),
            'Std Dev (β=0.7)': self.standard_deviation(0.7),
            'Exponential': self.exponential_principle(),
            'Wang Transform (λ=0.25)': self.wang_transform(0.25),
            'Wang Transform (λ=0.50)': self.wang_transform(0.50),
            'Swiss Solvency (99%)': self.swiss_solvency(0.99),
            'TVaR (99%)': self.tail_value_at_risk(0.99),
        }

        # Calculate loadings and display
        for name, premium in results.items():
            if np.isnan(premium) or np.isinf(premium):
                print(f"{name:28} ${'N/A':>15} (Loading:     N/A)")
            else:
                loading = (premium / self.mean - 1) * 100
                print(f"{name:28} ${premium:15,.2f} (Loading: {loading:7.2f}%)")

        return results

print("Lognormal distribution")
loss_dist = stats.lognorm(s=0.9, scale=100_000)
prem_principles = PremiumPrinciples(loss_dist)
premiums = prem_principles.compare_all()
```

#### Sample Output

```
import numpy as np
from scipy import stats
from scipy.integrate import quad, quad_vec
from scipy.special import ndtr
import warnings

class PremiumPrinciples:
    """
    Compare different premium calculation methods for P&C insurance.
    Enhanced with numerical stability and realistic loss distributions.
    """

    def __init__(self, loss_dist, max_loss=None):
        self.loss_dist = loss_dist
        self.mean = loss_dist.mean()
        self.std = loss_dist.std()
        self.var = loss_dist.var()

        # Set practical upper bound for integration (99.99th percentile or specified max)
        self.max_loss = max_loss if max_loss else loss_dist.ppf(0.9999)

        # Store some percentiles for context
        self.percentiles = {
            'p50': loss_dist.ppf(0.50),
            'p75': loss_dist.ppf(0.75),
            'p90': loss_dist.ppf(0.90),
            'p95': loss_dist.ppf(0.95),
            'p99': loss_dist.ppf(0.99),
            'p99.5': loss_dist.ppf(0.995),
        }

    def pure_premium(self):
        """Net premium with no loading."""
        return self.mean

    def expected_value(self, loading=0.3):
        """Expected value principle with proportional loading."""
        return self.mean * (1 + loading)

    def variance_principle(self, alpha=0.00001):
        """Variance principle: μ + α·σ²
        Alpha typically small for P&C (e.g., 0.00001 to 0.0001)"""
        return self.mean + alpha * self.var

    def standard_deviation(self, beta=0.5):
        """Standard deviation principle: μ + β·σ
        Beta typically 0.3 to 0.7 for P&C"""
        return self.mean + beta * self.std

    def exponential_principle(self, alpha=None):
        """
        Exponential/Esscher principle using limited integration range.
        For heavy-tailed distributions, we use a small alpha and bounded integration.
        """
        # Auto-select alpha if not provided (smaller for higher variance)
        if alpha is None:
            cv = self.std / self.mean  # Coefficient of variation
            alpha = min(0.5 / self.std, 0.001)  # Adaptive alpha based on volatility

        try:
            # Use bounded integration to avoid numerical issues
            def integrand(x):
                return np.exp(alpha * x) * self.loss_dist.pdf(x)

            # Integrate over practical range
            mgf, error = quad(integrand, 0, self.max_loss, limit=100)

            if mgf <= 0 or np.isnan(mgf) or np.isinf(mgf):
                # Fallback to approximation using cumulants
                return self.mean + alpha * self.var / 2  # Second-order approximation

            return np.log(mgf) / alpha

        except Exception as e:
            # Fallback to Taylor approximation
            return self.mean + alpha * self.var / 2

    def wang_transform(self, lambda_param=0.25):
        """
        Wang transform with power distortion function.
        g(S(x)) = S(x)^(1/(1+λ)) where S(x) is survival function
        Lambda typically 0.1 to 0.5 for P&C
        """
        try:
            def distorted_survival_density(x):
                # Survival function
                S_x = 1 - self.loss_dist.cdf(x)

                # Avoid numerical issues near boundaries
                if S_x <= 1e-10:
                    return 0

                # Power distortion: g(u) = u^(1/(1+λ))
                exponent = 1 / (1 + lambda_param)
                return S_x ** exponent

            # Integrate the distorted survival function
            premium, error = quad(distorted_survival_density, 0, self.max_loss,
                                 limit=100, epsabs=1e-8)

            return premium

        except Exception as e:
            print(f"Wang Transform calculation error: {e}")
            return np.nan

    def swiss_solvency(self, confidence=0.99):
        """
        Swiss Solvency Test principle (simplified).
        Premium = Mean + Loading based on tail risk
        """
        tvar = self.tail_value_at_risk(confidence)
        loading_factor = 0.06  # Typical SST cost of capital rate
        return self.mean + loading_factor * (tvar - self.mean)

    def tail_value_at_risk(self, confidence=0.99):
        """Calculate TVaR (Conditional Tail Expectation)."""
        var = self.loss_dist.ppf(confidence)

        def tail_expectation(x):
            return x * self.loss_dist.pdf(x)

        tail_integral, _ = quad(tail_expectation, var, self.max_loss, limit=100)
        tail_prob = 1 - confidence

        if tail_prob > 0:
            return tail_integral / tail_prob
        else:
            return var

    def distortion_principle_general(self, distortion_func):
        """
        General distortion principle for any distortion function.
        Premium = ∫[0,∞] g(S(x)) dx where g is the distortion function
        """
        def integrand(x):
            survival = 1 - self.loss_dist.cdf(x)
            return distortion_func(survival)

        premium, _ = quad(integrand, 0, self.max_loss, limit=100)
        return premium

    def compare_all(self, show_stats=True):
        """Compare all premium principles with enhanced output."""

        if show_stats:
            print("=" * 70)
            print("LOSS DISTRIBUTION STATISTICS")
            print("=" * 70)
            print(f"Mean Loss:              ${self.mean:15,.2f}")
            print(f"Standard Deviation:     ${self.std:15,.2f}")
            print(f"Coefficient of Var:     {self.std/self.mean:15.3f}")
            print(f"Skewness:               {self.loss_dist.stats(moments='s'):15.3f}")
            print("\nKey Percentiles:")
            for p_name, p_val in self.percentiles.items():
                print(f"  {p_name:5}:               ${p_val:15,.2f}")
            print(f"\nIntegration bound:      ${self.max_loss:15,.2f}")
            print("\n" + "=" * 70)
            print("PREMIUM CALCULATIONS")
            print("=" * 70)

        results = {
            'Pure Premium': self.pure_premium(),
            'Expected Value (30%)': self.expected_value(0.3),
            'Expected Value (40%)': self.expected_value(0.4),
            'Variance (α=0.00001)': self.variance_principle(0.00001),
            'Std Dev (β=0.5)': self.standard_deviation(0.5),
            'Std Dev (β=0.7)': self.standard_deviation(0.7),
            'Exponential': self.exponential_principle(),
            'Wang Transform (λ=0.25)': self.wang_transform(0.25),
            'Wang Transform (λ=0.50)': self.wang_transform(0.50),
            'Swiss Solvency (99%)': self.swiss_solvency(0.99),
            'TVaR (99%)': self.tail_value_at_risk(0.99),
        }

        # Calculate loadings and display
        for name, premium in results.items():
            if np.isnan(premium) or np.isinf(premium):
                print(f"{name:28} ${'N/A':>15} (Loading:     N/A)")
            else:
                loading = (premium / self.mean - 1) * 100
                print(f"{name:28} ${premium:15,.2f} (Loading: {loading:7.2f}%)")

        return results

print("Lognormal distribution")
loss_dist = stats.lognorm(s=0.9, scale=100_000)
prem_principles = PremiumPrinciples(loss_dist)
premiums = prem_principles.compare_all()
```

(claims-development)=
## Claims Development

### Development Triangles

Claims develop over time:
| Year | Dev 0 | Dev 1 | Dev 2 | Dev 3 | Ultimate |
|------|-------|-------|-------|-------|----------|
| 2020 | 100   | 150   | 170   | 175   | 175      |
| 2021 | 110   | 165   | 187   | ?     | ?        |
| 2022 | 120   | 180   | ?     | ?     | ?        |
| 2023 | 130   | ?     | ?     | ?     | ?        |

### Chain Ladder Method

Development factors:

$$
f_j = \frac{\sum_{i} C_{i,j+1}}{\sum_{i} C_{i,j}}
$$

Ultimate loss:

$$
\hat{C}_{i,\infty} = C_{i,k} \prod_{j=k}^{\infty} f_j
$$

### Bornhuetter-Ferguson Method

Combines prior estimate with actual:

$$
\hat{C}_{i,\infty} = C_{i,k} + \text{Prior}_i \cdot (1 - \text{DevPattern}_k)
$$

### Implementation

```python
class ClaimsDevelopment:
"""Model claims development patterns."""

def __init__(self, triangle):
self.triangle = np.array(triangle)
self.n_years, self.n_dev = triangle.shape

def chain_ladder(self):
"""Apply chain ladder method."""

        # Calculate development factors
factors = []
for j in range(self.n_dev - 1):
numerator = np.nansum(self.triangle[:, j + 1])
denominator = np.nansum(self.triangle[:self.n_years - j - 1, j])
factors.append(numerator / denominator)

        # Apply factors to complete triangle
completed = self.triangle.copy()

for i in range(self.n_years):
for j in range(self.n_years - i, self.n_dev):
if np.isnan(completed[i, j]):
completed[i, j] = completed[i, j - 1] * factors[j - 1]

return completed, factors

def plot_development(self, completed):
"""Visualize development patterns."""

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Development by year
for i in range(self.n_years):
dev_pattern = completed[i, :] / completed[i, -1]
axes[0].plot(dev_pattern, marker='o', label=f'Year {i}')

axes[0].set_xlabel('Development Period')
axes[0].set_ylabel('Proportion of Ultimate')
axes[0].set_title('Development Patterns')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

        # Ultimate losses
ultimate = completed[:, -1]
axes[1].bar(range(self.n_years), ultimate)
axes[1].set_xlabel('Accident Year')
axes[1].set_ylabel('Ultimate Loss')
axes[1].set_title('Ultimate Loss Estimates')

plt.tight_layout()
return fig

# Example triangle (with NaN for future)
triangle = [
[1000, 1500, 1700, 1750],
[1100, 1650, 1870, np.nan],
[1200, 1800, np.nan, np.nan],
[1300, np.nan, np.nan, np.nan]
]

dev_model = ClaimsDevelopment(triangle)
completed, factors = dev_model.chain_ladder()

print("Development Factors:", factors)
print("\nCompleted Triangle:")
print(completed)
```

(reinsurance-structures)=
## Reinsurance Structures

### Types of Reinsurance

1. **Proportional (Pro-Rata)**
   - Quota Share: Fixed percentage
   - Surplus: Variable percentage by risk
2. **Non-Proportional (Excess of Loss)**
   - Per Risk: Each individual loss
   - Per Occurrence: Each event
   - Aggregate: Annual total

### Quota Share

Cede fixed percentage $q$:

- Retained loss: $(1-q) \cdot L$
- Ceded loss: $q \cdot L$
- Premium: $q \cdot P \cdot (1 + c)$

where $c$ is ceding commission.

### Surplus Treaty

Cede above retention line $R$:

- Retention: $\min(S, R)$
- Cession: $\max(0, S - R)$

where $S$ is sum insured.

### Aggregate Excess

Annual aggregate deductible $D$ and limit $L$:

$$
\text{Recovery} = \min(L, \max(0, S_{\text{annual}}
- D))
$$

### Optimization Example

```python
def optimize_reinsurance_program(base_losses, budget, risk_tolerance):
    """Optimize multi-layer reinsurance program."""

    from scipy.optimize import differential_evolution

    def objective(params):
        # Unpack parameters
        xs_retention = params[0]
        xs_limit = params[1]
        agg_deductible = params[2]
        agg_limit = params[3]
        quota_share = params[4]

        # Simulate net losses
        net_losses = []
        total_premium = 0

        for gross_loss in base_losses:
            # Apply quota share first
            after_qs = gross_loss * (1 - quota_share)

            # Apply per-occurrence excess
            if after_qs > xs_retention:
                xs_recovery = min(xs_limit, after_qs - xs_retention)
                after_xs = after_qs - xs_recovery
            else:
                after_xs = after_qs

            net_losses.append(after_xs)

        # Apply aggregate excess
        annual_total = sum(net_losses)
        if annual_total > agg_deductible:
            agg_recovery = min(agg_limit, annual_total - agg_deductible)
            final_net = annual_total - agg_recovery
        else:
            final_net = annual_total

        # Calculate premiums (simplified)
        xs_premium = xs_limit * 0.05 * (1 - xs_retention / 1e6)
        agg_premium = agg_limit * 0.03
        qs_premium = quota_share * np.mean(base_losses) * len(base_losses) * 1.2
        total_premium = xs_premium + agg_premium + qs_premium

        # Check constraints
        if total_premium > budget:
            return 1e10

        # Objective: minimize VaR subject to premium constraint
        return np.percentile(net_losses, 99)

    # Optimization bounds
    bounds = [
        (0, 1e6),      # xs_retention
        (0, 5e6),      # xs_limit
        (0, 10e6),     # agg_deductible
        (0, 20e6),     # agg_limit
        (0, 0.5)       # quota_share
    ]

    result = differential_evolution(objective, bounds, maxiter=100)

    optimal_params = {
        'xs_retention': result.x[0],
        'xs_limit': result.x[1],
        'agg_deductible': result.x[2],
        'agg_limit': result.x[3],
        'quota_share': result.x[4]
    }

    return optimal_params

# Example optimization
np.random.seed(42)
base_losses = stats.lognorm(s=2, scale=100_000).rvs(100)
optimal = optimize_reinsurance_program(base_losses, budget=1e6, risk_tolerance=0.01)

print("Optimal Reinsurance Program:")
for key, value in optimal.items():
    if 'retention' in key or 'limit' in key or 'deductible' in key:
        print(f"{key}: ${value:,.0f}")
    else:
        print(f"{key}: {value:.1%}")
```


(practical-applications)=
## Practical Applications

### Application 1: Manufacturing Company

![Factory Floor](../../../assets/photos/factory_floor_1_small.jpg)

```python
def manufacturing_insurance_analysis():
    """Analyze insurance needs for widget manufacturer."""

    # Company parameters
    revenue = 50_000_000  # $50M annual revenue
    assets = 30_000_000   # $30M total assets
    margin = 0.08         # 8% operating margin

    # Risk profile
    risks = {
        'property': {
            'frequency': stats.poisson(mu=2),
            'severity': stats.lognorm(s=1.5, scale=200_000),
            'max_loss': assets * 0.5
        },
        'liability': {
            'frequency': stats.poisson(mu=5),
            'severity': stats.lognorm(s=2, scale=50_000),
            'max_loss': revenue * 2
        },
        'business_interruption': {
            'frequency': stats.poisson(mu=0.5),
            'severity': stats.uniform(loc=revenue*0.1, scale=revenue*0.4),
            'max_loss': revenue
        }
    }

    # Simulate annual losses
    n_sims = 10000
    results = {}

    for risk_type, risk_params in risks.items():
        annual_losses = []

        for _ in range(n_sims):
            n_claims = risk_params['frequency'].rvs()
            if n_claims > 0:
                claims = risk_params['severity'].rvs(n_claims)
                total = min(sum(claims), risk_params['max_loss'])
            else:
                total = 0
            annual_losses.append(total)

        results[risk_type] = {
            'mean': np.mean(annual_losses),
            'p95': np.percentile(annual_losses, 95),
            'p99': np.percentile(annual_losses, 99),
            'max': np.max(annual_losses)
        }

    # Recommend limits
    recommendations = {}
    for risk_type, stats in results.items():
        # Primary layer at 95th percentile
        primary = stats['p95']

        # Excess layer to 99.5th percentile
        excess = stats['p99'] - primary

        # Catastrophic layer
        cat = stats['max'] - stats['p99']

        recommendations[risk_type] = {
            'primary': primary,
            'excess': excess,
            'catastrophic': cat,
            'total_limit': primary + excess + cat
        }

    return results, recommendations

# Run analysis
loss_stats, recommendations = manufacturing_insurance_analysis()

print("Loss Statistics by Risk Type:")
for risk_type, stats in loss_stats.items():
    print(f"\n{risk_type.upper()}:")
    for metric, value in stats.items():
        print(f"  {metric}: ${value:,.0f}")

print("\n\nRecommended Insurance Structure:")
for risk_type, limits in recommendations.items():
    print(f"\n{risk_type.upper()}:")
    print(f"  Primary (0 - ${limits['primary']:,.0f})")
    print(f"  Excess (${limits['primary']:,.0f} - ${limits['primary'] + limits['excess']:,.0f})")
    print(f"  Cat (${limits['primary'] + limits['excess']:,.0f} - ${limits['total_limit']:,.0f})")
```


### Application 2: Portfolio Insurance

![Office Space](../../../assets/photos/conference_room_1_small.jpg)

```python
def portfolio_tail_risk_hedging(portfolio_value=100_000_000):
"""Design tail risk hedging for investment portfolio."""

    # Market scenarios
scenarios = {
'normal': {'prob': 0.85, 'return': 0.08, 'vol': 0.15},
'correction': {'prob': 0.10, 'return': -0.10, 'vol': 0.25},
'crisis': {'prob': 0.04, 'return': -0.30, 'vol': 0.40},
'black_swan': {'prob': 0.01, 'return': -0.50, 'vol': 0.60}
}

    # Simulate with and without hedging
n_sims = 10000
results_unhedged = []
results_hedged = []
hedge_cost = portfolio_value * 0.01
# 1% annual cost

for _ in range(n_sims):
        # Select scenario
rand = np.random.rand()
cumsum = 0
for scenario, params in scenarios.items():
cumsum += params['prob']
if rand < cumsum:
selected = params
break

        # Generate return
annual_return = np.random.normal(selected['return'], selected['vol'])

        # Unhedged portfolio
unhedged_value = portfolio_value
* (1 + annual_return)
results_unhedged.append(unhedged_value)

        # Hedged portfolio (put option at 90% strike)
hedged_return = max(annual_return, -0.10)
# Floor at -10%
hedged_value = portfolio_value * (1 + hedged_return) - hedge_cost
results_hedged.append(hedged_value)

    # Compare strategies
comparison = pd.DataFrame({
'Metric': ['Mean', 'Std Dev', '5% VaR', '1% CVaR', 'Worst Case',
'Prob(Loss>20%)', 'Sharpe Ratio'],
'Unhedged': [
np.mean(results_unhedged),
np.std(results_unhedged),
portfolio_value - np.percentile(results_unhedged, 5),
portfolio_value - np.mean([x for x in results_unhedged
if x < np.percentile(results_unhedged, 1)]),
portfolio_value - np.min(results_unhedged),
np.mean(np.array(results_unhedged) < portfolio_value
* 0.8),
(np.mean(results_unhedged) - portfolio_value) / np.std(results_unhedged)
],
'Hedged': [
np.mean(results_hedged),
np.std(results_hedged),
portfolio_value - np.percentile(results_hedged, 5),
portfolio_value - np.mean([x for x in results_hedged
if x < np.percentile(results_hedged, 1)]),
portfolio_value - np.min(results_hedged),
np.mean(np.array(results_hedged) < portfolio_value * 0.8),
(np.mean(results_hedged) - portfolio_value) / np.std(results_hedged)
]
})

return comparison

# Analyze hedging strategies
hedging_analysis = portfolio_tail_risk_hedging()
print(hedging_analysis.to_string())
```

(key-takeaways)=
## Key Takeaways

1. **Frequency-severity framework**: Foundation of insurance modeling
2. **Heavy tails matter**: Extreme events dominate risk
3. **Layers reduce cost**: Structured coverage optimizes premium spend
4. **Retention optimization**: Balance premium savings with risk tolerance
5. **Multiple premium principles**: Different approaches for different risks
6. **Claims develop over time**: Reserve adequacy crucial
7. **Reinsurance complexity**: Multiple structures serve different purposes
8. **Practical applications**: Theory guides real-world decisions

## Next Steps

- [Chapter 4: Optimization Theory](04_optimization_theory.md) - Mathematical optimization methods
- [Chapter 5: Statistical Methods](05_statistical_methods.md) - Validation and testing
- [Chapter 1: Ergodic Economics](01_ergodic_economics.md) - Foundational concepts
