# Statistical Methods for Insurance Analysis

<div style="flex: 1; padding: 15px; border: 2px solid #2196F3; border-radius: 8px; background-color: #E3F2FD;">
    <h3 style="margin-top: 0; color: #1e82d3ff !important;">📊 Why This Matters</h3>
    <p>Statistical methods reveal that standard validation techniques fail catastrophically for insurance data. Using regular K-fold cross-validation on time series creates data leakage, with future information predicting past events, leading to overconfident models that collapse in production. Monte Carlo variance reduction techniques like antithetic variates and importance sampling can cut computational requirements by more than half while maintaining accuracy, critical when simulating rare catastrophic events with probabilities below 1%. Convergence diagnostics through Gelman-Rubin statistics and effective sample size calculations prevent the dangerous assumption that simulations have converged when they haven't; undetected non-convergence can underestimate tail risks by orders of magnitude. Bootstrap methods provide confidence intervals for complex insurance metrics where analytical solutions don't exist, with BCa adjustments correcting for the bias and skewness inherent in loss distributions. Walk-forward validation and purged K-fold cross-validation respect temporal structure, showing that models with 90% accuracy in standard validation often achieve only 60% in proper time-series validation. Backtesting reveals that strategies optimized on historical data frequently fail out-of-sample due to regime changes. Fortunately, the statistical methods here detect this overfitting before capital is at risk. The framework proves that robust statistical validation can mean the difference between sustainable profitability and bankruptcy, transforming insurance from educated gambling to scientific risk management through rigorous quantification of model uncertainty and validation of ergodic advantages.</p>
</div>

## Table of Contents
1. [Monte Carlo Methods](#monte-carlo-methods)
2. [Convergence Diagnostics](#convergence-diagnostics)
3. [Confidence Intervals](#confidence-intervals)
4. [Hypothesis Testing](#hypothesis-testing)
5. [Bootstrap Methods](#bootstrap-methods)
6. [Walk-Forward Validation](#walk-forward-validation)
7. [Backtesting](#backtesting)
8. [Model Validation](#model-validation)

![Sand Dune](photos/sand_dune.jpg)

(monte-carlo-methods)=
## Monte Carlo Methods

### Basic Monte Carlo Simulation

Monte Carlo methods estimate expectations through random sampling, providing a powerful tool for evaluating complex insurance structures where analytical solutions are intractable:

$$
E[f(X)] \approx \frac{1}{N} \sum_{i=1}^N f(X_i)
$$

where $X_i$ are independent samples from the distribution of $X$.

#### Convergence Considerations:

The standard error of the Monte Carlo estimate decreases as $O(1/\sqrt{N})$:

$$
SE(\hat{\mu}) = \frac{\sigma}{\sqrt{N}}
$$

For insurance applications with heavy-tailed distributions, larger $N$ (typically 10,000-100,000 simulations) is necessary to adequately capture tail events that drive insurance buying decisions.

### MCMC (Markov Chain Monte Carlo)

Sequential, dependent sampling creating a Markov chain, with each sample depending on the previous one. A Markov Chain is a stochastic model where the current state depends only on the immediate previous state (meaning it's "memoryless," not looking back over the full history of the path). MCMC requires a "burn-in" period to converge to the target distribution.

### When Each Is Used in Insurance Applications

**Basic Monte Carlo**:

- Simulating losses from known distributions (given parameters)
- Pricing layers when severity/frequency distributions are specified
- Evaluating retention levels with established loss distributions
- Bootstrap resampling of historical losses

**MCMC**:

- Parameter estimation with complex likelihoods (Bayesian inference)
- Fitting copulas for dependency modeling between lines
- Hierarchical models (e.g., credibility across multiple entities)
- Missing data problems in loss triangles
- Situations where the normalizing constant is unknown

### Variance Reduction Techniques

These techniques improve estimation efficiency, crucial when evaluating rare but severe events that determine insurance program effectiveness.

#### Antithetic Variates

Use negatively correlated pairs to reduce variance while maintaining unbiased estimates:

$$
\hat{\mu}_{\text{AV}} = \frac{1}{2N} \sum_{i=1}^N [f(X_i) + f(X_i')]
$$

where $X_i'$ is antithetic to $X_i$.

This technique particularly benefits:

- Excess layer pricing where both attachment and exhaustion probabilities matter
- Aggregate stop-loss evaluations where both frequency and severity variations impact results

#### Control Variates

Reduce variance using a correlated variable with known expectation:

$$
\hat{\mu}_{\text{CV}} = \hat{\mu} - c(\hat{\mu}_Y - \mu_Y)
$$

where $c$ is optimally chosen as $\text{Cov}(f(X), Y)/\text{Var}(Y)$.

This technique excels when:

- Comparing similar program structures (the control variate cancels common variation)
- Evaluating the marginal benefit of additional coverage layers
- Assessing the impact of inflation or trend uncertainty on long-term programs

#### Importance Sampling

Sample from an alternative distribution to improve efficiency for rare event estimation:

$$
E[f(X)] = E_Q\left[f(X)\frac{p(X)}{q(X)}\right]
$$

where $p(X)$ is the original density and $q(X)$ is the importance sampling density.

For example, importance sampling may use a Pareto distribution with lower threshold to simulate tail losses, where base distribution would produce very few simulated excess losses.

- Weight adjustments ensure unbiased estimates while dramatically reducing variance

(convergence-diagnostics)=
## Convergence Diagnostics

When using iterative simulation methods (MCMC, bootstrapping, or sequential Monte Carlo), convergence diagnostics ensure reliable estimates for insurance program evaluation. These tools are critical when modeling complex dependencies between lines of coverage or multi-year loss development patterns.

### Gelman-Rubin Statistic

For multiple chains, assess whether independent simulations have converged to the same distribution:

$$
\hat{R} = \sqrt{\frac{\hat{V}}{W}}
$$

where:
- $W$ = Within-chain variance (average variance within each independent simulation)
- $\hat{V}$ = Estimated total variance (combines within-chain and between-chain variance)
- $B$ = Between-chain variance

More specifically:
$$
\hat{V} = \frac{n-1}{n}W + \frac{1}{n}B
$$

#### Interpretation for Insurance Applications:

- $\hat{R} \approx 1.0$: Chains have converged (typically require $\hat{R} < 1.1$ for confidence)
- $\hat{R} > 1.1$: Insufficient convergence, requiring longer runs
- $\hat{R} \gg 1.5$: Indicates fundamental issues with model specification or starting values

### Effective Sample Size

Account for autocorrelation in sequential samples to determine the true information content:

$$
\text{ESS} = \frac{N}{1 + 2\sum_{k=1}^K \rho_k}
$$

where:
- $N$ = Total number of samples
- $\rho_k$ = Lag-$k$ autocorrelation
- $K$ = Maximum lag considered (typically where $\rho_k$ becomes negligible)

#### Practical Interpretation:

- $\text{ESS} \approx N$: Independent samples (ideal case)
- $\text{ESS} \ll N$: High autocorrelation reducing information content
- $\text{ESS}/N$ = Efficiency ratio (target > 0.1 for practical applications)

#### Decision Thresholds for Insurance Buyers:

Minimum ESS requirements depend on the decision context:

- **Strategic program design**: ESS > 1,000 for major retention decisions
- **Layer pricing comparison**: ESS > 5,000 for distinguishing between similar options
- **Tail risk metrics** (TVaR, probable maximum loss): ESS > 10,000 for 99.5th percentile estimates
- **Regulatory capital calculations**: Follow specific guidelines (e.g., Solvency II requires demonstrable convergence)

### Warning Signs Requiring Investigation:

- Persistent $\hat{R} > 1.1$ after extended runs: Model misspecification or multimodal posteriors
- ESS plateaus despite increasing $N$: Fundamental correlation structure requiring reparameterization
- Divergent chains for tail parameters: Insufficient data for extreme value estimation
- Cyclic behavior in trace plots: Indicates need for alternative sampling methods

### Implementation Example

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class ConvergenceDiagnostics:
    """Diagnose Monte Carlo convergence for insurance optimization models."""

    def __init__(self, chains):
        """
        Parameters:
        chains: list of arrays, each representing a Markov chain
                (e.g., samples of retention levels, loss parameters, or program costs)
        """
        self.chains = [np.array(chain) for chain in chains]
        self.n_chains = len(chains)
        self.n_samples = len(chains[0])

    def gelman_rubin(self):
        """
        Calculate Gelman-Rubin convergence diagnostic.
        Critical for validating parameter estimates in insurance models.
        """
        # Chain means
        chain_means = [np.mean(chain) for chain in self.chains]

        # Overall mean
        overall_mean = np.mean(chain_means)

        # Between-chain variance
        B = self.n_samples * np.var(chain_means, ddof=1)

        # Within-chain variance
        W = np.mean([np.var(chain, ddof=1) for chain in self.chains])

        # Estimated variance
        V_hat = ((self.n_samples - 1) / self.n_samples) * W + B / self.n_samples

        # R-hat statistic
        R_hat = np.sqrt(V_hat / W) if W > 0 else 1.0

        return {
            'R_hat': R_hat,
            'converged': R_hat < 1.1,
            'B': B,
            'W': W,
            'V_hat': V_hat
        }

    def effective_sample_size(self, chain=None):
        """
        Calculate effective sample size accounting for autocorrelation.
        Essential for determining if we have enough independent information
        for reliable retention decisions.
        """
        if chain is None:
            # Combine all chains
            chain = np.concatenate(self.chains)

        n = len(chain)

        # Calculate autocorrelation
        mean = np.mean(chain)
        c0 = np.sum((chain - mean)**2) / n

        # Calculate autocorrelation at each lag
        max_lag = min(int(n/4), 100)
        autocorr = []

        for k in range(1, max_lag):
            ck = np.sum((chain[:-k] - mean) * (chain[k:] - mean)) / n
            rho_k = ck / c0

            if rho_k < 0.05:
                # Stop when autocorrelation becomes negligible
                break

            autocorr.append(rho_k)

        # ESS calculation
        sum_autocorr = sum(autocorr) if autocorr else 0
        ess = n / (1 + 2 * sum_autocorr)

        return {
            'ess': ess,
            'efficiency': ess / n,
            'autocorrelation': autocorr,
            'n_lags': len(autocorr)
        }

    def geweke_diagnostic(self, chain=None, first_prop=0.1, last_prop=0.5):
        """
        Geweke convergence diagnostic comparing early and late portions.
        Useful for detecting drift in parameter estimates.
        """
        if chain is None:
            chain = self.chains[0]

        n = len(chain)
        n_first = int(n * first_prop)
        n_last = int(n * last_prop)

        first_portion = chain[:n_first]
        last_portion = chain[-n_last:]

        # Calculate means and variances
        mean_first = np.mean(first_portion)
        mean_last = np.mean(last_portion)
        var_first = np.var(first_portion, ddof=1)
        var_last = np.var(last_portion, ddof=1)

        # Geweke z-score
        se = np.sqrt(var_first/n_first + var_last/n_last)
        z_score = (mean_first - mean_last) / se if se > 0 else 0

        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return {
            'z_score': z_score,
            'p_value': p_value,
            'converged': abs(z_score) < 2,
            'mean_diff': mean_first - mean_last
        }

    def heidelberger_welch(self, chain=None, alpha=0.05):
        """
        Heidelberger-Welch stationarity and interval halfwidth test.
        Validates stability of insurance program cost estimates.
        """
        if chain is None:
            chain = self.chains[0]

        n = len(chain)

        # Stationarity test using Cramer-von Mises
        cumsum = np.cumsum(chain - np.mean(chain))
        test_stat = np.sum(cumsum**2) / (n**2 * np.var(chain))

        # Critical value approximation
        critical_value = 0.461  # For alpha=0.05
        stationary = test_stat < critical_value

        # Halfwidth test
        mean = np.mean(chain)
        se = np.std(chain, ddof=1) / np.sqrt(n)
        halfwidth = stats.t.ppf(1 - alpha/2, n-1) * se
        relative_halfwidth = halfwidth / abs(mean) if mean != 0 else np.inf

        return {
            'stationary': stationary,
            'test_statistic': test_stat,
            'mean': mean,
            'halfwidth': halfwidth,
            'relative_halfwidth': relative_halfwidth,
            'halfwidth_test_passed': relative_halfwidth < 0.1
        }

    def plot_diagnostics(self):
        """Visualize convergence diagnostics for insurance optimization."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Trace plots
        for i, chain in enumerate(self.chains[:3]):
            axes[0, 0].plot(chain, alpha=0.7, label=f'Chain {i+1}')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].set_title('Trace Plots')
        axes[0, 0].legend()

        # Running mean
        for chain in self.chains[:3]:
            running_mean = np.cumsum(chain) / np.arange(1, len(chain) + 1)
            axes[0, 1].plot(running_mean, alpha=0.7)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Running Mean')
        axes[0, 1].set_title('Running Mean Convergence')

        # Autocorrelation
        chain = self.chains[0]
        lags = range(1, min(50, len(chain)//4))
        autocorr = [np.corrcoef(chain[:-lag], chain[lag:])[0, 1] for lag in lags]
        axes[0, 2].bar(lags, autocorr)
        axes[0, 2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[0, 2].set_xlabel('Lag')
        axes[0, 2].set_ylabel('Autocorrelation')
        axes[0, 2].set_title('Autocorrelation Function')

        # Density plots
        for chain in self.chains[:3]:
            axes[1, 0].hist(chain, bins=30, alpha=0.5, density=True)
        axes[1, 0].set_xlabel('Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Marginal Distributions')

        # Gelman-Rubin evolution (FIXED INDENTATION)
        r_hats = []
        check_points = range(100, len(self.chains[0]), 100)
        for n in check_points:
            truncated_chains = [chain[:n] for chain in self.chains]
            diag = ConvergenceDiagnostics(truncated_chains)
            r_hats.append(diag.gelman_rubin()['R_hat'])

        # Plot outside the loop
        axes[1, 1].plot(check_points, r_hats)
        axes[1, 1].axhline(y=1.1, color='r', linestyle='--', label='Threshold')
        axes[1, 1].set_xlabel('Sample Size')
        axes[1, 1].set_ylabel('R-hat')
        axes[1, 1].set_title('Gelman-Rubin Statistic')
        axes[1, 1].legend()

        # Cumulative mean comparison
        for chain in self.chains[:3]:
            cumulative_mean = np.cumsum(chain) / np.arange(1, len(chain) + 1)
            cumulative_std = [np.std(chain[:i]) for i in range(1, len(chain) + 1)]
            axes[1, 2].fill_between(range(len(chain)),
                                    cumulative_mean - cumulative_std,
                                    cumulative_mean + cumulative_std,
                                    alpha=0.3)
            axes[1, 2].plot(cumulative_mean, linewidth=2)

        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('Cumulative Mean ± Std')
        axes[1, 2].set_title('Convergence Bands')

        plt.tight_layout()
        plt.show()


# ============================================================================
# INSURANCE-SPECIFIC EXAMPLE: Optimal Retention Level Estimation
# ============================================================================

class InsuranceRetentionMCMC:
    """
    MCMC for estimating optimal retention levels considering:
    - Historical loss data uncertainty
    - Premium savings vs. volatility trade-off
    - Multiple lines of coverage correlation
    """

    def __init__(self, historical_losses, premium_schedule):
        """
        Parameters:
        historical_losses: dict with keys as coverage lines, values as loss arrays
        premium_schedule: function mapping retention to premium savings
        """
        self.losses = historical_losses
        self.premium_schedule = premium_schedule

    def log_posterior(self, params):
        """
        Calculate log posterior for retention optimization.
        params = [retention, risk_aversion, correlation_factor]
        """
        retention, risk_aversion, corr_factor = params

        # Prior constraints (e.g., retention between 100k and 5M)
        if retention < 100000 or retention > 5000000:
            return -np.inf
        if risk_aversion < 0.5 or risk_aversion > 3:
            return -np.inf
        if corr_factor < 0 or corr_factor > 1:
            return -np.inf

        # Calculate expected retained losses
        retained_losses = []
        for line, losses in self.losses.items():
            retained = np.minimum(losses, retention)
            retained_losses.extend(retained)

        # Total cost = retained losses - premium savings + risk charge
        expected_retained = np.mean(retained_losses)
        volatility = np.std(retained_losses)
        premium_savings = self.premium_schedule(retention)

        # Risk-adjusted total cost
        total_cost = expected_retained - premium_savings + risk_aversion * volatility

        # Log posterior (negative cost since we want to minimize)
        return -total_cost / 100000  # Scale for numerical stability

    def run_chain(self, seed, n_samples=5000, initial_params=None):
        """Run a single MCMC chain for retention optimization."""
        np.random.seed(seed)

        if initial_params is None:
            # Random initialization
            initial_params = [
                np.random.uniform(250000, 2000000),  # retention
                np.random.uniform(1, 2),              # risk_aversion
                np.random.uniform(0.2, 0.8)           # correlation
            ]

        chain = []
        current_params = initial_params
        current_log_p = self.log_posterior(current_params)

        # Metropolis-Hastings sampling
        for i in range(n_samples):
            # Propose new parameters
            proposal = current_params + np.random.randn(3) * [50000, 0.1, 0.05]
            proposal_log_p = self.log_posterior(proposal)

            # Accept/reject
            log_ratio = proposal_log_p - current_log_p
            if np.log(np.random.rand()) < log_ratio:
                current_params = proposal
                current_log_p = proposal_log_p

            # Store only retention level for diagnostics
            chain.append(current_params[0])

        return chain


# ============================================================================
# EXAMPLE USAGE: Corporate Insurance Program Optimization
# ============================================================================

# Simulate historical loss data for a manufacturer
np.random.seed(42)
historical_losses = {
    'property': np.random.pareto(2, 100) * 100000,  # Heavy-tailed property losses
    'liability': np.random.lognormal(11, 1.5, 80),   # GL/Products losses
    'auto': np.random.gamma(2, 50000, 120),          # Fleet losses
    'workers_comp': np.random.lognormal(9, 2, 150)   # WC claims
}

# Premium schedule (savings increase with retention)
def premium_schedule(retention):
    """Premium savings as function of retention level."""
    base_premium = 2000000
    discount = min(0.4, (retention / 1000000) * 0.15)
    return base_premium * discount

# Initialize MCMC sampler
sampler = InsuranceRetentionMCMC(historical_losses, premium_schedule)

# Run multiple chains with different starting points
print("Running MCMC for Optimal Retention Analysis...")
print("=" * 50)

chains = []
for i in range(4):
    print(f"Running chain {i+1}/4...")
    chain = sampler.run_chain(seed=i, n_samples=5000)
    chains.append(chain)

# Diagnose convergence
print("\nConvergence Diagnostics for Retention Optimization:")
print("-" * 50)

diagnostics = ConvergenceDiagnostics(chains)

# Gelman-Rubin
gr = diagnostics.gelman_rubin()
print(f"Gelman-Rubin R-hat: {gr['R_hat']:.3f}")
print(f"  → Converged: {gr['converged']}")
print(f"  → Interpretation: {'Chains agree on optimal retention' if gr['converged'] else 'Need longer runs'}")

# Effective Sample Size
ess = diagnostics.effective_sample_size()
print(f"\nEffective Sample Size: {ess['ess']:.0f} ({ess['efficiency']:.1%} efficiency)")
print(f"  → Interpretation: {'Sufficient for decision' if ess['ess'] > 1000 else 'Need more samples'}")

# Geweke
geweke = diagnostics.geweke_diagnostic()
print(f"\nGeweke Diagnostic:")
print(f"  → z-score: {geweke['z_score']:.3f} (p-value: {geweke['p_value']:.3f})")
print(f"  → Interpretation: {'Stable estimates' if geweke['converged'] else 'Potential drift detected'}")

# Heidelberger-Welch
hw = diagnostics.heidelberger_welch()
print(f"\nHeidelberger-Welch Test:")
print(f"  → Stationary: {hw['stationary']}")
print(f"  → Halfwidth test: {hw['halfwidth_test_passed']}")
print(f"  → Mean retention estimate: ${hw['mean']:,.0f}")
print(f"  → Confidence interval halfwidth: ${hw['halfwidth']:,.0f}")

# Final recommendation
combined_chain = np.concatenate(chains)
optimal_retention = np.mean(combined_chain)
retention_std = np.std(combined_chain)
percentiles = np.percentile(combined_chain, [25, 50, 75])

print(f"\n" + "=" * 50)
print("RETENTION RECOMMENDATION:")
print(f"  → Optimal Retention: ${optimal_retention:,.0f}")
print(f"  → Standard Deviation: ${retention_std:,.0f}")
print(f"  → 50% Confidence Range: ${percentiles[0]:,.0f} - ${percentiles[2]:,.0f}")
print(f"  → Decision Quality: {'High confidence' if gr['converged'] and ess['ess'] > 1000 else 'Consider additional analysis'}")

# Visualize diagnostics
print("\nGenerating diagnostic plots...")
diagnostics.plot_diagnostics()
```

#### Sample Output

![Convergence Diagnostics Example](figures/convergence_diagnostics_example.png)

```
Convergence Diagnostics for Retention Optimization:
--------------------------------------------------
Gelman-Rubin R-hat: 1.134
  → Converged: False
  → Interpretation: Need longer runs

Effective Sample Size: 107 (0.5% efficiency)
  → Interpretation: Need more samples

Geweke Diagnostic:
  → z-score: -87.302 (p-value: 0.000)
  → Interpretation: Potential drift detected

Heidelberger-Welch Test:
  → Stationary: False
  → Halfwidth test: True
  → Mean retention estimate: $2,634,781
  → Confidence interval halfwidth: $18,035

==================================================
RETENTION RECOMMENDATION:
  → Optimal Retention: $3,362,652
  → Standard Deviation: $1,126,258
  → 50% Confidence Range: $2,701,352 - $4,404,789
  → Decision Quality: Consider additional analysis
```

(confidence-intervals)=
## Confidence Intervals

Confidence intervals quantify parameter uncertainty in insurance estimates, crucial for setting retention levels, evaluating program adequacy, and regulatory compliance. The choice of method depends on sample size, distribution characteristics, and the specific insurance application.

### Classical Confidence Intervals

For large samples with approximately normal sampling distributions, use Central Limit Theorem:

$$
\bar{X} \pm z_{\alpha/2} \frac{s}{\sqrt{n}}
$$

**Limitations in Insurance Context:**
- Assumes normality of sampling distribution (often violated for insurance losses)
- Underestimates uncertainty for heavy-tailed distributions
- Poor coverage for extreme percentiles (important for high limit attachment points)

**Appropriate Uses:**
- Frequency estimates with sufficient data (n > 30)
- Average severity for high-frequency lines after log transformation
- Loss ratio estimates for stable, mature books

### Student's *t* Intervals

For smaller samples or unknown population variance:

$$
\bar{X} \pm t_{n-1,\alpha/2} \frac{s}{\sqrt{n}}
$$

**Insurance Applications:**
- New coverage lines with limited history (n < 30)
- Credibility-weighted estimates for small accounts
- Emerging risk categories with sparse data

### Bootstrap Confidence Intervals

Bootstrap methods handle complex statistics and non-normal distributions common in insurance, providing distribution-free (nonparametric) inference.

#### Percentile Method

Direct empirical quantiles from bootstrap distribution:

$$
[\hat{\theta}^*_{\alpha/2}, \hat{\theta}^*_{1-\alpha/2}]
$$

where $\hat{\theta}^*_p$ is the $p$-th percentile of $B$ bootstrap estimates.

**Implementation for Insurance:**

1. Resample claims data with replacement $B$ times (at least 1,000 resamples for confidence intervals, typically 5,000-10,000)
2. Calculate statistic of interest for each resample (e.g., 99.5% VaR)
3. Use empirical percentiles as confidence bounds

**Best for:**

- Median loss estimates
- Interquartile ranges for pricing bands
- Simple reserve estimates

#### BCa (Bias-Corrected and Accelerated)

Adjusts for bias and skewness inherent in insurance loss distributions:

$$
[\hat{\theta}^*_{\alpha_1}, \hat{\theta}^*_{\alpha_2}]
$$

where adjusted percentiles are:

$$
\alpha_1 = \Phi\left(\hat{z}_0 + \frac{\hat{z}_0 + z_{\alpha/2}}{1 - \hat{a}(\hat{z}_0 + z_{\alpha/2})}\right)
$$

$$
\alpha_2 = \Phi\left(\hat{z}_0 + \frac{\hat{z}_0 + z_{1-\alpha/2}}{1 - \hat{a}(\hat{z}_0 + z_{1-\alpha/2})}\right)
$$

with:
- $\hat{z}_0$ = bias correction factor (proportion of bootstrap estimates < original estimate)
- $\hat{a}$ = acceleration constant (measures rate of change in standard error)

**Critical for:**
- Tail risk metrics (TVaR, probable maximum loss)
- Excess layer loss costs with limited data
- Catastrophe model uncertainty quantification

#### Bootstrap-*t* Intervals

Studentized bootstrap for improved coverage:

$$
[\hat{\theta} - t^*_{1-\alpha/2} \cdot SE^*, \hat{\theta} - t^*_{\alpha/2} \cdot SE^*]
$$

where $t^*$ values come from bootstrap distribution of $(\hat{\theta}^* - \hat{\theta})/SE^*$.

**Superior for:**
- Highly skewed distributions (cyber, catastrophe losses)
- Ratio estimates (loss ratios, combined ratios)
- Correlation estimates between coverage lines

### Credibility-Weighted Confidence Intervals

Combines company-specific and industry experience where company experience has low credibility:

$$
\hat{\theta}_{\text{cred}} = Z \cdot \hat{\theta}_{\text{company}} + (1-Z) \cdot \hat{\theta}_{\text{industry}}
$$

with variance:

$$
\text{Var}(\hat{\theta}_{\text{cred}}) = Z^2 \cdot \text{Var}(\hat{\theta}_{\text{company}}) + (1-Z)^2 \cdot \text{Var}(\hat{\theta}_{\text{industry}})
$$

where credibility factor $Z = n/(n + k)$ with $k$ representing prior variance.

### Bayesian Credible Intervals

Bayesian credible intervals provide direct probability statements about where a parameter lies, incorporating both observed data and prior knowledge. This approach is particularly valuable when dealing with limited loss data or incorporating industry benchmarks.

#### Core Concept

A 95% credible interval means there's a 95% probability the parameter falls within the interval:

$$
P(\theta \in [a, b] | \text{data}) = 0.95
$$

**Key Distinction from Confidence Intervals:**
- **Confidence Interval**: "If we repeated this procedure many times, 95% of intervals would contain the true value"
- **Credible Interval**: "Given our data and prior knowledge, there's a 95% probability the parameter is in this range"

The credible interval directly answers: "What's the probability our retention is adequate?"

#### How Credible Intervals Work

Starting with:

1. **Prior distribution**: Industry benchmark or regulatory assumption
2. **Likelihood**: How well different parameter values explain your data
3. **Posterior distribution**: Updated belief combining prior and data

The credible interval captures the central 95% (or other level) of the posterior distribution.

#### Two Types of Credible Intervals

##### Equal-Tailed Intervals
Uses the 2.5th and 97.5th percentiles of posterior distribution (or for some other level, take $\alpha / 2$ from each side):

$$
[\theta_{0.025}, \theta_{0.975}]
$$

**When to use**: Symmetric distributions, standard reporting

##### Highest Posterior Density (HPD) Intervals
The shortest interval containing 95% probability:
- Always shorter than equal-tailed for skewed distributions
- All points inside have higher probability than points outside

**When to use**: Skewed distributions (severity modeling), optimization decisions

**Example - Cyber Loss Severity:**

- Equal-tailed 95%: [\$50K, \$2.8M]  (width: \$2.75M)
- HPD 95%: [\$25K, \$2.3M]  (width: \$2.275M)
  - HPD provides tighter bounds for decision-making

**Insurance-Specific Advantages**:

1. Natural Incorporation of Industry Priors
2. Handles Parameter Uncertainty in Hierarchical Models
3. Updates Systematically with Emerging Experience

#### When to Use Bayesian Credible Intervals

**Ideal situations:**

- Limited company data (< 3 years history)
- New coverage lines or emerging risks
- Incorporating industry benchmarks or expert judgment
- Multi-level modeling (location, division, company)
- When stakeholders want probability statements for decisions

**May not be suitable when:**

- Extensive historical data makes priors irrelevant
- Regulatory requirements mandate frequentist methods
- Computational resources are limited
- Prior information is controversial or unavailable

### Profile Likelihood Intervals

Profile likelihood intervals provide more accurate confidence intervals for maximum likelihood estimates, especially when standard methods fail due to skewness, small samples, or boundary constraints.

#### The Problem with Standard (Wald) Intervals

Maximum likelihood estimates typically use Wald intervals based on asymptotic normality:

$$
\hat{\theta} \pm z_{\alpha/2} \cdot SE(\hat{\theta})
$$

These assume the likelihood is approximately quadratic (normal) near the MLE, which often fails for:
- Small samples (n < 50)
- Parameters near boundaries (probabilities near 0 or 1)
- Skewed likelihood surfaces (common in tail estimation)

#### Profile Likelihood Solution

Profile likelihood intervals use the actual shape of the likelihood function rather than assuming normality:

$$
\{θ : 2[\ell(\hat{θ}) - \ell(θ)] ≤ χ^2_{1,1-α}\}
$$

where:
- $\ell(\hat{θ})$ = log-likelihood at maximum
- $\ell(θ)$ = log-likelihood at any value θ
- $χ^2_{1,1-α}$ = critical value from chi-squared distribution

#### Why Profile Likelihood Works Better

1. **Respects parameter constraints**: Won't give negative variances or probabilities > 1
2. **Captures asymmetry**: Naturally wider on the side with more uncertainty
3. **Better small-sample coverage**: Achieves closer to nominal 95% coverage
4. **Invariant to parameterization**: Same interval whether using $\theta$ or $\log(\theta)$

**Essential for:**
- Tail index estimation in extreme value models
- Copula parameter estimation for dependency modeling
- Non-linear trend parameters in reserve development

(hypothesis-testing)=
## Hypothesis Testing

### Framework

Hypothesis testing provides a formal framework for making data-driven decisions about insurance program changes, comparing options, and validating actuarial assumptions. The process quantifies evidence against a baseline assumption to support strategic decisions.

#### Core Components

Test null hypothesis $H_0$ against alternative $H_1$:

1. **Test statistic**: $T = T(X_1, ..., X_n)$ - Summary measure of evidence
2. **P-value**: $P(T \geq T_{\text{obs}} | H_0)$ - Probability of seeing data this extreme if $H_0$ true
3. **Decision**: Reject $H_0$ if p-value < $\alpha$ (significance level, typically 0.05)

#### Interpretation for Insurance Decisions

**P-value meaning**: "If our current program is adequate (null hypothesis), what's the probability of seeing losses this bad or worse?"
   - P-value = 0.03: Only 3% chance of seeing these results if program adequate, consider changing programs
   - P-value = 0.15: 15% chance, insufficient evidence to change program

**Common significance levels (α) in insurance:**
   - 0.10: Preliminary analysis, early warning systems
   - 0.05: Standard business decisions, pricing changes
   - 0.01: Major strategic changes, regulatory compliance
   - 0.001: Safety-critical decisions, catastrophe modeling

### Types of Hypothesis Tests for Insurance

- **One-Sample Tests**: Testing whether observed experience differs from expected
- **Two-Sample Tests**: Comparing programs, periods, or segments
- **Goodness-of-Fit Tests**: Validating distributional assumptions
  - **Kolmogorov-Smirnov Test**: Maximum distance between empirical and theoretical CDF
  - **Anderson-Darling Test**: More sensitive to tail differences than K-S

### Bayesian Hypothesis Testing

Alternative framework using posterior probabilities:

#### Bayes Factors

Ratio of evidence for competing hypotheses:

$$
BF_{10} = \frac{P(\text{data}|H_1)}{P(\text{data}|H_0)}
$$

#### Interpretation scale:

- BF < 1: Evidence favors $H_0$
- BF = 1-3: Weak evidence for $H_1​$
- BF = 3-10: Moderate evidence for $H_1​$
- BF > 10: Strong evidence for $H_1​$

### Hypothesis Testing Example

```python
import numpy as np
from scipy import stats
import pandas as pd

def retention_analysis(losses, current_retention, proposed_retention,
                      premium_difference, alpha=0.05):
    """
    Complete hypothesis testing framework for retention decision.
    """

    # Calculate retained losses under each retention
    current_retained = np.minimum(losses, current_retention)
    proposed_retained = np.minimum(losses, proposed_retention)

    # Test 1: Paired t-test for mean difference
    t_stat, p_value_mean = stats.ttest_rel(current_retained, proposed_retained)

    # Test 2: F-test for variance difference
    f_stat = np.var(current_retained) / np.var(proposed_retained)
    p_value_var = stats.f.sf(f_stat, len(losses)-1, len(losses)-1) * 2

    # Test 3: Bootstrap test for tail risk (95th percentile)
    n_bootstrap = 10000
    percentile_diffs = []

    for _ in range(n_bootstrap):
        sample_idx = np.random.choice(len(losses), len(losses), replace=True)
        sample_losses = losses[sample_idx]

        current_p95 = np.percentile(np.minimum(sample_losses, current_retention), 95)
        proposed_p95 = np.percentile(np.minimum(sample_losses, proposed_retention), 95)
        percentile_diffs.append(proposed_p95 - current_p95)

    p_value_tail = np.mean(np.array(percentile_diffs) > premium_difference)

    # Economic significance test
    expected_savings = np.mean(current_retained - proposed_retained)
    roi = (expected_savings - premium_difference) / premium_difference

    # Decision matrix
    decisions = {
        'mean_test': {
            'statistic': t_stat,
            'p_value': p_value_mean,
            'significant': p_value_mean < alpha,
            'interpretation': 'Average retained losses differ' if p_value_mean < alpha
                           else 'No significant difference in average'
        },
        'variance_test': {
            'statistic': f_stat,
            'p_value': p_value_var,
            'significant': p_value_var < alpha,
            'interpretation': 'Volatility changes significantly' if p_value_var < alpha
                           else 'Similar volatility'
        },
        'tail_test': {
            'p_value': p_value_tail,
            'significant': p_value_tail < alpha,
            'interpretation': 'Tail risk justifies premium' if p_value_tail < alpha
                           else 'Premium exceeds tail risk benefit'
        },
        'economic_test': {
            'expected_savings': expected_savings,
            'roi': roi,
            'recommendation': 'Change retention' if roi > 0.1 and p_value_mean < alpha
                           else 'Keep current retention'
        }
    }

    return decisions

# Example usage
losses = np.random.pareto(2, 1000) * 100000  # Historical losses
results = retention_analysis(losses, 500000, 1000000, 75000)

print("Retention Analysis Results:")
for test, result in results.items():
    print(f"\n{test}:")
    for key, value in result.items():
        print(f"  {key}: {value}")
```

#### Sample Output

```
Retention Analysis Results:

mean_test:
  statistic: -4.836500411778284
  p_value: 1.5295080832844058e-06
  significant: True
  interpretation: Average retained losses differ

variance_test:
  statistic: 0.5405383468155659
  p_value: 1.9999999999999998
  significant: False
  interpretation: Similar volatility

tail_test:
  p_value: 0.0
  significant: True
  interpretation: Tail risk justifies premium

economic_test:
  expected_savings: -9644.033668628837
  roi: -1.1285871155817178
  recommendation: Keep current retention
```

(bootstrap-methods)=
## Bootstrap Methods

Bootstrap methods provide powerful, flexible tools for quantifying uncertainty without restrictive distributional assumptions. For insurance applications with limited data, heavy tails, complex dependencies, or complex policy mechanics, bootstrap techniques often outperform traditional methods.

### Core Concept

Bootstrap resampling mimics the sampling process by treating the observed data as the population, enabling inference about parameter uncertainty, confidence intervals, and hypothesis testing without assuming normality or other specific distributions.

### Standard (Nonparametric) Bootstrap Algorithm

The classical bootstrap resamples from observed data with replacement:

1. Draw $B$ samples of size $n$ *with replacement* from original data
2. Calculate statistic $\hat{\theta}^*_b$ for each bootstrap sample $b = 1, ..., B$
3. Use empirical distribution of $\{\hat{\theta}^*_1, ..., \hat{\theta}^*_B\}$ for inference

#### Insurance Implementation:

```python
import numpy as np

def bootstrap_loss_metrics(losses, n_bootstrap=10000, seed=42):
    """
    Bootstrap key metrics for insurance loss analysis.

    Parameters:
    losses: array of historical loss amounts
    n_bootstrap: number of bootstrap samples
    """
    np.random.seed(seed)
    n = len(losses)

    # Storage for bootstrap statistics
    means = []
    medians = []
    percentile_95s = []
    tvar_95s = []  # Tail Value at Risk

    for b in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(losses, size=n, replace=True)

        # Calculate statistics
        means.append(np.mean(sample))
        medians.append(np.median(sample))
        percentile_95s.append(np.percentile(sample, 95))

        # TVaR (average of losses above 95th percentile)
        threshold = np.percentile(sample, 95)
        tail_losses = sample[sample > threshold]
        tvar_95s.append(np.mean(tail_losses) if len(tail_losses) > 0 else threshold)

    # Return bootstrap distributions
    return {
        'mean': means,
        'median': medians,
        'var_95': percentile_95s,
        'tvar_95': tvar_95s
    }

# Example: Property losses for manufacturer
property_losses = np.array([15000, 28000, 45000, 52000, 78000, 95000,
                           120000, 185000, 340000, 580000, 1250000])

bootstrap_results = bootstrap_loss_metrics(property_losses)

# Confidence intervals
print("95% Bootstrap Confidence Intervals:")
for metric, values in bootstrap_results.items():
    ci_lower = np.percentile(values, 2.5)
    ci_upper = np.percentile(values, 97.5)
    print(f"{metric}: [{ci_lower:,.0f}, {ci_upper:,.0f}]")
```

##### Sample Output:

```
95% Bootstrap Confidence Intervals:
mean: [86,539, 486,189]
median: [45,000, 340,000]
var_95: [185,000, 1,250,000]
tvar_95: [185,000, 1,250,000]
```

#### Advantages for Insurance:

- No distributional assumptions (critical for heavy-tailed losses)
- Captures actual data characteristics including skewness
- Works for any statistic (ratios, percentiles, complex metrics)

#### Limitations:

- Underestimates tail uncertainty if largest loss not representative
- Requires sufficient data to represent distribution (typically n > 30)
- Cannot extrapolate beyond observed range

### Parametric Bootstrap

Resamples from fitted parametric distribution rather than empirical distribution:

1. Fit parametric model to data: $\hat{F}(x; \hat{\theta})$
2. Generate $B$ samples of size $n$ from $\hat{F}$
3. Calculate statistic for each parametric bootstrap sample
4. Use distribution for inference

#### When to Use:

- Strong theoretical basis for distribution (e.g., Poisson frequency)
- Small samples where nonparametric bootstrap unreliable
- Extrapolation beyond observed data needed

#### Advantages:

- Can extrapolate beyond observed data
- Smoother estimates for tail probabilities
- More efficient if model correct

#### Risks:

- Model misspecification bias
- Overconfidence if model wrong
- Should validate with goodness-of-fit tests

### Block Bootstrap

Preserves temporal or spatial dependencies by resampling blocks of consecutive observations:

1. Divide data into overlapping or non-overlapping blocks of length $l$
2. Resample blocks with replacement
3. Concatenate blocks to form bootstrap sample
4. Calculate statistics from blocked samples

#### Block Length Selection:

- Too short: Doesn't capture dependencies
- Too long: Reduces effective sample size
- Rule of thumb: $l \approx n^{1/3}$ for time series
- For insurance: Often use 3-6 months for monthly data

##### Applications:

- Loss development patterns with correlation
- Natural catastrophe clustering
- Economic cycle effects on liability claims
- Seasonal patterns in frequency

### Wild Bootstrap

Preserves heteroskedasticity (varying variance) by resampling residuals with random weights:

- Fit initial model: $y_i = f(x_i; \hat{\theta}) + \hat{\epsilon}_i$
- Generate bootstrap samples: $y_i^* = f(x_i; \hat{\theta}) + w_i \cdot \hat{\epsilon}_i$
- Weights $w_i$​ drawn from distribution with $E[w] = 0$, $Var[w] = 1$
- Refit model to bootstrap data

#### Advantages:

- Preserves heteroskedastic structure
- Valid for regression with non-constant variance
- Better coverage for predictions across size ranges

#### Weight Distributions:

- Rademacher: $P(w = 1) = P(w = -1) = 0.5$
- Mammen: $P(w = -(\sqrt{5}-1)/2) = (\sqrt{5}+1)/(2\sqrt{5})$
- Normal: $w \sim N(0, 1)$

### Smoothed Bootstrap

For small samples or discrete distributions, adds small random noise (jitter) to avoid discreteness.

### Bootstrap Selection Guidelines

1. Data Structure:
   - Independent observations → Standard bootstrap
   - Time series/dependencies → Block bootstrap
   - Regression/varying variance → Wild bootstrap
   - Small sample/discrete → Smoothed bootstrap
   - Known distribution family → Parametric bootstrap

3. Objective:
   - General inference → Standard bootstrap
   - Tail risk/extremes → Parametric (GPD/Pareto)
   - Trend analysis → Block bootstrap
   - Predictive intervals → Wild or parametric

4. Sample Size:
   - n < 20: Parametric or smoothed
   - 20 < n < 100: Any method, validate with alternatives
   - n > 100: Standard usually sufficient

### Computational Considerations

Number of Bootstrap Samples ($B$):

- Confidence intervals: B = 5,000-10,000
- Standard errors: B = 1,000
- Hypothesis tests: B = 10,000-20,000
- Extreme percentiles: B = 20,000-50,000

(walk-forward-validation)=
## Walk-Forward Validation

### Methodology

1. Train on historical window
2. Test on next period
3. Roll window forward
4. Aggregate results

### Implementation

```python
class WalkForwardValidator:
"""Walk-forward validation for insurance strategies."""

def __init__(self, data, window_size, test_size):
self.data = data
self.window_size = window_size
self.test_size = test_size
self.n_periods = len(data)

def validate_strategy(self, strategy_func, params_optimizer):
"""Validate insurance strategy using walk-forward analysis."""

results = []

        # Calculate number of folds
n_folds = (self.n_periods - self.window_size) // self.test_size

for fold in range(n_folds):
            # Define training window
train_start = fold
* self.test_size
train_end = train_start + self.window_size

            # Define test window
test_start = train_end
test_end = test_start + self.test_size

if test_end > self.n_periods:
break

            # Training data
train_data = self.data[train_start:train_end]

            # Optimize parameters on training data
optimal_params = params_optimizer(train_data)

            # Test data
test_data = self.data[test_start:test_end]

            # Apply strategy with optimal parameters
test_performance = strategy_func(test_data, optimal_params)

results.append({
'fold': fold,
'train_period': (train_start, train_end),
'test_period': (test_start, test_end),
'optimal_params': optimal_params,
'test_performance': test_performance
})

return results

def analyze_results(self, results):
"""Analyze walk-forward validation results."""

        # Extract performance metrics
performances = [r['test_performance'] for r in results]

        # Calculate statistics
analysis = {
'mean_performance': np.mean(performances),
'std_performance': np.std(performances),
'min_performance': np.min(performances),
'max_performance': np.max(performances),
'sharpe_ratio': np.mean(performances) / np.std(performances),
'win_rate': np.mean(np.array(performances) > 0),
'n_folds': len(results)
}

        # Parameter stability
all_params = [r['optimal_params'] for r in results]
if isinstance(all_params[0], dict):
param_stability = {}
for key in all_params[0].keys():
values = [p[key] for p in all_params]
param_stability[key] = {
'mean': np.mean(values),
'std': np.std(values),
'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf
}
analysis['parameter_stability'] = param_stability

return analysis

def plot_validation(self, results):
"""Visualize walk-forward validation results."""

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Performance over time
performances = [r['test_performance'] for r in results]
test_periods = [(r['test_period'][0] + r['test_period'][1])/2 for r in results]

axes[0].plot(test_periods, performances, 'o-')
axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[0].axhline(y=np.mean(performances), color='g', linestyle='--',
label=f'Mean: {np.mean(performances):.3f}')
axes[0].fill_between(test_periods,
np.mean(performances) - np.std(performances),
np.mean(performances) + np.std(performances),
alpha=0.3, color='green')
axes[0].set_xlabel('Time Period')
axes[0].set_ylabel('Performance')
axes[0].set_title('Out-of-Sample Performance')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

        # Cumulative performance
cumulative = np.cumsum(performances)
axes[1].plot(range(len(cumulative)), cumulative, 'b-', linewidth=2)
axes[1].fill_between(range(len(cumulative)), 0, cumulative, alpha=0.3)
axes[1].set_xlabel('Fold Number')
axes[1].set_ylabel('Cumulative Performance')
axes[1].set_title('Cumulative Out-of-Sample Performance')
axes[1].grid(True, alpha=0.3)

        # Parameter evolution
if 'optimal_params' in results[0] and isinstance(results[0]['optimal_params'], dict):
param_names = list(results[0]['optimal_params'].keys())

for param_name in param_names[:3]:
# Plot first 3 parameters
param_values = [r['optimal_params'][param_name] for r in results]
axes[2].plot(range(len(param_values)), param_values,
'o-', label=param_name, alpha=0.7)

axes[2].set_xlabel('Fold Number')
axes[2].set_ylabel('Parameter Value')
axes[2].set_title('Parameter Evolution')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Example: Walk-forward validation of insurance strategy
def insurance_strategy(data, params):
"""Simple insurance strategy."""
retention = params['retention']
limit = params['limit']

    # Calculate returns with insurance
returns = []
for value in data:
loss = max(0, -value) if value < 0 else 0
retained_loss = min(loss, retention)
premium = 0.02 * limit

net_return = value - retained_loss - premium
returns.append(net_return)

return np.mean(returns)

def optimize_params(data):
"""Optimize insurance parameters."""
best_performance = -np.inf
best_params = None

    # Grid search (simplified)
for retention in [0.01, 0.05, 0.10]:
for limit in [0.20, 0.30, 0.40]:
params = {'retention': retention, 'limit': limit}
performance = insurance_strategy(data, params)

if performance > best_performance:
best_performance = performance
best_params = params

return best_params

# Generate sample data
np.random.seed(42)
returns_data = np.random.normal(0.08, 0.15, 1000) + \
np.random.standard_t(5, 1000)
* 0.05
# Add fat tails

# Walk-forward validation
validator = WalkForwardValidator(returns_data, window_size=200, test_size=50)
results = validator.validate_strategy(insurance_strategy, optimize_params)

# Analyze
analysis = validator.analyze_results(results)
print("\nWalk-Forward Validation Results:")
print("-" * 40)
for key, value in analysis.items():
if key != 'parameter_stability':
print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

# Visualize
validator.plot_validation(results)
```

#### Sample Output

![Walk-Forward Validation](../../../theory/figures/walk_forward_validation.png)

```
Walk-Forward Validation Results:
----------------------------------------
mean_performance: 0.0757
std_performance: 0.0214
min_performance: 0.0354
max_performance: 0.1136
sharpe_ratio: 3.5398
win_rate: 1.0000
n_folds: 16
```

(backtesting)=
## Backtesting

### Backtesting Framework

```python
class InsuranceBacktester:
"""Comprehensive backtesting for insurance strategies."""

def __init__(self, historical_data):
self.data = historical_data
self.results = {}

def backtest(self, strategy, initial_capital=1000000,
start_date=None, end_date=None):
"""Run backtest of insurance strategy."""

        # Filter data by date if specified
test_data = self.data
if start_date:
test_data = test_data[test_data.index >= start_date]
if end_date:
test_data = test_data[test_data.index <= end_date]

        # Initialize tracking variables
capital = initial_capital
positions = []
metrics = {
'dates': [],
'capital': [],
'returns': [],
'premiums': [],
'claims': [],
'retained_losses': []
}

        # Run strategy
for date, row in test_data.iterrows():
            # Get strategy decision
decision = strategy(date, row, capital, positions)

            # Calculate losses and premiums
loss = row.get('loss', 0)
premium = decision['premium']
retention = decision['retention']
limit = decision['limit']

            # Calculate retained loss
retained_loss = min(loss, retention)
insured_loss = min(max(0, loss - retention), limit)

            # Update capital
capital = capital
* (1 + row.get('return', 0)) - premium - retained_loss

            # Track metrics
metrics['dates'].append(date)
metrics['capital'].append(capital)
metrics['returns'].append(row.get('return', 0))
metrics['premiums'].append(premium)
metrics['claims'].append(loss)
metrics['retained_losses'].append(retained_loss)

            # Check for bankruptcy
if capital <= 0:
print(f"Bankruptcy on {date}")
break

        # Calculate performance metrics
self.results = self.calculate_metrics(metrics, initial_capital)
return self.results

def calculate_metrics(self, metrics, initial_capital):
"""Calculate comprehensive performance metrics."""

capital_series = np.array(metrics['capital'])
returns = np.diff(capital_series) / capital_series[:-1]

        # Basic metrics
total_return = (capital_series[-1] / initial_capital - 1) * 100
annual_return = (capital_series[-1] / initial_capital) ** (252/len(capital_series)) - 1

        # Risk metrics
volatility = np.std(returns)
* np.sqrt(252)
downside_returns = returns[returns < 0]
downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0

        # Drawdown analysis
cummax = np.maximum.accumulate(capital_series)
drawdown = (cummax - capital_series) / cummax
max_drawdown = np.max(drawdown)

        # Risk-adjusted returns
sharpe = annual_return / volatility if volatility > 0 else 0
sortino = annual_return / downside_vol if downside_vol > 0 else 0

        # Insurance-specific metrics
total_premiums = np.sum(metrics['premiums'])
total_claims = np.sum(metrics['claims'])
total_retained = np.sum(metrics['retained_losses'])
loss_ratio = total_retained / total_premiums if total_premiums > 0 else 0

return {
'total_return_%': total_return,
'annual_return_%': annual_return
* 100,
'volatility_%': volatility * 100,
'sharpe_ratio': sharpe,
'sortino_ratio': sortino,
'max_drawdown_%': max_drawdown
* 100,
'total_premiums': total_premiums,
'total_claims': total_claims,
'total_retained': total_retained,
'loss_ratio': loss_ratio,
'final_capital': capital_series[-1],
'survival': capital_series[-1] > 0,
'metrics_history': metrics
}

def plot_backtest(self):
"""Visualize backtest results."""

if not self.results:
print("No results to plot. Run backtest first.")
return

metrics = self.results['metrics_history']

fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        # Capital evolution
axes[0, 0].plot(metrics['dates'], metrics['capital'], 'b-', linewidth=2)
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Capital')
axes[0, 0].set_title('Capital Evolution')
axes[0, 0].grid(True, alpha=0.3)

        # Drawdown
capital_series = np.array(metrics['capital'])
cummax = np.maximum.accumulate(capital_series)
drawdown = (cummax - capital_series) / cummax * 100

axes[0, 1].fill_between(metrics['dates'], 0, drawdown, color='red', alpha=0.3)
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Drawdown (%)')
axes[0, 1].set_title('Drawdown Analysis')
axes[0, 1].grid(True, alpha=0.3)

        # Premium vs Claims
cumulative_premiums = np.cumsum(metrics['premiums'])
cumulative_claims = np.cumsum(metrics['retained_losses'])

axes[1, 0].plot(metrics['dates'], cumulative_premiums, label='Premiums', linewidth=2)
axes[1, 0].plot(metrics['dates'], cumulative_claims, label='Retained Losses', linewidth=2)
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Cumulative Amount')
axes[1, 0].set_title('Premiums vs Retained Losses')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

        # Return distribution
returns = np.diff(capital_series) / capital_series[:-1]
axes[1, 1].hist(returns, bins=50, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(np.mean(returns), color='r', linestyle='--',
label=f'Mean: {np.mean(returns):.4f}')
axes[1, 1].set_xlabel('Returns')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Return Distribution')
axes[1, 1].legend()

        # Rolling metrics
window = min(60, len(returns) // 4)
rolling_mean = pd.Series(returns).rolling(window).mean()
rolling_vol = pd.Series(returns).rolling(window).std()

axes[2, 0].plot(metrics['dates'][1:], rolling_mean, label='Rolling Mean')
axes[2, 0].plot(metrics['dates'][1:], rolling_vol, label='Rolling Volatility')
axes[2, 0].set_xlabel('Date')
axes[2, 0].set_ylabel('Value')
axes[2, 0].set_title(f'{window}-Period Rolling Metrics')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

        # Performance summary
summary_text = f"""
Total Return: {self.results['total_return_%']:.2f}%
Annual Return: {self.results['annual_return_%']:.2f}%
Volatility: {self.results['volatility_%']:.2f}%
Sharpe Ratio: {self.results['sharpe_ratio']:.2f}
Max Drawdown: {self.results['max_drawdown_%']:.2f}%
Loss Ratio: {self.results['loss_ratio']:.2f}
"""

axes[2, 1].text(0.1, 0.5, summary_text, fontsize=12,
verticalalignment='center', family='monospace')
axes[2, 1].set_xlim(0, 1)
axes[2, 1].set_ylim(0, 1)
axes[2, 1].axis('off')
axes[2, 1].set_title('Performance Summary')

plt.tight_layout()
plt.show()

# Example backtest
import pandas as pd

# Generate sample historical data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=1000, freq='D')

# Market returns with occasional shocks
returns = np.random.normal(0.0003, 0.01, len(dates))
shocks = np.random.choice([0, -0.05], size=len(dates), p=[0.99, 0.01])
returns = returns + shocks

# Loss events
losses = np.random.exponential(10000, len(dates))
* (np.random.rand(len(dates)) < 0.05)

historical_data = pd.DataFrame({
'return': returns,
'loss': losses
}, index=dates)

# Define strategy
def adaptive_insurance_strategy(date, row, capital, positions):
"""Adaptive insurance strategy based on market conditions."""

    # Base parameters
base_retention = capital * 0.001
# 0.1% of capital
base_limit = capital
* 0.05
# 5% of capital

    # Adjust based on recent volatility
if len(positions) > 20:
recent_returns = [p.get('return', 0) for p in positions[-20:]]
recent_vol = np.std(recent_returns)

if recent_vol > 0.015:
# High volatility
retention = base_retention * 0.5
limit = base_limit
* 1.5
else:
retention = base_retention
limit = base_limit
else:
retention = base_retention
limit = base_limit

    # Calculate premium (simplified)
premium = limit * 0.001 + (limit - retention)
* 0.0005

return {
'retention': retention,
'limit': limit,
'premium': premium
}

# Run backtest
backtester = InsuranceBacktester(historical_data)
results = backtester.backtest(adaptive_insurance_strategy, initial_capital=10000000)

# Display results
print("\nBacktest Results:")
print("-" * 40)
for key, value in results.items():
if key != 'metrics_history':
if isinstance(value, float):
print(f"{key}: {value:.2f}")
else:
print(f"{key}: {value}")

# Visualize
backtester.plot_backtest()
```

#### Sample Output

![Backtesting Example](../../../theory/figures/backtesting_example.png)

```
Backtest Results:
----------------------------------------
total_return_%: -24.80
annual_return_%: -6.93
volatility_%: 18.12
sharpe_ratio: -0.38
sortino_ratio: -0.48
max_drawdown_%: 49.10
total_premiums: 568714.83
total_claims: 567571.65
total_retained: 282088.83
loss_ratio: 0.50
final_capital: 7520085.91
survival: True
```

(model-validation)=
## Model Validation

### Cross-Validation for Time Series

#### Critical Points for Insurance Model Validation

1. **Standard K-Fold is WRONG for Time Series**
- Randomly shuffles data, breaking temporal order
- Uses future data to predict past (data leakage)
- Gives overly optimistic performance estimates


2. **Time Series Split (Expanding Window)**
- Respects temporal order
- Training set grows with each fold
- Good for limited data

3. **Walk-Forward Analysis (Fixed Window)**
- Maintains consistent training size
- Better mimics production deployment
- Ideal for parameter stability testing


4. **Blocked CV with Gap**
- Prevents immediate lookahead bias
- Gap ensures no information leakage
- Good for high-frequency data

5. **Purged K-Fold**
- Removes data around test set
- Prevents temporal leakage in financial data
- Best for data with strong autocorrelation

#### Best Practices:

✓ Always respect temporal order
✓ Include gap/purge for autocorrelated data
✓ Use multiple CV strategies to validate robustness
✓ Monitor parameter stability across folds
✓ Check for distribution shift between folds

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


class TimeSeriesCrossValidation:
"""Comprehensive demonstration of time series cross-validation for insurance models."""

def __init__(self, n_periods=1000, seed=42):
np.random.seed(seed)
self.n_periods = n_periods

        # Generate realistic time series data with autocorrelation
self.generate_time_series_data()

def generate_time_series_data(self):
"""Generate correlated time series data typical of insurance metrics."""

        # Create dates
self.dates = pd.date_range('2020-01-01', periods=self.n_periods, freq='D')

        # Generate autocorrelated returns using AR(1) process
phi = 0.3
# Autocorrelation coefficient
sigma = 0.02
returns = [np.random.normal(0.0005, sigma)]

for t in range(1, self.n_periods):
            # AR(1) process with trend
innovation = np.random.normal(0, sigma)
new_return = 0.0005 + phi
* (returns[-1] - 0.0005) + innovation
returns.append(new_return)

        # Generate loss frequency (Poisson with time-varying rate)
base_rate = 3
seasonal_pattern = 1 + 0.3 * np.sin(2
* np.pi * np.arange(self.n_periods) / 365)
loss_counts = np.random.poisson(base_rate
* seasonal_pattern)

        # Generate loss severities (lognormal with trend)
severities = []
for t in range(self.n_periods):
trend_factor = 1 + 0.0002 * t
# Inflation trend
if loss_counts[t] > 0:
losses = np.random.lognormal(10, 2, loss_counts[t])
* trend_factor
severities.append(np.sum(losses))
else:
severities.append(0)

        # Create DataFrame
self.data = pd.DataFrame({
'date': self.dates,
'return': returns,
'loss_count': loss_counts,
'total_loss': severities,
'volatility': pd.Series(returns).rolling(20).std().fillna(sigma)
})

        # Add features for modeling
self.data['month'] = self.data['date'].dt.month
self.data['day_of_year'] = self.data['date'].dt.dayofyear
self.data['rolling_mean_return'] = self.data['return'].rolling(10).mean().fillna(0.0005)
self.data['rolling_loss_rate'] = self.data['total_loss'].rolling(30).mean().fillna(10000)

def standard_kfold_cv(self, n_splits=5):
"""Standard K-Fold (WRONG for time series) - for comparison."""
print("1. STANDARD K-FOLD CROSS-VALIDATION (Incorrect for Time Series)")
print("-" * 60)

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

        # Prepare features and target

X = self.data[['volatility', 'rolling_mean_return', 'rolling_loss_rate']].values
y = self.data['total_loss'].values

kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

scores = []
for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            # Train model
model = LinearRegression()
model.fit(X[train_idx], y[train_idx])

            # Predict
y_pred = model.predict(X[test_idx])

            # Score
mse = mean_squared_error(y[test_idx], y_pred)
r2 = r2_score(y[test_idx], y_pred)
scores.append({'fold': fold, 'mse': mse, 'r2': r2})

print(f"Average MSE: {np.mean([s['mse'] for s in scores]):,.0f}")
print(f"Average R²: {np.mean([s['r2'] for s in scores]):.4f}")
print("⚠️ WARNING: This method uses future data to predict past - invalid for time series!")

return scores

def time_series_split_cv(self, n_splits=5):
"""Time Series Split (CORRECT) - expanding window."""
print("\n2. TIME SERIES SPLIT (Expanding Window)")
print("-"
* 60)

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = self.data[['volatility', 'rolling_mean_return', 'rolling_loss_rate']].values
y = self.data['total_loss'].values

tscv = TimeSeriesSplit(n_splits=n_splits)

scores = []
train_sizes = []
test_sizes = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            # Train model
model = LinearRegression()
model.fit(X[train_idx], y[train_idx])

            # Predict
y_pred = model.predict(X[test_idx])

            # Score
mse = mean_squared_error(y[test_idx], y_pred)
r2 = r2_score(y[test_idx], y_pred)

scores.append({'fold': fold, 'mse': mse, 'r2': r2})
train_sizes.append(len(train_idx))
test_sizes.append(len(test_idx))

print(f"Fold {fold}: Train size: {len(train_idx):4d}, Test size: {len(test_idx):3d}, "
f"MSE: {mse:12,.0f}, R²: {r2:.4f}")

print(f"\nAverage MSE: {np.mean([s['mse'] for s in scores]):,.0f}")
print(f"Average R²: {np.mean([s['r2'] for s in scores]):.4f}")
print("✓ Each fold only uses past data to predict future")

return scores, train_sizes, test_sizes

def walk_forward_cv(self, window_size=200, step_size=50):
"""Walk-Forward Analysis with fixed window size."""
print("\n3. WALK-FORWARD ANALYSIS (Fixed Window)")
print("-" * 60)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

X = self.data[['volatility', 'rolling_mean_return', 'rolling_loss_rate', 'month']].values
y = self.data['total_loss'].values

n_windows = (len(X) - window_size - step_size) // step_size

scores = []
predictions = []
actuals = []

for i in range(n_windows):
            # Define window
train_start = i
* step_size
train_end = train_start + window_size
test_start = train_end
test_end = test_start + step_size

            # Train
model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
model.fit(X[train_start:train_end], y[train_start:train_end])

            # Test
y_pred = model.predict(X[test_start:test_end])
y_true = y[test_start:test_end]

            # Metrics
mse = mean_squared_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true + 1, y_pred + 1)
# Add 1 to avoid division by zero

scores.append({
'window': i + 1,
'train_period': (train_start, train_end),
'test_period': (test_start, test_end),
'mse': mse,
'mape': mape
})

predictions.extend(y_pred)
actuals.extend(y_true)

print(f"Number of windows: {n_windows}")
print(f"Window size: {window_size}, Step size: {step_size}")
print(f"Average MSE: {np.mean([s['mse'] for s in scores]):,.0f}")
print(f"Average MAPE: {np.mean([s['mape'] for s in scores]):.2%}")
print("✓ Fixed window maintains consistent training size")

return scores, predictions, actuals

def blocked_cv(self, n_splits=5, gap=10):
"""Blocked Time Series CV with gap between train and test."""
print("\n4. BLOCKED TIME SERIES CV (With Gap)")
print("-" * 60)

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

X = self.data[['volatility', 'rolling_mean_return', 'rolling_loss_rate']].values
y = self.data['total_loss'].values

block_size = len(X) // (n_splits + 1)

scores = []

for fold in range(n_splits):
            # Training: all blocks before current
train_end = (fold + 1)
* block_size

            # Gap to prevent lookahead bias
test_start = train_end + gap
test_end = min(test_start + block_size, len(X))

if test_end > len(X):
break

            # Train
model = Ridge(alpha=1.0)
model.fit(X[:train_end], y[:train_end])

            # Test
y_pred = model.predict(X[test_start:test_end])
y_true = y[test_start:test_end]

mse = mean_squared_error(y_true, y_pred)

scores.append({
'fold': fold + 1,
'train_size': train_end,
'test_size': test_end - test_start,
'gap': gap,
'mse': mse
})

print(f"Fold {fold+1}: Train: 0-{train_end:3d}, "
f"Gap: {gap}, Test: {test_start:3d}-{test_end:3d}, "
f"MSE: {mse:12,.0f}")

print(f"\nAverage MSE: {np.mean([s['mse'] for s in scores]):,.0f}")
print(f"✓ Gap of {gap} periods prevents information leakage")

return scores

def purged_cv(self, n_splits=5, purge_window=10):
"""Purged K-Fold for financial time series."""
print("\n5. PURGED K-FOLD CV (For Financial Data)")
print("-" * 60)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

X = self.data[['volatility', 'rolling_mean_return', 'rolling_loss_rate']].values
y = self.data['total_loss'].values
n = len(X)

        # Create fold indices
fold_size = n // n_splits
scores = []

for fold in range(n_splits):
            # Test fold
test_start = fold
* fold_size
test_end = min((fold + 1) * fold_size, n)

            # Training indices (exclude test fold and purge window around it)
train_indices = []
for i in range(n):
if i < test_start - purge_window or i >= test_end + purge_window:
train_indices.append(i)

if len(train_indices) < 100:
# Skip if too few training samples
continue

            # Train
model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
model.fit(X[train_indices], y[train_indices])

            # Test
y_pred = model.predict(X[test_start:test_end])
y_true = y[test_start:test_end]

mse = mean_squared_error(y_true, y_pred)

scores.append({
'fold': fold + 1,
'train_size': len(train_indices),
'test_size': test_end - test_start,
'mse': mse
})

print(f"Fold {fold+1}: Train size: {len(train_indices):3d}, "
f"Test: {test_start:3d}-{test_end:3d}, "
f"Purge: ±{purge_window}, MSE: {mse:12,.0f}")

print(f"\nAverage MSE: {np.mean([s['mse'] for s in scores]):,.0f}")
print(f"✓ Purge window of {purge_window} prevents temporal leakage")

return scores

def plot_cv_comparison(self):
"""Visualize different CV strategies."""

fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        # 1. Show the time series data
axes[0, 0].plot(self.data['date'], self.data['total_loss'], alpha=0.7, linewidth=0.5)
axes[0, 0].set_title('Insurance Loss Time Series')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Total Loss')
axes[0, 0].grid(True, alpha=0.3)

        # 2. Autocorrelation plot
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(self.data['total_loss'], ax=axes[0, 1])
axes[0, 1].set_title('Autocorrelation of Losses')
axes[0, 1].set_xlabel('Lag')
axes[0, 1].set_ylabel('Autocorrelation')

        # 3. Time Series Split visualization
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(tscv.split(self.data)):
            # Plot train/test splits
axes[1, 0].scatter(train_idx, [fold]*len(train_idx), c='blue', s=1, alpha=0.5)
axes[1, 0].scatter(test_idx, [fold]*len(test_idx), c='red', s=1, alpha=0.5)

axes[1, 0].set_title('Time Series Split Strategy')
axes[1, 0].set_xlabel('Time Index')
axes[1, 0].set_ylabel('Fold')
axes[1, 0].legend(['Train', 'Test'])
axes[1, 0].grid(True, alpha=0.3)

        # 4. Walk-Forward visualization
window_size = 200
step_size = 50
n_windows = min(8, (len(self.data) - window_size - step_size) // step_size)

for i in range(n_windows):
train_start = i * step_size
train_end = train_start + window_size
test_start = train_end
test_end = test_start + step_size

axes[1, 1].barh(i, window_size, left=train_start, height=0.8,
color='blue', alpha=0.6, label='Train' if i == 0 else '')
axes[1, 1].barh(i, step_size, left=test_start, height=0.8,
color='red', alpha=0.6, label='Test' if i == 0 else '')

axes[1, 1].set_title('Walk-Forward Analysis')
axes[1, 1].set_xlabel('Time Index')
axes[1, 1].set_ylabel('Window')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

        # 5. Purged K-Fold visualization
n_splits = 5
purge_window = 20
fold_size = len(self.data) // n_splits

for fold in range(n_splits):
test_start = fold
* fold_size
test_end = min((fold + 1) * fold_size, len(self.data))

            # Test region
axes[2, 0].barh(fold, test_end - test_start, left=test_start,
height=0.8, color='red', alpha=0.6)

            # Purge regions
if test_start - purge_window >= 0:
axes[2, 0].barh(fold, purge_window, left=test_start - purge_window,
height=0.8, color='yellow', alpha=0.6)
if test_end + purge_window <= len(self.data):
axes[2, 0].barh(fold, purge_window, left=test_end,
height=0.8, color='yellow', alpha=0.6)

            # Training regions (simplified visualization)
if test_start - purge_window > 0:
axes[2, 0].barh(fold, test_start - purge_window, left=0,
height=0.8, color='blue', alpha=0.3)
if test_end + purge_window < len(self.data):
axes[2, 0].barh(fold, len(self.data) - test_end - purge_window,
left=test_end + purge_window, height=0.8, color='blue', alpha=0.3)

axes[2, 0].set_title('Purged K-Fold with Gap')
axes[2, 0].set_xlabel('Time Index')
axes[2, 0].set_ylabel('Fold')
axes[2, 0].legend(['Test', 'Purge', 'Train'])
axes[2, 0].grid(True, alpha=0.3)

        # 6. Performance comparison
methods = ['Standard\nK-Fold', 'Time Series\nSplit', 'Walk\nForward',
'Blocked\nCV', 'Purged\nK-Fold']
        # These would be actual MSE values from the methods
mse_values = [250000, 180000, 160000, 170000, 165000]
# Example values
colors = ['red', 'green', 'green', 'green', 'green']

bars = axes[2, 1].bar(methods, mse_values, color=colors, alpha=0.7)
axes[2, 1].set_title('CV Method Performance Comparison')
axes[2, 1].set_ylabel('Average MSE')
axes[2, 1].set_xlabel('CV Method')
axes[2, 1].tick_params(axis='x', rotation=45)

        # Add warning for standard K-Fold
axes[2, 1].text(0, mse_values[0] + 10000, '⚠️ Invalid\nfor TS',
ha='center', fontsize=10, color='red')

plt.tight_layout()
plt.show()

# Run comprehensive demonstration
print("="*60)
print("TIME SERIES CROSS-VALIDATION DEMONSTRATION")
print("="*60)
print("\nGenerating correlated time series data with 1000 periods...\n")

# Initialize and run demonstrations
cv_demo = TimeSeriesCrossValidation(n_periods=1000, seed=42)

# Run all CV methods
standard_scores = cv_demo.standard_kfold_cv(n_splits=5)
ts_scores, train_sizes, test_sizes = cv_demo.time_series_split_cv(n_splits=5)
wf_scores, predictions, actuals = cv_demo.walk_forward_cv(window_size=200, step_size=50)
blocked_scores = cv_demo.blocked_cv(n_splits=5, gap=10)
purged_scores = cv_demo.purged_cv(n_splits=5, purge_window=10)

# Visualize
print("\n" + "="*60)
print("VISUALIZATION OF CV STRATEGIES")
print("="*60)
cv_demo.plot_cv_comparison()
```

#### Sample Output

![Cross-Validation Examples](../../../theory/figures/cv_examples.png)

```
Walk-Forward Validation Results:
----------------------------------------
mean_performance: 0.0757
std_performance: 0.0214
min_performance: 0.0354
max_performance: 0.1136
sharpe_ratio: 3.5398
win_rate: 1.0000
n_folds: 16

Backtest Results:
----------------------------------------
total_return_%: -24.80
annual_return_%: -6.93
volatility_%: 18.12
sharpe_ratio: -0.38
sortino_ratio: -0.48
max_drawdown_%: 49.10
total_premiums: 568714.83
total_claims: 567571.65
total_retained: 282088.83
loss_ratio: 0.50
final_capital: 7520085.91
survival: True
============================================================
TIME SERIES CROSS-VALIDATION DEMONSTRATION
============================================================

Generating correlated time series data with 1000 periods...

1. STANDARD K-FOLD CROSS-VALIDATION (Incorrect for Time Series)
------------------------------------------------------------
Average MSE: 2,384,621,964,008
Average R²: -0.0014
⚠️ WARNING: This method uses future data to predict past - invalid for time series!

2. TIME SERIES SPLIT (Expanding Window)
------------------------------------------------------------
Fold 1: Train size:
170, Test size: 166, MSE: 2,567,880,070,435, R²: 0.0060
Fold 2: Train size:
336, Test size: 166, MSE: 1,766,387,331,946, R²: 0.0230
Fold 3: Train size:
502, Test size: 166, MSE: 1,157,698,242,406, R²: -0.0073
Fold 4: Train size:
668, Test size: 166, MSE: 1,228,997,308,765, R²: 0.0401
Fold 5: Train size:
834, Test size: 166, MSE: 7,091,176,011,971, R²: 0.0284

Average MSE: 2,762,427,793,105
Average R²: 0.0180
✓ Each fold only uses past data to predict future

3. WALK-FORWARD ANALYSIS (Fixed Window)
------------------------------------------------------------
Number of windows: 15
Window size: 200, Step size: 50
Average MSE: 5,236,986,571,393
Average MAPE: 7332236.65%
✓ Fixed window maintains consistent training size

4. BLOCKED TIME SERIES CV (With Gap)
------------------------------------------------------------
Fold 1: Train: 0-166, Gap: 10, Test: 176-342, MSE: 2,478,133,892,977
Fold 2: Train: 0-332, Gap: 10, Test: 342-508, MSE: 1,999,680,046,316
Fold 3: Train: 0-498, Gap: 10, Test: 508-674, MSE: 879,386,307,408
Fold 4: Train: 0-664, Gap: 10, Test: 674-840, MSE: 1,237,952,734,707
Fold 5: Train: 0-830, Gap: 10, Test: 840-1000, MSE: 7,329,263,519,493

Average MSE: 2,784,883,300,180
✓ Gap of 10 periods prevents information leakage

5. PURGED K-FOLD CV (For Financial Data)
------------------------------------------------------------
Fold 1: Train size: 790, Test:
0-200, Purge: ±10, MSE: 931,398,609,705
Fold 2: Train size: 780, Test: 200-400, Purge: ±10, MSE: 1,773,539,840,096
Fold 3: Train size: 780, Test: 400-600, Purge: ±10, MSE: 7,021,802,954,931
Fold 4: Train size: 780, Test: 600-800, Purge: ±10, MSE: 808,735,067,736
Fold 5: Train size: 790, Test: 800-1000, Purge: ±10, MSE: 7,457,208,690,945

Average MSE: 3,598,537,032,683
✓ Purge window of 10 prevents temporal leakage
```

## Key Statistical Methods for Insurance Analysis:

1. **Monte Carlo Methods**

   - Basic simulation for complex systems
   - Variance reduction techniques (antithetic, control variates)
   - Importance sampling for rare events

2. **Convergence Diagnostics**

   - Gelman-Rubin statistic for MCMC
   - Effective sample size accounting for correlation
   - Multiple diagnostic tests for reliability

3. **Confidence Intervals**

   - Classical, bootstrap, and BCa methods
   - Appropriate for different data distributions
   - Bootstrap for complex statistics

4. **Hypothesis Testing**

   - Tests for ergodic advantage
   - Non-parametric alternatives
   - Permutation tests for flexibility

5. **Bootstrap Methods**

   - Resampling for uncertainty quantification
   - Block bootstrap for time series
   - Wild bootstrap for heteroskedasticity

6. **Walk-Forward Validation**

   - Realistic out-of-sample testing
   - Parameter stability assessment
   - Time series appropriate

7. **Backtesting**

   - Historical performance evaluation
   - Comprehensive risk metrics
   - Visual diagnostics

8. **Model Validation**

   - Cross-validation for time series
   - Multiple scoring metrics
   - Robustness checks

## Key Takeaways

1. **Monte Carlo is fundamental**: But use variance reduction for efficiency
2. **Convergence must be verified**: Multiple diagnostics prevent false conclusions
3. **Bootstrap provides flexibility**: Works for complex statistics without assumptions
4. **Time series need special methods**: Block bootstrap, walk-forward validation
5. **Backtesting is powerful**: But beware of overfitting to historical data
6. **Multiple validation approaches**: No single method captures all aspects
7. **Statistical rigor is essential**: Proper testing prevents costly mistakes

## Next Steps

- [Chapter 6: References](06_references.md) - Academic papers and resources
- [Chapter 4: Optimization Theory](04_optimization_theory.md) - Optimization methods
- [Chapter 1: Ergodic Economics](01_ergodic_economics.md) - Theoretical foundation
