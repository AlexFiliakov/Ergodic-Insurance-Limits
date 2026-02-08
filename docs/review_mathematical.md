# Mathematical Implementation Correctness Review

**Reviewer**: Math Review Agent (Pass 1 + Pass 2)
**Date**: 2026-02-06 (Pass 1), 2026-02-07 (Pass 2)
**Status**: COMPLETE

## Areas Reviewed

| Module | File | Status | Issues Found (Pass 1) | Issues Found (Pass 2) |
|--------|------|--------|-----------------------|-----------------------|
| HJB Solver | `hjb_solver.py` | Reviewed | 4 | 0 |
| Optimization | `optimization.py` | Reviewed | 2 | 0 |
| Risk Metrics | `risk_metrics.py` | Reviewed | 2 | 3 |
| Convergence (basic) | `convergence.py` | Reviewed | 0 | 1 |
| Convergence (advanced) | `convergence_advanced.py` | Reviewed | 2 | 1 |
| Ergodic Analyzer | `ergodic_analyzer.py` | Reviewed | 0 | 3 |
| Monte Carlo | `monte_carlo.py` | Reviewed | 0 | 0 |
| Loss Distributions | `loss_distributions.py` | Reviewed | 0 | 0 |
| Stochastic Processes | `stochastic_processes.py` | Reviewed | 0 | 0 |
| Bootstrap Analysis | `bootstrap_analysis.py` | Reviewed | 0 | 1 |
| Statistical Tests | `statistical_tests.py` | Reviewed | 0 | 0 |
| Ruin Probability | `ruin_probability.py` | Reviewed | 1 | 0 |
| Sensitivity | `sensitivity.py` | Reviewed | 0 | 0 |
| Decision Engine | `decision_engine.py` | Reviewed | 0 | 0 |
| Adaptive Stopping | `adaptive_stopping.py` | Reviewed | 0 | 0 |
| Business Optimizer | `business_optimizer.py` | Reviewed | 0 | 0 |
| Optimal Control | `optimal_control.py` | Reviewed | 0 | 0 |
| Walk-Forward Validator | `walk_forward_validator.py` | Reviewed | - | 0 |
| Pareto Frontier | `pareto_frontier.py` | Reviewed | - | 0 |
| Simulation | `simulation.py` | Reviewed | - | 0 |
| Decimal Utils | `decimal_utils.py` | Reviewed | - | 0 |

## Summary

- **Total issues identified (Pass 1)**: 11
- **Total issues identified (Pass 2)**: 10
- **Combined total**: 21 unique issues
- **Critical (Pass 1)**: 4 (HJB solver, optimization convergence)
- **High**: 6 (Sortino ratio, parametric VaR, convergence tau, growth rate clamping, spectral ESS)
- **Medium**: 8 (population std across modules, BCa bootstrap, Geweke test, Welch's t-test)
- **Low**: 3 (Hill estimator naming, summary_statistics std)

## Issues Filed

### Pass 1 Issues (2026-02-06)

#### Critical

1. **#373 - HJB upwind scheme only applies to dimension 0** - Multi-dimensional HJB problems silently return zeros for advection in all dimensions > 0.
2. **#376 - HJB policy improvement ignores value function gradient** - The Hamiltonian is computed using only the running cost, completely omitting the crucial drift dot grad(V) term.
3. **#381 - Augmented Lagrangian slack threshold sign error** - The slack variable threshold has a flipped sign on the multiplier term, causing incorrect constraint activation.
4. **#385 - Augmented Lagrangian penalty update operator precedence** - Python ternary operator precedence causes unintended comparison, making penalty always increase on first iteration.

#### High

5. **#377 - HJB grid spacing assumes uniform spacing for log-scale grids** - Finite difference operators use constant dx from first two grid points, which is wrong for logarithmic grids.
6. **#386 - Sortino ratio uses wrong downside deviation formula** (2 locations) - Averages squared deviations only over below-target observations instead of all observations.
7. **#389 - Parametric VaR uses population standard deviation** - Uses ddof=0 instead of ddof=1, biasing VaR downward for small samples.
8. **#391 - Advanced convergence: integrated autocorrelation time missing factor of 2** - Computes tau = 1 + sum(rho_k) instead of tau = 1 + 2*sum(rho_k).

#### Medium

9. **#395 - HJB absorbing boundary conditions implement Dirichlet** - Absorbing BCs set value to constant (Dirichlet) instead of zero second derivative.
10. **#396 - Advanced convergence: initial monotone sequence check is incorrect** - Compares wrong pair sums for Geyer (1992) monotonicity criterion.
11. **#400 - Ruin probability bootstrap CI ignores configured seed** - Creates unseeded RNG instead of using config.seed.

### Pass 2 Issues (2026-02-07)

#### High

12. **#474 - Time-average growth rate clamped to -1.0 biases ergodic comparisons upward** (`ergodic_analyzer.py:781`) - The `max(growth_rate, -1.0)` clamp hides ruin events and biases time-average growth rates upward. The time-average growth rate g = (1/T) ln(X(T)/X(0)) can legitimately be < -1.0 when X(T)/X(0) < e^(-T), and clamping removes the signal that distinguishes catastrophic scenarios from merely bad ones.

13. **#476 - Spectral density ESS overestimated by factor of 2** (`convergence_advanced.py:156`) - The integrated autocorrelation time formula `tau = s_zero / (2 * variance)` divides by 2, but scipy's `periodogram()` returns a one-sided PSD that already integrates the negative frequencies. The correct formula is `tau = s_zero / variance`, producing ESS values roughly half what is currently reported.

#### Medium

14. **#478 - Ergodic analyzer uses population std (ddof=0) for all growth rate statistics** (`ergodic_analyzer.py:827,834,867,872,1489,1502,1930`) - Seven call sites use `np.std()` without `ddof=1`. For simulated growth rates (sample data), Bessel's correction should be applied. Affects volatility ratios, ensemble statistics, and significance thresholds.

15. **#482 - BCa bootstrap bias correction produces -inf/+inf, causing NaN confidence intervals** (`bootstrap_analysis.py:401`) - `stats.norm.ppf(np.mean(bootstrap_dist < original_stat))` returns -inf when all bootstrap replicates exceed the original statistic (proportion = 0), or +inf when all are below (proportion = 1). This propagates NaN through the BCa interval calculation.

16. **#486 - Geweke convergence test ignores autocorrelation, producing anti-conservative p-values** (`convergence.py:338-340`) - Uses simple variance `np.var(first_portion)` instead of the spectral density at frequency zero. For positively autocorrelated chains (the common case), the true variance of the mean is larger, so the current implementation produces z-scores that are too large and p-values that are too small.

17. **#488 - Sharpe ratio in risk_adjusted_metrics uses population std (ddof=0)** (`risk_metrics.py:411`) - `np.std(returns)` in Sharpe ratio calculation uses population std, which slightly underestimates volatility and inflates the Sharpe ratio.

18. **#490 - ROEAnalyzer uses population std (ddof=0) in volatility_metrics and rolling_statistics** (`risk_metrics.py:799,803,807,849`) - Four call sites in `ROEAnalyzer` use `np.std()` without `ddof=1`, affecting annualized volatility, rolling mean, rolling std, and rolling Sharpe calculations.

19. **#504 - compare_scenarios uses Student's t-test instead of Welch's t-test** (`ergodic_analyzer.py:1708`) - `stats.ttest_ind()` uses default `equal_var=True` (Student's t-test), but insured vs uninsured growth-rate distributions have inherently different variances. Welch's t-test (`equal_var=False`) is recommended as the default per Ruxton (2006).

#### Low

20. **#507 - Hill estimator returns Pareto MLE alpha, not Hill's gamma** (`risk_metrics.py:387-388`) - The method computes alpha = k / sum(ln(X_i/threshold)), which is the Pareto shape parameter (reciprocal of the classical Hill estimator gamma). This is a naming/documentation issue, not a computation error.

21. **#508 - RiskMetrics.summary_statistics uses population std (ddof=0)** (`risk_metrics.py:487`) - `np.std(self.losses)` uses population std. Low impact for typical sample sizes but inconsistent with statistical convention.

## Cross-Reference with Existing Issues

| Existing Issue | Related Finding | Status |
|---------------|----------------|--------|
| #300 (HJB mathematical errors) | Issues 1, 2, 5, 9 confirm and extend | Confirmed + new detail |
| #350 (Convergence ESS incorrect) | Issues 8, 10 in convergence_advanced.py | Confirmed in advanced module |
| #351 (Optimization augmented Lagrangian) | Issues 3, 4 | Confirmed + new detail |
| #355 (Ruin probability bugs) | Issue 11 | New finding |
| #303 (Decision engine recursion) | Not reproduced in current code | May be fixed |
| #348 (Monte Carlo state reset) | Not reproduced in current code | May be fixed |
| #349 (Simulation inconsistencies) | Not reproduced in current code | May be fixed |
| #352 (BusinessOptimizer unreliable) | Root cause likely Issues 3, 4 (optimizer bugs) | Related |
| #389 (Parametric VaR ddof=0) | Pass 2 found 4 more ddof=0 locations (#478, #488, #490, #508) | Extended |
| #391 (Integrated autocorrelation time) | Pass 2 found related spectral ESS issue (#476) | Extended |

## Systemic Patterns Found

### Pattern 1: Population vs Sample Standard Deviation (ddof=0 vs ddof=1)

This is the most pervasive issue, appearing across 6 files and 14+ call sites:

| Issue | File | Lines |
|-------|------|-------|
| #389 | `risk_metrics.py` | 143 |
| #478 | `ergodic_analyzer.py` | 827, 834, 867, 872, 1489, 1502, 1930 |
| #488 | `risk_metrics.py` | 411 |
| #490 | `risk_metrics.py` | 799, 803, 807, 849 |
| #508 | `risk_metrics.py` | 487 |

**Recommendation**: A single sweep replacing `np.std(x)` with `np.std(x, ddof=1)` across the codebase, with a lint rule to prevent regressions.

### Pattern 2: Convergence Diagnostics Off by Factor of 2

Two separate convergence issues involve incorrect factors of 2:
- #391: Integrated autocorrelation time missing factor of 2
- #476: Spectral density ESS divides by 2 when it should not

These compound: if both are present in a diagnostic pipeline, they could partially cancel or compound depending on code path.

### Pattern 3: Edge Case Handling in Bootstrap Methods

- #400: Unseeded RNG breaks reproducibility
- #482: BCa bias correction produces -inf/+inf at boundary proportions

Bootstrap methods need systematic edge case review for degenerate samples.

## Modules with No Issues Found

- **loss_distributions.py**: Lognormal parameterization (mean/cv to mu/sigma) is correct: sigma = sqrt(log(1 + cv^2)), mu = log(mean) - sigma^2/2. Pareto inverse CDF x_m / u^(1/alpha) is correct. GPD correctly delegates to scipy.stats.genpareto.
- **stochastic_processes.py**: GBM exact solution exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*z) is correct. LognormalVolatility bias correction exp(sigma*z - 0.5*sigma^2) ensures E[shock]=1. Mean-reverting process uses Euler-Maruyama (acceptable for dt=1).
- **statistical_tests.py**: Bootstrap permutation test for difference in means is correctly implemented. Ratio test and paired comparison appear sound.
- **convergence.py** (basic): Gelman-Rubin R-hat formula is correct. Basic ESS using Geyer's initial positive sequence with factor of 2 is correct.
- **pareto_frontier.py**: Pareto dominance check, crowding distance, and 2D hypervolume calculation are correct.
- **walk_forward_validator.py**: Train/test split logic and walk-forward protocol are correct.
- **simulation.py**: SimulationResults container and geometric mean ROE calculation are correct.
- **decimal_utils.py**: Decimal conversion, quantization, and safe division are correct.
- **sensitivity.py**: Elasticity formula (metric_range/baseline_metric) / (param_range/baseline_value) is standard.

## Methodology

### Pass 1 (2026-02-06)
- Focused on HJB solver, optimization, and core numerical modules
- Compared implementations against academic references (Peters 2019, Geyer 1992, Efron 1987)
- Identified 11 issues including 4 critical HJB/optimization bugs

### Pass 2 (2026-02-07)
- Systematic line-by-line review of all 14 priority files
- Cross-referenced numpy/scipy function defaults against statistical conventions
- Identified 10 additional issues, primarily in statistical computation defaults
- Verified mathematical correctness of loss distributions, stochastic processes, and Pareto frontier algorithms
