Glossary
========

A
-

**Attachment Point**
   The level at which an insurance layer begins to provide coverage. For example, a layer with $5M attachment covers losses above $5M.

**Attritional Losses**
   High-frequency, low-severity claims that occur regularly in normal business operations. Typically modeled with Poisson frequency and limited severity.

B
-

**Bankruptcy Probability**
   The likelihood that assets fall below zero (or a threshold) within a given time horizon. A key constraint in ergodic optimization.

**Burn-in Period**
   Initial simulation period excluded from analysis to eliminate dependency on starting conditions.

C
-

**Claims Development**
   The process by which insurance claims mature from initial reporting to final settlement. Modeled using development triangles.

**Combined Ratio**
   Sum of loss ratio and expense ratio. Values below 100% indicate underwriting profit.

**Conditional Value at Risk (CVaR)**
   Expected loss given that the loss exceeds the VaR threshold. Also called Expected Shortfall.

**Convergence**
   The stabilization of simulation results as the number of iterations increases. Monitored to ensure reliable estimates.

E
-

**Ensemble Average**
   Expected value across many parallel scenarios at a given time. Distinguished from time average in ergodic theory.

**Ergodic Theory**
   Mathematical framework distinguishing between time averages and ensemble averages. Central to understanding optimal insurance.

**Excess Layer**
   Insurance coverage above a primary layer. Attaches at the exhaustion point of underlying coverage.

G
-

**Geometric Brownian Motion (GBM)**
   Stochastic process where logarithm follows Brownian motion. Used to model multiplicative growth processes.

**Growth Rate**
   Rate of change in wealth or assets. Time-average growth rate is the key metric in ergodic optimization.

L
-

**Large Losses**
   Low-frequency, high-severity events that can significantly impact financial position. Typically modeled with heavy-tailed distributions.

**Layer**
   A band of insurance coverage between attachment and exhaustion points. Programs often stack multiple layers.

**Limit**
   Maximum amount an insurance policy will pay for covered losses. Can apply per occurrence or in aggregate.

**Loss Ratio**
   Claims paid divided by premiums earned. Primary metric of insurance profitability.

M
-

**Maximum Drawdown**
   Largest peak-to-trough decline in wealth or asset value. Measures worst-case historical scenario.

**Mean Reversion**
   Tendency of a variable to return to its long-term average. Modeled with Ornstein-Uhlenbeck process.

**Monte Carlo Simulation**
   Computational technique using random sampling to estimate probability distributions and expected values.

**Multiplicative Process**
   Process where changes are proportional to current value. Wealth dynamics are typically multiplicative.

O
-

**Operating Margin**
   Profit margin from core business operations, excluding insurance and financial items.

**Optimization Metric**
   Objective function being maximized or minimized. Common choices: time-average growth, Sharpe ratio, utility.

P
-

**Pareto Frontier**
   Set of solutions that are optimal for multi-objective optimization. No solution dominates another on all objectives.

**Poisson Process**
   Stochastic process modeling random events occurring at a constant average rate. Used for claim frequency.

**Premium Rate**
   Cost of insurance expressed as percentage of limit or exposure. Varies by layer and market conditions.

**Primary Layer**
   First layer of insurance coverage, attaching at ground (zero) or after a deductible.

R
-

**Retention**
   Amount of risk retained by insured before insurance responds. Similar to deductible.

**Return on Equity (ROE)**
   Net income divided by shareholder equity. Key performance metric for businesses.

**Risk-Adjusted Return**
   Return metric that accounts for volatility or downside risk. Examples: Sharpe ratio, Sortino ratio.

**Ruin Probability**
   See Bankruptcy Probability.

S
-

**Severity**
   Size or amount of an individual loss. Often modeled with lognormal or Pareto distributions.

**Sharpe Ratio**
   Excess return per unit of volatility. Adapted for ergodic context as growth rate divided by growth volatility.

**Solvency Ratio**
   Available capital divided by required capital. Regulatory measure of financial strength.

**Stochastic Process**
   Mathematical model of randomly evolving systems. Examples: Brownian motion, Poisson process, jump diffusion.

T
-

**Tail Risk**
   Risk of extreme events in distribution tails. Critical for insurance and catastrophic loss modeling.

**Time Average**
   Average experienced by single entity over time. Central to ergodic analysis.

**Time Horizon**
   Period over which analysis is conducted. Longer horizons reveal ergodic properties.

U
-

**Umbrella Coverage**
   High-level excess coverage sitting above multiple underlying policies. Provides catastrophic protection.

**Utility Function**
   Mathematical representation of preferences over outcomes. Ergodic approach eliminates need for arbitrary utility.

V
-

**Value at Risk (VaR)**
   Loss level that won't be exceeded with given confidence over specified period. Standard risk metric.

**Variance Reduction**
   Techniques to reduce statistical variance in Monte Carlo simulations. Examples: antithetic variates, control variates.

**Volatility**
   Standard deviation of returns or growth rates. Measures uncertainty or risk.

W
-

**Widget Manufacturer**
   Archetypal business model used throughout the framework. Represents typical manufacturing operations.

**Working Capital**
   Current assets minus current liabilities. Funds available for day-to-day operations.

Mathematical Notation
---------------------

:math:`g`
   Time-average growth rate

:math:`\langle r \rangle`
   Ensemble average (expected value) of returns

:math:`W(t)`
   Wealth or asset value at time t

:math:`\sigma`
   Volatility (standard deviation)

:math:`\lambda`
   Rate parameter for Poisson process (claim frequency)

:math:`\mu`
   Mean or drift parameter

:math:`T`
   Time horizon for analysis

:math:`\alpha`
   Confidence level (e.g., 95% for VaR)

:math:`L`
   Insurance limit

:math:`D`
   Deductible or retention

:math:`p`
   Premium rate

:math:`\rho`
   Correlation coefficient

Acronyms
--------

**CLT**
   Central Limit Theorem

**CV**
   Coefficient of Variation

**EV**
   Expected Value

**GBM**
   Geometric Brownian Motion

**IID**
   Independent and Identically Distributed

**MLE**
   Maximum Likelihood Estimation

**OU**
   Ornstein-Uhlenbeck (mean-reverting process)

**PV**
   Present Value

**ROA**
   Return on Assets

**ROE**
   Return on Equity

**VaR**
   Value at Risk

**CVaR**
   Conditional Value at Risk
