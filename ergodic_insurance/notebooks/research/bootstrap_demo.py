"""Demo code for bootstrap confidence intervals to be added to notebooks.

This code can be copied into Jupyter notebooks to demonstrate the bootstrap functionality.
"""

# %% [markdown]
# ## 7. Bootstrap Confidence Intervals for Simulation Results
#
# We can now compute bootstrap confidence intervals for key metrics to understand the statistical uncertainty in our simulation results.

# %%
from ergodic_insurance.bootstrap_analysis import BootstrapAnalyzer, bootstrap_confidence_interval
from ergodic_insurance.statistical_tests import difference_in_means_test, ratio_of_metrics_test

# Enable bootstrap confidence intervals in simulation config
bootstrap_config = MonteCarloConfig(
    n_simulations=10_000,
    n_years=10,
    parallel=True,
    n_workers=4,
    seed=42,
    compute_bootstrap_ci=True,  # Enable bootstrap CI
    bootstrap_confidence_level=0.95,
    bootstrap_n_iterations=10_000,
    bootstrap_method="percentile",
    progress_bar=True,
)

print("Running simulation with bootstrap confidence intervals...")
print("=" * 50)

# %%
# Run simulation with bootstrap CIs
engine_with_bootstrap = setup_simulation_engine(
    n_simulations=bootstrap_config.n_simulations,
    n_years=bootstrap_config.n_years,
    parallel=bootstrap_config.parallel,
)

# Update engine config
engine_with_bootstrap.config = bootstrap_config

# Run simulation
results_with_bootstrap = engine_with_bootstrap.run()

print("\nSimulation complete!")
print(f"Execution time: {results_with_bootstrap.execution_time:.2f} seconds")

# %% [markdown]
# ### Display Bootstrap Confidence Intervals

# %%
# Display bootstrap confidence intervals
if results_with_bootstrap.bootstrap_confidence_intervals:
    print("\nBootstrap Confidence Intervals (95%)")
    print("=" * 50)

    for metric_name, (
        lower,
        upper,
    ) in results_with_bootstrap.bootstrap_confidence_intervals.items():
        if (
            "assets" in metric_name.lower()
            or "losses" in metric_name.lower()
            or "recoveries" in metric_name.lower()
        ):
            print(f"{metric_name:30} [{format_currency(lower):>15}, {format_currency(upper):>15}]")
        elif "probability" in metric_name.lower():
            print(f"{metric_name:30} [{lower:>15.2%}, {upper:>15.2%}]")
        else:
            print(f"{metric_name:30} [{lower:>15.6f}, {upper:>15.6f}]")

    # Calculate confidence interval width as percentage of estimate
    mean_assets_ci = results_with_bootstrap.bootstrap_confidence_intervals.get(
        "Mean Final Assets", (0, 0)
    )
    mean_assets = np.mean(results_with_bootstrap.final_assets)
    ci_width_pct = (mean_assets_ci[1] - mean_assets_ci[0]) / mean_assets * 100

    print(f"\nCI width for mean final assets: {ci_width_pct:.2f}% of estimate")

# %% [markdown]
# ### Compare Insurance Strategies Using Bootstrap

# %%
# Compare two insurance strategies
print("\nComparing Insurance Strategies")
print("=" * 50)

# Strategy A: Lower limits
strategy_a_config = MonteCarloConfig(
    n_simulations=5_000, n_years=10, parallel=True, seed=42, progress_bar=False
)

# Create insurance program with lower limits
layers_a = [
    EnhancedInsuranceLayer(0, 2_500_000, 0.012),  # Lower primary limit
    EnhancedInsuranceLayer(2_500_000, 10_000_000, 0.006),
]
insurance_a = InsuranceProgram(layers_a)

# Run simulation for Strategy A
engine_a = MonteCarloEngine(
    loss_generator=loss_generator,
    insurance_program=insurance_a,
    manufacturer=manufacturer,
    config=strategy_a_config,
)
results_a = engine_a.run()

# Strategy B: Higher limits (original)
results_b = results_with_bootstrap  # Use previous results

print(f"Strategy A - Mean Final Assets: {format_currency(np.mean(results_a.final_assets))}")
print(f"Strategy B - Mean Final Assets: {format_currency(np.mean(results_b.final_assets))}")

# %%
# Statistical test for difference in strategies

# Test if Strategy B (higher limits) produces higher final assets
test_result = difference_in_means_test(
    results_a.final_assets[:5000],  # Use first 5000 from each
    results_b.final_assets[:5000],
    alternative="less",  # Test if A < B
    n_bootstrap=5000,
    seed=42,
)

print("\nStatistical Test: Strategy A < Strategy B")
print("=" * 50)
print(f"Difference in means: {format_currency(test_result.test_statistic)}")
print(
    f"95% CI for difference: [{format_currency(test_result.confidence_interval[0])}, "
    f"{format_currency(test_result.confidence_interval[1])}]"
)
print(f"P-value: {test_result.p_value:.4f}")

if test_result.reject_null:
    print("\nConclusion: Strategy B (higher limits) significantly outperforms Strategy A")
else:
    print("\nConclusion: No significant difference between strategies")

# %% [markdown]
# ### Bootstrap Distribution Visualization

# %%
# Visualize bootstrap distribution for mean growth rate
analyzer = BootstrapAnalyzer(n_bootstrap=5000, seed=42, show_progress=True)

# Compute bootstrap distribution for mean growth rate
result = analyzer.confidence_interval(
    results_with_bootstrap.growth_rates, np.mean, method="percentile"
)

# Create visualization
fig = go.Figure()

# Add histogram of bootstrap distribution
fig.add_trace(
    go.Histogram(
        x=result.bootstrap_distribution,
        nbinsx=50,
        name="Bootstrap Distribution",
        marker_color=WSJ_COLORS["blue"],
        opacity=0.7,
    )
)

# Add vertical lines for confidence interval
fig.add_vline(
    x=result.confidence_interval[0],
    line_dash="dash",
    line_color=WSJ_COLORS["red"],
    annotation_text="Lower CI",
)
fig.add_vline(
    x=result.confidence_interval[1],
    line_dash="dash",
    line_color=WSJ_COLORS["red"],
    annotation_text="Upper CI",
)
fig.add_vline(
    x=result.statistic, line_color=WSJ_COLORS["dark_gray"], annotation_text="Observed Mean"
)

fig.update_layout(
    title="Bootstrap Distribution of Mean Growth Rate",
    xaxis_title="Mean Growth Rate",
    yaxis_title="Frequency",
    showlegend=True,
    height=400,
)

fig.show()

print(f"\nMean Growth Rate: {result.statistic:.6f}")
print(f"95% CI: [{result.confidence_interval[0]:.6f}, {result.confidence_interval[1]:.6f}]")
print(f"Bootstrap Std Error: {np.std(result.bootstrap_distribution):.6f}")

# %% [markdown]
# ### Testing for Ruin Probability Significance

# %%
# Test if ruin probability is significantly different from a threshold
from ergodic_insurance.statistical_tests import paired_comparison_test

# Create ruin indicator (1 if ruined, 0 otherwise)
ruin_indicator = (results_with_bootstrap.final_assets <= 0).astype(float)

# Test if ruin probability is significantly less than 1% threshold
ruin_prob_test = paired_comparison_test(
    ruin_indicator - 0.01,  # Difference from 1% threshold
    null_value=0.0,
    alternative="less",  # Test if actual < threshold
    n_bootstrap=5000,
    seed=42,
)

print("\nTesting Ruin Probability < 1% Threshold")
print("=" * 50)
print(f"Observed Ruin Probability: {np.mean(ruin_indicator):.3%}")
print(f"Difference from 1%: {np.mean(ruin_indicator) - 0.01:.3%}")
print(f"P-value: {ruin_prob_test.p_value:.4f}")

if ruin_prob_test.reject_null:
    print("\nConclusion: Ruin probability is significantly below 1% threshold")
else:
    print("\nConclusion: Cannot confirm ruin probability is below 1% threshold")

# %% [markdown]
# ### BCa Bootstrap for Skewed Distributions

# %%
# Use BCa method for potentially skewed metrics like VaR
print("\nComparing Bootstrap Methods for VaR(99%)")
print("=" * 50)

# Percentile method
_, ci_percentile = bootstrap_confidence_interval(
    results_with_bootstrap.final_assets,
    lambda x: np.percentile(x, 99),
    method="percentile",
    n_bootstrap=5000,
    seed=42,
)

# BCa method
_, ci_bca = bootstrap_confidence_interval(
    results_with_bootstrap.final_assets,
    lambda x: np.percentile(x, 99),
    method="bca",
    n_bootstrap=5000,
    seed=42,
)

print(
    f"Percentile Method CI: [{format_currency(ci_percentile[0])}, {format_currency(ci_percentile[1])}]"
)
print(f"BCa Method CI:        [{format_currency(ci_bca[0])}, {format_currency(ci_bca[1])}]")
print(f"\nCI Width (Percentile): {format_currency(ci_percentile[1] - ci_percentile[0])}")
print(f"CI Width (BCa):        {format_currency(ci_bca[1] - ci_bca[0])}")

# %% [markdown]
# ### Summary of Bootstrap Analysis
#
# The bootstrap confidence intervals provide valuable insights into:
#
# 1. **Statistical Uncertainty**: The confidence intervals quantify the uncertainty in our estimates due to simulation sampling
#
# 2. **Strategy Comparison**: We can statistically test whether differences between strategies are significant
#
# 3. **Risk Metrics**: Bootstrap CIs for VaR and other risk metrics help understand their reliability
#
# 4. **Convergence**: Narrow confidence intervals indicate good convergence of the simulation
#
# 5. **Method Selection**: BCa method provides better coverage for skewed distributions
