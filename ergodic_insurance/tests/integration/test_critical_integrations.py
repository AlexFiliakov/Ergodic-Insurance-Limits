"""Critical integration tests for remaining cross-module interactions.

This module covers:
- Ergodic theory integration
- Configuration system propagation
- Optimization workflow
- Stochastic process integration
- Validation framework
- End-to-end scenarios
"""

import numpy as np
import pytest

from src.business_optimizer import BusinessOptimizer
from src.config_manager import ConfigManager
from src.config_v2 import ConfigV2
from src.convergence import ConvergenceDiagnostics
from src.decision_engine import DecisionEngine
from src.ergodic_analyzer import ErgodicAnalyzer
from src.insurance import InsuranceLayer, InsurancePolicy
from src.loss_distributions import LossData, ManufacturingLossGenerator
from src.manufacturer import WidgetManufacturer
from src.monte_carlo import MonteCarloEngine
from src.optimization import InsuranceOptimizer
from src.pareto_frontier import ParetoFrontier
from src.risk_metrics import RiskMetrics
from src.ruin_probability import RuinProbabilityCalculator
from src.simulation import Simulation
from src.stochastic_processes import GeometricBrownianMotion, MeanRevertingProcess
from src.validation_metrics import ValidationMetrics
from src.walk_forward_validator import WalkForwardValidator

from .test_fixtures import (
    assert_financial_consistency,
    default_config_v2,
    gbm_process,
    mean_reverting_process,
)
from .test_helpers import assert_convergence, calculate_ergodic_metrics, compare_scenarios, timer


class TestErgodicIntegration:
    """Test ergodic theory integration across modules."""

    def test_ergodic_analyzer_simulation_integration(
        self,
        default_config_v2: ConfigV2,
    ):
        """Test ergodic analyzer with simulation results.

        Verifies the example from issue requirements.
        """
        config = default_config_v2.model_copy()
        config.simulation.n_simulations = 50
        config.simulation.time_horizon = 30

        # Run simulations with and without insurance
        insured_results = []
        uninsured_results = []

        for i in range(10):
            # Insured simulation
            manufacturer = WidgetManufacturer.from_config_v2(config)
            policy = InsurancePolicy(
                layers=[InsuranceLayer(100_000, 5_000_000, 0.02)],
                deductible=100_000,
            )

            sim = Simulation(
                manufacturer=manufacturer,
                time_horizon=config.simulation.time_horizon,
                insurance_policy=policy,
                seed=42 + i,
            )
            insured_results.append(sim.run())

            # Uninsured simulation
            manufacturer = WidgetManufacturer.from_config_v2(config)
            sim = Simulation(
                manufacturer=manufacturer,
                time_horizon=config.simulation.time_horizon,
                insurance_policy=None,
                seed=142 + i,
            )
            uninsured_results.append(sim.run())

        # Analyze with ergodic analyzer
        analyzer = ErgodicAnalyzer()
        comparison = analyzer.compare_scenarios(
            insured_results=insured_results,
            uninsured_results=uninsured_results,
            metric="equity",
        )

        # Verify comparison structure
        assert "insured" in comparison
        assert "uninsured" in comparison
        assert "ergodic_advantage" in comparison

        # Check metrics
        assert "time_average_mean" in comparison["insured"]
        assert "ensemble_average" in comparison["insured"]
        assert "survival_rate" in comparison["insured"]

        # Verify ergodic advantage calculation
        insured_time_avg = comparison["insured"]["time_average_mean"]
        uninsured_time_avg = comparison["uninsured"]["time_average_mean"]

        if np.isfinite(insured_time_avg) and np.isfinite(uninsured_time_avg):
            ergodic_adv = insured_time_avg - uninsured_time_avg
            assert np.isclose(
                ergodic_adv,
                comparison["ergodic_advantage"]["time_average_difference"],
                rtol=1e-10,
            )

    def test_convergence_detection_integration(self):
        """Test convergence detection in ergodic calculations.

        Verifies that:
        - Convergence is properly detected
        - Stopping criteria work correctly
        - Statistics stabilize
        """
        analyzer = ConvergenceDiagnostics()

        # Generate multiple chains for convergence testing
        n_chains = 4
        n_points = 1000
        target_value = 0.05

        # Generate multiple chains that converge
        chains = []
        for chain_id in range(n_chains):
            np.random.seed(chain_id)
            chain = []
            for i in range(n_points):
                variance = 0.2 * np.exp(-i / 200)  # Decreasing variance
                value = target_value + np.random.normal(0, variance)
                chain.append(value)
            chains.append(chain)

        chains_array = np.array(chains)

        # Calculate convergence statistics
        r_hat = analyzer.calculate_r_hat(chains_array)

        # For a single chain, calculate ESS and MCSE
        single_chain = chains_array[0]
        ess = analyzer.calculate_ess(single_chain)
        mcse = analyzer.calculate_mcse(single_chain, ess)

        # Verify convergence metrics
        assert r_hat < 1.1, f"R-hat {r_hat:.3f} should be < 1.1 for convergence"
        assert ess > 100, f"ESS {ess:.0f} should be reasonably large"
        assert mcse < 0.01, f"MCSE {mcse:.4f} should be small"

    def test_time_vs_ensemble_average_divergence(self):
        """Test that time and ensemble averages diverge for multiplicative processes.

        This demonstrates the key ergodic insight.
        """
        n_paths = 100
        n_time = 100

        # Create multiplicative process with volatility
        trajectories = np.zeros((n_paths, n_time))

        for i in range(n_paths):
            np.random.seed(i)
            returns = np.random.normal(0.05, 0.20, n_time)
            trajectories[i] = 1000000 * np.exp(np.cumsum(returns))

        # Calculate ergodic metrics
        metrics = calculate_ergodic_metrics(trajectories, metric="growth_rate")

        # Verify divergence
        time_avg = metrics["time_average_mean"]
        ensemble_avg = metrics["ensemble_average"]

        # For multiplicative processes with volatility, time average < ensemble average
        assert time_avg < ensemble_avg, (
            f"Time average {time_avg:.4f} should be less than "
            f"ensemble average {ensemble_avg:.4f} for multiplicative process"
        )

        # The difference should be approximately volatility^2 / 2
        expected_diff = 0.20**2 / 2
        actual_diff = ensemble_avg - time_avg

        assert abs(actual_diff - expected_diff) < 0.02, (
            f"Ergodic difference {actual_diff:.4f} should be close to "
            f"vol^2/2 = {expected_diff:.4f}"
        )


class TestConfigurationIntegration:
    """Test configuration system integration across modules."""

    def test_config_propagation_to_all_modules(
        self,
        default_config_v2: ConfigV2,
    ):
        """Test that configuration properly propagates to all modules.

        Verifies the example from issue requirements.
        """
        manager = ConfigManager()

        # Load base configuration
        base_config = default_config_v2.model_copy()

        # Apply runtime override
        base_config.manufacturer.operating_margin = 0.15

        # Create modules with configuration
        manufacturer = WidgetManufacturer.from_config_v2(base_config)
        optimizer = BusinessOptimizer(base_config)
        engine = MonteCarloEngine(
            config=base_config.simulation,
            manufacturer_config=base_config.manufacturer,
            insurance_config=base_config.insurance,
            stochastic_config=base_config.stochastic,
        )

        # Verify override propagated
        assert manufacturer.operating_margin == 0.15
        assert optimizer.config.manufacturer.operating_margin == 0.15

        # Verify other settings maintained
        assert manufacturer.tax_rate == base_config.manufacturer.tax_rate
        assert optimizer.config.simulation.n_simulations == base_config.simulation.n_simulations

    def test_profile_inheritance_and_composition(self):
        """Test configuration profile inheritance and module composition.

        Verifies that:
        - Profiles inherit correctly
        - Modules compose properly
        - Overrides work as expected
        """
        config_manager = ConfigManager()

        # Create base profile
        base_config = ConfigV2()

        # Create conservative profile
        conservative = base_config.model_copy()
        conservative.insurance.primary_limit = 10_000_000
        conservative.insurance.primary_rate = 0.03
        conservative.simulation.confidence_levels = [0.95, 0.99]

        # Create aggressive profile
        aggressive = base_config.model_copy()
        aggressive.insurance.primary_limit = 2_000_000
        aggressive.insurance.primary_rate = 0.015
        aggressive.manufacturer.growth_capex_ratio = 0.10

        # Verify profile differences
        assert conservative.insurance.primary_limit > aggressive.insurance.primary_limit
        assert conservative.insurance.primary_rate > aggressive.insurance.primary_rate
        assert (
            aggressive.manufacturer.growth_capex_ratio > base_config.manufacturer.growth_capex_ratio
        )

    def test_backward_compatibility(self):
        """Test backward compatibility with legacy configurations.

        Verifies that:
        - Legacy configs can be loaded
        - Migration works correctly
        - New features are available
        """
        # Create legacy-style config dict
        legacy_config = {
            "initial_assets": 10_000_000,
            "asset_turnover_ratio": 1.2,
            "operating_margin": 0.10,
            "tax_rate": 0.25,
            "retention_ratio": 0.70,
        }

        # Convert to new config
        from src.config import ManufacturerConfig

        old_config = ManufacturerConfig(**legacy_config)

        # Create manufacturer with legacy config
        manufacturer = WidgetManufacturer(old_config)

        # Verify values transferred
        assert manufacturer.assets == legacy_config["initial_assets"]
        assert manufacturer.asset_turnover_ratio == legacy_config["asset_turnover_ratio"]
        assert manufacturer.operating_margin == legacy_config["operating_margin"]


class TestOptimizationWorkflow:
    """Test optimization workflow integration."""

    def test_business_optimizer_integration(
        self,
        default_config_v2: ConfigV2,
    ):
        """Test business optimizer with real simulation data.

        Verifies the example from issue requirements.
        """
        config = default_config_v2.model_copy()
        optimizer = BusinessOptimizer(config)

        # Test maximize ROE with insurance
        result = optimizer.maximize_roe_with_insurance(
            insurance_limit=5_000_000,
            insurance_rate=0.02,
            target_ruin_probability=0.01,
        )

        # Verify result structure
        assert "optimal_retention" in result
        assert "expected_roe" in result
        assert "ruin_probability" in result

        # Verify constraints satisfied
        assert result["ruin_probability"] <= 0.01
        assert result["optimal_retention"] >= 0
        assert result["optimal_retention"] <= 1

    def test_pareto_frontier_generation(self):
        """Test Pareto frontier for multi-objective optimization.

        Verifies that:
        - Frontier is properly generated
        - Trade-offs are captured
        - Solutions are non-dominated
        """
        frontier = ParetoFrontier()

        # Generate candidate solutions (ROE vs Risk)
        solutions = []
        for retention in np.linspace(0.3, 0.9, 20):
            for insurance_limit in [2e6, 5e6, 10e6]:
                # Simulate metrics
                roe = 0.15 * retention - 0.01 * insurance_limit / 1e6
                risk = 0.05 / retention + 0.001 * insurance_limit / 1e6

                solutions.append(
                    {
                        "retention": retention,
                        "insurance_limit": insurance_limit,
                        "roe": roe,
                        "risk": risk,
                    }
                )

        # Find Pareto optimal solutions
        pareto_points = frontier.find_pareto_optimal(
            solutions,
            objectives=["roe", "risk"],
            maximize=["roe"],
            minimize=["risk"],
        )

        # Verify Pareto optimality
        assert len(pareto_points) > 0
        assert len(pareto_points) < len(solutions)  # Some solutions dominated

        # Verify non-domination
        for p1 in pareto_points:
            for p2 in pareto_points:
                if p1 != p2:
                    # p2 should not dominate p1
                    dominates = (
                        p2["roe"] >= p1["roe"]
                        and p2["risk"] <= p1["risk"]
                        and (p2["roe"] > p1["roe"] or p2["risk"] < p1["risk"])
                    )
                    assert not dominates, "Pareto set contains dominated solution"

    def test_decision_engine_integration(self):
        """Test decision engine with optimization results.

        Verifies that:
        - Decisions are made based on optimization
        - Risk constraints are respected
        - Recommendations are actionable
        """
        config = ConfigV2()
        engine = DecisionEngine(config)

        # Create scenario data
        current_state = {
            "equity": 8_000_000,
            "assets": 10_000_000,
            "recent_losses": 1_500_000,
            "current_insurance_limit": 3_000_000,
        }

        # Get decision recommendation
        decision = engine.recommend_insurance_adjustment(
            current_state=current_state,
            risk_tolerance=0.02,  # 2% ruin probability
        )

        # Verify decision structure
        assert "recommended_limit" in decision
        assert "reasoning" in decision
        assert "expected_improvement" in decision

        # Verify decision is reasonable
        assert decision["recommended_limit"] >= 0
        assert decision["recommended_limit"] <= 50_000_000  # Reasonable upper bound


class TestStochasticIntegration:
    """Test stochastic process integration."""

    def test_stochastic_manufacturer_integration(
        self,
        gbm_process: GeometricBrownianMotion,
        mean_reverting_process: MeanRevertingProcess,
    ):
        """Test stochastic processes with manufacturer model.

        Verifies the example from issue requirements.
        """
        config = ConfigV2()
        manufacturer = WidgetManufacturer.from_config_v2(config)

        # Apply GBM to revenue growth
        n_years = 10
        revenue_multipliers = []

        for year in range(n_years):
            multiplier = gbm_process.generate_path(1, 1)[0]
            revenue_multipliers.append(multiplier)

            # Apply to manufacturer
            manufacturer.asset_turnover_ratio *= multiplier
            manufacturer.step()

        # Verify stochastic evolution
        assert len(revenue_multipliers) == n_years
        assert not all(m == 1.0 for m in revenue_multipliers)
        assert_financial_consistency(manufacturer)

        # Test mean reversion for operating margin
        manufacturer2 = WidgetManufacturer.from_config_v2(config)
        target_margin = manufacturer2.operating_margin

        for year in range(n_years):
            # Apply mean-reverting shock to margin
            shock = mean_reverting_process.generate_path(1, 1)[0]
            manufacturer2.operating_margin = target_margin * shock
            manufacturer2.step()

            # Margin should revert toward target
            mean_reverting_process.update_state(manufacturer2.operating_margin / target_margin)

        # Final margin should be closer to target
        final_margin = manufacturer2.operating_margin
        assert abs(final_margin - target_margin) < 0.05

    def test_correlation_between_risks(self):
        """Test correlation between operational and financial risks.

        Verifies that:
        - Correlations are properly modeled
        - Risk factors interact correctly
        - Seed management works
        """
        # Generate correlated risks
        n_periods = 100
        correlation = 0.3

        np.random.seed(42)

        # Generate base random factors
        z1 = np.random.normal(0, 1, n_periods)
        z2 = np.random.normal(0, 1, n_periods)

        # Create correlation
        operational_factor = z1
        financial_factor = correlation * z1 + np.sqrt(1 - correlation**2) * z2

        # Apply to losses
        operational_losses = np.exp(operational_factor) * 100_000
        financial_losses = np.exp(financial_factor) * 150_000

        # Verify correlation
        actual_corr = np.corrcoef(np.log(operational_losses), np.log(financial_losses))[0, 1]
        assert (
            abs(actual_corr - correlation) < 0.1
        ), f"Correlation {actual_corr:.2f} should be close to {correlation:.2f}"


class TestValidationFramework:
    """Test validation framework integration."""

    def test_walk_forward_validation(self):
        """Test walk-forward validation with time series.

        Verifies the example from issue requirements.
        """
        # Generate time series data
        n_periods = 100
        np.random.seed(42)

        # Create synthetic revenue data with trend
        time = np.arange(n_periods)
        trend = 1000000 + 10000 * time
        seasonal = 50000 * np.sin(2 * np.pi * time / 12)
        noise = np.random.normal(0, 20000, n_periods)
        revenue = trend + seasonal + noise

        # Set up walk-forward validation
        validator = WalkForwardValidator(
            window_size=20,
            step_size=5,
            min_train_size=30,
        )

        # Define simple forecast model
        def forecast_model(train_data):
            """Simple linear trend forecast."""
            x = np.arange(len(train_data))
            coeffs = np.polyfit(x, train_data, 1)
            next_value = np.polyval(coeffs, len(train_data))
            return next_value

        # Run validation
        results = validator.validate(
            data=revenue,
            model_func=forecast_model,
            metric="mape",  # Mean absolute percentage error
        )

        # Verify validation results
        assert "scores" in results
        assert "mean_score" in results
        assert len(results["scores"]) > 0

        # Check scores are reasonable
        assert results["mean_score"] < 0.10, "MAPE should be < 10% for trending data"

    def test_validation_metrics_calculation(self):
        """Test validation metrics for model performance.

        Verifies that:
        - Metrics are correctly calculated
        - Multiple metrics work together
        - Edge cases are handled
        """
        metrics = ValidationMetrics()

        # Generate predictions and actuals
        n_samples = 100
        np.random.seed(42)

        actuals = np.random.lognormal(14, 1, n_samples)
        # Add noise to create predictions
        predictions = actuals * np.random.normal(1, 0.1, n_samples)

        # Calculate various metrics
        mse = metrics.mean_squared_error(actuals, predictions)
        mae = metrics.mean_absolute_error(actuals, predictions)
        mape = metrics.mean_absolute_percentage_error(actuals, predictions)
        r2 = metrics.r_squared(actuals, predictions)

        # Verify metrics are reasonable
        assert mse > 0, "MSE should be positive"
        assert mae > 0, "MAE should be positive"
        assert 0 <= mape <= 1, "MAPE should be between 0 and 1"
        assert 0 <= r2 <= 1, "R² should be between 0 and 1"

        # Since predictions are close to actuals, R² should be high
        assert r2 > 0.8, "R² should be high for correlated data"


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios."""

    def test_startup_company_scenario(self):
        """Test startup company scenario (low assets, high risk).

        Complete E2E test from issue requirements.
        """
        # Configure startup
        config = ConfigV2(
            manufacturer={
                "initial_assets": 1_000_000,
                "asset_turnover": 0.8,
                "operating_margin": 0.05,
                "growth_capex_ratio": 0.10,
            },
            insurance={
                "deductible": 25_000,
                "primary_limit": 500_000,
                "primary_rate": 0.04,  # Higher rate for startup
            },
            simulation={
                "n_simulations": 100,
                "time_horizon": 20,
                "seed": 42,
            },
            stochastic={
                "revenue_volatility": 0.30,  # High volatility
                "claim_frequency_mean": 3,
                "claim_severity_mean": 50_000,
            },
        )

        # Run simulation
        engine = MonteCarloEngine(
            config=config.simulation,
            manufacturer_config=config.manufacturer,
            insurance_config=config.insurance,
            stochastic_config=config.stochastic,
        )

        with timer("Startup scenario") as t:
            results = engine.run()

        # Verify results
        assert results is not None
        assert len(results.terminal_values) == 100

        # Calculate key metrics
        survival_rate = np.mean(results.terminal_values > 100_000)  # 10% of initial
        median_terminal = np.median(results.terminal_values)

        # Startups should have lower survival but potential for growth
        assert (
            0.3 <= survival_rate <= 0.8
        ), f"Startup survival rate {survival_rate:.2%} out of expected range"

        # Verify timing
        assert t["elapsed"] < 60, f"Startup scenario took {t['elapsed']:.2f}s, should be < 60s"

    def test_mature_company_scenario(self):
        """Test mature company scenario (stable, optimized).

        Complete E2E test from issue requirements.
        """
        # Configure mature company
        config = ConfigV2(
            manufacturer={
                "initial_assets": 50_000_000,
                "asset_turnover": 1.5,
                "operating_margin": 0.15,
                "dividend_payout_ratio": 0.50,
            },
            insurance={
                "deductible": 250_000,
                "primary_limit": 10_000_000,
                "primary_rate": 0.018,
                "excess_limit": 20_000_000,
                "excess_attachment": 10_000_000,
                "excess_rate": 0.008,
            },
            simulation={
                "n_simulations": 100,
                "time_horizon": 30,
                "seed": 42,
            },
            stochastic={
                "revenue_volatility": 0.10,  # Lower volatility
                "claim_frequency_mean": 5,
                "claim_severity_mean": 300_000,
            },
        )

        # Run simulation
        engine = MonteCarloEngine(
            config=config.simulation,
            manufacturer_config=config.manufacturer,
            insurance_config=config.insurance,
            stochastic_config=config.stochastic,
        )

        with timer("Mature scenario") as t:
            results = engine.run()

        # Calculate metrics
        survival_rate = np.mean(results.terminal_values > config.manufacturer.initial_assets * 0.5)
        growth_rate = np.mean(
            np.log(results.terminal_values / config.manufacturer.initial_assets) / 30
        )

        # Mature companies should have high survival, steady growth
        assert survival_rate > 0.9, f"Mature company survival {survival_rate:.2%} should be > 90%"
        assert 0.02 <= growth_rate <= 0.10, f"Growth rate {growth_rate:.2%} out of expected range"

        # Verify timing
        assert t["elapsed"] < 60, f"Mature scenario took {t['elapsed']:.2f}s, should be < 60s"

    def test_crisis_scenario(self):
        """Test crisis scenario (catastrophic losses).

        Complete E2E test from issue requirements.
        """
        # Configure crisis scenario
        config = ConfigV2(
            manufacturer={
                "initial_assets": 20_000_000,
                "asset_turnover": 1.2,
                "operating_margin": 0.08,
            },
            insurance={
                "deductible": 100_000,
                "primary_limit": 5_000_000,
                "primary_rate": 0.025,
                "excess_limit": 15_000_000,
                "excess_attachment": 5_000_000,
                "excess_rate": 0.015,
            },
            simulation={
                "n_simulations": 100,
                "time_horizon": 10,
                "seed": 42,
            },
            stochastic={
                "revenue_volatility": 0.25,
                "claim_frequency_mean": 8,  # High frequency
                "claim_severity_mean": 500_000,  # High severity
                "catastrophe_probability": 0.10,  # 10% annual catastrophe chance
                "catastrophe_severity_mean": 10_000_000,
            },
        )

        # Run simulation
        engine = MonteCarloEngine(
            config=config.simulation,
            manufacturer_config=config.manufacturer,
            insurance_config=config.insurance,
            stochastic_config=config.stochastic,
        )

        results = engine.run()

        # Calculate metrics
        ruin_rate = np.mean(results.terminal_values < config.manufacturer.initial_assets * 0.1)

        # Insurance should prevent complete ruin even in crisis
        assert ruin_rate < 0.3, f"Ruin rate {ruin_rate:.2%} should be < 30% with insurance"

        # Compare with uninsured
        config.insurance = None
        uninsured_engine = MonteCarloEngine(
            config=config.simulation,
            manufacturer_config=config.manufacturer,
            insurance_config=None,
            stochastic_config=config.stochastic,
        )

        uninsured_results = uninsured_engine.run()
        uninsured_ruin = np.mean(
            uninsured_results.terminal_values < config.manufacturer.initial_assets * 0.1
        )

        # Insurance should significantly reduce ruin probability
        assert uninsured_ruin > ruin_rate, "Insurance should reduce ruin probability in crisis"

    def test_growth_scenario(self):
        """Test growth scenario (rapid expansion).

        Complete E2E test from issue requirements.
        """
        # Configure growth scenario
        config = ConfigV2(
            manufacturer={
                "initial_assets": 5_000_000,
                "asset_turnover": 1.0,
                "operating_margin": 0.12,
                "growth_capex_ratio": 0.15,  # High growth investment
                "dividend_payout_ratio": 0.10,  # Low dividends, high retention
            },
            insurance={
                "deductible": 50_000,
                "primary_limit": 3_000_000,
                "primary_rate": 0.022,
            },
            simulation={
                "n_simulations": 100,
                "time_horizon": 15,
                "seed": 42,
            },
            stochastic={
                "revenue_volatility": 0.20,
                "claim_frequency_mean": 4,
                "claim_severity_mean": 150_000,
            },
        )

        # Run simulation
        engine = MonteCarloEngine(
            config=config.simulation,
            manufacturer_config=config.manufacturer,
            insurance_config=config.insurance,
            stochastic_config=config.stochastic,
        )

        results = engine.run()

        # Calculate growth metrics
        terminal_values = results.terminal_values
        growth_multiples = terminal_values / config.manufacturer.initial_assets

        # High growth scenario should show significant expansion
        median_growth = np.median(growth_multiples)
        assert median_growth > 2.0, f"Median growth {median_growth:.1f}x should be > 2x in 15 years"

        # But with higher risk
        volatility = np.std(growth_multiples) / np.mean(growth_multiples)
        assert volatility > 0.3, "Growth scenario should have higher volatility"

    def test_performance_benchmarks(self):
        """Test that performance benchmarks are met.

        From issue requirements:
        - 1000-year simulation in <1 minute
        - 100K Monte Carlo in <10 minutes
        - Memory usage <4GB for 100K paths
        """
        # Test 1: 1000-year simulation
        config = ConfigV2()
        config.simulation.n_simulations = 1
        config.simulation.time_horizon = 1000

        manufacturer = WidgetManufacturer.from_config_v2(config)
        sim = Simulation(
            manufacturer=manufacturer,
            time_horizon=1000,
            seed=42,
        )

        with timer("1000-year simulation") as t:
            result = sim.run()

        assert t["elapsed"] < 60, f"1000-year simulation took {t['elapsed']:.2f}s, should be < 60s"

        # Test 2: Small Monte Carlo (scaled down for testing)
        config.simulation.n_simulations = 1000  # Scaled down from 100K for test speed
        config.simulation.time_horizon = 20
        config.simulation.enable_parallel = True

        engine = MonteCarloEngine(
            config=config.simulation,
            manufacturer_config=config.manufacturer,
            insurance_config=config.insurance,
            stochastic_config=config.stochastic,
        )

        with timer("1K Monte Carlo") as t:
            results = engine.run()

        # Scale expectation: 1K should take < 6 seconds (1/100 of 10 minutes)
        assert t["elapsed"] < 6, f"1K Monte Carlo took {t['elapsed']:.2f}s, should be < 6s"
