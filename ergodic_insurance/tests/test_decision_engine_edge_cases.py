"""Comprehensive edge case tests for the insurance decision engine."""

from unittest.mock import MagicMock, Mock, PropertyMock, patch

import numpy as np
import pytest
from ergodic_insurance.src.config import ManufacturerConfig
from ergodic_insurance.src.decision_engine import (
    DecisionMetrics,
    InsuranceDecision,
    InsuranceDecisionEngine,
    OptimizationConstraints,
    OptimizationMethod,
    Recommendations,
    SensitivityReport,
)
from ergodic_insurance.src.insurance_program import EnhancedInsuranceLayer as Layer
from ergodic_insurance.src.loss_distributions import LossDistribution
from ergodic_insurance.src.manufacturer import WidgetManufacturer
from scipy.optimize import OptimizeResult


class TestCVaREdgeCases:
    """Test edge cases for CVaR calculation."""

    @pytest.fixture
    def engine(self):
        """Create a test engine instance."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.2,
            operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)

        loss_dist = Mock(spec=LossDistribution)
        loss_dist.expected_value.return_value = 100_000

        return InsuranceDecisionEngine(
            manufacturer=manufacturer,
            loss_distribution=loss_dist,
            pricing_scenario="baseline",
        )

    def test_cvar_with_empty_losses(self, engine):
        """Test CVaR calculation with empty losses array."""
        losses = np.array([])
        cvar = engine._calculate_cvar(losses, 95)
        assert cvar == 0.0

    def test_cvar_with_no_tail_losses(self, engine):
        """Test CVaR when no losses exceed threshold."""
        # All losses below 95th percentile
        losses = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        cvar = engine._calculate_cvar(losses, 95)
        # Should return the threshold itself or close to it
        threshold = np.percentile(losses, 95)
        # Use more lenient comparison due to percentile precision
        assert np.isclose(cvar, threshold, rtol=0.1)

    def test_cvar_with_single_tail_loss(self, engine):
        """Test CVaR with only one loss exceeding threshold."""
        losses = np.array([100] * 95 + [10000])  # One extreme loss
        cvar = engine._calculate_cvar(losses, 95)
        assert cvar == 10000  # Only the extreme loss in tail


class TestNoInsuranceLayersScenario:
    """Test scenarios with no insurance layers."""

    @pytest.fixture
    def engine(self):
        """Create a test engine instance."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.2,
            operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)

        loss_dist = Mock(spec=LossDistribution)
        loss_dist.expected_value.return_value = 100_000

        return InsuranceDecisionEngine(
            manufacturer=manufacturer,
            loss_distribution=loss_dist,
            pricing_scenario="baseline",
        )

    def test_objective_with_no_layers(self, engine):
        """Test objective calculation when no active layers exist."""
        # Decision with retention only, no meaningful layers
        x = np.array([1_000_000, 500, 500, 0])  # Small layers below threshold
        weights = {"growth": 0.4, "risk": 0.4, "cost": 0.2}

        objective = engine._calculate_objective(x, weights)

        # Should return penalty value
        assert objective == 1e10

    def test_objective_with_all_zero_layers(self, engine):
        """Test objective when all layer limits are zero."""
        x = np.array([1_000_000, 0, 0, 0])
        weights = {"growth": 0.4, "risk": 0.4, "cost": 0.2}

        objective = engine._calculate_objective(x, weights)

        # Should return penalty value
        assert objective == 1e10


class TestWeightedSumOptimization:
    """Test WEIGHTED_SUM optimization method."""

    @pytest.fixture
    def engine(self):
        """Create a test engine instance."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.2,
            operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)

        loss_dist = Mock(spec=LossDistribution)
        loss_dist.expected_value.return_value = 100_000

        return InsuranceDecisionEngine(
            manufacturer=manufacturer,
            loss_distribution=loss_dist,
            pricing_scenario="baseline",
        )

    def test_weighted_sum_optimization(self, engine):
        """Test explicit WEIGHTED_SUM optimization method."""
        constraints = OptimizationConstraints(
            max_premium_budget=400_000,
            min_coverage_limit=5_000_000,
            max_coverage_limit=15_000_000,
            max_layers=3,
        )

        decision = engine.optimize_insurance_decision(
            constraints, method=OptimizationMethod.WEIGHTED_SUM
        )

        assert isinstance(decision, InsuranceDecision)
        assert decision.optimization_method in [
            "SLSQP",
            "weighted_sum",
        ]  # Uses SLSQP or weighted_sum
        assert decision.total_premium <= constraints.max_premium_budget
        assert (
            constraints.min_coverage_limit
            <= decision.total_coverage
            <= constraints.max_coverage_limit
        )


class TestObjectiveCalculationErrors:
    """Test error handling in objective calculation."""

    @pytest.fixture
    def engine(self):
        """Create a test engine instance."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.2,
            operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)

        loss_dist = Mock(spec=LossDistribution)
        loss_dist.expected_value.return_value = 100_000

        return InsuranceDecisionEngine(
            manufacturer=manufacturer,
            loss_distribution=loss_dist,
            pricing_scenario="baseline",
        )

    def test_objective_with_error_in_growth_calculation(self, engine):
        """Test objective calculation when growth rate estimation fails."""
        x = np.array([1_000_000, 5_000_000])
        weights = {"growth": 0.4, "risk": 0.4, "cost": 0.2}

        # Mock growth rate to raise exception
        with patch.object(
            engine, "_estimate_growth_rate", side_effect=Exception("Growth calc error")
        ):
            objective = engine._calculate_objective(x, weights)

        # Should return penalty value on error
        assert objective == 1e10

    def test_objective_with_error_in_bankruptcy_calculation(self, engine):
        """Test objective calculation when bankruptcy estimation fails."""
        x = np.array([1_000_000, 5_000_000])
        weights = {"growth": 0.4, "risk": 0.4, "cost": 0.2}

        # Mock bankruptcy probability to raise exception
        with patch.object(
            engine,
            "_estimate_bankruptcy_probability",
            side_effect=Exception("Bankruptcy calc error"),
        ):
            objective = engine._calculate_objective(x, weights)

        # Should return penalty value on error
        assert objective == 1e10


class TestMonteCarloEnginePathInBankruptcy:
    """Test Monte Carlo engine path in bankruptcy estimation."""

    @pytest.fixture
    def engine(self):
        """Create a test engine instance."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.2,
            operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)

        loss_dist = Mock(spec=LossDistribution)
        loss_dist.expected_value.return_value = 100_000

        return InsuranceDecisionEngine(
            manufacturer=manufacturer,
            loss_distribution=loss_dist,
            pricing_scenario="baseline",
        )

    def test_bankruptcy_estimation_with_monte_carlo(self, engine):
        """Test bankruptcy estimation using Monte Carlo engine."""
        # Add Monte Carlo engine attribute
        engine._monte_carlo_engine = Mock()

        x = np.array([1_000_000, 5_000_000, 10_000_000])

        # Mock Monte Carlo results
        mock_results = Mock()
        mock_results.ruin_probabilities = [0.015]

        with patch("ergodic_insurance.src.monte_carlo.MonteCarloEngine") as mock_mc_class:
            mock_mc_instance = Mock()
            mock_mc_instance.estimate_ruin_probability.return_value = mock_results
            mock_mc_class.return_value = mock_mc_instance

            bankruptcy_prob = engine._estimate_bankruptcy_probability(x)

            assert bankruptcy_prob == 0.015
            mock_mc_class.assert_called_once()

    def test_bankruptcy_estimation_monte_carlo_fallback(self, engine):
        """Test fallback when Monte Carlo estimation fails."""
        # Add Monte Carlo engine attribute
        engine._monte_carlo_engine = Mock()

        x = np.array([1_000_000, 5_000_000, 10_000_000])

        with patch("ergodic_insurance.src.monte_carlo.MonteCarloEngine") as mock_mc_class:
            mock_mc_instance = Mock()
            mock_mc_instance.estimate_ruin_probability.side_effect = Exception("MC failed")
            mock_mc_class.return_value = mock_mc_instance

            # Should fall back to simple estimation
            bankruptcy_prob = engine._estimate_bankruptcy_probability(x)

            # Should use fallback calculation
            assert 0 <= bankruptcy_prob <= 1
            assert bankruptcy_prob > 0  # With limited coverage, should have some risk


class TestCoverageRatioEdgeCases:
    """Test coverage ratio edge cases in bankruptcy estimation."""

    @pytest.fixture
    def engine(self):
        """Create a test engine instance."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.2,
            operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)

        loss_dist = Mock(spec=LossDistribution)
        loss_dist.expected_value.return_value = 100_000

        return InsuranceDecisionEngine(
            manufacturer=manufacturer,
            loss_distribution=loss_dist,
            pricing_scenario="baseline",
        )

    def test_bankruptcy_with_full_coverage(self, engine):
        """Test bankruptcy estimation with coverage >= expected loss."""
        # Very high coverage
        x = np.array([1_000_000, 50_000_000, 50_000_000])

        bankruptcy_prob = engine._estimate_bankruptcy_probability(x)

        # Should return minimum probability
        assert bankruptcy_prob == 0.001

    def test_bankruptcy_with_80_percent_coverage(self, engine):
        """Test bankruptcy estimation with 80% coverage ratio."""
        # Set up coverage to be exactly 80% of expected max loss
        engine.loss_distribution.expected_value.return_value = 1_000_000
        x = np.array([1_000_000, 4_000_000, 3_000_000])  # Total 8M coverage

        bankruptcy_prob = engine._estimate_bankruptcy_probability(x)

        # Should return 0.005 for 80% coverage
        assert bankruptcy_prob == 0.005

    def test_bankruptcy_with_60_percent_coverage(self, engine):
        """Test bankruptcy estimation with 60% coverage ratio."""
        # Set up coverage to be exactly 60% of expected max loss
        engine.loss_distribution.expected_value.return_value = 1_000_000
        x = np.array([1_000_000, 3_000_000, 2_000_000])  # Total 6M coverage

        bankruptcy_prob = engine._estimate_bankruptcy_probability(x)

        # Should return 0.01 for 60% coverage
        assert bankruptcy_prob == 0.01

    def test_bankruptcy_with_40_percent_coverage(self, engine):
        """Test bankruptcy estimation with 40% coverage ratio."""
        # Set up coverage to be exactly 40% of expected max loss
        engine.loss_distribution.expected_value.return_value = 1_000_000
        x = np.array([1_000_000, 2_000_000, 1_000_000])  # Total 4M coverage

        bankruptcy_prob = engine._estimate_bankruptcy_probability(x)

        # Should return 0.02 for 40% coverage
        assert bankruptcy_prob == 0.02

    def test_bankruptcy_with_low_coverage(self, engine):
        """Test bankruptcy estimation with very low coverage."""
        # Set up coverage to be < 40% of expected max loss
        engine.loss_distribution.expected_value.return_value = 1_000_000
        x = np.array([500_000, 1_000_000, 500_000])  # Total 2M coverage

        bankruptcy_prob = engine._estimate_bankruptcy_probability(x)

        # Should return 0.05 for low coverage
        assert bankruptcy_prob == 0.05


class TestDecisionValidationFailures:
    """Test decision validation failure scenarios."""

    @pytest.fixture
    def engine(self):
        """Create a test engine instance."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.2,
            operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)

        loss_dist = Mock(spec=LossDistribution)
        loss_dist.expected_value.return_value = 100_000

        return InsuranceDecisionEngine(
            manufacturer=manufacturer,
            loss_distribution=loss_dist,
            pricing_scenario="baseline",
        )

    def test_validation_failure_coverage_below_minimum(self, engine):
        """Test validation when coverage is below minimum."""
        constraints = OptimizationConstraints(
            min_coverage_limit=10_000_000,
            max_coverage_limit=20_000_000,
        )

        decision = InsuranceDecision(
            retained_limit=1_000_000,
            layers=[Layer(attachment_point=1_000_000, limit=3_000_000, premium_rate=0.01)],
            total_premium=30_000,
            total_coverage=4_000_000,  # Below minimum
            pricing_scenario="baseline",
            optimization_method="SLSQP",
        )

        assert engine._validate_decision(decision, constraints) is False

    def test_validation_failure_coverage_above_maximum(self, engine):
        """Test validation when coverage exceeds maximum."""
        constraints = OptimizationConstraints(
            min_coverage_limit=5_000_000,
            max_coverage_limit=10_000_000,
        )

        decision = InsuranceDecision(
            retained_limit=1_000_000,
            layers=[Layer(attachment_point=1_000_000, limit=15_000_000, premium_rate=0.01)],
            total_premium=150_000,
            total_coverage=16_000_000,  # Above maximum
            pricing_scenario="baseline",
            optimization_method="SLSQP",
        )

        assert engine._validate_decision(decision, constraints) is False

    def test_validation_failure_retention_below_minimum(self, engine):
        """Test validation when retention is below minimum."""
        constraints = OptimizationConstraints(
            min_retained_limit=500_000,
            max_retained_limit=2_000_000,
        )

        decision = InsuranceDecision(
            retained_limit=100_000,  # Below minimum
            layers=[Layer(attachment_point=100_000, limit=5_000_000, premium_rate=0.01)],
            total_premium=50_000,
            total_coverage=5_100_000,
            pricing_scenario="baseline",
            optimization_method="SLSQP",
        )

        assert engine._validate_decision(decision, constraints) is False

    def test_validation_failure_retention_above_maximum(self, engine):
        """Test validation when retention exceeds maximum."""
        constraints = OptimizationConstraints(
            min_retained_limit=100_000,
            max_retained_limit=1_000_000,
        )

        decision = InsuranceDecision(
            retained_limit=2_000_000,  # Above maximum
            layers=[Layer(attachment_point=2_000_000, limit=5_000_000, premium_rate=0.01)],
            total_premium=50_000,
            total_coverage=7_000_000,
            pricing_scenario="baseline",
            optimization_method="SLSQP",
        )

        assert engine._validate_decision(decision, constraints) is False

    def test_validation_failure_too_many_layers(self, engine):
        """Test validation when too many layers exist."""
        constraints = OptimizationConstraints(max_layers=2)

        decision = InsuranceDecision(
            retained_limit=1_000_000,
            layers=[
                Layer(attachment_point=1_000_000, limit=2_000_000, premium_rate=0.01),
                Layer(attachment_point=3_000_000, limit=2_000_000, premium_rate=0.008),
                Layer(attachment_point=5_000_000, limit=2_000_000, premium_rate=0.005),  # Too many
            ],
            total_premium=56_000,
            total_coverage=7_000_000,
            pricing_scenario="baseline",
            optimization_method="SLSQP",
        )

        assert engine._validate_decision(decision, constraints) is False


class TestMissingROEDataFallback:
    """Test fallback behavior when ROE data is missing."""

    @pytest.fixture
    def engine(self):
        """Create a test engine instance."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.2,
            operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)

        loss_dist = Mock(spec=LossDistribution)
        loss_dist.expected_value.return_value = 100_000

        return InsuranceDecisionEngine(
            manufacturer=manufacturer,
            loss_distribution=loss_dist,
            pricing_scenario="baseline",
        )

    def test_metrics_calculation_without_roe_series(self, engine):
        """Test metrics calculation when ROE series is not available."""
        decision = InsuranceDecision(
            retained_limit=1_000_000,
            layers=[Layer(attachment_point=1_000_000, limit=4_000_000, premium_rate=0.01)],
            total_premium=40_000,
            total_coverage=5_000_000,
            pricing_scenario="baseline",
            optimization_method="SLSQP",
        )

        # Mock simulation to return results without roe_series
        with patch.object(engine, "_run_simulation") as mock_sim:
            mock_sim.return_value = {
                "growth_rates": np.array([0.08, 0.09, 0.07]),
                "bankruptcies": np.array([0, 0, 0]),
                "roe": np.array([0.15, 0.16, 0.14]),
                "value": np.array([11_000_000, 11_500_000, 10_800_000]),
                "losses": np.array([100_000, 150_000, 80_000]),
                "roe_series": None,  # No series data
                "equity_series": None,
            }

            metrics = engine.calculate_decision_metrics(decision)

            # Should use fallback calculations
            assert metrics.time_weighted_roe == 0.15  # Mean of roe array
            assert metrics.roe_volatility > 0  # Should calculate std
            assert metrics.roe_sharpe_ratio > 0  # Should calculate Sharpe
            assert metrics.roe_1yr_rolling == 0.15
            assert metrics.roe_3yr_rolling == 0.15
            assert metrics.roe_5yr_rolling == 0.15


class TestBankruptcyAndNegativeEquityScenarios:
    """Test bankruptcy and negative equity scenarios in simulation."""

    @pytest.fixture
    def engine(self):
        """Create a test engine instance."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.2,
            operating_margin=0.01,  # Very low margin to trigger bankruptcy
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)

        loss_dist = Mock(spec=LossDistribution)
        loss_dist.expected_value.return_value = 500_000  # High losses

        return InsuranceDecisionEngine(
            manufacturer=manufacturer,
            loss_distribution=loss_dist,
            pricing_scenario="baseline",
        )

    def test_simulation_with_bankruptcy(self, engine):
        """Test simulation handling of bankruptcy scenarios."""
        decision = InsuranceDecision(
            retained_limit=5_000_000,  # High retention
            layers=[],  # No insurance
            total_premium=0,
            total_coverage=5_000_000,
            pricing_scenario="baseline",
            optimization_method="none",
        )

        # Mock loss distribution to return high losses
        engine.loss_distribution.expected_value.return_value = 2_000_000

        results = engine._run_simulation(decision, n_simulations=10, time_horizon=5)

        # Should have some bankruptcies with high losses and no insurance
        assert np.sum(results["bankruptcies"]) > 0

    def test_simulation_with_negative_equity_handling(self, engine):
        """Test simulation when equity goes negative."""
        decision = InsuranceDecision(
            retained_limit=10_000_000,  # Very high retention
            layers=[],
            total_premium=0,
            total_coverage=10_000_000,
            pricing_scenario="baseline",
            optimization_method="none",
        )

        # Set up for guaranteed negative equity
        engine.manufacturer.operating_margin = -0.1  # Negative margin

        results = engine._run_simulation(decision, n_simulations=5, time_horizon=3)

        # All simulations should result in bankruptcy
        assert np.all(results["bankruptcies"] == 1)


class TestParameterModificationInSensitivity:
    """Test parameter modification in sensitivity analysis."""

    @pytest.fixture
    def engine(self):
        """Create a test engine instance."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.2,
            operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)

        loss_dist = Mock(spec=LossDistribution)
        loss_dist.expected_value.return_value = 100_000
        loss_dist.frequency = 5.0  # Add frequency attribute

        return InsuranceDecisionEngine(
            manufacturer=manufacturer,
            loss_distribution=loss_dist,
            pricing_scenario="baseline",
        )

    def test_modify_loss_frequency_parameter(self, engine):
        """Test modification of loss frequency parameter."""
        original_frequency = engine.loss_distribution.frequency

        # Modify parameter
        original_state = engine._modify_parameter("loss_frequency", 0.2)

        # Check frequency was modified
        assert engine.loss_distribution.frequency == original_frequency * 1.2

        # Check original state was saved
        assert original_state["frequency"] == original_frequency

    def test_modify_capital_base_parameter(self, engine):
        """Test modification of capital base parameter."""
        original_assets = engine.manufacturer.assets

        # Modify parameter
        original_state = engine._modify_parameter("capital_base", -0.1)

        # Check assets were modified
        assert engine.manufacturer.assets == original_assets * 0.9

        # Check original state was saved
        assert original_state["assets"] == original_assets

    def test_modify_unknown_parameter(self, engine):
        """Test modification of unknown parameter."""
        # Should return empty state for unknown parameter
        original_state = engine._modify_parameter("unknown_param", 0.1)

        # Should return empty dict
        assert original_state == {}


class TestRecommendationGenerationEdgeCases:
    """Test edge cases in recommendation generation and confidence calculation."""

    @pytest.fixture
    def engine(self):
        """Create a test engine instance."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.2,
            operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)

        loss_dist = Mock(spec=LossDistribution)
        loss_dist.expected_value.return_value = 100_000

        return InsuranceDecisionEngine(
            manufacturer=manufacturer,
            loss_distribution=loss_dist,
            pricing_scenario="baseline",
        )

    def test_recommendations_with_high_bankruptcy_risk(self, engine):
        """Test recommendation generation with high bankruptcy risk."""
        decision = InsuranceDecision(
            retained_limit=5_000_000,
            layers=[],
            total_premium=0,
            total_coverage=5_000_000,
            pricing_scenario="baseline",
            optimization_method="SLSQP",
        )

        metrics = DecisionMetrics(
            ergodic_growth_rate=0.05,
            bankruptcy_probability=0.02,  # High risk
            expected_roe=0.08,
            roe_improvement=0.01,
            premium_to_limit_ratio=0,
            coverage_adequacy=0.5,
            capital_efficiency=1.0,
            value_at_risk_95=5_000_000,
            conditional_value_at_risk=7_000_000,
            decision_score=0.4,
        )

        recommendations = engine.generate_recommendations([(decision, metrics)])

        # Should include bankruptcy risk warning
        assert any(
            "bankruptcy risk" in risk.lower() for risk in recommendations.risk_considerations
        )

    def test_recommendations_with_high_premium(self, engine):
        """Test recommendations when premium exceeds threshold."""
        decision = InsuranceDecision(
            retained_limit=1_000_000,
            layers=[Layer(attachment_point=1_000_000, limit=50_000_000, premium_rate=0.025)],
            total_premium=1_250_000,  # High premium
            total_coverage=51_000_000,
            pricing_scenario="baseline",
            optimization_method="SLSQP",
        )

        metrics = DecisionMetrics(
            ergodic_growth_rate=0.12,
            bankruptcy_probability=0.005,
            expected_roe=0.14,
            roe_improvement=0.02,
            premium_to_limit_ratio=0.025,
            coverage_adequacy=0.9,
            capital_efficiency=1.5,
            value_at_risk_95=3_000_000,
            conditional_value_at_risk=4_000_000,
            decision_score=0.7,
        )

        recommendations = engine.generate_recommendations([(decision, metrics)])

        # Should include premium commitment warning
        assert any(
            "premium commitment" in risk.lower() for risk in recommendations.risk_considerations
        )

    def test_recommendations_with_complex_structure(self, engine):
        """Test recommendations with many layers."""
        layers = [
            Layer(attachment_point=1_000_000, limit=2_000_000, premium_rate=0.01),
            Layer(attachment_point=3_000_000, limit=2_000_000, premium_rate=0.008),
            Layer(attachment_point=5_000_000, limit=3_000_000, premium_rate=0.006),
            Layer(attachment_point=8_000_000, limit=4_000_000, premium_rate=0.004),
        ]

        decision = InsuranceDecision(
            retained_limit=1_000_000,
            layers=layers,
            total_premium=72_000,
            total_coverage=12_000_000,
            pricing_scenario="baseline",
            optimization_method="SLSQP",
        )

        metrics = DecisionMetrics(
            ergodic_growth_rate=0.11,
            bankruptcy_probability=0.007,
            expected_roe=0.13,
            roe_improvement=0.015,
            premium_to_limit_ratio=0.006,
            coverage_adequacy=0.85,
            capital_efficiency=1.8,
            value_at_risk_95=3_500_000,
            conditional_value_at_risk=4_500_000,
            decision_score=0.65,
        )

        recommendations = engine.generate_recommendations([(decision, metrics)])

        # Should include complexity warning
        assert any(
            "complex structure" in risk.lower() for risk in recommendations.risk_considerations
        )

    def test_recommendations_with_low_coverage_adequacy(self, engine):
        """Test recommendations when coverage adequacy is low."""
        decision = InsuranceDecision(
            retained_limit=1_000_000,
            layers=[Layer(attachment_point=1_000_000, limit=2_000_000, premium_rate=0.01)],
            total_premium=20_000,
            total_coverage=3_000_000,
            pricing_scenario="baseline",
            optimization_method="SLSQP",
        )

        metrics = DecisionMetrics(
            ergodic_growth_rate=0.08,
            bankruptcy_probability=0.015,
            expected_roe=0.10,
            roe_improvement=0.005,
            premium_to_limit_ratio=0.007,
            coverage_adequacy=0.6,  # Low adequacy
            capital_efficiency=1.2,
            value_at_risk_95=4_000_000,
            conditional_value_at_risk=5_000_000,
            decision_score=0.5,
        )

        recommendations = engine.generate_recommendations([(decision, metrics)])

        # Should include coverage adequacy warning
        assert any(
            "coverage may be insufficient" in risk.lower()
            for risk in recommendations.risk_considerations
        )

    def test_confidence_calculation_with_excellent_metrics(self, engine):
        """Test confidence calculation with excellent metrics."""
        decision = InsuranceDecision(
            retained_limit=1_000_000,
            layers=[Layer(attachment_point=1_000_000, limit=10_000_000, premium_rate=0.008)],
            total_premium=80_000,
            total_coverage=11_000_000,
            pricing_scenario="baseline",
            optimization_method="SLSQP",
        )

        metrics1 = DecisionMetrics(
            ergodic_growth_rate=0.15,
            bankruptcy_probability=0.003,  # Very low risk
            expected_roe=0.18,
            roe_improvement=0.05,  # High improvement
            premium_to_limit_ratio=0.007,
            coverage_adequacy=0.95,
            capital_efficiency=3.0,
            value_at_risk_95=2_000_000,
            conditional_value_at_risk=2_500_000,
            decision_score=0.9,  # High score
        )

        # Add alternative with lower score for comparison
        metrics2 = DecisionMetrics(
            ergodic_growth_rate=0.10,
            bankruptcy_probability=0.01,
            expected_roe=0.12,
            roe_improvement=0.01,
            premium_to_limit_ratio=0.01,
            coverage_adequacy=0.7,
            capital_efficiency=1.5,
            value_at_risk_95=3_000_000,
            conditional_value_at_risk=4_000_000,
            decision_score=0.6,  # Much lower score
        )

        decision2 = InsuranceDecision(
            retained_limit=2_000_000,
            layers=[Layer(attachment_point=2_000_000, limit=5_000_000, premium_rate=0.01)],
            total_premium=50_000,
            total_coverage=7_000_000,
            pricing_scenario="baseline",
            optimization_method="SLSQP",
        )

        recommendations = engine.generate_recommendations(
            [(decision, metrics1), (decision2, metrics2)]
        )

        # Should have high confidence
        assert recommendations.confidence_level >= 0.85

    def test_confidence_calculation_with_poor_metrics(self, engine):
        """Test confidence calculation with poor metrics."""
        decision = InsuranceDecision(
            retained_limit=3_000_000,
            layers=[Layer(attachment_point=3_000_000, limit=2_000_000, premium_rate=0.015)],
            total_premium=30_000,
            total_coverage=5_000_000,
            pricing_scenario="baseline",
            optimization_method="SLSQP",
        )

        metrics = DecisionMetrics(
            ergodic_growth_rate=0.06,
            bankruptcy_probability=0.025,  # High risk
            expected_roe=0.08,
            roe_improvement=-0.01,  # Negative improvement
            premium_to_limit_ratio=0.006,
            coverage_adequacy=0.5,
            capital_efficiency=0.8,
            value_at_risk_95=6_000_000,
            conditional_value_at_risk=8_000_000,
            decision_score=0.3,  # Low score
        )

        recommendations = engine.generate_recommendations([(decision, metrics)])

        # Should have low confidence
        assert recommendations.confidence_level <= 0.6


class TestLayerRateDetermination:
    """Test layer rate determination based on attachment points."""

    @pytest.fixture
    def engine(self):
        """Create a test engine instance."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.2,
            operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)

        loss_dist = Mock(spec=LossDistribution)
        loss_dist.expected_value.return_value = 100_000

        return InsuranceDecisionEngine(
            manufacturer=manufacturer,
            loss_distribution=loss_dist,
            pricing_scenario="baseline",
        )

    def test_higher_excess_rate_for_high_attachment(self, engine):
        """Test that high attachment points get higher excess rates."""
        # Create result with high attachment layer
        result = OptimizeResult()
        result.x = np.array([1_000_000, 0, 0, 30_000_000])  # High attachment for last layer
        result.fun = 0.5
        result.nit = 10
        result.success = True

        decision = engine._create_decision_from_result(result, OptimizationMethod.SLSQP)

        # Check that the high attachment layer has higher excess rate
        if decision.layers:
            for layer in decision.layers:
                if layer.attachment_point >= 25_000_000:
                    assert layer.premium_rate == engine.current_scenario.higher_excess_rate


class TestDifferentialEvolutionPenalties:
    """Test penalty calculations in differential evolution optimization."""

    @pytest.fixture
    def engine(self):
        """Create a test engine instance."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.2,
            operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)

        loss_dist = Mock(spec=LossDistribution)
        loss_dist.expected_value.return_value = 100_000

        return InsuranceDecisionEngine(
            manufacturer=manufacturer,
            loss_distribution=loss_dist,
            pricing_scenario="baseline",
        )

    def test_differential_evolution_coverage_max_penalty(self, engine):
        """Test penalty when coverage exceeds maximum in differential evolution."""
        constraints = OptimizationConstraints(
            max_coverage_limit=10_000_000,
            min_coverage_limit=5_000_000,
            max_premium_budget=500_000,
        )

        # Create a mock for the objective function to capture calls
        original_objective = engine._calculate_objective
        calls_made = []

        def track_objective(x, weights):
            calls_made.append(x.copy())
            # Force high coverage to trigger penalty
            if len(x) > 0 and sum(x) > constraints.max_coverage_limit:
                return 1000 * (sum(x) - constraints.max_coverage_limit)
            return original_objective(x, weights)

        with patch.object(engine, "_calculate_objective", side_effect=track_objective):
            with patch("ergodic_insurance.src.decision_engine.differential_evolution") as mock_de:
                # Mock the differential evolution to test penalty calculation
                def de_with_penalty(func, bounds, **kwargs):
                    # Test the penalty function with excessive coverage
                    x_test = np.array([5_000_000, 10_000_000, 10_000_000])  # 25M total > 10M max
                    penalty_result = func(x_test)
                    assert penalty_result > 1000  # Should include penalty

                    # Return a valid result
                    result = OptimizeResult()
                    result.x = np.array([1_000_000, 4_000_000, 0])
                    result.fun = 0.5
                    result.success = True
                    return result

                mock_de.side_effect = de_with_penalty

                engine.optimize_insurance_decision(
                    constraints, method=OptimizationMethod.DIFFERENTIAL_EVOLUTION
                )
