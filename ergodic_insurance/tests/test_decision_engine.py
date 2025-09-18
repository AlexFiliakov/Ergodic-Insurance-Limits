"""Tests for the insurance decision engine."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decision_engine import (
    DecisionMetrics,
    InsuranceDecision,
    InsuranceDecisionEngine,
    OptimizationConstraints,
    OptimizationMethod,
    Recommendations,
    SensitivityReport,
)
from ergodic_insurance.insurance_program import EnhancedInsuranceLayer as Layer
from ergodic_insurance.loss_distributions import LossDistribution
from ergodic_insurance.manufacturer import WidgetManufacturer


class TestOptimizationConstraints:
    """Test OptimizationConstraints dataclass."""

    def test_default_constraints(self):
        """Test default constraint values."""
        constraints = OptimizationConstraints()

        assert constraints.max_premium_budget == 1_000_000
        assert constraints.min_coverage_limit == 5_000_000
        assert constraints.max_coverage_limit == 100_000_000
        assert constraints.max_bankruptcy_probability == 0.01
        assert constraints.min_retained_limit == 100_000
        assert constraints.max_retained_limit == 10_000_000
        assert constraints.max_layers == 5
        assert constraints.min_layers == 1
        assert constraints.required_roi_improvement == 0.0

    def test_custom_constraints(self):
        """Test custom constraint values."""
        constraints = OptimizationConstraints(
            max_premium_budget=2_000_000,
            min_coverage_limit=10_000_000,
            max_bankruptcy_probability=0.005,
        )

        assert constraints.max_premium_budget == 2_000_000
        assert constraints.min_coverage_limit == 10_000_000
        assert constraints.max_bankruptcy_probability == 0.005

    def test_enhanced_constraints(self):
        """Test enhanced constraint fields."""
        constraints = OptimizationConstraints(
            max_debt_to_equity=1.5,
            max_insurance_cost_ratio=0.02,
            min_coverage_requirement=1_000_000,
            max_retention_limit=5_000_000,
        )

        assert constraints.max_debt_to_equity == 1.5
        assert constraints.max_insurance_cost_ratio == 0.02
        assert constraints.min_coverage_requirement == 1_000_000
        assert constraints.max_retention_limit == 5_000_000


class TestInsuranceDecision:
    """Test InsuranceDecision dataclass."""

    def test_decision_creation(self):
        """Test creating an insurance decision."""
        layers = [
            Layer(attachment_point=1_000_000, limit=4_000_000, premium_rate=0.01),
            Layer(attachment_point=5_000_000, limit=20_000_000, premium_rate=0.005),
        ]

        decision = InsuranceDecision(
            retained_limit=1_000_000,
            layers=layers,
            total_premium=140_000,  # 4M * 0.01 + 20M * 0.005
            total_coverage=26_000_000,
            pricing_scenario="baseline",
            optimization_method="SLSQP",
        )

        assert decision.retained_limit == 1_000_000
        assert len(decision.layers) == 2
        assert decision.total_premium == 140_000
        assert decision.total_coverage == 26_000_000
        assert decision.pricing_scenario == "baseline"

    def test_decision_auto_calculation(self):
        """Test automatic calculation of totals."""
        layers = [
            Layer(
                attachment_point=500_000,
                limit=4_500_000,
                premium_rate=0.01,
            ),
        ]

        decision = InsuranceDecision(
            retained_limit=500_000,
            layers=layers,
            total_premium=0,  # Should be calculated
            total_coverage=0,  # Should be calculated
            pricing_scenario="soft",
            optimization_method="SLSQP",
        )

        assert decision.total_premium == 45_000
        assert decision.total_coverage == 5_000_000


class TestDecisionMetrics:
    """Test DecisionMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating decision metrics."""
        metrics = DecisionMetrics(
            ergodic_growth_rate=0.12,
            bankruptcy_probability=0.005,
            expected_roe=0.15,
            roe_improvement=0.03,
            premium_to_limit_ratio=0.02,
            coverage_adequacy=0.85,
            capital_efficiency=2.5,
            value_at_risk_95=5_000_000,
            conditional_value_at_risk=7_000_000,
        )

        assert metrics.ergodic_growth_rate == 0.12
        assert metrics.bankruptcy_probability == 0.005
        assert metrics.expected_roe == 0.15
        assert metrics.capital_efficiency == 2.5

        # Test new ROE fields with defaults
        assert metrics.time_weighted_roe == 0.0
        assert metrics.roe_volatility == 0.0
        assert metrics.roe_sharpe_ratio == 0.0
        assert metrics.roe_downside_deviation == 0.0
        assert metrics.roe_1yr_rolling == 0.0
        assert metrics.roe_3yr_rolling == 0.0
        assert metrics.roe_5yr_rolling == 0.0
        assert metrics.operating_roe == 0.0
        assert metrics.insurance_impact_roe == 0.0
        assert metrics.tax_effect_roe == 0.0

    def test_enhanced_roe_metrics(self):
        """Test creation with enhanced ROE metrics."""
        metrics = DecisionMetrics(
            ergodic_growth_rate=0.12,
            bankruptcy_probability=0.005,
            expected_roe=0.15,
            roe_improvement=0.03,
            premium_to_limit_ratio=0.02,
            coverage_adequacy=0.85,
            capital_efficiency=2.5,
            value_at_risk_95=5_000_000,
            conditional_value_at_risk=7_000_000,
            # Enhanced ROE metrics
            time_weighted_roe=0.14,
            roe_volatility=0.05,
            roe_sharpe_ratio=2.4,
            roe_downside_deviation=0.03,
            roe_1yr_rolling=0.15,
            roe_3yr_rolling=0.14,
            roe_5yr_rolling=0.13,
            operating_roe=0.20,
            insurance_impact_roe=-0.04,
            tax_effect_roe=-0.01,
        )

        assert metrics.time_weighted_roe == 0.14
        assert metrics.roe_volatility == 0.05
        assert metrics.roe_sharpe_ratio == 2.4
        assert metrics.roe_downside_deviation == 0.03
        assert metrics.roe_1yr_rolling == 0.15
        assert metrics.roe_3yr_rolling == 0.14
        assert metrics.roe_5yr_rolling == 0.13
        assert metrics.operating_roe == 0.20
        assert metrics.insurance_impact_roe == -0.04
        assert metrics.tax_effect_roe == -0.01

    def test_calculate_score_default_weights(self):
        """Test score calculation with default weights."""
        metrics = DecisionMetrics(
            ergodic_growth_rate=0.15,  # 15% growth
            bankruptcy_probability=0.01,  # 1% risk
            expected_roe=0.18,
            roe_improvement=0.05,
            premium_to_limit_ratio=0.02,
            coverage_adequacy=0.9,
            capital_efficiency=0.8,
            value_at_risk_95=5_000_000,
            conditional_value_at_risk=7_000_000,
        )

        score = metrics.calculate_score()

        assert 0 <= score <= 1
        assert metrics.decision_score == score
        # With good metrics, score should be relatively high
        assert score > 0.6

    def test_calculate_score_custom_weights(self):
        """Test score calculation with custom weights."""
        metrics = DecisionMetrics(
            ergodic_growth_rate=0.10,
            bankruptcy_probability=0.02,
            expected_roe=0.12,
            roe_improvement=0.02,
            premium_to_limit_ratio=0.03,
            coverage_adequacy=0.7,
            capital_efficiency=0.6,
            value_at_risk_95=5_000_000,
            conditional_value_at_risk=7_000_000,
        )

        custom_weights = {
            "growth": 0.5,  # Emphasize growth
            "risk": 0.2,
            "efficiency": 0.2,
            "adequacy": 0.1,
        }

        score = metrics.calculate_score(custom_weights)

        assert 0 <= score <= 1
        assert metrics.decision_score == score


class TestInsuranceDecisionEngine:
    """Test InsuranceDecisionEngine class."""

    @pytest.fixture
    def mock_manufacturer(self):
        """Create mock manufacturer."""
        manufacturer = Mock(spec=WidgetManufacturer)
        manufacturer.current_assets = 10_000_000
        manufacturer.total_assets = 10_000_000
        manufacturer.equity = 10_000_000
        manufacturer.asset_turnover_ratio = 1.0
        manufacturer.base_operating_margin = 0.15  # Higher margin for positive growth
        manufacturer.step = Mock(return_value={"roe": 0.15, "assets": 10_000_000})
        manufacturer.reset = Mock()

        # Create a copy that returns a new mock with the same attributes
        manufacturer_copy = Mock(spec=WidgetManufacturer)
        manufacturer_copy.current_assets = 10_000_000
        manufacturer_copy.total_assets = 10_000_000
        manufacturer_copy.equity = 10_000_000
        manufacturer_copy.asset_turnover_ratio = 1.0
        manufacturer_copy.base_operating_margin = 0.15
        manufacturer_copy.step = Mock(return_value={"roe": 0.15, "assets": 10_000_000})
        manufacturer_copy.reset = Mock()
        manufacturer.copy = Mock(return_value=manufacturer_copy)

        return manufacturer

    @pytest.fixture
    def mock_loss_distribution(self):
        """Create mock loss distribution."""
        loss_dist = Mock(spec=LossDistribution)
        loss_dist.rvs = Mock(return_value=100_000)
        loss_dist.ppf = Mock(side_effect=lambda p: p * 10_000_000)
        loss_dist.expected_value = Mock(return_value=500_000)
        return loss_dist

    @pytest.fixture
    def engine(self, mock_manufacturer, mock_loss_distribution):
        """Create decision engine with mocks."""
        with patch("ergodic_insurance.decision_engine.ConfigLoader") as mock_loader:
            # Mock the config loader and pricing config
            mock_loader.return_value.load_pricing_scenarios.return_value.get_scenario.return_value = Mock(
                primary_layer_rate=0.01,
                first_excess_rate=0.005,
                higher_excess_rate=0.002,
            )

            engine = InsuranceDecisionEngine(
                manufacturer=mock_manufacturer,
                loss_distribution=mock_loss_distribution,
                pricing_scenario="baseline",
            )
            return engine

    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine.manufacturer is not None
        assert engine.loss_distribution is not None
        assert engine.pricing_scenario == "baseline"
        assert hasattr(engine, "insurance_program")
        assert hasattr(engine, "ergodic_analyzer")

    def test_optimize_insurance_decision_slsqp(self, engine):
        """Test optimization using SLSQP method."""
        constraints = OptimizationConstraints(
            max_premium_budget=500_000,
            min_coverage_limit=5_000_000,
            max_coverage_limit=50_000_000,
        )

        decision = engine.optimize_insurance_decision(constraints, method=OptimizationMethod.SLSQP)

        assert isinstance(decision, InsuranceDecision)
        assert decision.retained_limit >= constraints.min_retained_limit
        assert decision.retained_limit <= constraints.max_retained_limit
        assert decision.total_premium <= constraints.max_premium_budget
        assert decision.optimization_method == "SLSQP"

    def test_optimize_insurance_decision_differential_evolution(self, engine):
        """Test optimization using differential evolution."""
        constraints = OptimizationConstraints(
            max_premium_budget=300_000,
            min_coverage_limit=5_000_000,
            max_layers=3,
        )

        decision = engine.optimize_insurance_decision(
            constraints, method=OptimizationMethod.DIFFERENTIAL_EVOLUTION
        )

        assert isinstance(decision, InsuranceDecision)
        assert len(decision.layers) <= constraints.max_layers
        assert decision.total_premium <= constraints.max_premium_budget
        assert decision.optimization_method == "differential_evolution"

    def test_optimize_with_custom_weights(self, engine):
        """Test optimization with custom objective weights."""
        constraints = OptimizationConstraints()
        weights = {"growth": 0.6, "risk": 0.3, "cost": 0.1}

        decision = engine.optimize_insurance_decision(constraints, weights=weights)

        assert isinstance(decision, InsuranceDecision)
        assert decision.total_coverage >= constraints.min_coverage_limit

    def test_decision_caching(self, engine):
        """Test that decisions are cached."""
        constraints = OptimizationConstraints(max_premium_budget=400_000)

        # First call
        decision1 = engine.optimize_insurance_decision(constraints)

        # Second call with same constraints should return cached result
        decision2 = engine.optimize_insurance_decision(constraints)

        # Should be the same object (cached)
        assert decision1.objective_value == decision2.objective_value
        assert decision1.convergence_iterations == decision2.convergence_iterations

    def test_calculate_decision_metrics(self, engine):
        """Test metrics calculation for a decision."""
        decision = InsuranceDecision(
            retained_limit=1_000_000,
            layers=[
                Layer(
                    attachment_point=1_000_000,
                    limit=4_000_000,
                    premium_rate=0.01,
                )
            ],
            total_premium=40_000,
            total_coverage=5_000_000,
            pricing_scenario="baseline",
            optimization_method="SLSQP",
        )

        metrics = engine.calculate_decision_metrics(decision)

        assert isinstance(metrics, DecisionMetrics)
        assert metrics.ergodic_growth_rate >= 0
        assert 0 <= metrics.bankruptcy_probability <= 1
        assert metrics.premium_to_limit_ratio == 40_000 / 5_000_000
        assert metrics.coverage_adequacy > 0

    def test_run_sensitivity_analysis(self, engine):
        """Test sensitivity analysis."""
        base_decision = InsuranceDecision(
            retained_limit=500_000,
            layers=[
                Layer(
                    attachment_point=500_000,
                    limit=4_500_000,
                    premium_rate=0.01,
                )
            ],
            total_premium=45_000,
            total_coverage=5_000_000,
            pricing_scenario="baseline",
            optimization_method="SLSQP",
        )

        # Run sensitivity analysis on subset of parameters
        report = engine.run_sensitivity_analysis(
            base_decision, parameters=["premium_rates", "capital_base"]
        )

        assert isinstance(report, SensitivityReport)
        assert report.base_decision == base_decision
        assert "premium_rates" in report.parameter_sensitivities
        assert "capital_base" in report.parameter_sensitivities
        assert len(report.key_drivers) > 0
        assert len(report.stress_test_results) > 0

    def test_generate_recommendations(self, engine):
        """Test recommendation generation."""
        # Create sample analysis results
        decision1 = InsuranceDecision(
            retained_limit=1_000_000,
            layers=[
                Layer(
                    attachment_point=1_000_000,
                    limit=4_000_000,
                    premium_rate=0.01,
                )
            ],
            total_premium=40_000,
            total_coverage=5_000_000,
            pricing_scenario="baseline",
            optimization_method="SLSQP",
        )

        metrics1 = DecisionMetrics(
            ergodic_growth_rate=0.12,
            bankruptcy_probability=0.008,
            expected_roe=0.15,
            roe_improvement=0.03,
            premium_to_limit_ratio=0.008,
            coverage_adequacy=0.85,
            capital_efficiency=2.0,
            value_at_risk_95=3_000_000,
            conditional_value_at_risk=4_000_000,
            decision_score=0.75,
        )

        decision2 = InsuranceDecision(
            retained_limit=2_000_000,
            layers=[
                Layer(
                    attachment_point=2_000_000,
                    limit=8_000_000,
                    premium_rate=0.008,
                )
            ],
            total_premium=64_000,
            total_coverage=10_000_000,
            pricing_scenario="baseline",
            optimization_method="SLSQP",
        )

        metrics2 = DecisionMetrics(
            ergodic_growth_rate=0.10,
            bankruptcy_probability=0.012,
            expected_roe=0.13,
            roe_improvement=0.02,
            premium_to_limit_ratio=0.0064,
            coverage_adequacy=0.90,
            capital_efficiency=1.5,
            value_at_risk_95=4_000_000,
            conditional_value_at_risk=5_000_000,
            decision_score=0.65,
        )

        analysis_results = [(decision1, metrics1), (decision2, metrics2)]

        recommendations = engine.generate_recommendations(analysis_results)

        assert isinstance(recommendations, Recommendations)
        assert recommendations.primary_recommendation == decision1  # Higher score
        assert len(recommendations.alternative_options) > 0
        assert len(recommendations.implementation_timeline) > 0
        assert len(recommendations.risk_considerations) > 0
        assert "ROE Improvement" in recommendations.expected_benefits
        assert 0 <= recommendations.confidence_level <= 1

    def test_empty_recommendations_error(self, engine):
        """Test that empty analysis results raise error."""
        with pytest.raises(ValueError, match="No analysis results"):
            engine.generate_recommendations([])

    def test_validate_decision(self, engine):
        """Test decision validation against constraints."""
        constraints = OptimizationConstraints(
            max_premium_budget=50_000,
            min_coverage_limit=5_000_000,
            max_coverage_limit=20_000_000,
        )

        # Valid decision
        valid_decision = InsuranceDecision(
            retained_limit=1_000_000,
            layers=[
                Layer(
                    attachment_point=1_000_000,
                    limit=9_000_000,
                    premium_rate=0.005,
                )
            ],
            total_premium=45_000,
            total_coverage=10_000_000,
            pricing_scenario="baseline",
            optimization_method="SLSQP",
        )

        assert engine._validate_decision(valid_decision, constraints) is True

        # Invalid decision (premium too high)
        invalid_decision = InsuranceDecision(
            retained_limit=1_000_000,
            layers=[
                Layer(
                    attachment_point=1_000_000,
                    limit=9_000_000,
                    premium_rate=0.01,
                )
            ],
            total_premium=90_000,  # Exceeds budget
            total_coverage=10_000_000,
            pricing_scenario="baseline",
            optimization_method="SLSQP",
        )

        assert engine._validate_decision(invalid_decision, constraints) is False

    def test_calculate_objective(self, engine):
        """Test objective function calculation."""
        x = np.array([1_000_000, 4_000_000, 5_000_000])  # retention + 2 layers
        weights = {"growth": 0.4, "risk": 0.4, "cost": 0.2}

        objective_value = engine._calculate_objective(x, weights)

        assert isinstance(objective_value, float)
        # Should not be the penalty value
        assert objective_value < 1e10

    def test_calculate_premium(self, engine):
        """Test premium calculation for structure."""
        x = np.array([1_000_000, 4_000_000, 20_000_000])  # retention + 2 layers

        premium = engine._calculate_premium(x)

        assert isinstance(premium, float)
        assert premium > 0
        # Primary layer: 4M * 0.01 = 40K
        # Excess layer: 20M * 0.005 = 100K
        # Total: 140K
        assert premium == 140_000

    def test_estimate_growth_rate(self, engine):
        """Test growth rate estimation."""
        retained_limit = 1_000_000
        layer_limits = [4_000_000, 20_000_000]

        growth_rate = engine._estimate_growth_rate(retained_limit, layer_limits)

        assert isinstance(growth_rate, float)
        assert growth_rate > 0
        assert growth_rate < 1  # Should be reasonable percentage

    def test_estimate_bankruptcy_probability(self, engine):
        """Test bankruptcy probability estimation."""
        x = np.array([1_000_000, 4_000_000, 20_000_000])  # Good coverage

        bankruptcy_prob = engine._estimate_bankruptcy_probability(x)

        assert isinstance(bankruptcy_prob, float)
        assert 0 <= bankruptcy_prob <= 1
        # With good coverage, should be low
        assert bankruptcy_prob < 0.02


class TestIntegration:
    """Integration tests for decision engine."""

    def test_full_optimization_workflow(self):
        """Test complete optimization workflow."""
        # Create real components

        manufacturer_config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.6,
        )
        manufacturer = WidgetManufacturer(config=manufacturer_config)

        # Create simple loss distribution
        loss_dist = Mock(spec=LossDistribution)
        loss_dist.rvs = Mock(return_value=np.random.lognormal(11, 2))  # ~100K mean
        loss_dist.ppf = Mock(
            side_effect=lambda p: np.exp(11 + 2 * np.sqrt(2) * np.log(p / (1 - p + 1e-10)))
        )
        loss_dist.expected_value = Mock(return_value=100_000)

        # Create engine
        with patch("ergodic_insurance.decision_engine.ConfigLoader") as mock_loader:
            mock_loader.return_value.load_pricing_scenarios.return_value.get_scenario.return_value = Mock(
                primary_layer_rate=0.01,
                first_excess_rate=0.005,
                higher_excess_rate=0.002,
            )

            engine = InsuranceDecisionEngine(
                manufacturer=manufacturer,
                loss_distribution=loss_dist,
                pricing_scenario="baseline",
            )

        # Define constraints
        constraints = OptimizationConstraints(
            max_premium_budget=500_000,
            min_coverage_limit=5_000_000,
            max_coverage_limit=50_000_000,
            max_bankruptcy_probability=0.01,
        )

        # Optimize decision
        decision = engine.optimize_insurance_decision(constraints)

        # Calculate metrics
        metrics = engine.calculate_decision_metrics(decision)

        # Run sensitivity analysis
        sensitivity = engine.run_sensitivity_analysis(decision, parameters=["premium_rates"])

        # Generate recommendations
        recommendations = engine.generate_recommendations([(decision, metrics)])

        # Verify complete workflow
        assert isinstance(decision, InsuranceDecision)
        assert isinstance(metrics, DecisionMetrics)
        assert isinstance(sensitivity, SensitivityReport)
        assert isinstance(recommendations, Recommendations)

        # Verify decision quality
        assert decision.total_premium <= constraints.max_premium_budget
        assert decision.total_coverage >= constraints.min_coverage_limit
        assert metrics.bankruptcy_probability <= constraints.max_bankruptcy_probability

    def test_multi_scenario_optimization(self):
        """Test optimization across different pricing scenarios."""

        manufacturer_config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.6,
        )
        manufacturer = WidgetManufacturer(config=manufacturer_config)

        loss_dist = Mock(spec=LossDistribution)
        loss_dist.rvs = Mock(return_value=100_000)
        loss_dist.ppf = Mock(return_value=1_000_000)
        loss_dist.expected_value = Mock(return_value=500_000)

        constraints = OptimizationConstraints(
            max_premium_budget=400_000,
            min_coverage_limit=5_000_000,
        )

        decisions = {}

        for scenario in ["inexpensive", "baseline", "expensive"]:
            with patch("ergodic_insurance.decision_engine.ConfigLoader") as mock_loader:
                # Mock different rates for each scenario
                rates = {
                    "inexpensive": (0.005, 0.003, 0.001),
                    "baseline": (0.01, 0.005, 0.002),
                    "expensive": (0.015, 0.008, 0.004),
                }

                mock_scenario = Mock(
                    primary_layer_rate=rates[scenario][0],
                    first_excess_rate=rates[scenario][1],
                    higher_excess_rate=rates[scenario][2],
                )
                mock_loader.return_value.load_pricing_scenarios.return_value.get_scenario.return_value = (
                    mock_scenario
                )

                engine = InsuranceDecisionEngine(
                    manufacturer=manufacturer,
                    loss_distribution=loss_dist,
                    pricing_scenario=scenario,
                )

                decision = engine.optimize_insurance_decision(constraints)
                decisions[scenario] = decision

        # Verify different scenarios produce different decisions
        # Soft market should have lowest premiums
        assert decisions["inexpensive"].total_premium <= decisions["baseline"].total_premium
        # Hard market may have budget-constrained premium (fallback to differential evolution)
        # so we can't assert it's higher than baseline

        # In soft market, should buy more coverage for same or lower cost
        assert decisions["inexpensive"].total_coverage >= decisions["expensive"].total_coverage
        # Coverage should be inversely related to market hardness when budget-constrained
        assert all(d.total_premium <= constraints.max_premium_budget for d in decisions.values())


class TestEnhancedOptimizationMethods:
    """Test the new enhanced optimization methods."""

    @pytest.fixture
    def engine(self):
        """Create a test engine instance."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.2,
            base_operating_margin=0.08,
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

    def test_enhanced_slsqp_optimization(self, engine):
        """Test enhanced SLSQP optimization method."""
        constraints = OptimizationConstraints(
            max_premium_budget=500_000,
            min_coverage_limit=5_000_000,
            max_coverage_limit=20_000_000,
            max_layers=3,
        )

        decision = engine.optimize_insurance_decision(
            constraints, method=OptimizationMethod.ENHANCED_SLSQP
        )

        assert isinstance(decision, InsuranceDecision)
        assert decision.optimization_method == "enhanced_slsqp"
        assert decision.total_premium <= constraints.max_premium_budget
        assert (
            constraints.min_coverage_limit
            <= decision.total_coverage
            <= constraints.max_coverage_limit
        )
        assert len(decision.layers) <= constraints.max_layers

    @pytest.mark.filterwarnings("ignore:delta_grad == 0.0:UserWarning")
    def test_trust_region_optimization(self, engine):
        """Test trust-region optimization method."""
        constraints = OptimizationConstraints(
            max_premium_budget=400_000,
            min_coverage_limit=5_000_000,
            max_coverage_limit=15_000_000,
            max_layers=2,
        )

        decision = engine.optimize_insurance_decision(
            constraints, method=OptimizationMethod.TRUST_REGION
        )

        assert isinstance(decision, InsuranceDecision)
        assert decision.optimization_method == "trust_region"
        assert decision.total_premium <= constraints.max_premium_budget
        assert (
            constraints.min_coverage_limit
            <= decision.total_coverage
            <= constraints.max_coverage_limit
        )

    def test_penalty_method_optimization(self, engine):
        """Test penalty method optimization."""
        constraints = OptimizationConstraints(
            max_premium_budget=300_000,
            min_coverage_limit=4_000_000,
            max_coverage_limit=12_000_000,
            max_bankruptcy_probability=0.02,
        )

        decision = engine.optimize_insurance_decision(
            constraints, method=OptimizationMethod.PENALTY_METHOD
        )

        assert isinstance(decision, InsuranceDecision)
        assert decision.optimization_method == "penalty_method"
        assert decision.total_premium <= constraints.max_premium_budget
        assert (
            constraints.min_coverage_limit
            <= decision.total_coverage
            <= constraints.max_coverage_limit
        )

    def test_augmented_lagrangian_optimization(self, engine):
        """Test augmented Lagrangian optimization."""
        constraints = OptimizationConstraints(
            max_premium_budget=450_000,
            min_coverage_limit=6_000_000,
            max_coverage_limit=18_000_000,
            max_layers=3,
        )

        decision = engine.optimize_insurance_decision(
            constraints, method=OptimizationMethod.AUGMENTED_LAGRANGIAN
        )

        assert isinstance(decision, InsuranceDecision)
        assert decision.optimization_method == "augmented_lagrangian"
        assert decision.total_premium <= constraints.max_premium_budget
        assert (
            constraints.min_coverage_limit
            <= decision.total_coverage
            <= constraints.max_coverage_limit
        )

    def test_multi_start_optimization(self, engine):
        """Test multi-start global optimization."""
        constraints = OptimizationConstraints(
            max_premium_budget=500_000,
            min_coverage_limit=5_000_000,
            max_coverage_limit=20_000_000,
            max_layers=4,
        )

        decision = engine.optimize_insurance_decision(
            constraints, method=OptimizationMethod.MULTI_START
        )

        assert isinstance(decision, InsuranceDecision)
        assert decision.optimization_method == "multi_start"
        assert decision.total_premium <= constraints.max_premium_budget
        assert (
            constraints.min_coverage_limit
            <= decision.total_coverage
            <= constraints.max_coverage_limit
        )

    def test_enhanced_constraint_handling(self, engine):
        """Test the enhanced constraints (debt-to-equity, insurance cost ceiling, etc.)."""
        constraints = OptimizationConstraints(
            max_premium_budget=600_000,
            min_coverage_limit=5_000_000,
            max_coverage_limit=25_000_000,
            max_debt_to_equity=1.5,
            max_insurance_cost_ratio=0.025,  # 2.5% of revenue
            min_coverage_requirement=3_000_000,
            max_retention_limit=2_000_000,
        )

        decision = engine.optimize_insurance_decision(
            constraints, method=OptimizationMethod.ENHANCED_SLSQP
        )

        assert isinstance(decision, InsuranceDecision)
        assert decision.retained_limit <= constraints.max_retention_limit

        # Verify insurance cost ceiling
        revenue = engine.manufacturer.total_assets * engine.manufacturer.asset_turnover_ratio
        cost_ratio = decision.total_premium / revenue
        assert cost_ratio <= constraints.max_insurance_cost_ratio

        # Verify minimum coverage requirement
        coverage_from_layers = sum(layer.limit for layer in decision.layers)
        assert coverage_from_layers >= constraints.min_coverage_requirement

    def test_optimization_convergence_info(self, engine):
        """Test that optimization methods provide convergence information."""
        constraints = OptimizationConstraints(
            max_premium_budget=400_000,
            min_coverage_limit=5_000_000,
            max_coverage_limit=15_000_000,
        )

        # Test with enhanced SLSQP (which should have convergence info)
        decision = engine.optimize_insurance_decision(
            constraints, method=OptimizationMethod.ENHANCED_SLSQP
        )

        assert isinstance(decision, InsuranceDecision)
        # Convergence iterations should be set
        assert decision.convergence_iterations >= 0
        # Objective value should be set
        assert decision.objective_value != 0.0

    @pytest.mark.filterwarnings("ignore:delta_grad == 0.0:UserWarning")
    def test_fallback_optimization_on_failure(self, engine):
        """Test that optimization falls back to alternative method on failure."""
        # Create moderately restrictive constraints
        constraints = OptimizationConstraints(
            max_premium_budget=100_000,  # Low but not impossible budget
            min_coverage_limit=10_000_000,  # High coverage requirement
            max_coverage_limit=20_000_000,
            max_bankruptcy_probability=0.01,  # Strict but achievable risk constraint
        )

        # This should complete without hanging
        decision = engine.optimize_insurance_decision(
            constraints, method=OptimizationMethod.TRUST_REGION
        )

        # Should still return a decision (even if not optimal)
        assert isinstance(decision, InsuranceDecision)
        # Method might have changed due to fallback
        assert decision.optimization_method in [
            "trust_region",
            "differential_evolution",
            "weighted_sum",
        ]
