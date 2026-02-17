"""Tests for the insurance decision engine."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decision_engine import (
    DecisionMetrics,
    DecisionOptimizationConstraints,
    InsuranceDecision,
    InsuranceDecisionEngine,
    OptimizationMethod,
    Recommendations,
    SensitivityReport,
)
from ergodic_insurance.insurance_program import EnhancedInsuranceLayer as Layer
from ergodic_insurance.loss_distributions import LognormalLoss, LossDistribution
from ergodic_insurance.manufacturer import WidgetManufacturer


class TestDecisionOptimizationConstraints:
    """Test DecisionOptimizationConstraints dataclass."""

    def test_default_constraints(self):
        """Test default constraint values."""
        constraints = DecisionOptimizationConstraints()

        assert constraints.max_premium_budget == 1_000_000
        assert constraints.min_total_coverage == 5_000_000
        assert constraints.max_total_coverage == 100_000_000
        assert constraints.max_bankruptcy_probability == 0.01
        assert constraints.min_retained_limit == 100_000
        assert constraints.max_retained_limit == 10_000_000
        assert constraints.max_layers == 5
        assert constraints.min_layers == 1
        assert constraints.required_roi_improvement == 0.0

    def test_custom_constraints(self):
        """Test custom constraint values."""
        constraints = DecisionOptimizationConstraints(
            max_premium_budget=2_000_000,
            min_total_coverage=10_000_000,
            max_bankruptcy_probability=0.005,
        )

        assert constraints.max_premium_budget == 2_000_000
        assert constraints.min_total_coverage == 10_000_000
        assert constraints.max_bankruptcy_probability == 0.005

    def test_enhanced_constraints(self):
        """Test enhanced constraint fields."""
        constraints = DecisionOptimizationConstraints(
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
            Layer(attachment_point=1_000_000, limit=4_000_000, base_premium_rate=0.01),
            Layer(attachment_point=5_000_000, limit=20_000_000, base_premium_rate=0.005),
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
                base_premium_rate=0.01,
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

    def test_calculate_score_custom_targets(self):
        """Test score calculation with custom normalization targets."""
        metrics = DecisionMetrics(
            ergodic_growth_rate=0.08,
            bankruptcy_probability=0.01,
            expected_roe=0.10,
            roe_improvement=0.02,
            premium_to_limit_ratio=0.02,
            coverage_adequacy=0.9,
            capital_efficiency=0.8,
            value_at_risk_95=5_000_000,
            conditional_value_at_risk=7_000_000,
        )

        # A conservative CFO targeting 8% growth should score 1.0 on growth
        targets = {"growth_target": 0.08, "max_acceptable_risk": 0.03}
        score = metrics.calculate_score(targets=targets)

        assert 0 <= score <= 1
        assert metrics.decision_score == score
        # growth_score = 0.08/0.08 = 1.0, risk_score = 1 - 0.01/0.03 = 0.667
        # With default weights: 0.3*1.0 + 0.3*0.667 + 0.2*0.8 + 0.2*0.9 = 0.84
        assert score > 0.8

    def test_calculate_score_targets_backward_compatible(self):
        """Test that omitting targets still works (backward compatibility)."""
        metrics = DecisionMetrics(
            ergodic_growth_rate=0.15,
            bankruptcy_probability=0.01,
            expected_roe=0.18,
            roe_improvement=0.05,
            premium_to_limit_ratio=0.02,
            coverage_adequacy=0.9,
            capital_efficiency=0.8,
            value_at_risk_95=5_000_000,
            conditional_value_at_risk=7_000_000,
        )

        # Calling without targets should use defaults and not raise
        score_no_targets = metrics.calculate_score()
        assert 0 <= score_no_targets <= 1

        # Explicitly passing the defaults should give the same result
        score_with_defaults = metrics.calculate_score(
            targets={"growth_target": 0.10, "max_acceptable_risk": 0.05}
        )
        assert score_no_targets == score_with_defaults

    def test_calculate_score_conservative_vs_aggressive_targets(self):
        """Test that targets properly calibrate scores for different risk appetites."""
        metrics = DecisionMetrics(
            ergodic_growth_rate=0.08,
            bankruptcy_probability=0.02,
            expected_roe=0.10,
            roe_improvement=0.02,
            premium_to_limit_ratio=0.02,
            coverage_adequacy=0.8,
            capital_efficiency=0.7,
            value_at_risk_95=5_000_000,
            conditional_value_at_risk=7_000_000,
        )

        # Conservative company: 8% growth target
        conservative_score = metrics.calculate_score(targets={"growth_target": 0.08})
        # Aggressive company: 20% growth target
        aggressive_score = metrics.calculate_score(targets={"growth_target": 0.20})

        # Same metrics should score higher against conservative targets
        assert conservative_score > aggressive_score

    def test_calculate_score_partial_targets(self):
        """Test that partial targets dict uses defaults for missing keys."""
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

        # Only override growth_target, max_acceptable_risk should default to 0.05
        score_partial = metrics.calculate_score(targets={"growth_target": 0.10})
        score_full = metrics.calculate_score(
            targets={"growth_target": 0.10, "max_acceptable_risk": 0.05}
        )
        assert score_partial == score_full


class TestDecisionMetricsGrouping:
    """Test sub-group access and new methods on DecisionMetrics."""

    @staticmethod
    def _make_metrics(**overrides):
        defaults = {
            "ergodic_growth_rate": 0.12,
            "bankruptcy_probability": 0.005,
            "expected_roe": 0.15,
            "roe_improvement": 0.03,
            "premium_to_limit_ratio": 0.02,
            "coverage_adequacy": 0.85,
            "capital_efficiency": 2.5,
            "value_at_risk_95": 5_000_000,
            "conditional_value_at_risk": 7_000_000,
        }
        defaults.update(overrides)
        return DecisionMetrics(**defaults)

    def test_grouped_access_growth(self):
        m = self._make_metrics()
        assert m.growth.ergodic_growth_rate == 0.12

    def test_grouped_access_risk(self):
        m = self._make_metrics()
        assert m.risk.bankruptcy_probability == 0.005
        assert m.risk.value_at_risk_95 == 5_000_000
        assert m.risk.conditional_value_at_risk == 7_000_000

    def test_grouped_access_roe(self):
        m = self._make_metrics(
            time_weighted_roe=0.14,
            roe_volatility=0.05,
            operating_roe=0.20,
            insurance_impact_roe=-0.04,
            tax_effect_roe=-0.01,
        )
        assert m.roe.expected_roe == 0.15
        assert m.roe.roe_improvement == 0.03
        assert m.roe.time_weighted_roe == 0.14
        assert m.roe.roe_volatility == 0.05
        assert m.roe.components.operating_roe == 0.20
        assert m.roe.components.insurance_impact_roe == -0.04
        assert m.roe.components.tax_effect_roe == -0.01

    def test_grouped_access_efficiency(self):
        m = self._make_metrics()
        assert m.efficiency.premium_to_limit_ratio == 0.02
        assert m.efficiency.coverage_adequacy == 0.85
        assert m.efficiency.capital_efficiency == 2.5

    def test_flat_access_still_works(self):
        """Every flat field name resolves via __getattr__."""
        m = self._make_metrics(
            time_weighted_roe=0.14,
            operating_roe=0.20,
        )
        assert m.ergodic_growth_rate == 0.12
        assert m.bankruptcy_probability == 0.005
        assert m.expected_roe == 0.15
        assert m.roe_improvement == 0.03
        assert m.premium_to_limit_ratio == 0.02
        assert m.coverage_adequacy == 0.85
        assert m.capital_efficiency == 2.5
        assert m.value_at_risk_95 == 5_000_000
        assert m.conditional_value_at_risk == 7_000_000
        assert m.time_weighted_roe == 0.14
        assert m.operating_roe == 0.20

    def test_decision_score_top_level(self):
        m = self._make_metrics(decision_score=0.75)
        assert m.decision_score == 0.75

    def test_calculate_score_uses_getattr(self):
        m = self._make_metrics()
        score = m.calculate_score()
        assert 0 <= score <= 1
        assert m.decision_score == score

    def test_repr(self):
        m = self._make_metrics(decision_score=0.8)
        r = repr(m)
        assert "DecisionMetrics(" in r
        assert "score=0.800" in r
        assert "growth=" in r
        assert "risk=" in r
        assert "roe=" in r
        assert "efficiency=" in r

    def test_eq(self):
        m1 = self._make_metrics()
        m2 = self._make_metrics()
        assert m1 == m2
        m3 = self._make_metrics(ergodic_growth_rate=0.99)
        assert m1 != m3

    def test_to_dict_all(self):
        m = self._make_metrics(decision_score=0.7, operating_roe=0.20)
        d = m.to_dict()
        assert d["ergodic_growth_rate"] == 0.12
        assert d["bankruptcy_probability"] == 0.005
        assert d["operating_roe"] == 0.20
        assert d["decision_score"] == 0.7
        # Should be flat, no nested dicts
        assert all(not isinstance(v, dict) for v in d.values())
        # Should contain all 20 fields
        assert len(d) == 20

    def test_to_dict_group_growth(self):
        m = self._make_metrics()
        gd = m.to_dict(group="growth")
        assert gd == {"ergodic_growth_rate": 0.12}

    def test_to_dict_group_risk(self):
        m = self._make_metrics()
        rd = m.to_dict(group="risk")
        assert set(rd.keys()) == {
            "bankruptcy_probability",
            "value_at_risk_95",
            "conditional_value_at_risk",
        }

    def test_to_dict_roe_inlines_components(self):
        m = self._make_metrics(operating_roe=0.20)
        rd = m.to_dict(group="roe")
        assert "operating_roe" in rd
        assert rd["operating_roe"] == 0.20
        assert "components" not in rd  # inlined, not nested

    def test_to_dict_invalid_group(self):
        m = self._make_metrics()
        with pytest.raises(ValueError, match="Unknown group"):
            m.to_dict(group="nonexistent")

    def test_unknown_attribute_raises(self):
        m = self._make_metrics()
        with pytest.raises(AttributeError):
            _ = m.totally_fake_attribute

    def test_setattr_delegated_field(self):
        """Setting a delegated field propagates to the sub-group."""
        m = self._make_metrics()
        m.ergodic_growth_rate = 0.99
        assert m.ergodic_growth_rate == 0.99
        assert m.growth.ergodic_growth_rate == 0.99

    def test_deepcopy(self):
        """DecisionMetrics must support deepcopy (used in sensitivity analysis)."""
        import copy

        m = self._make_metrics(decision_score=0.8, operating_roe=0.20)
        m2 = copy.deepcopy(m)
        assert m == m2
        m2.ergodic_growth_rate = 0.99
        assert m.ergodic_growth_rate == 0.12  # original unchanged


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
        manufacturer.tax_rate = 0.25
        manufacturer.step = Mock(return_value={"roe": 0.15, "assets": 10_000_000})
        manufacturer.reset = Mock()

        # Create a copy that returns a new mock with the same attributes
        manufacturer_copy = Mock(spec=WidgetManufacturer)
        manufacturer_copy.current_assets = 10_000_000
        manufacturer_copy.total_assets = 10_000_000
        manufacturer_copy.equity = 10_000_000
        manufacturer_copy.asset_turnover_ratio = 1.0
        manufacturer_copy.base_operating_margin = 0.15
        manufacturer_copy.tax_rate = 0.25
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
        with patch.object(
            InsuranceDecisionEngine,
            "_load_pricing_scenarios",
        ) as mock_load:
            # Mock the pricing config
            mock_pricing = Mock()
            mock_pricing.get_scenario.return_value = Mock(
                primary_layer_rate=0.01,
                first_excess_rate=0.005,
                higher_excess_rate=0.002,
            )
            mock_load.return_value = mock_pricing

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
        constraints = DecisionOptimizationConstraints(
            max_premium_budget=500_000,
            min_total_coverage=5_000_000,
            max_total_coverage=50_000_000,
        )

        decision = engine.optimize_insurance_decision(constraints, method=OptimizationMethod.SLSQP)

        assert isinstance(decision, InsuranceDecision)
        assert decision.retained_limit >= constraints.min_retained_limit
        assert decision.retained_limit <= constraints.max_retained_limit
        assert decision.total_premium <= constraints.max_premium_budget
        assert decision.optimization_method == "SLSQP"

    def test_optimize_insurance_decision_differential_evolution(self, engine):
        """Test optimization using differential evolution."""
        constraints = DecisionOptimizationConstraints(
            max_premium_budget=300_000,
            min_total_coverage=5_000_000,
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
        constraints = DecisionOptimizationConstraints()
        weights = {"growth": 0.6, "risk": 0.3, "cost": 0.1}

        decision = engine.optimize_insurance_decision(constraints, weights=weights)

        assert isinstance(decision, InsuranceDecision)
        assert decision.total_coverage >= constraints.min_total_coverage

    def test_decision_caching(self, engine):
        """Test that decisions are cached."""
        constraints = DecisionOptimizationConstraints(max_premium_budget=400_000)

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
                    base_premium_rate=0.01,
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

    @pytest.mark.slow
    def test_run_sensitivity_analysis(self):
        """Test sensitivity analysis.

        Uses real objects (not mocks) because sensitivity analysis deepcopies
        engine state to prevent mutation, and Mock objects don't support deepcopy.
        """
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.6,
        )
        manufacturer = WidgetManufacturer(config=config)

        loss_dist = Mock(spec=LossDistribution)
        loss_dist.expected_value = Mock(return_value=500_000)

        with patch.object(
            InsuranceDecisionEngine,
            "_load_pricing_scenarios",
            return_value=Mock(
                get_scenario=Mock(
                    return_value=Mock(
                        primary_layer_rate=0.01,
                        first_excess_rate=0.005,
                        higher_excess_rate=0.002,
                    )
                )
            ),
        ):
            engine = InsuranceDecisionEngine(
                manufacturer=manufacturer,
                loss_distribution=loss_dist,
                pricing_scenario="baseline",
            )

        base_decision = InsuranceDecision(
            retained_limit=500_000,
            layers=[
                Layer(
                    attachment_point=500_000,
                    limit=4_500_000,
                    base_premium_rate=0.01,
                )
            ],
            total_premium=45_000,
            total_coverage=5_000_000,
            pricing_scenario="baseline",
            optimization_method="SLSQP",
        )

        # Run sensitivity analysis on subset of parameters
        report = engine.run_sensitivity_analysis(
            base_decision, parameters=["base_premium_rate", "capital_base"]
        )

        assert isinstance(report, SensitivityReport)
        assert report.base_decision == base_decision
        assert "base_premium_rate" in report.parameter_sensitivities
        assert "capital_base" in report.parameter_sensitivities
        assert len(report.key_drivers) > 0
        assert len(report.stress_test_results) > 0

        # Verify engine state was restored after sensitivity analysis
        assert float(manufacturer.total_assets) == float(engine.manufacturer.total_assets)

    def test_generate_recommendations(self, engine):
        """Test recommendation generation."""
        # Create sample analysis results
        decision1 = InsuranceDecision(
            retained_limit=1_000_000,
            layers=[
                Layer(
                    attachment_point=1_000_000,
                    limit=4_000_000,
                    base_premium_rate=0.01,
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
                    base_premium_rate=0.008,
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
        constraints = DecisionOptimizationConstraints(
            max_premium_budget=50_000,
            min_total_coverage=5_000_000,
            max_total_coverage=20_000_000,
        )

        # Valid decision
        valid_decision = InsuranceDecision(
            retained_limit=1_000_000,
            layers=[
                Layer(
                    attachment_point=1_000_000,
                    limit=9_000_000,
                    base_premium_rate=0.005,
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
                    base_premium_rate=0.01,
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

    @pytest.mark.slow
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
        loss_dist.rvs = Mock(return_value=np.random.default_rng(42).lognormal(11, 2))  # ~100K mean
        loss_dist.ppf = Mock(
            side_effect=lambda p: np.exp(11 + 2 * np.sqrt(2) * np.log(p / (1 - p + 1e-10)))
        )
        loss_dist.expected_value = Mock(return_value=100_000)

        # Create engine
        with patch.object(
            InsuranceDecisionEngine,
            "_load_pricing_scenarios",
            return_value=Mock(
                get_scenario=Mock(
                    return_value=Mock(
                        primary_layer_rate=0.01,
                        first_excess_rate=0.005,
                        higher_excess_rate=0.002,
                    )
                )
            ),
        ):

            engine = InsuranceDecisionEngine(
                manufacturer=manufacturer,
                loss_distribution=loss_dist,
                pricing_scenario="baseline",
            )

        # Define constraints
        constraints = DecisionOptimizationConstraints(
            max_premium_budget=500_000,
            min_total_coverage=5_000_000,
            max_total_coverage=50_000_000,
            max_bankruptcy_probability=0.01,
        )

        # Optimize decision
        decision = engine.optimize_insurance_decision(constraints)

        # Calculate metrics
        metrics = engine.calculate_decision_metrics(decision)

        # Run sensitivity analysis
        sensitivity = engine.run_sensitivity_analysis(decision, parameters=["base_premium_rates"])

        # Generate recommendations
        recommendations = engine.generate_recommendations([(decision, metrics)])

        # Verify complete workflow
        assert isinstance(decision, InsuranceDecision)
        assert isinstance(metrics, DecisionMetrics)
        assert isinstance(sensitivity, SensitivityReport)
        assert isinstance(recommendations, Recommendations)

        # Verify decision quality
        assert decision.total_premium <= constraints.max_premium_budget
        assert decision.total_coverage >= constraints.min_total_coverage
        assert metrics.bankruptcy_probability <= constraints.max_bankruptcy_probability

    @pytest.mark.slow
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

        constraints = DecisionOptimizationConstraints(
            max_premium_budget=400_000,
            min_total_coverage=5_000_000,
        )

        decisions = {}

        for scenario in ["inexpensive", "baseline", "expensive"]:
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
            with patch.object(
                InsuranceDecisionEngine,
                "_load_pricing_scenarios",
                return_value=Mock(get_scenario=Mock(return_value=mock_scenario)),
            ):
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
        constraints = DecisionOptimizationConstraints(
            max_premium_budget=500_000,
            min_total_coverage=5_000_000,
            max_total_coverage=20_000_000,
            max_layers=3,
        )

        decision = engine.optimize_insurance_decision(
            constraints, method=OptimizationMethod.ENHANCED_SLSQP
        )

        assert isinstance(decision, InsuranceDecision)
        assert decision.optimization_method == "enhanced_slsqp"
        assert decision.total_premium <= constraints.max_premium_budget
        assert (
            constraints.min_total_coverage
            <= decision.total_coverage
            <= constraints.max_total_coverage
        )
        assert len(decision.layers) <= constraints.max_layers

    @pytest.mark.filterwarnings("ignore:delta_grad == 0.0:UserWarning")
    def test_trust_region_optimization(self, engine):
        """Test trust-region optimization method."""
        constraints = DecisionOptimizationConstraints(
            max_premium_budget=400_000,
            min_total_coverage=5_000_000,
            max_total_coverage=15_000_000,
            max_layers=2,
        )

        decision = engine.optimize_insurance_decision(
            constraints, method=OptimizationMethod.TRUST_REGION
        )

        assert isinstance(decision, InsuranceDecision)
        assert decision.optimization_method == "trust_region"
        assert decision.total_premium <= constraints.max_premium_budget
        assert (
            constraints.min_total_coverage
            <= decision.total_coverage
            <= constraints.max_total_coverage
        )

    def test_penalty_method_optimization(self, engine):
        """Test penalty method optimization."""
        constraints = DecisionOptimizationConstraints(
            max_premium_budget=300_000,
            min_total_coverage=4_000_000,
            max_total_coverage=12_000_000,
            max_bankruptcy_probability=0.02,
        )

        decision = engine.optimize_insurance_decision(
            constraints, method=OptimizationMethod.PENALTY_METHOD
        )

        assert isinstance(decision, InsuranceDecision)
        assert decision.optimization_method == "penalty_method"
        assert decision.total_premium <= constraints.max_premium_budget
        assert (
            constraints.min_total_coverage
            <= decision.total_coverage
            <= constraints.max_total_coverage
        )

    def test_augmented_lagrangian_optimization(self, engine):
        """Test augmented Lagrangian optimization."""
        constraints = DecisionOptimizationConstraints(
            max_premium_budget=450_000,
            min_total_coverage=6_000_000,
            max_total_coverage=18_000_000,
            max_layers=3,
        )

        decision = engine.optimize_insurance_decision(
            constraints, method=OptimizationMethod.AUGMENTED_LAGRANGIAN
        )

        assert isinstance(decision, InsuranceDecision)
        assert decision.optimization_method == "augmented_lagrangian"
        assert decision.total_premium <= constraints.max_premium_budget
        assert (
            constraints.min_total_coverage
            <= decision.total_coverage
            <= constraints.max_total_coverage
        )

    def test_multi_start_optimization(self, engine):
        """Test multi-start global optimization."""
        constraints = DecisionOptimizationConstraints(
            max_premium_budget=500_000,
            min_total_coverage=5_000_000,
            max_total_coverage=20_000_000,
            max_layers=4,
        )

        decision = engine.optimize_insurance_decision(
            constraints, method=OptimizationMethod.MULTI_START
        )

        assert isinstance(decision, InsuranceDecision)
        assert decision.optimization_method == "multi_start"
        assert decision.total_premium <= constraints.max_premium_budget
        assert (
            constraints.min_total_coverage
            <= decision.total_coverage
            <= constraints.max_total_coverage
        )

    def test_enhanced_constraint_handling(self, engine):
        """Test the enhanced constraints (debt-to-equity, insurance cost ceiling, etc.)."""
        constraints = DecisionOptimizationConstraints(
            max_premium_budget=600_000,
            min_total_coverage=5_000_000,
            max_total_coverage=25_000_000,
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
        revenue = float(engine.manufacturer.total_assets) * engine.manufacturer.asset_turnover_ratio
        cost_ratio = decision.total_premium / revenue
        assert cost_ratio <= constraints.max_insurance_cost_ratio

        # Verify minimum coverage requirement
        coverage_from_layers = sum(layer.limit for layer in decision.layers)
        assert coverage_from_layers >= constraints.min_coverage_requirement

    def test_optimization_convergence_info(self, engine):
        """Test that optimization methods provide convergence information."""
        constraints = DecisionOptimizationConstraints(
            max_premium_budget=400_000,
            min_total_coverage=5_000_000,
            max_total_coverage=15_000_000,
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
        constraints = DecisionOptimizationConstraints(
            max_premium_budget=100_000,  # Low but not impossible budget
            min_total_coverage=10_000_000,  # High coverage requirement
            max_total_coverage=20_000_000,
            max_bankruptcy_probability=0.01,  # Strict but achievable risk constraint
        )

        # This should complete without hanging
        decision = engine.optimize_insurance_decision(
            constraints, method=OptimizationMethod.TRUST_REGION
        )

        # Should still return a decision (even if not optimal)
        assert isinstance(decision, InsuranceDecision)
        # Method might have changed due to fallback — any method is acceptable
        valid_methods = [m.value for m in OptimizationMethod]
        assert decision.optimization_method in valid_methods


class TestDecisionEngineConfigNewFields:
    """Test new DecisionEngineConfig fields from Issue #517."""

    def test_new_defaults(self):
        """Test that new fields have correct default values."""
        from ergodic_insurance.config.optimizer import DecisionEngineConfig

        cfg = DecisionEngineConfig()
        assert cfg.loss_cv == 0.5
        assert cfg.default_optimization_weights == {"growth": 0.4, "risk": 0.4, "cost": 0.2}
        assert cfg.layer_attachment_thresholds == (5_000_000, 25_000_000)

    def test_custom_new_fields(self):
        """Test that new fields can be overridden."""
        from ergodic_insurance.config.optimizer import DecisionEngineConfig

        cfg = DecisionEngineConfig(
            loss_cv=0.8,
            default_optimization_weights={"growth": 0.6, "risk": 0.3, "cost": 0.1},
            layer_attachment_thresholds=(3_000_000, 20_000_000),
        )
        assert cfg.loss_cv == 0.8
        assert cfg.default_optimization_weights == {"growth": 0.6, "risk": 0.3, "cost": 0.1}
        assert cfg.layer_attachment_thresholds == (3_000_000, 20_000_000)


class TestSimulationTaxApplication:
    """Test that Monte Carlo simulation applies taxes per ASC 740 (Issue #500)."""

    def _make_engine(self, tax_rate=0.25, operating_margin=0.15):
        """Helper to create a decision engine with configurable tax rate."""
        manufacturer = Mock(spec=WidgetManufacturer)
        manufacturer.current_assets = 10_000_000
        manufacturer.total_assets = 10_000_000
        manufacturer.equity = 10_000_000
        manufacturer.asset_turnover_ratio = 1.0
        manufacturer.base_operating_margin = operating_margin
        manufacturer.tax_rate = tax_rate
        manufacturer.step = Mock(return_value={"roe": 0.15, "assets": 10_000_000})
        manufacturer.reset = Mock()
        manufacturer_copy = Mock(spec=WidgetManufacturer)
        manufacturer_copy.current_assets = 10_000_000
        manufacturer_copy.total_assets = 10_000_000
        manufacturer_copy.equity = 10_000_000
        manufacturer_copy.asset_turnover_ratio = 1.0
        manufacturer_copy.base_operating_margin = operating_margin
        manufacturer_copy.tax_rate = tax_rate
        manufacturer_copy.step = Mock(return_value={"roe": 0.15, "assets": 10_000_000})
        manufacturer_copy.reset = Mock()
        manufacturer.copy = Mock(return_value=manufacturer_copy)

        loss_dist = Mock(spec=LossDistribution)
        loss_dist.rvs = Mock(return_value=100_000)
        loss_dist.ppf = Mock(side_effect=lambda p: p * 10_000_000)
        loss_dist.expected_value = Mock(return_value=500_000)

        with patch.object(
            InsuranceDecisionEngine,
            "_load_pricing_scenarios",
            return_value=Mock(
                get_scenario=Mock(
                    return_value=Mock(
                        primary_layer_rate=0.01,
                        first_excess_rate=0.005,
                        higher_excess_rate=0.002,
                    )
                )
            ),
        ):
            engine = InsuranceDecisionEngine(
                manufacturer=manufacturer,
                loss_distribution=loss_dist,
                pricing_scenario="baseline",
            )
        return engine

    def _make_decision(self):
        """Helper to create a simple insurance decision."""
        return InsuranceDecision(
            retained_limit=500_000,
            layers=[Layer(attachment_point=500_000, limit=5_000_000, base_premium_rate=0.01)],
            total_premium=50_000,
            total_coverage=5_500_000,
            pricing_scenario="baseline",
            optimization_method="SLSQP",
        )

    def test_tax_reduces_simulation_growth_rates(self):
        """Simulation with higher tax rate should produce lower growth rates."""
        decision = self._make_decision()

        engine_low_tax = self._make_engine(tax_rate=0.10)
        engine_high_tax = self._make_engine(tax_rate=0.40)

        results_low = engine_low_tax._run_simulation(decision, n_simulations=100, time_horizon=5)
        results_high = engine_high_tax._run_simulation(decision, n_simulations=100, time_horizon=5)

        assert np.mean(results_low["growth_rates"]) > np.mean(
            results_high["growth_rates"]
        ), "Higher tax rate should reduce average growth rate"

    def test_tax_reduces_simulation_roe(self):
        """Simulation with higher tax rate should produce lower ROE."""
        decision = self._make_decision()

        engine_low_tax = self._make_engine(tax_rate=0.10)
        engine_high_tax = self._make_engine(tax_rate=0.40)

        results_low = engine_low_tax._run_simulation(decision, n_simulations=100, time_horizon=5)
        results_high = engine_high_tax._run_simulation(decision, n_simulations=100, time_horizon=5)

        assert np.mean(results_low["roe"]) > np.mean(
            results_high["roe"]
        ), "Higher tax rate should reduce average ROE"

    def test_zero_tax_matches_pretax_income(self):
        """With tax_rate=0, results should match original pretax behavior."""
        decision = self._make_decision()

        engine_zero = self._make_engine(tax_rate=0.0)
        engine_25 = self._make_engine(tax_rate=0.25)

        results_zero = engine_zero._run_simulation(decision, n_simulations=100, time_horizon=5)
        results_25 = engine_25._run_simulation(decision, n_simulations=100, time_horizon=5)

        # Zero tax should give higher terminal equity than 25% tax
        assert np.mean(results_zero["value"]) > np.mean(
            results_25["value"]
        ), "Zero tax rate should produce higher terminal equity than 25% tax"

    def test_tax_not_applied_to_losses(self):
        """Tax expense should be zero when income_before_tax is negative."""
        # Very high loss scenario: operating margin = 1%, losses dominate
        decision = self._make_decision()
        engine = self._make_engine(tax_rate=0.25, operating_margin=0.001)

        # With tiny margin and substantial losses/premium, income_before_tax < 0
        # Tax should not make losses worse (no negative tax expense)
        results = engine._run_simulation(decision, n_simulations=100, time_horizon=3)

        # The key assertion: bankruptcy rate shouldn't be worse than with 0% tax
        engine_zero = self._make_engine(tax_rate=0.0, operating_margin=0.001)
        results_zero = engine_zero._run_simulation(decision, n_simulations=100, time_horizon=3)

        # Bankruptcy rates should be the same because tax doesn't apply to losses
        assert np.mean(results["bankruptcies"]) == np.mean(
            results_zero["bankruptcies"]
        ), "Tax should not apply to negative income; bankruptcy rates should match"

    def test_tax_rate_sourced_from_manufacturer(self):
        """Tax rate should come from manufacturer, not be hardcoded."""
        decision = self._make_decision()

        # Use a distinctive tax rate
        engine_custom = self._make_engine(tax_rate=0.42)
        engine_zero = self._make_engine(tax_rate=0.0)

        results_custom = engine_custom._run_simulation(decision, n_simulations=100, time_horizon=5)
        results_zero = engine_zero._run_simulation(decision, n_simulations=100, time_horizon=5)

        # The 42% rate should produce noticeably lower equity
        ratio = np.mean(results_custom["value"]) / np.mean(results_zero["value"])
        assert (
            ratio < 0.90
        ), f"42% tax should reduce terminal equity substantially vs 0%, got ratio={ratio:.3f}"

    def test_tax_increases_bankruptcy_probability(self):
        """Tax reduces net income, which should increase bankruptcy probability."""
        decision = self._make_decision()

        # Use a margin that's barely profitable so tax can tip into bankruptcy
        engine_zero = self._make_engine(tax_rate=0.0, operating_margin=0.02)
        engine_high = self._make_engine(tax_rate=0.50, operating_margin=0.02)

        results_zero = engine_zero._run_simulation(decision, n_simulations=200, time_horizon=10)
        results_high = engine_high._run_simulation(decision, n_simulations=200, time_horizon=10)

        assert np.mean(results_high["bankruptcies"]) >= np.mean(
            results_zero["bankruptcies"]
        ), "Higher tax rate should increase or maintain bankruptcy probability"


class TestSimulationFinancialModel:
    """Test GPU-engine-aligned financial model in _run_simulation() (Issue #1300).

    Validates depreciation, working capital, NOL carryforward, DTL/DTA,
    dividends, cash-based insolvency, and backward compatibility with mocks.
    """

    def _make_engine(
        self,
        tax_rate=0.25,
        operating_margin=0.15,
        initial_assets=10_000_000,
        equity=10_000_000,
        **extra_attrs,
    ):
        """Create a decision engine with configurable manufacturer attributes."""
        manufacturer = Mock(spec=WidgetManufacturer)
        manufacturer.current_assets = initial_assets
        manufacturer.total_assets = initial_assets
        manufacturer.equity = equity
        manufacturer.asset_turnover_ratio = 1.0
        manufacturer.base_operating_margin = operating_margin
        manufacturer.tax_rate = tax_rate
        manufacturer.step = Mock(return_value={"roe": 0.15, "assets": initial_assets})
        manufacturer.reset = Mock()

        # Apply extra attributes (e.g., ppe_ratio, retention_ratio)
        for attr, val in extra_attrs.items():
            setattr(manufacturer, attr, val)

        # Create a copy mock
        manufacturer_copy = Mock(spec=WidgetManufacturer)
        manufacturer_copy.current_assets = initial_assets
        manufacturer_copy.total_assets = initial_assets
        manufacturer_copy.equity = equity
        manufacturer_copy.asset_turnover_ratio = 1.0
        manufacturer_copy.base_operating_margin = operating_margin
        manufacturer_copy.tax_rate = tax_rate
        manufacturer_copy.step = Mock(return_value={"roe": 0.15, "assets": initial_assets})
        manufacturer_copy.reset = Mock()
        manufacturer.copy = Mock(return_value=manufacturer_copy)

        loss_dist = Mock(spec=LossDistribution)
        loss_dist.rvs = Mock(return_value=100_000)
        loss_dist.ppf = Mock(side_effect=lambda p: p * 10_000_000)
        loss_dist.expected_value = Mock(return_value=500_000)

        with patch.object(
            InsuranceDecisionEngine,
            "_load_pricing_scenarios",
            return_value=Mock(
                get_scenario=Mock(
                    return_value=Mock(
                        primary_layer_rate=0.01,
                        first_excess_rate=0.005,
                        higher_excess_rate=0.002,
                    )
                )
            ),
        ):
            engine = InsuranceDecisionEngine(
                manufacturer=manufacturer,
                loss_distribution=loss_dist,
                pricing_scenario="baseline",
            )
        return engine

    def _make_decision(self, premium=50_000):
        """Create a simple insurance decision."""
        return InsuranceDecision(
            retained_limit=500_000,
            layers=[
                Layer(
                    attachment_point=500_000,
                    limit=5_000_000,
                    base_premium_rate=0.01,
                )
            ],
            total_premium=premium,
            total_coverage=5_500_000,
            pricing_scenario="baseline",
            optimization_method="SLSQP",
        )

    def test_depreciation_affects_terminal_equity(self):
        """PP&E ratio + tax depreciation should produce DTL that affects equity."""
        decision = self._make_decision(premium=0)
        # Zero losses for deterministic comparison
        n_sims, n_years = 5, 5
        zero_losses = np.zeros((n_sims, n_years))

        # High PP&E with accelerated tax depreciation → DTL effect
        engine_high = self._make_engine(
            ppe_ratio=0.5,
            tax_depreciation_life_years=5.0,
            ppe_useful_life_years=10.0,
        )
        results_high = engine_high._run_simulation(
            decision,
            n_simulations=n_sims,
            time_horizon=n_years,
            loss_sequence=zero_losses,
        )

        # Minimal PP&E → negligible DTL
        engine_low = self._make_engine(
            ppe_ratio=0.01,
            tax_depreciation_life_years=5.0,
            ppe_useful_life_years=10.0,
        )
        results_low = engine_low._run_simulation(
            decision,
            n_simulations=n_sims,
            time_horizon=n_years,
            loss_sequence=zero_losses,
        )

        # DTL from timing difference should cause different terminal equity
        mean_high = np.mean(results_high["value"])
        mean_low = np.mean(results_low["value"])
        assert mean_high != pytest.approx(
            mean_low, rel=0.001
        ), f"Depreciation DTL should affect equity: high_ppe={mean_high:.0f}, low_ppe={mean_low:.0f}"

    def test_nol_carryforward_improves_recovery(self):
        """NOL carryforward should produce higher terminal equity after a loss year."""
        n_sims, n_years = 10, 5
        # Year 0: loss that exceeds operating income, Years 1-4: zero losses
        losses = np.zeros((n_sims, n_years))
        losses[:, 0] = 3_000_000  # Exceeds operating income → negative pre-tax

        # No-insurance decision so full loss hits the company
        decision = InsuranceDecision(
            retained_limit=10_000_000,
            layers=[],
            total_premium=0,
            total_coverage=10_000_000,
            pricing_scenario="baseline",
            optimization_method="SLSQP",
        )

        # Minimize working capital to isolate NOL effect
        engine_nol = self._make_engine(
            nol_carryforward_enabled=True,
            operating_margin=0.15,
            dso=0.0,
            dio=0.0,
            dpo=0.0,
            ppe_ratio=0.1,
        )
        results_nol = engine_nol._run_simulation(
            decision,
            n_simulations=n_sims,
            time_horizon=n_years,
            loss_sequence=losses,
        )

        engine_no_nol = self._make_engine(
            nol_carryforward_enabled=False,
            operating_margin=0.15,
            dso=0.0,
            dio=0.0,
            dpo=0.0,
            ppe_ratio=0.1,
        )
        results_no_nol = engine_no_nol._run_simulation(
            decision,
            n_simulations=n_sims,
            time_horizon=n_years,
            loss_sequence=losses,
        )

        # NOL should shelter future income from tax → higher terminal equity
        assert np.mean(results_nol["value"]) > np.mean(
            results_no_nol["value"]
        ), "NOL carryforward should improve recovery after loss years"

    def test_nol_80pct_limitation(self):
        """NOL utilization should be limited to 80% of taxable income per TCJA."""
        n_sims, n_years = 1, 5
        # Year 0: loss generating NOL; Years 1-4: profitable (zero losses)
        losses = np.zeros((n_sims, n_years))
        losses[0, 0] = 3_000_000  # Exceeds operating income → generates NOL

        # No-insurance decision so full loss hits the company
        decision = InsuranceDecision(
            retained_limit=10_000_000,
            layers=[],
            total_premium=0,
            total_coverage=10_000_000,
            pricing_scenario="baseline",
            optimization_method="SLSQP",
        )

        # Minimize working capital to isolate NOL effect
        common_kwargs = {
            "operating_margin": 0.15,
            "dso": 0.0,
            "dio": 0.0,
            "dpo": 0.0,
            "ppe_ratio": 0.1,
        }

        # With 80% limitation (TCJA default)
        engine_80 = self._make_engine(
            nol_carryforward_enabled=True,
            nol_limitation_pct=0.80,
            **common_kwargs,
        )
        results_80 = engine_80._run_simulation(
            decision,
            n_simulations=n_sims,
            time_horizon=n_years,
            loss_sequence=losses,
        )

        # With 100% limitation (pre-TCJA)
        engine_100 = self._make_engine(
            nol_carryforward_enabled=True,
            nol_limitation_pct=1.0,
            **common_kwargs,
        )
        results_100 = engine_100._run_simulation(
            decision,
            n_simulations=n_sims,
            time_horizon=n_years,
            loss_sequence=losses,
        )

        # 100% deduction should shelter more income → higher equity
        assert np.mean(results_100["value"]) >= np.mean(
            results_80["value"]
        ), "100% NOL deduction should produce >= equity than 80% limited"

    def test_cash_insolvency_detection(self):
        """High PP&E + working capital should trigger cash insolvency even with positive equity."""
        decision = self._make_decision(premium=0)
        n_sims, n_years = 20, 10

        # Very high PP&E ratio and working capital days → cash starved
        engine_cash_tight = self._make_engine(
            operating_margin=0.03,
            ppe_ratio=0.80,
            dso=120.0,
            dio=120.0,
            dpo=10.0,
            insolvency_tolerance=0.0,
        )
        results_tight = engine_cash_tight._run_simulation(
            decision,
            n_simulations=n_sims,
            time_horizon=n_years,
        )

        # Comfortable cash position
        engine_cash_ok = self._make_engine(
            operating_margin=0.03,
            ppe_ratio=0.10,
            dso=30.0,
            dio=30.0,
            dpo=60.0,
            insolvency_tolerance=0.0,
        )
        results_ok = engine_cash_ok._run_simulation(
            decision,
            n_simulations=n_sims,
            time_horizon=n_years,
        )

        # Cash-tight scenario should have more bankruptcies
        assert np.mean(results_tight["bankruptcies"]) >= np.mean(
            results_ok["bankruptcies"]
        ), "Cash-strapped scenario should trigger more insolvencies"

    def test_backward_compat_with_mocks(self):
        """Existing mock pattern (no config, no extra attrs) should complete without error."""
        # Vanilla mock — no ppe_ratio, no retention_ratio, etc.
        manufacturer = Mock(spec=WidgetManufacturer)
        manufacturer.current_assets = 10_000_000
        manufacturer.total_assets = 10_000_000
        manufacturer.equity = 10_000_000
        manufacturer.asset_turnover_ratio = 1.0
        manufacturer.base_operating_margin = 0.15
        manufacturer.tax_rate = 0.25
        manufacturer.step = Mock(return_value={"roe": 0.15, "assets": 10_000_000})
        manufacturer.reset = Mock()
        manufacturer_copy = Mock(spec=WidgetManufacturer)
        manufacturer_copy.current_assets = 10_000_000
        manufacturer_copy.total_assets = 10_000_000
        manufacturer_copy.equity = 10_000_000
        manufacturer_copy.asset_turnover_ratio = 1.0
        manufacturer_copy.base_operating_margin = 0.15
        manufacturer_copy.tax_rate = 0.25
        manufacturer_copy.step = Mock(return_value={"roe": 0.15, "assets": 10_000_000})
        manufacturer_copy.reset = Mock()
        manufacturer.copy = Mock(return_value=manufacturer_copy)

        loss_dist = Mock(spec=LossDistribution)
        loss_dist.rvs = Mock(return_value=100_000)
        loss_dist.ppf = Mock(side_effect=lambda p: p * 10_000_000)
        loss_dist.expected_value = Mock(return_value=500_000)

        with patch.object(
            InsuranceDecisionEngine,
            "_load_pricing_scenarios",
            return_value=Mock(
                get_scenario=Mock(
                    return_value=Mock(
                        primary_layer_rate=0.01,
                        first_excess_rate=0.005,
                        higher_excess_rate=0.002,
                    )
                )
            ),
        ):
            engine = InsuranceDecisionEngine(
                manufacturer=manufacturer,
                loss_distribution=loss_dist,
                pricing_scenario="baseline",
            )

        decision = self._make_decision()
        # Must not raise
        results = engine._run_simulation(decision, n_simulations=10, time_horizon=5)
        assert results["value"].shape == (10,)
        assert results["bankruptcies"].shape == (10,)

    def test_financial_model_divergence_vs_analytical(self):
        """Zero-loss deterministic scenario should track analytical balance sheet.

        With zero losses, zero premium, retention_ratio=1.0, no tax depreciation
        timing difference, and capex_ratio=1.0, equity should grow by
        net_income = revenue * margin * (1 - tax_rate) each period.
        """
        n_sims, n_years = 1, 5
        zero_losses = np.zeros((n_sims, n_years))

        margin = 0.10
        tax_rate = 0.25
        initial_assets = 10_000_000.0

        engine = self._make_engine(
            operating_margin=margin,
            tax_rate=tax_rate,
            initial_assets=initial_assets,
            equity=initial_assets,
            ppe_ratio=0.3,
            ppe_useful_life_years=10.0,
            tax_depreciation_life_years=None,  # no DTL
            capex_to_depreciation_ratio=1.0,
            retention_ratio=1.0,
            nol_carryforward_enabled=True,
            insolvency_tolerance=0.0,
            dpo=30.0,
            dso=45.0,
            dio=60.0,
        )

        decision = self._make_decision(premium=0)
        results = engine._run_simulation(
            decision,
            n_simulations=n_sims,
            time_horizon=n_years,
            loss_sequence=zero_losses,
        )

        # Analytically trace equity: each year equity grows by
        # retained_earnings (= revenue * margin * (1-tax))
        # and assets derive from equity + liabilities.
        # With no tax timing diff, delta_net_dtl ≈ 0, so equity += net_income.
        equity = initial_assets
        prev_ap = 0.0
        prev_net_dtl = 0.0
        for _ in range(n_years):
            assets = equity + prev_ap + prev_net_dtl
            revenue = assets * 1.0  # asset_turnover=1
            operating_income = revenue * margin
            net_income = operating_income * (1.0 - tax_rate)
            equity += net_income
            cogs = revenue * (1.0 - margin)
            prev_ap = cogs * (30.0 / 365.0)

        sim_equity = results["value"][0]
        # Allow < 10% divergence
        rel_diff = abs(sim_equity - equity) / max(abs(equity), 1.0)
        assert rel_diff < 0.10, (
            f"Financial model divergence too large: simulated={sim_equity:.0f}, "
            f"analytical={equity:.0f}, rel_diff={rel_diff:.4f}"
        )


class TestFromCompany:
    """Test InsuranceDecisionEngine.from_company() factory method."""

    def test_default_parameters(self):
        """Factory with all defaults returns a usable engine."""
        engine = InsuranceDecisionEngine.from_company()

        assert isinstance(engine, InsuranceDecisionEngine)
        assert engine.manufacturer.config.initial_assets == 10_000_000
        assert isinstance(engine.loss_distribution, LognormalLoss)
        assert engine.loss_distribution.mean == 1_000_000
        assert engine.loss_distribution.cv == 1.5
        assert engine.pricing_scenario == "baseline"

    def test_custom_assets_and_losses(self):
        """Factory accepts custom asset and loss parameters."""
        engine = InsuranceDecisionEngine.from_company(
            initial_assets=50_000_000,
            loss_mean=2_000_000,
            loss_cv=2.0,
        )

        assert engine.manufacturer.config.initial_assets == 50_000_000
        assert isinstance(engine.loss_distribution, LognormalLoss)
        assert engine.loss_distribution.mean == 2_000_000
        assert engine.loss_distribution.cv == 2.0

    def test_custom_company_parameters(self):
        """Factory passes company parameters to ManufacturerConfig."""
        engine = InsuranceDecisionEngine.from_company(
            initial_assets=25_000_000,
            operating_margin=0.12,
            tax_rate=0.21,
        )

        assert engine.manufacturer.config.initial_assets == 25_000_000
        assert engine.manufacturer.config.base_operating_margin == 0.12
        assert engine.manufacturer.config.tax_rate == 0.21

    def test_pricing_scenario(self):
        """Factory accepts pricing scenario."""
        engine = InsuranceDecisionEngine.from_company(
            pricing_scenario="inexpensive",
        )

        assert engine.pricing_scenario == "inexpensive"

    def test_seed_reproducibility(self):
        """Seed parameter produces reproducible loss distributions."""
        engine1 = InsuranceDecisionEngine.from_company(seed=42)
        engine2 = InsuranceDecisionEngine.from_company(seed=42)

        samples1 = engine1.loss_distribution.generate_severity(100)
        samples2 = engine2.loss_distribution.generate_severity(100)
        np.testing.assert_array_equal(samples1, samples2)

    def test_returns_engine_type(self):
        """Factory returns InsuranceDecisionEngine instance."""
        engine = InsuranceDecisionEngine.from_company()
        assert isinstance(engine, InsuranceDecisionEngine)
        assert engine.config_manager is not None
        assert engine.engine_config is not None

    def test_existing_init_unchanged(self):
        """Original __init__ API still works for advanced users."""
        config = ManufacturerConfig(initial_assets=5_000_000)
        manufacturer = WidgetManufacturer(config)
        loss_dist = Mock(spec=LossDistribution)
        loss_dist.expected_value.return_value = 500_000

        engine = InsuranceDecisionEngine(manufacturer, loss_dist)

        assert engine.manufacturer is manufacturer
        assert engine.loss_distribution is loss_dist


class TestOptimizeConvenience:
    """Test InsuranceDecisionEngine.optimize() convenience method."""

    def _make_engine(self):
        """Create engine with mocked loss distribution for fast tests."""
        config = ManufacturerConfig(initial_assets=10_000_000)
        manufacturer = WidgetManufacturer(config)
        loss_dist = Mock(spec=LossDistribution)
        loss_dist.expected_value.return_value = 500_000
        loss_dist.generate_severity.return_value = np.full(1000, 500_000.0)
        return InsuranceDecisionEngine(manufacturer, loss_dist)

    def test_optimize_with_max_premium(self):
        """optimize() with explicit max_premium builds correct constraints."""
        engine = self._make_engine()

        with patch.object(engine, "optimize_insurance_decision") as mock_opt:
            mock_opt.return_value = InsuranceDecision(
                retained_limit=1_000_000,
                layers=[],
                total_premium=0,
                total_coverage=0,
                pricing_scenario="baseline",
                optimization_method="SLSQP",
            )

            engine.optimize(max_premium=500_000)

            call_args = mock_opt.call_args
            constraints = call_args[0][0]
            assert constraints.max_premium_budget == 500_000
            assert constraints.max_bankruptcy_probability == 0.01

    def test_optimize_default_premium(self):
        """optimize() without max_premium defaults to 10% of revenue."""
        engine = self._make_engine()
        # revenue = initial_assets * asset_turnover_ratio = 10M * 0.8 = 8M
        # default max_premium = 8M * 0.10 = 800_000
        expected_premium = (
            engine.manufacturer.config.initial_assets
            * engine.manufacturer.config.asset_turnover_ratio
            * 0.10
        )

        with patch.object(engine, "optimize_insurance_decision") as mock_opt:
            mock_opt.return_value = InsuranceDecision(
                retained_limit=1_000_000,
                layers=[],
                total_premium=0,
                total_coverage=0,
                pricing_scenario="baseline",
                optimization_method="SLSQP",
            )

            engine.optimize()

            constraints = mock_opt.call_args[0][0]
            assert constraints.max_premium_budget == expected_premium

    def test_optimize_passes_method(self):
        """optimize() forwards method parameter."""
        engine = self._make_engine()

        with patch.object(engine, "optimize_insurance_decision") as mock_opt:
            mock_opt.return_value = InsuranceDecision(
                retained_limit=1_000_000,
                layers=[],
                total_premium=0,
                total_coverage=0,
                pricing_scenario="baseline",
                optimization_method="DE",
            )

            engine.optimize(
                max_premium=500_000,
                method=OptimizationMethod.DIFFERENTIAL_EVOLUTION,
            )

            call_kwargs = mock_opt.call_args[1]
            assert call_kwargs["method"] == OptimizationMethod.DIFFERENTIAL_EVOLUTION

    def test_optimize_passes_weights(self):
        """optimize() forwards weights parameter."""
        engine = self._make_engine()
        custom_weights = {"growth": 0.5, "risk": 0.3, "cost": 0.2}

        with patch.object(engine, "optimize_insurance_decision") as mock_opt:
            mock_opt.return_value = InsuranceDecision(
                retained_limit=1_000_000,
                layers=[],
                total_premium=0,
                total_coverage=0,
                pricing_scenario="baseline",
                optimization_method="SLSQP",
            )

            engine.optimize(max_premium=500_000, weights=custom_weights)

            call_kwargs = mock_opt.call_args[1]
            assert call_kwargs["weights"] == custom_weights

    def test_optimize_constraint_overrides(self):
        """optimize() passes extra kwargs to DecisionOptimizationConstraints."""
        engine = self._make_engine()

        with patch.object(engine, "optimize_insurance_decision") as mock_opt:
            mock_opt.return_value = InsuranceDecision(
                retained_limit=1_000_000,
                layers=[],
                total_premium=0,
                total_coverage=0,
                pricing_scenario="baseline",
                optimization_method="SLSQP",
            )

            engine.optimize(
                max_premium=500_000,
                min_total_coverage=10_000_000,
                max_layers=3,
            )

            constraints = mock_opt.call_args[0][0]
            assert constraints.min_total_coverage == 10_000_000
            assert constraints.max_layers == 3
