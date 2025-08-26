"""Tests for premium pricing scenario framework."""

from pathlib import Path
from typing import Any, Dict

import pytest

from ergodic_insurance.src.config import (
    MarketCycles,
    PricingScenario,
    PricingScenarioConfig,
    TransitionProbabilities,
)
from ergodic_insurance.src.config_loader import ConfigLoader


class TestPricingScenario:
    """Test individual pricing scenario configuration."""

    def test_soft_market_scenario(self):
        """Test soft market scenario creation."""
        scenario = PricingScenario(
            name="Soft Market",
            description="Buyer's market",
            market_condition="soft",
            primary_layer_rate=0.005,
            first_excess_rate=0.003,
            higher_excess_rate=0.001,
            capacity_factor=1.5,
            competition_level="high",
            retention_discount=0.1,
            volume_discount=0.05,
            loss_ratio_target=0.65,
            expense_ratio=0.30,
            new_business_appetite="aggressive",
            renewal_retention_focus="low",
            coverage_enhancement_willingness="high",
        )

        assert scenario.name == "Soft Market"
        assert scenario.market_condition == "soft"
        assert scenario.primary_layer_rate == 0.005
        assert scenario.capacity_factor == 1.5

    def test_rate_ordering_validation(self):
        """Test that rate ordering is validated."""
        # Valid ordering
        scenario = PricingScenario(
            name="Test",
            description="Test scenario",
            market_condition="normal",
            primary_layer_rate=0.010,
            first_excess_rate=0.005,
            higher_excess_rate=0.002,
            capacity_factor=1.0,
            competition_level="moderate",
            retention_discount=0.05,
            volume_discount=0.025,
            loss_ratio_target=0.60,
            expense_ratio=0.28,
            new_business_appetite="selective",
            renewal_retention_focus="balanced",
            coverage_enhancement_willingness="moderate",
        )
        assert scenario.primary_layer_rate >= scenario.first_excess_rate
        assert scenario.first_excess_rate >= scenario.higher_excess_rate

        # Invalid ordering should raise error
        with pytest.raises(ValueError, match="Rate ordering violation"):
            PricingScenario(
                name="Invalid",
                description="Invalid scenario",
                market_condition="normal",
                primary_layer_rate=0.002,  # Lower than excess!
                first_excess_rate=0.005,
                higher_excess_rate=0.001,
                capacity_factor=1.0,
                competition_level="moderate",
                retention_discount=0.05,
                volume_discount=0.025,
                loss_ratio_target=0.60,
                expense_ratio=0.28,
                new_business_appetite="selective",
                renewal_retention_focus="balanced",
                coverage_enhancement_willingness="moderate",
            )

    def test_capacity_factor_bounds(self):
        """Test capacity factor validation bounds."""
        # Valid capacity factor
        scenario = PricingScenario(
            name="Test",
            description="Test",
            market_condition="normal",
            primary_layer_rate=0.010,
            first_excess_rate=0.005,
            higher_excess_rate=0.002,
            capacity_factor=1.0,
            competition_level="moderate",
            retention_discount=0.05,
            volume_discount=0.025,
            loss_ratio_target=0.60,
            expense_ratio=0.28,
            new_business_appetite="selective",
            renewal_retention_focus="balanced",
            coverage_enhancement_willingness="moderate",
        )
        assert 0.5 <= scenario.capacity_factor <= 2.0

        # Test bounds - capacity_factor is validated at creation time, not assignment
        with pytest.raises(ValueError):
            PricingScenario(
                name="Test",
                description="Test",
                market_condition="normal",
                primary_layer_rate=0.010,
                first_excess_rate=0.005,
                higher_excess_rate=0.002,
                capacity_factor=0.3,  # Too low
                competition_level="moderate",
                retention_discount=0.05,
                volume_discount=0.025,
                loss_ratio_target=0.60,
                expense_ratio=0.28,
                new_business_appetite="selective",
                renewal_retention_focus="balanced",
                coverage_enhancement_willingness="moderate",
            )


class TestTransitionProbabilities:
    """Test market transition probability configuration."""

    def test_valid_transitions(self):
        """Test valid transition probability configuration."""
        transitions = TransitionProbabilities(
            soft_to_soft=0.6,
            soft_to_normal=0.35,
            soft_to_hard=0.05,
            normal_to_soft=0.15,
            normal_to_normal=0.65,
            normal_to_hard=0.20,
            hard_to_soft=0.0,
            hard_to_normal=0.45,
            hard_to_hard=0.55,
        )

        # Check individual probabilities
        assert transitions.soft_to_soft == 0.6
        assert transitions.normal_to_hard == 0.20
        assert transitions.hard_to_soft == 0.0

    def test_probability_sum_validation(self):
        """Test that transition probabilities sum to 1.0."""
        # Invalid soft market transitions (sum != 1.0)
        with pytest.raises(ValueError, match="Soft market transitions"):
            TransitionProbabilities(
                soft_to_soft=0.5,
                soft_to_normal=0.3,
                soft_to_hard=0.1,  # Sum = 0.9, not 1.0
                normal_to_soft=0.15,
                normal_to_normal=0.65,
                normal_to_hard=0.20,
                hard_to_soft=0.0,
                hard_to_normal=0.45,
                hard_to_hard=0.55,
            )

        # Invalid normal market transitions
        with pytest.raises(ValueError, match="Normal market transitions"):
            TransitionProbabilities(
                soft_to_soft=0.6,
                soft_to_normal=0.35,
                soft_to_hard=0.05,
                normal_to_soft=0.2,
                normal_to_normal=0.7,
                normal_to_hard=0.2,  # Sum = 1.1, not 1.0
                hard_to_soft=0.0,
                hard_to_normal=0.45,
                hard_to_hard=0.55,
            )

    def test_probability_bounds(self):
        """Test that probabilities are between 0 and 1."""
        with pytest.raises(ValueError):
            TransitionProbabilities(
                soft_to_soft=-0.1,  # Negative probability
                soft_to_normal=0.6,
                soft_to_hard=0.5,
                normal_to_soft=0.15,
                normal_to_normal=0.65,
                normal_to_hard=0.20,
                hard_to_soft=0.0,
                hard_to_normal=0.45,
                hard_to_hard=0.55,
            )


class TestMarketCycles:
    """Test market cycle configuration."""

    def test_cycle_creation(self):
        """Test market cycle configuration creation."""
        transitions = TransitionProbabilities(
            soft_to_soft=0.6,
            soft_to_normal=0.35,
            soft_to_hard=0.05,
            normal_to_soft=0.15,
            normal_to_normal=0.65,
            normal_to_hard=0.20,
            hard_to_soft=0.0,
            hard_to_normal=0.45,
            hard_to_hard=0.55,
        )

        cycles = MarketCycles(
            average_duration_years=8,
            soft_market_duration=3.5,
            normal_market_duration=3.0,
            hard_market_duration=1.5,
            transition_probabilities=transitions,
        )

        assert cycles.average_duration_years == 8
        assert cycles.soft_market_duration == 3.5
        assert cycles.transition_probabilities.soft_to_normal == 0.35

    def test_duration_validation_warning(self, capsys):
        """Test that duration validation produces warning."""
        transitions = TransitionProbabilities(
            soft_to_soft=0.6,
            soft_to_normal=0.35,
            soft_to_hard=0.05,
            normal_to_soft=0.15,
            normal_to_normal=0.65,
            normal_to_hard=0.20,
            hard_to_soft=0.0,
            hard_to_normal=0.45,
            hard_to_hard=0.55,
        )

        # Create cycles with mismatched durations
        cycles = MarketCycles(
            average_duration_years=15,  # Much higher than component average
            soft_market_duration=3.0,
            normal_market_duration=3.0,
            hard_market_duration=2.0,  # Component avg = 8/3 = 2.67
            transition_probabilities=transitions,
        )

        # Check warning was printed
        captured = capsys.readouterr()
        assert "Warning: Average duration" in captured.out


class TestPricingScenarioConfig:
    """Test complete pricing scenario configuration."""

    @pytest.fixture
    def sample_config(self) -> PricingScenarioConfig:
        """Create sample pricing scenario configuration."""
        soft_scenario = PricingScenario(
            name="Soft Market",
            description="Buyer's market",
            market_condition="soft",
            primary_layer_rate=0.005,
            first_excess_rate=0.003,
            higher_excess_rate=0.001,
            capacity_factor=1.5,
            competition_level="high",
            retention_discount=0.1,
            volume_discount=0.05,
            loss_ratio_target=0.65,
            expense_ratio=0.30,
            new_business_appetite="aggressive",
            renewal_retention_focus="low",
            coverage_enhancement_willingness="high",
        )

        normal_scenario = PricingScenario(
            name="Normal Market",
            description="Balanced market",
            market_condition="normal",
            primary_layer_rate=0.010,
            first_excess_rate=0.005,
            higher_excess_rate=0.002,
            capacity_factor=1.0,
            competition_level="moderate",
            retention_discount=0.05,
            volume_discount=0.025,
            loss_ratio_target=0.60,
            expense_ratio=0.28,
            new_business_appetite="selective",
            renewal_retention_focus="balanced",
            coverage_enhancement_willingness="moderate",
        )

        transitions = TransitionProbabilities(
            soft_to_soft=0.6,
            soft_to_normal=0.35,
            soft_to_hard=0.05,
            normal_to_soft=0.15,
            normal_to_normal=0.65,
            normal_to_hard=0.20,
            hard_to_soft=0.0,
            hard_to_normal=0.45,
            hard_to_hard=0.55,
        )

        cycles = MarketCycles(
            average_duration_years=8,
            soft_market_duration=3.5,
            normal_market_duration=3.0,
            hard_market_duration=1.5,
            transition_probabilities=transitions,
        )

        return PricingScenarioConfig(
            scenarios={"inexpensive": soft_scenario, "baseline": normal_scenario},
            market_cycles=cycles,
        )

    def test_get_scenario(self, sample_config):
        """Test retrieving scenarios by name."""
        soft = sample_config.get_scenario("inexpensive")
        assert soft.name == "Soft Market"
        assert soft.primary_layer_rate == 0.005

        normal = sample_config.get_scenario("baseline")
        assert normal.name == "Normal Market"
        assert normal.primary_layer_rate == 0.010

        # Test invalid scenario name
        with pytest.raises(KeyError, match="Scenario 'invalid' not found"):
            sample_config.get_scenario("invalid")

    def test_get_rate_multiplier(self, sample_config):
        """Test calculating rate multipliers between scenarios."""
        # Soft to Normal should increase rates
        multiplier = sample_config.get_rate_multiplier("inexpensive", "baseline")
        assert multiplier > 1.0  # Rates should increase

        # Calculate expected multiplier
        primary_mult = 0.010 / 0.005  # 2.0
        excess_mult = 0.005 / 0.003  # 1.667
        higher_mult = 0.002 / 0.001  # 2.0
        expected = (primary_mult + excess_mult + higher_mult) / 3

        assert abs(multiplier - expected) < 0.001

        # Normal to Soft should decrease rates
        reverse_mult = sample_config.get_rate_multiplier("baseline", "inexpensive")
        assert reverse_mult < 1.0  # Rates should decrease
        assert abs(reverse_mult * multiplier - 1.0) < 0.01  # Should be reciprocal


@pytest.mark.filterwarnings("ignore:ConfigLoader is deprecated:DeprecationWarning")
class TestConfigLoaderIntegration:
    """Test ConfigLoader integration with pricing scenarios."""

    def test_load_pricing_scenarios(self):
        """Test loading pricing scenarios from YAML file."""
        loader = ConfigLoader()

        # Load the actual scenario file we created
        pricing_config = loader.load_pricing_scenarios("insurance_pricing_scenarios")

        # Verify scenarios loaded correctly
        assert "inexpensive" in pricing_config.scenarios
        assert "baseline" in pricing_config.scenarios
        assert "expensive" in pricing_config.scenarios

        # Check soft market scenario
        soft = pricing_config.scenarios["inexpensive"]
        assert soft.market_condition == "soft"
        assert soft.primary_layer_rate == 0.005
        assert soft.capacity_factor == 1.5

        # Check normal market scenario
        normal = pricing_config.scenarios["baseline"]
        assert normal.market_condition == "normal"
        assert normal.primary_layer_rate == 0.010
        assert normal.capacity_factor == 1.0

        # Check hard market scenario
        hard = pricing_config.scenarios["expensive"]
        assert hard.market_condition == "hard"
        assert hard.primary_layer_rate == 0.015
        assert hard.capacity_factor == 0.7

        # Check market cycles
        assert pricing_config.market_cycles.average_duration_years == 8
        assert pricing_config.market_cycles.soft_market_duration == 3.5

        # Check transition probabilities
        trans = pricing_config.market_cycles.transition_probabilities
        assert trans.soft_to_normal == 0.35
        assert trans.normal_to_hard == 0.20
        assert trans.hard_to_soft == 0.0

    def test_scenario_switching(self):
        """Test switching between pricing scenarios."""
        loader = ConfigLoader()

        # Load base configuration
        config = loader.load("baseline")

        # Switch to soft market scenario
        soft_config = loader.switch_pricing_scenario(config, "inexpensive")

        # Config should still be valid
        assert soft_config.manufacturer.initial_assets == config.manufacturer.initial_assets

        # Switch to hard market scenario
        hard_config = loader.switch_pricing_scenario(config, "expensive")

        # Config should still be valid
        assert hard_config.manufacturer.initial_assets == config.manufacturer.initial_assets

    def test_invalid_scenario_file(self):
        """Test loading non-existent scenario file."""
        loader = ConfigLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_pricing_scenarios("non_existent_file")

    def test_scenario_comparison(self):
        """Test comparing scenarios for analysis."""
        loader = ConfigLoader()
        pricing_config = loader.load_pricing_scenarios()

        # Compare soft and hard markets
        soft = pricing_config.scenarios["inexpensive"]
        hard = pricing_config.scenarios["expensive"]

        # Premium differences
        primary_diff = hard.primary_layer_rate / soft.primary_layer_rate
        assert primary_diff == 3.0  # 1.5% / 0.5% = 3x

        # Capacity differences
        capacity_ratio = soft.capacity_factor / hard.capacity_factor
        assert abs(capacity_ratio - 2.14) < 0.01  # 1.5 / 0.7 ≈ 2.14

        # Competition differences
        assert soft.competition_level == "high"
        assert hard.competition_level == "low"
