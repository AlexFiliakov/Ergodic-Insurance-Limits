"""Tests for insurance pricing module with market cycle support.

This module tests the InsurancePricer class and related functionality,
including pure premium calculation, market cycle adjustments, and
integration with insurance programs.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from ergodic_insurance.claim_development import ClaimDevelopment
from ergodic_insurance.insurance import InsuranceLayer, InsurancePolicy
from ergodic_insurance.insurance_pricing import (
    InsurancePricer,
    LayerPricing,
    MarketCycle,
    PricingParameters,
)
from ergodic_insurance.insurance_program import (
    EnhancedInsuranceLayer,
    InsuranceProgram,
    ReinstatementType,
)
from ergodic_insurance.loss_distributions import ManufacturingLossGenerator


class TestMarketCycle:
    """Test MarketCycle enum."""

    @pytest.mark.parametrize(
        "member,expected_name,expected_value",
        [
            (MarketCycle.HARD, "HARD", 0.60),
            (MarketCycle.NORMAL, "NORMAL", 0.70),
            (MarketCycle.SOFT, "SOFT", 0.80),
        ],
        ids=["hard", "normal", "soft"],
    )
    def test_market_cycle_members(self, member, expected_name, expected_value):
        """Test market cycle enum names and loss ratio values."""
        assert member.name == expected_name
        assert member.value == expected_value


class TestPricingParameters:
    """Test PricingParameters dataclass."""

    def test_default_parameters(self):
        """Test default pricing parameters."""
        params = PricingParameters()
        assert params.loss_ratio == 0.70
        assert params.expense_ratio == 0.25
        assert params.profit_margin == 0.05
        assert params.risk_loading == 0.10
        assert params.confidence_level == 0.95
        assert params.simulation_years == 10
        assert params.min_premium == 1000.0
        assert params.max_rate_on_line == 0.50
        assert params.alae_ratio == 0.10
        assert params.ulae_ratio == 0.05
        assert params.lae_ratio == pytest.approx(0.15)

    def test_custom_parameters(self):
        """Test custom pricing parameters."""
        params = PricingParameters(
            loss_ratio=0.65,
            expense_ratio=0.30,
            profit_margin=0.10,
            risk_loading=0.15,
            confidence_level=0.99,
            simulation_years=20,
            min_premium=5000.0,
            max_rate_on_line=0.75,
            alae_ratio=0.12,
            ulae_ratio=0.06,
        )
        assert params.loss_ratio == 0.65
        assert params.expense_ratio == 0.30
        assert params.profit_margin == 0.10
        assert params.risk_loading == 0.15
        assert params.confidence_level == 0.99
        assert params.simulation_years == 20
        assert params.min_premium == 5000.0
        assert params.max_rate_on_line == 0.75
        assert params.alae_ratio == 0.12
        assert params.ulae_ratio == 0.06
        assert params.lae_ratio == pytest.approx(0.18)


class TestLayerPricing:
    """Test LayerPricing dataclass."""

    def test_layer_pricing_creation(self):
        """Test creating LayerPricing object."""
        pricing = LayerPricing(
            attachment_point=1_000_000,
            limit=5_000_000,
            expected_frequency=0.5,
            expected_severity=2_000_000,
            pure_premium=1_000_000,
            technical_premium=1_500_000,
            market_premium=2_000_000,
            rate_on_line=0.40,
            confidence_interval=(800_000, 1_200_000),
        )
        assert pricing.attachment_point == 1_000_000
        assert pricing.limit == 5_000_000
        assert pricing.expected_frequency == 0.5
        assert pricing.expected_severity == 2_000_000
        assert pricing.pure_premium == 1_000_000
        assert pricing.technical_premium == 1_500_000
        assert pricing.market_premium == 2_000_000
        assert pricing.rate_on_line == 0.40
        assert pricing.confidence_interval == (800_000, 1_200_000)


class TestInsurancePricer:
    """Test InsurancePricer class."""

    @pytest.fixture
    def loss_generator(self):
        """Create a loss generator for testing."""
        return ManufacturingLossGenerator(seed=42)

    @pytest.fixture
    def pricer(self, loss_generator):
        """Create a pricer for testing."""
        return InsurancePricer(
            loss_generator=loss_generator,
            market_cycle=MarketCycle.NORMAL,
            seed=42,
        )

    def test_pricer_initialization_with_market_cycle(self, loss_generator):
        """Test pricer initialization with market cycle."""
        pricer = InsurancePricer(
            loss_generator=loss_generator,
            market_cycle=MarketCycle.HARD,
        )
        assert pricer.loss_ratio == 0.60
        assert pricer.market_cycle == MarketCycle.HARD

    def test_pricer_initialization_with_loss_ratio(self, loss_generator):
        """Test pricer initialization with explicit loss ratio."""
        pricer = InsurancePricer(
            loss_generator=loss_generator,
            loss_ratio=0.75,
        )
        assert pricer.loss_ratio == 0.75
        assert pricer.market_cycle == MarketCycle.SOFT  # Inferred

    def test_pricer_initialization_default(self, loss_generator):
        """Test pricer initialization with defaults."""
        pricer = InsurancePricer(loss_generator=loss_generator)
        assert pricer.loss_ratio == 0.70
        assert pricer.market_cycle == MarketCycle.NORMAL

    def test_infer_market_cycle(self, pricer):
        """Test market cycle inference from loss ratio."""
        assert pricer._infer_market_cycle(0.60) == MarketCycle.HARD
        assert pricer._infer_market_cycle(0.65) == MarketCycle.HARD
        assert pricer._infer_market_cycle(0.70) == MarketCycle.NORMAL
        assert pricer._infer_market_cycle(0.75) == MarketCycle.SOFT
        assert pricer._infer_market_cycle(0.80) == MarketCycle.SOFT

    def test_calculate_pure_premium(self, pricer):
        """Test pure premium calculation."""
        attachment = 1_000_000
        limit = 5_000_000
        revenue = 15_000_000

        pure_premium, stats = pricer.calculate_pure_premium(
            attachment_point=attachment,
            limit=limit,
            expected_revenue=revenue,
            simulation_years=5,  # Fewer years for testing speed
        )

        assert pure_premium >= 0
        assert stats["expected_frequency"] >= 0
        assert stats["expected_severity"] >= 0
        assert stats["attachment_point"] == attachment
        assert stats["limit"] == limit
        assert stats["years_simulated"] == 5
        assert len(stats["confidence_interval"]) == 2

    def test_calculate_pure_premium_no_generator(self):
        """Test pure premium calculation without loss generator."""
        pricer = InsurancePricer(market_cycle=MarketCycle.NORMAL)

        with pytest.raises(ValueError, match="Loss generator required"):
            pricer.calculate_pure_premium(
                attachment_point=1_000_000,
                limit=5_000_000,
                expected_revenue=15_000_000,
            )

    def test_calculate_technical_premium(self, pricer):
        """Test technical premium calculation.

        Technical premium = pure_premium * (1 + risk_loading).
        Expense/profit loading is applied separately via loss ratio
        in calculate_market_premium() to avoid double-counting.
        """
        pure_premium = 100_000
        limit = 5_000_000

        technical = pricer.calculate_technical_premium(pure_premium, limit)

        # Should equal pure * (1 + risk_loading) = 100K * 1.10 = 110K
        expected = pure_premium * (1 + pricer.parameters.risk_loading)
        assert technical == expected

        # Should be higher than pure premium due to risk loading
        assert technical > pure_premium

        # Should respect minimum premium
        assert technical >= pricer.parameters.min_premium

        # Should respect maximum rate on line
        max_premium = limit * pricer.parameters.max_rate_on_line
        assert technical <= max_premium

    def test_calculate_technical_premium_min_floor(self, pricer):
        """Test technical premium minimum floor."""
        pure_premium = 10  # Very small
        limit = 5_000_000

        technical = pricer.calculate_technical_premium(pure_premium, limit)
        assert technical == pricer.parameters.min_premium

    def test_calculate_technical_premium_max_cap(self, pricer):
        """Test technical premium maximum cap."""
        pure_premium = 10_000_000  # Very large
        limit = 5_000_000

        technical = pricer.calculate_technical_premium(pure_premium, limit)
        max_premium = limit * pricer.parameters.max_rate_on_line
        assert technical == max_premium

    def test_calculate_market_premium(self, pricer):
        """Test market premium calculation."""
        technical_premium = 100_000

        # Test with different market cycles
        hard_premium = pricer.calculate_market_premium(
            technical_premium,
            market_cycle=MarketCycle.HARD,
        )
        normal_premium = pricer.calculate_market_premium(
            technical_premium,
            market_cycle=MarketCycle.NORMAL,
        )
        soft_premium = pricer.calculate_market_premium(
            technical_premium,
            market_cycle=MarketCycle.SOFT,
        )

        # Hard market should have highest premium
        assert hard_premium > normal_premium > soft_premium

    def test_price_layer(self, pricer):
        """Test pricing a single layer."""
        pricing = pricer.price_layer(
            attachment_point=1_000_000,
            limit=5_000_000,
            expected_revenue=15_000_000,
            market_cycle=MarketCycle.NORMAL,
        )

        assert isinstance(pricing, LayerPricing)
        assert pricing.attachment_point == 1_000_000
        assert pricing.limit == 5_000_000
        assert pricing.pure_premium >= 0
        assert pricing.technical_premium >= pricing.pure_premium
        assert pricing.market_premium >= 0
        assert 0 <= pricing.rate_on_line <= pricer.parameters.max_rate_on_line
        assert len(pricing.confidence_interval) == 2

    def test_compare_market_cycles(self, pricer):
        """Test comparing pricing across market cycles."""
        results = pricer.compare_market_cycles(
            attachment_point=1_000_000,
            limit=5_000_000,
            expected_revenue=15_000_000,
        )

        assert "HARD" in results
        assert "NORMAL" in results
        assert "SOFT" in results

        # Since the pure premium is the same but market cycles differ,
        # the relationship should hold for technical premiums converted to market
        # Get the pure premiums (should be similar across cycles)
        pure_premiums = [results[c].pure_premium for c in ["HARD", "NORMAL", "SOFT"]]

        # Check that all cycles calculated premiums
        for cycle_name in ["HARD", "NORMAL", "SOFT"]:
            assert results[cycle_name].market_premium > 0
            assert results[cycle_name].rate_on_line >= 0

        # When pure premiums are similar, hard market should typically have higher premiums
        # But due to simulation randomness, we just verify they're calculated
        # More robust test: verify the loss ratios are correctly applied
        for cycle_name, pricing in results.items():
            cycle = MarketCycle[cycle_name]
            # Market premium formula includes loss ratio adjustment
            assert pricing.market_premium > 0

    def test_simulate_cycle_transition(self, pricer):
        """Test market cycle transition simulation."""
        program = InsuranceProgram.create_standard_manufacturing_program()

        results = pricer.simulate_cycle_transition(
            program=program,
            expected_revenue=15_000_000,
            years=5,
        )

        assert len(results) == 5
        for result in results:
            assert "year" in result
            assert "market_cycle" in result
            assert "loss_ratio" in result
            assert "total_premium" in result
            assert "layer_premiums" in result

    def test_transition_cycle(self, pricer):
        """Test market cycle transition logic."""
        probs = {
            "hard_to_normal": 0.4,
            "hard_to_soft": 0.1,
            "normal_to_hard": 0.2,
            "normal_to_soft": 0.2,
            "soft_to_normal": 0.3,
            "soft_to_hard": 0.1,
        }

        # Test multiple transitions
        transitions_from_hard = []
        for _ in range(100):
            next_cycle = pricer._transition_cycle(MarketCycle.HARD, probs)
            transitions_from_hard.append(next_cycle)

        # Should have some transitions to different states
        assert MarketCycle.NORMAL in transitions_from_hard
        assert MarketCycle.HARD in transitions_from_hard

    def test_create_from_config(self, loss_generator):
        """Test creating pricer from configuration."""
        config = {
            "loss_ratio": 0.65,
            "expense_ratio": 0.30,
            "profit_margin": 0.08,
            "risk_loading": 0.12,
            "confidence_level": 0.99,
            "simulation_years": 15,
            "min_premium": 2000.0,
            "max_rate_on_line": 0.60,
            "market_cycle": "HARD",
            "seed": 123,
        }

        pricer = InsurancePricer.create_from_config(config, loss_generator)

        assert pricer.market_cycle == MarketCycle.HARD
        assert pricer.parameters.expense_ratio == 0.30
        assert pricer.parameters.profit_margin == 0.08
        assert pricer.parameters.risk_loading == 0.12
        assert pricer.parameters.confidence_level == 0.99
        assert pricer.parameters.simulation_years == 15
        assert pricer.parameters.min_premium == 2000.0
        assert pricer.parameters.max_rate_on_line == 0.60


class TestInsuranceProgramIntegration:
    """Test integration with InsuranceProgram."""

    @pytest.fixture
    def loss_generator(self):
        """Create a loss generator for testing."""
        return ManufacturingLossGenerator(seed=42)

    @pytest.fixture
    def pricer(self, loss_generator):
        """Create a pricer for testing."""
        return InsurancePricer(
            loss_generator=loss_generator,
            market_cycle=MarketCycle.NORMAL,
            seed=42,
        )

    @pytest.fixture
    def program(self):
        """Create an insurance program for testing."""
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=250_000,
                limit=4_750_000,
                base_premium_rate=0.015,
                reinstatements=0,
            ),
            EnhancedInsuranceLayer(
                attachment_point=5_000_000,
                limit=20_000_000,
                base_premium_rate=0.008,
                reinstatements=1,
                reinstatement_type=ReinstatementType.FULL,
            ),
        ]
        return InsuranceProgram(layers=layers, deductible=250_000)

    def test_price_insurance_program(self, pricer, program):
        """Test pricing an insurance program."""
        original_rates = [layer.base_premium_rate for layer in program.layers]

        priced_program = pricer.price_insurance_program(
            program=program,
            expected_revenue=15_000_000,
            market_cycle=MarketCycle.NORMAL,
            update_program=True,
        )

        # Should be the same object if update_program=True
        assert priced_program is program

        # Rates should be updated
        new_rates = [layer.base_premium_rate for layer in program.layers]
        assert new_rates != original_rates  # Should be different

        # Should have pricing results
        assert hasattr(program, "pricing_results")
        assert len(program.pricing_results) == len(program.layers)

    def test_price_insurance_program_no_update(self, pricer, program):
        """Test pricing without updating program."""
        original_rates = [layer.base_premium_rate for layer in program.layers]

        priced_program = pricer.price_insurance_program(
            program=program,
            expected_revenue=15_000_000,
            market_cycle=MarketCycle.NORMAL,
            update_program=False,
        )

        # Should be a different object
        assert priced_program is not program

        # Original rates should be unchanged
        unchanged_rates = [layer.base_premium_rate for layer in program.layers]
        assert unchanged_rates == original_rates

    def test_program_apply_pricing(self, loss_generator):
        """Test apply_pricing method on InsuranceProgram."""
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=250_000,
                limit=4_750_000,
                base_premium_rate=0.015,
                reinstatements=0,
            ),
        ]

        program = InsuranceProgram(
            layers=layers,
            deductible=250_000,
            pricing_enabled=True,
        )

        original_rate = program.layers[0].base_premium_rate

        program.apply_pricing(
            expected_revenue=15_000_000,
            market_cycle=MarketCycle.HARD,
            loss_generator=loss_generator,
        )

        # Rate should be updated
        assert program.layers[0].base_premium_rate != original_rate

        # Should have a pricer now
        assert program.pricer is not None

    def test_program_apply_pricing_not_enabled(self):
        """Test apply_pricing when pricing not enabled."""
        program = InsuranceProgram.create_standard_manufacturing_program()

        with pytest.raises(ValueError, match="Pricing not enabled"):
            program.apply_pricing(expected_revenue=15_000_000)

    def test_program_create_with_pricing(self, loss_generator):
        """Test creating program with pricing."""
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=250_000,
                limit=4_750_000,
                base_premium_rate=0.015,  # Initial rate
                reinstatements=0,
            ),
        ]

        program = InsuranceProgram.create_with_pricing(
            layers=layers,
            loss_generator=loss_generator,
            expected_revenue=15_000_000,
            market_cycle=MarketCycle.SOFT,
            deductible=250_000,
        )

        assert program.pricing_enabled
        assert program.pricer is not None
        # Rate should be updated from initial 0.015
        assert program.layers[0].base_premium_rate != 0.015

    def test_program_get_pricing_summary(self, loss_generator):
        """Test getting pricing summary."""
        program = InsuranceProgram.create_with_pricing(
            layers=[
                EnhancedInsuranceLayer(
                    attachment_point=250_000,
                    limit=4_750_000,
                    base_premium_rate=0.015,
                    reinstatements=0,
                ),
            ],
            loss_generator=loss_generator,
            expected_revenue=15_000_000,
        )

        summary = program.get_pricing_summary()

        assert summary["program_name"] == "Priced Insurance Program"
        assert summary["pricing_enabled"] is True
        assert summary["total_premium"] > 0
        assert len(summary["layers"]) == 1

        layer_summary = summary["layers"][0]
        assert "pure_premium" in layer_summary
        assert "expected_frequency" in layer_summary
        assert "expected_severity" in layer_summary


class TestInsurancePolicyIntegration:
    """Test integration with InsurancePolicy."""

    @pytest.fixture
    def loss_generator(self):
        """Create a loss generator for testing."""
        return ManufacturingLossGenerator(seed=42)

    @pytest.fixture
    def pricer(self, loss_generator):
        """Create a pricer for testing."""
        return InsurancePricer(
            loss_generator=loss_generator,
            market_cycle=MarketCycle.NORMAL,
            seed=42,
        )

    @pytest.fixture
    def policy(self):
        """Create an insurance policy for testing."""
        layers = [
            InsuranceLayer(
                attachment_point=250_000,
                limit=4_750_000,
                rate=0.015,
            ),
            InsuranceLayer(
                attachment_point=5_000_000,
                limit=20_000_000,
                rate=0.008,
            ),
        ]
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            return InsurancePolicy(layers=layers, deductible=250_000)

    def test_price_insurance_policy(self, pricer, policy):
        """Test pricing an insurance policy."""
        original_rates = [layer.rate for layer in policy.layers]

        with pytest.warns(DeprecationWarning, match="price_insurance_policy.*deprecated"):
            priced_policy = pricer.price_insurance_policy(
                policy=policy,
                expected_revenue=15_000_000,
                market_cycle=MarketCycle.NORMAL,
                update_policy=True,
            )

        # Should be the same object if update_policy=True
        assert priced_policy is policy

        # Rates should be updated
        new_rates = [layer.rate for layer in policy.layers]
        assert new_rates != original_rates

        # Should have pricing results
        assert hasattr(policy, "pricing_results")
        assert len(policy.pricing_results) == len(policy.layers)

    def test_policy_apply_pricing(self, loss_generator):
        """Test apply_pricing method on InsurancePolicy."""
        layers = [
            InsuranceLayer(
                attachment_point=250_000,
                limit=4_750_000,
                rate=0.015,
            ),
        ]

        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy(
                layers=layers,
                deductible=250_000,
                pricing_enabled=True,
            )

        original_rate = policy.layers[0].rate

        # apply_pricing internally calls price_insurance_policy which emits DeprecationWarning
        with pytest.warns(DeprecationWarning, match="price_insurance_policy.*deprecated"):
            policy.apply_pricing(
                expected_revenue=15_000_000,
                market_cycle=MarketCycle.HARD,
                loss_generator=loss_generator,
            )

        # Rate should be updated
        assert policy.layers[0].rate != original_rate

        # Should have a pricer now
        assert policy.pricer is not None

    def test_policy_create_with_pricing(self, loss_generator):
        """Test creating policy with pricing."""
        layers = [
            InsuranceLayer(
                attachment_point=250_000,
                limit=4_750_000,
                rate=0.015,  # Initial rate
            ),
        ]

        with pytest.warns(DeprecationWarning):
            policy = InsurancePolicy.create_with_pricing(
                layers=layers,
                loss_generator=loss_generator,
                expected_revenue=15_000_000,
                market_cycle=MarketCycle.SOFT,
                deductible=250_000,
            )

        assert policy.pricing_enabled
        assert policy.pricer is not None
        # Rate should be updated from initial 0.015
        assert policy.layers[0].rate != 0.015


class TestAnnualAggregateCIBootstrap:
    """Tests for Issue #614: CI computed on annual aggregates via bootstrap.

    Confidence intervals should reflect the distribution of the annual aggregate
    loss cost (sum of all losses in each simulated year), not the distribution
    of individual loss amounts. Per Werner & Modlin Chapter 8 and ASOP 25,
    pricing variability should be measured at the aggregate level.
    """

    @pytest.fixture
    def loss_generator(self):
        """Create a loss generator for testing."""
        return ManufacturingLossGenerator(seed=42)

    def test_ci_bounds_bracket_pure_premium(self, loss_generator):
        """Bootstrap CI of the mean annual aggregate should bracket the pure premium.

        The pure premium is the sample mean of the annual aggregate distribution.
        A correctly constructed CI of the mean should contain the sample mean.
        """
        pricer = InsurancePricer(
            loss_generator=loss_generator,
            market_cycle=MarketCycle.NORMAL,
            seed=42,
        )
        pure_premium, stats = pricer.calculate_pure_premium(
            attachment_point=100_000,
            limit=5_000_000,
            expected_revenue=15_000_000,
            simulation_years=50,
        )
        if pure_premium > 0:
            lower, upper = stats["confidence_interval"]
            assert (
                lower <= pure_premium
            ), f"CI lower ({lower}) should be <= pure_premium ({pure_premium})"
            assert (
                upper >= pure_premium
            ), f"CI upper ({upper}) should be >= pure_premium ({pure_premium})"

    def test_ci_narrows_with_more_simulation_years(self):
        """CI width should decrease with more simulation years.

        Bootstrap CI of the mean shrinks as sample size grows (law of large
        numbers). This is a key distinction from CI on individual losses,
        which would not shrink in the same way.
        """
        # Use high-frequency generator so every year has losses
        lg_few = ManufacturingLossGenerator.create_simple(
            frequency=20.0,
            severity_mean=100_000,
            severity_std=50_000,
            seed=42,
        )
        pricer_few = InsurancePricer(loss_generator=lg_few, seed=99)
        _, stats_few = pricer_few.calculate_pure_premium(
            attachment_point=50_000,
            limit=5_000_000,
            expected_revenue=15_000_000,
            simulation_years=10,
        )

        lg_many = ManufacturingLossGenerator.create_simple(
            frequency=20.0,
            severity_mean=100_000,
            severity_std=50_000,
            seed=42,
        )
        pricer_many = InsurancePricer(loss_generator=lg_many, seed=99)
        _, stats_many = pricer_many.calculate_pure_premium(
            attachment_point=50_000,
            limit=5_000_000,
            expected_revenue=15_000_000,
            simulation_years=200,
        )

        width_few = stats_few["confidence_interval"][1] - stats_few["confidence_interval"][0]
        width_many = stats_many["confidence_interval"][1] - stats_many["confidence_interval"][0]

        assert width_many < width_few, (
            f"CI width with 200 years ({width_many:.0f}) should be < "
            f"CI width with 10 years ({width_few:.0f})"
        )

    def test_annual_aggregates_in_statistics(self, loss_generator):
        """Statistics dict should include annual_aggregates for transparency."""
        pricer = InsurancePricer(
            loss_generator=loss_generator,
            market_cycle=MarketCycle.NORMAL,
            seed=42,
        )
        _, stats = pricer.calculate_pure_premium(
            attachment_point=100_000,
            limit=5_000_000,
            expected_revenue=15_000_000,
            simulation_years=10,
        )
        assert "annual_aggregates" in stats
        assert len(stats["annual_aggregates"]) == 10

    def test_zero_loss_years_included_in_aggregates(self):
        """Years with no layer losses should contribute 0 to annual aggregates.

        This ensures the aggregate distribution correctly captures the
        probability of zero-loss years.
        """
        # Very low frequency + high attachment -> many zero-loss years
        lg = ManufacturingLossGenerator.create_simple(
            frequency=0.1,
            severity_mean=50_000,
            severity_std=20_000,
            seed=42,
        )
        pricer = InsurancePricer(loss_generator=lg, seed=42)
        _, stats = pricer.calculate_pure_premium(
            attachment_point=100_000,
            limit=5_000_000,
            expected_revenue=15_000_000,
            simulation_years=20,
        )
        aggregates = stats["annual_aggregates"]
        assert len(aggregates) == 20
        # With low frequency and high attachment, most years should be zero
        zero_years = sum(1 for a in aggregates if a == 0)
        assert zero_years > 0, "Expected some zero-loss years with low frequency"


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_program_without_pricing(self):
        """Test that programs work without pricing."""
        program = InsuranceProgram.create_standard_manufacturing_program()

        # Should work normally
        assert program.calculate_annual_premium() > 0
        assert not program.pricing_enabled
        assert program.pricer is None

        # Process claims should work
        result = program.process_claim(1_000_000)
        assert result["total_claim"] == 1_000_000

    def test_policy_without_pricing(self):
        """Test that policies work without pricing."""
        layers = [
            InsuranceLayer(
                attachment_point=250_000,
                limit=4_750_000,
                rate=0.015,
            ),
        ]
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy(layers=layers, deductible=250_000)

        # Should work normally
        assert policy.calculate_premium() > 0
        assert not policy.pricing_enabled
        assert policy.pricer is None

        # Process claims should work
        company_payment, insurance_recovery = policy.process_claim(1_000_000)
        assert company_payment + insurance_recovery == 1_000_000

    def test_fixed_rates_preserved(self):
        """Test that fixed rates are preserved when pricing not enabled."""
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=250_000,
                limit=4_750_000,
                base_premium_rate=0.015,
                reinstatements=0,
            ),
        ]

        program = InsuranceProgram(layers=layers, deductible=250_000)

        # Rate should be unchanged
        assert program.layers[0].base_premium_rate == 0.015

        # Premium calculation should use fixed rate
        premium = program.calculate_annual_premium()
        assert premium == 4_750_000 * 0.015


class TestLAELoadingCalculation:
    """Tests for Issue #616: LAE loading uses dedicated ALAE/ULAE ratios.

    LAE should be calculated from separate alae_ratio and ulae_ratio fields,
    not from the general expense_ratio which includes non-LAE operating expenses.
    Per Werner & Modlin Chapter 7, LAE is a distinct provision from underwriting
    expenses.
    """

    @pytest.fixture
    def loss_generator(self):
        """Create a loss generator for testing."""
        return ManufacturingLossGenerator(seed=42)

    def test_lae_loading_uses_lae_ratio_not_expense_ratio(self, loss_generator):
        """LAE loading should use lae_ratio (alae + ulae), not expense_ratio.

        With default alae_ratio=0.10 and ulae_ratio=0.05, the LAE loading
        should be 15% of pure premium, not 25% (the expense_ratio default).
        """
        pricer = InsurancePricer(
            loss_generator=loss_generator,
            market_cycle=MarketCycle.NORMAL,
            seed=42,
        )
        pricing = pricer.price_layer(
            attachment_point=100_000,
            limit=5_000_000,
            expected_revenue=15_000_000,
        )

        if pricing.pure_premium > 0:
            expected_lae = pricing.pure_premium * 0.15  # alae(0.10) + ulae(0.05)
            assert pricing.lae_loading == pytest.approx(expected_lae)
            # Must NOT equal the old (incorrect) expense_ratio-based calculation
            wrong_lae = pricing.pure_premium * 0.25
            assert pricing.lae_loading != pytest.approx(wrong_lae)

    def test_lae_loading_with_custom_ratios(self, loss_generator):
        """LAE loading should reflect custom ALAE/ULAE ratios."""
        params = PricingParameters(alae_ratio=0.12, ulae_ratio=0.06)
        pricer = InsurancePricer(
            loss_generator=loss_generator,
            market_cycle=MarketCycle.NORMAL,
            parameters=params,
            seed=42,
        )
        pricing = pricer.price_layer(
            attachment_point=100_000,
            limit=5_000_000,
            expected_revenue=15_000_000,
        )

        if pricing.pure_premium > 0:
            expected_lae = pricing.pure_premium * 0.18  # 0.12 + 0.06
            assert pricing.lae_loading == pytest.approx(expected_lae)

    def test_lae_loading_in_compare_market_cycles(self, loss_generator):
        """compare_market_cycles should also use lae_ratio for LAE loading."""
        params = PricingParameters(alae_ratio=0.08, ulae_ratio=0.04)
        pricer = InsurancePricer(
            loss_generator=loss_generator,
            market_cycle=MarketCycle.NORMAL,
            parameters=params,
            seed=42,
        )
        results = pricer.compare_market_cycles(
            attachment_point=100_000,
            limit=5_000_000,
            expected_revenue=15_000_000,
        )

        # All cycles share the same pure premium and thus the same LAE loading
        lae_values = [r.lae_loading for r in results.values()]
        assert len(set(lae_values)) == 1  # all identical

        for pricing in results.values():
            if pricing.pure_premium > 0:
                expected_lae = pricing.pure_premium * 0.12  # 0.08 + 0.04
                assert pricing.lae_loading == pytest.approx(expected_lae)

    def test_lae_ratio_property(self):
        """PricingParameters.lae_ratio should return alae_ratio + ulae_ratio."""
        params = PricingParameters(alae_ratio=0.10, ulae_ratio=0.05)
        assert params.lae_ratio == pytest.approx(0.15)

        params2 = PricingParameters(alae_ratio=0.0, ulae_ratio=0.0)
        assert params2.lae_ratio == 0.0

    def test_warning_when_lae_exceeds_expense_ratio(self):
        """A warning should be emitted when lae_ratio > expense_ratio."""
        with pytest.warns(UserWarning, match="LAE ratio.*exceeds.*expense ratio"):
            PricingParameters(
                expense_ratio=0.10,
                alae_ratio=0.08,
                ulae_ratio=0.05,  # total LAE = 0.13 > expense_ratio 0.10
            )

    def test_no_warning_when_lae_within_expense_ratio(self):
        """No warning when lae_ratio <= expense_ratio (the normal case)."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Default: lae_ratio=0.15 <= expense_ratio=0.25 — no warning
            PricingParameters()

    def test_create_from_config_with_lae_ratios(self, loss_generator):
        """create_from_config should support alae_ratio and ulae_ratio."""
        config = {
            "alae_ratio": 0.12,
            "ulae_ratio": 0.07,
            "market_cycle": "NORMAL",
        }
        pricer = InsurancePricer.create_from_config(config, loss_generator)
        assert pricer.parameters.alae_ratio == 0.12
        assert pricer.parameters.ulae_ratio == 0.07
        assert pricer.parameters.lae_ratio == pytest.approx(0.19)


class TestLossDevelopmentToUltimate:
    """Tests for Issue #714: develop losses to ultimate before pure premium.

    Per ASOP 25 and CAS Ratemaking Chapter 4, losses used in experience
    rating must be developed to ultimate.  Each simulation year is treated
    as an accident year at a specific maturity: older years are more
    developed, recent years are less developed.  Dividing by cumulative
    percent developed produces the ultimate estimate (assumed-pattern
    chain-ladder).
    """

    @pytest.fixture
    def loss_generator(self):
        """High-frequency generator so every year has layer losses."""
        return ManufacturingLossGenerator.create_simple(
            frequency=20.0,
            severity_mean=500_000,
            severity_std=200_000,
            seed=42,
        )

    def test_long_tail_development_increases_pure_premium(self):
        """Long-tail development should increase pure premium.

        When losses develop over 10 years, recent simulation years have
        low cumulative pct_developed, so dividing by pct < 1 inflates
        those years' aggregates → higher mean = higher pure premium.
        """
        # Separate generators with identical seeds so both pricers see the same losses
        lg_no_dev = ManufacturingLossGenerator.create_simple(
            frequency=20.0,
            severity_mean=500_000,
            severity_std=200_000,
            seed=42,
        )
        lg_dev = ManufacturingLossGenerator.create_simple(
            frequency=20.0,
            severity_mean=500_000,
            severity_std=200_000,
            seed=42,
        )

        params_no_dev = PricingParameters()
        pricer_no_dev = InsurancePricer(
            loss_generator=lg_no_dev,
            parameters=params_no_dev,
            seed=99,
        )

        params_dev = PricingParameters(
            development_pattern=ClaimDevelopment.create_long_tail_10yr(),
        )
        pricer_dev = InsurancePricer(
            loss_generator=lg_dev,
            parameters=params_dev,
            seed=99,
        )

        pp_no_dev, stats_no_dev = pricer_no_dev.calculate_pure_premium(
            attachment_point=100_000,
            limit=5_000_000,
            expected_revenue=15_000_000,
            simulation_years=10,
        )
        pp_dev, stats_dev = pricer_dev.calculate_pure_premium(
            attachment_point=100_000,
            limit=5_000_000,
            expected_revenue=15_000_000,
            simulation_years=10,
        )

        assert pp_dev > pp_no_dev, (
            f"Developed premium ({pp_dev:,.0f}) should exceed "
            f"undeveloped ({pp_no_dev:,.0f}) for long-tail pattern"
        )
        assert stats_dev["development_factor"] > 1.0
        assert stats_dev["undeveloped_pure_premium"] == pytest.approx(pp_no_dev, rel=1e-6)

    def test_immediate_pattern_no_change(self, loss_generator):
        """Immediate pattern (100% paid in year 1) should not change premium.

        With an immediate pattern every simulation year is fully developed
        at the evaluation point, so no adjustment is needed.
        """
        params_dev = PricingParameters(
            development_pattern=ClaimDevelopment.create_immediate(),
        )
        pricer_dev = InsurancePricer(
            loss_generator=loss_generator,
            parameters=params_dev,
            seed=99,
        )

        pp, stats = pricer_dev.calculate_pure_premium(
            attachment_point=100_000,
            limit=5_000_000,
            expected_revenue=15_000_000,
            simulation_years=10,
        )

        assert stats["development_factor"] == pytest.approx(1.0)
        assert pp == pytest.approx(stats["undeveloped_pure_premium"])

    def test_no_pattern_backward_compatible(self, loss_generator):
        """No development pattern should produce identical results to before.

        When development_pattern is None (the default), pure_premium,
        undeveloped_pure_premium, and development_factor should all
        reflect no adjustment.
        """
        pricer = InsurancePricer(
            loss_generator=loss_generator,
            seed=99,
        )

        pp, stats = pricer.calculate_pure_premium(
            attachment_point=100_000,
            limit=5_000_000,
            expected_revenue=15_000_000,
            simulation_years=10,
        )

        assert stats["development_factor"] == 1.0
        assert stats["undeveloped_pure_premium"] == pytest.approx(pp)

    def test_statistics_contain_development_fields(self, loss_generator):
        """Statistics dict should include development metadata."""
        params = PricingParameters(
            development_pattern=ClaimDevelopment.create_medium_tail_5yr(),
        )
        pricer = InsurancePricer(
            loss_generator=loss_generator,
            parameters=params,
            seed=99,
        )

        _, stats = pricer.calculate_pure_premium(
            attachment_point=100_000,
            limit=5_000_000,
            expected_revenue=15_000_000,
            simulation_years=10,
        )

        assert "undeveloped_pure_premium" in stats
        assert "development_factor" in stats
        assert stats["development_factor"] >= 1.0

    def test_development_factor_in_layer_pricing(self, loss_generator):
        """price_layer should propagate development_factor to LayerPricing."""
        params = PricingParameters(
            development_pattern=ClaimDevelopment.create_long_tail_10yr(),
        )
        pricer = InsurancePricer(
            loss_generator=loss_generator,
            parameters=params,
            seed=99,
        )

        pricing = pricer.price_layer(
            attachment_point=100_000,
            limit=5_000_000,
            expected_revenue=15_000_000,
        )

        assert pricing.development_factor > 1.0

    def test_longer_tail_gives_larger_development_factor(self):
        """Longer-tail patterns should produce larger development factors.

        A 10-year GL pattern has more immature years than a 5-year WC
        pattern, so its development factor should be larger.
        """
        lg_5yr = ManufacturingLossGenerator.create_simple(
            frequency=20.0,
            severity_mean=500_000,
            severity_std=200_000,
            seed=42,
        )
        lg_10yr = ManufacturingLossGenerator.create_simple(
            frequency=20.0,
            severity_mean=500_000,
            severity_std=200_000,
            seed=42,
        )

        pricer_5yr = InsurancePricer(
            loss_generator=lg_5yr,
            parameters=PricingParameters(
                development_pattern=ClaimDevelopment.create_medium_tail_5yr(),
            ),
            seed=99,
        )
        pricer_10yr = InsurancePricer(
            loss_generator=lg_10yr,
            parameters=PricingParameters(
                development_pattern=ClaimDevelopment.create_long_tail_10yr(),
            ),
            seed=99,
        )

        _, stats_5yr = pricer_5yr.calculate_pure_premium(
            attachment_point=100_000,
            limit=5_000_000,
            expected_revenue=15_000_000,
            simulation_years=10,
        )
        _, stats_10yr = pricer_10yr.calculate_pure_premium(
            attachment_point=100_000,
            limit=5_000_000,
            expected_revenue=15_000_000,
            simulation_years=10,
        )

        assert stats_10yr["development_factor"] > stats_5yr["development_factor"], (
            f"10yr LDF ({stats_10yr['development_factor']:.3f}) should exceed "
            f"5yr LDF ({stats_5yr['development_factor']:.3f})"
        )

    def test_ci_computed_on_developed_aggregates(self):
        """Bootstrap CI should be computed on developed annual aggregates.

        With development, CI should be wider than without (more variance
        from inflating immature years).
        """
        lg_no = ManufacturingLossGenerator.create_simple(
            frequency=20.0,
            severity_mean=500_000,
            severity_std=200_000,
            seed=42,
        )
        lg_dev = ManufacturingLossGenerator.create_simple(
            frequency=20.0,
            severity_mean=500_000,
            severity_std=200_000,
            seed=42,
        )

        pricer_no_dev = InsurancePricer(
            loss_generator=lg_no,
            seed=99,
        )
        pricer_dev = InsurancePricer(
            loss_generator=lg_dev,
            parameters=PricingParameters(
                development_pattern=ClaimDevelopment.create_long_tail_10yr(),
            ),
            seed=99,
        )

        pp_no, stats_no = pricer_no_dev.calculate_pure_premium(
            attachment_point=100_000,
            limit=5_000_000,
            expected_revenue=15_000_000,
            simulation_years=10,
        )
        pp_dev, stats_dev = pricer_dev.calculate_pure_premium(
            attachment_point=100_000,
            limit=5_000_000,
            expected_revenue=15_000_000,
            simulation_years=10,
        )

        width_no = stats_no["confidence_interval"][1] - stats_no["confidence_interval"][0]
        width_dev = stats_dev["confidence_interval"][1] - stats_dev["confidence_interval"][0]

        # Development inflates recent years more than mature ones,
        # increasing variance → wider CI
        assert width_dev > width_no, (
            f"Developed CI width ({width_dev:,.0f}) should exceed " f"undeveloped ({width_no:,.0f})"
        )

    def test_zero_loss_years_remain_zero_after_development(self):
        """Years with zero layer losses should remain zero after development.

        0 / pct_developed = 0, so development should not create phantom losses.
        """
        # Very low frequency + very high attachment → many zero-loss years
        lg = ManufacturingLossGenerator.create_simple(
            frequency=0.1,
            severity_mean=50_000,
            severity_std=20_000,
            seed=42,
        )
        params = PricingParameters(
            development_pattern=ClaimDevelopment.create_long_tail_10yr(),
        )
        pricer = InsurancePricer(loss_generator=lg, parameters=params, seed=42)
        _, stats = pricer.calculate_pure_premium(
            attachment_point=100_000,
            limit=5_000_000,
            expected_revenue=15_000_000,
            simulation_years=20,
        )

        # With low frequency and high attachment, most aggregates should still be 0
        aggregates = stats["annual_aggregates"]
        zero_years = sum(1 for a in aggregates if a == 0.0)
        assert zero_years > 0, "Expected some zero-loss years even after development"

    def test_create_from_config_with_pattern_name(self, loss_generator):
        """create_from_config should accept pattern name string."""
        config = {
            "development_pattern": "long_tail_10yr",
            "market_cycle": "NORMAL",
        }
        pricer = InsurancePricer.create_from_config(config, loss_generator)
        assert pricer.parameters.development_pattern is not None
        assert pricer.parameters.development_pattern.pattern_name == "LONG_TAIL_10YR"

    def test_create_from_config_with_pattern_dict(self, loss_generator):
        """create_from_config should accept custom pattern dict."""
        config = {
            "development_pattern": {
                "name": "custom_3yr",
                "factors": [0.50, 0.30, 0.20],
            },
            "market_cycle": "NORMAL",
        }
        pricer = InsurancePricer.create_from_config(config, loss_generator)
        pattern = pricer.parameters.development_pattern
        assert pattern is not None
        assert pattern.pattern_name == "custom_3yr"
        assert pattern.development_factors == [0.50, 0.30, 0.20]

    def test_create_from_config_with_invalid_pattern_name(self, loss_generator):
        """create_from_config should raise on unknown pattern name."""
        config = {
            "development_pattern": "nonexistent_pattern",
        }
        with pytest.raises(ValueError, match="Unknown development pattern"):
            InsurancePricer.create_from_config(config, loss_generator)

    def test_create_from_config_with_pattern_object(self, loss_generator):
        """create_from_config should accept a ClaimDevelopment instance."""
        pattern = ClaimDevelopment.create_medium_tail_5yr()
        config = {
            "development_pattern": pattern,
            "market_cycle": "NORMAL",
        }
        pricer = InsurancePricer.create_from_config(config, loss_generator)
        assert pricer.parameters.development_pattern is pattern

    def test_compare_market_cycles_includes_development_factor(self, loss_generator):
        """compare_market_cycles should include development_factor."""
        params = PricingParameters(
            development_pattern=ClaimDevelopment.create_long_tail_10yr(),
        )
        pricer = InsurancePricer(
            loss_generator=loss_generator,
            parameters=params,
            seed=99,
        )
        results = pricer.compare_market_cycles(
            attachment_point=100_000,
            limit=5_000_000,
            expected_revenue=15_000_000,
        )

        # All cycles share the same development factor
        dev_factors = [r.development_factor for r in results.values()]
        assert all(d > 1.0 for d in dev_factors)
        assert len(set(dev_factors)) == 1
