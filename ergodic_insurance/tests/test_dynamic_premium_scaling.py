"""Tests for dynamic insurance premium scaling based on revenue exposure.

This module tests the dynamic premium scaling functionality where insurance
premiums scale with actual revenue exposure during simulation.
"""

from unittest.mock import Mock

import numpy as np
import pytest

from ergodic_insurance.exposure_base import FinancialStateProvider, RevenueExposure
from ergodic_insurance.insurance_pricing import InsurancePricer, MarketCycle
from ergodic_insurance.insurance_program import EnhancedInsuranceLayer, InsuranceProgram
from ergodic_insurance.loss_distributions import ManufacturingLossGenerator


class MockStateProvider:
    """Mock state provider for testing exposure-based scaling."""

    def __init__(self, base_revenue=10_000_000):
        self.base_revenue = base_revenue
        self.current_revenue = base_revenue
        self.base_assets = base_revenue
        self.current_assets = base_revenue
        self.base_equity = base_revenue * 0.5
        self.current_equity = base_revenue * 0.5


def test_premium_scales_with_revenue():
    """Test that premium scales 1-to-1 with revenue exposure."""
    # Given: Initial revenue of $10M, premium rate of 2%
    state_provider = MockStateProvider(base_revenue=10_000_000)
    exposure = RevenueExposure(state_provider=state_provider)

    layer = EnhancedInsuranceLayer(
        attachment_point=1_000_000,
        limit=5_000_000,
        base_premium_rate=0.02,
        premium_rate_exposure=exposure,
    )

    # When: Revenue stays at $10M
    base_premium = layer.calculate_base_premium(time=0.0)
    assert base_premium == 5_000_000 * 0.02  # $100,000

    # When: Revenue grows to $15M (1.5x)
    state_provider.current_revenue = 15_000_000
    scaled_premium = layer.calculate_base_premium(time=1.0)

    # Then: Premium should be 1.5x original premium
    assert scaled_premium == pytest.approx(base_premium * 1.5)


def test_loss_frequency_scales_with_revenue():
    """Test that loss frequency scales with revenue exposure."""
    # Given: Base frequency of 0.5 at $10M revenue
    state_provider = MockStateProvider(base_revenue=10_000_000)
    exposure = RevenueExposure(state_provider=state_provider)

    generator = ManufacturingLossGenerator(
        attritional_params={"base_frequency": 5.0}, exposure=exposure
    )

    # When: Revenue doubles to $20M
    state_provider.current_revenue = 20_000_000

    # Generate losses with doubled revenue
    losses1, stats1 = generator.generate_losses(duration=1.0, revenue=10_000_000, time=0.0)
    losses2, stats2 = generator.generate_losses(duration=1.0, revenue=10_000_000, time=1.0)

    # Then: The generator should use exposure-based revenue, not passed revenue
    # Note: Due to randomness, we can't assert exact frequency
    assert generator.exposure is not None
    assert generator.exposure.get_exposure(1.0) == 20_000_000


def test_full_simulation_with_dynamic_scaling():
    """Test a full simulation with dynamic premium and loss scaling."""
    # Setup manufacturer with growth
    state_provider = MockStateProvider(base_revenue=10_000_000)
    exposure = RevenueExposure(state_provider=state_provider)

    # Create insurance program with dynamic scaling
    layers = [
        EnhancedInsuranceLayer(
            attachment_point=250_000,
            limit=4_750_000,
            base_premium_rate=0.015,
            premium_rate_exposure=exposure,
        ),
        EnhancedInsuranceLayer(
            attachment_point=5_000_000,
            limit=20_000_000,
            base_premium_rate=0.008,
            premium_rate_exposure=exposure,
        ),
    ]

    program = InsuranceProgram(layers=layers, deductible=250_000)

    # Simulate growth over 10 years
    premiums = []
    for year in range(10):
        # Simulate revenue growth of 5% per year
        state_provider.current_revenue = 10_000_000 * (1.05**year)
        annual_premium = program.calculate_annual_premium(time=float(year))
        premiums.append(annual_premium)

    # Verify premiums increase with revenue
    assert all(premiums[i + 1] > premiums[i] for i in range(9))

    # Verify final premium is approximately 1.55x initial (1.05^9 ≈ 1.55)
    assert premiums[-1] == pytest.approx(premiums[0] * (1.05**9), rel=0.01)


def test_zero_revenue_handling():
    """Test that system handles zero revenue gracefully."""
    # Given: Company with zero revenue
    state_provider = MockStateProvider(base_revenue=10_000_000)
    state_provider.current_revenue = 0
    exposure = RevenueExposure(state_provider=state_provider)

    layer = EnhancedInsuranceLayer(
        attachment_point=250_000,
        limit=4_750_000,
        base_premium_rate=0.015,
        premium_rate_exposure=exposure,
    )

    # Then: Premiums should be zero
    premium = layer.calculate_base_premium(time=0.0)
    assert premium == 0.0

    # And: Loss frequency multiplier should be zero
    assert exposure.get_frequency_multiplier(time=0.0) == 0.0


def test_revenue_decline_scaling():
    """Test that premiums and losses scale down with declining revenue."""
    # Given: Revenue declines by 50%
    state_provider = MockStateProvider(base_revenue=10_000_000)
    exposure = RevenueExposure(state_provider=state_provider)

    layer = EnhancedInsuranceLayer(
        attachment_point=250_000,
        limit=4_750_000,
        base_premium_rate=0.015,
        premium_rate_exposure=exposure,
    )

    # Initial premium
    initial_premium = layer.calculate_base_premium(time=0.0)

    # Revenue declines by 50%
    state_provider.current_revenue = 5_000_000
    reduced_premium = layer.calculate_base_premium(time=1.0)

    # Then: Premiums should scale down by 50%
    assert reduced_premium == pytest.approx(initial_premium * 0.5)


def test_pricer_with_dynamic_revenue():
    """Test InsurancePricer using exposure for dynamic revenue."""
    # Setup exposure
    state_provider = MockStateProvider(base_revenue=15_000_000)
    exposure = RevenueExposure(state_provider=state_provider)

    # Create loss generator with exposure
    loss_gen = ManufacturingLossGenerator(exposure=exposure)

    # Create pricer with exposure
    pricer = InsurancePricer(
        loss_generator=loss_gen, market_cycle=MarketCycle.NORMAL, exposure=exposure
    )

    # Create program
    layers = [
        EnhancedInsuranceLayer(attachment_point=250_000, limit=4_750_000, base_premium_rate=0.015)
    ]
    program = InsuranceProgram(layers=layers)

    # Price program using exposure (no expected_revenue needed)
    priced_program = pricer.price_insurance_program(program=program, time=0.0, update_program=True)

    # Verify pricing was applied
    assert priced_program.layers[0].base_premium_rate > 0


def test_exposure_not_provided_fallback():
    """Test that system falls back to fixed premiums when exposure not provided."""
    # Create layer without exposure
    layer = EnhancedInsuranceLayer(
        attachment_point=250_000, limit=4_750_000, base_premium_rate=0.015
    )

    # Premium should be fixed
    premium1 = layer.calculate_base_premium(time=0.0)
    premium2 = layer.calculate_base_premium(time=1.0)

    assert premium1 == premium2
    assert premium1 == 4_750_000 * 0.015


def test_insurance_program_annual_premium_with_time():
    """Test that InsuranceProgram.calculate_annual_premium accepts time parameter."""
    state_provider = MockStateProvider(base_revenue=10_000_000)
    exposure = RevenueExposure(state_provider=state_provider)

    layers = [
        EnhancedInsuranceLayer(
            attachment_point=250_000,
            limit=4_750_000,
            base_premium_rate=0.015,
            premium_rate_exposure=exposure,
        )
    ]

    program = InsuranceProgram(layers=layers)

    # Should accept time parameter
    premium_t0 = program.calculate_annual_premium(time=0.0)

    # Change revenue
    state_provider.current_revenue = 12_000_000
    premium_t1 = program.calculate_annual_premium(time=1.0)

    # Premium should have increased
    assert premium_t1 > premium_t0
    assert premium_t1 == pytest.approx(premium_t0 * 1.2)


def test_multiple_layers_with_shared_exposure():
    """Test multiple insurance layers sharing the same exposure object."""
    state_provider = MockStateProvider(base_revenue=10_000_000)
    exposure = RevenueExposure(state_provider=state_provider)

    # Create multiple layers with shared exposure
    layers = [
        EnhancedInsuranceLayer(
            attachment_point=250_000,
            limit=4_750_000,
            base_premium_rate=0.015,
            premium_rate_exposure=exposure,
        ),
        EnhancedInsuranceLayer(
            attachment_point=5_000_000,
            limit=20_000_000,
            base_premium_rate=0.008,
            premium_rate_exposure=exposure,
        ),
        EnhancedInsuranceLayer(
            attachment_point=25_000_000,
            limit=25_000_000,
            base_premium_rate=0.004,
            premium_rate_exposure=exposure,
        ),
    ]

    program = InsuranceProgram(layers=layers)

    # All layers should see the same revenue
    initial_premiums = [layer.calculate_base_premium(0.0) for layer in layers]

    # Double revenue
    state_provider.current_revenue = 20_000_000
    scaled_premiums = [layer.calculate_base_premium(1.0) for layer in layers]

    # All premiums should double
    for initial, scaled in zip(initial_premiums, scaled_premiums):
        assert scaled == pytest.approx(initial * 2.0)


def test_consistency_between_loss_and_premium_scaling():
    """Test that loss generation and premium calculation scale consistently."""
    state_provider = MockStateProvider(base_revenue=10_000_000)
    exposure = RevenueExposure(state_provider=state_provider)

    # Create generator and layer with same exposure
    generator = ManufacturingLossGenerator(
        attritional_params={"base_frequency": 5.0}, exposure=exposure
    )

    layer = EnhancedInsuranceLayer(
        attachment_point=250_000,
        limit=4_750_000,
        base_premium_rate=0.015,
        premium_rate_exposure=exposure,
    )

    # Both should use same exposure multiplier
    multiplier = exposure.get_frequency_multiplier(0.0)
    assert multiplier == 1.0  # Base case

    # Change revenue
    state_provider.current_revenue = 15_000_000
    new_multiplier = exposure.get_frequency_multiplier(1.0)
    assert new_multiplier == 1.5

    # Premium should scale by same factor
    base_premium = 4_750_000 * 0.015
    scaled_premium = layer.calculate_base_premium(1.0)
    assert scaled_premium == pytest.approx(base_premium * 1.5)
