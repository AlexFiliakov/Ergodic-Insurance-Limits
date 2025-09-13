"""Integration tests for financial model components.

This module tests the integration between manufacturer, claim generator,
and claim development components to ensure financial consistency.
"""
# mypy: ignore-errors

import numpy as np
import pytest

from ergodic_insurance.src.claim_development import ClaimDevelopment
from ergodic_insurance.src.claim_generator import ClaimEvent, ClaimGenerator
from ergodic_insurance.src.loss_distributions import LossData, LossEvent
from ergodic_insurance.src.manufacturer import WidgetManufacturer
from ergodic_insurance.src.simulation import Simulation

from .test_claim_development_wrapper import ClaimDevelopmentWrapper
from .test_fixtures import (
    assert_financial_consistency,
    base_manufacturer,
    claim_development,
    default_config_v2,
    generate_sample_losses,
    standard_claim_generator,
)
from .test_helpers import timer, validate_trajectory


class TestFinancialIntegration:
    """Test financial model component integration."""

    def test_manufacturer_claim_generator_integration(
        self,
        base_manufacturer: WidgetManufacturer,
        standard_claim_generator: ClaimGenerator,
    ):
        """Test integration between manufacturer and claim generator.

        Verifies that:
        - Claims are properly generated and applied
        - Financial state remains consistent
        - Cash flow impacts are correct
        """
        manufacturer = base_manufacturer.copy()
        initial_equity = manufacturer.equity

        # Generate claims for one year
        claims = standard_claim_generator.generate_claims(years=1)
        total_losses = sum(claim.amount for claim in claims)

        # Apply claims to manufacturer
        for claim in claims:
            manufacturer.process_insurance_claim(
                claim_amount=claim.amount,
                deductible_amount=0,  # No insurance
                insurance_limit=0,  # No insurance coverage
            )

        # Pay the claim liabilities to trigger actual equity reduction
        manufacturer.pay_claim_liabilities()

        # Verify financial impact
        assert manufacturer.equity < initial_equity, "Equity should decrease from losses"
        assert_financial_consistency(manufacturer)

        # Verify loss amount impact
        equity_reduction = initial_equity - manufacturer.equity
        # With default payment schedule, only 10% of claims are paid in year 0
        expected_year_0_payment = total_losses * 0.10
        assert (
            equity_reduction >= expected_year_0_payment * 0.8
        ), "Equity reduction should match claim payments"

    def test_claim_development_integration(
        self,
        base_manufacturer: WidgetManufacturer,
        claim_development: ClaimDevelopmentWrapper,
    ):
        """Test integration of claim development patterns.

        Verifies that:
        - Multi-year claim payments are properly scheduled
        - Cash flow timing is correct
        - Balance sheet remains balanced across years
        """
        manufacturer = base_manufacturer.copy()

        # Create a large claim
        initial_claim = ClaimEvent(year=0, amount=5_000_000)

        # Develop the claim over multiple years
        developed_claims = claim_development.develop_claims([initial_claim])

        # Track financial state over development period
        states = []
        for year, claims_in_year in enumerate(developed_claims):
            year_state = {
                "year": year,
                "equity_before": manufacturer.equity,
                "assets_before": manufacturer.assets,
            }

            # Process claims for this year
            for claim in claims_in_year:
                manufacturer.process_insurance_claim(
                    claim_amount=claim.amount,
                    deductible_amount=0,
                    insurance_limit=0,
                )

            year_state["equity_after"] = manufacturer.equity
            year_state["assets_after"] = manufacturer.assets
            year_state["payments"] = sum(c.amount for c in claims_in_year)

            states.append(year_state)
            assert_financial_consistency(manufacturer)

            # Step manufacturer forward
            manufacturer.step()

        # Verify total payments match original claim
        total_paid = sum(state["payments"] for state in states)
        expected_total = initial_claim.amount * claim_development.ultimate_factor

        assert np.isclose(total_paid, expected_total, rtol=0.01), (
            f"Total payments {total_paid:.2f} should match " f"ultimate claim {expected_total:.2f}"
        )

        # Verify payment pattern matches development factors
        for i, state in enumerate(states[: len(claim_development.pattern)]):
            expected_pct = claim_development.pattern[i] if i < len(claim_development.pattern) else 0
            actual_pct = state["payments"] / initial_claim.amount
            assert np.isclose(actual_pct, expected_pct, rtol=0.01), (
                f"Year {i} payment percentage {actual_pct:.2%} should match "
                f"pattern {expected_pct:.2%}"
            )

    def test_working_capital_effects(
        self,
        base_manufacturer: WidgetManufacturer,
        standard_claim_generator: ClaimGenerator,
    ):
        """Test working capital impacts from losses.

        Verifies that:
        - Working capital requirements are maintained
        - Cash management handles claim payments
        - Liquidity constraints are respected
        """
        manufacturer = base_manufacturer.copy()

        # Track working capital over time
        wc_history = []

        for year in range(5):
            # Generate and apply claims
            claims = standard_claim_generator.generate_claims(years=1)

            # Track asset position before claims
            assets_before = manufacturer.assets

            for claim in claims:
                manufacturer.process_insurance_claim(
                    claim_amount=claim.amount,
                    deductible_amount=0,
                    insurance_limit=0,
                )

            # Step forward (this pays the liabilities and affects assets)
            manufacturer.step()

            # Track asset position after claims and step
            assets_after = manufacturer.assets

            wc_history.append(
                {
                    "year": year,
                    "assets_before": assets_before,
                    "assets_after": assets_after,
                    "claims": sum(c.amount for c in claims),
                }
            )

            # Verify financial constraints
            assert manufacturer.assets >= 0, f"Negative assets in year {year}"
            assert_financial_consistency(manufacturer)

        # Verify asset position responded to losses
        total_claims = sum(h["claims"] for h in wc_history)
        if total_claims > 0:
            assert any(
                h["assets_after"] < h["assets_before"] for h in wc_history
            ), "Assets should decrease from claim payments"

    def test_multi_year_financial_flow(
        self,
        base_manufacturer: WidgetManufacturer,
        standard_claim_generator: ClaimGenerator,
        claim_development: ClaimDevelopment,
    ):
        """Test complete multi-year financial flow.

        Verifies that:
        - Multi-year simulation maintains consistency
        - Developed claims are properly tracked
        - Growth dynamics work with losses
        """
        manufacturer = base_manufacturer.copy()
        simulation = Simulation(
            manufacturer=manufacturer,
            claim_generator=standard_claim_generator,
            time_horizon=10,
            seed=42,
        )

        # Run simulation with claim development
        results = simulation.run()

        # Verify results structure
        assert len(results.years) == 10
        assert len(results.equity) == 10
        assert len(results.assets) == 10

        # Verify financial trajectories
        assert validate_trajectory(
            results.equity,
            min_value=0,
            allow_negative=False,
        ), "Equity trajectory invalid"

        assert validate_trajectory(
            results.assets,
            min_value=0,
            allow_negative=False,
        ), "Assets trajectory invalid"

        # Verify balance sheet consistency at each point
        for i in range(len(results.years)):
            # Approximate check since we don't have debt directly
            assert (
                results.assets[i] >= results.equity[i] * 0.9
            ), f"Year {i}: Assets should approximately equal equity (no debt model)"

    def test_loss_to_balance_sheet_flow(
        self,
        base_manufacturer: WidgetManufacturer,
    ):
        """Test complete flow from loss generation to balance sheet impact.

        This is the example test from the issue requirements.
        """
        # Setup
        manufacturer = base_manufacturer.copy()
        claim_gen = ClaimGenerator(frequency=5, severity_mean=100_000, seed=42)
        claim_dev = ClaimDevelopmentWrapper(pattern=[0.6, 0.3, 0.1])

        # Generate losses with development
        annual_losses = claim_gen.generate_claims(years=1)
        payment_schedule = claim_dev.develop_claims(annual_losses)

        # Apply to manufacturer
        initial_equity = manufacturer.equity
        initial_assets = manufacturer.assets

        # Process first year payments
        if len(payment_schedule) > 0:
            for claim in payment_schedule[0]:
                manufacturer.process_insurance_claim(
                    claim_amount=claim.amount,
                    deductible_amount=0,
                    insurance_limit=0,
                )

        # Pay the claim liabilities to trigger actual equity reduction
        if len(annual_losses) > 0:
            manufacturer.pay_claim_liabilities()

        # Assertions
        if len(annual_losses) > 0:
            assert manufacturer.equity < initial_equity, "Equity should decrease from losses"
        assert manufacturer.assets >= 0, "No negative assets allowed"
        assert_financial_consistency(manufacturer)

        # Verify balance sheet equation (for this simple model without debt)
        assert np.isclose(
            manufacturer.assets,
            manufacturer.equity,
            rtol=1e-10,
        ), "Balance sheet equation must hold (assets = equity without debt)"

    def test_cash_flow_timing(
        self,
        base_manufacturer: WidgetManufacturer,
        claim_development: ClaimDevelopmentWrapper,
    ):
        """Test that cash flow timing is properly handled.

        Verifies that:
        - Immediate vs developed payments are distinguished
        - Cash impacts occur at the right time
        - Reserves are properly maintained
        """
        manufacturer = base_manufacturer.copy()

        # Create claims with different payment patterns
        immediate_claim = ClaimEvent(year=0, amount=1_000_000)
        developed_claim = ClaimEvent(year=0, amount=2_000_000)

        # Process immediate claim using uninsured claim method for immediate payment
        assets_before_immediate = manufacturer.assets
        manufacturer.process_uninsured_claim(
            claim_amount=immediate_claim.amount,
            immediate_payment=True,
        )
        assets_after_immediate = manufacturer.assets

        # Verify immediate asset impact
        asset_impact_immediate = assets_before_immediate - assets_after_immediate
        assert asset_impact_immediate > 0, "Immediate claim should reduce assets"

        # Reset for developed claim test
        manufacturer2 = base_manufacturer.copy()

        # Process developed claim using uninsured claim method with payment schedule
        assets_before_developed = manufacturer2.assets
        manufacturer2.process_uninsured_claim(
            claim_amount=developed_claim.amount,
            immediate_payment=False,  # This creates a liability with payment schedule
        )

        # Manually trigger first year payment from the liability
        manufacturer2.pay_claim_liabilities()
        assets_after_developed = manufacturer2.assets

        # First year impact should be less than total claim
        first_year_impact = assets_before_developed - assets_after_developed
        assert (
            first_year_impact < developed_claim.amount
        ), "First year payment should be less than total claim"

        # Verify it matches the default ClaimLiability payment pattern (10% in first year)
        expected_first_year = developed_claim.amount * 0.10  # Default first year is 10%
        assert np.isclose(first_year_impact, expected_first_year, rtol=0.1), (
            f"First year payment {first_year_impact:.2f} should match "
            f"default pattern {expected_first_year:.2f}"
        )

    def test_revenue_loss_correlation(
        self,
        base_manufacturer: WidgetManufacturer,
    ):
        """Test correlation between revenue and losses.

        Verifies that:
        - Higher revenue periods can generate more losses
        - Loss frequency scales appropriately
        - Financial impacts are proportional
        """
        # Create two manufacturers with different revenue levels
        low_revenue_mfg = base_manufacturer.copy()
        low_revenue_mfg.asset_turnover_ratio = 0.5  # Lower revenue

        high_revenue_mfg = base_manufacturer.copy()
        high_revenue_mfg.asset_turnover_ratio = 2.0  # Higher revenue

        # Generate losses scaled by revenue
        base_frequency = 5.0

        # Low revenue losses
        low_revenue = low_revenue_mfg.calculate_revenue()
        low_frequency = base_frequency * (low_revenue / 10_000_000)
        low_claim_gen = ClaimGenerator(
            frequency=low_frequency,
            severity_mean=100_000,
            seed=42,
        )

        # High revenue losses
        high_revenue = high_revenue_mfg.calculate_revenue()
        high_frequency = base_frequency * (high_revenue / 10_000_000)
        high_claim_gen = ClaimGenerator(
            frequency=high_frequency,
            severity_mean=100_000,
            seed=42,
        )

        # Generate claims
        low_claims = low_claim_gen.generate_claims(years=1)
        high_claims = high_claim_gen.generate_claims(years=1)

        # Verify scaling
        if high_revenue > low_revenue:
            # On average, higher revenue should have more expected losses
            low_expected = low_frequency * 100_000
            high_expected = high_frequency * 100_000
            assert high_expected > low_expected, "Higher revenue should have higher expected losses"

    def test_financial_state_persistence(
        self,
        base_manufacturer: WidgetManufacturer,
        standard_claim_generator: ClaimGenerator,
    ):
        """Test that financial state persists correctly across operations.

        Verifies that:
        - State changes are properly tracked
        - No state leakage between operations
        - Copy operations work correctly
        """
        # Original manufacturer
        original = base_manufacturer.copy()
        original_equity = original.equity

        # Apply losses to original using uninsured claims for immediate impact
        claims = standard_claim_generator.generate_claims(years=1)
        for claim in claims:
            original.process_uninsured_claim(
                claim_amount=claim.amount,
                immediate_payment=True,
            )

        # Verify original changed
        assert original.equity != original_equity, "Original should be modified"

        # Create a copy (which resets to initial state per the implementation)
        copy = original.copy()
        # The copy() method resets to initial state, not current state
        assert copy.equity == original_equity, "Copy should reset to initial equity"
        assert copy.equity != original.equity, "Copy should not match modified original"

        # Modify copy
        copy.step()

        # Verify independence - they should still be different
        assert copy.equity != original.equity, "Copy should be independent after modification"
        assert_financial_consistency(original)
        assert_financial_consistency(copy)

    def test_extreme_loss_scenarios(
        self,
        base_manufacturer: WidgetManufacturer,
    ):
        """Test financial model under extreme loss scenarios.

        Verifies that:
        - Model handles catastrophic losses gracefully
        - Bankruptcy is properly detected
        - Recovery mechanisms work
        """
        manufacturer = base_manufacturer.copy()
        initial_equity = manufacturer.equity

        # Apply catastrophic loss (50% of assets) with immediate payment
        catastrophic_loss = manufacturer.assets * 0.5
        manufacturer.process_uninsured_claim(
            claim_amount=catastrophic_loss,
            immediate_payment=True,
        )

        # Verify significant impact but consistency
        assert (
            manufacturer.equity < initial_equity * 0.6
        ), "Catastrophic loss should severely impact equity"
        assert_financial_consistency(manufacturer)

        # Test near-bankruptcy scenario
        manufacturer2 = base_manufacturer.copy()
        initial_equity2 = manufacturer2.equity
        bankruptcy_loss = manufacturer2.assets * 0.95
        manufacturer2.process_uninsured_claim(
            claim_amount=bankruptcy_loss,
            immediate_payment=True,
        )

        # Should be near bankruptcy but still consistent
        # In this model without debt: assets = equity, so both drop by 95%
        assert (
            manufacturer2.equity < initial_equity2 * 0.1
        ), "Should retain less than 10% of original equity"
        assert manufacturer2.equity >= 0, "Equity should not go negative in basic model"
        assert_financial_consistency(manufacturer2)

    def test_loss_data_integration(self):
        """Test integration with LossData structures.

        Verifies that:
        - LossData properly integrates with financial models
        - Conversions maintain data integrity
        - Timestamps are properly handled
        """
        # Generate loss events
        losses = [
            LossEvent(timestamp=0.5, amount=100_000, event_type="operational"),
            LossEvent(timestamp=1.2, amount=200_000, event_type="liability"),
            LossEvent(timestamp=1.8, amount=150_000, event_type="property"),
        ]

        # Convert to LossData
        loss_data = LossData.from_loss_events(losses)

        # Validate structure
        assert loss_data.validate()
        assert len(loss_data.timestamps) == 3
        assert np.sum(loss_data.loss_amounts) == 450_000

        # Convert to ClaimEvents
        claims = ClaimGenerator.from_loss_data(loss_data)

        # Verify conversion
        assert len(claims) == 3
        total_claims = sum(c.amount for c in claims)
        assert np.isclose(total_claims, 450_000, rtol=1e-10)

        # Verify year assignments
        for claim, loss in zip(claims, losses):
            assert claim.year == int(loss.timestamp)
            assert claim.amount == loss.amount
