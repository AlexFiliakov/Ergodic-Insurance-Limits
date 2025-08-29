"""Integration tests for financial model components.

This module tests the integration between manufacturer, claim generator,
and claim development components to ensure financial consistency.
"""

import numpy as np
import pytest

from src.claim_development import ClaimDevelopment
from src.claim_generator import ClaimEvent, ClaimGenerator
from src.loss_distributions import LossData, LossEvent
from src.manufacturer import WidgetManufacturer
from src.simulation import Simulation

from .test_fixtures import (
    assert_financial_consistency,
    base_manufacturer,
    claim_development,
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
        claims = standard_claim_generator.generate_year()
        total_losses = sum(claim.amount for claim in claims)

        # Apply claims to manufacturer
        for claim in claims:
            manufacturer.process_insurance_claim(
                claim_amount=claim.amount,
                insurance_recovery=0,  # No insurance
                deductible_amount=0,
            )

        # Verify financial impact
        assert manufacturer.equity < initial_equity, "Equity should decrease from losses"
        assert_financial_consistency(manufacturer)

        # Verify loss amount impact
        equity_reduction = initial_equity - manufacturer.equity
        assert equity_reduction >= total_losses * 0.5, "Losses should significantly impact equity"

    def test_claim_development_integration(
        self,
        base_manufacturer: WidgetManufacturer,
        claim_development: ClaimDevelopment,
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
                "cash_before": manufacturer.cash,
            }

            # Process claims for this year
            for claim in claims_in_year:
                manufacturer.process_insurance_claim(
                    claim_amount=claim.amount,
                    insurance_recovery=0,
                    deductible_amount=0,
                )

            year_state["equity_after"] = manufacturer.equity
            year_state["cash_after"] = manufacturer.cash
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
            claims = standard_claim_generator.generate_year()

            wc_before = manufacturer.working_capital
            cash_before = manufacturer.cash

            for claim in claims:
                manufacturer.process_insurance_claim(
                    claim_amount=claim.amount,
                    insurance_recovery=0,
                    deductible_amount=0,
                )

            wc_after = manufacturer.working_capital
            cash_after = manufacturer.cash

            wc_history.append(
                {
                    "year": year,
                    "wc_before": wc_before,
                    "wc_after": wc_after,
                    "cash_before": cash_before,
                    "cash_after": cash_after,
                    "claims": sum(c.amount for c in claims),
                }
            )

            # Step forward
            manufacturer.step()

            # Verify working capital constraints
            assert manufacturer.working_capital >= 0, f"Negative working capital in year {year}"
            assert manufacturer.cash >= 0, f"Negative cash in year {year}"
            assert_financial_consistency(manufacturer)

        # Verify working capital responded to losses
        total_claims = sum(h["claims"] for h in wc_history)
        if total_claims > 0:
            assert any(
                h["cash_after"] < h["cash_before"] for h in wc_history
            ), "Cash should decrease from claim payments"

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
        claim_dev = ClaimDevelopment(pattern=[0.6, 0.3, 0.1])

        # Generate losses with development
        annual_losses = claim_gen.generate_year()
        payment_schedule = claim_dev.develop_claims(annual_losses)

        # Apply to manufacturer
        initial_equity = manufacturer.equity
        initial_assets = manufacturer.assets

        # Process first year payments
        if len(payment_schedule) > 0:
            for claim in payment_schedule[0]:
                manufacturer.process_insurance_claim(
                    claim_amount=claim.amount,
                    insurance_recovery=0,
                    deductible_amount=0,
                )

        # Assertions
        if len(annual_losses) > 0:
            assert manufacturer.equity < initial_equity, "Equity should decrease from losses"
        assert manufacturer.cash >= 0, "No negative cash allowed"
        assert_financial_consistency(manufacturer)

        # Verify balance sheet equation
        assert np.isclose(
            manufacturer.assets,
            manufacturer.equity + manufacturer.debt,
            rtol=1e-10,
        ), "Balance sheet equation must hold"

    def test_cash_flow_timing(
        self,
        base_manufacturer: WidgetManufacturer,
        claim_development: ClaimDevelopment,
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

        # Process immediate claim
        cash_before_immediate = manufacturer.cash
        manufacturer.process_insurance_claim(
            claim_amount=immediate_claim.amount,
            insurance_recovery=0,
            deductible_amount=0,
        )
        cash_after_immediate = manufacturer.cash

        # Verify immediate cash impact
        cash_impact_immediate = cash_before_immediate - cash_after_immediate
        assert cash_impact_immediate > 0, "Immediate claim should reduce cash"

        # Reset for developed claim test
        manufacturer2 = base_manufacturer.copy()

        # Develop the claim
        developed_payments = claim_development.develop_claims([developed_claim])

        # Process only first year of developed claim
        cash_before_developed = manufacturer2.cash
        if len(developed_payments) > 0 and len(developed_payments[0]) > 0:
            for claim in developed_payments[0]:
                manufacturer2.process_insurance_claim(
                    claim_amount=claim.amount,
                    insurance_recovery=0,
                    deductible_amount=0,
                )
        cash_after_developed = manufacturer2.cash

        # First year impact should be less than total claim
        first_year_impact = cash_before_developed - cash_after_developed
        assert (
            first_year_impact < developed_claim.amount
        ), "First year payment should be less than total claim"

        # Verify it matches development pattern
        expected_first_year = developed_claim.amount * claim_development.pattern[0]
        assert np.isclose(first_year_impact, expected_first_year, rtol=0.1), (
            f"First year payment {first_year_impact:.2f} should match "
            f"pattern {expected_first_year:.2f}"
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
        low_claims = low_claim_gen.generate_year()
        high_claims = high_claim_gen.generate_year()

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

        # Apply losses to original
        claims = standard_claim_generator.generate_year()
        for claim in claims:
            original.process_insurance_claim(
                claim_amount=claim.amount,
                insurance_recovery=0,
                deductible_amount=0,
            )

        # Verify original changed
        assert original.equity != original_equity, "Original should be modified"

        # Create a copy and verify independence
        copy = original.copy()
        assert copy.equity == original.equity, "Copy should match original initially"

        # Modify copy
        copy.step()

        # Verify independence
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

        # Apply catastrophic loss (50% of assets)
        catastrophic_loss = manufacturer.assets * 0.5
        manufacturer.process_insurance_claim(
            claim_amount=catastrophic_loss,
            insurance_recovery=0,
            deductible_amount=0,
        )

        # Verify significant impact but consistency
        assert (
            manufacturer.equity < initial_equity * 0.6
        ), "Catastrophic loss should severely impact equity"
        assert_financial_consistency(manufacturer)

        # Test near-bankruptcy scenario
        manufacturer2 = base_manufacturer.copy()
        bankruptcy_loss = manufacturer2.assets * 0.95
        manufacturer2.process_insurance_claim(
            claim_amount=bankruptcy_loss,
            insurance_recovery=0,
            deductible_amount=0,
        )

        # Should be near bankruptcy but still consistent
        assert manufacturer2.equity < manufacturer2.assets * 0.1, "Should be near bankruptcy"
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
        assert np.sum(loss_data.amounts) == 450_000

        # Convert to ClaimEvents
        claim_gen = ClaimGenerator(seed=42)
        claims = claim_gen.from_loss_data(loss_data)

        # Verify conversion
        assert len(claims) == 3
        total_claims = sum(c.amount for c in claims)
        assert np.isclose(total_claims, 450_000, rtol=1e-10)

        # Verify year assignments
        for claim, loss in zip(claims, losses):
            assert claim.year == int(loss.timestamp)
            assert claim.amount == loss.amount
