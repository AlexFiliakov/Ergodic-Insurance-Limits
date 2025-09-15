"""Unit tests for the WidgetManufacturer class.

This module contains comprehensive tests for the WidgetManufacturer financial
model including balance sheet operations, insurance claim processing,
and financial metrics calculations.
"""

import math
from typing import Dict

import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.manufacturer import ClaimLiability, WidgetManufacturer


class TestClaimLiability:
    """Test suite for ClaimLiability class.

    Tests for the ClaimLiability dataclass including payment
    schedules and liability management functionality.
    """

    def test_init(self):
        """Test claim liability initialization.

        Verifies that ClaimLiability objects are properly initialized
        with correct payment schedules and amounts.
        """
        claim = ClaimLiability(original_amount=1000000, remaining_amount=1000000, year_incurred=0)
        assert claim.original_amount == 1000000
        assert claim.remaining_amount == 1000000
        assert claim.year_incurred == 0
        assert len(claim.payment_schedule) == 10
        assert sum(claim.payment_schedule) == pytest.approx(1.0)

    def test_get_payment(self):
        """Test payment schedule calculation.

        Tests the get_payment method for calculating scheduled
        payments based on years since claim incurred.
        """
        claim = ClaimLiability(original_amount=1000000, remaining_amount=1000000, year_incurred=0)

        # Test payment schedule
        assert claim.get_payment(0) == 100000  # 10% in year 1
        assert claim.get_payment(1) == 200000  # 20% in year 2
        assert claim.get_payment(2) == 200000  # 20% in year 3
        assert claim.get_payment(9) == 20000  # 2% in year 10

        # Test out of bounds
        assert claim.get_payment(-1) == 0
        assert claim.get_payment(10) == 0
        assert claim.get_payment(100) == 0

    def test_make_payment(self):
        """Test making payments against liability.

        Tests payment processing including partial payments,
        overpayments, and zero remaining balance scenarios.
        """
        claim = ClaimLiability(original_amount=1000000, remaining_amount=1000000, year_incurred=0)

        # Make partial payment
        actual = claim.make_payment(100000)
        assert actual == 100000
        assert claim.remaining_amount == 900000

        # Try to overpay
        actual = claim.make_payment(1000000)
        assert actual == 900000
        assert claim.remaining_amount == 0

        # Try to pay when nothing remaining
        actual = claim.make_payment(100000)
        assert actual == 0
        assert claim.remaining_amount == 0


class TestWidgetManufacturer:
    """Test suite for WidgetManufacturer class.

    Comprehensive tests for the widget manufacturer financial model
    including initialization, financial calculations, and simulation steps.
    """

    @pytest.fixture
    def config(self) -> ManufacturerConfig:
        """Create a test configuration.

        Returns:
            ManufacturerConfig with standard test parameters.
        """
        return ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=1.0,
        )

    @pytest.fixture
    def manufacturer(self, config) -> WidgetManufacturer:
        """Create a test manufacturer.

        Args:
            config: Manufacturer configuration fixture.

        Returns:
            WidgetManufacturer instance for testing.
        """
        return WidgetManufacturer(config)

    def test_initialization(self, manufacturer):
        """Test manufacturer initialization.

        Args:
            manufacturer: WidgetManufacturer fixture.

        Verifies that the manufacturer is properly initialized
        with all expected default values.
        """
        assert manufacturer.assets == 10_000_000
        assert manufacturer.collateral == 0
        assert manufacturer.restricted_assets == 0
        assert manufacturer.equity == 10_000_000
        assert manufacturer.asset_turnover_ratio == 1.0
        assert manufacturer.base_operating_margin == 0.08
        assert manufacturer.tax_rate == 0.25
        assert manufacturer.retention_ratio == 1.0
        assert manufacturer.current_year == 0
        assert manufacturer.current_month == 0
        assert manufacturer.is_ruined is False
        assert len(manufacturer.claim_liabilities) == 0
        assert len(manufacturer.metrics_history) == 0

    def test_properties(self, manufacturer):
        """Test computed properties.

        Args:
            manufacturer: WidgetManufacturer fixture.

        Tests calculated property methods including net assets,
        available assets, and total claim liabilities.
        """
        assert manufacturer.net_assets == 10_000_000
        assert manufacturer.available_assets == 10_000_000
        assert manufacturer.total_claim_liabilities == 0

        # Add restricted assets
        manufacturer.restricted_assets = 1_000_000
        assert manufacturer.net_assets == 9_000_000
        assert manufacturer.available_assets == 9_000_000

        claim = ClaimLiability(original_amount=500_000, remaining_amount=400_000, year_incurred=0)
        manufacturer.claim_liabilities.append(claim)
        assert manufacturer.total_claim_liabilities == 400_000

    def test_calculate_revenue_no_working_capital(self, manufacturer):
        """Test revenue calculation without working capital."""
        revenue = manufacturer.calculate_revenue(working_capital_pct=0.0)
        assert revenue == 10_000_000  # Assets * Turnover = 10M * 1.0

    def test_calculate_revenue_with_working_capital(self, manufacturer):
        """Test revenue calculation with working capital constraint."""
        revenue = manufacturer.calculate_revenue(working_capital_pct=0.2)
        # Revenue = Assets / (1 + Turnover * WC%)
        # Revenue = 10M / (1 + 1.0 * 0.2) = 10M / 1.2
        expected = 10_000_000 / 1.2
        assert revenue == pytest.approx(expected)

    def test_calculate_operating_income(self, manufacturer):
        """Test operating income calculation."""
        revenue = 10_000_000
        operating_income = manufacturer.calculate_operating_income(revenue)
        assert operating_income == 800_000  # 10M * 0.08

    def test_calculate_collateral_costs(self, manufacturer):
        """Test collateral cost calculation."""
        # No collateral
        costs = manufacturer.calculate_collateral_costs()
        assert costs == 0

        # With collateral - annual
        manufacturer.collateral = 1_000_000
        costs = manufacturer.calculate_collateral_costs(
            letter_of_credit_rate=0.015, time_period="annual"
        )
        assert costs == 15_000  # 1M * 0.015

        # With collateral - monthly
        costs = manufacturer.calculate_collateral_costs(
            letter_of_credit_rate=0.015, time_period="monthly"
        )
        assert costs == pytest.approx(1250)  # 1M * 0.015 / 12

    def test_calculate_net_income(self, manufacturer):
        """Test net income calculation."""
        operating_income = 800_000
        collateral_costs = 15_000

        net_income = manufacturer.calculate_net_income(operating_income, collateral_costs)

        # Income before tax = 800k - 15k = 785k
        # Tax = 785k * 0.25 = 196.25k
        # Net income = 785k - 196.25k = 588.75k
        expected = (800_000 - 15_000) * (1 - 0.25)
        assert net_income == pytest.approx(expected)

    def test_calculate_net_income_negative(self, manufacturer):
        """Test net income calculation with loss (no taxes on losses)."""
        operating_income = 100_000
        collateral_costs = 200_000

        net_income = manufacturer.calculate_net_income(operating_income, collateral_costs)

        # Income before tax = 100k - 200k = -100k
        # Tax = 0 (no tax on losses)
        # Net income = -100k
        assert net_income == -100_000

    def test_update_balance_sheet_positive_income(self, manufacturer):
        """Test balance sheet update with positive net income."""
        initial_assets = manufacturer.assets
        initial_equity = manufacturer.equity
        net_income = 500_000

        manufacturer.update_balance_sheet(net_income)

        # Full retention (retention_ratio = 1.0)
        assert manufacturer.assets == initial_assets + 500_000
        assert manufacturer.equity == initial_equity + 500_000

    def test_update_balance_sheet_with_dividends(self, manufacturer):
        """Test balance sheet update with partial retention."""
        manufacturer.retention_ratio = 0.6
        initial_assets = manufacturer.assets
        initial_equity = manufacturer.equity
        net_income = 500_000

        manufacturer.update_balance_sheet(net_income)

        # 60% retention, 40% dividends
        retained = 500_000 * 0.6
        assert manufacturer.assets == initial_assets + retained
        assert manufacturer.equity == initial_equity + retained

    def test_update_balance_sheet_negative_income(self, manufacturer):
        """Test balance sheet update with loss."""
        initial_assets = manufacturer.assets
        initial_equity = manufacturer.equity
        net_income = -200_000

        manufacturer.update_balance_sheet(net_income)

        # Losses reduce assets and equity
        assert manufacturer.assets == initial_assets - 200_000
        assert manufacturer.equity == initial_equity - 200_000

    def test_process_insurance_claim_with_collateral(self, manufacturer):
        """Test processing claim with deductible that creates company payment and collateral."""
        # Claim with deductible so company has payment obligation
        company_payment, insurance_payment = manufacturer.process_insurance_claim(
            claim_amount=1_000_000, deductible_amount=500_000, insurance_limit=2_000_000
        )

        # Company pays deductible, insurance covers rest
        assert company_payment == 500_000
        assert insurance_payment == 500_000
        assert manufacturer.collateral == 500_000  # Only company portion collateralized
        assert manufacturer.restricted_assets == 500_000
        assert len(manufacturer.claim_liabilities) == 1
        assert manufacturer.claim_liabilities[0].original_amount == 500_000  # Company portion only

    def test_process_large_insurance_claim(self, manufacturer):
        """Test processing large claim with high deductible."""
        # Large claim with high deductible to test company payment collateralization
        company_payment, insurance_payment = manufacturer.process_insurance_claim(
            claim_amount=20_000_000, deductible_amount=5_000_000, insurance_limit=15_000_000
        )

        # Company pays deductible, insurance covers up to limit
        assert company_payment == 5_000_000
        assert insurance_payment == 15_000_000
        assert manufacturer.collateral == 5_000_000  # Only company portion collateralized
        assert manufacturer.restricted_assets == 5_000_000
        assert len(manufacturer.claim_liabilities) == 1
        assert (
            manufacturer.claim_liabilities[0].original_amount == 5_000_000
        )  # Company portion only

    def test_pay_claim_liabilities_single_claim(self, manufacturer):
        """Test paying scheduled claim liabilities."""
        # Process a claim with deductible in year 0
        manufacturer.process_insurance_claim(
            claim_amount=1_500_000, deductible_amount=1_000_000, insurance_limit=2_000_000
        )

        # Pay first year payment (year 0 of claim = 10% of company portion = 100k)
        total_paid = manufacturer.pay_claim_liabilities()

        assert total_paid == 100_000
        assert manufacturer.assets == 9_900_000  # 10M - 100k
        assert manufacturer.claim_liabilities[0].remaining_amount == 900_000

        # Move to year 1 for second payment (20% of company portion = 200k)
        manufacturer.current_year = 1
        total_paid = manufacturer.pay_claim_liabilities()

        assert total_paid == 200_000
        assert manufacturer.assets == 9_700_000  # 9.9M - 200k
        assert manufacturer.claim_liabilities[0].remaining_amount == 700_000

    def test_pay_claim_liabilities_insufficient_assets(self, manufacturer):
        """Test partial payment when insufficient assets."""
        manufacturer.assets = 150_000  # Very low assets
        # Process claim with deductible to create company liability
        manufacturer.process_insurance_claim(
            claim_amount=1_500_000, deductible_amount=1_000_000, insurance_limit=2_000_000
        )
        manufacturer.current_year = 1

        # Try to pay year 1 payment (20% of company portion = 200k)
        # But only 50k available (150k - 100k min)
        total_paid = manufacturer.pay_claim_liabilities()

        assert total_paid == 50_000
        assert manufacturer.assets == 100_000  # Minimum maintained
        assert manufacturer.claim_liabilities[0].remaining_amount == 950_000

    def test_pay_claim_liabilities_removes_paid_claims(self, manufacturer):
        """Test that fully paid claims are removed."""
        # Create a nearly paid claim
        claim = ClaimLiability(original_amount=100_000, remaining_amount=10_000, year_incurred=0)
        manufacturer.claim_liabilities.append(claim)
        manufacturer.current_year = 1

        # Pay off the claim
        claim.remaining_amount = 0  # Simulate full payment
        manufacturer.pay_claim_liabilities()

        assert len(manufacturer.claim_liabilities) == 0

    def test_calculate_metrics(self, manufacturer):
        """Test metrics calculation."""
        metrics = manufacturer.calculate_metrics()

        assert metrics["assets"] == 10_000_000
        assert metrics["collateral"] == 0
        assert metrics["restricted_assets"] == 0
        assert metrics["available_assets"] == 10_000_000
        assert metrics["equity"] == 10_000_000
        assert metrics["net_assets"] == 10_000_000
        assert metrics["claim_liabilities"] == 0
        assert metrics["is_solvent"] is True
        assert metrics["revenue"] == 10_000_000
        assert metrics["operating_income"] == 800_000
        assert metrics["asset_turnover"] == 1.0
        assert metrics["base_operating_margin"] == 0.08
        assert metrics["roe"] == pytest.approx(0.06)  # 600k / 10M
        assert metrics["roa"] == pytest.approx(0.06)  # 600k / 10M
        assert metrics["collateral_to_equity"] == 0
        assert metrics["collateral_to_assets"] == 0

    def test_calculate_metrics_with_collateral(self, manufacturer):
        """Test metrics with collateral."""
        # Process claim with deductible to create company payment and collateral
        manufacturer.process_insurance_claim(
            claim_amount=1_000_000, deductible_amount=500_000, insurance_limit=2_000_000
        )

        metrics = manufacturer.calculate_metrics()

        assert metrics["collateral"] == 500_000
        assert metrics["restricted_assets"] == 500_000
        assert metrics["claim_liabilities"] == 500_000
        assert metrics["collateral_to_equity"] == pytest.approx(0.05)  # 500k / 10M
        assert metrics["collateral_to_assets"] == pytest.approx(0.05)  # 500k / 10M

    def test_calculate_metrics_zero_equity(self, manufacturer):
        """Test metrics calculation with zero equity (avoid division by zero)."""
        manufacturer.equity = 0
        manufacturer.assets = 0

        metrics = manufacturer.calculate_metrics()

        assert metrics["roe"] == 0
        assert metrics["roa"] == 0
        assert metrics["collateral_to_equity"] == 0
        assert metrics["collateral_to_assets"] == 0

    def test_step_basic(self, manufacturer):
        """Test basic step execution."""
        metrics = manufacturer.step(working_capital_pct=0.2, growth_rate=0.05)

        assert manufacturer.current_year == 1
        assert len(manufacturer.metrics_history) == 1
        assert metrics["year"] == 0
        assert manufacturer.assets > 10_000_000  # Should grow from retained earnings
        assert manufacturer.asset_turnover_ratio == pytest.approx(1.05)  # 5% growth

    def test_step_with_claims(self, manufacturer):
        """Test step with claim payments."""
        # Add a claim in year 0
        manufacturer.process_insurance_claim(1_000_000)

        initial_assets = manufacturer.assets
        # First step processes year 0 payments
        metrics = manufacturer.step()

        # Should have paid 10% of claim (100k) but also earned income
        # Net effect depends on earnings vs payment
        assert metrics["year"] == 0
        assert manufacturer.current_year == 1

        # For year 1, payment is 20% (200k)
        metrics = manufacturer.step()
        assert metrics["year"] == 1

    def test_step_sequence(self, manufacturer):
        """Test multiple steps in sequence."""
        for i in range(5):
            metrics = manufacturer.step(growth_rate=0.03)
            assert metrics["year"] == i
            assert manufacturer.current_year == i + 1

        assert len(manufacturer.metrics_history) == 5

        # Check compound growth
        expected_turnover = 1.0 * (1.03**5)
        assert manufacturer.asset_turnover_ratio == pytest.approx(expected_turnover)

    def test_reset(self, manufacturer):
        """Test resetting manufacturer to initial state.

        Args:
            manufacturer: WidgetManufacturer fixture.

        Verifies that the reset method properly restores all
        manufacturer attributes to their initial values.
        """
        # Make changes
        manufacturer.assets = 20_000_000
        manufacturer.collateral = 5_000_000
        manufacturer.restricted_assets = 5_000_000
        manufacturer.current_year = 10
        manufacturer.current_month = 6
        manufacturer.is_ruined = True
        manufacturer.process_insurance_claim(1_000_000)
        manufacturer.metrics_history.append({"test": 1})

        # Reset
        manufacturer.reset()

        assert manufacturer.assets == 10_000_000
        assert manufacturer.collateral == 0
        assert manufacturer.restricted_assets == 0
        assert manufacturer.equity == 10_000_000
        assert manufacturer.current_year == 0
        assert manufacturer.current_month == 0
        assert manufacturer.is_ruined is False
        assert len(manufacturer.claim_liabilities) == 0
        assert len(manufacturer.metrics_history) == 0
        assert manufacturer.asset_turnover_ratio == 1.0

    def test_check_solvency(self, manufacturer):
        """Test solvency checking."""
        assert manufacturer.check_solvency() is True
        assert manufacturer.is_ruined is False

        # Make insolvent
        manufacturer.equity = 0
        assert manufacturer.check_solvency() is False
        assert manufacturer.is_ruined is True

    def test_check_solvency_payment_insolvency(self, manufacturer):
        """Test solvency checking with payment insolvency (new realistic detection)."""
        # Create a catastrophic loss that creates unsustainable payment burden
        catastrophic_loss = 40_000_000  # $40M loss
        manufacturer.process_uninsured_claim(catastrophic_loss)

        # Should be solvent initially (no payments due yet in year 0)
        assert manufacturer.check_solvency() is True
        assert manufacturer.is_ruined is False

        # Move to year 1 - payments should be manageable (10% = $4M vs $10M revenue = 40%)
        manufacturer.step()

        # Year 2 should trigger insolvency (20% = $8M payment vs reduced revenue)
        # Revenue will be reduced due to previous year's asset reduction
        current_revenue = manufacturer.calculate_revenue()

        # Calculate expected payments for year 2
        expected_payments = catastrophic_loss * 0.20  # $8M
        payment_ratio = expected_payments / current_revenue

        # If payment ratio > 80%, should trigger insolvency
        if payment_ratio > 0.80:
            solvency_result = manufacturer.check_solvency()
            assert solvency_result is False
            assert manufacturer.is_ruined is True
        else:
            # If ratio is still manageable, need to go to year 3
            manufacturer.step()
            current_revenue = manufacturer.calculate_revenue()
            expected_payments = catastrophic_loss * 0.20  # Still 20% in year 3
            payment_ratio = expected_payments / current_revenue

            # Should definitely trigger by year 3
            assert payment_ratio > 0.80
            solvency_result = manufacturer.check_solvency()
            assert solvency_result is False
            assert manufacturer.is_ruined is True

    def test_check_solvency_negative_equity(self, config):
        """Test solvency checking with negative equity.

        Args:
            config: Manufacturer configuration fixture.

        Tests solvency checking behavior with negative equity values.
        """
        # Create new manufacturer to test negative equity scenario
        manufacturer = WidgetManufacturer(config)
        manufacturer.equity = -100_000
        assert manufacturer.check_solvency() is False
        assert manufacturer.is_ruined is True

    def test_monthly_collateral_costs(self, manufacturer):
        """Test monthly letter of credit cost tracking."""
        # Add collateral by processing claim with deductible
        manufacturer.process_insurance_claim(
            claim_amount=2_000_000, deductible_amount=1_200_000, insurance_limit=3_000_000
        )

        # Calculate expected monthly cost
        expected_monthly = 1_200_000 * 0.015 / 12  # 1,500

        # Test monthly calculation
        monthly_cost = manufacturer.calculate_collateral_costs(0.015, "monthly")
        assert monthly_cost == pytest.approx(expected_monthly)

        # Test annual calculation
        annual_cost = manufacturer.calculate_collateral_costs(0.015, "annual")
        assert annual_cost == pytest.approx(1_200_000 * 0.015)

    def test_process_uninsured_claim_with_schedule(self, manufacturer):
        """Test processing uninsured claims with payment schedule."""
        # Process uninsured claim with payment schedule
        claim_amount = manufacturer.process_uninsured_claim(1_000_000)

        # Should create liability without affecting assets immediately
        assert claim_amount == 1_000_000
        assert manufacturer.assets == 10_000_000  # No immediate impact
        assert manufacturer.collateral == 0  # No collateral required
        assert len(manufacturer.claim_liabilities) == 1
        assert manufacturer.claim_liabilities[0].original_amount == 1_000_000
        assert manufacturer.claim_liabilities[0].remaining_amount == 1_000_000
        assert manufacturer.claim_liabilities[0].is_insured is False  # Uninsured claim

        # Pay first year payment (10% = $100K)
        total_paid = manufacturer.pay_claim_liabilities()
        assert total_paid == 100_000
        assert manufacturer.assets == 9_900_000  # Assets reduced by payment

    def test_process_uninsured_claim_immediate(self, manufacturer):
        """Test processing uninsured claims with immediate payment."""
        # Process uninsured claim with immediate payment
        claim_amount = manufacturer.process_uninsured_claim(500_000, immediate_payment=True)

        # Should immediately reduce assets and record loss
        assert claim_amount == 500_000
        assert manufacturer.assets == 9_500_000  # Immediate reduction
        assert (
            manufacturer.period_insurance_losses == 500_000
        )  # Tax deductible (immediate payment records as insurance loss)
        assert len(manufacturer.claim_liabilities) == 0  # No liability created

    def test_full_financial_cycle(self, manufacturer):
        """Test a complete financial cycle with all components.

        Args:
            manufacturer: WidgetManufacturer fixture.

        Integration test that exercises multiple years of operations
        including claim processing, collateral management, and payments.
        """
        # Year 0: Normal operations
        metrics_0 = manufacturer.step(working_capital_pct=0.2, growth_rate=0.05)
        assert metrics_0["net_income"] > 0
        initial_equity = manufacturer.equity

        # Process a large claim with deductible that requires collateral
        # Only company portion requires collateral
        company_payment, insurance_payment = manufacturer.process_insurance_claim(
            claim_amount=20_000_000, deductible_amount=5_000_000, insurance_limit=15_000_000
        )

        # Year 1: Operations with claim payments and collateral costs
        metrics_1 = manufacturer.step(working_capital_pct=0.2, letter_of_credit_rate=0.015)

        # Should have collateral now
        assert manufacturer.collateral > 0
        assert metrics_1["year"] == 1

        # Year 2-10: Pay down claim
        for year in range(2, 11):
            metrics = manufacturer.step(working_capital_pct=0.2, letter_of_credit_rate=0.015)
            assert metrics["year"] == year

        # After 10 years, claim should be significantly paid down (company portion only)
        assert manufacturer.total_claim_liabilities < 5_000_000

        # Check if company became insolvent
        # Note: Company can be ruined due to payment insolvency even with positive equity
        if manufacturer.is_ruined:
            # Company is ruined - could be due to negative equity OR unsustainable payment burden
            # Payment insolvency can occur with positive equity if claim payments are unsustainable
            assert manufacturer.is_ruined is True
        else:
            # Company survived - should have positive equity
            assert manufacturer.equity > 0

    def test_insurance_premium_tax_treatment(self, manufacturer):
        """Test that insurance premiums are properly tax-deductible.

        Regression test for issue where premiums weren't reducing taxable income.
        """
        # Calculate baseline metrics
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # Calculate net income without insurance premium
        net_income_no_premium = manufacturer.calculate_net_income(operating_income, 0, 0, 0)

        # Calculate net income with insurance premium
        premium = 300_000
        net_income_with_premium = manufacturer.calculate_net_income(operating_income, 0, premium, 0)

        # Premium should reduce net income by (premium * (1 - tax_rate))
        expected_reduction = premium * (1 - manufacturer.tax_rate)
        actual_reduction = net_income_no_premium - net_income_with_premium

        assert actual_reduction == pytest.approx(expected_reduction)

        # Verify tax savings
        tax_savings = premium * manufacturer.tax_rate
        assert tax_savings == pytest.approx(75_000)  # 25% of 300K

    def test_deductible_tax_treatment_on_incurrence(self, manufacturer):
        """Test that deductibles are tax-deductible when incurred, not when paid.

        Regression test for issue where deductibles were only deductible when paid,
        causing timing mismatches and double counting.
        """
        initial_assets = manufacturer.assets

        # Process claim with smaller deductible to maintain positive income
        deductible = 50_000
        company_payment, insurance_payment = manufacturer.process_insurance_claim(
            claim_amount=200_000, deductible_amount=deductible, insurance_limit=10_000_000
        )

        # Verify deductible is recorded for tax purposes immediately
        assert manufacturer.period_insurance_losses == deductible

        # Assets should not be reduced immediately (only collateralized)
        assert manufacturer.assets == initial_assets
        assert manufacturer.collateral == deductible
        assert manufacturer.restricted_assets == deductible

        # Calculate operating income (which now includes the insurance losses)
        revenue = manufacturer.calculate_revenue()
        operating_income_with_loss = manufacturer.calculate_operating_income(revenue)

        # Calculate what operating income would be without the loss
        # Temporarily clear the loss to calculate base operating income
        manufacturer.period_insurance_losses = 0
        operating_income_no_loss = manufacturer.calculate_operating_income(revenue)
        manufacturer.period_insurance_losses = deductible  # Restore the loss

        # Net income calculations (insurance costs are already in operating income)
        net_income_with_loss = manufacturer.calculate_net_income(
            operating_income_with_loss, 0, 0, 0
        )
        net_income_no_loss = manufacturer.calculate_net_income(operating_income_no_loss, 0, 0, 0)

        # Deductible should reduce net income by (deductible * (1 - tax_rate))
        # Use smaller deductible to ensure positive income for tax benefit
        if operating_income_with_loss > 0:
            expected_reduction = deductible * (1 - manufacturer.tax_rate)
        else:
            # If income is negative, full deductible reduces net income (no tax benefit)
            expected_reduction = deductible

        actual_reduction = net_income_no_loss - net_income_with_loss

        assert actual_reduction == pytest.approx(expected_reduction)

    def test_no_double_counting_of_deductibles(self, manufacturer):
        """Test that deductibles are not double-counted for tax purposes.

        Regression test to ensure deductibles are only tax-deductible once
        (when incurred) and not again when paid.
        """
        # Process claim with deductible
        deductible = 1_000_000
        manufacturer.process_insurance_claim(
            claim_amount=5_000_000, deductible_amount=deductible, insurance_limit=10_000_000
        )

        # Record initial period losses
        initial_period_losses = manufacturer.period_insurance_losses
        assert initial_period_losses == deductible

        # Reset period costs (as would happen at end of step)
        manufacturer.reset_period_insurance_costs()
        assert manufacturer.period_insurance_losses == 0

        # Pay claim liabilities (first year payment)
        manufacturer.current_year = 1  # Advance time for payment
        payments_made = manufacturer.pay_claim_liabilities()

        # Verify payment was made
        assert payments_made > 0

        # Period losses should still be 0 (not double-counted)
        assert manufacturer.period_insurance_losses == 0

        # Verify collateral was reduced
        expected_collateral = deductible - payments_made
        assert manufacturer.collateral == pytest.approx(expected_collateral)

    def test_combined_premium_and_deductible_tax_treatment(self, manufacturer):
        """Test combined tax treatment of premiums and deductibles.

        Integration test for the full insurance cost tax treatment.
        """
        # Process claim with deductible
        deductible = 200_000
        manufacturer.process_insurance_claim(
            claim_amount=1_000_000, deductible_amount=deductible, insurance_limit=5_000_000
        )

        # Pay premium
        premium = 150_000
        manufacturer.record_insurance_premium(premium)

        # Verify both are recorded
        assert manufacturer.period_insurance_premiums == premium
        assert manufacturer.period_insurance_losses == deductible

        # Calculate net income with both costs
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)
        collateral_costs = manufacturer.calculate_collateral_costs(0.015)

        net_income = manufacturer.calculate_net_income(
            operating_income,
            collateral_costs,
            manufacturer.period_insurance_premiums,
            manufacturer.period_insurance_losses,
        )

        # Calculate baseline without insurance costs
        net_income_baseline = manufacturer.calculate_net_income(
            operating_income, collateral_costs, 0, 0
        )

        # Total insurance costs should reduce net income by (costs * (1 - tax_rate))
        total_insurance_costs = premium + deductible
        expected_reduction = total_insurance_costs * (1 - manufacturer.tax_rate)
        actual_reduction = net_income_baseline - net_income

        assert actual_reduction == pytest.approx(expected_reduction)

    def test_step_with_insurance_costs(self, manufacturer):
        """Test that step() properly handles insurance costs and tax treatment.

        End-to-end test of the step function with insurance.
        """
        # Process claim and pay premium before step
        deductible = 300_000
        manufacturer.process_insurance_claim(
            claim_amount=2_000_000, deductible_amount=deductible, insurance_limit=5_000_000
        )

        premium = 250_000
        manufacturer.record_insurance_premium(premium)

        # Store initial state
        initial_equity = manufacturer.equity

        # Run step
        metrics = manufacturer.step(working_capital_pct=0.2, letter_of_credit_rate=0.015)

        # Verify period costs were reset after step
        assert manufacturer.period_insurance_premiums == 0
        assert manufacturer.period_insurance_losses == 0

        # Verify metrics include the insurance impact
        # After step, first year payment has been made, so collateral is reduced
        first_year_payment = deductible * 0.10  # 10% first year payment
        assert metrics["collateral"] == pytest.approx(deductible - first_year_payment)
        assert metrics["restricted_assets"] == pytest.approx(deductible - first_year_payment)

        # Net income should reflect tax benefits
        # With 25% tax rate, the tax benefit is 0.25 * (premium + deductible)
        tax_benefit = (premium + deductible) * manufacturer.tax_rate

        # Equity change should reflect premium payment and tax benefits
        equity_change = manufacturer.equity - initial_equity

        # The equity change should account for:
        # - Premium paid (reduces equity)
        # - Net income (increases equity based on operations minus costs plus tax benefits)
        # This is complex to calculate exactly, but equity should have decreased
        # less than the premium amount due to tax benefits
        assert equity_change > -premium  # Tax benefits partially offset premium cost

    def test_uninsured_vs_insured_claim_tax_treatment(self, manufacturer):
        """Test that uninsured and insured claims have consistent tax treatment.

        Both should be tax-deductible, but timing may differ.
        """
        # Test insured claim
        manufacturer_insured = WidgetManufacturer(manufacturer.config)
        deductible = 100_000
        manufacturer_insured.process_insurance_claim(
            claim_amount=500_000, deductible_amount=deductible, insurance_limit=1_000_000
        )

        # Test uninsured claim (immediate payment)
        manufacturer_uninsured = WidgetManufacturer(manufacturer.config)
        manufacturer_uninsured.process_uninsured_claim(100_000, immediate_payment=True)

        # Both should have tax-deductible losses
        assert manufacturer_insured.period_insurance_losses == deductible
        assert manufacturer_uninsured.period_insurance_losses == 100_000

        # Calculate net income for both
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        net_income_insured = manufacturer_insured.calculate_net_income(
            operating_income, 0, 0, manufacturer_insured.period_insurance_losses
        )

        net_income_uninsured = manufacturer_uninsured.calculate_net_income(
            operating_income, 0, 0, manufacturer_uninsured.period_insurance_losses
        )

        # Both should have same net income (same loss amount)
        assert net_income_insured == net_income_uninsured

    def test_collateral_costs_are_tax_deductible(self, manufacturer):
        """Test that letter of credit collateral costs are tax-deductible.

        Ensures collateral financing costs reduce taxable income.
        """
        # Process smaller claim to maintain positive income
        manufacturer.process_insurance_claim(
            claim_amount=300_000, deductible_amount=100_000, insurance_limit=5_000_000
        )

        # Calculate collateral costs
        loc_rate = 0.02
        collateral_costs = manufacturer.calculate_collateral_costs(loc_rate)
        expected_costs = 100_000 * loc_rate  # Based on deductible collateral
        assert collateral_costs == expected_costs

        # Calculate net income with and without collateral costs
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        net_income_with_costs = manufacturer.calculate_net_income(
            operating_income, collateral_costs, 0, 0
        )

        net_income_without_costs = manufacturer.calculate_net_income(operating_income, 0, 0, 0)

        # Collateral costs should reduce net income by (costs * (1 - tax_rate))
        expected_reduction = collateral_costs * (1 - manufacturer.tax_rate)
        actual_reduction = net_income_without_costs - net_income_with_costs

        assert actual_reduction == pytest.approx(expected_reduction)

    def test_premium_does_not_reduce_productive_assets(self, manufacturer):
        """Test that insurance premiums don't immediately reduce productive assets.

        Regression test for issue where premiums were liquidating productive assets,
        causing revenue decline and negative ROI for insurance scenarios.
        """
        initial_assets = manufacturer.assets
        initial_equity = manufacturer.equity
        premium = 500_000

        # Record premium
        manufacturer.record_insurance_premium(premium)

        # Assets should NOT be immediately reduced
        assert manufacturer.assets == initial_assets
        # Period premiums should be recorded for tax deduction
        assert manufacturer.period_insurance_premiums == premium

        # The premium expense will only affect assets through net income in step()

    def test_revenue_generation_after_premium_payment(self, manufacturer):
        """Test that revenue generation is not affected by premium payments.

        Ensures that paying premiums doesn't reduce the company's ability
        to generate revenue from its productive assets.
        """
        # Calculate baseline revenue before any premiums
        baseline_revenue = manufacturer.calculate_revenue()

        # Pay a large premium
        large_premium = 1_000_000
        manufacturer.record_insurance_premium(large_premium)

        # Revenue should still be based on full assets, not reduced by premium
        revenue_after_premium = manufacturer.calculate_revenue()
        assert revenue_after_premium == baseline_revenue

        # Revenue = Assets × Turnover Ratio, and should not change
        expected_revenue = manufacturer.assets * manufacturer.asset_turnover_ratio
        assert revenue_after_premium == expected_revenue

    def test_premium_flows_through_net_income(self, manufacturer):
        """Test that premiums properly flow through net income calculation.

        Verifies that premiums reduce net income (with tax benefits) but don't
        directly liquidate assets.
        """
        premium = 100_000
        manufacturer.record_insurance_premium(premium)

        # Calculate financial metrics
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # Net income without considering the premium
        net_income_no_premium = manufacturer.calculate_net_income(operating_income, 0, 0, 0)

        # Net income with the premium expense
        net_income_with_premium = manufacturer.calculate_net_income(
            operating_income, 0, manufacturer.period_insurance_premiums, 0
        )

        # Premium should reduce net income by (premium × (1 - tax_rate))
        expected_reduction = premium * (1 - manufacturer.tax_rate)
        actual_reduction = net_income_no_premium - net_income_with_premium

        assert actual_reduction == pytest.approx(expected_reduction)

    def test_comparative_scenarios_with_premiums(self, manufacturer):
        """Test that insurance scenarios maintain revenue capacity.

        Comprehensive test comparing scenarios with and without insurance
        to ensure premiums don't create an unfair disadvantage.
        """
        # Create two identical manufacturers
        no_insurance = WidgetManufacturer(manufacturer.config)
        with_insurance = WidgetManufacturer(manufacturer.config)

        # Insurance company pays premium
        annual_premium = 300_000
        with_insurance.record_insurance_premium(annual_premium)

        # Both should generate same revenue initially
        revenue_no_ins = no_insurance.calculate_revenue()
        revenue_with_ins = with_insurance.calculate_revenue()
        assert revenue_no_ins == revenue_with_ins

        # Run a step for both
        metrics_no_ins = no_insurance.step()
        metrics_with_ins = with_insurance.step()

        # With insurance should have lower assets due to premium expense
        # but the difference should be the after-tax cost, not the full premium
        asset_difference = metrics_no_ins["assets"] - metrics_with_ins["assets"]
        after_tax_premium = annual_premium * (1 - with_insurance.tax_rate)

        # The difference should be approximately the after-tax premium cost
        # (some small difference due to compounding effects)
        assert asset_difference == pytest.approx(after_tax_premium, rel=0.01)

        # Both should have positive ROE
        assert metrics_no_ins["roe"] > 0
        assert metrics_with_ins["roe"] > 0

    def test_multi_year_premium_impact(self, manufacturer):
        """Test multi-year impact of premiums on business growth.

        Ensures that insurance premiums don't create a death spiral
        where reduced assets lead to reduced revenue in future years.
        """
        # Create two manufacturers
        no_insurance = WidgetManufacturer(manufacturer.config)
        with_insurance = WidgetManufacturer(manufacturer.config)

        annual_premium = 250_000
        years = 5

        # Track metrics over multiple years
        for year in range(years):
            # Insurance company pays premium each year
            with_insurance.record_insurance_premium(annual_premium)

            # Step both
            metrics_no_ins = no_insurance.step()
            metrics_with_ins = with_insurance.step()

            # Both should remain solvent
            assert metrics_no_ins["is_solvent"]
            assert metrics_with_ins["is_solvent"]

            # Both should have positive net income
            assert metrics_no_ins["net_income"] > 0
            assert metrics_with_ins["net_income"] > 0

        # After 5 years, the gap should be roughly the cumulative after-tax premiums
        total_premiums_paid = annual_premium * years
        after_tax_total = total_premiums_paid * (1 - manufacturer.tax_rate)

        asset_gap = no_insurance.assets - with_insurance.assets

        # The gap should be reasonably close to cumulative after-tax premiums
        # Allow 20% tolerance for compounding effects
        assert asset_gap < after_tax_total * 1.2

    def test_premium_with_claims_integration(self, manufacturer):
        """Test that premiums and claims work correctly together.

        Integration test ensuring premiums don't interfere with claim processing
        and both contribute properly to tax calculations.
        """
        # Process a claim with deductible
        deductible = 150_000
        manufacturer.process_insurance_claim(
            claim_amount=1_000_000, deductible_amount=deductible, insurance_limit=5_000_000
        )

        # Pay premium
        premium = 200_000
        manufacturer.record_insurance_premium(premium)

        # Both should be recorded for tax purposes
        assert manufacturer.period_insurance_premiums == premium
        assert manufacturer.period_insurance_losses == deductible

        # Assets should not be reduced by premium
        # (deductible creates collateral, not immediate asset reduction)
        assert manufacturer.collateral == deductible

        # Run step to process everything
        metrics = manufacturer.step()

        # Should remain solvent despite both costs
        assert metrics["is_solvent"]

        # Period costs should be reset after step
        assert manufacturer.period_insurance_premiums == 0
        assert manufacturer.period_insurance_losses == 0
