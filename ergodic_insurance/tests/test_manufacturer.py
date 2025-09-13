"""Unit tests for the WidgetManufacturer class.

This module contains comprehensive tests for the WidgetManufacturer financial
model including balance sheet operations, insurance claim processing,
and financial metrics calculations.
"""

import math
from typing import Dict

import pytest

from ergodic_insurance.src.config import ManufacturerConfig
from ergodic_insurance.src.manufacturer import ClaimLiability, WidgetManufacturer


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
            operating_margin=0.08,
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
        assert manufacturer.operating_margin == 0.08
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
        assert metrics["operating_margin"] == 0.08
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
        if manufacturer.is_ruined:
            assert manufacturer.equity <= 0
        else:
            assert manufacturer.equity > 0
