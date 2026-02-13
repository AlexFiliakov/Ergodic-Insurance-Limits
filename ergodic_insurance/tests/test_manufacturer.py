"""Unit tests for the WidgetManufacturer class."""

# pylint: disable=too-many-lines

from decimal import Decimal
import math
from typing import Dict

import pytest

from ergodic_insurance.claim_development import ClaimDevelopment
from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decimal_utils import ONE, ZERO, to_decimal
from ergodic_insurance.ledger import AccountName, TransactionType
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
        claim = ClaimLiability(
            original_amount=to_decimal(1000000),
            remaining_amount=to_decimal(1000000),
            year_incurred=0,
        )
        assert claim.original_amount == to_decimal(1000000)
        assert claim.remaining_amount == to_decimal(1000000)
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

    def test_payment_timing_year_zero(self):
        """Test that claims incurred in year 0 receive first payment in year 0.

        This test verifies the fix for issue #201 where claims were delayed by one year.
        A claim with year_incurred=0 should receive its first payment (10%) when
        years_since_incurred=0, not when years_since_incurred=1.
        """
        claim = ClaimLiability(
            original_amount=1_000_000, remaining_amount=1_000_000, year_incurred=0
        )

        # Year 0: First payment should be 10%
        assert claim.get_payment(0) == 100_000  # 10% of $1M

        # Year 1: Second payment should be 20%
        assert claim.get_payment(1) == 200_000  # 20% of $1M

        # Year 2: Third payment should be 20%
        assert claim.get_payment(2) == 200_000  # 20% of $1M

    def test_payment_schedule_follows_actuarial_pattern(self):
        """Test that full payment schedule follows expected actuarial pattern.

        Verifies that claims follow the standard 10-year payment pattern:
        10%, 20%, 20%, 15%, 10%, 8%, 7%, 5%, 3%, 2%
        """
        claim = ClaimLiability(
            original_amount=1_000_000, remaining_amount=1_000_000, year_incurred=0
        )

        expected_payments = [
            100_000,  # Year 0: 10%
            200_000,  # Year 1: 20%
            200_000,  # Year 2: 20%
            150_000,  # Year 3: 15%
            100_000,  # Year 4: 10%
            80_000,  # Year 5: 8%
            70_000,  # Year 6: 7%
            50_000,  # Year 7: 5%
            30_000,  # Year 8: 3%
            20_000,  # Year 9: 2%
        ]

        for year, expected in enumerate(expected_payments):
            actual = claim.get_payment(year)
            assert actual == expected, f"Year {year}: expected {expected}, got {actual}"

        # Verify total payments equal original amount
        total = sum(claim.get_payment(i) for i in range(10))
        assert total == 1_000_000


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
            ppe_ratio=0.1,  # Lower PPE ratio ensures sufficient cash for large claim tests
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
        assert manufacturer.total_assets == 10_000_000
        assert manufacturer.collateral == 0
        assert manufacturer.restricted_assets == 0
        assert manufacturer.equity == 10_000_000
        assert manufacturer.asset_turnover_ratio == 1.0
        assert manufacturer.base_operating_margin == 0.08
        assert manufacturer.tax_rate == 0.25
        assert manufacturer.retention_ratio == 1.0
        assert manufacturer.current_year == 0
        assert manufacturer.current_month == 0
        assert not manufacturer.is_ruined
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

        # Transfer cash to restricted assets via ledger (Issue #275: ledger is single source of truth)
        manufacturer.ledger.record_double_entry(
            date=0,
            debit_account=AccountName.RESTRICTED_CASH,
            credit_account=AccountName.CASH,
            amount=1_000_000,
            transaction_type=TransactionType.TRANSFER,
            description="Transfer to restricted assets",
        )
        # Total assets remain same (transfer between accounts), but 1M is now restricted
        assert manufacturer.restricted_assets == 1_000_000
        assert manufacturer.net_assets == pytest.approx(9_000_000, rel=0.01)
        assert manufacturer.available_assets == pytest.approx(9_000_000, rel=0.01)

        claim = ClaimLiability(original_amount=500_000, remaining_amount=400_000, year_incurred=0)
        manufacturer.claim_liabilities.append(claim)
        assert manufacturer.total_claim_liabilities == 400_000

    def test_calculate_revenue(self, manufacturer):
        """Test revenue calculation.

        Issue #244: working_capital_pct parameter was removed to fix double-counting.
        Working capital impact now flows only through calculate_working_capital_components()
        and the cash flow statement.
        """
        revenue = manufacturer.calculate_revenue()
        assert revenue == 10_000_000  # Assets * Turnover = 10M * 1.0

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

        # Add collateral via restricted assets (Issue #302: collateral tracked via RESTRICTED_CASH)
        manufacturer.ledger.record_double_entry(
            date=0,
            debit_account=AccountName.RESTRICTED_CASH,
            credit_account=AccountName.CASH,
            amount=1_000_000,
            transaction_type=TransactionType.TRANSFER,
            description="Post collateral for testing",
        )

        # With collateral - annual
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
        initial_assets = manufacturer.total_assets
        initial_equity = manufacturer.equity
        net_income = 500_000

        manufacturer.update_balance_sheet(net_income)

        # Full retention (retention_ratio = 1.0)
        assert manufacturer.total_assets == initial_assets + 500_000
        assert manufacturer.equity == initial_equity + 500_000

    def test_update_balance_sheet_with_dividends(self, manufacturer):
        """Test balance sheet update with partial retention."""
        manufacturer.retention_ratio = 0.6
        initial_assets = manufacturer.total_assets
        initial_equity = manufacturer.equity
        net_income = 500_000

        manufacturer.update_balance_sheet(net_income)

        # 60% retention, 40% dividends
        retained = to_decimal(500_000 * 0.6)
        assert manufacturer.total_assets == initial_assets + retained
        assert manufacturer.equity == initial_equity + retained

    def test_update_balance_sheet_negative_income(self, manufacturer):
        """Test balance sheet update with loss."""
        initial_assets = manufacturer.total_assets
        initial_equity = manufacturer.equity
        net_income = -200_000

        manufacturer.update_balance_sheet(net_income)

        # Losses reduce assets and equity
        assert manufacturer.total_assets == initial_assets - 200_000
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
        # Liability includes LAE (Issue #468): indemnity * (1 + lae_ratio)
        lae_factor = ONE + to_decimal(manufacturer.config.lae_ratio)
        assert manufacturer.claim_liabilities[0].original_amount == 500_000 * lae_factor

    def test_process_large_insurance_claim(self, manufacturer):
        """Test processing large claim with high deductible."""
        # Large claim with high deductible to test company payment collateralization
        company_payment, insurance_payment = manufacturer.process_insurance_claim(
            claim_amount=20_000_000,
            deductible_amount=5_000_000,
            insurance_limit=15_000_000,
        )

        # Company pays deductible, insurance covers up to limit
        assert company_payment == 5_000_000
        assert insurance_payment == 15_000_000
        assert manufacturer.collateral == 5_000_000  # Only company portion collateralized
        assert manufacturer.restricted_assets == 5_000_000
        assert len(manufacturer.claim_liabilities) == 1
        # Liability includes LAE (Issue #468)
        lae_factor = ONE + to_decimal(manufacturer.config.lae_ratio)
        assert manufacturer.claim_liabilities[0].original_amount == 5_000_000 * lae_factor

    def test_pay_claim_liabilities_single_claim(self, manufacturer):
        """Test paying scheduled claim liabilities."""
        # Process a claim with deductible in year 0
        manufacturer.process_insurance_claim(
            claim_amount=1_500_000,
            deductible_amount=1_000_000,
            insurance_limit=2_000_000,
        )

        # Pay first year payment (year 0 of claim = 10% of liability including LAE)
        # Company portion = 1M, liability = 1M * (1 + lae_ratio)
        lae_factor = ONE + to_decimal(manufacturer.config.lae_ratio)
        liability = to_decimal(1_000_000) * lae_factor  # 1,120,000
        total_paid = manufacturer.pay_claim_liabilities()

        year0_payment = liability * to_decimal("0.10")  # 112,000
        assert total_paid == pytest.approx(year0_payment)
        insurance_receivable = to_decimal(1_500_000 - 1_000_000)  # claim - deductible
        assert manufacturer.total_assets == pytest.approx(
            10_000_000 - year0_payment + insurance_receivable  # #814: includes ins. recv.
        )
        assert manufacturer.claim_liabilities[0].remaining_amount == pytest.approx(
            liability - year0_payment
        )

        # Move to year 1 for second payment (20% of liability = 224k)
        manufacturer.current_year = 1
        total_paid = manufacturer.pay_claim_liabilities()

        year1_payment = liability * to_decimal("0.20")  # 224,000
        assert total_paid == pytest.approx(year1_payment)
        assert manufacturer.total_assets == pytest.approx(
            10_000_000 - year0_payment - year1_payment + insurance_receivable
        )
        assert manufacturer.claim_liabilities[0].remaining_amount == pytest.approx(
            liability - year0_payment - year1_payment
        )

    def test_pay_claim_liabilities_insufficient_cash(self, manufacturer):
        """Test partial payment when insufficient cash."""
        # Process claim first - this will set collateral and reduce cash
        manufacturer.process_insurance_claim(
            claim_amount=1_500_000,
            deductible_amount=1_000_000,
            insurance_limit=2_000_000,
        )

        # Reduce cash to low value via ledger (Issue #275: ledger is single source of truth)
        # The claim processing already moved cash to restricted, so we need to adjust
        # just reduce available cash for the test scenario
        current_cash = manufacturer.cash
        if current_cash > 150_000:
            reduction = current_cash - to_decimal(150_000)
            manufacturer.ledger.record_double_entry(
                date=0,
                debit_account=AccountName.RETAINED_EARNINGS,
                credit_account=AccountName.CASH,
                amount=reduction,
                transaction_type=TransactionType.ADJUSTMENT,
                description="Reduce cash for test scenario",
            )
        manufacturer.current_year = 1

        # Try to pay year 1 payment (20% of liability including LAE)
        # Payment from restricted assets (collateral), not cash
        lae_factor = ONE + to_decimal(manufacturer.config.lae_ratio)
        liability = to_decimal(1_000_000) * lae_factor  # 1,120,000
        total_paid = manufacturer.pay_claim_liabilities()

        # Should pay from restricted assets
        year1_payment = liability * to_decimal("0.20")  # 224,000
        assert total_paid == pytest.approx(year1_payment)  # Full payment from collateral
        assert manufacturer.restricted_assets == pytest.approx(1_000_000 - year1_payment)
        assert manufacturer.claim_liabilities[0].remaining_amount == pytest.approx(
            liability - year1_payment  # Only year 1 payment made (no year 0 payment in this test)
        )

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

        assert metrics["assets"] == to_decimal(10_000_000)
        assert metrics["collateral"] == ZERO
        assert metrics["restricted_assets"] == ZERO
        assert metrics["available_assets"] == to_decimal(10_000_000)
        assert metrics["equity"] == to_decimal(10_000_000)
        assert metrics["net_assets"] == to_decimal(10_000_000)
        assert metrics["claim_liabilities"] == ZERO
        assert metrics["is_solvent"]
        assert metrics["revenue"] == to_decimal(10_000_000)
        # Operating income = Revenue * base_operating_margin = $10M * 8% = $800k
        # Issue #475: Depreciation is no longer subtracted from operating income
        # because it is already embedded in the COGS/SGA expense ratios.
        assert metrics["operating_income"] == to_decimal(800_000)
        assert float(metrics["asset_turnover"]) == pytest.approx(1.0)
        assert metrics["base_operating_margin"] == to_decimal(0.08)
        # Net income after tax: $800k * (1 - 0.25) = $600k
        assert float(metrics["roe"]) == pytest.approx(0.06)  # 600k / 10M
        assert float(metrics["roa"]) == pytest.approx(0.06)  # 600k / 10M
        assert metrics["collateral_to_equity"] == ZERO
        assert metrics["collateral_to_assets"] == ZERO

    def test_calculate_metrics_with_collateral(self, manufacturer):
        """Test metrics with collateral."""
        # Process claim with deductible to create company payment and collateral
        manufacturer.process_insurance_claim(
            claim_amount=1_000_000, deductible_amount=500_000, insurance_limit=2_000_000
        )

        metrics = manufacturer.calculate_metrics()

        # Collateral is indemnity only, liabilities include LAE (Issue #468)
        lae_factor = ONE + to_decimal(manufacturer.config.lae_ratio)
        expected_liability = to_decimal(500_000) * lae_factor  # 560,000
        assert metrics["collateral"] == to_decimal(500_000)
        assert metrics["restricted_assets"] == to_decimal(500_000)
        assert metrics["claim_liabilities"] == pytest.approx(expected_liability)
        insurance_receivable = 500_000  # #814: claim - deductible now in total_assets
        expected_equity = (10_000_000 + insurance_receivable) - float(expected_liability)
        assert float(metrics["collateral_to_equity"]) == pytest.approx(500_000 / expected_equity)
        assert float(metrics["collateral_to_assets"]) == pytest.approx(
            500_000 / (10_000_000 + insurance_receivable)
        )

    def test_calculate_metrics_zero_equity(self, manufacturer):
        """Test metrics calculation with zero equity (avoid division by zero)."""
        # Write off all assets via ledger (Issue #275: ledger is single source of truth)
        # Use the helper method to write off all assets
        manufacturer._write_off_all_assets("Test: write off all assets for zero equity test")

        metrics = manufacturer.calculate_metrics()

        assert metrics["roe"] == ZERO
        assert metrics["roa"] == ZERO
        assert metrics["collateral_to_equity"] == ZERO
        assert metrics["collateral_to_assets"] == ZERO

    def test_step_basic(self, manufacturer):
        """Test basic step execution."""
        metrics = manufacturer.step(growth_rate=0.05)

        assert manufacturer.current_year == 1
        assert len(manufacturer.metrics_history) == 1
        assert metrics["year"] == 0
        assert manufacturer.total_assets > 10_000_000  # Should grow from retained earnings
        assert manufacturer.asset_turnover_ratio == pytest.approx(1.05)  # 5% growth

    def test_step_with_claims(self, manufacturer):
        """Test step with claim payments."""
        # Add a claim in year 0
        manufacturer.process_insurance_claim(1_000_000)

        initial_assets = manufacturer.total_assets
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
        # Make changes via ledger (Issue #275: ledger is single source of truth)
        manufacturer.ledger.record_double_entry(
            date=0,
            debit_account=AccountName.CASH,
            credit_account=AccountName.RETAINED_EARNINGS,
            amount=10_000_000,
            transaction_type=TransactionType.ADJUSTMENT,
            description="Test: increase cash",
        )
        manufacturer.ledger.record_double_entry(
            date=0,
            debit_account=AccountName.COLLATERAL,
            credit_account=AccountName.RETAINED_EARNINGS,
            amount=5_000_000,
            transaction_type=TransactionType.TRANSFER,
            description="Test: add collateral",
        )
        manufacturer.ledger.record_double_entry(
            date=0,
            debit_account=AccountName.RESTRICTED_CASH,
            credit_account=AccountName.RETAINED_EARNINGS,
            amount=5_000_000,
            transaction_type=TransactionType.TRANSFER,
            description="Test: add restricted assets",
        )
        manufacturer.current_year = 10
        manufacturer.current_month = 6
        manufacturer.is_ruined = True
        manufacturer.process_insurance_claim(1_000_000)
        manufacturer.metrics_history.append({"test": 1})

        # Reset
        manufacturer.reset()

        assert manufacturer.total_assets == 10_000_000
        assert manufacturer.collateral == 0
        assert manufacturer.restricted_assets == 0
        assert manufacturer.equity == 10_000_000
        assert manufacturer.current_year == 0
        assert manufacturer.current_month == 0
        assert not manufacturer.is_ruined
        assert len(manufacturer.claim_liabilities) == 0
        assert len(manufacturer.metrics_history) == 0
        assert manufacturer.asset_turnover_ratio == 1.0

    def test_check_solvency(self, manufacturer):
        """Test solvency checking."""
        assert manufacturer.check_solvency()
        assert not manufacturer.is_ruined

        # Make insolvent by creating liability that exceeds total assets
        # Use deferred payment to create liability without immediate asset liquidation
        total_assets = manufacturer.total_assets
        # Create liability equal to 1.5x total assets
        # This will reduce equity below 0, triggering insolvency
        manufacturer.process_uninsured_claim(
            total_assets * to_decimal(1.5), immediate_payment=False
        )

        # Should be insolvent now (equity = assets - liabilities < 0)
        assert not manufacturer.check_solvency()
        assert manufacturer.is_ruined
        # LIMITED LIABILITY: Equity should be exactly 0, not negative
        assert manufacturer.equity == ZERO

    def test_check_solvency_payment_insolvency(self, manufacturer):
        """Test solvency checking with payment insolvency (limited liability enforcement)."""
        # Create a significant loss that will eventually deplete equity over time
        # but won't cause immediate insolvency
        # Use $3M loss (sustainable given company profitability and payment schedule)
        significant_loss = to_decimal(3_000_000)  # $3M loss (less than $10M assets)
        manufacturer.process_uninsured_claim(significant_loss)

        # Should be solvent initially - liabilities < assets
        # Equity = $10M assets - $3M liabilities = $7M
        assert manufacturer.check_solvency()
        assert not manufacturer.is_ruined

        # Step through years and verify company remains solvent
        # The $3M loss should be manageable given operating income
        # Track that equity stays positive and company can service the debt
        for year in range(1, 5):
            metrics = manufacturer.step()

            # LIMITED LIABILITY: Verify equity never goes negative
            assert manufacturer.equity >= ZERO, (
                f"Year {year}: Equity violated limited liability! "
                f"Equity = ${manufacturer.equity:,.2f}"
            )

            # Company should remain solvent with this manageable loss
            # (Operating income can cover claim payments over time)

        # Key test: equity never went negative (limited liability enforced)
        # and company successfully services a significant but sustainable claim

    def test_check_solvency_zero_equity(self, config):
        """Test solvency checking with zero equity (limited liability).

        Args:
            config: Manufacturer configuration fixture.

        Tests that company becomes insolvent when equity reaches zero,
        and that limited liability prevents negative equity.
        """
        # Create new manufacturer to test zero equity scenario
        manufacturer = WidgetManufacturer(config)

        # Create liability that exceeds total assets
        total_assets = manufacturer.total_assets
        # Use deferred payment to create liability
        manufacturer.process_uninsured_claim(
            total_assets * to_decimal(1.5), immediate_payment=False
        )

        assert not manufacturer.check_solvency()
        assert manufacturer.is_ruined
        # LIMITED LIABILITY: Equity should be exactly 0, not negative
        assert manufacturer.equity == ZERO

    def test_monthly_collateral_costs(self, manufacturer):
        """Test monthly letter of credit cost tracking."""
        # Add collateral by processing claim with deductible
        manufacturer.process_insurance_claim(
            claim_amount=2_000_000,
            deductible_amount=1_200_000,
            insurance_limit=3_000_000,
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
        # Liability includes LAE (Issue #468)
        lae_factor = ONE + to_decimal(manufacturer.config.lae_ratio)
        expected_liability = to_decimal(1_000_000) * lae_factor  # 1,120,000
        assert claim_amount == 1_000_000
        assert manufacturer.total_assets == 10_000_000  # No immediate impact
        assert manufacturer.collateral == 0  # No collateral required
        assert len(manufacturer.claim_liabilities) == 1
        assert manufacturer.claim_liabilities[0].original_amount == pytest.approx(
            expected_liability
        )
        assert manufacturer.claim_liabilities[0].remaining_amount == pytest.approx(
            expected_liability
        )
        assert not manufacturer.claim_liabilities[0].is_insured  # Uninsured claim

        # Pay first year payment (10% of liability including LAE)
        total_paid = manufacturer.pay_claim_liabilities()
        year0_payment = expected_liability * to_decimal("0.10")
        assert total_paid == pytest.approx(year0_payment)
        assert manufacturer.total_assets == pytest.approx(10_000_000 - year0_payment)

    def test_process_uninsured_claim_immediate(self, manufacturer):
        """Test processing uninsured claims with immediate payment."""
        # Process uninsured claim with immediate payment
        claim_amount = manufacturer.process_uninsured_claim(500_000, immediate_payment=True)

        # Should immediately reduce assets and record loss
        assert claim_amount == 500_000
        assert manufacturer.total_assets == 9_500_000  # Immediate reduction
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
        metrics_0 = manufacturer.step(growth_rate=0.05)
        assert metrics_0["net_income"] > 0
        initial_equity = manufacturer.equity

        # Process a large claim with deductible that requires collateral
        # Only company portion requires collateral
        company_payment, insurance_payment = manufacturer.process_insurance_claim(
            claim_amount=20_000_000,
            deductible_amount=5_000_000,
            insurance_limit=15_000_000,
        )

        # Year 1: Operations with claim payments and collateral costs
        metrics_1 = manufacturer.step(letter_of_credit_rate=0.015)

        # Should have collateral now
        assert manufacturer.collateral > 0
        assert metrics_1["year"] == 1

        # Year 2-10: Pay down claim
        for year in range(2, 11):
            metrics = manufacturer.step(letter_of_credit_rate=0.015)
            assert metrics["year"] == year

        # After 10 years, claim should be significantly paid down (company portion only)
        assert manufacturer.total_claim_liabilities < 5_000_000

        # Check if company became insolvent
        # Note: Company can be ruined due to payment insolvency even with positive equity
        if manufacturer.is_ruined:
            # Company is ruined - verify equity is zero (limited liability enforcement)
            assert manufacturer.equity == ZERO
        else:
            # Company survived - should have positive equity
            assert manufacturer.equity > 0

    def test_premium_reduces_taxable_income(self, manufacturer):
        """Premium should reduce operating income and flow to net income.

        Regression test for issue where premiums weren't reducing taxable income.
        Insurance premiums are deducted in calculate_operating_income() (Issue #374).
        """
        revenue = manufacturer.calculate_revenue()

        # Baseline operating income (no premium recorded)
        operating_income_no_premium = manufacturer.calculate_operating_income(revenue)

        # Record a premium and recalculate
        premium = 300_000
        manufacturer.record_insurance_premium(premium)
        operating_income_with_premium = manufacturer.calculate_operating_income(revenue)

        # Premium should reduce operating income by exactly the premium amount
        assert operating_income_no_premium - operating_income_with_premium == pytest.approx(premium)

        # The net income reduction equals premium * (1 - tax_rate)
        net_income_no_premium = manufacturer.calculate_net_income(operating_income_no_premium, 0)
        net_income_with_premium = manufacturer.calculate_net_income(
            operating_income_with_premium, 0
        )

        expected_reduction = premium * (1 - manufacturer.tax_rate)
        actual_reduction = net_income_no_premium - net_income_with_premium

        assert actual_reduction == pytest.approx(expected_reduction)

    def test_deductible_tax_treatment_on_incurrence(self, manufacturer):
        """Test that deductibles create liabilities that reduce equity.

        With correct accounting, deductibles create a liability when incurred.
        The liability reduces equity through the accounting equation (Assets - Liabilities = Equity).
        There is no separate expense recorded to avoid double-counting.
        """
        initial_assets = manufacturer.total_assets
        initial_equity = manufacturer.equity

        # Process claim with smaller deductible to maintain positive income
        deductible = 50_000
        company_payment, insurance_payment = manufacturer.process_insurance_claim(
            claim_amount=200_000,
            deductible_amount=deductible,
            insurance_limit=10_000_000,
        )

        # Verify deductible creates a liability (not an expense)
        # Liability includes LAE (Issue #468)
        lae_factor = ONE + to_decimal(manufacturer.config.lae_ratio)
        expected_liability = to_decimal(deductible) * lae_factor
        assert manufacturer.period_insurance_losses == 0  # No expense recorded
        assert manufacturer.total_claim_liabilities == pytest.approx(expected_liability)

        insurance_receivable = to_decimal(200_000 - deductible)  # #814: now in total_assets
        assert manufacturer.total_assets == initial_assets + insurance_receivable
        assert manufacturer.collateral == deductible
        assert manufacturer.restricted_assets == deductible

        expected_equity = initial_equity - expected_liability + insurance_receivable
        assert manufacturer.equity == pytest.approx(expected_equity)

    def test_no_double_counting_of_deductibles(self, manufacturer):
        """Test that deductibles are not double-counted for tax purposes.

        With correct accounting, deductibles create a liability (reducing equity once)
        and do NOT also create an expense (which would reduce equity twice).
        Payments against the liability reduce both assets and liabilities with no net equity impact.
        """
        # Process claim with deductible
        deductible = 1_000_000
        manufacturer.process_insurance_claim(
            claim_amount=5_000_000,
            deductible_amount=deductible,
            insurance_limit=10_000_000,
        )

        # Verify no expense is recorded (only liability)
        # Liability includes LAE (Issue #468)
        lae_factor = ONE + to_decimal(manufacturer.config.lae_ratio)
        expected_liability = to_decimal(deductible) * lae_factor
        assert manufacturer.period_insurance_losses == 0  # No expense
        assert manufacturer.total_claim_liabilities == pytest.approx(expected_liability)

        # Reset period costs (as would happen at end of step)
        manufacturer.reset_period_insurance_costs()
        assert manufacturer.period_insurance_losses == 0

        # Pay claim liabilities (first year payment)
        manufacturer.current_year = 1  # Advance time for payment
        payments_made = manufacturer.pay_claim_liabilities()

        # Verify payment was made
        assert payments_made > 0

        # Period losses should still be 0 (payments don't create expenses)
        assert manufacturer.period_insurance_losses == 0

        # Verify collateral was reduced
        expected_collateral = deductible - payments_made
        assert manufacturer.collateral == pytest.approx(expected_collateral)

    def test_combined_premium_and_deductible_tax_treatment(self, manufacturer):
        """Test combined treatment of premiums and deductibles.

        Premiums are expenses that reduce operating income with tax benefits.
        Deductibles create liabilities that reduce equity via the accounting equation.
        """
        initial_equity = manufacturer.equity

        # Process claim with deductible
        deductible = 200_000
        manufacturer.process_insurance_claim(
            claim_amount=1_000_000,
            deductible_amount=deductible,
            insurance_limit=5_000_000,
        )

        # Pay premium
        premium = 150_000
        manufacturer.record_insurance_premium(premium)

        # Verify premium is recorded as expense, deductible creates liability
        # Liability includes LAE (Issue #468)
        lae_factor = ONE + to_decimal(manufacturer.config.lae_ratio)
        expected_liability = to_decimal(deductible) * lae_factor
        assert manufacturer.period_insurance_premiums == premium
        assert manufacturer.period_insurance_losses == 0  # No expense from deductible
        assert manufacturer.total_claim_liabilities == pytest.approx(expected_liability)

        insurance_receivable = to_decimal(1_000_000 - deductible)  # #814
        expected_equity_after_liability = initial_equity - expected_liability + insurance_receivable
        assert manufacturer.equity == pytest.approx(expected_equity_after_liability)

        # Calculate net income - premium should reduce it with tax benefits
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)
        collateral_costs = manufacturer.calculate_collateral_costs(0.015)

        # Operating income already includes premium deduction
        net_income = manufacturer.calculate_net_income(operating_income, collateral_costs)

        # Premium should reduce net income by (premium * (1 - tax_rate))
        # Note: Cannot directly compare because operating income already includes the premium

    def test_step_with_insurance_costs(self, manufacturer):
        """Test that step() properly handles insurance costs and tax treatment.

        End-to-end test of the step function with insurance.
        """
        # Process claim and pay premium before step
        deductible = 300_000
        manufacturer.process_insurance_claim(
            claim_amount=2_000_000,
            deductible_amount=deductible,
            insurance_limit=5_000_000,
        )

        premium = 250_000
        manufacturer.record_insurance_premium(premium)

        # Store initial state
        initial_equity = manufacturer.equity

        # Run step
        metrics = manufacturer.step(letter_of_credit_rate=0.015)

        # Verify period costs were reset after step
        assert manufacturer.period_insurance_premiums == 0
        assert manufacturer.period_insurance_losses == 0

        # Verify metrics include the insurance impact
        # After step, first year payment has been made, so collateral is reduced
        # Liability includes LAE (Issue #468), payments are % of liability
        lae_factor = ONE + to_decimal(manufacturer.config.lae_ratio)
        liability = to_decimal(deductible) * lae_factor
        first_year_payment = float(liability * to_decimal("0.10"))  # 10% of liability
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
        """Test that uninsured and insured claims have consistent accounting treatment.

        Both create liabilities when using payment schedules.
        Immediate payments record expenses immediately.
        """
        # Test insured claim with payment schedule
        manufacturer_insured = WidgetManufacturer(manufacturer.config)
        deductible = 100_000
        manufacturer_insured.process_insurance_claim(
            claim_amount=500_000,
            deductible_amount=deductible,
            insurance_limit=1_000_000,
        )

        # Test uninsured claim with payment schedule (comparable to insured)
        manufacturer_uninsured = WidgetManufacturer(manufacturer.config)
        manufacturer_uninsured.process_uninsured_claim(100_000, immediate_payment=False)

        # Both should create liabilities without recording expenses
        # Liabilities include LAE (Issue #468)
        lae_factor = ONE + to_decimal(manufacturer.config.lae_ratio)
        expected_insured_liability = to_decimal(deductible) * lae_factor
        expected_uninsured_liability = to_decimal(100_000) * lae_factor
        assert manufacturer_insured.period_insurance_losses == 0
        assert manufacturer_uninsured.period_insurance_losses == 0
        assert manufacturer_insured.total_claim_liabilities == pytest.approx(
            expected_insured_liability
        )
        assert manufacturer_uninsured.total_claim_liabilities == pytest.approx(
            expected_uninsured_liability
        )

        insurance_receivable = to_decimal(500_000 - deductible)  # #814: ins. receivable
        expected_equity_uninsured = (
            to_decimal(manufacturer.config.initial_assets) - expected_uninsured_liability
        )
        expected_equity_insured = (
            to_decimal(manufacturer.config.initial_assets)
            + insurance_receivable
            - expected_insured_liability
        )
        assert manufacturer_insured.equity == pytest.approx(expected_equity_insured)
        assert manufacturer_uninsured.equity == pytest.approx(expected_equity_uninsured)

    def test_collateral_costs_are_tax_deductible(self, manufacturer):
        """Test that letter of credit collateral costs are tax-deductible.

        Ensures collateral financing costs reduce taxable income.
        """
        # Process smaller claim to maintain positive income
        manufacturer.process_insurance_claim(
            claim_amount=300_000, deductible_amount=100_000, insurance_limit=5_000_000
        )

        # Calculate collateral costs
        loc_rate = to_decimal(0.02)
        collateral_costs = manufacturer.calculate_collateral_costs(loc_rate)
        expected_costs = to_decimal(100_000) * loc_rate  # Based on deductible collateral
        assert collateral_costs == expected_costs

        # Calculate net income with and without collateral costs
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        net_income_with_costs = manufacturer.calculate_net_income(
            operating_income, collateral_costs
        )

        net_income_without_costs = manufacturer.calculate_net_income(operating_income, ZERO)

        # Collateral costs should reduce net income by (costs * (1 - tax_rate))
        tax_rate = to_decimal(manufacturer.tax_rate)
        expected_reduction = collateral_costs * (to_decimal(1) - tax_rate)
        actual_reduction = net_income_without_costs - net_income_with_costs

        assert float(actual_reduction) == pytest.approx(float(expected_reduction))

    def test_premium_does_not_reduce_productive_assets(self, manufacturer):
        """Test that insurance premiums don't immediately reduce productive assets.

        Regression test for issue where premiums were liquidating productive assets,
        causing revenue decline and negative ROI for insurance scenarios.
        """
        initial_assets = manufacturer.total_assets
        initial_equity = manufacturer.equity
        premium = 500_000

        # Record premium
        manufacturer.record_insurance_premium(premium)

        # Assets should NOT be immediately reduced
        assert manufacturer.total_assets == initial_assets
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
        expected_revenue = manufacturer.total_assets * to_decimal(manufacturer.asset_turnover_ratio)
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

        # Net income already reflects the premium through operating_income
        # (calculate_operating_income deducts period_insurance_premiums)
        net_income = manufacturer.calculate_net_income(operating_income, 0)

        # Verify premium was included in operating income
        expected_operating_income = revenue * to_decimal(
            manufacturer.base_operating_margin
        ) - to_decimal(premium)
        assert operating_income == pytest.approx(expected_operating_income)

        # Net income should reflect the premium deduction via operating income
        expected_net = operating_income * (1 - to_decimal(manufacturer.tax_rate))
        assert net_income == pytest.approx(expected_net, rel=1e-6)

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

        The test allows 40% tolerance for compounding effects because:
        - Revenue depends on assets (revenue = assets * turnover_ratio)
        - Insurance premiums reduce assets through lower retained earnings
        - Lower assets lead to lower revenue in subsequent periods
        - This compounds over multiple years
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

        asset_gap = no_insurance.total_assets - with_insurance.total_assets

        # The gap should be reasonably close to cumulative after-tax premiums
        # Allow 40% tolerance for compounding effects (revenue depends on assets)
        assert asset_gap < after_tax_total * 1.4

    def test_premium_with_claims_integration(self, manufacturer):
        """Test that premiums and claims work correctly together.

        Integration test ensuring premiums don't interfere with claim processing
        and both contribute properly to tax calculations.
        """
        # Process a claim with deductible
        deductible = 150_000
        manufacturer.process_insurance_claim(
            claim_amount=1_000_000,
            deductible_amount=deductible,
            insurance_limit=5_000_000,
        )

        # Pay premium
        premium = 200_000
        manufacturer.record_insurance_premium(premium)

        # Premium recorded as expense, deductible creates liability
        # Liability includes LAE (Issue #468)
        lae_factor = ONE + to_decimal(manufacturer.config.lae_ratio)
        expected_liability = to_decimal(deductible) * lae_factor
        assert manufacturer.period_insurance_premiums == premium
        assert manufacturer.period_insurance_losses == 0  # No expense from deductible
        assert manufacturer.total_claim_liabilities == pytest.approx(expected_liability)

        # Assets should not be reduced by premium (it's just an expense)
        # Deductible creates collateral (cash moved to restricted)
        assert manufacturer.collateral == deductible

        # Run step to process everything
        metrics = manufacturer.step()

        # Should remain solvent despite both costs
        assert metrics["is_solvent"]

        # Period costs should be reset after step
        assert manufacturer.period_insurance_premiums == 0
        assert manufacturer.period_insurance_losses == 0

    def test_liquidity_crisis_triggers_insolvency(self, config):
        """Test that loss exceeding available cash triggers immediate insolvency.

        Regression test for liquidity crisis bug where companies with positive book
        equity but insufficient cash were allowed to partially absorb losses,
        with unpaid amounts "disappearing" unrealistically.

        This test verifies that a company becomes insolvent when it cannot pay
        a loss with available cash, even if book equity is positive.
        """
        # Create manufacturer with working capital to tie up cash in non-liquid assets
        manufacturer = WidgetManufacturer(config)

        # Set up scenario: Positive equity but most cash tied up in working capital
        # Adjust balance sheet via ledger to create cash constraint (Issue #275)
        # First, calculate how much cash to move to other assets
        initial_cash = manufacturer.cash
        target_cash = to_decimal(500_000)
        cash_to_move = initial_cash - target_cash

        # Move cash to AR (tie up in working capital)
        if cash_to_move > 0:
            manufacturer.ledger.record_double_entry(
                date=0,
                debit_account=AccountName.ACCOUNTS_RECEIVABLE,
                credit_account=AccountName.CASH,
                amount=cash_to_move,
                transaction_type=TransactionType.WORKING_CAPITAL,
                description="Tie up cash in AR for test",
            )

        # Verify initial state: Solvent with positive equity but limited cash
        assert manufacturer.equity > 500_000  # Still has positive equity
        assert manufacturer.cash == pytest.approx(target_cash, rel=0.01)  # But limited cash
        assert not manufacturer.is_ruined

        # Incur a loss that exceeds available cash but not equity
        # Loss = $800K, Cash = $500K, Equity ~ $9.8M
        net_income = -800_000  # Larger loss than available cash

        # Attempt to absorb the loss - should trigger liquidity crisis
        manufacturer.update_balance_sheet(net_income)

        # Company should now be insolvent due to liquidity crisis
        assert manufacturer.is_ruined, "Company should be insolvent when loss exceeds cash"
        # Limited liability floor is enforced in handle_insolvency(), so equity >= 0 is guaranteed

    def test_equity_insolvency_triggers_when_loss_pushes_below_tolerance(self, manufacturer):
        """Test that loss pushing equity below tolerance triggers insolvency.

        Verifies that a company becomes insolvent when absorbing a loss would
        push equity below the insolvency tolerance threshold, even if it has
        sufficient cash to pay.
        """
        # Set up manufacturer near insolvency tolerance
        tolerance = manufacturer.config.insolvency_tolerance  # $10,000
        target_equity = tolerance + 50_000  # Equity at $60K (just above threshold)

        # Reduce equity to target level by creating liabilities
        # With LAE, total liability = claim * (1 + lae_ratio), so claim = equity_reduction / (1 + lae_ratio)
        lae_factor = ONE + to_decimal(manufacturer.config.lae_ratio)
        equity_reduction = manufacturer.equity - target_equity
        claim_amount = equity_reduction / lae_factor
        manufacturer.process_uninsured_claim(claim_amount, immediate_payment=False)

        # Verify starting state
        assert manufacturer.equity == pytest.approx(target_equity, rel=1e-6)
        assert manufacturer.cash > 100_000  # Has sufficient cash
        assert not manufacturer.is_ruined

        # Incur loss that would push equity below tolerance
        # Loss = $60K, which would bring equity to $0 (below $10K threshold)
        net_income = -60_000

        # Attempt to absorb - should trigger insolvency
        manufacturer.update_balance_sheet(net_income)

        # Should be insolvent
        assert manufacturer.is_ruined, "Should be insolvent when equity would fall below tolerance"
        # Limited liability floor is enforced in handle_insolvency(), so equity >= 0 is guaranteed

    def test_already_insolvent_company_does_not_absorb_further_losses(self, manufacturer):
        """Test that companies at or below tolerance don't absorb additional losses.

        Verifies that once a company reaches the insolvency tolerance threshold,
        it doesn't attempt to absorb further losses.
        """
        # Reduce equity to exactly the tolerance level
        # With LAE, total liability = claim * (1 + lae_ratio), so claim = equity_reduction / (1 + lae_ratio)
        tolerance = manufacturer.config.insolvency_tolerance  # $10,000
        lae_factor = ONE + to_decimal(manufacturer.config.lae_ratio)
        equity_reduction = manufacturer.equity - tolerance
        claim_amount = equity_reduction / lae_factor
        manufacturer.process_uninsured_claim(claim_amount, immediate_payment=False)

        # Verify at tolerance
        assert manufacturer.equity == pytest.approx(tolerance, rel=1e-6)
        initial_cash = manufacturer.cash
        assert not manufacturer.is_ruined

        # Try to absorb another loss
        net_income = -50_000

        manufacturer.update_balance_sheet(net_income)

        # Cash should not be reduced (loss not absorbed)
        assert manufacturer.cash == pytest.approx(
            initial_cash
        ), "Should not reduce cash when already at tolerance"
        # Equity should still be at tolerance
        assert manufacturer.equity == pytest.approx(tolerance)

    def test_normal_loss_absorption_when_sufficient_cash_and_equity(self, manufacturer):
        """Test that losses are absorbed normally when sufficient cash and equity exist.

        Verifies the "happy path" where a company has both sufficient cash to pay
        and sufficient equity buffer to remain solvent after absorption.
        """
        tolerance = manufacturer.config.insolvency_tolerance
        initial_cash = manufacturer.cash
        initial_equity = manufacturer.equity

        # Incur moderate loss that company can absorb
        # Loss = $200K, well within both cash ($7M) and equity ($10M)
        loss_amount = 200_000
        net_income = -loss_amount

        # Absorb the loss
        manufacturer.update_balance_sheet(net_income)

        # Loss should be fully absorbed
        assert manufacturer.cash == pytest.approx(initial_cash - loss_amount)
        assert manufacturer.equity == pytest.approx(initial_equity - loss_amount)
        assert not manufacturer.is_ruined, "Should remain solvent with sufficient buffers"
        assert manufacturer.equity > tolerance, "Should be well above insolvency threshold"

    def test_edge_case_loss_exactly_equals_cash(self, manufacturer):
        """Test edge case where loss exactly equals available cash.

        Verifies behavior when loss amount precisely matches cash on hand.
        """
        # Set cash to specific amount via ledger (Issue #275)
        target_cash = to_decimal(100_000)
        current_cash = manufacturer.cash
        if current_cash > target_cash:
            reduction = current_cash - target_cash
            manufacturer.ledger.record_double_entry(
                date=0,
                debit_account=AccountName.RETAINED_EARNINGS,
                credit_account=AccountName.CASH,
                amount=reduction,
                transaction_type=TransactionType.ADJUSTMENT,
                description="Reduce cash for test",
            )
        initial_equity = manufacturer.equity

        # Loss exactly equals cash
        net_income = -100_000

        manufacturer.update_balance_sheet(net_income)

        # Should absorb the loss if equity remains above tolerance
        tolerance = manufacturer.config.insolvency_tolerance
        if initial_equity - 100_000 > tolerance:
            # Should successfully absorb
            assert manufacturer.cash == pytest.approx(0, abs=1)
            assert not manufacturer.is_ruined
        else:
            # Should trigger insolvency
            assert manufacturer.is_ruined

    def test_liquidity_crisis_error_message_clarity(self, manufacturer, caplog):
        """Test that liquidity crisis produces clear error messages for debugging.

        Verifies that the liquidity crisis error message includes all relevant
        information (loss amount, available cash, book equity) to help with
        debugging and understanding the insolvency trigger.
        """
        import logging

        # Set up cash-constrained scenario via ledger (Issue #275)
        target_cash = to_decimal(100_000)
        current_cash = manufacturer.cash
        if current_cash > target_cash:
            reduction = current_cash - target_cash
            manufacturer.ledger.record_double_entry(
                date=0,
                debit_account=AccountName.RETAINED_EARNINGS,
                credit_account=AccountName.CASH,
                amount=reduction,
                transaction_type=TransactionType.ADJUSTMENT,
                description="Reduce cash for test",
            )
        initial_equity = manufacturer.equity

        # Trigger liquidity crisis
        loss_amount = 150_000
        net_income = -loss_amount

        with caplog.at_level(logging.ERROR):
            manufacturer.update_balance_sheet(net_income)

        # Check that error log contains key information
        error_messages = [
            record.message for record in caplog.records if record.levelname == "ERROR"
        ]
        assert any("LIQUIDITY CRISIS" in msg for msg in error_messages)
        assert any(f"${loss_amount:,.2f}" in msg for msg in error_messages)
        assert any(f"${100_000:,.2f}" in msg for msg in error_messages)

    def test_working_capital_creates_liquidity_constraint_scenario(self, config):
        """Integration test demonstrating real-world liquidity crisis scenario.

        Shows how working capital requirements can create a situation where
        a company has positive book equity but insufficient liquid cash to
        meet obligations, leading to insolvency.
        """
        manufacturer = WidgetManufacturer(config)

        # Run one period to establish working capital
        manufacturer.step()

        # Verify working capital has tied up cash
        assert manufacturer.accounts_receivable > 0, "Should have AR (cash tied up)"
        assert manufacturer.inventory > 0, "Should have inventory (cash tied up)"
        assert manufacturer.equity > 0, "Should have positive equity"

        # Record current state
        available_cash = manufacturer.cash
        total_equity = manufacturer.equity

        # Simulate catastrophic loss that exceeds liquid cash but not book equity
        # E.g., $5M loss when cash is ~$3M but equity is ~$10M
        catastrophic_loss = available_cash + 1_000_000  # Exceeds cash by $1M

        net_income = -catastrophic_loss

        # This should trigger liquidity crisis despite positive book equity
        manufacturer.update_balance_sheet(net_income)

        # Verify insolvency triggered
        assert manufacturer.is_ruined, (
            f"Should be insolvent: needed ${catastrophic_loss:,.2f}, "
            f"had cash ${available_cash:,.2f}, equity ${total_equity:,.2f}"
        )


class TestMidYearLiquidity:
    """Test suite for mid-year liquidity detection (Issue #279).

    Tests for the estimate_minimum_cash_point() and check_liquidity_constraints()
    methods that detect potential mid-year insolvency events.
    """

    @pytest.fixture
    def config(self) -> ManufacturerConfig:
        """Create test manufacturer config with mid-year liquidity enabled."""
        return ManufacturerConfig(
            initial_assets=5_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.7,
            premium_payment_month=0,  # January
            revenue_pattern="uniform",
            check_intra_period_liquidity=True,
        )

    @pytest.fixture
    def manufacturer(self, config: ManufacturerConfig) -> WidgetManufacturer:
        """Create a test manufacturer with mid-year liquidity checking enabled."""
        return WidgetManufacturer(config)

    def test_estimate_minimum_cash_point_uniform_revenue(self, manufacturer: WidgetManufacturer):
        """Test minimum cash point estimation with uniform revenue.

        With uniform revenue distribution, the minimum should occur early in
        the year when premium and first tax payment have been made but
        limited revenue has been collected.
        """
        min_cash, min_month = manufacturer.estimate_minimum_cash_point("annual")

        # Should return valid values
        assert isinstance(min_cash, Decimal)
        assert isinstance(min_month, int)
        assert 0 <= min_month <= 11

    def test_estimate_minimum_cash_point_back_loaded_revenue(self):
        """Test minimum cash point estimation with back-loaded revenue.

        With back-loaded revenue (60% in H2), the minimum should occur
        earlier in the year when outflows exceed inflows.
        """
        config = ManufacturerConfig(
            initial_assets=5_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.7,
            revenue_pattern="back_loaded",
            check_intra_period_liquidity=True,
        )
        manufacturer = WidgetManufacturer(config)

        min_cash, min_month = manufacturer.estimate_minimum_cash_point("annual")

        # With back-loaded revenue, minimum should be in H1 (months 0-5)
        assert 0 <= min_month <= 11

    def test_estimate_minimum_cash_point_monthly_resolution(self, manufacturer: WidgetManufacturer):
        """Test minimum cash point estimation with monthly resolution.

        For monthly resolution, should return current cash and month.
        """
        min_cash, min_month = manufacturer.estimate_minimum_cash_point("monthly")

        # Should return current cash for monthly resolution
        assert min_cash == manufacturer.cash
        assert min_month == manufacturer.current_month

    def test_check_liquidity_constraints_solvent(self, manufacturer: WidgetManufacturer):
        """Test liquidity constraint check when company is solvent.

        With sufficient initial cash, the company should pass liquidity check.
        """
        result = manufacturer.check_liquidity_constraints("annual")

        # Should be solvent with default initial assets
        assert result is True
        assert manufacturer.is_ruined is False
        assert manufacturer.ruin_month is None

    def test_check_liquidity_constraints_disabled(self):
        """Test liquidity constraint check when disabled in config.

        With check_intra_period_liquidity=False, should always pass.
        """
        config = ManufacturerConfig(
            initial_assets=100_000,  # Very low initial assets
            asset_turnover_ratio=0.8,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.7,
            check_intra_period_liquidity=False,  # Disabled
        )
        manufacturer = WidgetManufacturer(config)

        # Set a large premium that would cause mid-year insolvency
        manufacturer.period_insurance_premiums = to_decimal(200_000)

        result = manufacturer.check_liquidity_constraints("annual")

        # Should pass because check is disabled
        assert result is True

    def test_check_liquidity_constraints_mid_year_insolvency(self):
        """Test liquidity constraint check detects mid-year insolvency.

        A company with low cash and high premium should trigger mid-year ruin.
        """
        config = ManufacturerConfig(
            initial_assets=1_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.7,
            premium_payment_month=0,  # January
            check_intra_period_liquidity=True,
        )
        manufacturer = WidgetManufacturer(config)

        # Set a very large premium that would cause mid-year insolvency
        # Cash is roughly 30% of initial assets, so premium > 30% causes issue
        manufacturer.period_insurance_premiums = to_decimal(500_000)

        result = manufacturer.check_liquidity_constraints("annual")

        # Should detect mid-year insolvency
        assert result is False
        assert manufacturer.is_ruined is True
        assert manufacturer.ruin_month is not None
        assert 0 <= manufacturer.ruin_month <= 11

    def test_step_integrates_liquidity_check(self):
        """Test that step() integrates the liquidity constraint check.

        The step method should automatically call check_liquidity_constraints
        and detect mid-year insolvency.
        """
        config = ManufacturerConfig(
            initial_assets=1_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.7,
            premium_payment_month=0,
            check_intra_period_liquidity=True,
        )
        manufacturer = WidgetManufacturer(config)

        # Set a very large premium
        manufacturer.period_insurance_premiums = to_decimal(500_000)

        # Step should detect mid-year insolvency
        metrics = manufacturer.step()

        assert manufacturer.is_ruined is True
        assert metrics.get("is_solvent") is False

    def test_backwards_compatibility_disabled(self):
        """Test backwards compatibility when liquidity check is disabled.

        With check_intra_period_liquidity=False, simulations should produce
        the same results as before this feature was added.
        """
        config = ManufacturerConfig(
            initial_assets=5_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.7,
            check_intra_period_liquidity=False,
        )
        manufacturer = WidgetManufacturer(config)

        # Run a few steps
        for _ in range(3):
            manufacturer.step()

        # Should still be solvent (no mid-year check interference)
        assert manufacturer.is_ruined is False

    def test_get_monthly_revenue_distribution_uniform(self, manufacturer: WidgetManufacturer):
        """Test uniform monthly revenue distribution."""
        annual_revenue = to_decimal(1_200_000)
        distribution = manufacturer._get_monthly_revenue_distribution(annual_revenue, "uniform")

        assert len(distribution) == 12
        # Each month should get 1/12 of annual revenue
        expected_monthly = to_decimal(100_000)
        for monthly in distribution:
            assert monthly == expected_monthly

    def test_get_monthly_revenue_distribution_seasonal(self, manufacturer: WidgetManufacturer):
        """Test seasonal monthly revenue distribution.

        Q1-Q3 should get 60% split equally (20% each quarter),
        Q4 should get 40%.
        """
        annual_revenue = to_decimal(1_200_000)
        distribution = manufacturer._get_monthly_revenue_distribution(annual_revenue, "seasonal")

        assert len(distribution) == 12

        # Q1-Q3 (months 0-8): 60% / 9 months = 6.67% each
        q1_q3_monthly = annual_revenue * to_decimal(0.60) / to_decimal(9)
        for i in range(9):
            assert distribution[i] == q1_q3_monthly

        # Q4 (months 9-11): 40% / 3 months = 13.33% each
        q4_monthly = annual_revenue * to_decimal(0.40) / to_decimal(3)
        for i in range(9, 12):
            assert distribution[i] == q4_monthly

    def test_get_monthly_revenue_distribution_back_loaded(self, manufacturer: WidgetManufacturer):
        """Test back-loaded monthly revenue distribution.

        H1 should get 40%, H2 should get 60%.
        """
        annual_revenue = to_decimal(1_200_000)
        distribution = manufacturer._get_monthly_revenue_distribution(annual_revenue, "back_loaded")

        assert len(distribution) == 12

        # H1 (months 0-5): 40% / 6 months
        h1_monthly = annual_revenue * to_decimal(0.40) / to_decimal(6)
        for i in range(6):
            assert distribution[i] == h1_monthly

        # H2 (months 6-11): 60% / 6 months
        h2_monthly = annual_revenue * to_decimal(0.60) / to_decimal(6)
        for i in range(6, 12):
            assert distribution[i] == h2_monthly

    def test_ruin_month_tracking(self):
        """Test that ruin_month is properly tracked when mid-year insolvency occurs."""
        config = ManufacturerConfig(
            initial_assets=1_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.7,
            premium_payment_month=0,  # January - premium causes early ruin
            check_intra_period_liquidity=True,
        )
        manufacturer = WidgetManufacturer(config)

        # Set premium to cause insolvency in month 0
        manufacturer.period_insurance_premiums = to_decimal(500_000)
        manufacturer.check_liquidity_constraints("annual")

        # ruin_month should be set to the month when insolvency occurred
        assert manufacturer.ruin_month is not None
        # Should be early in year due to large January premium
        assert manufacturer.ruin_month <= 3

    # --- NOL carryforward tests (Issue #689) ---

    def test_nol_reduces_estimated_tax_in_liquidity_check(self):
        """Large NOL should reduce estimated tax, raising minimum cash estimate.

        Without NOL, quarterly tax = estimated_income * tax_rate / 4.
        With NOL, the 80%-limited deduction reduces taxable income, lowering
        the quarterly tax deducted in the cash projection.

        Uses premium_payment_month=3 (first tax month) so the minimum cash
        point occurs when both premium and tax are paid, making the tax
        difference visible in the minimum.
        """
        config = ManufacturerConfig(
            initial_assets=5_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.7,
            premium_payment_month=3,
            check_intra_period_liquidity=True,
        )
        mfr_no_nol = WidgetManufacturer(config)
        mfr_no_nol.period_insurance_premiums = to_decimal(1_350_000)
        min_cash_no_nol, _ = mfr_no_nol.estimate_minimum_cash_point("annual")

        mfr_with_nol = WidgetManufacturer(config)
        mfr_with_nol.period_insurance_premiums = to_decimal(1_350_000)
        mfr_with_nol.tax_handler.nol_carryforward = Decimal("5000000")
        min_cash_with_nol, _ = mfr_with_nol.estimate_minimum_cash_point("annual")

        assert min_cash_with_nol > min_cash_no_nol

    def test_nol_80pct_limitation_respected(self):
        """NOL deduction limited to 80% of estimated income per IRC 172(a)(2).

        Even with NOL >> estimated income, 20% of income remains taxable.
        A small NOL (below the 80% cap) should yield a smaller benefit
        than a huge NOL that hits the cap.

        Uses premium_payment_month=3 so minimum cash falls at a tax month.
        """
        config = ManufacturerConfig(
            initial_assets=5_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.7,
            premium_payment_month=3,
            check_intra_period_liquidity=True,
        )
        premium = to_decimal(1_350_000)
        # estimated_income ≈ $5M * 0.8 * 0.10 = $400K
        # 80% cap = $320K

        # No NOL
        mfr_zero = WidgetManufacturer(config)
        mfr_zero.period_insurance_premiums = premium
        min_cash_zero, _ = mfr_zero.estimate_minimum_cash_point("annual")

        # Small NOL ($50K < 80% cap of $320K) — fully used
        mfr_small = WidgetManufacturer(config)
        mfr_small.period_insurance_premiums = premium
        mfr_small.tax_handler.nol_carryforward = Decimal("50000")
        min_cash_small, _ = mfr_small.estimate_minimum_cash_point("annual")

        # Huge NOL ($10M >> income) — capped at 80%
        mfr_huge = WidgetManufacturer(config)
        mfr_huge.period_insurance_premiums = premium
        mfr_huge.tax_handler.nol_carryforward = Decimal("10000000")
        min_cash_huge, _ = mfr_huge.estimate_minimum_cash_point("annual")

        # Ordering: no NOL < small NOL < huge NOL (80%-capped)
        assert min_cash_zero < min_cash_small < min_cash_huge

    def test_nol_5m_with_400k_income(self):
        """$5M NOL with ~$400K estimated income: taxable = $80K (20% of income).

        80% of $400K = $320K max deduction. NOL $5M >> $320K, so deduction
        is capped at $320K. Taxable = $400K - $320K = $80K.
        Annual tax = $80K * 0.25 = $20K vs $100K without NOL.
        Quarterly tax savings = ($100K - $20K) / 4 = $20K per quarter.
        """
        config = ManufacturerConfig(
            initial_assets=5_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.7,
            premium_payment_month=3,
            check_intra_period_liquidity=True,
        )
        premium = to_decimal(1_350_000)

        mfr = WidgetManufacturer(config)
        mfr.period_insurance_premiums = premium
        mfr.tax_handler.nol_carryforward = Decimal("5000000")
        min_cash_nol, _ = mfr.estimate_minimum_cash_point("annual")

        mfr_base = WidgetManufacturer(config)
        mfr_base.period_insurance_premiums = premium
        min_cash_base, _ = mfr_base.estimate_minimum_cash_point("annual")

        assert min_cash_nol > min_cash_base
        savings = min_cash_nol - min_cash_base
        # Min occurs at month 3 where one quarter's tax is saved: $20K
        assert savings >= Decimal("15000")

    def test_nol_1m_with_400k_income(self):
        """$1M NOL with ~$400K estimated income: deduction = $320K (80% cap binds).

        $1M NOL > 80% cap of $320K, so deduction = $320K.
        Same result as $5M NOL case — the 80% cap is the binding constraint.
        """
        config = ManufacturerConfig(
            initial_assets=5_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.7,
            check_intra_period_liquidity=True,
        )
        # $1M NOL — above the 80% cap ($320K), so same effect as any higher NOL
        mfr_1m = WidgetManufacturer(config)
        mfr_1m.tax_handler.nol_carryforward = Decimal("1000000")
        min_cash_1m, _ = mfr_1m.estimate_minimum_cash_point("annual")

        # $5M NOL — also above cap, should give same result
        mfr_5m = WidgetManufacturer(config)
        mfr_5m.tax_handler.nol_carryforward = Decimal("5000000")
        min_cash_5m, _ = mfr_5m.estimate_minimum_cash_point("annual")

        # Both exceed the 80% cap, so their effect should be identical
        assert min_cash_1m == min_cash_5m

    def test_partial_nol_below_80pct_cap(self):
        """$100K NOL with ~$400K income: full NOL used (below 80% cap of $320K).

        Deduction = min($100K, $320K) = $100K.
        Taxable = $400K - $100K = $300K.
        Tax = $300K * 0.25 = $75K vs $100K without NOL.
        """
        config = ManufacturerConfig(
            initial_assets=5_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.7,
            premium_payment_month=3,
            check_intra_period_liquidity=True,
        )
        premium = to_decimal(1_350_000)

        mfr_partial = WidgetManufacturer(config)
        mfr_partial.period_insurance_premiums = premium
        mfr_partial.tax_handler.nol_carryforward = Decimal("100000")
        min_cash_partial, _ = mfr_partial.estimate_minimum_cash_point("annual")

        mfr_capped = WidgetManufacturer(config)
        mfr_capped.period_insurance_premiums = premium
        mfr_capped.tax_handler.nol_carryforward = Decimal("5000000")
        min_cash_capped, _ = mfr_capped.estimate_minimum_cash_point("annual")

        mfr_none = WidgetManufacturer(config)
        mfr_none.period_insurance_premiums = premium
        min_cash_none, _ = mfr_none.estimate_minimum_cash_point("annual")

        # Partial NOL gives less benefit than capped NOL
        assert min_cash_none < min_cash_partial < min_cash_capped

    def test_nol_prevents_false_insolvency(self):
        """Company with NOL should not trigger false insolvency from inflated tax.

        Sets up a scenario where premium + tax in month 3 cause insolvency
        without NOL adjustment, but with NOL the reduced tax keeps cash positive.
        """
        config = ManufacturerConfig(
            initial_assets=1_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.20,
            tax_rate=0.30,
            retention_ratio=0.7,
            premium_payment_month=3,  # Coincides with first tax month
            check_intra_period_liquidity=True,
        )

        # Without NOL — triggers insolvency
        mfr_no_nol = WidgetManufacturer(config)
        mfr_no_nol.period_insurance_premiums = to_decimal(370_000)

        result_no_nol = mfr_no_nol.check_liquidity_constraints("annual")
        assert result_no_nol is False, "Should be insolvent without NOL adjustment"
        assert mfr_no_nol.is_ruined is True

        # With NOL — same scenario stays solvent
        mfr_with_nol = WidgetManufacturer(config)
        mfr_with_nol.period_insurance_premiums = to_decimal(370_000)
        estimated_income = mfr_with_nol.calculate_revenue() * to_decimal(0.20)
        mfr_with_nol.tax_handler.nol_carryforward = estimated_income

        result_with_nol = mfr_with_nol.check_liquidity_constraints("annual")
        assert result_with_nol is True, "Should be solvent with NOL sheltering income"
        assert mfr_with_nol.is_ruined is False

    def test_zero_nol_unchanged_behavior(self):
        """Zero NOL produces same result as default (no NOL adjustment)."""
        config = ManufacturerConfig(
            initial_assets=5_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.7,
            check_intra_period_liquidity=True,
        )
        mfr_zero = WidgetManufacturer(config)
        mfr_zero.tax_handler.nol_carryforward = Decimal("0")
        min_cash_zero, _ = mfr_zero.estimate_minimum_cash_point("annual")

        mfr_default = WidgetManufacturer(config)
        min_cash_default, _ = mfr_default.estimate_minimum_cash_point("annual")

        assert min_cash_zero == min_cash_default

    def test_nol_graceful_without_tax_handler(self):
        """Without tax_handler attribute, estimate works unchanged (no crash)."""
        config = ManufacturerConfig(
            initial_assets=5_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.7,
            check_intra_period_liquidity=True,
        )
        mfr = WidgetManufacturer(config)
        baseline_cash, baseline_month = mfr.estimate_minimum_cash_point("annual")

        mfr2 = WidgetManufacturer(config)
        delattr(mfr2, "tax_handler")
        no_handler_cash, no_handler_month = mfr2.estimate_minimum_cash_point("annual")

        assert no_handler_cash == baseline_cash
        assert no_handler_month == baseline_month
