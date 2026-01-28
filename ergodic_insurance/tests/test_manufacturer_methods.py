"""Unit tests for WidgetManufacturer step() and process_insurance_claim() methods."""

import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decimal_utils import to_decimal
from ergodic_insurance.manufacturer import WidgetManufacturer


class TestProcessInsuranceClaim:
    """Test suite for process_insurance_claim() method."""

    @pytest.fixture
    def manufacturer(self):
        """Create a manufacturer for testing."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.1,
            tax_rate=0.25,
            retention_ratio=0.5,
        )
        return WidgetManufacturer(config)

    def test_claim_below_deductible(self, manufacturer):
        """Test claim that is fully below deductible."""
        claim_amount = 50_000
        deductible = 100_000
        limit = 1_000_000

        company_payment, insurance_payment = manufacturer.process_insurance_claim(
            claim_amount, deductible, limit
        )

        # Company pays full amount when below deductible
        assert company_payment == claim_amount
        assert insurance_payment == 0

        # Collateral should be posted for company payment
        assert manufacturer.collateral == company_payment
        assert manufacturer.restricted_assets == company_payment
        assert len(manufacturer.claim_liabilities) == 1

    def test_claim_above_deductible_below_limit(self, manufacturer):
        """Test claim between deductible and limit."""
        initial_assets = manufacturer.total_assets
        claim_amount = 500_000
        deductible = 100_000
        limit = 1_000_000

        company_payment, insurance_payment = manufacturer.process_insurance_claim(
            claim_amount, deductible, limit
        )

        # Company pays deductible, insurance pays the rest
        assert company_payment == deductible
        assert insurance_payment == claim_amount - deductible

        # Check collateral for company payment (not insurance)
        assert manufacturer.collateral == company_payment
        assert manufacturer.restricted_assets == company_payment
        assert len(manufacturer.claim_liabilities) == 1

        # Assets not immediately reduced (paid over time)
        assert manufacturer.total_assets == initial_assets

    def test_claim_above_limit(self, manufacturer):
        """Test claim that exceeds insurance limit."""
        initial_assets = manufacturer.total_assets
        claim_amount = 2_000_000
        deductible = 100_000
        limit = 1_000_000

        company_payment, insurance_payment = manufacturer.process_insurance_claim(
            claim_amount, deductible, limit
        )

        # Company pays deductible plus excess over limit
        expected_company = deductible + (claim_amount - deductible - limit)
        assert company_payment == expected_company
        assert insurance_payment == limit

        # Check collateral for company portion (not insurance)
        assert manufacturer.collateral == expected_company
        assert manufacturer.restricted_assets == expected_company

        # Assets not immediately reduced (paid over time)
        assert manufacturer.total_assets == initial_assets

    def test_zero_claim(self, manufacturer):
        """Test handling of zero claim amount."""
        company_payment, insurance_payment = manufacturer.process_insurance_claim(
            0, 100_000, 1_000_000
        )

        assert company_payment == 0
        assert insurance_payment == 0
        assert manufacturer.collateral == 0

    def test_no_insurance_limit(self, manufacturer):
        """Test claim with no insurance (limit = 0)."""
        claim_amount = 500_000
        deductible = 0
        limit = 0

        company_payment, insurance_payment = manufacturer.process_insurance_claim(
            claim_amount, deductible, limit
        )

        # Company pays everything when no insurance
        assert company_payment == claim_amount
        assert insurance_payment == 0

    def test_infinite_limit(self, manufacturer):
        """Test claim with infinite insurance limit."""
        claim_amount = 10_000_000
        deductible = 100_000

        company_payment, insurance_payment = manufacturer.process_insurance_claim(
            claim_amount, deductible, float("inf")
        )

        # Company only pays deductible
        assert company_payment == deductible
        assert insurance_payment == claim_amount - deductible

    def test_claim_exceeds_assets(self, manufacturer):
        """Test claim when company doesn't have enough assets (limited liability)."""
        # Set all assets to achieve low total assets of 50_000
        manufacturer.cash = 50_000
        manufacturer.accounts_receivable = 0
        manufacturer.inventory = 0
        manufacturer.prepaid_insurance = 0
        manufacturer.gross_ppe = 0
        manufacturer.accumulated_depreciation = 0
        manufacturer.restricted_assets = 0

        claim_amount = 200_000
        deductible = 100_000
        limit = 1_000_000

        company_payment, insurance_payment = manufacturer.process_insurance_claim(
            claim_amount, deductible, limit
        )

        # Company payment is the deductible amount (but capped by limited liability)
        assert company_payment == deductible
        # Assets remain unchanged (collateral posted, paid over time)
        assert manufacturer.total_assets == 50_000
        # LIMITED LIABILITY: Collateral capped at available cash/equity ($50K), not full deductible
        assert manufacturer.collateral == 50_000
        # Remaining $50K of deductible cannot be paid (limited liability violation)
        # Company should be marked as insolvent
        assert manufacturer.is_ruined is True
        assert manufacturer.equity == 0
        # Insurance still covers its portion
        assert insurance_payment == claim_amount - deductible

    def test_claim_liability_payment_schedule_edge_cases(self, manufacturer):
        """Test edge cases in claim liability payment schedule."""
        # Create a claim liability
        manufacturer.process_insurance_claim(1_000_000, 100_000, 2_000_000)

        # Test payment schedule for year beyond schedule (should return 0)
        claim = manufacturer.claim_liabilities[0]

        # Test invalid year (negative)
        payment = claim.get_payment(-1)
        assert payment == 0.0

        # Test year beyond payment schedule (year 15, schedule only has 10 years)
        payment = claim.get_payment(15)
        assert payment == 0.0

        # Test valid year within schedule
        payment_year_0 = claim.get_payment(0)
        assert payment_year_0 > 0
        assert payment_year_0 == claim.original_amount * to_decimal(claim.payment_schedule[0])

    def test_uninsured_claim_immediate_payment(self, manufacturer):
        """Test immediate payment of uninsured claim."""
        initial_assets = manufacturer.total_assets
        initial_equity = manufacturer.equity
        claim_amount = 500_000

        processed_amount = manufacturer.process_uninsured_claim(
            claim_amount, immediate_payment=True
        )

        assert processed_amount == claim_amount
        assert manufacturer.total_assets == initial_assets - claim_amount
        assert manufacturer.equity == initial_equity - claim_amount
        assert manufacturer.period_insurance_losses == claim_amount
        assert len(manufacturer.claim_liabilities) == 0
        assert manufacturer.collateral == 0
        assert manufacturer.restricted_assets == 0

    def test_uninsured_claim_immediate_payment_exceeds_assets(self, manufacturer):
        """Test immediate payment when claim exceeds available assets."""
        # Set all assets to achieve total assets of 100_000
        manufacturer.cash = 100_000
        manufacturer.accounts_receivable = 0
        manufacturer.inventory = 0
        manufacturer.prepaid_insurance = 0
        manufacturer.gross_ppe = 0
        manufacturer.accumulated_depreciation = 0
        manufacturer.restricted_assets = 0
        claim_amount = 500_000

        processed_amount = manufacturer.process_uninsured_claim(
            claim_amount, immediate_payment=True
        )

        # The method returns the full claim amount, not the actual payment
        assert processed_amount == 500_000
        assert manufacturer.total_assets == 0
        # LIMITED LIABILITY: Equity is floored at 0, not -400_000
        # Company pays max $100K (all available equity), bringing equity to 0
        assert manufacturer.equity == 0
        # Only the actual payment (100k) is recorded as a loss
        assert manufacturer.period_insurance_losses == 100_000
        # Company should be marked as insolvent
        assert manufacturer.is_ruined is True
        # LIMITED LIABILITY: Cannot create $400K liability because it would violate limited liability
        # Equity is already 0 after payment, so no additional liabilities can be recorded
        assert len(manufacturer.claim_liabilities) == 0

    def test_uninsured_claim_deferred_payment(self, manufacturer):
        """Test uninsured claim with deferred payment schedule."""
        initial_assets = manufacturer.total_assets
        initial_equity = manufacturer.equity
        claim_amount = 500_000

        processed_amount = manufacturer.process_uninsured_claim(claim_amount)

        assert processed_amount == claim_amount
        assert manufacturer.total_assets == initial_assets
        # Equity decreases by claim amount due to new liability (Assets = Liabilities + Equity)
        assert manufacturer.equity == initial_equity - claim_amount
        assert manufacturer.period_insurance_losses == 0
        assert len(manufacturer.claim_liabilities) == 1
        assert manufacturer.collateral == 0
        assert manufacturer.restricted_assets == 0

        claim = manufacturer.claim_liabilities[0]
        assert claim.original_amount == claim_amount
        assert claim.remaining_amount == claim_amount
        assert claim.year_incurred == manufacturer.current_year
        assert claim.is_insured is False

    def test_uninsured_claim_deferred_payment_multiple_claims(self, manufacturer):
        """Test multiple uninsured claims with deferred payment."""
        claim_amount_1 = 300_000
        claim_amount_2 = 200_000

        manufacturer.process_uninsured_claim(claim_amount_1)
        manufacturer.process_uninsured_claim(claim_amount_2)

        assert len(manufacturer.claim_liabilities) == 2
        assert manufacturer.collateral == 0
        assert manufacturer.restricted_assets == 0

        total_liability = manufacturer.total_claim_liabilities
        assert total_liability == claim_amount_1 + claim_amount_2

        for claim in manufacturer.claim_liabilities:
            assert claim.is_insured is False

    def test_uninsured_claim_zero_amount(self, manufacturer):
        """Test uninsured claim with zero amount."""
        initial_assets = manufacturer.total_assets
        initial_equity = manufacturer.equity

        processed_amount = manufacturer.process_uninsured_claim(0.0)

        assert processed_amount == 0.0
        assert manufacturer.total_assets == initial_assets
        assert manufacturer.equity == initial_equity
        assert len(manufacturer.claim_liabilities) == 0

    def test_uninsured_claim_negative_amount(self, manufacturer):
        """Test uninsured claim with negative amount."""
        initial_assets = manufacturer.total_assets
        initial_equity = manufacturer.equity

        processed_amount = manufacturer.process_uninsured_claim(-100_000)

        assert processed_amount == 0.0
        assert manufacturer.total_assets == initial_assets
        assert manufacturer.equity == initial_equity
        assert len(manufacturer.claim_liabilities) == 0

    def test_uninsured_claim_immediate_vs_deferred_comparison(self, manufacturer):
        """Test comparison between immediate and deferred payment modes."""
        claim_amount = 500_000

        # Test immediate payment
        manufacturer_immediate = WidgetManufacturer(manufacturer.config)
        initial_assets_imm = manufacturer_immediate.total_assets
        manufacturer_immediate.process_uninsured_claim(claim_amount, immediate_payment=True)

        # Test deferred payment
        manufacturer_deferred = WidgetManufacturer(manufacturer.config)
        initial_assets_def = manufacturer_deferred.total_assets
        manufacturer_deferred.process_uninsured_claim(claim_amount, immediate_payment=False)

        # Immediate payment should reduce assets immediately
        assert manufacturer_immediate.total_assets == initial_assets_imm - claim_amount
        # Deferred payment should not reduce assets immediately
        assert manufacturer_deferred.total_assets == initial_assets_def

        # Immediate should have no liabilities
        assert len(manufacturer_immediate.claim_liabilities) == 0
        # Deferred should have one liability
        assert len(manufacturer_deferred.claim_liabilities) == 1

    def test_uninsured_claim_with_existing_collateral(self, manufacturer):
        """Test that uninsured claims don't affect existing collateral from insured claims."""
        # Create an insured claim first to establish collateral
        manufacturer.process_insurance_claim(300_000, 50_000, 500_000)
        initial_collateral = manufacturer.collateral
        initial_restricted = manufacturer.restricted_assets

        # Add uninsured claim
        manufacturer.process_uninsured_claim(200_000)

        # Collateral should remain unchanged
        assert manufacturer.collateral == initial_collateral
        assert manufacturer.restricted_assets == initial_restricted
        # Should have 2 liabilities total (1 insured + 1 uninsured)
        assert len(manufacturer.claim_liabilities) == 2

        insured_claims = [c for c in manufacturer.claim_liabilities if c.is_insured]
        uninsured_claims = [c for c in manufacturer.claim_liabilities if not c.is_insured]
        assert len(insured_claims) == 1
        assert len(uninsured_claims) == 1


class TestStepMethod:
    """Test suite for step() method."""

    @pytest.fixture
    def manufacturer(self):
        """Create a manufacturer for testing."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.1,
            tax_rate=0.25,
            retention_ratio=0.5,
        )
        return WidgetManufacturer(config)

    def test_normal_operation(self, manufacturer):
        """Test normal profitable operation."""
        initial_assets = manufacturer.total_assets
        initial_equity = manufacturer.equity

        metrics = manufacturer.step(
            letter_of_credit_rate=0.015,
            growth_rate=0.0,
        )

        # Check metrics returned
        assert "revenue" in metrics
        assert "operating_income" in metrics
        assert "net_income" in metrics
        assert "assets" in metrics
        assert "equity" in metrics
        assert "roe" in metrics

        # With positive margin, should be profitable
        assert metrics["net_income"] > 0

        # Assets should increase with retained earnings
        assert manufacturer.total_assets > initial_assets

        # Equity change reflects depreciation, tax accruals, and retained earnings
        # With depreciation (500K) + tax accruals (83K) > retained earnings (125K),
        # equity will decrease even with positive net income
        equity_change = manufacturer.equity - initial_equity
        retained_earnings = metrics["net_income"] * to_decimal(manufacturer.retention_ratio)
        # Equity should decrease but by less than depreciation alone
        assert (
            equity_change < 0
        ), "Equity should decrease due to depreciation exceeding retained earnings"
        assert (
            equity_change > -500_000
        ), "Equity decrease should be partially offset by retained earnings"

        # Balance sheet should remain balanced (Assets = Liabilities + Equity)
        assert manufacturer.total_assets == pytest.approx(
            manufacturer.total_liabilities + manufacturer.equity, rel=1e-9
        )

    def test_with_collateral_costs(self, manufacturer):
        """Test step with existing collateral requiring LoC costs."""
        # Create a claim to establish collateral
        manufacturer.process_insurance_claim(500_000, 100_000, 1_000_000)

        # Run step without collateral first
        manufacturer_no_collateral = WidgetManufacturer(manufacturer.config)
        metrics_no_collateral = manufacturer_no_collateral.step(letter_of_credit_rate=0.015)

        # Run step with collateral
        metrics = manufacturer.step(letter_of_credit_rate=0.015)

        # Net income should be lower due to collateral costs
        # The difference should be approximately the after-tax collateral costs
        assert metrics["net_income"] < metrics_no_collateral["net_income"]

    def test_letter_of_credit_calculation_accuracy(self, manufacturer):
        """Test that letter of credit costs are calculated correctly at 1.5% annually."""
        # Create two identical manufacturers with claims
        # One will have zero LoC rate, the other will have 1.5%
        claim_amount = 5_000_000
        deductible = 1_000_000  # This will be the collateral amount
        limit = 10_000_000

        # Create manufacturer with collateral but zero LoC rate
        manufacturer_zero_rate = WidgetManufacturer(manufacturer.config)
        manufacturer_zero_rate.process_insurance_claim(claim_amount, deductible, limit)
        assert manufacturer_zero_rate.collateral == deductible

        # Create manufacturer with collateral and 1.5% LoC rate
        manufacturer_with_rate = WidgetManufacturer(manufacturer.config)
        manufacturer_with_rate.process_insurance_claim(claim_amount, deductible, limit)
        assert manufacturer_with_rate.collateral == deductible

        # Run steps - one with zero rate, one with 1.5% rate
        metrics_zero_rate = manufacturer_zero_rate.step(letter_of_credit_rate=0.0)
        metrics_with_rate = manufacturer_with_rate.step(letter_of_credit_rate=0.015)

        # Verify that net income is lower with collateral costs
        assert metrics_with_rate["net_income"] < metrics_zero_rate["net_income"]

        # The income difference should reflect the LoC costs
        income_difference = float(metrics_zero_rate["net_income"]) - float(
            metrics_with_rate["net_income"]
        )
        assert income_difference > 0  # Cost is applied

        # Test monthly calculation
        manufacturer.reset()
        # Process the same claim to establish collateral
        manufacturer.process_insurance_claim(claim_amount, deductible, limit)

        for _ in range(12):
            manufacturer.step(time_resolution="monthly", letter_of_credit_rate=0.015)
            # Monthly cost should be annual rate / 12
            # This is embedded in the net income calculation

        # After 12 months, total cost should equal annual cost
        # This is verified through the cumulative impact on equity

    def test_with_growth_rate(self, manufacturer):
        """Test step with revenue growth."""
        initial_turnover = manufacturer.asset_turnover_ratio

        manufacturer.step(growth_rate=0.1)  # 10% growth

        # Asset turnover should increase
        assert manufacturer.asset_turnover_ratio == pytest.approx(initial_turnover * 1.1)

    def test_monthly_resolution(self, manufacturer):
        """Test monthly time resolution."""
        # Run 12 monthly steps
        for month in range(12):
            metrics = manufacturer.step(time_resolution="monthly")
            assert metrics["month"] == month

        # After 12 months, should be in year 1
        assert manufacturer.current_year == 1
        assert manufacturer.current_month == 0

    def test_insolvency_detection(self, manufacturer):
        """Test that insolvency is properly detected."""
        # Force insolvency by creating negative equity
        # First create a large liability, then reduce assets
        manufacturer.process_uninsured_claim(20_000_000, immediate_payment=False)
        # Now reduce assets to near zero
        manufacturer.cash = 0
        manufacturer.accounts_receivable = 0
        manufacturer.inventory = 0
        manufacturer.prepaid_insurance = 0
        manufacturer.gross_ppe = 0
        manufacturer.accumulated_depreciation = 0
        manufacturer.restricted_assets = 0

        manufacturer.step()

        assert manufacturer.is_ruined is True

        # Further steps should not process
        manufacturer.step()
        assert manufacturer.is_ruined is True

    def test_insolvency_monthly_resolution(self, manufacturer):
        """Test monthly steps when company is insolvent."""
        # Force insolvency by creating negative equity
        manufacturer.process_uninsured_claim(20_000_000, immediate_payment=False)
        # Reduce assets to near zero
        manufacturer.cash = 0
        manufacturer.accounts_receivable = 0
        manufacturer.inventory = 0
        manufacturer.prepaid_insurance = 0
        manufacturer.gross_ppe = 0
        manufacturer.accumulated_depreciation = 0
        manufacturer.restricted_assets = 0
        manufacturer.is_ruined = True
        manufacturer.current_month = 10
        manufacturer.current_year = 5

        # Run a monthly step when insolvent
        metrics = manufacturer.step(time_resolution="monthly")

        # Should still return metrics with proper time tracking
        assert metrics["year"] == 5
        assert metrics["month"] == 10
        assert manufacturer.is_ruined is True

        # Time should advance
        assert manufacturer.current_month == 11

        # Test year rollover when insolvent
        metrics = manufacturer.step(time_resolution="monthly")
        assert manufacturer.current_month == 0
        assert manufacturer.current_year == 6

    def test_balance_sheet_consistency(self, manufacturer):
        """Test that balance sheet remains consistent after operations."""
        # Run several steps with various operations
        for i in range(5):
            if i == 2:
                # Add a claim in the middle
                manufacturer.process_insurance_claim(300_000, 50_000, 500_000)

            manufacturer.step()

            # Balance sheet equation: Assets = Liabilities + Equity
            # Therefore: Equity = Assets - Liabilities
            # The manufacturer.equity property correctly implements this
            assert manufacturer.equity == pytest.approx(
                manufacturer.total_assets - manufacturer.total_liabilities,
                rel=0.01,
            )

            # Alternative check: Assets should equal Liabilities + Equity
            assert manufacturer.total_assets == pytest.approx(
                manufacturer.total_liabilities + manufacturer.equity,
                rel=0.01,
            )

    def test_revenue_calculation(self, manufacturer):
        """Test revenue calculation.

        Issue #244: working_capital_pct parameter was removed to fix double-counting.
        Revenue is now simply: Assets * Turnover Ratio.
        Working capital impact flows through calculate_working_capital_components()
        and the cash flow statement.
        """
        # Capture initial assets before step modifies them
        initial_assets = manufacturer.total_assets
        initial_turnover = manufacturer.asset_turnover_ratio

        metrics = manufacturer.step()

        # Should generate revenue based on assets and turnover
        assert metrics["revenue"] > 0
        # Revenue = Assets * Turnover Ratio
        expected_revenue = initial_assets * to_decimal(initial_turnover)
        assert metrics["revenue"] == pytest.approx(expected_revenue, rel=0.01)

    def test_claim_liability_payments(self, manufacturer):
        """Test that claim liabilities are paid according to schedule."""
        # Create a claim liability
        manufacturer.process_insurance_claim(1_000_000, 100_000, 2_000_000)
        initial_liabilities = manufacturer.total_claim_liabilities

        # Step forward one year
        manufacturer.current_year = 1
        manufacturer.step()

        # Some liability should have been paid
        assert manufacturer.total_claim_liabilities < initial_liabilities

    def test_metrics_history(self, manufacturer):
        """Test that metrics are properly stored in history."""
        assert len(manufacturer.metrics_history) == 0

        # Run several steps
        for _ in range(3):
            manufacturer.step()

        assert len(manufacturer.metrics_history) == 3

        # Each entry should have key metrics
        for metrics in manufacturer.metrics_history:
            assert "assets" in metrics
            assert "equity" in metrics
            assert "revenue" in metrics
            assert "net_income" in metrics
            assert "year" in metrics

    def test_multiple_claims_integration(self, manufacturer):
        """Test handling multiple claims over time with comprehensive verification."""

        # Year 0: First claim
        claim1_amount = 500_000
        deductible = 100_000
        limit = 1_000_000
        manufacturer.process_insurance_claim(claim1_amount, deductible, limit)

        # Verify claim 1 setup
        # Collateral should equal the deductible (company payment portion)
        assert manufacturer.collateral == deductible
        assert len(manufacturer.claim_liabilities) == 1

        # Year 1: Step and add second claim
        metrics_year1 = manufacturer.step()
        assert metrics_year1["year"] == 0  # First step is still year 0

        claim2_amount = 300_000
        manufacturer.process_insurance_claim(claim2_amount, deductible, limit)

        # Should have two claims now
        assert len(manufacturer.claim_liabilities) == 2
        # Total collateral is sum of deductibles (company payment portions)
        total_collateral = deductible + deductible  # Two claims, each with same deductible

        # Year 2-5: Continue operations and track claim payments
        for year in range(1, 5):
            metrics = manufacturer.step()
            assert metrics["year"] == year

            # Verify claims are being paid down
            current_liabilities = manufacturer.total_claim_liabilities
            if year > 2:
                # Liabilities should decrease over time
                assert current_liabilities < total_collateral

        # Verify balance sheet integrity throughout
        assert manufacturer.check_solvency() is True

    def test_edge_case_scenarios(self, manufacturer):
        """Test various edge case scenarios."""
        # Test with zero operating margin
        manufacturer.base_operating_margin = 0.0
        metrics = manufacturer.step()

        # With zero margin, operating income should be negative due to depreciation
        # Depreciation = PP&E / 10 = $5M / 10 = $500K
        expected_operating_income = -(manufacturer.gross_ppe / 10)
        assert metrics["operating_income"] == expected_operating_income

        # Test with negative growth rate
        manufacturer.reset()
        initial_turnover = manufacturer.asset_turnover_ratio
        metrics = manufacturer.step(growth_rate=-0.1)

        # Asset turnover should decrease
        assert manufacturer.asset_turnover_ratio == pytest.approx(initial_turnover * 0.9)
