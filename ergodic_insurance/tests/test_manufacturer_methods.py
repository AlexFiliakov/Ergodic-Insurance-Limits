"""Unit tests for WidgetManufacturer step() and process_insurance_claim() methods."""

import pytest

from ergodic_insurance.src.config import ManufacturerConfig
from ergodic_insurance.src.manufacturer import WidgetManufacturer


class TestProcessInsuranceClaim:
    """Test suite for process_insurance_claim() method."""

    @pytest.fixture
    def manufacturer(self):
        """Create a manufacturer for testing."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            operating_margin=0.1,
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

        # No collateral should be posted
        assert manufacturer.collateral == 0
        assert manufacturer.restricted_assets == 0
        assert len(manufacturer.claim_liabilities) == 0

    def test_claim_above_deductible_below_limit(self, manufacturer):
        """Test claim between deductible and limit."""
        initial_assets = manufacturer.assets
        claim_amount = 500_000
        deductible = 100_000
        limit = 1_000_000

        company_payment, insurance_payment = manufacturer.process_insurance_claim(
            claim_amount, deductible, limit
        )

        # Company pays deductible, insurance pays the rest
        assert company_payment == deductible
        assert insurance_payment == claim_amount - deductible

        # Check collateral and liabilities
        assert manufacturer.collateral == insurance_payment
        assert manufacturer.restricted_assets == insurance_payment
        assert len(manufacturer.claim_liabilities) == 1

        # Check assets reduced by company payment
        assert manufacturer.assets == initial_assets - company_payment

    def test_claim_above_limit(self, manufacturer):
        """Test claim that exceeds insurance limit."""
        initial_assets = manufacturer.assets
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

        # Check collateral for insurance portion
        assert manufacturer.collateral == limit
        assert manufacturer.restricted_assets == limit

        # Check assets reduced
        assert manufacturer.assets == initial_assets - expected_company

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
        """Test claim when company doesn't have enough assets."""
        manufacturer.assets = 50_000  # Set low assets
        manufacturer.equity = 50_000
        initial_assets = manufacturer.assets

        claim_amount = 200_000
        deductible = 100_000
        limit = 1_000_000

        company_payment, insurance_payment = manufacturer.process_insurance_claim(
            claim_amount, deductible, limit
        )

        # Company tries to pay deductible but is limited by assets
        # The actual payment made is limited to available assets
        assert manufacturer.assets == 0  # All assets used
        # Insurance still covers its portion
        assert insurance_payment == claim_amount - deductible


class TestStepMethod:
    """Test suite for step() method."""

    @pytest.fixture
    def manufacturer(self):
        """Create a manufacturer for testing."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            operating_margin=0.1,
            tax_rate=0.25,
            retention_ratio=0.5,
        )
        return WidgetManufacturer(config)

    def test_normal_operation(self, manufacturer):
        """Test normal profitable operation."""
        initial_assets = manufacturer.assets
        initial_equity = manufacturer.equity

        metrics = manufacturer.step(
            working_capital_pct=0.2,
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
        assert manufacturer.assets > initial_assets
        assert manufacturer.equity > initial_equity

        # Balance sheet should remain balanced
        assert manufacturer.equity == manufacturer.assets  # No debt

    def test_with_collateral_costs(self, manufacturer):
        """Test step with existing collateral requiring LoC costs."""
        # Create a claim to establish collateral
        manufacturer.process_insurance_claim(500_000, 100_000, 1_000_000)
        collateral_amount = manufacturer.collateral

        # Run step without collateral first
        manufacturer_no_collateral = WidgetManufacturer(manufacturer.config)
        metrics_no_collateral = manufacturer_no_collateral.step(letter_of_credit_rate=0.015)

        # Run step with collateral
        metrics = manufacturer.step(letter_of_credit_rate=0.015)

        # Should have collateral costs reducing net income
        expected_costs = collateral_amount * 0.015

        # Net income should be lower due to collateral costs
        # The difference should be approximately the after-tax collateral costs
        assert metrics["net_income"] < metrics_no_collateral["net_income"]

    def test_with_growth_rate(self, manufacturer):
        """Test step with revenue growth."""
        initial_turnover = manufacturer.asset_turnover_ratio

        metrics = manufacturer.step(growth_rate=0.1)  # 10% growth

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
        # Force insolvency by setting very low assets
        manufacturer.assets = 100
        manufacturer.equity = -1000  # Negative equity

        metrics = manufacturer.step()

        assert manufacturer.is_ruined is True

        # Further steps should not process
        metrics2 = manufacturer.step()
        assert manufacturer.is_ruined is True

    def test_balance_sheet_consistency(self, manufacturer):
        """Test that balance sheet remains consistent after operations."""
        # Run several steps with various operations
        for i in range(5):
            if i == 2:
                # Add a claim in the middle
                manufacturer.process_insurance_claim(300_000, 50_000, 500_000)

            metrics = manufacturer.step()

            # Without debt, equity should equal assets minus restricted assets
            # plus claim liabilities
            total_liabilities = manufacturer.total_claim_liabilities
            net_assets = manufacturer.assets - manufacturer.restricted_assets

            # Balance sheet equation (simplified for no debt case)
            # Assets = Equity + Liabilities (represented by restricted assets)
            assert manufacturer.equity == pytest.approx(
                manufacturer.assets - manufacturer.restricted_assets + total_liabilities, rel=0.01
            )

    def test_zero_working_capital(self, manufacturer):
        """Test operation with zero working capital."""
        metrics = manufacturer.step(working_capital_pct=0.0)

        # Should still generate revenue
        assert metrics["revenue"] > 0
        # Revenue should be higher with no working capital constraint
        expected_revenue = manufacturer.assets * manufacturer.asset_turnover_ratio
        assert metrics["revenue"] == pytest.approx(expected_revenue, rel=0.01)

    def test_high_working_capital(self, manufacturer):
        """Test operation with high working capital requirement."""
        # Get baseline revenue with no working capital
        metrics_zero_wc = manufacturer.step(working_capital_pct=0.0)
        manufacturer.reset()  # Reset for next test

        # Now test with high working capital
        metrics = manufacturer.step(working_capital_pct=0.5)

        # Revenue should be reduced due to working capital constraint
        assert metrics["revenue"] > 0
        # But less than with zero working capital
        assert metrics["revenue"] < metrics_zero_wc["revenue"]

    def test_claim_liability_payments(self, manufacturer):
        """Test that claim liabilities are paid according to schedule."""
        # Create a claim liability
        manufacturer.process_insurance_claim(1_000_000, 100_000, 2_000_000)
        initial_liabilities = manufacturer.total_claim_liabilities

        # Step forward one year
        manufacturer.current_year = 1
        metrics = manufacturer.step()

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
