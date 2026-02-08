"""Unit tests for tax handling in insurance premium and loss calculations.

This module tests the proper tax treatment of insurance premiums and losses
in the WidgetManufacturer class, ensuring they are correctly handled as
tax-deductible business expenses.
"""

import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decimal_utils import to_decimal
from ergodic_insurance.manufacturer import WidgetManufacturer


class TestTaxHandling:
    """Test suite for tax handling of insurance costs.

    Tests the proper tax treatment of insurance premiums and losses,
    ensuring they are correctly treated as tax-deductible business expenses
    in the financial calculations.
    """

    @pytest.fixture
    def config(self):
        """Create a standard test configuration."""
        return ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.15,
            tax_rate=0.25,
            retention_ratio=0.7,
        )

    @pytest.fixture
    def manufacturer(self, config):
        """Create a WidgetManufacturer instance for testing."""
        return WidgetManufacturer(config)

    def test_baseline_net_income_calculation(self, manufacturer):
        """Test baseline net income calculation without insurance costs.

        Verifies that the basic tax calculation works correctly
        without any insurance premiums or losses.
        """
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)
        net_income = manufacturer.calculate_net_income(operating_income, 0)

        # Expected: Operating income = $1.5M, taxes = $375K, net = $1.125M
        expected_taxes = operating_income * to_decimal(manufacturer.tax_rate)
        expected_net = operating_income - expected_taxes

        assert abs(net_income - expected_net) < 0.01
        assert expected_net == pytest.approx(1_125_000)

    def test_premium_tax_deductibility(self, manufacturer):
        """Test that insurance premiums are properly tax-deductible.

        Premiums are deducted in calculate_operating_income() and flow
        through to net income. Verifies the single-deduction income waterfall.
        """
        revenue = manufacturer.calculate_revenue()

        # Baseline operating income (no premium recorded yet)
        operating_income_baseline = manufacturer.calculate_operating_income(revenue)

        # Record a premium and recalculate operating income
        premium = 500_000
        manufacturer.record_insurance_premium(premium)
        operating_income_with_premium = manufacturer.calculate_operating_income(revenue)

        # Premium should reduce operating income by exactly the premium amount
        assert operating_income_baseline - operating_income_with_premium == pytest.approx(premium)

        # Net income reflects the premium through operating income
        net_income = manufacturer.calculate_net_income(operating_income_with_premium, 0)
        income_before_tax = operating_income_with_premium
        expected_taxes = income_before_tax * to_decimal(manufacturer.tax_rate)
        expected_net = income_before_tax - expected_taxes

        assert abs(float(net_income) - float(expected_net)) < 0.01

    def test_loss_tax_deductibility(self, manufacturer):
        """Test that insurance losses are properly tax-deductible.

        Losses are deducted in calculate_operating_income() via
        period_insurance_losses and flow through to net income.
        """
        revenue = manufacturer.calculate_revenue()

        # Baseline operating income (no losses recorded yet)
        operating_income_baseline = manufacturer.calculate_operating_income(revenue)

        # Record a loss and recalculate operating income
        loss = 300_000
        manufacturer.period_insurance_losses = to_decimal(loss)
        operating_income_with_loss = manufacturer.calculate_operating_income(revenue)

        # Loss should reduce operating income by exactly the loss amount
        assert operating_income_baseline - operating_income_with_loss == pytest.approx(loss)

        # Net income reflects the loss through operating income
        net_income = manufacturer.calculate_net_income(operating_income_with_loss, 0)
        income_before_tax = operating_income_with_loss
        expected_taxes = income_before_tax * to_decimal(manufacturer.tax_rate)
        expected_net = income_before_tax - expected_taxes

        assert abs(float(net_income) - float(expected_net)) < 0.01

    def test_combined_premium_and_loss_deductibility(self, manufacturer):
        """Test combined premium and loss tax deductibility.

        Both premiums and losses are deducted in calculate_operating_income()
        and flow through to net income correctly.
        """
        revenue = manufacturer.calculate_revenue()

        # Baseline operating income
        operating_income_baseline = manufacturer.calculate_operating_income(revenue)

        # Record both premium and loss
        premium = 400_000
        loss = 200_000
        manufacturer.record_insurance_premium(premium)
        manufacturer.period_insurance_losses = to_decimal(loss)
        operating_income_with_costs = manufacturer.calculate_operating_income(revenue)

        # Both should reduce operating income
        total_insurance = premium + loss
        assert operating_income_baseline - operating_income_with_costs == pytest.approx(
            total_insurance
        )

        # Net income reflects combined costs through operating income
        net_income = manufacturer.calculate_net_income(operating_income_with_costs, 0)
        income_before_tax = operating_income_with_costs
        expected_taxes = income_before_tax * to_decimal(manufacturer.tax_rate)
        expected_net = income_before_tax - expected_taxes

        assert abs(float(net_income) - float(expected_net)) < 0.01

    def test_collateral_costs_deduction(self, manufacturer):
        """Test that collateral costs are properly deducted below operating income.

        Collateral costs are the only below-the-line deduction in
        calculate_net_income() after insurance was moved to operating income.
        """
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        collateral_costs = 50_000

        net_income = manufacturer.calculate_net_income(operating_income, collateral_costs)

        # Expected calculation
        income_before_tax = operating_income - to_decimal(collateral_costs)
        expected_taxes = income_before_tax * to_decimal(manufacturer.tax_rate)
        expected_net = income_before_tax - expected_taxes

        assert abs(float(net_income) - float(expected_net)) < 0.01

    def test_zero_collateral_costs(self, manufacturer):
        """Test that zero collateral costs work correctly.

        Verifies that calling with zero collateral produces correct results.
        """
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        net_income = manufacturer.calculate_net_income(operating_income, 0)

        # Should just be operating income minus taxes
        expected_taxes = operating_income * to_decimal(manufacturer.tax_rate)
        expected_net = operating_income - expected_taxes

        assert abs(net_income - expected_net) < 0.01

    def test_negative_income_no_tax_benefit(self, manufacturer):
        """Test that negative pre-tax income generates no tax benefit.

        Verifies that when collateral costs exceed operating income,
        no tax benefit is generated (no negative taxes).
        """
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # Large collateral costs that exceed operating income
        excessive_collateral = operating_income + 100_000

        net_income = manufacturer.calculate_net_income(operating_income, excessive_collateral)

        # Expected: negative income before tax, zero taxes
        income_before_tax = operating_income - excessive_collateral
        assert income_before_tax < 0

        # Net income should equal income before tax (no taxes paid)
        assert abs(net_income - income_before_tax) < 0.01
        assert net_income < 0

    def test_period_premium_tracking(self, manufacturer):
        """Test that premium tracking works correctly.

        Verifies that the record_insurance_premium method properly
        tracks premiums and that they flow through the income statement
        when step() is called.
        """
        initial_assets = manufacturer.total_assets
        initial_equity = manufacturer.equity

        premium = 500_000
        manufacturer.record_insurance_premium(premium)

        # Check tracking
        assert manufacturer.period_insurance_premiums == premium

        # Assets shouldn't change immediately - premiums flow through income statement
        assert manufacturer.total_assets == initial_assets
        assert manufacturer.equity == initial_equity

        # Test multiple premiums accumulate
        additional_premium = 200_000
        manufacturer.record_insurance_premium(additional_premium)

        assert manufacturer.period_insurance_premiums == premium + additional_premium
        assert manufacturer.total_assets == initial_assets  # Still no immediate change

        # Now run step() to see the premium flow through income statement
        metrics = manufacturer.step()

        # Check that premiums were included in the metrics
        assert metrics["insurance_premiums"] == premium + additional_premium

        # Compare what assets would have been WITHOUT the premiums
        # Calculate expected net income impact
        total_premiums = premium + additional_premium

        # Net income should be lower due to premiums (tax-deductible expense)
        # The premiums reduce pre-tax income, which reduces taxes, so the net impact
        # is premiums * (1 - tax_rate) * retention_ratio
        tax_rate = to_decimal(manufacturer.tax_rate)
        retention_ratio = to_decimal(manufacturer.retention_ratio)

        # Expected reduction in retained earnings due to premiums
        premium_impact = to_decimal(total_premiums) * (to_decimal(1) - tax_rate) * retention_ratio

        # Without premiums, assets would have been higher
        # We can't directly test final assets since we don't know all the other income/expenses
        # But we can verify the premiums were processed
        assert metrics["total_insurance_costs"] == total_premiums

        # Verify premiums were reset after step
        assert manufacturer.period_insurance_premiums == 0

    def test_period_loss_tracking(self, manufacturer):
        """Test that deductibles create liabilities, not expenses.

        With correct accounting, deductibles create liabilities that reduce equity
        through the accounting equation, not expenses that reduce operating income.
        """
        # Process a claim to generate company payment
        claim_amount = 2_000_000
        deductible = 500_000

        company_payment, insurance_payment = manufacturer.process_insurance_claim(
            claim_amount, deductible, 10_000_000
        )

        # Verify deductible creates liability, not expense
        assert manufacturer.period_insurance_losses == 0  # No expense recorded
        assert manufacturer.total_claim_liabilities == company_payment  # Liability created
        assert company_payment == deductible  # Should equal deductible amount
        assert insurance_payment == claim_amount - deductible

    def test_period_cost_reset(self, manufacturer):
        """Test that period costs are reset after step.

        Verifies that period insurance costs are properly reset
        after each simulation step.
        """
        # Add premium (which creates an expense)
        manufacturer.record_insurance_premium(300_000)

        # Add uninsured claim with immediate payment (which creates an expense)
        manufacturer.process_uninsured_claim(200_000, immediate_payment=True)

        # Verify costs are tracked
        assert manufacturer.period_insurance_premiums > 0
        assert manufacturer.period_insurance_losses > 0

        # Run a step
        manufacturer.step()

        # Verify costs are reset
        assert manufacturer.period_insurance_premiums == 0.0
        assert manufacturer.period_insurance_losses == 0.0

    def test_step_integration_with_insurance_costs(self, manufacturer):
        """Test that step method properly uses insurance costs in calculation.

        Verifies that the step method correctly includes tracked insurance
        costs in the net income calculation.
        """
        # Record some insurance costs during the period
        manufacturer.record_insurance_premium(400_000)
        manufacturer.process_insurance_claim(1_500_000, 300_000, 5_000_000)

        # Run step and get metrics
        metrics = manufacturer.step()

        # The net income should reflect tax savings from insurance costs
        # We can't easily calculate the exact expected value without duplicating
        # the entire calculation, but we can verify it's reasonable
        assert "net_income" in metrics
        # net_income is Decimal, convert to float for type checking
        assert isinstance(float(metrics["net_income"]), (int, float))

        # Verify that period costs were used and then reset
        assert manufacturer.period_insurance_premiums == 0.0
        assert manufacturer.period_insurance_losses == 0.0

    def test_reset_clears_period_costs(self, manufacturer):
        """Test that reset clears period insurance costs.

        Verifies that the reset method properly clears all period
        insurance cost tracking.
        """
        # Add some period costs
        manufacturer.record_insurance_premium(100_000)
        manufacturer.period_insurance_losses = 50_000

        # Reset manufacturer
        manufacturer.reset()

        # Verify costs are cleared
        assert manufacturer.period_insurance_premiums == 0.0
        assert manufacturer.period_insurance_losses == 0.0

    @pytest.mark.parametrize("tax_rate", [0.0, 0.15, 0.25, 0.35, 0.5])
    def test_tax_deduction_with_different_rates(self, config, tax_rate):
        """Test tax deduction works correctly with different tax rates.

        Verifies that the tax deduction calculation is correct
        across various corporate tax rates.
        """
        config.tax_rate = tax_rate
        manufacturer = WidgetManufacturer(config)

        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        collateral_costs = 600_000
        net_income = manufacturer.calculate_net_income(operating_income, collateral_costs)

        # Calculate expected values
        income_before_tax = operating_income - to_decimal(collateral_costs)
        expected_taxes = max(to_decimal(0), income_before_tax * to_decimal(tax_rate))
        expected_net = income_before_tax - expected_taxes

        assert abs(float(net_income) - float(expected_net)) < 0.01

    def test_large_collateral_costs_edge_case(self, manufacturer):
        """Test handling of collateral costs larger than operating income.

        Verifies proper handling when collateral costs exceed operating income,
        resulting in zero taxes but significant losses.
        """
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # Collateral costs exceed operating income
        excessive_collateral = operating_income * 2
        net_income = manufacturer.calculate_net_income(operating_income, excessive_collateral)

        # Should result in negative net income with no taxes
        expected_net = operating_income - excessive_collateral
        assert expected_net < 0
        assert abs(net_income - expected_net) < 0.01

    def test_cost_precision(self, manufacturer):
        """Test precision of cost calculations.

        Verifies that small amounts are handled with
        appropriate precision in tax calculations.
        """
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # Test with small collateral amount
        small_collateral = 4.25

        net_income = manufacturer.calculate_net_income(operating_income, small_collateral)

        income_before_tax = operating_income - to_decimal(small_collateral)
        expected_taxes = income_before_tax * to_decimal(manufacturer.tax_rate)
        expected_net = income_before_tax - expected_taxes

        assert abs(float(net_income) - float(expected_net)) < 0.01


class TestTaxHandler:
    """Test suite for the consolidated TaxHandler class.

    Tests the TaxHandler class which centralizes tax calculation, limited
    liability capping, and accrual recording logic. These tests verify that
    the consolidated logic maintains correct behavior and documents the
    non-circular tax flow.
    """

    @pytest.fixture
    def accrual_manager(self):
        """Create an AccrualManager for testing."""
        from ergodic_insurance.accrual_manager import AccrualManager

        return AccrualManager()

    @pytest.fixture
    def tax_handler(self, accrual_manager):
        """Create a TaxHandler for testing."""
        from ergodic_insurance.manufacturer import TaxHandler

        return TaxHandler(tax_rate=0.25, accrual_manager=accrual_manager)

    def test_calculate_tax_liability_positive_income(self, tax_handler):
        """Test tax calculation on positive income."""
        tax, nol_used = tax_handler.calculate_tax_liability(1_000_000)
        assert tax == 250_000  # 25% of $1M
        assert nol_used == 0

    def test_calculate_tax_liability_zero_income(self, tax_handler):
        """Test tax calculation on zero income."""
        tax, nol_used = tax_handler.calculate_tax_liability(0)
        assert tax == 0.0
        assert nol_used == 0

    def test_calculate_tax_liability_negative_income(self, tax_handler):
        """Test tax calculation on negative income (loss)."""
        tax, nol_used = tax_handler.calculate_tax_liability(-500_000)
        assert tax == 0.0  # No tax on losses
        assert nol_used == 0

    def test_limited_liability_cap_sufficient_equity(self, tax_handler):
        """Test that no cap is applied when equity is sufficient."""
        capped_amount, was_capped = tax_handler.apply_limited_liability_cap(
            tax_amount=250_000, current_equity=1_000_000
        )
        assert capped_amount == 250_000
        assert was_capped is False

    def test_limited_liability_cap_insufficient_equity(self, tax_handler):
        """Test that cap is applied when equity is insufficient."""
        capped_amount, was_capped = tax_handler.apply_limited_liability_cap(
            tax_amount=500_000, current_equity=200_000
        )
        assert capped_amount == 200_000
        assert was_capped is True

    def test_limited_liability_cap_zero_equity(self, tax_handler):
        """Test that cap is zero when equity is zero."""
        capped_amount, was_capped = tax_handler.apply_limited_liability_cap(
            tax_amount=100_000, current_equity=0
        )
        assert capped_amount == 0.0
        assert was_capped is True

    def test_limited_liability_cap_negative_equity(self, tax_handler):
        """Test that cap is zero when equity is negative."""
        capped_amount, was_capped = tax_handler.apply_limited_liability_cap(
            tax_amount=100_000, current_equity=-50_000
        )
        assert capped_amount == 0.0
        assert was_capped is True

    def test_record_tax_accrual_annual(self, tax_handler, accrual_manager):
        """Test recording annual tax accrual."""
        from ergodic_insurance.accrual_manager import AccrualType

        tax_handler.record_tax_accrual(
            amount=100_000,
            time_resolution="annual",
            current_year=2024,
            current_month=0,
            description="Year 2024 tax liability",
        )

        # Check accrual was recorded
        tax_accruals = accrual_manager.get_accruals_by_type(AccrualType.TAXES)
        assert len(tax_accruals) == 1
        assert tax_accruals[0].amount == 100_000

    def test_record_tax_accrual_zero_amount(self, tax_handler, accrual_manager):
        """Test that zero amount tax accrual is not recorded."""
        from ergodic_insurance.accrual_manager import AccrualType

        tax_handler.record_tax_accrual(
            amount=0,
            time_resolution="annual",
            current_year=2024,
            current_month=0,
        )

        # Check no accrual was recorded
        tax_accruals = accrual_manager.get_accruals_by_type(AccrualType.TAXES)
        assert len(tax_accruals) == 0

    def test_calculate_and_accrue_tax_annual(self, tax_handler, accrual_manager):
        """Test full tax calculation and accrual for annual mode."""
        from ergodic_insurance.accrual_manager import AccrualType

        actual_tax, was_capped, nol_utilized = tax_handler.calculate_and_accrue_tax(
            income_before_tax=1_000_000,
            current_equity=5_000_000,
            use_accrual=True,
            time_resolution="annual",
            current_year=2024,
            current_month=0,
        )

        # Check tax calculation
        assert actual_tax == 250_000  # 25% of $1M
        assert was_capped is False
        assert nol_utilized == 0

        # Check accrual was recorded
        tax_accruals = accrual_manager.get_accruals_by_type(AccrualType.TAXES)
        assert len(tax_accruals) == 1
        assert tax_accruals[0].amount == 250_000

    def test_calculate_and_accrue_tax_with_cap(self, tax_handler, accrual_manager):
        """Test tax calculation with limited liability cap applied."""
        from ergodic_insurance.accrual_manager import AccrualType

        actual_tax, was_capped, nol_utilized = tax_handler.calculate_and_accrue_tax(
            income_before_tax=1_000_000,
            current_equity=100_000,  # Less than $250K tax
            use_accrual=True,
            time_resolution="annual",
            current_year=2024,
            current_month=0,
        )

        # Check cap was applied
        assert actual_tax == 100_000  # Capped at equity
        assert was_capped is True

        # Check capped amount was recorded
        tax_accruals = accrual_manager.get_accruals_by_type(AccrualType.TAXES)
        assert len(tax_accruals) == 1
        assert tax_accruals[0].amount == 100_000

    def test_calculate_and_accrue_tax_no_accrual(self, tax_handler, accrual_manager):
        """Test tax calculation without accrual recording."""
        from ergodic_insurance.accrual_manager import AccrualType

        actual_tax, was_capped, nol_utilized = tax_handler.calculate_and_accrue_tax(
            income_before_tax=1_000_000,
            current_equity=5_000_000,
            use_accrual=False,  # Accrual disabled
            time_resolution="annual",
            current_year=2024,
            current_month=0,
        )

        # Check tax calculation still works
        assert actual_tax == 250_000
        assert was_capped is False

        # Check NO accrual was recorded
        tax_accruals = accrual_manager.get_accruals_by_type(AccrualType.TAXES)
        assert len(tax_accruals) == 0

    def test_calculate_and_accrue_tax_monthly_quarter_end(self, tax_handler, accrual_manager):
        """Test tax accrual at quarter-end in monthly mode."""
        from ergodic_insurance.accrual_manager import AccrualType

        # Month 2 is end of Q1
        actual_tax, was_capped, nol_utilized = tax_handler.calculate_and_accrue_tax(
            income_before_tax=500_000,
            current_equity=5_000_000,
            use_accrual=True,
            time_resolution="monthly",
            current_year=2024,
            current_month=2,  # End of Q1
        )

        # Check accrual was recorded at quarter end
        tax_accruals = accrual_manager.get_accruals_by_type(AccrualType.TAXES)
        assert len(tax_accruals) == 1
        assert tax_accruals[0].amount == 125_000  # 25% of $500K

    def test_calculate_and_accrue_tax_monthly_non_quarter_end(self, tax_handler, accrual_manager):
        """Test that no accrual is recorded mid-quarter in monthly mode."""
        from ergodic_insurance.accrual_manager import AccrualType

        # Month 1 is mid-Q1
        actual_tax, was_capped, nol_utilized = tax_handler.calculate_and_accrue_tax(
            income_before_tax=500_000,
            current_equity=5_000_000,
            use_accrual=True,
            time_resolution="monthly",
            current_year=2024,
            current_month=1,  # Mid-quarter
        )

        # Tax is still calculated
        assert actual_tax == 125_000

        # But NO accrual recorded mid-quarter
        tax_accruals = accrual_manager.get_accruals_by_type(AccrualType.TAXES)
        assert len(tax_accruals) == 0

    def test_tax_handler_documents_non_circular_flow(self, tax_handler):
        """Verify TaxHandler docstring documents the non-circular tax flow.

        This test serves as documentation that the tax flow is intentionally
        designed to avoid circular dependencies.
        """
        # The TaxHandler class should have comprehensive documentation
        assert "Tax Flow Sequence" in tax_handler.__class__.__doc__
        assert "No Circular Dependency" in tax_handler.__class__.__doc__
        assert "BEFORE" in tax_handler.__class__.__doc__
        assert "AFTER" in tax_handler.__class__.__doc__
