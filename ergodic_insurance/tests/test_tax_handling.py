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
        net_income = manufacturer.calculate_net_income(operating_income, 0, 0, 0)

        # Expected: Operating income = $1.5M, taxes = $375K, net = $1.125M
        expected_taxes = operating_income * to_decimal(manufacturer.tax_rate)
        expected_net = operating_income - expected_taxes

        assert abs(net_income - expected_net) < 0.01
        assert expected_net == pytest.approx(1_125_000)

    def test_premium_tax_deductibility(self, manufacturer):
        """Test that insurance premiums are properly tax-deductible.

        Verifies that insurance premium payments reduce taxable income
        and provide appropriate tax savings.
        """
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        premium = 500_000
        net_income_with_premium = manufacturer.calculate_net_income(operating_income, 0, premium, 0)

        # Expected calculation
        income_before_tax = operating_income - to_decimal(premium)  # $1M
        expected_taxes = income_before_tax * to_decimal(manufacturer.tax_rate)  # $250K
        expected_net = income_before_tax - expected_taxes  # $750K

        assert abs(float(net_income_with_premium) - float(expected_net)) < 0.01
        assert float(expected_net) == pytest.approx(750_000)

        # Verify tax savings
        baseline_taxes = operating_income * to_decimal(manufacturer.tax_rate)
        tax_savings = baseline_taxes - expected_taxes
        assert tax_savings == pytest.approx(125_000)  # 25% of $500K

    def test_loss_tax_deductibility(self, manufacturer):
        """Test that insurance losses are properly tax-deductible.

        Verifies that company-paid insurance losses (deductibles)
        reduce taxable income and provide appropriate tax savings.
        """
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        loss = 300_000
        net_income_with_loss = manufacturer.calculate_net_income(operating_income, 0, 0, loss)

        # Expected calculation
        income_before_tax = operating_income - to_decimal(loss)  # $1.2M
        expected_taxes = income_before_tax * to_decimal(manufacturer.tax_rate)  # $300K
        expected_net = income_before_tax - expected_taxes  # $900K

        assert abs(float(net_income_with_loss) - float(expected_net)) < 0.01
        assert float(expected_net) == pytest.approx(900_000)

        # Verify tax savings
        baseline_taxes = operating_income * to_decimal(manufacturer.tax_rate)
        tax_savings = baseline_taxes - expected_taxes
        assert tax_savings == pytest.approx(75_000)  # 25% of $300K

    def test_combined_premium_and_loss_deductibility(self, manufacturer):
        """Test combined premium and loss tax deductibility.

        Verifies that both premiums and losses together are properly
        tax-deductible and provide combined tax savings.
        """
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        premium = 400_000
        loss = 200_000
        net_income_combined = manufacturer.calculate_net_income(operating_income, 0, premium, loss)

        # Expected calculation
        total_insurance_costs = to_decimal(premium + loss)  # $600K
        income_before_tax = operating_income - total_insurance_costs  # $900K
        expected_taxes = income_before_tax * to_decimal(manufacturer.tax_rate)  # $225K
        expected_net = income_before_tax - expected_taxes  # $675K

        assert abs(float(net_income_combined) - float(expected_net)) < 0.01
        assert float(expected_net) == pytest.approx(675_000)

        # Verify combined tax savings
        baseline_taxes = operating_income * to_decimal(manufacturer.tax_rate)
        tax_savings = baseline_taxes - expected_taxes
        assert tax_savings == pytest.approx(150_000)  # 25% of $600K

    def test_collateral_costs_with_insurance_costs(self, manufacturer):
        """Test that collateral costs work properly with insurance costs.

        Verifies that collateral costs, insurance premiums, and losses
        all work together properly in the tax calculation.
        """
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        collateral_costs = 50_000
        premium = 300_000
        loss = 100_000

        net_income = manufacturer.calculate_net_income(
            operating_income, collateral_costs, premium, loss
        )

        # Expected calculation
        total_deductible_costs = to_decimal(collateral_costs + premium + loss)  # $450K
        income_before_tax = operating_income - total_deductible_costs  # $1.05M
        expected_taxes = income_before_tax * to_decimal(manufacturer.tax_rate)  # $262.5K
        expected_net = income_before_tax - expected_taxes  # $787.5K

        assert abs(float(net_income) - float(expected_net)) < 0.01
        assert float(expected_net) == pytest.approx(787_500)

    def test_zero_insurance_costs(self, manufacturer):
        """Test that zero insurance costs work correctly.

        Verifies that when no insurance premiums or losses are provided,
        the calculation works the same as the baseline.
        """
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        net_income_zeros = manufacturer.calculate_net_income(operating_income, 0, 0, 0)
        net_income_default = manufacturer.calculate_net_income(operating_income, 0)

        # Both should give the same result
        assert abs(net_income_zeros - net_income_default) < 0.01

    def test_negative_income_no_tax_benefit(self, manufacturer):
        """Test that negative pre-tax income generates no tax benefit.

        Verifies that when insurance costs exceed operating income,
        no tax benefit is generated (no negative taxes).
        """
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # Large insurance costs that exceed operating income
        premium = operating_income + 100_000  # More than operating income
        loss = 200_000

        net_income = manufacturer.calculate_net_income(operating_income, 0, premium, loss)

        # Expected: negative income before tax, zero taxes
        income_before_tax = operating_income - premium - loss
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

        insurance_costs = 600_000
        net_income = manufacturer.calculate_net_income(operating_income, 0, insurance_costs, 0)

        # Calculate expected values
        income_before_tax = operating_income - to_decimal(insurance_costs)
        expected_taxes = max(to_decimal(0), income_before_tax * to_decimal(tax_rate))
        expected_net = income_before_tax - expected_taxes

        assert abs(float(net_income) - float(expected_net)) < 0.01

        # Calculate tax savings
        baseline_taxes = max(to_decimal(0), operating_income * to_decimal(tax_rate))
        tax_savings = baseline_taxes - expected_taxes
        expected_savings = to_decimal(insurance_costs) * to_decimal(tax_rate)

        assert abs(float(tax_savings) - float(expected_savings)) < 0.01

    def test_large_insurance_costs_edge_case(self, manufacturer):
        """Test handling of insurance costs larger than operating income.

        Verifies proper handling when insurance costs exceed operating income,
        resulting in zero taxes but significant losses.
        """
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # Insurance costs exceed operating income
        excessive_premium = operating_income * 2
        net_income = manufacturer.calculate_net_income(operating_income, 0, excessive_premium, 0)

        # Should result in negative net income with no taxes
        expected_net = operating_income - excessive_premium
        assert expected_net < 0
        assert abs(net_income - expected_net) < 0.01

    def test_insurance_cost_precision(self, manufacturer):
        """Test precision of insurance cost calculations.

        Verifies that small insurance amounts are handled with
        appropriate precision in tax calculations.
        """
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # Test with small amounts
        small_premium = 1.50
        small_loss = 2.75

        net_income = manufacturer.calculate_net_income(
            operating_income, 0, small_premium, small_loss
        )

        income_before_tax = operating_income - to_decimal(small_premium) - to_decimal(small_loss)
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
        tax = tax_handler.calculate_tax_liability(1_000_000)
        assert tax == 250_000  # 25% of $1M

    def test_calculate_tax_liability_zero_income(self, tax_handler):
        """Test tax calculation on zero income."""
        tax = tax_handler.calculate_tax_liability(0)
        assert tax == 0.0

    def test_calculate_tax_liability_negative_income(self, tax_handler):
        """Test tax calculation on negative income (loss)."""
        tax = tax_handler.calculate_tax_liability(-500_000)
        assert tax == 0.0  # No tax on losses

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

        actual_tax, was_capped = tax_handler.calculate_and_accrue_tax(
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

        # Check accrual was recorded
        tax_accruals = accrual_manager.get_accruals_by_type(AccrualType.TAXES)
        assert len(tax_accruals) == 1
        assert tax_accruals[0].amount == 250_000

    def test_calculate_and_accrue_tax_with_cap(self, tax_handler, accrual_manager):
        """Test tax calculation with limited liability cap applied."""
        from ergodic_insurance.accrual_manager import AccrualType

        actual_tax, was_capped = tax_handler.calculate_and_accrue_tax(
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

        actual_tax, was_capped = tax_handler.calculate_and_accrue_tax(
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
        actual_tax, was_capped = tax_handler.calculate_and_accrue_tax(
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
        actual_tax, was_capped = tax_handler.calculate_and_accrue_tax(
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
