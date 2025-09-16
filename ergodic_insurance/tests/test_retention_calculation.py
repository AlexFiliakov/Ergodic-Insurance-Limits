"""
Test suite for validating retention ratio calculations.

This module ensures that retention_ratio is correctly applied to net income
(profit after all costs) rather than revenue or other intermediate values.
"""

import numpy as np
import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.manufacturer import WidgetManufacturer


class TestRetentionCalculation:
    """Comprehensive tests for retention ratio application."""

    def test_retention_applies_to_net_income_not_revenue(self):
        """Ensure retention ratio is applied to profit, not revenue."""
        # Setup manufacturer with known parameters
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.2,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.70,
        )
        manufacturer = WidgetManufacturer(config)

        # Calculate expected values
        expected_revenue = manufacturer.assets * manufacturer.asset_turnover_ratio
        expected_operating_income = expected_revenue * manufacturer.base_operating_margin

        # No additional costs for this test
        expected_income_before_tax = expected_operating_income
        expected_taxes = expected_income_before_tax * manufacturer.tax_rate
        expected_net_income = expected_income_before_tax - expected_taxes

        # Expected retained earnings (70% of net income, NOT revenue)
        expected_retained = expected_net_income * manufacturer.retention_ratio
        expected_dividends = expected_net_income * (1 - manufacturer.retention_ratio)

        # Run actual calculation
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)
        net_income = manufacturer.calculate_net_income(operating_income, 0, 0, 0)

        # Verify net income calculation
        assert (
            abs(net_income - expected_net_income) < 0.01
        ), f"Net income {net_income} != expected {expected_net_income}"

        # Verify retention is applied to net income, not revenue
        initial_equity = manufacturer.equity
        manufacturer.update_balance_sheet(net_income)
        equity_increase = manufacturer.equity - initial_equity

        assert (
            abs(equity_increase - expected_retained) < 0.01
        ), f"Retained earnings {equity_increase} != expected {expected_retained}"

        # Ensure it's NOT applied to revenue
        wrong_retained = expected_revenue * manufacturer.retention_ratio
        assert (
            abs(equity_increase - wrong_retained) > 1000
        ), "Retention seems to be applied to revenue instead of net income"

    def test_all_costs_deducted_before_retention(self):
        """Verify all cost components are deducted before applying retention."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.2,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.70,
        )
        manufacturer = WidgetManufacturer(config)

        # Calculate base values
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # Add various costs
        insurance_premiums = 100_000
        insurance_losses = 50_000
        collateral_costs = 75_000

        # Calculate expected net income with all costs
        total_insurance = insurance_premiums + insurance_losses
        income_before_tax = operating_income - collateral_costs - total_insurance
        expected_taxes = max(0, income_before_tax * manufacturer.tax_rate)
        expected_net_income = income_before_tax - expected_taxes

        # Run actual calculation
        actual_net_income = manufacturer.calculate_net_income(
            operating_income, collateral_costs, insurance_premiums, insurance_losses
        )

        # Verify all costs were deducted
        assert (
            abs(actual_net_income - expected_net_income) < 0.01
        ), f"Net income {actual_net_income} != expected {expected_net_income}"

        # Verify retention applies to this fully-costed net income
        expected_retained = expected_net_income * manufacturer.retention_ratio
        initial_equity = manufacturer.equity
        manufacturer.update_balance_sheet(actual_net_income)
        equity_increase = manufacturer.equity - initial_equity

        assert (
            abs(equity_increase - expected_retained) < 0.01
        ), f"Retained earnings {equity_increase} != expected {expected_retained}"

    def test_profit_waterfall_calculation(self):
        """Validate the complete profit waterfall from revenue to retained earnings."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.2,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.70,
        )
        manufacturer = WidgetManufacturer(config)

        # Track each step of the waterfall
        waterfall = {}

        # Step 1: Revenue
        waterfall["revenue"] = manufacturer.calculate_revenue()
        assert waterfall["revenue"] == manufacturer.assets * manufacturer.asset_turnover_ratio

        # Step 2: Operating Income (after operating costs)
        waterfall["operating_income"] = manufacturer.calculate_operating_income(
            waterfall["revenue"]
        )
        waterfall["operating_costs"] = waterfall["revenue"] - waterfall["operating_income"]

        # Step 3: Add other costs
        waterfall["insurance_premiums"] = 80_000
        waterfall["insurance_losses"] = 40_000
        waterfall["collateral_costs"] = 60_000

        # Step 4: Income before tax
        waterfall["income_before_tax"] = (
            waterfall["operating_income"]
            - waterfall["collateral_costs"]
            - waterfall["insurance_premiums"]
            - waterfall["insurance_losses"]
        )

        # Step 5: Taxes
        waterfall["taxes"] = max(0, waterfall["income_before_tax"] * manufacturer.tax_rate)

        # Step 6: Net income
        waterfall["net_income"] = waterfall["income_before_tax"] - waterfall["taxes"]

        # Step 7: Retained earnings and dividends
        waterfall["retained_earnings"] = waterfall["net_income"] * manufacturer.retention_ratio
        waterfall["dividends"] = waterfall["net_income"] * (1 - manufacturer.retention_ratio)

        # Verify actual calculation matches waterfall
        actual_net_income = manufacturer.calculate_net_income(
            waterfall["operating_income"],
            waterfall["collateral_costs"],
            waterfall["insurance_premiums"],
            waterfall["insurance_losses"],
        )

        assert (
            abs(actual_net_income - waterfall["net_income"]) < 0.01
        ), f"Waterfall net income {waterfall['net_income']} != actual {actual_net_income}"

        # Verify balance sheet update
        initial_equity = manufacturer.equity
        manufacturer.update_balance_sheet(actual_net_income)
        equity_increase = manufacturer.equity - initial_equity

        assert (
            abs(equity_increase - waterfall["retained_earnings"]) < 0.01
        ), f"Retained {equity_increase} != waterfall {waterfall['retained_earnings']}"

        # Verify the complete chain
        assert (
            waterfall["retained_earnings"] < waterfall["net_income"]
        ), "Retained earnings should be less than net income"
        assert (
            waterfall["net_income"] < waterfall["operating_income"]
        ), "Net income should be less than operating income after costs"
        assert (
            waterfall["operating_income"] < waterfall["revenue"]
        ), "Operating income should be less than revenue"

    def test_negative_income_retention(self):
        """Test retention behavior with losses."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.2,
            base_operating_margin=0.02,  # Low margin
            tax_rate=0.25,
            retention_ratio=0.70,
        )
        manufacturer = WidgetManufacturer(config)

        # Calculate base values
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # Add large costs to create a loss
        large_insurance_loss = 500_000
        collateral_costs = 100_000

        # Calculate expected loss
        income_before_tax = operating_income - collateral_costs - large_insurance_loss
        assert income_before_tax < 0, "Should create a loss scenario"

        # No taxes on losses
        expected_taxes = 0
        expected_net_loss = income_before_tax  # Negative value

        # Loss retention (reduces equity)
        expected_retained_loss = expected_net_loss * manufacturer.retention_ratio

        # Run actual calculation
        actual_net_income = manufacturer.calculate_net_income(
            operating_income, collateral_costs, 0, large_insurance_loss
        )

        assert actual_net_income < 0, "Should result in a net loss"
        assert abs(actual_net_income - expected_net_loss) < 0.01

        # Verify equity reduction
        initial_equity = manufacturer.equity
        manufacturer.update_balance_sheet(actual_net_income)
        equity_change = manufacturer.equity - initial_equity

        # Equity should decrease by retained portion of loss
        assert equity_change < 0, "Equity should decrease with losses"
        assert abs(equity_change - expected_retained_loss) < 0.01

    def test_retention_with_insurance_in_operating_income(self):
        """Test that insurance costs in operating income aren't double-counted."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.2,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.70,
        )
        manufacturer = WidgetManufacturer(config)

        # Set period insurance costs (these affect operating_income)
        manufacturer.period_insurance_premiums = 100_000
        manufacturer.period_insurance_losses = 50_000

        # Calculate with insurance already in operating income
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # These should already be reflected in operating_income
        # So we pass 0 for insurance costs to avoid double-counting
        collateral_costs = 75_000
        net_income = manufacturer.calculate_net_income(
            operating_income,
            collateral_costs,
            0,  # Don't double-count premiums
            0,  # Don't double-count losses
        )

        # Verify retention applies correctly
        expected_retained = net_income * manufacturer.retention_ratio
        initial_equity = manufacturer.equity
        manufacturer.update_balance_sheet(net_income)
        equity_increase = manufacturer.equity - initial_equity

        assert abs(equity_increase - expected_retained) < 0.01

    def test_period_cost_accumulation(self):
        """Verify period costs are properly accumulated and don't affect retention ratio."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.2,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.70,
        )
        manufacturer = WidgetManufacturer(config)

        # Multiple operations in a period
        for i in range(3):
            # Each operation adds to period costs
            manufacturer.period_insurance_premiums += 30_000
            manufacturer.period_insurance_losses += 10_000

        # Final calculation should use accumulated costs
        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # Verify total costs are reflected
        assert manufacturer.period_insurance_premiums == 90_000
        assert manufacturer.period_insurance_losses == 30_000

        # Calculate net income with all accumulated costs
        collateral_costs = 50_000
        net_income = manufacturer.calculate_net_income(
            operating_income,
            collateral_costs,
            0,  # Already in operating_income
            0,  # Already in operating_income
        )

        # Retention should apply to final net income after all costs
        expected_retained = net_income * manufacturer.retention_ratio
        initial_equity = manufacturer.equity
        manufacturer.update_balance_sheet(net_income)
        equity_increase = manufacturer.equity - initial_equity

        assert abs(equity_increase - expected_retained) < 0.01

        # Reset for next period
        manufacturer.reset_period_insurance_costs()
        assert manufacturer.period_insurance_premiums == 0
        assert manufacturer.period_insurance_losses == 0

    def test_retention_ratio_bounds(self):
        """Test edge cases for retention ratio values."""
        # Test 0% retention (all dividends)
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.2,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.0,  # No retention
        )
        manufacturer = WidgetManufacturer(config)

        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)
        net_income = manufacturer.calculate_net_income(operating_income, 0, 0, 0)

        assert net_income > 0, "Should have positive income"

        initial_equity = manufacturer.equity
        manufacturer.update_balance_sheet(net_income)
        equity_change = manufacturer.equity - initial_equity

        assert abs(equity_change) < 0.01, "No equity change with 0% retention"

        # Test 100% retention (no dividends)
        config2 = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.2,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=1.0,  # Full retention
        )
        manufacturer2 = WidgetManufacturer(config2)

        revenue2 = manufacturer2.calculate_revenue()
        operating_income2 = manufacturer2.calculate_operating_income(revenue2)
        net_income2 = manufacturer2.calculate_net_income(operating_income2, 0, 0, 0)

        initial_equity2 = manufacturer2.equity
        manufacturer2.update_balance_sheet(net_income2)
        equity_change2 = manufacturer2.equity - initial_equity2

        assert (
            abs(equity_change2 - net_income2) < 0.01
        ), "Full net income retained with 100% retention"


class TestRetentionValidation:
    """Tests for validation and error handling of retention calculations."""

    def test_validate_retention_not_applied_to_gross_values(self):
        """Ensure retention is never mistakenly applied to gross values."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.2,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.70,
        )
        manufacturer = WidgetManufacturer(config)

        revenue = manufacturer.calculate_revenue()
        operating_income = manufacturer.calculate_operating_income(revenue)

        # Add significant costs
        total_costs = 300_000
        net_income = manufacturer.calculate_net_income(operating_income, total_costs, 0, 0)

        # Calculate what retention would be if mistakenly applied to wrong values
        wrong_applications = {
            "revenue": revenue * manufacturer.retention_ratio,
            "operating_income": operating_income * manufacturer.retention_ratio,
            "income_before_tax": (operating_income - total_costs) * manufacturer.retention_ratio,
        }

        # Correct application
        correct_retained = net_income * manufacturer.retention_ratio

        initial_equity = manufacturer.equity
        manufacturer.update_balance_sheet(net_income)
        actual_retained = manufacturer.equity - initial_equity

        # Verify it matches correct calculation
        assert abs(actual_retained - correct_retained) < 0.01

        # Verify it doesn't match any wrong calculation
        for name, wrong_value in wrong_applications.items():
            assert (
                abs(actual_retained - wrong_value) > 100
            ), f"Retention seems to be applied to {name} instead of net income"

    def test_retention_calculation_consistency(self):
        """Test that retention calculation is consistent across multiple periods."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.2,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.70,
        )
        manufacturer = WidgetManufacturer(config)

        retained_ratios = []

        for period in range(5):
            revenue = manufacturer.calculate_revenue()
            operating_income = manufacturer.calculate_operating_income(revenue)

            # Varying costs per period
            collateral_costs = 50_000 + period * 10_000
            net_income = manufacturer.calculate_net_income(operating_income, collateral_costs, 0, 0)

            if net_income > 0:  # Only check ratio for profitable periods
                initial_equity = manufacturer.equity
                manufacturer.update_balance_sheet(net_income)
                actual_retained = manufacturer.equity - initial_equity

                actual_ratio = actual_retained / net_income
                retained_ratios.append(actual_ratio)

                # Each period should have the same retention ratio
                assert (
                    abs(actual_ratio - manufacturer.retention_ratio) < 0.001
                ), f"Period {period}: retention ratio {actual_ratio} != {manufacturer.retention_ratio}"

        # All periods should have consistent ratio
        assert all(abs(r - manufacturer.retention_ratio) < 0.001 for r in retained_ratios)
