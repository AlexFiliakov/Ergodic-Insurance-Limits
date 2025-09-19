"""Test working capital component calculations using DSO/DIO/DPO ratios."""

import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.config_v2 import WorkingCapitalRatiosConfig
from ergodic_insurance.manufacturer import WidgetManufacturer


class TestWorkingCapitalCalculation:
    """Test working capital calculations with industry-standard ratios."""

    @pytest.fixture
    def manufacturer(self):
        """Create a manufacturer with standard configuration."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        return WidgetManufacturer(config)

    @pytest.fixture
    def wc_ratios_config(self):
        """Create working capital ratios configuration."""
        return WorkingCapitalRatiosConfig(
            days_sales_outstanding=45, days_inventory_outstanding=60, days_payable_outstanding=30
        )

    def test_accounts_receivable_calculation(self, manufacturer):
        """Test accounts receivable calculation using DSO."""
        revenue = 10_000_000
        dso = 45  # 45 days sales outstanding

        components = manufacturer.calculate_working_capital_components(revenue, dso=dso)

        # AR = Revenue * (DSO / 365)
        expected_ar = revenue * (dso / 365)
        assert components["accounts_receivable"] == pytest.approx(expected_ar, rel=0.01)
        assert manufacturer.accounts_receivable == pytest.approx(expected_ar, rel=0.01)

    def test_inventory_calculation(self, manufacturer):
        """Test inventory calculation using DIO."""
        revenue = 10_000_000
        dio = 60  # 60 days inventory outstanding

        # COGS = Revenue * (1 - operating_margin)
        cogs = revenue * (1 - manufacturer.base_operating_margin)

        components = manufacturer.calculate_working_capital_components(revenue, dio=dio)

        # Inventory = COGS * (DIO / 365)
        expected_inventory = cogs * (dio / 365)
        assert components["inventory"] == pytest.approx(expected_inventory, rel=0.01)
        assert manufacturer.inventory == pytest.approx(expected_inventory, rel=0.01)

    def test_accounts_payable_calculation(self, manufacturer):
        """Test accounts payable calculation using DPO."""
        revenue = 10_000_000
        dpo = 30  # 30 days payable outstanding

        # COGS = Revenue * (1 - operating_margin)
        cogs = revenue * (1 - manufacturer.base_operating_margin)

        components = manufacturer.calculate_working_capital_components(revenue, dpo=dpo)

        # AP = COGS * (DPO / 365)
        expected_ap = cogs * (dpo / 365)
        assert components["accounts_payable"] == pytest.approx(expected_ap, rel=0.01)
        assert manufacturer.accounts_payable == pytest.approx(expected_ap, rel=0.01)

    def test_net_working_capital_calculation(self, manufacturer):
        """Test net working capital calculation."""
        revenue = 10_000_000
        dso, dio, dpo = 45, 60, 30

        components = manufacturer.calculate_working_capital_components(
            revenue, dso=dso, dio=dio, dpo=dpo
        )

        # Net WC = AR + Inventory - AP
        expected_net_wc = (
            components["accounts_receivable"]
            + components["inventory"]
            - components["accounts_payable"]
        )
        assert components["net_working_capital"] == pytest.approx(expected_net_wc, rel=0.01)

    def test_cash_conversion_cycle(self, manufacturer):
        """Test cash conversion cycle calculation."""
        revenue = 10_000_000
        dso, dio, dpo = 45, 60, 30

        components = manufacturer.calculate_working_capital_components(
            revenue, dso=dso, dio=dio, dpo=dpo
        )

        # CCC = DSO + DIO - DPO
        expected_ccc = dso + dio - dpo
        assert components["cash_conversion_cycle"] == expected_ccc

    def test_working_capital_with_different_margins(self, manufacturer):
        """Test that working capital adjusts with operating margin."""
        revenue = 10_000_000
        dio = 60
        dpo = 30

        # Test with different operating margins
        margins = [0.05, 0.10, 0.15]
        inventories = []
        payables = []

        for margin in margins:
            manufacturer.base_operating_margin = margin
            components = manufacturer.calculate_working_capital_components(
                revenue, dio=dio, dpo=dpo
            )
            inventories.append(components["inventory"])
            payables.append(components["accounts_payable"])

        # Higher margin means lower COGS, thus lower inventory and AP
        assert inventories[0] > inventories[1] > inventories[2]
        assert payables[0] > payables[1] > payables[2]

    def test_working_capital_integration_with_step(self, manufacturer):
        """Test that working capital is calculated during step() method."""
        # Initial state should have zero working capital components
        assert manufacturer.accounts_receivable == 0
        assert manufacturer.inventory == 0
        assert manufacturer.accounts_payable == 0

        # Run a step
        metrics = manufacturer.step()

        # After step, working capital components should be calculated
        assert manufacturer.accounts_receivable > 0
        assert manufacturer.inventory > 0
        assert manufacturer.accounts_payable > 0

        # Components should be in metrics
        assert metrics["accounts_receivable"] > 0
        assert metrics["inventory"] > 0
        assert metrics["accounts_payable"] > 0

    def test_working_capital_persistence_across_steps(self, manufacturer):
        """Test that working capital components update correctly across multiple steps."""
        wc_components_history = []

        # Run multiple steps
        for _ in range(3):
            metrics = manufacturer.step()
            wc_components_history.append(
                {
                    "ar": metrics["accounts_receivable"],
                    "inv": metrics["inventory"],
                    "ap": metrics["accounts_payable"],
                }
            )

        # Components should exist and potentially change based on revenue
        for components in wc_components_history:
            assert components["ar"] > 0
            assert components["inv"] > 0
            assert components["ap"] > 0

    def test_working_capital_ratios_validation(self):
        """Test WorkingCapitalRatiosConfig validation."""
        # Valid configuration
        config = WorkingCapitalRatiosConfig(
            days_sales_outstanding=45, days_inventory_outstanding=60, days_payable_outstanding=30
        )
        assert config.days_sales_outstanding == 45
        assert config.days_inventory_outstanding == 60
        assert config.days_payable_outstanding == 30

        # Edge cases
        config_zero = WorkingCapitalRatiosConfig(
            days_sales_outstanding=0, days_inventory_outstanding=0, days_payable_outstanding=0
        )
        assert config_zero.days_sales_outstanding == 0

        # Maximum values
        config_max = WorkingCapitalRatiosConfig(
            days_sales_outstanding=365, days_inventory_outstanding=365, days_payable_outstanding=365
        )
        assert config_max.days_sales_outstanding == 365

    def test_cash_conversion_cycle_warnings(self, capsys):
        """Test that appropriate warnings are issued for unusual cash conversion cycles."""
        # Negative cash conversion cycle (good but unusual)
        config_negative = WorkingCapitalRatiosConfig(
            days_sales_outstanding=30, days_inventory_outstanding=20, days_payable_outstanding=60
        )
        captured = capsys.readouterr()
        assert "Negative cash conversion cycle" in captured.out

        # Very long cash conversion cycle (problematic)
        config_long = WorkingCapitalRatiosConfig(
            days_sales_outstanding=120, days_inventory_outstanding=150, days_payable_outstanding=30
        )
        captured = capsys.readouterr()
        assert "Very long cash conversion cycle" in captured.out

    def test_working_capital_impact_on_cash(self, manufacturer):
        """Test that working capital affects cash position and balance sheet balances."""
        initial_cash = manufacturer.cash

        # Calculate working capital components
        revenue = manufacturer.calculate_revenue()
        manufacturer.calculate_working_capital_components(revenue)

        # Run a step to update cash
        manufacturer.step()

        # Verify that the accounting equation holds: Assets = Liabilities + Equity
        assert manufacturer.total_assets == pytest.approx(
            manufacturer.total_liabilities + manufacturer.equity, rel=0.01
        ), "Accounting equation should balance: Assets = Liabilities + Equity"

        # Verify that cash is correctly calculated as part of total assets
        # Total Assets = Cash + AR + Inventory + Prepaid + Net PPE + Restricted
        expected_cash = (
            manufacturer.total_assets
            - manufacturer.accounts_receivable
            - manufacturer.inventory
            - manufacturer.prepaid_insurance
            - manufacturer.net_ppe
            - manufacturer.restricted_assets
        )
        assert manufacturer.cash == pytest.approx(expected_cash, rel=0.01)

    def test_working_capital_with_zero_revenue(self, manufacturer):
        """Test working capital calculation with zero revenue."""
        components = manufacturer.calculate_working_capital_components(revenue=0)

        # All components should be zero with zero revenue
        assert components["accounts_receivable"] == 0
        assert components["inventory"] == 0
        assert components["accounts_payable"] == 0
        assert components["net_working_capital"] == 0

    def test_working_capital_monthly_vs_annual(self, manufacturer):
        """Test working capital calculation consistency between monthly and annual steps."""
        # Annual step
        manufacturer.reset()
        annual_metrics = manufacturer.step(time_resolution="annual")
        annual_ar = annual_metrics["accounts_receivable"]

        # Monthly steps (12 months)
        manufacturer.reset()
        for month in range(12):
            monthly_metrics = manufacturer.step(time_resolution="monthly")

        # Final AR should be similar (based on annualized revenue)
        final_monthly_ar = monthly_metrics["accounts_receivable"]

        # Should be reasonably close
        assert annual_ar == pytest.approx(final_monthly_ar, rel=0.1)
