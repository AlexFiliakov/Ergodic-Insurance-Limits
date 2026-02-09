"""Test depreciation and amortization tracking functionality."""

import pytest

from ergodic_insurance.config import DepreciationConfig, ManufacturerConfig
from ergodic_insurance.decimal_utils import to_decimal
from ergodic_insurance.manufacturer import WidgetManufacturer


class TestDepreciationTracking:
    """Test depreciation and amortization calculations."""

    @pytest.fixture
    def manufacturer(self):
        """Create a manufacturer with standard configuration.

        Uses capex_to_depreciation_ratio=0.0 to isolate depreciation behavior
        from capex effects (capex is tested separately in test_capex.py).
        """
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
            capex_to_depreciation_ratio=0.0,
        )
        return WidgetManufacturer(config)

    @pytest.fixture
    def depreciation_config(self):
        """Create depreciation configuration."""
        return DepreciationConfig(
            ppe_useful_life_years=10,
            prepaid_insurance_amortization_months=12,
            initial_accumulated_depreciation=0,
        )

    def test_initial_ppe_allocation(self, manufacturer):
        """Test that initial assets are properly allocated to PP&E."""
        # PPE ratio is now configurable, with defaults based on operating margin
        expected_ppe_ratio = manufacturer.config.ppe_ratio
        expected_ppe = to_decimal(manufacturer.config.initial_assets * expected_ppe_ratio)
        assert manufacturer.gross_ppe == expected_ppe
        assert manufacturer.accumulated_depreciation == 0
        assert manufacturer.net_ppe == expected_ppe

        # Cash should be the remainder minus working capital assets (AR + inventory)
        # Working capital is initialized to steady-state to avoid warm-up distortion
        working_capital_assets = manufacturer.accounts_receivable + manufacturer.inventory
        expected_cash = (
            to_decimal(manufacturer.config.initial_assets) - expected_ppe - working_capital_assets
        )
        assert float(manufacturer.cash) == pytest.approx(float(expected_cash), rel=0.01)

    def test_custom_ppe_ratio(self):
        """Test that custom PPE ratio overrides the default."""
        # Create manufacturer with custom PPE ratio
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,  # Would normally default to 0.3 ratio
            tax_rate=0.25,
            retention_ratio=0.7,
            ppe_ratio=0.6,  # Custom PPE ratio
        )
        manufacturer = WidgetManufacturer(config)

        # Verify custom ratio is used
        assert config.ppe_ratio == 0.6
        expected_ppe = to_decimal(config.initial_assets * 0.6)
        assert manufacturer.gross_ppe == expected_ppe
        # Cash = remaining assets minus working capital (AR + inventory initialized to steady-state)
        working_capital_assets = manufacturer.accounts_receivable + manufacturer.inventory
        expected_cash = to_decimal(config.initial_assets * 0.4) - working_capital_assets
        assert float(manufacturer.cash) == pytest.approx(float(expected_cash), rel=0.01)

    def test_default_ppe_ratio_based_on_margin(self):
        """Test that default PPE ratio is correctly set based on operating margin."""
        # Test low margin (<10%)
        config_low = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.05,  # 5% margin
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        assert config_low.ppe_ratio == 0.3

        # Test medium margin (10-15%)
        config_med = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.12,  # 12% margin
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        assert config_med.ppe_ratio == 0.5

        # Test high margin (>15%)
        config_high = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.20,  # 20% margin
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        assert config_high.ppe_ratio == 0.7

    def test_straight_line_depreciation(self, manufacturer):
        """Test straight-line depreciation calculation."""
        useful_life = 10  # years
        expected_annual_depreciation = manufacturer.gross_ppe / useful_life

        # Record one year of depreciation
        depreciation = manufacturer.record_depreciation(useful_life_years=useful_life)

        assert float(depreciation) == pytest.approx(float(expected_annual_depreciation), rel=0.01)
        assert float(manufacturer.accumulated_depreciation) == pytest.approx(
            float(expected_annual_depreciation), rel=0.01
        )
        assert float(manufacturer.net_ppe) == pytest.approx(
            float(manufacturer.gross_ppe - expected_annual_depreciation), rel=0.01
        )

    def test_depreciation_accumulation_over_time(self, manufacturer):
        """Test that depreciation accumulates correctly over multiple periods."""
        useful_life = 10
        annual_depreciation = manufacturer.gross_ppe / useful_life

        # Record 5 years of depreciation
        total_depreciation = 0
        for year in range(5):
            depreciation = manufacturer.record_depreciation(useful_life_years=useful_life)
            total_depreciation += depreciation

        assert float(manufacturer.accumulated_depreciation) == pytest.approx(
            float(annual_depreciation * 5), rel=0.01
        )
        assert float(manufacturer.net_ppe) == pytest.approx(
            float(manufacturer.gross_ppe - (annual_depreciation * 5)), rel=0.01
        )

    def test_depreciation_cannot_exceed_asset_value(self, manufacturer):
        """Test that accumulated depreciation cannot exceed gross PP&E."""
        useful_life = 2  # Short useful life for testing

        # Depreciate for more years than useful life
        for year in range(5):
            manufacturer.record_depreciation(useful_life_years=useful_life)

        # Accumulated depreciation should not exceed gross PP&E
        assert manufacturer.accumulated_depreciation <= manufacturer.gross_ppe
        assert manufacturer.net_ppe >= 0

    def test_depreciation_in_annual_step(self, manufacturer):
        """Test that depreciation is recorded during annual steps."""
        initial_net_ppe = manufacturer.net_ppe

        # Run annual step
        metrics = manufacturer.step(time_resolution="annual")

        # Depreciation should have been recorded
        assert manufacturer.accumulated_depreciation > 0
        assert manufacturer.net_ppe < initial_net_ppe
        assert metrics["accumulated_depreciation"] > 0
        assert metrics["net_ppe"] < initial_net_ppe

    def test_depreciation_in_monthly_steps(self, manufacturer):
        """Test that depreciation is properly scaled for monthly steps."""
        # Run 12 monthly steps
        monthly_depreciation_total = 0
        for month in range(12):
            initial_accumulated = manufacturer.accumulated_depreciation
            metrics = manufacturer.step(time_resolution="monthly")
            monthly_depreciation = manufacturer.accumulated_depreciation - initial_accumulated
            monthly_depreciation_total += monthly_depreciation

        # Reset and run one annual step
        manufacturer.reset()
        annual_metrics = manufacturer.step(time_resolution="annual")
        annual_depreciation = manufacturer.accumulated_depreciation

        # Total monthly depreciation should approximately equal annual
        assert float(monthly_depreciation_total) == pytest.approx(
            float(annual_depreciation), rel=0.01
        )

    def test_prepaid_insurance_recording(self, manufacturer):
        """Test recording of prepaid insurance premiums."""
        initial_cash = manufacturer.cash
        premium = 1_200_000

        # Record prepaid insurance
        manufacturer.record_prepaid_insurance(premium)

        assert manufacturer.prepaid_insurance == premium
        assert manufacturer.cash == initial_cash - premium

    def test_prepaid_insurance_amortization(self, manufacturer):
        """Test amortization of prepaid insurance over time."""
        premium = 1_200_000
        manufacturer.record_prepaid_insurance(premium)

        # Amortize for 1 month
        amortized = manufacturer.amortize_prepaid_insurance(months=1)
        expected_monthly_amortization = premium / 12

        assert float(amortized) == pytest.approx(float(expected_monthly_amortization), rel=0.01)
        assert float(manufacturer.prepaid_insurance) == pytest.approx(
            float(premium - expected_monthly_amortization), rel=0.01
        )
        assert float(manufacturer.period_insurance_premiums) == pytest.approx(
            float(expected_monthly_amortization), rel=0.01
        )

    def test_prepaid_insurance_full_amortization(self, manufacturer):
        """Test that prepaid insurance fully amortizes over 12 months."""
        premium = 1_200_000
        manufacturer.record_prepaid_insurance(premium)

        # Amortize for 12 months
        total_amortized = 0
        for month in range(12):
            amortized = manufacturer.amortize_prepaid_insurance(months=1)
            total_amortized += amortized

        assert float(total_amortized) == pytest.approx(float(premium), rel=0.01)
        assert float(manufacturer.prepaid_insurance) == pytest.approx(0, abs=0.01)

    def test_prepaid_insurance_cannot_over_amortize(self, manufacturer):
        """Test that prepaid insurance cannot be amortized below zero."""
        premium = 100_000
        manufacturer.record_prepaid_insurance(premium)

        # Try to amortize for 24 months at once (should stop at balance)
        total_amortized = manufacturer.amortize_prepaid_insurance(months=24)

        assert float(total_amortized) == pytest.approx(float(premium), rel=1e-9)
        assert float(manufacturer.prepaid_insurance) == pytest.approx(0, abs=1e-9)

        # Further amortization should return 0
        additional_amortization = manufacturer.amortize_prepaid_insurance(months=1)
        assert additional_amortization == 0

    def test_prepaid_insurance_amortization_in_monthly_steps(self, manufacturer):
        """Test that prepaid insurance is amortized during monthly steps."""
        premium = 600_000
        manufacturer.record_prepaid_insurance(premium)

        # Run 6 monthly steps
        for month in range(6):
            metrics = manufacturer.step(time_resolution="monthly")

        # Half of the premium should be amortized
        assert float(manufacturer.prepaid_insurance) == pytest.approx(float(premium / 2), rel=0.01)

    def test_depreciation_config_properties(self, depreciation_config):
        """Test DepreciationConfig calculated properties."""
        # Test annual depreciation rate
        expected_rate = 1.0 / depreciation_config.ppe_useful_life_years
        assert depreciation_config.annual_depreciation_rate == expected_rate

        # Test monthly amortization rate
        expected_monthly_rate = 1.0 / depreciation_config.prepaid_insurance_amortization_months
        assert depreciation_config.monthly_insurance_amortization_rate == expected_monthly_rate

    def test_depreciation_impact_on_metrics(self, manufacturer):
        """Test that depreciation affects financial metrics correctly."""
        # Run multiple years
        for _ in range(3):
            metrics = manufacturer.step()

        # Verify depreciation is tracked in metrics
        assert metrics["accumulated_depreciation"] > 0
        assert metrics["net_ppe"] < metrics["gross_ppe"]
        assert metrics["net_ppe"] == metrics["gross_ppe"] - metrics["accumulated_depreciation"]

    def test_reset_clears_depreciation(self, manufacturer):
        """Test that reset() properly clears depreciation and prepaid insurance."""
        # Accumulate some depreciation and prepaid insurance
        manufacturer.record_prepaid_insurance(500_000)
        manufacturer.record_depreciation(useful_life_years=10)

        assert manufacturer.accumulated_depreciation > 0
        assert manufacturer.prepaid_insurance > 0

        # Reset
        manufacturer.reset()

        # Should be back to initial state
        assert manufacturer.accumulated_depreciation == 0
        assert manufacturer.prepaid_insurance == 0
        # PP&E allocation depends on margin - with 8% margin, it's 30%
        margin = manufacturer.config.base_operating_margin
        if margin < 0.10:
            expected_ppe_ratio = 0.3
        elif margin < 0.15:
            expected_ppe_ratio = 0.5
        else:
            expected_ppe_ratio = 0.7
        assert manufacturer.gross_ppe == to_decimal(
            manufacturer.config.initial_assets * expected_ppe_ratio
        )

    def test_different_useful_lives(self, manufacturer):
        """Test depreciation with different useful life assumptions."""
        useful_lives = [5, 10, 20]
        expected_depreciations = [manufacturer.gross_ppe / life for life in useful_lives]

        for life, expected in zip(useful_lives, expected_depreciations):
            manufacturer.reset()
            depreciation = manufacturer.record_depreciation(useful_life_years=life)
            assert float(depreciation) == pytest.approx(float(expected), rel=0.01)

    def test_depreciation_reduces_equity(self):
        """Regression test: depreciation must reduce equity via total_assets.

        Depreciation is a non-cash expense recorded as a credit to
        accumulated_depreciation (a contra-asset). This reduces net PP&E,
        which reduces total_assets, which reduces equity (Assets - Liabilities).

        Before the fix for Issue #285, accumulated_depreciation returned a
        negative value (raw ledger balance), causing total_assets to
        INCREASE instead of decrease when depreciation was recorded. This
        made equity rise even when depreciation exceeded retained earnings.

        Regression for GitHub Issue #286.
        """
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.1,
            tax_rate=0.25,
            retention_ratio=0.5,
        )
        manufacturer = WidgetManufacturer(config)

        initial_equity = manufacturer.equity
        initial_total_assets = manufacturer.total_assets
        initial_net_ppe = manufacturer.net_ppe

        # Record depreciation directly (outside step) to isolate its effect
        depreciation_expense = manufacturer.record_depreciation(useful_life_years=10)
        assert depreciation_expense > 0, "Should record positive depreciation"

        # Accumulated depreciation must be positive (contra-asset convention)
        assert manufacturer.accumulated_depreciation > 0, (
            "accumulated_depreciation must return a positive value "
            "(Issue #285 fix: contra-asset stored as negative in ledger)"
        )

        # Net PP&E must decrease by the depreciation amount
        assert float(manufacturer.net_ppe) == pytest.approx(
            float(initial_net_ppe - depreciation_expense), rel=1e-9
        )

        # Total assets must decrease (depreciation reduces net PP&E)
        assert (
            manufacturer.total_assets < initial_total_assets
        ), "Depreciation must reduce total_assets via accumulated_depreciation"

        # Equity must decrease (Assets - Liabilities, liabilities unchanged)
        assert manufacturer.equity < initial_equity, (
            "Depreciation must reduce equity since it reduces total_assets "
            "while liabilities remain unchanged"
        )

        # Equity decrease should equal the depreciation amount
        equity_change = manufacturer.equity - initial_equity
        assert float(equity_change) == pytest.approx(float(-depreciation_expense), rel=1e-9)

    def test_depreciation_reduces_equity_through_step(self):
        """Test that step() produces equity decrease when depreciation dominates.

        With a configuration where depreciation exceeds retained earnings,
        equity should decrease even when net income is positive.

        Regression for GitHub Issue #286.
        """
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.1,
            tax_rate=0.25,
            retention_ratio=0.5,
        )
        manufacturer = WidgetManufacturer(config)

        initial_equity = manufacturer.equity
        metrics = manufacturer.step(letter_of_credit_rate=0.015, growth_rate=0.0)

        # Net income should be positive (profitable operation)
        assert metrics["net_income"] > 0

        # But equity should decrease because depreciation (500K) + tax accrual
        # exceed retained earnings
        equity_change = manufacturer.equity - initial_equity
        assert equity_change < 0, (
            f"Equity should decrease when depreciation exceeds retained earnings, "
            f"but equity_change={equity_change}"
        )
