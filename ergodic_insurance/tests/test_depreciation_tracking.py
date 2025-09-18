"""Test depreciation and amortization tracking functionality."""

import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.config_v2 import DepreciationConfig
from ergodic_insurance.manufacturer import WidgetManufacturer


class TestDepreciationTracking:
    """Test depreciation and amortization calculations."""

    @pytest.fixture
    def manufacturer(self):
        """Create a manufacturer with standard configuration."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
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
        # PP&E allocation now depends on operating margin
        # < 10% margin: 30% PP&E
        # 10-15% margin: 50% PP&E
        # > 15% margin: 70% PP&E
        margin = manufacturer.config.base_operating_margin
        if margin < 0.10:
            expected_ppe_ratio = 0.3
        elif margin < 0.15:
            expected_ppe_ratio = 0.5
        else:
            expected_ppe_ratio = 0.7

        expected_ppe = manufacturer.config.initial_assets * expected_ppe_ratio
        assert manufacturer.gross_ppe == expected_ppe
        assert manufacturer.accumulated_depreciation == 0
        assert manufacturer.net_ppe == expected_ppe

        # Cash should be the remainder
        expected_cash = manufacturer.config.initial_assets - expected_ppe
        assert manufacturer.cash == expected_cash

    def test_straight_line_depreciation(self, manufacturer):
        """Test straight-line depreciation calculation."""
        useful_life = 10  # years
        expected_annual_depreciation = manufacturer.gross_ppe / useful_life

        # Record one year of depreciation
        depreciation = manufacturer.record_depreciation(useful_life_years=useful_life)

        assert depreciation == pytest.approx(expected_annual_depreciation, rel=0.01)
        assert manufacturer.accumulated_depreciation == pytest.approx(
            expected_annual_depreciation, rel=0.01
        )
        assert manufacturer.net_ppe == pytest.approx(
            manufacturer.gross_ppe - expected_annual_depreciation, rel=0.01
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

        assert manufacturer.accumulated_depreciation == pytest.approx(
            annual_depreciation * 5, rel=0.01
        )
        assert manufacturer.net_ppe == pytest.approx(
            manufacturer.gross_ppe - (annual_depreciation * 5), rel=0.01
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
        assert monthly_depreciation_total == pytest.approx(annual_depreciation, rel=0.01)

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

        assert amortized == pytest.approx(expected_monthly_amortization, rel=0.01)
        assert manufacturer.prepaid_insurance == pytest.approx(
            premium - expected_monthly_amortization, rel=0.01
        )
        assert manufacturer.period_insurance_premiums == pytest.approx(
            expected_monthly_amortization, rel=0.01
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

        assert total_amortized == pytest.approx(premium, rel=0.01)
        assert manufacturer.prepaid_insurance == pytest.approx(0, abs=0.01)

    def test_prepaid_insurance_cannot_over_amortize(self, manufacturer):
        """Test that prepaid insurance cannot be amortized below zero."""
        premium = 100_000
        manufacturer.record_prepaid_insurance(premium)

        # Try to amortize for 24 months at once (should stop at balance)
        total_amortized = manufacturer.amortize_prepaid_insurance(months=24)

        assert total_amortized == premium
        assert manufacturer.prepaid_insurance == 0

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
        assert manufacturer.prepaid_insurance == pytest.approx(premium / 2, rel=0.01)

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
        assert manufacturer.gross_ppe == manufacturer.config.initial_assets * expected_ppe_ratio

    def test_different_useful_lives(self, manufacturer):
        """Test depreciation with different useful life assumptions."""
        useful_lives = [5, 10, 20]
        expected_depreciations = [manufacturer.gross_ppe / life for life in useful_lives]

        for life, expected in zip(useful_lives, expected_depreciations):
            manufacturer.reset()
            depreciation = manufacturer.record_depreciation(useful_life_years=life)
            assert depreciation == pytest.approx(expected, rel=0.01)
