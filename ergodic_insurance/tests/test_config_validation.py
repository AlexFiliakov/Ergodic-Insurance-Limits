"""Tests for configuration validation logic."""

import pytest

from ergodic_insurance.config import (
    IndustryConfig,
    ManufacturingConfig,
    RetailConfig,
    ServiceConfig,
)


class TestParameterBounds:
    """Test parameter validation and bounds checking."""

    def test_valid_margin_ranges(self):
        """Test that margins are properly bounded."""
        # Valid configurations should work
        config = IndustryConfig(gross_margin=0.5, operating_expense_ratio=0.3)
        assert config.gross_margin == 0.5
        assert config.operating_expense_ratio == 0.3

        # Edge cases - exactly 0 and 1
        config_zero = IndustryConfig(gross_margin=0.0, operating_expense_ratio=0.0)
        assert config_zero.gross_margin == 0.0

        config_one = IndustryConfig(gross_margin=1.0, operating_expense_ratio=1.0)
        assert config_one.gross_margin == 1.0

    def test_invalid_margin_ranges(self):
        """Test that invalid margins are rejected."""
        # Negative gross margin
        with pytest.raises(AssertionError):
            IndustryConfig(gross_margin=-0.1)

        # Gross margin > 1
        with pytest.raises(AssertionError):
            IndustryConfig(gross_margin=1.1)

        # Negative operating expense ratio
        with pytest.raises(AssertionError):
            IndustryConfig(operating_expense_ratio=-0.05)

        # Operating expense ratio > 1
        with pytest.raises(AssertionError):
            IndustryConfig(operating_expense_ratio=1.5)

    def test_asset_ratio_validation(self):
        """Test asset composition ratio validation."""
        # Valid - sums to 1.0
        config = IndustryConfig(current_asset_ratio=0.3, ppe_ratio=0.5, intangible_ratio=0.2)
        assert config.current_asset_ratio == 0.3

        # Invalid - sums to less than 1.0
        with pytest.raises(AssertionError, match="Asset ratios must sum to 1.0"):
            IndustryConfig(
                current_asset_ratio=0.2, ppe_ratio=0.3, intangible_ratio=0.4  # Sum = 0.9
            )

        # Invalid - sums to more than 1.0
        with pytest.raises(AssertionError, match="Asset ratios must sum to 1.0"):
            IndustryConfig(
                current_asset_ratio=0.4, ppe_ratio=0.4, intangible_ratio=0.3  # Sum = 1.1
            )

    def test_working_capital_days_validation(self):
        """Test working capital days validation."""
        # Valid positive values
        config = IndustryConfig(
            days_sales_outstanding=60, days_inventory_outstanding=90, days_payables_outstanding=45
        )
        assert config.days_sales_outstanding == 60

        # Zero values should be valid (e.g., no inventory)
        config = IndustryConfig(days_inventory_outstanding=0)
        assert config.days_inventory_outstanding == 0

        # Negative values should be invalid
        with pytest.raises(AssertionError, match="Days sales outstanding must be non-negative"):
            IndustryConfig(days_sales_outstanding=-10)

        with pytest.raises(AssertionError, match="Days inventory outstanding must be non-negative"):
            IndustryConfig(days_inventory_outstanding=-5)

        with pytest.raises(AssertionError, match="Days payables outstanding must be non-negative"):
            IndustryConfig(days_payables_outstanding=-15)

    def test_depreciation_validation(self):
        """Test depreciation parameter validation."""
        # Valid useful life
        config = IndustryConfig(ppe_useful_life=15)
        assert config.ppe_useful_life == 15

        # Zero useful life should be invalid
        with pytest.raises(AssertionError, match="PPE useful life must be positive"):
            IndustryConfig(ppe_useful_life=0)

        # Negative useful life should be invalid
        with pytest.raises(AssertionError, match="PPE useful life must be positive"):
            IndustryConfig(ppe_useful_life=-5)

        # Valid depreciation methods
        config_sl = IndustryConfig(depreciation_method="straight_line")
        assert config_sl.depreciation_method == "straight_line"

        config_db = IndustryConfig(depreciation_method="declining_balance")
        assert config_db.depreciation_method == "declining_balance"

        # Invalid depreciation method
        with pytest.raises(AssertionError, match="Unknown depreciation method"):
            IndustryConfig(depreciation_method="sum_of_years")


class TestIndustrySpecificValidation:
    """Test industry-specific validation rules."""

    def test_manufacturing_constraints(self):
        """Test manufacturing-specific constraints."""
        config = ManufacturingConfig()

        # Manufacturing should have inventory
        assert config.days_inventory_outstanding > 0

        # Manufacturing should have significant PP&E
        assert config.ppe_ratio >= 0.3

    def test_service_constraints(self):
        """Test service-specific constraints."""
        config = ServiceConfig()

        # Services typically have no inventory
        assert config.days_inventory_outstanding == 0

        # Services should have lower PP&E
        assert config.ppe_ratio <= 0.3

        # Services should have higher intangibles
        assert config.intangible_ratio >= 0.15

    def test_retail_constraints(self):
        """Test retail-specific constraints."""
        config = RetailConfig()

        # Retail should have fast cash collection
        assert config.days_sales_outstanding <= 10

        # Retail should have inventory
        assert config.days_inventory_outstanding > 0

        # Retail margins should be moderate
        assert 0.2 <= config.gross_margin <= 0.4


class TestValidationErrorMessages:
    """Test that validation errors provide helpful messages."""

    def test_asset_ratio_error_message(self):
        """Test asset ratio validation error message."""
        with pytest.raises(AssertionError) as exc_info:
            IndustryConfig(current_asset_ratio=0.2, ppe_ratio=0.2, intangible_ratio=0.2)
        assert "0.6" in str(exc_info.value)  # Should show actual sum

    def test_margin_error_message(self):
        """Test margin validation error message."""
        with pytest.raises(AssertionError) as exc_info:
            IndustryConfig(gross_margin=1.5)
        assert "1.5" in str(exc_info.value)  # Should show invalid value

    def test_depreciation_error_message(self):
        """Test depreciation validation error message."""
        with pytest.raises(AssertionError) as exc_info:
            IndustryConfig(depreciation_method="invalid")
        assert "invalid" in str(exc_info.value)  # Should show invalid method


class TestConfigurationCompleteness:
    """Test that configurations are complete and consistent."""

    def test_all_industries_have_complete_configs(self):
        """Test that all industry configs have complete parameters."""
        configs = [ManufacturingConfig(), ServiceConfig(), RetailConfig()]

        for config in configs:
            # Check all required attributes exist
            assert hasattr(config, "industry_type")
            assert hasattr(config, "days_sales_outstanding")
            assert hasattr(config, "days_inventory_outstanding")
            assert hasattr(config, "days_payables_outstanding")
            assert hasattr(config, "gross_margin")
            assert hasattr(config, "operating_expense_ratio")
            assert hasattr(config, "current_asset_ratio")
            assert hasattr(config, "ppe_ratio")
            assert hasattr(config, "intangible_ratio")
            assert hasattr(config, "ppe_useful_life")
            assert hasattr(config, "depreciation_method")

            # Check calculated properties work
            assert isinstance(config.working_capital_days, (int, float))
            assert isinstance(config.operating_margin, (int, float))

    def test_reasonable_default_values(self):
        """Test that default values are reasonable for each industry."""
        manufacturing = ManufacturingConfig()
        service = ServiceConfig()
        retail = RetailConfig()

        # All operating margins should be positive
        assert manufacturing.operating_margin > 0
        assert service.operating_margin > 0
        assert retail.operating_margin > 0

        # Working capital cycles should be reasonable
        assert 0 <= manufacturing.working_capital_days <= 120
        assert -30 <= service.working_capital_days <= 60
        assert -30 <= retail.working_capital_days <= 60

        # All should have positive useful life
        assert manufacturing.ppe_useful_life > 0
        assert service.ppe_useful_life > 0
        assert retail.ppe_useful_life > 0
