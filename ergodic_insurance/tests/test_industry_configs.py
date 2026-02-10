"""Tests for industry-specific configuration classes."""

from pydantic import ValidationError
import pytest

from ergodic_insurance.config import (
    IndustryConfig,
    ManufacturerConfig,
    ManufacturingConfig,
    RetailConfig,
    ServiceConfig,
)


class TestIndustryConfig:
    """Test base IndustryConfig class."""

    def test_default_initialization(self):
        """Test creating IndustryConfig with default values."""
        config = IndustryConfig()
        assert config.industry_type == "manufacturing"
        assert config.days_sales_outstanding == 45
        assert config.days_inventory_outstanding == 60
        assert config.days_payables_outstanding == 30
        assert config.gross_margin == 0.35
        assert config.operating_expense_ratio == 0.25

    def test_asset_composition_validation(self):
        """Test that asset ratios must sum to 1.0."""
        # This should fail validation
        with pytest.raises(ValidationError, match="Asset ratios must sum to 1.0"):
            IndustryConfig(
                current_asset_ratio=0.3, ppe_ratio=0.3, intangible_ratio=0.3  # Sum is 0.9, not 1.0
            )

    def test_margin_validation(self):
        """Test margin parameter validation."""
        # Invalid gross margin
        with pytest.raises(ValidationError):
            IndustryConfig(gross_margin=1.5)

        # Invalid operating expense ratio
        with pytest.raises(ValidationError):
            IndustryConfig(operating_expense_ratio=-0.1)

    def test_working_capital_days_calculation(self):
        """Test working capital cycle calculation."""
        config = IndustryConfig(
            days_sales_outstanding=50, days_inventory_outstanding=70, days_payables_outstanding=40
        )
        assert config.working_capital_days == 80  # 50 + 70 - 40

    def test_operating_margin_calculation(self):
        """Test operating margin calculation."""
        config = IndustryConfig(gross_margin=0.40, operating_expense_ratio=0.25)
        assert abs(config.operating_margin - 0.15) < 0.0001  # 0.40 - 0.25

    def test_depreciation_validation(self):
        """Test depreciation parameter validation."""
        # Invalid useful life
        with pytest.raises(ValidationError):
            IndustryConfig(ppe_useful_life=0)

        # Invalid depreciation method
        with pytest.raises(ValidationError):
            IndustryConfig(depreciation_method="invalid_method")  # type: ignore[arg-type]


class TestManufacturingConfig:
    """Test ManufacturingConfig class."""

    def test_manufacturing_defaults(self):
        """Test manufacturing-specific default values."""
        config = ManufacturingConfig()
        assert config.industry_type == "manufacturing"
        assert config.days_inventory_outstanding == 60  # Has inventory
        assert config.ppe_ratio == 0.5  # Moderate PP&E
        assert config.gross_margin == 0.35

    def test_manufacturing_override(self):
        """Test overriding manufacturing defaults."""
        config = ManufacturingConfig(
            gross_margin=0.45,
            ppe_ratio=0.6,
            current_asset_ratio=0.3,  # Adjust so total = 1.0
            intangible_ratio=0.1,
        )
        assert config.gross_margin == 0.45
        assert config.ppe_ratio == 0.6
        # Other values should still be manufacturing defaults
        assert config.days_inventory_outstanding == 60


class TestServiceConfig:
    """Test ServiceConfig class."""

    def test_service_defaults(self):
        """Test service-specific default values."""
        config = ServiceConfig()
        assert config.industry_type == "services"
        assert config.days_inventory_outstanding == 0  # No inventory
        assert config.ppe_ratio == 0.2  # Low PP&E
        assert config.intangible_ratio == 0.2  # Higher intangibles
        assert config.gross_margin == 0.60  # Higher margins

    def test_service_working_capital(self):
        """Test service company working capital cycle."""
        config = ServiceConfig()
        # Services have no inventory, so shorter cycle
        expected_days = 30 + 0 - 20  # DSO + DIO - DPO
        assert config.working_capital_days == expected_days


class TestRetailConfig:
    """Test RetailConfig class."""

    def test_retail_defaults(self):
        """Test retail-specific default values."""
        config = RetailConfig()
        assert config.industry_type == "retail"
        assert config.days_sales_outstanding == 5  # Cash sales
        assert config.days_inventory_outstanding == 45
        assert config.gross_margin == 0.30  # Lower margins

    def test_retail_working_capital(self):
        """Test retail company working capital cycle."""
        config = RetailConfig()
        # Retail has fast cash collection
        expected_days = 5 + 45 - 35  # DSO + DIO - DPO
        assert config.working_capital_days == expected_days


class TestIndustryConfigIntegration:
    """Test integration with existing configuration classes."""

    def test_manufacturer_from_industry_config(self):
        """Test creating ManufacturerConfig from IndustryConfig."""
        industry = ManufacturingConfig()
        manufacturer_config = ManufacturerConfig.from_industry_config(
            industry, initial_assets=15_000_000
        )

        # Check that values are properly transferred
        assert manufacturer_config.base_operating_margin == industry.operating_margin
        assert manufacturer_config.ppe_ratio == industry.ppe_ratio
        assert manufacturer_config.initial_assets == 15_000_000

    def test_service_to_manufacturer_config(self):
        """Test converting service config to manufacturer config."""
        service = ServiceConfig()
        manufacturer_config = ManufacturerConfig.from_industry_config(service)

        # Service has higher margins but lower PP&E
        assert abs(manufacturer_config.base_operating_margin - 0.15) < 0.0001  # 0.60 - 0.45
        assert manufacturer_config.ppe_ratio == 0.2

    def test_retail_to_manufacturer_config(self):
        """Test converting retail config to manufacturer config."""
        retail = RetailConfig()
        manufacturer_config = ManufacturerConfig.from_industry_config(
            retail, asset_turnover_ratio=1.5  # Override with retail-specific turnover
        )

        assert abs(manufacturer_config.base_operating_margin - 0.08) < 0.0001  # 0.30 - 0.22
        assert manufacturer_config.ppe_ratio == 0.4
        assert manufacturer_config.asset_turnover_ratio == 1.5


class TestIndustryConfigSwitching:
    """Test switching between different industry configurations."""

    def test_switch_industry_types(self):
        """Test switching between different industry types."""
        configs = [ManufacturingConfig(), ServiceConfig(), RetailConfig()]

        for config in configs:
            # Each should have valid configuration
            assert config.industry_type in ["manufacturing", "services", "retail"]
            # Asset ratios should sum to 1
            asset_sum = config.current_asset_ratio + config.ppe_ratio + config.intangible_ratio
            assert abs(asset_sum - 1.0) < 0.01

    def test_industry_config_comparison(self):
        """Test comparing different industry configurations."""
        manufacturing = ManufacturingConfig()
        service = ServiceConfig()
        retail = RetailConfig()

        # Services should have lowest PP&E
        assert service.ppe_ratio < manufacturing.ppe_ratio
        assert service.ppe_ratio < retail.ppe_ratio

        # Services should have highest gross margin
        assert service.gross_margin > manufacturing.gross_margin
        assert service.gross_margin > retail.gross_margin

        # Retail should have shortest DSO (cash sales)
        assert retail.days_sales_outstanding < manufacturing.days_sales_outstanding
        assert retail.days_sales_outstanding < service.days_sales_outstanding


class TestIndustryConfigValidationRobustness:
    """Test that IndustryConfig validation uses Pydantic, not assert.

    Issue #462: IndustryConfig must not use assert for validation because
    assert statements are silently removed when Python runs with -O.
    These tests verify the Pydantic Field validators and model_validator
    are always active regardless of optimization flags.
    """

    def test_is_pydantic_model(self):
        """IndustryConfig and subclasses are Pydantic BaseModel instances."""
        from pydantic import BaseModel

        assert issubclass(IndustryConfig, BaseModel)
        assert issubclass(ManufacturingConfig, BaseModel)
        assert issubclass(ServiceConfig, BaseModel)
        assert issubclass(RetailConfig, BaseModel)

    def test_no_assert_in_source(self):
        """IndustryConfig source has no assert-based validation."""
        import inspect

        source = inspect.getsource(IndustryConfig)
        # Filter out docstrings/comments; check for bare assert statements
        lines = source.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'"):
                continue
            assert not stripped.startswith(
                "assert "
            ), f"Found assert-based validation in IndustryConfig: {stripped}"

    def test_gross_margin_out_of_range_raises_validation_error(self):
        """gross_margin > 1.0 is rejected by Pydantic, not assert."""
        with pytest.raises(ValidationError):
            IndustryConfig(gross_margin=5.0)

    def test_negative_gross_margin_rejected(self):
        """Negative gross_margin is rejected."""
        with pytest.raises(ValidationError):
            IndustryConfig(gross_margin=-0.1)

    def test_operating_expense_ratio_out_of_range(self):
        """operating_expense_ratio > 1.0 is rejected."""
        with pytest.raises(ValidationError):
            IndustryConfig(operating_expense_ratio=1.5)

    def test_negative_days_sales_outstanding(self):
        """Negative DSO is rejected."""
        with pytest.raises(ValidationError):
            IndustryConfig(days_sales_outstanding=-10)

    def test_negative_ppe_useful_life(self):
        """ppe_useful_life <= 0 is rejected."""
        with pytest.raises(ValidationError):
            IndustryConfig(ppe_useful_life=-10)

    def test_zero_ppe_useful_life(self):
        """ppe_useful_life == 0 is rejected (must be > 0)."""
        with pytest.raises(ValidationError):
            IndustryConfig(ppe_useful_life=0)

    def test_invalid_depreciation_method(self):
        """Only 'straight_line' and 'declining_balance' are accepted."""
        with pytest.raises(ValidationError):
            IndustryConfig(depreciation_method="accelerated")  # type: ignore[arg-type]

    def test_asset_ratios_not_summing_to_one(self):
        """Asset composition that doesn't sum to 1.0 is rejected."""
        with pytest.raises(ValidationError, match="Asset ratios must sum to 1.0"):
            IndustryConfig(
                current_asset_ratio=0.5,
                ppe_ratio=0.5,
                intangible_ratio=0.5,
            )

    def test_subclass_validation_manufacturing(self):
        """ManufacturingConfig also validates via Pydantic."""
        with pytest.raises(ValidationError):
            ManufacturingConfig(gross_margin=2.0)

    def test_subclass_validation_service(self):
        """ServiceConfig also validates via Pydantic."""
        with pytest.raises(ValidationError):
            ServiceConfig(ppe_useful_life=-5)

    def test_subclass_validation_retail(self):
        """RetailConfig also validates via Pydantic."""
        with pytest.raises(ValidationError):
            RetailConfig(operating_expense_ratio=-0.5)
