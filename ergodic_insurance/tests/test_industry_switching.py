"""Tests for switching between different industry configurations."""

import pytest

from ergodic_insurance.config import (
    Config,
    DebtConfig,
    GrowthConfig,
    IndustryConfig,
    LoggingConfig,
    ManufacturerConfig,
    ManufacturingConfig,
    OutputConfig,
    RetailConfig,
    ServiceConfig,
    SimulationConfig,
    WorkingCapitalConfig,
)


class TestIndustrySwitching:
    """Test switching between different industry configurations."""

    def create_base_config_v2(self, industry_config=None):
        """Helper to create a basic Config with required fields."""
        from ergodic_insurance.config import ProfileMetadata

        return Config(
            profile=ProfileMetadata(
                name="test-profile", description="Test profile for industry switching"
            ),
            manufacturer=ManufacturerConfig(
                initial_assets=10_000_000,
                asset_turnover_ratio=0.8,
                base_operating_margin=0.1,
                tax_rate=0.25,
                retention_ratio=0.7,
            ),
            working_capital=WorkingCapitalConfig(percent_of_sales=0.2),
            growth=GrowthConfig(annual_growth_rate=0.05, volatility=0.02),
            debt=DebtConfig(
                interest_rate=0.05, max_leverage_ratio=2.0, minimum_cash_balance=500_000
            ),
            simulation=SimulationConfig(time_horizon_years=10),
            output=OutputConfig(),
            logging=LoggingConfig(level="INFO"),
            industry_config=industry_config,
        )

    def test_switch_to_manufacturing(self):
        """Test switching to manufacturing configuration."""
        config = self.create_base_config_v2()

        # Add manufacturing config
        manufacturing = ManufacturingConfig()
        config = config.model_copy(update={"industry_config": manufacturing})

        assert config.industry_config is not None
        assert config.industry_config.industry_type == "manufacturing"
        assert config.industry_config.days_inventory_outstanding == 60

    def test_switch_to_service(self):
        """Test switching to service configuration."""
        config = self.create_base_config_v2()

        # Switch to service config
        service = ServiceConfig()
        config = config.model_copy(update={"industry_config": service})

        assert config.industry_config.industry_type == "services"
        assert config.industry_config.days_inventory_outstanding == 0
        assert config.industry_config.ppe_ratio == 0.2

    def test_switch_to_retail(self):
        """Test switching to retail configuration."""
        config = self.create_base_config_v2()

        # Switch to retail config
        retail = RetailConfig()
        config = config.model_copy(update={"industry_config": retail})

        assert config.industry_config.industry_type == "retail"
        assert config.industry_config.days_sales_outstanding == 5

    def test_switch_between_industries(self):
        """Test switching from one industry to another."""
        # Start with manufacturing
        config = self.create_base_config_v2(ManufacturingConfig())
        assert config.industry_config.industry_type == "manufacturing"

        # Switch to service
        config = config.model_copy(update={"industry_config": ServiceConfig()})
        assert config.industry_config.industry_type == "services"

        # Switch to retail
        config = config.model_copy(update={"industry_config": RetailConfig()})
        assert config.industry_config.industry_type == "retail"

        # Switch back to manufacturing
        config = config.model_copy(update={"industry_config": ManufacturingConfig()})
        assert config.industry_config.industry_type == "manufacturing"

    def test_industry_config_affects_manufacturer(self):
        """Test that industry config affects manufacturer configuration."""
        # Create manufacturer config from manufacturing industry
        manufacturing = ManufacturingConfig()
        manufacturer_config = ManufacturerConfig.from_industry_config(manufacturing)

        manufacturing_margin = manufacturer_config.base_operating_margin
        manufacturing_ppe = manufacturer_config.ppe_ratio

        # Create manufacturer config from service industry
        service = ServiceConfig()
        service_manufacturer = ManufacturerConfig.from_industry_config(service)

        service_margin = service_manufacturer.base_operating_margin
        service_ppe = service_manufacturer.ppe_ratio

        # Service should have different characteristics
        assert service_margin != manufacturing_margin
        assert service_ppe < manufacturing_ppe  # Services have less PP&E

    def test_preserve_custom_values_on_switch(self):
        """Test that custom values are preserved when switching industries."""
        config = self.create_base_config_v2()

        # Set custom manufacturer values
        custom_assets = 25_000_000
        config.manufacturer.initial_assets = custom_assets

        # Switch industry
        config = config.model_copy(update={"industry_config": ServiceConfig()})

        # Custom values should be preserved
        assert config.manufacturer.initial_assets == custom_assets

    def test_industry_specific_calculations(self):
        """Test that industry-specific calculations work correctly."""
        industries = [ManufacturingConfig(), ServiceConfig(), RetailConfig()]

        for industry in industries:
            # Working capital calculation should work
            wc_days = industry.working_capital_days
            assert isinstance(wc_days, (int, float))

            # Operating margin calculation should work
            op_margin = industry.operating_margin
            assert isinstance(op_margin, (int, float))
            assert op_margin == industry.gross_margin - industry.operating_expense_ratio


class TestIndustryConfigCompatibility:
    """Test backward compatibility with existing configurations."""

    def test_config_without_industry_works(self):
        """Test that configurations without industry_config still work."""
        from ergodic_insurance.config import ProfileMetadata

        # Create config without industry_config
        config = Config(
            profile=ProfileMetadata(
                name="test-profile", description="Test without industry config"
            ),
            manufacturer=ManufacturerConfig(
                initial_assets=10_000_000,
                asset_turnover_ratio=0.8,
                base_operating_margin=0.1,
                tax_rate=0.25,
                retention_ratio=0.7,
            ),
            working_capital=WorkingCapitalConfig(percent_of_sales=0.2),
            growth=GrowthConfig(annual_growth_rate=0.05, volatility=0.02),
            debt=DebtConfig(
                interest_rate=0.05, max_leverage_ratio=2.0, minimum_cash_balance=500_000
            ),
            simulation=SimulationConfig(time_horizon_years=10),
            output=OutputConfig(),
            logging=LoggingConfig(level="INFO"),
            # No industry_config specified
        )

        # Config should work without industry_config
        assert config.industry_config is None
        assert config.manufacturer.initial_assets == 10_000_000

    def test_add_industry_to_existing_config(self):
        """Test adding industry config to existing configuration."""
        from ergodic_insurance.config import ProfileMetadata

        # Start without industry config
        config = Config(
            profile=ProfileMetadata(name="test-profile", description="Test profile"),
            manufacturer=ManufacturerConfig(
                initial_assets=10_000_000,
                asset_turnover_ratio=0.8,
                base_operating_margin=0.1,
                tax_rate=0.25,
                retention_ratio=0.7,
            ),
            working_capital=WorkingCapitalConfig(percent_of_sales=0.2),
            growth=GrowthConfig(annual_growth_rate=0.05, volatility=0.02),
            debt=DebtConfig(
                interest_rate=0.05, max_leverage_ratio=2.0, minimum_cash_balance=500_000
            ),
            simulation=SimulationConfig(time_horizon_years=10),
            output=OutputConfig(),
            logging=LoggingConfig(level="INFO"),
        )

        # Add industry config later
        updated_config = config.model_copy(update={"industry_config": ManufacturingConfig()})

        assert updated_config.industry_config is not None
        assert updated_config.industry_config.industry_type == "manufacturing"


class TestIndustryConfigOverrides:
    """Test overriding industry config defaults."""

    def test_override_manufacturing_defaults(self):
        """Test overriding manufacturing default values."""
        custom_margin = 0.45
        custom_ppe = 0.6

        config = ManufacturingConfig(
            gross_margin=custom_margin,
            ppe_ratio=custom_ppe,
            current_asset_ratio=0.3,  # Adjust for sum = 1.0
            intangible_ratio=0.1,
        )

        assert config.gross_margin == custom_margin
        assert config.ppe_ratio == custom_ppe
        # Industry type should remain
        assert config.industry_type == "manufacturing"

    def test_override_service_defaults(self):
        """Test overriding service default values."""
        # Services usually have no inventory, but override it
        config = ServiceConfig(days_inventory_outstanding=10)  # Some inventory

        assert config.days_inventory_outstanding == 10
        assert config.industry_type == "services"

    def test_override_retail_defaults(self):
        """Test overriding retail default values."""
        config = RetailConfig(
            days_sales_outstanding=30, gross_margin=0.40  # Longer payment terms  # Higher margin
        )

        assert config.days_sales_outstanding == 30
        assert config.gross_margin == 0.40
        assert config.industry_type == "retail"

    def test_partial_overrides(self):
        """Test that partial overrides work correctly."""
        config = ManufacturingConfig(gross_margin=0.40)  # Override only this

        # Overridden value
        assert config.gross_margin == 0.40

        # Other values should be defaults
        assert config.days_inventory_outstanding == 60
        assert config.ppe_ratio == 0.5
