"""Integration tests for config_v2 module."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from pydantic import ValidationError
import pytest
import yaml

from ergodic_insurance.config import (
    Config,
    DebtConfig,
    GrowthConfig,
    InsuranceConfig,
    InsuranceLayerConfig,
    LoggingConfig,
    LossDistributionConfig,
    ManufacturerConfig,
    ModuleConfig,
    OutputConfig,
    PresetConfig,
    PresetLibrary,
    ProfileMetadata,
    SimulationConfig,
    WorkingCapitalConfig,
)


class TestIntegration:
    """Integration tests for config_v2 module."""

    def test_full_config_lifecycle(self):
        """Test complete configuration lifecycle with all features."""
        # Create base config
        config = Config(
            profile=ProfileMetadata(
                name="integration-test",
                description="Integration test profile",
                version="1.0.0",
                includes=["risk", "optimization"],
                tags=["test", "integration"],
            ),
            manufacturer=ManufacturerConfig(
                initial_assets=15000000,
                asset_turnover_ratio=0.9,
                base_operating_margin=0.09,
                tax_rate=0.25,
                retention_ratio=0.6,
            ),
            working_capital=WorkingCapitalConfig(percent_of_sales=0.15),
            growth=GrowthConfig(annual_growth_rate=0.06, volatility=0.15),
            debt=DebtConfig(
                interest_rate=0.045,
                max_leverage_ratio=3.0,
                minimum_cash_balance=100000,
            ),
            simulation=SimulationConfig(time_horizon_years=15, random_seed=123),
            output=OutputConfig(output_directory="./integration_output"),
            logging=LoggingConfig(level="DEBUG"),
            insurance=InsuranceConfig(
                enabled=True,
                layers=[
                    InsuranceLayerConfig(
                        name="Primary",
                        limit=2000000,
                        attachment=0,
                        base_premium_rate=0.02,
                        reinstatements=1,
                    ),
                    InsuranceLayerConfig(
                        name="Excess",
                        limit=5000000,
                        attachment=2000000,
                        base_premium_rate=0.01,
                    ),
                ],
                deductible=100000,
                coinsurance=0.9,
            ),
            losses=LossDistributionConfig(
                frequency_distribution="negative_binomial",
                frequency_annual=4.0,
                severity_distribution="pareto",
                severity_mean=75000,
                severity_std=25000,
                correlation_factor=0.15,
                tail_alpha=2.2,
            ),
            custom_modules={
                "stress_testing": ModuleConfig(
                    module_name="stress_testing",
                    module_version="1.0.0",
                    dependencies=["risk"],
                )
            },
        )

        # Validate completeness
        issues = config.validate_completeness()
        assert len(issues) == 0

        # Apply preset (returns new instance)
        preset_data = {
            "growth": {"volatility": 0.25},
            "losses": {"tail_alpha": 3.0},
        }
        config = config.with_preset("high_volatility", preset_data)
        assert config.growth.volatility == 0.25
        assert config.losses is not None and config.losses.tail_alpha == 3.0
        assert "high_volatility" in config.applied_presets

        # Create override version using dot notation
        override_config = config.with_overrides(
            {
                "manufacturer.initial_assets": 20000000,
                "simulation.time_horizon_years": 25,
                "insurance.deductible": 150000,
            }
        )

        assert override_config.manufacturer.initial_assets == 20000000
        assert override_config.simulation.time_horizon_years == 25
        assert (
            override_config.insurance is not None and override_config.insurance.deductible == 150000
        )

        # Original should be unchanged
        assert config.manufacturer.initial_assets == 15000000
        assert config.simulation.time_horizon_years == 15

        # Test custom module access
        assert "stress_testing" in config.custom_modules
        assert config.custom_modules["stress_testing"].module_version == "1.0.0"
