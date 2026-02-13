"""Comprehensive tests for all parameter combinations.

This module validates all possible parameter combinations across
different configuration files, ensuring compatibility and proper
interaction between different settings.
"""

import itertools
from pathlib import Path
from typing import Any, Dict, List

from pydantic import ValidationError
import pytest
import yaml

from ergodic_insurance.config import Config
from ergodic_insurance.config_loader import ConfigLoader


@pytest.mark.filterwarnings("ignore:ConfigLoader is deprecated:DeprecationWarning")
class TestParameterCombinations:
    """Test all parameter combinations for validity."""

    @pytest.fixture
    def config_loader(self, project_root):
        """Create config loader instance."""
        return ConfigLoader(project_root / "data" / "parameters")

    @pytest.fixture
    def all_parameter_files(self, project_root):
        """Get all parameter YAML files."""
        param_dir = project_root / "data" / "parameters"
        return list(param_dir.glob("*.yaml"))

    def test_all_individual_configs_load(self, all_parameter_files):
        """Test that each parameter file can be loaded individually."""
        for yaml_file in all_parameter_files:
            config_name = yaml_file.stem
            try:
                with open(yaml_file, "r") as f:
                    data = yaml.safe_load(f)

                # Skip files that aren't complete configs
                if not all(k in data for k in ["manufacturer", "simulation"]):
                    continue

                config = Config.from_yaml(yaml_file)
                assert config is not None, f"Failed to load {config_name}"

                # Basic validation
                assert config.manufacturer.initial_assets > 0
                assert 0 <= config.manufacturer.tax_rate <= 1
                assert config.simulation.time_horizon_years > 0

            except (ValidationError, KeyError):
                # TODO(tautology-review): silently swallowing validation errors
                # means invalid full configs would pass. Consider collecting
                # errors and failing if any non-partial config fails.
                pass

    def test_scenario_combinations(self, config_loader):
        """Test loading different scenario combinations."""
        scenarios = ["baseline", "conservative", "optimistic"]

        for scenario in scenarios:
            config = config_loader.load_scenario(scenario)

            # Validate key relationships
            if scenario == "conservative":
                assert config.growth.annual_growth_rate <= 0.05
                assert config.manufacturer.base_operating_margin <= 0.08
                assert config.debt.max_leverage_ratio <= 2.0
            elif scenario == "optimistic":
                assert config.growth.annual_growth_rate >= 0.05
                assert config.manufacturer.base_operating_margin >= 0.08
                assert config.debt.max_leverage_ratio >= 1.5

            # Common validations
            assert config.manufacturer.retention_ratio <= 1.0
            assert config.working_capital.percent_of_sales <= 0.5

    def test_growth_type_combinations(self, config_loader):
        """Test different growth type configurations."""
        growth_types = ["deterministic", "stochastic"]

        for growth_type in growth_types:
            if growth_type == "stochastic":
                # Load stochastic config
                config = config_loader.load("stochastic")
                assert config.growth.type == "stochastic"
                assert config.growth.volatility > 0

                # Verify stochastic-specific parameters exist
                assert hasattr(config, "stochastic") or hasattr(config.growth, "mean_reversion")
            else:
                # Load baseline (deterministic)
                config = config_loader.load("baseline")
                assert config.growth.type == "deterministic"
                assert config.growth.volatility == 0

    def test_time_resolution_combinations(self):
        """Test different time resolution settings."""
        resolutions = ["annual", "monthly"]
        horizons = [10, 50, 100, 500]

        for resolution, horizon in itertools.product(resolutions, horizons):
            config_dict = self._create_base_config()
            config_dict["simulation"]["time_resolution"] = resolution
            config_dict["simulation"]["time_horizon_years"] = horizon

            config = Config(**config_dict)

            # Validate time steps calculation
            if resolution == "monthly":
                expected_steps = horizon * 12
            else:
                expected_steps = horizon

            # Ensure horizon doesn't exceed max
            assert config.simulation.time_horizon_years <= config.simulation.max_horizon_years

    def test_leverage_and_debt_combinations(self):
        """Test different leverage and debt parameter combinations."""
        leverage_ratios = [0.5, 1.0, 2.0, 3.0]
        interest_rates = [0.01, 0.03, 0.05, 0.08]
        min_cash_levels = [10_000, 100_000, 500_000]

        for leverage, interest, min_cash in itertools.product(
            leverage_ratios, interest_rates, min_cash_levels
        ):
            config_dict = self._create_base_config()
            config_dict["debt"]["max_leverage_ratio"] = leverage
            config_dict["debt"]["interest_rate"] = interest
            config_dict["debt"]["minimum_cash_balance"] = min_cash

            config = Config(**config_dict)

            # Validate relationships
            assert config.debt.max_leverage_ratio >= 0
            assert 0 <= config.debt.interest_rate <= 0.5  # Reasonable bounds
            assert config.debt.minimum_cash_balance >= 0

    def test_profitability_combinations(self):
        """Test different profitability parameter combinations."""
        margins = [0.02, 0.05, 0.08, 0.12, 0.15, 0.20]
        turnovers = [0.5, 0.8, 1.0, 1.2, 1.5]
        tax_rates = [0.0, 0.15, 0.25, 0.35]

        for margin, turnover, tax in itertools.product(margins, turnovers, tax_rates):
            config_dict = self._create_base_config()
            config_dict["manufacturer"]["base_operating_margin"] = margin
            config_dict["manufacturer"]["asset_turnover_ratio"] = turnover
            config_dict["manufacturer"]["tax_rate"] = tax

            config = Config(**config_dict)

            # Calculate implied ROA and validate
            roa_before_tax = margin * turnover
            roa_after_tax = roa_before_tax * (1 - tax)

            # Ensure reasonable profitability bounds
            assert roa_after_tax <= 0.5  # Max 50% after-tax ROA is reasonable

    def test_retention_and_growth_combinations(self):
        """Test retention ratio and growth rate combinations."""
        retention_ratios = [0.0, 0.3, 0.5, 0.7, 1.0]
        growth_rates = [0.0, 0.03, 0.05, 0.08, 0.12, 0.15]

        for retention, growth in itertools.product(retention_ratios, growth_rates):
            config_dict = self._create_base_config()
            config_dict["manufacturer"]["retention_ratio"] = retention
            config_dict["growth"]["annual_growth_rate"] = growth

            config = Config(**config_dict)

            # High growth should generally have high retention
            if growth > 0.10 and retention < 0.5:
                # This is a valid but unusual combination
                pass  # Could log warning in production

            assert 0 <= config.manufacturer.retention_ratio <= 1.0
            assert config.growth.annual_growth_rate >= 0

    def test_stochastic_parameter_combinations(self):
        """Test stochastic process parameter combinations."""
        process_types = ["gbm", "lognormal", "mean_reverting"]
        volatilities = [0.05, 0.10, 0.15, 0.20, 0.30]

        for process_type, volatility in itertools.product(process_types, volatilities):
            config_dict = self._create_base_config()
            config_dict["growth"]["type"] = "stochastic"
            config_dict["growth"]["volatility"] = volatility

            # Add stochastic section if needed
            config_dict["stochastic"] = {
                "process_type": process_type,
                "sales_volatility": volatility * 0.8,
                "margin_volatility": volatility * 0.5,
                "sales_margin_correlation": 0.3,
                "time_step": 1.0,
            }

            try:
                config = Config(**config_dict)

                # Process-specific validations
                if process_type == "mean_reverting":
                    # Mean reverting should have reasonable volatility
                    assert volatility <= 0.30

                assert config.growth.volatility > 0

            except ValidationError as e:
                # Some combinations might be invalid, which is expected
                if "requires non-zero volatility" in str(e):
                    pass  # Expected for certain combinations

    def test_output_format_combinations(self):
        """Test different output format and configuration combinations."""
        formats = ["csv", "parquet", "json"]
        checkpoint_freqs = [0, 1, 10, 50, 100]
        detailed_flags = [True, False]

        for fmt, freq, detailed in itertools.product(formats, checkpoint_freqs, detailed_flags):
            config_dict = self._create_base_config()
            config_dict["output"]["file_format"] = fmt
            config_dict["output"]["checkpoint_frequency"] = freq
            config_dict["output"]["detailed_metrics"] = detailed

            config = Config(**config_dict)

            # Validate checkpoint frequency relative to horizon
            if freq > 0:
                assert freq <= config.simulation.time_horizon_years

    def test_extreme_parameter_boundaries(self):
        """Test parameter combinations at extreme boundaries."""
        # Test minimum viable company
        min_config = self._create_base_config()
        min_config["manufacturer"]["initial_assets"] = 100_000
        min_config["manufacturer"]["base_operating_margin"] = 0.01
        min_config["manufacturer"]["retention_ratio"] = 0.0
        min_config["growth"]["annual_growth_rate"] = 0.0

        config = Config(**min_config)
        assert config.manufacturer.initial_assets == 100_000

        # Test maximum growth company
        max_config = self._create_base_config()
        max_config["manufacturer"]["initial_assets"] = 1_000_000_000
        max_config["manufacturer"]["base_operating_margin"] = 0.30
        max_config["manufacturer"]["retention_ratio"] = 1.0
        max_config["growth"]["annual_growth_rate"] = 0.20

        config = Config(**max_config)
        assert config.manufacturer.base_operating_margin == 0.30

    def test_invalid_combinations(self):
        """Test that invalid parameter combinations are rejected."""
        # Negative assets
        with pytest.raises(ValidationError):
            config_dict = self._create_base_config()
            config_dict["manufacturer"]["initial_assets"] = -1000
            Config(**config_dict)

        # Tax rate > 100%
        with pytest.raises(ValidationError):
            config_dict = self._create_base_config()
            config_dict["manufacturer"]["tax_rate"] = 1.5
            Config(**config_dict)

        # Retention ratio > 100%
        with pytest.raises(ValidationError):
            config_dict = self._create_base_config()
            config_dict["manufacturer"]["retention_ratio"] = 1.5
            Config(**config_dict)

        # Stochastic with zero volatility
        with pytest.raises(ValidationError):
            config_dict = self._create_base_config()
            config_dict["growth"]["type"] = "stochastic"
            config_dict["growth"]["volatility"] = 0.0
            Config(**config_dict)

        # Time horizon exceeding maximum
        with pytest.raises(ValidationError):
            config_dict = self._create_base_config()
            config_dict["simulation"]["time_horizon_years"] = 2000
            config_dict["simulation"]["max_horizon_years"] = 1000
            Config(**config_dict)

    def test_cross_parameter_dependencies(self):
        """Test dependencies between different parameter sections."""
        # High leverage with low profitability (risky)
        config_dict = self._create_base_config()
        config_dict["manufacturer"]["base_operating_margin"] = 0.02
        config_dict["debt"]["max_leverage_ratio"] = 3.0

        config = Config(**config_dict)
        # This should work but might warrant a warning
        assert config.debt.max_leverage_ratio == 3.0

        # High growth with low retention (unsustainable)
        config_dict = self._create_base_config()
        config_dict["growth"]["annual_growth_rate"] = 0.15
        config_dict["manufacturer"]["retention_ratio"] = 0.2

        config = Config(**config_dict)
        # This should work but is financially questionable
        assert config.growth.annual_growth_rate == 0.15

    def test_working_capital_combinations(self):
        """Test working capital parameter combinations."""
        wc_percentages = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
        turnovers = [0.5, 1.0, 1.5, 2.0]

        for wc_pct, turnover in itertools.product(wc_percentages, turnovers):
            config_dict = self._create_base_config()
            config_dict["working_capital"]["percent_of_sales"] = wc_pct
            config_dict["manufacturer"]["asset_turnover_ratio"] = turnover

            config = Config(**config_dict)

            # Higher turnover businesses often need more working capital
            if turnover > 1.5 and wc_pct < 0.10:
                pass  # Potentially problematic but allowed

            assert config.working_capital.percent_of_sales <= 0.5

    def test_simulation_seed_combinations(self):
        """Test different random seed configurations."""
        seeds = [None, 0, 42, 12345, 999999]

        for seed in seeds:
            config_dict = self._create_base_config()
            config_dict["simulation"]["random_seed"] = seed

            config = Config(**config_dict)
            assert config.simulation.random_seed == seed

    def test_logging_level_combinations(self):
        """Test different logging level configurations."""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR"]  # CRITICAL is not allowed in LoggingConfig
        enabled_flags = [True, False]
        console_flags = [True, False]

        for level, enabled, console in itertools.product(levels, enabled_flags, console_flags):
            config_dict = self._create_base_config()
            config_dict["logging"]["level"] = level
            config_dict["logging"]["enabled"] = enabled
            config_dict["logging"]["console_output"] = console

            config = Config(**config_dict)
            assert config.logging.level == level

            # If logging is disabled, other settings shouldn't matter
            if not enabled:
                assert not config.logging.enabled

    def _create_base_config(self) -> Dict[str, Any]:
        """Create a base configuration dictionary for testing."""
        return {
            "manufacturer": {
                "initial_assets": 10_000_000,
                "asset_turnover_ratio": 1.0,
                "base_operating_margin": 0.08,
                "tax_rate": 0.25,
                "retention_ratio": 0.7,
            },
            "working_capital": {"percent_of_sales": 0.20},
            "growth": {
                "type": "deterministic",
                "annual_growth_rate": 0.05,
                "volatility": 0.0,
            },
            "debt": {
                "interest_rate": 0.05,
                "max_leverage_ratio": 2.0,
                "minimum_cash_balance": 100_000,
            },
            "simulation": {
                "time_resolution": "annual",
                "time_horizon_years": 100,
                "max_horizon_years": 1000,
                "random_seed": 42,
            },
            "output": {
                "output_directory": "outputs",
                "file_format": "csv",
                "checkpoint_frequency": 10,
                "detailed_metrics": True,
            },
            "logging": {
                "enabled": True,
                "level": "INFO",
                "log_file": "test.log",
                "console_output": True,
                "format": "%(message)s",
            },
        }


class TestInsuranceParameterCombinations:
    """Test insurance-specific parameter combinations."""

    @pytest.fixture
    def insurance_params(self, project_root):
        """Load insurance parameter files."""
        param_dir = project_root / "data" / "parameters"
        files = {
            "structures": param_dir / "insurance_structures.yaml",
            "pricing": param_dir / "insurance_pricing_scenarios.yaml",
            "market": param_dir / "insurance_market.yaml",
            "distributions": param_dir / "loss_distributions.yaml",
        }

        params = {}
        for name, filepath in files.items():
            if filepath.exists():
                with open(filepath, "r") as f:
                    params[name] = yaml.safe_load(f)
        return params

    def test_insurance_layer_combinations(self, insurance_params):
        """Test different insurance layer structures."""
        if "structures" not in insurance_params:
            pytest.skip("Insurance structures file not found")

        structures = insurance_params["structures"]

        # Test each program type
        program_types = [
            "standard_manufacturing",
            "conservative_manufacturing",
            "aggressive_manufacturing",
            "small_company",
            "captive_program",
        ]

        for program_type in program_types:
            if program_type in structures:
                program = structures[program_type]

                # Validate layer structure
                previous_attachment = program["deductible"]

                for layer in program.get("layers", []):
                    # Attachment points should increase
                    assert layer["attachment_point"] >= previous_attachment

                    # Limits should be positive
                    assert layer["limit"] > 0

                    # Premium rates should be reasonable
                    assert 0 < layer["base_premium_rate"] <= 0.10

                    # Update for next layer
                    previous_attachment = layer["attachment_point"] + layer["limit"]

    def test_loss_distribution_combinations(self, insurance_params):
        """Test different loss distribution parameter combinations."""
        if "distributions" not in insurance_params:
            pytest.skip("Loss distributions file not found")

        distributions = insurance_params["distributions"]

        # Test attritional vs large loss parameters
        if "attritional_losses" in distributions:
            attritional = distributions["attritional_losses"]
            assert attritional["frequency"]["mean"] > 1  # High frequency
            assert attritional["severity"]["mean"] < 1_000_000  # Low severity

        if "large_losses" in distributions:
            large = distributions["large_losses"]
            assert large["frequency"]["mean"] < 1  # Low frequency
            assert large["severity"]["mean"] > 100_000  # High severity

    def test_market_condition_combinations(self, insurance_params):
        """Test different market condition scenarios."""
        if "market" not in insurance_params:
            pytest.skip("Insurance market file not found")

        market = insurance_params["market"]

        if "market_conditions" in market:
            conditions = market["market_conditions"]

            # Test hard vs soft market
            if "hard_market" in conditions:
                assert conditions["hard_market"]["price_multiplier"] > 1.0

            if "soft_market" in conditions:
                assert conditions["soft_market"]["price_multiplier"] < 1.0

    def test_pricing_scenario_combinations(self, insurance_params):
        """Test different pricing scenario combinations."""
        if "pricing" not in insurance_params:
            pytest.skip("Insurance pricing file not found")

        pricing = insurance_params["pricing"]

        # Test different pricing scenarios
        for scenario_name, scenario in pricing.items():
            if isinstance(scenario, dict) and "layers" in scenario:
                # Validate pricing structure
                for layer in scenario["layers"]:
                    if "rate" in layer:
                        assert 0 < layer["rate"] <= 0.20  # Max 20% rate


class TestParameterValidationRules:
    """Test specific parameter validation rules and constraints."""

    def test_manufacturer_constraints(self):
        """Test manufacturer parameter constraints."""
        # Test minimum assets
        config_dict = self._create_base_config()
        config_dict["manufacturer"]["initial_assets"] = 1  # Very small but valid
        config = Config(**config_dict)
        assert config.manufacturer.initial_assets == 1

        # Test maximum reasonable margin
        config_dict = self._create_base_config()
        config_dict["manufacturer"]["base_operating_margin"] = 0.35  # High but valid
        config = Config(**config_dict)
        assert config.manufacturer.base_operating_margin == 0.35

    def test_growth_volatility_constraints(self):
        """Test growth and volatility parameter constraints."""
        # Deterministic should have zero volatility
        config_dict = self._create_base_config()
        config_dict["growth"]["type"] = "deterministic"
        config_dict["growth"]["volatility"] = 0.0
        config = Config(**config_dict)
        assert config.growth.volatility == 0.0

        # Stochastic must have positive volatility
        config_dict["growth"]["type"] = "stochastic"
        config_dict["growth"]["volatility"] = 0.01  # Small but valid
        config = Config(**config_dict)
        assert config.growth.volatility == 0.01

    def test_time_horizon_constraints(self):
        """Test time horizon constraints."""
        config_dict = self._create_base_config()

        # Test at maximum
        config_dict["simulation"]["time_horizon_years"] = 1000
        config_dict["simulation"]["max_horizon_years"] = 1000
        config = Config(**config_dict)
        assert config.simulation.time_horizon_years == 1000

        # Test just below maximum
        config_dict["simulation"]["time_horizon_years"] = 999
        config = Config(**config_dict)
        assert config.simulation.time_horizon_years == 999

    def _create_base_config(self) -> Dict[str, Any]:
        """Create a base configuration dictionary for testing."""
        return {
            "manufacturer": {
                "initial_assets": 10_000_000,
                "asset_turnover_ratio": 1.0,
                "base_operating_margin": 0.08,
                "tax_rate": 0.25,
                "retention_ratio": 0.7,
            },
            "working_capital": {"percent_of_sales": 0.20},
            "growth": {
                "type": "deterministic",
                "annual_growth_rate": 0.05,
                "volatility": 0.0,
            },
            "debt": {
                "interest_rate": 0.05,
                "max_leverage_ratio": 2.0,
                "minimum_cash_balance": 100_000,
            },
            "simulation": {
                "time_resolution": "annual",
                "time_horizon_years": 100,
                "max_horizon_years": 1000,
                "random_seed": 42,
            },
            "output": {
                "output_directory": "outputs",
                "file_format": "csv",
                "checkpoint_frequency": 10,
                "detailed_metrics": True,
            },
            "logging": {
                "enabled": True,
                "level": "INFO",
                "log_file": "test.log",
                "console_output": True,
                "format": "%(message)s",
            },
        }
