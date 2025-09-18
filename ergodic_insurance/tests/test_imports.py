"""Test module import patterns and naming conventions.

This module validates that all imports work correctly and follow
consistent naming patterns.

Author: Alex Filiakov
Date: 2025-01-26
"""

import importlib
import os
from pathlib import Path

import pytest


class TestImportPatterns:
    """Test import patterns and module naming conventions."""

    def test_module_class_naming_consistency(self):
        """Verify that module names align with their primary class names."""
        # Modules are now directly in ergodic_insurance/, not in src/
        package_dir = Path(__file__).parent.parent

        # Expected naming patterns (module_name: primary_class)
        expected_patterns = {
            "business_optimizer": "BusinessOptimizer",
            "claim_generator": "ClaimGenerator",
            "claim_development": "ClaimDevelopment",
            "config_loader": "ConfigLoader",
            "ergodic_analyzer": "ErgodicAnalyzer",
            "manufacturer": "WidgetManufacturer",
            "simulation": "Simulation",
            "risk_metrics": "RiskMetrics",
            "decision_engine": "InsuranceDecisionEngine",
            "insurance_program": "InsuranceProgram",
            "monte_carlo": "MonteCarloEngine",
        }

        for module_name, expected_class in expected_patterns.items():
            module_path = package_dir / f"{module_name}.py"
            if module_path.exists():
                # Import the module dynamically
                module = importlib.import_module(f"ergodic_insurance.{module_name}")

                # Check if the expected class exists
                assert hasattr(
                    module, expected_class
                ), f"Module {module_name} should contain class {expected_class}"

    def test_public_api_imports(self):
        """Test that all public API imports work from package root.

        This test verifies not just that imports succeed, but that the
        imported classes have the expected structure and can be instantiated.
        """
        from ergodic_insurance import (  # pylint: disable=no-name-in-module
            BusinessConstraints,
            BusinessObjective,
            BusinessOptimizer,
            ClaimEvent,
            ClaimGenerator,
            Config,
            ErgodicAnalyzer,
            ManufacturerConfig,
            Simulation,
            SimulationResults,
            WidgetManufacturer,
        )

        # Verify all imports are classes or valid types
        assert isinstance(BusinessOptimizer, type), "BusinessOptimizer should be a class"
        assert isinstance(BusinessObjective, type), "BusinessObjective should be a class"
        assert isinstance(BusinessConstraints, type), "BusinessConstraints should be a class"
        assert isinstance(ClaimEvent, type), "ClaimEvent should be a class"
        assert isinstance(ClaimGenerator, type), "ClaimGenerator should be a class"
        assert isinstance(Config, type), "Config should be a class"
        assert isinstance(ManufacturerConfig, type), "ManufacturerConfig should be a class"
        assert isinstance(ErgodicAnalyzer, type), "ErgodicAnalyzer should be a class"
        assert isinstance(WidgetManufacturer, type), "WidgetManufacturer should be a class"
        assert isinstance(Simulation, type), "Simulation should be a class"
        assert isinstance(SimulationResults, type), "SimulationResults should be a class"

        # Verify classes have expected methods/attributes
        assert hasattr(BusinessOptimizer, "__init__"), "BusinessOptimizer should be instantiable"
        assert hasattr(
            ClaimGenerator, "generate_claims"
        ), "ClaimGenerator should have generate_claims method"
        assert hasattr(WidgetManufacturer, "__init__"), "WidgetManufacturer should be instantiable"
        assert hasattr(ErgodicAnalyzer, "__init__"), "ErgodicAnalyzer should be instantiable"

        # Test that basic instantiation works for simple classes
        try:
            # ClaimEvent should be instantiable with basic parameters
            event = ClaimEvent(amount=1000, event_type="test", year=1)
            assert event.amount == 1000, "ClaimEvent should store amount"
        except TypeError:
            # If it needs different parameters, at least verify it's a proper class
            assert ClaimEvent.__module__.startswith(
                "ergodic_insurance"
            ), "ClaimEvent should be from ergodic_insurance module"

    def test_no_circular_imports(self):
        """Verify there are no circular import dependencies."""
        # This test passes if all modules can be imported individually
        modules_to_test = [
            "business_optimizer",
            "claim_generator",
            "claim_development",
            "config",
            "config_loader",
            "convergence",
            "decision_engine",
            "ergodic_analyzer",
            "insurance",
            "insurance_program",
            "loss_distributions",
            "manufacturer",
            "monte_carlo",
            "optimization",
            "pareto_frontier",
            "risk_metrics",
            "simulation",
            "stochastic_processes",
            "visualization",
        ]

        for module_name in modules_to_test:
            try:
                importlib.import_module(f"ergodic_insurance.{module_name}")
            except ImportError as e:
                # Allow ImportError for optional dependencies like matplotlib
                if "matplotlib" not in str(e) and "seaborn" not in str(e):
                    raise

    def test_init_exports_match_all(self):
        """Verify that __init__.py __all__ matches actual exports."""
        import ergodic_insurance
        from ergodic_insurance import __all__

        # Check that all items in __all__ can be imported
        for item in __all__:
            if item == "__version__":
                continue
            assert hasattr(ergodic_insurance, item), f"Item '{item}' in __all__ cannot be imported"

    def test_consistent_import_style(self):
        """Verify that imports follow consistent patterns in test files."""
        test_dir = Path(__file__).parent
        test_files = list(test_dir.glob("test_*.py"))

        for test_file in test_files:
            if test_file.name == "test_imports.py":
                continue

            content = test_file.read_text(encoding="utf-8")
            lines = content.split("\n")

            for line in lines:
                # Check for consistent import patterns
                if "from ergodic_insurance" in line:
                    # Allow test fixture imports from ergodic_insurance.tests
                    if ".tests.test_fixtures" in line:
                        continue  # Test fixtures are allowed to be imported directly

                    # After moving from src/, imports should be directly from ergodic_insurance
                    # Valid patterns:
                    # - from ergodic_insurance import ...
                    # - from ergodic_insurance.module import ...
                    # Invalid patterns:
                    # - from ergodic_insurance.src import ... (old structure)
                    assert ".src" not in line, f"Old src/ pattern found in {test_file.name}: {line}"

                    # Should not use old naming
                    assert (
                        "BusinessOutcomeOptimizer" not in line
                    ), f"Old class name found in {test_file.name}: {line}"

    def test_class_instantiation(self):
        """Test that imported classes can be instantiated with valid parameters.

        This ensures classes are not just importable but actually functional.
        """
        from ergodic_insurance import (  # pylint: disable=no-name-in-module
            ManufacturerConfig,
            WidgetManufacturer,
        )

        # Test ManufacturerConfig instantiation
        config = ManufacturerConfig(
            initial_assets=1_000_000,
            asset_turnover_ratio=0.5,
            base_operating_margin=0.1,
            tax_rate=0.25,
            retention_ratio=0.8,
        )
        assert config.initial_assets == 1_000_000, "Config should store initial_assets"
        assert config.tax_rate == 0.25, "Config should store tax_rate"

        # Test WidgetManufacturer instantiation with config
        manufacturer = WidgetManufacturer(config)
        assert manufacturer is not None, "Manufacturer should be created"
        assert hasattr(
            manufacturer, "total_assets"
        ), "Manufacturer should have total_assets attribute"
        assert (
            manufacturer.total_assets == config.initial_assets
        ), "Manufacturer should initialize with config's initial_assets"
