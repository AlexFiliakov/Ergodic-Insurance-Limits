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
        src_dir = Path(__file__).parent.parent / "src"

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
            module_path = src_dir / f"{module_name}.py"
            if module_path.exists():
                # Import the module dynamically
                module = importlib.import_module(f"ergodic_insurance.src.{module_name}")

                # Check if the expected class exists
                assert hasattr(
                    module, expected_class
                ), f"Module {module_name} should contain class {expected_class}"

    def test_public_api_imports(self):
        """Test that all public API imports work from package root."""
        from ergodic_insurance.src import (
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

        # Verify all imports are not None
        assert BusinessOptimizer is not None
        assert BusinessObjective is not None
        assert BusinessConstraints is not None
        assert ClaimEvent is not None
        assert ClaimGenerator is not None
        assert Config is not None
        assert ManufacturerConfig is not None
        assert ErgodicAnalyzer is not None
        assert WidgetManufacturer is not None
        assert Simulation is not None
        assert SimulationResults is not None

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
                importlib.import_module(f"ergodic_insurance.src.{module_name}")
            except ImportError as e:
                # Allow ImportError for optional dependencies like matplotlib
                if "matplotlib" not in str(e) and "seaborn" not in str(e):
                    raise

    def test_init_exports_match_all(self):
        """Verify that __init__.py __all__ matches actual exports."""
        import ergodic_insurance.src
        from ergodic_insurance.src import __all__

        # Check that all items in __all__ can be imported
        for item in __all__:
            if item == "__version__":
                continue
            assert hasattr(
                ergodic_insurance.src, item
            ), f"Item '{item}' in __all__ cannot be imported"

    def test_consistent_import_style(self):
        """Verify that imports follow consistent patterns in test files."""
        test_dir = Path(__file__).parent
        test_files = list(test_dir.glob("test_*.py"))

        for test_file in test_files:
            if test_file.name == "test_imports.py":
                continue

            content = test_file.read_text()
            lines = content.split("\n")

            for line in lines:
                # Check for consistent import patterns
                if "from ergodic_insurance" in line:
                    # Should use .src. pattern
                    assert (
                        ".src." in line or "ergodic_insurance import" in line
                    ), f"Inconsistent import pattern in {test_file.name}: {line}"

                    # Should not use old naming
                    assert (
                        "BusinessOutcomeOptimizer" not in line
                    ), f"Old class name found in {test_file.name}: {line}"
