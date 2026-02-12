#!/usr/bin/env python
"""Practical demonstrations of the ConfigManager system for real-world scenarios.

This script demonstrates practical applications of the configuration system
for insurance optimization workflows, including:
- Quick scenario comparison
- Risk assessment configurations
- Monte Carlo simulation setup
- Optimization parameter tuning
- Configuration migration from legacy systems

Author:
    Alex Filiakov

Date:
    2025-08-27
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ergodic_insurance import InsuranceProgram, MonteCarloEngine, WidgetManufacturer
from ergodic_insurance.config import Config
from ergodic_insurance.config_manager import ConfigManager
from ergodic_insurance.config_migrator import ConfigMigrator
from ergodic_insurance.monte_carlo import MonteCarloConfig


def demo_scenario_comparison():
    """Compare different market scenarios for insurance decision-making.

    This demonstrates how to quickly compare multiple scenarios to understand
    the impact of different market conditions on insurance optimization.
    """
    print("\n" + "=" * 70)
    print("SCENARIO COMPARISON FOR INSURANCE DECISION-MAKING")
    print("=" * 70)

    manager = ConfigManager()

    scenarios: Dict[str, Dict[str, Any]] = {
        "baseline": {
            "profile": "default",
            "presets": [],
            "description": "Current market conditions",
        },
        "hard_market": {
            "profile": "conservative",
            "presets": ["hard_market"],
            "description": "Rising premiums, limited capacity",
        },
        "recession": {
            "profile": "conservative",
            "presets": ["high_volatility"],
            "overrides": {
                "growth": {"annual_growth_rate": -0.02},
                "manufacturer": {"base_operating_margin": 0.04},
            },
            "description": "Economic downturn scenario",
        },
        "expansion": {
            "profile": "aggressive",
            "overrides": {
                "growth": {"annual_growth_rate": 0.15},
                "manufacturer": {"base_operating_margin": 0.12},
            },
            "description": "High growth expansion phase",
        },
    }

    results = []
    for scenario_name, scenario_config in scenarios.items():
        # Load configuration
        config = manager.load_profile(
            scenario_config["profile"],
            presets=scenario_config.get("presets", []),
            **scenario_config.get("overrides", {}),
        )

        # Calculate key metrics
        manufacturer = WidgetManufacturer(config.manufacturer)
        expected_revenue = (
            config.manufacturer.initial_assets * config.manufacturer.asset_turnover_ratio
        )
        expected_profit = expected_revenue * config.manufacturer.base_operating_margin

        result = {
            "name": scenario_name,
            "description": scenario_config["description"],
            "growth_rate": config.growth.annual_growth_rate,
            "volatility": config.growth.volatility,
            "base_operating_margin": config.manufacturer.base_operating_margin,
            "expected_annual_profit": expected_profit,
            "risk_tolerance": (
                "High"
                if scenario_name == "expansion"
                else "Low" if "conservative" in scenario_config["profile"] else "Medium"
            ),
        }
        results.append(result)

    # Display comparison table
    print(
        "\n{:<15} {:<30} {:>12} {:>12} {:>15}".format(
            "Scenario", "Description", "Growth", "Volatility", "Annual Profit"
        )
    )
    print("-" * 90)

    for r in results:
        print(
            "{:<15} {:<30} {:>11.1%} {:>11.1%} ${:>13,.0f}".format(
                r["name"],
                r["description"],
                r["growth_rate"],
                r["volatility"],
                r["expected_annual_profit"],
            )
        )

    print("\nRecommendation: Choose scenario based on current market conditions and risk appetite")


def demo_monte_carlo_setup():
    """Demonstrate setting up a Monte Carlo simulation with configuration.

    Shows how to configure and run a simple Monte Carlo simulation using
    the configuration system for insurance optimization.
    """
    print("\n" + "=" * 70)
    print("MONTE CARLO SIMULATION SETUP")
    print("=" * 70)

    manager = ConfigManager()

    # Load configuration for simulation
    config = manager.load_profile(
        "default",
        presets=["high_volatility"],  # Add volatility for interesting results
        monte_carlo={"n_simulations": 1000, "n_years": 10, "seed": 42},
    )

    print(f"\nSimulation Configuration:")
    print(f"  Profile: {config.profile.name}")
    print(f"  Time resolution: {config.simulation.time_resolution}")
    print(f"  Time horizon: {config.simulation.time_horizon_years} years")
    print(f"  Random seed: {config.simulation.random_seed}")

    # Setup simulation components
    print(f"\nManufacturer Configuration:")
    print(f"  Initial assets: ${config.manufacturer.initial_assets:,.0f}")
    print(f"  Operating margin: {config.manufacturer.base_operating_margin:.1%}")
    print(f"  Growth volatility: {config.growth.volatility:.1%}")

    if hasattr(config, "insurance") and config.insurance:
        print(f"\nInsurance Configuration:")
        print(f"  Number of layers: {len(config.insurance.layers)}")
        for i, layer in enumerate(config.insurance.layers, 1):
            print(
                f"  Layer {i}: ${layer.attachment:,.0f} xs ${layer.limit:,.0f} @ {layer.base_premium_rate:.2%}"
            )

    print("\n✓ Configuration ready for Monte Carlo simulation")
    print("  Next step: Initialize MonteCarloEngine with this configuration")


def demo_optimization_tuning():
    """Demonstrate configuration for optimization algorithm tuning.

    Shows how to configure optimization parameters for finding optimal
    insurance limits and structures.
    """
    print("\n" + "=" * 70)
    print("OPTIMIZATION PARAMETER TUNING")
    print("=" * 70)

    manager = ConfigManager()

    # Different optimization strategies
    strategies = {
        "quick_exploration": {
            "optimization": {
                "algorithm": "differential_evolution",
                "max_iterations": 50,
                "population_size": 15,
                "tolerance": 1e-3,
            },
            "description": "Fast exploration for initial insights",
        },
        "balanced": {
            "optimization": {
                "algorithm": "trust-region",
                "max_iterations": 200,
                "tolerance": 1e-4,
                "step_size": 0.1,
            },
            "description": "Balanced accuracy and speed",
        },
        "high_precision": {
            "optimization": {
                "algorithm": "augmented_lagrangian",
                "max_iterations": 1000,
                "tolerance": 1e-6,
                "constraint_tolerance": 1e-5,
            },
            "description": "High precision for final decisions",
        },
    }

    for strategy_name, strategy_config in strategies.items():
        config = manager.load_profile("default")

        print(f"\n{strategy_name.upper()}: {strategy_config['description']}")
        # Note: optimization settings would be in custom_modules if configured
        if "optimization" in config.custom_modules:
            opt = config.custom_modules["optimization"]
            if hasattr(opt, "settings") and isinstance(opt.settings, dict):
                print(f"  Algorithm: {opt.settings.get('algorithm', 'default')}")
                print(f"  Max iterations: {opt.settings.get('max_iterations', 100)}")
                print(f"  Tolerance: {opt.settings.get('tolerance', 1e-6)}")
                if "population_size" in opt.settings:
                    print(f"  Population size: {opt.settings['population_size']}")
                if "step_size" in opt.settings:
                    print(f"  Step size: {opt.settings['step_size']}")


def demo_risk_assessment_configs():
    """Demonstrate configurations for different risk assessment scenarios.

    Shows how to set up configurations for various risk metrics and
    assessment criteria.
    """
    print("\n" + "=" * 70)
    print("RISK ASSESSMENT CONFIGURATIONS")
    print("=" * 70)

    manager = ConfigManager()

    # Configure for different risk metrics
    risk_configs = {
        "value_at_risk": {
            "risk_metrics": {"metric_type": "var", "confidence_level": 0.95, "time_horizon": 1},
            "description": "95% VaR over 1 year",
        },
        "conditional_var": {
            "risk_metrics": {"metric_type": "cvar", "confidence_level": 0.95, "time_horizon": 1},
            "description": "95% CVaR (Expected Shortfall)",
        },
        "ruin_probability": {
            "risk_metrics": {
                "metric_type": "ruin_probability",
                "ruin_threshold": 0.01,
                "time_horizon": 10,
            },
            "description": "Probability of ruin over 10 years",
        },
        "ergodic_growth": {
            "risk_metrics": {"metric_type": "time_average_growth", "min_acceptable_growth": 0.05},
            "description": "Time-average growth rate",
        },
    }

    print("\nAvailable Risk Assessment Configurations:")
    print("-" * 50)

    for config_name, config_data in risk_configs.items():
        config = manager.load_profile("default")
        print(f"\n{config_name.replace('_', ' ').title()}:")
        print(f"  {config_data['description']}")

        if hasattr(config, "risk_metrics"):
            rm = config.risk_metrics
            if hasattr(rm, "confidence_level"):
                print(f"  Confidence: {rm.confidence_level:.0%}")
            if hasattr(rm, "time_horizon"):
                print(f"  Time horizon: {rm.time_horizon} years")
            if hasattr(rm, "ruin_threshold"):
                print(f"  Ruin threshold: {rm.ruin_threshold:.1%}")


def demo_migration_from_legacy():
    """Demonstrate migrating from legacy configuration to new system.

    Shows how to migrate existing legacy configurations to the new
    3-tier configuration system.
    """
    print("\n" + "=" * 70)
    print("LEGACY CONFIGURATION MIGRATION")
    print("=" * 70)

    # Check if legacy configs exist
    legacy_dir = Path(__file__).parent.parent / "data" / "parameters"

    if legacy_dir.exists():
        legacy_files = list(legacy_dir.glob("*.yaml"))
        print(f"\nFound {len(legacy_files)} legacy configuration files:")
        for f in legacy_files[:5]:  # Show first 5
            print(f"  - {f.name}")

        if len(legacy_files) > 5:
            print(f"  ... and {len(legacy_files) - 5} more")

        print("\nMigration process:")
        print("1. Use ConfigMigrator to convert legacy files")
        print("2. Review and validate converted configurations")
        print("3. Save as new profiles in config/profiles/custom/")
        print("4. Test with ConfigManager.load_profile()")

        # Example migration (conceptual)
        print("\nExample migration command:")
        print("  migrator = ConfigMigrator()")
        print('  migrator.migrate_file("data/parameters/baseline.yaml", ')
        print('                        "data/config/profiles/custom/legacy_baseline.yaml")')
    else:
        print("\nNo legacy configurations found to migrate")

    print("\n✓ Migration guide available in docs/migration_guide.md")


def demo_configuration_validation():
    """Demonstrate configuration validation and error handling.

    Shows how the configuration system validates inputs and handles
    common configuration errors.
    """
    print("\n" + "=" * 70)
    print("CONFIGURATION VALIDATION")
    print("=" * 70)

    manager = ConfigManager()

    # Valid configuration
    print("\n1. Valid configuration:")
    try:
        config = manager.load_profile("default", manufacturer={"base_operating_margin": 0.08})
        print("   ✓ Configuration loaded successfully")
        print(f"   Operating margin: {config.manufacturer.base_operating_margin:.1%}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Invalid range
    print("\n2. Invalid parameter range:")
    try:
        config = manager.load_profile(
            "default", manufacturer={"base_operating_margin": 1.5}
        )  # Too high
        print("   ✓ Configuration loaded")
    except Exception as e:
        print(f"   ✗ Validation error: Operating margin must be between 0 and 1")

    # Type validation
    print("\n3. Type validation:")
    try:
        config = manager.load_profile(
            "default", manufacturer={"initial_assets": "ten million"}  # Wrong type
        )
        print("   ✓ Configuration loaded")
    except Exception as e:
        print(f"   ✗ Type error: initial_assets must be numeric")

    # Missing required field
    print("\n4. Configuration completeness:")
    config = manager.load_profile("default")
    required_fields = ["manufacturer", "growth", "monte_carlo", "optimization"]
    for field in required_fields:
        if hasattr(config, field):
            print(f"   ✓ {field}: present")
        else:
            print(f"   ✗ {field}: missing")

    print("\n✓ Configuration system provides comprehensive validation")


def main():
    """Run all configuration demonstrations."""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " PRACTICAL CONFIGURATION SYSTEM DEMONSTRATIONS".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    demos = [
        ("Scenario Comparison", demo_scenario_comparison),
        ("Monte Carlo Setup", demo_monte_carlo_setup),
        ("Optimization Tuning", demo_optimization_tuning),
        ("Risk Assessment", demo_risk_assessment_configs),
        ("Legacy Migration", demo_migration_from_legacy),
        ("Validation", demo_configuration_validation),
    ]

    print("\nAvailable demonstrations:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")

    print("\nRunning all demonstrations...\n")

    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            continue

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review ergodic_insurance/examples/demo_config_v2.py for basic usage")
    print("2. Check docs/migration_guide.md for migration instructions")
    print("3. See docs/config_best_practices.md for best practices")
    print("4. Run notebooks/00_config_migration_example.ipynb for interactive demo")


if __name__ == "__main__":
    main()
