#!/usr/bin/env python
"""Demonstration of the new ConfigManager system.

This script shows how to use the new 3-tier configuration system
with profiles, modules, and presets.
"""

from ergodic_insurance import WidgetManufacturer
from ergodic_insurance.config import ConfigV2
from ergodic_insurance.config_manager import ConfigManager


def demo_basic_loading():
    """Demonstrate basic configuration loading."""
    print("\n" + "=" * 60)
    print("BASIC CONFIGURATION LOADING")
    print("=" * 60)

    manager = ConfigManager()

    # Load default profile
    config = manager.load_profile("default")
    print(f"\nLoaded profile: {config.profile.name}")
    print(f"Description: {config.profile.description}")
    print(f"Based on: {config.profile.extends or 'None'}")

    # Access configuration values
    print(f"\nManufacturer settings:")
    print(f"  Initial assets: ${config.manufacturer.initial_assets:,.0f}")
    print(f"  Operating margin: {config.manufacturer.base_operating_margin:.1%}")
    print(f"  Asset turnover: {config.manufacturer.asset_turnover_ratio:.2f}x")

    return config


def demo_profile_variants():
    """Demonstrate loading different profiles."""
    print("\n" + "=" * 60)
    print("PROFILE VARIANTS")
    print("=" * 60)

    manager = ConfigManager()
    profiles = ["default", "conservative", "aggressive"]

    for profile_name in profiles:
        config = manager.load_profile(profile_name)
        print(f"\n{profile_name.upper()} Profile:")
        print(f"  Growth rate: {config.growth.annual_growth_rate:.1%}")
        print(f"  Volatility: {config.growth.volatility:.1%}")
        print(f"  Operating margin: {config.manufacturer.base_operating_margin:.1%}")
        print(f"  Tax rate: {config.manufacturer.tax_rate:.1%}")


def demo_runtime_overrides():
    """Demonstrate runtime configuration overrides."""
    print("\n" + "=" * 60)
    print("RUNTIME OVERRIDES")
    print("=" * 60)

    manager = ConfigManager()

    # Load with overrides
    config = manager.load_profile(
        "default",
        manufacturer={"base_operating_margin": 0.12, "tax_rate": 0.30},
        growth={"annual_growth_rate": 0.08},
    )

    print("\nConfiguration with overrides:")
    print(f"  Operating margin: {config.manufacturer.base_operating_margin:.1%} (overridden)")
    print(f"  Tax rate: {config.manufacturer.tax_rate:.1%} (overridden)")
    print(f"  Growth rate: {config.growth.annual_growth_rate:.1%} (overridden)")
    print(f"  Initial assets: ${config.manufacturer.initial_assets:,.0f} (default)")


def demo_presets():
    """Demonstrate using preset libraries."""
    print("\n" + "=" * 60)
    print("PRESET LIBRARIES")
    print("=" * 60)

    manager = ConfigManager()

    # Load with market condition preset
    print("\nApplying 'hard_market' preset:")
    config = manager.load_profile("default", presets=["hard_market"])

    if hasattr(config, "insurance") and config.insurance:
        print(f"  Insurance module loaded with {len(config.insurance.layers)} layers")

    # Load with multiple presets
    print("\nApplying multiple presets ['soft_market', 'high_volatility']:")
    config = manager.load_profile("default", presets=["soft_market", "high_volatility"])

    if hasattr(config, "stochastic") and config.stochastic:
        print(f"  Revenue volatility: {config.stochastic.revenue_volatility:.1%}")
        print(f"  Mean reversion: {config.stochastic.mean_reversion_rate:.2f}")


def demo_module_composition():
    """Demonstrate selective module loading."""
    print("\n" + "=" * 60)
    print("MODULE COMPOSITION")
    print("=" * 60)

    manager = ConfigManager()

    # Load profile with specific modules only
    print("\nLoading with only 'insurance' and 'stochastic' modules:")
    config = manager.load_profile("default", modules=["insurance", "stochastic"])

    # Check what modules were loaded
    print("\nLoaded components:")
    if hasattr(config, "insurance") and config.insurance:
        print("  [OK] Insurance module loaded")
        print(f"       - Number of layers: {len(config.insurance.layers)}")

    if hasattr(config, "stochastic") and config.stochastic:
        print("  [OK] Stochastic module loaded")
        print(f"       - Process type: {config.stochastic.process_type}")

    if hasattr(config, "losses") and config.losses:
        print("  [OK] Losses module loaded")
    else:
        print("  [--] Losses module not loaded (as expected)")


def demo_config_caching():
    """Demonstrate configuration caching."""
    print("\n" + "=" * 60)
    print("CONFIGURATION CACHING")
    print("=" * 60)

    import time

    manager = ConfigManager()

    # First load (reads from disk)
    print("\nFirst load (from disk):")
    start = time.time()
    config1 = manager.load_profile("default")
    elapsed1 = time.time() - start
    print(f"  Time: {elapsed1*1000:.2f} ms")

    # Second load (from cache)
    print("\nSecond load (from cache):")
    start = time.time()
    config2 = manager.load_profile("default")
    elapsed2 = time.time() - start
    print(f"  Time: {elapsed2*1000:.2f} ms")
    print(f"  Speedup: {elapsed1/elapsed2:.1f}x")

    # Load without cache
    print("\nLoad without cache:")
    start = time.time()
    config3 = manager.load_profile("default", use_cache=False)
    elapsed3 = time.time() - start
    print(f"  Time: {elapsed3*1000:.2f} ms")

    # Verify they're equivalent
    assert config1.manufacturer.initial_assets == config2.manufacturer.initial_assets
    assert config2.manufacturer.initial_assets == config3.manufacturer.initial_assets
    print("\n  [OK] All configurations are equivalent")


def demo_validation():
    """Demonstrate configuration validation."""
    print("\n" + "=" * 60)
    print("CONFIGURATION VALIDATION")
    print("=" * 60)

    manager = ConfigManager()

    # Load valid configuration
    print("\nLoading valid configuration:")
    try:
        config = manager.load_profile("default")
        print("  [OK] Configuration is valid")
        print(f"  Type: {type(config).__name__}")
        assert isinstance(config, ConfigV2)
    except Exception as e:
        print(f"  [ERROR] {e}")

    # Try invalid override (will be caught by Pydantic)
    print("\nTrying invalid override (negative margin):")
    try:
        config = manager.load_profile(
            "default", manufacturer={"base_operating_margin": -0.5}  # Invalid negative margin
        )
        print("  [OK] Configuration accepted")
    except Exception as e:
        print(f"  [ERROR] Validation failed: {e}")


def demo_manufacturer_integration():
    """Demonstrate integration with WidgetManufacturer."""
    print("\n" + "=" * 60)
    print("MANUFACTURER INTEGRATION")
    print("=" * 60)

    manager = ConfigManager()

    # Load configuration
    config = manager.load_profile("conservative")

    # Create manufacturer
    manufacturer = WidgetManufacturer(config.manufacturer)

    print(f"\nManufacturer created with conservative profile:")
    print(f"  Initial assets: ${manufacturer.assets:,.0f}")
    print(f"  Initial equity: ${manufacturer.equity:,.0f}")
    print(f"  Base operating margin: {manufacturer.config.base_operating_margin:.1%}")
    print(f"  Tax rate: {manufacturer.config.tax_rate:.1%}")

    print("\n[Demo complete - actual simulation would require process_year method]")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print(" ConfigManager v2 Demonstration")
    print(" Showing the new 3-tier configuration system")
    print("=" * 60)

    # Run demonstrations
    demo_basic_loading()
    demo_profile_variants()
    demo_runtime_overrides()
    demo_presets()
    demo_module_composition()
    demo_config_caching()
    demo_validation()
    demo_manufacturer_integration()

    print("\n" + "=" * 60)
    print(" Demonstration Complete")
    print("=" * 60)
    print("\nThe new ConfigManager provides:")
    print("  - Simplified 3-tier architecture")
    print("  - Profile inheritance and composition")
    print("  - Preset libraries for common scenarios")
    print("  - Automatic caching for performance")
    print("  - Full backward compatibility")
    print("\nSee migration_guide.md for migration instructions.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
