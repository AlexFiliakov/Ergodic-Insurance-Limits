#!/usr/bin/env python
"""Script to migrate configurations from legacy to new 3-tier system."""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ergodic_insurance.config_manager import ConfigManager
from ergodic_insurance.config_migrator import ConfigMigrator


def main():
    """Run the configuration migration."""
    parser = argparse.ArgumentParser(
        description="Migrate legacy configurations to new 3-tier system"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate migration after completion"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--test-load", action="store_true", help="Test loading configurations after migration"
    )

    args = parser.parse_args()

    # Create migrator
    migrator = ConfigMigrator()

    if args.dry_run:
        print("DRY RUN MODE - No changes will be made\n")
        print("Would perform the following migrations:")
        print("- Convert baseline.yaml to profiles/default.yaml")
        print("- Convert conservative.yaml to profiles/conservative.yaml")
        print("- Convert optimistic.yaml to profiles/aggressive.yaml")
        print("- Extract insurance, losses, stochastic, and business modules")
        print("- Create market_conditions, layer_structures, and risk_scenarios presets")
        return 0

    # Run migration
    print("Starting configuration migration...")
    success = migrator.run_migration()

    # Print report
    print("\n" + migrator.generate_migration_report())

    if not success:
        print("\n[FAILED] Migration failed. Please check the report above.")
        return 1

    # Validate if requested
    if args.validate or args.test_load:
        print("\n" + "=" * 60)
        print("Testing new configuration system...")
        print("=" * 60)

        try:
            manager = ConfigManager()

            # Test loading each profile
            profiles = ["default", "conservative", "aggressive"]
            for profile_name in profiles:
                config = manager.load_profile(profile_name)
                print(f"[OK] Loaded profile: {profile_name}")

                # Validate
                issues = manager.validate(config)
                if issues:
                    print(f"  [WARNING] Validation issues: {', '.join(issues)}")
                else:
                    print(f"  [OK] Validation passed")

            # Test listing functions
            available_profiles = manager.list_profiles()
            print(f"\n[OK] Available profiles: {', '.join(available_profiles)}")

            available_modules = manager.list_modules()
            print(f"[OK] Available modules: {', '.join(available_modules)}")

            available_presets = manager.list_presets()
            print(f"[OK] Available preset types: {', '.join(available_presets.keys())}")

            print("\n[SUCCESS] Configuration system test successful!")

        except Exception as e:
            print(f"\n[FAILED] Configuration test failed: {e}")
            return 1

    print("\n[SUCCESS] Migration completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
