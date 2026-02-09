"""Configuration migration tools for converting legacy YAML files to new 3-tier system.

This module provides utilities to migrate from the old 12-file configuration
system to the new profiles/modules/presets architecture.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List

import yaml

try:
    from ergodic_insurance.config.utils import deep_merge_inplace as _deep_merge_inplace
except ImportError:
    try:
        from .config.utils import deep_merge_inplace as _deep_merge_inplace
    except ImportError:
        from config.utils import deep_merge_inplace as _deep_merge_inplace  # type: ignore[no-redef]


class ConfigMigrator:
    """Handles migration from legacy configuration to new 3-tier system."""

    def __init__(self):
        """Initialize the configuration migrator."""
        # Find the project root
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent  # Go up to project root
        self.legacy_dir = project_root / "ergodic_insurance" / "data" / "parameters"
        self.new_dir = project_root / "ergodic_insurance" / "data" / "config"
        self.converted_configs: Dict[str, Dict[str, Any]] = {}
        self.migration_report: List[str] = []

    def convert_baseline(self) -> Dict[str, Any]:
        """Convert baseline.yaml to new profile format.

        Returns:
            Converted configuration as a dictionary.
        """
        baseline_path = self.legacy_dir / "baseline.yaml"
        with open(baseline_path, "r") as f:
            baseline = yaml.safe_load(f)

        # Remove YAML anchors
        baseline = {k: v for k, v in baseline.items() if not k.startswith("_")}

        # Create new profile structure
        profile = {
            "profile": {
                "name": "default",
                "description": "Standard baseline configuration for widget manufacturer",
                "version": "2.0.0",
            }
        }

        # Merge baseline config
        profile.update(baseline)

        self.migration_report.append("[OK] Converted baseline.yaml to default profile")
        return profile

    def convert_conservative(self) -> Dict[str, Any]:
        """Convert conservative.yaml to new profile format.

        Returns:
            Converted configuration as a dictionary.
        """
        conservative_path = self.legacy_dir / "conservative.yaml"
        with open(conservative_path, "r") as f:
            conservative = yaml.safe_load(f)

        # Remove YAML anchors
        conservative = {k: v for k, v in conservative.items() if not k.startswith("_")}

        profile = {
            "profile": {
                "name": "conservative",
                "description": "Risk-averse configuration with higher safety margins",
                "extends": "default",
                "version": "2.0.0",
            }
        }

        # Only include differences from baseline
        profile.update(conservative)

        self.migration_report.append("[OK] Converted conservative.yaml to conservative profile")
        return profile

    def convert_optimistic(self) -> Dict[str, Any]:
        """Convert optimistic.yaml to new profile format.

        Returns:
            Converted configuration as a dictionary.
        """
        optimistic_path = self.legacy_dir / "optimistic.yaml"
        with open(optimistic_path, "r") as f:
            optimistic = yaml.safe_load(f)

        # Remove YAML anchors
        optimistic = {k: v for k, v in optimistic.items() if not k.startswith("_")}

        profile = {
            "profile": {
                "name": "aggressive",
                "description": "Growth-focused configuration with higher risk tolerance",
                "extends": "default",
                "version": "2.0.0",
            }
        }

        # Only include differences from baseline
        profile.update(optimistic)

        self.migration_report.append("[OK] Converted optimistic.yaml to aggressive profile")
        return profile

    def extract_modules(self) -> None:
        """Extract reusable modules from legacy configuration files."""
        # Insurance module
        insurance_configs = [
            "insurance.yaml",
            "insurance_market.yaml",
            "insurance_structures.yaml",
            "insurance_pricing_scenarios.yaml",
        ]

        insurance_module: Dict[str, Any] = {}
        for config_file in insurance_configs:
            path = self.legacy_dir / config_file
            if path.exists():
                with open(path, "r") as f:
                    data = yaml.safe_load(f)
                    # Deep merge the configurations
                    self._deep_merge(insurance_module, data)

        # Save insurance module
        module_path = self.new_dir / "modules" / "insurance.yaml"
        module_path.parent.mkdir(parents=True, exist_ok=True)
        with open(module_path, "w") as f:
            yaml.dump(insurance_module, f, default_flow_style=False, sort_keys=False)

        self.migration_report.append("[OK] Extracted insurance module from 4 config files")

        # Loss distributions module
        loss_configs = ["losses.yaml", "loss_distributions.yaml"]
        loss_module: Dict[str, Any] = {}
        for config_file in loss_configs:
            path = self.legacy_dir / config_file
            if path.exists():
                with open(path, "r") as f:
                    data = yaml.safe_load(f)
                    self._deep_merge(loss_module, data)

        # Save losses module
        module_path = self.new_dir / "modules" / "losses.yaml"
        with open(module_path, "w") as f:
            yaml.dump(loss_module, f, default_flow_style=False, sort_keys=False)

        self.migration_report.append("[OK] Extracted losses module from 2 config files")

        # Stochastic module
        stochastic_path = self.legacy_dir / "stochastic.yaml"
        if stochastic_path.exists():
            with open(stochastic_path, "r") as f:
                stochastic = yaml.safe_load(f)
            module_path = self.new_dir / "modules" / "stochastic.yaml"
            with open(module_path, "w") as f:
                yaml.dump(stochastic, f, default_flow_style=False, sort_keys=False)
            self.migration_report.append("[OK] Extracted stochastic module")

        # Business optimization module
        business_path = self.legacy_dir / "business_optimization.yaml"
        if business_path.exists():
            with open(business_path, "r") as f:
                business = yaml.safe_load(f)
            module_path = self.new_dir / "modules" / "business.yaml"
            with open(module_path, "w") as f:
                yaml.dump(business, f, default_flow_style=False, sort_keys=False)
            self.migration_report.append("[OK] Extracted business module")

    def create_presets(self) -> None:
        """Generate preset libraries from existing configurations."""
        # Market conditions presets
        market_presets = {
            "stable": {
                "manufacturer": {"revenue_volatility": 0.10, "cost_volatility": 0.08},
                "simulation": {"stochastic_revenue": False},
            },
            "volatile": {
                "manufacturer": {"revenue_volatility": 0.25, "cost_volatility": 0.20},
                "simulation": {"stochastic_revenue": True, "stochastic_costs": True},
            },
            "growth": {
                "manufacturer": {"revenue_growth": 0.15, "market_expansion": 0.10},
                "growth": {"annual_growth_rate": 0.12},
            },
            "recession": {
                "manufacturer": {"revenue_growth": -0.05, "cost_inflation": 0.08},
                "growth": {"annual_growth_rate": -0.02},
            },
        }

        preset_path = self.new_dir / "presets" / "market_conditions.yaml"
        preset_path.parent.mkdir(parents=True, exist_ok=True)
        with open(preset_path, "w") as f:
            yaml.dump(market_presets, f, default_flow_style=False, sort_keys=False)

        self.migration_report.append("[OK] Created market_conditions presets")

        # Layer structures presets
        layer_presets = {
            "basic": {
                "insurance": {
                    "layers": [
                        {
                            "name": "Primary",
                            "limit": 5_000_000,
                            "attachment": 0,
                            "base_premium_rate": 0.015,
                        }
                    ]
                }
            },
            "comprehensive": {
                "insurance": {
                    "layers": [
                        {
                            "name": "Primary",
                            "limit": 5_000_000,
                            "attachment": 0,
                            "base_premium_rate": 0.015,
                        },
                        {
                            "name": "Excess",
                            "limit": 20_000_000,
                            "attachment": 5_000_000,
                            "base_premium_rate": 0.008,
                        },
                        {
                            "name": "Umbrella",
                            "limit": 25_000_000,
                            "attachment": 25_000_000,
                            "base_premium_rate": 0.004,
                        },
                    ]
                }
            },
            "catastrophic": {
                "insurance": {
                    "layers": [
                        {
                            "name": "High Deductible",
                            "limit": 45_000_000,
                            "attachment": 5_000_000,
                            "base_premium_rate": 0.006,
                        }
                    ]
                }
            },
        }

        preset_path = self.new_dir / "presets" / "layer_structures.yaml"
        with open(preset_path, "w") as f:
            yaml.dump(layer_presets, f, default_flow_style=False, sort_keys=False)

        self.migration_report.append("[OK] Created layer_structures presets")

        # Risk scenarios presets
        risk_presets = {
            "low_risk": {
                "losses": {"frequency_annual": 2.0, "severity_mean": 50_000, "severity_std": 25_000}
            },
            "moderate_risk": {
                "losses": {
                    "frequency_annual": 5.0,
                    "severity_mean": 250_000,
                    "severity_std": 150_000,
                }
            },
            "high_risk": {
                "losses": {
                    "frequency_annual": 8.0,
                    "severity_mean": 1_000_000,
                    "severity_std": 750_000,
                }
            },
        }

        preset_path = self.new_dir / "presets" / "risk_scenarios.yaml"
        with open(preset_path, "w") as f:
            yaml.dump(risk_presets, f, default_flow_style=False, sort_keys=False)

        self.migration_report.append("[OK] Created risk_scenarios presets")

    def validate_migration(self) -> bool:
        """Validate that all configurations were successfully migrated.

        Returns:
            True if validation passes, False otherwise.
        """
        # Check that all profile files exist
        profiles = ["default.yaml", "conservative.yaml", "aggressive.yaml"]
        for profile in profiles:
            path = self.new_dir / "profiles" / profile
            if not path.exists():
                self.migration_report.append(f"[ERROR] Missing profile: {profile}")
                return False

        # Check that key modules exist
        modules = ["insurance.yaml", "losses.yaml", "stochastic.yaml", "business.yaml"]
        for module in modules:
            path = self.new_dir / "modules" / module
            if not path.exists():
                self.migration_report.append(f"[ERROR] Missing module: {module}")
                return False

        # Check that preset libraries exist
        presets = ["market_conditions.yaml", "layer_structures.yaml", "risk_scenarios.yaml"]
        for preset in presets:
            path = self.new_dir / "presets" / preset
            if not path.exists():
                self.migration_report.append(f"[ERROR] Missing preset: {preset}")
                return False

        self.migration_report.append("[OK] All configurations successfully migrated")
        return True

    def generate_migration_report(self) -> str:
        """Generate a detailed migration report.

        Returns:
            Formatted migration report as a string.
        """
        report = ["=" * 60]
        report.append("Configuration Migration Report")
        report.append("=" * 60)
        report.extend(self.migration_report)
        report.append("=" * 60)
        report.append(f"Total items processed: {len(self.migration_report)}")
        return "\n".join(report)

    def _deep_merge(self, target: Dict, source: Dict) -> None:
        """Deep merge source dictionary into target.

        Args:
            target: Target dictionary to merge into.
            source: Source dictionary to merge from.
        """
        _deep_merge_inplace(target, source)

    def run_migration(self) -> bool:
        """Run the complete migration process.

        Returns:
            True if migration successful, False otherwise.
        """
        try:
            # Convert main profiles
            default_profile = self.convert_baseline()
            profile_path = self.new_dir / "profiles" / "default.yaml"
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            with open(profile_path, "w") as f:
                yaml.dump(default_profile, f, default_flow_style=False, sort_keys=False)

            conservative_profile = self.convert_conservative()
            profile_path = self.new_dir / "profiles" / "conservative.yaml"
            with open(profile_path, "w") as f:
                yaml.dump(conservative_profile, f, default_flow_style=False, sort_keys=False)

            aggressive_profile = self.convert_optimistic()
            profile_path = self.new_dir / "profiles" / "aggressive.yaml"
            with open(profile_path, "w") as f:
                yaml.dump(aggressive_profile, f, default_flow_style=False, sort_keys=False)

            # Extract modules
            self.extract_modules()

            # Create presets
            self.create_presets()

            # Validate
            return self.validate_migration()

        except (FileNotFoundError, ValueError, KeyError, yaml.YAMLError) as e:
            self.migration_report.append(f"[ERROR] Migration failed: {str(e)}")
            return False


if __name__ == "__main__":
    # Run the migration when executed directly
    migrator = ConfigMigrator()
    success = migrator.run_migration()
    print(migrator.generate_migration_report())

    if not success:
        print("\n⚠️  Migration completed with errors. Please review the report above.")
        sys.exit(1)
    else:
        print("\n✓ Migration completed successfully!")
        sys.exit(0)
