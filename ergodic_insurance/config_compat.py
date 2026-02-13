"""Backward compatibility stub — this module is deprecated.

The Config and ConfigV2 classes have been unified into a single ``Config``
class (Issue #638).  All functionality previously provided by this module
(LegacyConfigAdapter, ConfigTranslator, load_config, migrate_config_usage)
is no longer needed.

Use ``ergodic_insurance.config.Config`` directly, or
``ergodic_insurance.config_manager.ConfigManager`` for profile management.

.. deprecated:: 0.10.0
    This module will be removed in a future version.
"""

import warnings

warnings.warn(
    "The config_compat module is deprecated and will be removed in a future version. "
    "Use ergodic_insurance.config.Config directly — it now includes all ConfigV2 features. "
    "Use ergodic_insurance.config_manager.ConfigManager for profile management.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export names for code that still imports from here
try:
    from ergodic_insurance.config import Config
    from ergodic_insurance.config_manager import ConfigManager
except ImportError:
    try:
        from .config import Config
        from .config_manager import ConfigManager
    except ImportError:
        from config import Config  # type: ignore[no-redef]
        from config_manager import ConfigManager  # type: ignore[no-redef]

# Deprecated aliases
ConfigV2 = Config
LegacyConfigAdapter = None
ConfigTranslator = None


def load_config(config_name="baseline", override_params=None):
    """Deprecated. Use ConfigManager.load_profile() instead."""
    warnings.warn(
        "config_compat.load_config() is deprecated. " "Use ConfigManager().load_profile() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    manager = ConfigManager()
    overrides = override_params or {}
    return manager.load_profile(config_name, **overrides)


def migrate_config_usage(file_path):
    """Deprecated. No longer needed — Config and ConfigV2 are unified."""
    warnings.warn(
        "migrate_config_usage() is deprecated and no longer needed. "
        "Config and ConfigV2 are now the same class.",
        DeprecationWarning,
        stacklevel=2,
    )
