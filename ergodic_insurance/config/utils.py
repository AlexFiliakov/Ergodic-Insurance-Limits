"""Shared configuration utilities.

Contains common helper functions used across the configuration system
to avoid duplication.

Since:
    Version 0.10.0 (Issue #306)
"""

from typing import Any, Dict


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dictionary into base dictionary.

    Creates a new dictionary with values from ``base`` updated by ``override``.
    Nested dictionaries are merged recursively rather than replaced wholesale.

    Args:
        base: Base dictionary providing default values.
        override: Override dictionary whose values take precedence.

    Returns:
        New merged dictionary (neither input is mutated).
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def deep_merge_inplace(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Recursively merge source dictionary into target dictionary in-place.

    Mutates ``target`` by applying values from ``source``.  Nested
    dictionaries are merged recursively rather than replaced wholesale.

    Args:
        target: Target dictionary to merge into (mutated).
        source: Source dictionary to merge from (not mutated).
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            deep_merge_inplace(target[key], value)
        else:
            target[key] = value
