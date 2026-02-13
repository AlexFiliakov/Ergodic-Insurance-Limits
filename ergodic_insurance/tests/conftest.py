"""Pytest configuration and shared fixtures."""

import os
from pathlib import Path
import sys

import matplotlib
import pytest

# pylint: disable=wrong-import-position
# We need to configure matplotlib before importing test fixtures

# Set matplotlib to use non-interactive backend for testing
matplotlib.use("Agg")

# Import all fixtures from integration test_fixtures to make them available globally
from ergodic_insurance.tests.integration.test_fixtures import (
    base_manufacturer,
    basic_insurance_policy,
    catastrophic_loss_generator,
    claim_development,
    config_manager,
    default_config_v2,
    enhanced_insurance_program,
    gbm_process,
    high_frequency_loss_generator,
    integration_test_dir,
    lognormal_volatility,
    manufacturing_loss_generator,
    mature_manufacturer,
    mean_reverting_process,
    monte_carlo_engine,
    multi_layer_insurance,
    standard_loss_generator,
    startup_manufacturer,
)

# pylint: enable=wrong-import-position


def pytest_collection_modifyitems(config, items):
    """Auto-skip requires_multiprocessing and benchmark tests in CI."""
    if os.environ.get("CI"):
        skip_mp = pytest.mark.skip(
            reason="Multiprocessing/shared memory tests crash xdist workers in CI"
        )
        skip_bench = pytest.mark.skip(
            reason="Benchmark tests skipped in CI (variable performance on shared runners)"
        )
        for item in items:
            if "requires_multiprocessing" in item.keywords:
                item.add_marker(skip_mp)
            if "benchmark" in item.keywords:
                item.add_marker(skip_bench)


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Auto-close all matplotlib figures after each test to prevent memory leaks."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def test_data_dir(project_root):
    """Return the test data directory."""
    return project_root / "tests" / "test_data"


@pytest.fixture
def parameters_dir(project_root):
    """Return the parameters directory."""
    return project_root / "data" / "parameters"
