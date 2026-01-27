"""Pytest configuration and shared fixtures."""

from pathlib import Path
import sys

import matplotlib
import pytest

# pylint: disable=wrong-import-position
# We need to configure matplotlib and update sys.path before importing test fixtures

# Set matplotlib to use non-interactive backend for testing
matplotlib.use("Agg")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
