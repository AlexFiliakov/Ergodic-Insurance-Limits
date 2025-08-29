"""Shared fixtures and utilities for integration tests.

This module provides reusable fixtures and configuration helpers
for integration testing across the ergodic insurance framework.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from src.claim_development import ClaimDevelopment
from src.claim_generator import ClaimEvent, ClaimGenerator
from src.config import ManufacturerConfig, SimulationConfig
from src.config_manager import ConfigManager
from src.config_v2 import ConfigV2, InsuranceConfig
from src.insurance import InsuranceLayer, InsurancePolicy
from src.insurance_program import EnhancedInsuranceLayer, InsuranceProgram
from src.loss_distributions import LossData, LossEvent, ManufacturingLossGenerator
from src.manufacturer import WidgetManufacturer
from src.monte_carlo import MonteCarloEngine
from src.stochastic_processes import (
    GeometricBrownianMotion,
    LognormalVolatility,
    MeanRevertingProcess,
)

# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def integration_test_dir(tmp_path) -> Path:
    """Create a temporary directory for integration test outputs.

    Returns:
        Path: Temporary directory path for test outputs.
    """
    test_dir = tmp_path / "integration_tests"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture
def default_config_v2() -> ConfigV2:
    """Create a default ConfigV2 for integration testing.

    Returns:
        ConfigV2: Default configuration for testing.
    """
    return ConfigV2(
        manufacturer=ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover=1.2,
            operating_margin=0.10,
            tax_rate=0.25,
            working_capital_ratio=0.20,
            growth_capex_ratio=0.05,
            maintenance_capex_ratio=0.03,
            dividend_payout_ratio=0.30,
        ),
        insurance=InsuranceConfig(
            deductible=100_000,
            primary_limit=5_000_000,
            primary_rate=0.02,
            excess_limit=10_000_000,
            excess_attachment=5_000_000,
            excess_rate=0.01,
        ),
        simulation=SimulationConfig(
            n_simulations=100,
            time_horizon=50,
            seed=42,
            confidence_levels=[0.90, 0.95, 0.99],
            enable_parallel=True,
            batch_size=20,
        ),
        stochastic=StochasticConfig(
            revenue_volatility=0.15,
            claim_frequency_mean=5.0,
            claim_severity_mean=200_000,
            claim_severity_cv=1.5,
            catastrophe_probability=0.02,
            catastrophe_severity_mean=5_000_000,
        ),
    )


@pytest.fixture
def config_manager(tmp_path) -> ConfigManager:
    """Create a ConfigManager with test configurations.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        ConfigManager: Configured manager for testing.
    """
    # Create test config directory
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)

    # Create a test profile
    profiles_dir = config_dir / "profiles"
    profiles_dir.mkdir(exist_ok=True)

    # Initialize manager with test directory
    os.environ["ERGODIC_CONFIG_DIR"] = str(config_dir)
    manager = ConfigManager()

    return manager


# ============================================================================
# Manufacturer Fixtures
# ============================================================================


@pytest.fixture
def base_manufacturer() -> WidgetManufacturer:
    """Create a base manufacturer with standard parameters.

    Returns:
        WidgetManufacturer: Standard manufacturer for testing.
    """
    config = ConfigV2()
    return WidgetManufacturer.from_config_v2(config)


@pytest.fixture
def startup_manufacturer() -> WidgetManufacturer:
    """Create a startup manufacturer (low assets, high growth potential).

    Returns:
        WidgetManufacturer: Startup configuration.
    """
    config = ConfigV2(
        manufacturer=ManufacturerConfig(
            initial_assets=1_000_000,
            asset_turnover=0.8,
            operating_margin=0.05,
            growth_capex_ratio=0.10,
        )
    )
    return WidgetManufacturer.from_config_v2(config)


@pytest.fixture
def mature_manufacturer() -> WidgetManufacturer:
    """Create a mature manufacturer (high assets, stable operations).

    Returns:
        WidgetManufacturer: Mature company configuration.
    """
    config = ConfigV2(
        manufacturer=ManufacturerConfig(
            initial_assets=50_000_000,
            asset_turnover=1.5,
            operating_margin=0.15,
            dividend_payout_ratio=0.50,
        )
    )
    return WidgetManufacturer.from_config_v2(config)


# ============================================================================
# Insurance Fixtures
# ============================================================================


@pytest.fixture
def basic_insurance_policy() -> InsurancePolicy:
    """Create a basic single-layer insurance policy.

    Returns:
        InsurancePolicy: Basic insurance for testing.
    """
    layer = InsuranceLayer(
        attachment_point=100_000,
        limit=5_000_000,
        rate=0.02,
    )
    return InsurancePolicy(layers=[layer], deductible=100_000)


@pytest.fixture
def multi_layer_insurance() -> InsurancePolicy:
    """Create a multi-layer insurance structure.

    Returns:
        InsurancePolicy: Multi-layer insurance for testing.
    """
    primary = InsuranceLayer(
        attachment_point=50_000,
        limit=2_000_000,
        rate=0.025,
    )
    excess1 = InsuranceLayer(
        attachment_point=2_050_000,
        limit=5_000_000,
        rate=0.015,
    )
    excess2 = InsuranceLayer(
        attachment_point=7_050_000,
        limit=10_000_000,
        rate=0.008,
    )
    return InsurancePolicy(
        layers=[primary, excess1, excess2],
        deductible=50_000,
    )


@pytest.fixture
def enhanced_insurance_program() -> InsuranceProgram:
    """Create an enhanced insurance program with multiple layers.

    Returns:
        InsuranceProgram: Enhanced program for advanced testing.
    """
    primary = EnhancedInsuranceLayer(
        attachment_point=100_000,
        limit=5_000_000,
        premium_rate=0.02,
        reinstatement_premium=0.01,
        aggregate_limit=15_000_000,
    )
    excess = EnhancedInsuranceLayer(
        attachment_point=5_100_000,
        limit=10_000_000,
        premium_rate=0.01,
        participation_rate=0.90,
    )
    return InsuranceProgram(layers=[primary, excess])


# ============================================================================
# Loss Generation Fixtures
# ============================================================================


@pytest.fixture
def standard_claim_generator() -> ClaimGenerator:
    """Create a standard claim generator with typical parameters.

    Returns:
        ClaimGenerator: Standard loss generator.
    """
    return ClaimGenerator(
        frequency=5.0,
        severity_mean=200_000,
        severity_std=300_000,
        seed=42,
    )


@pytest.fixture
def high_frequency_generator() -> ClaimGenerator:
    """Create a high-frequency, low-severity claim generator.

    Returns:
        ClaimGenerator: High frequency generator.
    """
    return ClaimGenerator(
        frequency=20.0,
        severity_mean=50_000,
        severity_std=30_000,
        seed=42,
    )


@pytest.fixture
def catastrophic_generator() -> ClaimGenerator:
    """Create a low-frequency, high-severity claim generator.

    Returns:
        ClaimGenerator: Catastrophic loss generator.
    """
    return ClaimGenerator(
        frequency=0.1,
        severity_mean=10_000_000,
        severity_std=5_000_000,
        seed=42,
    )


@pytest.fixture
def manufacturing_loss_generator() -> ManufacturingLossGenerator:
    """Create a manufacturing-specific loss generator.

    Returns:
        ManufacturingLossGenerator: Manufacturing loss generator.
    """
    return ManufacturingLossGenerator(
        base_frequency=3.0,
        revenue_scaling=True,
        correlation_factor=0.3,
        seed=42,
    )


@pytest.fixture
def claim_development() -> ClaimDevelopment:
    """Create a standard claim development pattern.

    Returns:
        ClaimDevelopment: Standard development pattern.
    """
    return ClaimDevelopment(
        pattern=[0.60, 0.25, 0.10, 0.05],
        tail_factor=1.02,
    )


# ============================================================================
# Stochastic Process Fixtures
# ============================================================================


@pytest.fixture
def gbm_process() -> GeometricBrownianMotion:
    """Create a Geometric Brownian Motion process.

    Returns:
        GeometricBrownianMotion: GBM for testing.
    """
    return GeometricBrownianMotion(
        drift=0.05,
        volatility=0.15,
        initial_value=1.0,
        seed=42,
    )


@pytest.fixture
def mean_reverting_process() -> MeanRevertingProcess:
    """Create a mean-reverting process.

    Returns:
        MeanRevertingProcess: Mean-reverting process for testing.
    """
    return MeanRevertingProcess(
        mean_level=1.0,
        reversion_speed=0.5,
        volatility=0.2,
        initial_value=1.0,
        seed=42,
    )


@pytest.fixture
def lognormal_volatility() -> LognormalVolatility:
    """Create a lognormal volatility process.

    Returns:
        LognormalVolatility: Lognormal volatility for testing.
    """
    return LognormalVolatility(
        volatility=0.20,
        seed=42,
    )


# ============================================================================
# Monte Carlo Engine Fixtures
# ============================================================================


@pytest.fixture
def monte_carlo_engine(default_config_v2: ConfigV2) -> MonteCarloEngine:
    """Create a Monte Carlo engine for integration testing.

    Args:
        default_config_v2: Default configuration.

    Returns:
        MonteCarloEngine: Configured engine for testing.
    """
    # Use smaller numbers for testing
    config = default_config_v2.model_copy()
    config.simulation.n_simulations = 10
    config.simulation.time_horizon = 20
    config.simulation.enable_parallel = False  # Disable for testing

    return MonteCarloEngine(
        config=config.simulation,
        manufacturer_config=config.manufacturer,
        insurance_config=config.insurance,
        stochastic_config=config.stochastic,
    )


# ============================================================================
# Test Data Generators
# ============================================================================


def generate_sample_losses(
    n_years: int = 10,
    frequency: float = 5.0,
    severity_mean: float = 200_000,
    seed: Optional[int] = None,
) -> List[ClaimEvent]:
    """Generate sample loss events for testing.

    Args:
        n_years: Number of years to generate.
        frequency: Average claims per year.
        severity_mean: Mean claim severity.
        seed: Random seed for reproducibility.

    Returns:
        List[ClaimEvent]: Generated claim events.
    """
    if seed is not None:
        np.random.seed(seed)

    claims = []
    for year in range(n_years):
        n_claims = np.random.poisson(frequency)
        for _ in range(n_claims):
            amount = np.random.lognormal(
                np.log(severity_mean),
                0.5,
            )
            claims.append(ClaimEvent(year=year, amount=amount))

    return claims


def generate_loss_data(
    duration: int = 10,
    base_frequency: float = 5.0,
    include_catastrophic: bool = True,
    seed: Optional[int] = None,
) -> LossData:
    """Generate LossData structure for testing.

    Args:
        duration: Duration in years.
        base_frequency: Base claim frequency.
        include_catastrophic: Include catastrophic losses.
        seed: Random seed.

    Returns:
        LossData: Generated loss data.
    """
    if seed is not None:
        np.random.seed(seed)

    losses = []
    for year in range(duration):
        # Regular losses
        n_regular = np.random.poisson(base_frequency)
        for _ in range(n_regular):
            losses.append(
                LossEvent(
                    timestamp=year + np.random.random(),
                    amount=np.random.lognormal(12, 1),  # ~200k mean
                    event_type="operational",
                    description=f"Loss at year {year}",
                )
            )

        # Catastrophic losses
        if include_catastrophic and np.random.random() < 0.02:
            losses.append(
                LossEvent(
                    timestamp=year + np.random.random(),
                    amount=np.random.lognormal(15, 1),  # ~5M mean
                    event_type="catastrophic",
                    description=f"Catastrophe at year {year}",
                )
            )

    return LossData(
        timestamps=np.array([loss.timestamp for loss in losses]),
        amounts=np.array([loss.amount for loss in losses]),
        event_types=[loss.event_type for loss in losses],
        descriptions=[loss.description for loss in losses],
    )


# ============================================================================
# Assertion Helpers
# ============================================================================


def assert_financial_consistency(manufacturer: WidgetManufacturer) -> None:
    """Assert that manufacturer's financial state is consistent.

    Args:
        manufacturer: Manufacturer to validate.

    Raises:
        AssertionError: If financial state is inconsistent.
    """
    # Basic balance sheet equation
    assert np.isclose(
        manufacturer.assets,
        manufacturer.equity + manufacturer.debt,
        rtol=1e-10,
    ), "Balance sheet equation violated"

    # Non-negative constraints
    assert manufacturer.assets >= 0, "Negative assets"
    assert (
        manufacturer.equity >= -1e-10
    ), "Significantly negative equity"  # Allow small numerical errors
    assert manufacturer.debt >= -1e-10, "Negative debt"
    assert manufacturer.cash >= -1e-10, "Negative cash"

    # Cash is part of assets
    assert manufacturer.cash <= manufacturer.assets * 1.01, "Cash exceeds assets"


def assert_loss_data_valid(loss_data: LossData) -> None:
    """Assert that LossData structure is valid.

    Args:
        loss_data: Loss data to validate.

    Raises:
        AssertionError: If loss data is invalid.
    """
    assert loss_data.validate(), "Loss data validation failed"
    assert len(loss_data.timestamps) == len(loss_data.amounts), "Mismatched arrays"
    assert np.all(loss_data.amounts >= 0), "Negative loss amounts"
    assert np.all(np.diff(loss_data.timestamps) >= 0), "Non-monotonic timestamps"


def assert_simulation_results_valid(results: Any) -> None:
    """Assert that simulation results are valid.

    Args:
        results: Simulation results to validate.

    Raises:
        AssertionError: If results are invalid.
    """
    assert results is not None, "Null results"
    assert hasattr(results, "years"), "Missing years attribute"
    assert hasattr(results, "equity"), "Missing equity attribute"
    assert len(results.years) > 0, "Empty results"
    assert len(results.years) == len(results.equity), "Mismatched result arrays"


# ============================================================================
# Performance Helpers
# ============================================================================


def measure_memory_usage(func, *args, **kwargs) -> Dict[str, Any]:
    """Measure memory usage of a function call.

    Args:
        func: Function to measure.
        *args: Positional arguments for func.
        **kwargs: Keyword arguments for func.

    Returns:
        Dict containing result and memory statistics.
    """
    import gc

    import psutil

    gc.collect()
    process = psutil.Process()

    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    result = func(*args, **kwargs)
    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    return {
        "result": result,
        "memory_before_mb": mem_before,
        "memory_after_mb": mem_after,
        "memory_increase_mb": mem_after - mem_before,
    }


def assert_performance_benchmark(
    elapsed_time: float,
    max_time: float,
    operation: str,
) -> None:
    """Assert that performance meets benchmark.

    Args:
        elapsed_time: Actual elapsed time.
        max_time: Maximum allowed time.
        operation: Description of operation.

    Raises:
        AssertionError: If performance benchmark not met.
    """
    assert elapsed_time <= max_time, (
        f"{operation} took {elapsed_time:.2f}s, " f"should be <= {max_time:.2f}s"
    )
