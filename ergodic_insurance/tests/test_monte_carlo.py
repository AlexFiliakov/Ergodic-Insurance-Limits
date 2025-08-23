"""Unit tests for Monte Carlo simulation engine.

This module contains comprehensive tests for the memory-efficient
Monte Carlo engine including batch processing, parallelization,
checkpointing, and streaming statistics.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ergodic_insurance.src.config import Config, ManufacturerConfig
from ergodic_insurance.src.config_loader import load_config
from ergodic_insurance.src.insurance import InsuranceLayer, InsurancePolicy
from ergodic_insurance.src.monte_carlo import (
    MonteCarloCheckpoint,
    MonteCarloEngine,
    StreamingStatistics,
    run_single_scenario,
)
from ergodic_insurance.src.simulation import Simulation


class TestStreamingStatistics:
    """Test suite for streaming statistics calculator."""

    def test_initialization(self):
        """Test StreamingStatistics initialization."""
        stats = StreamingStatistics()
        assert stats.count == 0
        assert stats.mean == 0.0
        assert stats.m2 == 0.0
        assert stats.min_val == float("inf")
        assert stats.max_val == float("-inf")
        assert stats.survival_count == 0

    def test_single_update(self):
        """Test updating with single value."""
        stats = StreamingStatistics()
        stats.update(10.0, survived=True)

        assert stats.count == 1
        assert stats.mean == 10.0
        assert stats.min_val == 10.0
        assert stats.max_val == 10.0
        assert stats.survival_count == 1

    def test_multiple_updates(self):
        """Test updating with multiple values."""
        stats = StreamingStatistics()
        values = [10.0, 20.0, 30.0, 40.0, 50.0]

        for val in values:
            stats.update(val, survived=True)

        assert stats.count == 5
        assert stats.mean == 30.0
        assert stats.min_val == 10.0
        assert stats.max_val == 50.0
        assert stats.survival_count == 5
        assert stats.std == pytest.approx(15.811, rel=1e-3)

    def test_geometric_mean(self):
        """Test geometric mean calculation."""
        stats = StreamingStatistics()
        values = [1.1, 1.2, 1.15, 1.05]  # Growth rates

        for val in values:
            stats.update(val, survived=True)

        expected_geo_mean = np.exp(np.mean(np.log(values)))
        assert stats.geometric_mean == pytest.approx(expected_geo_mean)

    def test_survival_rate(self):
        """Test survival rate calculation."""
        stats = StreamingStatistics()

        # 3 survived, 2 failed
        stats.update(100, survived=True)
        stats.update(150, survived=True)
        stats.update(0, survived=False)
        stats.update(200, survived=True)
        stats.update(0, survived=False)

        assert stats.survival_rate == 0.6

    def test_percentiles(self):
        """Test percentile calculation from reservoir."""
        stats = StreamingStatistics()
        np.random.seed(42)

        # Add 100 values
        values = np.random.normal(100, 20, 100)
        for val in values:
            stats.update(val, survived=True)

        # Check percentiles are reasonable
        assert 80 < stats.percentile(25) < 95
        assert 95 < stats.percentile(50) < 105
        assert 105 < stats.percentile(75) < 120

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = StreamingStatistics()
        stats.update(10.0, survived=True)
        stats.update(20.0, survived=True)

        result = stats.to_dict()
        assert "count" in result
        assert "mean" in result
        assert "std" in result
        assert "geometric_mean" in result
        assert "survival_rate" in result
        assert result["count"] == 2
        assert result["mean"] == 15.0


class TestMonteCarloCheckpoint:
    """Test suite for checkpoint functionality."""

    def test_checkpoint_save_load(self, tmp_path):
        """Test saving and loading checkpoints."""
        # Create checkpoint
        stats = {"metric1": StreamingStatistics(), "metric2": StreamingStatistics()}
        stats["metric1"].update(10.0)
        stats["metric2"].update(20.0)

        checkpoint = MonteCarloCheckpoint(
            scenario_start=0, scenario_end=100, statistics=stats, timestamp=12345.0
        )

        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.parquet"
        checkpoint.save(checkpoint_path)

        assert checkpoint_path.exists()

        # Load checkpoint
        loaded = MonteCarloCheckpoint.load(checkpoint_path)

        assert loaded.scenario_start == 0
        assert loaded.scenario_end == 100
        assert loaded.timestamp == 12345.0
        assert "metric1" in loaded.statistics
        assert "metric2" in loaded.statistics


class TestRunSingleScenario:
    """Test suite for single scenario execution."""

    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return Config(
            manufacturer=ManufacturerConfig(
                initial_assets=10_000_000,
                asset_turnover_ratio=1.0,
                operating_margin=0.08,
                tax_rate=0.25,
                retention_ratio=1.0,
            ),
            **{
                "working_capital": {"percent_of_sales": 0.2},
                "growth": {"type": "deterministic", "annual_growth_rate": 0.05, "volatility": 0.0},
                "debt": {
                    "interest_rate": 0.015,
                    "max_leverage_ratio": 2.0,
                    "minimum_cash_balance": 100_000,
                },
                "simulation": {
                    "time_resolution": "annual",
                    "time_horizon_years": 10,
                    "max_horizon_years": 1000,
                    "random_seed": 42,
                },
                "output": {
                    "output_directory": "outputs",
                    "file_format": "csv",
                    "checkpoint_frequency": 10,
                    "detailed_metrics": True,
                },
                "logging": {
                    "enabled": False,
                    "level": "INFO",
                    "log_file": None,
                    "console_output": False,
                    "format": "%(message)s",
                },
            },
        )

    @pytest.fixture
    def test_policy(self):
        """Create test insurance policy."""
        layers = [
            InsuranceLayer(attachment_point=500_000, limit=4_500_000, rate=0.015),
            InsuranceLayer(attachment_point=5_000_000, limit=20_000_000, rate=0.008),
        ]
        return InsurancePolicy(layers=layers, deductible=500_000)

    def test_run_single_scenario_basic(self, test_config, test_policy):
        """Test running a single scenario."""
        result = run_single_scenario(
            scenario_id=0,
            config=test_config,
            insurance_policy=test_policy,
            time_horizon=10,
            seed=42,
        )

        assert "scenario_id" in result
        assert "survived" in result
        assert "final_equity" in result
        assert "geometric_return" in result
        assert result["scenario_id"] == 0

    def test_run_single_scenario_with_seed(self, test_config, test_policy):
        """Test scenario reproducibility with seed."""
        result1 = run_single_scenario(
            scenario_id=0,
            config=test_config,
            insurance_policy=test_policy,
            time_horizon=10,
            seed=42,
        )

        result2 = run_single_scenario(
            scenario_id=0,
            config=test_config,
            insurance_policy=test_policy,
            time_horizon=10,
            seed=42,
        )

        # Results should be identical with same seed
        assert result1["final_equity"] == result2["final_equity"]
        assert result1["geometric_return"] == result2["geometric_return"]


class TestMonteCarloEngine:
    """Test suite for Monte Carlo engine."""

    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return load_config("baseline")

    @pytest.fixture
    def test_policy(self):
        """Create test insurance policy."""
        layers = [InsuranceLayer(attachment_point=500_000, limit=4_500_000, rate=0.015)]
        return InsurancePolicy(layers=layers, deductible=500_000)

    def test_engine_initialization(self, test_config, test_policy, tmp_path):
        """Test Monte Carlo engine initialization."""
        engine = MonteCarloEngine(
            config=test_config,
            insurance_policy=test_policy,
            n_scenarios=100,
            batch_size=10,
            n_jobs=2,
            checkpoint_dir=tmp_path,
            seed=42,
        )

        assert engine.n_scenarios == 100
        assert engine.batch_size == 10
        assert engine.n_jobs == 2
        assert engine.checkpoint_dir == tmp_path

    def test_run_batch(self, test_config, test_policy, tmp_path):
        """Test running a batch of scenarios."""
        engine = MonteCarloEngine(
            config=test_config,
            insurance_policy=test_policy,
            n_scenarios=10,
            batch_size=5,
            n_jobs=2,
            checkpoint_dir=tmp_path,
            seed=42,
        )

        results = engine.run_batch(0, 5)

        assert len(results) == 5
        assert all("scenario_id" in r for r in results)
        assert all("survived" in r for r in results)

    def test_update_statistics(self, test_config, test_policy, tmp_path):
        """Test updating streaming statistics."""
        engine = MonteCarloEngine(
            config=test_config,
            insurance_policy=test_policy,
            n_scenarios=10,
            batch_size=5,
            n_jobs=2,
            checkpoint_dir=tmp_path,
        )

        # Create mock results
        results = [
            {
                "survived": True,
                "final_equity": 15_000_000,
                "final_assets": 15_000_000,
                "geometric_return": 0.05,
                "arithmetic_return": 0.06,
                "years_survived": 10,
            },
            {
                "survived": False,
                "final_equity": 0,
                "final_assets": 0,
                "geometric_return": -0.1,
                "arithmetic_return": -0.08,
                "years_survived": 5,
            },
        ]

        engine.update_statistics(results)

        assert engine.statistics["final_equity"].count == 2
        assert engine.statistics["final_equity"].survival_count == 1

    def test_checkpoint_save_load(self, test_config, test_policy, tmp_path):
        """Test checkpoint saving and loading."""
        engine = MonteCarloEngine(
            config=test_config,
            insurance_policy=test_policy,
            n_scenarios=10,
            checkpoint_dir=tmp_path,
        )

        # Update some statistics
        results = [
            {
                "survived": True,
                "final_equity": 15_000_000,
                "final_assets": 15_000_000,
                "geometric_return": 0.05,
                "arithmetic_return": 0.06,
                "years_survived": 10,
            }
        ]
        engine.update_statistics(results)

        # Save checkpoint
        checkpoint_path = engine.save_checkpoint(5)
        assert checkpoint_path.exists()

        # Create new engine and load checkpoint
        new_engine = MonteCarloEngine(
            config=test_config,
            insurance_policy=test_policy,
            n_scenarios=10,
            checkpoint_dir=tmp_path,
        )

        loaded_scenarios = new_engine.load_checkpoint(checkpoint_path)
        assert loaded_scenarios == 5
        assert new_engine.statistics["final_equity"].count == 1

    def test_find_latest_checkpoint(self, test_config, test_policy, tmp_path):
        """Test finding the latest checkpoint."""
        engine = MonteCarloEngine(
            config=test_config,
            insurance_policy=test_policy,
            n_scenarios=10,
            checkpoint_dir=tmp_path,
        )

        # Save multiple checkpoints
        engine.save_checkpoint(100)
        engine.save_checkpoint(200)
        engine.save_checkpoint(150)

        latest = engine.find_latest_checkpoint()
        assert latest is not None
        assert "checkpoint_000200" in str(latest)

    @pytest.mark.slow
    def test_run_small_simulation(self, test_config, test_policy, tmp_path):
        """Test running a small Monte Carlo simulation."""
        engine = MonteCarloEngine(
            config=test_config,
            insurance_policy=test_policy,
            n_scenarios=20,
            batch_size=5,
            n_jobs=2,
            checkpoint_dir=tmp_path,
            checkpoint_frequency=10,
            seed=42,
        )

        results = engine.run(resume=False)

        assert results["n_scenarios"] == 20
        assert "statistics" in results
        assert "final_equity" in results["statistics"]
        assert results["statistics"]["final_equity"]["count"] == 20

    def test_results_dataframe(self, test_config, test_policy, tmp_path):
        """Test loading results as DataFrame."""
        engine = MonteCarloEngine(
            config=test_config,
            insurance_policy=test_policy,
            n_scenarios=10,
            batch_size=10,
            n_jobs=2,
            checkpoint_dir=tmp_path,
            checkpoint_frequency=10,
            seed=42,
        )

        # Run simulation
        engine.run(resume=False)

        # Get results DataFrame
        df = engine.get_results_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert "scenario_id" in df.columns
        assert "survived" in df.columns


class TestSimulationIntegration:
    """Test Monte Carlo integration with Simulation class."""

    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return load_config("baseline")

    @pytest.fixture
    def test_policy(self):
        """Create test insurance policy."""
        return InsurancePolicy.from_yaml(
            str(Path(__file__).parent.parent / "data" / "parameters" / "insurance.yaml")
        )

    @pytest.mark.slow
    def test_run_monte_carlo_class_method(self, test_config, test_policy, tmp_path):
        """Test running Monte Carlo via Simulation class method."""
        results = Simulation.run_monte_carlo(
            config=test_config,
            insurance_policy=test_policy,
            n_scenarios=10,
            batch_size=5,
            n_jobs=2,
            checkpoint_dir=tmp_path,
            seed=42,
            resume=False,
        )

        assert "statistics" in results
        assert "ergodic_analysis" in results
        assert results["n_scenarios"] == 10

    @pytest.mark.slow
    def test_compare_insurance_strategies(self, test_config, tmp_path):
        """Test comparing multiple insurance strategies."""
        # Create different policies
        policies = {
            "Low Coverage": InsurancePolicy(
                layers=[InsuranceLayer(attachment_point=1_000_000, limit=4_000_000, rate=0.01)],
                deductible=1_000_000,
            ),
            "Medium Coverage": InsurancePolicy(
                layers=[InsuranceLayer(attachment_point=500_000, limit=9_500_000, rate=0.015)],
                deductible=500_000,
            ),
            "High Coverage": InsurancePolicy(
                layers=[
                    InsuranceLayer(attachment_point=250_000, limit=4_750_000, rate=0.02),
                    InsuranceLayer(attachment_point=5_000_000, limit=20_000_000, rate=0.01),
                ],
                deductible=250_000,
            ),
        }

        comparison = Simulation.compare_insurance_strategies(
            config=test_config,
            insurance_policies=policies,
            n_scenarios=10,
            n_jobs=2,
            seed=42,
        )

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 3
        assert "survival_rate" in comparison.columns
        assert "geometric_return" in comparison.columns
        assert "annual_premium" in comparison.columns
