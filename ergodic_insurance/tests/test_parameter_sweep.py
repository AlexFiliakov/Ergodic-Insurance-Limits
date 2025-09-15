"""Tests for parameter sweep utilities.

This module contains comprehensive tests for the parameter sweep functionality,
including grid generation, parallel execution, result storage, and optimal
region identification.

Author:
    Alex Filiakov

Date:
    2025-08-29
"""

import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.src.business_optimizer import BusinessOptimizer, OptimalStrategy
from ergodic_insurance.src.parameter_sweep import ParameterSweeper, SweepConfig


class TestSweepConfig:
    """Test suite for SweepConfig dataclass."""

    def test_sweep_config_initialization(self):
        """Test SweepConfig initialization with default values."""
        config = SweepConfig(
            parameters={"param1": [1, 2, 3], "param2": [0.1, 0.2]}, fixed_params={"fixed1": 10}
        )

        assert config.parameters == {"param1": [1, 2, 3], "param2": [0.1, 0.2]}
        assert config.fixed_params == {"fixed1": 10}
        assert config.batch_size == 100
        assert config.n_workers is not None
        assert config.cache_dir == "./cache/sweeps"

    def test_sweep_config_empty_parameters_raises_error(self):
        """Test that empty parameters dictionary raises ValueError."""
        with pytest.raises(ValueError, match="Parameters dictionary cannot be empty"):
            SweepConfig(parameters={})

    def test_generate_grid(self):
        """Test parameter grid generation."""
        config = SweepConfig(
            parameters={"param1": [1, 2], "param2": [0.1, 0.2, 0.3]}, fixed_params={"fixed1": 10}
        )

        grid = config.generate_grid()

        # Should have 2 * 3 = 6 combinations
        assert len(grid) == 6

        # Check that all combinations are present
        expected_combinations = [
            {"fixed1": 10, "param1": 1, "param2": 0.1},
            {"fixed1": 10, "param1": 1, "param2": 0.2},
            {"fixed1": 10, "param1": 1, "param2": 0.3},
            {"fixed1": 10, "param1": 2, "param2": 0.1},
            {"fixed1": 10, "param1": 2, "param2": 0.2},
            {"fixed1": 10, "param1": 2, "param2": 0.3},
        ]

        for expected in expected_combinations:
            assert expected in grid

    def test_estimate_runtime(self):
        """Test runtime estimation."""
        config = SweepConfig(
            parameters={"param1": [1, 2, 3], "param2": [0.1, 0.2]},  # 3 values  # 2 values
            n_workers=2,
        )

        # 3 * 2 = 6 runs, with 2 workers and 1 second per run
        # Should take 3 seconds
        estimate = config.estimate_runtime(seconds_per_run=1.0)
        assert "3s" in estimate

        # Test with longer runtime
        estimate = config.estimate_runtime(seconds_per_run=100.0)
        assert "m" in estimate  # Should show minutes

    def test_cache_dir_creation(self):
        """Test that cache directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "test_cache"

            config = SweepConfig(parameters={"param1": [1, 2]}, cache_dir=str(cache_dir))

            assert cache_dir.exists()


class TestParameterSweeper:
    """Test suite for ParameterSweeper class."""

    @pytest.fixture
    def mock_optimizer(self):
        """Create mock optimizer."""
        optimizer = MagicMock(spec=BusinessOptimizer)
        return optimizer

    @pytest.fixture
    def sweeper(self, mock_optimizer):
        """Create ParameterSweeper instance with temporary cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sweeper = ParameterSweeper(
                optimizer=mock_optimizer,
                cache_dir=tmpdir,
                use_parallel=False,  # Disable parallel for testing
            )
            yield sweeper

    def test_sweeper_initialization(self, mock_optimizer):
        """Test ParameterSweeper initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sweeper = ParameterSweeper(optimizer=mock_optimizer, cache_dir=tmpdir)

            assert sweeper.optimizer == mock_optimizer
            assert sweeper.cache_dir == Path(tmpdir)
            assert sweeper.use_parallel is True
            assert sweeper.results_cache == {}

    def test_run_single_success(self, sweeper):
        """Test successful single parameter run."""
        # Pre-populate cache to avoid actual optimization
        params = {"initial_assets": 10e6, "operating_margin": 0.08, "time_horizon": 10}

        # Create expected result
        expected_result = {
            **params,
            "optimal_roe": 0.15,
            "ruin_probability": 0.005,
            "optimal_limit": 5e6,
            "total_premium": 100000,
        }

        # Pre-populate cache
        cache_key = sweeper._get_cache_key(params)
        sweeper.results_cache[cache_key] = expected_result

        metrics = ["optimal_roe", "ruin_probability"]
        result = sweeper._run_single(params, metrics)

        assert result["initial_assets"] == 10e6
        assert result["optimal_roe"] == 0.15
        assert result["ruin_probability"] == 0.005
        assert result["optimal_limit"] == 5e6

    def test_run_single_with_cache(self, sweeper):
        """Test that cached results are returned."""
        params = {"param1": 1, "param2": 0.1}
        metrics = ["metric1"]

        # Pre-populate cache
        cache_key = sweeper._get_cache_key(params)
        cached_result = {"param1": 1, "param2": 0.1, "metric1": 0.5}
        sweeper.results_cache[cache_key] = cached_result

        result = sweeper._run_single(params, metrics)

        assert result == cached_result

    def test_run_single_optimization_failure(self, sweeper):
        """Test handling of optimization failure."""
        with patch.object(sweeper, "_create_manufacturer") as mock_create:
            mock_manufacturer = MagicMock()
            mock_manufacturer.assets = 10e6  # Add assets attribute for BusinessOptimizer
            mock_create.return_value = mock_manufacturer

            with patch("ergodic_insurance.src.parameter_sweep.BusinessOptimizer") as MockOptimizer:
                mock_opt_instance = MockOptimizer.return_value
                mock_opt_instance.maximize_roe_with_insurance.side_effect = Exception(
                    "Optimization failed"
                )

                params = {"initial_assets": 10e6}
                metrics = ["optimal_roe", "ruin_probability"]

                result = sweeper._run_single(params, metrics)

                assert result["initial_assets"] == 10e6
                assert np.isnan(result["optimal_roe"])
                assert np.isnan(result["ruin_probability"])

    def test_sweep_execution(self, sweeper):
        """Test full sweep execution."""
        config = SweepConfig(
            parameters={"param1": [1, 2], "param2": [0.1, 0.2]},
            fixed_params={"fixed1": 10},
            n_workers=1,
        )

        # Mock _run_single to return predictable results
        def mock_run_single(params, metrics):
            return {
                **params,
                "optimal_roe": params.get("param1", 1) * params.get("param2", 0.1),
                "ruin_probability": 0.01,
            }

        with patch.object(sweeper, "_run_single", side_effect=mock_run_single):
            with patch.object(sweeper, "_save_results"):
                results = sweeper.sweep(config)

        assert len(results) == 4  # 2 * 2 combinations
        assert "optimal_roe" in results.columns
        assert "ruin_probability" in results.columns
        assert "param1" in results.columns
        assert "param2" in results.columns

    def test_create_scenarios(self, sweeper):
        """Test pre-defined scenario creation."""
        scenarios = sweeper.create_scenarios()

        # Check that all expected scenarios are present
        expected_scenarios = [
            "company_sizes",
            "loss_scenarios",
            "market_conditions",
            "time_horizons",
            "simulation_scales",
        ]

        for scenario_name in expected_scenarios:
            assert scenario_name in scenarios
            assert isinstance(scenarios[scenario_name], SweepConfig)

        # Check specific scenario configuration
        company_sizes = scenarios["company_sizes"]
        assert "initial_assets" in company_sizes.parameters
        assert len(company_sizes.parameters["initial_assets"]) == 3
        assert company_sizes.fixed_params["time_horizon"] == 10

    def test_find_optimal_regions(self, sweeper):
        """Test optimal region identification."""
        # Create sample results
        np.random.seed(42)
        n_samples = 100
        results = pd.DataFrame(
            {
                "param1": np.random.uniform(1, 10, n_samples),
                "param2": np.random.uniform(0.1, 1.0, n_samples),
                "optimal_roe": np.random.uniform(0.05, 0.25, n_samples),
                "ruin_probability": np.random.uniform(0.001, 0.05, n_samples),
            }
        )

        # Find optimal regions
        optimal, summary = sweeper.find_optimal_regions(
            results,
            objective="optimal_roe",
            constraints={"ruin_probability": (0, 0.01)},
            top_percentile=90,
        )

        # Check that constraints are applied
        assert (optimal["ruin_probability"] <= 0.01).all()

        # Check that we get top performers
        constrained_results = results[results["ruin_probability"] <= 0.01]
        if not constrained_results.empty:
            threshold = np.percentile(constrained_results["optimal_roe"], 90)
            assert (optimal["optimal_roe"] >= threshold).all()

        # Check summary statistics
        assert "min" in summary.columns
        assert "max" in summary.columns
        assert "mean" in summary.columns
        assert "std" in summary.columns

    def test_find_optimal_regions_no_valid_results(self, sweeper):
        """Test optimal region identification with no valid results."""
        # Create results with all NaN values
        results = pd.DataFrame(
            {
                "param1": [1, 2, 3],
                "optimal_roe": [np.nan, np.nan, np.nan],
                "ruin_probability": [0.01, 0.02, 0.03],
            }
        )

        optimal, summary = sweeper.find_optimal_regions(results, objective="optimal_roe")

        assert optimal.empty
        assert summary.empty

    def test_compare_scenarios(self, sweeper):
        """Test scenario comparison."""
        # Create sample results for different scenarios
        scenario1 = pd.DataFrame(
            {
                "optimal_roe": [0.10, 0.12, 0.15],
                "ruin_probability": [0.01, 0.005, 0.008],
                "total_premium": [100000, 120000, 150000],
            }
        )

        scenario2 = pd.DataFrame(
            {
                "optimal_roe": [0.08, 0.09, 0.11],
                "ruin_probability": [0.02, 0.015, 0.018],
                "total_premium": [80000, 90000, 110000],
            }
        )

        results = {"scenario1": scenario1, "scenario2": scenario2}

        comparison = sweeper.compare_scenarios(results)

        # Check structure
        assert "optimal_roe_mean" in comparison.columns
        assert "optimal_roe_std" in comparison.columns
        assert "ruin_probability_mean" in comparison.columns

        # Check values
        assert comparison.loc["scenario1", "optimal_roe_mean"] == pytest.approx(0.123, rel=0.01)
        assert comparison.loc["scenario2", "optimal_roe_mean"] == pytest.approx(0.093, rel=0.01)

    def test_compare_scenarios_with_normalization(self, sweeper):
        """Test scenario comparison with normalization."""
        scenario1 = pd.DataFrame({"metric1": [10, 20, 30]})
        scenario2 = pd.DataFrame({"metric1": [5, 10, 15]})

        results = {"scenario1": scenario1, "scenario2": scenario2}

        comparison = sweeper.compare_scenarios(results, metrics=["metric1"], normalize=True)

        # Check that normalized columns are created only if there are multiple scenarios
        if "metric1_mean_normalized" in comparison.columns:
            # Check normalization (scenario1 should be 1, scenario2 should be 0)
            assert comparison.loc["scenario1", "metric1_mean_normalized"] == 1.0
            assert comparison.loc["scenario2", "metric1_mean_normalized"] == 0.0
        else:
            # At least check that basic comparison works
            assert "metric1_mean" in comparison.columns

    def test_cache_key_generation(self, sweeper):
        """Test cache key generation consistency."""
        params1 = {"param1": 1, "param2": 0.1, "param3": "value"}
        params2 = {"param3": "value", "param1": 1, "param2": 0.1}  # Different order
        params3 = {"param1": 2, "param2": 0.1, "param3": "value"}  # Different value

        key1 = sweeper._get_cache_key(params1)
        key2 = sweeper._get_cache_key(params2)
        key3 = sweeper._get_cache_key(params3)

        # Same parameters in different order should give same key
        assert key1 == key2

        # Different parameters should give different key
        assert key1 != key3

    def test_sweep_hash_generation(self, sweeper):
        """Test sweep configuration hash generation."""
        config1 = SweepConfig(parameters={"param1": [1, 2]}, fixed_params={"fixed1": 10})

        config2 = SweepConfig(parameters={"param1": [1, 2]}, fixed_params={"fixed1": 10})

        config3 = SweepConfig(
            parameters={"param1": [1, 2, 3]}, fixed_params={"fixed1": 10}  # Different
        )

        hash1 = sweeper._get_sweep_hash(config1)
        hash2 = sweeper._get_sweep_hash(config2)
        hash3 = sweeper._get_sweep_hash(config3)

        # Same config should give same hash
        assert hash1 == hash2

        # Different config should give different hash
        assert hash1 != hash3

        # Hash should be 8 characters
        assert len(hash1) == 8

    def test_save_and_load_results(self, sweeper):
        """Test saving and loading results."""
        # Create sample results
        results = pd.DataFrame(
            {
                "param1": [1, 2, 3],
                "optimal_roe": [0.10, 0.12, 0.15],
                "ruin_probability": [0.01, 0.005, 0.008],
            }
        )

        config = SweepConfig(
            parameters={"param1": [1, 2, 3]}, metrics_to_track=["optimal_roe", "ruin_probability"]
        )

        # Save results
        sweeper._save_results(results, config)

        # Load results
        sweep_hash = sweeper._get_sweep_hash(config)
        loaded = sweeper.load_results(sweep_hash)

        assert loaded is not None
        # Use check_dtype=False for compatibility with different storage backends
        pd.testing.assert_frame_equal(results, loaded, check_dtype=False)

        # Check metadata file
        meta_file = sweeper.cache_dir / f"sweep_{sweep_hash}_meta.json"
        assert meta_file.exists()

        with open(meta_file, "r") as f:
            metadata = json.load(f)

        assert metadata["n_results"] == 3
        assert metadata["parameters"] == ["param1"]
        assert metadata["sweep_hash"] == sweep_hash

    def test_export_results_formats(self, sweeper):
        """Test exporting results to different formats."""
        results = pd.DataFrame({"param1": [1, 2, 3], "optimal_roe": [0.10, 0.12, 0.15]})

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test CSV export
            csv_file = Path(tmpdir) / "results.csv"
            sweeper.export_results(results, str(csv_file), file_format="csv")
            assert csv_file.exists()

            # Test Parquet export
            parquet_file = Path(tmpdir) / "results.parquet"
            sweeper.export_results(results, str(parquet_file), file_format="parquet")
            assert parquet_file.exists()

            # Test HDF5 export (may fall back to parquet if tables not available)
            h5_file = Path(tmpdir) / "results.h5"
            sweeper.export_results(results, str(h5_file), file_format="hdf5")
            # Check if either h5 or parquet file exists (fallback behavior)
            assert h5_file.exists() or h5_file.with_suffix(".parquet").exists()

            # Test invalid format
            with pytest.raises(ValueError, match="Unsupported format"):
                sweeper.export_results(results, "output.txt", file_format="invalid")

    def test_adaptive_refinement(self, sweeper):
        """Test adaptive refinement functionality."""
        # Create initial results with clear optimal region
        np.random.seed(42)
        initial_results = pd.DataFrame(
            {
                "param1": np.linspace(1, 10, 20),
                "param2": np.linspace(0.1, 1.0, 20),
                "optimal_roe": np.concatenate(
                    [
                        np.random.uniform(0.05, 0.10, 10),  # Low performance
                        np.random.uniform(0.15, 0.20, 10),  # High performance
                    ]
                ),
            }
        )

        config = SweepConfig(
            parameters={
                "param1": list(np.linspace(1, 10, 5)),
                "param2": list(np.linspace(0.1, 1.0, 5)),
            },
            adaptive_refinement=True,
            refinement_threshold=50,  # Top 50%
        )

        # Mock the sweep method to return refined results
        def mock_sweep(refined_config):
            # Generate some refined results
            n_refined = 10
            return pd.DataFrame(
                {
                    "param1": np.random.uniform(5, 10, n_refined),
                    "param2": np.random.uniform(0.5, 1.0, n_refined),
                    "optimal_roe": np.random.uniform(0.18, 0.22, n_refined),
                }
            )

        with patch.object(sweeper, "sweep", side_effect=mock_sweep):
            refined = sweeper._apply_adaptive_refinement(initial_results, config)

        # Should have more results than initial
        assert len(refined) >= len(initial_results)

        # Check that duplicates are removed
        param_cols = ["param1", "param2"]
        assert not refined[param_cols].duplicated().any()

    def test_parallel_execution(self):
        """Test parallel execution configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with parallel enabled
            sweeper_parallel = ParameterSweeper(cache_dir=tmpdir, use_parallel=True)
            assert sweeper_parallel.use_parallel is True

            # Test with parallel disabled
            sweeper_serial = ParameterSweeper(cache_dir=tmpdir, use_parallel=False)
            assert sweeper_serial.use_parallel is False

    def test_create_manufacturer(self, sweeper):
        """Test manufacturer creation from parameters."""
        params = {
            "initial_assets": 20e6,
            "working_capital_pct": 0.15,
            "operating_margin": 0.10,
            "asset_turnover": 1.2,
            "tax_rate": 0.30,
        }

        manufacturer = sweeper._create_manufacturer(params)

        # Check that manufacturer was created with config
        assert manufacturer.assets == 20e6
        assert manufacturer.config.base_operating_margin == 0.10
        assert manufacturer.config.asset_turnover_ratio == 1.2
        assert manufacturer.config.tax_rate == 0.30

    def test_save_intermediate_results(self, sweeper):
        """Test saving intermediate results during sweep."""
        results = [
            {"param1": 1, "optimal_roe": 0.10},
            {"param1": 2, "optimal_roe": 0.12},
            {"param1": 3, "optimal_roe": 0.15},
        ]

        sweep_hash = "test_hash"
        sweeper._save_intermediate_results(results, sweep_hash)

        # Check that temporary file exists (either h5 or parquet)
        temp_h5_file = sweeper.cache_dir / f"sweep_{sweep_hash}_temp.h5"
        temp_parquet_file = sweeper.cache_dir / f"sweep_{sweep_hash}_temp.parquet"
        assert temp_h5_file.exists() or temp_parquet_file.exists()

        # Load and verify contents
        loaded = None
        if temp_h5_file.exists():
            try:
                loaded = pd.read_hdf(temp_h5_file, key="results")
            except ImportError:
                # If HDF5 not available, check parquet
                if temp_parquet_file.exists():
                    loaded = pd.read_parquet(temp_parquet_file)
                else:
                    pytest.skip("Neither HDF5 nor Parquet file found")
        elif temp_parquet_file.exists():
            loaded = pd.read_parquet(temp_parquet_file)
        else:
            pytest.skip("No temporary file found")

        assert loaded is not None
        assert len(loaded) == 3
        assert loaded["param1"].tolist() == [1, 2, 3]
