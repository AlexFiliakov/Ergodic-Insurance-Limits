"""Extended tests for Monte Carlo simulation engine to improve coverage."""

import warnings
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import pickle

import numpy as np
import pytest
from ergodic_insurance.src.config import ManufacturerConfig
from ergodic_insurance.src.convergence import ConvergenceStats
from ergodic_insurance.src.insurance_program import EnhancedInsuranceLayer, InsuranceProgram
from ergodic_insurance.src.loss_distributions import LossEvent, ManufacturingLossGenerator
from ergodic_insurance.src.manufacturer import WidgetManufacturer
from ergodic_insurance.src.monte_carlo import MonteCarloEngine, SimulationConfig, SimulationResults


class TestMonteCarloExtended:
    """Extended tests for Monte Carlo engine."""

    @pytest.fixture
    def setup_simple_engine(self):
        """Set up a simple engine for testing."""
        # Simple loss generator
        loss_generator = ManufacturingLossGenerator(
            attritional_params={"base_frequency": 1.0, "severity_mean": 10_000, "severity_cv": 0.5},
            large_params={"base_frequency": 0.1, "severity_mean": 100_000, "severity_cv": 0.5},
            catastrophic_params=None,
            seed=42,
        )

        # Simple insurance
        layers = [EnhancedInsuranceLayer(attachment_point=0, limit=100_000, premium_rate=0.05)]
        insurance_program = InsuranceProgram(layers=layers)

        # Simple manufacturer
        manufacturer_config = ManufacturerConfig(
            initial_assets=1_000_000,
            asset_turnover_ratio=0.5,
            operating_margin=0.1,
            tax_rate=0.25,
            retention_ratio=0.8,
        )
        manufacturer = WidgetManufacturer(manufacturer_config)

        # Simple config
        config = SimulationConfig(
            n_simulations=100,
            n_years=2,
            parallel=False,
            cache_results=False,
            progress_bar=False,
            seed=42,
        )

        return MonteCarloEngine(
            loss_generator=loss_generator,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=config,
        )

    def test_summary_method(self, setup_simple_engine):
        """Test the summary method of SimulationResults."""
        engine = setup_simple_engine
        
        # Create mock results
        results = SimulationResults(
            final_assets=np.array([100_000, 200_000, 0, -50_000]),
            annual_losses=np.ones((4, 2)) * 10_000,
            insurance_recoveries=np.ones((4, 2)) * 5_000,
            retained_losses=np.ones((4, 2)) * 5_000,
            growth_rates=np.array([0.05, 0.10, -1.0, -1.0]),
            ruin_probability=0.5,
            metrics={"mean_loss": 10_000, "var_95": 20_000},
            convergence={"metric1": ConvergenceStats(
                r_hat=1.05, ess=1000, mcse=0.01, converged=True, 
                n_iterations=100, autocorrelation=0.1
            )},
            execution_time=1.5,
            config=engine.config,
        )
        
        # Test summary
        summary = results.summary()
        assert "Simulations: 100" in summary  # Uses config.n_simulations
        assert "Ruin Probability: 50.00%" in summary
        assert "Mean Final Assets:" in summary
        assert "Execution Time: 1.50s" in summary

    def test_run_sequential_with_progress(self, setup_simple_engine):
        """Test sequential run with progress bar."""
        engine = setup_simple_engine
        engine.config.progress_bar = True
        engine.config.n_simulations = 10
        
        results = engine.run()
        assert results is not None
        assert len(results.final_assets) == 10

    def test_parallel_with_different_chunks(self, setup_simple_engine):
        """Test parallel processing with various chunk configurations."""
        engine = setup_simple_engine
        engine.config.parallel = True
        engine.config.n_workers = 2
        engine.config.chunk_size = 25
        engine.config.n_simulations = 50
        
        results = engine.run()
        assert results is not None
        assert len(results.final_assets) == 50

    def test_cache_operations(self, setup_simple_engine):
        """Test cache save and load operations."""
        engine = setup_simple_engine
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine.cache_dir = Path(tmpdir)
            engine.config.cache_results = True
            
            # Run first time - should save to cache
            results1 = engine.run()
            
            # Check cache was saved
            cache_key = engine._get_cache_key()
            cache_file = engine.cache_dir / f"{cache_key}.pkl"
            assert cache_file.exists()
            
            # Run second time - should load from cache
            with patch.object(engine, '_run_sequential') as mock_run:
                results2 = engine.run()
                mock_run.assert_not_called()  # Should not run simulation
            
            assert np.array_equal(results1.final_assets, results2.final_assets)

    def test_cache_save_failure(self, setup_simple_engine):
        """Test cache save failure handling."""
        engine = setup_simple_engine
        engine.cache_dir = Path("/invalid/path/that/does/not/exist")
        engine.config.cache_results = True
        
        # Should warn but not fail
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = engine.run()
            assert any("Failed to save cache" in str(warning.message) for warning in w)
        
        assert results is not None

    def test_cache_load_failure(self, setup_simple_engine):
        """Test cache load failure handling."""
        engine = setup_simple_engine
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine.cache_dir = Path(tmpdir)
            engine.config.cache_results = True
            
            # Create corrupt cache file
            cache_key = engine._get_cache_key()
            cache_file = engine.cache_dir / f"{cache_key}.pkl"
            cache_file.write_text("corrupt data")
            
            # Should warn but continue with normal run
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                results = engine.run()
                assert any("Failed to load cache" in str(warning.message) for warning in w)
            
            assert results is not None

    def test_checkpoint_operations(self, setup_simple_engine):
        """Test checkpoint save operations."""
        engine = setup_simple_engine
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine.cache_dir = Path(tmpdir)
            
            # Save checkpoint
            test_array = np.array([1, 2, 3])
            engine._save_checkpoint(10, test_array)
            
            # Check checkpoint was saved
            checkpoint_file = engine.cache_dir / "checkpoint_10.npz"
            assert checkpoint_file.exists()
            
            # Load and verify
            with np.load(checkpoint_file) as data:
                assert data['iteration'] == 10
                assert np.array_equal(data['arr_0'], test_array)

    def test_checkpoint_save_failure(self, setup_simple_engine):
        """Test checkpoint save failure handling."""
        engine = setup_simple_engine
        engine.cache_dir = Path("/invalid/path")
        
        # Should warn but not fail
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine._save_checkpoint(10, np.array([1, 2, 3]))
            assert any("Failed to save checkpoint" in str(warning.message) for warning in w)

    def test_convergence_with_insufficient_data(self, setup_simple_engine):
        """Test convergence check with insufficient data."""
        engine = setup_simple_engine
        
        # Create results with very few simulations
        results = SimulationResults(
            final_assets=np.array([100_000, 200_000]),
            annual_losses=np.ones((2, 2)),
            insurance_recoveries=np.ones((2, 2)),
            retained_losses=np.ones((2, 2)),
            growth_rates=np.array([0.05, 0.10]),
            ruin_probability=0.0,
            metrics={},
            convergence={},
            execution_time=1.0,
            config=engine.config,
        )
        
        # Should return empty dict for insufficient data
        convergence = engine._check_convergence(results)
        assert convergence == {}

    def test_convergence_monitoring_max_iterations(self, setup_simple_engine):
        """Test convergence monitoring with max iterations limit."""
        engine = setup_simple_engine
        engine.config.n_simulations = 1000
        
        # Mock convergence check to never converge
        with patch.object(engine, '_check_convergence') as mock_check:
            mock_check.return_value = {
                "metric": ConvergenceStats(
                    r_hat=2.0,  # High R-hat, won't converge
                    ess=100,
                    mcse=0.1,
                    converged=False,
                    n_iterations=1000,
                    autocorrelation=0.5,
                )
            }
            
            results = engine.run_with_convergence_monitoring(
                target_r_hat=1.05,
                check_interval=100,
                max_iterations=500,  # Will hit this limit
            )
            
            assert results is not None
            # Should have run max_iterations
            assert len(results.final_assets) <= 500

    def test_convergence_monitoring_without_progress(self, setup_simple_engine):
        """Test convergence monitoring without progress bar."""
        engine = setup_simple_engine
        engine.config.progress_bar = False
        engine.config.n_simulations = 100
        
        results = engine.run_with_convergence_monitoring(
            target_r_hat=1.5,  # Easy target
            check_interval=50,
        )
        
        assert results is not None

    def test_use_float32_config(self, setup_simple_engine):
        """Test using float32 configuration."""
        engine = setup_simple_engine
        engine.config.use_float32 = True
        engine.config.n_simulations = 10
        
        results = engine._run_sequential()
        
        # Check that arrays use float32
        assert results.final_assets.dtype == np.float32
        assert results.annual_losses.dtype == np.float32

    def test_manufacturer_copy_in_simulation(self, setup_simple_engine):
        """Test that manufacturer is properly copied in simulation."""
        engine = setup_simple_engine
        original_assets = engine.manufacturer.assets
        
        # Run single simulation
        engine.config.n_simulations = 1
        results = engine.run()
        
        # Original manufacturer should be unchanged
        assert engine.manufacturer.assets == original_assets
        assert results is not None

    def test_edge_cases_in_growth_calculation(self, setup_simple_engine):
        """Test edge cases in growth rate calculation."""
        engine = setup_simple_engine
        
        # Test with zero initial assets (edge case)
        original_assets = engine.manufacturer.assets
        engine.manufacturer.assets = 0
        growth_rates = engine._calculate_growth_rates(np.array([100_000, 200_000]))
        assert np.all(growth_rates == 0)  # Should return zeros for invalid calculation
        
        # Restore
        engine.manufacturer.assets = original_assets
        
        # Test with negative final assets
        growth_rates = engine._calculate_growth_rates(np.array([-100_000, 0, 100_000]))
        assert growth_rates[0] == 0  # Negative should give 0
        assert growth_rates[1] == 0  # Zero should give 0
        assert growth_rates[2] != 0  # Positive should give non-zero

    def test_parallel_chunk_processing(self, setup_simple_engine):
        """Test the _run_chunk method directly."""
        engine = setup_simple_engine
        
        # Test chunk processing
        chunk = (0, 5, 42)  # start, end, seed
        results = engine._run_chunk(chunk)
        
        assert "final_assets" in results
        assert "annual_losses" in results
        assert len(results["final_assets"]) == 5

    def test_combine_chunk_results(self, setup_simple_engine):
        """Test combining chunk results."""
        engine = setup_simple_engine
        
        # Create mock chunk results
        chunk1 = {
            "final_assets": np.array([100_000, 200_000]),
            "annual_losses": np.ones((2, 2)),
            "insurance_recoveries": np.ones((2, 2)),
            "retained_losses": np.ones((2, 2)),
        }
        
        chunk2 = {
            "final_assets": np.array([150_000, 250_000]),
            "annual_losses": np.ones((2, 2)) * 2,
            "insurance_recoveries": np.ones((2, 2)) * 2,
            "retained_losses": np.ones((2, 2)) * 2,
        }
        
        results = engine._combine_chunk_results([chunk1, chunk2])
        
        assert len(results.final_assets) == 4
        assert results.annual_losses.shape == (4, 2)
        assert results.ruin_probability == 0.0  # No ruins in test data

    def test_metrics_with_zero_variance(self, setup_simple_engine):
        """Test metrics calculation with zero variance data."""
        engine = setup_simple_engine
        
        # Create results with no variance
        results = SimulationResults(
            final_assets=np.ones(10) * 100_000,
            annual_losses=np.ones((10, 2)) * 10_000,
            insurance_recoveries=np.ones((10, 2)) * 5_000,
            retained_losses=np.ones((10, 2)) * 5_000,
            growth_rates=np.zeros(10),  # Zero variance
            ruin_probability=0.0,
            metrics={},
            convergence={},
            execution_time=1.0,
            config=engine.config,
        )
        
        metrics = engine._calculate_metrics(results)
        
        assert metrics["sharpe_ratio"] == 0  # Should handle zero std dev
        assert metrics["mean_loss"] == 20_000  # Sum across 2 years
        assert metrics["std_loss"] == 0  # No variance

    def test_run_with_convergence_no_limit(self, setup_simple_engine):
        """Test convergence monitoring with no max iterations limit."""
        engine = setup_simple_engine
        engine.config.n_simulations = 100
        
        # Mock to converge quickly
        with patch.object(engine, '_check_convergence') as mock_check:
            mock_check.return_value = {
                "metric": ConvergenceStats(
                    r_hat=1.01,  # Good convergence
                    ess=1000,
                    mcse=0.001,
                    converged=True,
                    n_iterations=100,
                    autocorrelation=0.1,
                )
            }
            
            results = engine.run_with_convergence_monitoring(
                target_r_hat=1.05,
                check_interval=50,
                max_iterations=None,  # No limit
            )
            
            assert results is not None