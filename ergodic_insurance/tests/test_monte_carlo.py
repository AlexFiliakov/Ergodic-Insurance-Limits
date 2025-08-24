"""Tests for Monte Carlo simulation engine."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from ergodic_insurance.src.config import ManufacturerConfig
from ergodic_insurance.src.convergence import ConvergenceStats
from ergodic_insurance.src.insurance_program import EnhancedInsuranceLayer, InsuranceProgram
from ergodic_insurance.src.loss_distributions import ManufacturingLossGenerator
from ergodic_insurance.src.manufacturer import WidgetManufacturer
from ergodic_insurance.src.monte_carlo import MonteCarloEngine, SimulationConfig, SimulationResults


class TestSimulationConfig:
    """Test SimulationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SimulationConfig()
        assert config.n_simulations == 100_000
        assert config.n_years == 10
        assert config.n_chains == 4
        assert config.parallel is True
        assert config.use_float32 is True
        assert config.cache_results is True
        assert config.progress_bar is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SimulationConfig(n_simulations=50_000, n_years=5, parallel=False, seed=42)
        assert config.n_simulations == 50_000
        assert config.n_years == 5
        assert config.parallel is False
        assert config.seed == 42


class TestSimulationResults:
    """Test SimulationResults dataclass."""

    def test_results_summary(self):
        """Test results summary generation."""
        config = SimulationConfig(n_simulations=1000, n_years=5)
        results = SimulationResults(
            final_assets=np.array([100_000, 150_000, 80_000]),
            annual_losses=np.zeros((3, 5)),
            insurance_recoveries=np.zeros((3, 5)),
            retained_losses=np.zeros((3, 5)),
            growth_rates=np.array([0.05, 0.08, -0.02]),
            ruin_probability=0.1,
            metrics={"var_99": 1_000_000, "tvar_99": 1_500_000},
            convergence={"growth_rate": ConvergenceStats(1.02, 1000, 0.01, True, 1000, 0.1)},
            execution_time=10.5,
            config=config,
        )

        summary = results.summary()
        assert "Simulations: 1,000" in summary
        assert "Years: 5" in summary
        assert "Execution Time: 10.50s" in summary
        assert "Ruin Probability: 10.00%" in summary
        assert "Mean Growth Rate: 0.0367" in summary
        assert "VaR(99%): $1,000,000" in summary
        assert "TVaR(99%): $1,500,000" in summary
        assert "Convergence R-hat: 1.020" in summary


class TestMonteCarloEngine:
    """Test Monte Carlo simulation engine."""

    @pytest.fixture
    def setup_engine(self):
        """Set up test engine with mocked components."""
        # Create mock loss generator
        loss_generator = Mock(spec=ManufacturingLossGenerator)
        loss_generator.generate_losses.return_value = ([], {"total_amount": 100_000})

        # Create insurance program
        layer = EnhancedInsuranceLayer(attachment_point=0, limit=1_000_000, premium_rate=0.02)
        insurance_program = InsuranceProgram(layers=[layer])

        # Create manufacturer
        manufacturer_config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.5,
            operating_margin=0.1,
            tax_rate=0.25,
            retention_ratio=0.8,
        )
        manufacturer = WidgetManufacturer(manufacturer_config)

        # Create config with small values for testing
        config = SimulationConfig(
            n_simulations=100,
            n_years=2,
            parallel=False,
            cache_results=False,
            progress_bar=False,
            seed=42,
        )

        # Create engine
        engine = MonteCarloEngine(
            loss_generator=loss_generator,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=config,
        )

        return engine, loss_generator, insurance_program, manufacturer

    def test_engine_initialization(self, setup_engine):
        """Test engine initialization."""
        engine, _, _, _ = setup_engine

        assert engine.config.n_simulations == 100
        assert engine.config.n_years == 2
        assert engine.convergence_diagnostics is not None

    def test_sequential_run(self, setup_engine):
        """Test sequential simulation run."""
        engine, loss_generator, _, _ = setup_engine

        # Mock loss events
        from ergodic_insurance.src.loss_distributions import LossEvent

        loss_generator.generate_losses.return_value = (
            [LossEvent(time=0.5, amount=50_000, loss_type="test")],
            {"total_amount": 50_000},
        )

        # Run simulation
        results = engine.run()

        assert results is not None
        assert len(results.final_assets) == 100
        assert results.annual_losses.shape == (100, 2)
        assert results.insurance_recoveries.shape == (100, 2)
        assert results.retained_losses.shape == (100, 2)
        assert 0 <= results.ruin_probability <= 1
        assert results.execution_time > 0

    def test_parallel_run(self, setup_engine):
        """Test parallel simulation run."""
        engine, loss_generator, _, _ = setup_engine

        # Configure for parallel execution
        engine.config.n_simulations = 10_000
        engine.config.parallel = True
        engine.config.n_workers = 2
        engine.config.chunk_size = 5_000

        # Mock loss events
        from ergodic_insurance.src.loss_distributions import LossEvent

        loss_generator.generate_losses.return_value = (
            [LossEvent(time=0.5, amount=50_000, loss_type="test")],
            {"total_amount": 50_000},
        )

        # Mock parallel execution to avoid multiprocessing in tests
        with patch.object(engine, "_run_parallel", return_value=engine._run_sequential()):
            results = engine.run()

        assert results is not None
        assert len(results.final_assets) == 10_000

    def test_growth_rate_calculation(self, setup_engine):
        """Test growth rate calculation."""
        engine, _, _, _ = setup_engine

        # Test with positive final assets
        final_assets = np.array([15_000_000, 8_000_000, 12_000_000])
        engine.config.n_years = 5
        growth_rates = engine._calculate_growth_rates(final_assets)

        assert len(growth_rates) == 3
        assert growth_rates[0] > 0  # Growth
        assert growth_rates[1] < 0  # Decline

        # Test with zero/negative final assets
        final_assets = np.array([0, -1_000_000, 15_000_000])
        growth_rates = engine._calculate_growth_rates(final_assets)

        assert growth_rates[0] == 0  # Zero for ruin
        assert growth_rates[1] == 0  # Zero for negative
        assert growth_rates[2] > 0  # Positive for growth

    def test_metrics_calculation(self, setup_engine):
        """Test risk metrics calculation."""
        engine, _, _, _ = setup_engine

        # Create mock results
        results = SimulationResults(
            final_assets=np.random.normal(10_000_000, 2_000_000, 1000),
            annual_losses=np.random.exponential(100_000, (1000, 5)),
            insurance_recoveries=np.random.exponential(50_000, (1000, 5)),
            retained_losses=np.random.exponential(50_000, (1000, 5)),
            growth_rates=np.random.normal(0.05, 0.02, 1000),
            ruin_probability=0.05,
            metrics={},
            convergence={},
            execution_time=0,
            config=engine.config,
        )

        # Calculate metrics
        metrics = engine._calculate_metrics(results)

        assert "mean_loss" in metrics
        assert "var_99" in metrics
        assert "tvar_99" in metrics
        assert "mean_growth_rate" in metrics
        assert "sharpe_ratio" in metrics
        assert metrics["var_99"] > metrics["var_95"]
        assert metrics["tvar_99"] > metrics["var_99"]

    def test_convergence_check(self, setup_engine):
        """Test convergence checking."""
        engine, _, _, _ = setup_engine

        # Create results with enough data for convergence check
        n_sims = 1000
        results = SimulationResults(
            final_assets=np.random.normal(10_000_000, 2_000_000, n_sims),
            annual_losses=np.random.exponential(100_000, (n_sims, 5)),
            insurance_recoveries=np.random.exponential(50_000, (n_sims, 5)),
            retained_losses=np.random.exponential(50_000, (n_sims, 5)),
            growth_rates=np.random.normal(0.05, 0.02, n_sims),
            ruin_probability=0.05,
            metrics={},
            convergence={},
            execution_time=0,
            config=engine.config,
        )

        # Check convergence
        convergence = engine._check_convergence(results)

        if convergence:  # May be empty if not enough chains
            assert "growth_rate" in convergence
            assert isinstance(convergence["growth_rate"], ConvergenceStats)
            assert convergence["growth_rate"].r_hat >= 0

    def test_caching(self, setup_engine):
        """Test result caching."""
        engine, loss_generator, _, _ = setup_engine

        # Create temporary cache directory
        with tempfile.TemporaryDirectory() as tmpdir:
            engine.cache_dir = Path(tmpdir) / "cache"
            engine.cache_dir.mkdir(parents=True)
            engine.config.cache_results = True

            # Mock loss events
            from ergodic_insurance.src.loss_distributions import LossEvent

            loss_generator.generate_losses.return_value = (
                [LossEvent(time=0.5, amount=50_000, loss_type="test")],
                {"total_amount": 50_000},
            )

            # First run - should cache
            results1 = engine.run()

            # Second run - should load from cache
            with patch.object(engine, "_run_sequential") as mock_run:
                results2 = engine.run()
                mock_run.assert_not_called()  # Should not run simulation

            # Results should be identical
            assert np.array_equal(results1.final_assets, results2.final_assets)

    def test_convergence_monitoring(self, setup_engine):
        """Test convergence monitoring."""
        engine, loss_generator, _, _ = setup_engine

        # Mock loss events
        from ergodic_insurance.src.loss_distributions import LossEvent

        loss_generator.generate_losses.return_value = (
            [LossEvent(time=0.5, amount=50_000, loss_type="test")],
            {"total_amount": 50_000},
        )

        # Run with convergence monitoring
        engine.config.n_simulations = 100
        results = engine.run_with_convergence_monitoring(
            target_r_hat=1.1, check_interval=50, max_iterations=200
        )

        assert results is not None
        assert len(results.final_assets) >= 50  # At least one batch

    def test_single_simulation(self, setup_engine):
        """Test single simulation path."""
        engine, loss_generator, _, manufacturer = setup_engine

        # Mock loss events
        from ergodic_insurance.src.loss_distributions import LossEvent

        loss_generator.generate_losses.return_value = (
            [LossEvent(time=0.5, amount=50_000, loss_type="test")],
            {"total_amount": 50_000},
        )

        # Run single simulation
        sim_results = engine._run_single_simulation(0)

        assert "final_assets" in sim_results
        assert "annual_losses" in sim_results
        assert "insurance_recoveries" in sim_results
        assert "retained_losses" in sim_results
        assert len(sim_results["annual_losses"]) == engine.config.n_years

    def test_chunk_processing(self, setup_engine):
        """Test chunk processing for parallel execution."""
        engine, loss_generator, _, _ = setup_engine

        # Mock loss events
        from ergodic_insurance.src.loss_distributions import LossEvent

        loss_generator.generate_losses.return_value = (
            [LossEvent(time=0.5, amount=50_000, loss_type="test")],
            {"total_amount": 50_000},
        )

        # Process a chunk
        chunk = (0, 10, 42)  # start, end, seed
        chunk_results = engine._run_chunk(chunk)

        assert "final_assets" in chunk_results
        assert len(chunk_results["final_assets"]) == 10
        assert chunk_results["annual_losses"].shape == (10, engine.config.n_years)

    def test_combine_results(self, setup_engine):
        """Test combining multiple simulation results."""
        engine, _, _, _ = setup_engine

        # Create multiple result sets
        results1 = SimulationResults(
            final_assets=np.array([100_000, 150_000]),
            annual_losses=np.ones((2, 5)),
            insurance_recoveries=np.ones((2, 5)),
            retained_losses=np.ones((2, 5)),
            growth_rates=np.array([0.05, 0.08]),
            ruin_probability=0.0,
            metrics={},
            convergence={},
            execution_time=1.0,
            config=engine.config,
        )

        results2 = SimulationResults(
            final_assets=np.array([80_000, 120_000]),
            annual_losses=np.ones((2, 5)),
            insurance_recoveries=np.ones((2, 5)),
            retained_losses=np.ones((2, 5)),
            growth_rates=np.array([-0.02, 0.04]),
            ruin_probability=0.25,
            metrics={},
            convergence={},
            execution_time=1.0,
            config=engine.config,
        )

        # Combine results
        combined = engine._combine_multiple_results([results1, results2])

        assert len(combined.final_assets) == 4
        assert combined.annual_losses.shape == (4, 5)
        assert combined.execution_time == 2.0
