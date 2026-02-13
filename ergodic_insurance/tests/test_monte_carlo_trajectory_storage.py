"""Tests for Monte Carlo integration with trajectory storage."""

from pathlib import Path
import shutil
import tempfile
from unittest.mock import Mock

import numpy as np
import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.insurance_program import EnhancedInsuranceLayer, InsuranceProgram
from ergodic_insurance.loss_distributions import ManufacturingLossGenerator
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.monte_carlo import MonteCarloConfig, MonteCarloEngine
from ergodic_insurance.trajectory_storage import StorageConfig, TrajectoryStorage


class TestMonteCarloTrajectoryIntegration:
    """Test Monte Carlo engine with trajectory storage."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for storage."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def setup_engine_with_storage(self, temp_dir):
        """Set up Monte Carlo engine with trajectory storage."""
        # Create loss generator
        loss_generator = Mock(spec=ManufacturingLossGenerator)
        loss_generator.generate_losses.return_value = ([], {"total_amount": 100_000})

        # Create insurance program
        layer = EnhancedInsuranceLayer(attachment_point=0, limit=1_000_000, base_premium_rate=0.02)
        insurance_program = InsuranceProgram(layers=[layer])

        # Create manufacturer
        manufacturer_config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.5,
            base_operating_margin=0.1,
            tax_rate=0.25,
            retention_ratio=0.8,
        )
        manufacturer = WidgetManufacturer(manufacturer_config)

        # Configure trajectory storage
        storage_config = StorageConfig(
            storage_dir=temp_dir,
            backend="memmap",
            sample_interval=5,
            chunk_size=100,
            enable_summary_stats=True,
            enable_time_series=True,
        )

        # Create simulation config with storage enabled
        sim_config = MonteCarloConfig(
            n_simulations=100,
            n_years=10,
            parallel=False,  # Disable parallel for simpler testing
            seed=42,
            enable_trajectory_storage=True,
            trajectory_storage_config=storage_config,
        )

        # Create engine
        engine = MonteCarloEngine(
            loss_generator=loss_generator,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=sim_config,
        )

        return engine

    def test_storage_enabled_in_config(self):
        """Test that storage can be enabled in simulation config."""
        storage_config = StorageConfig(storage_dir="./test_storage")

        config = MonteCarloConfig(
            enable_trajectory_storage=True,
            trajectory_storage_config=storage_config,
        )

        assert config.enable_trajectory_storage is True
        assert config.trajectory_storage_config is not None
        assert config.trajectory_storage_config.storage_dir == "./test_storage"

    def test_engine_initializes_storage(self, setup_engine_with_storage):
        """Test that engine initializes trajectory storage when enabled."""
        engine = setup_engine_with_storage

        assert engine.trajectory_storage is not None
        assert isinstance(engine.trajectory_storage, TrajectoryStorage)
        assert engine.trajectory_storage.storage_path.exists()

    def test_simulation_stores_trajectories(self, setup_engine_with_storage):
        """Test that running simulation stores trajectories."""
        engine = setup_engine_with_storage

        # Run simulation
        results = engine.run()

        # Check that simulations were stored
        storage = engine.trajectory_storage
        assert storage is not None
        assert storage._total_simulations > 0

        # Verify we can load stored data
        data = storage.load_simulation(0)
        assert "summary" in data
        assert data["summary"]["sim_id"] == 0

    def test_storage_respects_sampling_interval(self, setup_engine_with_storage):
        """Test that storage respects sampling interval setting."""
        engine = setup_engine_with_storage

        # Verify sampling interval is set
        assert engine.trajectory_storage.config.sample_interval == 5

        # Run simulation
        results = engine.run()

        # Load a trajectory with time series
        data = engine.trajectory_storage.load_simulation(0, load_time_series=True)

        if "time_series" in data:
            # With 10 years and sampling interval of 5, should have 2 samples (0, 5)
            assert len(data["time_series"]["annual_losses"]) <= 10 // 5 + 1

    def test_storage_disabled_by_default(self):
        """Test that storage is disabled by default."""
        config = MonteCarloConfig()
        assert config.enable_trajectory_storage is False
        assert config.trajectory_storage_config is None

        # Create engine with default config
        loss_generator = Mock(spec=ManufacturingLossGenerator)
        loss_generator.generate_losses.return_value = ([], {"total_amount": 0})

        insurance_program = InsuranceProgram(layers=[])
        manufacturer_config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.5,
            base_operating_margin=0.1,
            tax_rate=0.25,
            retention_ratio=0.8,
        )
        manufacturer = WidgetManufacturer(manufacturer_config)

        engine = MonteCarloEngine(
            loss_generator=loss_generator,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=config,
        )

        assert engine.trajectory_storage is None

    def test_export_functionality(self, setup_engine_with_storage, temp_dir):
        """Test that stored data can be exported."""
        engine = setup_engine_with_storage

        # Run simulation
        results = engine.run()

        # Export to CSV
        csv_path = Path(temp_dir) / "export.csv"
        engine.trajectory_storage.export_summaries_csv(str(csv_path))
        assert csv_path.exists()

        # Export to JSON
        json_path = Path(temp_dir) / "export.json"
        engine.trajectory_storage.export_summaries_json(str(json_path))
        assert json_path.exists()

    def test_storage_statistics(self, setup_engine_with_storage):
        """Test that storage tracks statistics correctly."""
        engine = setup_engine_with_storage

        # Run simulation
        results = engine.run()

        # Get storage stats
        stats = engine.trajectory_storage.get_storage_stats()

        assert stats["total_simulations"] == engine.config.n_simulations
        assert stats["backend"] == "memmap"
        assert stats["sample_interval"] == 5
        assert "disk_usage_gb" in stats
        assert stats["disk_usage_gb"] < 1.0  # Should be well under 1GB for 100 sims
