"""Memory-efficient storage system for simulation trajectories.

This module provides a lightweight storage system for Monte Carlo simulation
trajectories that minimizes RAM usage while storing both partial time series
data and comprehensive summary statistics.

Features:
    - Memory-mapped numpy arrays for efficient storage
    - Optional HDF5 backend with compression
    - Configurable time series sampling (store every Nth year)
    - Lazy loading to minimize memory footprint
    - Automatic disk space management
    - CSV/JSON export for analysis tools
    - <2GB RAM usage for 100K simulations
    - <1GB disk usage with sampling

Example:
    >>> from ergodic_insurance.src.trajectory_storage import TrajectoryStorage
    >>> storage = TrajectoryStorage(
    ...     storage_dir="./trajectories",
    ...     sample_interval=5,  # Store every 5th year
    ...     max_disk_usage_gb=1.0
    ... )
    >>> # During simulation
    >>> storage.store_simulation(
    ...     sim_id=0,
    ...     annual_losses=losses,
    ...     final_assets=assets,
    ...     summary_stats=stats
    ... )
    >>> # Later retrieval
    >>> data = storage.load_simulation(sim_id=0)
"""

import csv
from dataclasses import dataclass, field
import gc
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import h5py
import numpy as np
import pandas as pd


@dataclass
class StorageConfig:
    """Configuration for trajectory storage system.

    Attributes:
        storage_dir: Directory for storing trajectory data
        backend: Storage backend ('memmap' or 'hdf5')
        sample_interval: Store every Nth year (1 = store all)
        max_disk_usage_gb: Maximum disk space to use
        compression: Enable compression (HDF5 only)
        compression_level: Compression level 0-9 (HDF5 only)
        chunk_size: Chunk size for batch operations
        enable_summary_stats: Store summary statistics
        enable_time_series: Store time series data
        dtype: Data type for storage (np.float32 or np.float64)
    """

    storage_dir: str = "./trajectory_storage"
    backend: str = "memmap"  # 'memmap' or 'hdf5'
    sample_interval: int = 10  # Store every 10th year by default
    max_disk_usage_gb: float = 1.0
    compression: bool = True
    compression_level: int = 4  # Medium compression
    chunk_size: int = 1000  # Process 1000 simulations at a time
    enable_summary_stats: bool = True
    enable_time_series: bool = True
    dtype: Any = np.float32  # Use float32 for memory efficiency


@dataclass
class SimulationSummary:
    """Summary statistics for a single simulation.

    Attributes:
        sim_id: Simulation identifier
        final_assets: Final asset value
        total_losses: Total losses over simulation
        total_recoveries: Total insurance recoveries
        mean_annual_loss: Average annual loss
        max_annual_loss: Maximum annual loss
        min_annual_loss: Minimum annual loss
        growth_rate: Realized growth rate
        ruin_occurred: Whether ruin occurred
        ruin_year: Year of ruin (if occurred)
        volatility: Standard deviation of returns
    """

    sim_id: int
    final_assets: float
    total_losses: float
    total_recoveries: float
    mean_annual_loss: float
    max_annual_loss: float
    min_annual_loss: float
    growth_rate: float
    ruin_occurred: bool
    ruin_year: Optional[int] = None
    volatility: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "sim_id": self.sim_id,
            "final_assets": float(self.final_assets),
            "total_losses": float(self.total_losses),
            "total_recoveries": float(self.total_recoveries),
            "mean_annual_loss": float(self.mean_annual_loss),
            "max_annual_loss": float(self.max_annual_loss),
            "min_annual_loss": float(self.min_annual_loss),
            "growth_rate": float(self.growth_rate),
            "ruin_occurred": self.ruin_occurred,
            "ruin_year": self.ruin_year,
            "volatility": float(self.volatility) if self.volatility else None,
        }


class TrajectoryStorage:
    """Memory-efficient storage for simulation trajectories.

    Provides lightweight storage using memory-mapped arrays or HDF5,
    with configurable sampling and automatic disk space management.
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        """Initialize trajectory storage.

        Args:
            config: Storage configuration
        """
        self.config = config or StorageConfig()
        self.storage_path = Path(self.config.storage_dir)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize storage structures
        self._summaries: Dict[int, SimulationSummary] = {}
        self._memmap_files: Dict[str, np.memmap] = {}
        self._hdf5_file: Optional[h5py.File] = None

        # Track storage statistics
        self._total_simulations = 0
        self._disk_usage = 0.0

        # Setup backend
        if self.config.backend == "hdf5":
            self._setup_hdf5()
        else:
            self._setup_memmap()

    def _setup_memmap(self) -> None:
        """Setup memory-mapped array storage."""
        # Create directories for different data types
        (self.storage_path / "summaries").mkdir(exist_ok=True)
        if self.config.enable_time_series:
            (self.storage_path / "time_series").mkdir(exist_ok=True)

    def _setup_hdf5(self) -> None:
        """Setup HDF5 storage backend."""
        hdf5_path = self.storage_path / "trajectories.h5"

        # Open or create HDF5 file
        self._hdf5_file = h5py.File(hdf5_path, "a")

        # Create groups if they don't exist
        if "summaries" not in self._hdf5_file:
            self._hdf5_file.create_group("summaries")
        if self.config.enable_time_series and "time_series" not in self._hdf5_file:
            self._hdf5_file.create_group("time_series")

    def store_simulation(
        self,
        sim_id: int,
        annual_losses: np.ndarray,
        insurance_recoveries: np.ndarray,
        retained_losses: np.ndarray,
        final_assets: float,
        initial_assets: float,
        ruin_occurred: bool = False,
        ruin_year: Optional[int] = None,
    ) -> None:
        """Store simulation trajectory with automatic sampling.

        Args:
            sim_id: Simulation identifier
            annual_losses: Array of annual losses
            insurance_recoveries: Array of insurance recoveries
            retained_losses: Array of retained losses
            final_assets: Final asset value
            initial_assets: Initial asset value
            ruin_occurred: Whether ruin occurred
            ruin_year: Year of ruin (if applicable)
        """
        # Check disk usage limit
        if not self._check_disk_space():
            warnings.warn(f"Disk usage limit ({self.config.max_disk_usage_gb}GB) exceeded")
            return

        # Calculate and store summary statistics
        if self.config.enable_summary_stats:
            summary = self._calculate_summary(
                sim_id=sim_id,
                annual_losses=annual_losses,
                insurance_recoveries=insurance_recoveries,
                final_assets=final_assets,
                initial_assets=initial_assets,
                ruin_occurred=ruin_occurred,
                ruin_year=ruin_year,
            )
            self._store_summary(summary)

        # Store sampled time series data
        if self.config.enable_time_series:
            self._store_time_series(
                sim_id=sim_id,
                annual_losses=annual_losses,
                insurance_recoveries=insurance_recoveries,
                retained_losses=retained_losses,
            )

        self._total_simulations += 1

        # Periodic cleanup to manage memory
        if self._total_simulations % self.config.chunk_size == 0:
            self._cleanup_memory()

    def _calculate_summary(
        self,
        sim_id: int,
        annual_losses: np.ndarray,
        insurance_recoveries: np.ndarray,
        final_assets: float,
        initial_assets: float,
        ruin_occurred: bool,
        ruin_year: Optional[int],
    ) -> SimulationSummary:
        """Calculate summary statistics for a simulation.

        Args:
            sim_id: Simulation identifier
            annual_losses: Annual loss amounts
            insurance_recoveries: Insurance recovery amounts
            final_assets: Final asset value
            initial_assets: Initial asset value
            ruin_occurred: Whether ruin occurred
            ruin_year: Year of ruin

        Returns:
            SimulationSummary with calculated statistics
        """
        n_years = len(annual_losses)

        # Calculate growth rate
        if final_assets > 0 and initial_assets > 0:
            growth_rate = np.log(final_assets / initial_assets) / n_years
        else:
            growth_rate = -np.inf

        # Calculate volatility (simplified)
        if n_years > 1:
            # Avoid division by zero
            non_zero_mask = annual_losses[:-1] != 0
            if np.any(non_zero_mask):
                returns = np.diff(annual_losses)[non_zero_mask] / annual_losses[:-1][non_zero_mask]
                volatility = np.std(returns) if len(returns) > 0 else 0.0
            else:
                volatility = 0.0
        else:
            volatility = 0.0

        return SimulationSummary(
            sim_id=sim_id,
            final_assets=float(final_assets),
            total_losses=float(np.sum(annual_losses)),
            total_recoveries=float(np.sum(insurance_recoveries)),
            mean_annual_loss=float(np.mean(annual_losses)),
            max_annual_loss=float(np.max(annual_losses)),
            min_annual_loss=float(np.min(annual_losses)),
            growth_rate=float(growth_rate),
            ruin_occurred=ruin_occurred,
            ruin_year=ruin_year,
            volatility=float(volatility),
        )

    def _store_summary(self, summary: SimulationSummary) -> None:
        """Store summary statistics.

        Args:
            summary: Simulation summary to store
        """
        self._summaries[summary.sim_id] = summary

        # Persist to disk periodically
        if len(self._summaries) >= self.config.chunk_size:
            self._persist_summaries()

    def _store_time_series(
        self,
        sim_id: int,
        annual_losses: np.ndarray,
        insurance_recoveries: np.ndarray,
        retained_losses: np.ndarray,
    ) -> None:
        """Store sampled time series data.

        Args:
            sim_id: Simulation identifier
            annual_losses: Annual loss amounts
            insurance_recoveries: Insurance recovery amounts
            retained_losses: Retained loss amounts
        """
        # Sample data according to interval
        sample_indices = np.arange(0, len(annual_losses), self.config.sample_interval)

        if self.config.backend == "hdf5":
            self._store_time_series_hdf5(
                sim_id, sample_indices, annual_losses, insurance_recoveries, retained_losses
            )
        else:
            self._store_time_series_memmap(
                sim_id, sample_indices, annual_losses, insurance_recoveries, retained_losses
            )

    def _store_time_series_memmap(
        self,
        sim_id: int,
        sample_indices: np.ndarray,
        annual_losses: np.ndarray,
        insurance_recoveries: np.ndarray,
        retained_losses: np.ndarray,
    ) -> None:
        """Store time series using memory-mapped arrays.

        Args:
            sim_id: Simulation identifier
            sample_indices: Indices to sample
            annual_losses: Annual loss amounts
            insurance_recoveries: Insurance recovery amounts
            retained_losses: Retained loss amounts
        """
        # Create memory-mapped file for this simulation
        ts_path = self.storage_path / "time_series" / f"sim_{sim_id}.dat"

        # Stack sampled data
        sampled_data = np.vstack(
            [
                annual_losses[sample_indices],
                insurance_recoveries[sample_indices],
                retained_losses[sample_indices],
            ]
        ).astype(self.config.dtype)

        # Write to memory-mapped file
        mmap = np.memmap(
            ts_path,
            dtype=self.config.dtype,
            mode="w+",
            shape=sampled_data.shape,
        )
        mmap[:] = sampled_data
        mmap.flush()
        del mmap  # Close the file

    def _store_time_series_hdf5(
        self,
        sim_id: int,
        sample_indices: np.ndarray,
        annual_losses: np.ndarray,
        insurance_recoveries: np.ndarray,
        retained_losses: np.ndarray,
    ) -> None:
        """Store time series using HDF5.

        Args:
            sim_id: Simulation identifier
            sample_indices: Indices to sample
            annual_losses: Annual loss amounts
            insurance_recoveries: Insurance recovery amounts
            retained_losses: Retained loss amounts
        """
        if not self._hdf5_file:
            return

        # Create dataset for this simulation
        ts_group = self._hdf5_file["time_series"]
        sim_group = ts_group.create_group(f"sim_{sim_id}")

        # Store sampled data with compression
        compression = "gzip" if self.config.compression else None
        compression_opts = self.config.compression_level if self.config.compression else None

        sim_group.create_dataset(
            "annual_losses",
            data=annual_losses[sample_indices].astype(self.config.dtype),
            compression=compression,
            compression_opts=compression_opts,
        )
        sim_group.create_dataset(
            "insurance_recoveries",
            data=insurance_recoveries[sample_indices].astype(self.config.dtype),
            compression=compression,
            compression_opts=compression_opts,
        )
        sim_group.create_dataset(
            "retained_losses",
            data=retained_losses[sample_indices].astype(self.config.dtype),
            compression=compression,
            compression_opts=compression_opts,
        )
        sim_group.attrs["sample_indices"] = sample_indices

        # Flush to disk
        self._hdf5_file.flush()

    def load_simulation(self, sim_id: int, load_time_series: bool = False) -> Dict[str, Any]:
        """Load simulation data with lazy loading.

        Args:
            sim_id: Simulation identifier
            load_time_series: Whether to load time series data

        Returns:
            Dictionary with simulation data
        """
        result = {}

        # Load summary if available
        if sim_id in self._summaries:
            result["summary"] = self._summaries[sim_id].to_dict()
        else:
            # Try loading from disk
            summary = self._load_summary_from_disk(sim_id)
            if summary:
                result["summary"] = summary.to_dict()

        # Load time series if requested
        if load_time_series:
            time_series = self._load_time_series(sim_id)
            if time_series:
                result["time_series"] = time_series

        return result

    def _load_summary_from_disk(self, sim_id: int) -> Optional[SimulationSummary]:
        """Load summary from disk storage.

        Args:
            sim_id: Simulation identifier

        Returns:
            SimulationSummary or None if not found
        """
        if self.config.backend == "hdf5" and self._hdf5_file:
            if f"sim_{sim_id}" in self._hdf5_file["summaries"]:
                data = self._hdf5_file[f"summaries/sim_{sim_id}"]
                return SimulationSummary(**{k: data.attrs[k] for k in data.attrs})
        else:
            summary_file = self.storage_path / "summaries" / f"sim_{sim_id}.json"
            if summary_file.exists():
                with open(summary_file, "r") as f:
                    data = json.load(f)
                    return SimulationSummary(**data)
        return None

    def _load_time_series(self, sim_id: int) -> Optional[Dict[str, np.ndarray]]:
        """Load time series data for a simulation.

        Args:
            sim_id: Simulation identifier

        Returns:
            Dictionary with time series arrays or None
        """
        if self.config.backend == "hdf5" and self._hdf5_file:
            if f"sim_{sim_id}" in self._hdf5_file["time_series"]:
                sim_group = self._hdf5_file[f"time_series/sim_{sim_id}"]
                return {
                    "annual_losses": np.array(sim_group["annual_losses"]),
                    "insurance_recoveries": np.array(sim_group["insurance_recoveries"]),
                    "retained_losses": np.array(sim_group["retained_losses"]),
                    "sample_indices": sim_group.attrs["sample_indices"],
                }
        else:
            ts_path = self.storage_path / "time_series" / f"sim_{sim_id}.dat"
            if ts_path.exists():
                # Determine shape from first file
                mmap = np.memmap(ts_path, dtype=self.config.dtype, mode="r")
                # Reshape assuming 3 rows (losses, recoveries, retained)
                n_samples = len(mmap) // 3
                reshaped = np.array(mmap).reshape((3, n_samples))
                return {
                    "annual_losses": reshaped[0],
                    "insurance_recoveries": reshaped[1],
                    "retained_losses": reshaped[2],
                }
        return None

    def export_summaries_csv(self, output_path: str) -> None:
        """Export all summary statistics to CSV.

        Args:
            output_path: Path for CSV output file
        """
        # Ensure all summaries are persisted
        self._persist_summaries()

        # Collect all summaries
        all_summaries: List[SimulationSummary] = []

        # From memory
        all_summaries.extend(self._summaries.values())

        # From disk (if using memmap)
        if self.config.backend == "memmap":
            summary_dir = self.storage_path / "summaries"
            for summary_file in summary_dir.glob("*.json"):
                with open(summary_file, "r") as f:
                    data = json.load(f)
                    all_summaries.append(SimulationSummary(**data))

        # Write to CSV
        if all_summaries:
            df = pd.DataFrame([s.to_dict() for s in all_summaries])
            df.to_csv(output_path, index=False)
            print(f"Exported {len(all_summaries)} summaries to {output_path}")

    def export_summaries_json(self, output_path: str) -> None:
        """Export all summary statistics to JSON.

        Args:
            output_path: Path for JSON output file
        """
        # Ensure all summaries are persisted
        self._persist_summaries()

        # Collect all summaries
        all_summaries: List[Dict[str, Any]] = []

        # From memory
        all_summaries.extend([s.to_dict() for s in self._summaries.values()])

        # From disk (if using memmap)
        if self.config.backend == "memmap":
            summary_dir = self.storage_path / "summaries"
            for summary_file in summary_dir.glob("*.json"):
                with open(summary_file, "r") as f:
                    all_summaries.append(json.load(f))

        # Write to JSON
        with open(output_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"Exported {len(all_summaries)} summaries to {output_path}")

    def _persist_summaries(self) -> None:
        """Persist in-memory summaries to disk."""
        if self.config.backend == "hdf5" and self._hdf5_file:
            # Store in HDF5
            summary_group = self._hdf5_file["summaries"]
            for sim_id, summary in self._summaries.items():
                if f"sim_{sim_id}" not in summary_group:
                    sim_group = summary_group.create_group(f"sim_{sim_id}")
                    for key, value in summary.to_dict().items():
                        sim_group.attrs[key] = value if value is not None else -1
            self._hdf5_file.flush()
        else:
            # Store as JSON files
            summary_dir = self.storage_path / "summaries"
            for sim_id, summary in self._summaries.items():
                summary_file = summary_dir / f"sim_{sim_id}.json"
                with open(summary_file, "w") as f:
                    json.dump(summary.to_dict(), f)

        # Clear memory cache after persisting
        self._summaries.clear()

    def _check_disk_space(self) -> bool:
        """Check if disk usage is within limits.

        Returns:
            True if within limits, False otherwise
        """
        # Calculate current disk usage
        total_size = 0
        for path in self.storage_path.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size

        self._disk_usage = total_size / (1024**3)  # Convert to GB
        return self._disk_usage < self.config.max_disk_usage_gb

    def _cleanup_memory(self) -> None:
        """Periodic cleanup to manage memory usage."""
        # Persist any cached summaries
        if self._summaries:
            self._persist_summaries()

        # Force garbage collection
        gc.collect()

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        self._check_disk_space()

        return {
            "total_simulations": self._total_simulations,
            "disk_usage_gb": self._disk_usage,
            "disk_limit_gb": self.config.max_disk_usage_gb,
            "backend": self.config.backend,
            "sample_interval": self.config.sample_interval,
            "compression_enabled": self.config.compression,
            "storage_directory": str(self.storage_path),
        }

    def clear_storage(self) -> None:
        """Clear all stored data."""
        # Close HDF5 file if open
        if self._hdf5_file:
            self._hdf5_file.close()
            self._hdf5_file = None

        # Clear memory caches
        self._summaries.clear()
        self._memmap_files.clear()

        # Remove storage directory
        if self.storage_path.exists():
            shutil.rmtree(self.storage_path)
            self.storage_path.mkdir(parents=True, exist_ok=True)

        # Reset counters
        self._total_simulations = 0
        self._disk_usage = 0.0

        # Reinitialize backend
        if self.config.backend == "hdf5":
            self._setup_hdf5()
        else:
            self._setup_memmap()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure data is persisted."""
        # Persist any remaining summaries
        if self._summaries:
            self._persist_summaries()

        # Close HDF5 file if open
        if self._hdf5_file:
            self._hdf5_file.close()
            self._hdf5_file = None

        # Clear memory-mapped files
        self._memmap_files.clear()
