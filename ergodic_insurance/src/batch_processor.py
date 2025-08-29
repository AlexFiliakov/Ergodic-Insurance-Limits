"""Batch processing engine for running multiple simulation scenarios.

This module provides a framework for executing multiple scenarios in parallel
or serial, with support for checkpointing, resumption, and result aggregation.
"""

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import pickle
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from .excel_reporter import ExcelReportConfig, ExcelReporter
from .insurance_program import InsuranceProgram
from .loss_distributions import ManufacturingLossGenerator
from .manufacturer import WidgetManufacturer
from .monte_carlo import MonteCarloEngine, SimulationConfig, SimulationResults
from .parallel_executor import ParallelExecutor
from .scenario_manager import ScenarioConfig, ScenarioManager


class ProcessingStatus(Enum):
    """Status of scenario processing."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BatchResult:
    """Result from a single scenario execution.

    Attributes:
        scenario_id: Unique scenario identifier
        scenario_name: Human-readable scenario name
        status: Processing status
        simulation_results: Monte Carlo simulation results
        execution_time: Time taken to execute scenario
        error_message: Error message if failed
        metadata: Additional result metadata
    """

    scenario_id: str
    scenario_name: str
    status: ProcessingStatus
    simulation_results: Optional[SimulationResults] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedResults:
    """Aggregated results from batch processing.

    Attributes:
        batch_results: Individual scenario results
        summary_statistics: Summary stats across scenarios
        comparison_metrics: Comparative metrics between scenarios
        sensitivity_analysis: Sensitivity analysis results
        execution_summary: Batch execution summary
    """

    batch_results: List[BatchResult]
    summary_statistics: pd.DataFrame
    comparison_metrics: Dict[str, pd.DataFrame]
    sensitivity_analysis: Optional[pd.DataFrame] = None
    execution_summary: Dict[str, Any] = field(default_factory=dict)

    def get_successful_results(self) -> List[BatchResult]:
        """Get only successful results."""
        return [r for r in self.batch_results if r.status == ProcessingStatus.COMPLETED]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis.

        Returns:
            DataFrame with scenario results
        """
        data = []
        for result in self.batch_results:
            row = {
                "scenario_id": result.scenario_id,
                "scenario_name": result.scenario_name,
                "status": result.status.value,
                "execution_time": result.execution_time,
            }

            # Add simulation metrics if available
            if result.simulation_results:
                row.update(
                    {
                        "ruin_probability": result.simulation_results.ruin_probability,
                        "mean_growth_rate": np.mean(result.simulation_results.growth_rates),
                        "mean_final_assets": np.mean(result.simulation_results.final_assets),
                        "var_99": result.simulation_results.metrics.get("var_99", np.nan),
                        "tvar_99": result.simulation_results.metrics.get("tvar_99", np.nan),
                    }
                )

            data.append(row)

        return pd.DataFrame(data)


@dataclass
class CheckpointData:
    """Checkpoint data for resumable batch processing."""

    completed_scenarios: Set[str]
    failed_scenarios: Set[str]
    batch_results: List[BatchResult]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class BatchProcessor:
    """Engine for batch processing multiple simulation scenarios."""

    def __init__(
        self,
        loss_generator: Optional[ManufacturingLossGenerator] = None,
        insurance_program: Optional[InsuranceProgram] = None,
        manufacturer: Optional[WidgetManufacturer] = None,
        n_workers: Optional[int] = None,
        checkpoint_dir: Optional[Path] = None,
        use_parallel: bool = True,
        progress_bar: bool = True,
    ):
        """Initialize batch processor.

        Args:
            loss_generator: Loss event generator
            insurance_program: Insurance program structure
            manufacturer: Manufacturing company model
            n_workers: Number of parallel workers
            checkpoint_dir: Directory for checkpoints
            use_parallel: Whether to use parallel processing
            progress_bar: Whether to show progress bar
        """
        self.loss_generator = loss_generator
        self.insurance_program = insurance_program
        self.manufacturer = manufacturer
        self.n_workers = n_workers
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints/batch")
        self.use_parallel = use_parallel
        self.progress_bar = progress_bar

        # Processing state
        self.batch_results: List[BatchResult] = []
        self.completed_scenarios: Set[str] = set()
        self.failed_scenarios: Set[str] = set()

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def process_batch(
        self,
        scenarios: List[ScenarioConfig],
        resume_from_checkpoint: bool = True,
        checkpoint_interval: int = 10,
        max_failures: Optional[int] = None,
        priority_threshold: Optional[int] = None,
    ) -> AggregatedResults:
        """Process a batch of scenarios.

        Args:
            scenarios: List of scenarios to process
            resume_from_checkpoint: Whether to resume from checkpoint
            checkpoint_interval: Save checkpoint every N scenarios
            max_failures: Maximum allowed failures before stopping
            priority_threshold: Only process scenarios up to this priority

        Returns:
            Aggregated results from batch processing
        """
        start_time = time.time()

        # Filter by priority if specified
        if priority_threshold is not None:
            scenarios = [s for s in scenarios if s.priority <= priority_threshold]

        # Sort by priority
        scenarios = sorted(scenarios, key=lambda x: x.priority)

        # Resume from checkpoint if requested
        if resume_from_checkpoint:
            self._load_checkpoint()

        # Filter out completed scenarios
        pending_scenarios = [s for s in scenarios if s.scenario_id not in self.completed_scenarios]

        if not pending_scenarios:
            print("All scenarios already completed.")
            return self._aggregate_results()

        print(f"Processing {len(pending_scenarios)} scenarios...")

        # Process scenarios
        if self.use_parallel and len(pending_scenarios) > 1:
            results = self._process_parallel(pending_scenarios, checkpoint_interval, max_failures)
        else:
            results = self._process_serial(pending_scenarios, checkpoint_interval, max_failures)

        # Add results
        self.batch_results.extend(results)

        # Final checkpoint
        self._save_checkpoint()

        # Aggregate results
        aggregated = self._aggregate_results()

        # Add execution summary
        aggregated.execution_summary = {
            "total_scenarios": len(scenarios),
            "completed": len(self.completed_scenarios),
            "failed": len(self.failed_scenarios),
            "skipped": len(scenarios) - len(pending_scenarios),
            "execution_time": time.time() - start_time,
            "average_time_per_scenario": (
                (time.time() - start_time) / len(pending_scenarios) if pending_scenarios else 0
            ),
        }

        return aggregated

    def _process_serial(
        self, scenarios: List[ScenarioConfig], checkpoint_interval: int, max_failures: Optional[int]
    ) -> List[BatchResult]:
        """Process scenarios serially.

        Args:
            scenarios: Scenarios to process
            checkpoint_interval: Checkpoint frequency
            max_failures: Maximum failures allowed

        Returns:
            List of batch results
        """
        results = []
        failures = 0

        iterator = tqdm(scenarios, desc="Processing scenarios") if self.progress_bar else scenarios

        for i, scenario in enumerate(iterator):
            # Check failure limit
            if max_failures and failures >= max_failures:
                print(f"Stopping batch: reached {max_failures} failures")
                break

            # Process scenario
            result = self._process_scenario(scenario)
            results.append(result)

            # Update state
            if result.status == ProcessingStatus.COMPLETED:
                self.completed_scenarios.add(scenario.scenario_id)
            elif result.status == ProcessingStatus.FAILED:
                self.failed_scenarios.add(scenario.scenario_id)
                failures += 1

            # Checkpoint periodically
            if (i + 1) % checkpoint_interval == 0:
                self.batch_results.extend(results)
                self._save_checkpoint()
                results.clear()

        return results

    def _process_parallel(
        self, scenarios: List[ScenarioConfig], checkpoint_interval: int, max_failures: Optional[int]
    ) -> List[BatchResult]:
        """Process scenarios in parallel.

        Args:
            scenarios: Scenarios to process
            checkpoint_interval: Checkpoint frequency
            max_failures: Maximum failures allowed

        Returns:
            List of batch results
        """
        results = []
        failures = 0

        # Use process pool for parallel execution
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all scenarios
            future_to_scenario = {
                executor.submit(self._process_scenario, scenario): scenario
                for scenario in scenarios
            }

            # Process completed futures
            iterator = as_completed(future_to_scenario)
            if self.progress_bar:
                iterator = tqdm(iterator, total=len(scenarios), desc="Processing scenarios")

            for i, future in enumerate(iterator):
                # Check failure limit
                if max_failures and failures >= max_failures:
                    # Cancel remaining futures
                    for f in future_to_scenario:
                        f.cancel()
                    print(f"Stopping batch: reached {max_failures} failures")
                    break

                # Get result
                scenario = future_to_scenario[future]
                try:
                    result = future.result()
                except (ValueError, RuntimeError, TypeError) as e:
                    result = BatchResult(
                        scenario_id=scenario.scenario_id,
                        scenario_name=scenario.name,
                        status=ProcessingStatus.FAILED,
                        error_message=str(e),
                    )

                results.append(result)

                # Update state
                if result.status == ProcessingStatus.COMPLETED:
                    self.completed_scenarios.add(scenario.scenario_id)
                elif result.status == ProcessingStatus.FAILED:
                    self.failed_scenarios.add(scenario.scenario_id)
                    failures += 1

                # Checkpoint periodically
                if (i + 1) % checkpoint_interval == 0:
                    self.batch_results.extend(results)
                    self._save_checkpoint()
                    results.clear()

        return results

    def _apply_overrides(self, obj: Any, prefix: str, overrides: Dict[str, Any]) -> Any:
        """Apply parameter overrides to an object.

        Args:
            obj: Object to apply overrides to
            prefix: Prefix to match in parameter paths
            overrides: Parameter overrides

        Returns:
            Modified copy of object or original if no overrides
        """
        if not obj or not overrides:
            return obj

        import copy

        obj_copy = copy.deepcopy(obj)
        for param_path, value in overrides.items():
            if param_path.startswith(prefix):
                param = param_path.replace(prefix, "")
                if hasattr(obj_copy, param):
                    setattr(obj_copy, param, value)
        return obj_copy

    def _process_scenario(self, scenario: ScenarioConfig) -> BatchResult:
        """Process a single scenario.

        Args:
            scenario: Scenario to process

        Returns:
            Batch result for the scenario
        """
        start_time = time.time()

        try:
            # Apply configuration overrides
            overrides = scenario.parameter_overrides or {}
            manufacturer = self._apply_overrides(self.manufacturer, "manufacturer.", overrides)
            insurance_program = self._apply_overrides(
                self.insurance_program, "insurance.", overrides
            )
            loss_generator = self._apply_overrides(self.loss_generator, "loss.", overrides)

            # Create Monte Carlo engine for this scenario
            if not all([loss_generator, insurance_program, manufacturer]):
                raise ValueError(
                    "BatchProcessor requires loss_generator, insurance_program, "
                    "and manufacturer to be initialized"
                )

            # Assert types for mypy
            assert loss_generator is not None
            assert insurance_program is not None
            assert manufacturer is not None

            monte_carlo_engine = MonteCarloEngine(
                loss_generator=loss_generator,
                insurance_program=insurance_program,
                manufacturer=manufacturer,
                config=scenario.simulation_config or SimulationConfig(),
            )

            # Run simulation
            simulation_results = monte_carlo_engine.run()

            return BatchResult(
                scenario_id=scenario.scenario_id,
                scenario_name=scenario.name,
                status=ProcessingStatus.COMPLETED,
                simulation_results=simulation_results,
                execution_time=time.time() - start_time,
                metadata={"tags": list(scenario.tags)},
            )

        except (ValueError, RuntimeError, TypeError) as e:
            return BatchResult(
                scenario_id=scenario.scenario_id,
                scenario_name=scenario.name,
                status=ProcessingStatus.FAILED,
                execution_time=time.time() - start_time,
                error_message=str(e),
            )

    def _aggregate_results(self) -> AggregatedResults:
        """Aggregate results across scenarios.

        Returns:
            Aggregated results
        """
        # Create summary statistics
        summary_data = []
        for result in self.batch_results:
            if result.status == ProcessingStatus.COMPLETED and result.simulation_results:
                sim_results = result.simulation_results
                summary_data.append(
                    {
                        "scenario": result.scenario_name,
                        "ruin_probability": sim_results.ruin_probability,
                        "mean_growth_rate": np.mean(sim_results.growth_rates),
                        "std_growth_rate": np.std(sim_results.growth_rates),
                        "mean_final_assets": np.mean(sim_results.final_assets),
                        "median_final_assets": np.median(sim_results.final_assets),
                        "var_95": sim_results.metrics.get("var_95", np.nan),
                        "var_99": sim_results.metrics.get("var_99", np.nan),
                        "tvar_95": sim_results.metrics.get("tvar_95", np.nan),
                        "tvar_99": sim_results.metrics.get("tvar_99", np.nan),
                        "execution_time": result.execution_time,
                    }
                )

        summary_df = pd.DataFrame(summary_data) if summary_data else pd.DataFrame()

        # Create comparison metrics
        comparison_metrics = {}

        if len(summary_data) > 1:
            # Relative performance matrix
            baseline_idx = 0  # First scenario as baseline
            if not summary_df.empty:
                relative_performance = pd.DataFrame(index=summary_df["scenario"])
                for metric in ["mean_growth_rate", "mean_final_assets", "ruin_probability"]:
                    if metric in summary_df.columns:
                        baseline_value = summary_df[metric].iloc[baseline_idx]
                        if baseline_value != 0:
                            relative_performance[f"{metric}_relative"] = (
                                summary_df[metric] / baseline_value
                            )
                comparison_metrics["relative_performance"] = relative_performance

            # Ranking by different metrics
            if not summary_df.empty:
                ranking_df = pd.DataFrame(index=summary_df["scenario"])
                for metric in ["mean_growth_rate", "mean_final_assets"]:
                    if metric in summary_df.columns:
                        ranking_df[f"{metric}_rank"] = (
                            summary_df[metric].rank(ascending=False).astype(int)
                        )

                # Ruin probability ranked ascending (lower is better)
                if "ruin_probability" in summary_df.columns:
                    ranking_df["ruin_probability_rank"] = (
                        summary_df["ruin_probability"].rank(ascending=True).astype(int)
                    )

                comparison_metrics["rankings"] = ranking_df

        # Perform sensitivity analysis if tagged scenarios exist
        sensitivity_results = self._perform_sensitivity_analysis()

        return AggregatedResults(
            batch_results=self.batch_results,
            summary_statistics=summary_df,
            comparison_metrics=comparison_metrics,
            sensitivity_analysis=sensitivity_results,
            execution_summary={},
        )

    def _perform_sensitivity_analysis(self) -> Optional[pd.DataFrame]:
        """Perform sensitivity analysis on results.

        Returns:
            Sensitivity analysis DataFrame or None
        """
        # Find baseline and sensitivity scenarios
        baseline_results = [
            r for r in self.batch_results if "baseline" in r.metadata.get("tags", [])
        ]

        if not baseline_results:
            return None

        baseline = baseline_results[0]
        if not baseline.simulation_results:
            return None

        sensitivity_data = []

        # Compare each sensitivity scenario to baseline
        for result in self.batch_results:
            if (
                result.status == ProcessingStatus.COMPLETED
                and result.simulation_results
                and "sensitivity" in result.metadata.get("tags", [])
                and "baseline" not in result.metadata.get("tags", [])
            ):
                # Calculate percentage changes
                baseline_growth = np.mean(baseline.simulation_results.growth_rates)
                scenario_growth = np.mean(result.simulation_results.growth_rates)

                sensitivity_data.append(
                    {
                        "scenario": result.scenario_name,
                        "growth_rate_change_pct": (
                            (scenario_growth - baseline_growth) / baseline_growth * 100
                            if baseline_growth != 0
                            else np.nan
                        ),
                        "ruin_prob_change_pct": (
                            (
                                result.simulation_results.ruin_probability
                                - baseline.simulation_results.ruin_probability
                            )
                            / baseline.simulation_results.ruin_probability
                            * 100
                            if baseline.simulation_results.ruin_probability != 0
                            else np.nan
                        ),
                        "final_assets_change_pct": (
                            (
                                np.mean(result.simulation_results.final_assets)
                                - np.mean(baseline.simulation_results.final_assets)
                            )
                            / np.mean(baseline.simulation_results.final_assets)
                            * 100
                        ),
                    }
                )

        return pd.DataFrame(sensitivity_data) if sensitivity_data else None

    def _save_checkpoint(self) -> None:
        """Save checkpoint to disk."""
        checkpoint = CheckpointData(
            completed_scenarios=self.completed_scenarios,
            failed_scenarios=self.failed_scenarios,
            batch_results=self.batch_results.copy(),
            timestamp=datetime.now(),
            metadata={
                "n_completed": len(self.completed_scenarios),
                "n_failed": len(self.failed_scenarios),
            },
        )

        checkpoint_path = self.checkpoint_dir / f"checkpoint_{datetime.now():%Y%m%d_%H%M%S}.pkl"
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)

        # Keep only the latest checkpoint
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        if len(checkpoints) > 3:  # Keep last 3 checkpoints
            for old_checkpoint in checkpoints[:-3]:
                old_checkpoint.unlink()

    def _load_checkpoint(self) -> bool:
        """Load checkpoint from disk.

        Returns:
            True if checkpoint loaded, False otherwise
        """
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        if not checkpoints:
            return False

        latest_checkpoint = checkpoints[-1]
        print(f"Loading checkpoint from {latest_checkpoint}")

        with open(latest_checkpoint, "rb") as f:
            checkpoint: CheckpointData = pickle.load(f)

        self.completed_scenarios = checkpoint.completed_scenarios
        self.failed_scenarios = checkpoint.failed_scenarios
        self.batch_results = checkpoint.batch_results

        print(
            f"Resumed from checkpoint: {len(self.completed_scenarios)} completed, "
            f"{len(self.failed_scenarios)} failed"
        )

        return True

    def clear_checkpoints(self) -> None:
        """Clear all checkpoints."""
        for checkpoint in self.checkpoint_dir.glob("checkpoint_*.pkl"):
            checkpoint.unlink()

        self.completed_scenarios.clear()
        self.failed_scenarios.clear()
        self.batch_results.clear()

    def export_results(self, path: Union[str, Path], export_format: str = "csv") -> None:
        """Export aggregated results to file.

        Args:
            path: Output file path
            export_format: Export format (csv, json, excel)
        """
        path = Path(path)
        aggregated = self._aggregate_results()

        if export_format == "csv":
            aggregated.to_dataframe().to_csv(path, index=False)
        elif export_format == "json":
            data = {
                "summary": aggregated.summary_statistics.to_dict("records"),
                "execution_summary": aggregated.execution_summary,
                "batch_results": [
                    {
                        "scenario_id": r.scenario_id,
                        "scenario_name": r.scenario_name,
                        "status": r.status.value,
                        "execution_time": r.execution_time,
                        "error_message": r.error_message,
                    }
                    for r in aggregated.batch_results
                ],
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        elif export_format == "excel":
            # Ensure proper file closure on Windows by explicitly saving
            writer = pd.ExcelWriter(path, engine="openpyxl")
            try:
                aggregated.summary_statistics.to_excel(writer, sheet_name="Summary", index=False)
                aggregated.to_dataframe().to_excel(writer, sheet_name="Details", index=False)
                if aggregated.sensitivity_analysis is not None:
                    aggregated.sensitivity_analysis.to_excel(
                        writer, sheet_name="Sensitivity", index=False
                    )
                for name, df in aggregated.comparison_metrics.items():
                    df.to_excel(writer, sheet_name=name[:31])  # Excel sheet name limit
            finally:
                writer.close()
        elif export_format == "excel_financial":
            # Use the comprehensive Excel reporter for financial statements
            self.export_financial_statements(path)

    def export_financial_statements(self, path: Union[str, Path]) -> None:
        """Export comprehensive financial statements to Excel.

        Generates detailed financial statements including balance sheets,
        income statements, cash flow statements, reconciliation reports,
        and metrics dashboards for each scenario.

        Args:
            path: Output directory path for Excel files
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Configure Excel reporter
        excel_config = ExcelReportConfig(
            output_path=path,
            include_balance_sheet=True,
            include_income_statement=True,
            include_cash_flow=True,
            include_reconciliation=True,
            include_metrics_dashboard=True,
            include_pivot_data=True,
        )

        reporter = ExcelReporter(excel_config)

        # Generate reports for each completed scenario
        for result in self.batch_results:
            if result.status == ProcessingStatus.COMPLETED and result.simulation_results:
                output_file = f"financial_report_{result.scenario_name}.xlsx"
                try:
                    # For now, generate Monte Carlo report since we have MC results
                    # TODO: Add support for extracting individual trajectories  # pylint: disable=fixme
                    reporter.generate_monte_carlo_report(
                        result.simulation_results,
                        output_file,
                        title=f"Financial Report - {result.scenario_name}",
                    )
                    print(f"Generated financial report: {path / output_file}")
                except (OSError, ValueError, KeyError, AttributeError) as e:
                    print(f"Error generating report for {result.scenario_name}: {e}")
