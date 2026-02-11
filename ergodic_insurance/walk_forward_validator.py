"""Walk-forward validation system for insurance strategy testing.

This module implements a rolling window validation framework that tests
insurance strategies across multiple time periods to detect overfitting
and ensure robustness of insurance decisions.

Example:
    >>> from walk_forward_validator import WalkForwardValidator
    >>> from strategy_backtester import ConservativeFixedStrategy, AdaptiveStrategy

    >>> # Create validator with 3-year windows
    >>> validator = WalkForwardValidator(
    ...     window_size=3,
    ...     step_size=1,
    ...     test_ratio=0.3
    ... )
    >>>
    >>> # Define strategies to test
    >>> strategies = [
    ...     ConservativeFixedStrategy(),
    ...     AdaptiveStrategy()
    ... ]
    >>>
    >>> # Run walk-forward validation
    >>> results = validator.validate_strategies(
    ...     strategies=strategies,
    ...     n_years=10,
    ...     n_simulations=1000
    ... )
    >>>
    >>> # Generate reports
    >>> validator.generate_report(results, output_dir="./reports")
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence

from jinja2 import Template
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import Config
from .manufacturer import WidgetManufacturer
from .monte_carlo import SimulationConfig
from .simulation import Simulation
from .strategy_backtester import InsuranceStrategy, StrategyBacktester
from .validation_metrics import (
    MetricCalculator,
    PerformanceTargets,
    StrategyPerformance,
    ValidationMetrics,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationWindow:
    """Represents a single validation window.

    Attributes:
        window_id: Unique identifier for the window
        train_start: Start year of training period
        train_end: End year of training period
        test_start: Start year of testing period
        test_end: End year of testing period
    """

    window_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    def get_train_years(self) -> int:
        """Get number of training years."""
        return self.train_end - self.train_start

    def get_test_years(self) -> int:
        """Get number of testing years."""
        return self.test_end - self.test_start

    def __str__(self) -> str:
        """String representation."""
        return (
            f"Window {self.window_id}: "
            f"Train[{self.train_start}-{self.train_end}], "
            f"Test[{self.test_start}-{self.test_end}]"
        )


@dataclass
class WindowResult:
    """Results from a single validation window.

    Attributes:
        window: The validation window
        strategy_performances: Performance by strategy name
        optimization_params: Optimized parameters if applicable
        execution_time: Time to process window
    """

    window: ValidationWindow
    strategy_performances: Dict[str, StrategyPerformance] = field(default_factory=dict)
    optimization_params: Dict[str, Dict[str, float]] = field(default_factory=dict)
    execution_time: float = 0.0


@dataclass
class ValidationResult:
    """Complete walk-forward validation results.

    Attributes:
        window_results: Results for each window
        strategy_rankings: Overall strategy rankings
        overfitting_analysis: Overfitting detection results
        consistency_scores: Strategy consistency across windows
        best_strategy: Recommended strategy based on validation
        metadata: Additional validation metadata
    """

    window_results: List[WindowResult] = field(default_factory=list)
    strategy_rankings: pd.DataFrame = field(default_factory=pd.DataFrame)
    overfitting_analysis: Dict[str, float] = field(default_factory=dict)
    consistency_scores: Dict[str, float] = field(default_factory=dict)
    best_strategy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _process_window_worker(
    window: ValidationWindow,
    strategies: Sequence[InsuranceStrategy],
    n_simulations: int,
    manufacturer: WidgetManufacturer,
    config: Config,
    simulation_engine: Optional[Simulation] = None,
    window_seed: Optional[np.random.SeedSequence] = None,
) -> WindowResult:
    """Process a single validation window in a worker process.

    Module-level function for ProcessPoolExecutor pickle compatibility.
    Follows the existing pattern from monte_carlo_worker.py.

    Args:
        window: Validation window to process
        strategies: Strategies to test
        n_simulations: Number of simulations
        manufacturer: Manufacturer instance
        config: Configuration object
        simulation_engine: Optional simulation engine for optimization
        window_seed: SeedSequence for this window, spawned from the
            validator's base SeedSequence. Guarantees statistically
            independent streams across windows.

    Returns:
        WindowResult for the window.
    """
    start_time = time.time()

    backtester = StrategyBacktester(simulation_engine=simulation_engine)
    window_result = WindowResult(window=window)

    # Derive independent train/test seeds via SeedSequence.spawn()
    if window_seed is None:
        window_seed = np.random.SeedSequence(window.window_id)
    train_ss, test_ss = window_seed.spawn(2)

    train_config = SimulationConfig(
        n_simulations=n_simulations,
        n_years=window.get_train_years(),
        seed=int(train_ss.generate_state(1)[0]),
    )
    test_config = SimulationConfig(
        n_simulations=n_simulations,
        n_years=window.get_test_years(),
        seed=int(test_ss.generate_state(1)[0]),
    )

    for strategy in strategies:
        strategy.reset()

        train_result = backtester.test_strategy(
            strategy=strategy,
            manufacturer=manufacturer,
            config=train_config,
            use_cache=False,
        )

        if hasattr(strategy, "optimized_params") and strategy.optimized_params:
            window_result.optimization_params[strategy.name] = strategy.optimized_params.copy()

        test_result = backtester.test_strategy(
            strategy=strategy,
            manufacturer=manufacturer,
            config=test_config,
            use_cache=False,
        )

        performance = StrategyPerformance(
            strategy_name=strategy.name,
            in_sample_metrics=train_result.metrics,
            out_sample_metrics=test_result.metrics,
        )
        performance.calculate_degradation()
        window_result.strategy_performances[strategy.name] = performance

    window_result.execution_time = time.time() - start_time
    return window_result


class WalkForwardValidator:
    """Walk-forward validation system for insurance strategies."""

    def __init__(
        self,
        window_size: int = 3,
        step_size: int = 1,
        test_ratio: float = 0.3,
        simulation_engine: Optional[Simulation] = None,
        backtester: Optional[StrategyBacktester] = None,
        performance_targets: Optional[PerformanceTargets] = None,
        max_workers: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """Initialize walk-forward validator.

        Args:
            window_size: Size of each window in years
            step_size: Step between windows in years
            test_ratio: Ratio of window used for testing (0-1)
            simulation_engine: Engine for running simulations
            backtester: Strategy backtesting engine
            performance_targets: Optional performance targets
            max_workers: Maximum worker processes for parallel window evaluation.
                None (default) auto-detects based on CPU count and window count.
                Set to 1 to force sequential processing.
            seed: Base random seed for reproducibility. Uses
                ``np.random.SeedSequence`` internally to derive statistically
                independent per-window seeds. None uses system entropy.
        """
        self.window_size = window_size
        self.step_size = step_size
        self.test_ratio = test_ratio
        self.simulation_engine = simulation_engine
        self.backtester = backtester or StrategyBacktester(self.simulation_engine)
        self.performance_targets = performance_targets
        self.metric_calculator = MetricCalculator()
        self.max_workers = max_workers
        self.seed = seed

    def generate_windows(self, total_years: int) -> List[ValidationWindow]:
        """Generate validation windows.

        Args:
            total_years: Total years of data available

        Returns:
            List of validation windows.
        """
        windows = []
        window_id = 0
        current_start = 0

        while current_start + self.window_size <= total_years:
            # Calculate train/test split
            train_years = int(self.window_size * (1 - self.test_ratio))
            _test_years = self.window_size - train_years

            # Create window
            window = ValidationWindow(
                window_id=window_id,
                train_start=current_start,
                train_end=current_start + train_years,
                test_start=current_start + train_years,
                test_end=current_start + self.window_size,
            )
            windows.append(window)

            # Move to next window
            current_start += self.step_size
            window_id += 1

        logger.info(f"Generated {len(windows)} validation windows")
        return windows

    def validate_strategies(
        self,
        strategies: List[InsuranceStrategy],
        n_years: int = 10,
        n_simulations: int = 1000,
        manufacturer: Optional[WidgetManufacturer] = None,
        config: Optional[Config] = None,
    ) -> ValidationResult:
        """Validate strategies using walk-forward analysis.

        Args:
            strategies: List of strategies to validate
            n_years: Total years for validation
            n_simulations: Number of simulations per test
            manufacturer: Manufacturer instance
            config: Configuration object

        Returns:
            ValidationResult with complete analysis.
        """
        logger.info(f"Starting walk-forward validation for {len(strategies)} strategies")

        # Initialize manufacturer and config if not provided
        if manufacturer is None:
            from ergodic_insurance.config import ManufacturerConfig

            default_mfg_config = ManufacturerConfig(
                initial_assets=10000000,
                asset_turnover_ratio=1.0,
                base_operating_margin=0.08,
                tax_rate=0.25,
                retention_ratio=0.7,
            )
            manufacturer = WidgetManufacturer(default_mfg_config)
        if config is None:
            from ergodic_insurance.config import (
                Config,
                DebtConfig,
                GrowthConfig,
                LoggingConfig,
                ManufacturerConfig,
                OutputConfig,
            )
            from ergodic_insurance.config import (
                WorkingCapitalConfig,
            )
            from ergodic_insurance.config import SimulationConfig as SimConfig

            config = Config(
                manufacturer=ManufacturerConfig(
                    initial_assets=10000000,
                    asset_turnover_ratio=1.0,
                    base_operating_margin=0.08,
                    tax_rate=0.25,
                    retention_ratio=0.7,
                ),
                working_capital=WorkingCapitalConfig(
                    percent_of_sales=0.15,
                ),
                growth=GrowthConfig(
                    type="deterministic",
                    annual_growth_rate=0.05,
                    volatility=0.15,
                ),
                debt=DebtConfig(
                    interest_rate=0.05,
                    max_leverage_ratio=2.0,
                    minimum_cash_balance=100000,
                ),
                simulation=SimConfig(
                    time_resolution="annual",
                    time_horizon_years=10,
                ),
                output=OutputConfig(
                    output_directory="./results",
                    file_format="csv",
                    checkpoint_frequency=0,
                    detailed_metrics=True,
                ),
                logging=LoggingConfig(
                    enabled=True,
                    level="INFO",
                    log_file=None,
                ),
            )

        # Generate windows
        windows = self.generate_windows(n_years)

        # Derive per-window SeedSequences via spawn() for statistical independence
        base_ss = np.random.SeedSequence(self.seed)
        window_seeds = base_ss.spawn(len(windows))

        # Determine parallelism
        n_workers = self.max_workers
        if n_workers is None:
            n_workers = min(os.cpu_count() or 1, len(windows))

        if n_workers > 1 and len(windows) > 1:
            # Parallel window processing
            logger.info(
                f"Processing {len(windows)} windows in parallel " f"with {n_workers} workers"
            )
            window_results = self._process_windows_parallel(
                windows=windows,
                strategies=strategies,
                n_simulations=n_simulations,
                manufacturer=manufacturer,
                config=config,
                n_workers=n_workers,
                window_seeds=window_seeds,
            )
        else:
            # Sequential window processing
            window_results = []
            for window, w_seed in zip(windows, window_seeds):
                logger.info(f"Processing {window}")
                window_result = self._process_window(
                    window=window,
                    strategies=strategies,
                    n_simulations=n_simulations,
                    manufacturer=manufacturer,
                    config=config,
                    window_seed=w_seed,
                )
                window_results.append(window_result)

        # Analyze results
        validation_result = ValidationResult(window_results=window_results)
        self._analyze_results(validation_result, strategies)

        return validation_result

    def _process_window(
        self,
        window: ValidationWindow,
        strategies: List[InsuranceStrategy],
        n_simulations: int,
        manufacturer: WidgetManufacturer,
        config: Config,
        window_seed: Optional[np.random.SeedSequence] = None,
    ) -> WindowResult:
        """Process a single validation window.

        Args:
            window: Validation window to process
            strategies: Strategies to test
            n_simulations: Number of simulations
            manufacturer: Manufacturer instance
            config: Configuration object
            window_seed: SeedSequence for this window, spawned from the
                validator's base SeedSequence. Guarantees statistically
                independent streams across windows.

        Returns:
            WindowResult for the window.
        """
        import time

        start_time = time.time()

        window_result = WindowResult(window=window)

        # Derive independent train/test seeds via SeedSequence.spawn()
        if window_seed is None:
            window_seed = np.random.SeedSequence(window.window_id)
        train_ss, test_ss = window_seed.spawn(2)

        train_config = SimulationConfig(
            n_simulations=n_simulations,
            n_years=window.get_train_years(),
            seed=int(train_ss.generate_state(1)[0]),
        )

        test_config = SimulationConfig(
            n_simulations=n_simulations,
            n_years=window.get_test_years(),
            seed=int(test_ss.generate_state(1)[0]),
        )

        # Test each strategy
        for strategy in strategies:
            logger.info(f"  Testing strategy: {strategy.name}")

            # Reset strategy for new window
            strategy.reset()

            # Run training period
            train_result = self.backtester.test_strategy(
                strategy=strategy,
                manufacturer=manufacturer,
                config=train_config,
                use_cache=False,  # Don't cache during validation
            )

            # For OptimizedStaticStrategy, capture optimization params
            if hasattr(strategy, "optimized_params") and strategy.optimized_params:
                window_result.optimization_params[strategy.name] = strategy.optimized_params.copy()

            # Run testing period
            test_result = self.backtester.test_strategy(
                strategy=strategy, manufacturer=manufacturer, config=test_config, use_cache=False
            )

            # Create performance record
            performance = StrategyPerformance(
                strategy_name=strategy.name,
                in_sample_metrics=train_result.metrics,
                out_sample_metrics=test_result.metrics,
            )
            performance.calculate_degradation()

            window_result.strategy_performances[strategy.name] = performance

        window_result.execution_time = time.time() - start_time
        logger.info(f"  Window processed in {window_result.execution_time:.2f} seconds")

        return window_result

    def _process_windows_parallel(
        self,
        windows: List[ValidationWindow],
        strategies: List[InsuranceStrategy],
        n_simulations: int,
        manufacturer: WidgetManufacturer,
        config: Config,
        n_workers: int,
        window_seeds: Optional[List[np.random.SeedSequence]] = None,
    ) -> List[WindowResult]:
        """Process validation windows in parallel using ProcessPoolExecutor.

        Args:
            windows: List of validation windows
            strategies: Strategies to test
            n_simulations: Number of simulations
            manufacturer: Manufacturer instance
            config: Configuration object
            n_workers: Number of worker processes
            window_seeds: Per-window SeedSequence objects (picklable).

        Returns:
            List of WindowResult in original window order.
        """
        results_by_idx: Dict[int, WindowResult] = {}

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {}
            for idx, window in enumerate(windows):
                w_seed = window_seeds[idx] if window_seeds else None
                future = executor.submit(
                    _process_window_worker,
                    window=window,
                    strategies=strategies,
                    n_simulations=n_simulations,
                    manufacturer=manufacturer,
                    config=config,
                    simulation_engine=self.simulation_engine,
                    window_seed=w_seed,
                )
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                result = future.result()
                results_by_idx[idx] = result
                logger.info(
                    f"Completed window {idx + 1}/{len(windows)} " f"in {result.execution_time:.2f}s"
                )

        return [results_by_idx[i] for i in range(len(windows))]

    def _analyze_results(  # pylint: disable=too-many-branches
        self, validation_result: ValidationResult, strategies: List[InsuranceStrategy]
    ):
        """Analyze validation results.

        Args:
            validation_result: Results to analyze
            strategies: List of strategies tested
        """
        # Collect metrics across windows
        strategy_metrics: Dict[str, List[ValidationMetrics]] = {s.name: [] for s in strategies}

        # Also collect from window results to handle all strategies
        for window_result in validation_result.window_results:
            for strategy_name in window_result.strategy_performances:
                if strategy_name not in strategy_metrics:
                    strategy_metrics[strategy_name] = []

        for window_result in validation_result.window_results:
            for strategy_name, performance in window_result.strategy_performances.items():
                if performance.out_sample_metrics:
                    if strategy_name not in strategy_metrics:
                        strategy_metrics[strategy_name] = []
                    strategy_metrics[strategy_name].append(performance.out_sample_metrics)

        # Calculate overfitting scores
        for strategy_name in strategy_metrics:
            performances = []
            for window_result in validation_result.window_results:
                if strategy_name in window_result.strategy_performances:
                    perf = window_result.strategy_performances[strategy_name]
                    if perf.overfitting_score is not None:
                        performances.append(perf.overfitting_score)

            if performances:
                validation_result.overfitting_analysis[strategy_name] = float(np.mean(performances))

        # Calculate consistency scores (coefficient of variation)
        for strategy_name, metrics_list in strategy_metrics.items():
            if metrics_list:
                roes = [m.roe for m in metrics_list]
                if len(roes) > 1 and np.mean(roes) != 0:
                    consistency = 1 - (np.std(roes) / abs(np.mean(roes)))
                    validation_result.consistency_scores[strategy_name] = max(
                        0.0, float(consistency)
                    )
                else:
                    validation_result.consistency_scores[strategy_name] = 1.0

        # Create strategy rankings
        ranking_data = []
        for strategy_name, metrics_list in strategy_metrics.items():
            if metrics_list:
                avg_metrics = self._average_metrics(metrics_list)
                ranking_data.append(
                    {
                        "strategy": strategy_name,
                        "avg_roe": avg_metrics.roe,
                        "avg_ruin_prob": avg_metrics.ruin_probability,
                        "avg_sharpe": avg_metrics.sharpe_ratio,
                        "avg_growth": avg_metrics.growth_rate,
                        "overfitting_score": validation_result.overfitting_analysis.get(
                            strategy_name, 0
                        ),
                        "consistency_score": validation_result.consistency_scores.get(
                            strategy_name, 0
                        ),
                        "composite_score": self._calculate_composite_score(
                            avg_metrics,
                            validation_result.overfitting_analysis.get(strategy_name, 0),
                            validation_result.consistency_scores.get(strategy_name, 0),
                        ),
                    }
                )

        if ranking_data:
            validation_result.strategy_rankings = pd.DataFrame(ranking_data)
            validation_result.strategy_rankings = validation_result.strategy_rankings.sort_values(
                "composite_score", ascending=False
            )

            # Select best strategy
            if not validation_result.strategy_rankings.empty:
                validation_result.best_strategy = validation_result.strategy_rankings.iloc[0][
                    "strategy"
                ]

    def _average_metrics(self, metrics_list: List[ValidationMetrics]) -> ValidationMetrics:
        """Calculate average metrics.

        Args:
            metrics_list: List of metrics to average

        Returns:
            Averaged ValidationMetrics.
        """
        if not metrics_list:
            return ValidationMetrics(0, 0, 0, 0)

        avg_metrics = ValidationMetrics(
            roe=float(np.mean([m.roe for m in metrics_list])),
            ruin_probability=float(np.mean([m.ruin_probability for m in metrics_list])),
            growth_rate=float(np.mean([m.growth_rate for m in metrics_list])),
            volatility=float(np.mean([m.volatility for m in metrics_list])),
            sharpe_ratio=float(np.mean([m.sharpe_ratio for m in metrics_list])),
            max_drawdown=float(np.mean([m.max_drawdown for m in metrics_list])),
            var_95=float(np.mean([m.var_95 for m in metrics_list])),
            cvar_95=float(np.mean([m.cvar_95 for m in metrics_list])),
            win_rate=float(np.mean([m.win_rate for m in metrics_list])),
            profit_factor=float(np.mean([m.profit_factor for m in metrics_list])),
            stability=float(np.mean([m.stability for m in metrics_list])),
        )

        return avg_metrics

    def _calculate_composite_score(
        self, metrics: ValidationMetrics, overfitting_score: float, consistency_score: float
    ) -> float:
        """Calculate composite score for ranking.

        Args:
            metrics: Performance metrics
            overfitting_score: Overfitting score (lower is better)
            consistency_score: Consistency score (higher is better)

        Returns:
            Composite score for ranking.
        """
        # Weights for different components
        weights = {"roe": 0.3, "sharpe": 0.2, "consistency": 0.2, "overfitting": 0.2, "ruin": 0.1}

        # Calculate components (normalize to 0-1 scale)
        components = {
            "roe": min(max(metrics.roe / 0.2, 0), 1),  # Normalize ROE (0.2 = 20% is excellent)
            "sharpe": min(max(metrics.sharpe_ratio / 2, 0), 1),  # Normalize Sharpe (2 is excellent)
            "consistency": consistency_score,
            "overfitting": 1 - min(overfitting_score, 1),  # Invert so lower is better
            "ruin": 1 - min(metrics.ruin_probability * 10, 1),  # Invert and scale
        }

        # Calculate weighted score
        score = sum(weight * components[component] for component, weight in weights.items())

        return score

    def generate_report(
        self,
        validation_result: ValidationResult,
        output_dir: str = "./reports",
        include_visualizations: bool = True,
    ) -> Dict[str, Any]:
        """Generate validation reports.

        Args:
            validation_result: Validation results to report
            output_dir: Directory for output files
            include_visualizations: Whether to include plots

        Returns:
            Dictionary of generated file paths.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_files: Dict[str, Any] = {}

        # Generate markdown summary
        md_path = output_path / f"validation_summary_{timestamp}.md"
        self._generate_markdown_report(validation_result, md_path)
        report_files["markdown"] = md_path

        # Generate HTML report
        html_path = output_path / f"validation_report_{timestamp}.html"
        self._generate_html_report(validation_result, html_path, include_visualizations)
        report_files["html"] = html_path

        # Generate visualizations if requested
        if include_visualizations:
            viz_dir = output_path / f"visualizations_{timestamp}"
            viz_dir.mkdir(exist_ok=True)
            viz_files = self._generate_visualizations(validation_result, viz_dir)
            report_files["visualizations"] = viz_files

        # Save raw results as JSON
        json_path = output_path / f"validation_results_{timestamp}.json"
        self._save_results_json(validation_result, json_path)
        report_files["json"] = json_path

        logger.info(f"Reports generated in {output_path}")
        return report_files

    def _generate_markdown_report(self, validation_result: ValidationResult, output_path: Path):
        """Generate markdown summary report.

        Args:
            validation_result: Results to report
            output_path: Output file path
        """
        lines = []
        lines.append("# Walk-Forward Validation Report")
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("\n## Configuration")
        lines.append(f"- Window Size: {self.window_size} years")
        lines.append(f"- Step Size: {self.step_size} years")
        lines.append(f"- Test Ratio: {self.test_ratio:.1%}")
        lines.append(f"- Total Windows: {len(validation_result.window_results)}")

        lines.append("\n## Strategy Rankings")
        if not validation_result.strategy_rankings.empty:
            lines.append("\n" + validation_result.strategy_rankings.to_markdown(index=False))

        lines.append(f"\n## Best Strategy: **{validation_result.best_strategy}**")

        lines.append("\n## Overfitting Analysis")
        for strategy, score in validation_result.overfitting_analysis.items():
            status = "✓ Low" if score < 0.2 else "⚠ Moderate" if score < 0.4 else "✗ High"
            lines.append(f"- {strategy}: {score:.3f} ({status})")

        lines.append("\n## Consistency Scores")
        for strategy, score in validation_result.consistency_scores.items():
            status = "✓ High" if score > 0.8 else "⚠ Moderate" if score > 0.6 else "✗ Low"
            lines.append(f"- {strategy}: {score:.3f} ({status})")

        lines.append("\n## Performance by Window")
        for window_result in validation_result.window_results:
            lines.append(f"\n### {window_result.window}")
            for strategy_name, performance in window_result.strategy_performances.items():
                lines.append(f"\n**{strategy_name}:**")
                if performance.in_sample_metrics:
                    lines.append(f"- In-Sample ROE: {performance.in_sample_metrics.roe:.2%}")
                if performance.out_sample_metrics:
                    lines.append(f"- Out-Sample ROE: {performance.out_sample_metrics.roe:.2%}")
                if performance.degradation:
                    deg = performance.degradation.get("roe_diff", 0)
                    lines.append(f"- Degradation: {deg:.2%}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _generate_html_report(
        self, validation_result: ValidationResult, output_path: Path, include_visualizations: bool
    ):
        """Generate HTML report with visualizations.

        Args:
            validation_result: Results to report
            output_path: Output file path
            include_visualizations: Whether to embed visualizations
        """
        html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Walk-Forward Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        h2 { color: #666; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .best { background-color: #d4edda; }
        .warning { background-color: #fff3cd; }
        .danger { background-color: #f8d7da; }
        .metric { font-family: monospace; }
        .chart { margin: 20px 0; text-align: center; }
    </style>
</head>
<body>
    <h1>Walk-Forward Validation Report</h1>
    <p>Generated: {{ timestamp }}</p>

    <h2>Configuration</h2>
    <ul>
        <li>Window Size: {{ window_size }} years</li>
        <li>Step Size: {{ step_size }} years</li>
        <li>Test Ratio: {{ test_ratio }}%</li>
        <li>Total Windows: {{ n_windows }}</li>
    </ul>

    <h2>Strategy Rankings</h2>
    {{ rankings_table }}

    <h2>Best Strategy</h2>
    <p style="font-size: 1.2em; font-weight: bold;">{{ best_strategy }}</p>

    <h2>Detailed Results</h2>
    {{ detailed_results }}

    {% if include_viz %}
    <h2>Visualizations</h2>
    <div class="chart">
        <!-- Visualization placeholders -->
        <p>See accompanying visualization files for detailed charts</p>
    </div>
    {% endif %}
</body>
</html>"""

        # Prepare template data
        template_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "window_size": self.window_size,
            "step_size": self.step_size,
            "test_ratio": f"{self.test_ratio * 100:.1f}",
            "n_windows": len(validation_result.window_results),
            "best_strategy": validation_result.best_strategy or "N/A",
            "include_viz": include_visualizations,
        }

        # Generate rankings table
        if not validation_result.strategy_rankings.empty:
            template_data["rankings_table"] = validation_result.strategy_rankings.to_html(
                index=False, classes="ranking-table", float_format=lambda x: f"{x:.4f}"
            )
        else:
            template_data["rankings_table"] = "<p>No rankings available</p>"

        # Generate detailed results
        detailed_html = []
        for window_result in validation_result.window_results:
            detailed_html.append(f"<h3>{window_result.window}</h3>")
            detailed_html.append("<table>")
            detailed_html.append(
                "<tr><th>Strategy</th><th>In-Sample ROE</th><th>Out-Sample ROE</th><th>Degradation</th></tr>"
            )

            for strategy_name, perf in window_result.strategy_performances.items():
                in_roe = f"{perf.in_sample_metrics.roe:.2%}" if perf.in_sample_metrics else "N/A"
                out_roe = f"{perf.out_sample_metrics.roe:.2%}" if perf.out_sample_metrics else "N/A"
                deg = f"{perf.degradation.get('roe_diff', 0):.2%}" if perf.degradation else "N/A"

                row_class = ""
                if perf.overfitting_score > 0.4:
                    row_class = "danger"
                elif perf.overfitting_score > 0.2:
                    row_class = "warning"

                detailed_html.append(f'<tr class="{row_class}">')
                detailed_html.append(f"<td>{strategy_name}</td>")
                detailed_html.append(f'<td class="metric">{in_roe}</td>')
                detailed_html.append(f'<td class="metric">{out_roe}</td>')
                detailed_html.append(f'<td class="metric">{deg}</td>')
                detailed_html.append("</tr>")

            detailed_html.append("</table>")

        template_data["detailed_results"] = "\n".join(detailed_html)

        # Render and save
        template = Template(html_template)
        html_content = template.render(**template_data)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def _generate_visualizations(  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
        self, validation_result: ValidationResult, output_dir: Path
    ) -> List[Path]:
        """Generate visualization plots.

        Args:
            validation_result: Results to visualize
            output_dir: Directory for plots

        Returns:
            List of generated plot files.
        """
        plot_files = []
        sns.set_style("whitegrid")

        # Generate performance plot
        perf_plot = self._plot_performance_across_windows(validation_result, output_dir)
        if perf_plot:
            plot_files.append(perf_plot)

        # Generate overfitting analysis plot
        overfit_plot = self._plot_overfitting_analysis(validation_result, output_dir)
        if overfit_plot:
            plot_files.append(overfit_plot)

        # Generate ranking heatmap
        heatmap_plot = self._plot_strategy_ranking_heatmap(validation_result, output_dir)
        if heatmap_plot:
            plot_files.append(heatmap_plot)

        return plot_files

    def _plot_performance_across_windows(
        self, validation_result: ValidationResult, output_dir: Path
    ) -> Optional[Path]:
        """Plot strategy performance across windows.

        Args:
            validation_result: Results to visualize
            output_dir: Directory for plot

        Returns:
            Path to generated plot or None.
        """
        _fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Collect data for plotting
        strategies = list(validation_result.window_results[0].strategy_performances.keys())
        windows = list(range(len(validation_result.window_results)))

        metrics_data: Dict[str, Dict[str, List[float]]] = {
            "ROE": {s: [] for s in strategies},
            "Sharpe": {s: [] for s in strategies},
            "Ruin Prob": {s: [] for s in strategies},
            "Growth": {s: [] for s in strategies},
        }

        for window_result in validation_result.window_results:
            for strategy in strategies:
                if strategy in window_result.strategy_performances:
                    perf = window_result.strategy_performances[strategy]
                    if perf.out_sample_metrics:
                        metrics_data["ROE"][strategy].append(perf.out_sample_metrics.roe)
                        metrics_data["Sharpe"][strategy].append(
                            perf.out_sample_metrics.sharpe_ratio
                        )
                        metrics_data["Ruin Prob"][strategy].append(
                            perf.out_sample_metrics.ruin_probability
                        )
                        metrics_data["Growth"][strategy].append(perf.out_sample_metrics.growth_rate)

        # Plot each metric
        for ax, (metric_name, metric_values) in zip(axes.flat, metrics_data.items()):
            for strategy in strategies:
                if metric_values[strategy]:
                    ax.plot(
                        windows[: len(metric_values[strategy])],
                        metric_values[strategy],
                        marker="o",
                        label=strategy,
                    )
            ax.set_title(f"{metric_name} Across Windows")
            ax.set_xlabel("Window")
            ax.set_ylabel(metric_name)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "performance_across_windows.png"
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()
        return plot_path

    def _plot_overfitting_analysis(
        self, validation_result: ValidationResult, output_dir: Path
    ) -> Optional[Path]:
        """Plot overfitting analysis bar chart.

        Args:
            validation_result: Results to visualize
            output_dir: Directory for plot

        Returns:
            Path to generated plot or None.
        """
        if not validation_result.overfitting_analysis:
            return None

        _fig, ax = plt.subplots(figsize=(10, 6))

        strategies = list(validation_result.overfitting_analysis.keys())
        scores = list(validation_result.overfitting_analysis.values())

        bars = ax.bar(strategies, scores)

        # Color bars based on severity
        for bar_element, score in zip(bars, scores):  # Renamed 'bar' to 'bar_element'
            if score < 0.2:
                bar_element.set_color("green")
            elif score < 0.4:
                bar_element.set_color("orange")
            else:
                bar_element.set_color("red")

        ax.set_title("Overfitting Scores by Strategy")
        ax.set_xlabel("Strategy")
        ax.set_ylabel("Overfitting Score")
        ax.axhline(y=0.2, color="orange", linestyle="--", alpha=0.5, label="Moderate threshold")
        ax.axhline(y=0.4, color="red", linestyle="--", alpha=0.5, label="High threshold")
        ax.legend()

        plt.tight_layout()
        plot_path = output_dir / "overfitting_analysis.png"
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()
        return plot_path

    def _plot_strategy_ranking_heatmap(
        self, validation_result: ValidationResult, output_dir: Path
    ) -> Optional[Path]:
        """Plot strategy ranking heatmap.

        Args:
            validation_result: Results to visualize
            output_dir: Directory for plot

        Returns:
            Path to generated plot or None.
        """
        if validation_result.strategy_rankings.empty:
            return None

        _fig, ax = plt.subplots(figsize=(10, 6))

        # Prepare data for heatmap - use available columns
        possible_cols = ["avg_roe", "avg_sharpe", "consistency_score", "composite_score"]
        available_cols = [
            col for col in possible_cols if col in validation_result.strategy_rankings.columns
        ]

        if available_cols:
            heatmap_data = validation_result.strategy_rankings.set_index("strategy")[
                available_cols
            ].T
        else:
            # Skip heatmap if no ranking columns available
            heatmap_data = None

        if heatmap_data is not None and not heatmap_data.empty:
            # Normalize for better visualization
            # Convert to numpy array to ensure reshape works
            row_mins = np.asarray(heatmap_data.min(axis=1).values).reshape(-1, 1)
            row_maxs = np.asarray(heatmap_data.max(axis=1).values).reshape(-1, 1)
            row_ranges = row_maxs - row_mins

            # Avoid division by zero for rows with no variation
            with np.errstate(divide="ignore", invalid="ignore"):
                heatmap_norm = np.where(
                    row_ranges != 0,
                    (heatmap_data - row_mins) / row_ranges,
                    0.5,  # Use 0.5 for rows with no variation (centered value)
                )

            # Handle any remaining NaN values
            heatmap_norm = np.nan_to_num(heatmap_norm, nan=0.5)

            sns.heatmap(
                heatmap_norm,
                annot=True,
                fmt=".2f",
                cmap="RdYlGn",
                ax=ax,
                cbar_kws={"label": "Normalized Score"},
            )
            ax.set_title("Strategy Performance Heatmap (Normalized)")
            ax.set_xlabel("Strategy")
            ax.set_ylabel("Metric")

            plt.tight_layout()
            plot_path = output_dir / "strategy_ranking_heatmap.png"
            plt.savefig(plot_path, dpi=100, bbox_inches="tight")
            plt.close()
            return plot_path

        return None

    def _save_results_json(self, validation_result: ValidationResult, output_path: Path):
        """Save results as JSON.

        Args:
            validation_result: Results to save
            output_path: Output file path
        """
        # Convert to serializable format
        results_dict: Dict[str, Any] = {
            "metadata": validation_result.metadata,
            "best_strategy": validation_result.best_strategy,
            "overfitting_analysis": validation_result.overfitting_analysis,
            "consistency_scores": validation_result.consistency_scores,
            "strategy_rankings": (
                validation_result.strategy_rankings.to_dict()
                if not validation_result.strategy_rankings.empty
                else {}
            ),
            "windows": [],
        }

        for window_result in validation_result.window_results:
            window_dict: Dict[str, Any] = {
                "window_id": window_result.window.window_id,
                "train_start": window_result.window.train_start,
                "train_end": window_result.window.train_end,
                "test_start": window_result.window.test_start,
                "test_end": window_result.window.test_end,
                "execution_time": window_result.execution_time,
                "optimization_params": window_result.optimization_params,
                "performances": {},
            }

            performances_dict: Dict[str, Any] = {}
            for strategy_name, perf in window_result.strategy_performances.items():
                performances_dict[strategy_name] = {
                    "in_sample": (
                        perf.in_sample_metrics.to_dict() if perf.in_sample_metrics else None
                    ),
                    "out_sample": (
                        perf.out_sample_metrics.to_dict() if perf.out_sample_metrics else None
                    ),
                    "overfitting_score": perf.overfitting_score,
                }

            window_dict["performances"] = performances_dict

            windows_list = results_dict.get("windows")
            if isinstance(windows_list, list):
                windows_list.append(window_dict)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2, default=str)
