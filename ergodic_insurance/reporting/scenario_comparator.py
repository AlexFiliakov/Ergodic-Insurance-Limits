"""Scenario comparison framework for analyzing multiple simulation results.

This module provides comprehensive tools for comparing different scenarios,
highlighting parameter differences, and performing statistical comparisons.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..visualization.annotations import add_value_labels
from ..visualization.core import WSJ_COLORS, set_wsj_style


@dataclass
class ScenarioComparison:
    """Container for scenario comparison results.

    Attributes:
        scenarios: List of scenario names
        metrics: Dictionary of metric values by scenario
        parameters: Dictionary of parameter values by scenario
        statistics: Statistical comparison results
        diffs: Parameter differences from baseline
        rankings: Scenario rankings by metric
    """

    scenarios: List[str]
    metrics: Dict[str, Dict[str, float]]
    parameters: Dict[str, Dict[str, Any]]
    statistics: Dict[str, Any] = field(default_factory=dict)
    diffs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    rankings: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)

    def get_metric_df(self, metric: str) -> pd.DataFrame:
        """Get metric values as DataFrame.

        Args:
            metric: Metric name

        Returns:
            DataFrame with scenarios and metric values
        """
        if metric not in self.metrics:
            raise ValueError(f"Metric {metric} not found")

        data = []
        for scenario in self.scenarios:
            if scenario in self.metrics[metric]:
                data.append({"scenario": scenario, metric: self.metrics[metric][scenario]})

        return pd.DataFrame(data)

    def get_top_performers(
        self, metric: str, n: int = 3, ascending: bool = False
    ) -> List[Tuple[str, float]]:
        """Get top performing scenarios for a metric.

        Args:
            metric: Metric name
            n: Number of top performers
            ascending: If True, lower values are better

        Returns:
            List of (scenario, value) tuples
        """
        if metric not in self.rankings:
            self._compute_rankings()

        rankings = self.rankings[metric]
        if ascending:
            rankings = rankings[::-1]

        return rankings[:n]

    def _compute_rankings(self):
        """Compute scenario rankings for each metric."""
        for metric, values in self.metrics.items():
            sorted_scenarios = sorted(values.items(), key=lambda x: x[1], reverse=True)
            self.rankings[metric] = sorted_scenarios


class ScenarioComparator:
    """Framework for comparing multiple simulation scenarios."""

    def __init__(self):
        """Initialize scenario comparator."""
        self.baseline_scenario: Optional[str] = None
        self.comparison_data: Optional[ScenarioComparison] = None

    def compare_scenarios(
        self,
        results: Dict[str, Any],
        baseline: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        parameters: Optional[List[str]] = None,
    ) -> ScenarioComparison:
        """Compare multiple scenarios with comprehensive analysis.

        Args:
            results: Dictionary of scenario results
            baseline: Baseline scenario name for comparison
            metrics: Metrics to compare (default: all numeric)
            parameters: Parameters to track (default: all)

        Returns:
            ScenarioComparison object with analysis results

        Examples:
            >>> comparator = ScenarioComparator()
            >>> results = {'base': {...}, 'optimized': {...}}
            >>> comparison = comparator.compare_scenarios(results)
        """
        scenarios = list(results.keys())

        if baseline:
            self.baseline_scenario = baseline
        elif scenarios:
            self.baseline_scenario = scenarios[0]

        # Extract metrics
        metric_data = self._extract_metrics(results, metrics)

        # Extract parameters
        param_data = self._extract_parameters(results, parameters)

        # Create comparison object
        comparison = ScenarioComparison(
            scenarios=scenarios, metrics=metric_data, parameters=param_data
        )

        # Compute parameter diffs
        if self.baseline_scenario:
            comparison.diffs = self._compute_diffs(param_data, self.baseline_scenario)

        # Perform statistical tests
        comparison.statistics = self._perform_statistical_tests(metric_data)

        self.comparison_data = comparison
        return comparison

    def _extract_metrics(
        self, results: Dict[str, Any], metrics: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Extract metric values from results.

        Args:
            results: Scenario results
            metrics: Specific metrics to extract

        Returns:
            Dictionary of metrics by scenario
        """
        metric_data: Dict[str, Dict[str, float]] = {}

        for scenario, data in results.items():
            # Handle different result formats
            if hasattr(data, "summary_statistics"):
                stats = data.summary_statistics
            elif isinstance(data, dict) and "summary_statistics" in data:
                stats = data["summary_statistics"]
            elif isinstance(data, pd.DataFrame):
                stats = data.to_dict("records")[0] if not data.empty else {}
            else:
                stats = data if isinstance(data, dict) else {}

            # Extract numeric metrics
            for key, value in stats.items():
                if metrics and key not in metrics:
                    continue

                if isinstance(value, (int, float, np.number)):
                    if key not in metric_data:
                        metric_data[key] = {}
                    metric_data[key][scenario] = float(value)

        return metric_data

    def _extract_parameters(
        self, results: Dict[str, Any], parameters: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Extract parameter values from results.

        Args:
            results: Scenario results
            parameters: Specific parameters to extract

        Returns:
            Dictionary of parameters by scenario
        """
        param_data = {}

        for scenario, data in results.items():
            # Handle different result formats
            if hasattr(data, "config"):
                config = data.config
            elif isinstance(data, dict) and "config" in data:
                config = data["config"]
            elif hasattr(data, "parameters"):
                config = data.parameters
            elif isinstance(data, dict) and "parameters" in data:
                config = data["parameters"]
            else:
                config = {}

            # Extract parameters
            if config:
                param_data[scenario] = self._flatten_config(config, parameters)

        return param_data

    def _flatten_config(self, config: Any, keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Flatten nested configuration to flat dictionary.

        Args:
            config: Configuration object or dict
            keys: Specific keys to extract

        Returns:
            Flat dictionary of parameters
        """
        flat = {}

        def flatten_recursive(obj, prefix=""):
            if hasattr(obj, "__dict__"):
                obj = obj.__dict__

            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_key = f"{prefix}.{k}" if prefix else k

                    if keys and new_key not in keys:
                        continue

                    if isinstance(v, (dict, object)) and not isinstance(
                        v, (str, int, float, bool, list)
                    ):
                        flatten_recursive(v, new_key)
                    else:
                        flat[new_key] = v
            elif hasattr(obj, "__dict__"):
                flatten_recursive(obj.__dict__, prefix)

        flatten_recursive(config)
        return flat

    def _compute_diffs(
        self, param_data: Dict[str, Dict[str, Any]], baseline: str
    ) -> Dict[str, Dict[str, Any]]:
        """Compute parameter differences from baseline.

        Args:
            param_data: Parameter data by scenario
            baseline: Baseline scenario name

        Returns:
            Dictionary of parameter differences
        """
        diffs: Dict[str, Dict[str, Any]] = {}

        if baseline not in param_data:
            return diffs

        baseline_params = param_data[baseline]

        for scenario, params in param_data.items():
            if scenario == baseline:
                continue

            scenario_diffs = {}
            for key, value in params.items():
                if key in baseline_params:
                    baseline_val = baseline_params[key]

                    if isinstance(value, (int, float)) and isinstance(baseline_val, (int, float)):
                        diff = value - baseline_val
                        pct_diff = (diff / baseline_val * 100) if baseline_val != 0 else 0
                        scenario_diffs[key] = {
                            "absolute": diff,
                            "percentage": pct_diff,
                            "baseline": baseline_val,
                            "value": value,
                        }
                    elif value != baseline_val:
                        scenario_diffs[key] = {
                            "changed": True,
                            "baseline": baseline_val,
                            "value": value,
                        }

            diffs[scenario] = scenario_diffs

        return diffs

    def _perform_statistical_tests(
        self, metric_data: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Perform statistical tests between scenarios.

        Args:
            metric_data: Metric values by scenario

        Returns:
            Dictionary of statistical test results
        """
        statistics = {}

        for metric, values in metric_data.items():
            if len(values) < 2:
                continue

            # Prepare data for tests
            scenario_values = list(values.values())

            # Basic statistics
            statistics[metric] = {
                "mean": np.mean(scenario_values),
                "std": np.std(scenario_values),
                "min": np.min(scenario_values),
                "max": np.max(scenario_values),
                "range": np.max(scenario_values) - np.min(scenario_values),
                "cv": np.std(scenario_values) / np.mean(scenario_values)
                if np.mean(scenario_values) != 0
                else 0,
            }

            # ANOVA if more than 2 scenarios
            if len(values) > 2:
                # Simplified ANOVA using scenario means
                # Note: This is a simplified version for demonstration
                grand_mean = np.mean(scenario_values)
                scenario_values_array = np.array(scenario_values)
                ss_between = len(scenario_values) * np.sum(
                    (scenario_values_array - grand_mean) ** 2
                )
                ss_total = np.sum((scenario_values_array - grand_mean) ** 2)

                statistics[metric]["anova"] = {
                    "significant_difference": ss_between > 0.5 * ss_total
                }

        return statistics

    def create_comparison_grid(
        self,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (16, 10),
        show_diff: bool = True,
    ) -> Figure:
        """Create comprehensive comparison visualization grid.

        Args:
            metrics: Metrics to display (default: top 6)
            figsize: Figure size
            show_diff: Show difference from baseline

        Returns:
            Matplotlib figure with comparison grid
        """
        if not self.comparison_data:
            raise ValueError("No comparison data available. Run compare_scenarios first.")

        set_wsj_style()

        # Select metrics
        if metrics is None:
            available_metrics = list(self.comparison_data.metrics.keys())
            metrics = available_metrics[:6]

        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        # Create figure with GridSpec
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.25)

        for idx, metric in enumerate(metrics):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])

            self._plot_metric_comparison(ax, metric, show_diff)

        # Add title
        fig.suptitle("Scenario Comparison Analysis", fontsize=16, fontweight="bold", y=0.98)

        return fig

    def _plot_metric_comparison(
        self, ax: plt.Axes, metric: str, show_diff: bool = True
    ):  # pylint: disable=too-many-locals
        """Plot single metric comparison.

        Args:
            ax: Matplotlib axes
            metric: Metric name
            show_diff: Show difference from baseline
        """
        if self.comparison_data is None or metric not in self.comparison_data.metrics:
            return

        values = self.comparison_data.metrics[metric]
        scenarios = list(values.keys())
        metric_values = [values[s] for s in scenarios]

        # Determine if lower is better based on metric name
        lower_is_better = any(x in metric.lower() for x in ["risk", "probability", "loss", "cost"])

        # Create bar plot
        bars = ax.bar(range(len(scenarios)), metric_values, alpha=0.8)

        # Color code bars
        if lower_is_better:
            best_idx = np.argmin(metric_values)
            worst_idx = np.argmax(metric_values)
        else:
            best_idx = np.argmax(metric_values)
            worst_idx = np.argmin(metric_values)

        for i, bar_patch in enumerate(bars):
            if i == best_idx:
                bar_patch.set_color(WSJ_COLORS["green"])
                bar_patch.set_alpha(1.0)
            elif i == worst_idx:
                bar_patch.set_color(WSJ_COLORS["red"])
                bar_patch.set_alpha(0.7)
            else:
                bar_patch.set_color(WSJ_COLORS["blue"])

        # Add value labels
        add_value_labels(ax, bars)

        # Show difference from baseline
        if show_diff and self.baseline_scenario in scenarios:
            baseline_idx = scenarios.index(self.baseline_scenario)
            baseline_value = metric_values[baseline_idx]

            for i, (scenario, value) in enumerate(zip(scenarios, metric_values)):
                if scenario != self.baseline_scenario:
                    diff_pct = (
                        ((value - baseline_value) / baseline_value * 100)
                        if baseline_value != 0
                        else 0
                    )

                    # Add diff annotation
                    y_pos = value + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                    color = (
                        WSJ_COLORS["green"]
                        if (diff_pct > 0) != lower_is_better
                        else WSJ_COLORS["red"]
                    )
                    ax.text(
                        i,
                        y_pos,
                        f"{diff_pct:+.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color=color,
                        fontweight="bold",
                    )

        # Format axes
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, rotation=45, ha="right")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3, axis="y")

        # Add significance indicator if available
        if self.comparison_data and metric in self.comparison_data.statistics:
            stats_data = self.comparison_data.statistics[metric]
            if "anova" in stats_data and stats_data["anova"]["significant_difference"]:
                ax.text(
                    0.95,
                    0.95,
                    "*",
                    transform=ax.transAxes,
                    fontsize=16,
                    color=WSJ_COLORS["red"],
                    fontweight="bold",
                    ha="right",
                    va="top",
                )

    def create_parameter_diff_table(self, scenario: str, threshold: float = 5.0) -> pd.DataFrame:
        """Create table showing parameter differences from baseline.

        Args:
            scenario: Scenario to compare
            threshold: Minimum percentage change to highlight

        Returns:
            DataFrame with parameter differences
        """
        if not self.comparison_data or scenario not in self.comparison_data.diffs:
            return pd.DataFrame()

        diffs = self.comparison_data.diffs[scenario]

        rows = []
        for param, diff_data in diffs.items():
            if "percentage" in diff_data:
                if abs(diff_data["percentage"]) >= threshold:
                    rows.append(
                        {
                            "Parameter": param,
                            "Baseline": diff_data["baseline"],
                            "Scenario": diff_data["value"],
                            "Change": diff_data["absolute"],
                            "Change %": diff_data["percentage"],
                        }
                    )
            elif "changed" in diff_data and diff_data["changed"]:
                rows.append(
                    {
                        "Parameter": param,
                        "Baseline": str(diff_data["baseline"]),
                        "Scenario": str(diff_data["value"]),
                        "Change": "Modified",
                        "Change %": None,
                    }
                )

        df = pd.DataFrame(rows)

        # Sort by absolute percentage change
        if not df.empty and "Change %" in df.columns:
            df["abs_change"] = df["Change %"].abs()
            df = df.sort_values("abs_change", ascending=False)
            df = df.drop("abs_change", axis=1)

        return df

    def export_comparison_report(
        self, output_path: str, include_plots: bool = True
    ) -> Dict[str, Any]:
        """Export comprehensive comparison report.

        Args:
            output_path: Base path for output files
            include_plots: Whether to generate and save plots

        Returns:
            Dictionary with paths to generated files
        """
        if not self.comparison_data:
            raise ValueError("No comparison data available")

        outputs = {}

        # Export metrics summary
        metrics_df = pd.DataFrame(self.comparison_data.metrics)
        metrics_path = f"{output_path}_metrics.csv"
        metrics_df.to_csv(metrics_path)
        outputs["metrics"] = metrics_path

        # Export parameter differences
        if self.comparison_data.diffs:
            for scenario in self.comparison_data.scenarios:
                if scenario in self.comparison_data.diffs:
                    diff_df = self.create_parameter_diff_table(scenario)
                    if not diff_df.empty:
                        diff_path = f"{output_path}_diffs_{scenario}.csv"
                        diff_df.to_csv(diff_path, index=False)
                        outputs[f"diffs_{scenario}"] = diff_path

        # Generate and save plots
        if include_plots:
            fig = self.create_comparison_grid()
            plot_path = f"{output_path}_comparison.png"
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            outputs["plot"] = plot_path

        # Export statistical summary
        if self.comparison_data.statistics:
            stats_df = pd.DataFrame(self.comparison_data.statistics).T
            stats_path = f"{output_path}_statistics.csv"
            stats_df.to_csv(stats_path)
            outputs["statistics"] = stats_path

        return outputs
