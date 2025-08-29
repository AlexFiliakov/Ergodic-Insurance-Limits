"""Insight extraction and natural language generation for analysis results.

This module provides tools for automatically extracting key insights from
simulation results and generating natural language descriptions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class Insight:
    """Container for a single insight.

    Attributes:
        category: Type of insight (trend, outlier, correlation, etc.)
        importance: Importance score (0-100)
        title: Short title for the insight
        description: Detailed description
        data: Supporting data for the insight
        metrics: Related metrics
        confidence: Confidence level (0-1)
    """

    category: str
    importance: float
    title: str
    description: str
    data: Dict[str, Any] = field(default_factory=dict)
    metrics: List[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_bullet_point(self) -> str:
        """Convert insight to bullet point format.

        Returns:
            Formatted bullet point string
        """
        return f"• {self.title}: {self.description}"

    def to_executive_summary(self) -> str:
        """Convert to executive summary format.

        Returns:
            Executive-friendly description
        """
        # Simplify technical language
        summary = self.description
        replacements = {
            "ruin probability": "risk of failure",
            "growth rate": "return",
            "assets": "capital",
            "mean": "average",
            "variance": "volatility",
            "optimal": "best",
            "suboptimal": "poor",
        }

        for technical, simple in replacements.items():
            summary = summary.replace(technical, simple)

        return summary


class InsightExtractor:
    """Extract and generate insights from simulation results."""

    # Templates for natural language generation
    TEMPLATES = {
        "best_performer": "The {scenario} scenario achieved the best {metric} at {value}, outperforming the baseline by {improvement}%",
        "worst_performer": "The {scenario} scenario showed the weakest {metric} at {value}, underperforming by {decline}%",
        "trend_positive": "{metric} shows a strong positive trend, increasing by {change}% from {start} to {end}",
        "trend_negative": "{metric} displays concerning decline, decreasing by {change}% from {start} to {end}",
        "outlier_high": "{scenario} exhibits exceptionally high {metric} ({value}), {std_dev} standard deviations above mean",
        "outlier_low": "{scenario} shows unusually low {metric} ({value}), {std_dev} standard deviations below mean",
        "threshold_exceeded": "{metric} exceeded the critical threshold of {threshold} in {count} scenarios",
        "convergence": "{metric} converges to {value} after {period} periods, indicating stability",
        "volatility_high": "{scenario} demonstrates high volatility in {metric} with coefficient of variation {cv}",
        "correlation": "Strong {direction} correlation ({corr}) detected between {metric1} and {metric2}",
        "inflection": "Key inflection point identified at period {period} where {metric} shifts from {before} to {after}",
        "dominance": "{scenario} dominates on {count} out of {total} metrics, suggesting overall superiority",
    }

    def __init__(self):
        """Initialize insight extractor."""
        self.insights: List[Insight] = []
        self.data: Optional[Any] = None

    def extract_insights(
        self,
        data: Any,
        focus_metrics: Optional[List[str]] = None,
        threshold_importance: float = 50.0,
    ) -> List[Insight]:
        """Extract insights from simulation data.

        Args:
            data: Simulation results or comparison data
            focus_metrics: Metrics to focus on
            threshold_importance: Minimum importance threshold

        Returns:
            List of extracted insights

        Examples:
            >>> extractor = InsightExtractor()
            >>> insights = extractor.extract_insights(results)
            >>> for insight in insights[:3]:
            ...     print(insight.to_bullet_point())
        """
        self.data = data
        self.insights = []

        # Extract different types of insights
        self._extract_performance_insights(data, focus_metrics)
        self._extract_trend_insights(data, focus_metrics)
        self._extract_outlier_insights(data, focus_metrics)
        self._extract_threshold_insights(data, focus_metrics)
        self._extract_correlation_insights(data, focus_metrics)

        # Filter by importance
        filtered_insights = [i for i in self.insights if i.importance >= threshold_importance]

        # Sort by importance
        filtered_insights.sort(key=lambda x: x.importance, reverse=True)

        return filtered_insights

    def _extract_performance_insights(self, data: Any, focus_metrics: Optional[List[str]] = None):
        """Extract performance-related insights.

        Args:
            data: Analysis data
            focus_metrics: Metrics to focus on
        """
        # Handle different data formats
        if hasattr(data, "metrics"):
            metrics_data = data.metrics
        elif isinstance(data, dict) and "metrics" in data:
            metrics_data = data["metrics"]
        else:
            return

        for metric_name, values in metrics_data.items():
            if focus_metrics and metric_name not in focus_metrics:
                continue

            if not values:
                continue

            # Find best and worst performers
            sorted_scenarios = sorted(values.items(), key=lambda x: x[1])

            # Determine if lower is better
            lower_is_better = any(
                x in metric_name.lower() for x in ["risk", "probability", "loss", "cost", "ruin"]
            )

            if lower_is_better:
                best = sorted_scenarios[0]
                worst = sorted_scenarios[-1]
            else:
                best = sorted_scenarios[-1]
                worst = sorted_scenarios[0]

            # Calculate baseline comparison if available
            baseline_value = None
            if hasattr(data, "baseline_scenario") and data.baseline_scenario:
                baseline_value = values.get(data.baseline_scenario)

            # Best performer insight
            if baseline_value and best[0] != data.baseline_scenario:
                improvement = (
                    abs((best[1] - baseline_value) / baseline_value * 100)
                    if baseline_value != 0
                    else 0
                )

                insight = Insight(
                    category="performance",
                    importance=80
                    + min(20, improvement / 5),  # Higher improvement = higher importance
                    title=f"Top Performance in {metric_name.replace('_', ' ').title()}",
                    description=self.TEMPLATES["best_performer"].format(
                        scenario=best[0],
                        metric=metric_name.replace("_", " "),
                        value=self._format_value(best[1], metric_name),
                        improvement=f"{improvement:.1f}",
                    ),
                    data={"scenario": best[0], "value": best[1], "improvement": improvement},
                    metrics=[metric_name],
                    confidence=0.95,
                )
                self.insights.append(insight)

            # Worst performer insight (if significantly different)
            if len(sorted_scenarios) > 2:
                mean_value = np.mean(list(values.values()))
                if abs(worst[1] - mean_value) > 2 * np.std(list(values.values())):
                    decline = (
                        abs((worst[1] - mean_value) / mean_value * 100) if mean_value != 0 else 0
                    )

                    insight = Insight(
                        category="performance",
                        importance=70,
                        title=f"Underperformance in {metric_name.replace('_', ' ').title()}",
                        description=self.TEMPLATES["worst_performer"].format(
                            scenario=worst[0],
                            metric=metric_name.replace("_", " "),
                            value=self._format_value(worst[1], metric_name),
                            decline=f"{decline:.1f}",
                        ),
                        data={"scenario": worst[0], "value": worst[1], "decline": decline},
                        metrics=[metric_name],
                        confidence=0.90,
                    )
                    self.insights.append(insight)

    def _extract_trend_insights(self, data: Any, focus_metrics: Optional[List[str]] = None):
        """Extract trend-related insights.

        Args:
            data: Analysis data
            focus_metrics: Metrics to focus on
        """
        # Handle time series data if available
        if hasattr(data, "time_series") or (isinstance(data, dict) and "time_series" in data):
            time_series = data.time_series if hasattr(data, "time_series") else data["time_series"]

            for metric_name, series in time_series.items():
                if focus_metrics and metric_name not in focus_metrics:
                    continue

                if len(series) < 10:
                    continue

                # Calculate trend
                x = np.arange(len(series))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)

                # Significant trend detection
                if abs(r_value) > 0.7 and p_value < 0.05:
                    start_val = series[0]
                    end_val = series[-1]
                    change = abs((end_val - start_val) / start_val * 100) if start_val != 0 else 0

                    template = "trend_positive" if slope > 0 else "trend_negative"

                    insight = Insight(
                        category="trend",
                        importance=60 + min(30, change / 10),
                        title=f"{'Growth' if slope > 0 else 'Decline'} Trend in {metric_name.replace('_', ' ').title()}",
                        description=self.TEMPLATES[template].format(
                            metric=metric_name.replace("_", " "),
                            change=f"{change:.1f}",
                            start=self._format_value(start_val, metric_name),
                            end=self._format_value(end_val, metric_name),
                        ),
                        data={"slope": slope, "r_squared": r_value**2, "change": change},
                        metrics=[metric_name],
                        confidence=abs(r_value),
                    )
                    self.insights.append(insight)

    def _extract_outlier_insights(self, data: Any, focus_metrics: Optional[List[str]] = None):
        """Extract outlier-related insights.

        Args:
            data: Analysis data
            focus_metrics: Metrics to focus on
        """
        if hasattr(data, "metrics"):
            metrics_data = data.metrics
        elif isinstance(data, dict) and "metrics" in data:
            metrics_data = data["metrics"]
        else:
            return

        for metric_name, values in metrics_data.items():
            if focus_metrics and metric_name not in focus_metrics:
                continue

            if len(values) < 3:
                continue

            # Calculate statistics
            metric_values = list(values.values())
            mean_val = np.mean(metric_values)
            std_val = np.std(metric_values)

            if std_val == 0:
                continue

            # Detect outliers (>2 standard deviations)
            for scenario, value in values.items():
                z_score = (value - mean_val) / std_val

                if abs(z_score) > 2:
                    template = "outlier_high" if z_score > 0 else "outlier_low"

                    insight = Insight(
                        category="outlier",
                        importance=50 + min(30, abs(z_score) * 10),
                        title=f"Outlier Detected in {metric_name.replace('_', ' ').title()}",
                        description=self.TEMPLATES[template].format(
                            scenario=scenario,
                            metric=metric_name.replace("_", " "),
                            value=self._format_value(value, metric_name),
                            std_dev=f"{abs(z_score):.1f}",
                        ),
                        data={"scenario": scenario, "value": value, "z_score": z_score},
                        metrics=[metric_name],
                        confidence=0.85,
                    )
                    self.insights.append(insight)

    def _extract_threshold_insights(self, data: Any, focus_metrics: Optional[List[str]] = None):
        """Extract threshold-related insights.

        Args:
            data: Analysis data
            focus_metrics: Metrics to focus on
        """
        # Define critical thresholds for common metrics
        thresholds = {
            "ruin_probability": 0.01,  # 1% ruin probability
            "var_95": 0,  # Negative VaR
            "var_99": 0,  # Negative VaR
            "growth_rate": 0,  # Negative growth
            "sharpe_ratio": 1.0,  # Below 1.0 Sharpe
            "max_drawdown": 0.2,  # 20% drawdown
        }

        if hasattr(data, "metrics"):
            metrics_data = data.metrics
        elif isinstance(data, dict) and "metrics" in data:
            metrics_data = data["metrics"]
        else:
            return

        for metric_name, values in metrics_data.items():
            if focus_metrics and metric_name not in focus_metrics:
                continue

            # Check if we have a threshold for this metric
            for threshold_metric, threshold_value in thresholds.items():
                if threshold_metric in metric_name.lower():
                    # Count violations
                    if "probability" in metric_name.lower() or "drawdown" in metric_name.lower():
                        violations = [s for s, v in values.items() if v > threshold_value]
                    else:
                        violations = [s for s, v in values.items() if v < threshold_value]

                    if violations:
                        insight = Insight(
                            category="threshold",
                            importance=70 + len(violations) * 5,
                            title=f"Critical Threshold Exceeded for {metric_name.replace('_', ' ').title()}",
                            description=self.TEMPLATES["threshold_exceeded"].format(
                                metric=metric_name.replace("_", " "),
                                threshold=self._format_value(threshold_value, metric_name),
                                count=len(violations),
                            ),
                            data={"violations": violations, "threshold": threshold_value},
                            metrics=[metric_name],
                            confidence=0.95,
                        )
                        self.insights.append(insight)

    def _extract_correlation_insights(self, data: Any, focus_metrics: Optional[List[str]] = None):
        """Extract correlation-related insights.

        Args:
            data: Analysis data
            focus_metrics: Metrics to focus on
        """
        if hasattr(data, "metrics"):
            metrics_data = data.metrics
        elif isinstance(data, dict) and "metrics" in data:
            metrics_data = data["metrics"]
        else:
            return

        # Check correlations between metrics
        metric_names = list(metrics_data.keys())

        for i, metric1 in enumerate(metric_names):
            if focus_metrics and metric1 not in focus_metrics:
                continue

            for metric2 in metric_names[i + 1 :]:
                if focus_metrics and metric2 not in focus_metrics:
                    continue

                # Get common scenarios
                common_scenarios = set(metrics_data[metric1].keys()) & set(
                    metrics_data[metric2].keys()
                )

                if len(common_scenarios) < 3:
                    continue

                # Calculate correlation
                values1 = [metrics_data[metric1][s] for s in common_scenarios]
                values2 = [metrics_data[metric2][s] for s in common_scenarios]

                if np.std(values1) == 0 or np.std(values2) == 0:
                    continue

                corr, p_value = stats.pearsonr(values1, values2)

                # Strong correlation detection
                if abs(corr) > 0.7 and p_value < 0.05:
                    direction = "positive" if corr > 0 else "negative"

                    insight = Insight(
                        category="correlation",
                        importance=50 + abs(corr) * 30,
                        title=f"Strong Correlation Between Metrics",
                        description=self.TEMPLATES["correlation"].format(
                            direction=direction,
                            corr=f"{corr:.2f}",
                            metric1=metric1.replace("_", " "),
                            metric2=metric2.replace("_", " "),
                        ),
                        data={"correlation": corr, "p_value": p_value},
                        metrics=[metric1, metric2],
                        confidence=1 - p_value,
                    )
                    self.insights.append(insight)

    def _format_value(self, value: float, metric_name: str) -> str:
        """Format value based on metric type.

        Args:
            value: Numeric value
            metric_name: Name of the metric

        Returns:
            Formatted string
        """
        if "probability" in metric_name.lower() or "rate" in metric_name.lower():
            return f"{value:.2%}"
        elif "assets" in metric_name.lower() or "value" in metric_name.lower():
            if abs(value) > 1e6:
                return f"${value/1e6:.1f}M"
            elif abs(value) > 1e3:
                return f"${value/1e3:.1f}K"
            else:
                return f"${value:.2f}"
        else:
            return f"{value:.3g}"

    def generate_executive_summary(self, max_points: int = 5, focus_positive: bool = True) -> str:
        """Generate executive summary from insights.

        Args:
            max_points: Maximum number of points
            focus_positive: Focus on positive insights

        Returns:
            Executive summary text
        """
        if not self.insights:
            return "No significant insights were identified in the analysis."

        # Filter insights
        if focus_positive:
            filtered = [
                i
                for i in self.insights
                if i.category in ["performance", "trend"]
                and "best" in i.title.lower()
                or "growth" in i.title.lower()
            ]
        else:
            filtered = self.insights

        # Take top insights
        top_insights = filtered[:max_points] if filtered else self.insights[:max_points]

        # Generate summary
        summary_lines = ["## Executive Summary\n"]
        summary_lines.append("### Key Findings:\n")

        for insight in top_insights:
            summary_lines.append(f"- {insight.to_executive_summary()}")

        # Add recommendation if applicable
        if any(i.category == "performance" for i in top_insights):
            best_performer = next((i for i in top_insights if "best" in i.title.lower()), None)
            if best_performer:
                summary_lines.append(f"\n### Recommendation:")
                summary_lines.append(
                    f"Based on the analysis, the {best_performer.data.get('scenario', 'optimal')} "
                    f"configuration provides the best overall performance."
                )

        return "\n".join(summary_lines)

    def generate_technical_notes(self) -> List[str]:
        """Generate technical notes from insights.

        Returns:
            List of technical observation strings
        """
        notes = []

        # Group insights by category
        by_category = {}
        for insight in self.insights:
            if insight.category not in by_category:
                by_category[insight.category] = []
            by_category[insight.category].append(insight)

        # Generate notes by category
        for category, category_insights in by_category.items():
            if category == "correlation":
                notes.append(
                    f"Correlation Analysis: {len(category_insights)} significant relationships identified"
                )
            elif category == "outlier":
                notes.append(
                    f"Outlier Detection: {len(category_insights)} anomalous scenarios detected"
                )
            elif category == "threshold":
                notes.append(
                    f"Risk Assessment: {len(category_insights)} metrics exceeded critical thresholds"
                )
            elif category == "trend":
                notes.append(
                    f"Trend Analysis: {len(category_insights)} significant patterns observed"
                )

        # Add confidence notes
        high_confidence = [i for i in self.insights if i.confidence > 0.9]
        if high_confidence:
            notes.append(
                f"High Confidence: {len(high_confidence)} insights with >90% statistical confidence"
            )

        return notes

    def export_insights(self, output_path: str, format: str = "markdown") -> str:
        """Export insights to file.

        Args:
            output_path: Path for output file
            format: Output format (markdown, json, csv)

        Returns:
            Path to exported file
        """
        if format == "markdown":
            content = self.generate_executive_summary()
            content += "\n\n## Detailed Insights\n\n"

            for i, insight in enumerate(self.insights, 1):
                content += f"### {i}. {insight.title}\n"
                content += f"{insight.description}\n"
                content += f"- Category: {insight.category}\n"
                content += f"- Importance: {insight.importance:.0f}/100\n"
                content += f"- Confidence: {insight.confidence:.1%}\n\n"

            with open(output_path, "w") as f:
                f.write(content)

        elif format == "json":
            import json

            data = [
                {
                    "title": i.title,
                    "description": i.description,
                    "category": i.category,
                    "importance": i.importance,
                    "confidence": i.confidence,
                    "metrics": i.metrics,
                    "data": i.data,
                }
                for i in self.insights
            ]

            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        elif format == "csv":
            df = pd.DataFrame(
                [
                    {
                        "Title": i.title,
                        "Category": i.category,
                        "Importance": i.importance,
                        "Confidence": i.confidence,
                        "Description": i.description,
                    }
                    for i in self.insights
                ]
            )

            df.to_csv(output_path, index=False)

        return output_path
