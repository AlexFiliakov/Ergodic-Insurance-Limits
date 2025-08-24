"""Comprehensive risk metrics suite for tail risk analysis.

This module provides industry-standard risk metrics including VaR, TVaR, PML,
and Expected Shortfall to quantify tail risk and support insurance optimization
decisions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


@dataclass
class RiskMetricsResult:
    """Container for risk metric calculation results."""

    metric_name: str
    value: float
    confidence_level: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Optional[Dict[str, any]] = None


class RiskMetrics:
    """Calculate comprehensive risk metrics for loss distributions.

    This class provides industry-standard risk metrics for analyzing
    tail risk in insurance and financial applications.
    """

    def __init__(
        self,
        losses: np.ndarray,
        weights: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ):
        """Initialize risk metrics calculator.

        Args:
            losses: Array of loss values (positive values represent losses).
            weights: Optional importance sampling weights.
            seed: Random seed for bootstrap calculations.

        Raises:
            ValueError: If losses array is empty or contains invalid values.
        """
        if len(losses) == 0:
            raise ValueError("Losses array cannot be empty")

        # Handle NaN and infinite values
        valid_mask = np.isfinite(losses)
        if not np.all(valid_mask):
            print(f"Warning: Removing {np.sum(~valid_mask)} non-finite values")
            losses = losses[valid_mask]
            if weights is not None:
                weights = weights[valid_mask]

        self.losses = np.asarray(losses)
        self.weights = weights
        self.rng = np.random.RandomState(seed)

        # Pre-calculate sorted losses for percentile-based metrics
        if weights is None:
            self._sorted_losses = np.sort(self.losses)
        else:
            # Weighted sorting
            sort_idx = np.argsort(self.losses)
            self._sorted_losses = self.losses[sort_idx]
            self._sorted_weights = self.weights[sort_idx]
            self._cumulative_weights = np.cumsum(self._sorted_weights)
            self._cumulative_weights /= self._cumulative_weights[-1]

    def var(
        self,
        confidence: float = 0.99,
        method: str = "empirical",
        bootstrap_ci: bool = False,
        n_bootstrap: int = 1000,
    ) -> Union[float, RiskMetricsResult]:
        """Calculate Value at Risk (VaR).

        VaR represents the loss amount that will not be exceeded with
        a given confidence level over a specific time period.

        Args:
            confidence: Confidence level (e.g., 0.99 for 99% VaR).
            method: 'empirical' or 'parametric' (assumes normal distribution).
            bootstrap_ci: Whether to calculate bootstrap confidence intervals.
            n_bootstrap: Number of bootstrap samples for CI calculation.

        Returns:
            VaR value or RiskMetricsResult with confidence intervals.

        Raises:
            ValueError: If confidence level is not in (0, 1).
        """
        if not 0 < confidence < 1:
            raise ValueError(f"Confidence must be in (0, 1), got {confidence}")

        if method == "empirical":
            var_value = self._empirical_var(confidence)
        elif method == "parametric":
            var_value = self._parametric_var(confidence)
        else:
            raise ValueError(f"Method must be 'empirical' or 'parametric', got {method}")

        if bootstrap_ci:
            ci = self._bootstrap_var_ci(confidence, n_bootstrap)
            return RiskMetricsResult(
                metric_name="VaR",
                value=var_value,
                confidence_level=confidence,
                confidence_interval=ci,
                metadata={"method": method},
            )

        return var_value

    def _empirical_var(self, confidence: float) -> float:
        """Calculate empirical VaR using percentiles."""
        if self.weights is None:
            return np.percentile(self.losses, confidence * 100)
        else:
            # Weighted percentile
            idx = np.searchsorted(self._cumulative_weights, confidence)
            if idx >= len(self._sorted_losses):
                idx = len(self._sorted_losses) - 1
            return self._sorted_losses[idx]

    def _parametric_var(self, confidence: float) -> float:
        """Calculate parametric VaR assuming normal distribution."""
        mean = np.average(self.losses, weights=self.weights)
        if self.weights is None:
            std = np.std(self.losses)
        else:
            variance = np.average((self.losses - mean) ** 2, weights=self.weights)
            std = np.sqrt(variance)

        return mean + std * stats.norm.ppf(confidence)

    def _bootstrap_var_ci(self, confidence: float, n_bootstrap: int) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for VaR."""
        n = len(self.losses)
        var_bootstrap = []

        for _ in range(n_bootstrap):
            if self.weights is None:
                idx = self.rng.choice(n, size=n, replace=True)
                sample = self.losses[idx]
                var_bootstrap.append(np.percentile(sample, confidence * 100))
            else:
                idx = self.rng.choice(n, size=n, replace=True)
                sample = self.losses[idx]
                weights = self.weights[idx]
                # Recalculate weighted percentile
                sort_idx = np.argsort(sample)
                sorted_sample = sample[sort_idx]
                sorted_weights = weights[sort_idx]
                cum_weights = np.cumsum(sorted_weights)
                cum_weights /= cum_weights[-1]
                idx_var = np.searchsorted(cum_weights, confidence)
                if idx_var >= len(sorted_sample):
                    idx_var = len(sorted_sample) - 1
                var_bootstrap.append(sorted_sample[idx_var])

        return np.percentile(var_bootstrap, [2.5, 97.5])

    def tvar(
        self,
        confidence: float = 0.99,
        var_value: Optional[float] = None,
    ) -> float:
        """Calculate Tail Value at Risk (TVaR/CVaR).

        TVaR represents the expected loss given that the loss exceeds VaR.
        It's a coherent risk measure that satisfies sub-additivity.

        Args:
            confidence: Confidence level for VaR threshold.
            var_value: Pre-calculated VaR value (if None, will calculate).

        Returns:
            TVaR value.
        """
        if var_value is None:
            var_value = self.var(confidence)

        if self.weights is None:
            tail_losses = self.losses[self.losses >= var_value]
            if len(tail_losses) == 0:
                return var_value
            return np.mean(tail_losses)
        else:
            mask = self.losses >= var_value
            if not np.any(mask):
                return var_value
            tail_losses = self.losses[mask]
            tail_weights = self.weights[mask]
            return np.average(tail_losses, weights=tail_weights)

    def expected_shortfall(
        self,
        threshold: float,
    ) -> float:
        """Calculate Expected Shortfall (ES) above a threshold.

        ES is the average of all losses that exceed a given threshold.

        Args:
            threshold: Loss threshold.

        Returns:
            Expected shortfall value.
        """
        if self.weights is None:
            tail_losses = self.losses[self.losses >= threshold]
            if len(tail_losses) == 0:
                return 0.0
            return np.mean(tail_losses)
        else:
            mask = self.losses >= threshold
            if not np.any(mask):
                return 0.0
            tail_losses = self.losses[mask]
            tail_weights = self.weights[mask]
            return np.average(tail_losses, weights=tail_weights)

    def pml(self, return_period: int) -> float:
        """Calculate Probable Maximum Loss (PML) for a given return period.

        PML represents the loss amount expected to be equaled or exceeded
        once every 'return_period' years on average.

        Args:
            return_period: Return period in years (e.g., 100 for 100-year event).

        Returns:
            PML value.

        Raises:
            ValueError: If return period is less than 1.
        """
        if return_period < 1:
            raise ValueError(f"Return period must be >= 1, got {return_period}")

        # PML corresponds to the (1 - 1/return_period) percentile
        confidence = 1 - 1 / return_period
        return self.var(confidence)

    def conditional_tail_expectation(
        self,
        confidence: float = 0.99,
    ) -> float:
        """Calculate Conditional Tail Expectation (CTE).

        CTE is similar to TVaR but uses a slightly different calculation method.
        It's the expected value of losses that exceed the VaR threshold.

        Args:
            confidence: Confidence level.

        Returns:
            CTE value.
        """
        # CTE is essentially the same as TVaR in our implementation
        return self.tvar(confidence)

    def maximum_drawdown(self) -> float:
        """Calculate Maximum Drawdown.

        Maximum drawdown measures the largest peak-to-trough decline
        in cumulative value.

        Returns:
            Maximum drawdown value.
        """
        if self.weights is not None:
            # For weighted data, use weighted cumulative sum
            cumsum = np.cumsum(self.losses * self.weights)
        else:
            cumsum = np.cumsum(self.losses)

        # Calculate running maximum
        running_max = np.maximum.accumulate(cumsum)

        # Calculate drawdown
        drawdown = running_max - cumsum

        return np.max(drawdown)

    def economic_capital(
        self,
        confidence: float = 0.999,
        expected_loss: Optional[float] = None,
    ) -> float:
        """Calculate Economic Capital requirement.

        Economic capital is the amount of capital needed to cover
        unexpected losses at a given confidence level.

        Args:
            confidence: Confidence level (typically 99.9% for regulatory).
            expected_loss: Expected loss (if None, will calculate mean).

        Returns:
            Economic capital requirement.
        """
        var_value = self.var(confidence)

        if expected_loss is None:
            expected_loss = np.average(self.losses, weights=self.weights)

        # Economic capital = VaR - Expected Loss
        return max(0, var_value - expected_loss)

    def return_period_curve(
        self,
        return_periods: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate return period curve (exceedance probability curve).

        Args:
            return_periods: Array of return periods to calculate.
                           If None, uses standard periods.

        Returns:
            Tuple of (return_periods, loss_values).
        """
        if return_periods is None:
            return_periods = np.array([2, 5, 10, 25, 50, 100, 200, 250, 500, 1000])

        loss_values = []
        for period in return_periods:
            loss_values.append(self.pml(period))

        return return_periods, np.array(loss_values)

    def tail_index(self, threshold: Optional[float] = None) -> float:
        """Estimate tail index using Hill estimator.

        The tail index characterizes the heaviness of the tail.
        Lower values indicate heavier tails.

        Args:
            threshold: Threshold for tail definition (if None, uses 90th percentile).

        Returns:
            Estimated tail index.
        """
        if threshold is None:
            threshold = np.percentile(self.losses, 90)

        tail_losses = self.losses[self.losses > threshold]
        if len(tail_losses) < 2:
            return np.nan

        # Hill estimator
        k = len(tail_losses)
        hill_estimate = k / np.sum(np.log(tail_losses / threshold))

        return hill_estimate

    def risk_adjusted_metrics(
        self,
        returns: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.02,
    ) -> Dict[str, float]:
        """Calculate risk-adjusted return metrics.

        Args:
            returns: Array of returns (if None, uses negative of losses).
            risk_free_rate: Risk-free rate for Sharpe ratio calculation.

        Returns:
            Dictionary of risk-adjusted metrics.
        """
        if returns is None:
            # Convert losses to returns (negative losses)
            returns = -self.losses

        if self.weights is None:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
        else:
            mean_return = np.average(returns, weights=self.weights)
            variance = np.average((returns - mean_return) ** 2, weights=self.weights)
            std_return = np.sqrt(variance)

        # Sharpe ratio
        sharpe = (mean_return - risk_free_rate) / std_return if std_return > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < risk_free_rate]
        if len(downside_returns) > 0:
            if self.weights is None:
                downside_std = np.std(downside_returns)
            else:
                downside_weights = self.weights[returns < risk_free_rate]
                downside_mean = np.average(downside_returns, weights=downside_weights)
                downside_var = np.average(
                    (downside_returns - downside_mean) ** 2, weights=downside_weights
                )
                downside_std = np.sqrt(downside_var)
            sortino = (mean_return - risk_free_rate) / downside_std if downside_std > 0 else 0
        else:
            sortino = np.inf if mean_return > risk_free_rate else 0

        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "mean_return": mean_return,
            "volatility": std_return,
        }

    def coherence_test(self) -> Dict[str, bool]:
        """Test coherence properties of risk measures.

        A coherent risk measure satisfies:
        1. Monotonicity
        2. Sub-additivity
        3. Positive homogeneity
        4. Translation invariance

        Returns:
            Dictionary indicating which properties are satisfied.
        """
        # This is a simplified test - full testing would require multiple portfolios
        results = {}

        # Test positive homogeneity for TVaR
        tvar_original = self.tvar(0.99)
        scaled_losses = self.losses * 2
        metrics_scaled = RiskMetrics(scaled_losses, self.weights)
        tvar_scaled = metrics_scaled.tvar(0.99)

        results["tvar_positive_homogeneity"] = np.isclose(tvar_scaled, 2 * tvar_original, rtol=0.01)

        # Test translation invariance
        shift = 1000
        shifted_losses = self.losses + shift
        metrics_shifted = RiskMetrics(shifted_losses, self.weights)
        tvar_shifted = metrics_shifted.tvar(0.99)

        results["tvar_translation_invariance"] = np.isclose(
            tvar_shifted, tvar_original + shift, rtol=0.01
        )

        return results

    def summary_statistics(self) -> Dict[str, float]:
        """Calculate comprehensive summary statistics.

        Returns:
            Dictionary of summary statistics.
        """
        if self.weights is None:
            mean = np.mean(self.losses)
            std = np.std(self.losses)
            skew = stats.skew(self.losses)
            kurt = stats.kurtosis(self.losses)
            median = np.median(self.losses)
        else:
            mean = np.average(self.losses, weights=self.weights)
            variance = np.average((self.losses - mean) ** 2, weights=self.weights)
            std = np.sqrt(variance)
            # Weighted skewness
            m3 = np.average((self.losses - mean) ** 3, weights=self.weights)
            skew = m3 / (std**3) if std > 0 else 0
            # Weighted kurtosis
            m4 = np.average((self.losses - mean) ** 4, weights=self.weights)
            kurt = (m4 / (std**4) - 3) if std > 0 else 0
            # Weighted median
            idx = np.searchsorted(self._cumulative_weights, 0.5)
            median = self._sorted_losses[idx]

        return {
            "mean": mean,
            "median": median,
            "std": std,
            "skewness": skew,
            "kurtosis": kurt,
            "min": np.min(self.losses),
            "max": np.max(self.losses),
            "count": len(self.losses),
        }

    def plot_distribution(
        self,
        bins: int = 50,
        show_metrics: bool = True,
        confidence_levels: List[float] = None,
        figsize: Tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """Plot loss distribution with risk metrics overlay.

        Args:
            bins: Number of bins for histogram.
            show_metrics: Whether to show VaR and TVaR lines.
            confidence_levels: Confidence levels for metrics to show.
            figsize: Figure size.

        Returns:
            Matplotlib figure object.
        """
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Histogram
        ax = axes[0, 0]
        if self.weights is None:
            ax.hist(self.losses, bins=bins, density=True, alpha=0.7, edgecolor="black")
        else:
            ax.hist(
                self.losses,
                bins=bins,
                weights=self.weights,
                density=True,
                alpha=0.7,
                edgecolor="black",
            )

        if show_metrics:
            colors = ["red", "orange", "yellow"]
            for i, conf in enumerate(confidence_levels[:3]):
                var_val = self.var(conf)
                tvar_val = self.tvar(conf)
                color = colors[i % len(colors)]
                ax.axvline(var_val, color=color, linestyle="--", label=f"VaR {conf:.0%}")
                ax.axvline(tvar_val, color=color, linestyle=":", label=f"TVaR {conf:.0%}")

        ax.set_xlabel("Loss Amount")
        ax.set_ylabel("Density")
        ax.set_title("Loss Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Q-Q plot
        ax = axes[0, 1]
        if self.weights is None:
            stats.probplot(self.losses, dist="norm", plot=ax)
        else:
            # Weighted Q-Q plot approximation
            theoretical_quantiles = stats.norm.ppf(self._cumulative_weights)
            theoretical_quantiles = theoretical_quantiles[np.isfinite(theoretical_quantiles)]
            empirical_quantiles = self._sorted_losses[: len(theoretical_quantiles)]
            ax.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.5)
            ax.plot(
                [theoretical_quantiles.min(), theoretical_quantiles.max()],
                [theoretical_quantiles.min(), theoretical_quantiles.max()],
                "r--",
            )
            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel("Sample Quantiles")
        ax.set_title("Q-Q Plot (Normal)")
        ax.grid(True, alpha=0.3)

        # Return period curve
        ax = axes[1, 0]
        periods, losses = self.return_period_curve()
        ax.semilogx(periods, losses, "o-", linewidth=2, markersize=6)
        ax.set_xlabel("Return Period (years)")
        ax.set_ylabel("Loss Amount")
        ax.set_title("Return Period Curve")
        ax.grid(True, alpha=0.3, which="both")

        # Risk metrics summary
        ax = axes[1, 1]
        ax.axis("off")
        metrics_text = "Risk Metrics Summary\n" + "=" * 30 + "\n"

        for conf in confidence_levels:
            var_val = self.var(conf)
            tvar_val = self.tvar(conf)
            metrics_text += f"\nConfidence Level: {conf:.1%}\n"
            metrics_text += f"  VaR:  ${var_val:,.0f}\n"
            metrics_text += f"  TVaR: ${tvar_val:,.0f}\n"

        pml_periods = [100, 250]
        metrics_text += f"\nPML Values:\n"
        for period in pml_periods:
            pml_val = self.pml(period)
            metrics_text += f"  {period}-year: ${pml_val:,.0f}\n"

        es_99 = self.expected_shortfall(self.var(0.99))
        metrics_text += f"\nExpected Shortfall (99%): ${es_99:,.0f}\n"

        ec = self.economic_capital(0.999)
        metrics_text += f"Economic Capital (99.9%): ${ec:,.0f}\n"

        ax.text(
            0.1,
            0.5,
            metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="center",
            family="monospace",
        )

        plt.tight_layout()
        return fig


def compare_risk_metrics(
    scenarios: Dict[str, np.ndarray],
    confidence_levels: List[float] = None,
) -> pd.DataFrame:
    """Compare risk metrics across multiple scenarios.

    Args:
        scenarios: Dictionary mapping scenario names to loss arrays.
        confidence_levels: Confidence levels to evaluate.

    Returns:
        DataFrame with comparative metrics.
    """
    import pandas as pd

    if confidence_levels is None:
        confidence_levels = [0.95, 0.99, 0.995]

    results = []

    for scenario_name, losses in scenarios.items():
        metrics = RiskMetrics(losses)
        stats = metrics.summary_statistics()

        row = {"scenario": scenario_name, **stats}

        for conf in confidence_levels:
            row[f"var_{conf:.1%}"] = metrics.var(conf)
            row[f"tvar_{conf:.1%}"] = metrics.tvar(conf)

        row["pml_100yr"] = metrics.pml(100)
        row["pml_250yr"] = metrics.pml(250)
        row["max_drawdown"] = metrics.maximum_drawdown()
        row["economic_capital"] = metrics.economic_capital(0.999)

        results.append(row)

    return pd.DataFrame(results)
