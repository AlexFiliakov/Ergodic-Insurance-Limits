"""Comprehensive risk metrics suite for tail risk analysis.

This module provides industry-standard risk metrics including VaR, TVaR, PML,
and Expected Shortfall to quantify tail risk and support insurance optimization
decisions.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, overload
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .config import DEFAULT_RISK_FREE_RATE

logger = logging.getLogger(__name__)


@dataclass
class RiskMetricsResult:
    """Container for risk metric calculation results."""

    metric_name: str
    value: float
    confidence_level: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Optional[Dict[str, Any]] = None


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
            logger.warning("Removing %d non-finite values", np.sum(~valid_mask))
            losses = losses[valid_mask]
            if weights is not None:
                weights = weights[valid_mask]

        self.losses = np.asarray(losses)
        self.weights = weights
        self.rng = np.random.default_rng(seed)

        # Pre-calculate sorted losses for percentile-based metrics
        if weights is None:
            self._sorted_losses = np.sort(self.losses)
            self._sorted_weights = None
        else:
            # Weighted sorting
            sort_idx = np.argsort(self.losses)
            self._sorted_losses = self.losses[sort_idx]
            if self.weights is not None:
                self._sorted_weights = self.weights[sort_idx]
                self._cumulative_weights = np.cumsum(self._sorted_weights)
                self._cumulative_weights /= self._cumulative_weights[-1]
            else:
                self._sorted_weights = None

    @overload
    def var(
        self,
        confidence: float = ...,
        method: str = ...,
        *,
        bootstrap_ci: Literal[False] = ...,
        n_bootstrap: int = ...,
    ) -> float: ...

    @overload
    def var(
        self,
        confidence: float = ...,
        method: str = ...,
        *,
        bootstrap_ci: Literal[True],
        n_bootstrap: int = ...,
    ) -> "RiskMetricsResult": ...

    def var(
        self,
        confidence: float = 0.99,
        method: str = "empirical",
        bootstrap_ci: bool = False,
        n_bootstrap: int = 1000,
    ) -> Union[float, "RiskMetricsResult"]:
        """Calculate Value at Risk (VaR).

        VaR represents the loss amount that will not be exceeded with
        a given confidence level over a specific time period.

        Args:
            confidence: Confidence level (e.g., 0.99 for 99% VaR).
            method: 'empirical' or 'parametric' (assumes normal distribution).
            bootstrap_ci: Deprecated. Use ``var_with_ci()`` instead.
                When True, delegates to ``var_with_ci()`` and returns a
                ``RiskMetricsResult``.  Will be removed in a future release.
            n_bootstrap: Deprecated. Use ``var_with_ci(n_bootstrap=...)`` instead.

        Returns:
            VaR value as a float.  (When the deprecated *bootstrap_ci* flag is
            True, returns ``RiskMetricsResult`` for backward compatibility.)

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
            warnings.warn(
                "The 'bootstrap_ci' parameter is deprecated. " "Use var_with_ci() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return self.var_with_ci(
                confidence=confidence,
                method=method,
                n_bootstrap=n_bootstrap,
            )

        return var_value

    def var_with_ci(
        self,
        confidence: float = 0.99,
        method: str = "empirical",
        n_bootstrap: int = 1000,
    ) -> "RiskMetricsResult":
        """Calculate Value at Risk (VaR) with bootstrap confidence intervals.

        Args:
            confidence: Confidence level (e.g., 0.99 for 99% VaR).
            method: 'empirical' or 'parametric' (assumes normal distribution).
            n_bootstrap: Number of bootstrap samples for CI calculation.

        Returns:
            RiskMetricsResult containing the VaR value and confidence interval.

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

        ci = self._bootstrap_var_ci(confidence, n_bootstrap)
        return RiskMetricsResult(
            metric_name="VaR",
            value=var_value,
            confidence_level=confidence,
            confidence_interval=ci,
            metadata={"method": method},
        )

    def _empirical_var(self, confidence: float) -> float:
        """Calculate empirical VaR using percentiles."""
        if self.weights is None:
            return float(np.percentile(self.losses, confidence * 100))
        # Weighted percentile
        idx = np.searchsorted(self._cumulative_weights, confidence)
        if idx >= len(self._sorted_losses):
            idx = len(self._sorted_losses) - 1  # type: ignore
        return float(self._sorted_losses[idx])

    def _parametric_var(self, confidence: float) -> float:
        """Calculate parametric VaR assuming normal distribution."""
        mean = np.average(self.losses, weights=self.weights)
        if self.weights is None:
            std = np.std(self.losses, ddof=1)
        else:
            variance = np.average((self.losses - mean) ** 2, weights=self.weights)
            std = np.sqrt(variance)

        return float(mean + std * stats.norm.ppf(confidence))

    def _bootstrap_var_ci(self, confidence: float, n_bootstrap: int) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for VaR."""
        n = len(self.losses)
        chunk_size = max(1, min(n_bootstrap, 10_000_000 // max(1, n)))
        var_bootstrap = np.empty(n_bootstrap)

        if self.weights is None:
            for start in range(0, n_bootstrap, chunk_size):
                end = min(start + chunk_size, n_bootstrap)
                batch = end - start
                all_idx = self.rng.choice(n, size=(batch, n), replace=True)
                all_samples = self.losses[all_idx]
                var_bootstrap[start:end] = np.percentile(all_samples, confidence * 100, axis=1)
        else:
            for start in range(0, n_bootstrap, chunk_size):
                end = min(start + chunk_size, n_bootstrap)
                batch = end - start
                all_idx = self.rng.choice(n, size=(batch, n), replace=True)
                all_samples = self.losses[all_idx]
                all_weights = self.weights[all_idx]
                sort_idx = np.argsort(all_samples, axis=1)
                sorted_samples = np.take_along_axis(all_samples, sort_idx, axis=1)
                sorted_weights = np.take_along_axis(all_weights, sort_idx, axis=1)
                cum_weights = np.cumsum(sorted_weights, axis=1)
                cum_weights /= cum_weights[:, -1:]
                # Vectorized searchsorted: find first index where cum_weights >= confidence
                found = cum_weights >= confidence
                idx_var = np.argmax(found, axis=1)
                # argmax returns 0 when no match; guard those rows
                no_match = ~found.any(axis=1)
                idx_var[no_match] = n - 1
                var_bootstrap[start:end] = sorted_samples[np.arange(batch), idx_var]

        result = np.percentile(var_bootstrap, [2.5, 97.5])
        return (float(result[0]), float(result[1]))

    def _bootstrap_tvar_ci(self, confidence: float, n_bootstrap: int) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for TVaR."""
        n = len(self.losses)
        chunk_size = max(1, min(n_bootstrap, 10_000_000 // max(1, n)))
        tvar_bootstrap = np.empty(n_bootstrap)

        if self.weights is None:
            for start in range(0, n_bootstrap, chunk_size):
                end = min(start + chunk_size, n_bootstrap)
                batch = end - start
                all_idx = self.rng.choice(n, size=(batch, n), replace=True)
                all_samples = self.losses[all_idx]
                var_values = np.percentile(all_samples, confidence * 100, axis=1)
                # Boolean mask: which samples are in the tail
                mask = all_samples >= var_values[:, np.newaxis]
                masked_samples = all_samples * mask
                tail_sums = masked_samples.sum(axis=1)
                tail_counts = mask.sum(axis=1)
                # Fallback to VaR where no tail samples exist
                has_tail = tail_counts > 0
                tvar_bootstrap[start:end] = np.where(
                    has_tail, tail_sums / np.maximum(tail_counts, 1), var_values
                )
        else:
            for start in range(0, n_bootstrap, chunk_size):
                end = min(start + chunk_size, n_bootstrap)
                batch = end - start
                all_idx = self.rng.choice(n, size=(batch, n), replace=True)
                all_samples = self.losses[all_idx]
                all_weights = self.weights[all_idx]
                # Compute weighted VaR for each bootstrap sample
                sort_idx = np.argsort(all_samples, axis=1)
                sorted_samples = np.take_along_axis(all_samples, sort_idx, axis=1)
                sorted_weights = np.take_along_axis(all_weights, sort_idx, axis=1)
                cum_weights = np.cumsum(sorted_weights, axis=1)
                cum_weights /= cum_weights[:, -1:]
                found = cum_weights >= confidence
                idx_var = np.argmax(found, axis=1)
                no_match = ~found.any(axis=1)
                idx_var[no_match] = n - 1
                var_values = sorted_samples[np.arange(batch), idx_var]
                # Weighted tail average
                mask = all_samples >= var_values[:, np.newaxis]
                weighted_tail_sums = (all_samples * all_weights * mask).sum(axis=1)
                weight_sums = (all_weights * mask).sum(axis=1)
                has_tail = weight_sums > 0
                tvar_bootstrap[start:end] = np.where(
                    has_tail,
                    weighted_tail_sums / np.maximum(weight_sums, 1e-300),
                    var_values,
                )

        result = np.percentile(tvar_bootstrap, [2.5, 97.5])
        return (float(result[0]), float(result[1]))

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
            TVaR value as a float.
        """
        if var_value is None:
            var_value = self.var(confidence)

        if self.weights is None:
            tail_losses = self.losses[self.losses >= var_value]
            if len(tail_losses) == 0:
                return float(var_value) if var_value is not None else 0.0
            return float(np.mean(tail_losses))
        mask = self.losses >= var_value
        if not np.any(mask):
            return float(var_value) if var_value is not None else 0.0
        tail_losses = self.losses[mask]
        tail_weights = self.weights[mask]
        return float(np.average(tail_losses, weights=tail_weights))

    def tvar_with_ci(
        self,
        confidence: float = 0.99,
        n_bootstrap: int = 1000,
    ) -> "RiskMetricsResult":
        """Calculate Tail Value at Risk (TVaR/CVaR) with bootstrap confidence intervals.

        Args:
            confidence: Confidence level for VaR threshold.
            n_bootstrap: Number of bootstrap samples for CI calculation.

        Returns:
            RiskMetricsResult containing the TVaR value and confidence interval.
        """
        tvar_value = self.tvar(confidence)
        ci = self._bootstrap_tvar_ci(confidence, n_bootstrap)
        return RiskMetricsResult(
            metric_name="TVaR",
            value=tvar_value,
            confidence_level=confidence,
            confidence_interval=ci,
        )

    def expected_shortfall(
        self,
        threshold: float,
    ) -> float:
        """Calculate Expected Shortfall (ES) above a threshold.

        ES is the average of all losses that exceed a given threshold.
        Delegates to tvar() with a pre-computed VaR value.

        Args:
            threshold: Loss threshold.

        Returns:
            Expected shortfall value, or 0.0 if no losses exceed threshold.
        """
        # Check if any losses exceed the threshold first
        if self.weights is None:
            tail_losses = self.losses[self.losses >= threshold]
            if len(tail_losses) == 0:
                return 0.0
        else:
            mask = self.losses >= threshold
            if not np.any(mask):
                return 0.0
        # Delegate to tvar with pre-computed threshold as the VaR value
        return self.tvar(var_value=threshold)

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
        """Calculate Maximum Drawdown on cumulative losses.

        Computes the largest peak-to-trough decline in the **cumulative sum
        of losses**, not in portfolio value.  This measures the worst
        stretch of accumulated losses and is useful for sizing reserves.
        It is *not* the same as the standard portfolio-return drawdown
        commonly used in asset management.

        Returns:
            Maximum drawdown value (non-negative).
        """
        # Calculate cumulative sum with overflow protection
        with np.errstate(over="ignore"):
            if self.weights is not None:
                # For weighted data, use weighted cumulative sum
                cumsum = np.cumsum(self.losses * self.weights)
            else:
                cumsum = np.cumsum(self.losses)

        # Handle any overflow by replacing inf values
        if not np.all(np.isfinite(cumsum)):
            max_val = np.finfo(np.float64).max / 100  # Leave some headroom
            cumsum = np.where(np.isfinite(cumsum), cumsum, max_val)

        # Calculate running maximum with overflow protection
        with np.errstate(over="ignore"):
            running_max = np.maximum.accumulate(cumsum)

        # Calculate drawdown
        drawdown = running_max - cumsum

        return float(np.max(drawdown))

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
        var_val = self.var(confidence)

        if expected_loss is None:
            expected_loss = np.average(self.losses, weights=self.weights)

        # Economic capital = VaR - Expected Loss
        return max(0, var_val - expected_loss)

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

        # Validate all return periods at once
        if np.any(return_periods < 1):
            raise ValueError(
                f"Return period must be >= 1, got {return_periods[return_periods < 1][0]}"
            )

        # Compute all percentiles in a single vectorized call instead of
        # looping through pml() -> var() -> np.percentile() one at a time.
        # Fixes #506.
        confidences = 1 - 1 / return_periods
        if self.weights is None:
            loss_values = np.percentile(self.losses, confidences * 100)
        else:
            # Weighted percentile: reuse the pre-sorted data
            indices = np.searchsorted(self._cumulative_weights, confidences)
            indices = np.clip(indices, 0, len(self._sorted_losses) - 1)
            loss_values = self._sorted_losses[indices].astype(float)

        return return_periods, loss_values

    def tail_index(self, threshold: Optional[float] = None) -> float:
        """Estimate the Pareto tail index alpha via Hill's method.

        Computes the Pareto shape parameter alpha (= 1 / gamma), where
        gamma is the extreme value index from Hill (1975).  Larger alpha
        means thinner tails; smaller alpha means heavier tails.

        Note:
            The classical Hill estimator returns gamma = (1/k) * sum(ln(X_i/u)).
            This method returns its reciprocal, alpha = k / sum(ln(X_i/u)),
            which is the maximum-likelihood estimate of the Pareto shape
            parameter.  To recover the Hill gamma, compute ``1 / tail_index()``.

        Args:
            threshold: Threshold for tail definition (if None, uses 90th percentile).

        Returns:
            Estimated Pareto shape parameter alpha (= 1 / Hill gamma).
        """
        if threshold is None:
            threshold = np.percentile(self.losses, 90)

        tail_losses = self.losses[self.losses > threshold]
        if len(tail_losses) < 2:
            return np.nan

        # Pareto alpha MLE via Hill's method (reciprocal of Hill gamma)
        k = len(tail_losses)
        hill_estimate = k / np.sum(np.log(tail_losses / threshold))

        return float(hill_estimate)

    def risk_adjusted_metrics(
        self,
        returns: Optional[np.ndarray] = None,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
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
            std_return = np.std(returns, ddof=1)
        else:
            mean_return = np.average(returns, weights=self.weights)
            variance = np.average((returns - mean_return) ** 2, weights=self.weights)
            # Bessel correction for reliability weights:
            # corrected = pop_variance * V1^2 / (V1^2 - V2)
            # where V1 = sum(w), V2 = sum(w^2)
            v1 = np.sum(self.weights)
            v2 = np.sum(self.weights**2)
            denom = v1**2 - v2
            if denom > 0:
                variance = variance * v1**2 / denom
            std_return = np.sqrt(variance)

        # Sharpe ratio
        sharpe = (mean_return - risk_free_rate) / std_return if std_return > 0 else 0

        # Sortino ratio (downside deviation over all observations)
        # DD = sqrt( (1/N) * sum( min(r_i - target, 0)^2 ) )
        # Reference: Sortino & Price (1994)
        downside_deviations = np.minimum(returns - risk_free_rate, 0)
        if self.weights is None:
            downside_std = np.sqrt(np.mean(downside_deviations**2))
        else:
            downside_std = np.sqrt(np.average(downside_deviations**2, weights=self.weights))
        if downside_std > 0:
            sortino = (mean_return - risk_free_rate) / downside_std
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

        return {k: bool(v) for k, v in results.items()}

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
            idx = int(np.searchsorted(self._cumulative_weights, 0.5))
            idx = min(idx, len(self._sorted_losses) - 1)
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

    def plot_distribution(  # pylint: disable=too-many-locals
        self,
        bins: int = 50,
        show_metrics: bool = True,
        confidence_levels: Optional[List[float]] = None,
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
        metrics_text += "\nPML Values:\n"
        for period in pml_periods:
            pml_val = self.pml(period)
            metrics_text += f"  {period}-year: ${pml_val:,.0f}\n"

        var_99 = self.var(0.99)
        es_99 = self.expected_shortfall(var_99)
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
    confidence_levels: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Compare risk metrics across multiple scenarios.

    Args:
        scenarios: Dictionary mapping scenario names to loss arrays.
        confidence_levels: Confidence levels to evaluate.

    Returns:
        DataFrame with comparative metrics.
    """
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


class ROEAnalyzer:
    """Comprehensive ROE analysis framework.

    This class provides specialized metrics and analysis tools for
    Return on Equity (ROE) calculations, including time-weighted averages,
    component breakdowns, and volatility analysis.
    """

    def __init__(self, roe_series: np.ndarray, equity_series: Optional[np.ndarray] = None):
        """Initialize ROE analyzer.

        Args:
            roe_series: Array of ROE values over time.
            equity_series: Optional array of equity values for weighted calculations.
        """
        self.roe_series = np.asarray(roe_series)
        self.equity_series = np.asarray(equity_series) if equity_series is not None else None

        # Filter out NaN values for clean analysis
        self.valid_mask = ~np.isnan(self.roe_series)
        self.valid_roe = self.roe_series[self.valid_mask]

    def time_weighted_average(self) -> float:
        """Calculate time-weighted average ROE using geometric mean.

        Time-weighted average gives equal weight to each period regardless
        of the equity level, providing a measure of consistent performance.

        Returns:
            Time-weighted average ROE.
        """
        if len(self.valid_roe) == 0:
            return 0.0

        # Convert to growth factors and compute geometric mean
        growth_factors = 1 + self.valid_roe

        # Handle negative growth factors by using arithmetic mean as fallback
        if np.any(growth_factors <= 0):
            return float(np.mean(self.valid_roe))

        return float(np.exp(np.mean(np.log(growth_factors))) - 1)

    def equity_weighted_average(self) -> float:
        """Calculate equity-weighted average ROE.

        Equity-weighted average gives more weight to periods with higher
        equity levels, reflecting the actual dollar impact.

        Returns:
            Equity-weighted average ROE.
        """
        if self.equity_series is None or len(self.valid_roe) == 0:
            return self.time_weighted_average()

        valid_equity = self.equity_series[self.valid_mask]

        if np.sum(valid_equity) == 0:
            return 0.0

        weights = valid_equity / np.sum(valid_equity)
        return float(np.sum(self.valid_roe * weights))

    def rolling_statistics(self, window: int) -> Dict[str, np.ndarray]:
        """Calculate rolling window statistics for ROE.

        Args:
            window: Window size in periods.

        Returns:
            Dictionary with rolling mean, std, min, max arrays.
        """
        n = len(self.roe_series)

        if window > n:
            raise ValueError(f"Window {window} larger than series length {n}")

        # Use pandas rolling for vectorized computation instead of a Python
        # for-loop.  min_periods=1 mirrors the original NaN-skipping behaviour
        # while pd.Series.rolling(min_periods=window) ensures the first
        # (window-1) entries stay NaN.  Fixes #483.
        series = pd.Series(self.roe_series)
        rolling = series.rolling(window, min_periods=window)

        roll_mean = rolling.mean().to_numpy()
        # ddof=0 matches the original np.std (population std)
        roll_std = rolling.std(ddof=0).to_numpy()
        roll_min = rolling.min().to_numpy()
        roll_max = rolling.max().to_numpy()

        risk_free_rate = DEFAULT_RISK_FREE_RATE

        # Sharpe = (mean - risk_free) / std, only where std > 0
        roll_sharpe = np.full(n, np.nan)
        valid = (~np.isnan(roll_std)) & (roll_std > 0)
        roll_sharpe[valid] = (roll_mean[valid] - risk_free_rate) / roll_std[valid]

        return {
            "mean": roll_mean,
            "std": roll_std,
            "min": roll_min,
            "max": roll_max,
            "sharpe": roll_sharpe,
        }

    def volatility_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive volatility metrics for ROE.

        Returns:
            Dictionary with volatility measures.
        """
        if len(self.valid_roe) < 2:
            return {
                "standard_deviation": 0.0,
                "downside_deviation": 0.0,
                "upside_deviation": 0.0,
                "semi_variance": 0.0,
                "coefficient_variation": 0.0,
            }

        mean_roe = np.mean(self.valid_roe)
        std_roe = np.std(self.valid_roe)

        # Downside deviation (below mean, over all observations)
        downside_deviations = np.minimum(self.valid_roe - mean_roe, 0)
        downside_dev = np.sqrt(np.mean(downside_deviations**2))

        # Upside deviation (above mean)
        above_mean = self.valid_roe[self.valid_roe > mean_roe]
        upside_dev = np.std(above_mean) if len(above_mean) > 0 else 0.0

        # Semi-variance (below target, using 0 as target, over ALL observations)
        # SV = (1/N) * sum( min(r_i - target, 0)^2 )
        semi_var = float(np.mean(np.minimum(self.valid_roe, 0) ** 2))

        # Coefficient of variation
        cv = std_roe / abs(mean_roe) if mean_roe != 0 else float("inf")

        return {
            "standard_deviation": std_roe,
            "downside_deviation": downside_dev,
            "upside_deviation": upside_dev,
            "semi_variance": semi_var,
            "coefficient_variation": cv,
        }

    def performance_ratios(
        self, risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    ) -> Dict[str, float]:  # pylint: disable=too-many-locals
        """Calculate performance ratios for ROE.

        Args:
            risk_free_rate: Risk-free rate for Sharpe/Sortino calculations.

        Returns:
            Dictionary with performance ratios.
        """
        if len(self.valid_roe) < 2:
            return {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "information_ratio": 0.0,
                "omega_ratio": 0.0,
            }

        mean_roe = np.mean(self.valid_roe)
        std_roe = np.std(self.valid_roe)

        # Sharpe ratio
        sharpe = (mean_roe - risk_free_rate) / std_roe if std_roe > 0 else 0.0

        # Sortino ratio (downside deviation over all observations)
        # DD = sqrt( (1/N) * sum( min(r_i - target, 0)^2 ) )
        downside_deviations = np.minimum(self.valid_roe - risk_free_rate, 0)
        downside_dev = np.sqrt(np.mean(downside_deviations**2))
        sortino = (mean_roe - risk_free_rate) / downside_dev if downside_dev > 0 else 0.0

        # Calmar ratio (return over max drawdown)
        max_dd = self._calculate_max_drawdown()
        calmar = mean_roe / abs(max_dd) if max_dd != 0 else 0.0

        # Information ratio (vs benchmark, using median as benchmark)
        benchmark = np.median(self.valid_roe)
        active_return = mean_roe - benchmark
        tracking_error = np.std(self.valid_roe - benchmark)
        info_ratio = active_return / tracking_error if tracking_error > 0 else 0.0

        # Omega ratio (probability-weighted gains vs losses)
        threshold = risk_free_rate
        gains = self.valid_roe[self.valid_roe > threshold] - threshold
        losses = threshold - self.valid_roe[self.valid_roe <= threshold]

        omega = np.sum(gains) / np.sum(losses) if np.sum(losses) > 0 else float("inf")

        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "information_ratio": info_ratio,
            "omega_ratio": omega,
        }

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown for ROE series.

        Returns:
            Maximum drawdown value.
        """
        if len(self.valid_roe) < 2:
            return 0.0

        # Calculate cumulative returns with overflow protection
        # Use log-space calculation to avoid overflow
        try:
            # Clip extreme values to prevent overflow
            clipped_roe = np.clip(self.valid_roe, -0.99, 10.0)

            # Calculate cumulative returns
            with np.errstate(over="raise"):
                cumulative = np.cumprod(1 + clipped_roe)
        except (FloatingPointError, OverflowError):
            # Fallback to log-space calculation
            log_returns = np.log1p(np.clip(self.valid_roe, -0.99, 10.0))
            with np.errstate(over="ignore"):
                cumulative = np.exp(np.cumsum(log_returns))
            # Handle overflow in exp
            if not np.all(np.isfinite(cumulative)):
                cumulative = np.where(np.isfinite(cumulative), cumulative, 1e10)

        # Handle any remaining inf/nan values
        if not np.all(np.isfinite(cumulative)):
            # Replace inf/nan with large but finite values
            cumulative = np.where(np.isfinite(cumulative), cumulative, 1e10)

        running_max = np.maximum.accumulate(cumulative)

        # Avoid division by zero or near-zero
        with np.errstate(divide="ignore", invalid="ignore"):
            drawdown = (cumulative - running_max) / np.maximum(running_max, 1e-10)
            drawdown = np.where(np.isfinite(drawdown), drawdown, 0.0)

        return float(np.min(drawdown))

    def distribution_analysis(self) -> Dict[str, float]:
        """Analyze the distribution of ROE values.

        Returns:
            Dictionary with distribution statistics.
        """
        if len(self.valid_roe) == 0:
            return {
                "mean": 0.0,
                "median": 0.0,
                "skewness": 0.0,
                "kurtosis": 0.0,
                "percentile_5": 0.0,
                "percentile_25": 0.0,
                "percentile_75": 0.0,
                "percentile_95": 0.0,
            }

        return {
            "mean": np.mean(self.valid_roe),
            "median": np.median(self.valid_roe),
            "skewness": stats.skew(self.valid_roe) if len(self.valid_roe) > 2 else 0.0,
            "kurtosis": stats.kurtosis(self.valid_roe) if len(self.valid_roe) > 3 else 0.0,
            "percentile_5": np.percentile(self.valid_roe, 5),
            "percentile_25": np.percentile(self.valid_roe, 25),
            "percentile_75": np.percentile(self.valid_roe, 75),
            "percentile_95": np.percentile(self.valid_roe, 95),
        }

    def stability_analysis(self, periods: Optional[List[int]] = None) -> Dict[str, Any]:
        """Analyze ROE stability across different time periods.

        Args:
            periods: List of period lengths to analyze (default: [1, 3, 5, 10]).

        Returns:
            Dictionary with stability metrics for each period.
        """
        if periods is None:
            periods = [1, 3, 5, 10]

        stability_metrics = {}

        for period in periods:
            if period > len(self.roe_series):
                continue

            rolling_stats = self.rolling_statistics(period)

            stability_metrics[f"{period}yr"] = {
                "mean_stability": 1
                - np.nanstd(rolling_stats["mean"]) / (np.nanmean(rolling_stats["mean"]) + 1e-10),
                "volatility_stability": 1
                - np.nanstd(rolling_stats["std"]) / (np.nanmean(rolling_stats["std"]) + 1e-10),
                "range": np.nanmax(rolling_stats["max"]) - np.nanmin(rolling_stats["min"]),
                "consistency": np.sum(rolling_stats["mean"] > 0)
                / np.sum(~np.isnan(rolling_stats["mean"])),
            }

        return stability_metrics
