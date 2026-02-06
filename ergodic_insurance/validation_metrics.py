"""Validation metrics for walk-forward analysis and strategy backtesting.

This module provides performance metrics and comparison tools for evaluating
insurance strategies across training and testing periods in walk-forward validation.

Example:
    >>> from validation_metrics import ValidationMetrics, MetricCalculator
    >>> import numpy as np

    >>> # Calculate metrics for a strategy's performance
    >>> returns = np.random.normal(0.08, 0.02, 1000)
    >>> losses = np.random.exponential(100000, 1000)
    >>>
    >>> calculator = MetricCalculator()
    >>> metrics = calculator.calculate_metrics(
    ...     returns=returns,
    ...     losses=losses,
    ...     final_assets=10000000
    ... )
    >>>
    >>> print(f"ROE: {metrics.roe:.2%}")
    >>> print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ValidationMetrics:
    """Container for validation performance metrics.

    Attributes:
        roe: Return on equity (annualized)
        ruin_probability: Probability of insolvency
        growth_rate: Compound annual growth rate
        volatility: Standard deviation of returns
        sharpe_ratio: Risk-adjusted return metric
        max_drawdown: Maximum peak-to-trough decline
        var_95: Value at Risk at 95% confidence
        cvar_95: Conditional Value at Risk at 95% confidence
        win_rate: Percentage of profitable periods
        profit_factor: Ratio of gross profits to gross losses
        recovery_time: Average time to recover from drawdown
        stability: R-squared of equity curve
    """

    roe: float
    ruin_probability: float
    growth_rate: float
    volatility: float
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    recovery_time: float = 0.0
    stability: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary.

        Returns:
            Dictionary of metric values.
        """
        return {
            "roe": self.roe,
            "ruin_probability": self.ruin_probability,
            "growth_rate": self.growth_rate,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "recovery_time": self.recovery_time,
            "stability": self.stability,
        }

    def compare(self, other: "ValidationMetrics") -> Dict[str, float]:
        """Compare metrics with another set.

        Args:
            other: Metrics to compare against.

        Returns:
            Dictionary of percentage differences.
        """
        comparisons = {}
        for key, value in self.to_dict().items():
            other_value = getattr(other, key)
            if other_value != 0:
                comparisons[f"{key}_diff"] = (value - other_value) / abs(other_value)
            else:
                comparisons[f"{key}_diff"] = 0.0 if value == 0 else float("inf")
        return comparisons


@dataclass
class StrategyPerformance:
    """Performance tracking for a single strategy.

    Attributes:
        strategy_name: Name of the strategy
        in_sample_metrics: Metrics from training period
        out_sample_metrics: Metrics from testing period
        degradation: Performance degradation from in-sample to out-sample
        overfitting_score: Degree of overfitting (0 = none, 1 = severe)
        consistency_score: Consistency across multiple windows
        metadata: Additional strategy-specific data
    """

    strategy_name: str
    in_sample_metrics: Optional[ValidationMetrics] = None
    out_sample_metrics: Optional[ValidationMetrics] = None
    degradation: Dict[str, float] = field(default_factory=dict)
    overfitting_score: float = 0.0
    consistency_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_degradation(self):
        """Calculate performance degradation from in-sample to out-of-sample."""
        if self.in_sample_metrics and self.out_sample_metrics:
            self.degradation = self.out_sample_metrics.compare(self.in_sample_metrics)

            # Calculate overfitting score based on key metrics
            key_metrics = ["roe", "sharpe_ratio", "growth_rate"]
            degradations = [abs(self.degradation.get(f"{m}_diff", 0)) for m in key_metrics]
            self.overfitting_score = float(np.mean(degradations))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert performance to DataFrame for reporting.

        Returns:
            DataFrame with performance metrics.
        """
        data = []

        if self.in_sample_metrics:
            row: Dict[str, Any] = {"period": "in_sample", "strategy": self.strategy_name}
            metrics_dict = self.in_sample_metrics.to_dict()
            row.update({k: str(v) if isinstance(v, float) else v for k, v in metrics_dict.items()})
            data.append(row)

        if self.out_sample_metrics:
            row2: Dict[str, Any] = {"period": "out_sample", "strategy": self.strategy_name}
            metrics_dict = self.out_sample_metrics.to_dict()
            row2.update({k: str(v) if isinstance(v, float) else v for k, v in metrics_dict.items()})
            data.append(row2)

        return pd.DataFrame(data) if data else pd.DataFrame()


class MetricCalculator:
    """Calculator for performance metrics from simulation results."""

    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize metric calculator.

        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation.
        """
        self.risk_free_rate = risk_free_rate

    def calculate_metrics(  # pylint: disable=too-many-locals
        self,
        returns: np.ndarray,
        losses: Optional[np.ndarray] = None,
        final_assets: Optional[np.ndarray] = None,
        initial_assets: float = 10000000,
        n_years: Optional[int] = None,
    ) -> ValidationMetrics:
        """Calculate comprehensive performance metrics.

        Args:
            returns: Array of period returns
            losses: Array of loss amounts (optional)
            final_assets: Array of final asset values (optional)
            initial_assets: Initial asset value
            n_years: Number of years for annualization

        Returns:
            ValidationMetrics object with calculated metrics.
        """
        # Basic return metrics
        roe = float(np.mean(returns))
        volatility = float(np.std(returns))

        # Growth rate
        if final_assets is not None and len(final_assets) > 0:
            if n_years:
                growth_rate = float(np.mean((final_assets / initial_assets) ** (1 / n_years) - 1))
            else:
                growth_rate = float(np.mean(final_assets / initial_assets - 1))
        else:
            growth_rate = roe

        # Risk metrics
        sharpe_ratio = (roe - self.risk_free_rate) / volatility if volatility > 0 else 0.0

        # Drawdown analysis
        if len(returns) > 1:
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = float(abs(np.min(drawdown)))
        else:
            max_drawdown = 0.0

        # Value at Risk
        var_95 = float(np.percentile(returns, 5))
        cvar_95 = (
            float(np.mean(returns[returns <= var_95]))
            if len(returns[returns <= var_95]) > 0
            else var_95
        )

        # Win rate and profit factor
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0.0

        if len(negative_returns) > 0:
            profit_factor = abs(np.sum(positive_returns) / np.sum(negative_returns))
        else:
            profit_factor = float("inf") if len(positive_returns) > 0 else 1.0

        # Ruin probability
        if final_assets is not None and len(final_assets) > 0:
            ruin_probability = float(np.mean(final_assets <= 0))
        else:
            ruin_probability = 0.0

        # Stability (R-squared of cumulative returns)
        if len(returns) > 2:
            cumulative = np.cumprod(1 + returns)
            x = np.arange(len(cumulative))
            _slope, _intercept, r_value, _, _ = stats.linregress(x, np.log(cumulative))
            stability = r_value**2
        else:
            stability = 0.0

        return ValidationMetrics(
            roe=roe,
            ruin_probability=ruin_probability,
            growth_rate=growth_rate,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            win_rate=win_rate,
            profit_factor=profit_factor,
            recovery_time=0.0,  # Would require more detailed drawdown analysis
            stability=stability,
        )

    def calculate_rolling_metrics(
        self, returns: np.ndarray, window_size: int = 252
    ) -> pd.DataFrame:
        """Calculate rolling window metrics.

        Args:
            returns: Array of returns
            window_size: Size of rolling window

        Returns:
            DataFrame with rolling metrics.
        """
        n_windows = len(returns) - window_size + 1
        metrics_list = []

        for i in range(n_windows):
            window_returns = returns[i : i + window_size]
            metrics = self.calculate_metrics(window_returns)
            metrics_dict = metrics.to_dict()
            metrics_dict["window_start"] = i
            metrics_dict["window_end"] = i + window_size
            metrics_list.append(metrics_dict)

        return pd.DataFrame(metrics_list)


class PerformanceTargets:
    """User-defined performance targets for strategy evaluation.

    Attributes:
        min_roe: Minimum acceptable ROE
        max_ruin_probability: Maximum acceptable ruin probability
        min_sharpe_ratio: Minimum acceptable Sharpe ratio
        max_drawdown: Maximum acceptable drawdown
        min_growth_rate: Minimum acceptable growth rate
    """

    def __init__(
        self,
        min_roe: Optional[float] = None,
        max_ruin_probability: Optional[float] = None,
        min_sharpe_ratio: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        min_growth_rate: Optional[float] = None,
    ):
        """Initialize performance targets.

        Args:
            min_roe: Minimum ROE target
            max_ruin_probability: Maximum ruin probability target
            min_sharpe_ratio: Minimum Sharpe ratio target
            max_drawdown: Maximum drawdown target
            min_growth_rate: Minimum growth rate target
        """
        self.min_roe = min_roe
        self.max_ruin_probability = max_ruin_probability
        self.min_sharpe_ratio = min_sharpe_ratio
        self.max_drawdown = max_drawdown
        self.min_growth_rate = min_growth_rate

    def evaluate(self, metrics: ValidationMetrics) -> Tuple[bool, List[str]]:
        """Evaluate metrics against targets.

        Args:
            metrics: Metrics to evaluate

        Returns:
            Tuple of (meets_all_targets, list_of_failures)
        """
        failures = []

        if self.min_roe is not None and metrics.roe < self.min_roe:
            failures.append(f"ROE {metrics.roe:.2%} < target {self.min_roe:.2%}")

        if (
            self.max_ruin_probability is not None
            and metrics.ruin_probability > self.max_ruin_probability
        ):
            failures.append(
                f"Ruin probability {metrics.ruin_probability:.2%} > target {self.max_ruin_probability:.2%}"
            )

        if self.min_sharpe_ratio is not None and metrics.sharpe_ratio < self.min_sharpe_ratio:
            failures.append(
                f"Sharpe ratio {metrics.sharpe_ratio:.2f} < target {self.min_sharpe_ratio:.2f}"
            )

        if self.max_drawdown is not None and metrics.max_drawdown > self.max_drawdown:
            failures.append(
                f"Max drawdown {metrics.max_drawdown:.2%} > target {self.max_drawdown:.2%}"
            )

        if self.min_growth_rate is not None and metrics.growth_rate < self.min_growth_rate:
            failures.append(
                f"Growth rate {metrics.growth_rate:.2%} < target {self.min_growth_rate:.2%}"
            )

        return len(failures) == 0, failures
