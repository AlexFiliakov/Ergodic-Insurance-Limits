"""Enhanced loss distributions for manufacturing risk modeling.

This module provides parametric loss distributions for realistic insurance claim
modeling, including attritional losses, large losses, and catastrophic events
with revenue-dependent frequency scaling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats


class LossDistribution(ABC):
    """Abstract base class for loss severity distributions.

    Provides a common interface for generating loss amounts and calculating
    statistical properties of the distribution.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize the loss distribution.

        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def generate_severity(self, n_samples: int) -> np.ndarray:
        """Generate loss severity samples.

        Args:
            n_samples: Number of samples to generate.

        Returns:
            Array of loss amounts.
        """

    @abstractmethod
    def expected_value(self) -> float:
        """Calculate the analytical expected value of the distribution.

        Returns:
            Expected value when analytically available, otherwise estimated.
        """

    def reset_seed(self, seed: int) -> None:
        """Reset the random seed for reproducibility.

        Args:
            seed: New random seed to use.
        """
        self.rng = np.random.RandomState(seed)


class LognormalLoss(LossDistribution):
    """Lognormal loss severity distribution.

    Common for attritional and large losses in manufacturing.
    Parameters can be specified as either (mean, cv) or (mu, sigma).
    """

    def __init__(
        self,
        mean: Optional[float] = None,
        cv: Optional[float] = None,
        mu: Optional[float] = None,
        sigma: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        """Initialize lognormal distribution.

        Args:
            mean: Mean of the lognormal distribution.
            cv: Coefficient of variation (std/mean).
            mu: Log-space mean parameter (alternative to mean/cv).
            sigma: Log-space standard deviation (alternative to mean/cv).
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If invalid parameter combinations are provided.
        """
        super().__init__(seed)

        if mean is not None and cv is not None:
            # Convert from mean/cv parameterization
            if mean <= 0:
                raise ValueError(f"Mean must be positive, got {mean}")
            if cv < 0:
                raise ValueError(f"CV must be non-negative, got {cv}")

            self.mean = mean
            self.cv = cv
            variance = (mean * cv) ** 2

            # Calculate lognormal parameters
            self.sigma = np.sqrt(np.log(1 + variance / mean**2))
            self.mu = np.log(mean) - self.sigma**2 / 2

        elif mu is not None and sigma is not None:
            # Direct lognormal parameters
            if sigma < 0:
                raise ValueError(f"Sigma must be non-negative, got {sigma}")

            self.mu = mu
            self.sigma = sigma
            self.mean = np.exp(mu + sigma**2 / 2)
            self.cv = np.sqrt(np.exp(sigma**2) - 1)
        else:
            raise ValueError("Must provide either (mean, cv) or (mu, sigma) parameters")

    def generate_severity(self, n_samples: int) -> np.ndarray:
        """Generate lognormal loss samples.

        Args:
            n_samples: Number of samples to generate.

        Returns:
            Array of loss amounts.
        """
        if n_samples <= 0:
            return np.array([])

        return self.rng.lognormal(self.mu, self.sigma, size=n_samples)

    def expected_value(self) -> float:
        """Calculate expected value of lognormal distribution.

        Returns:
            Analytical expected value.
        """
        return self.mean


class ParetoLoss(LossDistribution):
    """Pareto loss severity distribution for catastrophic events.

    Heavy-tailed distribution suitable for modeling extreme losses
    with potentially unbounded severity.
    """

    def __init__(self, alpha: float, xm: float, seed: Optional[int] = None):
        """Initialize Pareto distribution.

        Args:
            alpha: Shape parameter (tail index). Lower = heavier tail.
            xm: Scale parameter (minimum value).
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If parameters are invalid.
        """
        super().__init__(seed)

        if alpha <= 0:
            raise ValueError(f"Alpha must be positive, got {alpha}")
        if xm <= 0:
            raise ValueError(f"Minimum value xm must be positive, got {xm}")

        self.alpha = alpha
        self.xm = xm

    def generate_severity(self, n_samples: int) -> np.ndarray:
        """Generate Pareto loss samples.

        Args:
            n_samples: Number of samples to generate.

        Returns:
            Array of loss amounts.
        """
        if n_samples <= 0:
            return np.array([])

        # Generate using inverse transform method
        u = self.rng.uniform(0, 1, size=n_samples)
        return self.xm / (u ** (1 / self.alpha))

    def expected_value(self) -> float:
        """Calculate expected value of Pareto distribution.

        Returns:
            Analytical expected value if it exists (alpha > 1), else inf.
        """
        if self.alpha <= 1:
            return np.inf
        return self.alpha * self.xm / (self.alpha - 1)


@dataclass
class LossEvent:
    """Represents a single loss event with timing and amount."""

    time: float  # Time of occurrence (in years)
    amount: float  # Loss amount
    loss_type: str  # Type of loss (attritional, large, catastrophic)


class FrequencyGenerator:
    """Base class for generating loss event frequencies.

    Supports revenue-dependent scaling of claim frequencies.
    """

    def __init__(
        self,
        base_frequency: float,
        revenue_scaling_exponent: float = 0.0,
        reference_revenue: float = 10_000_000,
        seed: Optional[int] = None,
    ):
        """Initialize frequency generator.

        Args:
            base_frequency: Base expected events per year (at reference revenue).
            revenue_scaling_exponent: Exponent for revenue scaling (0 = no scaling).
            reference_revenue: Reference revenue level for base frequency.
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If parameters are invalid.
        """
        if base_frequency < 0:
            raise ValueError(f"Base frequency must be non-negative, got {base_frequency}")
        if reference_revenue <= 0:
            raise ValueError(f"Reference revenue must be positive, got {reference_revenue}")

        self.base_frequency = base_frequency
        self.revenue_scaling_exponent = revenue_scaling_exponent
        self.reference_revenue = reference_revenue
        self.rng = np.random.RandomState(seed)

    def get_scaled_frequency(self, revenue: float) -> float:
        """Calculate revenue-scaled frequency.

        Args:
            revenue: Current revenue level.

        Returns:
            Scaled frequency parameter.
        """
        if revenue <= 0:
            return 0.0

        scaling_factor = (revenue / self.reference_revenue) ** self.revenue_scaling_exponent
        return float(self.base_frequency * scaling_factor)

    def generate_event_times(self, duration: float, revenue: float) -> np.ndarray:
        """Generate event times using Poisson process.

        Args:
            duration: Time period in years.
            revenue: Revenue level for frequency scaling.

        Returns:
            Array of event times.
        """
        if duration <= 0 or revenue <= 0:
            return np.array([])

        frequency = self.get_scaled_frequency(revenue)
        if frequency <= 0:
            return np.array([])

        # Generate number of events
        n_events = self.rng.poisson(frequency * duration)

        if n_events == 0:
            return np.array([])

        # Generate uniform event times
        event_times = self.rng.uniform(0, duration, size=n_events)
        return np.sort(event_times)


class AttritionalLossGenerator:
    """Generator for high-frequency, low-severity attritional losses.

    Typical for widget manufacturing: worker injuries, quality defects,
    minor property damage.
    """

    def __init__(
        self,
        base_frequency: float = 5.0,
        severity_mean: float = 25_000,
        severity_cv: float = 1.5,
        revenue_scaling_exponent: float = 0.5,
        reference_revenue: float = 10_000_000,
        seed: Optional[int] = None,
    ):
        """Initialize attritional loss generator.

        Args:
            base_frequency: Base events per year (3-8 typical).
            severity_mean: Mean loss amount ($3K-$100K typical).
            severity_cv: Coefficient of variation (0.6-1.0 typical).
            revenue_scaling_exponent: Revenue scaling power (0.5 = sqrt scaling).
            reference_revenue: Reference revenue for base frequency.
            seed: Random seed for reproducibility.
        """
        self.frequency_generator = FrequencyGenerator(
            base_frequency=base_frequency,
            revenue_scaling_exponent=revenue_scaling_exponent,
            reference_revenue=reference_revenue,
            seed=seed,
        )

        self.severity_distribution = LognormalLoss(mean=severity_mean, cv=severity_cv, seed=seed)

        self.loss_type = "attritional"

    def generate_losses(self, duration: float, revenue: float) -> List[LossEvent]:
        """Generate attritional loss events.

        Args:
            duration: Simulation period in years.
            revenue: Current revenue level.

        Returns:
            List of loss events.
        """
        event_times = self.frequency_generator.generate_event_times(duration, revenue)

        if len(event_times) == 0:
            return []

        severities = self.severity_distribution.generate_severity(len(event_times))

        return [
            LossEvent(time=t, amount=s, loss_type=self.loss_type)
            for t, s in zip(event_times, severities)
        ]


class LargeLossGenerator:
    """Generator for medium-frequency, medium-severity large losses.

    Typical for manufacturing: product recalls, major equipment failures,
    litigation settlements.
    """

    def __init__(
        self,
        base_frequency: float = 0.3,
        severity_mean: float = 2_000_000,
        severity_cv: float = 2.0,
        revenue_scaling_exponent: float = 0.7,
        reference_revenue: float = 10_000_000,
        seed: Optional[int] = None,
    ):
        """Initialize large loss generator.

        Args:
            base_frequency: Base events per year (0.1-0.5 typical).
            severity_mean: Mean loss amount ($500K-$50M typical).
            severity_cv: Coefficient of variation (1.5-2.0 typical).
            revenue_scaling_exponent: Revenue scaling power (0.7 typical).
            reference_revenue: Reference revenue for base frequency.
            seed: Random seed for reproducibility.
        """
        self.frequency_generator = FrequencyGenerator(
            base_frequency=base_frequency,
            revenue_scaling_exponent=revenue_scaling_exponent,
            reference_revenue=reference_revenue,
            seed=seed,
        )

        self.severity_distribution = LognormalLoss(mean=severity_mean, cv=severity_cv, seed=seed)

        self.loss_type = "large"

    def generate_losses(self, duration: float, revenue: float) -> List[LossEvent]:
        """Generate large loss events.

        Args:
            duration: Simulation period in years.
            revenue: Current revenue level.

        Returns:
            List of loss events.
        """
        event_times = self.frequency_generator.generate_event_times(duration, revenue)

        if len(event_times) == 0:
            return []

        severities = self.severity_distribution.generate_severity(len(event_times))

        return [
            LossEvent(time=t, amount=s, loss_type=self.loss_type)
            for t, s in zip(event_times, severities)
        ]


class CatastrophicLossGenerator:
    """Generator for low-frequency, high-severity catastrophic losses.

    Uses Pareto distribution for heavy-tailed severity modeling.
    Examples: major equipment failure, facility damage, environmental disasters.
    """

    def __init__(
        self,
        base_frequency: float = 0.03,
        severity_alpha: float = 2.5,
        severity_xm: float = 1_000_000,
        seed: Optional[int] = None,
    ):
        """Initialize catastrophic loss generator.

        Args:
            base_frequency: Base events per year (0.01-0.05 typical).
            severity_alpha: Pareto shape parameter (2.5 typical).
            severity_xm: Pareto minimum value ($1M+ typical).
            seed: Random seed for reproducibility.
        """
        # Catastrophic events typically don't scale with revenue
        self.frequency_generator = FrequencyGenerator(
            base_frequency=base_frequency,
            revenue_scaling_exponent=0.0,  # No revenue scaling
            seed=seed,
        )

        self.severity_distribution = ParetoLoss(alpha=severity_alpha, xm=severity_xm, seed=seed)

        self.loss_type = "catastrophic"

    def generate_losses(self, duration: float, revenue: float = 10_000_000) -> List[LossEvent]:
        """Generate catastrophic loss events.

        Args:
            duration: Simulation period in years.
            revenue: Current revenue level (not used for scaling).

        Returns:
            List of loss events.
        """
        event_times = self.frequency_generator.generate_event_times(duration, revenue)

        if len(event_times) == 0:
            return []

        severities = self.severity_distribution.generate_severity(len(event_times))

        return [
            LossEvent(time=t, amount=s, loss_type=self.loss_type)
            for t, s in zip(event_times, severities)
        ]


class ManufacturingLossGenerator:
    """Composite loss generator for widget manufacturing risks.

    Combines attritional, large, and catastrophic loss generators
    to provide comprehensive risk modeling.
    """

    def __init__(
        self,
        attritional_params: Optional[dict] = None,
        large_params: Optional[dict] = None,
        catastrophic_params: Optional[dict] = None,
        seed: Optional[int] = None,
    ):
        """Initialize manufacturing loss generator.

        Args:
            attritional_params: Parameters for attritional losses.
            large_params: Parameters for large losses.
            catastrophic_params: Parameters for catastrophic losses.
            seed: Random seed for reproducibility.
        """
        # Use provided parameters or defaults
        attritional_params = attritional_params or {}
        large_params = large_params or {}
        catastrophic_params = catastrophic_params or {}

        # Set seed for all generators
        if seed is not None:
            attritional_params["seed"] = seed
            large_params["seed"] = seed + 1  # Different seed for each
            catastrophic_params["seed"] = seed + 2

        self.attritional = AttritionalLossGenerator(**attritional_params)
        self.large = LargeLossGenerator(**large_params)
        self.catastrophic = CatastrophicLossGenerator(**catastrophic_params)

    def generate_losses(
        self, duration: float, revenue: float, include_catastrophic: bool = True
    ) -> Tuple[List[LossEvent], Dict[str, Any]]:
        """Generate all types of losses for manufacturing operations.

        Args:
            duration: Simulation period in years.
            revenue: Current revenue level.
            include_catastrophic: Whether to include catastrophic events.

        Returns:
            Tuple of (all_losses, statistics_dict).
        """
        # Generate each type of loss
        attritional_losses = self.attritional.generate_losses(duration, revenue)
        large_losses = self.large.generate_losses(duration, revenue)

        if include_catastrophic:
            catastrophic_losses = self.catastrophic.generate_losses(duration, revenue)
        else:
            catastrophic_losses = []

        # Combine all losses
        all_losses = attritional_losses + large_losses + catastrophic_losses

        # Sort by time
        all_losses.sort(key=lambda x: x.time)

        # Calculate statistics
        statistics = {
            "total_losses": len(all_losses),
            "attritional_count": len(attritional_losses),
            "large_count": len(large_losses),
            "catastrophic_count": len(catastrophic_losses),
            "total_amount": sum(loss.amount for loss in all_losses),
            "attritional_amount": sum(loss.amount for loss in attritional_losses),
            "large_amount": sum(loss.amount for loss in large_losses),
            "catastrophic_amount": sum(loss.amount for loss in catastrophic_losses),
            "average_loss": (
                sum(loss.amount for loss in all_losses) / len(all_losses) if all_losses else 0
            ),
            "max_loss": max((loss.amount for loss in all_losses), default=0),
            "annual_frequency": len(all_losses) / duration if duration > 0 else 0,
            "annual_expected_loss": sum(loss.amount for loss in all_losses) / duration
            if duration > 0
            else 0,
        }

        return all_losses, statistics

    def validate_distributions(
        self, n_simulations: int = 10000, duration: float = 1.0, revenue: float = 10_000_000
    ) -> Dict[str, Dict[str, float]]:
        """Validate distribution properties through simulation.

        Args:
            n_simulations: Number of simulations to run.
            duration: Duration of each simulation.
            revenue: Revenue level for testing.

        Returns:
            Dictionary of validation statistics.
        """
        results: Dict[str, List[float]] = {
            "attritional": [],
            "large": [],
            "catastrophic": [],
            "total": [],
        }

        for _ in range(n_simulations):
            losses, stats = self.generate_losses(duration, revenue)

            results["attritional"].append(stats["attritional_amount"])
            results["large"].append(stats["large_amount"])
            results["catastrophic"].append(stats["catastrophic_amount"])
            results["total"].append(stats["total_amount"])

        # Calculate validation statistics
        validation = {}
        for loss_type, amounts in results.items():
            amounts_array = np.array(amounts)
            validation[loss_type] = {
                "mean": np.mean(amounts_array),
                "std": np.std(amounts_array),
                "cv": np.std(amounts_array) / np.mean(amounts_array)
                if np.mean(amounts_array) > 0
                else 0,
                "min": np.min(amounts_array),
                "p25": np.percentile(amounts_array, 25),
                "median": np.median(amounts_array),
                "p75": np.percentile(amounts_array, 75),
                "p95": np.percentile(amounts_array, 95),
                "p99": np.percentile(amounts_array, 99),
                "max": np.max(amounts_array),
            }

        return validation


def perform_statistical_tests(
    samples: np.ndarray,
    distribution_type: str,
    params: Dict[str, Any],
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """Perform statistical tests to validate distribution fit.

    Args:
        samples: Generated samples to test.
        distribution_type: Type of distribution ('lognormal' or 'pareto').
        params: Distribution parameters.
        significance_level: Significance level for tests.

    Returns:
        Dictionary with test results.
    """
    results: Dict[str, Any] = {}

    if len(samples) < 20:
        results["error"] = "Insufficient samples for statistical testing"
        return results

    if distribution_type == "lognormal":
        # Kolmogorov-Smirnov test
        if "mu" in params and "sigma" in params:
            ks_stat, ks_pvalue = stats.kstest(
                samples,
                lambda x: stats.lognorm.cdf(x, s=params["sigma"], scale=np.exp(params["mu"])),
            )
            results["ks_test"] = {
                "statistic": ks_stat,
                "p_value": ks_pvalue,
                "reject_null": ks_pvalue < significance_level,
                "interpretation": "Data fits lognormal"
                if ks_pvalue >= significance_level
                else "Data does not fit lognormal",
            }

        # Anderson-Darling test for lognormality
        log_samples = np.log(samples[samples > 0])
        if len(log_samples) > 0:
            ad_result = stats.anderson(log_samples, dist="norm")
            results["anderson_darling"] = {
                "statistic": ad_result.statistic,
                "critical_values": dict(
                    zip(ad_result.significance_level, ad_result.critical_values)
                ),
                "reject_at_5%": ad_result.statistic > ad_result.critical_values[2],  # 5% level
            }

    elif distribution_type == "pareto":
        # Fit Pareto and test
        if "alpha" in params and "xm" in params:
            # Define Pareto CDF properly for vectorized inputs
            def pareto_cdf(x):
                x = np.atleast_1d(x)
                result = np.zeros_like(x)
                mask = x >= params["xm"]
                result[mask] = 1 - (params["xm"] / x[mask]) ** params["alpha"]
                return result if len(x) > 1 else result[0]

            # Simple KS test for Pareto
            ks_stat, ks_pvalue = stats.kstest(samples, pareto_cdf)
            results["ks_test"] = {
                "statistic": ks_stat,
                "p_value": ks_pvalue,
                "reject_null": ks_pvalue < significance_level,
                "interpretation": "Data fits Pareto"
                if ks_pvalue >= significance_level
                else "Data does not fit Pareto",
            }

    # Shapiro-Wilk test for normality of log-transformed data (if applicable)
    if distribution_type == "lognormal" and len(samples) <= 5000:
        log_samples = np.log(samples[samples > 0])
        if len(log_samples) >= 3:
            shapiro_stat, shapiro_pvalue = stats.shapiro(log_samples)
            results["shapiro_wilk"] = {
                "statistic": shapiro_stat,
                "p_value": shapiro_pvalue,
                "reject_null": shapiro_pvalue < significance_level,
                "interpretation": "Log-transformed data is normal"
                if shapiro_pvalue >= significance_level
                else "Log-transformed data is not normal",
            }

    return results
