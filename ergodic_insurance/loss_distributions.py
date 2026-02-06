"""Enhanced loss distributions for manufacturing risk modeling.

This module provides parametric loss distributions for realistic insurance claim
modeling, including attritional losses, large losses, and catastrophic events
with revenue-dependent frequency scaling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from .ergodic_analyzer import ErgodicData
    from .exposure_base import ExposureBase
    from .insurance_program import InsuranceProgram


class LossDistribution(ABC):
    """Abstract base class for loss severity distributions.

    Provides a common interface for generating loss amounts and calculating
    statistical properties of the distribution.
    """

    def __init__(self, seed: Optional[Union[int, np.random.SeedSequence]] = None):
        """Initialize the loss distribution.

        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)

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

    def reset_seed(self, seed) -> None:
        """Reset the random seed for reproducibility.

        Args:
            seed: New random seed to use (int or SeedSequence).
        """
        self.rng = np.random.default_rng(seed)


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


class GeneralizedParetoLoss(LossDistribution):
    """Generalized Pareto distribution for modeling excesses over threshold.

    Implements the GPD using scipy.stats.genpareto for Peaks Over Threshold (POT)
    extreme value modeling. According to the Pickands-Balkema-de Haan theorem,
    excesses over a sufficiently high threshold asymptotically follow a GPD.

    The distribution models: P(X - u | X > u) ~ GPD(ξ, β)

    Shape parameter interpretation:
    - ξ < 0: Bounded distribution (Type III - short-tailed)
    - ξ = 0: Exponential distribution (Type I - medium-tailed)
    - ξ > 0: Pareto-type distribution (Type II - heavy-tailed)
    """

    def __init__(
        self,
        severity_shape: float,
        severity_scale: float,
        seed: Optional[Union[int, np.random.SeedSequence]] = None,
    ):
        """Initialize Generalized Pareto distribution.

        Args:
            severity_shape: Shape parameter ξ (any real value).
                - Negative: bounded tail
                - Zero: exponential tail
                - Positive: Pareto-type heavy tail
            severity_scale: Scale parameter β (must be positive).
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If severity_scale <= 0.
        """
        super().__init__(seed)

        if severity_scale <= 0:
            raise ValueError(f"Scale parameter must be positive, got {severity_scale}")

        self.severity_shape = severity_shape
        self.severity_scale = severity_scale

    def generate_severity(self, n_samples: int) -> np.ndarray:
        """Generate GPD samples (excesses above threshold).

        Args:
            n_samples: Number of samples to generate.

        Returns:
            Array of excess amounts above threshold.
        """
        if n_samples <= 0:
            return np.array([])

        # Use scipy.stats.genpareto with c=shape, scale=scale, loc=0
        result = stats.genpareto.rvs(
            c=self.severity_shape,
            scale=self.severity_scale,
            loc=0,
            size=n_samples,
            random_state=self.rng,
        )
        return np.asarray(result)

    def expected_value(self) -> float:
        """Calculate expected excess above threshold.

        Returns:
            Analytical expected value if it exists (ξ < 1), else inf.
            E[X - u | X > u] = β / (1 - ξ) for ξ < 1
        """
        if self.severity_shape >= 1:
            return np.inf
        return self.severity_scale / (1 - self.severity_shape)


@dataclass
class LossEvent:
    """Represents a single loss event with timing and amount."""

    amount: float  # Loss amount
    time: float = 0.0  # Time of occurrence (in years)
    loss_type: str = "operational"  # Type of loss (attritional, large, catastrophic)
    timestamp: Optional[float] = None  # Alternative name for time
    event_type: Optional[str] = None  # Alternative name for loss_type
    description: Optional[str] = None  # Optional description

    def __post_init__(self):
        """Handle alternative parameter names."""
        if self.timestamp is not None and self.time == 0.0:
            self.time = self.timestamp
        if self.event_type is not None and self.loss_type == "operational":
            self.loss_type = self.event_type

    def __le__(self, other):
        """Support ordering by amount."""
        if isinstance(other, (int, float)):
            return self.amount <= other
        if isinstance(other, LossEvent):
            return self.amount <= other.amount
        return NotImplemented

    def __lt__(self, other):
        """Support ordering by amount."""
        if isinstance(other, (int, float)):
            return self.amount < other
        if isinstance(other, LossEvent):
            return self.amount < other.amount
        return NotImplemented


@dataclass
class LossData:
    """Unified loss data structure for cross-module compatibility.

    This dataclass provides a standardized interface for loss data
    that can be used consistently across all modules in the framework.
    """

    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    loss_amounts: np.ndarray = field(default_factory=lambda: np.array([]))
    loss_types: List[str] = field(default_factory=list)
    claim_ids: List[str] = field(default_factory=list)
    development_factors: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate data consistency.

        Returns:
            True if data is valid and consistent, False otherwise.
        """
        # Check array lengths match
        if len(self.timestamps) != len(self.loss_amounts):
            return False

        # Check optional fields match array length if provided
        n_losses = len(self.timestamps)

        # Check each optional field separately for better readability
        if self.loss_types and len(self.loss_types) != n_losses:
            return False

        if self.claim_ids and len(self.claim_ids) != n_losses:
            return False

        if self.development_factors is not None and len(self.development_factors) != n_losses:
            return False

        # Check for valid amounts and timestamps (non-negative)
        if np.any(self.loss_amounts < 0) or (
            len(self.timestamps) > 0 and np.any(self.timestamps < 0)
        ):
            return False

        return True

    def to_ergodic_format(self) -> "ErgodicData":
        """Convert to ergodic analyzer format.

        Returns:
            Data formatted for ergodic analysis.
        """
        from .ergodic_analyzer import ErgodicData

        return ErgodicData(
            time_series=self.timestamps, values=self.loss_amounts, metadata=self.metadata
        )

    def apply_insurance(self, program: "InsuranceProgram") -> "LossData":
        """Apply insurance recoveries to losses.

        Args:
            program: Insurance program to apply.

        Returns:
            New LossData with insurance recoveries applied.
        """
        # Import here to avoid circular dependency
        from .insurance_program import InsuranceProgram

        # Create copy of data
        recovered_amounts = self.loss_amounts.copy()
        recovery_metadata = self.metadata.copy()

        # Apply insurance to each loss
        total_recoveries = 0.0
        total_premiums = 0.0

        for i, amount in enumerate(self.loss_amounts):
            # Process claim through insurance program
            result = program.process_claim(amount)
            recovery = result["insurance_recovery"]
            recovered_amounts[i] = amount - recovery
            total_recoveries += recovery

        # Calculate total premiums
        if hasattr(program, "calculate_annual_premium"):
            total_premiums = program.calculate_annual_premium()

        # Update metadata
        recovery_metadata.update(
            {
                "insurance_applied": True,
                "total_recoveries": total_recoveries,
                "total_premiums": total_premiums,
                "net_benefit": total_recoveries - total_premiums,
            }
        )

        return LossData(
            timestamps=self.timestamps.copy(),
            loss_amounts=recovered_amounts,
            loss_types=self.loss_types.copy() if self.loss_types else [],
            claim_ids=self.claim_ids.copy() if self.claim_ids else [],
            development_factors=self.development_factors.copy()
            if self.development_factors is not None
            else None,
            metadata=recovery_metadata,
        )

    @classmethod
    def from_loss_events(cls, events: List[LossEvent]) -> "LossData":
        """Create LossData from a list of LossEvent objects.

        Args:
            events: List of LossEvent objects.

        Returns:
            LossData instance with consolidated event data.
        """
        if not events:
            return cls()

        timestamps = np.array([e.time for e in events])
        amounts = np.array([e.amount for e in events])
        types = [e.loss_type for e in events]

        # Sort by time
        sort_idx = np.argsort(timestamps)

        return cls(
            timestamps=timestamps[sort_idx],
            loss_amounts=amounts[sort_idx],
            loss_types=[types[i] for i in sort_idx],
            claim_ids=[f"claim_{i}" for i in range(len(events))],
            metadata={"source": "loss_events", "n_events": len(events)},
        )

    def to_loss_events(self) -> List[LossEvent]:
        """Convert LossData back to LossEvent list.

        Returns:
            List of LossEvent objects.
        """
        events = []
        for i, timestamp in enumerate(self.timestamps):
            loss_type = self.loss_types[i] if i < len(self.loss_types) else "unknown"
            events.append(
                LossEvent(time=timestamp, amount=self.loss_amounts[i], loss_type=loss_type)
            )
        return events

    def get_annual_aggregates(self, years: int) -> Dict[int, float]:
        """Aggregate losses by year.

        Args:
            years: Number of years to aggregate over.

        Returns:
            Dictionary mapping year to total loss amount.
        """
        annual_losses = {year: 0.0 for year in range(years)}

        for time, amount in zip(self.timestamps, self.loss_amounts):
            year = int(time)
            if 0 <= year < years:
                annual_losses[year] += amount

        return annual_losses

    def calculate_statistics(self) -> Dict[str, float]:
        """Calculate comprehensive statistics for the loss data.

        Returns:
            Dictionary of statistical metrics.
        """
        if len(self.loss_amounts) == 0:
            return {
                "count": 0,
                "total": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        return {
            "count": len(self.loss_amounts),
            "total": float(np.sum(self.loss_amounts)),  # Boundary: float for NumPy
            "mean": float(np.mean(self.loss_amounts)),  # Boundary: float for NumPy
            "std": float(np.std(self.loss_amounts)),  # Boundary: float for NumPy
            "min": float(np.min(self.loss_amounts)),  # Boundary: float for NumPy
            "max": float(np.max(self.loss_amounts)),  # Boundary: float for NumPy
            "p50": float(np.percentile(self.loss_amounts, 50)),  # Boundary: float for NumPy
            "p95": float(np.percentile(self.loss_amounts, 95)),  # Boundary: float for NumPy
            "p99": float(np.percentile(self.loss_amounts, 99)),  # Boundary: float for NumPy
        }


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
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed) -> None:
        """Re-seed the random state.

        Args:
            seed: New random seed (int or SeedSequence).
        """
        self.rng = np.random.default_rng(seed)

    def get_scaled_frequency(self, revenue: float) -> float:
        """Calculate revenue-scaled frequency.

        Args:
            revenue: Current revenue level (can be float or Decimal).

        Returns:
            Scaled frequency parameter.
        """
        if revenue <= 0:
            return 0.0

        # Boundary: float for NumPy - power operation requires float
        revenue_float = float(revenue)
        scaling_factor = (revenue_float / self.reference_revenue) ** self.revenue_scaling_exponent
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
        exposure: Optional["ExposureBase"] = None,
        seed: Optional[int] = None,
    ):
        """Initialize attritional loss generator.

        Args:
            base_frequency: Base events per year (3-8 typical).
            severity_mean: Mean loss amount ($3K-$100K typical).
            severity_cv: Coefficient of variation (0.6-1.0 typical).
            revenue_scaling_exponent: Revenue scaling power (0.5 = sqrt scaling).
            reference_revenue: Reference revenue for base frequency.
            exposure: Optional exposure object for dynamic frequency scaling.
            seed: Random seed for reproducibility.
        """
        self.exposure = exposure
        self.frequency_generator = FrequencyGenerator(
            base_frequency=base_frequency,
            revenue_scaling_exponent=revenue_scaling_exponent,
            reference_revenue=reference_revenue,
            seed=seed,
        )

        self.severity_distribution = LognormalLoss(mean=severity_mean, cv=severity_cv, seed=seed)

        self.loss_type = "attritional"

    def reseed(self, seed) -> None:
        """Re-seed all internal random states.

        Args:
            seed: New random seed (int or SeedSequence). A SeedSequence is
                used internally to derive independent child seeds for
                frequency and severity.
        """
        ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
        child_seeds = ss.spawn(2)
        self.frequency_generator.reseed(child_seeds[0])
        self.severity_distribution.reset_seed(child_seeds[1])

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
        exposure: Optional["ExposureBase"] = None,
        seed: Optional[int] = None,
    ):
        """Initialize large loss generator.

        Args:
            base_frequency: Base events per year (0.1-0.5 typical).
            severity_mean: Mean loss amount ($500K-$50M typical).
            severity_cv: Coefficient of variation (1.5-2.0 typical).
            revenue_scaling_exponent: Revenue scaling power (0.7 typical).
            reference_revenue: Reference revenue for base frequency.
            exposure: Optional exposure object for dynamic frequency scaling.
            seed: Random seed for reproducibility.
        """
        self.exposure = exposure
        self.frequency_generator = FrequencyGenerator(
            base_frequency=base_frequency,
            revenue_scaling_exponent=revenue_scaling_exponent,
            reference_revenue=reference_revenue,
            seed=seed,
        )

        self.severity_distribution = LognormalLoss(mean=severity_mean, cv=severity_cv, seed=seed)

        self.loss_type = "large"

    def reseed(self, seed) -> None:
        """Re-seed all internal random states.

        Args:
            seed: New random seed (int or SeedSequence). A SeedSequence is
                used internally to derive independent child seeds for
                frequency and severity.
        """
        ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
        child_seeds = ss.spawn(2)
        self.frequency_generator.reseed(child_seeds[0])
        self.severity_distribution.reset_seed(child_seeds[1])

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
        revenue_scaling_exponent: float = 0.0,
        reference_revenue: float = 10_000_000,
        exposure: Optional["ExposureBase"] = None,
        seed: Optional[int] = None,
    ):
        """Initialize catastrophic loss generator.

        Args:
            base_frequency: Base events per year (0.01-0.05 typical).
            severity_alpha: Pareto shape parameter (2.5 typical).
            severity_xm: Pareto minimum value ($1M+ typical).
            exposure: Optional exposure object for dynamic frequency scaling.
            seed: Random seed for reproducibility.
        """
        # Store exposure for dynamic scaling
        self.exposure = exposure
        # Catastrophic events typically don't scale with revenue
        self.frequency_generator = FrequencyGenerator(
            base_frequency=base_frequency,
            revenue_scaling_exponent=revenue_scaling_exponent,
            reference_revenue=reference_revenue,
            seed=seed,
        )

        self.severity_distribution = ParetoLoss(alpha=severity_alpha, xm=severity_xm, seed=seed)

        self.loss_type = "catastrophic"

    def reseed(self, seed) -> None:
        """Re-seed all internal random states.

        Args:
            seed: New random seed (int or SeedSequence). A SeedSequence is
                used internally to derive independent child seeds for
                frequency and severity.
        """
        ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
        child_seeds = ss.spawn(2)
        self.frequency_generator.reseed(child_seeds[0])
        self.severity_distribution.reset_seed(child_seeds[1])

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
        extreme_params: Optional[dict] = None,
        exposure: Optional["ExposureBase"] = None,
        seed: Optional[int] = None,
    ):
        """Initialize manufacturing loss generator.

        Args:
            attritional_params: Parameters for attritional losses.
            large_params: Parameters for large losses.
            catastrophic_params: Parameters for catastrophic losses.
            extreme_params: Optional parameters for extreme value modeling.
                - threshold_value (float): Threshold for GPD application (required if extreme_params provided)
                - severity_shape (float): GPD shape parameter ξ (required)
                - severity_scale (float): GPD scale parameter β > 0 (required)
            exposure: Optional exposure object for dynamic frequency scaling.
            seed: Random seed for reproducibility.
        """
        # Use provided parameters or defaults
        attritional_params = attritional_params or {}
        large_params = large_params or {}
        catastrophic_params = catastrophic_params or {}

        # Store exposure object
        self.exposure = exposure

        # Pass exposure to each generator if provided
        if exposure is not None:
            attritional_params["exposure"] = exposure
            large_params["exposure"] = exposure
            catastrophic_params["exposure"] = exposure

        # Set seed for all generators using SeedSequence for statistical independence
        if seed is not None:
            # Use SeedSequence to spawn statistically independent child sequences
            # This ensures proper independence between generator streams
            ss = np.random.SeedSequence(seed)
            child_seeds = ss.spawn(4)  # 4 generators: attritional, large, catastrophic, gpd

            # Pass child SeedSequences directly to default_rng via sub-generators
            attritional_params["seed"] = child_seeds[0]
            large_params["seed"] = child_seeds[1]
            catastrophic_params["seed"] = child_seeds[2]
            gpd_child_seed = child_seeds[3]
        else:
            gpd_child_seed = None

        self.attritional = AttritionalLossGenerator(**attritional_params)
        self.large = LargeLossGenerator(**large_params)
        self.catastrophic = CatastrophicLossGenerator(**catastrophic_params)

        # Initialize extreme value modeling if parameters provided
        self.extreme_params = extreme_params
        self.threshold_value = None
        self.gpd_generator = None

        if extreme_params is not None:
            self.threshold_value = extreme_params.get("threshold_value")
            if self.threshold_value is not None:
                # Create GPD generator with independent seed from SeedSequence
                gpd_seed = gpd_child_seed if seed is not None else None
                self.gpd_generator = GeneralizedParetoLoss(
                    severity_shape=extreme_params["severity_shape"],
                    severity_scale=extreme_params["severity_scale"],
                    seed=gpd_seed,
                )

    def reseed(self, seed: int) -> None:
        """Re-seed all internal random states using SeedSequence.

        Derives independent child seeds for each sub-generator so that
        parallel workers produce statistically distinct loss sequences.

        Args:
            seed: New random seed.
        """
        ss = np.random.SeedSequence(seed)
        n_children = 4 if self.gpd_generator is not None else 3
        child_seeds = ss.spawn(n_children)

        self.attritional.reseed(child_seeds[0])
        self.large.reseed(child_seeds[1])
        self.catastrophic.reseed(child_seeds[2])

        if self.gpd_generator is not None:
            self.gpd_generator.reset_seed(child_seeds[3])

    @classmethod
    def create_simple(
        cls,
        frequency: float = 0.1,
        severity_mean: float = 5_000_000,
        severity_std: float = 2_000_000,
        seed: Optional[int] = None,
    ) -> "ManufacturingLossGenerator":
        """Create a simple loss generator (migration helper from ClaimGenerator).

        This factory method provides a simplified interface similar to ClaimGenerator,
        making migration easier. It creates a generator with mostly attritional losses
        and minimal catastrophic risk.

        Args:
            frequency: Annual frequency of losses (Poisson lambda).
            severity_mean: Mean loss amount in dollars.
            severity_std: Standard deviation of loss amount.
            seed: Random seed for reproducibility.

        Returns:
            ManufacturingLossGenerator configured for simple use case.

        Examples:
            Simple usage (equivalent to ClaimGenerator)::

                generator = ManufacturingLossGenerator.create_simple(
                    frequency=0.1,
                    severity_mean=5_000_000,
                    severity_std=2_000_000,
                    seed=42
                )
                losses, stats = generator.generate_losses(duration=10, revenue=10_000_000)

            Accessing loss amounts::

                total_loss = sum(loss.amount for loss in losses)
                print(f"Total losses: ${total_loss:,.0f}")
                print(f"Number of events: {stats['total_losses']}")

        Note:
            For advanced features (multiple loss types, extreme value modeling),
            use the standard __init__ method with explicit parameters.

        See Also:
            Migration guide: docs/migration_guides/claim_generator_migration.md
        """
        # Convert coefficient of variation to lognormal parameters
        if severity_mean <= 0:
            raise ValueError(f"severity_mean must be positive, got {severity_mean}")
        if severity_std < 0:
            raise ValueError(f"severity_std must be non-negative, got {severity_std}")

        cv = severity_std / severity_mean if severity_mean > 0 else 0

        # Configure as primarily attritional with small large loss component
        # Attritional losses: 90% of events, use mean/cv approach
        attritional_params = {
            "base_frequency": frequency * 0.9,  # 90% of frequency
            "severity_mean": severity_mean * 0.5,  # Smaller attritional losses
            "severity_cv": cv,
        }

        # Large losses: 10% of events, use mean/cv approach
        large_params = {
            "base_frequency": frequency * 0.1,  # 10% of frequency
            "severity_mean": severity_mean * 2,  # Larger losses
            "severity_cv": cv * 1.5,  # More variable
        }

        # Catastrophic losses: very rare, use Pareto distribution
        catastrophic_params = {
            "base_frequency": 0.001,  # Very rare
            "severity_alpha": 2.5,  # Pareto shape
            "severity_xm": severity_mean * 5,  # Much larger severities
        }

        return cls(
            attritional_params=attritional_params,
            large_params=large_params,
            catastrophic_params=catastrophic_params,
            seed=seed,
        )

    def generate_losses(
        self, duration: float, revenue: float, include_catastrophic: bool = True, time: float = 0.0
    ) -> Tuple[List[LossEvent], Dict[str, Any]]:
        """Generate all types of losses for manufacturing operations.

        Args:
            duration: Simulation period in years.
            revenue: Current revenue level.
            include_catastrophic: Whether to include catastrophic events.
            time: Current time for exposure calculation (default 0.0).

        Returns:
            Tuple of (all_losses, statistics_dict).
        """
        # Use exposure if available to get actual revenue
        if self.exposure is not None:
            actual_revenue = self.exposure.get_exposure(time)
        else:
            actual_revenue = revenue

        # Generate each type of loss
        attritional_losses = self.attritional.generate_losses(duration, actual_revenue)
        large_losses = self.large.generate_losses(duration, actual_revenue)

        if include_catastrophic:
            catastrophic_losses = self.catastrophic.generate_losses(duration, actual_revenue)
        else:
            catastrophic_losses = []

        # Combine all losses
        all_losses = attritional_losses + large_losses + catastrophic_losses

        # Sort by time
        all_losses.sort(key=lambda x: x.time)

        # Apply extreme value transformation if configured
        extreme_losses = []
        if self.extreme_params is not None and self.threshold_value is not None:
            assert self.gpd_generator is not None  # For type checking
            # Identify losses exceeding threshold
            for loss in all_losses:
                if loss.amount > self.threshold_value:
                    # Generate GPD excess
                    gpd_excess = self.gpd_generator.generate_severity(1)[0]
                    # Create new extreme loss with transformed amount
                    extreme_loss = LossEvent(
                        amount=self.threshold_value + gpd_excess,
                        time=loss.time,
                        loss_type="extreme",
                    )
                    extreme_losses.append(extreme_loss)
                    # Remove original loss from its category
                    if loss in attritional_losses:
                        attritional_losses.remove(loss)
                    elif loss in large_losses:
                        large_losses.remove(loss)
                    elif loss in catastrophic_losses:
                        catastrophic_losses.remove(loss)

            # Add extreme losses to combined list
            all_losses = attritional_losses + large_losses + catastrophic_losses + extreme_losses
            # Re-sort by time after adding extreme losses
            all_losses.sort(key=lambda x: x.time)

        # Calculate statistics
        statistics = {
            "total_losses": len(all_losses),
            "attritional_count": len(attritional_losses),
            "large_count": len(large_losses),
            "catastrophic_count": len(catastrophic_losses),
            "extreme_count": len(extreme_losses),
            "total_amount": sum(loss.amount for loss in all_losses),
            "attritional_amount": sum(loss.amount for loss in attritional_losses),
            "large_amount": sum(loss.amount for loss in large_losses),
            "catastrophic_amount": sum(loss.amount for loss in catastrophic_losses),
            "extreme_amount": sum(loss.amount for loss in extreme_losses),
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
            "extreme": [],
            "total": [],
        }

        for _ in range(n_simulations):
            _losses, stats = self.generate_losses(duration, revenue)

            results["attritional"].append(stats["attritional_amount"])
            results["large"].append(stats["large_amount"])
            results["catastrophic"].append(stats["catastrophic_amount"])
            results["extreme"].append(stats.get("extreme_amount", 0))
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
                """Calculate Pareto cumulative distribution function.

                Args:
                    x: Value(s) at which to evaluate the CDF

                Returns:
                    CDF value(s) at x
                """
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
