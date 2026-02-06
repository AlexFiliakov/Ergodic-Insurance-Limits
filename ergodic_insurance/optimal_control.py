"""Optimal control strategies for insurance decisions.

This module provides implementations of various control strategies derived from
the HJB solver, including feedback control laws, state-dependent insurance limits,
and integration with the simulation framework.

Key Components:
    - ControlSpace: Defines feasible insurance control parameters
    - ControlStrategy: Abstract base for control strategies
    - StaticControl: Fixed insurance parameters throughout simulation
    - HJBFeedbackControl: State-dependent optimal control from HJB solution
    - TimeVaryingControl: Predetermined time-based control schedule
    - OptimalController: Integrates control strategies with simulations

Typical Workflow:
    1. Solve HJB equation to get optimal policy
    2. Create control strategy (e.g., HJBFeedbackControl)
    3. Initialize OptimalController with strategy
    4. Apply controls in simulation loop
    5. Track and analyze performance

Example:
    >>> # Solve HJB problem
    >>> solver = HJBSolver(problem, config)
    >>> value_func, policy = solver.solve()
    >>>
    >>> # Create feedback control
    >>> control_space = ControlSpace(
    ...     limits=[(1e6, 5e7)],
    ...     retentions=[(1e5, 1e7)]
    ... )
    >>> strategy = HJBFeedbackControl(solver, control_space)
    >>>
    >>> # Apply in simulation
    >>> controller = OptimalController(strategy, control_space)
    >>> insurance = controller.apply_control(manufacturer, time=t)

Author: Alex Filiakov
Date: 2025-01-26
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import interpolate

from .hjb_solver import HJBSolver
from .insurance_program import EnhancedInsuranceLayer, InsuranceProgram
from .manufacturer import WidgetManufacturer

logger = logging.getLogger(__name__)


class ControlMode(Enum):
    """Mode of control application.

    Attributes:
        STATIC: Fixed control parameters that never change.
        STATE_FEEDBACK: Control depends on current system state.
        TIME_VARYING: Control follows predetermined time schedule.
        ADAPTIVE: Control adapts based on observed history.
    """

    STATIC = "static"  # Fixed control parameters
    STATE_FEEDBACK = "state_feedback"  # Control depends on current state
    TIME_VARYING = "time_varying"  # Control depends on time
    ADAPTIVE = "adaptive"  # Control adapts based on history


@dataclass
class ControlSpace:
    """Definition of the control space for insurance decisions."""

    limits: List[Tuple[float, float]]  # (min, max) for each layer
    retentions: List[Tuple[float, float]]  # (min, max) for each layer
    coverage_percentages: List[Tuple[float, float]] = field(default_factory=list)
    reinsurance_limits: Optional[List[Tuple[float, float]]] = None

    def __post_init__(self):
        """Validate control space configuration.

        Raises:
            ValueError: If limits and retentions have different number of layers.
            ValueError: If coverage percentages don't match number of layers.
            ValueError: If any bounds are invalid (min >= max).
            ValueError: If coverage percentages are outside [0, 1] range.
        """
        # Ensure all lists have same length
        n_layers = len(self.limits)
        if len(self.retentions) != n_layers:
            raise ValueError("Limits and retentions must have same number of layers")

        # Set default coverage percentages if not provided
        if not self.coverage_percentages:
            self.coverage_percentages = [(0.8, 1.0)] * n_layers
        elif len(self.coverage_percentages) != n_layers:
            raise ValueError("Coverage percentages must match number of layers")

        # Validate bounds
        for i, ((lmin, lmax), (rmin, rmax), (cmin, cmax)) in enumerate(
            zip(self.limits, self.retentions, self.coverage_percentages)
        ):
            if lmin >= lmax:
                raise ValueError(f"Invalid limit bounds for layer {i}")
            if rmin >= rmax:
                raise ValueError(f"Invalid retention bounds for layer {i}")
            if not 0 <= cmin <= cmax <= 1:
                raise ValueError(f"Invalid coverage percentage bounds for layer {i}")

    def get_dimensions(self) -> int:
        """Get total number of control dimensions.

        Returns:
            int: Total number of control variables across all layers
                and control types.

        Note:
            Used for determining the size of control vectors in
            optimization algorithms.
        """
        n = len(self.limits) * 2  # Limits and retentions
        if self.coverage_percentages:
            n += len(self.coverage_percentages)
        if self.reinsurance_limits:
            n += len(self.reinsurance_limits)
        return n

    def to_array(
        self, limits: List[float], retentions: List[float], coverages: Optional[List[float]] = None
    ) -> np.ndarray:
        """Convert control parameters to array format.

        Args:
            limits: Insurance limits for each layer.
            retentions: Retention levels for each layer.
            coverages: Optional coverage percentages. If None,
                defaults to full coverage.

        Returns:
            np.ndarray: Flattened control array suitable for
                optimization algorithms.

        Note:
            Array order is: [limits, retentions, coverages].
        """
        controls = []
        controls.extend(limits)
        controls.extend(retentions)
        if coverages:
            controls.extend(coverages)
        return np.array(controls)

    def from_array(self, control_array: np.ndarray) -> Dict[str, List[float]]:
        """Convert control array back to named parameters.

        Args:
            control_array: Flattened control array from optimization.

        Returns:
            Dict[str, List[float]]: Dictionary with keys 'limits',
                'retentions', and 'coverages' mapping to lists of
                values for each layer.

        Note:
            Inverse operation of to_array().
        """
        n_layers = len(self.limits)
        idx = 0

        # Extract limits
        limits = control_array[idx : idx + n_layers].tolist()
        idx += n_layers

        # Extract retentions
        retentions = control_array[idx : idx + n_layers].tolist()
        idx += n_layers

        # Extract coverages if present
        coverages = []
        if idx < len(control_array) and self.coverage_percentages:
            coverages = control_array[idx : idx + n_layers].tolist()

        return {"limits": limits, "retentions": retentions, "coverages": coverages}


class ControlStrategy(ABC):
    """Abstract base class for control strategies.

    All control strategies must implement methods to:
    1. Determine control actions based on state/time
    2. Update internal parameters based on outcomes
    """

    @abstractmethod
    def get_control(self, state: Dict[str, float], time: float = 0.0) -> Dict[str, Any]:
        """Get control action for current state and time.

        Args:
            state: Current state dictionary containing keys like
                'wealth', 'assets', 'cumulative_losses', etc.
            time: Current simulation time.

        Returns:
            Dict[str, Any]: Control actions with keys 'limits',
                'retentions', and 'coverages', each mapping to
                lists of values.
        """

    @abstractmethod
    def update(self, state: Dict[str, float], outcome: Dict[str, float]):
        """Update strategy based on observed outcome.

        Args:
            state: State where control was applied.
            outcome: Observed outcome containing keys like
                'losses', 'premium_costs', 'claim_payments', etc.

        Note:
            May be no-op for non-adaptive strategies.
        """


class StaticControl(ControlStrategy):
    """Static control strategy with fixed parameters.

    This is the simplest control strategy where insurance parameters
    remain constant throughout the simulation.
    """

    def __init__(
        self, limits: List[float], retentions: List[float], coverages: Optional[List[float]] = None
    ):
        """Initialize static control.

        Args:
            limits: Fixed insurance limits for each layer.
            retentions: Fixed retention levels for each layer.
            coverages: Optional fixed coverage percentages. If None,
                defaults to 100% coverage for all layers.
        """
        self.limits = limits
        self.retentions = retentions
        self.coverages = coverages if coverages else [1.0] * len(limits)

        logger.info(f"Initialized static control with {len(limits)} layers")

    def get_control(self, state: Dict[str, float], time: float = 0.0) -> Dict[str, Any]:
        """Return fixed control parameters.

        Args:
            state: Current state (ignored for static control).
            time: Current time (ignored for static control).

        Returns:
            Dict[str, Any]: Fixed control parameters.
        """
        return {"limits": self.limits, "retentions": self.retentions, "coverages": self.coverages}

    def update(self, state: Dict[str, float], outcome: Dict[str, float]):
        """No updates for static control.

        Args:
            state: State where control was applied (ignored).
            outcome: Observed outcome (ignored).
        """


class HJBFeedbackControl(ControlStrategy):
    """State-feedback control derived from HJB solution.

    This strategy uses the optimal policy computed by the HJB solver
    to determine insurance parameters based on the current state.
    """

    def __init__(
        self,
        hjb_solver: HJBSolver,
        control_space: ControlSpace,
        state_mapping: Optional[Callable[[Dict[str, float]], np.ndarray]] = None,
    ):
        """Initialize HJB feedback control.

        Args:
            hjb_solver: Solved HJB problem containing optimal policy.
            control_space: Definition of feasible control space.
            state_mapping: Optional function to map simulation state
                dictionary to HJB state array. If None, uses default
                mapping based on common state variable names.

        Raises:
            ValueError: If HJB solver hasn't been solved yet.
        """
        if hjb_solver.optimal_policy is None:
            raise ValueError("HJB solver must be solved before creating feedback control")

        self.hjb_solver = hjb_solver
        self.control_space = control_space
        self.state_mapping = state_mapping if state_mapping else self._default_state_mapping

        # Create interpolators for each control
        self._create_interpolators()

        logger.info("Initialized HJB feedback control")

    def _default_state_mapping(self, state: Dict[str, float]) -> np.ndarray:
        """Default mapping from simulation state to HJB state.

        Args:
            state: Simulation state dictionary with keys like
                'assets', 'wealth', 'equity', 'time', etc.

        Returns:
            np.ndarray: HJB state array with mapped values in order
                expected by the HJB solver.

        Note:
            Searches for common state variable names and maps them
            to HJB state dimensions. Order: wealth, time.
            The HJB problem in create_hjb_controller uses exactly 2 dimensions.
        """
        # Map common state variables - must match HJB problem dimensions
        hjb_state = []

        # Wealth/assets (first dimension)
        if "assets" in state:
            hjb_state.append(state["assets"])
        elif "wealth" in state:
            hjb_state.append(state["wealth"])
        elif "equity" in state:
            hjb_state.append(state["equity"])
        else:
            hjb_state.append(1e7)  # Default value

        # Time (second dimension)
        if "time" in state:
            hjb_state.append(state["time"])
        else:
            hjb_state.append(0.0)  # Default time

        # Return exactly 2 dimensions to match the HJB problem
        return np.array(hjb_state[:2])

    def _create_interpolators(self):
        """Create interpolation functions for optimal controls.

        Creates a RegularGridInterpolator for each control variable
        to enable fast evaluation of the optimal policy at arbitrary
        state points.
        """
        self.interpolators = {}

        # Get state grids from HJB solver
        grids = self.hjb_solver.problem.state_space.grids

        # Create interpolator for each control variable
        if self.hjb_solver.optimal_policy is None:
            return
        for control_name, control_values in self.hjb_solver.optimal_policy.items():
            self.interpolators[control_name] = interpolate.RegularGridInterpolator(
                grids, control_values, method="linear", bounds_error=False, fill_value=None
            )

    def get_control(self, state: Dict[str, float], time: float = 0.0) -> Dict[str, Any]:
        """Get optimal control from HJB policy.

        Args:
            state: Current simulation state dictionary.
            time: Current time (may be included in state mapping).

        Returns:
            Dict[str, Any]: Optimal control parameters with keys
                'limits', 'retentions', and 'coverages'.

        Note:
            Uses linear interpolation of the HJB optimal policy
            between grid points.
        """
        # Map to HJB state space
        hjb_state = self.state_mapping(state)

        # Ensure state is a 1D array for interpolation
        if hjb_state.ndim > 1:
            # If it's already 2D (e.g., shape (1, 2)), flatten to 1D
            hjb_state = hjb_state.flatten()

        # Extract controls from interpolators
        controls = {}
        for name, interp in self.interpolators.items():
            # Pass state directly as 1D array - interpolator will handle it
            # Use item() to extract scalar from 0-d array to avoid deprecation warning
            result = interp(hjb_state)
            controls[name] = float(result.item() if hasattr(result, "item") else result)

        # Map back to insurance parameters format
        n_layers = len(self.control_space.limits)

        # Parse control names to extract limits and retentions
        limits = []
        retentions = []
        coverages = []

        for i in range(n_layers):
            # Look for controls named like "limit_0", "retention_0", etc.
            if f"limit_{i}" in controls:
                limits.append(controls[f"limit_{i}"])
            elif "limit" in controls and i == 0:
                limits.append(controls["limit"])
            else:
                # Use midpoint if not found
                limits.append(
                    (self.control_space.limits[i][0] + self.control_space.limits[i][1]) / 2
                )

            if f"retention_{i}" in controls:
                retentions.append(controls[f"retention_{i}"])
            elif "retention" in controls and i == 0:
                retentions.append(controls["retention"])
            else:
                retentions.append(
                    (self.control_space.retentions[i][0] + self.control_space.retentions[i][1]) / 2
                )

            if f"coverage_{i}" in controls:
                coverages.append(controls[f"coverage_{i}"])
            else:
                coverages.append(1.0)  # Default to full coverage

        return {"limits": limits, "retentions": retentions, "coverages": coverages}

    def update(self, state: Dict[str, float], outcome: Dict[str, float]):
        """No updates needed for HJB feedback control.

        Args:
            state: State where control was applied (ignored).
            outcome: Observed outcome (ignored).

        Note:
            HJB policy is precomputed and doesn't adapt online.
        """


class TimeVaryingControl(ControlStrategy):
    """Time-varying control strategy with predetermined schedule.

    This strategy adjusts insurance parameters according to a
    predetermined time schedule, useful for seasonal or cyclical risks.
    """

    def __init__(
        self,
        time_schedule: List[float],
        limits_schedule: List[List[float]],
        retentions_schedule: List[List[float]],
        coverages_schedule: Optional[List[List[float]]] = None,
    ):
        """Initialize time-varying control.

        Args:
            time_schedule: Time points where control parameters change.
            limits_schedule: Insurance limits at each time point,
                shape (n_times, n_layers).
            retentions_schedule: Retentions at each time point,
                shape (n_times, n_layers).
            coverages_schedule: Optional coverages at each time point,
                shape (n_times, n_layers). Defaults to full coverage.

        Raises:
            ValueError: If schedule lengths don't match.
        """
        self.time_schedule = np.array(time_schedule)
        self.limits_schedule = np.array(limits_schedule)
        self.retentions_schedule = np.array(retentions_schedule)

        if coverages_schedule:
            self.coverages_schedule = np.array(coverages_schedule)
        else:
            # Default to full coverage
            self.coverages_schedule = np.ones_like(self.limits_schedule)

        # Validate dimensions
        if len(self.time_schedule) != len(self.limits_schedule):
            raise ValueError("Schedule lengths must match")

        logger.info(f"Initialized time-varying control with {len(time_schedule)} time points")

    def get_control(self, state: Dict[str, float], time: float = 0.0) -> Dict[str, Any]:
        """Get control parameters for current time.

        Args:
            state: Current state (ignored for time-based control).
            time: Current simulation time.

        Returns:
            Dict[str, Any]: Control parameters interpolated linearly
                between scheduled time points.

        Note:
            Uses nearest value for times outside the schedule range.
        """
        # Find interpolation weights
        if time <= self.time_schedule[0]:
            # Before first time point
            limits = self.limits_schedule[0]
            retentions = self.retentions_schedule[0]
            coverages = self.coverages_schedule[0]
        elif time >= self.time_schedule[-1]:
            # After last time point
            limits = self.limits_schedule[-1]
            retentions = self.retentions_schedule[-1]
            coverages = self.coverages_schedule[-1]
        else:
            # Interpolate each layer separately
            n_layers = self.limits_schedule.shape[1]
            limits = np.zeros(n_layers)
            retentions = np.zeros(n_layers)
            coverages = np.zeros(n_layers)

            for i in range(n_layers):
                limits[i] = np.interp(time, self.time_schedule, self.limits_schedule[:, i])
                retentions[i] = np.interp(time, self.time_schedule, self.retentions_schedule[:, i])
                coverages[i] = np.interp(time, self.time_schedule, self.coverages_schedule[:, i])

        return {
            "limits": limits.tolist(),
            "retentions": retentions.tolist(),
            "coverages": coverages.tolist(),
        }

    def update(self, state: Dict[str, float], outcome: Dict[str, float]):
        """No updates for predetermined schedule.

        Args:
            state: State where control was applied (ignored).
            outcome: Observed outcome (ignored).
        """


class OptimalController:
    """Controller that applies optimal strategies in simulation.

    This class integrates control strategies with the simulation framework,
    managing the application of controls and tracking performance.
    """

    def __init__(self, strategy: ControlStrategy, control_space: ControlSpace):
        """Initialize optimal controller.

        Args:
            strategy: Control strategy to apply during simulation.
            control_space: Definition of feasible control space.

        Attributes:
            control_history: List of applied controls.
            state_history: List of states where controls were applied.
            outcome_history: List of observed outcomes.
        """
        self.strategy = strategy
        self.control_space = control_space

        # Performance tracking
        self.control_history: list[Dict[str, Any]] = []
        self.state_history: list[Dict[str, float]] = []
        self.outcome_history: list[Dict[str, float]] = []

        logger.info(f"Initialized optimal controller with {strategy.__class__.__name__}")

    def apply_control(
        self,
        manufacturer: WidgetManufacturer,
        state: Optional[Dict[str, float]] = None,
        time: float = 0.0,
    ) -> InsuranceProgram:
        """Apply control strategy to create insurance program.

        Args:
            manufacturer: Current manufacturer instance for extracting
                state if not provided.
            state: Optional state override. If None, state is extracted
                from manufacturer using _extract_state().
            time: Current simulation time.

        Returns:
            InsuranceProgram: Insurance program with layers configured
                according to the control strategy.

        Note:
            Records control and state in history for later analysis.
        """
        # Extract state if not provided
        if state is None:
            state = self._extract_state(manufacturer)

        # Get optimal control
        control = self.strategy.get_control(state, time)

        # Record for history
        self.control_history.append(control)
        self.state_history.append(state)

        # Create insurance program
        layers = []

        # Ensure lists
        limits = control["limits"] if isinstance(control["limits"], list) else [control["limits"]]
        retentions = (
            control["retentions"]
            if isinstance(control["retentions"], list)
            else [control["retentions"]]
        )
        coverages = control.get("coverages", [1.0] * len(limits))
        if not isinstance(coverages, list):
            coverages = [coverages]

        for _i, (limit, retention, coverage) in enumerate(zip(limits, retentions, coverages)):
            layer = EnhancedInsuranceLayer(
                attachment_point=retention,
                limit=limit * coverage,  # Apply coverage percentage to limit
                base_premium_rate=self._estimate_premium_rate(limit, retention, coverage),
            )
            layers.append(layer)

        # Create and return insurance program
        program = InsuranceProgram(
            layers=layers,
            deductible=min(retentions)
            if retentions
            else 0.0,  # Use minimum retention as deductible
        )

        return program

    def _extract_state(self, manufacturer: WidgetManufacturer) -> Dict[str, float]:
        """Extract state from manufacturer.

        Args:
            manufacturer: Manufacturer instance to extract state from.

        Returns:
            Dict[str, float]: State dictionary with keys including
                'assets', 'equity', 'wealth', 'revenue', etc.

        Note:
            Provides multiple naming conventions for compatibility
            with different control strategies.
        """
        metrics = manufacturer.calculate_metrics()

        return {
            "assets": float(manufacturer.total_assets),
            "equity": float(manufacturer.equity),
            "wealth": float(manufacturer.total_assets),  # Alternative naming
            "debt": 0.0,  # WidgetManufacturer doesn't track debt separately
            "revenue": float(metrics.get("revenue", 0)),
            "cumulative_losses": getattr(manufacturer, "cumulative_losses", 0),
            "time": getattr(manufacturer, "current_period", 0),
        }

    def _estimate_premium_rate(self, limit: float, retention: float, coverage: float) -> float:
        """Estimate fair premium rate for layer.

        Args:
            limit: Insurance limit in dollars.
            retention: Retention/deductible level in dollars.
            coverage: Coverage percentage (0 to 1).

        Returns:
            float: Estimated annual premium rate as percentage
                of insured value.

        Note:
            Uses simplified heuristic. Production systems would use
            actuarial models based on loss history and risk factors.
        """
        # Simple heuristic - would use actuarial model in production
        base_rate = 0.02  # 2% base rate

        # Adjust for retention (higher retention = lower rate)
        retention_factor = 1.0 - min(0.5, retention / 1e7)

        # Adjust for limit (higher limit = slightly higher rate)
        limit_factor = 1.0 + 0.1 * np.log10(max(1, limit / 1e6))

        # Adjust for coverage
        coverage_factor = coverage

        return float(base_rate * retention_factor * limit_factor * coverage_factor)

    def update_outcome(self, outcome: Dict[str, float]):
        """Update controller with observed outcome.

        Args:
            outcome: Observed outcome dictionary with keys like
                'losses', 'premium_costs', 'claim_payments', etc.

        Note:
            Calls strategy.update() if strategy is adaptive.
        """
        self.outcome_history.append(outcome)

        # Update strategy if it's adaptive
        if self.state_history:
            self.strategy.update(self.state_history[-1], outcome)

    def get_performance_summary(self) -> pd.DataFrame:
        """Get summary of controller performance.

        Returns:
            pd.DataFrame: DataFrame with columns for step number,
                state variables (prefixed with ``state_``),
                control variables (prefixed with ``control_``),
                and outcomes (prefixed with ``outcome_``).

        Note:
            Useful for analyzing control strategy effectiveness
            and creating visualizations.
        """
        if not self.control_history:
            return pd.DataFrame()

        # Combine histories into DataFrame
        records = []
        for i, (control, state) in enumerate(zip(self.control_history, self.state_history)):
            record: Dict[str, Any] = {"step": i}
            record.update({f"state_{k}": v for k, v in state.items()})
            record.update({f"control_{k}": v for k, v in control.items()})
            if i < len(self.outcome_history):
                record.update({f"outcome_{k}": v for k, v in self.outcome_history[i].items()})
            records.append(record)

        return pd.DataFrame(records)

    def reset(self):
        """Reset controller history.

        Clears all recorded history to prepare for new simulation run.
        """
        self.control_history = []
        self.state_history = []
        self.outcome_history = []


def create_hjb_controller(  # pylint: disable=too-many-locals
    manufacturer: WidgetManufacturer,
    simulation_years: int = 10,
    utility_type: str = "log",
    risk_aversion: float = 2.0,
) -> OptimalController:
    """Convenience function to create HJB-based controller.

    Creates and solves a simplified HJB problem for insurance optimization,
    then returns a controller configured with the optimal policy.

    Args:
        manufacturer: Manufacturer instance for extracting model parameters
            like growth rates and risk characteristics.
        simulation_years: Time horizon for optimization. Longer horizons
            may require more grid points for accuracy.
        utility_type: Type of utility function:
            - 'log': Logarithmic utility (Kelly criterion)
            - 'power': Power/CRRA utility with risk aversion
            - 'linear': Risk-neutral expected wealth
        risk_aversion: Coefficient of relative risk aversion for power
            utility. Higher values imply more conservative policies.
            Ignored for log and linear utilities.

    Returns:
        OptimalController: Controller with HJB feedback strategy configured
            for the specified problem.

    Raises:
        ValueError: If utility_type is not recognized.

    Example:
        >>> from ergodic_insurance.manufacturer import WidgetManufacturer
        >>> from ergodic_insurance.config import ManufacturerConfig
        >>>
        >>> # Set up manufacturer
        >>> config = ManufacturerConfig()
        >>> manufacturer = WidgetManufacturer(config)
        >>>
        >>> # Create HJB controller with power utility
        >>> controller = create_hjb_controller(
        ...     manufacturer,
        ...     simulation_years=10,
        ...     utility_type="power",
        ...     risk_aversion=2.0
        ... )
        >>>
        >>> # Apply control at current state
        >>> insurance = controller.apply_control(manufacturer, time=0)
        >>>
        >>> # Run simulation step
        >>> losses = manufacturer.generate_losses()
        >>> manufacturer.apply_losses(losses, insurance)
        >>>
        >>> # Update controller with outcome
        >>> outcome = {'losses': losses, 'premium': insurance.total_premium}
        >>> controller.update_outcome(outcome)

    Note:
        This function creates a simplified 2D state space (wealth, time)
        and single-layer insurance for demonstration. Production systems
        would use higher-dimensional state spaces and multiple layers.
    """
    from .hjb_solver import (
        ControlVariable,
        ExpectedWealth,
        HJBProblem,
        HJBSolverConfig,
        LogUtility,
        PowerUtility,
        StateSpace,
        StateVariable,
        UtilityFunction,
    )

    # Define state space (simplified 2D for demonstration)
    state_variables = [
        StateVariable(name="wealth", min_value=1e6, max_value=1e8, num_points=10, log_scale=True),
        StateVariable(
            name="time", min_value=0, max_value=simulation_years, num_points=5, log_scale=False
        ),
    ]
    state_space = StateSpace(state_variables)

    # Define control variables (single layer for simplicity)
    control_variables = [
        ControlVariable(name="limit", min_value=1e6, max_value=5e7, num_points=5),
        ControlVariable(name="retention", min_value=1e5, max_value=1e7, num_points=5),
    ]

    # Select utility function
    utility: UtilityFunction
    if utility_type == "log":
        utility = LogUtility()
    elif utility_type == "power":
        utility = PowerUtility(risk_aversion=risk_aversion)
    elif utility_type == "linear":
        utility = ExpectedWealth()
    else:
        raise ValueError(f"Unknown utility type: {utility_type}")

    # Define dynamics (simplified)
    def dynamics(state, control, time):
        """Wealth dynamics with insurance."""
        wealth = state[..., 0]
        _limit = control[..., 0]
        _retention = control[..., 1]

        # Expected growth rate (simplified)
        growth_rate = 0.08  # 8% baseline growth

        # Insurance cost reduces growth
        premium_rate = 0.02 * (_limit / 1e7)  # Simplified premium

        # Wealth drift
        wealth_drift = wealth * (growth_rate - premium_rate)

        # Time drift (always 1 since time moves forward)
        time_drift = np.ones_like(wealth)

        # Stack the drifts for both state variables
        result = np.stack([wealth_drift, time_drift], axis=-1)
        return result

    # Define running cost (negative for reward)
    def running_cost(state, control, time):
        """Running reward function."""
        wealth = state[..., 0]
        _limit = control[..., 0]

        # Reward is utility of wealth minus insurance cost
        reward = utility.evaluate(wealth)

        # The reward should have the same shape as wealth (number of state points)
        # which is state.shape[:-1] when state is (n_points, n_state_dims)
        return reward

    # Terminal value
    def terminal_value(state):
        """Terminal value function."""
        wealth = state[..., 0]
        return utility.evaluate(wealth)

    # Create HJB problem
    problem = HJBProblem(
        state_space=state_space,
        control_variables=control_variables,
        utility_function=utility,
        dynamics=dynamics,
        running_cost=running_cost,
        terminal_value=terminal_value,
        discount_rate=0.05,  # 5% discount rate
        time_horizon=simulation_years,
    )

    # Solve HJB equation
    config = HJBSolverConfig(time_step=0.1, max_iterations=10, tolerance=1e-3, verbose=False)

    solver = HJBSolver(problem, config)
    logger.info("Solving HJB equation...")
    solver.solve()

    # Create control space (single layer)
    control_space = ControlSpace(
        limits=[(1e6, 5e7)], retentions=[(1e5, 1e7)], coverage_percentages=[(0.9, 1.0)]
    )

    # Create feedback control strategy
    strategy = HJBFeedbackControl(solver, control_space)

    # Return configured controller
    return OptimalController(strategy, control_space)
