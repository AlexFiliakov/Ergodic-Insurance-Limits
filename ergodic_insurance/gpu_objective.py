"""GPU-accelerated objective evaluation for business optimization.

Implements GPU-batched evaluation of insurance optimization objectives so that
multiple parameter sets (coverage_limit, deductible, premium_rate) can be
scored in a single vectorized pass.  This is the core acceleration layer
for the ``BusinessOptimizer`` hot loop.

Depends on:
    - Issue #960 — GPU Backend Abstraction Layer (``gpu_backend`` module)
    - Issue #966 — GPU-accelerate optimization objective evaluation with batch MC

The module provides four public classes:

* ``GPUBatchObjective`` — vectorized objective functions (ROE, bankruptcy risk,
  growth rate, capital efficiency) over arrays of parameter sets.
* ``GPUObjectiveWrapper`` — thin wrapper that makes the batch evaluator
  compatible with ``scipy.optimize`` (single-point + cached evaluation,
  batched finite-difference gradients).
* ``GPUMultiStartScreener`` — screens a large set of starting points and
  returns the top-k by objective value using a single batch call.
* ``GPUDifferentialEvolution`` — GPU-native differential evolution optimizer
  that evaluates entire populations per generation via batch calls.

Since:
    Version 0.10.0 (Issue #966)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config.optimizer import BusinessOptimizerConfig
from .gpu_backend import (
    GPUConfig,
    get_array_module,
    is_gpu_available,
    set_random_seed,
    to_numpy,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  GPUBatchObjective
# ---------------------------------------------------------------------------


class GPUBatchObjective:
    """Vectorized evaluation of business-optimizer objectives on GPU or CPU.

    Evaluates multiple ``(coverage_limit, deductible, premium_rate)`` tuples
    simultaneously using array operations from CuPy (GPU) or NumPy (CPU
    fallback).  The financial formulas mirror the scalar implementations in
    ``BusinessOptimizer`` exactly, but operate on 1-D arrays of length
    *n_sets* instead of single floats.

    Args:
        equity: Company equity value (float, > 0).
        total_assets: Company total assets (float, > 0).
        revenue: Company revenue (float, > 0).
        optimizer_config: Pydantic model with optimizer heuristic parameters.
        gpu_config: Optional GPU configuration.  When ``None``, GPU is
            disabled and all operations use NumPy.
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        equity: float,
        total_assets: float,
        revenue: float,
        optimizer_config: BusinessOptimizerConfig,
        gpu_config: Optional[GPUConfig] = None,
    ) -> None:
        self.equity = float(equity)
        self.total_assets = float(total_assets)
        self.revenue = float(revenue)
        self.optimizer_config = optimizer_config
        self.gpu_config = gpu_config or GPUConfig()

        self.use_gpu: bool = self.gpu_config.enabled and is_gpu_available()
        self.xp = get_array_module(gpu=self.use_gpu)

        if self.use_gpu:
            logger.info("GPUBatchObjective: using GPU acceleration")
        else:
            logger.debug("GPUBatchObjective: using NumPy (CPU) backend")

    # -- helpers -----------------------------------------------------------

    def _deductible_ratio(self, deductible, coverage_limit):
        """Vectorized deductible-to-coverage ratio, clamped to [0, 1]."""
        xp = self.xp
        safe_limit = xp.where(coverage_limit > 0, coverage_limit, 1.0)
        ratio = xp.where(
            coverage_limit > 0,
            xp.minimum(deductible / safe_limit, 1.0),
            0.0,
        )
        return ratio

    def _unpack(self, param_sets):
        """Convert *param_sets* to device arrays and split columns."""
        xp = self.xp
        arr = xp.asarray(param_sets, dtype=xp.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        coverage_limit = arr[:, 0]
        deductible = arr[:, 1]
        premium_rate = arr[:, 2]
        return coverage_limit, deductible, premium_rate

    # -- public batch evaluators -------------------------------------------

    def evaluate_batch_roe(  # pylint: disable=unused-argument
        self,
        param_sets: np.ndarray,
        time_horizon: int,
        n_simulations: int = 100,
    ) -> np.ndarray:
        """Vectorized ROE simulation for all parameter sets at once.

        Mirrors ``BusinessOptimizer._simulate_roe`` but operates on arrays.
        Monte-Carlo noise is applied by broadcasting over the simulation
        axis and averaging.

        Args:
            param_sets: Array of shape ``(n_sets, 3)`` with columns
                ``[coverage_limit, deductible, premium_rate]``.
            time_horizon: Planning horizon in years (unused in the heuristic
                but accepted for API symmetry).
            n_simulations: Number of noisy ROE draws to average per set.

        Returns:
            1-D numpy array of shape ``(n_sets,)`` with mean adjusted ROE.
        """
        xp = self.xp
        cfg = self.optimizer_config

        coverage_limit, deductible, premium_rate = self._unpack(param_sets)
        n_sets = coverage_limit.shape[0]

        ded_ratio = self._deductible_ratio(deductible, coverage_limit)
        annual_premium = coverage_limit * premium_rate * (1.0 - ded_ratio)

        premium_cost = annual_premium / self.equity
        protection_benefit = cfg.protection_benefit_factor * (coverage_limit / self.total_assets)
        retained_loss_drag = (deductible / self.equity) * cfg.protection_benefit_factor

        adjusted_roe = cfg.base_roe - premium_cost + protection_benefit - retained_loss_drag

        # Monte-Carlo noise: shape (n_simulations, n_sets)
        seed = cfg.seed
        set_random_seed(seed, gpu=self.use_gpu)
        n_sims = min(n_simulations, 100)
        noise = xp.random.normal(1.0, cfg.roe_noise_std, size=(n_sims, n_sets))

        # Broadcast: (n_sims, n_sets) * (n_sets,) -> (n_sims, n_sets)
        simulated = noise * adjusted_roe[None, :]
        mean_roe = xp.mean(simulated, axis=0)

        return to_numpy(mean_roe)

    def evaluate_batch_bankruptcy_risk(
        self,
        param_sets: np.ndarray,
        time_horizon: int,
    ) -> np.ndarray:
        """Vectorized bankruptcy-risk estimation for all parameter sets.

        Mirrors ``BusinessOptimizer._estimate_bankruptcy_risk``.

        Args:
            param_sets: Array of shape ``(n_sets, 3)`` with columns
                ``[coverage_limit, deductible, premium_rate]``.
            time_horizon: Planning horizon in years.

        Returns:
            1-D numpy array of shape ``(n_sets,)`` with bankruptcy
            probabilities clamped to [0, 1].
        """
        xp = self.xp
        cfg = self.optimizer_config

        coverage_limit, deductible, premium_rate = self._unpack(param_sets)

        ded_ratio = self._deductible_ratio(deductible, coverage_limit)
        annual_premium = coverage_limit * premium_rate * (1.0 - ded_ratio)

        coverage_ratio = coverage_limit / self.total_assets
        effective_coverage_ratio = coverage_ratio * (1.0 - ded_ratio)

        risk_reduction = xp.minimum(
            effective_coverage_ratio * cfg.max_risk_reduction,
            cfg.max_risk_reduction,
        )

        retained_risk = (deductible / self.total_assets) * cfg.max_risk_reduction

        premium_burden = annual_premium / self.revenue
        risk_increase = premium_burden * cfg.premium_burden_risk_factor

        time_factor = 1.0 - xp.exp(-time_horizon / cfg.time_risk_constant)

        bankruptcy_risk = (
            cfg.base_bankruptcy_risk - risk_reduction + retained_risk + risk_increase
        ) * time_factor

        bankruptcy_risk = xp.clip(bankruptcy_risk, 0.0, 1.0)
        return to_numpy(bankruptcy_risk)

    def evaluate_batch_growth_rate(  # pylint: disable=unused-argument
        self,
        param_sets: np.ndarray,
        time_horizon: int,
        metric: str = "revenue",
    ) -> np.ndarray:
        """Vectorized growth-rate estimation for all parameter sets.

        Mirrors ``BusinessOptimizer._estimate_growth_rate``.

        Args:
            param_sets: Array of shape ``(n_sets, 3)`` with columns
                ``[coverage_limit, deductible, premium_rate]``.
            time_horizon: Planning horizon in years (unused in the heuristic
                but accepted for API symmetry).
            metric: Growth metric — ``"revenue"`` (default), ``"assets"``,
                or ``"equity"``.

        Returns:
            1-D numpy array of shape ``(n_sets,)`` with non-negative growth
            rates.
        """
        xp = self.xp
        cfg = self.optimizer_config

        coverage_limit, deductible, premium_rate = self._unpack(param_sets)

        ded_ratio = self._deductible_ratio(deductible, coverage_limit)
        annual_premium = coverage_limit * premium_rate * (1.0 - ded_ratio)

        coverage_ratio = coverage_limit / self.total_assets
        effective_coverage_ratio = coverage_ratio * (1.0 - ded_ratio)

        growth_boost = effective_coverage_ratio * cfg.growth_boost_factor
        premium_drag = (annual_premium / self.revenue) * cfg.premium_drag_factor
        retained_risk_drag = (deductible / self.total_assets) * cfg.growth_boost_factor

        adjusted_growth = cfg.base_growth_rate + growth_boost - premium_drag - retained_risk_drag

        if metric == "assets":
            adjusted_growth = adjusted_growth * cfg.asset_growth_factor
        elif metric == "equity":
            adjusted_growth = adjusted_growth * cfg.equity_growth_factor

        adjusted_growth = xp.maximum(adjusted_growth, 0.0)
        return to_numpy(adjusted_growth)

    def evaluate_batch_capital_efficiency(
        self,
        param_sets: np.ndarray,
    ) -> np.ndarray:
        """Vectorized capital-efficiency calculation for all parameter sets.

        Mirrors ``BusinessOptimizer._calculate_capital_efficiency``.

        Args:
            param_sets: Array of shape ``(n_sets, 3)`` with columns
                ``[coverage_limit, deductible, premium_rate]``.

        Returns:
            1-D numpy array of shape ``(n_sets,)`` with non-negative
            efficiency ratios.
        """
        xp = self.xp
        cfg = self.optimizer_config

        coverage_limit, deductible, premium_rate = self._unpack(param_sets)

        ded_ratio = self._deductible_ratio(deductible, coverage_limit)
        annual_premium = coverage_limit * premium_rate * (1.0 - ded_ratio)

        risk_transfer_benefit = coverage_limit * (1.0 - ded_ratio) * cfg.risk_transfer_benefit_rate

        net_benefit = risk_transfer_benefit - annual_premium
        efficiency_ratio = 1.0 + (net_benefit / self.total_assets)

        efficiency_ratio = xp.maximum(efficiency_ratio, 0.0)
        return to_numpy(efficiency_ratio)


# ---------------------------------------------------------------------------
#  GPUObjectiveWrapper
# ---------------------------------------------------------------------------


class GPUObjectiveWrapper:
    """Wrap :class:`GPUBatchObjective` for ``scipy.optimize`` compatibility.

    Provides a ``__call__`` interface for single-point evaluation (with
    optional caching) and a ``gradient`` method that computes central
    finite-difference gradients in a single batched evaluation.

    Sign convention:
        * *Maximization* objectives (ROE, growth_rate, capital_efficiency)
          return the **negated** value from ``__call__`` so that
          ``scipy.optimize.minimize`` maximizes them.
        * *Minimization* objectives (bankruptcy_risk) return the raw value.

    Args:
        batch_objective: A :class:`GPUBatchObjective` instance.
        objective_name: One of ``"roe"``, ``"bankruptcy_risk"``,
            ``"growth_rate"``, ``"capital_efficiency"``.
        time_horizon: Planning horizon in years.
        n_simulations: Number of MC simulations (ROE only).
        cache_enabled: Whether to cache evaluation results.
    """

    # Objectives where higher is better — negate for scipy.minimize
    _MAXIMIZE_OBJECTIVES = {"roe", "growth_rate", "capital_efficiency"}

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        batch_objective: GPUBatchObjective,
        objective_name: str = "roe",
        time_horizon: int = 10,
        n_simulations: int = 100,
        cache_enabled: bool = True,
    ) -> None:
        self.batch_objective = batch_objective
        self.objective_name = objective_name.lower()
        self.time_horizon = time_horizon
        self.n_simulations = n_simulations
        self.cache_enabled = cache_enabled

        self._cache: Dict[Tuple[float, ...], float] = {}

    # -- evaluation dispatch -----------------------------------------------

    def _evaluate_batch(self, param_sets: np.ndarray) -> np.ndarray:
        """Dispatch to the appropriate batch evaluator."""
        name = self.objective_name
        bo = self.batch_objective
        if name == "roe":
            return bo.evaluate_batch_roe(param_sets, self.time_horizon, self.n_simulations)
        if name == "bankruptcy_risk":
            return bo.evaluate_batch_bankruptcy_risk(param_sets, self.time_horizon)
        if name == "growth_rate":
            return bo.evaluate_batch_growth_rate(param_sets, self.time_horizon)
        if name == "capital_efficiency":
            return bo.evaluate_batch_capital_efficiency(param_sets)
        raise ValueError(f"Unknown objective: {name!r}")

    # -- public interface --------------------------------------------------

    def evaluate_single(self, x: np.ndarray) -> float:
        """Evaluate objective without sign flip (raw value).

        Args:
            x: 1-D array ``[coverage_limit, deductible, premium_rate]``.

        Returns:
            The raw objective value (not negated).
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        key = tuple(x.tolist())
        if self.cache_enabled and key in self._cache:
            return self._cache[key]
        result = float(self._evaluate_batch(x.reshape(1, -1))[0])
        if self.cache_enabled:
            self._cache[key] = result
        return result

    def __call__(self, x: np.ndarray) -> float:
        """Single-point evaluation for ``scipy.optimize``.

        Returns a value suitable for *minimization*: negated for
        maximization objectives, raw for minimization objectives.

        Args:
            x: 1-D array ``[coverage_limit, deductible, premium_rate]``.

        Returns:
            Scalar objective value (negated for maximization objectives).
        """
        raw = self.evaluate_single(x)
        if self.objective_name in self._MAXIMIZE_OBJECTIVES:
            return -raw
        return raw

    def gradient(self, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
        """Batched central finite-difference gradient.

        Creates ``2 * n_params`` perturbation points and evaluates them
        all in a single batch call through the GPU evaluator.

        Args:
            x: 1-D array ``[coverage_limit, deductible, premium_rate]``.
            h: Perturbation step size (absolute).

        Returns:
            1-D numpy array of shape ``(n_params,)`` with the gradient.
            The sign convention matches ``__call__``.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        n_params = x.shape[0]

        # Build perturbation matrix: 2*n_params rows
        perturbed = np.tile(x, (2 * n_params, 1))
        for i in range(n_params):
            perturbed[2 * i, i] += h
            perturbed[2 * i + 1, i] -= h

        raw_vals = self._evaluate_batch(perturbed)

        # Apply sign convention
        if self.objective_name in self._MAXIMIZE_OBJECTIVES:
            raw_vals = -raw_vals

        grad = np.empty(n_params, dtype=np.float64)
        for i in range(n_params):
            grad[i] = (raw_vals[2 * i] - raw_vals[2 * i + 1]) / (2.0 * h)

        return grad

    def clear_cache(self) -> None:
        """Clear the evaluation cache."""
        self._cache.clear()


# ---------------------------------------------------------------------------
#  GPUMultiStartScreener
# ---------------------------------------------------------------------------


class GPUMultiStartScreener:  # pylint: disable=too-few-public-methods
    """Screen starting points for multi-start optimization using batch GPU evaluation.

    Evaluates all candidate starting points in a single batch call and
    returns the top-k points sorted by objective value.

    Args:
        batch_objective: A :class:`GPUBatchObjective` instance.
        objective_name: One of ``"roe"``, ``"bankruptcy_risk"``,
            ``"growth_rate"``, ``"capital_efficiency"``.
        time_horizon: Planning horizon in years.
        n_simulations: Number of MC simulations (ROE only).
    """

    _MAXIMIZE_OBJECTIVES = {"roe", "growth_rate", "capital_efficiency"}

    def __init__(
        self,
        batch_objective: GPUBatchObjective,
        objective_name: str = "roe",
        time_horizon: int = 10,
        n_simulations: int = 100,
    ) -> None:
        self.batch_objective = batch_objective
        self.objective_name = objective_name.lower()
        self.time_horizon = time_horizon
        self.n_simulations = n_simulations

    def _evaluate_batch(self, param_sets: np.ndarray) -> np.ndarray:
        name = self.objective_name
        bo = self.batch_objective
        if name == "roe":
            return bo.evaluate_batch_roe(param_sets, self.time_horizon, self.n_simulations)
        if name == "bankruptcy_risk":
            return bo.evaluate_batch_bankruptcy_risk(param_sets, self.time_horizon)
        if name == "growth_rate":
            return bo.evaluate_batch_growth_rate(param_sets, self.time_horizon)
        if name == "capital_efficiency":
            return bo.evaluate_batch_capital_efficiency(param_sets)
        raise ValueError(f"Unknown objective: {name!r}")

    def screen_starting_points(
        self,
        starting_points: np.ndarray,
        top_k: int = 5,
    ) -> List[np.ndarray]:
        """Evaluate all starting points in one batch and return the best.

        Args:
            starting_points: Array of shape ``(n_points, 3)``.
            top_k: Number of best starting points to return.

        Returns:
            List of 1-D numpy arrays (length 3), sorted best-first.
            For maximization objectives, "best" = highest value.
            For minimization objectives, "best" = lowest value.
        """
        starting_points = np.asarray(starting_points, dtype=np.float64)
        if starting_points.ndim == 1:
            starting_points = starting_points.reshape(1, -1)

        values = self._evaluate_batch(starting_points)

        # Sort: descending for maximize, ascending for minimize
        if self.objective_name in self._MAXIMIZE_OBJECTIVES:
            order = np.argsort(-values)
        else:
            order = np.argsort(values)

        top_k = min(top_k, len(order))
        return [starting_points[i].copy() for i in order[:top_k]]


# ---------------------------------------------------------------------------
#  GPUDifferentialEvolution
# ---------------------------------------------------------------------------


@dataclass
class _DEResult:
    """Minimal result object compatible with ``scipy.optimize.OptimizeResult``."""

    x: np.ndarray
    fun: float
    nit: int
    nfev: int
    success: bool
    message: str
    population: np.ndarray = field(default_factory=lambda: np.empty(0))
    population_values: np.ndarray = field(default_factory=lambda: np.empty(0))


class GPUDifferentialEvolution:
    """GPU-native differential evolution optimizer.

    Evaluates the entire population each generation using
    :class:`GPUBatchObjective`, so the dominant cost is one batch kernel
    launch per generation instead of ``pop_size`` sequential evaluations.

    The optimizer uses the ``DE/rand/1/bin`` strategy: for each member of
    the population a mutant is created from three distinct random members,
    crossover is applied, and the trial replaces the target if it has a
    better (lower) objective value.

    Sign convention: internally the optimizer always *minimizes*.  For
    maximization objectives the raw values are negated before comparison.

    Args:
        batch_objective: A :class:`GPUBatchObjective` instance.
        bounds: Sequence of ``(low, high)`` pairs — one per parameter.
        objective_name: One of ``"roe"``, ``"bankruptcy_risk"``,
            ``"growth_rate"``, ``"capital_efficiency"``.
        time_horizon: Planning horizon in years.
        n_simulations: Number of MC simulations (ROE only).
        seed: Optional RNG seed for reproducibility.
    """

    _MAXIMIZE_OBJECTIVES = {"roe", "growth_rate", "capital_efficiency"}

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        batch_objective: GPUBatchObjective,
        bounds: List[Tuple[float, float]],
        objective_name: str = "roe",
        time_horizon: int = 10,
        n_simulations: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        self.batch_objective = batch_objective
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.objective_name = objective_name.lower()
        self.time_horizon = time_horizon
        self.n_simulations = n_simulations
        self.seed = seed

    # -- helpers -----------------------------------------------------------

    def _evaluate_batch(self, param_sets: np.ndarray) -> np.ndarray:
        name = self.objective_name
        bo = self.batch_objective
        if name == "roe":
            return bo.evaluate_batch_roe(param_sets, self.time_horizon, self.n_simulations)
        if name == "bankruptcy_risk":
            return bo.evaluate_batch_bankruptcy_risk(param_sets, self.time_horizon)
        if name == "growth_rate":
            return bo.evaluate_batch_growth_rate(param_sets, self.time_horizon)
        if name == "capital_efficiency":
            return bo.evaluate_batch_capital_efficiency(param_sets)
        raise ValueError(f"Unknown objective: {name!r}")

    def _to_minimization(self, values: np.ndarray) -> np.ndarray:
        if self.objective_name in self._MAXIMIZE_OBJECTIVES:
            return -values
        return values

    # -- public interface --------------------------------------------------

    def optimize(
        self,
        pop_size: int = 50,
        n_generations: int = 100,
        mutation_factor: float = 0.8,
        crossover_prob: float = 0.9,
    ) -> _DEResult:
        """Run differential evolution with batch GPU evaluation.

        Args:
            pop_size: Population size.
            n_generations: Number of generations.
            mutation_factor: DE mutation scale factor ``F`` in (0, 2].
            crossover_prob: Crossover probability ``CR`` in [0, 1].

        Returns:
            A result object with fields ``x``, ``fun``, ``nit``, ``nfev``,
            ``success``, ``message``, ``population``, and
            ``population_values``.
        """
        rng = np.random.default_rng(self.seed)
        n_params = self.bounds.shape[0]
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]

        # Initialize population uniformly within bounds
        population = rng.uniform(lower, upper, size=(pop_size, n_params))

        # Evaluate initial population
        raw_values = self._evaluate_batch(population)
        fitness = self._to_minimization(raw_values)
        nfev = pop_size

        best_idx = int(np.argmin(fitness))
        best_x = population[best_idx].copy()
        best_fit = fitness[best_idx]

        for _gen in range(n_generations):
            # --- Mutation: DE/rand/1 ---
            # For each member, pick three distinct others
            indices = np.arange(pop_size)
            # Generate random triplets (avoiding self)
            r1 = np.empty(pop_size, dtype=int)
            r2 = np.empty(pop_size, dtype=int)
            r3 = np.empty(pop_size, dtype=int)
            for i in range(pop_size):
                choices = rng.choice(np.delete(indices, i), size=3, replace=False)
                r1[i], r2[i], r3[i] = choices

            mutant = population[r1] + mutation_factor * (population[r2] - population[r3])

            # Clip mutant to bounds
            mutant = np.clip(mutant, lower, upper)

            # --- Crossover: binomial ---
            cross_mask = rng.random(size=(pop_size, n_params)) < crossover_prob
            # Ensure at least one parameter is crossed over
            j_rand = rng.integers(0, n_params, size=pop_size)
            for i in range(pop_size):
                cross_mask[i, j_rand[i]] = True

            trial = np.where(cross_mask, mutant, population)

            # --- Selection ---
            trial_raw = self._evaluate_batch(trial)
            trial_fitness = self._to_minimization(trial_raw)
            nfev += pop_size

            improved = trial_fitness < fitness
            population[improved] = trial[improved]
            fitness[improved] = trial_fitness[improved]

            # Track best
            gen_best_idx = int(np.argmin(fitness))
            if fitness[gen_best_idx] < best_fit:
                best_fit = fitness[gen_best_idx]
                best_x = population[gen_best_idx].copy()

        # Convert best fitness back to raw objective value
        if self.objective_name in self._MAXIMIZE_OBJECTIVES:
            best_raw = -best_fit
        else:
            best_raw = best_fit

        # Final population raw values
        final_raw = self._evaluate_batch(population)

        return _DEResult(
            x=best_x,
            fun=float(best_raw),
            nit=n_generations,
            nfev=nfev,
            success=True,
            message="Differential evolution converged",
            population=population.copy(),
            population_values=final_raw,
        )
