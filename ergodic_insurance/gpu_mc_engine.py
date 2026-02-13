"""GPU-accelerated Monte Carlo simulation engine.

Provides a vectorized, reduced-form financial model that processes ALL
simulation paths simultaneously using array operations.  When CuPy is
available the arrays live on the GPU; otherwise plain NumPy is used with
the same code paths.

The GPU path uses a **simplified but statistically equivalent** model
compared to the full OOP manufacturer/ledger simulation:

* No Decimal accounting, no ledger, no depreciation, no LoC collateral
* Revenue, operating income, tax, and retained earnings expressed as
  array ops on ``(n_sims,)`` vectors
* Insurance applied per-occurrence with vectorized layer logic

Since:
    Version 0.11.0 (Issue #961)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .gpu_backend import get_array_module, gpu_memory_pool, is_gpu_available, to_numpy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  GPUSimulationParams — flat scalar container
# ---------------------------------------------------------------------------


@dataclass
class GPUSimulationParams:
    """Flat scalar parameters extracted from OOP model objects.

    This dataclass holds every numeric knob needed by the vectorized
    simulation so that the hot loop never touches Python objects.

    Attributes:
        initial_assets: Starting total assets of the manufacturer.
        asset_turnover_ratio: Revenue per dollar of assets.
        base_operating_margin: EBIT / Revenue ratio.
        tax_rate: Corporate tax rate.
        retention_ratio: Fraction of net income retained.
        deductible: Insurance program self-insured retention.
        layer_attachments: Per-layer attachment points.
        layer_limits: Per-layer coverage limits.
        layer_limit_types: Per-layer limit type (0=per-occurrence, 1=aggregate, 2=hybrid).
        layer_aggregate_limits: Per-layer annual aggregate caps (inf if none).
        base_annual_premium: Total annual premium across all layers.
        attritional_base_freq: Attritional loss base frequency.
        attritional_scaling_exp: Revenue-scaling exponent for attritional frequency.
        attritional_ref_revenue: Reference revenue for attritional scaling.
        attritional_mu: Lognormal mu for attritional severity.
        attritional_sigma: Lognormal sigma for attritional severity.
        large_base_freq: Large loss base frequency.
        large_scaling_exp: Revenue-scaling exponent for large frequency.
        large_ref_revenue: Reference revenue for large scaling.
        large_mu: Lognormal mu for large severity.
        large_sigma: Lognormal sigma for large severity.
        cat_base_freq: Catastrophic loss base frequency.
        cat_scaling_exp: Revenue-scaling exponent for catastrophic frequency.
        cat_ref_revenue: Reference revenue for catastrophic scaling.
        cat_alpha: Pareto alpha (shape) for catastrophic severity.
        cat_xm: Pareto xm (scale/minimum) for catastrophic severity.
        n_simulations: Number of simulation paths.
        n_years: Number of years to simulate.
        insolvency_tolerance: Equity threshold for ruin.
        letter_of_credit_rate: Annual LoC rate (unused in simplified model, kept for parity).
        growth_rate: Exogenous revenue growth rate per year.
        use_float32: Whether to use float32 for arrays.
        seed: Random seed for reproducibility.
    """

    # Manufacturer
    initial_assets: float = 10_000_000.0
    asset_turnover_ratio: float = 1.0
    base_operating_margin: float = 0.10
    tax_rate: float = 0.21
    retention_ratio: float = 0.80

    # Insurance layers
    deductible: float = 0.0
    layer_attachments: List[float] = field(default_factory=list)
    layer_limits: List[float] = field(default_factory=list)
    layer_limit_types: List[int] = field(default_factory=list)  # 0=per-occ, 1=agg, 2=hybrid
    layer_aggregate_limits: List[float] = field(default_factory=list)
    base_annual_premium: float = 0.0

    # Loss generators — attritional (lognormal)
    attritional_base_freq: float = 3.0
    attritional_scaling_exp: float = 0.7
    attritional_ref_revenue: float = 10_000_000.0
    attritional_mu: float = 10.0
    attritional_sigma: float = 1.5

    # Loss generators — large (lognormal)
    large_base_freq: float = 0.3
    large_scaling_exp: float = 0.5
    large_ref_revenue: float = 10_000_000.0
    large_mu: float = 13.0
    large_sigma: float = 1.0

    # Loss generators — catastrophic (Pareto)
    cat_base_freq: float = 0.05
    cat_scaling_exp: float = 0.0
    cat_ref_revenue: float = 10_000_000.0
    cat_alpha: float = 1.5
    cat_xm: float = 1_000_000.0

    # Engine
    n_simulations: int = 10_000
    n_years: int = 10
    insolvency_tolerance: float = 10_000.0
    letter_of_credit_rate: float = 0.015
    growth_rate: float = 0.0
    use_float32: bool = False
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
#  Parameter extraction
# ---------------------------------------------------------------------------


def extract_params(
    manufacturer,
    insurance_program,
    loss_generator,
    mc_config,
) -> GPUSimulationParams:
    """Extract flat parameters from OOP model objects.

    Args:
        manufacturer: A :class:`WidgetManufacturer` instance.
        insurance_program: An :class:`InsuranceProgram` instance.
        loss_generator: A :class:`ManufacturingLossGenerator` instance.
        mc_config: A :class:`MonteCarloConfig` instance.

    Returns:
        A populated :class:`GPUSimulationParams`.
    """
    cfg = manufacturer.config

    # --- Insurance layers ---
    attachments: List[float] = []
    limits: List[float] = []
    limit_types: List[int] = []
    agg_limits: List[float] = []

    _LIMIT_TYPE_MAP = {"per-occurrence": 0, "aggregate": 1, "hybrid": 2}

    for layer in insurance_program.layers:
        attachments.append(float(layer.attachment_point))
        limits.append(float(layer.limit))
        lt = getattr(layer, "limit_type", "per-occurrence")
        limit_types.append(_LIMIT_TYPE_MAP.get(lt, 0))
        agg = getattr(layer, "aggregate_limit", None)
        agg_limits.append(float(agg) if agg is not None else float("inf"))

    base_premium = float(insurance_program.calculate_premium())

    # --- Loss generators ---
    def _freq_params(gen):
        fg = gen.frequency_generator
        return (
            float(fg.base_frequency),
            float(fg.revenue_scaling_exponent),
            float(fg.reference_revenue),
        )

    att_freq, att_exp, att_ref = _freq_params(loss_generator.attritional)
    lg_freq, lg_exp, lg_ref = _freq_params(loss_generator.large)
    cat_freq, cat_exp, cat_ref = _freq_params(loss_generator.catastrophic)

    att_sev = loss_generator.attritional.severity_distribution
    lg_sev = loss_generator.large.severity_distribution
    cat_sev = loss_generator.catastrophic.severity_distribution

    return GPUSimulationParams(
        # Manufacturer
        initial_assets=float(cfg.initial_assets),
        asset_turnover_ratio=float(cfg.asset_turnover_ratio),
        base_operating_margin=float(cfg.base_operating_margin),
        tax_rate=float(cfg.tax_rate),
        retention_ratio=float(cfg.retention_ratio),
        # Insurance
        deductible=float(insurance_program.deductible),
        layer_attachments=attachments,
        layer_limits=limits,
        layer_limit_types=limit_types,
        layer_aggregate_limits=agg_limits,
        base_annual_premium=base_premium,
        # Attritional
        attritional_base_freq=att_freq,
        attritional_scaling_exp=att_exp,
        attritional_ref_revenue=att_ref,
        attritional_mu=float(att_sev.mu),
        attritional_sigma=float(att_sev.sigma),
        # Large
        large_base_freq=lg_freq,
        large_scaling_exp=lg_exp,
        large_ref_revenue=lg_ref,
        large_mu=float(lg_sev.mu),
        large_sigma=float(lg_sev.sigma),
        # Catastrophic
        cat_base_freq=cat_freq,
        cat_scaling_exp=cat_exp,
        cat_ref_revenue=cat_ref,
        cat_alpha=float(cat_sev.alpha),
        cat_xm=float(cat_sev.xm),
        # Engine
        n_simulations=mc_config.n_simulations,
        n_years=mc_config.n_years,
        insolvency_tolerance=mc_config.insolvency_tolerance,
        letter_of_credit_rate=mc_config.letter_of_credit_rate,
        growth_rate=mc_config.growth_rate,
        use_float32=mc_config.use_float32,
        seed=mc_config.seed,
    )


# ---------------------------------------------------------------------------
#  Loss generation (vectorized)
# ---------------------------------------------------------------------------

# Hard cap on event slots per simulation-year.  Poisson means rarely
# exceed ~30 even for aggressive frequencies, so 64 is generous.
_MAX_EVENTS_PER_YEAR = 64


def _scatter_events(
    loss_amounts_cpu: np.ndarray,
    att_counts: np.ndarray,
    lg_counts: np.ndarray,
    cat_counts: np.ndarray,
    att_sevs: np.ndarray,
    lg_sevs: np.ndarray,
    cat_sevs: np.ndarray,
    max_events: int,
) -> None:
    """Scatter per-generator severities into the padded (n_sims, max_events) array."""
    n_sims = loss_amounts_cpu.shape[0]
    att_ptr = 0
    lg_ptr = 0
    cat_ptr = 0

    for i in range(n_sims):
        col = 0
        for counts, sevs_arr, ptr_name in (
            (att_counts[i], att_sevs, "att"),
            (lg_counts[i], lg_sevs, "lg"),
            (cat_counts[i], cat_sevs, "cat"),
        ):
            if counts > 0:
                end = min(col + counts, max_events)
                count = end - col
                ptr = {"att": att_ptr, "lg": lg_ptr, "cat": cat_ptr}[ptr_name]
                loss_amounts_cpu[i, col:end] = sevs_arr[ptr : ptr + count]
                if ptr_name == "att":
                    att_ptr += counts
                elif ptr_name == "lg":
                    lg_ptr += counts
                else:
                    cat_ptr += counts
                col = end


def generate_losses_for_year(
    params: GPUSimulationParams,
    revenues: Any,
    xp: Any,
    rng: np.random.Generator,
    dtype: Any,
) -> tuple:
    """Generate per-event loss amounts for one simulated year.

    All operations are vectorized across *n_sims* paths.

    Args:
        params: Flat simulation parameters.
        revenues: ``(n_sims,)`` array of current-year revenues.
        xp: Array module (``numpy`` or ``cupy``).
        rng: NumPy random generator (used even for GPU — counts are
            sampled on CPU then transferred).
        dtype: Numpy dtype for arrays.

    Returns:
        ``(loss_amounts, n_events)`` where *loss_amounts* has shape
        ``(n_sims, max_events)`` and *n_events* is ``(n_sims,)``.
    """
    n_sims = revenues.shape[0]
    # We need revenues on CPU for Poisson sampling
    rev_cpu = to_numpy(revenues).astype(np.float64)

    # ------ sample event counts per generator type ------
    def _scaled_freq(base_freq, scaling_exp, ref_revenue):
        safe_rev = np.maximum(rev_cpu, 1.0)
        return base_freq * (safe_rev / ref_revenue) ** scaling_exp

    att_freq = _scaled_freq(
        params.attritional_base_freq, params.attritional_scaling_exp, params.attritional_ref_revenue
    )
    lg_freq = _scaled_freq(
        params.large_base_freq, params.large_scaling_exp, params.large_ref_revenue
    )
    cat_freq = _scaled_freq(params.cat_base_freq, params.cat_scaling_exp, params.cat_ref_revenue)

    att_counts = rng.poisson(att_freq).astype(np.int32)
    lg_counts = rng.poisson(lg_freq).astype(np.int32)
    cat_counts = rng.poisson(cat_freq).astype(np.int32)

    total_counts = att_counts + lg_counts + cat_counts
    max_events = min(
        int(np.max(total_counts)) if total_counts.size > 0 else 0, _MAX_EVENTS_PER_YEAR
    )

    # Edge case: no events at all
    if max_events == 0:
        loss_amounts = xp.zeros((n_sims, 1), dtype=dtype)
        n_events = xp.zeros(n_sims, dtype=xp.int32)
        return loss_amounts, n_events

    # ------ sample severities in bulk on CPU then transfer ------
    total_att = int(np.sum(att_counts))
    total_lg = int(np.sum(lg_counts))
    total_cat = int(np.sum(cat_counts))

    # Attritional — lognormal
    att_sevs = (
        rng.lognormal(params.attritional_mu, params.attritional_sigma, size=total_att).astype(
            np.float64
        )
        if total_att > 0
        else np.empty(0, dtype=np.float64)
    )

    # Large — lognormal
    lg_sevs = (
        rng.lognormal(params.large_mu, params.large_sigma, size=total_lg).astype(np.float64)
        if total_lg > 0
        else np.empty(0, dtype=np.float64)
    )

    # Catastrophic — Pareto via inverse transform: X = xm * U^(-1/alpha)
    if total_cat > 0:
        u = rng.uniform(0.0, 1.0, size=total_cat)
        cat_sevs = (params.cat_xm * u ** (-1.0 / params.cat_alpha)).astype(np.float64)
    else:
        cat_sevs = np.empty(0, dtype=np.float64)

    # ------ scatter into (n_sims, max_events) padded array ------
    loss_amounts_cpu = np.zeros((n_sims, max_events), dtype=np.float64)
    _scatter_events(
        loss_amounts_cpu, att_counts, lg_counts, cat_counts, att_sevs, lg_sevs, cat_sevs, max_events
    )

    # Clip total counts to max_events
    n_events_cpu = np.minimum(total_counts, max_events).astype(np.int32)

    # Transfer to device
    loss_amounts_dev = xp.asarray(loss_amounts_cpu.astype(dtype))
    n_events_dev = xp.asarray(n_events_cpu)

    return loss_amounts_dev, n_events_dev


# ---------------------------------------------------------------------------
#  Insurance application (vectorized)
# ---------------------------------------------------------------------------


def apply_insurance_vectorized(
    loss_amounts: Any,
    n_events: Any,
    params: GPUSimulationParams,
    xp: Any,
    dtype: Any,
) -> tuple:
    """Apply insurance layers to per-event losses, vectorized across sims.

    Handles per-occurrence layers fully vectorized and aggregate layers
    with a small loop over events (typically < 15 iterations) that is
    vectorized across all simulations.

    Args:
        loss_amounts: ``(n_sims, max_events)`` padded loss array.
        n_events: ``(n_sims,)`` actual event count per sim.
        params: Flat simulation parameters.
        xp: Array module.
        dtype: Array dtype.

    Returns:
        ``(total_losses, recoveries, retained)`` each ``(n_sims,)``.
    """
    n_sims = loss_amounts.shape[0]
    max_events = loss_amounts.shape[1]

    # Event mask: (n_sims, max_events)
    event_idx = xp.arange(max_events, dtype=xp.int32)
    event_mask = event_idx[None, :] < n_events[:, None]

    # Ensure padding zeros remain zero
    masked_losses = loss_amounts * event_mask

    # Total loss per sim
    total_losses = xp.sum(masked_losses, axis=1)

    n_layers = len(params.layer_attachments)
    if n_layers == 0:
        # No insurance layers — everything retained above deductible
        per_event_ded = xp.asarray(params.deductible, dtype=dtype)
        recovery_per_event = xp.minimum(masked_losses, per_event_ded) * 0.0  # zero recovery
        total_recoveries = xp.zeros(n_sims, dtype=dtype)
        total_retained = total_losses
        return total_losses, total_recoveries, total_retained

    # Convert layer params to device arrays
    attachments = xp.asarray(params.layer_attachments, dtype=dtype)
    limits = xp.asarray(params.layer_limits, dtype=dtype)
    limit_types = params.layer_limit_types
    agg_limits_list = params.layer_aggregate_limits

    # Identify which layers are aggregate-limited
    has_aggregate = [
        lt in (1, 2) or agg < float("inf") for lt, agg in zip(limit_types, agg_limits_list)
    ]
    any_aggregate = any(has_aggregate)

    if not any_aggregate:
        # ---- Fast path: all per-occurrence, fully vectorized ----
        # loss_amounts: (n_sims, max_events) → expand for layers
        # attachments: (n_layers,) → (1, 1, n_layers)
        losses_3d = masked_losses[:, :, None]  # (n_sims, max_events, 1)
        att_3d = attachments[None, None, :]  # (1, 1, n_layers)
        lim_3d = limits[None, None, :]  # (1, 1, n_layers)

        excess = xp.maximum(losses_3d - att_3d, 0.0)
        layer_recovery = xp.minimum(excess, lim_3d)
        # Sum across layers → (n_sims, max_events)
        per_event_recovery = xp.sum(layer_recovery, axis=2)

        # Cap: recovery cannot exceed (loss - deductible) per event
        deductible_dev = dtype.type(params.deductible)
        max_recovery = xp.maximum(masked_losses - deductible_dev, 0.0)
        per_event_recovery = xp.minimum(per_event_recovery, max_recovery)

        total_recoveries = xp.sum(per_event_recovery * event_mask, axis=1)
    else:
        # ---- Aggregate path: loop over events, vectorized across sims ----
        n_agg = sum(1 for h in has_aggregate if h)
        agg_used = xp.zeros((n_sims, n_layers), dtype=dtype)
        total_recoveries = xp.zeros(n_sims, dtype=dtype)
        agg_limits_dev = xp.asarray(agg_limits_list, dtype=dtype)
        deductible_dev = dtype.type(params.deductible)

        for ev in range(max_events):
            ev_loss = loss_amounts[:, ev]  # (n_sims,)
            ev_active = (ev < n_events).astype(dtype)  # (n_sims,)

            ev_recovery = xp.zeros(n_sims, dtype=dtype)

            for j in range(n_layers):
                excess = xp.maximum(ev_loss - attachments[j], 0.0)
                layer_rec = xp.minimum(excess, limits[j])

                # Aggregate cap
                if has_aggregate[j]:
                    remaining = xp.maximum(agg_limits_dev[j] - agg_used[:, j], 0.0)
                    layer_rec = xp.minimum(layer_rec, remaining)
                    agg_used[:, j] += layer_rec * ev_active

                ev_recovery += layer_rec

            # Cap at loss - deductible
            max_rec = xp.maximum(ev_loss - deductible_dev, 0.0)
            ev_recovery = xp.minimum(ev_recovery, max_rec)
            total_recoveries += ev_recovery * ev_active

    total_retained = total_losses - total_recoveries
    return total_losses, total_recoveries, total_retained


# ---------------------------------------------------------------------------
#  Financial state update (vectorized)
# ---------------------------------------------------------------------------


def update_financial_state(
    assets: Any,
    retained: Any,
    revenues: Any,
    premium: Any,
    params: GPUSimulationParams,
    xp: Any,
    dtype: Any,
) -> tuple:
    """Update financial state for one year, vectorized across sims.

    Simplified model:
        operating_income = revenue * margin
        pre_tax_income   = operating_income - retained_losses - premium
        tax              = max(0, pre_tax_income * tax_rate)
        net_income        = pre_tax_income - tax
        assets           += net_income * retention_ratio

    Args:
        assets: ``(n_sims,)`` current total assets.
        retained: ``(n_sims,)`` retained losses this year.
        revenues: ``(n_sims,)`` revenue this year.
        premium: scalar or ``(n_sims,)`` insurance premium.
        params: Simulation parameters.
        xp: Array module.
        dtype: Array dtype.

    Returns:
        ``(new_assets, new_equity)`` both ``(n_sims,)``.  In the
        simplified model equity equals assets (no liabilities tracked).
    """
    operating_income = revenues * dtype.type(params.base_operating_margin)
    pre_tax_income = operating_income - retained - premium

    # Tax only on positive income
    tax = xp.maximum(pre_tax_income, 0.0) * dtype.type(params.tax_rate)
    net_income = pre_tax_income - tax

    retained_earnings = net_income * dtype.type(params.retention_ratio)
    new_assets = assets + retained_earnings

    # In the simplified model, equity == assets (no liabilities)
    new_equity = new_assets
    return new_assets, new_equity


# ---------------------------------------------------------------------------
#  Main simulation loop
# ---------------------------------------------------------------------------


def run_gpu_simulation(
    params: GPUSimulationParams,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, np.ndarray]:
    """Execute the full vectorized Monte Carlo simulation.

    All *n_sims* paths are processed simultaneously each year.  The year
    loop is sequential (year Y depends on Y-1 assets) but each iteration
    is a batch of array operations across all simulations.

    Args:
        params: Flat simulation parameters.
        progress_callback: ``(completed_years, total_years, elapsed)``
        cancel_event: Threading event for cancellation.

    Returns:
        Dict with NumPy arrays matching the CPU engine output format:
        ``final_assets``, ``final_equity``, ``annual_losses``,
        ``insurance_recoveries``, ``retained_losses``.
    """
    import time as _time

    use_gpu = is_gpu_available()
    xp = get_array_module(gpu=use_gpu)
    dtype = np.dtype(np.float32 if params.use_float32 else np.float64)

    n_sims = params.n_simulations
    n_years = params.n_years

    # Seed the RNG
    rng = np.random.default_rng(params.seed)

    with gpu_memory_pool():
        # Pre-allocate result arrays
        annual_losses = xp.zeros((n_sims, n_years), dtype=dtype)
        insurance_recoveries = xp.zeros((n_sims, n_years), dtype=dtype)
        retained_losses = xp.zeros((n_sims, n_years), dtype=dtype)

        # State vectors
        assets = xp.full(n_sims, params.initial_assets, dtype=dtype)
        active_mask = xp.ones(n_sims, dtype=xp.bool_)

        # Frozen values for ruined paths
        frozen_assets = assets.copy()

        start_time = _time.time()

        for year in range(n_years):
            # Check cancellation
            if cancel_event is not None and cancel_event.is_set():
                logger.info("GPU simulation cancelled at year %d/%d", year, n_years)
                break

            # Revenue = max(assets, 0) * turnover * (1 + growth_rate)^year
            growth_factor = dtype.type((1.0 + params.growth_rate) ** year)
            revenues = (
                xp.maximum(assets, 0.0) * dtype.type(params.asset_turnover_ratio) * growth_factor
            )

            # Generate losses
            loss_amounts, n_events = generate_losses_for_year(params, revenues, xp, rng, dtype)

            # Apply insurance
            year_total_losses, year_recoveries, year_retained = apply_insurance_vectorized(
                loss_amounts, n_events, params, xp, dtype
            )

            # Premium scaled by revenue relative to initial
            base_revenue = dtype.type(params.initial_assets * params.asset_turnover_ratio)
            safe_base_revenue = dtype.type(max(float(base_revenue), 1.0))
            premium = dtype.type(params.base_annual_premium) * (revenues / safe_base_revenue)

            # Financial update
            new_assets, new_equity = update_financial_state(
                assets, year_retained, revenues, premium, params, xp, dtype
            )

            # Freeze ruined paths
            ruin_this_year = new_assets <= dtype.type(params.insolvency_tolerance)
            newly_ruined = ruin_this_year & active_mask
            active_mask = active_mask & ~ruin_this_year

            # Store frozen values before overwrite
            frozen_assets = xp.where(newly_ruined, new_assets, frozen_assets)

            # Active paths get new values; ruined paths keep frozen
            assets = xp.where(active_mask, new_assets, frozen_assets)

            # Record results — active paths get real values, ruined get this year's
            annual_losses[:, year] = xp.where(active_mask | newly_ruined, year_total_losses, 0.0)
            insurance_recoveries[:, year] = xp.where(
                active_mask | newly_ruined, year_recoveries, 0.0
            )
            retained_losses[:, year] = xp.where(active_mask | newly_ruined, year_retained, 0.0)

            # Progress callback
            if progress_callback is not None:
                elapsed = _time.time() - start_time
                progress_callback(year + 1, n_years, elapsed)

        # Transfer everything to CPU
        result = {
            "final_assets": to_numpy(assets),
            "final_equity": to_numpy(assets),  # simplified: equity == assets
            "annual_losses": to_numpy(annual_losses),
            "insurance_recoveries": to_numpy(insurance_recoveries),
            "retained_losses": to_numpy(retained_losses),
        }

    return result
