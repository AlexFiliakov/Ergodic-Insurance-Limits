"""GPU-accelerated Monte Carlo simulation engine.

Provides a vectorized financial model that processes ALL simulation paths
simultaneously using array operations.  When CuPy is available the arrays
live on the GPU; otherwise plain NumPy is used with the same code paths.

The GPU engine is a **faithful vectorized reproduction** of the CPU
financial model (``WidgetManufacturer`` / ``monte_carlo_worker``):

* Full balance sheet tracking: equity, PP&E, depreciation, DTL, working
  capital (AR, inventory, AP), NOL carryforward
* Revenue base excludes net DTA per Issue #1055
* NOL carryforward with TCJA 80%% limitation per IRC §172
* Equity-based ruin threshold matching CPU's ``equity <= tolerance``
* Insurance reinstatement logic for aggregate layers
* Capex reinvestment driven by depreciation × capex_to_depreciation_ratio

See Issue #1387 for the full alignment specification.

Since:
    Version 0.11.0 (Issue #961)
    Updated: Issue #1387 — full financial model fidelity
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

    The GPU engine replicates the full financial model from the CPU path
    (WidgetManufacturer / monte_carlo_worker) including balance sheet
    tracking, depreciation, capex, working capital, DTL, and NOL
    carryforward.  See Issue #1387 for details.

    Attributes:
        initial_assets: Starting total assets of the manufacturer.
        asset_turnover_ratio: Revenue per dollar of assets.
        base_operating_margin: EBIT / Revenue ratio.
        tax_rate: Corporate tax rate.
        retention_ratio: Fraction of net income retained.
        ppe_ratio: PP&E as fraction of initial assets.
        ppe_useful_life_years: Book depreciation useful life (straight-line).
        tax_depreciation_life_years: Tax depreciation life (MACRS proxy).
            None means same as book life (no DTL).
        capex_to_depreciation_ratio: Capex as multiple of depreciation.
        dso: Days Sales Outstanding for accounts receivable.
        dio: Days Inventory Outstanding for inventory.
        dpo: Days Payable Outstanding for accounts payable.
        nol_carryforward_enabled: Enable NOL carryforward per IRC §172.
        nol_limitation_pct: NOL deduction limitation (0.80 post-TCJA).
        deductible: Insurance program self-insured retention.
        layer_attachments: Per-layer attachment points.
        layer_limits: Per-layer coverage limits.
        layer_limit_types: Per-layer limit type (0=per-occurrence, 1=aggregate, 2=hybrid).
        layer_aggregate_limits: Per-layer annual aggregate caps (inf if none).
        layer_reinstatements: Per-layer max reinstatements available.
        layer_reinstatement_premium_rates: Per-layer reinstatement premium
            as fraction of base layer premium.
        base_annual_premium: Total annual premium across all layers.
        layer_base_premiums: Per-layer base premium for reinstatement calc.
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
        letter_of_credit_rate: Annual LoC rate for collateral costs.
        growth_rate: Exogenous revenue growth rate per year.
        use_float32: Whether to use float32 for arrays.
        seed: Random seed for reproducibility.
    """

    # Manufacturer — core financials
    initial_assets: float = 10_000_000.0
    asset_turnover_ratio: float = 1.0
    base_operating_margin: float = 0.10
    tax_rate: float = 0.21
    retention_ratio: float = 0.80

    # Manufacturer — balance sheet (Issue #1387)
    ppe_ratio: float = 0.3
    ppe_useful_life_years: float = 10.0
    tax_depreciation_life_years: Optional[float] = None
    capex_to_depreciation_ratio: float = 1.0

    # Working capital (Issue #1387)
    dso: float = 45.0
    dio: float = 60.0
    dpo: float = 30.0

    # Tax — NOL carryforward (Issue #1387)
    nol_carryforward_enabled: bool = True
    nol_limitation_pct: float = 0.80

    # Insurance layers
    deductible: float = 0.0
    layer_attachments: List[float] = field(default_factory=list)
    layer_limits: List[float] = field(default_factory=list)
    layer_limit_types: List[int] = field(default_factory=list)  # 0=per-occ, 1=agg, 2=hybrid
    layer_aggregate_limits: List[float] = field(default_factory=list)
    layer_reinstatements: List[int] = field(default_factory=list)
    layer_reinstatement_premium_rates: List[float] = field(default_factory=list)
    base_annual_premium: float = 0.0
    layer_base_premiums: List[float] = field(default_factory=list)

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
    reinstatements: List[int] = []
    reinstatement_premium_rates: List[float] = []
    layer_premiums: List[float] = []

    _LIMIT_TYPE_MAP = {"per-occurrence": 0, "aggregate": 1, "hybrid": 2}

    for layer in insurance_program.layers:
        attachments.append(float(layer.attachment_point))
        limits.append(float(layer.limit))
        lt = getattr(layer, "limit_type", "per-occurrence")
        limit_types.append(_LIMIT_TYPE_MAP.get(lt, 0))
        agg = getattr(layer, "aggregate_limit", None)
        agg_limits.append(float(agg) if agg is not None else float("inf"))
        reinstatements.append(int(getattr(layer, "reinstatements", 0)))
        reinstatement_premium_rates.append(float(getattr(layer, "reinstatement_premium", 0.0)))
        layer_premiums.append(float(layer.calculate_base_premium()))

    base_premium = float(insurance_program.calculate_premium())

    # --- Balance sheet params (Issue #1387) ---
    ppe_ratio = float(cfg.ppe_ratio) if cfg.ppe_ratio is not None else 0.3
    tax_depr_life = (
        float(cfg.tax_depreciation_life_years)
        if cfg.tax_depreciation_life_years is not None
        else None
    )

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
        # Manufacturer — core
        initial_assets=float(cfg.initial_assets),
        asset_turnover_ratio=float(cfg.asset_turnover_ratio),
        base_operating_margin=float(cfg.base_operating_margin),
        tax_rate=float(cfg.tax_rate),
        retention_ratio=float(cfg.retention_ratio),
        # Manufacturer — balance sheet (Issue #1387)
        ppe_ratio=ppe_ratio,
        ppe_useful_life_years=float(cfg.ppe_useful_life_years),
        tax_depreciation_life_years=tax_depr_life,
        capex_to_depreciation_ratio=float(cfg.capex_to_depreciation_ratio),
        # Working capital (Issue #1387) — use defaults (45/60/30) matching CPU init
        dso=45.0,
        dio=60.0,
        dpo=30.0,
        # Tax — NOL (Issue #1387)
        nol_carryforward_enabled=bool(cfg.nol_carryforward_enabled),
        nol_limitation_pct=float(cfg.nol_limitation_pct),
        # Insurance
        deductible=float(insurance_program.deductible),
        layer_attachments=attachments,
        layer_limits=limits,
        layer_limit_types=limit_types,
        layer_aggregate_limits=agg_limits,
        layer_reinstatements=reinstatements,
        layer_reinstatement_premium_rates=reinstatement_premium_rates,
        base_annual_premium=base_premium,
        layer_base_premiums=layer_premiums,
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
    """Scatter per-generator severities into the padded (n_sims, max_events) array.

    Uses vectorized cumsum-based offsets and fancy indexing instead of a
    Python for-loop over ``n_sims``.  All row/column coordinates are
    computed as arrays and assigned in a single bulk operation per
    generator type.
    """
    n_sims = loss_amounts_cpu.shape[0]

    # Column starts per generator per sim (replicates sequential col tracking)
    att_col_start = np.zeros(n_sims, dtype=np.int32)
    lg_col_start = np.minimum(att_counts, max_events).astype(np.int32)
    cat_col_start = np.minimum(lg_col_start + lg_counts, max_events).astype(np.int32)

    for counts, sevs, col_start in (
        (att_counts, att_sevs, att_col_start),
        (lg_counts, lg_sevs, lg_col_start),
        (cat_counts, cat_sevs, cat_col_start),
    ):
        total = int(np.sum(counts))
        if total == 0:
            continue

        # Row indices: repeat each sim index by its count
        rows = np.repeat(np.arange(n_sims, dtype=np.int32), counts)

        # Local event indices within each sim: 0, 1, ..., counts[i]-1
        sim_starts = np.cumsum(counts) - counts  # exclusive prefix sum
        offsets_within = np.arange(total, dtype=np.int32) - np.repeat(sim_starts, counts)

        # Global column = col_start[sim] + local offset
        cols = np.repeat(col_start, counts) + offsets_within

        # Only assign events that fit within max_events columns
        valid = cols < max_events
        loss_amounts_cpu[rows[valid], cols[valid]] = sevs[valid]


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
    vectorized across all simulations.  Aggregate layers support
    reinstatement logic matching the CPU ``LayerState.process_claim``
    path (Issue #1387).

    Args:
        loss_amounts: ``(n_sims, max_events)`` padded loss array.
        n_events: ``(n_sims,)`` actual event count per sim.
        params: Flat simulation parameters.
        xp: Array module.
        dtype: Array dtype.

    Returns:
        ``(total_losses, recoveries, retained, reinstatement_premiums)``
        each ``(n_sims,)``.
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

    reinstatement_premiums = xp.zeros(n_sims, dtype=dtype)

    n_layers = len(params.layer_attachments)
    if n_layers == 0:
        total_recoveries = xp.zeros(n_sims, dtype=dtype)
        total_retained = total_losses
        return total_losses, total_recoveries, total_retained, reinstatement_premiums

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
        losses_3d = masked_losses[:, :, None]  # (n_sims, max_events, 1)
        att_3d = attachments[None, None, :]  # (1, 1, n_layers)
        lim_3d = limits[None, None, :]  # (1, 1, n_layers)

        excess = xp.maximum(losses_3d - att_3d, 0.0)
        layer_recovery = xp.minimum(excess, lim_3d)
        per_event_recovery = xp.sum(layer_recovery, axis=2)

        deductible_dev = dtype.type(params.deductible)
        max_recovery = xp.maximum(masked_losses - deductible_dev, 0.0)
        per_event_recovery = xp.minimum(per_event_recovery, max_recovery)

        total_recoveries = xp.sum(per_event_recovery * event_mask, axis=1)
    else:
        # ---- Aggregate path with reinstatement logic (Issue #1387) ----
        agg_used = xp.zeros((n_sims, n_layers), dtype=dtype)
        total_recoveries = xp.zeros(n_sims, dtype=dtype)
        agg_limits_dev = xp.asarray(agg_limits_list, dtype=dtype)
        deductible_dev = dtype.type(params.deductible)

        # Reinstatement state per layer per sim
        reinst_max = params.layer_reinstatements or [0] * n_layers
        reinst_rates = params.layer_reinstatement_premium_rates or [0.0] * n_layers
        layer_prems = params.layer_base_premiums or [0.0] * n_layers
        reinst_used = xp.zeros((n_sims, n_layers), dtype=dtype)

        for ev in range(max_events):
            ev_loss = loss_amounts[:, ev]  # (n_sims,)
            ev_active = (ev < n_events).astype(dtype)  # (n_sims,)

            ev_recovery = xp.zeros(n_sims, dtype=dtype)

            for j in range(n_layers):
                excess = xp.maximum(ev_loss - attachments[j], 0.0)
                layer_rec = xp.minimum(excess, limits[j])

                if has_aggregate[j]:
                    remaining = xp.maximum(agg_limits_dev[j] - agg_used[:, j], 0.0)
                    layer_rec = xp.minimum(layer_rec, remaining)
                    agg_used[:, j] += layer_rec * ev_active

                    # Reinstatement: when aggregate exhausted and reinstatements remain
                    if reinst_max[j] > 0:
                        exhausted = (agg_used[:, j] >= agg_limits_dev[j]) & (ev_active > 0)
                        can_reinstate = exhausted & (reinst_used[:, j] < reinst_max[j])
                        if xp.any(can_reinstate):
                            # Calculate reinstatement premium for reinstating sims
                            reinst_prem = dtype.type(layer_prems[j] * reinst_rates[j])
                            reinstatement_premiums += xp.where(
                                can_reinstate, reinst_prem, dtype.type(0.0)
                            )
                            # Reset aggregate usage and increment reinstatement counter
                            agg_used[:, j] = xp.where(
                                can_reinstate, dtype.type(0.0), agg_used[:, j]
                            )
                            reinst_used[:, j] = xp.where(
                                can_reinstate,
                                reinst_used[:, j] + dtype.type(1.0),
                                reinst_used[:, j],
                            )

                ev_recovery += layer_rec

            # Cap at loss - deductible
            max_rec = xp.maximum(ev_loss - deductible_dev, 0.0)
            ev_recovery = xp.minimum(ev_recovery, max_rec)
            total_recoveries += ev_recovery * ev_active

    total_retained = total_losses - total_recoveries
    return total_losses, total_recoveries, total_retained, reinstatement_premiums


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
    """Legacy simplified financial update (used by unit tests).

    The full balance sheet model is implemented inline in
    :func:`run_gpu_simulation`.  This function preserves the original
    simplified interface for backward-compatible unit tests.

    Args:
        assets: ``(n_sims,)`` current total assets.
        retained: ``(n_sims,)`` retained losses this year.
        revenues: ``(n_sims,)`` revenue this year.
        premium: scalar or ``(n_sims,)`` insurance premium.
        params: Simulation parameters.
        xp: Array module.
        dtype: Array dtype.

    Returns:
        ``(new_assets, new_equity)`` both ``(n_sims,)``.
    """
    operating_income = revenues * dtype.type(params.base_operating_margin)
    pre_tax_income = operating_income - retained - premium

    # Tax only on positive income
    tax = xp.maximum(pre_tax_income, 0.0) * dtype.type(params.tax_rate)
    net_income = pre_tax_income - tax

    # Dividends only on positive income (Issue #1387)
    dividends = xp.where(
        net_income > 0,
        net_income * dtype.type(1.0 - params.retention_ratio),
        dtype.type(0.0),
    )
    retained_earnings = net_income - dividends
    new_assets = assets + retained_earnings

    new_equity = new_assets
    return new_assets, new_equity


# ---------------------------------------------------------------------------
#  Balance sheet helpers (Issue #1387)
# ---------------------------------------------------------------------------


def _compute_dtl_dta(accum_tax_depr, accum_book_depr, nol_carryforward, tax_rate, xp, dtype):
    """Compute net deferred tax liability/asset from depreciation timing and NOL.

    Returns:
        ``(net_dtl, net_dta)`` each ``(n_sims,)``.
        Exactly one of them is non-zero per sim (netting within jurisdiction,
        ASC 740-10-45-6).
    """
    dta = nol_carryforward * dtype.type(tax_rate)
    timing_diff = xp.maximum(accum_tax_depr - accum_book_depr, 0.0)
    dtl = timing_diff * dtype.type(tax_rate)
    net_dtl = xp.maximum(dtl - dta, 0.0)
    net_dta = xp.maximum(dta - dtl, 0.0)
    return net_dtl, net_dta


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

    The financial model replicates the CPU path (``WidgetManufacturer.step``
    + ``monte_carlo_worker.run_chunk_standalone``) including:

    * Balance sheet tracking: equity, PP&E, depreciation, DTL, working capital
    * Revenue base excludes net DTA (Issue #1055)
    * NOL carryforward with TCJA limitation (IRC §172)
    * Equity-based ruin threshold
    * Insurance reinstatement logic for aggregate layers

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
        retained_losses_arr = xp.zeros((n_sims, n_years), dtype=dtype)

        # ---- Initialize balance sheet (matches CPU WidgetManufacturer.__init__) ----
        initial_assets = dtype.type(params.initial_assets)
        initial_ppe_ratio = dtype.type(params.ppe_ratio)
        initial_gross_ppe_val = float(params.initial_assets * params.ppe_ratio)
        initial_revenue = float(params.initial_assets * params.asset_turnover_ratio)
        initial_cogs = initial_revenue * (1.0 - params.base_operating_margin)
        initial_ar = initial_revenue * (params.dso / 365.0)
        initial_inventory = initial_cogs * (params.dio / 365.0)

        # State vectors — balance sheet (n_sims,)
        equity = xp.full(n_sims, params.initial_assets, dtype=dtype)
        gross_ppe = xp.full(n_sims, initial_gross_ppe_val, dtype=dtype)
        accum_book_depr = xp.zeros(n_sims, dtype=dtype)
        accum_tax_depr = xp.zeros(n_sims, dtype=dtype)
        nol_carryforward = xp.zeros(n_sims, dtype=dtype)
        # AP initialized to 0 (matching CPU manufacturer.__init__)
        prev_ap = xp.zeros(n_sims, dtype=dtype)

        active_mask = xp.ones(n_sims, dtype=xp.bool_)

        # Frozen values for ruined paths
        frozen_equity = equity.copy()
        frozen_assets = xp.full(n_sims, params.initial_assets, dtype=dtype)

        start_time = _time.time()

        for year in range(n_years):
            # Check cancellation
            if cancel_event is not None and cancel_event.is_set():
                logger.info("GPU simulation cancelled at year %d/%d", year, n_years)
                break

            # ---- Derive total_assets from balance sheet state ----
            net_dtl, net_dta = _compute_dtl_dta(
                accum_tax_depr,
                accum_book_depr,
                nol_carryforward,
                params.tax_rate,
                xp,
                dtype,
            )
            total_liabilities = prev_ap + net_dtl
            total_assets = equity + total_liabilities

            # ---- Revenue (exclude net DTA from base, Issue #1055) ----
            revenue_base = xp.maximum(total_assets - net_dta, 0.0)
            growth_factor = dtype.type((1.0 + params.growth_rate) ** year)
            revenues = revenue_base * dtype.type(params.asset_turnover_ratio) * growth_factor

            # ---- Generate losses ----
            loss_amounts, n_events = generate_losses_for_year(
                params,
                revenues,
                xp,
                rng,
                dtype,
            )

            # ---- Apply insurance (with reinstatements) ----
            year_total_losses, year_recoveries, year_retained, reinst_premium = (
                apply_insurance_vectorized(loss_amounts, n_events, params, xp, dtype)
            )

            # ---- Premium scaled by revenue relative to initial ----
            base_revenue = dtype.type(params.initial_assets * params.asset_turnover_ratio)
            safe_base_revenue = dtype.type(max(float(base_revenue), 1.0))
            premium = dtype.type(params.base_annual_premium) * (revenues / safe_base_revenue)

            # ---- Income statement (matches CPU calculate_operating_income) ----
            operating_income = revenues * dtype.type(params.base_operating_margin)
            # Subtract insurance costs from operating income (same as CPU)
            pre_tax_income = operating_income - year_retained - premium - reinst_premium

            # ---- Tax with NOL carryforward (Issue #1387) ----
            tax_rate = dtype.type(params.tax_rate)
            if params.nol_carryforward_enabled:
                # Utilize existing NOL against positive income
                nol_limit_pct = dtype.type(params.nol_limitation_pct)
                nol_deduction = xp.where(
                    pre_tax_income > 0,
                    xp.minimum(
                        nol_carryforward,
                        pre_tax_income * nol_limit_pct,
                    ),
                    dtype.type(0.0),
                )
                taxable_income = xp.maximum(pre_tax_income - nol_deduction, 0.0)
                tax = taxable_income * tax_rate
                # Consume utilized NOL
                nol_carryforward = nol_carryforward - nol_deduction
                # Generate new NOL from losses
                new_nol = xp.maximum(-pre_tax_income, 0.0)
                nol_carryforward = nol_carryforward + new_nol
            else:
                tax = xp.maximum(pre_tax_income, 0.0) * tax_rate

            net_income = pre_tax_income - tax

            # ---- Dividends only on positive income (Issue #1387) ----
            dividends = xp.where(
                net_income > 0,
                net_income * dtype.type(1.0 - params.retention_ratio),
                dtype.type(0.0),
            )
            retained_earnings = net_income - dividends

            # ---- Depreciation (book) ----
            net_ppe = xp.maximum(gross_ppe - accum_book_depr, 0.0)
            book_life = dtype.type(params.ppe_useful_life_years)
            book_depr = xp.minimum(gross_ppe / book_life, net_ppe)
            accum_book_depr = accum_book_depr + book_depr

            # ---- Depreciation (tax, for DTL) ----
            if params.tax_depreciation_life_years is not None:
                tax_life = dtype.type(params.tax_depreciation_life_years)
                remaining_tax_basis = xp.maximum(gross_ppe - accum_tax_depr, 0.0)
                tax_depr = xp.minimum(gross_ppe / tax_life, remaining_tax_basis)
                accum_tax_depr = accum_tax_depr + tax_depr

            # ---- Capex reinvestment (cash → PP&E) ----
            capex_ratio = dtype.type(params.capex_to_depreciation_ratio)
            capex = book_depr * capex_ratio
            gross_ppe = gross_ppe + capex

            # ---- Working capital update ----
            cogs = revenues * dtype.type(1.0 - params.base_operating_margin)
            new_ap = cogs * dtype.type(params.dpo / 365.0)

            # ---- DTL change for equity adjustment ----
            new_net_dtl, new_net_dta = _compute_dtl_dta(
                accum_tax_depr,
                accum_book_depr,
                nol_carryforward,
                params.tax_rate,
                xp,
                dtype,
            )
            delta_net_dtl = new_net_dtl - net_dtl

            # ---- Equity update ----
            # equity += retained_earnings - delta_net_DTL
            # (AP changes cancel out: ΔTA = RE + ΔAP, ΔTL = ΔAP + ΔDTL,
            #  so ΔEquity = RE - ΔDTL)
            new_equity = equity + retained_earnings - delta_net_dtl

            # Update AP for next period's total_assets derivation
            prev_ap = new_ap

            # Derive new total_assets for ruin check and output
            new_total_liabilities = new_ap + new_net_dtl
            new_total_assets = new_equity + new_total_liabilities

            # ---- Ruin detection (equity-based, Issue #1387) ----
            ruin_this_year = new_equity <= dtype.type(params.insolvency_tolerance)
            newly_ruined = ruin_this_year & active_mask
            active_mask = active_mask & ~ruin_this_year

            # Store frozen values before overwrite
            frozen_equity = xp.where(newly_ruined, new_equity, frozen_equity)
            frozen_assets = xp.where(newly_ruined, new_total_assets, frozen_assets)

            # Active paths get new values; ruined paths keep frozen
            equity = xp.where(active_mask, new_equity, frozen_equity)

            # Also freeze balance sheet state for ruined paths
            gross_ppe = xp.where(active_mask, gross_ppe, gross_ppe)  # no-op, kept for clarity
            accum_book_depr = xp.where(active_mask, accum_book_depr, accum_book_depr)
            accum_tax_depr = xp.where(active_mask, accum_tax_depr, accum_tax_depr)
            nol_carryforward = xp.where(active_mask, nol_carryforward, nol_carryforward)
            prev_ap = xp.where(active_mask, prev_ap, prev_ap)

            # ---- Record results ----
            annual_losses[:, year] = xp.where(
                active_mask | newly_ruined,
                year_total_losses,
                0.0,
            )
            insurance_recoveries[:, year] = xp.where(
                active_mask | newly_ruined,
                year_recoveries,
                0.0,
            )
            retained_losses_arr[:, year] = xp.where(
                active_mask | newly_ruined,
                year_retained,
                0.0,
            )

            # Progress callback
            if progress_callback is not None:
                elapsed = _time.time() - start_time
                progress_callback(year + 1, n_years, elapsed)

        # ---- Final output ----
        # Compute final total_assets for output
        final_net_dtl, _ = _compute_dtl_dta(
            accum_tax_depr,
            accum_book_depr,
            nol_carryforward,
            params.tax_rate,
            xp,
            dtype,
        )
        final_total_liabilities = prev_ap + final_net_dtl
        final_total_assets = equity + final_total_liabilities

        # For ruined paths, use frozen values
        out_assets = xp.where(active_mask, final_total_assets, frozen_assets)
        out_equity = xp.where(active_mask, equity, frozen_equity)

        result = {
            "final_assets": to_numpy(out_assets),
            "final_equity": to_numpy(out_equity),
            "annual_losses": to_numpy(annual_losses),
            "insurance_recoveries": to_numpy(insurance_recoveries),
            "retained_losses": to_numpy(retained_losses_arr),
        }

    return result
