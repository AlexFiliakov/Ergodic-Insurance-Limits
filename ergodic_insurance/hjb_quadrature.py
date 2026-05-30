"""Quadrature primitives for HJB jump-term expectations.

This module implements the per-event severity discretization used by the
PIDE jump operator in the HJB optimal-control notebook
(``notebooks/optimization/07_hjb_insurance_optimization.ipynb``).

The discretization replaces the analytical expectation
``E_X[V(w - L_retained(X, SIR))]`` with a finite weighted sum
``sum_k w_k V(w - L_retained(x_k, SIR))`` over a small set of quadrature
points ``(x_k, w_k)``. The notebook then assembles a sparse jump operator
``J`` from these points and applies it once per inner iteration.

The hybrid scheme (issue #1572 R4b) dispatches on severity component type:

- **Lognormal components** use Gauss-Hermite quadrature on
  ``Y = log(X) ~ Normal(mu, sigma^2)``. With ``n_nodes = 16-32`` this is
  exact for any smooth function of ``X`` to machine precision -- vastly
  more efficient than equiprobable-bin atoms, and crucially with no
  ``SIR``-grid-induced piecewise-affine artifacts in the Hamiltonian.
- **Pareto components** use stratified atoms: half equiprobable in the
  body ``[0, F^{-1}(0.95)]``, half logarithmically spaced in the upper
  tail ``[F^{-1}(0.95), F^{-1}(0.99999)]``. The within-bin atom value is
  the conditional mean (exact via differences of the limited expected
  value), which preserves the per-bin first moment.

References:
    Issue #1572 R4b for the rationale; ``feedback_hjb_pide_convention.md``
    for the broader PIDE convention these atoms feed into.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "severity_cdf",
    "lognormal_gauss_hermite_nodes",
    "pareto_stratified_atoms",
    "component_atoms",
    "build_loss_atoms",
    "build_equity_jump_operator_2d",
    "build_equity_jump_operator_2d_sizescaled",
    "make_jump_term_2d",
]


def severity_cdf(severity, x: float) -> float:
    """Analytical CDF for the supported severity distributions.

    Args:
        severity: A ``LognormalLoss`` (has ``mu`` and ``sigma``) or a
            ``ParetoLoss`` (has ``alpha`` and ``xm``).
        x: The point at which to evaluate ``F(x) = Pr(X <= x)``.

    Returns:
        ``F(x)`` in ``[0, 1]``.

    Raises:
        TypeError: if the severity type is not supported.
    """
    from scipy import stats as _stats

    if not np.isfinite(x):
        return 1.0
    if x <= 0:
        return 0.0
    if hasattr(severity, "mu") and hasattr(severity, "sigma"):
        return float(_stats.norm.cdf((np.log(x) - severity.mu) / severity.sigma))
    if hasattr(severity, "alpha") and hasattr(severity, "xm"):
        if x <= severity.xm:
            return 0.0
        return float(1.0 - (severity.xm / x) ** severity.alpha)
    raise TypeError(f"Unsupported severity for CDF: {type(severity)}")


def lognormal_gauss_hermite_nodes(severity, n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite quadrature points for ``E_X[g(X)]`` with ``X`` lognormal.

    For ``X = exp(Y)`` with ``Y ~ Normal(mu, sigma^2)``, the substitution
    ``Y = mu + sigma * sqrt(2) * eta`` reduces the expectation to the
    physicists' Gauss-Hermite form ``(1/sqrt(pi)) integral exp(-eta^2)
    g(exp(mu + sigma*sqrt(2)*eta)) d(eta)``. Standard Gauss-Hermite nodes
    and weights then give an exact quadrature for any polynomial of order
    up to ``2*n_nodes - 1`` in ``Y``, which translates to extremely high
    accuracy for any smooth function of ``X``.

    Args:
        severity: A ``LognormalLoss`` (must expose ``mu`` and ``sigma``).
        n_nodes: Number of Gauss-Hermite nodes. ``16-32`` is typically
            sufficient for full machine precision on smooth integrands.

    Returns:
        ``(x_nodes, w_nodes)`` where ``x_nodes`` are positive samples of
        ``X`` and ``w_nodes`` are normalized weights summing to ``1``.
    """
    if not (hasattr(severity, "mu") and hasattr(severity, "sigma")):
        raise TypeError(
            f"lognormal_gauss_hermite_nodes requires LognormalLoss-like "
            f"severity; got {type(severity)}"
        )
    eta_nodes, eta_weights = np.polynomial.hermite.hermgauss(n_nodes)
    x_nodes = np.exp(severity.mu + severity.sigma * np.sqrt(2.0) * eta_nodes)
    w_nodes = eta_weights / np.sqrt(np.pi)
    # Renormalize against floating-point drift so weights sum to exactly 1.
    w_nodes = w_nodes / w_nodes.sum()
    return x_nodes, w_nodes


def pareto_stratified_atoms(severity, n_atoms: int) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified body + log-tail atoms for a Pareto severity.

    The heavy upper tail of Pareto Type I severities carries
    disproportionate mass per atom, so equiprobable binning under-samples
    the tail. This routine puts half the atoms in the body
    ``[0, F^{-1}(0.95)]`` equiprobably and the other half in the upper
    tail ``[F^{-1}(0.95), F^{-1}(1 - 1e-5)]`` with logarithmically spaced
    survival probability ``1 - F``.

    Within each bin, the atom value is the conditional mean computed
    exactly from differences of the limited expected value
    ``E[min(X, x)]``, matching the convention from the original notebook
    discretization.

    Args:
        severity: A ``ParetoLoss`` (must expose ``alpha`` and ``xm`` and
            ``limited_expected_value`` / ``expected_value`` methods).
        n_atoms: Total number of atoms. Split half/half between body and
            tail; minimum sensible value is around ``20``.

    Returns:
        ``(x_atoms, p_atoms)`` where ``p_atoms`` are non-negative and
        sum to ``1`` (within floating-point precision).
    """
    if not (hasattr(severity, "alpha") and hasattr(severity, "xm")):
        raise TypeError(
            f"pareto_stratified_atoms requires ParetoLoss-like severity; " f"got {type(severity)}"
        )
    if n_atoms < 4:
        raise ValueError(f"n_atoms must be at least 4; got {n_atoms}")

    n_body = n_atoms // 2
    n_tail = n_atoms - n_body
    # Body edges: equiprobable in [0, 0.95]
    body_edges = np.linspace(0.0, 0.95, n_body + 1)
    # Tail edges: log-spaced in (1 - F), from 0.05 down to 1e-5
    tail_one_minus = np.logspace(np.log10(0.05), -5.0, n_tail + 1)
    tail_edges = 1.0 - tail_one_minus  # ascending in p
    # Concatenate and dedupe (body[-1] == tail_edges[0] == 0.95)
    edge_probs = np.unique(np.concatenate([body_edges, tail_edges]))

    # Invert the Pareto CDF analytically: x = xm / (1 - p)^(1/alpha).
    inner = edge_probs[1:-1]
    x_edges = np.concatenate(
        [[0.0], severity.xm / (1.0 - inner) ** (1.0 / severity.alpha), [np.inf]]
    )

    ev = severity.expected_value()
    lev_edges = np.array(
        [severity.limited_expected_value(x) if np.isfinite(x) else ev for x in x_edges]
    )
    f_edges = np.array([severity_cdf(severity, x) for x in x_edges])

    delta_lev = lev_edges[1:] - lev_edges[:-1]
    delta_f = f_edges[1:] - f_edges[:-1]
    delta_f_safe = np.where(delta_f > 0, delta_f, 1.0)
    x_atoms = delta_lev / delta_f_safe
    p_atoms = delta_f / delta_f.sum()
    return x_atoms, p_atoms


def component_atoms(severity, n_atoms: int, gh_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch to the right quadrature for a single severity component.

    Lognormal severities use Gauss-Hermite quadrature with ``gh_nodes``
    nodes; Pareto severities use ``pareto_stratified_atoms`` with
    ``n_atoms`` total atoms. The hybrid choice (issue #1572 R4b) is what
    eliminates the piecewise-affine artifacts in the SIR-Hamiltonian
    that caused the notebook-07 boundary-pinning bug.

    Args:
        severity: A LognormalLoss or ParetoLoss instance.
        n_atoms: Number of stratified atoms for Pareto components
            (ignored for lognormal).
        gh_nodes: Number of Gauss-Hermite nodes for lognormal components
            (ignored for Pareto).

    Returns:
        ``(x_atoms, p_atoms)`` where ``p_atoms`` sums to ``1`` and
        approximates the within-component severity distribution.
    """
    if hasattr(severity, "mu") and hasattr(severity, "sigma"):
        return lognormal_gauss_hermite_nodes(severity, gh_nodes)
    if hasattr(severity, "alpha") and hasattr(severity, "xm"):
        return pareto_stratified_atoms(severity, n_atoms)
    raise TypeError(f"Unsupported severity type for component_atoms: {type(severity)}")


def build_loss_atoms(
    pricers: Sequence, n_per_component: int, gh_nodes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the mixture atom table for a tuple of layer pricers.

    Each pricer's severity component is discretized via ``component_atoms``
    (Gauss-Hermite for lognormal, stratified for Pareto). Within-component
    probabilities are then weighted by the pricer's mixture share
    ``frequency / total_frequency``, and the per-component atom tables are
    concatenated into flat arrays suitable for the sparse jump-operator
    construction in the notebook.

    Args:
        pricers: Sequence of pricers, each exposing ``.severity`` and
            ``.frequency``.
        n_per_component: Number of stratified atoms for each Pareto
            component (lognormals use ``gh_nodes`` instead).
        gh_nodes: Number of Gauss-Hermite nodes for each lognormal
            component.

    Returns:
        ``(x_atoms, p_atoms)`` flat arrays. ``p_atoms`` sums to ``1`` and
        encodes the per-event severity mixture across all components.
    """
    if not pricers:
        raise ValueError("pricers must be non-empty")
    total_freq = sum(p.frequency for p in pricers)
    if total_freq <= 0:
        raise ValueError(f"total frequency must be positive; got {total_freq}")
    x_parts: list[np.ndarray] = []
    p_parts: list[np.ndarray] = []
    for p in pricers:
        xa, pa = component_atoms(p.severity, n_per_component, gh_nodes)
        x_parts.append(xa)
        p_parts.append(pa * (p.frequency / total_freq))
    return np.concatenate(x_parts), np.concatenate(p_parts)


# =====================================================
# 2-D (assets, equity-ratio) equity-only PIDE jump operator
# =====================================================
# These two functions assemble and apply the block-diagonal equity-jump operator
# used by the 2-D HJB solve in notebook 07. They were lifted verbatim from that
# notebook (cell 16) so the interpolation math is unit-testable without a 25-year
# solve; the only behavioural additions are an explicit ``phi_retention`` parameter
# (was a notebook global) and an opt-in ``interp="linear"`` mode on the returned
# callback for continuous off-grid / diagnostic evaluation (issue #1612). The
# default ``interp="nearest"`` path is the original code, so the solve -- which
# always evaluates on grid nodes, where linear and nearest coincide -- is
# bit-for-bit unchanged.


def build_equity_jump_operator_2d(
    a_grid: np.ndarray,
    e_grid_l: np.ndarray,
    sir_grid_l: np.ndarray,
    x_atoms: np.ndarray,
    p_atoms: np.ndarray,
    tower_top: float,
    lae_ratio: float,
    e_floor: float,
    phi_retention: float,
) -> Tuple[Any, np.ndarray]:
    """Build the block-diagonal equity-jump operator (``J_big``, ``p_ruin_2d``).

    A loss reduces EQUITY only (assets fixed at impact): for assets-slice ``A_i``,
    post-equity-ratio is ``e - L_ret/A_i`` with ``L_ret = (1+LAE)*(min(X,SIR) +
    (X-TOWER_TOP)+)``. Each slice is an independent 1-D-in-e interpolation operator
    (the post-jump assets index never changes), so the full operator is
    block-diagonal across assets.

    Args:
        a_grid: Assets grid (log-spaced), shape ``(n_a,)``.
        e_grid_l: Equity-ratio grid (linear-spaced, ascending), shape ``(n_e,)``.
        sir_grid_l: SIR control grid (log-spaced), shape ``(n_sir,)``.
        x_atoms: Per-event severity atom values, shape ``(n_atoms,)``.
        p_atoms: Per-event severity atom probabilities (sum to 1), shape ``(n_atoms,)``.
        tower_top: Top of the insurance tower; losses above it are retained.
        lae_ratio: Loss-adjustment-expense markup on retained loss.
        e_floor: Equity operational ruin floor (post-jump equity below it = ruin).
        phi_retention: After-tax/after-dividend retention factor scaling the
            SURVIVING (non-ruining) equity reduction; the GROSS retained loss
            (no ``phi``) drives the single-loss ruin test (issue #1598).

    Returns:
        ``(J_big, p_ruin_2d)``. ``J_big`` is a CSR matrix of shape
        ``(n_a*n_e*n_sir, n_a*n_e)``: row ``((a*n_e+e)*n_sir+sir)`` maps
        ``V_flat`` (``= V.ravel()``, C-order on ``(n_a, n_e)``) to ``E_X[V(post)]``
        for the non-ruining atoms (mass ``1 - p_ruin``). ``p_ruin_2d`` has shape
        ``(n_a, n_e, n_sir)`` -- the per-event probability the loss drives post-jump
        equity below ``e_floor``.
    """
    from scipy.sparse import block_diag, coo_matrix

    n_a, n_e, n_sir = len(a_grid), len(e_grid_l), len(sir_grid_l)
    n_atoms = len(x_atoms)
    # Retained loss per (sir, atom) -- independent of state. Two versions (issue #1598):
    #   * GROSS retained loss (incl. LAE) drives the single-loss RUIN test: the firm must
    #     absorb the full retained cost at the instant of the claim (limited liability), so a
    #     loss exceeding current equity bankrupts it NOW regardless of the tax shield.
    #   * AFTER-TAX retained loss (x phi_retention) drives the SURVIVING equity reduction: a
    #     non-ruining loss is booked as a deferred claim liability costing phi_retention x face.
    L_gross_atom_sir = (1.0 + lae_ratio) * (
        np.minimum(x_atoms[None, :], sir_grid_l[:, None])
        + np.maximum(x_atoms[None, :] - tower_top, 0.0)
    )  # (n_sir, n_atoms), GROSS retained
    L_per_atom_sir = phi_retention * L_gross_atom_sir  # (n_sir, n_atoms), after-tax
    row_tmpl = (np.arange(n_e)[:, None] * n_sir + np.arange(n_sir)[None, :]).ravel()

    blocks = []
    p_ruin_2d = np.empty((n_a, n_e, n_sir))
    for i, A_i in enumerate(a_grid):
        E_grid_i = e_grid_l * A_i  # equity at this slice
        # Ruin mask on the GROSS retained loss (#1598): insolvent the instant a single loss
        # exceeds current equity, before any tax deferral (gross mask >= the after-tax mask).
        raw_post_E_gross = (
            E_grid_i[:, None, None] - L_gross_atom_sir[None, :, :]
        )  # (n_e,n_sir,n_atoms)
        ruin = raw_post_E_gross < e_floor
        p_ruin_2d[i] = np.sum(p_atoms[None, None, :] * ruin, axis=2)
        # Surviving atoms reduce equity by only the AFTER-TAX amount (interpolation target).
        raw_post_E = E_grid_i[:, None, None] - L_per_atom_sir[None, :, :]
        post_e = np.where(ruin, e_grid_l[0], raw_post_E / A_i)
        post_e = np.clip(post_e, e_grid_l[0], e_grid_l[-1])
        idx_lo = np.clip(np.searchsorted(e_grid_l, post_e, side="right") - 1, 0, n_e - 2)
        idx_hi = idx_lo + 1
        w_hi = np.clip(
            (post_e - e_grid_l[idx_lo]) / (e_grid_l[idx_hi] - e_grid_l[idx_lo]), 0.0, 1.0
        )
        w_lo = 1.0 - w_hi
        interior = (~ruin).astype(float)
        p_lo = (p_atoms[None, None, :] * w_lo * interior).reshape(n_e * n_sir, n_atoms)
        p_hi = (p_atoms[None, None, :] * w_hi * interior).reshape(n_e * n_sir, n_atoms)
        rows = np.tile(np.repeat(row_tmpl, n_atoms), 2)
        cols = np.concatenate(
            [
                idx_lo.reshape(n_e * n_sir, n_atoms).ravel(),
                idx_hi.reshape(n_e * n_sir, n_atoms).ravel(),
            ]
        )
        data = np.concatenate([p_lo.ravel(), p_hi.ravel()])
        J_i = coo_matrix((data, (rows, cols)), shape=(n_e * n_sir, n_e)).tocsr()
        J_i.sum_duplicates()
        blocks.append(J_i)
    J_big = block_diag(blocks, format="csr")
    return J_big, p_ruin_2d


def build_equity_jump_operator_2d_sizescaled(
    a_grid: np.ndarray,
    e_grid_l: np.ndarray,
    sir_grid_l: np.ndarray,
    component_x_atoms: Sequence[np.ndarray],
    component_p_within: Sequence[np.ndarray],
    component_freq_ref: Sequence[float],
    component_freq_exponent: Sequence[float],
    component_sev_exponent: Sequence[float],
    atr: float,
    reference_revenue: float,
    sev_reference_base: float,
    tower_top: float,
    lae_ratio: float,
    e_floor: float,
    phi_retention: float,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Size-scaled sibling of :func:`build_equity_jump_operator_2d` (issue #1607).

    Identical block-diagonal-in-assets equity-jump construction, except the per-event
    severity atoms and the loss intensity are made **component-aware and per-assets-slice**
    so the catastrophe/large ceiling tracks the exposed value as the firm grows:

    - Per assets-slice ``A_i`` with revenue ``rev_i = A_i * atr``, component ``c`` has
      frequency ``freq_ref_c * (rev_i / reference_revenue) ** freq_exponent_c`` and its
      atom values are scaled by ``(A_i / sev_reference_base) ** sev_exponent_c``
      (attritional ``sev_exponent = 0`` keeps its severity fixed).
    - The per-slice mixture weight of component ``c``'s atoms is ``freq_i_c / lambda_i``
      (``lambda_i = sum_c freq_i_c``), so the intensity is returned per slice as
      ``lambda_2d`` rather than a single scalar power law.

    With ``sev_exponent == 0`` for every component and a shared ``freq_exponent`` this
    reduces **exactly** (to floating-point) to :func:`build_equity_jump_operator_2d` fed
    the frequency-mixture-weighted concatenation of the components, and ``lambda_2d``
    collapses to ``(sum_c freq_ref_c) * (rev_i / reference_revenue) ** freq_exponent`` --
    the scalar power law :func:`make_jump_term_2d` uses on the flat path.

    Because a loss reduces EQUITY only (assets fixed at impact), severity scaling changes
    only the within-slice equity shift ``L_ret / A_i``; the operator stays block-diagonal
    across assets (no mass moves between assets-slices).

    Args:
        a_grid, e_grid_l, sir_grid_l: State (assets, equity-ratio) and control (SIR) grids;
            see :func:`build_equity_jump_operator_2d`.
        component_x_atoms: Per-component reference (unscaled) severity atom values.
        component_p_within: Per-component within-component atom probabilities (each sums to 1).
        component_freq_ref: Per-component frequency (events/yr) at ``reference_revenue``.
        component_freq_exponent: Per-component revenue exponent for frequency scaling.
        component_sev_exponent: Per-component size exponent for severity scaling (``0`` = fixed).
        atr: Asset-turnover ratio (``revenue = A * atr``).
        reference_revenue: Reference revenue for frequency scaling.
        sev_reference_base: Reference asset base for severity scaling (e.g. initial assets).
        tower_top, lae_ratio, e_floor, phi_retention: Retained-loss parameters; see
            :func:`build_equity_jump_operator_2d`.

    Returns:
        ``(J_big, p_ruin_2d, lambda_2d)``. ``J_big`` and ``p_ruin_2d`` match the flat
        function's shapes/contract; ``lambda_2d`` has shape ``(n_a,)`` -- the loss intensity
        per assets-slice (pass it to :func:`make_jump_term_2d` via ``lambda_2d=``).

    Raises:
        ValueError: if the per-component sequences do not all share one length.
    """
    from scipy.sparse import block_diag, coo_matrix

    n_comp = len(component_x_atoms)
    if not (
        len(component_p_within)
        == len(component_freq_ref)
        == len(component_freq_exponent)
        == len(component_sev_exponent)
        == n_comp
    ):
        raise ValueError("all per-component sequences must share one length")

    n_a, n_e, n_sir = len(a_grid), len(e_grid_l), len(sir_grid_l)
    # Reference atom values + a component index per atom, concatenated in component order
    # (matching build_loss_atoms' concatenation so the gamma=0 reduction is exact).
    x_ref = np.concatenate([np.asarray(x, dtype=float) for x in component_x_atoms])
    p_within = np.concatenate([np.asarray(p, dtype=float) for p in component_p_within])
    comp_idx = np.concatenate(
        [np.full(len(component_x_atoms[c]), c, dtype=int) for c in range(n_comp)]
    )
    n_atoms = len(x_ref)
    sev_exp_per_atom = np.asarray(component_sev_exponent, dtype=float)[comp_idx]
    freq_ref_arr = np.asarray(component_freq_ref, dtype=float)
    freq_exp_arr = np.asarray(component_freq_exponent, dtype=float)
    row_tmpl = (np.arange(n_e)[:, None] * n_sir + np.arange(n_sir)[None, :]).ravel()

    blocks = []
    p_ruin_2d = np.empty((n_a, n_e, n_sir))
    lambda_2d = np.empty(n_a)
    for i, A_i in enumerate(a_grid):
        rev_i = A_i * atr
        freq_i = freq_ref_arr * (rev_i / reference_revenue) ** freq_exp_arr
        lam_i = float(freq_i.sum())
        lambda_2d[i] = lam_i
        # Per-slice mixture probability of each atom = (freq_i_c / lambda_i) * p_within.
        comp_weight = (freq_i / max(lam_i, 1e-300))[comp_idx]
        p_atoms_i = p_within * comp_weight  # sums to 1
        # Per-slice scaled severities: large/cat by (A_i / sev_ref)^sev_exp; attritional exp 0.
        x_i = x_ref * (A_i / sev_reference_base) ** sev_exp_per_atom
        # GROSS retained (drives the single-loss ruin test) and after-tax retained (the
        # surviving equity reduction); scale the atoms BEFORE the tower-top re-entry so a
        # grown firm's loss above tower_top is retained at scaled dollars.
        L_gross_atom_sir = (1.0 + lae_ratio) * (
            np.minimum(x_i[None, :], sir_grid_l[:, None])
            + np.maximum(x_i[None, :] - tower_top, 0.0)
        )  # (n_sir, n_atoms)
        L_per_atom_sir = phi_retention * L_gross_atom_sir  # (n_sir, n_atoms)
        E_grid_i = e_grid_l * A_i
        raw_post_E_gross = E_grid_i[:, None, None] - L_gross_atom_sir[None, :, :]
        ruin = raw_post_E_gross < e_floor
        p_ruin_2d[i] = np.sum(p_atoms_i[None, None, :] * ruin, axis=2)
        raw_post_E = E_grid_i[:, None, None] - L_per_atom_sir[None, :, :]
        post_e = np.where(ruin, e_grid_l[0], raw_post_E / A_i)
        post_e = np.clip(post_e, e_grid_l[0], e_grid_l[-1])
        idx_lo = np.clip(np.searchsorted(e_grid_l, post_e, side="right") - 1, 0, n_e - 2)
        idx_hi = idx_lo + 1
        w_hi = np.clip(
            (post_e - e_grid_l[idx_lo]) / (e_grid_l[idx_hi] - e_grid_l[idx_lo]), 0.0, 1.0
        )
        w_lo = 1.0 - w_hi
        interior = (~ruin).astype(float)
        p_lo = (p_atoms_i[None, None, :] * w_lo * interior).reshape(n_e * n_sir, n_atoms)
        p_hi = (p_atoms_i[None, None, :] * w_hi * interior).reshape(n_e * n_sir, n_atoms)
        rows = np.tile(np.repeat(row_tmpl, n_atoms), 2)
        cols = np.concatenate(
            [
                idx_lo.reshape(n_e * n_sir, n_atoms).ravel(),
                idx_hi.reshape(n_e * n_sir, n_atoms).ravel(),
            ]
        )
        data = np.concatenate([p_lo.ravel(), p_hi.ravel()])
        J_i = coo_matrix((data, (rows, cols)), shape=(n_e * n_sir, n_e)).tocsr()
        J_i.sum_duplicates()
        blocks.append(J_i)
    J_big = block_diag(blocks, format="csr")
    return J_big, p_ruin_2d, lambda_2d


def make_jump_term_2d(
    a_grid: np.ndarray,
    e_grid_l: np.ndarray,
    sir_grid_l: np.ndarray,
    x_atoms: np.ndarray,
    p_atoms: np.ndarray,
    tower_top: float,
    lae_ratio: float,
    e_floor: float,
    atr_local: float,
    reference_revenue_local: float,
    freq_scaling_exponent_local: float,
    lambda_base_local: float,
    v_ruin: float,
    phi_retention: Optional[float] = None,
    prebuilt: Optional[Tuple[Any, np.ndarray]] = None,
    lambda_2d: Optional[np.ndarray] = None,
) -> Tuple[Callable, Any, np.ndarray]:
    """Build a 2-D ``jump_term(state, control, t, V, grids)`` callback (Option A).

    ``E_X[V(post)] = J_big @ V_flat + p_ruin * v_ruin``. ``J_big @ V_flat`` is
    computed ONCE per distinct ``V`` (cached on object identity) and reused across
    the many per-chunk control-scan calls within a single backward step.

    ``prebuilt=(J_big, p_ruin_2d)`` skips the (state-independent) operator build --
    the V_RUIN robustness sweep reuses the baseline operator and only varies the
    scalar ``v_ruin``, so it need not rebuild the block-diagonal map. When
    ``prebuilt`` is supplied, ``phi_retention`` is ignored (the operator already
    encodes it); otherwise ``phi_retention`` is required.

    The returned callback takes an extra keyword ``interp`` (issue #1612):

    - ``interp="nearest"`` (default): snap state ``(A, e)`` and control ``SIR`` to
      the nearest grid nodes. Exact on grid (states/controls always sit on nodes at
      solve time), so this is the fast solve path and is bit-for-bit unchanged.
    - ``interp="linear"``: continuous off-grid path for diagnostics. ``V_cur`` is
      bilinearly interpolated over ``(A, e)`` and the per-SIR-node jump value
      ``J_big@V + p_ruin*v_ruin`` (including the ``p_ruin*V_RUIN`` ruin-atom term)
      is interpolated trilinearly across ``(A, e, SIR)`` -- linear in ``log(A)``,
      linear in ``e``, linear in ``log(SIR)`` (the axes the nearest path snaps on
      and the operator's post-jump split use). At grid nodes it equals the nearest
      result exactly; between nodes it is continuous.

    Args:
        a_grid, e_grid_l, sir_grid_l: State (assets, equity-ratio) and control (SIR)
            grids; see :func:`build_equity_jump_operator_2d`.
        x_atoms, p_atoms, tower_top, lae_ratio, e_floor: Severity atoms and retained-loss
            parameters; see :func:`build_equity_jump_operator_2d`.
        atr_local: Asset-turnover ratio (revenue = ``A * atr_local``).
        reference_revenue_local: Reference revenue for frequency scaling.
        freq_scaling_exponent_local: Exponent for revenue-scaled loss frequency.
        lambda_base_local: Loss intensity (events/yr) at the reference revenue.
        v_ruin: Terminal log-equity assigned to the ruin atom (``= log(e_floor)``).
        phi_retention: After-tax retention factor; required when ``prebuilt is None``
            (forwarded to :func:`build_equity_jump_operator_2d`), ignored otherwise.
        prebuilt: Optional ``(J_big, p_ruin_2d)`` to reuse instead of rebuilding.
        lambda_2d: Optional per-assets-slice loss intensity ``(n_a,)`` from
            :func:`build_equity_jump_operator_2d_sizescaled` (issue #1607). When provided,
            the callback uses it -- nearest snaps to the assets node, linear interpolates
            in ``log(A)`` -- instead of the scalar power law
            ``lambda_base_local * (A*atr/ref) ** freq_scaling_exponent_local``
            (``lambda_base_local`` / ``freq_scaling_exponent_local`` are then ignored).

    Returns:
        ``(jump_term, J_big, p_ruin_2d)``.

    Raises:
        ValueError: If ``prebuilt is None`` and ``phi_retention is None``; or if the
            callback is invoked with an unknown ``interp`` mode.
    """
    if prebuilt is not None:
        J_big, p_ruin_2d = prebuilt
    else:
        if phi_retention is None:
            raise ValueError(
                "phi_retention is required when prebuilt is None (it scales the "
                "after-tax retained loss baked into the jump operator)"
            )
        J_big, p_ruin_2d = build_equity_jump_operator_2d(
            a_grid,
            e_grid_l,
            sir_grid_l,
            x_atoms,
            p_atoms,
            tower_top,
            lae_ratio,
            e_floor,
            phi_retention,
        )
    p_ruin_flat = p_ruin_2d.ravel()
    n_a, n_e, n_sir = len(a_grid), len(e_grid_l), len(sir_grid_l)
    log_a_grid = np.log(a_grid)
    log_sir_grid = np.log(sir_grid_l)
    cache: dict = {}  # lazily holds {"V": last value_function, "EV": J_big @ V_flat}

    def _nearest_log(values, log_grid):
        log_v = np.log(values)
        idx = np.searchsorted(log_grid, log_v, side="left")
        lo = np.clip(idx - 1, 0, len(log_grid) - 1)
        hi = np.clip(idx, 0, len(log_grid) - 1)
        return np.where(np.abs(log_v - log_grid[hi]) < np.abs(log_v - log_grid[lo]), hi, lo)

    def _nearest_lin(values, grid):
        idx = np.searchsorted(grid, values, side="left")
        lo = np.clip(idx - 1, 0, len(grid) - 1)
        hi = np.clip(idx, 0, len(grid) - 1)
        return np.where(np.abs(values - grid[hi]) < np.abs(values - grid[lo]), hi, lo)

    def _lin_bracket(values, grid):
        """Bracket ``values`` in the sorted ``grid`` with linear blend weights.

        Returns ``(lo, hi, w_lo, w_hi)`` where ``w_hi`` is clamped to ``[0, 1]`` so
        out-of-range queries flat-extrapolate to the edge node (matching the nearest
        path's index clip). At a grid node ``w_hi`` is exactly 0 or 1, so the linear
        blend collapses to the node value.
        """
        v = np.asarray(values, dtype=float)
        lo = np.clip(np.searchsorted(grid, v, side="left") - 1, 0, len(grid) - 2)
        hi = lo + 1
        w_hi = np.clip((v - grid[lo]) / (grid[hi] - grid[lo]), 0.0, 1.0)
        return lo, hi, 1.0 - w_hi, w_hi

    def jump_term(state, control, t, value_function, state_grids, interp="nearest"):
        if interp not in ("nearest", "linear"):
            raise ValueError(f"interp must be 'nearest' or 'linear', got {interp!r}")
        if cache.get("V") is not value_function:
            cache["EV"] = np.asarray(J_big @ value_function.ravel())
            cache["V"] = value_function
        EV_interior_all = cache["EV"]
        A = state[..., 0]
        e = state[..., 1]
        sir = control[..., 0]
        # Loss intensity: per-slice lambda_2d when supplied (issue #1607), else the
        # scalar revenue power law.  lambda_2d wins because under per-component frequency
        # exponents lambda(A) is a SUM of power laws, not a single power.
        lam_powerlaw = (
            lambda_base_local
            * (A * atr_local / reference_revenue_local) ** freq_scaling_exponent_local
        )
        if interp == "nearest":
            a_idx = _nearest_log(A, log_a_grid)
            e_idx = _nearest_lin(e, e_grid_l)
            sir_idx = _nearest_log(sir, log_sir_grid)
            flat_state = a_idx * n_e + e_idx
            row = flat_state * n_sir + sir_idx
            EV_post = EV_interior_all[row] + p_ruin_flat[row] * v_ruin
            V_cur = value_function.ravel()[flat_state]
            lam = lam_powerlaw if lambda_2d is None else lambda_2d[a_idx]
            return lam * (EV_post - V_cur)
        # interp == "linear": continuous off-grid path (issue #1612).
        V_flat = value_function.ravel()
        # Per-node jump value: surviving-mass E_X[V(post)] plus the ruin-atom term.
        # Interpolating the bundle == interpolating the parts (interp is linear).
        Jflat = EV_interior_all + p_ruin_flat * v_ruin
        aL, aH, waL, waH = _lin_bracket(np.log(A), log_a_grid)
        eL, eH, weL, weH = _lin_bracket(e, e_grid_l)
        sL, sH, wsL, wsH = _lin_bracket(np.log(sir), log_sir_grid)
        # Bilinear V_cur over the 4 (a, e) corners (C-order flat index a*n_e + e).
        V_cur = (
            waL * weL * np.take(V_flat, aL * n_e + eL)
            + waL * weH * np.take(V_flat, aL * n_e + eH)
            + waH * weL * np.take(V_flat, aH * n_e + eL)
            + waH * weH * np.take(V_flat, aH * n_e + eH)
        )
        # Trilinear EV_post over the 8 (a, e, sir) corners
        # (flat index ((a*n_e + e)*n_sir + sir)).
        EV_post = 0.0
        for a_idx, w_a in ((aL, waL), (aH, waH)):
            for e_idx, w_e in ((eL, weL), (eH, weH)):
                base = (a_idx * n_e + e_idx) * n_sir
                for s_idx, w_s in ((sL, wsL), (sH, wsH)):
                    EV_post = EV_post + w_a * w_e * w_s * np.take(Jflat, base + s_idx)
        # Intensity: interpolate lambda_2d in log(A) (issue #1607), else the power law.
        lam = lam_powerlaw if lambda_2d is None else (waL * lambda_2d[aL] + waH * lambda_2d[aH])
        return lam * (EV_post - V_cur)

    return jump_term, J_big, p_ruin_2d
