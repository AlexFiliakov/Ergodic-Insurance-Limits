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

from typing import Sequence, Tuple

import numpy as np

__all__ = [
    "severity_cdf",
    "lognormal_gauss_hermite_nodes",
    "pareto_stratified_atoms",
    "component_atoms",
    "build_loss_atoms",
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
