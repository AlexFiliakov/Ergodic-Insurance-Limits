"""Quick-start factory for common actuarial workflows.

Provides ``run_analysis()`` — a single-function entry point that bundles
company setup, loss generation, insurance configuration, Monte Carlo
simulation, and ergodic comparison into one call.

Examples:
    Minimal usage::

        from ergodic_insurance import run_analysis

        results = run_analysis(
            initial_assets=10_000_000,
            loss_frequency=2.5,
            loss_severity_mean=1_000_000,
            deductible=500_000,
            coverage_limit=10_000_000,
            premium_rate=0.025,
        )
        print(results.summary())

    With custom simulation parameters::

        results = run_analysis(
            initial_assets=50_000_000,
            operating_margin=0.12,
            loss_frequency=1.0,
            loss_severity_mean=5_000_000,
            deductible=1_000_000,
            coverage_limit=25_000_000,
            premium_rate=0.02,
            n_simulations=500,
            time_horizon=30,
            seed=42,
        )
        df = results.to_dataframe()

Since:
    Version 0.5.0
"""

import copy
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config, ManufacturerConfig
from .ergodic_analyzer import ErgodicAnalyzer
from .insurance import InsurancePolicy
from .loss_distributions import ManufacturingLossGenerator
from .manufacturer import WidgetManufacturer
from .simulation import Simulation, SimulationResults

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResults:
    """Container for ``run_analysis()`` output.

    Holds insured and uninsured simulation results together with the
    ergodic comparison, and provides convenience methods for inspection.

    Attributes:
        insured_results: List of SimulationResults from insured runs.
        uninsured_results: List of SimulationResults from uninsured runs
            (empty when *compare_uninsured* was ``False``).
        comparison: Ergodic comparison dict from
            :meth:`ErgodicAnalyzer.compare_scenarios`, or ``None`` when
            *compare_uninsured* was ``False``.
        config: The Config used for the simulations.
        insurance_policy: The InsurancePolicy used for insured runs.

    Examples:
        Quick inspection::

            results = run_analysis(...)
            print(results.summary())

        Export to DataFrame::

            df = results.to_dataframe()
            df.to_csv("analysis.csv", index=False)

        Visualize::

            results.plot()
    """

    insured_results: List[SimulationResults]
    uninsured_results: List[SimulationResults]
    comparison: Optional[Dict[str, Any]]
    config: Config
    insurance_policy: InsurancePolicy

    # Cached summary text
    _summary_cache: Optional[str] = field(default=None, repr=False)

    def summary(self) -> str:
        """Return a human-readable summary of the analysis.

        Includes survival rates, growth metrics, and the ergodic
        advantage of insurance.

        Returns:
            str: Multi-line formatted summary.
        """
        if self._summary_cache is not None:
            return self._summary_cache

        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("Ergodic Insurance Analysis Summary")
        lines.append("=" * 60)

        n_sims = len(self.insured_results)
        lines.append(f"Simulations: {n_sims}")

        # Determine time horizon from first result
        if self.insured_results:
            horizon = len(self.insured_results[0].years)
            lines.append(f"Time Horizon: {horizon} years")

        # --- Insured scenario ---
        lines.append("")
        lines.append("--- Insured Scenario ---")
        insured_survived = sum(1 for r in self.insured_results if r.insolvency_year is None)
        lines.append(
            f"Survival Rate: {insured_survived}/{n_sims} " f"({insured_survived / n_sims:.1%})"
        )

        insured_final_equity = [
            float(r.equity[-1]) for r in self.insured_results if r.insolvency_year is None
        ]
        if insured_final_equity:
            lines.append(
                f"Mean Final Equity (survivors): " f"${np.mean(insured_final_equity):,.0f}"
            )
            lines.append(
                f"Median Final Equity (survivors): " f"${np.median(insured_final_equity):,.0f}"
            )

        insured_tw_roe = [r.calculate_time_weighted_roe() for r in self.insured_results]
        lines.append(f"Mean Time-Weighted ROE: {np.mean(insured_tw_roe):.2%}")

        premium = self.insurance_policy.calculate_premium()
        lines.append(f"Annual Premium: ${premium:,.0f}")

        # --- Uninsured scenario ---
        if self.uninsured_results:
            lines.append("")
            lines.append("--- Uninsured Scenario ---")
            uninsured_survived = sum(1 for r in self.uninsured_results if r.insolvency_year is None)
            lines.append(
                f"Survival Rate: {uninsured_survived}/{n_sims} "
                f"({uninsured_survived / n_sims:.1%})"
            )

            uninsured_final_equity = [
                float(r.equity[-1]) for r in self.uninsured_results if r.insolvency_year is None
            ]
            if uninsured_final_equity:
                lines.append(
                    f"Mean Final Equity (survivors): " f"${np.mean(uninsured_final_equity):,.0f}"
                )

            uninsured_tw_roe = [r.calculate_time_weighted_roe() for r in self.uninsured_results]
            lines.append(f"Mean Time-Weighted ROE: {np.mean(uninsured_tw_roe):.2%}")

        # --- Ergodic advantage ---
        if self.comparison is not None:
            adv = self.comparison.get("ergodic_advantage", {})
            lines.append("")
            lines.append("--- Ergodic Advantage (Insured - Uninsured) ---")
            ta_gain = adv.get("time_average_gain")
            if ta_gain is not None:
                lines.append(f"Time-Average Growth Gain: {ta_gain:+.2%}")
            surv_gain = adv.get("survival_gain")
            if surv_gain is not None:
                lines.append(f"Survival Rate Gain: {surv_gain:+.1%}")
            sig = adv.get("significant")
            if sig is not None:
                lines.append(f"Statistically Significant: {'Yes' if sig else 'No'}")

        lines.append("=" * 60)
        text = "\n".join(lines)
        self._summary_cache = text
        return text

    def to_dataframe(self) -> pd.DataFrame:
        """Export per-simulation summary metrics to a DataFrame.

        Returns:
            pd.DataFrame: One row per simulation with columns for
            scenario, survival, final equity, and time-weighted ROE.
        """
        rows: List[Dict[str, Any]] = []

        for i, r in enumerate(self.insured_results):
            rows.append(
                {
                    "scenario": "insured",
                    "simulation": i,
                    "survived": r.insolvency_year is None,
                    "insolvency_year": r.insolvency_year,
                    "final_assets": float(r.assets[-1]),
                    "final_equity": float(r.equity[-1]),
                    "mean_roe": float(np.nanmean(r.roe)),
                    "time_weighted_roe": r.calculate_time_weighted_roe(),
                    "total_claims": float(np.sum(r.claim_amounts)),
                }
            )

        for i, r in enumerate(self.uninsured_results):
            rows.append(
                {
                    "scenario": "uninsured",
                    "simulation": i,
                    "survived": r.insolvency_year is None,
                    "insolvency_year": r.insolvency_year,
                    "final_assets": float(r.assets[-1]),
                    "final_equity": float(r.equity[-1]),
                    "mean_roe": float(np.nanmean(r.roe)),
                    "time_weighted_roe": r.calculate_time_weighted_roe(),
                    "total_claims": float(np.sum(r.claim_amounts)),
                }
            )

        return pd.DataFrame(rows)

    def plot(self, show: bool = True):
        """Generate a quick comparison visualization.

        Creates a 2x2 figure comparing insured vs uninsured outcomes:
        survival rate, final equity distribution, time-weighted ROE
        distribution, and equity trajectory fan chart.

        Args:
            show: If ``True``, call ``plt.show()``. Set to ``False``
                when saving the figure or embedding in notebooks.

        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        try:
            import matplotlib.pyplot as plt  # noqa: E402
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plotting. " "Install it with: pip install matplotlib"
            ) from exc

        has_uninsured = bool(self.uninsured_results)
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle("Ergodic Insurance Analysis", fontsize=14, fontweight="bold")

        # --- 1. Survival rates ---
        ax = axes[0, 0]
        insured_surv = sum(1 for r in self.insured_results if r.insolvency_year is None) / len(
            self.insured_results
        )
        labels = ["Insured"]
        values = [insured_surv * 100]
        colors = ["#2196F3"]
        if has_uninsured:
            uninsured_surv = sum(
                1 for r in self.uninsured_results if r.insolvency_year is None
            ) / len(self.uninsured_results)
            labels.append("Uninsured")
            values.append(uninsured_surv * 100)
            colors.append("#FF5722")
        ax.bar(labels, values, color=colors, alpha=0.8)
        ax.set_ylabel("Survival Rate (%)")
        ax.set_title("Survival Rate")
        ax.set_ylim(0, 105)
        for j, v in enumerate(values):
            ax.text(j, v + 1, f"{v:.1f}%", ha="center", fontsize=10)

        # --- 2. Final equity distribution ---
        ax = axes[0, 1]
        insured_eq = [float(r.equity[-1]) / 1e6 for r in self.insured_results]
        ax.hist(insured_eq, bins=30, alpha=0.6, label="Insured", color="#2196F3")
        if has_uninsured:
            uninsured_eq = [float(r.equity[-1]) / 1e6 for r in self.uninsured_results]
            ax.hist(
                uninsured_eq,
                bins=30,
                alpha=0.6,
                label="Uninsured",
                color="#FF5722",
            )
        ax.set_xlabel("Final Equity ($M)")
        ax.set_ylabel("Count")
        ax.set_title("Final Equity Distribution")
        ax.legend()

        # --- 3. Time-weighted ROE distribution ---
        ax = axes[1, 0]
        insured_roe = [r.calculate_time_weighted_roe() * 100 for r in self.insured_results]
        ax.hist(insured_roe, bins=30, alpha=0.6, label="Insured", color="#2196F3")
        if has_uninsured:
            uninsured_roe = [r.calculate_time_weighted_roe() * 100 for r in self.uninsured_results]
            ax.hist(
                uninsured_roe,
                bins=30,
                alpha=0.6,
                label="Uninsured",
                color="#FF5722",
            )
        ax.set_xlabel("Time-Weighted ROE (%)")
        ax.set_ylabel("Count")
        ax.set_title("Time-Weighted ROE Distribution")
        ax.legend()

        # --- 4. Equity fan chart (percentile bands) ---
        ax = axes[1, 1]
        self._plot_fan_chart(ax, self.insured_results, "#2196F3", "Insured")
        if has_uninsured:
            self._plot_fan_chart(ax, self.uninsured_results, "#FF5722", "Uninsured")
        ax.set_xlabel("Year")
        ax.set_ylabel("Equity ($M)")
        ax.set_title("Equity Trajectories (Median + 25th-75th)")
        ax.legend()

        plt.tight_layout()
        if show:
            plt.show()
        return fig

    @staticmethod
    def _plot_fan_chart(
        ax,
        results: List[SimulationResults],
        color: str,
        label: str,
    ) -> None:
        """Plot median line with percentile band on *ax*."""
        # Build matrix of equity paths, padding short paths with NaN
        max_len = max(len(r.equity) for r in results)
        matrix = np.full((len(results), max_len), np.nan)
        for i, r in enumerate(results):
            matrix[i, : len(r.equity)] = r.equity

        years = np.arange(max_len)
        median = np.nanmedian(matrix, axis=0) / 1e6
        p25 = np.nanpercentile(matrix, 25, axis=0) / 1e6
        p75 = np.nanpercentile(matrix, 75, axis=0) / 1e6

        ax.plot(years, median, color=color, label=label, linewidth=2)
        ax.fill_between(years, p25, p75, color=color, alpha=0.15)


def run_analysis(
    # Company parameters
    initial_assets: float = 10_000_000,
    operating_margin: float = 0.08,
    # Loss parameters
    loss_frequency: float = 2.5,
    loss_severity_mean: float = 1_000_000,
    loss_severity_std: Optional[float] = None,
    # Insurance parameters
    deductible: float = 500_000,
    coverage_limit: float = 10_000_000,
    premium_rate: float = 0.025,
    # Simulation parameters
    n_simulations: int = 1000,
    time_horizon: int = 20,
    seed: Optional[int] = None,
    # Advanced options
    growth_rate: float = 0.05,
    tax_rate: float = 0.25,
    compare_uninsured: bool = True,
) -> AnalysisResults:
    """Run a complete insured-vs-uninsured ergodic analysis.

    This is the recommended quick-start entry point.  It builds all
    internal objects (Config, Manufacturer, LossGenerator, Insurance,
    Simulation, ErgodicAnalyzer), runs *n_simulations* Monte Carlo
    paths for both the insured and uninsured scenarios, and returns a
    rich :class:`AnalysisResults` container.

    Args:
        initial_assets: Company starting assets in dollars.
        operating_margin: Annual operating margin as a decimal
            (e.g. 0.08 for 8%).
        loss_frequency: Expected number of losses per year
            (Poisson lambda).
        loss_severity_mean: Mean loss size in dollars.
        loss_severity_std: Standard deviation of loss size.
            Defaults to *loss_severity_mean* if not provided, implying a
            coefficient of variation (CV) of 1.0.  Typical CV ranges by
            line of business: property 0.5–1.5, general liability 1.0–3.0,
            workers' compensation 1.0–2.0.  Set explicitly when your loss
            data suggests a different CV.
        deductible: Self-insured retention in dollars.
        coverage_limit: Maximum insurance payout per occurrence.
        premium_rate: Annual premium as a fraction of
            *coverage_limit* (e.g. 0.025 for 2.5%).
        n_simulations: Number of Monte Carlo paths to run.
        time_horizon: Simulation length in years.
        seed: Base random seed for reproducibility.
        growth_rate: Annual revenue growth rate.
        tax_rate: Corporate tax rate.
        compare_uninsured: If ``True`` (default), also run an uninsured
            scenario and compute the ergodic comparison.

    Returns:
        AnalysisResults: Container with simulation results, comparison
        metrics, and convenience methods (``.summary()``,
        ``.to_dataframe()``, ``.plot()``).

    Examples:
        Basic usage::

            from ergodic_insurance import run_analysis

            results = run_analysis(
                initial_assets=10_000_000,
                loss_frequency=2.5,
                loss_severity_mean=1_000_000,
                deductible=500_000,
                coverage_limit=10_000_000,
                premium_rate=0.025,
            )
            print(results.summary())

        Minimal (all defaults)::

            results = run_analysis()
            print(results.summary())

        Export and plot::

            df = results.to_dataframe()
            df.to_csv("results.csv")
            results.plot()
    """
    if loss_severity_std is None:
        loss_severity_std = loss_severity_mean
        logger.info(
            "loss_severity_std not provided; defaulting to loss_severity_mean " "(CV=1.0): %.2f",
            loss_severity_std,
        )

    # --- Build configuration ---
    config = Config.from_company(
        initial_assets=initial_assets,
        operating_margin=operating_margin,
        tax_rate=tax_rate,
        growth_rate=growth_rate,
        time_horizon_years=time_horizon,
    )

    # --- Build insurance policy ---
    policy = InsurancePolicy.from_simple(
        deductible=deductible,
        limit=coverage_limit,
        premium_rate=premium_rate,
    )

    # --- Run insured simulations ---
    logger.info(
        "Running %d insured simulations over %d years ...",
        n_simulations,
        time_horizon,
    )
    insured_results = _run_batch(
        config=config,
        loss_frequency=loss_frequency,
        loss_severity_mean=loss_severity_mean,
        loss_severity_std=loss_severity_std,
        insurance_policy=policy,
        n_simulations=n_simulations,
        time_horizon=time_horizon,
        growth_rate=growth_rate,
        base_seed=seed,
    )

    # --- Run uninsured simulations ---
    uninsured_results: List[SimulationResults] = []
    comparison: Optional[Dict[str, Any]] = None

    if compare_uninsured:
        logger.info(
            "Running %d uninsured simulations over %d years ...",
            n_simulations,
            time_horizon,
        )
        uninsured_results = _run_batch(
            config=config,
            loss_frequency=loss_frequency,
            loss_severity_mean=loss_severity_mean,
            loss_severity_std=loss_severity_std,
            insurance_policy=None,
            n_simulations=n_simulations,
            time_horizon=time_horizon,
            growth_rate=growth_rate,
            base_seed=seed,
        )

        # --- Ergodic comparison ---
        analyzer = ErgodicAnalyzer()
        comparison = analyzer.compare_scenarios(insured_results, uninsured_results, metric="equity")

    return AnalysisResults(
        insured_results=insured_results,
        uninsured_results=uninsured_results,
        comparison=comparison,
        config=config,
        insurance_policy=policy,
    )


def _run_batch(
    *,
    config: Config,
    loss_frequency: float,
    loss_severity_mean: float,
    loss_severity_std: float,
    insurance_policy: Optional[InsurancePolicy],
    n_simulations: int,
    time_horizon: int,
    growth_rate: float,
    base_seed: Optional[int],
) -> List[SimulationResults]:
    """Run *n_simulations* independent single-path simulations.

    Each simulation gets a unique seed derived from *base_seed* so
    that insured and uninsured batches share the same loss sequences
    for fair comparison when both use the same *base_seed*.
    """
    results: List[SimulationResults] = []

    for i in range(n_simulations):
        sim_seed = (base_seed + i) if base_seed is not None else None

        manufacturer = WidgetManufacturer(config.manufacturer)
        loss_gen = ManufacturingLossGenerator.create_simple(
            frequency=loss_frequency,
            severity_mean=loss_severity_mean,
            severity_std=loss_severity_std,
            seed=sim_seed,
        )

        sim = Simulation(
            manufacturer=manufacturer,
            loss_generator=loss_gen,
            insurance_policy=insurance_policy,
            time_horizon=time_horizon,
            seed=sim_seed,
            growth_rate=growth_rate,
        )
        results.append(sim.run())

    return results
