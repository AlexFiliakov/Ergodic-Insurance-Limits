"""Simulation engine for time evolution of widget manufacturer model.

This module provides the main simulation engine that orchestrates the
time evolution of the widget manufacturer financial model, managing
loss events, financial calculations, and result collection.

The simulation framework supports both single-path and Monte Carlo simulations,
enabling comprehensive analysis of insurance strategies and business outcomes
under uncertainty. It tracks detailed financial metrics, processes insurance claims,
and handles bankruptcy conditions appropriately.

Key Features:
    - Single-path trajectory simulation with detailed metrics
    - Monte Carlo simulation support through integration
    - Insurance claim processing with policy application
    - Financial statement tracking and ROE calculation
    - Bankruptcy detection and proper termination
    - Comprehensive result analysis and export capabilities

Examples:
    Basic simulation::

        from ergodic_insurance import Simulation, Config
        from ergodic_insurance.manufacturer import WidgetManufacturer
        from ergodic_insurance.loss_distributions import ManufacturingLossGenerator

        config = Config()
        manufacturer = WidgetManufacturer(config.manufacturer)
        loss_generator = ManufacturingLossGenerator.create_simple(
            frequency=0.1, severity_mean=5_000_000, seed=42
        )

        sim = Simulation(
            manufacturer=manufacturer,
            loss_generator=loss_generator,
            time_horizon=50
        )
        results = sim.run()

        print(f"Mean ROE: {results.summary_stats()['mean_roe']:.2%}")

Note:
    This module is thread-safe for parallel Monte Carlo simulations when
    each thread has its own Simulation instance.

Since:
    Version 0.1.0
"""

from copy import deepcopy
from dataclasses import dataclass
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Union
import warnings

import numpy as np
import pandas as pd

from ._compare_strategies import StrategyComparisonResult  # re-export for backward compat
from ._compare_strategies import compare_strategies as _compare_strategies_func
from ._compare_strategies import run_monte_carlo as _run_monte_carlo_func
from ._warnings import ErgodicInsuranceDeprecationWarning
from .config import DEFAULT_RISK_FREE_RATE, Config
from .decimal_utils import ZERO, to_decimal
from .insurance import InsurancePolicy
from .insurance_program import EnhancedInsuranceLayer, InsuranceProgram
from .loss_distributions import LossData, LossEvent, ManufacturingLossGenerator
from .manufacturer import WidgetManufacturer
from .monte_carlo import MonteCarloEngine

logger = logging.getLogger(__name__)


@dataclass
class SimulationResults:
    """Container for simulation trajectory data.

    Holds the complete time series of financial metrics and events
    from a single simulation run, with methods for analysis and export.

    This dataclass provides comprehensive storage for all simulation outputs
    and includes utility methods for calculating derived metrics, performing
    statistical analysis, and exporting data for further processing.

    Attributes:
        years: Array of simulation years (0 to time_horizon-1).
        assets: Total assets at each year.
        equity: Shareholder equity at each year.
        roe: Return on equity for each year.
        revenue: Annual revenue for each year.
        net_income: Annual net income for each year.
        claim_counts: Number of claims in each year.
        claim_amounts: Total claim amount in each year.
        insolvency_year: Year when bankruptcy occurred (None if survived).

    Examples:
        Analyzing simulation results::

            results = simulation.run()

            # Get summary statistics
            stats = results.summary_stats()
            print(f"Survival: {stats['survived']}")
            print(f"Mean ROE: {stats['mean_roe']:.2%}")

            # Export to DataFrame
            df = results.to_dataframe()
            df.to_csv('simulation_results.csv')

            # Calculate volatility metrics
            volatility = results.calculate_roe_volatility()
            print(f"ROE Sharpe Ratio: {volatility['roe_sharpe']:.2f}")

    Note:
        All financial values are in nominal dollars without inflation adjustment.
        ROE calculations handle edge cases like zero equity appropriately.
    """

    years: np.ndarray
    assets: np.ndarray
    equity: np.ndarray
    roe: np.ndarray
    revenue: np.ndarray
    net_income: np.ndarray
    claim_counts: np.ndarray
    claim_amounts: np.ndarray
    insolvency_year: Optional[int] = None

    @property
    def survived(self) -> bool:
        """Whether the entity survived the full simulation without insolvency."""
        return self.insolvency_year is None

    @property
    def n_years(self) -> int:
        """Number of simulation years."""
        return len(self.years)

    @property
    def mean_roe(self) -> float:
        """Arithmetic mean of non-NaN ROE values."""
        valid = self.roe[~np.isnan(self.roe)]
        return float(np.mean(valid)) if len(valid) > 0 else 0.0

    @property
    def final_equity(self) -> float:
        """Equity at the end of the simulation."""
        return float(self.equity[-1])

    @property
    def final_assets(self) -> float:
        """Total assets at the end of the simulation."""
        return float(self.assets[-1])

    @property
    def total_claims(self) -> float:
        """Cumulative claim amounts over the simulation."""
        return float(np.sum(self.claim_amounts))

    def __repr__(self) -> str:
        status = "survived" if self.survived else f"insolvent@yr{self.insolvency_year}"
        return (
            f"SimulationResults(n_years={self.n_years}, {status}, "
            f"mean_roe={self.mean_roe:.2%}, "
            f"final_equity=${self.final_equity:,.0f})"
        )

    def __str__(self) -> str:
        lines = [
            f"SimulationResults — {self.n_years} years",
            f"  Survived: {self.survived}"
            + (f"  (insolvent year {self.insolvency_year})" if not self.survived else ""),
            f"  Mean ROE: {self.mean_roe:.2%}",
            f"  Final Equity: ${self.final_equity:,.0f}",
            f"  Final Assets: ${self.final_assets:,.0f}",
            f"  Total Claims: ${self.total_claims:,.0f}",
        ]
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        status_color = "#27ae60" if self.survived else "#e74c3c"
        status_text = "Survived" if self.survived else f"Insolvent (year {self.insolvency_year})"
        return (
            "<div style='font-family: monospace; padding: 8px; "
            "border: 1px solid #ddd; border-radius: 4px; display: inline-block;'>"
            "<b>SimulationResults</b><br>"
            f"Years: {self.n_years} &nbsp;|&nbsp; "
            f"Status: <span style='color:{status_color}'>{status_text}</span><br>"
            f"Mean ROE: {self.mean_roe:.2%} &nbsp;|&nbsp; "
            f"Final Equity: ${self.final_equity:,.0f}<br>"
            f"Final Assets: ${self.final_assets:,.0f} &nbsp;|&nbsp; "
            f"Total Claims: ${self.total_claims:,.0f}"
            "</div>"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert simulation results to pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame with columns for year, assets, equity, roe,
                revenue, net_income, claim_count, and claim_amount.

        Examples:
            Export to Excel::

                df = results.to_dataframe()
                df.to_excel('results.xlsx', index=False)
        """
        return pd.DataFrame(
            {
                "year": self.years,
                "assets": self.assets,
                "equity": self.equity,
                "roe": self.roe,
                "revenue": self.revenue,
                "net_income": self.net_income,
                "claim_count": self.claim_counts,
                "claim_amount": self.claim_amounts,
            }
        )

    def calculate_time_weighted_roe(self) -> float:
        """Calculate time-weighted average ROE.

        Time-weighted ROE gives equal weight to each period regardless
        of the equity level, providing a better measure of consistent
        performance over time. Uses geometric mean for proper compounding.

        Returns:
            float: Time-weighted average ROE as a decimal (e.g., 0.08 for 8%).

        Note:
            This method uses geometric mean of growth factors (1 + ROE) to
            properly account for compounding effects. NaN values are excluded
            from the calculation.

        Examples:
            Compare different ROE measures::

                simple_avg = np.mean(results.roe)
                time_weighted = results.calculate_time_weighted_roe()
                print(f"Simple average: {simple_avg:.2%}")
                print(f"Time-weighted: {time_weighted:.2%}")
        """
        valid_roe = self.roe[~np.isnan(self.roe)]
        if len(valid_roe) == 0:
            return 0.0

        # For time-weighted average, we use geometric mean for compounding returns
        # Guard against total loss scenarios where ROE <= -1 would cause log domain errors
        # Clip to -0.99 (99% loss) to maintain valid growth factors >= 0.01
        clipped_roe = np.clip(valid_roe, -0.99, None)

        # Convert ROE to growth factors (1 + roe)
        growth_factors = 1 + clipped_roe

        # Geometric mean minus 1 to get average return
        time_weighted_roe = np.exp(np.mean(np.log(growth_factors))) - 1
        return float(time_weighted_roe)

    def calculate_rolling_roe(self, window: int) -> np.ndarray:
        """Calculate rolling window ROE.

        Args:
            window: Window size in years (e.g., 1, 3, 5). Must be positive
                and not exceed the data length.

        Returns:
            np.ndarray: Array of rolling ROE values. Values are NaN for
                positions where the full window is not available.

        Raises:
            ValueError: If window size exceeds data length.

        Examples:
            Calculate and plot rolling ROE::

                rolling_3yr = results.calculate_rolling_roe(3)
                plt.plot(results.years, rolling_3yr, label='3-Year Rolling ROE')
                plt.axhline(y=0.08, color='r', linestyle='--', label='Target')
        """
        if window > len(self.roe):
            raise ValueError(f"Window {window} larger than data length {len(self.roe)}")

        rolling_roe = np.full(len(self.roe), np.nan)

        for i in range(window - 1, len(self.roe)):
            window_data = self.roe[i - window + 1 : i + 1]
            valid_data = window_data[~np.isnan(window_data)]
            if len(valid_data) > 0:
                rolling_roe[i] = np.mean(valid_data)

        return rolling_roe

    def calculate_roe_components(
        self, base_operating_margin: float = 0.08, tax_rate: float = 0.25
    ) -> Dict[str, np.ndarray]:
        """Calculate ROE component breakdown.

        Decomposes ROE into operating, insurance, and tax components
        using DuPont-style analysis. This helps identify the drivers
        of ROE performance and the impact of insurance decisions.

        Args:
            base_operating_margin: Baseline operating margin for the business.
                Defaults to 0.08 (8%). Can be sourced from
                manufacturer.config.base_operating_margin.
            tax_rate: Corporate tax rate. Defaults to 0.25 (25%). Can be
                sourced from manufacturer.config.tax_rate.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing:
                - 'operating_roe': Base business ROE without claims
                - 'insurance_impact': ROE reduction from claims/premiums
                - 'tax_effect': Impact of taxes on ROE
                - 'total_roe': Actual ROE for reference

        Note:
            This is a simplified decomposition. Actual implementation would
            require more detailed financial data for precise attribution.

        Examples:
            Analyze ROE drivers::

                components = results.calculate_roe_components()
                operating_avg = np.mean(components['operating_roe'])
                insurance_drag = np.mean(components['insurance_impact'])
                print(f"Operating ROE: {operating_avg:.2%}")
                print(f"Insurance drag: {insurance_drag:.2%}")

            Using manufacturer config values::

                components = results.calculate_roe_components(
                    base_operating_margin=manufacturer.config.base_operating_margin,
                    tax_rate=manufacturer.config.tax_rate,
                )
        """
        components = {
            "operating_roe": np.zeros(len(self.years)),
            "insurance_impact": np.zeros(len(self.years)),
            "tax_effect": np.zeros(len(self.years)),
            "total_roe": self.roe.copy(),
        }

        # Calculate components based on available data
        for i in range(len(self.years)):
            if self.equity[i] > 0 and not np.isnan(self.roe[i]):
                # Operating ROE = (Revenue - Operating Costs) / Equity
                # This is a simplified calculation; actual implementation
                # would need more detailed financial data
                base_margin = base_operating_margin
                components["operating_roe"][i] = (self.revenue[i] * base_margin) / self.equity[i]

                # Insurance impact = reduction in ROE due to premiums and retained losses
                if self.claim_amounts[i] > 0:
                    components["insurance_impact"][i] = -self.claim_amounts[i] / self.equity[i]

                # Tax effect (simplified)
                components["tax_effect"][i] = self.roe[i] * (1 - tax_rate) - self.roe[i]

        return components

    def calculate_roe_volatility(self) -> Dict[str, float]:
        """Calculate ROE volatility metrics.

        Computes various risk-adjusted performance metrics for ROE,
        including standard deviation, downside deviation, Sharpe ratio,
        and coefficient of variation.

        Returns:
            Dict[str, float]: Dictionary containing:
                - 'roe_std': Standard deviation of ROE
                - 'roe_downside_deviation': Downside deviation from mean
                - 'roe_sharpe': Sharpe ratio using 2% risk-free rate
                - 'roe_coefficient_variation': Coefficient of variation (std/mean)

        Note:
            Returns zeros for all metrics if insufficient data (< 2 observations).
            Sharpe ratio uses a 2% risk-free rate assumption.

        Examples:
            Risk-adjusted performance analysis::

                volatility = results.calculate_roe_volatility()
                if volatility['roe_sharpe'] > 1.0:
                    print("Strong risk-adjusted performance")
                print(f"Downside risk: {volatility['roe_downside_deviation']:.2%}")
        """
        valid_roe = self.roe[~np.isnan(self.roe)]
        if len(valid_roe) < 2:
            return {
                "roe_std": 0.0,
                "roe_downside_deviation": 0.0,
                "roe_sharpe": 0.0,
                "roe_coefficient_variation": 0.0,
            }

        mean_roe = np.mean(valid_roe)
        std_roe = np.std(valid_roe, ddof=1)

        # Downside deviation (over all observations)
        downside_deviations = np.minimum(valid_roe - mean_roe, 0)
        downside_dev = np.sqrt(np.mean(downside_deviations**2))

        # Sharpe ratio equivalent for ROE
        risk_free_rate = DEFAULT_RISK_FREE_RATE
        sharpe = (mean_roe - risk_free_rate) / std_roe if std_roe > 0 else 0.0

        # Coefficient of variation
        cv = std_roe / abs(mean_roe) if mean_roe != 0 else float("inf")

        return {
            "roe_std": std_roe,
            "roe_downside_deviation": downside_dev,
            "roe_sharpe": sharpe,
            "roe_coefficient_variation": cv,
        }

    def summary_stats(self) -> Dict[str, float]:
        """Calculate summary statistics for the simulation.

        Computes comprehensive summary statistics including ROE metrics,
        rolling averages, volatility measures, and survival indicators.

        Returns:
            Dict[str, float]: Dictionary containing:
                - Basic ROE metrics (mean, std, median, time-weighted)
                - Rolling averages (1, 3, 5 year)
                - Final state (assets, equity)
                - Claims statistics (total, frequency)
                - Survival indicators (survived, insolvency_year)
                - Volatility metrics (from calculate_roe_volatility)

        Examples:
            Generate summary report::

                stats = results.summary_stats()

                print("Performance Summary:")
                print(f"  Mean ROE: {stats['mean_roe']:.2%}")
                print(f"  Volatility: {stats['std_roe']:.2%}")
                print(f"  Sharpe Ratio: {stats['roe_sharpe']:.2f}")

                print("\nRisk Summary:")
                print(f"  Survived: {stats['survived']}")
                print(f"  Total Claims: ${stats['total_claims']:,.0f}")
        """
        # Filter out NaN values for ROE calculation
        valid_roe = self.roe[~np.isnan(self.roe)]

        # Calculate rolling ROE for different windows
        rolling_3yr = self.calculate_rolling_roe(3) if len(self.years) >= 3 else np.array([])
        rolling_5yr = self.calculate_rolling_roe(5) if len(self.years) >= 5 else np.array([])

        # Get volatility metrics
        volatility_metrics = self.calculate_roe_volatility()

        base_stats = {
            "mean_roe": np.mean(valid_roe) if len(valid_roe) > 0 else 0.0,
            "std_roe": np.std(valid_roe, ddof=1) if len(valid_roe) > 1 else 0.0,
            "median_roe": np.median(valid_roe) if len(valid_roe) > 0 else 0.0,
            "time_weighted_roe": self.calculate_time_weighted_roe(),
            "roe_3yr_avg": np.nanmean(rolling_3yr) if len(rolling_3yr) > 0 else 0.0,
            "roe_5yr_avg": np.nanmean(rolling_5yr) if len(rolling_5yr) > 0 else 0.0,
            "final_assets": self.assets[-1],
            "final_equity": self.equity[-1],
            "total_claims": np.sum(self.claim_amounts),
            "claim_frequency": np.mean(self.claim_counts),
            "survived": self.insolvency_year is None,
            "insolvency_year": (
                float(self.insolvency_year) if self.insolvency_year is not None else 0.0
            ),
        }

        # Add volatility metrics
        base_stats.update(volatility_metrics)

        return base_stats


class Simulation:
    """Simulation engine for widget manufacturer time evolution.

    The main simulation class that coordinates the time evolution of the
    widget manufacturer model, processing losses and tracking financial
    performance over the specified time horizon.

    Supports both single-path and Monte Carlo simulations, with comprehensive
    tracking of financial metrics, loss events, and bankruptcy conditions.

    Examples:
        Basic simulation setup and execution::

            from ergodic_insurance.config import ManufacturerConfig
            from ergodic_insurance.manufacturer import WidgetManufacturer
            from ergodic_insurance.loss_distributions import ManufacturingLossGenerator
            from ergodic_insurance.insurance_program import InsuranceProgram
            from ergodic_insurance.simulation import Simulation

            # Create manufacturer
            config = ManufacturerConfig(initial_assets=10_000_000)
            manufacturer = WidgetManufacturer(config)

            # Create insurance program
            program = InsuranceProgram.simple(
                deductible=500_000,
                limit=5_000_000,
                rate=0.02,
            )

            # Run simulation
            sim = Simulation(
                manufacturer=manufacturer,
                loss_generator=ManufacturingLossGenerator.create_simple(seed=42),
                insurance_policy=program,
                time_horizon=10
            )
            results = sim.run()

            # Analyze results
            print(f"Mean ROE: {results.summary_stats()['mean_roe']:.2%}")
            print(f"Survived: {results.insolvency_year is None}")

        Running Monte Carlo simulation::

            # Use MonteCarloEngine for multiple paths
            monte_carlo = MonteCarloEngine(
                base_simulation=sim,
                n_simulations=1000,
                parallel=True
            )
            mc_results = monte_carlo.run()

            print(f"Survival rate: {mc_results.survival_rate:.1%}")
            print(f"95% VaR: ${mc_results.var_95:,.0f}")

    Attributes:
        manufacturer: The widget manufacturer being simulated
        loss_generator: Generator for loss events
        insurance_policy: Optional insurance coverage
        time_horizon: Simulation duration in years
        seed: Random seed for reproducibility

    See Also:
        :class:`SimulationResults`: Container for simulation output
        :class:`MonteCarloEngine`: For running multiple simulation paths
        :class:`WidgetManufacturer`: The core financial model
        :class:`ManufacturingLossGenerator`: For generating loss events
    """

    def __init__(
        self,
        manufacturer: WidgetManufacturer,
        loss_generator: Optional[
            Union[ManufacturingLossGenerator, List[ManufacturingLossGenerator]]
        ] = None,
        insurance_policy: Optional[Union[InsuranceProgram, InsurancePolicy]] = None,
        time_horizon: int = 50,
        seed: Optional[int] = None,
        growth_rate: float = 0.0,
        letter_of_credit_rate: float = 0.015,
        copy: bool = True,
    ):
        """Initialize simulation.

        Args:
            manufacturer: WidgetManufacturer instance to simulate.
            loss_generator: ManufacturingLossGenerator or list of generators for
                creating loss events. If a list is provided, losses from all
                generators are combined. If None, a default generator is
                created with severity scaled to 5% of the manufacturer's
                initial assets (deprecated — pass explicitly instead).
            insurance_policy: Insurance program (or deprecated InsurancePolicy)
                for claim processing. Accepts :class:`InsuranceProgram`
                (preferred) or :class:`InsurancePolicy` (deprecated, auto-
                converted). If None, no insurance coverage is applied.
            time_horizon: Number of years to simulate. Must be positive.
            seed: Random seed for reproducibility. Passed to loss generator(s).
            growth_rate: Revenue growth rate per period (default 0.0).
            letter_of_credit_rate: Annual LoC rate for collateral costs (default 0.015).
            copy: If True (default), deep-copy the manufacturer so the caller's
                reference is never mutated. Set to False when the caller has
                already copied the manufacturer or mutation is acceptable
                (e.g., inside MonteCarloEngine loops).

        Examples:
            Setup with custom insurance::

                from ergodic_insurance.insurance_program import InsuranceProgram

                program = InsuranceProgram.simple(
                    deductible=1_000_000,
                    limit=10_000_000,
                    rate=0.03,
                )

                sim = Simulation(
                    manufacturer=manufacturer,
                    insurance_policy=program,
                    time_horizon=50
                )

            Setup with multiple loss generators::

                sim = Simulation(
                    manufacturer=manufacturer,
                    loss_generator=[standard_losses, catastrophic_risk],
                    insurance_policy=policy,
                    time_horizon=20
                )

            Opt out of copying for performance-critical paths::

                mfg_copy = deepcopy(manufacturer)
                sim = Simulation(manufacturer=mfg_copy, copy=False)
        """
        if copy:
            manufacturer = deepcopy(manufacturer)
        self.manufacturer = manufacturer
        # Deep-copy the initial manufacturer state for re-entrancy (Issue #349)
        self._initial_manufacturer = deepcopy(manufacturer)
        self.growth_rate = growth_rate
        self.letter_of_credit_rate = letter_of_credit_rate
        self._seed = seed

        # Handle single generator or list of generators
        if loss_generator is None:
            import warnings

            # Scale default severity to manufacturer size (5% mean, 2% std)
            initial_assets = float(manufacturer.config.initial_assets)
            default_frequency = 0.1
            default_severity_mean = initial_assets * 0.05
            default_severity_std = initial_assets * 0.02

            warnings.warn(
                f"No loss_generator provided — using default scaled to "
                f"initial_assets=${initial_assets:,.0f} "
                f"(frequency={default_frequency}, "
                f"severity_mean=${default_severity_mean:,.0f}, "
                f"severity_std=${default_severity_std:,.0f}). "
                f"Explicit construction is recommended:\n"
                f"  ManufacturingLossGenerator.create_simple(\n"
                f"      frequency={default_frequency}, "
                f"severity_mean={default_severity_mean:.0f}, "
                f"severity_std={default_severity_std:.0f}, seed=...)",
                ErgodicInsuranceDeprecationWarning,
                stacklevel=2,
            )
            logger.info(
                "Default loss generator: frequency=%.2f, "
                "severity_mean=$%s, severity_std=$%s "
                "(scaled to initial_assets=$%s)",
                default_frequency,
                f"{default_severity_mean:,.0f}",
                f"{default_severity_std:,.0f}",
                f"{initial_assets:,.0f}",
            )
            self.loss_generator = [
                ManufacturingLossGenerator.create_simple(
                    frequency=default_frequency,
                    severity_mean=default_severity_mean,
                    severity_std=default_severity_std,
                    seed=seed,
                )
            ]
        elif isinstance(loss_generator, list):
            self.loss_generator = loss_generator
        else:
            self.loss_generator = [loss_generator]

        # Normalize insurance input to InsuranceProgram
        if insurance_policy is not None and isinstance(insurance_policy, InsurancePolicy):
            import warnings

            warnings.warn(
                "Passing InsurancePolicy to Simulation is deprecated. "
                "Use InsuranceProgram instead (e.g., InsuranceProgram.simple(...)).",
                ErgodicInsuranceDeprecationWarning,
                stacklevel=2,
            )
            insurance_policy = insurance_policy.to_enhanced_program()

        self.insurance_policy = insurance_policy
        self.time_horizon = time_horizon
        self.seed = seed

        # Pre-allocate arrays for efficiency
        self.years = np.arange(time_horizon)
        self.assets = np.zeros(time_horizon)
        self.equity = np.zeros(time_horizon)
        self.roe = np.zeros(time_horizon)
        self.revenue = np.zeros(time_horizon)
        self.net_income = np.zeros(time_horizon)
        self.claim_counts = np.zeros(time_horizon)
        self.claim_amounts = np.zeros(time_horizon)

        self.insolvency_year: Optional[int] = None

    def step_annual(self, year: int, losses: List[LossEvent]) -> Dict[str, Any]:
        """Execute single annual time step.

        Processes losses for the year, applies insurance coverage,
        updates manufacturer financial state, and returns metrics.

        Args:
            year: Current simulation year (0-indexed).
            losses: List of LossEvent objects for this year.

        Returns:
            Dict[str, float]: Dictionary containing metrics:
                - All metrics from manufacturer.step()
                - 'claim_count': Number of losses this year
                - 'claim_amount': Total loss amount before insurance
                - 'company_payment': Amount paid by company after deductible
                - 'insurance_recovery': Amount recovered from insurance

        Note:
            This method modifies the manufacturer state in-place. Insurance
            premiums are deducted from both assets and equity to maintain
            balance sheet integrity.

        Side Effects:
            - Modifies manufacturer.assets and manufacturer.equity
            - Updates manufacturer internal state via step() method
        """
        # Unified ordering: claims → premium → step (Issue #349)
        # Process claims and premium BEFORE stepping the manufacturer so that
        # the metrics returned by step() reflect post-claim equity, and
        # insolvency is detected in the year it occurs.
        total_loss_amount = sum(loss.amount for loss in losses)
        total_company_payment = ZERO
        total_insurance_recovery = ZERO

        # Apply insurance
        if self.insurance_policy:
            for loss in losses:
                # Use manufacturer's built-in claim processing which handles collateral and payment schedules
                company_payment, insurance_recovery = self.manufacturer.process_insurance_claim(
                    claim_amount=loss.amount,
                    deductible_amount=self.insurance_policy.deductible,
                    insurance_limit=self.insurance_policy.get_total_coverage(),
                )
                total_company_payment += company_payment
                total_insurance_recovery += insurance_recovery

            # Record annual premium
            annual_premium = self.insurance_policy.calculate_premium()
            self.manufacturer.record_insurance_premium(annual_premium)
        else:
            # No insurance - process as uninsured claims
            for loss in losses:
                # Company pays full amount with no insurance coverage
                # Use deferred payment to avoid immediate double-hit on equity
                self.manufacturer.process_uninsured_claim(
                    claim_amount=loss.amount,
                    immediate_payment=False,  # Create liability with payment schedule starting next year
                )
                total_company_payment += to_decimal(loss.amount)

        # Step manufacturer AFTER claims and premium (Issue #349)
        # Uses configurable parameters instead of hardcoded values
        metrics = self.manufacturer.step(
            letter_of_credit_rate=self.letter_of_credit_rate,
            growth_rate=self.growth_rate,
        )

        # Add loss information to metrics
        metrics["claim_count"] = len(losses)
        metrics["claim_amount"] = total_loss_amount
        metrics["company_payment"] = total_company_payment
        metrics["insurance_recovery"] = total_insurance_recovery

        return metrics

    def run(
        self,
        progress_interval: int = 100,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> SimulationResults:
        """Run the full simulation over the specified time horizon.

        Executes a complete simulation trajectory, processing claims each year,
        updating the manufacturer's financial state, and tracking all metrics.
        The simulation terminates early if the manufacturer becomes insolvent.

        Args:
            progress_interval: How often to log progress (in years). Set to 0
                to disable progress logging. Useful for long simulations.
            progress_callback: Optional callback invoked with
                ``(completed_years, total_years, elapsed_seconds)`` after each
                year completes.  Useful for GUI progress bars, web dashboards,
                or any non-terminal environment.
            cancel_event: Optional :class:`threading.Event`.  When set, the
                simulation will stop after the current year and return partial
                results (same pattern as insolvency early-stop).

        Returns:
            SimulationResults object containing:
                - Complete time series of financial metrics
                - Claim history and amounts
                - ROE trajectory
                - Insolvency year (if bankruptcy occurred)

        Examples:
            Run simulation with progress updates::

                sim = Simulation(manufacturer, time_horizon=1000)
                results = sim.run(progress_interval=100)  # Log every 100 years

                # Check if company survived
                if results.insolvency_year:
                    print(f"Bankruptcy in year {results.insolvency_year}")
                else:
                    print(f"Survived {len(results.years)} years")

            Analyze simulation results::

                results = sim.run()
                df = results.to_dataframe()

                # Plot equity evolution
                import matplotlib.pyplot as plt
                plt.plot(df['year'], df['equity'])
                plt.xlabel('Year')
                plt.ylabel('Equity ($)')
                plt.title('Company Equity Over Time')
                plt.show()

        Note:
            The simulation uses pre-generated claims for efficiency. All claims
            are generated at the start based on the configured loss distributions
            and random seed.

        See Also:
            :meth:`step_annual`: Single year simulation step
            :class:`SimulationResults`: Output data structure
        """
        start_time = time.time()

        # Reset mutable state so the simulation is re-entrant
        # Reset manufacturer from initial state (Issue #349)
        self.manufacturer = deepcopy(self._initial_manufacturer)
        # Reseed loss generators so repeated runs produce identical sequences
        if self._seed is not None:
            for gen in self.loss_generator:
                if hasattr(gen, "reseed"):
                    gen.reseed(self._seed)
        self.insolvency_year = None
        self.assets = np.zeros(self.time_horizon)
        self.equity = np.zeros(self.time_horizon)
        self.roe = np.zeros(self.time_horizon)
        self.revenue = np.zeros(self.time_horizon)
        self.net_income = np.zeros(self.time_horizon)
        self.claim_counts = np.zeros(self.time_horizon)
        self.claim_amounts = np.zeros(self.time_horizon)

        logger.info(f"Starting {self.time_horizon}-year simulation with dynamic loss generation")

        # Run simulation
        completed_years = 0
        cancelled = False
        for year in range(self.time_horizon):
            # Check for cancellation
            if cancel_event is not None and cancel_event.is_set():
                logger.info("Cancellation requested at year %d/%d", year, self.time_horizon)
                cancelled = True
                # Fill remaining years with zeros
                self.assets[year:] = 0
                self.equity[year:] = 0
                self.roe[year:] = np.nan
                self.revenue[year:] = 0
                self.net_income[year:] = 0
                break

            # Log progress
            if year > 0 and year % progress_interval == 0:
                elapsed = time.time() - start_time
                rate = year / elapsed if elapsed > 0 else float("inf")
                remaining = (self.time_horizon - year) / rate
                logger.info(
                    f"Year {year}/{self.time_horizon} - {elapsed:.1f}s elapsed, {remaining:.1f}s remaining"
                )

            # Generate losses for this year based on current financial state
            year_losses: List[LossEvent] = []
            revenue = (
                self.manufacturer.revenue if hasattr(self.manufacturer, "revenue") else 10_000_000
            )
            for generator in self.loss_generator:
                # Generate losses for one year duration
                losses, _ = generator.generate_losses(duration=1, revenue=revenue, time=float(year))
                year_losses.extend(losses)

            # Execute time step
            metrics = self.step_annual(year, year_losses)

            # Store results
            self.assets[year] = metrics.get("assets", 0)
            self.equity[year] = metrics.get("equity", 0)
            self.roe[year] = metrics.get("roe", 0)
            self.revenue[year] = metrics.get("revenue", 0)
            self.net_income[year] = metrics.get("net_income", 0)
            self.claim_counts[year] = metrics.get("claim_count", 0)
            self.claim_amounts[year] = metrics.get("claim_amount", 0)

            completed_years = year + 1

            # Fire progress callback after each year
            if progress_callback is not None:
                progress_callback(completed_years, self.time_horizon, time.time() - start_time)

            # Check for insolvency using manufacturer's insolvency tolerance
            tolerance = self.manufacturer.config.insolvency_tolerance
            if metrics.get("equity", 0) <= tolerance and self.insolvency_year is None:
                self.insolvency_year = year
                logger.warning(
                    f"Manufacturer became insolvent in year {year} (equity <= ${tolerance:,.2f})"
                )
                # Fill remaining years with zeros
                self.assets[year + 1 :] = 0
                self.equity[year + 1 :] = 0
                self.roe[year + 1 :] = np.nan
                self.revenue[year + 1 :] = 0
                self.net_income[year + 1 :] = 0
                break  # Stop simulation — manufacturer is insolvent

        # Calculate total time
        total_time = time.time() - start_time
        logger.info(f"Simulation completed in {total_time:.2f} seconds")

        # Create and return results
        results = SimulationResults(
            years=self.years,
            assets=self.assets,
            equity=self.equity,
            roe=self.roe,
            revenue=self.revenue,
            net_income=self.net_income,
            claim_counts=self.claim_counts,
            claim_amounts=self.claim_amounts,
            insolvency_year=self.insolvency_year,
        )

        return results

    def run_with_loss_data(
        self, loss_data: LossData, validate: bool = True, progress_interval: int = 100
    ) -> SimulationResults:
        """Run simulation using standardized LossData.

        Args:
            loss_data: Standardized loss data.
            validate: Whether to validate loss data before running.
            progress_interval: How often to log progress.

        Returns:
            SimulationResults object with full trajectory.
        """
        # Validate if requested
        if validate and not loss_data.validate():
            logger.warning("Loss data validation failed")
            raise ValueError("Invalid loss data provided")

        # Convert to LossEvents
        losses = loss_data.to_loss_events()

        # Group losses by year
        losses_by_year: Dict[int, List[LossEvent]] = {year: [] for year in range(self.time_horizon)}
        for loss in losses:
            year = int(loss.time)
            if 0 <= year < self.time_horizon:
                losses_by_year[year].append(loss)

        # Reset mutable state so the simulation is re-entrant
        # Reset manufacturer from initial state (Issue #349)
        self.manufacturer = deepcopy(self._initial_manufacturer)
        if self._seed is not None:
            for gen in self.loss_generator:
                if hasattr(gen, "reseed"):
                    gen.reseed(self._seed)
        self.insolvency_year = None
        self.assets = np.zeros(self.time_horizon)
        self.equity = np.zeros(self.time_horizon)
        self.roe = np.zeros(self.time_horizon)
        self.revenue = np.zeros(self.time_horizon)
        self.net_income = np.zeros(self.time_horizon)
        self.claim_counts = np.zeros(self.time_horizon)
        self.claim_amounts = np.zeros(self.time_horizon)

        logger.info(
            f"Starting {self.time_horizon}-year simulation with {len(losses)} losses from LossData"
        )

        start_time = time.time()

        # Run simulation year by year
        for year in range(self.time_horizon):
            # Log progress
            if year > 0 and year % progress_interval == 0:
                elapsed = time.time() - start_time
                rate = year / elapsed if elapsed > 0 else float("inf")
                remaining = (self.time_horizon - year) / rate
                logger.info(
                    f"Year {year}/{self.time_horizon} - {elapsed:.1f}s elapsed, {remaining:.1f}s remaining"
                )

            # Get losses for this year
            year_losses = losses_by_year.get(year, [])

            # Execute time step
            metrics = self.step_annual(year, year_losses)

            # Store results
            self.assets[year] = metrics.get("assets", 0)
            self.equity[year] = metrics.get("equity", 0)
            self.roe[year] = metrics.get("roe", 0)
            self.revenue[year] = metrics.get("revenue", 0)
            self.net_income[year] = metrics.get("net_income", 0)
            self.claim_counts[year] = metrics.get("claim_count", 0)
            self.claim_amounts[year] = metrics.get("claim_amount", 0)

            # Check for insolvency using manufacturer's insolvency tolerance
            tolerance = self.manufacturer.config.insolvency_tolerance
            if metrics.get("equity", 0) <= tolerance and self.insolvency_year is None:
                self.insolvency_year = year
                logger.warning(
                    f"Manufacturer became insolvent in year {year} (equity <= ${tolerance:,.2f})"
                )
                # Fill remaining years with zeros
                self.assets[year + 1 :] = 0
                self.equity[year + 1 :] = 0
                self.roe[year + 1 :] = np.nan
                self.revenue[year + 1 :] = 0
                self.net_income[year + 1 :] = 0
                break  # Stop simulation — manufacturer is insolvent

        # Log completion
        total_time = time.time() - start_time
        logger.info(f"Simulation with LossData completed in {total_time:.2f} seconds")

        # Create and return results
        results = SimulationResults(
            years=self.years,
            assets=self.assets,
            equity=self.equity,
            roe=self.roe,
            revenue=self.revenue,
            net_income=self.net_income,
            claim_counts=self.claim_counts,
            claim_amounts=self.claim_amounts,
            insolvency_year=self.insolvency_year,
        )

        return results

    def get_trajectory(self) -> pd.DataFrame:
        """Get simulation trajectory as pandas DataFrame.

        This is a convenience method that runs the simulation if needed
        and returns the results as a DataFrame.

        Returns:
            DataFrame with simulation trajectory.
        """
        results = self.run()
        return results.to_dataframe()

    @classmethod
    def run_monte_carlo(
        cls,
        config: Config,
        insurance_policy: Optional[Union[InsuranceProgram, InsurancePolicy]] = None,
        n_scenarios: int = 10000,
        batch_size: int = 1000,
        n_jobs: int = 7,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_frequency: int = 5000,
        seed: Optional[int] = None,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation using the MonteCarloEngine.

        .. deprecated::
            Use :func:`ergodic_insurance.run_monte_carlo` instead.
        """
        warnings.warn(
            "Simulation.run_monte_carlo() is deprecated. "
            "Use the standalone run_monte_carlo() function instead:\n"
            "  from ergodic_insurance import run_monte_carlo",
            ErgodicInsuranceDeprecationWarning,
            stacklevel=2,
        )
        return _run_monte_carlo_func(
            config=config,
            insurance_policy=insurance_policy,
            n_scenarios=n_scenarios,
            batch_size=batch_size,
            n_jobs=n_jobs,
            checkpoint_dir=checkpoint_dir,
            checkpoint_frequency=checkpoint_frequency,
            seed=seed,
            resume=resume,
        )

    @classmethod
    def compare_insurance_strategies(
        cls,
        config: Config,
        insurance_policies: Mapping[str, Union[InsuranceProgram, InsurancePolicy]],
        n_scenarios: int = 1000,
        n_jobs: int = 7,
        seed: Optional[int] = None,
    ) -> StrategyComparisonResult:
        """Compare multiple insurance strategies via Monte Carlo.

        .. deprecated::
            Use :func:`ergodic_insurance.compare_strategies` instead.
        """
        warnings.warn(
            "Simulation.compare_insurance_strategies() is deprecated. "
            "Use the standalone compare_strategies() function instead:\n"
            "  from ergodic_insurance import compare_strategies",
            ErgodicInsuranceDeprecationWarning,
            stacklevel=2,
        )
        return _compare_strategies_func(
            config=config,
            insurance_policies=insurance_policies,
            n_scenarios=n_scenarios,
            n_jobs=n_jobs,
            seed=seed,
        )
