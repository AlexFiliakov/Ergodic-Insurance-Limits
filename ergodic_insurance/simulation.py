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

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .config import Config
from .insurance import InsurancePolicy
from .insurance_program import InsuranceProgram
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

    def calculate_roe_components(self) -> Dict[str, np.ndarray]:
        """Calculate ROE component breakdown.

        Decomposes ROE into operating, insurance, and tax components
        using DuPont-style analysis. This helps identify the drivers
        of ROE performance and the impact of insurance decisions.

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
                base_margin = 0.08  # Baseline operating margin
                components["operating_roe"][i] = (self.revenue[i] * base_margin) / self.equity[i]

                # Insurance impact = reduction in ROE due to premiums and retained losses
                if self.claim_amounts[i] > 0:
                    components["insurance_impact"][i] = -self.claim_amounts[i] / self.equity[i]

                # Tax effect (simplified)
                tax_rate = 0.25
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
        std_roe = np.std(valid_roe)

        # Downside deviation (only negative deviations from mean)
        negative_deviations = valid_roe[valid_roe < mean_roe] - mean_roe
        downside_dev = (
            np.sqrt(np.mean(negative_deviations**2)) if len(negative_deviations) > 0 else 0.0
        )

        # Sharpe ratio equivalent for ROE (using risk-free rate of 2%)
        risk_free_rate = 0.02
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
        rolling_1yr = self.calculate_rolling_roe(1) if len(self.years) >= 1 else np.array([])
        rolling_3yr = self.calculate_rolling_roe(3) if len(self.years) >= 3 else np.array([])
        rolling_5yr = self.calculate_rolling_roe(5) if len(self.years) >= 5 else np.array([])

        # Get volatility metrics
        volatility_metrics = self.calculate_roe_volatility()

        base_stats = {
            "mean_roe": np.mean(valid_roe) if len(valid_roe) > 0 else 0.0,
            "std_roe": np.std(valid_roe) if len(valid_roe) > 0 else 0.0,
            "median_roe": np.median(valid_roe) if len(valid_roe) > 0 else 0.0,
            "time_weighted_roe": self.calculate_time_weighted_roe(),
            "roe_1yr_avg": np.nanmean(rolling_1yr) if len(rolling_1yr) > 0 else 0.0,
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
            from ergodic_insurance.insurance import InsurancePolicy
            from ergodic_insurance.simulation import Simulation

            # Create manufacturer
            config = ManufacturerConfig(initial_assets=10_000_000)
            manufacturer = WidgetManufacturer(config)

            # Create insurance policy
            policy = InsurancePolicy(
                deductible=500_000,
                limit=5_000_000,
                premium_rate=0.02
            )

            # Run simulation
            sim = Simulation(
                manufacturer=manufacturer,
                loss_generator=ManufacturingLossGenerator.create_simple(seed=42),
                insurance_policy=policy,
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
        insurance_policy: Optional[InsurancePolicy] = None,
        time_horizon: int = 100,
        seed: Optional[int] = None,
    ):
        """Initialize simulation.

        Args:
            manufacturer: WidgetManufacturer instance to simulate. This object
                maintains the financial state and is modified during simulation.
            loss_generator: ManufacturingLossGenerator or list of generators for
                creating loss events. If a list is provided, losses from all
                generators are combined. If None, a default generator with
                standard parameters is created.
            insurance_policy: Insurance policy for claim processing. If None,
                legacy claim processing is used with fixed parameters.
            time_horizon: Number of years to simulate. Must be positive.
            seed: Random seed for reproducibility. Passed to loss generator(s).

        Note:
            The manufacturer object is modified in-place during simulation.
            Create a copy if you need to preserve the initial state.

        Examples:
            Setup with custom insurance::

                from ergodic_insurance.insurance import InsurancePolicy

                policy = InsurancePolicy(
                    deductible=1_000_000,
                    limit=10_000_000,
                    premium_rate=0.03
                )

                sim = Simulation(
                    manufacturer=manufacturer,
                    insurance_policy=policy,
                    time_horizon=50
                )

            Setup with multiple loss generators::

                sim = Simulation(
                    manufacturer=manufacturer,
                    loss_generator=[standard_losses, catastrophic_risk],
                    insurance_policy=policy,
                    time_horizon=20
                )
        """
        self.manufacturer = manufacturer

        # Handle single generator or list of generators
        if loss_generator is None:
            # Create default loss generator with reasonable parameters
            self.loss_generator = [
                ManufacturingLossGenerator.create_simple(
                    frequency=0.1,  # 10% chance per year
                    severity_mean=5_000_000,  # $5M mean claim
                    severity_std=2_000_000,  # $2M standard deviation
                    seed=seed,
                )
            ]
        elif isinstance(loss_generator, list):
            self.loss_generator = loss_generator
        else:
            self.loss_generator = [loss_generator]

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

    def step_annual(self, year: int, losses: List[LossEvent]) -> Dict[str, float]:
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
        # Step manufacturer forward FIRST (process year's normal operations)
        # Then apply losses at END of year to prevent newly-created liabilities
        # from being paid in the same year they're created
        metrics = self.manufacturer.step(
            working_capital_pct=0.2, letter_of_credit_rate=0.015, growth_rate=0.03
        )

        # Process losses at END of year
        total_loss_amount = sum(loss.amount for loss in losses)
        total_company_payment = 0.0
        total_insurance_recovery = 0.0

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

            # Record annual premium (will be handled in manufacturer's step method next year)
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
                total_company_payment += loss.amount

        # Add loss information to metrics
        metrics["claim_count"] = len(losses)
        metrics["claim_amount"] = total_loss_amount
        metrics["company_payment"] = total_company_payment
        metrics["insurance_recovery"] = total_insurance_recovery

        return metrics

    def run(self, progress_interval: int = 100) -> SimulationResults:
        """Run the full simulation over the specified time horizon.

        Executes a complete simulation trajectory, processing claims each year,
        updating the manufacturer's financial state, and tracking all metrics.
        The simulation terminates early if the manufacturer becomes insolvent.

        Args:
            progress_interval: How often to log progress (in years). Set to 0
                to disable progress logging. Useful for long simulations.

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

        logger.info(f"Starting {self.time_horizon}-year simulation with dynamic loss generation")

        # Run simulation
        for year in range(self.time_horizon):
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
                # Continue simulation to maintain full array length

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
                # Continue simulation to maintain full array length

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
    def run_monte_carlo(  # pylint: disable=too-many-locals
        cls,
        config: Config,
        insurance_policy: InsurancePolicy,
        n_scenarios: int = 10000,
        batch_size: int = 1000,
        n_jobs: int = 7,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_frequency: int = 5000,
        seed: Optional[int] = None,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation using the MonteCarloEngine.

        This is a convenience class method for running large-scale Monte Carlo
        simulations with the optimized engine.

        Args:
            config: Configuration object.
            insurance_policy: Insurance policy to simulate.
            n_scenarios: Number of scenarios to run.
            batch_size: Scenarios per batch.
            n_jobs: Number of parallel jobs.
            checkpoint_dir: Directory for checkpoints.
            checkpoint_frequency: Save checkpoint every N scenarios.
            seed: Random seed.
            resume: Whether to resume from checkpoint.

        Returns:
            Dictionary of Monte Carlo results and statistics.
        """
        # Create loss generator
        from .loss_distributions import ManufacturingLossGenerator

        loss_generator = ManufacturingLossGenerator(seed=seed)

        # Create insurance program
        # Always create a new program since InsurancePolicy and InsuranceProgram
        # have incompatible signatures
        insurance_program = InsuranceProgram(layers=[])
        if insurance_policy:
            # Add layer based on simple policy
            from .insurance_program import EnhancedInsuranceLayer

            layer = EnhancedInsuranceLayer(
                attachment_point=0,
                limit=(
                    insurance_policy.limit if hasattr(insurance_policy, "limit") else float("inf")
                ),
                base_premium_rate=0.01,
            )
            insurance_program.layers.append(layer)

        # Create manufacturer
        manufacturer = WidgetManufacturer(config=config.manufacturer)

        # Create simulation config
        from .monte_carlo import SimulationConfig

        sim_config = SimulationConfig(
            n_simulations=n_scenarios,
            n_years=getattr(config.simulation, "years", 10),
            parallel=n_jobs > 1 if n_jobs else True,
            n_workers=n_jobs,
            chunk_size=batch_size,
            checkpoint_interval=checkpoint_frequency,
            seed=seed,
        )

        # Run simulation WITH insurance
        engine_with_insurance = MonteCarloEngine(
            loss_generator=loss_generator,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=sim_config,
        )

        logger.info("Running Monte Carlo simulation WITH insurance...")
        results_with_insurance = engine_with_insurance.run()

        # Run simulation WITHOUT insurance using the same seed for fair comparison
        logger.info("Running Monte Carlo simulation WITHOUT insurance (same seed)...")

        # Create new loss generator with same seed for reproducibility
        loss_generator_no_insurance = ManufacturingLossGenerator(seed=seed)

        # Create new manufacturer instance (must be fresh for second run)
        manufacturer_no_insurance = WidgetManufacturer(config=config.manufacturer)

        # Create empty insurance program (no coverage)
        insurance_program_no_insurance = InsuranceProgram(layers=[])

        engine_without_insurance = MonteCarloEngine(
            loss_generator=loss_generator_no_insurance,
            insurance_program=insurance_program_no_insurance,
            manufacturer=manufacturer_no_insurance,
            config=sim_config,
        )

        results_without_insurance = engine_without_insurance.run()

        # Calculate empirical ergodic analysis by comparing the two runs
        ergodic_analysis = {}

        if hasattr(results_with_insurance, "statistics") and hasattr(
            results_without_insurance, "statistics"
        ):
            stats_with = results_with_insurance.statistics
            stats_without = results_without_insurance.statistics

            # Extract metrics from both runs
            if "geometric_return" in stats_with and "geometric_return" in stats_without:
                geo_with = stats_with["geometric_return"]
                geo_without = stats_without["geometric_return"]

                premium_cost = insurance_policy.calculate_premium()
                initial_assets = config.manufacturer.initial_assets
                premium_rate = premium_cost / initial_assets

                # Calculate empirical insurance impact
                ergodic_analysis = {
                    "premium_rate": premium_rate,
                    # With insurance metrics
                    "geometric_mean_return_with_insurance": geo_with.get("geometric_mean", 0.0),
                    "survival_rate_with_insurance": geo_with.get("survival_rate", 0.0),
                    "volatility_with_insurance": geo_with.get("std", 0.0),
                    # Without insurance metrics
                    "geometric_mean_return_without_insurance": geo_without.get(
                        "geometric_mean", 0.0
                    ),
                    "survival_rate_without_insurance": geo_without.get("survival_rate", 0.0),
                    "volatility_without_insurance": geo_without.get("std", 0.0),
                    # Empirical deltas (insurance impact)
                    "growth_impact": geo_with.get("geometric_mean", 0.0)
                    - geo_without.get("geometric_mean", 0.0),
                    "survival_benefit": geo_with.get("survival_rate", 0.0)
                    - geo_without.get("survival_rate", 0.0),
                    "volatility_reduction": geo_without.get("std", 0.0) - geo_with.get("std", 0.0),
                }

                logger.info(
                    f"Empirical insurance impact: growth={ergodic_analysis['growth_impact']:.2%}, "
                    f"survival={ergodic_analysis['survival_benefit']:.2%}, "
                    f"volatility_reduction={ergodic_analysis['volatility_reduction']:.2%}"
                )

                return {
                    "results_with_insurance": results_with_insurance,
                    "results_without_insurance": results_without_insurance,
                    "ergodic_analysis": ergodic_analysis,
                }

        # Fallback if statistics not available
        return {
            "results_with_insurance": results_with_insurance,
            "results_without_insurance": results_without_insurance,
        }

    @classmethod
    def compare_insurance_strategies(
        cls,
        config: Config,
        insurance_policies: Dict[str, InsurancePolicy],
        n_scenarios: int = 1000,
        n_jobs: int = 7,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compare multiple insurance strategies via Monte Carlo.

        Args:
            config: Configuration object.
            insurance_policies: Dictionary of policy name to InsurancePolicy.
            n_scenarios: Scenarios per policy.
            n_jobs: Number of parallel jobs.
            seed: Random seed.

        Returns:
            DataFrame comparing results across strategies.
        """
        results = []

        for policy_name, policy in insurance_policies.items():
            logger.info(f"Running Monte Carlo for policy: {policy_name}")

            # Run Monte Carlo
            mc_results = cls.run_monte_carlo(
                config=config,
                insurance_policy=policy,
                n_scenarios=n_scenarios,
                n_jobs=n_jobs,
                seed=seed,
                checkpoint_frequency=n_scenarios + 1,  # Don't checkpoint for comparisons
                resume=False,
            )

            # Extract key metrics
            stats = mc_results["statistics"]

            result_row = {
                "policy": policy_name,
                "annual_premium": policy.calculate_premium(),
                "total_coverage": policy.get_total_coverage(),
                "survival_rate": stats["final_equity"]["survival_rate"],
                "mean_final_equity": stats["final_equity"]["mean"],
                "std_final_equity": stats["final_equity"]["std"],
                "geometric_return": stats["geometric_return"]["geometric_mean"],
                "arithmetic_return": stats["arithmetic_return"]["mean"],
                "p95_final_equity": stats["final_equity"]["p95"],
                "p99_final_equity": stats["final_equity"]["p99"],
            }

            results.append(result_row)

        comparison_df = pd.DataFrame(results)

        # Add relative metrics
        if len(comparison_df) > 0:
            comparison_df["premium_to_coverage"] = (
                comparison_df["annual_premium"] / comparison_df["total_coverage"]
            )
            comparison_df["sharpe_ratio"] = (
                comparison_df["arithmetic_return"] / comparison_df["std_final_equity"]
            )

        return comparison_df
