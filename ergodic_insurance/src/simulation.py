"""Simulation engine for time evolution of widget manufacturer model.

This module provides the main simulation engine that orchestrates the
time evolution of the widget manufacturer financial model, managing
claim events, financial calculations, and result collection.

Includes both single-path simulation and Monte Carlo capabilities.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .claim_generator import ClaimEvent, ClaimGenerator
from .config import Config
from .insurance import InsurancePolicy
from .insurance_program import InsuranceProgram
from .manufacturer import WidgetManufacturer
from .monte_carlo import MonteCarloEngine

if TYPE_CHECKING:
    from .loss_distributions import LossData

logger = logging.getLogger(__name__)


@dataclass
class SimulationResults:
    """Container for simulation trajectory data.

    Holds the complete time series of financial metrics and events
    from a single simulation run, with methods for analysis and export.
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
            DataFrame with all trajectory data.
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
        performance over time.

        Returns:
            Time-weighted average ROE.
        """
        valid_roe = self.roe[~np.isnan(self.roe)]
        if len(valid_roe) == 0:
            return 0.0

        # For time-weighted average, we use geometric mean for compounding returns
        # Convert ROE to growth factors (1 + roe)
        growth_factors = 1 + valid_roe

        # Geometric mean minus 1 to get average return
        time_weighted_roe = np.exp(np.mean(np.log(growth_factors))) - 1
        return float(time_weighted_roe)

    def calculate_rolling_roe(self, window: int) -> np.ndarray:
        """Calculate rolling window ROE.

        Args:
            window: Window size in years (e.g., 1, 3, 5).

        Returns:
            Array of rolling ROE values.
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
        using DuPont-style analysis.

        Returns:
            Dictionary with component arrays.
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

        Returns:
            Dictionary with volatility metrics.
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

        Returns:
            Dictionary of key statistics including enhanced ROE metrics.
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
    widget manufacturer model, processing claims and tracking financial
    performance over the specified time horizon.

    Supports both single-path and Monte Carlo simulations.
    """

    def __init__(
        self,
        manufacturer: WidgetManufacturer,
        claim_generator: Optional[ClaimGenerator] = None,
        insurance_policy: Optional[InsurancePolicy] = None,
        time_horizon: int = 100,
        seed: Optional[int] = None,
    ):
        """Initialize simulation.

        Args:
            manufacturer: WidgetManufacturer instance to simulate.
            claim_generator: ClaimGenerator for creating insurance claims.
            insurance_policy: Insurance policy for claim processing.
            time_horizon: Number of years to simulate.
            seed: Random seed for reproducibility.
        """
        self.manufacturer = manufacturer
        self.claim_generator = claim_generator or ClaimGenerator(seed=seed)
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

    def step_annual(self, year: int, claims: List[ClaimEvent]) -> Dict[str, float]:
        """Execute single annual time step.

        Args:
            year: Current simulation year.
            claims: List of claims for this year.

        Returns:
            Dictionary of metrics for this year.
        """
        # Process claims
        total_claim_amount = sum(claim.amount for claim in claims)
        total_company_payment = 0.0
        total_insurance_recovery = 0.0

        # Apply insurance
        if self.insurance_policy:
            for claim in claims:
                company_payment, insurance_recovery = self.insurance_policy.process_claim(
                    claim.amount
                )
                self.manufacturer.assets -= company_payment
                total_company_payment += company_payment
                total_insurance_recovery += insurance_recovery

            # Pay annual premium
            annual_premium = self.insurance_policy.calculate_premium()
            self.manufacturer.assets -= annual_premium
        else:
            # Legacy behavior - process claims without policy
            for claim in claims:
                self.manufacturer.process_insurance_claim(
                    claim_amount=claim.amount, deductible=1_000_000, insurance_limit=10_000_000
                )

        # Step manufacturer forward
        metrics = self.manufacturer.step(
            working_capital_pct=0.2, letter_of_credit_rate=0.015, growth_rate=0.03
        )

        # Add claim information to metrics
        metrics["claim_count"] = len(claims)
        metrics["claim_amount"] = total_claim_amount
        metrics["company_payment"] = total_company_payment
        metrics["insurance_recovery"] = total_insurance_recovery

        return metrics

    def run(self, progress_interval: int = 100) -> SimulationResults:
        """Run the full simulation.

        Args:
            progress_interval: How often to log progress (in years).

        Returns:
            SimulationResults object with full trajectory.
        """
        start_time = time.time()

        # Generate all claims upfront for efficiency
        all_claims = self.claim_generator.generate_claims(self.time_horizon)

        # Group claims by year
        claims_by_year: Dict[int, List[ClaimEvent]] = {
            year: [] for year in range(self.time_horizon)
        }
        for claim in all_claims:
            if 0 <= claim.year < self.time_horizon:
                claims_by_year[claim.year].append(claim)

        logger.info(
            f"Starting {self.time_horizon}-year simulation with {len(all_claims)} total claims"
        )

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

            # Get claims for this year
            year_claims = claims_by_year.get(year, [])

            # Execute time step
            metrics = self.step_annual(year, year_claims)

            # Store results
            self.assets[year] = metrics.get("assets", 0)
            self.equity[year] = metrics.get("equity", 0)
            self.roe[year] = metrics.get("roe", 0)
            self.revenue[year] = metrics.get("revenue", 0)
            self.net_income[year] = metrics.get("net_income", 0)
            self.claim_counts[year] = metrics.get("claim_count", 0)
            self.claim_amounts[year] = metrics.get("claim_amount", 0)

            # Check for insolvency
            if metrics.get("equity", 0) <= 0 and self.insolvency_year is None:
                self.insolvency_year = year
                logger.warning(f"Manufacturer became insolvent in year {year}")
                # Fill remaining years with zeros
                self.assets[year + 1 :] = 0
                self.equity[year + 1 :] = 0
                self.roe[year + 1 :] = np.nan
                self.revenue[year + 1 :] = 0
                self.net_income[year + 1 :] = 0
                break

        # Calculate total time
        total_time = time.time() - start_time
        logger.info(f"Simulation completed in {total_time:.2f} seconds")

        # Create and return results
        results = SimulationResults(
            years=self.years[: year + 1] if self.insolvency_year else self.years,
            assets=self.assets[: year + 1] if self.insolvency_year else self.assets,
            equity=self.equity[: year + 1] if self.insolvency_year else self.equity,
            roe=self.roe[: year + 1] if self.insolvency_year else self.roe,
            revenue=self.revenue[: year + 1] if self.insolvency_year else self.revenue,
            net_income=self.net_income[: year + 1] if self.insolvency_year else self.net_income,
            claim_counts=(
                self.claim_counts[: year + 1] if self.insolvency_year else self.claim_counts
            ),
            claim_amounts=(
                self.claim_amounts[: year + 1] if self.insolvency_year else self.claim_amounts
            ),
            insolvency_year=self.insolvency_year,
        )

        return results

    def run_with_loss_data(
        self, loss_data: "LossData", validate: bool = True, progress_interval: int = 100
    ) -> SimulationResults:
        """Run simulation using standardized LossData.

        Args:
            loss_data: Standardized loss data.
            validate: Whether to validate loss data before running.
            progress_interval: How often to log progress.

        Returns:
            SimulationResults object with full trajectory.
        """
        # Import here to avoid circular dependency
        from .loss_distributions import LossData

        # Validate if requested
        if validate and not loss_data.validate():
            logger.warning("Loss data validation failed")
            raise ValueError("Invalid loss data provided")

        # Convert to ClaimEvents
        claims = ClaimGenerator.from_loss_data(loss_data)

        # Group claims by year
        claims_by_year: Dict[int, List[ClaimEvent]] = {
            year: [] for year in range(self.time_horizon)
        }
        for claim in claims:
            if 0 <= claim.year < self.time_horizon:
                claims_by_year[claim.year].append(claim)

        logger.info(
            f"Starting {self.time_horizon}-year simulation with {len(claims)} claims from LossData"
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

            # Get claims for this year
            year_claims = claims_by_year.get(year, [])

            # Execute time step
            metrics = self.step_annual(year, year_claims)

            # Store results
            self.assets[year] = metrics.get("assets", 0)
            self.equity[year] = metrics.get("equity", 0)
            self.roe[year] = metrics.get("roe", 0)
            self.revenue[year] = metrics.get("revenue", 0)
            self.net_income[year] = metrics.get("net_income", 0)
            self.claim_counts[year] = metrics.get("claim_count", 0)
            self.claim_amounts[year] = metrics.get("claim_amount", 0)

            # Check for insolvency
            if metrics.get("equity", 0) <= 0 and self.insolvency_year is None:
                self.insolvency_year = year
                logger.warning(f"Manufacturer became insolvent in year {year}")
                # Fill remaining years with zeros
                self.assets[year + 1 :] = 0
                self.equity[year + 1 :] = 0
                self.roe[year + 1 :] = np.nan
                self.revenue[year + 1 :] = 0
                self.net_income[year + 1 :] = 0
                break

        # Log completion
        total_time = time.time() - start_time
        logger.info(f"Simulation with LossData completed in {total_time:.2f} seconds")

        # Create and return results
        results = SimulationResults(
            years=self.years[: year + 1] if self.insolvency_year else self.years,
            assets=self.assets[: year + 1] if self.insolvency_year else self.assets,
            equity=self.equity[: year + 1] if self.insolvency_year else self.equity,
            roe=self.roe[: year + 1] if self.insolvency_year else self.roe,
            revenue=self.revenue[: year + 1] if self.insolvency_year else self.revenue,
            net_income=self.net_income[: year + 1] if self.insolvency_year else self.net_income,
            claim_counts=(
                self.claim_counts[: year + 1] if self.insolvency_year else self.claim_counts
            ),
            claim_amounts=(
                self.claim_amounts[: year + 1] if self.insolvency_year else self.claim_amounts
            ),
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
                limit=insurance_policy.limit
                if hasattr(insurance_policy, "limit")
                else float("inf"),
                premium_rate=0.01,
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

        engine = MonteCarloEngine(
            loss_generator=loss_generator,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            config=sim_config,
        )

        results = engine.run()

        # Add ergodic analysis
        if hasattr(results, "statistics"):
            stats = results.statistics
        else:
            stats = {}

        # Calculate ergodic premium justification
        if "geometric_return" in stats:
            geo_stats = stats["geometric_return"]
            premium_cost = insurance_policy.calculate_premium()
            initial_assets = config.manufacturer.initial_assets

            # Premium as percentage of assets
            premium_rate = premium_cost / initial_assets

            # Compare geometric returns with and without insurance
            # This is simplified - actual comparison would require running without insurance
            ergodic_analysis = {
                "premium_rate": premium_rate,
                "geometric_mean_return": geo_stats["geometric_mean"],
                "survival_rate": geo_stats["survival_rate"],
                "volatility_reduction": geo_stats["std"],
            }
            return {"results": results, "ergodic_analysis": ergodic_analysis}

        return {"results": results}

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
