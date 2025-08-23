"""Simulation engine for time evolution of widget manufacturer model.

This module provides the main simulation engine that orchestrates the
time evolution of the widget manufacturer financial model, managing
claim events, financial calculations, and result collection.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .claim_generator import ClaimEvent, ClaimGenerator
from .manufacturer import WidgetManufacturer

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

    def summary_stats(self) -> Dict[str, float]:
        """Calculate summary statistics for the simulation.

        Returns:
            Dictionary of key statistics.
        """
        # Filter out NaN values for ROE calculation
        valid_roe = self.roe[~np.isnan(self.roe)]

        return {
            "mean_roe": np.mean(valid_roe) if len(valid_roe) > 0 else 0.0,
            "std_roe": np.std(valid_roe) if len(valid_roe) > 0 else 0.0,
            "median_roe": np.median(valid_roe) if len(valid_roe) > 0 else 0.0,
            "final_assets": self.assets[-1],
            "final_equity": self.equity[-1],
            "total_claims": np.sum(self.claim_amounts),
            "claim_frequency": np.mean(self.claim_counts),
            "survived": self.insolvency_year is None,
            "insolvency_year": float(self.insolvency_year)
            if self.insolvency_year is not None
            else 0.0,
        }


class Simulation:
    """Simulation engine for widget manufacturer time evolution.

    The main simulation class that coordinates the time evolution of the
    widget manufacturer model, processing claims and tracking financial
    performance over the specified time horizon.
    """

    def __init__(
        self,
        manufacturer: WidgetManufacturer,
        claim_generator: Optional[ClaimGenerator] = None,
        time_horizon: int = 100,
        seed: Optional[int] = None,
    ):
        """Initialize simulation.

        Args:
            manufacturer: WidgetManufacturer instance to simulate.
            claim_generator: ClaimGenerator for creating insurance claims.
            time_horizon: Number of years to simulate.
            seed: Random seed for reproducibility.
        """
        self.manufacturer = manufacturer
        self.claim_generator = claim_generator or ClaimGenerator(seed=seed)
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

        # Apply insurance (simplified - actual implementation would use process_insurance_claim)
        for claim in claims:
            # Assuming deductible of 1M and limit of 10M for now
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
            claim_counts=self.claim_counts[: year + 1]
            if self.insolvency_year
            else self.claim_counts,
            claim_amounts=self.claim_amounts[: year + 1]
            if self.insolvency_year
            else self.claim_amounts,
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
