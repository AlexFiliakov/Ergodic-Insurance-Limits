"""Financial statement compilation and generation.

This module provides classes for generating standard financial statements
(Balance Sheet, Income Statement, Cash Flow Statement) from simulation data.
It supports both single trajectory and Monte Carlo aggregated reports with
reconciliation capabilities.

Example:
    Generate financial statements from a manufacturer simulation::

        from ergodic_insurance.src.manufacturer import WidgetManufacturer
        from ergodic_insurance.src.financial_statements import FinancialStatementGenerator

        # Run simulation
        manufacturer = WidgetManufacturer(config)
        for year in range(10):
            manufacturer.step()

        # Generate statements
        generator = FinancialStatementGenerator(manufacturer)
        balance_sheet = generator.generate_balance_sheet(year=5)
        income_statement = generator.generate_income_statement(year=5)
        cash_flow = generator.generate_cash_flow_statement(year=5)
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .manufacturer import WidgetManufacturer


@dataclass
class FinancialStatementConfig:
    """Configuration for financial statement generation.

    Attributes:
        currency_symbol: Symbol to use for currency formatting
        decimal_places: Number of decimal places for numeric values
        include_yoy_change: Whether to include year-over-year changes
        include_percentages: Whether to include percentage breakdowns
        fiscal_year_end: Month of fiscal year end (1-12)
        consolidate_monthly: Whether to consolidate monthly data into annual
    """

    currency_symbol: str = "$"
    decimal_places: int = 0
    include_yoy_change: bool = True
    include_percentages: bool = True
    fiscal_year_end: int = 12
    consolidate_monthly: bool = True


class FinancialStatementGenerator:
    """Generates financial statements from simulation data.

    This class compiles standard financial statements (Balance Sheet,
    Income Statement, Cash Flow) from manufacturer metrics history.
    It handles both annual and monthly data, performs reconciliation
    checks, and calculates derived financial metrics.

    Attributes:
        manufacturer_data: Raw simulation data from manufacturer
        config: Configuration for statement generation
        metrics_history: List of metrics dictionaries from simulation
        years_available: Number of years of data available
    """

    def __init__(
        self,
        manufacturer: Optional["WidgetManufacturer"] = None,
        manufacturer_data: Optional[Dict[str, Any]] = None,
        config: Optional[FinancialStatementConfig] = None,
    ):
        """Initialize financial statement generator.

        Args:
            manufacturer: WidgetManufacturer instance with simulation data
            manufacturer_data: Alternative dictionary of manufacturer data
            config: Configuration for statement generation

        Raises:
            ValueError: If neither manufacturer nor manufacturer_data provided
        """
        if manufacturer is not None:
            self.manufacturer_data = {
                "metrics_history": manufacturer.metrics_history,  # type: ignore[attr-defined]
                "initial_assets": manufacturer.initial_assets,  # type: ignore[attr-defined]
                "config": manufacturer.config,  # type: ignore[attr-defined]
            }
        elif manufacturer_data is not None:
            self.manufacturer_data = manufacturer_data
        else:
            raise ValueError("Either manufacturer or manufacturer_data must be provided")

        self.config = config or FinancialStatementConfig()
        self.metrics_history = self.manufacturer_data.get("metrics_history", [])
        self.years_available = len(self.metrics_history)

    def generate_balance_sheet(
        self, year: int, compare_years: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Generate balance sheet for specified year.

        Creates a standard balance sheet with assets, liabilities, and equity
        sections. Includes year-over-year comparisons if configured.

        Args:
            year: Year index (0-based) for balance sheet
            compare_years: Optional list of years to compare against

        Returns:
            DataFrame containing balance sheet data

        Raises:
            IndexError: If year is out of range
        """
        if year >= self.years_available or year < 0:
            raise IndexError(f"Year {year} out of range. Available: 0-{self.years_available-1}")

        metrics = self.metrics_history[year]

        # Build balance sheet structure
        balance_sheet_data: List[Tuple[str, Union[str, float, int], str, str]] = []

        # ASSETS SECTION
        balance_sheet_data.append(("ASSETS", "", "", ""))
        balance_sheet_data.append(("", "", "", ""))

        # Current Assets
        balance_sheet_data.append(("Current Assets", "", "", ""))
        working_capital = metrics.get("assets", 0) * 0.2  # Estimate if not directly available
        unrestricted_cash = metrics.get("available_assets", 0) - working_capital
        balance_sheet_data.append(("  Working Capital", working_capital, "", ""))
        balance_sheet_data.append(("  Unrestricted Cash", unrestricted_cash, "", ""))
        total_current = working_capital + unrestricted_cash
        balance_sheet_data.append(("  Total Current Assets", total_current, "", "subtotal"))
        balance_sheet_data.append(("", "", "", ""))

        # Fixed Assets
        balance_sheet_data.append(("Fixed Assets", "", "", ""))
        fixed_assets = (
            metrics.get("assets", 0) - total_current - metrics.get("restricted_assets", 0)
        )
        balance_sheet_data.append(("  Property, Plant & Equipment", fixed_assets, "", ""))
        balance_sheet_data.append(("  Total Fixed Assets", fixed_assets, "", "subtotal"))
        balance_sheet_data.append(("", "", "", ""))

        # Restricted Assets
        balance_sheet_data.append(("Restricted Assets", "", "", ""))
        collateral = metrics.get("collateral", 0)
        restricted_other = metrics.get("restricted_assets", 0) - collateral
        balance_sheet_data.append(("  Insurance Collateral", collateral, "", ""))
        balance_sheet_data.append(("  Letter of Credit", restricted_other, "", ""))
        total_restricted = metrics.get("restricted_assets", 0)
        balance_sheet_data.append(("  Total Restricted Assets", total_restricted, "", "subtotal"))
        balance_sheet_data.append(("", "", "", ""))

        # Total Assets
        total_assets = metrics.get("assets", 0)
        balance_sheet_data.append(("TOTAL ASSETS", total_assets, "", "total"))
        balance_sheet_data.append(("", "", "", ""))
        balance_sheet_data.append(("", "", "", ""))

        # LIABILITIES SECTION
        balance_sheet_data.append(("LIABILITIES", "", "", ""))
        balance_sheet_data.append(("", "", "", ""))

        # Current Liabilities
        balance_sheet_data.append(("Current Liabilities", "", "", ""))
        claim_liabilities = metrics.get("claim_liabilities", 0)
        # Estimate current portion (first year of payment schedule)
        current_claims = claim_liabilities * 0.1 if claim_liabilities > 0 else 0
        balance_sheet_data.append(("  Outstanding Claims (Current)", current_claims, "", ""))
        balance_sheet_data.append(("  Accounts Payable", 0, "", ""))  # Simplified
        balance_sheet_data.append(("  Total Current Liabilities", current_claims, "", "subtotal"))
        balance_sheet_data.append(("", "", "", ""))

        # Long-term Liabilities
        balance_sheet_data.append(("Long-term Liabilities", "", "", ""))
        long_term_claims = claim_liabilities - current_claims
        balance_sheet_data.append(("  Insurance Claim Reserves", long_term_claims, "", ""))
        balance_sheet_data.append(
            ("  Total Long-term Liabilities", long_term_claims, "", "subtotal")
        )
        balance_sheet_data.append(("", "", "", ""))

        # Total Liabilities
        total_liabilities = claim_liabilities
        balance_sheet_data.append(("TOTAL LIABILITIES", total_liabilities, "", "total"))
        balance_sheet_data.append(("", "", "", ""))
        balance_sheet_data.append(("", "", "", ""))

        # EQUITY SECTION
        balance_sheet_data.append(("EQUITY", "", "", ""))
        balance_sheet_data.append(("", "", "", ""))
        equity = metrics.get("equity", 0)
        balance_sheet_data.append(("  Retained Earnings", equity, "", ""))
        balance_sheet_data.append(("TOTAL EQUITY", equity, "", "total"))
        balance_sheet_data.append(("", "", "", ""))
        balance_sheet_data.append(("", "", "", ""))

        # Validation
        balance_sheet_data.append(
            ("TOTAL LIABILITIES + EQUITY", total_liabilities + equity, "", "total")
        )

        # Create DataFrame
        df = pd.DataFrame(
            balance_sheet_data, columns=["Item", f"Year {year}", "YoY Change %", "Type"]
        )

        # Add year-over-year comparison if requested
        if self.config.include_yoy_change and year > 0:
            self._add_yoy_comparison(df, year)

        # Add comparison years if specified
        if compare_years:
            for comp_year in compare_years:
                if comp_year < self.years_available:
                    self._add_comparison_year(df, comp_year)

        return df

    def generate_income_statement(
        self, year: int, compare_years: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Generate income statement for specified year.

        Creates a standard income statement showing revenue, expenses,
        and net income with optional year-over-year comparisons.

        Args:
            year: Year index (0-based) for income statement
            compare_years: Optional list of years to compare against

        Returns:
            DataFrame containing income statement data

        Raises:
            IndexError: If year is out of range
        """
        if year >= self.years_available or year < 0:
            raise IndexError(f"Year {year} out of range. Available: 0-{self.years_available-1}")

        metrics = self.metrics_history[year]

        # Build income statement structure
        income_data: List[Tuple[str, Union[str, float, int], str, str]] = []

        # Revenue Section
        income_data.append(("REVENUE", "", "", ""))
        revenue = metrics.get("revenue", 0)
        income_data.append(("  Sales Revenue", revenue, "", ""))
        income_data.append(("  Total Revenue", revenue, "", "subtotal"))
        income_data.append(("", "", "", ""))

        # Operating Expenses
        income_data.append(("OPERATING EXPENSES", "", "", ""))
        operating_income = metrics.get("operating_income", 0)
        operating_expenses = revenue - operating_income
        cogs = operating_expenses * 0.7  # Estimate COGS as 70% of operating expenses
        sga = operating_expenses * 0.3  # SG&A as 30%
        income_data.append(("  Cost of Goods Sold", cogs, "", ""))
        income_data.append(("  Selling, General & Admin", sga, "", ""))
        income_data.append(("  Total Operating Expenses", operating_expenses, "", "subtotal"))
        income_data.append(("", "", "", ""))

        # Operating Income
        income_data.append(("OPERATING INCOME", operating_income, "", "subtotal"))
        income_data.append(("", "", "", ""))

        # Other Income/Expenses
        income_data.append(("OTHER INCOME (EXPENSES)", "", "", ""))

        # Insurance costs - estimate from metrics
        insurance_premium = 0  # Would need to track separately
        insurance_claims = 0  # Company portion of claims

        income_data.append(("  Insurance Premiums", -insurance_premium, "", ""))
        income_data.append(("  Insurance Claim Expenses", -insurance_claims, "", ""))
        income_data.append(("  Interest Income", 0, "", ""))
        income_data.append(
            ("  Total Other", -(insurance_premium + insurance_claims), "", "subtotal")
        )
        income_data.append(("", "", "", ""))

        # Pre-tax Income
        pretax_income = operating_income - insurance_premium - insurance_claims
        income_data.append(("INCOME BEFORE TAXES", pretax_income, "", "subtotal"))
        income_data.append(("", "", "", ""))

        # Taxes
        tax_rate = (
            self.manufacturer_data.get("config", {}).tax_rate
            if hasattr(self.manufacturer_data.get("config", {}), "tax_rate")
            else 0.25
        )
        taxes = pretax_income * tax_rate if pretax_income > 0 else 0
        income_data.append(("  Income Tax Expense", taxes, "", ""))
        income_data.append(("", "", "", ""))

        # Net Income
        net_income = metrics.get("net_income", pretax_income - taxes)
        income_data.append(("NET INCOME", net_income, "", "total"))
        income_data.append(("", "", "", ""))

        # Per-share data (if applicable)
        income_data.append(("", "", "", ""))
        income_data.append(("KEY METRICS", "", "", ""))
        income_data.append(
            ("  Operating Margin %", metrics.get("operating_margin", 0) * 100, "%", "")
        )
        income_data.append(
            ("  Net Margin %", (net_income / revenue * 100) if revenue > 0 else 0, "%", "")
        )
        income_data.append(("  ROE %", metrics.get("roe", 0) * 100, "%", ""))
        income_data.append(("  ROA %", metrics.get("roa", 0) * 100, "%", ""))

        # Create DataFrame
        df = pd.DataFrame(income_data, columns=["Item", f"Year {year}", "Unit", "Type"])

        # Add year-over-year comparison if requested
        if self.config.include_yoy_change and year > 0:
            self._add_yoy_comparison_income(df, year)

        # Add comparison years if specified
        if compare_years:
            for comp_year in compare_years:
                if comp_year < self.years_available:
                    self._add_comparison_year_income(df, comp_year)

        return df

    def generate_cash_flow_statement(self, year: int, method: str = "indirect") -> pd.DataFrame:
        """Generate cash flow statement for specified year.

        Creates a cash flow statement showing operating, investing, and
        financing activities using either direct or indirect method.

        Args:
            year: Year index (0-based) for cash flow statement
            method: 'direct' or 'indirect' method for operating activities

        Returns:
            DataFrame containing cash flow statement data

        Raises:
            IndexError: If year is out of range
        """
        if year >= self.years_available or year < 0:
            raise IndexError(f"Year {year} out of range. Available: 0-{self.years_available-1}")

        metrics = self.metrics_history[year]
        prev_metrics = self.metrics_history[year - 1] if year > 0 else {}

        # Build cash flow statement
        cash_flow_data: List[Tuple[str, Union[str, float, int], str]] = []

        # Operating Activities
        cash_flow_data.append(("OPERATING ACTIVITIES", "", ""))

        if method == "indirect":
            # Start with net income
            net_income = metrics.get("net_income", 0)
            cash_flow_data.append(("  Net Income", net_income, ""))
            cash_flow_data.append(("  Adjustments to reconcile:", "", ""))

            # Add back non-cash items (simplified)
            depreciation = metrics.get("assets", 0) * 0.05  # Estimate 5% depreciation
            cash_flow_data.append(("    Depreciation", depreciation, ""))

            # Changes in working capital
            if year > 0:
                wc_change = (metrics.get("assets", 0) - prev_metrics.get("assets", 0)) * 0.2
                cash_flow_data.append(("    Change in Working Capital", -wc_change, ""))

                # Change in claim liabilities
                claim_change = metrics.get("claim_liabilities", 0) - prev_metrics.get(
                    "claim_liabilities", 0
                )
                cash_flow_data.append(("    Change in Claim Liabilities", claim_change, ""))
            else:
                wc_change = 0
                claim_change = 0

            operating_cash = net_income + depreciation - wc_change + claim_change
        else:
            # Direct method
            revenue = metrics.get("revenue", 0)
            cash_flow_data.append(("  Cash from Customers", revenue, ""))

            # Cash payments
            operating_income = metrics.get("operating_income", 0)
            operating_expenses = revenue - operating_income
            cash_flow_data.append(("  Cash to Suppliers/Employees", -operating_expenses, ""))

            # Other operating cash flows
            taxes_paid = operating_income * 0.25  # Estimate
            cash_flow_data.append(("  Taxes Paid", -taxes_paid, ""))

            operating_cash = revenue - operating_expenses - taxes_paid

        cash_flow_data.append(("  Net Cash from Operating Activities", operating_cash, "subtotal"))
        cash_flow_data.append(("", "", ""))

        # Investing Activities
        cash_flow_data.append(("INVESTING ACTIVITIES", "", ""))

        # Capital expenditures (simplified)
        if year > 0:
            asset_change = metrics.get("assets", 0) - prev_metrics.get("assets", 0)
            capex = max(0, asset_change * 0.3)  # Estimate 30% of asset growth as capex
        else:
            capex = 0

        cash_flow_data.append(("  Capital Expenditures", -capex, ""))
        cash_flow_data.append(("  Net Cash from Investing Activities", -capex, "subtotal"))
        cash_flow_data.append(("", "", ""))

        # Financing Activities
        cash_flow_data.append(("FINANCING ACTIVITIES", "", ""))

        # Simplified - no debt or equity transactions in base model
        financing_cash = 0
        cash_flow_data.append(("  Net Cash from Financing Activities", financing_cash, "subtotal"))
        cash_flow_data.append(("", "", ""))

        # Net Change in Cash
        net_cash_change = operating_cash - capex + financing_cash
        cash_flow_data.append(("NET CHANGE IN CASH", net_cash_change, "total"))

        # Cash reconciliation
        if year > 0:
            beginning_cash = prev_metrics.get("available_assets", 0)
        else:
            beginning_cash = self.manufacturer_data.get("initial_assets", 0)

        ending_cash = metrics.get("available_assets", 0)

        cash_flow_data.append(("", "", ""))
        cash_flow_data.append(("Cash - Beginning of Period", beginning_cash, ""))
        cash_flow_data.append(("Cash - End of Period", ending_cash, ""))

        # Create DataFrame
        df = pd.DataFrame(cash_flow_data, columns=["Item", f"Year {year}", "Type"])

        return df

    def generate_reconciliation_report(self, year: int) -> pd.DataFrame:
        """Generate reconciliation report for financial statements.

        Validates that financial statements balance and reconcile properly,
        checking key accounting identities and relationships.

        Args:
            year: Year index (0-based) for reconciliation

        Returns:
            DataFrame containing reconciliation checks and results
        """
        if year >= self.years_available or year < 0:
            raise IndexError(f"Year {year} out of range. Available: 0-{self.years_available-1}")

        metrics = self.metrics_history[year]
        reconciliation_data: List[Tuple[str, Union[str, float, int], str, str]] = []

        # Balance Sheet Equation Check
        reconciliation_data.append(("BALANCE SHEET RECONCILIATION", "", "", ""))
        assets = metrics.get("assets", 0)
        liabilities = metrics.get("claim_liabilities", 0)
        equity = metrics.get("equity", 0)

        balance_check = abs(assets - (liabilities + equity)) < 0.01
        reconciliation_data.append(("  Assets", assets, "", ""))
        reconciliation_data.append(("  Liabilities + Equity", liabilities + equity, "", ""))
        reconciliation_data.append(("  Difference", assets - (liabilities + equity), "", ""))
        reconciliation_data.append(
            ("  Status", "BALANCED" if balance_check else "IMBALANCED", "", "status")
        )
        reconciliation_data.append(("", "", "", ""))

        # Net Assets Check
        reconciliation_data.append(("NET ASSETS RECONCILIATION", "", "", ""))
        net_assets = metrics.get("net_assets", 0)
        calc_net_assets = assets - metrics.get("restricted_assets", 0)
        net_assets_check = abs(net_assets - calc_net_assets) < 0.01

        reconciliation_data.append(("  Reported Net Assets", net_assets, "", ""))
        reconciliation_data.append(("  Calculated (Assets - Restricted)", calc_net_assets, "", ""))
        reconciliation_data.append(("  Difference", net_assets - calc_net_assets, "", ""))
        reconciliation_data.append(
            ("  Status", "MATCHED" if net_assets_check else "MISMATCHED", "", "status")
        )
        reconciliation_data.append(("", "", "", ""))

        # Collateral Reconciliation
        reconciliation_data.append(("COLLATERAL RECONCILIATION", "", "", ""))
        collateral = metrics.get("collateral", 0)
        restricted = metrics.get("restricted_assets", 0)
        collateral_check = collateral <= restricted

        reconciliation_data.append(("  Insurance Collateral", collateral, "", ""))
        reconciliation_data.append(("  Total Restricted Assets", restricted, "", ""))
        reconciliation_data.append(
            ("  Status", "VALID" if collateral_check else "INVALID", "", "status")
        )
        reconciliation_data.append(("", "", "", ""))

        # Solvency Check
        reconciliation_data.append(("SOLVENCY CHECK", "", "", ""))
        is_solvent = metrics.get("is_solvent", True)
        equity_positive = equity > 0

        reconciliation_data.append(("  Equity", equity, "", ""))
        reconciliation_data.append(("  Solvency Flag", is_solvent, "", ""))
        reconciliation_data.append(
            ("  Status", "SOLVENT" if equity_positive else "INSOLVENT", "", "status")
        )

        # Create DataFrame
        df = pd.DataFrame(reconciliation_data, columns=["Check", "Value", "Expected", "Type"])

        return df

    def _add_yoy_comparison(self, df: pd.DataFrame, year: int) -> None:
        """Add year-over-year comparison to balance sheet DataFrame.

        Args:
            df: DataFrame to modify
            year: Current year for comparison
        """
        if year > 0:
            prev_metrics = self.metrics_history[year - 1]
            curr_metrics = self.metrics_history[year]

            # Calculate YoY changes for key items
            yoy_changes = {}

            # Assets
            if prev_metrics.get("assets", 0) > 0:
                yoy_changes["TOTAL ASSETS"] = (
                    (curr_metrics.get("assets", 0) - prev_metrics.get("assets", 0))
                    / prev_metrics.get("assets", 0)
                    * 100
                )

            # Equity
            if prev_metrics.get("equity", 0) > 0:
                yoy_changes["TOTAL EQUITY"] = (
                    (curr_metrics.get("equity", 0) - prev_metrics.get("equity", 0))
                    / prev_metrics.get("equity", 0)
                    * 100
                )

            # Update DataFrame with YoY changes
            for index, row in df.iterrows():
                if row["Item"].strip() in yoy_changes:
                    df.at[index, "YoY Change %"] = f"{yoy_changes[row['Item'].strip()]:.1f}%"

    def _add_yoy_comparison_income(self, df: pd.DataFrame, year: int) -> None:
        """Add year-over-year comparison to income statement DataFrame.

        Args:
            df: DataFrame to modify
            year: Current year for comparison
        """
        if year > 0:
            prev_metrics = self.metrics_history[year - 1]
            curr_metrics = self.metrics_history[year]

            # Add previous year column
            df[f"Year {year-1}"] = ""

            # Calculate YoY for revenue
            if prev_metrics.get("revenue", 0) > 0:
                revenue_yoy = (
                    (curr_metrics.get("revenue", 0) - prev_metrics.get("revenue", 0))
                    / prev_metrics.get("revenue", 0)
                    * 100
                )
                # Find revenue row and add YoY
                for index, row in df.iterrows():
                    if "Sales Revenue" in row["Item"]:
                        df.at[index, f"Year {year-1}"] = prev_metrics.get("revenue", 0)

    def _add_comparison_year(self, df: pd.DataFrame, comp_year: int) -> None:
        """Add comparison year data to balance sheet DataFrame.

        Args:
            df: DataFrame to modify
            comp_year: Year to add for comparison
        """
        if comp_year < self.years_available:
            comp_metrics = self.metrics_history[comp_year]
            df[f"Year {comp_year}"] = ""

            # Add comparison year values for key items
            for index, row in df.iterrows():
                item = row["Item"].strip()
                if item == "TOTAL ASSETS":
                    df.at[index, f"Year {comp_year}"] = comp_metrics.get("assets", 0)
                elif item == "TOTAL EQUITY":
                    df.at[index, f"Year {comp_year}"] = comp_metrics.get("equity", 0)
                elif item == "TOTAL LIABILITIES":
                    df.at[index, f"Year {comp_year}"] = comp_metrics.get("claim_liabilities", 0)

    def _add_comparison_year_income(self, df: pd.DataFrame, comp_year: int) -> None:
        """Add comparison year data to income statement DataFrame.

        Args:
            df: DataFrame to modify
            comp_year: Year to add for comparison
        """
        if comp_year < self.years_available:
            comp_metrics = self.metrics_history[comp_year]
            df[f"Year {comp_year}"] = ""

            # Add comparison year values for key items
            for index, row in df.iterrows():
                item = row["Item"].strip()
                if "Sales Revenue" in item:
                    df.at[index, f"Year {comp_year}"] = comp_metrics.get("revenue", 0)
                elif item == "NET INCOME":
                    df.at[index, f"Year {comp_year}"] = comp_metrics.get("net_income", 0)


class MonteCarloStatementAggregator:
    """Aggregates financial statements across Monte Carlo simulations.

    This class processes multiple simulation trajectories to create
    statistical summaries of financial statements, showing means,
    percentiles, and confidence intervals.

    Attributes:
        results: Monte Carlo simulation results
        config: Configuration for statement generation
    """

    def __init__(
        self,
        monte_carlo_results: Union[List[Dict], pd.DataFrame],
        config: Optional[FinancialStatementConfig] = None,
    ):
        """Initialize Monte Carlo statement aggregator.

        Args:
            monte_carlo_results: Results from Monte Carlo simulations
            config: Configuration for statement generation
        """
        self.results = monte_carlo_results
        self.config = config or FinancialStatementConfig()

    def aggregate_balance_sheets(
        self, year: int, percentiles: List[float] = [5, 25, 50, 75, 95]
    ) -> pd.DataFrame:
        """Aggregate balance sheets across simulations.

        Args:
            year: Year index to aggregate
            percentiles: Percentiles to calculate

        Returns:
            DataFrame with aggregated balance sheet statistics
        """
        # Implementation for aggregating balance sheets across simulations
        # This would process multiple trajectories and calculate statistics
        raise NotImplementedError("Monte Carlo aggregation not yet implemented")

    def aggregate_income_statements(
        self, year: int, percentiles: List[float] = [5, 25, 50, 75, 95]
    ) -> pd.DataFrame:
        """Aggregate income statements across simulations.

        Args:
            year: Year index to aggregate
            percentiles: Percentiles to calculate

        Returns:
            DataFrame with aggregated income statement statistics
        """
        # Implementation for aggregating income statements
        raise NotImplementedError("Monte Carlo aggregation not yet implemented")

    def generate_convergence_analysis(self) -> pd.DataFrame:
        """Analyze convergence of financial metrics across simulations.

        Returns:
            DataFrame showing convergence statistics
        """
        # Analyze how financial metrics converge with more simulations
        raise NotImplementedError("Convergence analysis not yet implemented")
