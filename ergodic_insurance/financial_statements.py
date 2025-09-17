"""Financial statement compilation and generation.

This module provides classes for generating standard financial statements
(Balance Sheet, Income Statement, Cash Flow Statement) from simulation data.
It supports both single trajectory and Monte Carlo aggregated reports with
reconciliation capabilities.

Example:
    Generate financial statements from a manufacturer simulation::

        from ergodic_insurance.manufacturer import WidgetManufacturer
        from ergodic_insurance.financial_statements import FinancialStatementGenerator

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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

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
        self.manufacturer = manufacturer
        if manufacturer is not None:
            self.manufacturer_data = {
                "metrics_history": manufacturer.metrics_history,
                "initial_assets": manufacturer.config.initial_assets,
                "config": manufacturer.config,
            }
        elif manufacturer_data is not None:
            self.manufacturer_data = manufacturer_data
        else:
            raise ValueError("Either manufacturer or manufacturer_data must be provided")

        self.config = config or FinancialStatementConfig()
        self._update_metrics_cache()

    def _update_metrics_cache(self):
        """Update the cached metrics from manufacturer data."""
        if self.manufacturer is not None:
            # Get fresh metrics from manufacturer
            self.manufacturer_data["metrics_history"] = self.manufacturer.metrics_history

        metrics = self.manufacturer_data.get("metrics_history", [])
        self.metrics_history: List[Dict[str, float]] = metrics if isinstance(metrics, list) else []
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
        # Update metrics cache to get latest data
        self._update_metrics_cache()

        if year >= self.years_available or year < 0:
            raise IndexError(f"Year {year} out of range. Available: 0-{self.years_available-1}")

        metrics = self.metrics_history[year]

        # Build balance sheet structure
        balance_sheet_data: List[Tuple[str, Union[str, float, int], str, str]] = []

        # Build main sections
        self._build_assets_section(balance_sheet_data, metrics)
        self._build_liabilities_section(balance_sheet_data, metrics)
        self._build_equity_section(balance_sheet_data, metrics)

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

    def _build_assets_section(
        self, data: List[Tuple[str, Union[str, float, int], str, str]], metrics: Dict[str, float]
    ) -> None:
        """Build assets section of balance sheet with GAAP structure."""
        # ASSETS SECTION
        data.append(("ASSETS", "", "", ""))
        data.append(("", "", "", ""))

        # Current Assets
        data.append(("Current Assets", "", "", ""))

        # Use detailed components if available, otherwise estimate
        cash = metrics.get("cash", metrics.get("available_assets", 0) * 0.3)
        accounts_receivable = metrics.get("accounts_receivable", 0)
        inventory = metrics.get("inventory", 0)
        prepaid_insurance = metrics.get("prepaid_insurance", 0)

        data.append(("  Cash and Cash Equivalents", cash, "", ""))
        data.append(("  Accounts Receivable", accounts_receivable, "", ""))
        data.append(("  Inventory", inventory, "", ""))
        data.append(("  Prepaid Insurance", prepaid_insurance, "", ""))

        total_current = cash + accounts_receivable + inventory + prepaid_insurance
        data.append(("  Total Current Assets", total_current, "", "subtotal"))
        data.append(("", "", "", ""))

        # Non-Current Assets
        data.append(("Non-Current Assets", "", "", ""))

        # Property, Plant & Equipment
        gross_ppe = metrics.get("gross_ppe", metrics.get("assets", 0) * 0.7)
        accumulated_depreciation = metrics.get("accumulated_depreciation", 0)
        net_ppe = gross_ppe - accumulated_depreciation

        data.append(("  Property, Plant & Equipment (Gross)", gross_ppe, "", ""))
        data.append(("  Less: Accumulated Depreciation", -accumulated_depreciation, "", ""))
        data.append(("  Net Property, Plant & Equipment", net_ppe, "", "subtotal"))
        data.append(("", "", "", ""))

        # Restricted Assets
        data.append(("Restricted Assets", "", "", ""))
        collateral = metrics.get("collateral", 0)
        restricted_other = metrics.get("restricted_assets", 0) - collateral
        data.append(("  Insurance Collateral", collateral, "", ""))
        if restricted_other > 0:
            data.append(("  Other Restricted Assets", restricted_other, "", ""))
        total_restricted = metrics.get("restricted_assets", 0)
        data.append(("  Total Restricted Assets", total_restricted, "", "subtotal"))
        data.append(("", "", "", ""))

        # Total Assets (recalculate from components for consistency)
        total_assets = total_current + net_ppe + total_restricted
        data.append(("TOTAL ASSETS", total_assets, "", "total"))
        data.append(("", "", "", ""))
        data.append(("", "", "", ""))

    def _build_liabilities_section(
        self, data: List[Tuple[str, Union[str, float, int], str, str]], metrics: Dict[str, float]
    ) -> None:
        """Build liabilities section of balance sheet with GAAP structure."""
        # LIABILITIES SECTION
        data.append(("LIABILITIES", "", "", ""))
        data.append(("", "", "", ""))

        # Current Liabilities
        data.append(("Current Liabilities", "", "", ""))

        # Use detailed components if available
        accounts_payable = metrics.get("accounts_payable", 0)
        accrued_expenses = metrics.get("accrued_expenses", 0)

        # Estimate current portion of claims (first year of payment schedule)
        claim_liabilities = metrics.get("claim_liabilities", 0)
        current_claims = claim_liabilities * 0.1 if claim_liabilities > 0 else 0

        data.append(("  Accounts Payable", accounts_payable, "", ""))
        data.append(("  Accrued Expenses", accrued_expenses, "", ""))
        data.append(("  Current Portion of Claim Liabilities", current_claims, "", ""))

        total_current_liabilities = accounts_payable + accrued_expenses + current_claims
        data.append(("  Total Current Liabilities", total_current_liabilities, "", "subtotal"))
        data.append(("", "", "", ""))

        # Non-Current Liabilities
        data.append(("Non-Current Liabilities", "", "", ""))
        long_term_claims = claim_liabilities - current_claims
        data.append(("  Long-Term Claim Reserves", long_term_claims, "", ""))
        data.append(("  Total Non-Current Liabilities", long_term_claims, "", "subtotal"))
        data.append(("", "", "", ""))

        # Total Liabilities
        total_liabilities = total_current_liabilities + long_term_claims
        data.append(("TOTAL LIABILITIES", total_liabilities, "", "total"))
        data.append(("", "", "", ""))
        data.append(("", "", "", ""))

    def _build_equity_section(
        self, data: List[Tuple[str, Union[str, float, int], str, str]], metrics: Dict[str, float]
    ) -> None:
        """Build equity section of balance sheet."""
        # EQUITY SECTION
        data.append(("EQUITY", "", "", ""))
        data.append(("", "", "", ""))

        # Calculate total assets (must match what's shown in assets section)
        cash = metrics.get("cash", 0)
        accounts_receivable = metrics.get("accounts_receivable", 0)
        inventory = metrics.get("inventory", 0)
        prepaid = metrics.get("prepaid_insurance", 0)
        total_current = cash + accounts_receivable + inventory + prepaid

        net_ppe = metrics.get("net_ppe", 0)
        restricted_assets = metrics.get("restricted_assets", 0)
        total_assets = total_current + net_ppe + restricted_assets

        # Calculate total liabilities
        accounts_payable = metrics.get("accounts_payable", 0)
        accrued_expenses = metrics.get("accrued_expenses", 0)
        claim_liabilities = metrics.get("claim_liabilities", 0)
        total_liabilities = accounts_payable + accrued_expenses + claim_liabilities

        # Equity = Assets - Liabilities (basic accounting equation)
        equity_for_balance_sheet = total_assets - total_liabilities

        data.append(("  Retained Earnings", equity_for_balance_sheet, "", ""))
        data.append(("TOTAL EQUITY", equity_for_balance_sheet, "", "total"))
        data.append(("", "", "", ""))
        data.append(("", "", "", ""))

        # Validation
        data.append(
            (
                "TOTAL LIABILITIES + EQUITY",
                total_liabilities + equity_for_balance_sheet,
                "",
                "total",
            )
        )

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
        # Update metrics cache to get latest data
        self._update_metrics_cache()

        if year >= self.years_available or year < 0:
            raise IndexError(f"Year {year} out of range. Available: 0-{self.years_available-1}")

        metrics = self.metrics_history[year]

        # Build income statement structure
        income_data: List[Tuple[str, Union[str, float, int], str, str]] = []

        # Build main sections
        revenue = self._build_revenue_section(income_data, metrics)
        pretax_income = self._build_expenses_section(income_data, metrics, revenue)
        self._build_income_bottom_section(income_data, metrics, pretax_income, revenue)

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

    def _build_revenue_section(
        self, data: List[Tuple[str, Union[str, float, int], str, str]], metrics: Dict[str, float]
    ) -> float:
        """Build revenue section of income statement and return total revenue."""
        # Revenue Section
        data.append(("REVENUE", "", "", ""))
        revenue = metrics.get("revenue", 0)
        data.append(("  Sales Revenue", revenue, "", ""))
        data.append(("  Total Revenue", revenue, "", "subtotal"))
        data.append(("", "", "", ""))
        return revenue

    def _build_expenses_section(
        self,
        data: List[Tuple[str, Union[str, float, int], str, str]],
        metrics: Dict[str, float],
        revenue: float,
    ) -> float:
        """Build expenses section and return pretax income."""
        # Operating Expenses
        data.append(("OPERATING EXPENSES", "", "", ""))
        operating_income = metrics.get("operating_income", 0)
        operating_expenses = revenue - operating_income
        cogs = operating_expenses * 0.7  # Estimate COGS as 70% of operating expenses
        sga = operating_expenses * 0.3  # SG&A as 30%
        data.append(("  Cost of Goods Sold", cogs, "", ""))
        data.append(("  Selling, General & Admin", sga, "", ""))
        data.append(("  Total Operating Expenses", operating_expenses, "", "subtotal"))
        data.append(("", "", "", ""))

        # Operating Income
        data.append(("OPERATING INCOME", operating_income, "", "subtotal"))
        data.append(("", "", "", ""))

        # Other Income/Expenses
        data.append(("OTHER INCOME (EXPENSES)", "", "", ""))

        # Pull actual insurance costs from metrics
        # These values are tracked in manufacturer.py via period_insurance_premiums
        # and period_insurance_losses, and included in the metrics dictionary
        insurance_premium = metrics.get("insurance_premiums", 0)
        insurance_claims = metrics.get("insurance_losses", 0)

        data.append(("  Insurance Premiums", -insurance_premium, "", ""))
        data.append(("  Insurance Claim Expenses", -insurance_claims, "", ""))
        data.append(("  Interest Income", 0, "", ""))
        data.append(("  Total Other", -(insurance_premium + insurance_claims), "", "subtotal"))
        data.append(("", "", "", ""))

        # Pre-tax Income
        pretax_income = operating_income - insurance_premium - insurance_claims
        data.append(("INCOME BEFORE TAXES", pretax_income, "", "subtotal"))
        data.append(("", "", "", ""))

        return pretax_income

    def _build_income_bottom_section(
        self,
        data: List[Tuple[str, Union[str, float, int], str, str]],
        metrics: Dict[str, float],
        pretax_income: float,
        revenue: float,
    ) -> None:
        """Build the bottom section with taxes, net income, and metrics."""
        # Taxes
        config = self.manufacturer_data.get("config")
        tax_rate = config.tax_rate if config and hasattr(config, "tax_rate") else 0.25
        taxes = pretax_income * tax_rate if pretax_income > 0 else 0
        data.append(("  Income Tax Expense", taxes, "", ""))
        data.append(("", "", "", ""))

        # Net Income
        net_income = metrics.get("net_income", pretax_income - taxes)
        data.append(("NET INCOME", net_income, "", "total"))
        data.append(("", "", "", ""))

        # Key metrics
        data.append(("", "", "", ""))
        data.append(("KEY METRICS", "", "", ""))
        data.append(
            ("  Base Operating Margin %", metrics.get("base_operating_margin", 0) * 100, "%", "")
        )
        data.append(
            (
                "  Actual Operating Margin %",
                metrics.get("actual_operating_margin", 0) * 100,
                "%",
                "",
            )
        )
        data.append(("  Net Margin %", (net_income / revenue * 100) if revenue > 0 else 0, "%", ""))
        data.append(("  ROE %", metrics.get("roe", 0) * 100, "%", ""))
        data.append(("  ROA %", metrics.get("roa", 0) * 100, "%", ""))

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
        # Update metrics cache to get latest data
        self._update_metrics_cache()

        if year >= self.years_available or year < 0:
            raise IndexError(f"Year {year} out of range. Available: 0-{self.years_available-1}")

        metrics = self.metrics_history[year]
        prev_metrics = self.metrics_history[year - 1] if year > 0 else {}

        # Build cash flow statement
        cash_flow_data: List[Tuple[str, Union[str, float, int], str]] = []

        # Build main sections
        operating_cash = self._build_operating_activities(
            cash_flow_data, metrics, prev_metrics, year, method=method
        )
        capex = self._build_investing_activities(cash_flow_data, metrics, prev_metrics, year)
        financing_cash = self._build_financing_activities(cash_flow_data)

        # Net Change in Cash
        net_cash_change = operating_cash - capex + financing_cash
        cash_flow_data.append(("NET CHANGE IN CASH", net_cash_change, "total"))

        # Cash reconciliation
        self._add_cash_reconciliation(cash_flow_data, metrics, prev_metrics, year)

        # Create DataFrame
        df = pd.DataFrame(cash_flow_data, columns=["Item", f"Year {year}", "Type"])

        return df

    def _build_operating_activities(
        self,
        data: List[Tuple[str, Union[str, float, int], str]],
        metrics: Dict[str, float],
        prev_metrics: Dict[str, float],
        year: int,
        **kwargs,
    ) -> float:
        """Build operating activities section and return operating cash flow."""
        method = kwargs.get("method", "indirect")
        data.append(("OPERATING ACTIVITIES", "", ""))

        if method == "indirect":
            operating_cash = self._build_indirect_operating(data, metrics, prev_metrics, year)
        else:
            operating_cash = self._build_direct_operating(data, metrics)

        data.append(("  Net Cash from Operating Activities", operating_cash, "subtotal"))
        data.append(("", "", ""))
        return operating_cash

    def _build_indirect_operating(
        self,
        data: List[Tuple[str, Union[str, float, int], str]],
        metrics: Dict[str, float],
        prev_metrics: Dict[str, float],
        year: int,
    ) -> float:
        """Build indirect method operating section."""
        # Start with net income
        net_income = metrics.get("net_income", 0)
        data.append(("  Net Income", net_income, ""))
        data.append(("  Adjustments to reconcile:", "", ""))

        # Add back non-cash items (simplified)
        depreciation = metrics.get("assets", 0) * 0.05  # Estimate 5% depreciation
        data.append(("    Depreciation", depreciation, ""))

        # Changes in working capital
        if year > 0:
            wc_change = (metrics.get("assets", 0) - prev_metrics.get("assets", 0)) * 0.2
            data.append(("    Change in Working Capital", -wc_change, ""))

            # Change in claim liabilities
            claim_change = metrics.get("claim_liabilities", 0) - prev_metrics.get(
                "claim_liabilities", 0
            )
            data.append(("    Change in Claim Liabilities", claim_change, ""))
        else:
            wc_change = 0
            claim_change = 0

        return net_income + depreciation - wc_change + claim_change

    def _build_direct_operating(
        self, data: List[Tuple[str, Union[str, float, int], str]], metrics: Dict[str, float]
    ) -> float:
        """Build direct method operating section."""
        revenue = metrics.get("revenue", 0)
        data.append(("  Cash from Customers", revenue, ""))

        # Cash payments
        operating_income = metrics.get("operating_income", 0)
        operating_expenses = revenue - operating_income
        data.append(("  Cash to Suppliers/Employees", -operating_expenses, ""))

        # Other operating cash flows
        taxes_paid = operating_income * 0.25  # Estimate
        data.append(("  Taxes Paid", -taxes_paid, ""))

        return revenue - operating_expenses - taxes_paid

    def _build_investing_activities(
        self,
        data: List[Tuple[str, Union[str, float, int], str]],
        metrics: Dict[str, float],
        prev_metrics: Dict[str, float],
        year: int,
    ) -> float:
        """Build investing activities section and return capex."""
        data.append(("INVESTING ACTIVITIES", "", ""))

        # Capital expenditures (simplified)
        if year > 0:
            asset_change = metrics.get("assets", 0) - prev_metrics.get("assets", 0)
            capex = max(0, asset_change * 0.3)  # Estimate 30% of asset growth as capex
        else:
            capex = 0

        data.append(("  Capital Expenditures", -capex, ""))
        data.append(("  Net Cash from Investing Activities", -capex, "subtotal"))
        data.append(("", "", ""))
        return capex

    def _build_financing_activities(
        self, data: List[Tuple[str, Union[str, float, int], str]]
    ) -> float:
        """Build financing activities section and return financing cash."""
        data.append(("FINANCING ACTIVITIES", "", ""))

        # Simplified - no debt or equity transactions in base model
        financing_cash = 0
        data.append(("  Net Cash from Financing Activities", financing_cash, "subtotal"))
        data.append(("", "", ""))
        return financing_cash

    def _add_cash_reconciliation(
        self,
        data: List[Tuple[str, Union[str, float, int], str]],
        metrics: Dict[str, float],
        prev_metrics: Dict[str, float],
        year: int,
    ) -> None:
        """Add cash reconciliation section."""
        if year > 0:
            beginning_cash = prev_metrics.get("available_assets", 0)
        else:
            initial_val = self.manufacturer_data.get("initial_assets", 0.0)
            beginning_cash = float(initial_val) if isinstance(initial_val, (int, float)) else 0.0

        ending_cash = metrics.get("available_assets", 0)

        data.append(("", "", ""))
        data.append(("Cash - Beginning of Period", beginning_cash, ""))
        data.append(("Cash - End of Period", ending_cash, ""))

    def generate_reconciliation_report(self, year: int) -> pd.DataFrame:
        """Generate reconciliation report for financial statements.

        Validates that financial statements balance and reconcile properly,
        checking key accounting identities and relationships.

        Args:
            year: Year index (0-based) for reconciliation

        Returns:
            DataFrame containing reconciliation checks and results
        """
        # Update metrics cache to get latest data
        self._update_metrics_cache()

        if year >= self.years_available or year < 0:
            raise IndexError(f"Year {year} out of range. Available: 0-{self.years_available-1}")

        metrics = self.metrics_history[year]
        reconciliation_data: List[Tuple[str, Union[str, float, int], str, str]] = []

        # Perform all reconciliation checks
        self._check_balance_sheet_equation(reconciliation_data, metrics)
        self._check_net_assets(reconciliation_data, metrics)
        self._check_collateral(reconciliation_data, metrics)
        self._check_solvency(reconciliation_data, metrics)

        # Create DataFrame
        df = pd.DataFrame(reconciliation_data, columns=["Check", "Value", "Expected", "Type"])

        return df

    def _check_balance_sheet_equation(
        self, data: List[Tuple[str, Union[str, float, int], str, str]], metrics: Dict[str, float]
    ) -> None:
        """Check if balance sheet equation balances."""
        data.append(("BALANCE SHEET RECONCILIATION", "", "", ""))
        assets = metrics.get("assets", 0)
        liabilities = metrics.get("claim_liabilities", 0)
        equity = metrics.get("equity", 0)

        balance_check = abs(assets - (liabilities + equity)) < 0.01
        data.append(("  Assets", assets, "", ""))
        data.append(("  Liabilities + Equity", liabilities + equity, "", ""))
        data.append(("  Difference", assets - (liabilities + equity), "", ""))
        data.append(("  Status", "BALANCED" if balance_check else "IMBALANCED", "", "status"))
        data.append(("", "", "", ""))

    def _check_net_assets(
        self, data: List[Tuple[str, Union[str, float, int], str, str]], metrics: Dict[str, float]
    ) -> None:
        """Check net assets reconciliation."""
        data.append(("NET ASSETS RECONCILIATION", "", "", ""))
        net_assets = metrics.get("net_assets", 0)
        calc_net_assets = metrics.get("assets", 0) - metrics.get("restricted_assets", 0)
        net_assets_check = abs(net_assets - calc_net_assets) < 0.01

        data.append(("  Reported Net Assets", net_assets, "", ""))
        data.append(("  Calculated (Assets - Restricted)", calc_net_assets, "", ""))
        data.append(("  Difference", net_assets - calc_net_assets, "", ""))
        data.append(("  Status", "MATCHED" if net_assets_check else "MISMATCHED", "", "status"))
        data.append(("", "", "", ""))

    def _check_collateral(
        self, data: List[Tuple[str, Union[str, float, int], str, str]], metrics: Dict[str, float]
    ) -> None:
        """Check collateral reconciliation."""
        data.append(("COLLATERAL RECONCILIATION", "", "", ""))
        collateral = metrics.get("collateral", 0)
        restricted = metrics.get("restricted_assets", 0)
        collateral_check = collateral <= restricted

        data.append(("  Insurance Collateral", collateral, "", ""))
        data.append(("  Total Restricted Assets", restricted, "", ""))
        data.append(("  Status", "VALID" if collateral_check else "INVALID", "", "status"))
        data.append(("", "", "", ""))

    def _check_solvency(
        self, data: List[Tuple[str, Union[str, float, int], str, str]], metrics: Dict[str, float]
    ) -> None:
        """Check solvency status."""
        data.append(("SOLVENCY CHECK", "", "", ""))
        is_solvent = metrics.get("is_solvent", True)
        equity = metrics.get("equity", 0)
        equity_positive = equity > 0

        data.append(("  Equity", equity, "", ""))
        data.append(("  Solvency Flag", is_solvent, "", ""))
        data.append(("  Status", "SOLVENT" if equity_positive else "INSOLVENT", "", "status"))

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

            # Add previous year column
            df[f"Year {year-1}"] = ""

            # Calculate YoY for revenue
            if prev_metrics.get("revenue", 0) > 0:
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
        self, year: int, percentiles: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """Aggregate balance sheets across simulations.

        Args:
            year: Year index to aggregate
            percentiles: Percentiles to calculate (defaults to [5, 25, 50, 75, 95])

        Returns:
            DataFrame with aggregated balance sheet statistics
        """
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]
        # Validate year parameter to avoid unused argument warning
        _ = year
        # Implementation for aggregating balance sheets across simulations
        # This would process multiple trajectories and calculate statistics
        raise NotImplementedError("Monte Carlo aggregation not yet implemented")

    def aggregate_income_statements(
        self, year: int, percentiles: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """Aggregate income statements across simulations.

        Args:
            year: Year index to aggregate
            percentiles: Percentiles to calculate (defaults to [5, 25, 50, 75, 95])

        Returns:
            DataFrame with aggregated income statement statistics
        """
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]
        # Validate year parameter to avoid unused argument warning
        _ = year
        # Implementation for aggregating income statements
        raise NotImplementedError("Monte Carlo aggregation not yet implemented")

    def generate_convergence_analysis(self) -> pd.DataFrame:
        """Analyze convergence of financial metrics across simulations.

        Returns:
            DataFrame showing convergence statistics
        """
        # Analyze how financial metrics converge with more simulations
        raise NotImplementedError("Convergence analysis not yet implemented")
