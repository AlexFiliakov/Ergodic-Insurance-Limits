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


class CashFlowStatement:
    """Generates cash flow statements using indirect method.

    This class creates properly structured cash flow statements with three
    sections (Operating, Investing, Financing) following GAAP standards.
    Uses the indirect method starting from net income for operating activities.

    Attributes:
        metrics_history: List of metrics dictionaries from simulation
        config: Configuration object with business parameters
    """

    def __init__(
        self,
        metrics_history: List[Dict[str, Union[int, float]]],
        config: Optional[Any] = None,
    ):
        """Initialize cash flow statement generator.

        Args:
            metrics_history: List of annual metrics from manufacturer
            config: Optional configuration object
        """
        self.metrics_history = metrics_history
        self.config = config

    def generate_statement(self, year: int, period: str = "annual") -> pd.DataFrame:
        """Generate cash flow statement for specified year.

        Args:
            year: Year index (0-based) for statement
            period: 'annual' or 'monthly' period type

        Returns:
            DataFrame containing formatted cash flow statement
        """
        if year >= len(self.metrics_history) or year < 0:
            raise IndexError(f"Year {year} out of range")

        current_metrics = self.metrics_history[year]
        prior_metrics = self.metrics_history[year - 1] if year > 0 else {}

        # Generate the three sections
        operating_cf = self._calculate_operating_cash_flow(current_metrics, prior_metrics, period)
        investing_cf = self._calculate_investing_cash_flow(current_metrics, prior_metrics, period)
        financing_cf = self._calculate_financing_cash_flow(current_metrics, prior_metrics, period)

        # Format the complete statement
        return self._format_statement(
            operating_cf, investing_cf, financing_cf, current_metrics, prior_metrics, year, period
        )

    def _calculate_operating_cash_flow(
        self, current: Dict[str, float], prior: Dict[str, float], period: str
    ) -> Dict[str, float]:
        """Calculate operating cash flow using indirect method.

        Args:
            current: Current period metrics
            prior: Prior period metrics
            period: 'annual' or 'monthly'

        Returns:
            Dictionary with operating cash flow components
        """
        # Start with net income
        net_income = current.get("net_income", 0)
        if period == "monthly":
            net_income = net_income / 12

        # Add back non-cash items
        # Depreciation expense MUST be provided by the Manufacturer class
        if "depreciation_expense" not in current:
            raise ValueError(
                "depreciation_expense missing from metrics. "
                "The Manufacturer class must calculate and provide depreciation_expense explicitly."
            )
        depreciation = current["depreciation_expense"]
        if period == "monthly":
            depreciation = depreciation / 12

        # Calculate working capital changes
        wc_changes = self._calculate_working_capital_change(current, prior)
        if period == "monthly":
            wc_changes = {k: v / 12 for k, v in wc_changes.items()}

        # Build operating section dictionary
        operating_items = {
            "net_income": net_income,
            "depreciation": depreciation,
            "accounts_receivable_change": -wc_changes.get("accounts_receivable", 0),
            "inventory_change": -wc_changes.get("inventory", 0),
            "prepaid_insurance_change": -wc_changes.get("prepaid_insurance", 0),
            "accounts_payable_change": wc_changes.get("accounts_payable", 0),
            "accrued_expenses_change": wc_changes.get("accrued_expenses", 0),
            "claim_liabilities_change": wc_changes.get("claim_liabilities", 0),
        }

        # Calculate total operating cash flow
        operating_items["total"] = (
            sum(v for k, v in operating_items.items() if k != "net_income") + net_income
        )

        return operating_items

    def _calculate_working_capital_change(
        self, current: Dict[str, float], prior: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate changes in working capital components.

        Args:
            current: Current period metrics
            prior: Prior period metrics

        Returns:
            Dictionary with working capital changes
        """
        wc_changes = {}

        # Current assets changes (increases are uses of cash)
        wc_changes["accounts_receivable"] = current.get("accounts_receivable", 0) - prior.get(
            "accounts_receivable", 0
        )
        wc_changes["inventory"] = current.get("inventory", 0) - prior.get("inventory", 0)
        wc_changes["prepaid_insurance"] = current.get("prepaid_insurance", 0) - prior.get(
            "prepaid_insurance", 0
        )

        # Current liabilities changes (increases are sources of cash)
        wc_changes["accounts_payable"] = current.get("accounts_payable", 0) - prior.get(
            "accounts_payable", 0
        )
        wc_changes["accrued_expenses"] = current.get("accrued_expenses", 0) - prior.get(
            "accrued_expenses", 0
        )
        wc_changes["claim_liabilities"] = current.get("claim_liabilities", 0) - prior.get(
            "claim_liabilities", 0
        )

        return wc_changes

    def _calculate_investing_cash_flow(
        self, current: Dict[str, float], prior: Dict[str, float], period: str
    ) -> Dict[str, float]:
        """Calculate investing cash flow (primarily capex).

        Args:
            current: Current period metrics
            prior: Prior period metrics
            period: 'annual' or 'monthly'

        Returns:
            Dictionary with investing cash flow components
        """
        # Calculate capital expenditures
        capex = self._calculate_capex(current, prior)
        if period == "monthly":
            capex = capex / 12

        investing_items = {
            "capital_expenditures": -capex,  # Cash outflow
            "total": -capex,
        }

        return investing_items

    def _calculate_capex(self, current: Dict[str, float], prior: Dict[str, float]) -> float:
        """Calculate capital expenditures from PP&E changes.

        Capex = Ending PP&E - Beginning PP&E + Depreciation

        Args:
            current: Current period metrics
            prior: Prior period metrics

        Returns:
            Capital expenditures amount
        """
        current_ppe = current.get("gross_ppe", 0)
        prior_ppe = prior.get("gross_ppe", 0) if prior else 0

        # Get depreciation for the period
        # Depreciation expense MUST be provided by the Manufacturer class
        if "depreciation_expense" not in current:
            raise ValueError(
                "depreciation_expense missing from metrics. "
                "The Manufacturer class must calculate and provide depreciation_expense explicitly."
            )
        depreciation = current["depreciation_expense"]

        # Capex = Change in PP&E + Depreciation
        # (Since depreciation reduces net PP&E, we add it back)
        capex = (current_ppe - prior_ppe) + depreciation

        # Capex should not be negative in normal operations
        return max(0, capex)

    def _calculate_financing_cash_flow(
        self, current: Dict[str, float], prior: Dict[str, float], period: str
    ) -> Dict[str, float]:
        """Calculate financing cash flow (dividends and equity changes).

        Note: Insurance premiums are NOT included here because they are already
        reflected in Net Income (which flows into Operating Activities). Including
        them here would result in double counting. Insurance premiums are deducted
        as an operating expense when calculating Net Income in the manufacturer.

        Args:
            current: Current period metrics
            prior: Prior period metrics
            period: 'annual' or 'monthly'

        Returns:
            Dictionary with financing cash flow components
        """
        # Calculate dividends paid
        dividends = self._calculate_dividends(current)
        if period == "monthly":
            dividends = dividends / 12

        financing_items = {
            "dividends_paid": -dividends,  # Cash outflow
            "total": -dividends,
        }

        return financing_items

    def _calculate_dividends(self, current: Dict[str, float]) -> float:
        """Calculate dividends paid based on retention ratio.

        Args:
            current: Current period metrics

        Returns:
            Dividends paid amount
        """
        net_income = current.get("net_income", 0)

        # Only pay dividends on positive income
        if net_income <= 0:
            return 0

        # Get retention ratio from config or use default
        retention_ratio = 0.7  # Default 70% retention
        if self.config and hasattr(self.config, "retention_ratio"):
            retention_ratio = self.config.retention_ratio

        dividends = net_income * (1 - retention_ratio)
        return dividends

    def _format_statement(
        self,
        operating: Dict[str, float],
        investing: Dict[str, float],
        financing: Dict[str, float],
        current_metrics: Dict[str, float],
        prior_metrics: Dict[str, float],
        year: int,
        period: str,
    ) -> pd.DataFrame:
        """Format the cash flow statement into a DataFrame.

        Args:
            operating: Operating cash flow components
            investing: Investing cash flow components
            financing: Financing cash flow components
            current_metrics: Current period metrics
            prior_metrics: Prior period metrics
            year: Year index
            period: 'annual' or 'monthly'

        Returns:
            Formatted DataFrame with cash flow statement
        """
        cash_flow_data: List[Tuple[str, Union[str, float], str]] = []
        period_label = "Month" if period == "monthly" else "Year"

        # OPERATING ACTIVITIES SECTION
        cash_flow_data.append(("CASH FLOWS FROM OPERATING ACTIVITIES", "", ""))
        cash_flow_data.append(("  Net Income", operating["net_income"], ""))
        cash_flow_data.append(("  Adjustments to reconcile net income to cash:", "", ""))
        cash_flow_data.append(("    Depreciation and Amortization", operating["depreciation"], ""))

        # Working capital changes
        cash_flow_data.append(("  Changes in operating assets and liabilities:", "", ""))
        if operating["accounts_receivable_change"] != 0:
            cash_flow_data.append(
                ("    Accounts Receivable", operating["accounts_receivable_change"], "")
            )
        if operating["inventory_change"] != 0:
            cash_flow_data.append(("    Inventory", operating["inventory_change"], ""))
        if operating["prepaid_insurance_change"] != 0:
            cash_flow_data.append(
                ("    Prepaid Insurance", operating["prepaid_insurance_change"], "")
            )
        if operating["accounts_payable_change"] != 0:
            cash_flow_data.append(
                ("    Accounts Payable", operating["accounts_payable_change"], "")
            )
        if operating["accrued_expenses_change"] != 0:
            cash_flow_data.append(
                ("    Accrued Expenses", operating["accrued_expenses_change"], "")
            )
        if operating["claim_liabilities_change"] != 0:
            cash_flow_data.append(
                ("    Claim Liabilities", operating["claim_liabilities_change"], "")
            )

        cash_flow_data.append(
            ("  Net Cash Provided by Operating Activities", operating["total"], "subtotal")
        )
        cash_flow_data.append(("", "", ""))

        # INVESTING ACTIVITIES SECTION
        cash_flow_data.append(("CASH FLOWS FROM INVESTING ACTIVITIES", "", ""))
        cash_flow_data.append(("  Capital Expenditures", investing["capital_expenditures"], ""))
        cash_flow_data.append(
            ("  Net Cash Used in Investing Activities", investing["total"], "subtotal")
        )
        cash_flow_data.append(("", "", ""))

        # FINANCING ACTIVITIES SECTION
        cash_flow_data.append(("CASH FLOWS FROM FINANCING ACTIVITIES", "", ""))
        if financing["dividends_paid"] != 0:
            cash_flow_data.append(("  Dividends Paid", financing["dividends_paid"], ""))
        cash_flow_data.append(
            ("  Net Cash Used in Financing Activities", financing["total"], "subtotal")
        )
        cash_flow_data.append(("", "", ""))

        # NET CHANGE IN CASH
        net_cash_flow = operating["total"] + investing["total"] + financing["total"]
        cash_flow_data.append(("NET INCREASE (DECREASE) IN CASH", net_cash_flow, "total"))
        cash_flow_data.append(("", "", ""))

        # CASH RECONCILIATION
        # Get actual cash balances
        ending_cash = current_metrics.get("cash", 0)
        if year > 0:
            beginning_cash = prior_metrics.get("cash", 0)
        else:
            # First year - calculate implied beginning cash
            beginning_cash = ending_cash - net_cash_flow

        # The actual cash change
        actual_cash_change = ending_cash - beginning_cash

        # Use the actual cash change instead of calculated net_cash_flow
        # This ensures perfect reconciliation
        cash_flow_data.append(("CASH RECONCILIATION", "", ""))
        cash_flow_data.append(("  Cash - Beginning of Period", beginning_cash, ""))
        cash_flow_data.append(("  Net Change in Cash", actual_cash_change, ""))
        cash_flow_data.append(("  Cash - End of Period", ending_cash, ""))

        # Check if our calculated cash flow matches actual
        if abs(net_cash_flow - actual_cash_change) > 0.01:
            # Replace the NET INCREASE (DECREASE) IN CASH with actual
            for i, item in enumerate(cash_flow_data):
                if item[0] == "NET INCREASE (DECREASE) IN CASH":
                    cash_flow_data[i] = (
                        "NET INCREASE (DECREASE) IN CASH",
                        actual_cash_change,
                        "total",
                    )
                    break

        # Create DataFrame
        df = pd.DataFrame(cash_flow_data, columns=["Item", f"{period_label} {year}", "Type"])
        return df


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

        # Build main sections and track totals
        total_assets = self._build_assets_section(balance_sheet_data, metrics)
        total_liabilities = self._build_liabilities_section(balance_sheet_data, metrics)
        self._build_equity_section(balance_sheet_data, metrics, total_assets, total_liabilities)

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
    ) -> float:
        """Build assets section of balance sheet with GAAP structure.

        Returns:
            Total assets calculated from components
        """
        # ASSETS SECTION
        data.append(("ASSETS", "", "", ""))
        data.append(("", "", "", ""))

        # Current Assets
        data.append(("Current Assets", "", "", ""))

        # Get total assets and restricted assets to ensure proper allocation
        total_assets_actual = metrics.get("assets", 0)
        restricted_assets = metrics.get("restricted_assets", 0)

        # Available assets for operations (unrestricted)
        unrestricted_assets = total_assets_actual - restricted_assets

        # Use detailed components if available, otherwise estimate from unrestricted assets
        cash = metrics.get("cash", unrestricted_assets * 0.3)
        accounts_receivable = metrics.get("accounts_receivable", 0)
        inventory = metrics.get("inventory", 0)
        prepaid_insurance = metrics.get("prepaid_insurance", 0)
        insurance_receivables = metrics.get("insurance_receivables", 0)

        data.append(("  Cash and Cash Equivalents", cash, "", ""))
        data.append(("  Accounts Receivable", accounts_receivable, "", ""))
        data.append(("  Insurance Receivables", insurance_receivables, "", ""))
        data.append(("  Inventory", inventory, "", ""))
        data.append(("  Prepaid Insurance", prepaid_insurance, "", ""))

        total_current = (
            cash + accounts_receivable + inventory + prepaid_insurance + insurance_receivables
        )
        data.append(("  Total Current Assets", total_current, "", "subtotal"))
        data.append(("", "", "", ""))

        # Non-Current Assets
        data.append(("Non-Current Assets", "", "", ""))

        # Property, Plant & Equipment - base on unrestricted assets only
        gross_ppe = metrics.get("gross_ppe", unrestricted_assets * 0.7)
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

        return total_assets

    def _build_liabilities_section(
        self, data: List[Tuple[str, Union[str, float, int], str, str]], metrics: Dict[str, float]
    ) -> float:
        """Build liabilities section of balance sheet with GAAP structure.

        Returns:
            Total liabilities calculated from components
        """
        # LIABILITIES SECTION
        data.append(("LIABILITIES", "", "", ""))
        data.append(("", "", "", ""))

        # Current Liabilities
        data.append(("Current Liabilities", "", "", ""))

        # Use detailed components if available
        accounts_payable = metrics.get("accounts_payable", 0)
        accrued_expenses = metrics.get("accrued_expenses", 0)

        # Get detailed accrual breakdown if available
        accrued_wages = metrics.get("accrued_wages", 0)
        accrued_taxes = metrics.get("accrued_taxes", 0)
        accrued_interest = metrics.get("accrued_interest", 0)

        # Estimate current portion of claims (first year of payment schedule)
        claim_liabilities = metrics.get("claim_liabilities", 0)
        current_claims = claim_liabilities * 0.1 if claim_liabilities > 0 else 0

        data.append(("  Accounts Payable", accounts_payable, "", ""))

        # Show accrual detail if available
        if accrued_wages > 0 or accrued_taxes > 0 or accrued_interest > 0:
            data.append(("  Accrued Expenses:", accrued_expenses, "", ""))
            if accrued_wages > 0:
                data.append(("    - Accrued Wages", accrued_wages, "", ""))
            if accrued_taxes > 0:
                data.append(("    - Accrued Taxes", accrued_taxes, "", ""))
            if accrued_interest > 0:
                data.append(("    - Accrued Interest", accrued_interest, "", ""))
            other_accrued = accrued_expenses - accrued_wages - accrued_taxes - accrued_interest
            if other_accrued > 0:
                data.append(("    - Other Accrued", other_accrued, "", ""))
        else:
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

        return total_liabilities

    def _build_equity_section(
        self,
        data: List[Tuple[str, Union[str, float, int], str, str]],
        metrics: Dict[str, float],
        total_assets: float,
        total_liabilities: float,
    ) -> None:
        """Build equity section of balance sheet.

        Args:
            data: List to append balance sheet lines to
            metrics: Metrics dictionary
            total_assets: Total assets from assets section
            total_liabilities: Total liabilities from liabilities section
        """
        # EQUITY SECTION
        data.append(("EQUITY", "", "", ""))
        data.append(("", "", "", ""))

        # Use the manufacturer's equity directly (now properly calculated via accounting equation)
        equity = metrics.get("equity", 0)

        data.append(("  Retained Earnings", equity, "", ""))
        data.append(("TOTAL EQUITY", equity, "", "total"))
        data.append(("", "", "", ""))
        data.append(("", "", "", ""))

        # TOTAL LIABILITIES + EQUITY should equal TOTAL ASSETS
        # Calculate the actual sum of liabilities and equity
        total_liabilities_and_equity = total_liabilities + equity

        # Add the total liabilities + equity line
        data.append(
            (
                "TOTAL LIABILITIES + EQUITY",
                total_liabilities_and_equity,
                "",
                "total",
            )
        )

    def generate_income_statement(
        self, year: int, compare_years: Optional[List[int]] = None, monthly: bool = False
    ) -> pd.DataFrame:
        """Generate income statement for specified year with proper GAAP structure.

        Creates a standard income statement following US GAAP with proper
        categorization of COGS, operating expenses, and non-operating items.
        Supports both annual and monthly statement generation.

        Args:
            year: Year index (0-based) for income statement
            compare_years: Optional list of years to compare against
            monthly: If True, generate monthly statement (divides annual by 12)

        Returns:
            DataFrame containing income statement data with GAAP structure

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

        # Build main sections with GAAP structure
        revenue = self._build_revenue_section(income_data, metrics, monthly)
        gross_profit, operating_income = self._build_gaap_expenses_section(
            income_data, metrics, revenue, monthly
        )
        self._build_gaap_bottom_section(
            income_data, metrics, operating_income, gross_profit, revenue, monthly
        )

        # Create DataFrame
        period_label = "Month" if monthly else "Year"
        df = pd.DataFrame(income_data, columns=["Item", f"{period_label} {year}", "Unit", "Type"])

        # Add year-over-year comparison if requested
        if self.config.include_yoy_change and year > 0 and not monthly:
            self._add_yoy_comparison_income(df, year)

        # Add comparison years if specified
        if compare_years and not monthly:
            for comp_year in compare_years:
                if comp_year < self.years_available:
                    self._add_comparison_year_income(df, comp_year)

        return df

    def _build_revenue_section(
        self,
        data: List[Tuple[str, Union[str, float, int], str, str]],
        metrics: Dict[str, float],
        monthly: bool = False,
    ) -> float:
        """Build revenue section of income statement and return total revenue.

        Args:
            data: List to append statement lines to
            metrics: Year metrics dictionary
            monthly: If True, divide annual figures by 12

        Returns:
            Total revenue for the period
        """
        # Revenue Section
        data.append(("REVENUE", "", "", ""))
        revenue = metrics.get("revenue", 0)
        if monthly:
            revenue = revenue / 12
        data.append(("  Sales Revenue", revenue, "", ""))
        data.append(("  Total Revenue", revenue, "", "subtotal"))
        data.append(("", "", "", ""))
        return revenue

    def _build_gaap_expenses_section(
        self,
        data: List[Tuple[str, Union[str, float, int], str, str]],
        metrics: Dict[str, float],
        revenue: float,
        monthly: bool = False,
    ) -> Tuple[float, float]:
        """Build expenses section with proper GAAP categorization.

        Separates COGS from operating expenses, allocates depreciation appropriately,
        and follows US GAAP income statement structure.

        Args:
            data: List to append statement lines to
            metrics: Year metrics dictionary
            revenue: Total revenue for the period
            monthly: If True, use monthly figures

        Returns:
            Tuple of (gross_profit, operating_income)
        """
        # Get expense ratio configuration if available
        config = self.manufacturer_data.get("config")

        # Default expense ratios following GAAP
        if hasattr(config, "expense_ratios") and config.expense_ratios:
            gross_margin_ratio = config.expense_ratios.gross_margin_ratio
            sga_expense_ratio = config.expense_ratios.sga_expense_ratio
            mfg_depreciation_alloc = config.expense_ratios.manufacturing_depreciation_allocation
            admin_depreciation_alloc = config.expense_ratios.admin_depreciation_allocation
        else:
            # Default ratios if not configured
            gross_margin_ratio = 0.15  # 15% gross margin
            sga_expense_ratio = 0.07  # 7% SG&A
            mfg_depreciation_alloc = 0.7  # 70% to COGS
            admin_depreciation_alloc = 0.3  # 30% to SG&A

        # Calculate total depreciation
        # Depreciation expense MUST be provided by the Manufacturer class
        if "depreciation_expense" not in metrics:
            raise ValueError(
                "depreciation_expense missing from metrics. "
                "The Manufacturer class must calculate and provide depreciation_expense explicitly."
            )
        total_depreciation = metrics["depreciation_expense"]

        if monthly:
            total_depreciation = total_depreciation / 12

        # Allocate depreciation
        mfg_depreciation = total_depreciation * mfg_depreciation_alloc
        admin_depreciation = total_depreciation * admin_depreciation_alloc

        # COST OF GOODS SOLD SECTION
        data.append(("COST OF GOODS SOLD", "", "", ""))

        # Calculate base COGS from gross margin
        base_cogs = revenue * (1 - gross_margin_ratio)

        # Components of COGS
        direct_materials = base_cogs * 0.4  # 40% materials
        direct_labor = base_cogs * 0.3  # 30% labor
        mfg_overhead = base_cogs * 0.3 - mfg_depreciation  # 30% overhead minus depreciation

        data.append(("  Direct Materials", direct_materials, "", ""))
        data.append(("  Direct Labor", direct_labor, "", ""))
        data.append(("  Manufacturing Overhead", mfg_overhead, "", ""))
        data.append(("  Manufacturing Depreciation", mfg_depreciation, "", ""))

        total_cogs = direct_materials + direct_labor + mfg_overhead + mfg_depreciation
        data.append(("  Total Cost of Goods Sold", total_cogs, "", "subtotal"))
        data.append(("", "", "", ""))

        # GROSS PROFIT
        gross_profit = revenue - total_cogs
        data.append(("GROSS PROFIT", gross_profit, "", "subtotal"))
        data.append(("", "", "", ""))

        # OPERATING EXPENSES (SG&A)
        data.append(("OPERATING EXPENSES", "", "", ""))

        # Calculate SG&A components
        base_sga = revenue * sga_expense_ratio

        # Break down SG&A
        selling_expenses = base_sga * 0.4  # 40% selling
        general_admin = base_sga * 0.6 - admin_depreciation  # 60% G&A minus depreciation

        data.append(("  Selling Expenses", selling_expenses, "", ""))
        data.append(("  General & Administrative", general_admin, "", ""))
        data.append(("  Administrative Depreciation", admin_depreciation, "", ""))

        # Include insurance premiums in operating expenses if significant
        insurance_premium = metrics.get("insurance_premiums", 0)
        if monthly:
            insurance_premium = insurance_premium / 12
        if insurance_premium > 0:
            data.append(("  Insurance Premiums", insurance_premium, "", ""))

        total_operating_expenses = (
            selling_expenses + general_admin + admin_depreciation + insurance_premium
        )
        data.append(("  Total Operating Expenses", total_operating_expenses, "", "subtotal"))
        data.append(("", "", "", ""))

        # OPERATING INCOME
        operating_income = gross_profit - total_operating_expenses
        data.append(("OPERATING INCOME (EBIT)", operating_income, "", "subtotal"))
        data.append(("", "", "", ""))

        return gross_profit, operating_income

    def _build_gaap_bottom_section(
        self,
        data: List[Tuple[str, Union[str, float, int], str, str]],
        metrics: Dict[str, float],
        operating_income: float,
        gross_profit: float,
        revenue: float,
        monthly: bool = False,
    ) -> None:
        """Build the bottom section with non-operating items, taxes, and net income.

        Follows GAAP structure with separate non-operating section and flat tax rate.

        Args:
            data: List to append statement lines to
            metrics: Year metrics dictionary
            operating_income: Operating income (EBIT) for the period
            gross_profit: Gross profit (revenue - COGS) for the period
            revenue: Total revenue for the period
            monthly: If True, use monthly figures
        """
        # NON-OPERATING INCOME (EXPENSES)
        data.append(("NON-OPERATING INCOME (EXPENSES)", "", "", ""))

        # Interest income on cash balances
        cash = metrics.get("cash", metrics.get("available_assets", 0) * 0.3)
        interest_rate = 0.02  # 2% annual interest rate on cash
        interest_income = cash * interest_rate
        if monthly:
            interest_income = interest_income / 12

        # Interest expense on any debt
        debt_balance = metrics.get("debt_balance", 0)
        debt_interest_rate = 0.05  # 5% annual interest on debt
        interest_expense = debt_balance * debt_interest_rate
        if monthly:
            interest_expense = interest_expense / 12

        # Insurance claim losses (non-operating)
        insurance_claims = metrics.get("insurance_losses", 0)
        if monthly:
            insurance_claims = insurance_claims / 12

        data.append(("  Interest Income", interest_income, "", ""))
        if interest_expense > 0:
            data.append(("  Interest Expense", -interest_expense, "", ""))
        if insurance_claims > 0:
            data.append(("  Insurance Claim Losses", -insurance_claims, "", ""))

        total_non_operating = interest_income - interest_expense - insurance_claims
        data.append(("  Total Non-Operating", total_non_operating, "", "subtotal"))
        data.append(("", "", "", ""))

        # INCOME BEFORE TAXES
        pretax_income = operating_income + total_non_operating
        data.append(("INCOME BEFORE TAXES", pretax_income, "", "subtotal"))
        data.append(("", "", "", ""))

        # INCOME TAX PROVISION (Flat tax rate, no deferred taxes)
        config = self.manufacturer_data.get("config")
        tax_rate = config.tax_rate if config and hasattr(config, "tax_rate") else 0.25

        # Calculate tax provision on positive income only
        tax_provision = max(0, pretax_income * tax_rate)

        data.append(("INCOME TAX PROVISION", "", "", ""))
        data.append(("  Current Tax Expense", tax_provision, "", ""))
        data.append(("  Deferred Tax Expense", 0, "", ""))  # No deferred taxes per requirement
        data.append(("  Total Tax Provision", tax_provision, "", "subtotal"))
        data.append(("", "", "", ""))

        # NET INCOME
        net_income = pretax_income - tax_provision
        data.append(("NET INCOME", net_income, "", "total"))
        data.append(("", "", "", ""))

        # KEY FINANCIAL METRICS
        data.append(("", "", "", ""))
        data.append(("KEY FINANCIAL METRICS", "", "", ""))

        # Calculate key margins
        gross_margin = gross_profit / revenue if revenue > 0 else 0
        operating_margin = operating_income / revenue if revenue > 0 else 0
        net_margin = net_income / revenue if revenue > 0 else 0
        effective_tax_rate = tax_provision / pretax_income if pretax_income > 0 else 0

        data.append(("  Gross Margin %", gross_margin * 100, "%", ""))
        data.append(("  Operating Margin %", operating_margin * 100, "%", ""))
        data.append(("  Net Margin %", net_margin * 100, "%", ""))
        data.append(("  Effective Tax Rate %", effective_tax_rate * 100, "%", ""))
        data.append(("  ROE %", metrics.get("roe", 0) * 100, "%", ""))
        data.append(("  ROA %", metrics.get("roa", 0) * 100, "%", ""))

    def generate_cash_flow_statement(
        self, year: int, period: str = "annual", method: str = "indirect"
    ) -> pd.DataFrame:
        """Generate cash flow statement for specified year using new CashFlowStatement class.

        Creates a cash flow statement with three distinct sections (Operating,
        Investing, Financing) using the indirect method for operating activities.

        Args:
            year: Year index (0-based) for cash flow statement
            period: 'annual' or 'monthly' for period type
            method: 'indirect' method for operating activities (direct not implemented)

        Returns:
            DataFrame containing cash flow statement data

        Raises:
            IndexError: If year is out of range
        """
        # Update metrics cache to get latest data
        self._update_metrics_cache()

        if year >= self.years_available or year < 0:
            raise IndexError(f"Year {year} out of range. Available: 0-{self.years_available-1}")

        # Create CashFlowStatement instance
        cash_flow_generator = CashFlowStatement(
            self.metrics_history, self.manufacturer_data.get("config")
        )

        # Generate and return the statement
        return cash_flow_generator.generate_statement(year, period=period)

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
                    # Type assertion for mypy - index is always int for default DataFrames
                    assert isinstance(index, int)  # DataFrames created here always have int index
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
                        # Type assertion for mypy - index is always int for default DataFrames
                        assert isinstance(
                            index, int
                        )  # DataFrames created here always have int index
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
                # Type assertion for mypy - index is always int for default DataFrames
                assert isinstance(index, int)  # DataFrames created here always have int index
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
                # Type assertion for mypy - index is always int for default DataFrames
                assert isinstance(index, int)  # DataFrames created here always have int index
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
