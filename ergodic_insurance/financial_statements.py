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
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .decimal_utils import ZERO, is_zero, to_decimal

if TYPE_CHECKING:
    from .ledger import Ledger
    from .manufacturer import WidgetManufacturer


@dataclass
class FinancialStatementConfig:
    """Configuration for financial statement generation.

    Attributes:
        currency_symbol: Symbol to use for currency formatting
        decimal_places: Number of decimal places for numeric values
        include_yoy_change: Whether to include year-over-year changes
        include_percentages: Whether to include percentage breakdowns
        fiscal_year_end: Month of fiscal year end (1-12). If None, inherits from
            the central Config.simulation.fiscal_year_end setting. Defaults to 12
            (December) if neither is set, for calendar year alignment.
        consolidate_monthly: Whether to consolidate monthly data into annual
        current_claims_ratio: Fraction of claim liabilities classified as current
            (due within one year). Defaults to 0.1 (10%). Should be derived from
            actual claim payment schedules when available.
    """

    currency_symbol: str = "$"
    decimal_places: int = 0
    include_yoy_change: bool = True
    include_percentages: bool = True
    fiscal_year_end: Optional[int] = None
    consolidate_monthly: bool = True
    current_claims_ratio: float = 0.1


class CashFlowStatement:
    """Generates cash flow statements using indirect or direct method.

    This class creates properly structured cash flow statements with three
    sections (Operating, Investing, Financing) following GAAP standards.
    Supports both the indirect method (starting from net income) and the
    direct method (summing ledger entries) for operating activities.

    When a ledger is provided, the direct method is available, which provides
    perfect reconciliation and audit trail for all cash flows.

    Attributes:
        metrics_history: List of metrics dictionaries from simulation
        config: Configuration object with business parameters
        ledger: Optional Ledger for direct method cash flow generation
    """

    def __init__(
        self,
        metrics_history: List[Dict[str, Union[int, float]]],
        config: Optional[Any] = None,
        ledger: Optional["Ledger"] = None,
    ):
        """Initialize cash flow statement generator.

        Args:
            metrics_history: List of annual metrics from manufacturer
            config: Optional configuration object
            ledger: Optional Ledger instance for direct method cash flow.
                When provided, enables direct method generation which sums
                actual cash transactions rather than inferring from balance
                sheet deltas.
        """
        self.metrics_history = metrics_history
        self.config = config
        self.ledger = ledger

    def generate_statement(
        self, year: int, period: str = "annual", method: str = "indirect"
    ) -> pd.DataFrame:
        """Generate cash flow statement for specified year.

        Args:
            year: Year index (0-based) for statement
            period: 'annual' or 'monthly' period type
            method: 'indirect' (default) or 'direct'. Direct method requires
                a ledger to be provided during initialization.

        Returns:
            DataFrame containing formatted cash flow statement

        Raises:
            IndexError: If year is out of range
            ValueError: If direct method requested but no ledger available
        """
        if year >= len(self.metrics_history) or year < 0:
            raise IndexError(f"Year {year} out of range")

        if method == "direct" and self.ledger is None:
            raise ValueError(
                "Direct method requires a ledger. Provide a Ledger instance "
                "when initializing CashFlowStatement."
            )

        current_metrics = self.metrics_history[year]
        prior_metrics = self.metrics_history[year - 1] if year > 0 else {}

        # Generate the three sections based on method
        if method == "direct" and self.ledger is not None:
            operating_cf = self._calculate_operating_cash_flow_direct(year, period)
            investing_cf = self._calculate_investing_cash_flow_direct(year, period)
            financing_cf = self._calculate_financing_cash_flow_direct(year, period)
        else:
            operating_cf = self._calculate_operating_cash_flow(
                current_metrics, prior_metrics, period
            )
            investing_cf = self._calculate_investing_cash_flow(
                current_metrics, prior_metrics, period
            )
            financing_cf = self._calculate_financing_cash_flow(
                current_metrics, prior_metrics, period
            )

        # Format the complete statement
        return self._format_statement(
            operating_cf,
            investing_cf,
            financing_cf,
            current_metrics,
            prior_metrics,
            year,
            period,
            method,
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
        """Get dividends paid from metrics or calculate as fallback.

        Issue #239: The WidgetManufacturer now tracks actual dividends_paid
        considering cash constraints. This method should read that value
        instead of calculating it, which may report phantom payments.

        Issue #243: Removed hardcoded retention_ratio=0.7 fallback to prevent
        inconsistent configuration. Now requires either dividends_paid in metrics
        or a config with retention_ratio attribute.

        Args:
            current: Current period metrics

        Returns:
            Dividends paid amount (actual from metrics if available)

        Raises:
            ValueError: If neither dividends_paid in metrics nor config with
                retention_ratio is available.
        """
        # Prefer actual dividends_paid from metrics (tracks cash constraints)
        if "dividends_paid" in current:
            return current["dividends_paid"]

        # Fallback: Calculate from net income (backward compatibility)
        # This path is used when metrics don't have dividends_paid
        net_income = current.get("net_income", 0)

        # Only pay dividends on positive income
        if net_income <= 0:
            return 0

        # Get retention ratio from config - no hardcoded default (Issue #243)
        if self.config and hasattr(self.config, "retention_ratio"):
            retention_ratio: float = float(self.config.retention_ratio)
        else:
            raise ValueError(
                "Cannot calculate dividends: config must have 'retention_ratio' attribute "
                "when 'dividends_paid' is not in metrics. Pass a ManufacturerConfig or "
                "ensure metrics include 'dividends_paid' from the simulation."
            )

        dividends: float = net_income * (1 - retention_ratio)
        return dividends

    def _calculate_operating_cash_flow_direct(
        self, year: int, period: str
    ) -> Dict[str, Union[float, Decimal]]:
        """Calculate operating cash flow using direct method from ledger.

        The direct method sums actual cash transactions from the ledger,
        providing perfect audit trail and reconciliation.

        Args:
            year: Period to extract cash flows for
            period: 'annual' or 'monthly'

        Returns:
            Dictionary with operating cash flow components
        """
        if self.ledger is None:
            raise ValueError("Direct method requires a ledger")

        flows = self.ledger.get_cash_flows(period=year)

        operating_items: Dict[str, Union[float, Decimal]] = {
            "cash_from_customers": flows.get("cash_from_customers", ZERO),
            "cash_from_insurance": flows.get("cash_from_insurance", ZERO),
            "cash_to_suppliers": -flows.get("cash_to_suppliers", ZERO),
            "cash_for_insurance": -flows.get("cash_for_insurance", ZERO),
            "cash_for_taxes": -flows.get("cash_for_taxes", ZERO),
            "cash_for_wages": -flows.get("cash_for_wages", ZERO),
        }

        # Calculate total
        operating_items["total"] = sum((to_decimal(v) for v in operating_items.values()), ZERO)

        if period == "monthly":
            operating_items = {k: to_decimal(v) / 12 for k, v in operating_items.items()}

        return operating_items

    def _calculate_investing_cash_flow_direct(
        self, year: int, period: str
    ) -> Dict[str, Union[float, Decimal]]:
        """Calculate investing cash flow using direct method from ledger.

        Args:
            year: Period to extract cash flows for
            period: 'annual' or 'monthly'

        Returns:
            Dictionary with investing cash flow components
        """
        if self.ledger is None:
            raise ValueError("Direct method requires a ledger")

        flows = self.ledger.get_cash_flows(period=year)

        investing_items: Dict[str, Union[float, Decimal]] = {
            "capital_expenditures": -flows.get("capital_expenditures", ZERO),
            "asset_sales": flows.get("asset_sales", ZERO),
        }

        investing_items["total"] = sum((to_decimal(v) for v in investing_items.values()), ZERO)

        if period == "monthly":
            investing_items = {k: to_decimal(v) / 12 for k, v in investing_items.items()}

        return investing_items

    def _calculate_financing_cash_flow_direct(
        self, year: int, period: str
    ) -> Dict[str, Union[float, Decimal]]:
        """Calculate financing cash flow using direct method from ledger.

        Args:
            year: Period to extract cash flows for
            period: 'annual' or 'monthly'

        Returns:
            Dictionary with financing cash flow components
        """
        if self.ledger is None:
            raise ValueError("Direct method requires a ledger")

        flows = self.ledger.get_cash_flows(period=year)

        financing_items: Dict[str, Union[float, Decimal]] = {
            "dividends_paid": -flows.get("dividends_paid", ZERO),
            "equity_issuance": flows.get("equity_issuance", ZERO),
        }

        financing_items["total"] = sum((to_decimal(v) for v in financing_items.values()), ZERO)

        if period == "monthly":
            financing_items = {k: to_decimal(v) / 12 for k, v in financing_items.items()}

        return financing_items

    def _format_statement(
        self,
        operating: Dict[str, float],
        investing: Dict[str, float],
        financing: Dict[str, float],
        current_metrics: Dict[str, float],
        prior_metrics: Dict[str, float],
        year: int,
        period: str,
        method: str = "indirect",
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
            method: 'indirect' or 'direct' - affects formatting

        Returns:
            Formatted DataFrame with cash flow statement
        """
        cash_flow_data: List[Tuple[str, Union[str, float], str]] = []
        period_label = "Month" if period == "monthly" else "Year"

        # OPERATING ACTIVITIES SECTION
        if method == "direct":
            cash_flow_data.append(("CASH FLOWS FROM OPERATING ACTIVITIES (Direct Method)", "", ""))
            # Direct method shows actual cash receipts and payments
            if operating.get("cash_from_customers", 0) != 0:
                cash_flow_data.append(
                    ("  Cash Received from Customers", operating["cash_from_customers"], "")
                )
            if operating.get("cash_from_insurance", 0) != 0:
                cash_flow_data.append(
                    ("  Cash Received from Insurance", operating["cash_from_insurance"], "")
                )
            if operating.get("cash_to_suppliers", 0) != 0:
                cash_flow_data.append(
                    ("  Cash Paid to Suppliers", operating["cash_to_suppliers"], "")
                )
            if operating.get("cash_for_insurance", 0) != 0:
                cash_flow_data.append(
                    ("  Cash Paid for Insurance", operating["cash_for_insurance"], "")
                )
            if operating.get("cash_for_taxes", 0) != 0:
                cash_flow_data.append(("  Cash Paid for Taxes", operating["cash_for_taxes"], ""))
            if operating.get("cash_for_wages", 0) != 0:
                cash_flow_data.append(("  Cash Paid for Wages", operating["cash_for_wages"], ""))
        else:
            cash_flow_data.append(("CASH FLOWS FROM OPERATING ACTIVITIES", "", ""))
            cash_flow_data.append(("  Net Income", operating["net_income"], ""))
            cash_flow_data.append(("  Adjustments to reconcile net income to cash:", "", ""))
            cash_flow_data.append(
                ("    Depreciation and Amortization", operating["depreciation"], "")
            )

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
        if method == "direct" and investing.get("asset_sales", 0) != 0:
            cash_flow_data.append(("  Proceeds from Asset Sales", investing["asset_sales"], ""))
        cash_flow_data.append(
            ("  Net Cash Used in Investing Activities", investing["total"], "subtotal")
        )
        cash_flow_data.append(("", "", ""))

        # FINANCING ACTIVITIES SECTION
        cash_flow_data.append(("CASH FLOWS FROM FINANCING ACTIVITIES", "", ""))
        if financing["dividends_paid"] != 0:
            cash_flow_data.append(("  Dividends Paid", financing["dividends_paid"], ""))
        if method == "direct" and financing.get("equity_issuance", 0) != 0:
            cash_flow_data.append(
                ("  Proceeds from Equity Issuance", financing["equity_issuance"], "")
            )
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

        # Use the actual cash change instead of calculated net_cash_flow
        # This ensures perfect reconciliation
        cash_flow_data.append(("CASH RECONCILIATION", "", ""))
        cash_flow_data.append(("  Cash - Beginning of Period", beginning_cash, ""))
        cash_flow_data.append(("  Net Change in Cash", net_cash_flow, ""))
        cash_flow_data.append(("  Cash - End of Period", ending_cash, ""))

        # Create DataFrame
        df = pd.DataFrame(cash_flow_data, columns=["Item", f"{period_label} {year}", "Type"])
        return df


class FinancialStatementGenerator:
    """Generates financial statements from simulation data.

    This class compiles standard financial statements (Balance Sheet,
    Income Statement, Cash Flow) from manufacturer metrics history.
    It handles both annual and monthly data, performs reconciliation
    checks, and calculates derived financial metrics.

    When a ledger is provided (either directly or via the manufacturer),
    direct method cash flow statements can be generated, providing perfect
    reconciliation and audit trail for all cash transactions.

    Attributes:
        manufacturer_data: Raw simulation data from manufacturer
        config: Configuration for statement generation
        metrics_history: List of metrics dictionaries from simulation
        years_available: Number of years of data available
        ledger: Optional Ledger for direct method cash flow generation
    """

    def __init__(
        self,
        manufacturer: Optional["WidgetManufacturer"] = None,
        manufacturer_data: Optional[Dict[str, Any]] = None,
        config: Optional[FinancialStatementConfig] = None,
        ledger: Optional["Ledger"] = None,
    ):
        """Initialize financial statement generator.

        Args:
            manufacturer: WidgetManufacturer instance with simulation data.
                If the manufacturer has a ledger attribute, it will be used
                for direct method cash flow generation.
            manufacturer_data: Alternative dictionary of manufacturer data
            config: Configuration for statement generation
            ledger: Optional Ledger instance for direct method cash flow.
                Overrides any ledger from manufacturer.

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
            # Use ledger from manufacturer if available and not overridden
            if ledger is None and hasattr(manufacturer, "ledger"):
                ledger = manufacturer.ledger
        elif manufacturer_data is not None:
            self.manufacturer_data = manufacturer_data
        else:
            raise ValueError("Either manufacturer or manufacturer_data must be provided")

        self.config = config or FinancialStatementConfig()
        self.ledger = ledger

        # Resolve fiscal_year_end from central config if not explicitly set
        if self.config.fiscal_year_end is None:
            central_config = self.manufacturer_data.get("config")
            if central_config is not None and hasattr(central_config, "simulation"):
                simulation_config = central_config.simulation
                if hasattr(simulation_config, "fiscal_year_end"):
                    self.config.fiscal_year_end = simulation_config.fiscal_year_end
            # Fall back to default of 12 (December) if still not set
            if self.config.fiscal_year_end is None:
                self.config.fiscal_year_end = 12

        self._update_metrics_cache()

    def _update_metrics_cache(self):
        """Update the cached metrics from manufacturer data."""
        if self.manufacturer is not None:
            # Get fresh metrics from manufacturer
            self.manufacturer_data["metrics_history"] = self.manufacturer.metrics_history

        metrics = self.manufacturer_data.get("metrics_history", [])
        self.metrics_history: List[Dict[str, float]] = metrics if isinstance(metrics, list) else []
        self.years_available = len(self.metrics_history)

    def _get_metrics_from_ledger(self, year: int) -> Dict[str, Union[float, Decimal]]:
        """Derive metrics dictionary from ledger balances.

        This method constructs a metrics-like dictionary from ledger account
        balances, providing the single source of truth for financial statements.
        When a ledger is available, this method should be used instead of
        metrics_history to ensure consistency.

        Args:
            year: Year index (0-based) for which to get metrics

        Returns:
            Dictionary of metrics derived from ledger balances

        Note:
            The ledger uses year as the 'as_of_date' parameter, so get_balance(account, year)
            returns the balance as of the end of that year (inclusive of all transactions
            with date <= year).
        """
        if self.ledger is None:
            raise ValueError("Ledger is required for ledger-based metrics")

        metrics: Dict[str, Union[float, Decimal]] = {}

        # For some accounts, the ledger may be incomplete. Fall back to manufacturer state
        # or metrics_history when ledger data is inconsistent.
        # Get metrics_history for fallback
        mfr_metrics = None
        if year < len(self.metrics_history):
            mfr_metrics = self.metrics_history[year]

        # Asset accounts (debit-normal, positive balance expected)
        # For most asset accounts, the ledger should be accurate. However, cash may
        # differ due to incomplete transaction recording. Use manufacturer state
        # when available for consistency with equity calculations.
        if mfr_metrics and "cash" in mfr_metrics:
            metrics["cash"] = mfr_metrics["cash"]
        else:
            metrics["cash"] = self.ledger.get_balance("cash", year)
        metrics["accounts_receivable"] = self.ledger.get_balance("accounts_receivable", year)
        metrics["inventory"] = self.ledger.get_balance("inventory", year)
        metrics["prepaid_insurance"] = self.ledger.get_balance("prepaid_insurance", year)
        metrics["insurance_receivables"] = self.ledger.get_balance("insurance_receivables", year)
        metrics["gross_ppe"] = self.ledger.get_balance("gross_ppe", year)
        # Accumulated depreciation is a contra-asset with credit-normal balance
        # The ledger treats it as debit-normal (ASSET), so credits make it negative
        # Convert to positive value for calculations
        raw_accumulated_dep = self.ledger.get_balance("accumulated_depreciation", year)
        metrics["accumulated_depreciation"] = abs(raw_accumulated_dep)

        # Restricted assets and collateral may not be fully tracked in ledger
        # Fall back to manufacturer state if ledger shows incorrect (negative or zero) values
        ledger_restricted = self.ledger.get_balance("restricted_cash", year)
        if ledger_restricted <= 0 and mfr_metrics and mfr_metrics.get("restricted_assets", 0) > 0:
            metrics["restricted_assets"] = mfr_metrics["restricted_assets"]
        else:
            metrics["restricted_assets"] = max(0, ledger_restricted)

        ledger_collateral = self.ledger.get_balance("collateral", year)
        if ledger_collateral <= 0 and mfr_metrics and mfr_metrics.get("collateral", 0) > 0:
            metrics["collateral"] = mfr_metrics["collateral"]
        else:
            metrics["collateral"] = max(0, ledger_collateral)

        # Calculate net PPE
        metrics["net_ppe"] = metrics["gross_ppe"] - metrics["accumulated_depreciation"]

        # Calculate total assets
        # Use manufacturer's total_assets if available to ensure consistency with
        # equity calculation (which is based on manufacturer's Assets - Liabilities)
        # This ensures the accounting equation (Assets = Liabilities + Equity) balances
        if mfr_metrics and "assets" in mfr_metrics:
            metrics["assets"] = mfr_metrics["assets"]
        else:
            current_assets = (
                metrics["cash"]
                + metrics["accounts_receivable"]
                + metrics["inventory"]
                + metrics["prepaid_insurance"]
                + metrics["insurance_receivables"]
            )
            metrics["assets"] = current_assets + metrics["net_ppe"] + metrics["restricted_assets"]
        metrics["available_assets"] = metrics["assets"] - metrics["restricted_assets"]

        # Liability accounts (credit-normal, positive balance expected)
        # Some accrued accounts may not be fully tracked in ledger (only payments recorded)
        # Fall back to manufacturer state for accuracy
        metrics["accounts_payable"] = self.ledger.get_balance("accounts_payable", year)

        # Accrued expenses may be negative in ledger if only payments recorded
        ledger_accrued_exp = self.ledger.get_balance("accrued_expenses", year)
        if ledger_accrued_exp <= 0 and mfr_metrics and mfr_metrics.get("accrued_expenses", 0) > 0:
            metrics["accrued_expenses"] = mfr_metrics["accrued_expenses"]
        else:
            metrics["accrued_expenses"] = max(0, ledger_accrued_exp)

        metrics["accrued_wages"] = self.ledger.get_balance("accrued_wages", year)

        # Accrued taxes may be negative in ledger if only payments recorded
        ledger_accrued_tax = self.ledger.get_balance("accrued_taxes", year)
        if ledger_accrued_tax <= 0 and mfr_metrics and mfr_metrics.get("accrued_taxes", 0) > 0:
            metrics["accrued_taxes"] = mfr_metrics["accrued_taxes"]
        else:
            metrics["accrued_taxes"] = max(0, ledger_accrued_tax)

        metrics["accrued_interest"] = self.ledger.get_balance("accrued_interest", year)
        metrics["unearned_revenue"] = self.ledger.get_balance("unearned_revenue", year)

        # Claim liabilities may be negative in ledger if only payments (debits) are recorded
        # without the initial liability setup (credit). Fall back to manufacturer state.
        ledger_claim_liabilities = self.ledger.get_balance("claim_liabilities", year)
        if ledger_claim_liabilities < 0 and mfr_metrics:
            metrics["claim_liabilities"] = mfr_metrics.get("claim_liabilities", 0)
        else:
            metrics["claim_liabilities"] = max(0, ledger_claim_liabilities)

        # Equity: The ledger doesn't record revenue and expense transactions,
        # so we can't compute equity from cumulative P&L. Instead, use the
        # manufacturer's equity from metrics_history, which is computed correctly.
        if mfr_metrics and "equity" in mfr_metrics:
            metrics["equity"] = mfr_metrics["equity"]
        else:
            # Fallback: compute from ledger balances (may not balance)
            retained_earnings_base = self.ledger.get_balance("retained_earnings", year)
            cumulative_revenue = self.ledger.get_balance("revenue", year)
            cumulative_depreciation_exp = self.ledger.get_balance("depreciation_expense", year)
            # Dividends is classified as EQUITY (credit-normal) but we debit it, so balance is negative
            dividends = abs(self.ledger.get_balance("dividends", year))
            metrics["equity"] = (
                retained_earnings_base
                + cumulative_revenue
                - cumulative_depreciation_exp
                - dividends
            )

        # Net assets = Assets - Restricted Assets (available for operations)
        # This matches the Manufacturer's net_assets property definition.
        # Issue #301: Previously used assets - total_liabilities (equity), which
        # is a different concept and caused reconciliation mismatches.
        metrics["net_assets"] = metrics["assets"] - metrics["restricted_assets"]

        # Income statement items (period flows, not cumulative balances)
        # The ledger doesn't record P&L transactions, so fall back to metrics_history
        if mfr_metrics:
            metrics["revenue"] = mfr_metrics.get("revenue", 0)
            metrics["depreciation_expense"] = mfr_metrics.get("depreciation_expense", 0)
            metrics["operating_income"] = mfr_metrics.get("operating_income", 0)
            metrics["net_income"] = mfr_metrics.get("net_income", 0)
            metrics["insurance_premiums"] = mfr_metrics.get("insurance_premiums", 0)
            metrics["insurance_losses"] = mfr_metrics.get("insurance_losses", 0)
            metrics["total_insurance_costs"] = mfr_metrics.get("total_insurance_costs", 0)
            metrics["dividends_paid"] = mfr_metrics.get("dividends_paid", 0)
            # COGS breakdown (Issue #255)
            metrics["direct_materials"] = mfr_metrics.get("direct_materials", 0)
            metrics["direct_labor"] = mfr_metrics.get("direct_labor", 0)
            metrics["manufacturing_overhead"] = mfr_metrics.get("manufacturing_overhead", 0)
            metrics["mfg_depreciation"] = mfr_metrics.get("mfg_depreciation", 0)
            metrics["total_cogs"] = mfr_metrics.get("total_cogs", 0)
            # SG&A breakdown (Issue #255)
            metrics["selling_expenses"] = mfr_metrics.get("selling_expenses", 0)
            metrics["general_admin_expenses"] = mfr_metrics.get("general_admin_expenses", 0)
            metrics["admin_depreciation"] = mfr_metrics.get("admin_depreciation", 0)
            metrics["total_sga"] = mfr_metrics.get("total_sga", 0)
            # Expense ratios for reporting reference
            metrics["gross_margin_ratio"] = mfr_metrics.get("gross_margin_ratio", 0.15)
            metrics["sga_expense_ratio"] = mfr_metrics.get("sga_expense_ratio", 0.07)
            # Tax expense (Issue #257)
            if "tax_expense" in mfr_metrics:
                metrics["tax_expense"] = mfr_metrics["tax_expense"]
        else:
            # Fallback: use ledger period changes (may be 0 if not recorded)
            metrics["revenue"] = self.ledger.get_period_change("revenue", year)
            metrics["depreciation_expense"] = self.ledger.get_period_change(
                "depreciation_expense", year
            )
            # Calculate from ledger if we have revenue
            cogs = self.ledger.get_period_change("cost_of_goods_sold", year)
            operating_exp = self.ledger.get_period_change("operating_expenses", year)
            wage_exp = self.ledger.get_period_change("wage_expense", year)
            metrics["operating_income"] = metrics["revenue"] - cogs - operating_exp - wage_exp
            insurance_exp = self.ledger.get_period_change("insurance_expense", year)
            insurance_loss = self.ledger.get_period_change("insurance_loss", year)
            tax_exp = self.ledger.get_period_change("tax_expense", year)
            interest_exp = self.ledger.get_period_change("interest_expense", year)
            collateral_exp = self.ledger.get_period_change("collateral_expense", year)
            interest_income = self.ledger.get_period_change("interest_income", year)
            insurance_recovery = self.ledger.get_period_change("insurance_recovery", year)
            total_revenue = metrics["revenue"] + interest_income + insurance_recovery
            total_expenses = (
                cogs
                + operating_exp
                + metrics["depreciation_expense"]
                + insurance_exp
                + insurance_loss
                + tax_exp
                + interest_exp
                + collateral_exp
                + wage_exp
            )
            metrics["net_income"] = total_revenue - total_expenses
            metrics["insurance_premiums"] = insurance_exp
            metrics["insurance_losses"] = insurance_loss
            metrics["total_insurance_costs"] = insurance_exp + insurance_loss
            metrics["dividends_paid"] = self.ledger.get_period_change("dividends", year)

        # Solvency check
        metrics["is_solvent"] = metrics["equity"] > 0

        return metrics

    def _get_year_metrics(self, year: int) -> Dict[str, float]:
        """Get metrics for a year, preferring ledger-derived metrics when available.

        This method provides the single entry point for getting financial metrics,
        ensuring that ledger-based calculations are used when a ledger is present.

        Args:
            year: Year index (0-based)

        Returns:
            Dictionary of metrics for the year
        """
        if self.ledger is not None:
            return self._get_metrics_from_ledger(year)
        elif year < len(self.metrics_history):
            return self.metrics_history[year]
        else:
            raise IndexError(f"Year {year} out of range")

    def generate_balance_sheet(
        self, year: int, compare_years: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Generate balance sheet for specified year.

        Creates a standard balance sheet with assets, liabilities, and equity
        sections. Includes year-over-year comparisons if configured.

        When a ledger is available, balances are derived directly from the ledger
        using get_balance() for each account, ensuring perfect reconciliation.
        Otherwise, falls back to metrics_history from the manufacturer.

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

        # Determine available years based on data source
        if self.ledger is not None:
            # When using ledger, we don't have a fixed years_available limit
            # The ledger can calculate balances for any year that has transactions
            if year < 0:
                raise IndexError(f"Year {year} must be non-negative")
        elif year >= self.years_available or year < 0:
            raise IndexError(f"Year {year} out of range. Available: 0-{self.years_available-1}")

        # Get metrics from ledger (preferred) or metrics_history (fallback)
        metrics = self._get_year_metrics(year)

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

        # Issue #256: Critical financial keys MUST be provided by the Manufacturer
        # Fabricating data hides simulation bugs and produces misleading reports
        if "cash" not in metrics:
            raise ValueError(
                "cash missing from metrics. "
                "The Manufacturer class must calculate and provide cash balance explicitly. "
                "(Issue #256: Removed unsafe data estimation from reporting layer)"
            )
        if "gross_ppe" not in metrics:
            raise ValueError(
                "gross_ppe missing from metrics. "
                "The Manufacturer class must calculate and provide gross PP&E explicitly. "
                "(Issue #256: Removed unsafe data estimation from reporting layer)"
            )

        cash = metrics["cash"]
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

        # Property, Plant & Equipment (Issue #256: gross_ppe must be provided)
        gross_ppe = metrics["gross_ppe"]
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

        # Total Assets: Use manufacturer's ledger-based value for consistency
        # with equity calculation (which is Assets - Liabilities from ledger)
        # This ensures the accounting equation (Assets = Liabilities + Equity) balances
        total_assets = total_assets_actual
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

        # Current portion of claims (configurable via FinancialStatementConfig)
        claim_liabilities = metrics.get("claim_liabilities", 0)
        claims_ratio = to_decimal(self.config.current_claims_ratio)
        current_claims = claim_liabilities * claims_ratio if claim_liabilities > 0 else ZERO

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
        total_liabilities_and_equity = total_liabilities + to_decimal(equity)

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

        When a ledger is available, revenue and expenses are derived from ledger
        period changes using get_period_change(), ensuring perfect reconciliation.
        Otherwise, falls back to metrics_history from the manufacturer.

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

        # Determine available years based on data source
        if self.ledger is not None:
            if year < 0:
                raise IndexError(f"Year {year} must be non-negative")
        elif year >= self.years_available or year < 0:
            raise IndexError(f"Year {year} out of range. Available: 0-{self.years_available-1}")

        # Get metrics from ledger (preferred) or metrics_history (fallback)
        metrics = self._get_year_metrics(year)

        # Build income statement structure
        income_data: List[Tuple[str, Union[str, float, int], str, str]] = []

        # Build main sections with GAAP structure
        revenue = self._build_revenue_section(income_data, metrics, monthly)
        gross_profit, operating_income = self._build_gaap_expenses_section(
            income_data, metrics, revenue, monthly
        )
        self._build_gaap_bottom_section(
            income_data, metrics, operating_income, gross_profit, revenue, monthly, year
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

        Issue #255: COGS and SG&A breakdown values are now read from metrics instead
        of being calculated with hardcoded ratios. This moves business logic to the
        Manufacturer model where it belongs, leaving the Reporting layer to only
        format existing data.

        Args:
            data: List to append statement lines to
            metrics: Year metrics dictionary
            revenue: Total revenue for the period
            monthly: If True, use monthly figures

        Returns:
            Tuple of (gross_profit, operating_income)
        """
        # Calculate total depreciation
        # Depreciation expense MUST be provided by the Manufacturer class
        if "depreciation_expense" not in metrics:
            raise ValueError(
                "depreciation_expense missing from metrics. "
                "The Manufacturer class must calculate and provide depreciation_expense explicitly."
            )

        # Issue #255: COGS breakdown MUST be provided by the Manufacturer class
        # This removes hardcoded ratios from the reporting layer
        required_cogs_fields = [
            "direct_materials",
            "direct_labor",
            "manufacturing_overhead",
            "mfg_depreciation",
        ]
        required_sga_fields = [
            "selling_expenses",
            "general_admin_expenses",
            "admin_depreciation",
        ]

        missing_cogs = [f for f in required_cogs_fields if f not in metrics]
        missing_sga = [f for f in required_sga_fields if f not in metrics]

        if missing_cogs:
            raise ValueError(
                f"COGS breakdown fields missing from metrics: {missing_cogs}. "
                "The Manufacturer class must calculate and provide COGS breakdown explicitly. "
                "(Issue #255: Removed hardcoded business logic from reporting layer)"
            )

        if missing_sga:
            raise ValueError(
                f"SG&A breakdown fields missing from metrics: {missing_sga}. "
                "The Manufacturer class must calculate and provide SG&A breakdown explicitly. "
                "(Issue #255: Removed hardcoded business logic from reporting layer)"
            )

        # Read COGS breakdown from metrics (Issue #255)
        direct_materials = metrics["direct_materials"]
        direct_labor = metrics["direct_labor"]
        manufacturing_overhead = metrics["manufacturing_overhead"]
        mfg_depreciation = metrics["mfg_depreciation"]

        # Read SG&A breakdown from metrics (Issue #255)
        selling_expenses = metrics["selling_expenses"]
        general_admin = metrics["general_admin_expenses"]
        admin_depreciation = metrics["admin_depreciation"]

        # Apply monthly scaling if needed
        if monthly:
            direct_materials = direct_materials / 12
            direct_labor = direct_labor / 12
            manufacturing_overhead = manufacturing_overhead / 12
            mfg_depreciation = mfg_depreciation / 12
            selling_expenses = selling_expenses / 12
            general_admin = general_admin / 12
            admin_depreciation = admin_depreciation / 12

        # COST OF GOODS SOLD SECTION
        data.append(("COST OF GOODS SOLD", "", "", ""))

        data.append(("  Direct Materials", direct_materials, "", ""))
        data.append(("  Direct Labor", direct_labor, "", ""))
        data.append(("  Manufacturing Overhead", manufacturing_overhead, "", ""))
        data.append(("  Manufacturing Depreciation", mfg_depreciation, "", ""))

        total_cogs = direct_materials + direct_labor + manufacturing_overhead + mfg_depreciation
        data.append(("  Total Cost of Goods Sold", total_cogs, "", "subtotal"))
        data.append(("", "", "", ""))

        # GROSS PROFIT
        gross_profit = revenue - total_cogs
        data.append(("GROSS PROFIT", gross_profit, "", "subtotal"))
        data.append(("", "", "", ""))

        # OPERATING EXPENSES (SG&A)
        data.append(("OPERATING EXPENSES", "", "", ""))

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
        year: int = 0,
    ) -> None:
        """Build the bottom section with non-operating items, taxes, and net income.

        Follows GAAP structure with separate non-operating section. Tax expense
        is read from the Ledger or metrics when available, falling back to flat
        rate calculation only when no actual tax data exists.

        Issue #257: Tax expense should be read from the Ledger (sum of TAX_ACCRUAL
        entries) rather than recalculated. This ensures the Income Statement
        reports what actually happened in the simulation, not what should have
        happened based on a flat rate.

        Args:
            data: List to append statement lines to
            metrics: Year metrics dictionary
            operating_income: Operating income (EBIT) for the period
            gross_profit: Gross profit (revenue - COGS) for the period
            revenue: Total revenue for the period
            monthly: If True, use monthly figures
            year: Year index for ledger queries
        """
        # NON-OPERATING INCOME (EXPENSES)
        data.append(("NON-OPERATING INCOME (EXPENSES)", "", "", ""))

        # Interest income/expense: read from metrics (Issue #301)
        # The reporting layer must not fabricate financial data with hardcoded rates.
        # Interest income and expense should be computed by the Manufacturer or Ledger.
        interest_income = metrics.get("interest_income", 0)
        if monthly:
            interest_income = interest_income / 12

        interest_expense = metrics.get("interest_expense", 0)
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

        # INCOME TAX PROVISION
        # Issue #257: Read tax expense from Ledger (preferred) or metrics (fallback)
        # The Reporting layer should report what happened, not recalculate it.
        # Priority: 1) Ledger TAX_ACCRUAL entries, 2) metrics["tax_expense"], 3) flat rate
        tax_provision: Optional[float] = None

        # Priority 1: Get tax expense from Ledger if available
        if self.ledger is not None:
            ledger_tax = self.ledger.get_period_change("tax_expense", year)
            if ledger_tax > ZERO:
                tax_provision = float(ledger_tax)
                if monthly:
                    tax_provision = tax_provision / 12

        # Priority 2: Get tax expense from metrics if provided by Manufacturer
        if tax_provision is None and "tax_expense" in metrics:
            tax_provision = float(metrics["tax_expense"])
            if monthly:
                tax_provision = tax_provision / 12

        # Priority 3: Fall back to flat rate calculation (backward compatibility)
        if tax_provision is None:
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
        # Issue #301: Use manufacturer's net_income directly to ensure consistency
        # between income statement and balance sheet (which uses manufacturer's equity).
        # The income statement presentation above may compute a different operating_income
        # due to GAAP categorization differences, but the bottom line must match.
        net_income = metrics.get("net_income", float(pretax_income) - float(tax_provision))
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
        """Generate cash flow statement for specified year using CashFlowStatement class.

        Creates a cash flow statement with three distinct sections (Operating,
        Investing, Financing). Supports both indirect method (starting from
        net income) and direct method (summing ledger entries) for operating
        activities.

        When a ledger is available, the direct method is preferred as it provides
        perfect reconciliation and audit trail for all cash transactions by
        summing actual ledger entries.

        Args:
            year: Year index (0-based) for cash flow statement
            period: 'annual' or 'monthly' for period type
            method: 'indirect' (default) or 'direct'. Direct method requires
                a ledger to be available. When ledger is present and no method
                specified, direct method may be preferred for better accuracy.

        Returns:
            DataFrame containing cash flow statement data

        Raises:
            IndexError: If year is out of range
            ValueError: If direct method requested but no ledger available
        """
        # Update metrics cache to get latest data
        self._update_metrics_cache()

        # Determine available years based on data source
        if self.ledger is not None:
            if year < 0:
                raise IndexError(f"Year {year} must be non-negative")
        elif year >= self.years_available or year < 0:
            raise IndexError(f"Year {year} out of range. Available: 0-{self.years_available-1}")

        # Create CashFlowStatement instance with ledger if available
        cash_flow_generator = CashFlowStatement(
            self.metrics_history,
            self.manufacturer_data.get("config"),
            ledger=self.ledger,
        )

        # Generate and return the statement with specified method
        return cash_flow_generator.generate_statement(year, period=period, method=method)

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
        """Check if balance sheet equation balances.

        Uses full total liabilities (accounts_payable + accrued_expenses +
        claim_liabilities) rather than just claim_liabilities, matching
        the liabilities section of the balance sheet (Issue #301).
        """
        data.append(("BALANCE SHEET RECONCILIATION", "", "", ""))
        assets = to_decimal(metrics.get("assets", 0))
        # Issue #301: Use full total liabilities matching _build_liabilities_section
        claims_ratio = to_decimal(self.config.current_claims_ratio)
        claim_liabilities = to_decimal(metrics.get("claim_liabilities", 0))
        current_claims = claim_liabilities * claims_ratio if claim_liabilities > 0 else ZERO
        long_term_claims = claim_liabilities - current_claims
        total_current = (
            to_decimal(metrics.get("accounts_payable", 0))
            + to_decimal(metrics.get("accrued_expenses", 0))
            + current_claims
        )
        liabilities = total_current + long_term_claims
        equity = to_decimal(metrics.get("equity", 0))

        difference = assets - (liabilities + equity)
        # Using is_zero() for precise comparison after quantization
        balance_check = is_zero(difference)
        data.append(("  Assets", float(assets), "", ""))
        data.append(("  Liabilities + Equity", float(liabilities + equity), "", ""))
        data.append(("  Difference", float(difference), "", ""))
        data.append(("  Status", "BALANCED" if balance_check else "IMBALANCED", "", "status"))
        data.append(("", "", "", ""))

    def _check_net_assets(
        self, data: List[Tuple[str, Union[str, float, int], str, str]], metrics: Dict[str, float]
    ) -> None:
        """Check net assets reconciliation.

        Net assets = Total Assets - Restricted Assets (available for operations).
        This matches the Manufacturer's net_assets property definition (Issue #301).
        """
        data.append(("NET ASSETS RECONCILIATION", "", "", ""))
        net_assets = to_decimal(metrics.get("net_assets", 0))
        calc_net_assets = to_decimal(metrics.get("assets", 0)) - to_decimal(
            metrics.get("restricted_assets", 0)
        )
        difference = net_assets - calc_net_assets
        # Using is_zero() for precise comparison after quantization
        net_assets_check = is_zero(difference)

        data.append(("  Reported Net Assets", float(net_assets), "", ""))
        data.append(("  Calculated (Assets - Restricted)", float(calc_net_assets), "", ""))
        data.append(("  Difference", float(difference), "", ""))
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
                    # Calculate total liabilities same as in _build_liabilities_section
                    # Includes accounts_payable, accrued_expenses, and claim_liabilities
                    accounts_payable = comp_metrics.get("accounts_payable", 0)
                    accrued_expenses = comp_metrics.get("accrued_expenses", 0)
                    claim_liabilities = comp_metrics.get("claim_liabilities", 0)
                    total_liabilities = accounts_payable + accrued_expenses + claim_liabilities
                    df.at[index, f"Year {comp_year}"] = total_liabilities

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

        # Default fiscal_year_end to 12 if not set (no central config available here)
        if self.config.fiscal_year_end is None:
            self.config.fiscal_year_end = 12

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
