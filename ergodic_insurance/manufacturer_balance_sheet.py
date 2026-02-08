# mypy: disable-error-code="attr-defined, has-type, no-any-return"
"""Balance sheet mixin for WidgetManufacturer.

This module contains the BalanceSheetMixin class, extracted from manufacturer.py
as part of the decomposition refactor (Issue #305). It provides all balance sheet
properties, ledger helper methods, working capital calculations, depreciation,
and growth logic.
"""

from decimal import Decimal
import logging
from typing import Dict, Union

try:
    from ergodic_insurance.decimal_utils import ONE, ZERO, quantize_currency, to_decimal
    from ergodic_insurance.ledger import AccountName, TransactionType
except ImportError:
    try:
        from .decimal_utils import ONE, ZERO, quantize_currency, to_decimal
        from .ledger import AccountName, TransactionType
    except ImportError:
        from decimal_utils import ONE, ZERO, quantize_currency, to_decimal  # type: ignore[no-redef]
        from ledger import AccountName, TransactionType  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class BalanceSheetMixin:
    """Mixin providing balance sheet properties and ledger helper methods.

    This mixin expects the host class to have:
        - self.ledger: Ledger instance
        - self.config: ManufacturerConfig instance
        - self.current_year: int
        - self.current_month: int
        - self.base_operating_margin: float
        - self.claim_liabilities: list
        - self.accrual_manager: AccrualManager instance
        - self.stochastic_process: Optional[StochasticProcess]
        - self.asset_turnover_ratio: float
    """

    # ========================================================================
    # Balance Sheet Properties - Ledger is Single Source of Truth (Issue #275)
    # ========================================================================

    @property
    def cash(self) -> Decimal:
        """Cash balance derived from ledger (single source of truth).

        Returns:
            Current cash balance from the ledger.
        """
        return self.ledger.get_balance(AccountName.CASH)

    @property
    def accounts_receivable(self) -> Decimal:
        """Accounts receivable balance derived from ledger (single source of truth).

        Returns:
            Current accounts receivable balance from the ledger.
        """
        return self.ledger.get_balance(AccountName.ACCOUNTS_RECEIVABLE)

    @property
    def inventory(self) -> Decimal:
        """Inventory balance derived from ledger (single source of truth).

        Returns:
            Current inventory balance from the ledger.
        """
        return self.ledger.get_balance(AccountName.INVENTORY)

    @property
    def prepaid_insurance(self) -> Decimal:
        """Prepaid insurance balance derived from ledger (single source of truth).

        Returns:
            Current prepaid insurance balance from the ledger.
        """
        return self.ledger.get_balance(AccountName.PREPAID_INSURANCE)

    @property
    def gross_ppe(self) -> Decimal:
        """Gross PP&E balance derived from ledger (single source of truth).

        Returns:
            Current gross property, plant & equipment balance from the ledger.
        """
        return self.ledger.get_balance(AccountName.GROSS_PPE)

    @property
    def accumulated_depreciation(self) -> Decimal:
        """Accumulated depreciation balance derived from ledger (single source of truth).

        Accumulated depreciation is a contra-asset account stored with credit-normal
        balance (negative in the ledger). This property returns the absolute value
        for intuitive usage - a positive value representing total depreciation.

        Returns:
            Current accumulated depreciation balance as a positive Decimal.
        """
        return abs(self.ledger.get_balance(AccountName.ACCUMULATED_DEPRECIATION))

    @property
    def restricted_assets(self) -> Decimal:
        """Restricted assets balance derived from ledger (single source of truth).

        Returns:
            Current restricted assets (restricted cash) balance from the ledger.
        """
        return self.ledger.get_balance(AccountName.RESTRICTED_CASH)

    @property
    def accounts_payable(self) -> Decimal:
        """Accounts payable balance derived from ledger (single source of truth).

        Returns:
            Current accounts payable balance from the ledger.
        """
        return self.ledger.get_balance(AccountName.ACCOUNTS_PAYABLE)

    @property
    def collateral(self) -> Decimal:
        """Letter of credit collateral balance derived from restricted assets.

        Collateral is tracked via RESTRICTED_CASH since restricted assets
        represent cash pledged as collateral for insurance claims (issue #302).

        Returns:
            Current collateral balance (equals restricted assets).
        """
        return self.restricted_assets

    @property
    def deferred_tax_asset(self) -> Decimal:
        """Deferred tax asset from NOL carryforward per ASC 740-10-25-3 (Issue #365).

        Returns:
            Current deferred tax asset balance from the ledger.
        """
        return self.ledger.get_balance(AccountName.DEFERRED_TAX_ASSET)

    @property
    def total_assets(self) -> Decimal:
        """Calculate total assets from all asset components.

        Total assets include all current and non-current assets following
        the accounting equation: Assets = Liabilities + Equity.

        Returns:
            Decimal: Total assets in dollars, sum of all asset components.
        """
        # Current assets
        current = self.cash + self.accounts_receivable + self.inventory + self.prepaid_insurance
        # Non-current assets
        net_ppe = self.gross_ppe - self.accumulated_depreciation
        # Total (includes DTA per ASC 740, Issue #365)
        return current + net_ppe + self.restricted_assets + self.deferred_tax_asset

    @property
    def total_liabilities(self) -> Decimal:
        """Calculate total liabilities from all liability components.

        Total liabilities include current liabilities and long-term claim liabilities.

        Note:
            ClaimLiability objects are the single source of truth for insurance claim
            liabilities. INSURANCE_CLAIMS in AccrualManager are excluded from this
            calculation to prevent double-counting. See GitHub issue #213.

        Returns:
            Decimal: Total liabilities in dollars, sum of all liability components.
        """
        from ergodic_insurance.accrual_manager import AccrualType

        # Get accrued expenses from accrual manager (excludes INSURANCE_CLAIMS to avoid
        # double-counting with ClaimLiability objects which are the source of truth)
        accrual_items = self.accrual_manager.get_balance_sheet_items()
        total_accrued_expenses = to_decimal(accrual_items.get("accrued_expenses", ZERO))

        # Subtract any insurance claims from accrual manager to prevent double-counting
        # ClaimLiability is the authoritative source for claim liabilities
        insurance_claims_in_accrual = sum(
            (
                to_decimal(a.remaining_balance)
                for a in self.accrual_manager.accrued_expenses.get(AccrualType.INSURANCE_CLAIMS, [])
                if not a.is_fully_paid
            ),
            ZERO,
        )
        adjusted_accrued_expenses = total_accrued_expenses - insurance_claims_in_accrual

        # Current liabilities - AccrualManager is single source of truth (issue #238)
        current_liabilities = self.accounts_payable + adjusted_accrued_expenses

        # Long-term liabilities (claim liabilities) - single source of truth
        claim_liability_total = sum(
            (liability.remaining_amount for liability in self.claim_liabilities), ZERO
        )

        return current_liabilities + claim_liability_total

    @property
    def equity(self) -> Decimal:
        """Calculate equity using the accounting equation.

        Equity is derived as Assets - Liabilities, ensuring the accounting
        equation always balances: Assets = Liabilities + Equity.

        Returns:
            Decimal: Shareholder equity in dollars.
        """
        return self.total_assets - self.total_liabilities

    @property
    def net_assets(self) -> Decimal:
        """Calculate net assets (total assets minus restricted assets).

        Net assets represent the portion of total assets that are available
        for operational use. Restricted assets are those pledged as collateral
        for insurance claims and cannot be used for general business purposes.

        Returns:
            Decimal: Net assets in dollars. Always non-negative as restricted
                assets cannot exceed total assets.
        """
        return self.total_assets - self.restricted_assets

    @property
    def available_assets(self) -> Decimal:
        """Calculate available (unrestricted) assets for operations.

        This property is an alias for net_assets, providing semantic clarity
        when referring to assets available for business operations.

        Returns:
            float: Available assets in dollars. Equal to total assets minus
                restricted assets pledged as insurance collateral.
        """
        return self.total_assets - self.restricted_assets

    @property
    def net_ppe(self) -> Decimal:
        """Calculate net property, plant & equipment after depreciation.

        Returns:
            Decimal: Net PP&E (gross PP&E minus accumulated depreciation).
        """
        return self.gross_ppe - self.accumulated_depreciation

    # ========================================================================
    # Ledger Helper Methods for Balance Sheet Modifications (Issue #275)
    # ========================================================================

    def _record_cash_adjustment(
        self,
        amount: Decimal,
        description: str,
        transaction_type: TransactionType = TransactionType.ADJUSTMENT,
    ) -> None:
        """Record a cash adjustment through the ledger.

        .. warning::
            This method creates phantom assets (Debit CASH, Credit RETAINED_EARNINGS)
            which inflates total_assets and equity. It should only be used in tests
            to set up specific cash/equity states. Production code should use proper
            accounting entries through specific expense or liability accounts.
            See Issue #319 for details.

        Args:
            amount: Positive to increase cash, negative to decrease
            description: Description of the adjustment
            transaction_type: Type of transaction (default: ADJUSTMENT)
        """
        if amount > ZERO:
            # Increase cash - debit cash, credit retained earnings
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.CASH,
                credit_account=AccountName.RETAINED_EARNINGS,
                amount=amount,
                transaction_type=transaction_type,
                description=description,
            )
        elif amount < ZERO:
            # Decrease cash - debit retained earnings, credit cash
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.RETAINED_EARNINGS,
                credit_account=AccountName.CASH,
                amount=-amount,
                transaction_type=transaction_type,
                description=description,
            )

    def _record_asset_transfer(
        self,
        from_account: AccountName,
        to_account: AccountName,
        amount: Decimal,
        description: str,
    ) -> None:
        """Record a transfer between asset accounts through the ledger.

        Used for moving cash to restricted assets, etc.

        Args:
            from_account: Source account
            to_account: Destination account
            amount: Amount to transfer (must be positive)
            description: Description of the transfer
        """
        if amount <= ZERO:
            return
        self.ledger.record_double_entry(
            date=self.current_year,
            debit_account=to_account,
            credit_account=from_account,
            amount=amount,
            transaction_type=TransactionType.TRANSFER,
            description=description,
        )

    def _write_off_all_assets(self, description: str = "Asset write-off") -> None:
        """Write off all asset balances to zero through the ledger.

        Args:
            description: Description of the write-off
        """
        # Write off each asset account that has a balance
        asset_accounts = [
            (AccountName.CASH, self.cash),
            (AccountName.ACCOUNTS_RECEIVABLE, self.accounts_receivable),
            (AccountName.INVENTORY, self.inventory),
            (AccountName.PREPAID_INSURANCE, self.prepaid_insurance),
            (AccountName.GROSS_PPE, self.gross_ppe),
            (AccountName.RESTRICTED_CASH, self.restricted_assets),
        ]

        for account, balance in asset_accounts:
            if balance > ZERO:
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.RETAINED_EARNINGS,
                    credit_account=account,
                    amount=balance,
                    transaction_type=TransactionType.WRITE_OFF,
                    description=f"{description} - {account.value}",
                )

        # Write off accumulated depreciation (contra-asset - has credit balance)
        accum_depr = self.accumulated_depreciation
        if accum_depr > ZERO:
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.ACCUMULATED_DEPRECIATION,
                credit_account=AccountName.RETAINED_EARNINGS,
                amount=accum_depr,
                transaction_type=TransactionType.WRITE_OFF,
                description=f"{description} - accumulated_depreciation",
            )

    def _record_liquidation(
        self,
        amount: Decimal,
        description: str = "Liquidation loss",
    ) -> None:
        """Record liquidation loss through the ledger.

        Used during bankruptcy to record asset liquidation costs.
        Routes the loss through INSURANCE_LOSS expense so it appears on the
        income statement before closing to retained earnings (Issue #319).

        Args:
            amount: Amount of liquidation loss
            description: Description of the liquidation
        """
        if amount <= ZERO:
            return
        self.ledger.record_double_entry(
            date=self.current_year,
            debit_account=AccountName.INSURANCE_LOSS,
            credit_account=AccountName.CASH,
            amount=amount,
            transaction_type=TransactionType.LIQUIDATION,
            description=description,
        )

    def _verify_accounting_equation(self) -> None:
        """Verify that the ledger's debits equal credits after each period.

        Asserts that the double-entry bookkeeping invariant holds: total debits
        must equal total credits across all ledger entries. This ensures no
        phantom assets or liabilities have been created (Issue #319).

        Raises:
            AssertionError: If total debits != total credits.
        """
        is_balanced, difference = self.ledger.verify_balance()
        assert is_balanced, (
            f"Accounting equation violation: ledger debits - credits = ${difference:,.2f} "
            f"at year {self.current_year}. This indicates phantom assets or "
            f"unbalanced entries were created."
        )

    def _record_liquid_asset_reduction(
        self,
        total_reduction: Decimal,
        description: str = "Liquid asset reduction for claim payment",
    ) -> None:
        """Reduce liquid assets proportionally to make a payment.

        Reduces cash, accounts receivable, and inventory proportionally
        to fund a payment amount.

        Args:
            total_reduction: Total amount to reduce from liquid assets
            description: Description of the reduction
        """
        if total_reduction <= ZERO:
            return

        # Get current liquid asset balances
        current_cash = self.cash
        current_ar = self.accounts_receivable
        current_inventory = self.inventory
        total_liquid = current_cash + current_ar + current_inventory

        if total_liquid <= ZERO:
            return

        # Calculate reduction ratio
        if total_liquid <= total_reduction:
            # Use all liquid assets
            reduction_ratio = ONE
        else:
            reduction_ratio = total_reduction / total_liquid

        # Reduce each account proportionally
        cash_reduction = current_cash * reduction_ratio
        ar_reduction = current_ar * reduction_ratio
        inventory_reduction = current_inventory * reduction_ratio

        if cash_reduction > ZERO:
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.INSURANCE_LOSS,
                credit_account=AccountName.CASH,
                amount=quantize_currency(cash_reduction),
                transaction_type=TransactionType.INSURANCE_CLAIM,
                description=f"{description} - cash",
            )

        if ar_reduction > ZERO:
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.INSURANCE_LOSS,
                credit_account=AccountName.ACCOUNTS_RECEIVABLE,
                amount=quantize_currency(ar_reduction),
                transaction_type=TransactionType.INSURANCE_CLAIM,
                description=f"{description} - accounts_receivable",
            )

        if inventory_reduction > ZERO:
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.INSURANCE_LOSS,
                credit_account=AccountName.INVENTORY,
                amount=quantize_currency(inventory_reduction),
                transaction_type=TransactionType.INSURANCE_CLAIM,
                description=f"{description} - inventory",
            )

    # ========================================================================
    # Working Capital and Balance Sheet Updates
    # ========================================================================

    def calculate_working_capital_components(
        self, revenue: Union[Decimal, float], dso: float = 45, dio: float = 60, dpo: float = 30
    ) -> Dict[str, Decimal]:
        """Calculate individual working capital components based on revenue and ratios.

        Uses standard financial ratios to calculate accounts receivable, inventory,
        and accounts payable from annual revenue.

        Args:
            revenue (float): Annual revenue in dollars.
            dso (float): Days Sales Outstanding. Defaults to 45.
            dio (float): Days Inventory Outstanding. Defaults to 60.
            dpo (float): Days Payable Outstanding. Defaults to 30.

        Returns:
            Dict[str, float]: Dictionary containing working capital components.
        """
        # Convert inputs to Decimal
        revenue_decimal = to_decimal(revenue)
        dso_decimal = to_decimal(dso)
        dio_decimal = to_decimal(dio)
        dpo_decimal = to_decimal(dpo)
        days_per_year = to_decimal(365)

        # Calculate cost of goods sold (approximate as % of revenue)
        cogs = revenue_decimal * to_decimal(1 - self.base_operating_margin)

        # Calculate new working capital components
        new_ar = revenue_decimal * (dso_decimal / days_per_year)
        new_inventory = cogs * (dio_decimal / days_per_year)  # Inventory based on COGS not revenue
        new_ap = cogs * (dpo_decimal / days_per_year)  # AP based on COGS not revenue

        # Calculate the change in working capital components
        ar_change = new_ar - self.accounts_receivable
        inventory_change = new_inventory - self.inventory
        ap_change = new_ap - self.accounts_payable

        # Record working capital changes in ledger
        # FIX (Issue #305): Use AccountName enum instead of string literals
        # AR increase: Debit AR, Credit Cash (collection cycle)
        if ar_change > 0:
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.ACCOUNTS_RECEIVABLE,
                credit_account=AccountName.CASH,
                amount=ar_change,
                transaction_type=TransactionType.WORKING_CAPITAL,
                description="AR increase from revenue growth",
                month=self.current_month,
            )
        elif ar_change < 0:
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.CASH,
                credit_account=AccountName.ACCOUNTS_RECEIVABLE,
                amount=abs(ar_change),
                transaction_type=TransactionType.COLLECTION,
                description="AR decrease (net collections)",
                month=self.current_month,
            )

        # Inventory increase: Debit Inventory, Credit Cash (purchase cycle)
        if inventory_change > 0:
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.INVENTORY,
                credit_account=AccountName.CASH,
                amount=inventory_change,
                transaction_type=TransactionType.INVENTORY_PURCHASE,
                description="Inventory increase",
                month=self.current_month,
            )
        elif inventory_change < 0:
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.CASH,
                credit_account=AccountName.INVENTORY,
                amount=abs(inventory_change),
                transaction_type=TransactionType.WORKING_CAPITAL,
                description="Inventory decrease",
                month=self.current_month,
            )

        # AP increase: Debit Cash, Credit AP (we owe more to vendors)
        if ap_change > 0:
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.CASH,
                credit_account=AccountName.ACCOUNTS_PAYABLE,
                amount=ap_change,
                transaction_type=TransactionType.WORKING_CAPITAL,
                description="AP increase from purchases",
                month=self.current_month,
            )
        elif ap_change < 0:
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.ACCOUNTS_PAYABLE,
                credit_account=AccountName.CASH,
                amount=abs(ap_change),
                transaction_type=TransactionType.PAYMENT,
                description="AP decrease (payments to vendors)",
                month=self.current_month,
            )

        # LIMITED LIABILITY: Check if ledger-based changes would make cash negative
        if self.cash < ZERO:
            shortfall = -self.cash
            logger.warning(
                f"WORKING CAPITAL FACILITY: Working capital changes pushed cash to ${self.cash:,.2f}. "
                f"Recording ${shortfall:,.2f} as accounts payable (vendor financing)."
            )
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.CASH,
                credit_account=AccountName.ACCOUNTS_PAYABLE,
                amount=shortfall,
                transaction_type=TransactionType.WORKING_CAPITAL,
                description="Working capital facility - vendor financing for cash shortfall",
                month=self.current_month,
            )

        # Calculate net working capital and cash conversion cycle
        net_working_capital = self.accounts_receivable + self.inventory - self.accounts_payable
        cash_conversion_cycle_days = dso_decimal + dio_decimal - dpo_decimal

        # Calculate cash impact for logging
        cash_impact = -(ar_change + inventory_change) + ap_change

        logger.debug(
            f"Working capital components: AR=${self.accounts_receivable:,.0f}, "
            f"Inv=${self.inventory:,.0f}, AP=${self.accounts_payable:,.0f}, "
            f"Net WC=${net_working_capital:,.0f}, Cash impact=${cash_impact:,.0f}"
        )

        return {
            "accounts_receivable": self.accounts_receivable,
            "inventory": self.inventory,
            "accounts_payable": self.accounts_payable,
            "net_working_capital": net_working_capital,
            "cash_conversion_cycle": cash_conversion_cycle_days,
            "cash_impact": cash_impact,
        }

    def update_balance_sheet(
        self, net_income: Union[Decimal, float], growth_rate: Union[Decimal, float] = 0.0
    ) -> None:
        """Update balance sheet with retained earnings and dividend distribution.

        This method processes the financial results of a period by allocating
        net income between retained earnings and dividend payments.

        Args:
            net_income (float): Net income for the period in dollars.
            growth_rate (float): Revenue growth rate parameter (currently unused).
        """
        # Convert input to Decimal
        net_income_decimal = to_decimal(net_income)

        # Validation: retention ratio should be applied to net income
        assert 0 <= self.retention_ratio <= 1, f"Invalid retention ratio: {self.retention_ratio}"

        # Calculate retained earnings
        retention_decimal = to_decimal(self.retention_ratio)
        retained_earnings = net_income_decimal * retention_decimal
        dividends = net_income_decimal * (to_decimal(1) - retention_decimal)

        # Log retention calculation details
        logger.info("===== RETENTION CALCULATION =====")
        logger.info(f"Net Income:              ${net_income_decimal:,.2f}")
        logger.info(f"Retention Ratio:         {self.retention_ratio:.1%}")
        logger.info(f"Retained Earnings:       ${retained_earnings:,.2f}")
        if net_income_decimal > ZERO:
            logger.info(f"Dividends Distributed:   ${dividends:,.2f}")
        else:
            logger.info(f"Loss Absorption:         ${retained_earnings:,.2f}")
        logger.info("=================================")

        # LIMITED LIABILITY: Check liquidity and solvency before absorbing losses
        if retained_earnings < ZERO:
            current_equity = self.equity
            available_cash = self.cash
            tolerance = to_decimal(self.config.insolvency_tolerance)
            loss_amount = abs(retained_earnings)

            # Check 1: Already insolvent by equity threshold
            if current_equity <= tolerance:
                logger.warning(
                    f"LIMITED LIABILITY: Equity too low (${current_equity:,.2f}) to absorb any losses. "
                    f"Cannot absorb loss of ${loss_amount:,.2f}. "
                    f"Company is already insolvent (threshold: ${tolerance:,.2f})."
                )
                self._last_dividends_paid = ZERO
                return

            # Check 2: LIQUIDITY CHECK - must have cash to pay loss
            if loss_amount > available_cash:
                logger.error(
                    f"LIQUIDITY CRISIS → INSOLVENCY: Loss ${loss_amount:,.2f} exceeds "
                    f"available cash ${available_cash:,.2f}. Equity=${current_equity:,.2f}. "
                    f"Company cannot meet obligations despite positive book equity."
                )
                self._last_dividends_paid = ZERO
                self.handle_insolvency()
                return

            # Check 3: Would paying the loss trigger equity insolvency?
            equity_after_loss = current_equity - loss_amount
            if equity_after_loss <= tolerance:
                logger.error(
                    f"EQUITY INSOLVENCY: Loss ${loss_amount:,.2f} would push equity to "
                    f"${equity_after_loss:,.2f}, below threshold ${tolerance:,.2f}. "
                    f"Current equity=${current_equity:,.2f}. Triggering insolvency."
                )
                # Apply the loss to the balance sheet via ledger (Issue #275)
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.RETAINED_EARNINGS,
                    credit_account=AccountName.CASH,
                    amount=loss_amount,
                    transaction_type=TransactionType.EXPENSE,
                    description=f"Year {self.current_year} operating loss (pre-insolvency)",
                    month=self.current_month,
                )
                self._last_dividends_paid = ZERO
                self.handle_insolvency()
                return

            # All checks passed - absorb the full loss via ledger (Issue #275)
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.RETAINED_EARNINGS,
                credit_account=AccountName.CASH,
                amount=loss_amount,
                transaction_type=TransactionType.EXPENSE,
                description=f"Year {self.current_year} operating loss",
                month=self.current_month,
            )

            self._last_dividends_paid = ZERO
            logger.info(
                f"Absorbed loss: ${loss_amount:,.2f}. "
                f"Remaining cash: ${self.cash:,.2f}, Remaining equity: ${self.equity:,.2f}"
            )
        else:
            # Positive retained earnings - check cash constraints for dividends
            projected_cash = self.cash + retained_earnings

            if projected_cash <= ZERO:
                actual_dividends = ZERO
                additional_retained = dividends
                logger.warning(
                    f"DIVIDEND CONSTRAINT: Projected cash ${projected_cash:,.2f} <= 0. "
                    f"No dividends can be paid. All ${dividends:,.2f} retained."
                )
            elif projected_cash < dividends:
                actual_dividends = projected_cash
                additional_retained = dividends - actual_dividends
                logger.warning(
                    f"DIVIDEND CONSTRAINT: Projected cash ${projected_cash:,.2f} "
                    f"< theoretical dividends ${dividends:,.2f}. "
                    f"Paying only ${actual_dividends:,.2f}."
                )
            else:
                actual_dividends = dividends
                additional_retained = ZERO

            self._last_dividends_paid = actual_dividends

            total_retained = retained_earnings + additional_retained

            if total_retained > ZERO:
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.CASH,
                    credit_account=AccountName.RETAINED_EARNINGS,
                    amount=total_retained,
                    transaction_type=TransactionType.RETAINED_EARNINGS,
                    description=f"Year {self.current_year} retained earnings",
                    month=self.current_month,
                )

        logger.info(
            f"Balance sheet updated: Assets=${self.total_assets:,.2f}, Equity=${self.equity:,.2f}"
        )
        if self._last_dividends_paid > ZERO:
            logger.info(f"Dividends paid: ${self._last_dividends_paid:,.2f}")

    def record_depreciation(self, useful_life_years: Union[Decimal, float, int] = 10) -> Decimal:
        """Record straight-line depreciation on PP&E.

        Args:
            useful_life_years: Average useful life of PP&E in years. Defaults to 10.

        Returns:
            Decimal: Annual depreciation expense recorded.
        """
        useful_life = to_decimal(useful_life_years)
        if self.gross_ppe > ZERO and useful_life > ZERO:
            annual_depreciation = self.gross_ppe / useful_life

            # Don't depreciate below zero net book value
            net_ppe = self.gross_ppe - self.accumulated_depreciation
            if net_ppe > ZERO:
                depreciation_expense = min(annual_depreciation, net_ppe)

                # Record depreciation via ledger only (Issue #275)
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.DEPRECIATION_EXPENSE,
                    credit_account=AccountName.ACCUMULATED_DEPRECIATION,
                    amount=depreciation_expense,
                    transaction_type=TransactionType.DEPRECIATION,
                    description=f"Year {self.current_year} depreciation",
                    month=self.current_month,
                )

                logger.debug(
                    f"Recorded depreciation: ${depreciation_expense:,.2f}, "
                    f"Accumulated: ${self.accumulated_depreciation:,.2f}"
                )
                return depreciation_expense
        return ZERO

    def record_capex(self, capex_amount: Union[Decimal, float]) -> Decimal:
        """Record capital expenditure (reinvestment in PP&E).

        Capitalizes the expenditure by debiting Gross PP&E and crediting Cash.
        Capex is not an expense — it increases the asset base rather than
        reducing net income (ASC 360-10).

        Args:
            capex_amount: Capital expenditure amount to record.

        Returns:
            Decimal: Actual capex recorded (may be less than requested if
            constrained by available cash).
        """
        amount = to_decimal(capex_amount)
        if amount <= ZERO:
            return ZERO

        # Cannot spend more cash than available (Issue #543)
        available_cash = self.cash
        if available_cash <= ZERO:
            logger.warning(
                f"Skipping capex: no cash available "
                f"(cash={available_cash:,.2f}, requested={amount:,.2f})"
            )
            return ZERO

        actual_capex = min(amount, available_cash)
        if actual_capex < amount:
            logger.warning(
                f"Capex constrained by cash: requested=${amount:,.2f}, "
                f"available=${available_cash:,.2f}, recording=${actual_capex:,.2f}"
            )

        self.ledger.record_double_entry(
            date=self.current_year,
            debit_account=AccountName.GROSS_PPE,
            credit_account=AccountName.CASH,
            amount=actual_capex,
            transaction_type=TransactionType.CAPEX,
            description=f"Year {self.current_year} capital expenditure",
            month=self.current_month,
        )

        logger.debug(
            f"Recorded capex: ${actual_capex:,.2f}, "
            f"Gross PP&E: ${self.gross_ppe:,.2f}, "
            f"Cash: ${self.cash:,.2f}"
        )
        return actual_capex

    def _apply_growth(
        self, growth_rate: Union[Decimal, float], time_resolution: str, apply_stochastic: bool
    ) -> None:
        """Apply revenue growth by adjusting asset turnover ratio.

        Args:
            growth_rate: Revenue growth rate for the period.
            time_resolution: "annual" or "monthly" for simulation step.
            apply_stochastic: Whether to apply stochastic shocks.
        """
        rate = to_decimal(growth_rate) if not isinstance(growth_rate, Decimal) else growth_rate
        if rate == ZERO or not (time_resolution == "annual" or self.current_month == 11):
            return

        base_growth = float(ONE + rate)  # Boundary: float for StochasticProcess

        # Add stochastic component to growth if enabled
        if apply_stochastic and self.stochastic_process is not None:
            growth_shock = self.stochastic_process.generate_shock(1.0)
            total_growth = base_growth * growth_shock
            self.asset_turnover_ratio *= total_growth
            logger.debug(
                f"Applied growth: {total_growth:.4f} (base={base_growth:.4f}, shock={growth_shock:.4f})"
            )
        else:
            self.asset_turnover_ratio *= base_growth
