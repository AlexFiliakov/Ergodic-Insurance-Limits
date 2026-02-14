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
    from ergodic_insurance.ledger import (
        CHART_OF_ACCOUNTS,
        AccountName,
        AccountType,
        TransactionType,
    )
except ImportError:
    try:
        from .decimal_utils import ONE, ZERO, quantize_currency, to_decimal
        from .ledger import (
            CHART_OF_ACCOUNTS,
            AccountName,
            AccountType,
            TransactionType,
        )
    except ImportError:
        from decimal_utils import ONE, ZERO, quantize_currency, to_decimal  # type: ignore[no-redef]
        from ledger import (  # type: ignore[no-redef]
            CHART_OF_ACCOUNTS,
            AccountName,
            AccountType,
            TransactionType,
        )

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
            Current cash balance from the ledger.  May be negative when the
            working capital facility has been drawn.  For balance-sheet
            presentation the negative portion is reclassified to
            :pyattr:`short_term_borrowings` — see :pyattr:`total_assets`
            and :pyattr:`total_liabilities` (Issue #496).
        """
        return self.ledger.get_balance(AccountName.CASH)

    @property
    def short_term_borrowings(self) -> Decimal:
        """Short-term borrowings from working capital facility per ASC 470-10 (Issue #496).

        Per ASC 210-10-45-1, a negative cash balance represents an obligation
        to the bank and must be classified as a current liability.  This
        property returns the reclassified overdraft plus any explicit
        short-term borrowing balance in the ledger.

        Returns:
            Total short-term borrowings as a non-negative Decimal.
        """
        explicit = self.ledger.get_balance(AccountName.SHORT_TERM_BORROWINGS)
        overdraft = max(-self.cash, to_decimal(0))
        return explicit + overdraft

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
    def insurance_receivables(self) -> Decimal:
        """Insurance receivables per ASC 310-10-45 (Issue #814).

        Represents amounts due from insurers for claims that exceed the
        deductible. Classified as a current asset on the balance sheet.

        Returns:
            Current insurance receivables balance from the ledger.
        """
        return self.ledger.get_balance(AccountName.INSURANCE_RECEIVABLES)

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
    def dta_valuation_allowance(self) -> Decimal:
        """Valuation allowance against DTA per ASC 740-10-30-5 (Issue #464).

        This is a contra-asset that reduces the gross DTA when realization
        is not more likely than not. Stored as a credit balance in the ledger
        (contra-asset), returned here as a positive value.

        Returns:
            Current valuation allowance balance as a positive Decimal.
        """
        return abs(self.ledger.get_balance(AccountName.DTA_VALUATION_ALLOWANCE))

    @property
    def deferred_tax_liability(self) -> Decimal:
        """Deferred tax liability from depreciation timing differences per ASC 740 (Issue #367).

        Returns:
            Current deferred tax liability balance from the ledger.
        """
        return self.ledger.get_balance(AccountName.DEFERRED_TAX_LIABILITY)

    @property
    def net_deferred_tax_asset(self) -> Decimal:
        """Net DTA per ASC 740-10-45-6 (Issue #821).

        DTA and DTL within the same tax jurisdiction are netted for
        balance sheet presentation. If the net position is an asset
        (DTA > DTL after valuation allowance), this returns the net amount.

        Returns:
            Net deferred tax asset (non-negative Decimal). Zero if net
            position is a liability.
        """
        gross_dta = self.deferred_tax_asset - self.dta_valuation_allowance
        return max(gross_dta - self.deferred_tax_liability, to_decimal(0))

    @property
    def net_deferred_tax_liability(self) -> Decimal:
        """Net DTL per ASC 740-10-45-6 (Issue #821).

        DTA and DTL within the same tax jurisdiction are netted for
        balance sheet presentation. If the net position is a liability
        (DTL > DTA after valuation allowance), this returns the net amount.

        Returns:
            Net deferred tax liability (non-negative Decimal). Zero if net
            position is an asset.
        """
        gross_dta = self.deferred_tax_asset - self.dta_valuation_allowance
        return max(self.deferred_tax_liability - gross_dta, to_decimal(0))

    @property
    def total_assets(self) -> Decimal:
        """Calculate total assets from all asset components.

        Total assets include all current and non-current assets following
        the accounting equation: Assets = Liabilities + Equity.

        Returns:
            Decimal: Total assets in dollars, sum of all asset components.
        """
        # Current assets — per ASC 210-10-45-1 (Issue #496), negative cash is
        # reclassified to short-term borrowings so it never reduces total assets.
        reported_cash = max(self.cash, to_decimal(0))
        current = (
            reported_cash
            + self.accounts_receivable
            + self.inventory
            + self.prepaid_insurance
            + self.insurance_receivables  # Issue #814: ASC 310-10-45
        )
        # Non-current assets
        net_ppe = self.gross_ppe - self.accumulated_depreciation
        # Issue #821: Net DTA per ASC 740-10-45-6 (DTA/DTL netted within jurisdiction)
        net_dta = self.net_deferred_tax_asset
        # Total (includes net DTA per ASC 740, Issue #365, #464, #821)
        return current + net_ppe + self.restricted_assets + net_dta

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
        total_accrued_expenses = to_decimal(accrual_items.get("accrued_expenses", to_decimal(0)))

        # Subtract any insurance claims from accrual manager to prevent double-counting
        # ClaimLiability is the authoritative source for claim liabilities
        insurance_claims_in_accrual = sum(
            (
                to_decimal(a.remaining_balance)
                for a in self.accrual_manager.accrued_expenses.get(AccrualType.INSURANCE_CLAIMS, [])
                if not a.is_fully_paid
            ),
            to_decimal(0),
        )
        adjusted_accrued_expenses = total_accrued_expenses - insurance_claims_in_accrual

        # Current liabilities - AccrualManager is single source of truth (issue #238)
        # Short-term borrowings from working capital facility (ASC 470-10, Issue #496)
        current_liabilities = (
            self.accounts_payable + adjusted_accrued_expenses + self.short_term_borrowings
        )

        # Long-term liabilities (claim liabilities) - single source of truth
        claim_liability_total = sum(
            (liability.remaining_amount for liability in self.claim_liabilities), to_decimal(0)
        )

        # Issue #821: Net DTL per ASC 740-10-45-6 (DTA/DTL netted within jurisdiction)
        return current_liabilities + claim_liability_total + self.net_deferred_tax_liability

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
            reduction_ratio = to_decimal(1)
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
        self,
        revenue: Union[Decimal, float],
        dso: float = 45,
        dio: float = 60,
        dpo: float = 30,
        period_revenue: Union[Decimal, float, None] = None,
    ) -> Dict[str, Decimal]:
        """Calculate individual working capital components based on revenue and ratios.

        Uses standard financial ratios to calculate accounts receivable, inventory,
        and accounts payable from annual revenue.

        Issue #1302: Revenue is recognised as Dr AR / Cr REVENUE (accrual basis,
        ASC 606).  This method records *cash collections* on AR
        (Dr CASH / Cr AR) rather than the old Dr AR / Cr CASH delta, which
        double-counted the cash impact when revenue also debited CASH.

        Args:
            revenue: Annual revenue in dollars (used for target computation).
            dso: Days Sales Outstanding. Defaults to 45.
            dio: Days Inventory Outstanding. Defaults to 60.
            dpo: Days Payable Outstanding. Defaults to 30.
            period_revenue: Revenue that will be recorded to AR in this period
                (annual in annual mode, annual/12 in monthly mode).  Defaults
                to ``revenue`` when *None* (backward compatibility / annual).

        Returns:
            Dict containing working capital components.
        """
        # Convert inputs to Decimal
        revenue_decimal = to_decimal(revenue)
        dso_decimal = to_decimal(dso)
        dio_decimal = to_decimal(dio)
        dpo_decimal = to_decimal(dpo)
        days_per_year = to_decimal(365)

        _use_accrual_ar = period_revenue is not None
        if _use_accrual_ar:
            period_revenue_decimal = to_decimal(period_revenue)
        else:
            period_revenue_decimal = ZERO

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

        if _use_accrual_ar:
            # Issue #1302: Record cash collections on AR instead of the old
            # Dr AR / Cr CASH delta.  Revenue will be recorded as Dr AR, so
            # the WC module handles the collection cycle (Dr CASH / Cr AR).
            #   collections = old_AR + period_revenue − target_AR
            # After both entries: AR = old_AR + period_revenue − collections
            #                       = target_AR
            collections = self.accounts_receivable + period_revenue_decimal - new_ar
            if collections > ZERO:
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.CASH,
                    credit_account=AccountName.ACCOUNTS_RECEIVABLE,
                    amount=collections,
                    transaction_type=TransactionType.COLLECTION,
                    description="Cash collections on accounts receivable",
                    month=self.current_month,
                )
            elif collections < ZERO:
                # Target AR exceeds current AR + period revenue (rapid growth
                # or first period with zero initial AR).
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.ACCOUNTS_RECEIVABLE,
                    credit_account=AccountName.CASH,
                    amount=abs(collections),
                    transaction_type=TransactionType.WORKING_CAPITAL,
                    description="AR buildup exceeds period collections",
                    month=self.current_month,
                )
        else:
            # Legacy path (backward compat): adjust AR to target via delta.
            collections = ZERO
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

        # WORKING CAPITAL FACILITY (ASC 470-10, Issue #496):
        # When working capital changes push the ledger cash negative, record the
        # shortfall as a short-term borrowing (credit-line draw) rather than
        # inflating accounts payable.  The negative cash is reclassified on the
        # balance sheet by the cash/short_term_borrowings properties.
        raw_cash = self.cash
        if raw_cash < ZERO:
            logger.warning(
                f"WORKING CAPITAL FACILITY: Working capital changes pushed cash to "
                f"${raw_cash:,.2f}. Negative balance will be presented as short-term "
                f"borrowings per ASC 470-10 (Issue #496)."
            )

        # Calculate net working capital and cash conversion cycle
        net_working_capital = self.accounts_receivable + self.inventory - self.accounts_payable
        cash_conversion_cycle_days = dso_decimal + dio_decimal - dpo_decimal

        # Calculate cash impact for logging
        if _use_accrual_ar:
            # Issue #1302: collections replace -ar_change
            cash_impact = collections - inventory_change + ap_change
        else:
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
        self,
        net_income: Union[Decimal, float],
        growth_rate: Union[Decimal, float] = 0.0,
        depreciation_expense: Union[Decimal, float, int] = 0,
        period_revenue: Union[Decimal, float, int] = 0,
        cogs_expense: Union[Decimal, float, int] = 0,
        opex_expense: Union[Decimal, float, int] = 0,
    ) -> None:
        """Update balance sheet with retained earnings and dividend distribution.

        Issues #687/#803/#1213: When called from step() with period_revenue,
        uses proper closing entries that close income statement accounts
        (SALES_REVENUE, DEPRECIATION_EXPENSE, COST_OF_GOODS_SOLD,
        OPERATING_EXPENSES) to RETAINED_EARNINGS and compute the cash outflow
        directly from net_income, avoiding the period_cash_expenses
        decomposition that double-counted depreciation embedded in
        base_operating_margin.

        When called without period_revenue (backward compatibility, tests),
        falls back to the legacy Dr CASH / Cr RETAINED_EARNINGS approach
        with depreciation add-back (Issue #637).

        Args:
            net_income: Net income for the period in dollars.
            growth_rate: Revenue growth rate parameter (currently unused).
            depreciation_expense: Period depreciation expense. Defaults to 0.
            period_revenue: Revenue recorded in step() via Dr AR / Cr SALES_REVENUE
                (Issue #1302).  Defaults to 0 (legacy mode).
            cogs_expense: COGS recorded in step() via Dr COGS / Cr CASH (Issue #1326).
                Defaults to 0 (backward compat).
            opex_expense: OPEX recorded in step() via Dr OPEX / Cr CASH (Issue #1326).
                Defaults to 0 (backward compat).
        """
        # Convert inputs to Decimal
        net_income_decimal = to_decimal(net_income)
        depreciation_addback = to_decimal(depreciation_expense)
        period_revenue_decimal = to_decimal(period_revenue)
        cogs_expense_decimal = to_decimal(cogs_expense)
        opex_expense_decimal = to_decimal(opex_expense)

        # Issue #803: Determine if we should use proper closing entries (new path)
        # or legacy Dr CASH / Cr RE behavior (backward compatibility).
        use_closing_entries = period_revenue_decimal > ZERO

        # Validation: retention ratio should be applied to net income
        assert 0 <= self.retention_ratio <= 1, f"Invalid retention ratio: {self.retention_ratio}"

        # Calculate retained earnings and dividends
        # Issue #1304: ASC 505-20-45 — dividends can only be declared from
        # positive earnings.  When net income is negative, 100 % of the loss
        # must be absorbed by retained earnings; no negative-dividend offset.
        retention_decimal = to_decimal(self.retention_ratio)
        if net_income_decimal > ZERO:
            retained_earnings = net_income_decimal * retention_decimal
            dividends = net_income_decimal * (to_decimal(1) - retention_decimal)
        else:
            retained_earnings = net_income_decimal  # full loss absorbed
            dividends = ZERO

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
                # Issue #1297: Close ALL temporary accounts even on insolvency
                if use_closing_entries:
                    self._record_closing_entries(net_income_decimal)
                self._last_dividends_paid = to_decimal(0)
                return

            # Issue #637: Depreciation did not consume cash, so actual cash
            # needed to cover the loss is reduced by the depreciation add-back.
            cash_consumed = max(loss_amount - depreciation_addback, to_decimal(0))

            # Check 2: LIQUIDITY CHECK - must have cash to pay loss
            # When using closing entries, cash is already updated by step() income
            # entries so available_cash reflects operating cash flows.
            if not use_closing_entries and cash_consumed > available_cash:
                logger.error(
                    f"LIQUIDITY CRISIS → INSOLVENCY: Cash drain ${cash_consumed:,.2f} "
                    f"(loss ${loss_amount:,.2f} - depreciation add-back "
                    f"${depreciation_addback:,.2f}) exceeds available cash "
                    f"${available_cash:,.2f}. Equity=${current_equity:,.2f}. "
                    f"Company cannot meet obligations despite positive book equity."
                )
                self._last_dividends_paid = to_decimal(0)
                self.handle_insolvency()
                return

            # Check 3: Would paying the loss trigger equity insolvency?
            if use_closing_entries:
                # With closing entries, equity change = closing entry effect on RE.
                # The income entries already affected cash; closing entries affect RE.
                equity_after_loss = current_equity - loss_amount
            else:
                # Legacy: Equity impact accounts for depreciation add-back to cash (Issue #637)
                equity_after_loss = current_equity - loss_amount + depreciation_addback
            if equity_after_loss <= tolerance:
                logger.error(
                    f"EQUITY INSOLVENCY: Loss ${loss_amount:,.2f} would push equity to "
                    f"${equity_after_loss:,.2f}, below threshold ${tolerance:,.2f}. "
                    f"Current equity=${current_equity:,.2f}. Triggering insolvency."
                )
                if use_closing_entries:
                    # Issue #803/#1297: Close ALL income statement accounts to RE
                    self._record_closing_entries(net_income_decimal)
                else:
                    # Legacy: Dr RE / Cr CASH
                    self.ledger.record_double_entry(
                        date=self.current_year,
                        debit_account=AccountName.RETAINED_EARNINGS,
                        credit_account=AccountName.CASH,
                        amount=loss_amount,
                        transaction_type=TransactionType.EXPENSE,
                        description=f"Year {self.current_year} operating loss (pre-insolvency)",
                        month=self.current_month,
                    )
                    if depreciation_addback > ZERO:
                        self.ledger.record_double_entry(
                            date=self.current_year,
                            debit_account=AccountName.CASH,
                            credit_account=AccountName.DEPRECIATION_EXPENSE,
                            amount=depreciation_addback,
                            transaction_type=TransactionType.DEPRECIATION,
                            description=f"Year {self.current_year} depreciation add-back (non-cash, Issue #637)",
                            month=self.current_month,
                        )
                self._last_dividends_paid = to_decimal(0)
                self.handle_insolvency()
                return

            # All checks passed - absorb the full loss
            if use_closing_entries:
                # Issue #803/#1297: Close ALL income statement accounts to RE
                self._record_closing_entries(net_income_decimal)
            else:
                # Legacy: Dr RE / Cr CASH
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.RETAINED_EARNINGS,
                    credit_account=AccountName.CASH,
                    amount=loss_amount,
                    transaction_type=TransactionType.EXPENSE,
                    description=f"Year {self.current_year} operating loss",
                    month=self.current_month,
                )
                if depreciation_addback > ZERO:
                    self.ledger.record_double_entry(
                        date=self.current_year,
                        debit_account=AccountName.CASH,
                        credit_account=AccountName.DEPRECIATION_EXPENSE,
                        amount=depreciation_addback,
                        transaction_type=TransactionType.DEPRECIATION,
                        description=f"Year {self.current_year} depreciation add-back (non-cash, Issue #637)",
                        month=self.current_month,
                    )

            self._last_dividends_paid = to_decimal(0)
            logger.info(
                f"Absorbed loss: ${loss_amount:,.2f}. "
                f"Remaining cash: ${self.cash:,.2f}, Remaining equity: ${self.equity:,.2f}"
            )
        else:
            # Positive retained earnings - check cash constraints for dividends
            if use_closing_entries:
                # Issue #803: Cash already reflects operating cash flows from step().
                # Projected cash for dividend check uses current cash balance.
                projected_cash = self.cash
            else:
                # Legacy: Include depreciation add-back in projected cash since
                # depreciation reduced net income but did not consume cash (Issue #637).
                projected_cash = self.cash + retained_earnings + depreciation_addback

            if projected_cash <= ZERO:
                actual_dividends = to_decimal(0)
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
                additional_retained = to_decimal(0)

            self._last_dividends_paid = actual_dividends

            total_retained = retained_earnings + additional_retained

            if use_closing_entries:
                # Issue #803/#1297: Close ALL income statement accounts to RE
                self._record_closing_entries(net_income_decimal)
            else:
                # Legacy: Issue #683: Record full net income to retained earnings
                if net_income_decimal > ZERO:
                    self.ledger.record_double_entry(
                        date=self.current_year,
                        debit_account=AccountName.CASH,
                        credit_account=AccountName.RETAINED_EARNINGS,
                        amount=net_income_decimal,
                        transaction_type=TransactionType.RETAINED_EARNINGS,
                        description=f"Year {self.current_year} retained earnings",
                        month=self.current_month,
                    )
                # Legacy: Issue #637: Add back non-cash depreciation to cash
                if depreciation_addback > ZERO:
                    self.ledger.record_double_entry(
                        date=self.current_year,
                        debit_account=AccountName.CASH,
                        credit_account=AccountName.DEPRECIATION_EXPENSE,
                        amount=depreciation_addback,
                        transaction_type=TransactionType.DEPRECIATION,
                        description=f"Year {self.current_year} depreciation add-back (non-cash, Issue #637)",
                        month=self.current_month,
                    )

            # Dividends are actual cash outflows regardless of closing entry mode
            if actual_dividends > ZERO:
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.RETAINED_EARNINGS,
                    credit_account=AccountName.CASH,
                    amount=actual_dividends,
                    transaction_type=TransactionType.DIVIDEND,
                    description=f"Year {self.current_year} dividend distribution",
                    month=self.current_month,
                )

        logger.info(
            f"Balance sheet updated: Assets=${self.total_assets:,.2f}, Equity=${self.equity:,.2f}"
        )
        if self._last_dividends_paid > ZERO:
            logger.info(f"Dividends paid: ${self._last_dividends_paid:,.2f}")

    def _record_closing_entries(
        self,
        net_income: Decimal,
    ) -> None:
        """Record period-end closing entries per GAAP (ASC 205-10).

        Issue #1297: Close ALL temporary accounts (revenue and expense) to
        RETAINED_EARNINGS at period end.  Previous implementation only closed
        SALES_REVENUE, DEPRECIATION_EXPENSE, COGS, and OPEX; remaining
        temporary accounts (INSURANCE_EXPENSE, TAX_EXPENSE, INSURANCE_LOSS,
        LAE_EXPENSE, RESERVE_DEVELOPMENT, etc.) were left unclosed, violating
        the GAAP closing process and causing cumulative balances.

        The method reads each temporary account's current balance from the
        ledger and closes it individually, providing a full audit trail.

        After these entries:
        - RE changes by net_income
        - All temporary accounts are zeroed
        - CASH is adjusted for the residual between ledger net and
          computed net_income (indirect-method OCF, ASC 230-10-28)

        Args:
            net_income: Full net income for the period (after all deductions).
        """
        total_revenue_closed = ZERO
        total_expense_closed = ZERO

        for account, acct_type in CHART_OF_ACCOUNTS.items():
            if acct_type == AccountType.REVENUE:
                balance = self.ledger.get_balance(account)
                if balance > ZERO:
                    # Normal revenue (credit balance): Dr REVENUE / Cr RE
                    self.ledger.record_double_entry(
                        date=self.current_year,
                        debit_account=account,
                        credit_account=AccountName.RETAINED_EARNINGS,
                        amount=balance,
                        transaction_type=TransactionType.RETAINED_EARNINGS,
                        description=(
                            f"Year {self.current_year} close "
                            f"{account.value} to retained earnings"
                        ),
                        month=self.current_month,
                    )
                    total_revenue_closed += balance
                elif balance < ZERO:
                    # Unusual debit balance on revenue account: Dr RE / Cr REVENUE
                    self.ledger.record_double_entry(
                        date=self.current_year,
                        debit_account=AccountName.RETAINED_EARNINGS,
                        credit_account=account,
                        amount=abs(balance),
                        transaction_type=TransactionType.RETAINED_EARNINGS,
                        description=(
                            f"Year {self.current_year} close "
                            f"{account.value} (debit balance) to retained earnings"
                        ),
                        month=self.current_month,
                    )
                    total_revenue_closed += balance  # negative, reduces total

            elif acct_type == AccountType.EXPENSE:
                balance = self.ledger.get_balance(account)
                if balance > ZERO:
                    # Normal expense (debit balance): Dr RE / Cr EXPENSE
                    self.ledger.record_double_entry(
                        date=self.current_year,
                        debit_account=AccountName.RETAINED_EARNINGS,
                        credit_account=account,
                        amount=balance,
                        transaction_type=TransactionType.RETAINED_EARNINGS,
                        description=(
                            f"Year {self.current_year} close "
                            f"{account.value} to retained earnings"
                        ),
                        month=self.current_month,
                    )
                    total_expense_closed += balance
                elif balance < ZERO:
                    # Unusual credit balance on expense (e.g., favorable
                    # reserve development): Dr EXPENSE / Cr RE
                    self.ledger.record_double_entry(
                        date=self.current_year,
                        debit_account=account,
                        credit_account=AccountName.RETAINED_EARNINGS,
                        amount=abs(balance),
                        transaction_type=TransactionType.RETAINED_EARNINGS,
                        description=(
                            f"Year {self.current_year} close "
                            f"{account.value} (credit balance) to retained earnings"
                        ),
                        month=self.current_month,
                    )
                    total_expense_closed += balance  # negative, reduces total

        # Net income per ledger (should equal computed net_income when all
        # income/expense flows are journaled; may differ when some costs
        # are only captured in the net_income calculation, not as entries).
        ledger_net = total_revenue_closed - total_expense_closed

        # Issue #1213/#1302: Residual cash adjustment.
        # Closing entries only shuffle between temporary accounts and RE;
        # they do not touch CASH.  The residual reconciles any gap between
        # ledger_net (from journal entries) and the computed net_income
        # (from the income model), ensuring RE changes by exactly net_income
        # and CASH adjusts accordingly (indirect-method OCF, ASC 230-10-28).
        cash_outflow = ledger_net - net_income

        if cash_outflow > ZERO:
            # Ledger tracked more income than computed — remove excess cash
            # Dr RETAINED_EARNINGS / Cr CASH
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.RETAINED_EARNINGS,
                credit_account=AccountName.CASH,
                amount=cash_outflow,
                transaction_type=TransactionType.RETAINED_EARNINGS,
                description=(
                    f"Year {self.current_year} close residual cash outflow " f"to retained earnings"
                ),
                month=self.current_month,
            )
        elif cash_outflow < ZERO:
            # Ledger tracked less income than computed — add cash
            # Dr CASH / Cr RETAINED_EARNINGS
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.CASH,
                credit_account=AccountName.RETAINED_EARNINGS,
                amount=abs(cash_outflow),
                transaction_type=TransactionType.RETAINED_EARNINGS,
                description=(
                    f"Year {self.current_year} close net cash inflow " f"to retained earnings"
                ),
                month=self.current_month,
            )

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
        return to_decimal(0)

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
            return to_decimal(0)

        # Cannot spend more cash than available (Issue #543)
        available_cash = self.cash
        if available_cash <= ZERO:
            logger.warning(
                f"Skipping capex: no cash available "
                f"(cash={available_cash:,.2f}, requested={amount:,.2f})"
            )
            return to_decimal(0)

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

        # Track vintage cohort for accurate DTL calculation (Issue #1321)
        if hasattr(self, "_ppe_cohorts"):
            from ergodic_insurance.manufacturer import _PPECohort

            self._ppe_cohorts.append(_PPECohort(amount=actual_capex))

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

        base_growth = float(to_decimal(1) + rate)  # Boundary: float for StochasticProcess

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
