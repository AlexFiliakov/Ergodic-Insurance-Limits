# mypy: disable-error-code="attr-defined, has-type, no-any-return"
"""Claim processing mixin for WidgetManufacturer.

This module contains the ClaimProcessingMixin class, extracted from manufacturer.py
as part of the decomposition refactor (Issue #305). It provides insurance recording,
claim processing, payments, recovery, and accrual methods.
"""

from decimal import Decimal
import logging
import random
from typing import TYPE_CHECKING, Dict, List, Optional, Union

try:
    from ergodic_insurance.claim_development import ClaimDevelopment
    from ergodic_insurance.claim_liability import ClaimLiability
    from ergodic_insurance.decimal_utils import ONE, ZERO, to_decimal
    from ergodic_insurance.ledger import AccountName, TransactionType
except ImportError:
    try:
        from .claim_development import ClaimDevelopment
        from .claim_liability import ClaimLiability
        from .decimal_utils import ONE, ZERO, to_decimal
        from .ledger import AccountName, TransactionType
    except ImportError:
        from claim_development import ClaimDevelopment  # type: ignore[no-redef]
        from claim_liability import ClaimLiability  # type: ignore[no-redef]
        from decimal_utils import ONE, ZERO, to_decimal  # type: ignore[no-redef]
        from ledger import AccountName, TransactionType  # type: ignore[no-redef]

if TYPE_CHECKING:
    from ergodic_insurance.claim_development import Claim

logger = logging.getLogger(__name__)


class ClaimProcessingMixin:
    """Mixin providing insurance claim processing methods.

    This mixin expects the host class to have:
        - self.ledger: Ledger instance
        - self.config: ManufacturerConfig instance
        - self.current_year: int
        - self.current_month: int
        - self.insurance_accounting: InsuranceAccounting instance
        - self.claim_liabilities: List[ClaimLiability]
        - self.period_insurance_premiums: Decimal
        - self.period_insurance_losses: Decimal
        - self.period_insurance_lae: Decimal (Issue #468)
        - self._original_prepaid_premium: Decimal
        - self.is_ruined: bool
        - Balance sheet properties from BalanceSheetMixin
        - self.handle_insolvency(): method from SolvencyMixin
        - self.check_solvency(): method from SolvencyMixin
        - self.calculate_revenue(): method from IncomeCalculationMixin
    """

    def record_insurance_premium(
        self, premium_amount: Union[Decimal, float], is_annual: bool = False
    ) -> None:
        """Record insurance premium payment with proper GAAP prepaid asset treatment.

        Args:
            premium_amount (float): Premium amount paid in the current period.
            is_annual (bool): Whether this is an annual premium payment (default False).
        """
        premium_decimal = to_decimal(premium_amount)
        if premium_decimal > ZERO:
            if is_annual:
                # COMPULSORY INSURANCE CHECK
                if self.cash < premium_decimal:
                    logger.error(
                        f"INSOLVENCY: Cannot afford compulsory annual insurance premium. "
                        f"Required: ${premium_decimal:,.2f}, Available cash: ${self.cash:,.2f}. "
                        f"Company cannot operate without insurance."
                    )
                    self.handle_insolvency()
                    return

                result = self.insurance_accounting.pay_annual_premium(premium_decimal)

                cash_outflow = result["cash_outflow"]
                if cash_outflow > ZERO:
                    self.ledger.record_double_entry(
                        date=self.current_year,
                        debit_account=AccountName.PREPAID_INSURANCE,
                        credit_account=AccountName.CASH,
                        amount=cash_outflow,
                        transaction_type=TransactionType.INSURANCE_PREMIUM,
                        description="Annual insurance premium payment",
                        month=self.current_month,
                    )

                self._original_prepaid_premium = premium_decimal

                logger.info(f"Paid annual insurance premium: ${premium_decimal:,.2f}")
                logger.info(f"Monthly expense will be: ${result['monthly_expense']:,.2f}")
            else:
                self.period_insurance_premiums += premium_decimal
                logger.info(f"Recorded insurance premium expense: ${premium_decimal:,.2f}")
                logger.debug(f"Period premiums total: ${self.period_insurance_premiums:,.2f}")

    def record_insurance_loss(self, loss_amount: Union[Decimal, float]) -> None:
        """Record insurance loss (deductible/retention) for tax deduction tracking.

        Args:
            loss_amount: Loss amount paid by company in the current period.
        """
        loss_decimal = to_decimal(loss_amount)
        if loss_decimal > ZERO:
            self.period_insurance_losses += loss_decimal
            logger.debug(f"Recorded insurance loss: ${loss_decimal:,.2f}")
            logger.debug(f"Period losses total: ${self.period_insurance_losses:,.2f}")

    def reset_period_insurance_costs(self) -> None:
        """Reset period insurance cost tracking for new period."""
        self.period_insurance_premiums = ZERO
        self.period_insurance_losses = ZERO
        self.period_insurance_lae = ZERO
        self.period_adverse_development = ZERO
        self.period_favorable_development = ZERO

    @property
    def total_claim_liabilities(self) -> Decimal:
        """Calculate total outstanding claim liabilities.

        Returns:
            Decimal: Total outstanding liability in dollars.
        """
        return sum((claim.remaining_amount for claim in self.claim_liabilities), ZERO)

    def record_prepaid_insurance(self, annual_premium: Union[Decimal, float]) -> None:
        """Record annual insurance premium payment as prepaid expense.

        Args:
            annual_premium (float): Annual insurance premium paid in advance.
        """
        annual_premium_decimal = to_decimal(annual_premium)
        if annual_premium_decimal > ZERO:
            if self.cash < annual_premium_decimal:
                logger.error(
                    f"INSOLVENCY: Cannot afford compulsory annual insurance premium. "
                    f"Required: ${annual_premium_decimal:,.2f}, Available cash: ${self.cash:,.2f}. "
                    f"Company cannot operate without insurance."
                )
                self.handle_insolvency()
                return

            result = self.insurance_accounting.pay_annual_premium(annual_premium_decimal)

            if result["cash_outflow"] > ZERO:
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.PREPAID_INSURANCE,
                    credit_account=AccountName.CASH,
                    amount=result["cash_outflow"],
                    transaction_type=TransactionType.INSURANCE_PREMIUM,
                    description="Annual insurance premium payment",
                    month=self.current_month,
                )

            logger.info(f"Recorded prepaid insurance: ${annual_premium_decimal:,.2f}")

    def amortize_prepaid_insurance(self, months: int = 1) -> Decimal:
        """Amortize prepaid insurance over time using GAAP straight-line method.

        Args:
            months (int): Number of months to amortize. Defaults to 1.

        Returns:
            Decimal: Amount amortized (insurance expense for the period).
        """
        total_amortized = ZERO

        for _ in range(months):
            if self.prepaid_insurance > ZERO:
                result = self.insurance_accounting.record_monthly_expense()

                self.period_insurance_premiums += result["insurance_expense"]
                total_amortized += result["insurance_expense"]

                if result["insurance_expense"] > ZERO:
                    self.ledger.record_double_entry(
                        date=self.current_year,
                        debit_account=AccountName.INSURANCE_EXPENSE,
                        credit_account=AccountName.PREPAID_INSURANCE,
                        amount=result["insurance_expense"],
                        transaction_type=TransactionType.EXPENSE,
                        description="Insurance premium amortization",
                        month=self.current_month,
                    )

                logger.debug(
                    f"Month {self.insurance_accounting.current_month}: "
                    f"Expense ${result['insurance_expense']:,.2f}, "
                    f"Remaining prepaid ${result['remaining_prepaid']:,.2f}"
                )

        return total_amortized

    def receive_insurance_recovery(
        self, amount: Union[Decimal, float], claim_id: Optional[str] = None
    ) -> Dict[str, Decimal]:
        """Receive payment from insurance for a claim recovery.

        Args:
            amount (float): Amount received from insurance company.
            claim_id (Optional[str]): Specific claim ID for the recovery.

        Returns:
            Dictionary with payment details.
        """
        if amount <= 0:
            return {
                "cash_received": ZERO,
                "receivable_reduction": ZERO,
                "remaining_receivables": ZERO,
            }

        result = self.insurance_accounting.receive_recovery_payment(amount, claim_id)

        if result["cash_received"] > ZERO:
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.CASH,
                credit_account=AccountName.INSURANCE_RECEIVABLES,
                amount=result["cash_received"],
                transaction_type=TransactionType.INSURANCE_CLAIM,
                description="Insurance recovery received",
                month=self.current_month,
            )

        logger.info(f"Received insurance recovery: ${result['cash_received']:,.2f}")
        logger.debug(f"Remaining receivables: ${result['remaining_receivables']:,.2f}")

        return result

    def process_insurance_claim(
        self,
        claim_amount: Union[Decimal, float],
        deductible_amount: Union[Decimal, float] = ZERO,
        insurance_limit: Union[Decimal, float, None] = None,
        insurance_recovery: Optional[Union[Decimal, float]] = None,
        record_period_loss: bool = False,
    ) -> tuple[Decimal, Decimal]:
        """Process an insurance claim with deductible and limit, setting up collateral.

        Args:
            claim_amount: Total amount of the loss/claim in dollars.
            deductible_amount: Amount company must pay before insurance kicks in.
            insurance_limit: Maximum amount insurance will pay per claim.
            insurance_recovery: Pre-calculated insurance recovery amount.
            record_period_loss: If True, also call record_insurance_loss().

        Returns:
            tuple[Decimal, Decimal]: Tuple of (company_payment, insurance_payment).
        """
        # Convert all inputs to Decimal
        claim = to_decimal(claim_amount)
        deductible = to_decimal(deductible_amount)
        limit = to_decimal(insurance_limit) if insurance_limit is not None else to_decimal(1e18)

        # Handle new style parameters if provided
        if insurance_recovery is not None:
            insurance_payment = to_decimal(insurance_recovery)
            company_payment = claim - insurance_payment
        else:
            if claim <= deductible:
                company_payment = claim
                insurance_payment = ZERO
            else:
                company_payment = deductible
                insurance_payment = min(claim - deductible, limit)
                if claim > deductible + limit:
                    company_payment += claim - deductible - limit

        # Compute LAE on company portion (Issue #468, ASC 944-40)
        lae_ratio = to_decimal(getattr(self.config, "lae_ratio", 0.12))

        # Company payment is collateralized and paid over time
        if company_payment > ZERO:
            lae_on_company = company_payment * lae_ratio

            current_equity = self.equity
            available_cash = self.cash
            # Cap at equity / (1 + lae_ratio) so total liability including LAE
            # does not exceed equity (limited liability, Issue #468)
            equity_cap = current_equity / (ONE + lae_ratio) if lae_ratio > ZERO else current_equity
            max_payable: Decimal = (
                min(company_payment, equity_cap, available_cash) if current_equity > ZERO else ZERO
            )
            unpayable_amount = company_payment - max_payable

            if max_payable > ZERO:
                lae_on_payable = max_payable * lae_ratio

                self._record_asset_transfer(
                    from_account=AccountName.CASH,
                    to_account=AccountName.RESTRICTED_CASH,
                    amount=max_payable,
                    description="Cash to restricted for insurance claim collateral",
                )

                # Claim liability includes indemnity + LAE
                claim_liability = ClaimLiability(
                    original_amount=max_payable + lae_on_payable,
                    remaining_amount=max_payable + lae_on_payable,
                    year_incurred=self.current_year,
                    is_insured=True,
                )
                self._apply_reserve_noise(claim_liability)
                self.claim_liabilities.append(claim_liability)

                # Record indemnity portion
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.INSURANCE_LOSS,
                    credit_account=AccountName.CLAIM_LIABILITIES,
                    amount=max_payable,
                    transaction_type=TransactionType.INSURANCE_CLAIM,
                    description="Recognize insured claim liability (company portion)",
                )

                # Record LAE portion separately (Issue #468)
                if lae_on_payable > ZERO:
                    self.ledger.record_double_entry(
                        date=self.current_year,
                        debit_account=AccountName.LAE_EXPENSE,
                        credit_account=AccountName.CLAIM_LIABILITIES,
                        amount=lae_on_payable,
                        transaction_type=TransactionType.INSURANCE_CLAIM,
                        description="LAE on insured claim (ALAE+ULAE per ASC 944-40)",
                    )
                    self.period_insurance_lae += lae_on_payable

                logger.info(
                    f"Company portion: ${max_payable:,.2f} + LAE ${lae_on_payable:,.2f} - collateralized with payment schedule"
                )
                logger.info(
                    f"Posted ${max_payable:,.2f} letter of credit as collateral for company portion"
                )

            # Handle unpayable portion
            if unpayable_amount > ZERO:
                lae_on_unpayable = unpayable_amount * lae_ratio
                current_equity_after_collateral = self.equity
                # Cap at equity / (1 + lae_ratio) for LAE headroom (Issue #468)
                equity_cap_after = (
                    current_equity_after_collateral / (ONE + lae_ratio)
                    if lae_ratio > ZERO
                    else current_equity_after_collateral
                )
                max_liability: Decimal = (
                    min(unpayable_amount, equity_cap_after)
                    if current_equity_after_collateral > ZERO
                    else ZERO
                )

                if max_liability > ZERO:
                    lae_on_max_liability = max_liability * lae_ratio
                    unpayable_claim = ClaimLiability(
                        original_amount=max_liability + lae_on_max_liability,
                        remaining_amount=max_liability + lae_on_max_liability,
                        year_incurred=self.current_year,
                        is_insured=False,
                    )
                    self._apply_reserve_noise(unpayable_claim)
                    self.claim_liabilities.append(unpayable_claim)

                    # Record indemnity portion
                    self.ledger.record_double_entry(
                        date=self.current_year,
                        debit_account=AccountName.INSURANCE_LOSS,
                        credit_account=AccountName.CLAIM_LIABILITIES,
                        amount=max_liability,
                        transaction_type=TransactionType.INSURANCE_CLAIM,
                        description="Recognize unpayable claim liability",
                    )

                    # Record LAE portion separately (Issue #468)
                    if lae_on_max_liability > ZERO:
                        self.ledger.record_double_entry(
                            date=self.current_year,
                            debit_account=AccountName.LAE_EXPENSE,
                            credit_account=AccountName.CLAIM_LIABILITIES,
                            amount=lae_on_max_liability,
                            transaction_type=TransactionType.INSURANCE_CLAIM,
                            description="LAE on unpayable claim portion (ALAE+ULAE per ASC 944-40)",
                        )
                        self.period_insurance_lae += lae_on_max_liability

                    logger.warning(
                        f"LIMITED LIABILITY: Company payment capped at ${max_payable:,.2f} (cash/equity). "
                        f"Additional liability recorded: ${max_liability:,.2f}"
                    )

                truly_unpayable = unpayable_amount - max_liability
                if truly_unpayable > ZERO:
                    logger.warning(
                        f"LIMITED LIABILITY: Cannot record ${truly_unpayable:,.2f} as liability "
                        f"(would violate limited liability). Company is insolvent."
                    )

                if self.equity <= ZERO:
                    self.check_solvency()

        # Insurance payment creates a receivable
        if insurance_payment > ZERO:
            claim_id = f"CLAIM_{self.current_year}_{len(self.claim_liabilities)}"
            self.insurance_accounting.record_claim_recovery(
                recovery_amount=insurance_payment, claim_id=claim_id, year=self.current_year
            )
            # Record receivable in ledger: Dr INSURANCE_RECEIVABLES / Cr INSURANCE_LOSS
            # per ASC 410-30 — recognize receivable at claim inception (Issue #625)
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.INSURANCE_RECEIVABLES,
                credit_account=AccountName.INSURANCE_LOSS,
                amount=insurance_payment,
                transaction_type=TransactionType.INSURANCE_CLAIM,
                description="Insurance receivable for claim recovery",
                month=self.current_month,
            )
            logger.info(f"Insurance covering ${insurance_payment:,.2f} - recorded as receivable")

        # Optionally record the loss in the income statement
        if record_period_loss and company_payment > ZERO:
            self.record_insurance_loss(company_payment)

        logger.info(
            f"Total claim: ${claim_amount:,.2f} (Company: ${company_payment:,.2f}, Insurance: ${insurance_payment:,.2f})"
        )

        return company_payment, insurance_payment

    def process_uninsured_claim(
        self, claim_amount: Union[Decimal, float], immediate_payment: bool = False
    ) -> Decimal:
        """Process an uninsured claim paid by company over time without collateral.

        Args:
            claim_amount: Total amount of the claim in dollars.
            immediate_payment: If True, pays entire amount immediately.

        Returns:
            Decimal: The claim amount processed.
        """
        claim = to_decimal(claim_amount)
        if claim <= ZERO:
            return ZERO

        if immediate_payment:
            equity_before_payment = self.equity
            max_payable: Decimal = (
                min(claim, equity_before_payment) if equity_before_payment > ZERO else ZERO
            )

            cash_payment: Decimal = min(max_payable, self.cash)
            remaining_to_pay: Decimal = max_payable - cash_payment

            if remaining_to_pay > ZERO:
                liquid_assets = self.cash + self.accounts_receivable + self.inventory
                if liquid_assets > ZERO:
                    actual_payment: Decimal = min(max_payable, liquid_assets)
                    self._record_liquid_asset_reduction(
                        total_reduction=actual_payment,
                        description="Liquid asset reduction for uninsured claim payment",
                    )
                else:
                    actual_payment = ZERO
            else:
                actual_payment = cash_payment
                if cash_payment > ZERO:
                    self._record_liquidation(
                        amount=cash_payment,
                        description="Cash payment for uninsured claim",
                    )

            self.period_insurance_losses += actual_payment

            # Track LAE on immediate-payment uninsured claim (Issue #468)
            # LAE is tracked for the income statement (like period_insurance_losses)
            # but not recorded as a balance-sheet liability for immediate payments.
            lae_ratio_imm = to_decimal(getattr(self.config, "lae_ratio", 0.12))
            lae_on_immediate = actual_payment * lae_ratio_imm
            if lae_on_immediate > ZERO:
                self.period_insurance_lae += lae_on_immediate

            shortfall = claim - actual_payment
            if shortfall > ZERO:
                current_equity_after_payment = self.equity
                # Cap at equity / (1 + lae_ratio) for LAE headroom (Issue #468)
                equity_cap_shortfall = (
                    current_equity_after_payment / (ONE + lae_ratio_imm)
                    if lae_ratio_imm > ZERO
                    else current_equity_after_payment
                )
                max_liability: Decimal = (
                    min(shortfall, equity_cap_shortfall)
                    if current_equity_after_payment > ZERO
                    else ZERO
                )

                if max_liability > ZERO:
                    lae_on_shortfall = max_liability * lae_ratio_imm
                    claim_liability = ClaimLiability(
                        original_amount=max_liability + lae_on_shortfall,
                        remaining_amount=max_liability + lae_on_shortfall,
                        year_incurred=self.current_year,
                        is_insured=False,
                    )
                    self._apply_reserve_noise(claim_liability)
                    self.claim_liabilities.append(claim_liability)

                    # Record indemnity portion
                    self.ledger.record_double_entry(
                        date=self.current_year,
                        debit_account=AccountName.INSURANCE_LOSS,
                        credit_account=AccountName.CLAIM_LIABILITIES,
                        amount=max_liability,
                        transaction_type=TransactionType.INSURANCE_CLAIM,
                        description="Recognize uninsured claim liability (shortfall)",
                    )

                    # Record LAE portion (Issue #468)
                    if lae_on_shortfall > ZERO:
                        self.ledger.record_double_entry(
                            date=self.current_year,
                            debit_account=AccountName.LAE_EXPENSE,
                            credit_account=AccountName.CLAIM_LIABILITIES,
                            amount=lae_on_shortfall,
                            transaction_type=TransactionType.INSURANCE_CLAIM,
                            description="LAE on uninsured shortfall (ALAE+ULAE per ASC 944-40)",
                        )
                        self.period_insurance_lae += lae_on_shortfall

                    logger.info(
                        f"LIMITED LIABILITY: Immediate payment ${actual_payment:,.2f}, "
                        f"created liability for ${claim_liability.original_amount:,.2f} (total claim: ${claim:,.2f})"
                    )

                truly_unpayable = shortfall - max_liability
                if truly_unpayable > ZERO:
                    logger.warning(
                        f"LIMITED LIABILITY: Cannot record ${truly_unpayable:,.2f} of ${claim:,.2f} claim as liability "
                        f"(would violate limited liability). "
                        f"Paid ${actual_payment:,.2f}, liability ${max_liability:,.2f}, shortfall ${truly_unpayable:,.2f}."
                    )

                if self.equity <= ZERO:
                    self.check_solvency()
            else:
                logger.info(f"Paid uninsured claim immediately: ${actual_payment:,.2f}")
            return claim

        # Create liability without collateral for payment over time
        lae_ratio = to_decimal(getattr(self.config, "lae_ratio", 0.12))
        current_equity = self.equity
        # Cap at equity / (1 + lae_ratio) so total liability including LAE
        # does not exceed equity (limited liability, Issue #468)
        equity_cap_deferred = (
            current_equity / (ONE + lae_ratio) if lae_ratio > ZERO else current_equity
        )
        deferred_max_liability: Decimal = (
            min(claim, equity_cap_deferred) if current_equity > ZERO else ZERO
        )

        if deferred_max_liability > ZERO:
            lae_on_deferred = deferred_max_liability * lae_ratio
            claim_liability = ClaimLiability(
                original_amount=deferred_max_liability + lae_on_deferred,
                remaining_amount=deferred_max_liability + lae_on_deferred,
                year_incurred=self.current_year,
                is_insured=False,
            )
            self._apply_reserve_noise(claim_liability)
            self.claim_liabilities.append(claim_liability)

            # Record indemnity portion
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.INSURANCE_LOSS,
                credit_account=AccountName.CLAIM_LIABILITIES,
                amount=deferred_max_liability,
                transaction_type=TransactionType.INSURANCE_CLAIM,
                description="Recognize uninsured deferred claim liability",
            )

            # Record LAE portion separately (Issue #468)
            if lae_on_deferred > ZERO:
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.LAE_EXPENSE,
                    credit_account=AccountName.CLAIM_LIABILITIES,
                    amount=lae_on_deferred,
                    transaction_type=TransactionType.INSURANCE_CLAIM,
                    description="LAE on uninsured deferred claim (ALAE+ULAE per ASC 944-40)",
                )
                self.period_insurance_lae += lae_on_deferred

            logger.info(
                f"Created uninsured claim liability: ${claim_liability.original_amount:,.2f} "
                f"(incl. LAE ${lae_on_deferred:,.2f}, no collateral required)"
            )

        unpayable = claim - deferred_max_liability
        if unpayable > ZERO:
            logger.warning(
                f"LIMITED LIABILITY: Cannot record ${unpayable:,.2f} as liability "
                f"(would violate limited liability). Company may become insolvent."
            )
            if self.equity <= ZERO:
                self.check_solvency()

        return claim

    def process_insurance_claim_with_development(
        self,
        claim_amount: Union[Decimal, float],
        deductible: Union[Decimal, float] = ZERO,
        insurance_limit: Union[Decimal, float, None] = None,
        development_pattern: Optional["ClaimDevelopment"] = None,
        claim_type: str = "general_liability",
    ) -> tuple[Decimal, Decimal, Optional["Claim"]]:
        """Process an insurance claim with custom development pattern integration.

        Args:
            claim_amount: Total amount of the loss/claim in dollars.
            deductible: Amount company must pay before insurance coverage begins.
            insurance_limit: Maximum amount insurance will pay per claim.
            development_pattern: Custom actuarial development pattern.
            claim_type: Classification of claim type.

        Returns:
            tuple: (company_payment, insurance_payment, claim_object)
        """
        company_payment, insurance_payment = self.process_insurance_claim(
            claim_amount, deductible, insurance_limit
        )

        claim_object = None
        if development_pattern is not None and insurance_payment > ZERO:
            from .claim_development import Claim

            claim_object = Claim(
                claim_id=f"CL_{self.current_year}_{len(self.claim_liabilities):04d}",
                accident_year=self.current_year,
                reported_year=self.current_year,
                initial_estimate=float(insurance_payment),
                claim_type=claim_type,
                development_pattern=development_pattern,
            )

            logger.info(
                f"Created claim with {development_pattern.pattern_name} development pattern"
            )

        return company_payment, insurance_payment, claim_object

    def pay_claim_liabilities(self, max_payable: Optional[Union[Decimal, float]] = None) -> Decimal:
        """Pay scheduled claim liabilities for the current year.

        Args:
            max_payable: Optional maximum amount that can be paid.

        Returns:
            Decimal: Total amount paid toward claims in dollars.
        """
        total_paid: Decimal = ZERO
        min_cash_balance = to_decimal(100_000)

        # Calculate total scheduled payments and cap
        total_scheduled: Decimal = ZERO
        for claim_item in self.claim_liabilities:
            years_since = self.current_year - claim_item.year_incurred
            scheduled_payment = claim_item.get_payment(years_since)
            total_scheduled += scheduled_payment

        if max_payable is not None:
            max_total_payable: Decimal = min(total_scheduled, to_decimal(max_payable))
        else:
            available_liquidity = self.cash + self.restricted_assets
            max_total_payable = (
                min(total_scheduled, available_liquidity) if available_liquidity > ZERO else ZERO
            )

        payment_ratio: Decimal = ONE
        if total_scheduled > max_total_payable and total_scheduled > ZERO:
            payment_ratio = max_total_payable / total_scheduled
            logger.warning(
                f"LIQUIDITY CONSTRAINT: Capping claim payments at ${max_total_payable:,.2f} "
                f"(scheduled: ${total_scheduled:,.2f}, available liquidity: ${self.cash + self.restricted_assets:,.2f})"
            )

        for claim_item in self.claim_liabilities:
            years_since = self.current_year - claim_item.year_incurred
            scheduled_payment = claim_item.get_payment(years_since)

            if scheduled_payment > ZERO:
                capped_scheduled = scheduled_payment * payment_ratio

                if claim_item.is_insured:
                    available_for_payment = min(capped_scheduled, self.restricted_assets)
                    actual_payment = available_for_payment

                    if actual_payment > ZERO:
                        claim_item.make_payment(actual_payment)
                        total_paid += actual_payment

                        self.ledger.record_double_entry(
                            date=self.current_year,
                            debit_account=AccountName.CLAIM_LIABILITIES,
                            credit_account=AccountName.RESTRICTED_CASH,
                            amount=actual_payment,
                            transaction_type=TransactionType.INSURANCE_CLAIM,
                            description="Insured claim payment from collateral",
                            month=self.current_month,
                        )

                        logger.debug(
                            f"Reduced collateral and restricted assets by ${actual_payment:,.2f}"
                        )
                else:
                    available_for_payment = max(ZERO, self.cash - min_cash_balance)
                    actual_payment = min(capped_scheduled, available_for_payment)

                    if actual_payment > ZERO:
                        claim_item.make_payment(actual_payment)
                        total_paid += actual_payment

                        self.ledger.record_double_entry(
                            date=self.current_year,
                            debit_account=AccountName.CLAIM_LIABILITIES,
                            credit_account=AccountName.CASH,
                            amount=actual_payment,
                            transaction_type=TransactionType.INSURANCE_CLAIM,
                            description="Uninsured claim payment",
                            month=self.current_month,
                        )

                        logger.debug(
                            f"Paid ${actual_payment:,.2f} toward uninsured claim (regular business expense)"
                        )

        # Remove fully paid claims
        self.claim_liabilities = [c for c in self.claim_liabilities if c.remaining_amount > ZERO]

        if total_paid > ZERO:
            logger.info(f"Paid ${total_paid:,.2f} toward claim liabilities")

        if payment_ratio < ONE or self.equity <= ZERO:
            self.check_solvency()

        return total_paid

    def record_claim_accrual(
        self,
        claim_amount: Union[Decimal, float],
        development_pattern: Optional[ClaimDevelopment] = None,
    ) -> None:
        """Record insurance claim with multi-year payment schedule.

        Inflates the claim amount by the configured LAE ratio and records
        a separate LAE ledger entry (Issue #468, ASC 944-40).

        Args:
            claim_amount: Total claim amount to be paid (indemnity only)
            development_pattern: Optional ClaimDevelopment strategy for payment timing.
        """
        amount = to_decimal(claim_amount)
        lae_ratio = to_decimal(getattr(self.config, "lae_ratio", 0.12))
        lae_amount = amount * lae_ratio
        total_with_lae = amount + lae_amount

        if development_pattern is not None:
            if isinstance(development_pattern, list):
                development_pattern = ClaimDevelopment(
                    pattern_name="custom",
                    development_factors=development_pattern,
                )
            new_claim = ClaimLiability(
                original_amount=total_with_lae,
                remaining_amount=total_with_lae,
                year_incurred=self.current_year,
                is_insured=False,
                development_strategy=development_pattern,
            )
        else:
            new_claim = ClaimLiability(
                original_amount=total_with_lae,
                remaining_amount=total_with_lae,
                year_incurred=self.current_year,
                is_insured=False,
            )
        self._apply_reserve_noise(new_claim)
        self.claim_liabilities.append(new_claim)

        # Record LAE expense separately (Issue #468)
        if lae_amount > ZERO:
            self.ledger.record_double_entry(
                date=self.current_year,
                debit_account=AccountName.LAE_EXPENSE,
                credit_account=AccountName.CLAIM_LIABILITIES,
                amount=lae_amount,
                transaction_type=TransactionType.INSURANCE_CLAIM,
                description="LAE on claim accrual (ALAE+ULAE per ASC 944-40)",
            )
            self.period_insurance_lae += lae_amount

        pattern_name = (
            getattr(development_pattern, "pattern_name", "custom")
            if development_pattern
            else "default"
        )
        logger.info(
            f"Created claim liability via record_claim_accrual: ${amount:,.2f} "
            f"(+ LAE ${lae_amount:,.2f}) with pattern {pattern_name}"
        )

    def _apply_reserve_noise(self, claim: ClaimLiability) -> None:
        """Apply initial estimation noise to a claim when reserve development is enabled.

        Stores the true ultimate amount and replaces original/remaining with
        a noisy estimate. No-op if reserve development is disabled.

        Args:
            claim: The ClaimLiability to apply noise to (modified in place).
        """
        if not getattr(self, "_enable_reserve_development", False):
            return
        rng = getattr(self, "_reserve_rng", None)
        if rng is None:
            return
        noise_std = getattr(self.config, "reserve_noise_std", 0.20)
        true_amount = claim.original_amount
        claim.true_ultimate = true_amount
        claim._noise_std = noise_std
        noise_factor = 1.0 + rng.gauss(0.0, noise_std)
        noisy_estimate = to_decimal(max(float(true_amount) * noise_factor, 0.0))
        claim.original_amount = noisy_estimate
        claim.remaining_amount = noisy_estimate

    def re_estimate_reserves(self) -> None:
        """Re-estimate all outstanding claim reserves per ASC 944-40-25.

        Iterates over claim liabilities, skips current-year claims, and
        applies stochastic re-estimation. Tracks adverse and favorable
        development separately and records ledger entries.
        """
        if not getattr(self, "_enable_reserve_development", False):
            return
        rng = getattr(self, "_reserve_rng", None)
        if rng is None:
            return

        for claim in self.claim_liabilities:
            # Skip current-year claims (not yet re-estimated)
            if claim.year_incurred >= self.current_year:
                continue
            change = claim.re_estimate(self.current_year, rng)
            if change > ZERO:
                # Adverse development: reserve increased
                self.period_adverse_development += change
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.RESERVE_DEVELOPMENT,
                    credit_account=AccountName.CLAIM_LIABILITIES,
                    amount=change,
                    transaction_type=TransactionType.RESERVE_DEVELOPMENT,
                    description="Adverse reserve development",
                    month=self.current_month,
                )
            elif change < ZERO:
                # Favorable development: reserve decreased
                favorable = abs(change)
                self.period_favorable_development += favorable
                self.ledger.record_double_entry(
                    date=self.current_year,
                    debit_account=AccountName.CLAIM_LIABILITIES,
                    credit_account=AccountName.RESERVE_DEVELOPMENT,
                    amount=favorable,
                    transaction_type=TransactionType.RESERVE_DEVELOPMENT,
                    description="Favorable reserve development",
                    month=self.current_month,
                )

    def get_reserve_reconciliation(self) -> Dict[str, Union[Decimal, int]]:
        """Generate a reserve reconciliation report.

        Returns:
            Dict with total_booked_reserves, total_true_residual,
            total_redundancy, total_deficiency, and claim_count.
        """
        total_booked = ZERO
        total_true_residual = ZERO
        count = 0
        for claim in self.claim_liabilities:
            total_booked += claim.remaining_amount
            if claim.true_ultimate is not None:
                true_residual = max(claim.true_ultimate - claim._total_paid, ZERO)
                total_true_residual += true_residual
            else:
                total_true_residual += claim.remaining_amount
            count += 1

        redundancy = max(total_booked - total_true_residual, ZERO)
        deficiency = max(total_true_residual - total_booked, ZERO)
        return {
            "total_booked_reserves": total_booked,
            "total_true_residual": total_true_residual,
            "total_redundancy": redundancy,
            "total_deficiency": deficiency,
            "claim_count": count,
        }
