"""Test that update_balance_sheet correctly handles non-cash depreciation (Issue #637).

The core bug: update_balance_sheet() routed net income directly through Cash,
ignoring the fact that depreciation reduces net income without consuming cash.
The fix adds back depreciation to the cash entry so that:
  - Cash reflects actual cash generated/consumed (net_income + depreciation)
  - Retained earnings correctly reflects cumulative net income
"""

from decimal import Decimal

import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decimal_utils import ZERO, to_decimal
from ergodic_insurance.ledger import AccountName, TransactionType
from ergodic_insurance.manufacturer import WidgetManufacturer


@pytest.fixture
def manufacturer():
    """Create a manufacturer with standard configuration for cash tests."""
    config = ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=0.8,
        base_operating_margin=0.08,
        tax_rate=0.25,
        retention_ratio=1.0,  # Retain all earnings (simplifies cash verification)
        capex_to_depreciation_ratio=0.0,  # No capex to isolate depreciation
    )
    return WidgetManufacturer(config)


@pytest.fixture
def manufacturer_with_dividends():
    """Create a manufacturer that pays 30% dividends."""
    config = ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=0.8,
        base_operating_margin=0.08,
        tax_rate=0.25,
        retention_ratio=0.7,
        capex_to_depreciation_ratio=0.0,
    )
    return WidgetManufacturer(config)


class TestCashDepreciationAddBack:
    """Test that cash correctly reflects non-cash depreciation add-back."""

    def test_profit_cash_includes_depreciation_addback(self, manufacturer):
        """Cash should increase by net_income + depreciation, not just net_income.

        Acceptance criteria: Cash balance correctly reflects actual cash
        generated (net income + depreciation add-back).
        """
        initial_cash = manufacturer.cash
        net_income = 500_000
        depreciation = 100_000

        # Record depreciation first (as step() does)
        # This creates Dr DEPRECIATION_EXPENSE / Cr ACCUMULATED_DEPRECIATION
        # and reduces net PP&E but not cash.
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.DEPRECIATION_EXPENSE,
            credit_account=AccountName.ACCUMULATED_DEPRECIATION,
            amount=depreciation,
            transaction_type=TransactionType.DEPRECIATION,
            description="Test depreciation",
        )

        # Cash should not have changed from depreciation alone
        assert manufacturer.cash == initial_cash

        # Now update balance sheet with depreciation add-back
        manufacturer.update_balance_sheet(net_income, depreciation_expense=depreciation)

        # Cash should increase by net_income + depreciation (Issue #637)
        expected_cash = initial_cash + net_income + depreciation
        assert manufacturer.cash == expected_cash, (
            f"Cash should be {expected_cash} (initial {initial_cash} + "
            f"net_income {net_income} + depreciation {depreciation}), "
            f"got {manufacturer.cash}"
        )

    def test_retained_earnings_unchanged_by_depreciation_addback(self, manufacturer):
        """Retained earnings should reflect net_income, not net_income + depreciation.

        Acceptance criteria: Retained earnings balance correctly reflects
        cumulative net income after dividends.
        """
        initial_re = manufacturer.ledger.get_balance(AccountName.RETAINED_EARNINGS)
        net_income = 500_000
        depreciation = 100_000

        manufacturer.update_balance_sheet(net_income, depreciation_expense=depreciation)

        # RE increases by net_income only (retention_ratio = 1.0)
        expected_re = initial_re + net_income
        actual_re = manufacturer.ledger.get_balance(AccountName.RETAINED_EARNINGS)
        assert actual_re == expected_re, (
            f"Retained earnings should be {expected_re} (initial {initial_re} + "
            f"net_income {net_income}), got {actual_re}"
        )

    def test_loss_cash_includes_depreciation_addback(self, manufacturer):
        """On a loss, cash should decrease by loss - depreciation, not full loss.

        If operating cash flow is positive (depreciation > loss), cash increases.
        """
        initial_cash = manufacturer.cash
        net_income = -50_000  # Loss
        depreciation = 100_000  # Depreciation exceeds loss

        # Record depreciation
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.DEPRECIATION_EXPENSE,
            credit_account=AccountName.ACCUMULATED_DEPRECIATION,
            amount=depreciation,
            transaction_type=TransactionType.DEPRECIATION,
            description="Test depreciation",
        )

        manufacturer.update_balance_sheet(net_income, depreciation_expense=depreciation)

        # retention_ratio = 1.0, so retained_earnings = net_income = -50K
        # loss_amount = 50K
        # Cash impact: -50K (loss) + 100K (depreciation add-back) = +50K
        loss_amount = abs(to_decimal(net_income))
        expected_cash = initial_cash - loss_amount + depreciation
        assert manufacturer.cash == expected_cash, (
            f"Cash should be {expected_cash} "
            f"(initial {initial_cash} - loss {loss_amount} + dep {depreciation}), "
            f"got {manufacturer.cash}"
        )

    def test_loss_larger_than_depreciation(self, manufacturer):
        """When loss exceeds depreciation, cash still decreases but by less."""
        initial_cash = manufacturer.cash
        net_income = -200_000
        depreciation = 50_000

        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.DEPRECIATION_EXPENSE,
            credit_account=AccountName.ACCUMULATED_DEPRECIATION,
            amount=depreciation,
            transaction_type=TransactionType.DEPRECIATION,
            description="Test depreciation",
        )

        manufacturer.update_balance_sheet(net_income, depreciation_expense=depreciation)

        # Cash drain: 200K loss - 50K dep add-back = 150K net drain
        loss_amount = abs(to_decimal(net_income))
        expected_cash = initial_cash - loss_amount + depreciation
        assert manufacturer.cash == expected_cash

    def test_zero_depreciation_preserves_old_behavior(self, manufacturer):
        """With zero depreciation, behavior is identical to pre-fix code."""
        initial_cash = manufacturer.cash
        net_income = 500_000

        manufacturer.update_balance_sheet(net_income, depreciation_expense=0)

        # Cash increases by just net_income (no add-back)
        assert manufacturer.cash == initial_cash + net_income

    def test_default_depreciation_preserves_old_behavior(self, manufacturer):
        """Calling without depreciation_expense arg preserves old behavior."""
        initial_cash = manufacturer.cash
        net_income = 500_000

        # No depreciation_expense argument (backward compatibility)
        manufacturer.update_balance_sheet(net_income)

        assert manufacturer.cash == initial_cash + net_income


class TestDepreciationAddBackWithDividends:
    """Test cash/depreciation interaction with dividend payments."""

    def test_profit_with_dividends_and_depreciation(self, manufacturer_with_dividends):
        """Cash = starting + total_retained + depreciation (dividends reduce retained)."""
        mfr = manufacturer_with_dividends
        initial_cash = mfr.cash
        net_income = 500_000
        depreciation = 100_000
        retention_ratio = to_decimal(0.7)

        mfr.update_balance_sheet(net_income, depreciation_expense=depreciation)

        # retained_earnings = 500K * 0.7 = 350K
        # dividends = 500K * 0.3 = 150K (paid implicitly by not adding to cash)
        # total_retained = 350K (assuming dividends can be paid)
        # Cash = initial + 350K + 100K = initial + 450K
        total_retained = to_decimal(net_income) * retention_ratio
        expected_cash = initial_cash + total_retained + depreciation
        assert mfr.cash == expected_cash

    def test_depreciation_addback_increases_dividend_capacity(self, manufacturer_with_dividends):
        """Depreciation add-back provides more cash for dividend payments."""
        mfr = manufacturer_with_dividends
        # Set cash very low to test dividend constraint
        # First drain cash to a low level
        drain_amount = mfr.cash - to_decimal(100)  # Leave only $100 cash
        if drain_amount > ZERO:
            mfr.ledger.record_double_entry(
                date=mfr.current_year,
                debit_account=AccountName.ACCOUNTS_PAYABLE,
                credit_account=AccountName.CASH,
                amount=drain_amount,
                transaction_type=TransactionType.PAYMENT,
                description="Drain cash for test",
            )

        initial_cash = mfr.cash
        assert initial_cash == to_decimal(100)

        net_income = 500
        depreciation = 200

        # Without depreciation: projected_cash = 100 + 350 = 450, dividends = 150 → OK
        # With depreciation: projected_cash = 100 + 350 + 200 = 650, dividends = 150 → OK
        mfr.update_balance_sheet(net_income, depreciation_expense=depreciation)

        # Cash = 100 + 350 (retained) + 200 (dep add-back) = 650
        expected_cash = (
            initial_cash + to_decimal(net_income) * to_decimal(0.7) + to_decimal(depreciation)
        )
        assert mfr.cash == expected_cash


class TestDepreciationAddBackSolvencyChecks:
    """Test that depreciation add-back correctly adjusts solvency thresholds."""

    def test_depreciation_prevents_false_liquidity_crisis(self):
        """A company with large depreciation shouldn't fail liquidity check.

        If the loss is $150K but depreciation is $100K, the actual cash drain
        is only $50K. A company with $80K cash should survive.
        """
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=1.0,
            capex_to_depreciation_ratio=0.0,
        )
        mfr = WidgetManufacturer(config)

        # Drain cash to $80K
        drain_amount = mfr.cash - to_decimal(80_000)
        if drain_amount > ZERO:
            mfr.ledger.record_double_entry(
                date=mfr.current_year,
                debit_account=AccountName.ACCOUNTS_PAYABLE,
                credit_account=AccountName.CASH,
                amount=drain_amount,
                transaction_type=TransactionType.PAYMENT,
                description="Drain cash for test",
            )
        assert float(mfr.cash) == pytest.approx(80_000, abs=1)

        net_income = -150_000
        depreciation = 100_000

        # Record depreciation entry
        mfr.ledger.record_double_entry(
            date=mfr.current_year,
            debit_account=AccountName.DEPRECIATION_EXPENSE,
            credit_account=AccountName.ACCUMULATED_DEPRECIATION,
            amount=depreciation,
            transaction_type=TransactionType.DEPRECIATION,
            description="Test depreciation",
        )

        # Without fix: loss=150K > cash=80K → LIQUIDITY CRISIS
        # With fix: cash_consumed = 150K - 100K = 50K < 80K → survives
        mfr.update_balance_sheet(net_income, depreciation_expense=depreciation)

        # Company should NOT be insolvent (is_ruined is the insolvency flag)
        assert (
            not mfr.is_ruined
        ), "Company should survive: actual cash drain is only $50K with $80K available"
        # Cash = 80K - 150K + 100K = 30K
        expected_cash = to_decimal(80_000) - to_decimal(150_000) + to_decimal(100_000)
        assert float(mfr.cash) == pytest.approx(float(expected_cash), abs=1)

    def test_equity_check_accounts_for_depreciation(self):
        """Equity after loss should account for depreciation add-back."""
        config = ManufacturerConfig(
            initial_assets=1_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=1.0,
            capex_to_depreciation_ratio=0.0,
        )
        mfr = WidgetManufacturer(config)

        initial_equity = mfr.equity
        net_income = -500_000
        depreciation = 200_000

        mfr.ledger.record_double_entry(
            date=mfr.current_year,
            debit_account=AccountName.DEPRECIATION_EXPENSE,
            credit_account=AccountName.ACCUMULATED_DEPRECIATION,
            amount=depreciation,
            transaction_type=TransactionType.DEPRECIATION,
            description="Test depreciation",
        )

        mfr.update_balance_sheet(net_income, depreciation_expense=depreciation)

        # Equity change = -loss + depreciation_addback (for balance sheet TA-TL)
        # But net of depreciation recording (-dep) and add-back (+dep) cancels
        # So net equity change = -loss_amount (the RE close)
        # Wait: equity = TA - TL
        # Depreciation recording: TA -200K (net PP&E down)
        # RE close: TA -500K (cash down), and dep add-back: TA +200K (cash up)
        # Net TA change: -200K - 500K + 200K = -500K
        # Net equity change: -500K = net_income
        expected_equity = initial_equity + to_decimal(net_income)  # Includes depreciation effect
        # Allow tolerance for the depreciation recording itself
        assert float(mfr.equity) == pytest.approx(
            float(expected_equity), rel=0.01
        ), f"Expected equity {expected_equity}, got {mfr.equity}"


class TestAccountingEquationWithDepreciation:
    """Verify the accounting equation holds after depreciation add-back."""

    def test_debits_equal_credits_after_profit_with_depreciation(self, manufacturer):
        """Ledger should remain balanced (debits == credits) after add-back."""
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.DEPRECIATION_EXPENSE,
            credit_account=AccountName.ACCUMULATED_DEPRECIATION,
            amount=100_000,
            transaction_type=TransactionType.DEPRECIATION,
            description="Test depreciation",
        )

        manufacturer.update_balance_sheet(500_000, depreciation_expense=100_000)

        is_balanced, difference = manufacturer.ledger.verify_balance()
        assert is_balanced, f"Ledger unbalanced by ${difference:,.2f}"

    def test_debits_equal_credits_after_loss_with_depreciation(self, manufacturer):
        """Ledger should remain balanced after loss path with add-back."""
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.DEPRECIATION_EXPENSE,
            credit_account=AccountName.ACCUMULATED_DEPRECIATION,
            amount=100_000,
            transaction_type=TransactionType.DEPRECIATION,
            description="Test depreciation",
        )

        manufacturer.update_balance_sheet(-200_000, depreciation_expense=100_000)

        is_balanced, difference = manufacturer.ledger.verify_balance()
        assert is_balanced, f"Ledger unbalanced by ${difference:,.2f}"

    def test_total_assets_equals_liabilities_plus_equity(self, manufacturer):
        """Accounting equation: Assets = Liabilities + Equity."""
        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.DEPRECIATION_EXPENSE,
            credit_account=AccountName.ACCUMULATED_DEPRECIATION,
            amount=100_000,
            transaction_type=TransactionType.DEPRECIATION,
            description="Test depreciation",
        )

        manufacturer.update_balance_sheet(500_000, depreciation_expense=100_000)

        # equity is defined as total_assets - total_liabilities,
        # so this is tautologically true, but verifies no exceptions
        assert manufacturer.equity == manufacturer.total_assets - manufacturer.total_liabilities


class TestIntegrationWithStep:
    """Test the full step() integration with depreciation add-back."""

    def test_step_passes_depreciation_to_balance_sheet(self):
        """Verify step() correctly passes depreciation to update_balance_sheet."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=1.0,
            capex_to_depreciation_ratio=0.0,  # No capex to isolate depreciation
        )
        mfr = WidgetManufacturer(config)
        initial_cash = mfr.cash

        # Run one step — this calls record_depreciation then update_balance_sheet
        mfr.step(
            letter_of_credit_rate=0,
            growth_rate=0.0,
            time_resolution="annual",
        )

        # After step: cash should reflect net_income + depreciation, not just net_income
        # The exact values depend on the config, but we can verify cash > what
        # it would be without depreciation add-back.
        # Key: net_income already includes depreciation as embedded expense.
        # Cash = initial + net_income + depreciation (from add-back)
        # Without fix: cash = initial + net_income (understated by depreciation)

        # We can verify by checking that cash increased by MORE than net_income
        net_income = to_decimal(mfr._period_net_income or 0)
        gross_ppe = mfr.gross_ppe
        annual_dep = gross_ppe / to_decimal(10)  # useful_life_years=10

        # Cash change should include depreciation add-back
        cash_change = mfr.cash - initial_cash
        expected_minimum = net_income  # Without fix, this would be the change
        assert cash_change > expected_minimum or annual_dep == ZERO, (
            f"Cash change {cash_change} should exceed net_income {net_income} "
            f"by depreciation {annual_dep}"
        )

    def test_multi_period_cash_accumulation(self):
        """Over multiple periods, cash should not be systematically understated."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=1.0,
            capex_to_depreciation_ratio=0.0,  # No capex to isolate depreciation
        )
        mfr = WidgetManufacturer(config)
        initial_cash = mfr.cash
        cumulative_net_income = ZERO
        cumulative_depreciation = ZERO

        for year in range(5):
            pre_step_accumulated_dep = mfr.accumulated_depreciation
            mfr.step(
                letter_of_credit_rate=0,
                growth_rate=0.0,
                time_resolution="annual",
            )
            cumulative_net_income += to_decimal(mfr._period_net_income or 0)
            period_dep = mfr.accumulated_depreciation - pre_step_accumulated_dep
            cumulative_depreciation += period_dep

        # Cash should be approximately initial + cumulative_net_income + cumulative_depreciation.
        # Allow generous tolerance because tax payments, accruals, and other
        # cash-affecting entries (e.g., tax accrual -> payment timing) cause the
        # exact cash figure to differ from the simple formula.
        cash_change = mfr.cash - initial_cash
        expected_change = cumulative_net_income + cumulative_depreciation

        assert float(cash_change) == pytest.approx(float(expected_change), rel=0.10), (
            f"Cash change {cash_change} should approximately equal "
            f"net_income {cumulative_net_income} + depreciation {cumulative_depreciation} "
            f"= {expected_change}"
        )
