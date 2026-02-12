"""Tests for claim re-estimation ledger entries (Issue #849).

Verifies that every change to ClaimLiability.remaining_amount has a
corresponding double-entry ledger record, ensuring:

- Adverse development: Dr RESERVE_DEVELOPMENT, Cr CLAIM_LIABILITIES
- Favorable development: Dr CLAIM_LIABILITIES, Cr RESERVE_DEVELOPMENT
- Ledger trial balance reflects current claim liability balances
- Reserve development appears in income statement from ledger
- Initial estimation noise is captured in the ledger
"""

from decimal import Decimal
import random

import pytest

from ergodic_insurance.claim_liability import ClaimLiability
from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decimal_utils import ZERO, to_decimal
from ergodic_insurance.ledger import AccountName, EntryType, TransactionType
from ergodic_insurance.manufacturer import WidgetManufacturer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manufacturer(noise_std=0.20, lae_ratio=0.0, initial_assets=10_000_000):
    """Build a WidgetManufacturer with reserve development enabled."""
    config = ManufacturerConfig(
        initial_assets=initial_assets,
        enable_reserve_development=True,
        reserve_noise_std=noise_std,
        lae_ratio=lae_ratio,
    )
    return WidgetManufacturer(config)


def _sum_claim_remaining(mfr) -> Decimal:
    """Sum remaining_amount across all open claim liabilities."""
    return sum((c.remaining_amount for c in mfr.claim_liabilities), ZERO)


def _get_reserve_dev_entries(mfr):
    """Get all RESERVE_DEVELOPMENT transaction-type entries."""
    return mfr.ledger.get_entries(transaction_type=TransactionType.RESERVE_DEVELOPMENT)


# ---------------------------------------------------------------------------
# Test: Adverse development ledger entries
# ---------------------------------------------------------------------------


class TestAdverseDevelopmentLedgerEntries:
    """Verify Dr RESERVE_DEVELOPMENT / Cr CLAIM_LIABILITIES for adverse development."""

    def test_adverse_entries_have_correct_accounts_and_amounts(self):
        """Force adverse development and verify entry accounts and amounts."""
        mfr = _make_manufacturer(noise_std=0.01)
        mfr.current_year = 1
        mfr.record_claim_accrual(500_000)

        claim = mfr.claim_liabilities[0]
        # Force remaining far below true → adverse development when re-estimated
        claim.remaining_amount = to_decimal(200_000)
        claim.true_ultimate = to_decimal(500_000)
        claim._noise_std = 0.001  # Near-zero noise → snaps to true residual

        entries_before = len(mfr.ledger.entries)
        mfr.current_year = 10  # Full maturity for convergence
        mfr.re_estimate_reserves()

        # Should have had adverse development
        assert (
            mfr.period_adverse_development > ZERO
        ), "Expected adverse development when remaining << true ultimate"

        dev_entries = _get_reserve_dev_entries(mfr)
        # Filter to only entries after our re-estimation (exclude initial noise entries)
        new_dev_entries = [e for e in dev_entries if e.date == 10]

        # Should have exactly one debit-credit pair
        debit_entries = [e for e in new_dev_entries if e.entry_type == EntryType.DEBIT]
        credit_entries = [e for e in new_dev_entries if e.entry_type == EntryType.CREDIT]
        assert len(debit_entries) >= 1, "Expected at least one debit entry for adverse development"
        assert (
            len(credit_entries) >= 1
        ), "Expected at least one credit entry for adverse development"

        # Debit should be to RESERVE_DEVELOPMENT (expense increases)
        debit_accounts = {e.account for e in debit_entries}
        assert AccountName.RESERVE_DEVELOPMENT.value in debit_accounts

        # Credit should be to CLAIM_LIABILITIES (liability increases)
        credit_accounts = {e.account for e in credit_entries}
        assert AccountName.CLAIM_LIABILITIES.value in credit_accounts

        # Amounts should match
        dev_debit = sum(
            e.amount for e in debit_entries if e.account == AccountName.RESERVE_DEVELOPMENT.value
        )
        liab_credit = sum(
            e.amount for e in credit_entries if e.account == AccountName.CLAIM_LIABILITIES.value
        )
        assert dev_debit == liab_credit, f"Debit ({dev_debit}) != Credit ({liab_credit})"
        assert dev_debit == mfr.period_adverse_development


# ---------------------------------------------------------------------------
# Test: Favorable development ledger entries
# ---------------------------------------------------------------------------


class TestFavorableDevelopmentLedgerEntries:
    """Verify Dr CLAIM_LIABILITIES / Cr RESERVE_DEVELOPMENT for favorable development."""

    def test_favorable_entries_have_correct_accounts_and_amounts(self):
        """Force favorable development and verify entry accounts and amounts."""
        mfr = _make_manufacturer(noise_std=0.01)
        mfr.current_year = 1
        mfr.record_claim_accrual(500_000)

        claim = mfr.claim_liabilities[0]
        # Force remaining far above true → favorable development when re-estimated
        claim.remaining_amount = to_decimal(800_000)
        claim.true_ultimate = to_decimal(500_000)
        claim._noise_std = 0.001  # Near-zero noise → snaps to true residual

        mfr.current_year = 10  # Full maturity for convergence
        mfr.re_estimate_reserves()

        # Should have had favorable development
        assert (
            mfr.period_favorable_development > ZERO
        ), "Expected favorable development when remaining >> true ultimate"

        dev_entries = _get_reserve_dev_entries(mfr)
        new_dev_entries = [e for e in dev_entries if e.date == 10]

        debit_entries = [e for e in new_dev_entries if e.entry_type == EntryType.DEBIT]
        credit_entries = [e for e in new_dev_entries if e.entry_type == EntryType.CREDIT]
        assert len(debit_entries) >= 1, "Expected at least one debit for favorable development"
        assert len(credit_entries) >= 1, "Expected at least one credit for favorable development"

        # Debit should be to CLAIM_LIABILITIES (liability decreases)
        debit_accounts = {e.account for e in debit_entries}
        assert AccountName.CLAIM_LIABILITIES.value in debit_accounts

        # Credit should be to RESERVE_DEVELOPMENT (expense decreases / benefit)
        credit_accounts = {e.account for e in credit_entries}
        assert AccountName.RESERVE_DEVELOPMENT.value in credit_accounts

        # Amounts should match
        liab_debit = sum(
            e.amount for e in debit_entries if e.account == AccountName.CLAIM_LIABILITIES.value
        )
        dev_credit = sum(
            e.amount for e in credit_entries if e.account == AccountName.RESERVE_DEVELOPMENT.value
        )
        assert liab_debit == dev_credit, f"Debit ({liab_debit}) != Credit ({dev_credit})"
        assert dev_credit == mfr.period_favorable_development


# ---------------------------------------------------------------------------
# Test: Trial balance reflects claim liabilities
# ---------------------------------------------------------------------------


class TestTrialBalanceReflectsClaimLiabilities:
    """Ledger CLAIM_LIABILITIES balance must equal sum of remaining_amounts."""

    def test_trial_balance_after_accrual_with_noise(self):
        """After recording a claim with noise, ledger matches claim state."""
        mfr = _make_manufacturer(noise_std=0.25, lae_ratio=0.0)
        mfr.current_year = 1
        mfr.record_claim_accrual(1_000_000)

        ledger_balance = mfr.ledger.get_balance(AccountName.CLAIM_LIABILITIES)
        claim_total = _sum_claim_remaining(mfr)
        assert ledger_balance == claim_total, (
            f"Ledger CLAIM_LIABILITIES ({ledger_balance}) != "
            f"sum of remaining_amounts ({claim_total})"
        )

    def test_trial_balance_after_reestimation(self):
        """After re-estimation, ledger still matches claim state."""
        mfr = _make_manufacturer(noise_std=0.25, lae_ratio=0.0)
        mfr.current_year = 1
        mfr.record_claim_accrual(500_000)
        mfr.record_claim_accrual(300_000)

        mfr.current_year = 3
        mfr.re_estimate_reserves()

        ledger_balance = mfr.ledger.get_balance(AccountName.CLAIM_LIABILITIES)
        claim_total = _sum_claim_remaining(mfr)
        assert ledger_balance == claim_total, (
            f"Ledger CLAIM_LIABILITIES ({ledger_balance}) != "
            f"sum of remaining_amounts ({claim_total}) after re-estimation"
        )

    def test_trial_balance_after_multiple_reestimations(self):
        """Ledger stays in sync across multiple re-estimation periods."""
        mfr = _make_manufacturer(noise_std=0.30, lae_ratio=0.0)
        mfr.current_year = 1
        mfr.record_claim_accrual(800_000)

        for year in range(2, 8):
            mfr.current_year = year
            mfr.re_estimate_reserves()

            ledger_balance = mfr.ledger.get_balance(AccountName.CLAIM_LIABILITIES)
            claim_total = _sum_claim_remaining(mfr)
            assert ledger_balance == claim_total, (
                f"Year {year}: Ledger ({ledger_balance}) != " f"Claims ({claim_total})"
            )

    def test_trial_balance_with_lae(self):
        """Trial balance accounts for LAE when lae_ratio > 0."""
        mfr = _make_manufacturer(noise_std=0.15, lae_ratio=0.12)
        mfr.current_year = 1
        mfr.record_claim_accrual(500_000)

        ledger_balance = mfr.ledger.get_balance(AccountName.CLAIM_LIABILITIES)
        claim_total = _sum_claim_remaining(mfr)
        assert (
            ledger_balance == claim_total
        ), f"With LAE: Ledger ({ledger_balance}) != Claims ({claim_total})"

    def test_trial_balance_via_process_uninsured_claim(self):
        """Trial balance correct when claims created via process_uninsured_claim."""
        mfr = _make_manufacturer(noise_std=0.20, lae_ratio=0.0)
        mfr.current_year = 1
        mfr.process_uninsured_claim(200_000)

        if mfr.claim_liabilities:
            ledger_balance = mfr.ledger.get_balance(AccountName.CLAIM_LIABILITIES)
            claim_total = _sum_claim_remaining(mfr)
            assert (
                ledger_balance == claim_total
            ), f"Uninsured: Ledger ({ledger_balance}) != Claims ({claim_total})"


# ---------------------------------------------------------------------------
# Test: Reserve development in income statement
# ---------------------------------------------------------------------------


class TestReserveDevelopmentInIncomeStatement:
    """Reserve development must appear in the income statement from the ledger."""

    def test_reserve_development_account_has_entries(self):
        """After re-estimation, RESERVE_DEVELOPMENT account has a non-zero balance."""
        mfr = _make_manufacturer(noise_std=0.01, lae_ratio=0.0)
        mfr.current_year = 1
        mfr.record_claim_accrual(500_000)

        claim = mfr.claim_liabilities[0]
        claim.remaining_amount = to_decimal(200_000)
        claim.true_ultimate = to_decimal(500_000)
        claim._noise_std = 0.001

        mfr.current_year = 10
        mfr.re_estimate_reserves()

        # RESERVE_DEVELOPMENT should have a non-zero balance from ledger
        dev_balance = mfr.ledger.get_balance(AccountName.RESERVE_DEVELOPMENT)
        assert (
            dev_balance != ZERO
        ), "Expected non-zero RESERVE_DEVELOPMENT balance after re-estimation"

    def test_net_reserve_development_in_metrics(self):
        """calculate_metrics includes net_reserve_development."""
        mfr = _make_manufacturer(noise_std=0.25, lae_ratio=0.0)
        mfr.current_year = 1
        mfr.record_claim_accrual(500_000)

        claim = mfr.claim_liabilities[0]
        claim.remaining_amount = to_decimal(200_000)
        claim.true_ultimate = to_decimal(500_000)
        claim._noise_std = 0.001

        mfr.current_year = 10
        mfr.re_estimate_reserves()

        metrics = mfr.calculate_metrics()
        assert "net_reserve_development" in metrics
        net_dev = metrics["net_reserve_development"]
        expected = mfr.period_adverse_development - mfr.period_favorable_development
        assert net_dev == expected


# ---------------------------------------------------------------------------
# Test: Initial noise recorded in ledger
# ---------------------------------------------------------------------------


class TestInitialNoiseRecordedInLedger:
    """Initial estimation noise from _apply_reserve_noise must have ledger entries."""

    def test_noise_creates_reserve_development_entries(self):
        """When noise changes remaining_amount, a ledger entry is recorded."""
        mfr = _make_manufacturer(noise_std=0.50, lae_ratio=0.0)
        mfr.current_year = 1

        entries_before = len(mfr.ledger.entries)
        mfr.record_claim_accrual(500_000)
        entries_after = len(mfr.ledger.entries)

        claim = mfr.claim_liabilities[0]
        noise_delta = claim.remaining_amount - claim.true_ultimate

        if noise_delta != ZERO:
            # Should have at least one pair of RESERVE_DEVELOPMENT entries for the noise
            dev_entries = mfr.ledger.get_entries(
                transaction_type=TransactionType.RESERVE_DEVELOPMENT
            )
            assert len(dev_entries) >= 2, (
                f"Expected RESERVE_DEVELOPMENT entries for noise delta "
                f"({noise_delta}), got {len(dev_entries)} entries"
            )

    def test_noise_disabled_no_extra_entries(self):
        """With noise_std=0.0, no noise adjustment entries are created."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            enable_reserve_development=True,
            reserve_noise_std=0.0,
            lae_ratio=0.0,
        )
        mfr = WidgetManufacturer(config)
        mfr.current_year = 1
        mfr.record_claim_accrual(500_000)

        dev_entries = mfr.ledger.get_entries(transaction_type=TransactionType.RESERVE_DEVELOPMENT)
        # With zero noise, remaining == true_ultimate, so no noise entries
        assert len(dev_entries) == 0, (
            f"Expected no RESERVE_DEVELOPMENT entries with zero noise, " f"got {len(dev_entries)}"
        )

    def test_noise_does_not_affect_period_development_tracking(self):
        """Initial noise should NOT update period_adverse/favorable_development."""
        mfr = _make_manufacturer(noise_std=0.50, lae_ratio=0.0)
        mfr.current_year = 1
        mfr.record_claim_accrual(500_000)

        # Period development should still be zero after initial claim recording
        assert mfr.period_adverse_development == ZERO
        assert mfr.period_favorable_development == ZERO


# ---------------------------------------------------------------------------
# Test: record_claim_accrual records indemnity entry
# ---------------------------------------------------------------------------


class TestRecordClaimAccrualLedgerEntries:
    """record_claim_accrual must record both indemnity and LAE entries."""

    def test_indemnity_entry_recorded(self):
        """Indemnity portion is recorded as Dr INSURANCE_LOSS / Cr CLAIM_LIABILITIES."""
        mfr = _make_manufacturer(noise_std=0.0, lae_ratio=0.0)
        mfr.current_year = 1
        mfr.record_claim_accrual(500_000)

        claim_entries = mfr.ledger.get_entries(transaction_type=TransactionType.INSURANCE_CLAIM)
        # Should have at least one debit to INSURANCE_LOSS
        loss_debits = [
            e
            for e in claim_entries
            if e.account == AccountName.INSURANCE_LOSS.value and e.entry_type == EntryType.DEBIT
        ]
        assert len(loss_debits) == 1
        assert loss_debits[0].amount == to_decimal(500_000)

        # Should have at least one credit to CLAIM_LIABILITIES
        liab_credits = [
            e
            for e in claim_entries
            if e.account == AccountName.CLAIM_LIABILITIES.value and e.entry_type == EntryType.CREDIT
        ]
        assert len(liab_credits) >= 1
        # At least one credit should be for the indemnity amount
        indemnity_credit = [e for e in liab_credits if e.amount == to_decimal(500_000)]
        assert len(indemnity_credit) == 1

    def test_indemnity_and_lae_both_recorded(self):
        """Both indemnity and LAE entries are present with lae_ratio > 0."""
        mfr = _make_manufacturer(noise_std=0.0, lae_ratio=0.10)
        mfr.current_year = 1
        mfr.record_claim_accrual(500_000)

        claim_entries = mfr.ledger.get_entries(transaction_type=TransactionType.INSURANCE_CLAIM)
        # Indemnity: Dr INSURANCE_LOSS 500,000
        loss_debits = [e for e in claim_entries if e.account == AccountName.INSURANCE_LOSS.value]
        assert len(loss_debits) == 1
        assert loss_debits[0].amount == to_decimal(500_000)

        # LAE: Dr LAE_EXPENSE 50,000
        lae_debits = [e for e in claim_entries if e.account == AccountName.LAE_EXPENSE.value]
        assert len(lae_debits) == 1
        assert lae_debits[0].amount == to_decimal(50_000)

        # CLAIM_LIABILITIES credits for both
        liab_credits = [
            e for e in claim_entries if e.account == AccountName.CLAIM_LIABILITIES.value
        ]
        total_credited = sum(e.amount for e in liab_credits)
        assert total_credited == to_decimal(550_000)  # 500K + 50K LAE
