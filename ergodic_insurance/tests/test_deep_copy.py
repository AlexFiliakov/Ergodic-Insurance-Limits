"""Tests for deep copy functionality of stateful classes.

This module tests the deep copy capabilities added to fix Issue #273:
Monte Carlo Worker State Corruption (Incomplete Copy).

The tests verify that:
1. All stateful classes support proper deep copy
2. Copied instances are completely independent from originals
3. Modifications to copies do not affect originals
4. Walk-forward simulations preserve warmed-up state
5. Deep copies are pickleable for Windows multiprocessing
"""

import copy
from decimal import Decimal
import pickle

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
import pytest

from ergodic_insurance.accrual_manager import (
    AccrualItem,
    AccrualManager,
    AccrualType,
    PaymentSchedule,
)
from ergodic_insurance.claim_development import ClaimDevelopment
from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decimal_utils import ZERO, to_decimal
from ergodic_insurance.insurance_accounting import InsuranceAccounting, InsuranceRecovery
from ergodic_insurance.ledger import EntryType, Ledger, LedgerEntry, TransactionType
from ergodic_insurance.manufacturer import ClaimLiability, WidgetManufacturer


class TestClaimLiabilityDeepCopy:
    """Tests for ClaimLiability deep copy functionality."""

    def test_deep_copy_preserves_all_fields(self):
        """Verify all fields are preserved in deep copy."""
        custom_strategy = ClaimDevelopment(
            pattern_name="CUSTOM",
            development_factors=[0.25, 0.35, 0.40],
        )
        original = ClaimLiability(
            original_amount=to_decimal(1_000_000),
            remaining_amount=to_decimal(750_000),
            year_incurred=5,
            is_insured=True,
            development_strategy=custom_strategy,
        )

        copied = copy.deepcopy(original)

        assert copied.original_amount == original.original_amount
        assert copied.remaining_amount == original.remaining_amount
        assert copied.year_incurred == original.year_incurred
        assert copied.is_insured == original.is_insured
        assert (
            copied.development_strategy.development_factors
            == original.development_strategy.development_factors
        )
        assert (
            copied.development_strategy.pattern_name == original.development_strategy.pattern_name
        )

    def test_deep_copy_independence(self):
        """Verify modifications to copy don't affect original."""
        custom_strategy = ClaimDevelopment(
            pattern_name="CUSTOM",
            development_factors=[0.25, 0.35, 0.40],
        )
        original = ClaimLiability(
            original_amount=to_decimal(1_000_000),
            remaining_amount=to_decimal(1_000_000),
            year_incurred=5,
            development_strategy=custom_strategy,
        )

        copied = copy.deepcopy(original)

        # Modify the copy
        copied.make_payment(to_decimal(100_000))
        copied.development_strategy.development_factors.append(0.10)

        # Original should be unchanged
        assert original.remaining_amount == to_decimal(1_000_000)
        assert len(original.development_strategy.development_factors) == 3

    def test_deep_copy_is_pickleable(self):
        """Verify deep copy can be pickled for multiprocessing."""
        original = ClaimLiability(
            original_amount=to_decimal(500_000),
            remaining_amount=to_decimal(500_000),
            year_incurred=3,
        )

        copied = copy.deepcopy(original)
        pickled = pickle.dumps(copied)
        restored = pickle.loads(pickled)

        assert restored.original_amount == original.original_amount


class TestAccrualItemDeepCopy:
    """Tests for AccrualItem deep copy functionality."""

    def test_deep_copy_preserves_all_fields(self):
        """Verify all fields are preserved in deep copy."""
        original = AccrualItem(
            item_type=AccrualType.TAXES,
            amount=to_decimal(50_000),
            period_incurred=12,
            payment_schedule=PaymentSchedule.QUARTERLY,
            payment_dates=[3, 6, 9, 12],
            amounts_paid=[to_decimal(12_500), to_decimal(12_500)],
            description="Q1 Tax Accrual",
        )

        copied = copy.deepcopy(original)

        assert copied.item_type == original.item_type
        assert copied.amount == original.amount
        assert copied.period_incurred == original.period_incurred
        assert copied.payment_schedule == original.payment_schedule
        assert copied.payment_dates == original.payment_dates
        assert copied.amounts_paid == original.amounts_paid
        assert copied.description == original.description

    def test_deep_copy_independence(self):
        """Verify modifications to copy don't affect original."""
        original = AccrualItem(
            item_type=AccrualType.WAGES,
            amount=to_decimal(100_000),
            period_incurred=1,
            payment_schedule=PaymentSchedule.IMMEDIATE,
            amounts_paid=[],
        )

        copied = copy.deepcopy(original)

        # Modify the copy
        copied.amounts_paid.append(to_decimal(50_000))

        # Original should be unchanged
        assert len(original.amounts_paid) == 0


class TestAccrualManagerDeepCopy:
    """Tests for AccrualManager deep copy functionality."""

    def test_deep_copy_preserves_state(self):
        """Verify all state is preserved in deep copy."""
        original = AccrualManager()
        original.current_period = 24

        # Add various accruals
        original.record_expense_accrual(
            AccrualType.TAXES, to_decimal(100_000), PaymentSchedule.QUARTERLY
        )
        original.record_expense_accrual(
            AccrualType.WAGES, to_decimal(50_000), PaymentSchedule.IMMEDIATE
        )
        original.record_revenue_accrual(to_decimal(200_000))

        copied = copy.deepcopy(original)

        assert copied.current_period == original.current_period
        assert len(copied.accrued_expenses[AccrualType.TAXES]) == 1
        assert len(copied.accrued_expenses[AccrualType.WAGES]) == 1
        assert len(copied.accrued_revenues) == 1

    def test_deep_copy_independence(self):
        """Verify modifications to copy don't affect original."""
        original = AccrualManager()
        original.record_expense_accrual(AccrualType.INSURANCE_CLAIMS, to_decimal(500_000))

        copied = copy.deepcopy(original)

        # Modify the copy
        copied.advance_period(12)
        copied.record_expense_accrual(AccrualType.OTHER, to_decimal(10_000))

        # Original should be unchanged
        assert original.current_period == 0
        assert len(original.accrued_expenses[AccrualType.OTHER]) == 0


class TestInsuranceRecoveryDeepCopy:
    """Tests for InsuranceRecovery deep copy functionality."""

    def test_deep_copy_preserves_all_fields(self):
        """Verify all fields are preserved in deep copy."""
        original = InsuranceRecovery(
            amount=to_decimal(250_000),
            claim_id="CLAIM_2024_001",
            year_approved=2024,
            amount_received=to_decimal(100_000),
        )

        copied = copy.deepcopy(original)

        assert copied.amount == original.amount
        assert copied.claim_id == original.claim_id
        assert copied.year_approved == original.year_approved
        assert copied.amount_received == original.amount_received


class TestInsuranceAccountingDeepCopy:
    """Tests for InsuranceAccounting deep copy functionality."""

    def test_deep_copy_preserves_state(self):
        """Verify all state is preserved in deep copy."""
        original = InsuranceAccounting()
        original.pay_annual_premium(to_decimal(120_000))

        # Advance a few months
        for _ in range(3):
            original.record_monthly_expense()

        # Add a recovery
        original.record_claim_recovery(to_decimal(50_000), "CLM001", 2024)

        copied = copy.deepcopy(original)

        assert copied.prepaid_insurance == original.prepaid_insurance
        assert copied.monthly_expense == original.monthly_expense
        assert copied.annual_premium == original.annual_premium
        assert copied.current_month == original.current_month
        assert len(copied.recoveries) == len(original.recoveries)

    def test_deep_copy_independence(self):
        """Verify modifications to copy don't affect original."""
        original = InsuranceAccounting()
        original.pay_annual_premium(to_decimal(60_000))

        copied = copy.deepcopy(original)

        # Modify the copy
        copied.record_monthly_expense()
        copied.record_claim_recovery(to_decimal(20_000))

        # Original should be unchanged
        assert original.current_month == 0
        assert len(original.recoveries) == 0


class TestLedgerEntryDeepCopy:
    """Tests for LedgerEntry deep copy functionality."""

    def test_deep_copy_preserves_all_fields(self):
        """Verify all fields are preserved in deep copy."""
        original = LedgerEntry(
            date=5,
            account="cash",
            amount=to_decimal(1_000_000),
            entry_type=EntryType.DEBIT,
            transaction_type=TransactionType.REVENUE,
            description="Cash sale",
            month=6,
        )

        copied = copy.deepcopy(original)

        assert copied.date == original.date
        assert copied.account == original.account
        assert copied.amount == original.amount
        assert copied.entry_type == original.entry_type
        assert copied.transaction_type == original.transaction_type
        assert copied.description == original.description
        assert copied.reference_id == original.reference_id
        assert copied.month == original.month


class TestLedgerDeepCopy:
    """Tests for Ledger deep copy functionality."""

    def test_deep_copy_preserves_entries(self):
        """Verify all entries are preserved in deep copy."""
        original = Ledger()

        # Record several transactions
        original.record_double_entry(
            date=1,
            debit_account="cash",
            credit_account="revenue",
            amount=to_decimal(500_000),
            transaction_type=TransactionType.REVENUE,
        )
        original.record_double_entry(
            date=1,
            debit_account="inventory",
            credit_account="cash",
            amount=to_decimal(200_000),
            transaction_type=TransactionType.INVENTORY_PURCHASE,
        )

        copied = copy.deepcopy(original)

        assert len(copied.entries) == len(original.entries)
        assert copied.get_balance("cash") == original.get_balance("cash")
        assert copied.get_balance("revenue") == original.get_balance("revenue")

    def test_deep_copy_preserves_balance_cache(self):
        """Verify balance cache is preserved in deep copy."""
        original = Ledger()

        original.record_double_entry(
            date=1,
            debit_account="cash",
            credit_account="retained_earnings",
            amount=to_decimal(1_000_000),
            transaction_type=TransactionType.EQUITY_ISSUANCE,
        )

        # Access balance to populate cache
        _ = original.get_balance("cash")

        copied = copy.deepcopy(original)

        # Both should have the same cached balance
        assert copied._balances == original._balances

    def test_deep_copy_independence(self):
        """Verify modifications to copy don't affect original."""
        original = Ledger()

        original.record_double_entry(
            date=1,
            debit_account="cash",
            credit_account="revenue",
            amount=to_decimal(100_000),
            transaction_type=TransactionType.REVENUE,
        )

        original_entry_count = len(original.entries)

        copied = copy.deepcopy(original)

        # Modify the copy
        copied.record_double_entry(
            date=2,
            debit_account="accounts_receivable",
            credit_account="revenue",
            amount=to_decimal(50_000),
            transaction_type=TransactionType.REVENUE,
        )

        # Original should be unchanged
        assert len(original.entries) == original_entry_count


class TestWidgetManufacturerDeepCopy:
    """Tests for WidgetManufacturer deep copy functionality."""

    @pytest.fixture
    def basic_config(self):
        """Create a basic manufacturer configuration."""
        return ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )

    def test_deep_copy_preserves_basic_state(self, basic_config):
        """Verify basic state is preserved in deep copy."""
        original = WidgetManufacturer(basic_config)

        copied = copy.deepcopy(original)

        assert copied.current_year == original.current_year
        assert copied.current_month == original.current_month
        assert copied.is_ruined == original.is_ruined
        assert copied.cash == original.cash
        assert copied.total_assets == original.total_assets

    def test_deep_copy_after_steps(self, basic_config):
        """Verify state is preserved after running simulation steps."""
        original = WidgetManufacturer(basic_config)

        # Advance to Year 5
        for _ in range(5):
            original.step()

        copied = copy.deepcopy(original)

        # Verify year preserved
        assert copied.current_year == 5
        assert original.current_year == 5

        # Verify metrics history preserved
        assert len(copied.metrics_history) == len(original.metrics_history) == 5

    def test_deep_copy_independence(self, basic_config):
        """Verify copied manufacturer is independent from original."""
        original = WidgetManufacturer(basic_config)

        # Advance to Year 5
        for _ in range(5):
            original.step()

        copied = copy.deepcopy(original)

        # Advance the copy
        copied.step()

        # Verify independence
        assert copied.current_year == 6
        assert original.current_year == 5  # Original unchanged

        # Verify metrics history independence
        assert len(copied.metrics_history) == 6
        assert len(original.metrics_history) == 5  # Original unchanged

    def test_deep_copy_preserves_nested_objects(self, basic_config):
        """Verify nested objects are properly deep copied."""
        original = WidgetManufacturer(basic_config)

        # Run a few steps to populate nested objects
        for _ in range(3):
            original.step()

        copied = copy.deepcopy(original)

        # Verify accrual_manager is independent
        assert copied.accrual_manager is not original.accrual_manager
        assert copied.accrual_manager.current_period == original.accrual_manager.current_period

        # Verify insurance_accounting is independent
        assert copied.insurance_accounting is not original.insurance_accounting

        # Verify ledger is independent
        assert copied.ledger is not original.ledger
        assert len(copied.ledger.entries) == len(original.ledger.entries)

    def test_deep_copy_preserves_claim_liabilities(self, basic_config):
        """Verify claim liabilities are properly deep copied."""
        original = WidgetManufacturer(basic_config)

        # Add a claim liability manually
        claim = ClaimLiability(
            original_amount=to_decimal(500_000),
            remaining_amount=to_decimal(500_000),
            year_incurred=1,
        )
        original.claim_liabilities.append(claim)

        copied = copy.deepcopy(original)

        # Verify claim is copied
        assert len(copied.claim_liabilities) == 1
        assert copied.claim_liabilities[0].original_amount == to_decimal(500_000)

        # Verify independence
        assert copied.claim_liabilities is not original.claim_liabilities
        assert copied.claim_liabilities[0] is not original.claim_liabilities[0]

        # Modify copy's claim
        copied.claim_liabilities[0].make_payment(to_decimal(100_000))

        # Original should be unchanged
        assert original.claim_liabilities[0].remaining_amount == to_decimal(500_000)

    def test_deep_copy_is_pickleable(self, basic_config):
        """Verify deep copy can be pickled for multiprocessing."""
        original = WidgetManufacturer(basic_config)

        # Run a few steps
        for _ in range(3):
            original.step()

        copied = copy.deepcopy(original)

        # Pickle and restore
        pickled = pickle.dumps(copied)
        restored = pickle.loads(pickled)

        assert restored.current_year == 3
        assert len(restored.metrics_history) == 3

    def test_getstate_setstate(self, basic_config):
        """Verify __getstate__ and __setstate__ work correctly."""
        original = WidgetManufacturer(basic_config)

        for _ in range(2):
            original.step()

        # Get and set state manually
        state = original.__getstate__()
        new_instance = WidgetManufacturer.__new__(WidgetManufacturer)
        new_instance.__setstate__(state)

        assert new_instance.current_year == original.current_year
        assert new_instance.total_assets == original.total_assets


class TestMonteCarloWalkForward:
    """Tests for walk-forward simulation state preservation.

    These tests verify that the Monte Carlo worker correctly preserves
    warmed-up state when forking simulations.
    """

    def test_walk_forward_preserves_year(self):
        """Verify forked simulation retains warm-up year."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        original = WidgetManufacturer(config)

        # Warm up to Year 5
        for _ in range(5):
            original.step()

        assert original.current_year == 5

        # Fork for simulation
        forked = copy.deepcopy(original)

        # Verify forked starts at Year 5
        assert forked.current_year == 5

        # Advance forked
        forked.step()
        assert forked.current_year == 6

        # Original unchanged
        assert original.current_year == 5

    def test_walk_forward_preserves_metrics_history(self):
        """Verify forked simulation retains pre-fork metrics history."""
        config = ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        original = WidgetManufacturer(config)

        # Warm up to Year 5
        for _ in range(5):
            original.step()

        assert len(original.metrics_history) == 5

        # Fork for simulation
        forked = copy.deepcopy(original)

        # Verify forked has history
        assert len(forked.metrics_history) == 5

        # Advance forked
        forked.step()
        assert len(forked.metrics_history) == 6

        # Original history unchanged
        assert len(original.metrics_history) == 5


class TestPropertyBasedDeepCopy:
    """Property-based tests using Hypothesis to verify copy independence."""

    @given(
        original_amount=st.decimals(
            min_value=1000, max_value=10_000_000, places=2, allow_nan=False
        ),
        year_incurred=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_claim_liability_copy_independence(self, original_amount, year_incurred):
        """Property test: copied claim is always independent."""
        original = ClaimLiability(
            original_amount=to_decimal(original_amount),
            remaining_amount=to_decimal(original_amount),
            year_incurred=year_incurred,
        )

        copied = copy.deepcopy(original)

        # Modify copy
        if copied.remaining_amount > ZERO:
            payment_amount = min(to_decimal(1000), copied.remaining_amount)
            copied.make_payment(payment_amount)

        # Original must be unchanged
        assert original.remaining_amount == to_decimal(original_amount)

    @given(
        initial_assets=st.integers(min_value=1_000_000, max_value=100_000_000),
        n_steps=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=10)
    def test_manufacturer_copy_independence(self, initial_assets, n_steps):
        """Property test: copied manufacturer is always independent."""
        config = ManufacturerConfig(
            initial_assets=initial_assets,
            asset_turnover_ratio=0.8,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        original = WidgetManufacturer(config)

        # Run some steps
        for _ in range(n_steps):
            original.step()

        original_year = original.current_year
        original_history_len = len(original.metrics_history)

        copied = copy.deepcopy(original)

        # Modify copy
        copied.step()

        # Original must be unchanged
        assert original.current_year == original_year
        assert len(original.metrics_history) == original_history_len
