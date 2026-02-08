"""Tests for Loss Adjustment Expense (LAE) tracking per ASC 944-40 (Issue #468).

Validates that LAE is:
- Applied as a configurable ratio on each claim's indemnity
- Tracked separately via period_insurance_lae
- Recorded in LAE_EXPENSE ledger account
- Included in operating income, metrics, and financial statements
- Backward compatible with default lae_ratio=0.12
"""

from decimal import Decimal

import pytest

from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decimal_utils import ZERO, to_decimal
from ergodic_insurance.ledger import AccountName, TransactionType
from ergodic_insurance.manufacturer import WidgetManufacturer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> ManufacturerConfig:
    """Config with default lae_ratio=0.12."""
    return ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=0.8,
        base_operating_margin=0.08,
        tax_rate=0.25,
        retention_ratio=0.7,
    )


@pytest.fixture
def zero_lae_config() -> ManufacturerConfig:
    """Config with lae_ratio=0 (no LAE)."""
    return ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=0.8,
        base_operating_margin=0.08,
        tax_rate=0.25,
        retention_ratio=0.7,
        lae_ratio=0.0,
    )


@pytest.fixture
def custom_lae_config() -> ManufacturerConfig:
    """Config with custom lae_ratio=0.20."""
    return ManufacturerConfig(
        initial_assets=10_000_000,
        asset_turnover_ratio=0.8,
        base_operating_margin=0.08,
        tax_rate=0.25,
        retention_ratio=0.7,
        lae_ratio=0.20,
    )


@pytest.fixture
def manufacturer(default_config: ManufacturerConfig) -> WidgetManufacturer:
    return WidgetManufacturer(default_config)


@pytest.fixture
def manufacturer_zero_lae(zero_lae_config: ManufacturerConfig) -> WidgetManufacturer:
    return WidgetManufacturer(zero_lae_config)


@pytest.fixture
def manufacturer_custom_lae(custom_lae_config: ManufacturerConfig) -> WidgetManufacturer:
    return WidgetManufacturer(custom_lae_config)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestLAEConfig:
    """Tests for LAE configuration field."""

    def test_default_lae_ratio(self, default_config: ManufacturerConfig):
        assert default_config.lae_ratio == 0.12

    def test_zero_lae_ratio(self, zero_lae_config: ManufacturerConfig):
        assert zero_lae_config.lae_ratio == 0.0

    def test_custom_lae_ratio(self, custom_lae_config: ManufacturerConfig):
        assert custom_lae_config.lae_ratio == 0.20

    def test_lae_ratio_validation_max(self):
        with pytest.raises(Exception):
            ManufacturerConfig(lae_ratio=1.5)

    def test_lae_ratio_validation_min(self):
        with pytest.raises(Exception):
            ManufacturerConfig(lae_ratio=-0.1)


# ---------------------------------------------------------------------------
# Insured claim tests
# ---------------------------------------------------------------------------


class TestLAEInsuredClaims:
    """Tests for LAE on insured (process_insurance_claim) claims."""

    def test_lae_applied_to_insured_claim(self, manufacturer: WidgetManufacturer):
        """LAE should be tracked on the company-retained portion."""
        claim_amount = to_decimal(100_000)
        deductible = to_decimal(50_000)
        limit = to_decimal(200_000)

        manufacturer.process_insurance_claim(claim_amount, deductible, limit)

        # Company pays deductible = 50_000.  LAE = 50_000 * 0.12 = 6_000
        expected_lae = to_decimal(50_000) * to_decimal(0.12)
        assert manufacturer.period_insurance_lae == expected_lae

    def test_lae_ledger_entries_insured(self, manufacturer: WidgetManufacturer):
        """LAE_EXPENSE should be debited for insured claims."""
        manufacturer.process_insurance_claim(
            claim_amount=100_000, deductible_amount=50_000, insurance_limit=200_000
        )

        lae_balance = manufacturer.ledger.get_balance(AccountName.LAE_EXPENSE)
        expected = to_decimal(50_000) * to_decimal(0.12)
        assert lae_balance == expected

    def test_lae_zero_when_ratio_zero(self, manufacturer_zero_lae: WidgetManufacturer):
        """No LAE should be recorded when lae_ratio=0."""
        manufacturer_zero_lae.process_insurance_claim(
            claim_amount=100_000, deductible_amount=50_000, insurance_limit=200_000
        )

        assert manufacturer_zero_lae.period_insurance_lae == ZERO
        assert manufacturer_zero_lae.ledger.get_balance(AccountName.LAE_EXPENSE) == ZERO

    def test_lae_custom_ratio(self, manufacturer_custom_lae: WidgetManufacturer):
        """LAE should use the configured custom ratio."""
        manufacturer_custom_lae.process_insurance_claim(
            claim_amount=100_000, deductible_amount=50_000, insurance_limit=200_000
        )

        expected_lae = to_decimal(50_000) * to_decimal(0.20)
        assert manufacturer_custom_lae.period_insurance_lae == expected_lae

    def test_claim_liability_includes_lae(self, manufacturer: WidgetManufacturer):
        """ClaimLiability original_amount should include LAE."""
        manufacturer.process_insurance_claim(
            claim_amount=100_000, deductible_amount=50_000, insurance_limit=200_000
        )

        # The payable claim liability should include indemnity + LAE
        assert len(manufacturer.claim_liabilities) >= 1
        claim = manufacturer.claim_liabilities[0]
        indemnity = to_decimal(50_000)
        lae = indemnity * to_decimal(0.12)
        assert claim.original_amount == indemnity + lae


# ---------------------------------------------------------------------------
# Uninsured claim tests
# ---------------------------------------------------------------------------


class TestLAEUninsuredClaims:
    """Tests for LAE on uninsured (process_uninsured_claim) claims."""

    def test_lae_on_deferred_uninsured_claim(self, manufacturer: WidgetManufacturer):
        """LAE should be applied to deferred uninsured claims."""
        claim_amount = to_decimal(100_000)
        manufacturer.process_uninsured_claim(claim_amount, immediate_payment=False)

        expected_lae = claim_amount * to_decimal(0.12)
        assert manufacturer.period_insurance_lae == expected_lae

    def test_lae_on_immediate_uninsured_claim(self, manufacturer: WidgetManufacturer):
        """LAE should be applied to immediate uninsured claims."""
        claim_amount = to_decimal(50_000)
        manufacturer.process_uninsured_claim(claim_amount, immediate_payment=True)

        # LAE is applied on whatever amount was actually paid
        assert manufacturer.period_insurance_lae > ZERO

    def test_uninsured_claim_liability_includes_lae(self, manufacturer: WidgetManufacturer):
        """Deferred uninsured claim liability should include LAE."""
        claim_amount = to_decimal(100_000)
        manufacturer.process_uninsured_claim(claim_amount, immediate_payment=False)

        assert len(manufacturer.claim_liabilities) >= 1
        claim = manufacturer.claim_liabilities[0]
        expected_total = claim_amount + claim_amount * to_decimal(0.12)
        assert claim.original_amount == expected_total

    def test_uninsured_claim_zero_lae(self, manufacturer_zero_lae: WidgetManufacturer):
        """No LAE for uninsured claims when ratio is 0."""
        manufacturer_zero_lae.process_uninsured_claim(100_000, immediate_payment=False)

        assert manufacturer_zero_lae.period_insurance_lae == ZERO


# ---------------------------------------------------------------------------
# record_claim_accrual tests
# ---------------------------------------------------------------------------


class TestLAEClaimAccrual:
    """Tests for LAE in record_claim_accrual."""

    def test_accrual_includes_lae(self, manufacturer: WidgetManufacturer):
        """record_claim_accrual should inflate amount by LAE ratio."""
        amount = to_decimal(200_000)
        manufacturer.record_claim_accrual(amount)

        assert len(manufacturer.claim_liabilities) == 1
        claim = manufacturer.claim_liabilities[0]
        expected = amount + amount * to_decimal(0.12)
        assert claim.original_amount == expected

    def test_accrual_lae_ledger_entry(self, manufacturer: WidgetManufacturer):
        """record_claim_accrual should create LAE_EXPENSE ledger entry."""
        amount = to_decimal(200_000)
        manufacturer.record_claim_accrual(amount)

        lae_balance = manufacturer.ledger.get_balance(AccountName.LAE_EXPENSE)
        expected_lae = amount * to_decimal(0.12)
        assert lae_balance == expected_lae

    def test_accrual_lae_period_tracking(self, manufacturer: WidgetManufacturer):
        """record_claim_accrual should track LAE in period_insurance_lae."""
        amount = to_decimal(200_000)
        manufacturer.record_claim_accrual(amount)

        expected_lae = amount * to_decimal(0.12)
        assert manufacturer.period_insurance_lae == expected_lae


# ---------------------------------------------------------------------------
# Operating income tests
# ---------------------------------------------------------------------------


class TestLAEOperatingIncome:
    """Tests for LAE in operating income calculation."""

    def test_lae_deducted_from_operating_income(self, manufacturer: WidgetManufacturer):
        """LAE should reduce operating income."""
        revenue = manufacturer.calculate_revenue()
        base_income = revenue * to_decimal(manufacturer.base_operating_margin)

        # Process a claim to generate LAE
        manufacturer.process_insurance_claim(
            claim_amount=100_000, deductible_amount=50_000, insurance_limit=200_000
        )

        operating_income = manufacturer.calculate_operating_income(revenue)

        # Operating income should be reduced by premiums + losses + LAE
        total_insurance = (
            manufacturer.period_insurance_premiums
            + manufacturer.period_insurance_losses
            + manufacturer.period_insurance_lae
        )
        expected = base_income - total_insurance
        assert operating_income == expected

    def test_lae_zero_does_not_affect_income(self, manufacturer_zero_lae: WidgetManufacturer):
        """With lae_ratio=0, operating income should not change from LAE."""
        revenue = manufacturer_zero_lae.calculate_revenue()
        income_before = manufacturer_zero_lae.calculate_operating_income(revenue)

        manufacturer_zero_lae.process_insurance_claim(
            claim_amount=100_000, deductible_amount=50_000, insurance_limit=200_000
        )

        income_after = manufacturer_zero_lae.calculate_operating_income(revenue)

        # The difference should only be from insurance losses, not LAE
        assert manufacturer_zero_lae.period_insurance_lae == ZERO
        expected_diff = manufacturer_zero_lae.period_insurance_losses
        assert income_before - income_after == expected_diff


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestLAEMetrics:
    """Tests for LAE in metrics calculation."""

    def test_insurance_lae_in_metrics(self, manufacturer: WidgetManufacturer):
        """Metrics should include insurance_lae field."""
        manufacturer.process_insurance_claim(
            claim_amount=100_000, deductible_amount=50_000, insurance_limit=200_000
        )

        metrics = manufacturer.calculate_metrics()

        assert "insurance_lae" in metrics
        expected_lae = to_decimal(50_000) * to_decimal(0.12)
        assert metrics["insurance_lae"] == expected_lae

    def test_lae_in_total_insurance_costs(self, manufacturer: WidgetManufacturer):
        """total_insurance_costs should include LAE."""
        manufacturer.process_insurance_claim(
            claim_amount=100_000, deductible_amount=50_000, insurance_limit=200_000
        )

        metrics = manufacturer.calculate_metrics()

        expected_total = (
            to_decimal(metrics["insurance_premiums"])
            + to_decimal(metrics["insurance_losses"])
            + to_decimal(metrics["insurance_lae"])
        )
        assert metrics["total_insurance_costs"] == expected_total


# ---------------------------------------------------------------------------
# Financial statement tests
# ---------------------------------------------------------------------------


class TestLAEFinancialStatements:
    """Tests for LAE in financial statement presentation."""

    def test_lae_appears_in_income_statement(self, manufacturer: WidgetManufacturer):
        """LAE line item should appear in the income statement."""
        from ergodic_insurance.financial_statements import FinancialStatementGenerator

        # Run a step with a claim (step populates full metrics including COGS breakdown)
        manufacturer.process_insurance_claim(
            claim_amount=100_000, deductible_amount=50_000, insurance_limit=200_000
        )
        metrics = manufacturer.step(letter_of_credit_rate=0.015)

        # Verify the metrics contain insurance_lae > 0 before generating statement
        assert metrics.get("insurance_lae", ZERO) > ZERO

        generator = FinancialStatementGenerator(manufacturer)
        # Year is 1 because step() incremented from 0 to 1; the metrics were for year 0
        statement = generator.generate_income_statement(year=0)

        # Check that LAE line item is in the statement text
        statement_text = str(statement)
        assert "Loss Adjustment Expenses" in statement_text or "LAE" in statement_text


# ---------------------------------------------------------------------------
# Reset and period boundary tests
# ---------------------------------------------------------------------------


class TestLAEReset:
    """Tests for LAE reset behavior."""

    def test_period_lae_reset(self, manufacturer: WidgetManufacturer):
        """period_insurance_lae should reset between periods."""
        manufacturer.process_insurance_claim(
            claim_amount=100_000, deductible_amount=50_000, insurance_limit=200_000
        )
        assert manufacturer.period_insurance_lae > ZERO

        manufacturer.reset_period_insurance_costs()
        assert manufacturer.period_insurance_lae == ZERO

    def test_full_reset(self, manufacturer: WidgetManufacturer):
        """Full manufacturer reset should clear period_insurance_lae."""
        manufacturer.process_insurance_claim(
            claim_amount=100_000, deductible_amount=50_000, insurance_limit=200_000
        )
        assert manufacturer.period_insurance_lae > ZERO

        manufacturer.reset()
        assert manufacturer.period_insurance_lae == ZERO


# ---------------------------------------------------------------------------
# Ledger balance tests
# ---------------------------------------------------------------------------


class TestLAELedgerEntries:
    """Tests for LAE ledger entry correctness."""

    def test_double_entry_balance(self, manufacturer: WidgetManufacturer):
        """Ledger should remain balanced after LAE entries."""
        manufacturer.process_insurance_claim(
            claim_amount=100_000, deductible_amount=50_000, insurance_limit=200_000
        )

        balanced, diff = manufacturer.ledger.verify_balance()
        assert balanced, f"Ledger out of balance by {diff}"

    def test_lae_entries_credit_claim_liabilities(self, manufacturer: WidgetManufacturer):
        """LAE entries should credit CLAIM_LIABILITIES."""
        manufacturer.process_insurance_claim(
            claim_amount=100_000, deductible_amount=50_000, insurance_limit=200_000
        )

        # Check that claim liabilities include the LAE component
        claim_liabilities = manufacturer.ledger.get_balance(AccountName.CLAIM_LIABILITIES)
        lae_expense = manufacturer.ledger.get_balance(AccountName.LAE_EXPENSE)

        # LAE expense should be positive (debited)
        assert lae_expense > ZERO
        # Claim liabilities should include both indemnity and LAE portions
        assert claim_liabilities > ZERO

    def test_lae_transaction_type(self, manufacturer: WidgetManufacturer):
        """LAE entries should use INSURANCE_CLAIM transaction type."""
        manufacturer.process_insurance_claim(
            claim_amount=100_000, deductible_amount=50_000, insurance_limit=200_000
        )

        lae_entries = manufacturer.ledger.get_entries(
            account=AccountName.LAE_EXPENSE,
        )
        assert len(lae_entries) > 0
        for entry in lae_entries:
            assert entry.transaction_type == TransactionType.INSURANCE_CLAIM


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------


class TestLAEBackwardCompatibility:
    """Tests that default LAE doesn't break existing flows."""

    def test_step_with_default_lae(self, manufacturer: WidgetManufacturer):
        """A full step with claims should succeed with default LAE."""
        manufacturer.process_insurance_claim(
            claim_amount=200_000, deductible_amount=100_000, insurance_limit=500_000
        )
        metrics = manufacturer.step(letter_of_credit_rate=0.015)

        assert "insurance_lae" in metrics
        assert metrics["is_solvent"]

    def test_step_with_zero_lae(self, manufacturer_zero_lae: WidgetManufacturer):
        """A full step should work with lae_ratio=0."""
        manufacturer_zero_lae.process_insurance_claim(
            claim_amount=200_000, deductible_amount=100_000, insurance_limit=500_000
        )
        metrics = manufacturer_zero_lae.step(letter_of_credit_rate=0.015)

        assert metrics["insurance_lae"] == ZERO
        assert metrics["is_solvent"]

    def test_no_claims_no_lae(self, manufacturer: WidgetManufacturer):
        """Without claims, LAE should be zero even with default ratio."""
        metrics = manufacturer.step(letter_of_credit_rate=0.015)

        assert metrics["insurance_lae"] == ZERO

    def test_import_works(self):
        """Verify the import chain doesn't break."""
        config = ManufacturerConfig()
        mfr = WidgetManufacturer(config)
        assert hasattr(mfr, "period_insurance_lae")
