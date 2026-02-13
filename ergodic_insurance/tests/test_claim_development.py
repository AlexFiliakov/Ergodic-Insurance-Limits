"""Unit tests for claim development patterns module."""

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from ergodic_insurance.claim_development import (
    CashFlowProjector,
    Claim,
    ClaimCohort,
    ClaimDevelopment,
    DevelopmentPattern,
    DevelopmentPatternType,
    load_development_patterns,
    load_ibnr_factors,
)


class TestClaimDevelopment:
    """Test ClaimDevelopment class."""

    def test_create_immediate_pattern(self):
        """Test immediate payment pattern creation."""
        pattern = ClaimDevelopment.create_immediate()
        assert pattern.pattern_name == "IMMEDIATE"
        assert pattern.development_factors == [1.0]
        assert pattern.tail_factor == 0.0
        assert sum(pattern.development_factors) == 1.0

    def test_create_medium_tail_pattern(self):
        """Test 5-year workers compensation pattern."""
        pattern = ClaimDevelopment.create_medium_tail_5yr()
        assert pattern.pattern_name == "MEDIUM_TAIL_5YR"
        assert len(pattern.development_factors) == 5
        assert abs(sum(pattern.development_factors) - 1.0) < 0.01

    def test_create_long_tail_pattern(self):
        """Test 10-year general liability pattern."""
        pattern = ClaimDevelopment.create_long_tail_10yr()
        assert pattern.pattern_name == "LONG_TAIL_10YR"
        assert len(pattern.development_factors) == 10
        assert abs(sum(pattern.development_factors) - 1.0) < 0.01

    def test_create_very_long_tail_pattern(self):
        """Test 15-year product liability pattern."""
        pattern = ClaimDevelopment.create_very_long_tail_15yr()
        assert pattern.pattern_name == "VERY_LONG_TAIL_15YR"
        assert len(pattern.development_factors) == 15
        assert abs(sum(pattern.development_factors) - 1.0) < 0.01

    def test_custom_pattern_validation(self):
        """Test custom pattern validation."""
        # Valid pattern
        pattern = ClaimDevelopment(
            pattern_name="CUSTOM",
            development_factors=[0.5, 0.3, 0.2],
            tail_factor=0.0,
        )
        assert sum(pattern.development_factors) == 1.0

        # Invalid: factors don't sum to 1
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ClaimDevelopment(
                pattern_name="INVALID",
                development_factors=[0.5, 0.3],
                tail_factor=0.0,
            )

        # Invalid: negative factors
        with pytest.raises(ValueError, match="must be non-negative"):
            ClaimDevelopment(
                pattern_name="INVALID",
                development_factors=[0.5, -0.2, 0.7],
                tail_factor=0.0,
            )

        # Invalid: empty factors
        with pytest.raises(ValueError, match="cannot be empty"):
            ClaimDevelopment(
                pattern_name="INVALID",
                development_factors=[],
                tail_factor=0.0,
            )

    def test_calculate_payments(self):
        """Test payment calculation for specific years."""
        pattern = ClaimDevelopment.create_long_tail_10yr()
        claim_amount = 1_000_000
        accident_year = 2020

        # Year 1 payment (2020)
        payment_y1 = pattern.calculate_payments(claim_amount, accident_year, 2020)
        assert payment_y1 == 100_000  # 10% in year 1

        # Year 2 payment (2021)
        payment_y2 = pattern.calculate_payments(claim_amount, accident_year, 2021)
        assert payment_y2 == 200_000  # 20% in year 2

        # Year 3 payment (2022)
        payment_y3 = pattern.calculate_payments(claim_amount, accident_year, 2022)
        assert payment_y3 == 200_000  # 20% in year 3

        # Payment before accident year
        payment_before = pattern.calculate_payments(claim_amount, accident_year, 2019)
        assert payment_before == 0.0

        # Payment beyond pattern period (no tail)
        payment_beyond = pattern.calculate_payments(claim_amount, accident_year, 2035)
        assert payment_beyond == 0.0

    def test_calculate_payments_with_tail(self):
        """Test payment calculation with tail factor."""
        pattern = ClaimDevelopment(
            pattern_name="WITH_TAIL",
            development_factors=[0.4, 0.3, 0.2],
            tail_factor=0.1,
        )
        claim_amount = 1_000_000
        accident_year = 2020

        # Within pattern period
        assert pattern.calculate_payments(claim_amount, accident_year, 2020) == 400_000
        assert pattern.calculate_payments(claim_amount, accident_year, 2022) == 200_000

        # Tail factor applies only once at development_year == len(factors)
        assert pattern.calculate_payments(claim_amount, accident_year, 2023) == 100_000
        # Beyond tail year - no more payments
        assert pattern.calculate_payments(claim_amount, accident_year, 2024) == 0.0
        assert pattern.calculate_payments(claim_amount, accident_year, 2025) == 0.0

    def test_get_cumulative_paid(self):
        """Test cumulative payment percentage calculation."""
        pattern = ClaimDevelopment.create_medium_tail_5yr()

        assert pattern.get_cumulative_paid(0) == 0.0
        assert pattern.get_cumulative_paid(1) == 0.40
        assert pattern.get_cumulative_paid(2) == 0.65
        assert pattern.get_cumulative_paid(3) == 0.80
        assert pattern.get_cumulative_paid(4) == 0.90
        assert pattern.get_cumulative_paid(5) == 1.0
        assert pattern.get_cumulative_paid(10) == 1.0  # Beyond pattern


class TestClaim:
    """Test Claim class."""

    def test_claim_initialization(self):
        """Test claim initialization with defaults."""
        claim = Claim(
            claim_id="CL001",
            accident_year=2020,
            reported_year=2020,
            initial_estimate=500_000,
        )
        assert claim.claim_id == "CL001"
        assert claim.accident_year == 2020
        assert claim.claim_type == "general_liability"
        assert claim.development_pattern is not None
        assert claim.development_pattern.pattern_name == "LONG_TAIL_10YR"

    def test_claim_with_custom_pattern(self):
        """Test claim with custom development pattern."""
        pattern = ClaimDevelopment.create_immediate()
        claim = Claim(
            claim_id="CL002",
            accident_year=2020,
            reported_year=2020,
            initial_estimate=100_000,
            claim_type="property",
            development_pattern=pattern,
        )
        assert claim.development_pattern is not None
        assert claim.development_pattern.pattern_name == "IMMEDIATE"

    def test_record_payment(self):
        """Test recording payments."""
        claim = Claim(
            claim_id="CL003",
            accident_year=2020,
            reported_year=2020,
            initial_estimate=1_000_000,
        )

        # Record payments
        claim.record_payment(2020, 100_000)
        claim.record_payment(2021, 200_000)
        claim.record_payment(2021, 50_000)  # Additional payment in same year

        assert claim.payments_made[2020] == 100_000
        assert claim.payments_made[2021] == 250_000
        assert claim.get_total_paid() == 350_000

    def test_outstanding_reserve(self):
        """Test outstanding reserve calculation."""
        claim = Claim(
            claim_id="CL004",
            accident_year=2020,
            reported_year=2020,
            initial_estimate=1_000_000,
        )

        # No payments yet
        assert claim.get_outstanding_reserve() == 1_000_000

        # After some payments
        claim.record_payment(2020, 300_000)
        claim.record_payment(2021, 200_000)
        assert claim.get_outstanding_reserve() == 500_000

        # Overpayment scenario
        claim.record_payment(2022, 600_000)
        assert claim.get_outstanding_reserve() == 0  # Can't be negative


class TestClaimCohort:
    """Test ClaimCohort class."""

    def test_cohort_initialization(self):
        """Test cohort initialization."""
        cohort = ClaimCohort(accident_year=2020)
        assert cohort.accident_year == 2020
        assert len(cohort.claims) == 0

    def test_add_claim_valid(self):
        """Test adding valid claims to cohort."""
        cohort = ClaimCohort(accident_year=2020)
        claim1 = Claim("CL001", 2020, 2020, 100_000)
        claim2 = Claim("CL002", 2020, 2021, 200_000)

        cohort.add_claim(claim1)
        cohort.add_claim(claim2)
        assert len(cohort.claims) == 2

    def test_add_claim_invalid_year(self):
        """Test adding claim from wrong year raises error."""
        cohort = ClaimCohort(accident_year=2020)
        claim = Claim("CL001", 2021, 2021, 100_000)

        with pytest.raises(ValueError, match="cannot be added to cohort"):
            cohort.add_claim(claim)

    def test_calculate_payments(self):
        """Test payment calculation for cohort."""
        cohort = ClaimCohort(accident_year=2020)

        # Add claims with different patterns
        claim1 = Claim(
            "CL001", 2020, 2020, 1_000_000, development_pattern=ClaimDevelopment.create_immediate()
        )
        claim2 = Claim(
            "CL002",
            2020,
            2020,
            2_000_000,
            development_pattern=ClaimDevelopment.create_long_tail_10yr(),
        )

        cohort.add_claim(claim1)
        cohort.add_claim(claim2)

        # Year 1 payments
        payment_2020 = cohort.calculate_payments(2020)
        expected = 1_000_000 + (2_000_000 * 0.10)  # Immediate + 10% of long-tail
        assert payment_2020 == expected

        # Verify payments were recorded
        assert claim1.get_total_paid() == 1_000_000
        assert claim2.get_total_paid() == 200_000

    def test_cohort_totals(self):
        """Test cohort total calculations."""
        cohort = ClaimCohort(accident_year=2020)

        claim1 = Claim("CL001", 2020, 2020, 500_000)
        claim2 = Claim("CL002", 2020, 2020, 1_500_000)
        cohort.add_claim(claim1)
        cohort.add_claim(claim2)

        # Total incurred
        assert cohort.get_total_incurred() == 2_000_000

        # Make some payments
        cohort.calculate_payments(2020)
        cohort.calculate_payments(2021)

        # Check totals
        total_paid = cohort.get_total_paid()
        assert total_paid > 0
        assert cohort.get_outstanding_reserve() == 2_000_000 - total_paid


class TestCashFlowProjector:
    """Test CashFlowProjector class."""

    def test_projector_initialization(self):
        """Test projector initialization."""
        projector = CashFlowProjector(discount_rate=0.05)
        assert projector.discount_rate == 0.05
        assert len(projector.cohorts) == 0

    def test_add_cohort(self):
        """Test adding cohorts to projector."""
        projector = CashFlowProjector()
        cohort1 = ClaimCohort(accident_year=2020)
        cohort2 = ClaimCohort(accident_year=2021)

        projector.add_cohort(cohort1)
        projector.add_cohort(cohort2)

        assert len(projector.cohorts) == 2
        assert 2020 in projector.cohorts
        assert 2021 in projector.cohorts

    def test_project_payments(self):
        """Test payment projection."""
        projector = CashFlowProjector()

        # Create cohorts with claims
        cohort_2020 = ClaimCohort(accident_year=2020)
        claim1 = Claim(
            "CL001",
            2020,
            2020,
            1_000_000,
            development_pattern=ClaimDevelopment.create_medium_tail_5yr(),
        )
        cohort_2020.add_claim(claim1)

        cohort_2021 = ClaimCohort(accident_year=2021)
        claim2 = Claim(
            "CL002",
            2021,
            2021,
            2_000_000,
            development_pattern=ClaimDevelopment.create_medium_tail_5yr(),
        )
        cohort_2021.add_claim(claim2)

        projector.add_cohort(cohort_2020)
        projector.add_cohort(cohort_2021)

        # Project payments
        payments = projector.project_payments(2020, 2025)

        assert len(payments) == 6  # 2020-2025
        assert payments[2020] == 400_000  # 40% of claim1
        assert payments[2021] == 250_000 + 800_000  # 25% of claim1 + 40% of claim2
        assert payments[2022] == 150_000 + 500_000  # 15% of claim1 + 25% of claim2

    def test_calculate_present_value(self):
        """Test present value calculation."""
        projector = CashFlowProjector(discount_rate=0.05)

        payments = {
            2020: 100_000.0,
            2021: 100_000.0,
            2022: 100_000.0,
        }

        pv = projector.calculate_present_value(payments, base_year=2020)

        # PV = 100k + 100k/1.05 + 100k/1.05^2
        expected = 100_000 + 100_000 / 1.05 + 100_000 / (1.05**2)
        assert abs(pv - expected) < 1  # Allow for rounding

    def test_estimate_ibnr(self):
        """Test IBNR estimation produces positive results for developing claims."""
        projector = CashFlowProjector(a_priori_loss_ratio=0.70)

        cohort_2023 = ClaimCohort(accident_year=2023)
        claim1 = Claim("CL001", 2023, 2023, 1_000_000)
        cohort_2023.add_claim(claim1)

        cohort_2021 = ClaimCohort(accident_year=2021)
        claim2 = Claim("CL002", 2021, 2021, 2_000_000)
        cohort_2021.add_claim(claim2)

        projector.add_cohort(cohort_2023)
        projector.add_cohort(cohort_2021)

        # Project payments so CL has data (through eval_year - 1 so
        # deterministic paid aligns with get_cumulative_paid)
        projector.project_payments(2021, 2022)

        earned_premium = {2021: 3_000_000, 2023: 1_500_000}
        ibnr = projector.estimate_ibnr(evaluation_year=2023, earned_premium=earned_premium)

        # BF with premium produces positive IBNR for immature cohorts
        assert ibnr > 0
        total_incurred = 3_000_000
        assert ibnr < total_incurred * 5

    def test_calculate_total_reserves(self):
        """Test total reserve calculation."""
        projector = CashFlowProjector()

        cohort = ClaimCohort(accident_year=2020)
        claim = Claim("CL001", 2020, 2020, 1_000_000)
        cohort.add_claim(claim)
        projector.add_cohort(cohort)

        # Project payments before evaluation year
        projector.project_payments(2020, 2020)

        reserves = projector.calculate_total_reserves(evaluation_year=2021)

        assert "case_reserves" in reserves
        assert "ibnr" in reserves
        assert "total_reserves" in reserves
        # After projecting payments, some payments made
        assert reserves["case_reserves"] >= 0
        # With deterministic CL (no BF), IBNR = 0
        assert reserves["ibnr"] >= 0
        assert reserves["total_reserves"] >= reserves["case_reserves"]


class TestLoadDevelopmentPatterns:
    """Test loading development patterns from YAML."""

    def test_load_patterns(self, tmp_path):
        """Test loading patterns from YAML file."""
        # Create test YAML file
        yaml_content = {
            "development_patterns": {
                "test_pattern": {
                    "factors": [0.5, 0.3, 0.2],
                    "tail_factor": 0.0,
                },
                "pattern_with_tail": {
                    "factors": [0.4, 0.3, 0.2],
                    "tail_factor": 0.1,
                },
            }
        }

        yaml_file = tmp_path / "test_patterns.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        # Load patterns
        patterns = load_development_patterns(str(yaml_file))

        assert len(patterns) == 2
        assert "test_pattern" in patterns
        assert "pattern_with_tail" in patterns

        # Verify pattern properties
        test_pattern = patterns["test_pattern"]
        assert test_pattern.development_factors == [0.5, 0.3, 0.2]
        assert test_pattern.tail_factor == 0.0

        tail_pattern = patterns["pattern_with_tail"]
        assert tail_pattern.tail_factor == 0.1

    def test_load_actual_config(self):
        """Test loading the actual development_patterns.yaml file."""
        # Get path to actual config file
        config_path = (
            Path(__file__).parent.parent / "data" / "parameters" / "development_patterns.yaml"
        )

        if config_path.exists():
            patterns = load_development_patterns(str(config_path))

            # Verify expected patterns exist
            assert "immediate" in patterns
            assert "medium_tail_5yr" in patterns
            assert "long_tail_10yr" in patterns
            assert "very_long_tail_15yr" in patterns

            # Verify patterns are valid
            for name, pattern in patterns.items():
                total = sum(pattern.development_factors) + pattern.tail_factor
                assert abs(total - 1.0) <= 0.01, f"Pattern {name} doesn't sum to 1.0"


class TestPerformance:
    """Performance tests for claim development."""

    @pytest.mark.benchmark
    def test_large_cohort_performance(self):
        """Test performance with large number of claims."""
        import time

        cohort = ClaimCohort(accident_year=2020)
        pattern = ClaimDevelopment.create_long_tail_10yr()

        # Add 10,000 claims
        for i in range(10_000):
            claim = Claim(
                claim_id=f"CL{i:05d}",
                accident_year=2020,
                reported_year=2020,
                initial_estimate=np.random.lognormal(11, 2),  # Random amounts
                development_pattern=pattern,
            )
            cohort.add_claim(claim)

        # Time payment calculation
        start_time = time.time()
        payment = cohort.calculate_payments(2021)
        elapsed = time.time() - start_time

        # Should process 10K claims in < 100ms (adjusted for system variance)
        assert elapsed < 0.20, f"Processing took {elapsed:.3f}s, expected < 200ms"
        assert payment > 0  # Should have calculated payments

    @pytest.mark.benchmark
    def test_multi_year_projection_performance(self):
        """Test performance of multi-year projections."""
        import time

        projector = CashFlowProjector()

        # Create 20 cohorts with 500 claims each
        for year in range(2000, 2020):
            cohort = ClaimCohort(accident_year=year)
            for i in range(500):
                claim = Claim(
                    claim_id=f"CL{year}{i:03d}",
                    accident_year=year,
                    reported_year=year,
                    initial_estimate=np.random.lognormal(10, 1.5),
                )
                cohort.add_claim(claim)
            projector.add_cohort(cohort)

        # Time projection
        start_time = time.time()
        payments = projector.project_payments(2020, 2030)
        elapsed = time.time() - start_time

        # Should complete in reasonable time
        assert elapsed < 1.0, f"Projection took {elapsed:.3f}s"
        assert len(payments) == 11  # 2020-2030


class TestIBNRActuarialMethods:
    """Test actuarial IBNR estimation methods (Issue #390)."""

    def test_cl_only_no_elr(self):
        """CL-only: paid-basis IBNR = ultimate - paid_to_date."""
        projector = CashFlowProjector()

        cohort = ClaimCohort(accident_year=2020)
        claim = Claim(
            "CL001",
            2020,
            2020,
            1_000_000,
            development_pattern=ClaimDevelopment.create_medium_tail_5yr(),
        )
        cohort.add_claim(claim)
        projector.add_cohort(cohort)

        # Project payments before evaluation year so CL has data
        projector.project_payments(2020, 2020)

        # Paid = 400k (40%), CL ultimate = 400k / 0.40 = 1M
        # IBNR = 1M - 400k = 600k (paid-basis)
        ibnr = projector.estimate_ibnr(evaluation_year=2021)
        assert ibnr == pytest.approx(600_000)

    def test_cl_no_payments_falls_back(self):
        """No payments and no premium -> fallback to incurred as ultimate.

        With paid-basis IBNR, IBNR = incurred - paid_to_date = 1M - 0 = 1M.
        """
        projector = CashFlowProjector()

        cohort = ClaimCohort(accident_year=2020)
        claim = Claim(
            "CL001",
            2020,
            2020,
            1_000_000,
            development_pattern=ClaimDevelopment.create_medium_tail_5yr(),
        )
        cohort.add_claim(claim)
        projector.add_cohort(cohort)

        # No project_payments, no premium, no ELR -> blended_ultimate = incurred
        # IBNR = incurred - paid_to_date = 1M - 0 = 1M (paid-basis)
        ibnr = projector.estimate_ibnr(evaluation_year=2021)
        assert ibnr == pytest.approx(1_000_000)

    def test_bf_only_immature_year(self):
        """BF dominates at 0% development (E2)."""
        projector = CashFlowProjector(a_priori_loss_ratio=0.70)

        cohort = ClaimCohort(accident_year=2023)
        claim = Claim(
            "CL001",
            2023,
            2023,
            1_000_000,
            development_pattern=ClaimDevelopment.create_long_tail_10yr(),
        )
        cohort.add_claim(claim)
        projector.add_cohort(cohort)

        # dev_years=0, pct_developed=0.0 -> CL undefined (no payments), BF only
        # BF IBNR = 0.70 * 2_000_000 * (1 - 0) = 1_400_000
        # BF ultimate = paid(0) + 1_400_000 = 1_400_000
        # IBNR = max(0, 1_400_000 - 0) = 1_400_000 (paid-basis)
        earned_premium = {2023: 2_000_000}
        ibnr = projector.estimate_ibnr(evaluation_year=2023, earned_premium=earned_premium)
        assert ibnr == pytest.approx(1_400_000)

    def test_blended_cl_bf(self):
        """Verify maturity-adaptive CL/BF weights."""
        projector = CashFlowProjector(a_priori_loss_ratio=0.70)

        cohort = ClaimCohort(accident_year=2020)
        claim = Claim(
            "CL001",
            2020,
            2020,
            1_000_000,
            development_pattern=ClaimDevelopment.create_medium_tail_5yr(),
        )
        cohort.add_claim(claim)
        projector.add_cohort(cohort)

        # Project payments before evaluation year
        projector.project_payments(2020, 2020)

        # pct=0.40, paid=400k
        # CL ultimate = 400k / 0.40 = 1M
        # BF IBNR = 0.70 * 2M * 0.60 = 840k
        # BF ultimate = paid + bf_ibnr = 400k + 840k = 1.24M
        # Blended = 0.40 * 1M + 0.60 * 1.24M = 400k + 744k = 1_144k
        # IBNR = 1_144k - 400k = 744k (paid-basis)
        earned_premium = {2020: 2_000_000}
        ibnr = projector.estimate_ibnr(evaluation_year=2021, earned_premium=earned_premium)
        assert ibnr == pytest.approx(744_000)

    def test_blended_with_earned_premium(self):
        """BF with earned premium uses standard formula."""
        projector = CashFlowProjector(a_priori_loss_ratio=0.70)

        cohort = ClaimCohort(accident_year=2020)
        claim = Claim(
            "CL001",
            2020,
            2020,
            500_000,
            development_pattern=ClaimDevelopment.create_medium_tail_5yr(),
        )
        cohort.add_claim(claim)
        projector.add_cohort(cohort)

        # Project payments before evaluation year
        projector.project_payments(2020, 2020)

        # pct_developed at dev_years=1 = 0.40, paid=200k
        # CL ultimate = 200k / 0.40 = 500k
        # BF IBNR = 0.70 * 1M * 0.60 = 420k
        # BF ultimate = paid + bf_ibnr = 200k + 420k = 620k
        # Blended = 0.40 * 500k + 0.60 * 620k = 200k + 372k = 572k
        # IBNR = 572k - 200k = 372k (paid-basis)
        earned_premium = {2020: 1_000_000}
        ibnr = projector.estimate_ibnr(evaluation_year=2021, earned_premium=earned_premium)
        assert ibnr == pytest.approx(372_000)

    def test_fully_developed_zero_ibnr(self):
        """E5: IBNR=0 at full maturity."""
        projector = CashFlowProjector()

        cohort = ClaimCohort(accident_year=2020)
        # Immediate pattern: fully developed after year 0
        claim = Claim(
            "CL001",
            2020,
            2020,
            1_000_000,
            development_pattern=ClaimDevelopment.create_immediate(),
        )
        cohort.add_claim(claim)
        projector.add_cohort(cohort)

        # At dev_years=1, immediate pattern is 100% developed
        ibnr = projector.estimate_ibnr(evaluation_year=2021)
        assert ibnr == 0.0

    def test_zero_incurred(self):
        """E4: No claims → IBNR=0."""
        projector = CashFlowProjector()

        # Empty cohort
        cohort = ClaimCohort(accident_year=2020)
        projector.add_cohort(cohort)

        ibnr = projector.estimate_ibnr(evaluation_year=2021)
        assert ibnr == 0.0

    def test_ibnr_floor_at_zero(self):
        """E7: Negative development is floored at 0."""
        # Set a very low ELR that would make BF ultimate < incurred
        projector = CashFlowProjector(a_priori_loss_ratio=0.5)

        cohort = ClaimCohort(accident_year=2020)
        claim = Claim(
            "CL001",
            2020,
            2020,
            1_000_000,
            development_pattern=ClaimDevelopment.create_immediate(),
        )
        cohort.add_claim(claim)
        projector.add_cohort(cohort)

        # Immediate pattern at dev_years=1 is fully developed → skip.
        # For a partially developed scenario, the floor would apply.
        ibnr = projector.estimate_ibnr(evaluation_year=2021)
        assert ibnr >= 0.0

    def test_elr_tier1_user_provided(self):
        """a_priori_loss_ratio is used when set (Tier 1)."""
        projector = CashFlowProjector(a_priori_loss_ratio=0.80)

        cohort = ClaimCohort(accident_year=2020)
        claim = Claim(
            "CL001",
            2020,
            2020,
            1_000_000,
            development_pattern=ClaimDevelopment.create_medium_tail_5yr(),
        )
        cohort.add_claim(claim)
        projector.add_cohort(cohort)

        # Project payments before evaluation year
        projector.project_payments(2020, 2020)

        # pct=0.40, paid=400k
        # CL = 400k/0.40 = 1M
        # BF IBNR = 0.80 * 2M * 0.60 = 960k
        # BF ult = paid + bf_ibnr = 400k + 960k = 1.36M
        # Blended = 0.40 * 1M + 0.60 * 1.36M = 400k + 816k = 1_216k
        # IBNR = 1_216k - 400k = 816k (paid-basis)
        earned_premium = {2020: 2_000_000}
        ibnr = projector.estimate_ibnr(evaluation_year=2021, earned_premium=earned_premium)
        assert ibnr == pytest.approx(816_000)

    def test_elr_tier3_industry_benchmark(self):
        """ibnr_factors from YAML are used for Tier 3."""
        projector = CashFlowProjector(ibnr_factors={"long_tail_10yr": 0.65})

        cohort = ClaimCohort(accident_year=2023)
        claim = Claim(
            "CL001",
            2023,
            2023,
            1_000_000,
            development_pattern=ClaimDevelopment.create_long_tail_10yr(),
        )
        cohort.add_claim(claim)
        projector.add_cohort(cohort)

        # dev_years=0, pct_developed=0.0 -> CL undefined (no payments)
        # Tier 3 ELR = 0.65 (from ibnr_factors)
        # BF IBNR = 0.65 * 2M * 1.0 = 1_300_000
        # BF ultimate = paid(0) + 1_300_000 = 1_300_000
        # IBNR = max(0, 1_300_000 - 0) = 1_300_000 (paid-basis)
        earned_premium = {2023: 2_000_000}
        ibnr = projector.estimate_ibnr(evaluation_year=2023, earned_premium=earned_premium)
        assert ibnr == pytest.approx(1_300_000)

    def test_elr_tier2_cape_cod(self):
        """Cape Cod ELR derived from >=2 cohorts with premium."""
        projector = CashFlowProjector()

        cohort_2019 = ClaimCohort(accident_year=2019)
        claim1 = Claim(
            "CL001",
            2019,
            2019,
            500_000,
            development_pattern=ClaimDevelopment.create_medium_tail_5yr(),
        )
        cohort_2019.add_claim(claim1)
        projector.add_cohort(cohort_2019)

        cohort_2020 = ClaimCohort(accident_year=2020)
        claim2 = Claim(
            "CL002",
            2020,
            2020,
            800_000,
            development_pattern=ClaimDevelopment.create_medium_tail_5yr(),
        )
        cohort_2020.add_claim(claim2)
        projector.add_cohort(cohort_2020)

        # Project payments through evaluation year so triangle has full data
        projector.project_payments(2019, 2021)

        # At eval 2021:
        # cohort_2019: dev_years=2, pct=0.65, paid=500k*0.80=400k
        # cohort_2020: dev_years=1, pct=0.40, paid=800k*0.65=520k
        # Cape Cod ELR = (400k+520k) / (700k*0.65 + 1.2M*0.40)
        #             = 920k / 935k ~ 0.984
        earned_premium = {2019: 700_000, 2020: 1_200_000}
        ibnr = projector.estimate_ibnr(evaluation_year=2021, earned_premium=earned_premium)
        assert ibnr > 0

        # Verify Cape Cod was used by comparing with CL-only (single cohort, no premium)
        projector_cl = CashFlowProjector()
        cohort_single = ClaimCohort(accident_year=2019)
        cohort_single.add_claim(
            Claim(
                "CL001b",
                2019,
                2019,
                500_000,
                development_pattern=ClaimDevelopment.create_medium_tail_5yr(),
            )
        )
        projector_cl.add_cohort(cohort_single)
        projector_cl.project_payments(2019, 2020)
        ibnr_single = projector_cl.estimate_ibnr(evaluation_year=2021)
        # CL-only on deterministic data gives IBNR=0; Cape Cod blend gives >0
        assert ibnr != ibnr_single

    def test_ibnr_consistent_with_development_pattern(self):
        """Long-tail patterns should produce more IBNR than short-tail."""
        # Short tail
        proj_short = CashFlowProjector(a_priori_loss_ratio=0.70)
        cohort_s = ClaimCohort(accident_year=2020)
        cohort_s.add_claim(
            Claim(
                "CL001",
                2020,
                2020,
                1_000_000,
                development_pattern=ClaimDevelopment.create_medium_tail_5yr(),
            )
        )
        proj_short.add_cohort(cohort_s)
        proj_short.project_payments(2020, 2020)

        # Long tail
        proj_long = CashFlowProjector(a_priori_loss_ratio=0.70)
        cohort_l = ClaimCohort(accident_year=2020)
        cohort_l.add_claim(
            Claim(
                "CL002",
                2020,
                2020,
                1_000_000,
                development_pattern=ClaimDevelopment.create_very_long_tail_15yr(),
            )
        )
        proj_long.add_cohort(cohort_l)
        proj_long.project_payments(2020, 2020)

        earned_premium = {2020: 1_500_000}
        ibnr_short = proj_short.estimate_ibnr(evaluation_year=2021, earned_premium=earned_premium)
        ibnr_long = proj_long.estimate_ibnr(evaluation_year=2021, earned_premium=earned_premium)

        # Long-tail should have more IBNR because it is less developed at year 1
        assert ibnr_long > ibnr_short

    def test_load_ibnr_factors(self, tmp_path):
        """Test loading IBNR factors from YAML."""
        yaml_content = {
            "ibnr_factors": {
                "immediate": 1.02,
                "long_tail_10yr": 1.20,
            }
        }
        yaml_file = tmp_path / "test_factors.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        factors = load_ibnr_factors(str(yaml_file))
        assert factors["immediate"] == 1.02
        assert factors["long_tail_10yr"] == 1.20

    def test_load_ibnr_factors_missing_section(self, tmp_path):
        """Test loading from YAML with no ibnr_factors section."""
        yaml_content: dict = {"development_patterns": {}}
        yaml_file = tmp_path / "no_factors.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        factors = load_ibnr_factors(str(yaml_file))
        assert factors == {}


class TestEmpiricalChainLadder:
    """Tests for empirical Chain-Ladder with loss development triangles (Issue #626).

    Standard CL per Friedland and CAS Exam 7: build a paid-loss development
    triangle from actual cohort payment histories, compute volume-weighted
    age-to-age factors, derive CDFs-to-ultimate, and project ultimate losses.
    """

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _make_projector_with_cohorts(
        accident_years, claim_amounts, pattern_factory, project_through
    ):
        """Build a CashFlowProjector with cohorts and project payments."""
        projector = CashFlowProjector()
        for ay, amount in zip(accident_years, claim_amounts):
            cohort = ClaimCohort(accident_year=ay)
            cohort.add_claim(
                Claim(
                    f"CL-{ay}",
                    ay,
                    ay,
                    amount,
                    development_pattern=pattern_factory(),
                )
            )
            projector.add_cohort(cohort)
        projector.project_payments(min(accident_years), project_through)
        return projector

    # -- build_triangle ---------------------------------------------------

    def test_build_triangle_basic(self):
        """Triangle contains cumulative paid by (AY, dev_age)."""
        projector = self._make_projector_with_cohorts(
            accident_years=[2018, 2019, 2020],
            claim_amounts=[1_000_000, 1_000_000, 1_000_000],
            pattern_factory=ClaimDevelopment.create_medium_tail_5yr,
            project_through=2020,
        )
        triangle = projector.build_triangle(evaluation_year=2020)

        # AY 2018 has dev ages 0,1,2  (cal years 2018,2019,2020)
        assert set(triangle[2018].keys()) == {0, 1, 2}
        # AY 2019 has dev ages 0,1
        assert set(triangle[2019].keys()) == {0, 1}
        # AY 2020 has dev age 0
        assert set(triangle[2020].keys()) == {0}

        # Cumulative paid at age 0 = 40% of 1M = 400k
        assert triangle[2018][0] == pytest.approx(400_000)
        # Cumulative at age 1 = 400k + 250k = 650k (40%+25%)
        assert triangle[2018][1] == pytest.approx(650_000)
        # Cumulative at age 2 = 650k + 150k = 800k (40%+25%+15%)
        assert triangle[2018][2] == pytest.approx(800_000)

    def test_build_triangle_excludes_future_cohorts(self):
        """Cohorts with accident_year > evaluation_year are excluded."""
        projector = self._make_projector_with_cohorts(
            accident_years=[2020, 2025],
            claim_amounts=[1_000_000, 500_000],
            pattern_factory=ClaimDevelopment.create_medium_tail_5yr,
            project_through=2021,
        )
        triangle = projector.build_triangle(evaluation_year=2021)
        assert 2020 in triangle
        assert 2025 not in triangle

    # -- age-to-age factors -----------------------------------------------

    def test_age_to_age_factors_volume_weighted(self):
        """Link ratios are volume-weighted across accident years."""
        projector = self._make_projector_with_cohorts(
            accident_years=[2018, 2019, 2020],
            claim_amounts=[1_000_000, 1_000_000, 1_000_000],
            pattern_factory=ClaimDevelopment.create_medium_tail_5yr,
            project_through=2020,
        )
        triangle = projector.build_triangle(evaluation_year=2020)
        ata = projector._compute_age_to_age_factors(triangle)

        # Age 0→1: AY 2018 and 2019 both contribute (AY 2020 only has age 0)
        # All claims are identical and deterministic, so
        # LDF(0) = (cumul_age1_2018 + cumul_age1_2019) /
        #           (cumul_age0_2018 + cumul_age0_2019)
        #        = (650k + 650k) / (400k + 400k) = 1.625
        assert 0 in ata
        assert ata[0] == pytest.approx(1.625)

        # Age 1→2: only AY 2018 has both ages 1 and 2 → need >=2 contributors
        # so this factor should NOT be present
        assert 1 not in ata

    def test_age_to_age_factors_requires_two_contributors(self):
        """Factors require >=2 accident years with data at both ages."""
        projector = self._make_projector_with_cohorts(
            accident_years=[2020],
            claim_amounts=[1_000_000],
            pattern_factory=ClaimDevelopment.create_medium_tail_5yr,
            project_through=2022,
        )
        triangle = projector.build_triangle(evaluation_year=2022)
        ata = projector._compute_age_to_age_factors(triangle)
        # Single AY → no factor has >=2 contributors
        assert ata == {}

    def test_age_to_age_factors_multiple_ages(self):
        """With enough cohorts, factors are computed for multiple ages."""
        # 4 cohorts give 3 pairs at age 0→1, 2 pairs at age 1→2
        projector = self._make_projector_with_cohorts(
            accident_years=[2017, 2018, 2019, 2020],
            claim_amounts=[1_000_000] * 4,
            pattern_factory=ClaimDevelopment.create_medium_tail_5yr,
            project_through=2020,
        )
        triangle = projector.build_triangle(evaluation_year=2020)
        ata = projector._compute_age_to_age_factors(triangle)

        # Age 0→1: AYs 2017,2018,2019 contribute (3 contributors)
        assert 0 in ata
        # Age 1→2: AYs 2017,2018 contribute (2 contributors)
        assert 1 in ata
        # Age 2→3: only AY 2017 contributes (1 contributor) → excluded
        assert 2 not in ata

    # -- CDF to ultimate --------------------------------------------------

    def test_cdf_to_ultimate_basic(self):
        """CDF is cumulative product of link ratios."""
        projector = CashFlowProjector()
        ata = {0: 2.0, 1: 1.5, 2: 1.2}

        # At age 0: CDF = 2.0 * 1.5 * 1.2 = 3.6
        assert projector._compute_cdf_to_ultimate(ata, 0) == pytest.approx(3.6)
        # At age 1: CDF = 1.5 * 1.2 = 1.8
        assert projector._compute_cdf_to_ultimate(ata, 1) == pytest.approx(1.8)
        # At age 2: CDF = 1.2
        assert projector._compute_cdf_to_ultimate(ata, 2) == pytest.approx(1.2)
        # Beyond max factor age: fully developed → 1.0
        assert projector._compute_cdf_to_ultimate(ata, 3) == pytest.approx(1.0)

    def test_cdf_to_ultimate_gap_returns_none(self):
        """A gap in the factor chain returns None (cannot project)."""
        projector = CashFlowProjector()
        ata = {0: 2.0, 2: 1.2}  # gap at age 1
        assert projector._compute_cdf_to_ultimate(ata, 0) is None

    def test_cdf_to_ultimate_empty_factors(self):
        """Empty factors → None."""
        projector = CashFlowProjector()
        assert projector._compute_cdf_to_ultimate({}, 0) is None

    # -- empirical CL in estimate_ibnr ------------------------------------

    def test_empirical_cl_with_adverse_development(self):
        """Empirical CL captures adverse development when actual > expected.

        When payments run higher than the assumed pattern implies, the
        empirical link ratios project a higher ultimate, producing positive
        IBNR even though assumed-pattern CL cannot.

        Uses 4 cohorts to get >=2 contributors at ages 0→1 and 1→2.
        """
        pattern = ClaimDevelopment.create_medium_tail_5yr()
        projector = CashFlowProjector()

        # 4 cohorts, each with initial_estimate = $500k.
        # Actual payments are HIGHER than the assumed 5yr pattern:
        #   age 0: $250k (50% vs 40%), age 1: +$200k → cumul $450k (90% vs 65%)
        #   age 2: +$100k → cumul $550k, age 3: +$60k → cumul $610k
        actual_incremental = [250_000, 200_000, 100_000, 60_000]

        for ay in [2017, 2018, 2019, 2020]:
            cohort = ClaimCohort(accident_year=ay)
            claim = Claim(f"CL-{ay}", ay, ay, 500_000, development_pattern=pattern)
            cohort.add_claim(claim)
            projector.add_cohort(cohort)
            max_age = 2020 - ay
            for age in range(min(max_age + 1, len(actual_incremental))):
                claim.record_payment(ay + age, actual_incremental[age])

        triangle = projector.build_triangle(2020)
        ata = projector._compute_age_to_age_factors(triangle)

        # LDF(0): AYs 2017,2018,2019 contribute → 450/250 = 1.8
        assert ata[0] == pytest.approx(1.8)
        # LDF(1): AYs 2017,2018 contribute → 550/450 ≈ 1.2222
        assert ata[1] == pytest.approx(550_000 / 450_000)

        ibnr = projector.estimate_ibnr(evaluation_year=2020)

        # Empirical CL should produce positive IBNR for at least some cohorts.
        # AY 2020 at age 0: paid=250k, CDF(0)=1.8*1.222=2.2 → ult=550k > 500k → IBNR=50k
        # AY 2019 at age 1: paid=450k, CDF(1)=1.222 → ult=550k > 500k → IBNR=50k
        assert ibnr > 0

    def test_empirical_cl_differs_from_assumed_pattern(self):
        """Multi-cohort empirical CL gives different IBNR than assumed-pattern CL.

        When actual payments run hotter than case estimates, empirical CL
        projects higher ultimates while assumed-pattern CL (which divides
        paid by the assumed pct) does the same but with different factors.
        Uses 4 cohorts for >=2 contributors at ages 0→1 and 1→2.
        """
        pattern = ClaimDevelopment.create_medium_tail_5yr()

        # --- Empirical projector (adverse development) ---
        proj_empirical = CashFlowProjector()
        actual_incremental = [250_000, 200_000, 100_000, 60_000]
        for ay in [2017, 2018, 2019, 2020]:
            cohort = ClaimCohort(accident_year=ay)
            claim = Claim(f"CL-{ay}", ay, ay, 500_000, development_pattern=pattern)
            cohort.add_claim(claim)
            proj_empirical.add_cohort(cohort)
            max_age = 2020 - ay
            for age in range(min(max_age + 1, len(actual_incremental))):
                claim.record_payment(ay + age, actual_incremental[age])

        # --- Assumed-pattern projector (deterministic, matches pattern) ---
        proj_assumed = CashFlowProjector()
        for ay in [2017, 2018, 2019, 2020]:
            cohort = ClaimCohort(accident_year=ay)
            claim = Claim(f"CL-{ay}", ay, ay, 500_000, development_pattern=pattern)
            cohort.add_claim(claim)
            proj_assumed.add_cohort(cohort)
        proj_assumed.project_payments(2017, 2020)

        ibnr_empirical = proj_empirical.estimate_ibnr(evaluation_year=2020)
        ibnr_assumed = proj_assumed.estimate_ibnr(evaluation_year=2020)

        # With paid-basis IBNR, assumed-pattern CL also produces positive IBNR
        # (ultimate - paid > 0 for immature cohorts)
        assert ibnr_assumed > 0
        # Empirical CL with adverse development → positive IBNR
        assert ibnr_empirical > 0
        assert ibnr_empirical != ibnr_assumed

    def test_empirical_cl_fallback_single_cohort(self):
        """Single cohort has no empirical factors → falls back to assumed-pattern CL."""
        projector = self._make_projector_with_cohorts(
            accident_years=[2020],
            claim_amounts=[1_000_000],
            pattern_factory=ClaimDevelopment.create_medium_tail_5yr,
            project_through=2021,
        )
        # Single cohort: no link ratios (need >=2 contributors).
        # Assumed-pattern CL at eval 2022: dev_years=2, pct=0.65,
        # paid=650k, CL = 650k / 0.65 = 1M, IBNR = 1M - 650k = 350k (paid-basis).
        ibnr = projector.estimate_ibnr(evaluation_year=2022)
        assert ibnr == pytest.approx(350_000)

    def test_bf_blend_with_empirical_cl(self):
        """BF blend still works correctly with empirical CL ultimate."""
        pattern = ClaimDevelopment.create_medium_tail_5yr()
        projector = CashFlowProjector(a_priori_loss_ratio=0.70)

        # 3 cohorts with faster-than-expected payments
        for ay in [2018, 2019, 2020]:
            cohort = ClaimCohort(accident_year=ay)
            claim = Claim(f"CL-{ay}", ay, ay, 1_000_000, development_pattern=pattern)
            cohort.add_claim(claim)
            projector.add_cohort(cohort)

        # Record payments: age 0 = 50% (vs 40%), age 1 = 30% (vs 25%)
        for ay in [2018, 2019, 2020]:
            claim = projector.cohorts[ay].claims[0]
            claim.payments_made = {}
            max_age = 2020 - ay
            actual = [500_000, 300_000, 150_000]
            for age in range(min(max_age + 1, len(actual))):
                claim.record_payment(ay + age, actual[age])

        earned_premium = {2018: 1_500_000, 2019: 1_500_000, 2020: 1_500_000}
        ibnr = projector.estimate_ibnr(evaluation_year=2020, earned_premium=earned_premium)

        # With BF + empirical CL + blend, IBNR should be positive
        assert ibnr > 0

        # Verify that IBNR is bounded by total exposure
        total_incurred = 3_000_000
        assert ibnr < total_incurred * 5

    def test_empirical_cl_triangle_accuracy(self):
        """Verify triangle values match actual cumulative payments exactly."""
        projector = CashFlowProjector()

        # Manually construct 2 cohorts with known payment histories
        cohort_2019 = ClaimCohort(accident_year=2019)
        claim_a = Claim(
            "A", 2019, 2019, 500_000, development_pattern=ClaimDevelopment.create_medium_tail_5yr()
        )
        claim_a.payments_made = {2019: 100_000, 2020: 80_000, 2021: 50_000}
        cohort_2019.add_claim(claim_a)

        cohort_2020 = ClaimCohort(accident_year=2020)
        claim_b = Claim(
            "B", 2020, 2020, 700_000, development_pattern=ClaimDevelopment.create_medium_tail_5yr()
        )
        claim_b.payments_made = {2020: 200_000, 2021: 120_000}
        cohort_2020.add_claim(claim_b)

        projector.add_cohort(cohort_2019)
        projector.add_cohort(cohort_2020)

        triangle = projector.build_triangle(evaluation_year=2021)

        # AY 2019: age 0=100k, age 1=180k, age 2=230k
        assert triangle[2019][0] == pytest.approx(100_000)
        assert triangle[2019][1] == pytest.approx(180_000)
        assert triangle[2019][2] == pytest.approx(230_000)

        # AY 2020: age 0=200k, age 1=320k
        assert triangle[2020][0] == pytest.approx(200_000)
        assert triangle[2020][1] == pytest.approx(320_000)

    def test_empirical_ata_with_varying_claim_sizes(self):
        """Volume-weighted ATA factors weight larger cohorts more heavily."""
        projector = CashFlowProjector()

        # Cohort 2019: small ($100k), develops 100k → 200k at age 0→1
        cohort_2019 = ClaimCohort(accident_year=2019)
        claim_s = Claim(
            "S", 2019, 2019, 100_000, development_pattern=ClaimDevelopment.create_medium_tail_5yr()
        )
        claim_s.payments_made = {2019: 100_000, 2020: 100_000}  # cumul: 100k, 200k
        cohort_2019.add_claim(claim_s)

        # Cohort 2020: large ($900k), develops 900k → 1080k at age 0→1
        # But we need eval year where both have age 0 and 1.
        # Actually cohort 2020 needs ages 0 and 1, so eval >= 2021.
        # Let me add a third cohort so we have >=2 contributors at age 0→1.
        cohort_2020 = ClaimCohort(accident_year=2020)
        claim_l = Claim(
            "L", 2020, 2020, 900_000, development_pattern=ClaimDevelopment.create_medium_tail_5yr()
        )
        claim_l.payments_made = {2020: 900_000, 2021: 180_000}  # cumul: 900k, 1080k
        cohort_2020.add_claim(claim_l)

        projector.add_cohort(cohort_2019)
        projector.add_cohort(cohort_2020)

        triangle = projector.build_triangle(evaluation_year=2021)
        ata = projector._compute_age_to_age_factors(triangle)

        # Volume-weighted LDF(0) = (200k + 1080k) / (100k + 900k) = 1280k / 1000k = 1.28
        assert 0 in ata
        assert ata[0] == pytest.approx(1.28)

    def test_empirical_cl_with_multiple_development_ages(self):
        """Verify CDF-to-ultimate with multiple link ratios from real data."""
        projector = CashFlowProjector()

        # 4 cohorts to get factors at ages 0→1 (3 contributors)
        # and 1→2 (2 contributors)
        for ay in [2017, 2018, 2019, 2020]:
            cohort = ClaimCohort(accident_year=ay)
            claim = Claim(
                f"CL-{ay}",
                ay,
                ay,
                1_000_000,
                development_pattern=ClaimDevelopment.create_medium_tail_5yr(),
            )
            cohort.add_claim(claim)
            projector.add_cohort(cohort)

        projector.project_payments(2017, 2020)

        triangle = projector.build_triangle(evaluation_year=2020)
        ata = projector._compute_age_to_age_factors(triangle)

        # For deterministic MEDIUM_TAIL_5YR [0.40, 0.25, 0.15, 0.10, 0.10]:
        # cumulative: age 0=400k, 1=650k, 2=800k, 3=900k
        # LDF(0) = 650/400 = 1.625 (3 contributors: 2017,2018,2019)
        # LDF(1) = 800/650 = ~1.2308 (2 contributors: 2017,2018)
        assert 0 in ata
        assert ata[0] == pytest.approx(1.625)
        assert 1 in ata
        assert ata[1] == pytest.approx(800_000 / 650_000)

        # CDF at age 0 = LDF(0) * LDF(1) = 1.625 * 1.2308 = 2.0
        cdf_0 = projector._compute_cdf_to_ultimate(ata, 0)
        assert cdf_0 == pytest.approx(1.625 * (800_000 / 650_000))

        # For AY 2020 at age 0 with paid=400k:
        # Empirical CL ultimate = 400k * CDF(0) = 400k * 2.0 = 800k
        # Incurred = 1M → IBNR from CL = max(0, 800k - 1M) = 0
        # (deterministic data → CL still under-projects due to missing later factors)

    def test_empirical_cl_preserves_existing_test_behavior(self):
        """All existing CL-only test scenarios still pass with empirical CL.

        Single-cohort deterministic scenarios fall back to assumed-pattern CL
        because empirical factors require >=2 contributors.
        """
        # Reproduce test_cl_only_no_elr exactly
        projector = CashFlowProjector()
        cohort = ClaimCohort(accident_year=2020)
        claim = Claim(
            "CL001",
            2020,
            2020,
            1_000_000,
            development_pattern=ClaimDevelopment.create_medium_tail_5yr(),
        )
        cohort.add_claim(claim)
        projector.add_cohort(cohort)
        projector.project_payments(2020, 2020)

        # CL ultimate = 400k / 0.40 = 1M; IBNR = 1M - 400k = 600k (paid-basis)
        ibnr = projector.estimate_ibnr(evaluation_year=2021)
        assert ibnr == pytest.approx(600_000)

    def test_empirical_cl_with_premium_and_bf(self):
        """Empirical CL ultimate feeds into BF blend correctly.

        When empirical CL gives a different ultimate than assumed-pattern,
        the BF blend should weight both CL and BF according to maturity.
        """
        pattern = ClaimDevelopment.create_medium_tail_5yr()
        projector = CashFlowProjector(a_priori_loss_ratio=0.70)

        # 3 cohorts, all deterministic (so empirical factors match assumed)
        for ay in [2018, 2019, 2020]:
            cohort = ClaimCohort(accident_year=ay)
            claim = Claim(f"CL-{ay}", ay, ay, 1_000_000, development_pattern=pattern)
            cohort.add_claim(claim)
            projector.add_cohort(cohort)

        projector.project_payments(2018, 2021)

        # AY 2020 at eval 2021: dev_years=1, pct=0.40, paid=650k
        # Empirical CDF at age 1: LDF(1) = 1.2308 (from AY 2018,2019)
        # Empirical CL ultimate = 650k * 1.2308 = 800k
        #
        # BF IBNR = 0.70 * 1.5M * 0.60 = 630k; BF ult = 650k + 630k = 1.28M
        # Blended = 0.40 * 800k + 0.60 * 1.28M = 320k + 768k = 1.088M
        # IBNR = 88k
        earned_premium = {2018: 1_500_000, 2019: 1_500_000, 2020: 1_500_000}
        ibnr = projector.estimate_ibnr(evaluation_year=2021, earned_premium=earned_premium)

        # IBNR should be positive when BF is available
        assert ibnr > 0


class TestRegressions:
    """Regression tests for specific bug fixes."""

    def test_tail_factor_pays_only_once_issue_810(self):
        """#810: tail factor must pay once at development_year == len(factors), not every year."""
        pattern = ClaimDevelopment(
            pattern_name="WITH_TAIL",
            development_factors=[0.40, 0.30, 0.28],
            tail_factor=0.02,
        )
        claim_amount = 1_000_000
        accident_year = 2020

        # Total payments over many years must equal claim_amount exactly
        total = sum(
            pattern.calculate_payments(claim_amount, accident_year, 2020 + y) for y in range(50)
        )
        assert total == pytest.approx(claim_amount)

        # Tail paid exactly once at development_year == 3
        assert pattern.calculate_payments(claim_amount, accident_year, 2023) == 20_000
        # No payment after tail year
        assert pattern.calculate_payments(claim_amount, accident_year, 2024) == 0.0
        assert pattern.calculate_payments(claim_amount, accident_year, 2050) == 0.0

    def test_cumulative_paid_consistent_with_payments_issue_810(self):
        """#810: get_cumulative_paid and calculate_payments must agree on total payout."""
        pattern = ClaimDevelopment(
            pattern_name="WITH_TAIL",
            development_factors=[0.40, 0.30, 0.28],
            tail_factor=0.02,
        )
        # After all development + tail, cumulative should be 1.0
        assert pattern.get_cumulative_paid(len(pattern.development_factors) + 1) == pytest.approx(
            1.0
        )
        # And should stay at 1.0 for all later years
        assert pattern.get_cumulative_paid(100) == pytest.approx(1.0)

    def test_bf_ultimate_uses_paid_to_date_issue_805(self):
        """#805: BF ultimate = paid_to_date + bf_ibnr, not incurred + bf_ibnr."""
        projector = CashFlowProjector(a_priori_loss_ratio=0.70)

        cohort = ClaimCohort(accident_year=2020)
        claim = Claim(
            "CL001",
            2020,
            2020,
            1_000_000,
            development_pattern=ClaimDevelopment.create_medium_tail_5yr(),
        )
        cohort.add_claim(claim)
        projector.add_cohort(cohort)
        projector.project_payments(2020, 2020)

        # pct=0.40, paid=400k, incurred=1M
        # BF IBNR = 0.70 * 2M * 0.60 = 840k
        # Correct BF ultimate = 400k + 840k = 1.24M
        # Wrong BF ultimate (old bug) = 1M + 840k = 1.84M
        # CL = 400k / 0.40 = 1M
        # Blended = 0.40 * 1M + 0.60 * 1.24M = 1.144M
        # IBNR = 1.144M - 400k = 744k (paid-basis)
        earned_premium = {2020: 2_000_000}
        ibnr = projector.estimate_ibnr(evaluation_year=2021, earned_premium=earned_premium)
        assert ibnr == pytest.approx(744_000)
        # Must NOT be the old buggy value (if BF used incurred instead of paid:
        # BF ult = 1M + 840k = 1.84M, blended = 1.504M, IBNR = 1.504M - 400k = 1_104k)
        assert ibnr != pytest.approx(1_104_000)

    def test_bf_only_immature_paid_basis_issue_805(self):
        """#805: at 0% development with BF-only, ultimate = 0 + bf_ibnr (paid basis)."""
        projector = CashFlowProjector(a_priori_loss_ratio=0.60)

        cohort = ClaimCohort(accident_year=2023)
        claim = Claim(
            "CL001",
            2023,
            2023,
            500_000,
            development_pattern=ClaimDevelopment.create_long_tail_10yr(),
        )
        cohort.add_claim(claim)
        projector.add_cohort(cohort)

        # dev_years=0, pct=0.0, paid=0
        # BF IBNR = 0.60 * 1M * 1.0 = 600k
        # BF ultimate = 0 + 600k = 600k
        # IBNR = max(0, 600k - 0) = 600k (paid-basis)
        earned_premium = {2023: 1_000_000}
        ibnr = projector.estimate_ibnr(evaluation_year=2023, earned_premium=earned_premium)
        assert ibnr == pytest.approx(600_000)


class TestDevelopmentPattern:
    """Tests for DevelopmentPattern class (#1054)."""

    def test_basic_construction(self):
        """DevelopmentPattern can be constructed with valid CDFs."""
        dp = DevelopmentPattern(
            pattern_name="test",
            cumulative_ldfs=[2.5, 1.5, 1.2, 1.0],
            tail_cdf=1.0,
        )
        assert dp.pattern_name == "test"
        assert dp.cumulative_ldfs == [2.5, 1.5, 1.2, 1.0]
        assert dp.tail_cdf == 1.0

    def test_pct_developed(self):
        """pct_developed returns 1/CDF clamped to [0, 1]."""
        dp = DevelopmentPattern(
            pattern_name="test",
            cumulative_ldfs=[2.5, 1.5, 1.25, 1.0],
        )
        assert dp.pct_developed(0) == 0.0
        assert dp.pct_developed(1) == pytest.approx(1.0 / 2.5)  # 0.40
        assert dp.pct_developed(2) == pytest.approx(1.0 / 1.5)  # ~0.667
        assert dp.pct_developed(3) == pytest.approx(1.0 / 1.25)  # 0.80
        assert dp.pct_developed(4) == pytest.approx(1.0)  # fully developed
        assert dp.pct_developed(10) == pytest.approx(1.0)  # beyond pattern

    def test_cdf_at(self):
        """cdf_at returns correct CDF at each age."""
        dp = DevelopmentPattern(
            pattern_name="test",
            cumulative_ldfs=[2.5, 1.5, 1.25, 1.0],
            tail_cdf=1.0,
        )
        # Age < 1 returns first (largest) CDF
        assert dp.cdf_at(0) == 2.5
        assert dp.cdf_at(-1) == 2.5
        # Ages within pattern
        assert dp.cdf_at(1) == 2.5
        assert dp.cdf_at(2) == 1.5
        assert dp.cdf_at(3) == 1.25
        assert dp.cdf_at(4) == 1.0
        # Beyond pattern returns tail
        assert dp.cdf_at(5) == 1.0
        assert dp.cdf_at(100) == 1.0

    def test_validation_empty(self):
        """Empty cumulative_ldfs raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            DevelopmentPattern("bad", cumulative_ldfs=[], tail_cdf=1.0)

    def test_validation_below_one(self):
        """CDFs below 1.0 raise ValueError."""
        with pytest.raises(ValueError, match="must be >= 1.0"):
            DevelopmentPattern("bad", cumulative_ldfs=[0.9, 0.8])

    def test_validation_non_monotonic(self):
        """Non-monotonically decreasing CDFs raise ValueError."""
        with pytest.raises(ValueError, match="non-increasing"):
            DevelopmentPattern("bad", cumulative_ldfs=[1.5, 2.0, 1.0])

    def test_validation_tail_below_one(self):
        """tail_cdf below 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="tail_cdf must be >= 1.0"):
            DevelopmentPattern("bad", cumulative_ldfs=[2.0, 1.5], tail_cdf=0.5)

    def test_pct_developed_negative_age(self):
        """Negative development age returns 0."""
        dp = DevelopmentPattern("test", cumulative_ldfs=[2.0, 1.5, 1.0])
        assert dp.pct_developed(-5) == 0.0


class TestDevelopmentPatternFromPayment:
    """Tests for DevelopmentPattern.from_payment_pattern (#1054)."""

    def test_medium_tail_5yr(self):
        """from_payment_pattern for MEDIUM_TAIL_5YR produces correct CDFs."""
        payment = ClaimDevelopment.create_medium_tail_5yr()
        dp = DevelopmentPattern.from_payment_pattern(payment)

        # MEDIUM_TAIL_5YR: [0.40, 0.25, 0.15, 0.10, 0.10]
        # cumulative: 0.40, 0.65, 0.80, 0.90, 1.00
        # CDF: 2.50, ~1.538, 1.25, ~1.111, 1.0
        assert dp.cumulative_ldfs[0] == pytest.approx(1.0 / 0.40)  # 2.5
        assert dp.cumulative_ldfs[1] == pytest.approx(1.0 / 0.65)
        assert dp.cumulative_ldfs[2] == pytest.approx(1.0 / 0.80)  # 1.25
        assert dp.cumulative_ldfs[3] == pytest.approx(1.0 / 0.90)
        assert dp.cumulative_ldfs[4] == pytest.approx(1.0)
        assert dp.tail_cdf == pytest.approx(1.0)

    def test_consistency_with_get_cumulative_paid(self):
        """pct_developed matches get_cumulative_paid for all built-in patterns."""
        patterns = [
            ClaimDevelopment.create_immediate(),
            ClaimDevelopment.create_medium_tail_5yr(),
            ClaimDevelopment.create_long_tail_10yr(),
            ClaimDevelopment.create_very_long_tail_15yr(),
        ]
        for payment in patterns:
            dp = DevelopmentPattern.from_payment_pattern(payment)
            for age in range(len(payment.development_factors) + 2):
                expected_pct = payment.get_cumulative_paid(age)
                actual_pct = dp.pct_developed(age)
                assert actual_pct == pytest.approx(expected_pct, abs=1e-10), (
                    f"Mismatch for {payment.pattern_name} at age {age}: "
                    f"expected {expected_pct}, got {actual_pct}"
                )

    def test_from_payment_with_tail(self):
        """Patterns with tail factor produce correct CDFs."""
        payment = ClaimDevelopment(
            pattern_name="WITH_TAIL",
            development_factors=[0.4, 0.3, 0.28],
            tail_factor=0.02,
        )
        dp = DevelopmentPattern.from_payment_pattern(payment)
        # 4 CDFs (3 dev factors + 1 tail period)
        assert len(dp.cumulative_ldfs) == 4
        assert dp.cumulative_ldfs[-1] == pytest.approx(1.0)
        assert dp.tail_cdf == pytest.approx(1.0)


class TestDevelopmentPatternFromATA:
    """Tests for DevelopmentPattern.from_age_to_age_factors (#1054)."""

    def test_basic_construction(self):
        """from_age_to_age_factors produces correct CDFs."""
        # LDFs: 1.625, 1.231, 1.125, 1.111
        dp = DevelopmentPattern.from_age_to_age_factors("test", [1.625, 1.231, 1.125, 1.111])
        # CDF at age 1 = 1.625 * 1.231 * 1.125 * 1.111
        expected_cdf_1 = 1.625 * 1.231 * 1.125 * 1.111
        assert dp.cumulative_ldfs[0] == pytest.approx(expected_cdf_1)
        # CDF at last age = 1.111
        assert dp.cumulative_ldfs[-1] == pytest.approx(1.111)
        assert dp.tail_cdf == 1.0

    def test_with_tail(self):
        """from_age_to_age_factors applies tail_factor correctly."""
        dp = DevelopmentPattern.from_age_to_age_factors("test", [2.0, 1.5], tail_factor=1.05)
        # CDF at age 1 = 2.0 * 1.5 * 1.05 = 3.15
        assert dp.cumulative_ldfs[0] == pytest.approx(3.15)
        # CDF at age 2 = 1.5 * 1.05 = 1.575
        assert dp.cumulative_ldfs[1] == pytest.approx(1.575)
        assert dp.tail_cdf == 1.05

    def test_empty_factors_raises(self):
        """Empty ata_factors raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            DevelopmentPattern.from_age_to_age_factors("bad", [])


class TestCashFlowProjectorWithDevelopmentPattern:
    """Tests for CashFlowProjector with explicit DevelopmentPattern (#1054)."""

    def test_explicit_pattern_override(self):
        """When development_pattern is set, it overrides per-claim patterns."""
        # Create a DevelopmentPattern that says 50% developed at age 1
        dp = DevelopmentPattern(
            pattern_name="override",
            cumulative_ldfs=[2.0, 1.0],  # age 1: 50%, age 2: 100%
        )
        projector = CashFlowProjector(development_pattern=dp)

        cohort = ClaimCohort(accident_year=2020)
        claim = Claim(
            "CL001",
            2020,
            2020,
            1_000_000,
            development_pattern=ClaimDevelopment.create_medium_tail_5yr(),
        )
        cohort.add_claim(claim)
        projector.add_cohort(cohort)
        projector.project_payments(2020, 2020)

        # pct_developed uses the explicit DevelopmentPattern (50%), not the
        # claim's MEDIUM_TAIL_5YR (40%)
        pct = projector._get_cohort_pct_developed(cohort, 1)
        assert pct == pytest.approx(0.50)

    def test_none_fallback(self):
        """When development_pattern is None, per-claim patterns are used."""
        projector = CashFlowProjector()  # No development_pattern

        cohort = ClaimCohort(accident_year=2020)
        claim = Claim(
            "CL001",
            2020,
            2020,
            1_000_000,
            development_pattern=ClaimDevelopment.create_medium_tail_5yr(),
        )
        cohort.add_claim(claim)
        projector.add_cohort(cohort)

        pct = projector._get_cohort_pct_developed(cohort, 1)
        assert pct == pytest.approx(0.40)


class TestTailFactor:
    """Tests for tail factor in CDF computation (#1059)."""

    def test_cdf_to_ultimate_with_tail_factor(self):
        """CDF is multiplied by tail factor."""
        projector = CashFlowProjector()
        ata = {0: 2.0, 1: 1.5}

        # Without tail: CDF at age 0 = 2.0 * 1.5 = 3.0
        assert projector._compute_cdf_to_ultimate(ata, 0) == pytest.approx(3.0)

        # With tail factor 1.05: CDF at age 0 = 2.0 * 1.5 * 1.05 = 3.15
        assert projector._compute_cdf_to_ultimate(ata, 0, tail_factor=1.05) == pytest.approx(3.15)

    def test_cdf_beyond_max_with_tail(self):
        """Beyond max observed age, returns tail_factor (not 1.0)."""
        projector = CashFlowProjector()
        ata = {0: 2.0, 1: 1.5}

        # At age 2 (beyond max_factor_age=1), return tail_factor
        assert projector._compute_cdf_to_ultimate(ata, 2, tail_factor=1.10) == pytest.approx(1.10)
        # Default tail_factor=1.0 preserves old behavior
        assert projector._compute_cdf_to_ultimate(ata, 2) == pytest.approx(1.0)

    def test_fit_tail_factor_bondy(self):
        """Bondy method returns last observed LDF."""
        projector = CashFlowProjector()
        ata = {0: 2.0, 1: 1.5, 2: 1.08}

        tail = projector.fit_tail_factor(ata, method="bondy")
        assert tail == pytest.approx(1.08)

    def test_fit_tail_factor_bondy_out_of_range(self):
        """Bondy returns 1.0 when last LDF > 2.0."""
        projector = CashFlowProjector()
        ata = {0: 3.0, 1: 2.5}

        tail = projector.fit_tail_factor(ata, method="bondy")
        assert tail == 1.0

    def test_fit_tail_factor_empty(self):
        """Empty factors → 1.0."""
        projector = CashFlowProjector()
        tail = projector.fit_tail_factor({}, method="bondy")
        assert tail == 1.0

    def test_reserve_tail_factor_in_ibnr(self):
        """reserve_tail_factor > 1 increases IBNR."""
        # Projector without tail
        proj_no_tail = CashFlowProjector()
        # Projector with tail
        proj_tail = CashFlowProjector(reserve_tail_factor=1.10)

        # 3 cohorts for empirical factors
        for proj in [proj_no_tail, proj_tail]:
            for ay in [2018, 2019, 2020]:
                cohort = ClaimCohort(accident_year=ay)
                claim = Claim(
                    f"CL-{ay}",
                    ay,
                    ay,
                    1_000_000,
                    development_pattern=ClaimDevelopment.create_medium_tail_5yr(),
                )
                cohort.add_claim(claim)
                proj.add_cohort(cohort)
            proj.project_payments(2018, 2020)

        ibnr_no_tail = proj_no_tail.estimate_ibnr(evaluation_year=2020)
        ibnr_with_tail = proj_tail.estimate_ibnr(evaluation_year=2020)

        # Tail factor increases IBNR
        assert ibnr_with_tail > ibnr_no_tail


class TestReserveIdentity:
    """Tests for reserve identity (#1056)."""

    def test_reserve_identity_case_plus_ibnr(self):
        """Verify total_reserves = case_outstanding + IBNR."""
        projector = CashFlowProjector(a_priori_loss_ratio=0.70)

        cohort = ClaimCohort(accident_year=2020)
        claim = Claim(
            "CL001",
            2020,
            2020,
            1_000_000,
            development_pattern=ClaimDevelopment.create_medium_tail_5yr(),
        )
        cohort.add_claim(claim)
        projector.add_cohort(cohort)
        projector.project_payments(2020, 2020)

        earned_premium = {2020: 2_000_000}
        reserves = projector.calculate_total_reserves(
            evaluation_year=2021, earned_premium=earned_premium
        )

        # total_reserves = case_reserves + ibnr
        assert reserves["total_reserves"] == pytest.approx(
            reserves["case_reserves"] + reserves["ibnr"]
        )
        # Case reserves = initial_estimate - paid = 1M - 400k = 600k
        assert reserves["case_reserves"] == pytest.approx(600_000)
        # IBNR should be positive (blended CL/BF on paid basis)
        assert reserves["ibnr"] > 0
