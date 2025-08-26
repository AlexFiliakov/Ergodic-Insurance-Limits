"""Unit tests for claim development patterns module."""

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml
from ergodic_insurance.src.claim_development import (
    CashFlowProjector,
    Claim,
    ClaimCohort,
    ClaimDevelopment,
    DevelopmentPatternType,
    load_development_patterns,
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

        # Beyond pattern period - tail factor applies
        assert pattern.calculate_payments(claim_amount, accident_year, 2023) == 100_000
        assert pattern.calculate_payments(claim_amount, accident_year, 2025) == 100_000

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
        """Test IBNR estimation."""
        projector = CashFlowProjector()

        # Recent accident year cohort
        cohort_2023 = ClaimCohort(accident_year=2023)
        claim1 = Claim("CL001", 2023, 2023, 1_000_000)
        cohort_2023.add_claim(claim1)

        # Older cohort
        cohort_2021 = ClaimCohort(accident_year=2021)
        claim2 = Claim("CL002", 2021, 2021, 2_000_000)
        cohort_2021.add_claim(claim2)

        projector.add_cohort(cohort_2023)
        projector.add_cohort(cohort_2021)

        ibnr = projector.estimate_ibnr(evaluation_year=2023, reporting_lag=3)

        # Recent cohort should have significant IBNR
        assert ibnr > 0
        # IBNR should be reasonable relative to incurred
        total_incurred = 3_000_000
        assert ibnr < total_incurred * 0.5  # Less than 50% of total

    def test_calculate_total_reserves(self):
        """Test total reserve calculation."""
        projector = CashFlowProjector()

        cohort = ClaimCohort(accident_year=2020)
        claim = Claim("CL001", 2020, 2020, 1_000_000)
        claim.record_payment(2020, 400_000)  # Partial payment
        cohort.add_claim(claim)
        projector.add_cohort(cohort)

        reserves = projector.calculate_total_reserves(evaluation_year=2021)

        assert "case_reserves" in reserves
        assert "ibnr" in reserves
        assert "total_reserves" in reserves
        assert reserves["case_reserves"] == 600_000  # Outstanding from claim
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
        assert elapsed < 0.10, f"Processing took {elapsed:.3f}s, expected < 100ms"
        assert payment > 0  # Should have calculated payments

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
