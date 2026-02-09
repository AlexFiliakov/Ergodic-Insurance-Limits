"""Regression tests for issue #400: Bootstrap CI ignores configured seed.

Verifies that _calculate_bootstrap_ci uses the seed parameter so that
bootstrap confidence intervals are reproducible when a seed is set.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from ergodic_insurance.ruin_probability import RuinProbabilityAnalyzer


@pytest.fixture
def analyzer():
    """Create a minimal RuinProbabilityAnalyzer for testing."""
    return RuinProbabilityAnalyzer(
        manufacturer=MagicMock(),
        loss_generator=MagicMock(),
        insurance_program=MagicMock(),
        config=MagicMock(),
    )


@pytest.fixture
def bankruptcy_years():
    """Deterministic bankruptcy years for reproducible tests."""
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 100)


class TestBootstrapCISeed:
    """Tests for issue #400: seed must be forwarded to bootstrap CI."""

    def test_seed_produces_reproducible_ci(self, analyzer, bankruptcy_years):
        """With seed=42, two identical calls produce identical bootstrap CIs."""
        ci1 = analyzer._calculate_bootstrap_ci(
            bankruptcy_years, [5, 10], n_bootstrap=200, confidence_level=0.95, seed=42
        )
        ci2 = analyzer._calculate_bootstrap_ci(
            bankruptcy_years, [5, 10], n_bootstrap=200, confidence_level=0.95, seed=42
        )
        np.testing.assert_array_equal(ci1, ci2)

    def test_no_seed_is_non_deterministic(self, analyzer, bankruptcy_years):
        """With seed=None, CIs should (almost certainly) differ between runs."""
        ci1 = analyzer._calculate_bootstrap_ci(
            bankruptcy_years, [5], n_bootstrap=200, confidence_level=0.95, seed=None
        )
        ci2 = analyzer._calculate_bootstrap_ci(
            bankruptcy_years, [5], n_bootstrap=200, confidence_level=0.95, seed=None
        )
        # Extremely unlikely that two unseeded runs produce identical CIs
        assert not np.array_equal(ci1, ci2)

    def test_different_seeds_produce_different_ci(self, analyzer, bankruptcy_years):
        """Different seeds should produce different CIs."""
        ci1 = analyzer._calculate_bootstrap_ci(
            bankruptcy_years, [5, 10], n_bootstrap=200, confidence_level=0.95, seed=42
        )
        ci2 = analyzer._calculate_bootstrap_ci(
            bankruptcy_years, [5, 10], n_bootstrap=200, confidence_level=0.95, seed=99
        )
        assert not np.array_equal(ci1, ci2)
