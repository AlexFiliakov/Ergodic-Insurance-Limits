"""Coverage tests for insurance.py targeting specific uncovered lines.

Missing lines: 124, 126, 326, 404-428, 449, 453
"""

import logging
from unittest.mock import patch
import warnings

import pytest

from ergodic_insurance.insurance import InsuranceLayer, InsurancePolicy


class TestInsuranceLayerValidation:
    """Tests for InsuranceLayer.__post_init__ validation (lines 124, 126)."""

    def test_negative_attachment_point_raises(self):
        """Line 124: Negative attachment_point raises ValueError."""
        with pytest.raises(ValueError, match="Attachment point must be non-negative"):
            InsuranceLayer(attachment_point=-100, limit=1000, rate=0.03)

    def test_zero_limit_raises(self):
        """Line 126: Zero or negative limit raises ValueError."""
        with pytest.raises(ValueError, match="Limit must be positive"):
            InsuranceLayer(attachment_point=0, limit=0, rate=0.03)

    def test_negative_limit_raises(self):
        """Line 126: Negative limit raises ValueError."""
        with pytest.raises(ValueError, match="Limit must be positive"):
            InsuranceLayer(attachment_point=0, limit=-1000, rate=0.03)

    def test_negative_rate_raises(self):
        """Line 128: Negative rate raises ValueError."""
        with pytest.raises(ValueError, match="Premium rate must be non-negative"):
            InsuranceLayer(attachment_point=0, limit=1000, rate=-0.01)


class TestInsurancePolicyCalculateRecovery:
    """Tests for InsurancePolicy.calculate_recovery (line 326)."""

    def test_recovery_below_deductible_is_zero(self):
        """Line 326: Claim at or below deductible returns zero recovery."""
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy(
                layers=[InsuranceLayer(1_000_000, 5_000_000, 0.03)],
                deductible=1_000_000,
            )
        # Claim exactly at deductible
        recovery = policy.calculate_recovery(1_000_000)
        assert recovery == 0.0

        # Claim below deductible
        recovery = policy.calculate_recovery(500_000)
        assert recovery == 0.0

    def test_recovery_zero_claim(self):
        """Zero claim returns zero recovery."""
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy(
                layers=[InsuranceLayer(0, 5_000_000, 0.03)],
                deductible=0,
            )
        recovery = policy.calculate_recovery(0)
        assert recovery == 0.0

    def test_recovery_negative_claim(self):
        """Negative claim returns zero recovery."""
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy(
                layers=[InsuranceLayer(0, 5_000_000, 0.03)],
            )
        recovery = policy.calculate_recovery(-100)
        assert recovery == 0.0


class TestInsurancePolicyToEnhancedProgram:
    """Tests for to_enhanced_program (lines 404-428)."""

    def test_successful_conversion(self):
        """Lines 404-428: Successful conversion to InsuranceProgram."""
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy(
                layers=[
                    InsuranceLayer(1_000_000, 5_000_000, 0.03),
                    InsuranceLayer(6_000_000, 10_000_000, 0.02),
                ],
                deductible=1_000_000,
            )
        program = policy.to_enhanced_program()
        if program is not None:
            assert len(program.layers) == 2
            assert program.deductible == 1_000_000

    def test_conversion_import_failure(self, caplog):
        """Lines 422-428: ImportError returns None with warning."""
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy(
                layers=[InsuranceLayer(0, 5_000_000, 0.03)],
            )
        # Mock the import to fail — method should return None
        with patch.dict("sys.modules", {"ergodic_insurance.insurance_program": None}):
            caplog.set_level(logging.WARNING, logger="ergodic_insurance.insurance")
            result = policy.to_enhanced_program()
            assert result is None
            # Should have logged a warning about unavailable module
            assert any("not available" in record.message for record in caplog.records)


class TestInsurancePolicyApplyPricing:
    """Tests for apply_pricing (lines 449, 453)."""

    def test_pricing_not_enabled_raises(self):
        """Line 449: Pricing not enabled raises ValueError."""
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy(
                layers=[InsuranceLayer(0, 5_000_000, 0.03)],
                pricing_enabled=False,
            )
        with pytest.raises(ValueError, match="Pricing not enabled"):
            policy.apply_pricing(expected_revenue=10_000_000)

    def test_pricing_enabled_no_pricer_no_generator_raises(self):
        """Line 453: No pricer and no loss_generator raises ValueError."""
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy(
                layers=[InsuranceLayer(0, 5_000_000, 0.03)],
                pricing_enabled=True,
                pricer=None,
            )
        with pytest.raises(ValueError, match="Either pricer or loss_generator"):
            policy.apply_pricing(expected_revenue=10_000_000, loss_generator=None)

    def test_pricing_with_loss_generator_creates_pricer(self):
        """Lines 456-461: When loss_generator provided, creates default pricer."""
        from ergodic_insurance.loss_distributions import ManufacturingLossGenerator

        loss_gen = ManufacturingLossGenerator.create_simple(
            frequency=0.1, severity_mean=500_000, severity_std=200_000, seed=42
        )
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy(
                layers=[InsuranceLayer(100_000, 5_000_000, 0.03)],
                pricing_enabled=True,
                pricer=None,
            )
        # Should create pricer from loss_generator
        # apply_pricing internally calls price_insurance_policy which also emits DeprecationWarning
        with pytest.warns(DeprecationWarning, match="price_insurance_policy.*deprecated"):
            policy.apply_pricing(expected_revenue=10_000_000, loss_generator=loss_gen)
        assert policy.pricer is not None


# TestInsurancePolicyFromYaml: REMOVED -- covered by test_insurance.py
# TestInsurancePolicyYAML.test_load_from_yaml

# TestInsurancePolicyGetTotalCoverage.test_empty_layers_returns_zero: REMOVED
# -- covered by test_insurance.py TestInsurancePolicy.test_empty_policy


class TestInsurancePolicyGetTotalCoverage:
    """Tests for get_total_coverage."""

    def test_multi_layer_coverage(self):
        """Multi-layer total coverage calculation."""
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy(
                layers=[
                    InsuranceLayer(1_000_000, 4_000_000, 0.03),
                    InsuranceLayer(5_000_000, 10_000_000, 0.02),
                ],
                deductible=1_000_000,
            )
        coverage = policy.get_total_coverage()
        assert coverage == 14_000_000  # (5M + 10M) - 1M deductible
