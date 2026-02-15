"""Tests for per-occurrence, aggregate, and hybrid insurance limit types.

This module tests the new functionality for different limit types in insurance layers,
ensuring proper behavior for per-occurrence (default), aggregate, and hybrid configurations.
"""

from typing import Any, Dict
import warnings

import numpy as np
import pytest

from ergodic_insurance.insurance_program import (
    EnhancedInsuranceLayer,
    InsuranceProgram,
    LayerState,
    ReinstatementType,
)


class TestPerOccurrenceLimits:
    """Test per-occurrence limit functionality."""

    def test_per_occurrence_default(self):
        """Test that per-occurrence is the default limit type."""
        layer = EnhancedInsuranceLayer(
            limit=1_000_000, attachment_point=100_000, base_premium_rate=0.02
        )
        assert layer.limit_type == "per-occurrence"

        # Process multiple large losses
        state = LayerState(layer)
        for _ in range(5):
            payment, _ = state.process_claim(2_000_000)
            assert payment == 1_000_000  # Each claim gets full limit

    def test_per_occurrence_no_exhaustion(self):
        """Test that per-occurrence limits never exhaust."""
        layer = EnhancedInsuranceLayer(
            limit=500_000, attachment_point=0, base_premium_rate=0.01, limit_type="per-occurrence"
        )

        state = LayerState(layer)

        # Process many claims
        total_paid = 0.0  # Fix: Declare as float
        for i in range(10):
            payment, reinstatement_premium = state.process_claim(1_000_000)
            assert payment == 500_000  # Each claim limited to 500K
            assert reinstatement_premium == 0  # No reinstatement premiums
            total_paid += payment

        assert total_paid == 5_000_000  # 10 claims * 500K each
        assert not state.is_exhausted  # Never exhausts

    def test_per_occurrence_reinstatement_warning(self, caplog):
        """Test that reinstatements generate warning for per-occurrence limits."""
        import logging

        with caplog.at_level(logging.WARNING, logger="ergodic_insurance.insurance_program"):
            layer = EnhancedInsuranceLayer(
                limit=1_000_000,
                attachment_point=100_000,
                base_premium_rate=0.02,
                limit_type="per-occurrence",
                reinstatements=2,  # Should trigger warning
            )

            # Check that a warning was raised
            warning_records = [
                r for r in caplog.records if "not used for per-occurrence limits" in r.message
            ]
            assert len(warning_records) == 1


class TestAggregateLimits:
    """Test aggregate limit functionality."""

    def test_aggregate_limit_exhaustion(self):
        """Test that aggregate limits exhaust after cumulative claims."""
        layer = EnhancedInsuranceLayer(
            limit=5_000_000,
            attachment_point=100_000,
            base_premium_rate=0.02,
            limit_type="aggregate",
        )

        state = LayerState(layer)
        payments = []
        for _ in range(3):
            payment, _ = state.process_claim(2_000_000)
            payments.append(payment)

        assert sum(payments) == 5_000_000  # Total limited to aggregate
        assert payments[2] == 1_000_000  # Last payment partial

        # Additional claims should not be paid
        payment, _ = state.process_claim(1_000_000)
        assert payment == 0
        assert state.is_exhausted

    def test_aggregate_with_reinstatements(self):
        """Test aggregate limits with reinstatements."""
        layer = EnhancedInsuranceLayer(
            limit=1_000_000,
            attachment_point=0,
            base_premium_rate=0.02,
            limit_type="aggregate",
            reinstatements=2,
            reinstatement_type=ReinstatementType.FULL,
        )

        state = LayerState(layer)

        # First claim exhausts the limit
        payment1, premium1 = state.process_claim(1_000_000)
        assert payment1 == 1_000_000
        assert premium1 == 20_000  # Full reinstatement premium (1M * 0.02)
        assert state.reinstatements_used == 1

        # Second claim uses reinstated limit
        payment2, premium2 = state.process_claim(1_000_000)
        assert payment2 == 1_000_000
        assert premium2 == 20_000  # Another reinstatement
        assert state.reinstatements_used == 2

        # Third claim uses last reinstated limit
        payment3, premium3 = state.process_claim(1_000_000)
        assert payment3 == 1_000_000
        assert premium3 == 0  # No more reinstatements
        assert state.is_exhausted


class TestHybridLimits:
    """Test hybrid limit functionality."""

    def test_hybrid_limits(self):
        """Test that hybrid limits apply both constraints."""
        layer = EnhancedInsuranceLayer(
            limit_type="hybrid",
            per_occurrence_limit=1_000_000,
            aggregate_limit=3_000_000,
            attachment_point=100_000,
            base_premium_rate=0.02,
            limit=1_000_000,  # Required field, used as per-occurrence if not specified
        )

        state = LayerState(layer)

        # First loss: 2M claim, limited to 1M per-occurrence
        payment1, _ = state.process_claim(2_000_000)
        assert payment1 == 1_000_000

        # Second loss: Another 2M claim, limited to 1M
        payment2, _ = state.process_claim(2_000_000)
        assert payment2 == 1_000_000

        # Third loss: 2M claim, but aggregate is nearly exhausted
        payment3, _ = state.process_claim(2_000_000)
        assert payment3 == 1_000_000

        # Fourth loss: Aggregate exhausted
        payment4, _ = state.process_claim(2_000_000)
        assert payment4 == 0
        assert state.is_exhausted

    def test_hybrid_partial_aggregate_exhaustion(self):
        """Test hybrid limits when aggregate partially exhausts."""
        layer = EnhancedInsuranceLayer(
            limit_type="hybrid",
            per_occurrence_limit=1_000_000,
            aggregate_limit=2_500_000,
            attachment_point=0,
            base_premium_rate=0.02,
            limit=1_000_000,
        )

        state = LayerState(layer)

        # First two claims: 1M each
        payment1, _ = state.process_claim(1_000_000)
        payment2, _ = state.process_claim(1_000_000)
        assert payment1 == 1_000_000
        assert payment2 == 1_000_000

        # Third claim: Only 500K remaining in aggregate
        payment3, _ = state.process_claim(1_000_000)
        assert payment3 == 500_000  # Limited by remaining aggregate
        assert state.is_exhausted

    def test_hybrid_with_reinstatements(self):
        """Test hybrid limits with reinstatements on aggregate portion."""
        layer = EnhancedInsuranceLayer(
            limit_type="hybrid",
            per_occurrence_limit=500_000,
            aggregate_limit=1_000_000,
            attachment_point=0,
            base_premium_rate=0.02,
            reinstatements=1,
            reinstatement_type=ReinstatementType.FULL,
            limit=500_000,
        )

        state = LayerState(layer)

        # First two claims exhaust aggregate
        payment1, premium1 = state.process_claim(600_000)  # Limited to 500K per-occurrence
        payment2, premium2 = state.process_claim(600_000)  # Limited to 500K per-occurrence

        assert payment1 == 500_000
        assert payment2 == 500_000
        assert premium1 == 0  # No reinstatement yet
        assert premium2 == 10_000  # Reinstatement triggered (500K * 0.02)

        # Aggregate is reinstated
        assert not state.is_exhausted

        # Third claim uses reinstated aggregate
        payment3, _ = state.process_claim(600_000)
        assert payment3 == 500_000


class TestReinstatementBehavior:
    """Test reinstatement behavior for different limit types."""

    def test_reinstatements_with_limit_types(self, caplog):
        """Test reinstatement behavior for different limit types."""
        import logging

        # Per-occurrence: reinstatements ignored with warning
        with caplog.at_level(logging.WARNING, logger="ergodic_insurance.insurance_program"):
            layer_po = EnhancedInsuranceLayer(
                limit=1_000_000,
                attachment_point=100_000,
                base_premium_rate=0.02,
                limit_type="per-occurrence",
                reinstatements=2,  # Should be ignored with a warning
            )
            warning_records = [
                r for r in caplog.records if "not used for per-occurrence limits" in r.message
            ]
            assert len(warning_records) == 1

        # Aggregate: reinstatements work
        layer_agg = EnhancedInsuranceLayer(
            limit=1_000_000,
            attachment_point=100_000,
            base_premium_rate=0.02,
            limit_type="aggregate",
            reinstatements=2,
        )

        state_agg = LayerState(layer_agg)

        # Exhaust and reinstate
        payment1, premium1 = state_agg.process_claim(1_000_000)
        assert payment1 == 1_000_000
        assert premium1 > 0  # Reinstatement premium charged

        # Can still claim after reinstatement
        payment2, _ = state_agg.process_claim(500_000)
        assert payment2 == 500_000


class TestConfigurationMigration:
    """Test configuration migration and backward compatibility."""

    def test_configuration_migration(self):
        """Test that existing configs default to per-occurrence limits."""
        old_config = {
            "limit": 5_000_000.0,
            "attachment_point": 100_000.0,
            "base_premium_rate": 0.02,
            # No limit_type specified
        }

        layer = EnhancedInsuranceLayer(**old_config)  # type: ignore[arg-type]
        # Should default to per-occurrence for new behavior
        assert layer.limit_type == "per-occurrence"

    def test_explicit_aggregate_configuration(self):
        """Test explicit aggregate configuration."""
        config: Dict[str, Any] = {
            "limit": 5_000_000.0,
            "attachment_point": 100_000.0,
            "base_premium_rate": 0.02,
            "limit_type": "aggregate",
        }

        layer = EnhancedInsuranceLayer(**config)
        assert layer.limit_type == "aggregate"
        assert layer.aggregate_limit == 5_000_000  # Set automatically

    def test_hybrid_configuration_validation(self):
        """Test hybrid configuration validation."""
        # Valid hybrid config
        valid_config: Dict[str, Any] = {
            "limit": 1_000_000.0,
            "attachment_point": 100_000.0,
            "base_premium_rate": 0.02,
            "limit_type": "hybrid",
            "per_occurrence_limit": 1_000_000.0,
            "aggregate_limit": 5_000_000.0,
        }

        layer = EnhancedInsuranceLayer(**valid_config)
        assert layer.limit_type == "hybrid"
        assert layer.per_occurrence_limit == 1_000_000
        assert layer.aggregate_limit == 5_000_000

        # Invalid hybrid config (missing aggregate)
        invalid_config: Dict[str, Any] = {
            "limit": 1_000_000.0,
            "attachment_point": 100_000.0,
            "base_premium_rate": 0.02,
            "limit_type": "hybrid",
            "per_occurrence_limit": 1_000_000.0,
            # Missing aggregate_limit
        }

        with pytest.raises(ValueError, match="Hybrid limit type requires"):
            EnhancedInsuranceLayer(**invalid_config)


class TestIntegrationScenarios:
    """Test integration scenarios with insurance programs."""

    def test_crisis_scenario_with_per_occurrence(self):
        """Test that crisis scenario improves with per-occurrence limits."""
        # Create program with per-occurrence limits
        layers_po = [
            EnhancedInsuranceLayer(
                attachment_point=100_000,
                limit=1_000_000,
                base_premium_rate=0.02,
                limit_type="per-occurrence",
            ),
            EnhancedInsuranceLayer(
                attachment_point=1_100_000,
                limit=5_000_000,
                base_premium_rate=0.01,
                limit_type="per-occurrence",
            ),
        ]

        program_po = InsuranceProgram(layers=layers_po, deductible=100_000)

        # Simulate crisis with multiple large losses
        crisis_losses = [2_000_000.0, 3_000_000.0, 2_500_000.0, 4_000_000.0]

        result_po = program_po.process_annual_claims(crisis_losses)

        # All losses should be covered up to per-occurrence limits
        assert result_po["total_recovery"] > 0

        # Compare with aggregate limits
        layers_agg = [
            EnhancedInsuranceLayer(
                attachment_point=100_000,
                limit=1_000_000,
                base_premium_rate=0.02,
                limit_type="aggregate",
            ),
            EnhancedInsuranceLayer(
                attachment_point=1_100_000,
                limit=5_000_000,
                base_premium_rate=0.01,
                limit_type="aggregate",
            ),
        ]

        program_agg = InsuranceProgram(layers=layers_agg, deductible=100_000)
        result_agg = program_agg.process_annual_claims(crisis_losses)

        # Per-occurrence should provide better coverage in crisis
        assert result_po["total_recovery"] > result_agg["total_recovery"]

    def test_premium_calculation_consistency(self):
        """Test that premiums calculate correctly for all limit types."""
        base_premium_rate = 0.02
        limit = 5_000_000
        expected_premium = limit * base_premium_rate

        # Per-occurrence
        layer_po = EnhancedInsuranceLayer(
            attachment_point=100_000,
            limit=limit,
            base_premium_rate=base_premium_rate,
            limit_type="per-occurrence",
        )
        assert layer_po.calculate_base_premium() == expected_premium

        # Aggregate
        layer_agg = EnhancedInsuranceLayer(
            attachment_point=100_000,
            limit=limit,
            base_premium_rate=base_premium_rate,
            limit_type="aggregate",
        )
        assert layer_agg.calculate_base_premium() == expected_premium

        # Hybrid
        layer_hybrid = EnhancedInsuranceLayer(
            attachment_point=100_000,
            limit=limit,
            base_premium_rate=base_premium_rate,
            limit_type="hybrid",
            per_occurrence_limit=limit,
            aggregate_limit=limit * 3,
        )
        assert layer_hybrid.calculate_base_premium() == expected_premium
