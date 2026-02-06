"""Additional coverage tests for insurance_program.py.

Targets specific uncovered lines to improve coverage from 86.63% toward 100%.
Focuses on: validation edge cases, hybrid/aggregate limit types, reinstatement
types, _process_claim_aggregate fallback, get_available_limit/get_utilization_rate
edge cases, get_program_summary, get_total_coverage empty, ergodic benefit empty,
_round_attachment_point ranges, _get_layer_capacity, _get_base_premium_rate,
optimize_layer_widths empty, get_pricing_summary, ProgramState empty stats.
"""

from pathlib import Path
import tempfile
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from ergodic_insurance.insurance_program import (
    EnhancedInsuranceLayer,
    InsuranceProgram,
    LayerState,
    OptimalStructure,
    OptimizationConstraints,
    ProgramState,
    ReinstatementType,
)

# ===========================================================================
# EnhancedInsuranceLayer: validation edge cases (lines 101, 108, 132-147)
# ===========================================================================


class TestEnhancedInsuranceLayerValidation:
    """Test validation logic in EnhancedInsuranceLayer.__post_init__."""

    def test_negative_reinstatement_premium_raises(self):
        """Line 101: negative reinstatement_premium raises ValueError."""
        with pytest.raises(ValueError, match="Reinstatement premium must be non-negative"):
            EnhancedInsuranceLayer(
                attachment_point=0,
                limit=1_000_000,
                base_premium_rate=0.01,
                reinstatement_premium=-0.5,
            )

    def test_invalid_limit_type_raises(self):
        """Line 108: invalid limit_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid limit_type"):
            EnhancedInsuranceLayer(
                attachment_point=0,
                limit=1_000_000,
                base_premium_rate=0.01,
                limit_type="invalid_type",
            )

    def test_hybrid_default_per_occurrence_limit(self):
        """Line 132: hybrid type sets per_occurrence_limit to limit when None."""
        layer = EnhancedInsuranceLayer(
            attachment_point=0,
            limit=1_000_000,
            base_premium_rate=0.01,
            limit_type="hybrid",
            aggregate_limit=5_000_000,
        )
        assert layer.per_occurrence_limit == 1_000_000

    def test_hybrid_missing_aggregate_limit_raises(self):
        """Line 135-137: hybrid without aggregate_limit raises ValueError."""
        with pytest.raises(ValueError, match="Hybrid limit type requires"):
            EnhancedInsuranceLayer(
                attachment_point=0,
                limit=1_000_000,
                base_premium_rate=0.01,
                limit_type="hybrid",
                per_occurrence_limit=500_000,
                aggregate_limit=None,
            )

    def test_hybrid_negative_per_occurrence_limit_raises(self):
        """Line 139: negative per_occurrence_limit raises ValueError."""
        with pytest.raises(ValueError, match="Per-occurrence limit must be positive"):
            EnhancedInsuranceLayer(
                attachment_point=0,
                limit=1_000_000,
                base_premium_rate=0.01,
                limit_type="hybrid",
                per_occurrence_limit=-100,
                aggregate_limit=5_000_000,
            )

    def test_hybrid_negative_aggregate_limit_raises(self):
        """Line 143: negative aggregate_limit in hybrid raises ValueError."""
        with pytest.raises(ValueError, match="Aggregate limit must be positive"):
            EnhancedInsuranceLayer(
                attachment_point=0,
                limit=1_000_000,
                base_premium_rate=0.01,
                limit_type="hybrid",
                per_occurrence_limit=500_000,
                aggregate_limit=-100,
            )

    def test_aggregate_limit_negative_standalone_raises(self):
        """Line 147: negative aggregate_limit on per-occurrence type raises ValueError."""
        with pytest.raises(ValueError, match="Aggregate limit must be positive"):
            EnhancedInsuranceLayer(
                attachment_point=0,
                limit=1_000_000,
                base_premium_rate=0.01,
                limit_type="per-occurrence",
                aggregate_limit=-500,
            )


# ===========================================================================
# EnhancedInsuranceLayer: reinstatement type NONE (line 184)
# ===========================================================================


class TestReinstatementTypeNone:
    """Test reinstatement premium returns 0 for NONE type."""

    def test_none_reinstatement_type_returns_zero(self):
        """Line 184: ReinstatementType.NONE returns 0.0."""
        layer = EnhancedInsuranceLayer(
            attachment_point=0,
            limit=1_000_000,
            base_premium_rate=0.02,
            reinstatement_type=ReinstatementType.NONE,
        )
        premium = layer.calculate_reinstatement_premium(timing_factor=0.5)
        assert premium == 0.0


# ===========================================================================
# EnhancedInsuranceLayer: calculate_layer_loss for different types (219-224)
# ===========================================================================


class TestCalculateLayerLossTypes:
    """Test calculate_layer_loss for aggregate and hybrid types."""

    def test_aggregate_layer_loss(self):
        """Lines 215-218: aggregate type returns excess up to limit."""
        layer = EnhancedInsuranceLayer(
            attachment_point=1_000_000,
            limit=5_000_000,
            base_premium_rate=0.01,
            limit_type="aggregate",
        )
        # Loss exceeding layer: should cap at limit
        assert layer.calculate_layer_loss(10_000_000) == 5_000_000
        # Loss within layer
        assert layer.calculate_layer_loss(3_000_000) == 2_000_000
        # Loss below attachment
        assert layer.calculate_layer_loss(500_000) == 0.0

    def test_hybrid_layer_loss(self):
        """Lines 219-222: hybrid type uses per_occurrence_limit."""
        layer = EnhancedInsuranceLayer(
            attachment_point=1_000_000,
            limit=5_000_000,
            base_premium_rate=0.01,
            limit_type="hybrid",
            per_occurrence_limit=2_000_000,
            aggregate_limit=10_000_000,
        )
        # Loss exceeds per_occurrence_limit: capped at 2M
        assert layer.calculate_layer_loss(10_000_000) == 2_000_000
        # Loss within per_occurrence_limit
        assert layer.calculate_layer_loss(2_500_000) == 1_500_000

    def test_unknown_limit_type_fallback(self):
        """Lines 223-224: unknown limit_type falls back to min(excess, limit)."""
        layer = EnhancedInsuranceLayer(
            attachment_point=1_000_000,
            limit=5_000_000,
            base_premium_rate=0.01,
        )
        # Directly override limit_type to trigger fallback
        layer.limit_type = "unknown"
        assert layer.calculate_layer_loss(10_000_000) == 5_000_000


# ===========================================================================
# LayerState: hybrid and aggregate edge cases (259, 286-287, 356-362, 380-381)
# ===========================================================================


class TestLayerStateHybrid:
    """Test LayerState.process_claim for hybrid limit_type."""

    def test_hybrid_exhausted_returns_zero(self):
        """Line 369-370: hybrid when already exhausted returns (0, 0)."""
        layer = EnhancedInsuranceLayer(
            attachment_point=0,
            limit=1_000_000,
            base_premium_rate=0.01,
            limit_type="hybrid",
            per_occurrence_limit=500_000,
            aggregate_limit=1_000_000,
        )
        state = LayerState(layer)
        state.is_exhausted = True
        payment, premium = state.process_claim(500_000)
        assert payment == 0.0
        assert premium == 0.0

    def test_hybrid_aggregate_exhausted_returns_zero(self):
        """Lines 380-381: hybrid aggregate remaining <= 0 returns (0, 0)."""
        layer = EnhancedInsuranceLayer(
            attachment_point=0,
            limit=1_000_000,
            base_premium_rate=0.01,
            limit_type="hybrid",
            per_occurrence_limit=500_000,
            aggregate_limit=1_000_000,
        )
        state = LayerState(layer)
        # Use up the aggregate
        state.aggregate_used = 1_000_000
        payment, premium = state.process_claim(500_000)
        assert payment == 0.0
        assert state.is_exhausted

    def test_hybrid_reinstatement_after_aggregate_exhaust(self):
        """Lines 398-403: hybrid triggers reinstatement when aggregate exhausted."""
        layer = EnhancedInsuranceLayer(
            attachment_point=0,
            limit=1_000_000,
            base_premium_rate=0.02,
            limit_type="hybrid",
            per_occurrence_limit=500_000,
            aggregate_limit=1_000_000,
            reinstatements=1,
            reinstatement_premium=1.0,
            reinstatement_type=ReinstatementType.FULL,
        )
        state = LayerState(layer)
        # First two claims exhaust aggregate (500K + 500K = 1M)
        state.process_claim(500_000)
        payment, reinstatement_premium = state.process_claim(500_000)
        # Should exhaust aggregate and trigger reinstatement
        assert payment == 500_000
        assert reinstatement_premium == 20_000  # Full base premium
        assert not state.is_exhausted  # Reinstated


class TestLayerStateAggregateEdgeCases:
    """Test aggregate edge cases in process_claim."""

    def test_aggregate_remaining_zero_exhausts(self):
        """Lines 286-287: remaining aggregate <= 0 sets exhausted."""
        layer = EnhancedInsuranceLayer(
            attachment_point=0,
            limit=1_000_000,
            base_premium_rate=0.01,
            limit_type="aggregate",
            aggregate_limit=1_000_000,
            reinstatements=0,
        )
        state = LayerState(layer)
        state.aggregate_used = 1_000_000  # already exhausted
        payment, _ = state.process_claim(100_000)
        assert payment == 0.0
        assert state.is_exhausted

    def test_aggregate_claim_fully_processed_with_reinstatement(self):
        """Lines 356-362: claim fully processed with limit at 0 triggers reinstatement."""
        layer = EnhancedInsuranceLayer(
            attachment_point=0,
            limit=500_000,
            base_premium_rate=0.02,
            limit_type="aggregate",
            aggregate_limit=500_000,
            reinstatements=2,
            reinstatement_premium=0.5,
            reinstatement_type=ReinstatementType.PRO_RATA,
        )
        state = LayerState(layer)
        # Exactly exhaust the limit
        payment, premium = state.process_claim(500_000, timing_factor=0.8)
        assert payment == 500_000
        # Should trigger reinstatement since limit exhausted
        assert state.reinstatements_used >= 1
        assert state.current_limit == 500_000  # Reinstated


# ===========================================================================
# LayerState: _process_claim_aggregate fallback (409, 415-469)
# ===========================================================================


class TestProcessClaimAggregateFallback:
    """Test _process_claim_aggregate method directly."""

    def test_fallback_aggregate_processing(self):
        """Lines 409, 415-469: process via _process_claim_aggregate."""
        layer = EnhancedInsuranceLayer(
            attachment_point=0,
            limit=2_000_000,
            base_premium_rate=0.01,
            limit_type="aggregate",
            aggregate_limit=3_000_000,
            reinstatements=1,
            reinstatement_premium=1.0,
            reinstatement_type=ReinstatementType.FULL,
        )
        state = LayerState(layer)
        # Directly call _process_claim_aggregate
        payment, premium = state._process_claim_aggregate(1_500_000, 0.8)
        assert payment == 1_500_000

    def test_fallback_aggregate_exhausted(self):
        """Line 415: _process_claim_aggregate returns (0,0) when exhausted."""
        layer = EnhancedInsuranceLayer(
            attachment_point=0,
            limit=1_000_000,
            base_premium_rate=0.01,
            limit_type="aggregate",
        )
        state = LayerState(layer)
        state.is_exhausted = True
        payment, premium = state._process_claim_aggregate(500_000)
        assert payment == 0.0
        assert premium == 0.0

    def test_fallback_aggregate_zero_claim(self):
        """Line 415: zero claim returns (0,0)."""
        layer = EnhancedInsuranceLayer(
            attachment_point=0,
            limit=1_000_000,
            base_premium_rate=0.01,
            limit_type="aggregate",
        )
        state = LayerState(layer)
        payment, premium = state._process_claim_aggregate(0)
        assert payment == 0.0
        assert premium == 0.0

    def test_fallback_aggregate_with_reinstatement(self):
        """Lines 451-460: reinstatement triggered in _process_claim_aggregate."""
        layer = EnhancedInsuranceLayer(
            attachment_point=0,
            limit=500_000,
            base_premium_rate=0.02,
            reinstatements=1,
            reinstatement_premium=1.0,
            reinstatement_type=ReinstatementType.FULL,
        )
        state = LayerState(layer)
        # This should trigger the fallback path
        # We need to force the fallback by using an invalid limit_type
        layer.limit_type = "fallback_test"
        payment, premium = state.process_claim(700_000)
        # Should go through _process_claim_aggregate and exhaust + reinstate
        assert payment == 700_000
        assert premium > 0

    def test_fallback_aggregate_no_reinstatement_exhaustion(self):
        """Lines 461-464: exhaustion with no more reinstatements."""
        layer = EnhancedInsuranceLayer(
            attachment_point=0,
            limit=500_000,
            base_premium_rate=0.01,
            reinstatements=0,
        )
        state = LayerState(layer)
        layer.limit_type = "fallback_test"
        payment, _ = state.process_claim(600_000)
        assert payment == 500_000
        assert state.is_exhausted

    def test_fallback_aggregate_limit_reached(self):
        """Lines 442-448: aggregate limit reached in fallback path."""
        layer = EnhancedInsuranceLayer(
            attachment_point=0,
            limit=2_000_000,
            base_premium_rate=0.01,
            reinstatements=5,
            aggregate_limit=3_000_000,
        )
        state = LayerState(layer)
        # Call _process_claim_aggregate directly with a large claim
        payment1, _ = state._process_claim_aggregate(2_000_000)
        assert payment1 == 2_000_000
        payment2, _ = state._process_claim_aggregate(2_000_000)
        assert payment2 == 1_000_000  # Only 1M aggregate remaining
        assert state.is_exhausted


# ===========================================================================
# LayerState: get_available_limit exhausted (line 487)
# ===========================================================================


class TestGetAvailableLimit:
    """Test get_available_limit when layer is exhausted."""

    def test_exhausted_returns_zero(self):
        """Line 487: get_available_limit returns 0 when exhausted."""
        layer = EnhancedInsuranceLayer(
            attachment_point=0,
            limit=1_000_000,
            base_premium_rate=0.01,
        )
        state = LayerState(layer)
        state.is_exhausted = True
        assert state.get_available_limit() == 0.0


# ===========================================================================
# LayerState: get_utilization_rate zero total (line 497)
# ===========================================================================


class TestGetUtilizationRate:
    """Test get_utilization_rate edge cases."""

    def test_zero_total_available_returns_zero(self):
        """Line 497: returns 0 when total_available is 0."""
        layer = EnhancedInsuranceLayer(
            attachment_point=0,
            limit=1_000_000,
            base_premium_rate=0.01,
            reinstatements=0,
        )
        state = LayerState(layer)
        # Override to make limit * (1 + reinstatements) = limit * 1
        # But we need total_available == 0. Only possible if limit == 0?
        # Actually the layer validates limit > 0, so we mock:
        original_limit = layer.limit
        layer.limit = 0  # bypass - only for testing the method
        layer.reinstatements = 0
        # Manually set to trigger the zero path
        result = state.get_utilization_rate()
        # limit is 0, so total_available = 0 * (1 + 0) = 0
        assert result == 0.0
        layer.limit = original_limit  # restore


# ===========================================================================
# InsuranceProgram: get_program_summary (line 691)
# ===========================================================================


class TestGetProgramSummary:
    """Test get_program_summary."""

    def test_program_summary_structure(self):
        """Line 691: verify summary dict structure."""
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=250_000, limit=5_000_000, base_premium_rate=0.015
            ),
            EnhancedInsuranceLayer(
                attachment_point=5_000_000, limit=20_000_000, base_premium_rate=0.008
            ),
        ]
        program = InsuranceProgram(layers, deductible=250_000, name="Test Program")
        summary = program.get_program_summary()

        assert summary["program_name"] == "Test Program"
        assert summary["deductible"] == 250_000
        assert summary["num_layers"] == 2
        assert summary["total_coverage"] > 0
        assert summary["annual_base_premium"] > 0
        assert len(summary["layers"]) == 2
        assert "attachment" in summary["layers"][0]
        assert "exhaustion_point" in summary["layers"][0]


# ===========================================================================
# InsuranceProgram: get_total_coverage empty (line 718)
# ===========================================================================


class TestGetTotalCoverageEmpty:
    """Test get_total_coverage with no layers."""

    def test_empty_layers_returns_zero(self):
        """Line 718: no layers returns 0.0."""
        program = InsuranceProgram(layers=[], deductible=0)
        assert program.get_total_coverage() == 0.0


# ===========================================================================
# InsuranceProgram: calculate_ergodic_benefit empty (line 795)
# ===========================================================================


class TestErgodicBenefitEmpty:
    """Test ergodic benefit with empty loss history."""

    def test_empty_loss_history_returns_zeros(self):
        """Line 795: empty loss_history returns zeros."""
        program = InsuranceProgram.create_standard_manufacturing_program()
        result = program.calculate_ergodic_benefit([])
        assert result["time_average_benefit"] == 0.0
        assert result["ensemble_average_cost"] == 0.0
        assert result["ergodic_ratio"] == 0.0
        assert result["volatility_reduction"] == 0.0


# ===========================================================================
# InsuranceProgram: _round_attachment_point ranges (line 874)
# ===========================================================================


class TestRoundAttachmentPointEdgeCases:
    """Test _round_attachment_point for boundary ranges."""

    @pytest.fixture
    def program(self):
        return InsuranceProgram.create_standard_manufacturing_program()

    def test_below_100k(self, program):
        """Line 894-895: values < 100K round to nearest 10K."""
        assert program._round_attachment_point(55_000) == 60_000
        # Use 6_000 instead of 5_000 to avoid banker's rounding (round(0.5)=0)
        assert program._round_attachment_point(6_000) == 10_000

    def test_between_100k_and_1m(self, program):
        """Line 896-897: values 100K-1M round to nearest 50K."""
        assert program._round_attachment_point(275_000) == 300_000
        # Use 730_000 instead of 725_000 to avoid banker's rounding (round(14.5)=14)
        assert program._round_attachment_point(730_000) == 750_000

    def test_between_1m_and_10m(self, program):
        """Line 898-899: values 1M-10M round to nearest 250K."""
        # Use 3_200_000 instead of 3_125_000 to avoid banker's rounding (round(12.5)=12)
        assert program._round_attachment_point(3_200_000) == 3_250_000

    def test_above_10m(self, program):
        """Line 900: values > 10M round to nearest 1M."""
        assert program._round_attachment_point(15_500_000) == 16_000_000


# ===========================================================================
# InsuranceProgram: _get_layer_capacity (line 913)
# ===========================================================================


class TestGetLayerCapacity:
    """Test _get_layer_capacity for different attachment thresholds."""

    @pytest.fixture
    def program(self):
        return InsuranceProgram.create_standard_manufacturing_program()

    def test_below_1m(self, program):
        """Line 910-912: attachment < 1M returns 5M."""
        assert program._get_layer_capacity(500_000) == 5_000_000

    def test_below_10m(self, program):
        """Line 910-912: attachment 1M-10M returns 25M."""
        assert program._get_layer_capacity(5_000_000) == 25_000_000

    def test_below_50m(self, program):
        """Line 910-912: attachment 10M-50M returns 50M."""
        assert program._get_layer_capacity(25_000_000) == 50_000_000

    def test_above_50m(self, program):
        """Line 910-912: attachment >= 50M returns 100M."""
        assert program._get_layer_capacity(75_000_000) == 100_000_000

    def test_default_fallback(self, program):
        """Line 913: default fallback returns 100M."""
        # This tests the unreachable fallback at the end
        # The for loop always returns for float("inf"), so this is technically
        # unreachable, but we test the boundary
        assert program._get_layer_capacity(float("inf")) == 100_000_000


# ===========================================================================
# InsuranceProgram: optimize_layer_widths empty (line 934)
# ===========================================================================


class TestOptimizeLayerWidthsEmpty:
    """Test optimize_layer_widths with empty inputs."""

    def test_empty_attachment_points_returns_empty(self):
        """Line 934: empty attachment_points returns []."""
        program = InsuranceProgram.create_standard_manufacturing_program()
        result = program.optimize_layer_widths([], 100_000)
        assert result == []


# ===========================================================================
# InsuranceProgram: optimize_layer_widths zero weight (line 960)
# ===========================================================================


class TestOptimizeLayerWidthsZeroWeight:
    """Test optimize_layer_widths with zero severity weights."""

    def test_zero_severity_weights_fallback(self):
        """Line 960: all zero weights distribute evenly."""
        program = InsuranceProgram.create_standard_manufacturing_program()
        # Pass loss_data where no losses exceed any attachment point
        # so excess_losses will be empty, and avg_excess = ap (fallback)
        # total_weight will be sum of attachment points
        attachment_points: list[float] = [250_000.0, 1_000_000.0]
        result = program.optimize_layer_widths(
            attachment_points, 200_000, loss_data=[1.0, 2.0, 3.0]
        )
        assert len(result) == 2
        assert all(w > 0 for w in result)


# ===========================================================================
# InsuranceProgram: _get_base_premium_rate (line 1002)
# ===========================================================================


class TestGetBasePremiumRate:
    """Test _get_base_premium_rate for different attachment thresholds."""

    @pytest.fixture
    def program(self):
        return InsuranceProgram.create_standard_manufacturing_program()

    def test_below_1m(self, program):
        """Line 999-1001: attachment < 1M returns 0.015."""
        assert program._get_base_premium_rate(500_000) == 0.015

    def test_between_1m_and_5m(self, program):
        """Line 999-1001: attachment 1M-5M returns 0.010."""
        assert program._get_base_premium_rate(3_000_000) == 0.010

    def test_between_5m_and_25m(self, program):
        """Line 999-1001: attachment 5M-25M returns 0.006."""
        assert program._get_base_premium_rate(15_000_000) == 0.006

    def test_above_25m(self, program):
        """Line 999-1001: attachment >= 25M returns 0.003."""
        assert program._get_base_premium_rate(50_000_000) == 0.003

    def test_default_fallback(self, program):
        """Line 1002: fallback returns 0.003."""
        assert program._get_base_premium_rate(float("inf")) == 0.003


# ===========================================================================
# InsuranceProgram: find_optimal_attachment_points edge case (line 888)
# ===========================================================================


class TestFindOptimalAttachmentPointsEdge:
    """Test find_optimal_attachment_points edge cases."""

    def test_non_increasing_corrected(self):
        """Line 887-888: non-increasing attachment points are corrected."""
        program = InsuranceProgram.create_standard_manufacturing_program()
        # Use data where most values are the same, causing ties in percentiles
        losses = [100_000.0] * 100  # All same value
        result = program.find_optimal_attachment_points(losses, num_layers=3)
        assert len(result) == 3
        # Should be strictly increasing after correction
        for i in range(1, len(result)):
            assert result[i] > result[i - 1]


# ===========================================================================
# InsuranceProgram: apply_pricing (line 1249)
# ===========================================================================


class TestApplyPricing:
    """Test apply_pricing error paths."""

    def test_pricing_not_enabled_raises(self):
        """Line 1244-1245: pricing_enabled=False raises ValueError."""
        program = InsuranceProgram.create_standard_manufacturing_program()
        with pytest.raises(ValueError, match="Pricing not enabled"):
            program.apply_pricing(expected_revenue=10_000_000)

    def test_pricing_no_pricer_no_generator_raises(self):
        """Line 1248-1249: no pricer and no loss_generator raises ValueError."""
        layers = [
            EnhancedInsuranceLayer(attachment_point=0, limit=1_000_000, base_premium_rate=0.01)
        ]
        program = InsuranceProgram(layers, pricing_enabled=True, pricer=None)
        with pytest.raises(ValueError, match="Either pricer or loss_generator"):
            program.apply_pricing(expected_revenue=10_000_000)


# ===========================================================================
# InsuranceProgram: get_pricing_summary without pricing_results (1297-1298)
# ===========================================================================


class TestGetPricingSummary:
    """Test get_pricing_summary with and without pricing results."""

    def test_summary_without_pricing_results(self):
        """Lines 1297-1298: summary without pricing data shows base info."""
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=250_000, limit=5_000_000, base_premium_rate=0.015
            ),
        ]
        program = InsuranceProgram(layers, deductible=250_000)
        summary = program.get_pricing_summary()

        assert summary["program_name"] == "Manufacturing Insurance Program"
        assert summary["pricing_enabled"] is False
        assert len(summary["layers"]) == 1
        assert "premium" in summary["layers"][0]
        assert summary["layers"][0]["premium"] == 5_000_000 * 0.015

    def test_summary_with_pricing_results(self):
        """Lines 1280-1295: summary with pricing results shows detailed info."""
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=250_000, limit=5_000_000, base_premium_rate=0.015
            ),
        ]
        program = InsuranceProgram(layers, pricing_enabled=True)

        # Mock pricing results
        pricing_result = Mock()
        pricing_result.market_premium = 80_000
        pricing_result.pure_premium = 60_000
        pricing_result.expected_frequency = 0.5
        pricing_result.expected_severity = 120_000
        program.pricing_results = [pricing_result]

        summary = program.get_pricing_summary()

        assert summary["layers"][0]["market_premium"] == 80_000
        assert summary["layers"][0]["pure_premium"] == 60_000
        assert summary["layers"][0]["expected_frequency"] == 0.5


# ===========================================================================
# ProgramState: get_summary_statistics empty (line 1407)
# ===========================================================================


class TestProgramStateEmpty:
    """Test ProgramState with no simulated years."""

    def test_empty_summary_statistics(self):
        """Line 1407: returns empty dict when no years simulated."""
        layers = [
            EnhancedInsuranceLayer(attachment_point=0, limit=5_000_000, base_premium_rate=0.01)
        ]
        program = InsuranceProgram(layers)
        state = ProgramState(program)
        stats = state.get_summary_statistics()
        assert stats == {}


# ===========================================================================
# InsuranceProgram: _calculate_reinstatements (line 1004-1010)
# ===========================================================================


class TestCalculateReinstatements:
    """Test _calculate_reinstatements for different layer positions."""

    @pytest.fixture
    def program(self):
        return InsuranceProgram.create_standard_manufacturing_program()

    def test_primary_layer_no_reinstatements(self, program):
        """Line 1006-1007: primary layer (index 0) gets 0."""
        assert program._calculate_reinstatements(0, 4) == 0

    def test_top_layer_unlimited_reinstatements(self, program):
        """Line 1008-1009: top layer gets 999."""
        assert program._calculate_reinstatements(3, 4) == 999

    def test_middle_layer_reinstatements(self, program):
        """Line 1010: middle layers get decreasing reinstatements."""
        assert program._calculate_reinstatements(1, 4) == 2
        assert program._calculate_reinstatements(2, 4) == 1


# ===========================================================================
# InsuranceProgram: create_standard_manufacturing_program (line 1224)
# ===========================================================================


class TestCreateStandardProgram:
    """Test create_standard_manufacturing_program returns correct structure."""

    def test_default_deductible(self):
        """Line 1224: uses default deductible and returns program."""
        program = InsuranceProgram.create_standard_manufacturing_program()
        assert program.deductible == 250_000
        assert program.name == "Standard Manufacturing Program"
        assert len(program.layers) == 4


# ===========================================================================
# InsuranceProgram: per-occurrence warning (line 116-122)
# ===========================================================================


class TestPerOccurrenceWarning:
    """Test per-occurrence layer with reinstatements emits warning."""

    def test_reinstatements_warning_for_per_occurrence(self):
        """Lines 116-122: per-occurrence with reinstatements warns."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            layer = EnhancedInsuranceLayer(
                attachment_point=0,
                limit=1_000_000,
                base_premium_rate=0.01,
                reinstatements=2,
                limit_type="per-occurrence",
            )
            assert len(w) >= 1
            assert "Reinstatements parameter" in str(w[0].message)


# ===========================================================================
# EnhancedInsuranceLayer: base premium with exposure (line 161-163)
# ===========================================================================


class TestBasePremiumWithExposure:
    """Test base premium calculation with exposure scaling."""

    def test_premium_with_exposure_multiplier(self):
        """Lines 161-163: exposure scaling applied to base premium."""
        mock_exposure = Mock()
        mock_exposure.get_frequency_multiplier.return_value = 2.0

        layer = EnhancedInsuranceLayer(
            attachment_point=0,
            limit=1_000_000,
            base_premium_rate=0.02,
            premium_rate_exposure=mock_exposure,
        )
        premium = layer.calculate_base_premium(time=1.0)
        # Base = 1M * 0.02 = 20K, with 2x multiplier = 40K
        assert premium == 40_000


# ===========================================================================
# InsuranceProgram: optimize_layer_structure no convergence (line 1110-1132)
# ===========================================================================


class TestOptimizeLayerStructureNoConvergence:
    """Test optimize_layer_structure fallback when no structure found."""

    def test_fallback_basic_structure(self):
        """Lines 1110-1132: returns basic structure when no optimization works."""
        program = InsuranceProgram.create_standard_manufacturing_program()
        # Use zero losses which will produce no positive all_losses
        result = program.optimize_layer_structure([[0], [0]], constraints=None)
        assert isinstance(result, OptimalStructure)


# ===========================================================================
# InsuranceProgram: from_yaml alt key names (line 1162-1164)
# ===========================================================================


class TestFromYamlAlternateKeys:
    """Test from_yaml with alternative premium rate key names."""

    def test_yaml_with_premium_rate_key(self):
        """Line 1162-1164: YAML with 'premium_rate' instead of 'base_premium_rate'."""
        config = {
            "program_name": "Alt Key Program",
            "deductible": 100_000,
            "layers": [
                {
                    "attachment_point": 100_000,
                    "limit": 1_000_000,
                    "premium_rate": 0.025,
                },
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            program = InsuranceProgram.from_yaml(temp_path)
            assert program.layers[0].base_premium_rate == 0.025
        finally:
            Path(temp_path).unlink()

    def test_yaml_with_rate_key(self):
        """Line 1162-1164: YAML with 'rate' key fallback."""
        config = {
            "program_name": "Rate Key Program",
            "deductible": 50_000,
            "layers": [
                {
                    "attachment_point": 50_000,
                    "limit": 500_000,
                    "rate": 0.03,
                },
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            program = InsuranceProgram.from_yaml(temp_path)
            assert program.layers[0].base_premium_rate == 0.03
        finally:
            Path(temp_path).unlink()
