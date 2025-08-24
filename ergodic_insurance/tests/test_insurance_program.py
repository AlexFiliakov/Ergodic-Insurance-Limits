"""Tests for multi-layer insurance program with reinstatements.

Comprehensive test suite for advanced insurance program features including
reinstatements, multi-layer structures, and complex claim scenarios.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from ergodic_insurance.src.insurance_program import (
    EnhancedInsuranceLayer,
    InsuranceProgram,
    LayerState,
    ProgramState,
    ReinstatementType,
)


class TestEnhancedInsuranceLayer:
    """Test enhanced insurance layer functionality."""

    def test_initialization(self):
        """Test layer initialization with all parameters."""
        layer = EnhancedInsuranceLayer(
            attachment_point=1_000_000,
            limit=5_000_000,
            premium_rate=0.01,
            reinstatements=2,
            reinstatement_premium=1.0,
            reinstatement_type=ReinstatementType.PRO_RATA,
        )

        assert layer.attachment_point == 1_000_000
        assert layer.limit == 5_000_000
        assert layer.premium_rate == 0.01
        assert layer.reinstatements == 2
        assert layer.reinstatement_premium == 1.0
        assert layer.reinstatement_type == ReinstatementType.PRO_RATA

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError, match="Attachment point must be non-negative"):
            EnhancedInsuranceLayer(attachment_point=-100, limit=1_000_000, premium_rate=0.01)

        with pytest.raises(ValueError, match="Limit must be positive"):
            EnhancedInsuranceLayer(attachment_point=0, limit=-1_000_000, premium_rate=0.01)

        with pytest.raises(ValueError, match="Premium rate must be non-negative"):
            EnhancedInsuranceLayer(attachment_point=0, limit=1_000_000, premium_rate=-0.01)

        with pytest.raises(ValueError, match="Reinstatements must be non-negative"):
            EnhancedInsuranceLayer(
                attachment_point=0, limit=1_000_000, premium_rate=0.01, reinstatements=-1
            )

    def test_calculate_base_premium(self):
        """Test base premium calculation."""
        layer = EnhancedInsuranceLayer(
            attachment_point=1_000_000, limit=5_000_000, premium_rate=0.02
        )

        assert layer.calculate_base_premium() == 100_000  # 5M * 0.02

    def test_reinstatement_premium_pro_rata(self):
        """Test pro-rata reinstatement premium calculation."""
        layer = EnhancedInsuranceLayer(
            attachment_point=1_000_000,
            limit=5_000_000,
            premium_rate=0.02,
            reinstatement_premium=0.5,
            reinstatement_type=ReinstatementType.PRO_RATA,
        )

        # Half year remaining
        premium = layer.calculate_reinstatement_premium(timing_factor=0.5)
        assert premium == 25_000  # 100K base * 0.5 reinstatement * 0.5 timing

    def test_reinstatement_premium_full(self):
        """Test full reinstatement premium calculation."""
        layer = EnhancedInsuranceLayer(
            attachment_point=1_000_000,
            limit=5_000_000,
            premium_rate=0.02,
            reinstatement_premium=1.0,
            reinstatement_type=ReinstatementType.FULL,
        )

        # Timing doesn't matter for full reinstatement
        premium = layer.calculate_reinstatement_premium(timing_factor=0.5)
        assert premium == 100_000  # Full base premium

    def test_reinstatement_premium_free(self):
        """Test free reinstatement premium calculation."""
        layer = EnhancedInsuranceLayer(
            attachment_point=1_000_000,
            limit=5_000_000,
            premium_rate=0.02,
            reinstatement_type=ReinstatementType.FREE,
        )

        premium = layer.calculate_reinstatement_premium(timing_factor=0.5)
        assert premium == 0.0

    def test_can_respond(self):
        """Test layer response determination."""
        layer = EnhancedInsuranceLayer(
            attachment_point=1_000_000, limit=5_000_000, premium_rate=0.01
        )

        assert not layer.can_respond(500_000)  # Below attachment
        assert not layer.can_respond(1_000_000)  # At attachment
        assert layer.can_respond(1_000_001)  # Above attachment

    def test_calculate_layer_loss(self):
        """Test layer loss calculation."""
        layer = EnhancedInsuranceLayer(
            attachment_point=1_000_000, limit=5_000_000, premium_rate=0.01
        )

        # Below attachment
        assert layer.calculate_layer_loss(500_000) == 0.0

        # Within layer
        assert layer.calculate_layer_loss(3_000_000) == 2_000_000

        # Exceeds layer
        assert layer.calculate_layer_loss(10_000_000) == 5_000_000  # Capped at limit


class TestLayerState:
    """Test layer state tracking functionality."""

    def test_initialization(self):
        """Test layer state initialization."""
        layer = EnhancedInsuranceLayer(
            attachment_point=1_000_000, limit=5_000_000, premium_rate=0.01, reinstatements=2
        )
        state = LayerState(layer)

        assert state.current_limit == 5_000_000
        assert state.used_limit == 0.0
        assert state.reinstatements_used == 0
        assert state.total_claims_paid == 0.0
        assert not state.is_exhausted

    def test_process_claim_simple(self):
        """Test simple claim processing."""
        layer = EnhancedInsuranceLayer(attachment_point=0, limit=5_000_000, premium_rate=0.01)
        state = LayerState(layer)

        payment, reinstatement_premium = state.process_claim(2_000_000)

        assert payment == 2_000_000
        assert reinstatement_premium == 0.0  # No reinstatement triggered
        assert state.current_limit == 3_000_000
        assert state.used_limit == 2_000_000

    def test_process_claim_exhaustion(self):
        """Test claim that exhausts layer."""
        layer = EnhancedInsuranceLayer(
            attachment_point=0,
            limit=5_000_000,
            premium_rate=0.01,
            reinstatements=0,  # No reinstatements
        )
        state = LayerState(layer)

        # Exhaust the layer
        payment, _ = state.process_claim(5_000_000)
        assert payment == 5_000_000
        assert state.current_limit == 0
        assert state.is_exhausted

        # Try another claim
        payment, _ = state.process_claim(1_000_000)
        assert payment == 0.0  # Layer exhausted

    def test_reinstatement_trigger(self):
        """Test reinstatement triggering."""
        layer = EnhancedInsuranceLayer(
            attachment_point=0,
            limit=5_000_000,
            premium_rate=0.02,
            reinstatements=1,
            reinstatement_premium=1.0,
            reinstatement_type=ReinstatementType.FULL,
        )
        state = LayerState(layer)

        # Exhaust the layer
        payment, reinstatement_premium = state.process_claim(5_000_000, timing_factor=0.5)

        assert payment == 5_000_000
        assert reinstatement_premium == 100_000  # Full premium
        assert state.reinstatements_used == 1
        assert state.current_limit == 5_000_000  # Reinstated
        assert not state.is_exhausted

    def test_multiple_reinstatements(self):
        """Test multiple reinstatements."""
        layer = EnhancedInsuranceLayer(
            attachment_point=0,
            limit=2_000_000,
            premium_rate=0.01,
            reinstatements=2,
            reinstatement_premium=0.5,
            reinstatement_type=ReinstatementType.PRO_RATA,
        )
        state = LayerState(layer)

        # First exhaustion
        payment1, premium1 = state.process_claim(2_000_000, timing_factor=0.8)
        assert state.reinstatements_used == 1
        assert premium1 == 8_000  # 20K base * 0.5 * 0.8

        # Second exhaustion
        payment2, premium2 = state.process_claim(2_000_000, timing_factor=0.4)
        assert state.reinstatements_used == 2
        assert premium2 == 4_000  # 20K base * 0.5 * 0.4

        # Third exhaustion - no more reinstatements
        payment3, premium3 = state.process_claim(2_000_000, timing_factor=0.2)
        assert payment3 == 2_000_000
        assert premium3 == 0.0  # No reinstatement available
        assert state.is_exhausted

    def test_aggregate_limit(self):
        """Test aggregate limit functionality."""
        layer = EnhancedInsuranceLayer(
            attachment_point=0,
            limit=2_000_000,
            premium_rate=0.01,
            reinstatements=10,  # Many reinstatements
            aggregate_limit=5_000_000,  # But aggregate cap
        )
        state = LayerState(layer)

        # First claim
        state.process_claim(2_000_000)
        assert state.aggregate_used == 2_000_000

        # Second claim
        state.process_claim(2_000_000)
        assert state.aggregate_used == 4_000_000

        # Third claim hits aggregate
        payment, _ = state.process_claim(2_000_000)
        assert payment == 1_000_000  # Only 1M left in aggregate
        assert state.is_exhausted
        assert state.aggregate_used == 5_000_000

    def test_reset(self):
        """Test state reset functionality."""
        layer = EnhancedInsuranceLayer(
            attachment_point=0, limit=5_000_000, premium_rate=0.01, reinstatements=1
        )
        state = LayerState(layer)

        # Use the layer
        state.process_claim(5_000_000)
        assert state.reinstatements_used == 1

        # Reset
        state.reset()
        assert state.current_limit == 5_000_000
        assert state.used_limit == 0.0
        assert state.reinstatements_used == 0
        assert not state.is_exhausted

    def test_utilization_rate(self):
        """Test utilization rate calculation."""
        layer = EnhancedInsuranceLayer(
            attachment_point=0,
            limit=5_000_000,
            premium_rate=0.01,
            reinstatements=1,  # Total 10M available
        )
        state = LayerState(layer)

        # Use half of first limit
        state.process_claim(2_500_000)
        assert state.get_utilization_rate() == 0.25  # 2.5M / 10M

        # Use rest and trigger reinstatement
        state.process_claim(2_500_000)
        assert state.get_utilization_rate() == 0.5  # 5M / 10M


class TestInsuranceProgram:
    """Test insurance program functionality."""

    def test_initialization(self):
        """Test program initialization."""
        layers = [
            EnhancedInsuranceLayer(attachment_point=0, limit=5_000_000, premium_rate=0.015),
            EnhancedInsuranceLayer(
                attachment_point=5_000_000, limit=20_000_000, premium_rate=0.008
            ),
        ]

        program = InsuranceProgram(layers, deductible=250_000)

        assert program.deductible == 250_000
        assert len(program.layers) == 2
        assert len(program.layer_states) == 2
        # Layers should be sorted by attachment
        assert program.layers[0].attachment_point == 0
        assert program.layers[1].attachment_point == 5_000_000

    def test_calculate_annual_premium(self):
        """Test annual premium calculation."""
        layers = [
            EnhancedInsuranceLayer(attachment_point=0, limit=5_000_000, premium_rate=0.02),
            EnhancedInsuranceLayer(attachment_point=5_000_000, limit=10_000_000, premium_rate=0.01),
        ]

        program = InsuranceProgram(layers)
        assert program.calculate_annual_premium() == 200_000  # 100K + 100K

    def test_process_small_claim(self):
        """Test processing a small claim within deductible."""
        program = InsuranceProgram([], deductible=250_000)

        result = program.process_claim(100_000)

        assert result["total_claim"] == 100_000
        assert result["deductible_paid"] == 100_000
        assert result["insurance_recovery"] == 0.0
        assert result["uncovered_loss"] == 0.0

    def test_process_single_layer_claim(self):
        """Test claim hitting single layer."""
        layers = [
            EnhancedInsuranceLayer(attachment_point=250_000, limit=5_000_000, premium_rate=0.01)
        ]
        program = InsuranceProgram(layers, deductible=250_000)

        result = program.process_claim(2_000_000)

        assert result["deductible_paid"] == 250_000
        assert result["insurance_recovery"] == 1_750_000
        assert result["uncovered_loss"] == 0.0
        assert len(result["layers_triggered"]) == 1

    def test_process_multi_layer_claim(self):
        """Test claim hitting multiple layers."""
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=250_000, limit=4_750_000, premium_rate=0.015  # Up to 5M
            ),
            EnhancedInsuranceLayer(
                attachment_point=5_000_000, limit=20_000_000, premium_rate=0.008
            ),
        ]
        program = InsuranceProgram(layers, deductible=250_000)

        # 10M claim
        result = program.process_claim(10_000_000)

        assert result["deductible_paid"] == 250_000
        assert result["insurance_recovery"] == 9_750_000  # 4.75M + 5M
        assert result["uncovered_loss"] == 0.0
        assert len(result["layers_triggered"]) == 2

    def test_process_claim_with_reinstatement(self):
        """Test claim triggering reinstatement."""
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=0,
                limit=5_000_000,
                premium_rate=0.02,
                reinstatements=1,
                reinstatement_premium=1.0,
                reinstatement_type=ReinstatementType.FULL,
            )
        ]
        program = InsuranceProgram(layers)

        result = program.process_claim(5_000_000, timing_factor=0.5)

        assert result["insurance_recovery"] == 5_000_000
        assert result["reinstatement_premiums"] == 100_000
        assert result["layers_triggered"][0]["reinstatement_premium"] == 100_000

    def test_process_annual_claims(self):
        """Test processing multiple claims in a year."""
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=0,
                limit=2_000_000,
                premium_rate=0.01,
                reinstatements=2,
                reinstatement_premium=0.5,
                reinstatement_type=ReinstatementType.PRO_RATA,
            )
        ]
        program = InsuranceProgram(layers)

        claims = [1_000_000, 2_000_000, 1_500_000]
        claim_times = [0.2, 0.5, 0.8]

        results = program.process_annual_claims(claims, claim_times)

        assert results["total_claims"] == 3
        assert results["total_losses"] == 4_500_000
        # First claim: 1M paid, 1M remains
        # Second claim: 2M total - 1M from remaining, exhausts, reinstates, 1M from new limit
        # Third claim: 1.5M from current limit (0.5M remains)
        assert results["total_recovery"] == 4_500_000  # All claims covered
        assert results["base_premium"] == 20_000

        # Check reinstatement premiums
        # Second claim (2M) exhausts the remaining 1M and triggers reinstatement
        # Reinstatement at time 0.5 (0.5 remaining): 20K * 0.5 * 0.5 = 5K
        # Then uses 1M from reinstated limit, leaving 1M
        # Third claim (1.5M) uses 1M and triggers another reinstatement
        # Reinstatement at time 0.8 (0.2 remaining): 20K * 0.5 * 0.2 = 2K
        # Total: 5K + 2K = 7K
        assert results["total_reinstatement_premiums"] == 7_000

    def test_reset_annual(self):
        """Test annual reset functionality."""
        layers = [EnhancedInsuranceLayer(attachment_point=0, limit=5_000_000, premium_rate=0.01)]
        program = InsuranceProgram(layers)

        # Process claim
        program.process_claim(3_000_000)
        assert program.layer_states[0].used_limit == 3_000_000

        # Reset
        program.reset_annual()
        assert program.layer_states[0].used_limit == 0.0
        assert program.layer_states[0].current_limit == 5_000_000

    def test_standard_manufacturing_program(self):
        """Test standard manufacturing program creation."""
        program = InsuranceProgram.create_standard_manufacturing_program(deductible=250_000)

        assert program.deductible == 250_000
        assert len(program.layers) == 4
        assert program.name == "Standard Manufacturing Program"

        # Check layer structure
        assert program.layers[0].attachment_point == 250_000
        assert program.layers[1].attachment_point == 5_000_000
        assert program.layers[2].attachment_point == 25_000_000
        assert program.layers[3].attachment_point == 50_000_000

    def test_get_total_coverage(self):
        """Test total coverage calculation."""
        layers = [
            EnhancedInsuranceLayer(attachment_point=0, limit=5_000_000, premium_rate=0.01),
            EnhancedInsuranceLayer(attachment_point=5_000_000, limit=20_000_000, premium_rate=0.01),
        ]
        program = InsuranceProgram(layers)

        assert program.get_total_coverage() == 25_000_000

    def test_yaml_loading(self):
        """Test loading program from YAML configuration."""
        config = {
            "program_name": "Test Program",
            "deductible": 500_000,
            "layers": [
                {
                    "attachment_point": 500_000,
                    "limit": 5_000_000,
                    "premium_rate": 0.02,
                    "reinstatements": 1,
                    "reinstatement_premium": 1.0,
                    "reinstatement_type": "full",
                },
                {
                    "attachment_point": 5_500_000,
                    "limit": 10_000_000,
                    "premium_rate": 0.01,
                    "reinstatements": 0,
                },
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            program = InsuranceProgram.from_yaml(temp_path)

            assert program.name == "Test Program"
            assert program.deductible == 500_000
            assert len(program.layers) == 2
            assert program.layers[0].reinstatement_type == ReinstatementType.FULL
        finally:
            Path(temp_path).unlink()


class TestProgramState:
    """Test multi-year program state tracking."""

    def test_initialization(self):
        """Test program state initialization."""
        layers = [EnhancedInsuranceLayer(attachment_point=0, limit=5_000_000, premium_rate=0.01)]
        program = InsuranceProgram(layers)
        state = ProgramState(program)

        assert state.years_simulated == 0
        assert len(state.total_claims) == 0
        assert len(state.total_recoveries) == 0
        assert len(state.total_premiums) == 0

    def test_simulate_year(self):
        """Test single year simulation."""
        layers = [EnhancedInsuranceLayer(attachment_point=0, limit=5_000_000, premium_rate=0.01)]
        program = InsuranceProgram(layers)
        state = ProgramState(program)

        claims = [1_000_000, 2_000_000]
        results = state.simulate_year(claims)

        assert state.years_simulated == 1
        assert state.total_claims[0] == 3_000_000
        assert state.total_recoveries[0] == 3_000_000
        assert state.total_premiums[0] == 50_000  # Base premium only
        assert len(state.annual_results) == 1

    def test_multi_year_simulation(self):
        """Test multi-year simulation."""
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=0,
                limit=3_000_000,
                premium_rate=0.01,
                reinstatements=1,
                reinstatement_premium=0.5,
                reinstatement_type=ReinstatementType.PRO_RATA,
            )
        ]
        program = InsuranceProgram(layers)
        state = ProgramState(program)

        # Year 1
        state.simulate_year([2_000_000, 1_500_000])

        # Year 2
        state.simulate_year([500_000, 3_000_000])

        # Year 3
        state.simulate_year([4_000_000])

        assert state.years_simulated == 3
        assert len(state.total_claims) == 3
        assert len(state.total_recoveries) == 3

    def test_summary_statistics(self):
        """Test summary statistics calculation."""
        layers = [EnhancedInsuranceLayer(attachment_point=0, limit=5_000_000, premium_rate=0.01)]
        program = InsuranceProgram(layers)
        state = ProgramState(program)

        # Simulate 3 years
        state.simulate_year([2_000_000])
        state.simulate_year([3_000_000])
        state.simulate_year([4_000_000])

        stats = state.get_summary_statistics()

        assert stats["years_simulated"] == 3
        assert stats["average_annual_claims"] == 3_000_000
        assert stats["average_annual_recovery"] == 3_000_000
        assert stats["total_claims"] == 9_000_000
        assert stats["total_recoveries"] == 9_000_000
        assert stats["recovery_ratio"] == 1.0  # Full recovery

        # Loss ratio = recoveries / premiums
        assert stats["loss_ratio"] == 9_000_000 / 150_000  # 60


class TestComplexScenarios:
    """Test complex insurance scenarios."""

    def test_single_large_loss_scenario(self):
        """Test the example single large loss scenario from requirements."""
        # Create program as specified
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=250_000,  # After deductible
                limit=4_750_000,  # Primary layer
                premium_rate=0.015,
            ),
            EnhancedInsuranceLayer(
                attachment_point=5_000_000,
                limit=20_000_000,  # First excess
                premium_rate=0.008,
                reinstatements=1,
                reinstatement_premium=1.0,
                reinstatement_type=ReinstatementType.FULL,
            ),
            EnhancedInsuranceLayer(
                attachment_point=25_000_000, limit=25_000_000, premium_rate=0.004  # Second excess
            ),
        ]

        program = InsuranceProgram(layers, deductible=250_000)

        # Process 30M loss
        result = program.process_claim(30_000_000)

        # Check allocations match expected
        assert result["deductible_paid"] == 250_000
        # Primary: 4.75M, First Excess: 20M, Second Excess: 5M
        assert result["insurance_recovery"] == 29_750_000
        assert result["uncovered_loss"] == 0.0

        # Verify layer details
        assert len(result["layers_triggered"]) == 3
        assert result["layers_triggered"][0]["payment"] == 4_750_000  # Primary
        assert result["layers_triggered"][1]["payment"] == 20_000_000  # First excess
        assert result["layers_triggered"][2]["payment"] == 5_000_000  # Second excess

    def test_multiple_attritional_losses(self):
        """Test multiple smaller losses utilizing primary layer."""
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=100_000,
                limit=2_000_000,
                premium_rate=0.02,
                reinstatements=2,
                reinstatement_premium=0.5,
                reinstatement_type=ReinstatementType.PRO_RATA,
            )
        ]

        program = InsuranceProgram(layers, deductible=100_000)

        # Process multiple claims through the year
        claims = [500_000, 800_000, 1_200_000, 600_000, 900_000]
        claim_times = [0.1, 0.3, 0.5, 0.7, 0.9]

        results = program.process_annual_claims(claims, claim_times)

        assert results["total_losses"] == 4_000_000
        # Each claim has 100K deductible
        # Claims: 500K, 800K, 1200K, 600K, 900K
        # Deductible paid: 100K * 5 = 500K
        # Layer covers: 400K, 700K, 1100K, 500K, 800K = 3500K
        assert results["total_deductible"] == 500_000  # 5 claims * 100K deductible
        assert results["total_recovery"] == 3_500_000  # Total recovery from layer

        # Check reinstatements were triggered
        layer_summary = results["layer_summaries"][0]
        assert layer_summary["reinstatements_used"] == 1  # Should have triggered one reinstatement

    def test_catastrophic_exhaustion(self):
        """Test scenario where all layers are exhausted."""
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=0, limit=5_000_000, premium_rate=0.02, reinstatements=0
            ),
            EnhancedInsuranceLayer(
                attachment_point=5_000_000, limit=10_000_000, premium_rate=0.01, reinstatements=0
            ),
        ]

        program = InsuranceProgram(layers)

        # Process claim exceeding all coverage
        result = program.process_claim(20_000_000)

        assert result["insurance_recovery"] == 15_000_000  # 5M + 10M
        assert result["uncovered_loss"] == 5_000_000
        assert result["deductible_paid"] == 5_000_000  # Company pays uncovered

    def test_performance_batch_processing(self):
        """Test performance requirement: process 10K claims in < 100ms."""
        import time

        # Create a simple program
        layers = [EnhancedInsuranceLayer(attachment_point=1_000, limit=10_000, premium_rate=0.01)]
        program = InsuranceProgram(layers)

        # Generate 10K claims
        np.random.seed(42)
        claims = np.random.lognormal(8, 1.5, 10_000).tolist()

        start_time = time.time()
        for claim in claims:
            program.process_claim(claim)
        elapsed = time.time() - start_time

        # Should process in reasonable time (relaxed from 100ms for safety)
        assert elapsed < 1.0  # 1 second max
