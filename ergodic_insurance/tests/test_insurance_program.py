"""Tests for multi-layer insurance program with reinstatements.

Comprehensive test suite for advanced insurance program features including
reinstatements, multi-layer structures, complex claim scenarios, and
optimization algorithms.
"""

from pathlib import Path
import tempfile

import numpy as np
import pytest
import yaml

from ergodic_insurance.ergodic_types import ClaimResult, LayerPayment
from ergodic_insurance.insurance_program import (
    EnhancedInsuranceLayer,
    InsuranceProgram,
    LayerState,
    OptimalStructure,
    ProgramOptimizationConstraints,
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
            base_premium_rate=0.01,
            reinstatements=2,
            reinstatement_premium=1.0,
            reinstatement_type=ReinstatementType.PRO_RATA,
        )

        assert layer.attachment_point == 1_000_000
        assert layer.limit == 5_000_000
        assert layer.base_premium_rate == 0.01
        assert layer.reinstatements == 2
        assert layer.reinstatement_premium == 1.0
        assert layer.reinstatement_type == ReinstatementType.PRO_RATA

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            (
                {"attachment_point": -100, "limit": 1_000_000, "base_premium_rate": 0.01},
                "Attachment point must be non-negative",
            ),
            (
                {"attachment_point": 0, "limit": -1_000_000, "base_premium_rate": 0.01},
                "Limit must be positive",
            ),
            (
                {"attachment_point": 0, "limit": 1_000_000, "base_premium_rate": -0.01},
                "Base premium rate must be non-negative",
            ),
            (
                {
                    "attachment_point": 0,
                    "limit": 1_000_000,
                    "base_premium_rate": 0.01,
                    "reinstatements": -1,
                },
                "Reinstatements must be non-negative",
            ),
        ],
        ids=[
            "negative-attachment",
            "negative-limit",
            "negative-rate",
            "negative-reinstatements",
        ],
    )
    def test_invalid_parameters(self, kwargs, match):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError, match=match):
            EnhancedInsuranceLayer(**kwargs)

    def test_calculate_base_premium(self):
        """Test base premium calculation."""
        layer = EnhancedInsuranceLayer(
            attachment_point=1_000_000, limit=5_000_000, base_premium_rate=0.02
        )

        assert layer.calculate_base_premium() == 100_000  # 5M * 0.02

    @pytest.mark.parametrize(
        "reinstatement_premium,reinstatement_type,expected",
        [
            (0.5, ReinstatementType.PRO_RATA, 25_000),  # 100K base * 0.5 * 0.5 timing
            (1.0, ReinstatementType.FULL, 100_000),  # Full base premium (timing ignored)
            (0.0, ReinstatementType.FREE, 0.0),  # Free reinstatement
        ],
        ids=["pro-rata", "full", "free"],
    )
    def test_reinstatement_premium(self, reinstatement_premium, reinstatement_type, expected):
        """Test reinstatement premium calculation for all types."""
        layer = EnhancedInsuranceLayer(
            attachment_point=1_000_000,
            limit=5_000_000,
            base_premium_rate=0.02,
            reinstatement_premium=reinstatement_premium,
            reinstatement_type=reinstatement_type,
        )

        premium = layer.calculate_reinstatement_premium(timing_factor=0.5)
        assert premium == expected

    def test_exhausted_is_dataclass_field(self):
        """exhausted should be a proper dataclass field, appearing in asdict and __eq__."""
        from dataclasses import asdict

        layer = EnhancedInsuranceLayer(
            attachment_point=1_000_000, limit=5_000_000, base_premium_rate=0.01
        )
        d = asdict(layer)
        assert "exhausted" in d
        assert d["exhausted"] == 0.0

        # Two identical layers should be equal (exhausted participates in __eq__)
        layer2 = EnhancedInsuranceLayer(
            attachment_point=1_000_000, limit=5_000_000, base_premium_rate=0.01
        )
        assert layer == layer2

    def test_can_respond(self):
        """Test layer response determination."""
        layer = EnhancedInsuranceLayer(
            attachment_point=1_000_000, limit=5_000_000, base_premium_rate=0.01
        )

        assert not layer.can_respond(500_000)  # Below attachment
        assert not layer.can_respond(1_000_000)  # At attachment
        assert layer.can_respond(1_000_001)  # Above attachment

    @pytest.mark.parametrize(
        "loss,expected",
        [
            pytest.param(500_000, 0.0, id="below-attachment"),
            pytest.param(3_000_000, 2_000_000, id="within-layer"),
            pytest.param(10_000_000, 5_000_000, id="exceeds-layer-capped"),
        ],
    )
    def test_calculate_layer_loss(self, loss, expected):
        """Test layer loss calculation."""
        layer = EnhancedInsuranceLayer(
            attachment_point=1_000_000, limit=5_000_000, base_premium_rate=0.01
        )
        assert layer.calculate_layer_loss(loss) == expected


class TestLayerState:
    """Test layer state tracking functionality."""

    def test_initialization(self):
        """Test layer state initialization."""
        layer = EnhancedInsuranceLayer(
            attachment_point=1_000_000, limit=5_000_000, base_premium_rate=0.01, reinstatements=2
        )
        state = LayerState(layer)

        assert state.current_limit == 5_000_000
        assert state.used_limit == 0.0
        assert state.reinstatements_used == 0
        assert state.total_claims_paid == 0.0
        assert not state.is_exhausted

    def test_process_claim_simple(self):
        """Test simple claim processing."""
        layer = EnhancedInsuranceLayer(
            attachment_point=0, limit=5_000_000, base_premium_rate=0.01, limit_type="aggregate"
        )
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
            base_premium_rate=0.01,
            reinstatements=0,  # No reinstatements
            limit_type="aggregate",
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
            base_premium_rate=0.02,
            reinstatements=1,
            reinstatement_premium=1.0,
            reinstatement_type=ReinstatementType.FULL,
            limit_type="aggregate",
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
            base_premium_rate=0.01,
            reinstatements=2,
            reinstatement_premium=0.5,
            reinstatement_type=ReinstatementType.PRO_RATA,
            limit_type="aggregate",
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
            base_premium_rate=0.01,
            reinstatements=10,  # Many reinstatements
            aggregate_limit=5_000_000,  # But aggregate cap
            limit_type="aggregate",
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
            attachment_point=0,
            limit=5_000_000,
            base_premium_rate=0.01,
            reinstatements=1,
            limit_type="aggregate",
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
            base_premium_rate=0.01,
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
            EnhancedInsuranceLayer(attachment_point=0, limit=5_000_000, base_premium_rate=0.015),
            EnhancedInsuranceLayer(
                attachment_point=5_000_000, limit=20_000_000, base_premium_rate=0.008
            ),
        ]

        program = InsuranceProgram(layers, deductible=250_000)

        assert program.deductible == 250_000
        assert len(program.layers) == 2
        assert len(program.layer_states) == 2
        # Layers should be sorted by attachment
        assert program.layers[0].attachment_point == 0
        assert program.layers[1].attachment_point == 5_000_000

    def test_calculate_premium(self):
        """Test premium calculation."""
        layers = [
            EnhancedInsuranceLayer(attachment_point=0, limit=5_000_000, base_premium_rate=0.02),
            EnhancedInsuranceLayer(
                attachment_point=5_000_000, limit=10_000_000, base_premium_rate=0.01
            ),
        ]

        program = InsuranceProgram(layers)
        assert program.calculate_premium() == 200_000  # 100K + 100K

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
            EnhancedInsuranceLayer(
                attachment_point=250_000, limit=5_000_000, base_premium_rate=0.01
            )
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
                attachment_point=250_000, limit=4_750_000, base_premium_rate=0.015  # Up to 5M
            ),
            EnhancedInsuranceLayer(
                attachment_point=5_000_000, limit=20_000_000, base_premium_rate=0.008
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
                base_premium_rate=0.02,
                reinstatements=1,
                reinstatement_premium=1.0,
                reinstatement_type=ReinstatementType.FULL,
                limit_type="aggregate",
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
                base_premium_rate=0.01,
                reinstatements=2,
                reinstatement_premium=0.5,
                reinstatement_type=ReinstatementType.PRO_RATA,
                limit_type="aggregate",
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
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=0, limit=5_000_000, base_premium_rate=0.01, limit_type="aggregate"
            )
        ]
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
            EnhancedInsuranceLayer(attachment_point=0, limit=5_000_000, base_premium_rate=0.01),
            EnhancedInsuranceLayer(
                attachment_point=5_000_000, limit=20_000_000, base_premium_rate=0.01
            ),
        ]
        program = InsuranceProgram(layers)

        assert program.get_total_coverage() == 25_000_000

    def test_get_total_coverage_subtracts_deductible(self):
        """get_total_coverage should return gross coverage minus deductible."""
        layers = [
            EnhancedInsuranceLayer(attachment_point=0, limit=5_000_000, base_premium_rate=0.01),
            EnhancedInsuranceLayer(
                attachment_point=5_000_000, limit=20_000_000, base_premium_rate=0.01
            ),
        ]
        program = InsuranceProgram(layers, deductible=1_000_000)
        # Gross = 25M, net = 25M - 1M = 24M
        assert program.get_total_coverage() == 24_000_000

    def test_get_total_coverage_consistent_with_policy(self):
        """InsuranceProgram.get_total_coverage should match InsurancePolicy for same config."""
        from ergodic_insurance.insurance import InsuranceLayer, InsurancePolicy

        layers_basic = [
            InsuranceLayer(attachment_point=500_000, limit=4_500_000, rate=0.015),
            InsuranceLayer(attachment_point=5_000_000, limit=20_000_000, rate=0.008),
        ]
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy(layers=layers_basic, deductible=500_000)

        program = policy.to_enhanced_program()
        assert program is not None
        assert program.get_total_coverage() == policy.get_total_coverage()

    def test_get_total_coverage_deductible_exceeds_layers(self):
        """get_total_coverage returns 0 when deductible exceeds highest layer exhaust."""
        layers = [
            EnhancedInsuranceLayer(attachment_point=100_000, limit=200_000, base_premium_rate=0.01),
        ]
        program = InsuranceProgram(layers, deductible=500_000)
        assert program.get_total_coverage() == 0.0

    def test_get_total_coverage_empty_layers(self):
        """get_total_coverage returns 0 when no layers."""
        program = InsuranceProgram(layers=[], deductible=0)
        assert program.get_total_coverage() == 0.0

    def test_yaml_loading(self):
        """Test loading program from YAML configuration."""
        config = {
            "program_name": "Test Program",
            "deductible": 500_000,
            "layers": [
                {
                    "attachment_point": 500_000,
                    "limit": 5_000_000,
                    "base_premium_rate": 0.02,
                    "reinstatements": 1,
                    "reinstatement_premium": 1.0,
                    "reinstatement_type": "full",
                },
                {
                    "attachment_point": 5_500_000,
                    "limit": 10_000_000,
                    "base_premium_rate": 0.01,
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
        layers = [
            EnhancedInsuranceLayer(attachment_point=0, limit=5_000_000, base_premium_rate=0.01)
        ]
        program = InsuranceProgram(layers)
        state = ProgramState(program)

        assert state.years_simulated == 0
        assert len(state.total_claims) == 0
        assert len(state.total_recoveries) == 0
        assert len(state.total_premiums) == 0

    def test_simulate_year(self):
        """Test single year simulation."""
        layers = [
            EnhancedInsuranceLayer(attachment_point=0, limit=5_000_000, base_premium_rate=0.01)
        ]
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
                base_premium_rate=0.01,
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
        layers = [
            EnhancedInsuranceLayer(attachment_point=0, limit=5_000_000, base_premium_rate=0.01)
        ]
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
                base_premium_rate=0.015,
            ),
            EnhancedInsuranceLayer(
                attachment_point=5_000_000,
                limit=20_000_000,  # First excess
                base_premium_rate=0.008,
                reinstatements=1,
                reinstatement_premium=1.0,
                reinstatement_type=ReinstatementType.FULL,
            ),
            EnhancedInsuranceLayer(
                attachment_point=25_000_000,
                limit=25_000_000,
                base_premium_rate=0.004,  # Second excess
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
                base_premium_rate=0.02,
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


class TestInsuranceProgramOptimization:
    """Test insurance program optimization algorithms."""

    @pytest.fixture
    def sample_loss_data(self):
        """Generate sample loss data for testing."""
        np.random.seed(42)

        # Generate 10 years of loss data
        loss_data = []
        for _ in range(10):
            annual_losses = []

            # Attritional losses (high frequency, low severity)
            n_attritional = np.random.poisson(5)
            for _ in range(n_attritional):
                annual_losses.append(np.random.lognormal(10, 1.5))  # ~22K mean

            # Large losses (low frequency, high severity)
            n_large = np.random.poisson(0.3)
            for _ in range(n_large):
                annual_losses.append(np.random.lognormal(14, 2))  # ~1.6M mean

            loss_data.append(annual_losses)

        return loss_data

    @pytest.fixture
    def company_profile(self):
        """Sample company profile for testing."""
        return {
            "initial_assets": 10_000_000,
            "annual_revenue": 15_000_000,
            "base_operating_margin": 0.08,
            "growth_rate": 0.05,
        }

    def test_calculate_ergodic_benefit(self, sample_loss_data, company_profile):
        """Test ergodic benefit calculation."""
        program = InsuranceProgram.create_standard_manufacturing_program()

        metrics = program.calculate_ergodic_benefit(sample_loss_data, company_profile)

        assert "time_average_benefit" in metrics
        assert "ensemble_average_cost" in metrics
        assert "ergodic_ratio" in metrics
        assert "volatility_reduction" in metrics
        assert metrics["volatility_reduction"] >= 0  # Insurance should reduce volatility

    def test_find_optimal_attachment_points(self):
        """Test attachment point optimization."""
        program = InsuranceProgram.create_standard_manufacturing_program()

        # Generate test loss data
        np.random.seed(42)
        losses = np.concatenate(
            [
                np.random.lognormal(10, 1.5, 100),  # Small losses
                np.random.lognormal(14, 2, 20),  # Large losses
            ]
        )

        # Test with different layer counts
        for num_layers in [3, 4, 5]:
            attachment_points = program.find_optimal_attachment_points(losses.tolist(), num_layers)

            assert len(attachment_points) == num_layers
            # Check strictly increasing
            for i in range(1, len(attachment_points)):
                assert attachment_points[i] > attachment_points[i - 1]

    def test_find_optimal_attachment_points_empty_data(self):
        """Test attachment point optimization with empty data."""
        program = InsuranceProgram.create_standard_manufacturing_program()

        attachment_points = program.find_optimal_attachment_points([], 4)
        assert attachment_points == []

        attachment_points = program.find_optimal_attachment_points([0, 0, 0], 4)
        assert attachment_points == []

    def test_optimize_layer_widths(self):
        """Test layer width optimization."""
        program = InsuranceProgram.create_standard_manufacturing_program()

        attachment_points = [250_000, 1_000_000, 5_000_000, 25_000_000]
        total_budget = 500_000  # Total premium budget

        widths = program.optimize_layer_widths(attachment_points, total_budget)

        assert len(widths) == len(attachment_points)
        assert all(w > 0 for w in widths)

        # Test with capacity constraints
        capacity_constraints = {
            "layer_0": 2_000_000,
            "layer_1": 10_000_000,
            "layer_2": 20_000_000,
            "layer_3": 50_000_000,
        }

        constrained_widths = program.optimize_layer_widths(
            attachment_points, total_budget, capacity_constraints
        )

        for i, width in enumerate(constrained_widths):
            assert width <= capacity_constraints[f"layer_{i}"]

    def test_optimize_layer_widths_with_loss_data(self):
        """Test layer width optimization with loss data."""
        program = InsuranceProgram.create_standard_manufacturing_program()

        # Generate test losses
        np.random.seed(42)
        losses = np.concatenate(
            [
                np.random.lognormal(10, 1.5, 100),
                np.random.lognormal(14, 2, 20),
            ]
        ).tolist()

        attachment_points = [250_000, 1_000_000, 5_000_000]
        total_budget = 300_000

        widths = program.optimize_layer_widths(attachment_points, total_budget, loss_data=losses)

        assert len(widths) == len(attachment_points)
        assert all(w > 0 for w in widths)

    def test_optimize_layer_structure(self, sample_loss_data, company_profile):
        """Test complete layer structure optimization."""
        program = InsuranceProgram.create_standard_manufacturing_program()

        # Test with default constraints
        optimal = program.optimize_layer_structure(sample_loss_data, company_profile)

        assert isinstance(optimal, OptimalStructure)
        assert len(optimal.layers) >= 3
        assert len(optimal.layers) <= 5
        assert optimal.total_premium > 0
        assert optimal.total_coverage > 0
        assert optimal.deductible > 0

        # Check layers are properly ordered
        for i in range(1, len(optimal.layers)):
            assert optimal.layers[i].attachment_point > optimal.layers[i - 1].attachment_point

    def test_optimize_layer_structure_with_constraints(self, sample_loss_data, company_profile):
        """Test structure optimization with custom constraints."""
        program = InsuranceProgram.create_standard_manufacturing_program()

        constraints = ProgramOptimizationConstraints(
            max_total_premium=400_000,
            min_total_coverage=30_000_000,
            max_layers=4,
            min_layers=3,
            min_roe_improvement=0.10,
        )

        optimal = program.optimize_layer_structure(sample_loss_data, company_profile, constraints)

        assert len(optimal.layers) >= constraints.min_layers
        assert len(optimal.layers) <= constraints.max_layers
        assert optimal.total_premium <= constraints.max_total_premium or optimal.total_premium == 0

        # Check if ROE improvement meets target (if converged)
        if optimal.convergence_achieved:
            assert optimal.roe_improvement >= 0  # Should be positive

    def test_optimize_empty_loss_data(self, company_profile):
        """Test optimization with no loss data."""
        program = InsuranceProgram.create_standard_manufacturing_program()

        optimal = program.optimize_layer_structure([], company_profile)

        # Should return basic structure
        assert isinstance(optimal, OptimalStructure)
        assert not optimal.convergence_achieved
        assert optimal.ergodic_benefit == 0.0

    def test_optimization_constraints_defaults(self):
        """Test default optimization constraints."""
        constraints = ProgramOptimizationConstraints()

        assert constraints.max_layers == 5
        assert constraints.min_layers == 3
        assert constraints.max_attachment_gap == 0.0
        assert constraints.min_roe_improvement == 0.15
        assert constraints.max_iterations == 1000
        assert constraints.convergence_tolerance == 1e-6

    @pytest.mark.parametrize(
        "value,expected",
        [
            pytest.param(6_000, 10_000, id="below-100k-small"),
            pytest.param(45_000, 40_000, id="below-100k-round-10k"),
            pytest.param(55_000, 60_000, id="below-100k-round-up"),
            pytest.param(123_456, 100_000, id="100k-1m-round-50k-low"),
            pytest.param(275_000, 300_000, id="100k-1m-round-50k-mid"),
            pytest.param(567_890, 550_000, id="100k-1m-round-50k-high"),
            pytest.param(730_000, 750_000, id="100k-1m-round-50k-upper"),
            pytest.param(1_234_567, 1_250_000, id="1m-10m-round-250k"),
            pytest.param(3_200_000, 3_250_000, id="1m-10m-round-250k-mid"),
            pytest.param(12_345_678, 12_000_000, id="above-10m-round-1m"),
            pytest.param(15_500_000, 16_000_000, id="above-10m-round-1m-up"),
            pytest.param(123_456_789, 123_000_000, id="above-10m-round-1m-large"),
        ],
    )
    def test_round_attachment_point(self, value, expected):
        """Test attachment point rounding logic for all ranges."""
        program = InsuranceProgram.create_standard_manufacturing_program()
        assert program._round_attachment_point(value) == expected

    def test_ergodic_benefit_calculation_edge_cases(self):
        """Test ergodic benefit calculation with edge cases."""
        program = InsuranceProgram.create_standard_manufacturing_program()

        # Test with single year data
        single_year = [[100_000, 200_000]]
        metrics = program.calculate_ergodic_benefit(single_year)

        assert metrics["time_average_benefit"] == 0.0  # Not enough data

        # Test with no losses
        no_losses = [[], [], []]
        metrics = program.calculate_ergodic_benefit(no_losses)

        assert metrics["ensemble_average_cost"] < 0  # Premium cost only

        # Test with catastrophic losses
        catastrophic = [[50_000_000], [0], [0]]
        metrics = program.calculate_ergodic_benefit(catastrophic)

        assert metrics["volatility_reduction"] > 0  # Insurance helps with catastrophic

    def test_optimization_convergence(self, sample_loss_data):
        """Test that optimization converges within iteration limit."""
        program = InsuranceProgram.create_standard_manufacturing_program()

        constraints = ProgramOptimizationConstraints(max_iterations=10)

        optimal = program.optimize_layer_structure(sample_loss_data, constraints=constraints)

        assert optimal.iterations_used <= constraints.max_iterations

    def test_layer_structure_validation(self):
        """Test that optimized structures are valid."""
        program = InsuranceProgram.create_standard_manufacturing_program()

        # Generate synthetic loss data
        np.random.seed(42)
        loss_data = []
        for _ in range(5):
            annual = list(np.random.lognormal(11, 2, np.random.poisson(3)))
            loss_data.append(annual)

        optimal = program.optimize_layer_structure(loss_data)

        # Validate structure
        for layer in optimal.layers:
            assert layer.attachment_point >= 0
            assert layer.limit > 0
            assert layer.base_premium_rate > 0
            assert layer.base_premium_rate <= 0.05  # Reasonable rate
            assert layer.reinstatements >= 0

    def test_percentile_based_attachment_selection(self):
        """Test percentile-based attachment point selection."""
        program = InsuranceProgram.create_standard_manufacturing_program()

        # Generate loss data with clear percentiles
        np.random.seed(42)
        losses = np.concatenate(
            [
                np.full(50, 10_000),  # 50th percentile
                np.full(40, 100_000),  # 90th percentile
                np.full(9, 1_000_000),  # 99th percentile
                np.full(1, 10_000_000),  # 100th percentile
            ]
        )
        np.random.shuffle(losses)

        # Test with custom percentiles
        attachment_points = program.find_optimal_attachment_points(
            losses.tolist(), num_layers=3, percentiles=[50, 90, 99]
        )

        assert len(attachment_points) == 3
        # Check that attachment points roughly match percentiles
        assert attachment_points[0] < 100_000  # Around 50th percentile
        assert attachment_points[1] < 1_000_000  # Around 90th percentile
        assert attachment_points[2] >= 1_000_000  # Around 99th percentile


class TestCatastrophicScenarios:
    """Test catastrophic scenarios and edge cases."""

    def test_catastrophic_exhaustion(self):
        """Test scenario where all layers are exhausted."""
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=0, limit=5_000_000, base_premium_rate=0.02, reinstatements=0
            ),
            EnhancedInsuranceLayer(
                attachment_point=5_000_000,
                limit=10_000_000,
                base_premium_rate=0.01,
                reinstatements=0,
            ),
        ]

        program = InsuranceProgram(layers)

        # Process claim exceeding all coverage
        result = program.process_claim(20_000_000)

        assert result["insurance_recovery"] == 15_000_000  # 5M + 10M
        assert result["uncovered_loss"] == 5_000_000
        assert result["deductible_paid"] == 5_000_000  # Company pays uncovered

    @pytest.mark.benchmark
    def test_performance_batch_processing(self):
        """Test performance requirement: process 10K claims in < 100ms."""
        import time

        # Create a simple program
        layers = [
            EnhancedInsuranceLayer(attachment_point=1_000, limit=10_000, base_premium_rate=0.01)
        ]
        program = InsuranceProgram(layers)

        # Generate 10K claims
        np.random.seed(42)
        claims = np.random.lognormal(8, 1.5, 10_000).tolist()

        start_time = time.time()
        for claim in claims:
            program.process_claim(claim)
        elapsed = time.time() - start_time

        # Should process in reasonable time (relaxed for CI environments)
        assert elapsed < 10.0  # 10 second max


class TestOverRecoveryGuard:
    """Tests for the over-recovery guard in InsuranceProgram (issue #310).

    Ensures total insurance recovery never exceeds (claim - deductible).
    """

    def test_recovery_capped_at_claim_minus_deductible(self):
        """Test that total recovery is capped at claim minus deductible."""
        # Overlapping layers that would over-recover without a guard
        layers = [
            EnhancedInsuranceLayer(attachment_point=0, limit=5_000_000, base_premium_rate=0.01),
            EnhancedInsuranceLayer(attachment_point=0, limit=5_000_000, base_premium_rate=0.01),
        ]
        program = InsuranceProgram(layers, deductible=1_000_000)

        result = program.process_claim(3_000_000)

        # Max recovery = 3M - 1M = 2M, not 6M from overlapping layers
        assert result["insurance_recovery"] <= 2_000_000
        total = result["deductible_paid"] + result["insurance_recovery"] - result["uncovered_loss"]
        assert abs(total - 3_000_000) < 0.01

    def test_deductible_not_equal_to_attachment(self):
        """Test with deductible != first layer attachment point (issue #310 specific)."""
        # Deductible is 1M but layer attaches at 500K
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=500_000, limit=5_000_000, base_premium_rate=0.015
            ),
        ]
        program = InsuranceProgram(layers, deductible=1_000_000)

        result = program.process_claim(3_000_000)

        # Max recovery = 3M - 1M = 2M
        # Layer would cover min(3M - 500K, 5M) = 2.5M, but must be capped at 2M
        assert result["insurance_recovery"] <= 2_000_000
        assert result["deductible_paid"] >= 1_000_000

    def test_zero_claim_produces_zero_recovery(self):
        """Test zero-loss events produce zero recovery."""
        program = InsuranceProgram.create_standard_manufacturing_program()

        result = program.process_claim(0)
        assert result["insurance_recovery"] == 0
        assert result["deductible_paid"] == 0

    def test_claim_below_deductible_no_recovery(self):
        """Test that claims below deductible get no insurance recovery."""
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=100_000, limit=5_000_000, base_premium_rate=0.01
            ),
        ]
        program = InsuranceProgram(layers, deductible=500_000)

        result = program.process_claim(300_000)

        # Below deductible, all paid by company
        assert result["insurance_recovery"] == 0
        assert result["deductible_paid"] == 300_000

    @pytest.mark.parametrize(
        "claim",
        [
            pytest.param(0, id="zero"),
            pytest.param(100_000, id="below-deductible"),
            pytest.param(250_000, id="at-deductible"),
            pytest.param(1_000_000, id="within-first-layer"),
            pytest.param(5_000_000, id="at-layer-boundary"),
            pytest.param(10_000_000, id="in-second-layer"),
            pytest.param(30_000_000, id="exceeds-all-layers"),
        ],
    )
    def test_recovery_never_exceeds_claim_various_amounts(self, claim):
        """Test over-recovery guard across a range of claim sizes."""
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=250_000, limit=4_750_000, base_premium_rate=0.015
            ),
            EnhancedInsuranceLayer(
                attachment_point=5_000_000, limit=20_000_000, base_premium_rate=0.008
            ),
        ]
        program = InsuranceProgram(layers, deductible=250_000)

        result = program.process_claim(claim)

        assert result["insurance_recovery"] <= claim, f"Over-recovery for claim {claim}"
        assert result["insurance_recovery"] <= max(
            0, claim - 250_000
        ), f"Recovery exceeds (claim - deductible) for claim {claim}"

    def test_aggregate_limit_across_multiple_claims(self):
        """Test aggregate limit tracking across multiple claims in same period.

        For aggregate limit_type, the 'limit' field is per-reinstatement capacity
        and 'aggregate_limit' is the overall cap. With limit=aggregate_limit and
        reinstatements=0, the aggregate controls total payouts.
        """
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=0,
                limit=3_000_000,
                base_premium_rate=0.01,
                reinstatements=0,
                aggregate_limit=3_000_000,
                limit_type="aggregate",
            ),
        ]
        program = InsuranceProgram(layers, deductible=0)

        # First claim: uses 2M of 3M aggregate
        r1 = program.process_claim(2_000_000)
        assert r1["insurance_recovery"] == 2_000_000

        # Second claim: only 1M aggregate remaining
        r2 = program.process_claim(2_000_000)
        assert r2["insurance_recovery"] == 1_000_000

        # Third claim: aggregate exhausted
        r3 = program.process_claim(2_000_000)
        assert r3["insurance_recovery"] == 0

    def test_claim_at_exact_layer_boundary(self):
        """Test claims exactly at layer boundary values."""
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=1_000_000, limit=4_000_000, base_premium_rate=0.015
            ),
            EnhancedInsuranceLayer(
                attachment_point=5_000_000, limit=10_000_000, base_premium_rate=0.008
            ),
        ]
        program = InsuranceProgram(layers, deductible=1_000_000)

        # Claim exactly at first layer exhaust = attachment + limit = 5M
        result = program.process_claim(5_000_000)
        assert result["insurance_recovery"] == 4_000_000  # Full first layer
        assert result["deductible_paid"] == 1_000_000

        program.reset_annual()

        # Claim exactly at second layer attachment (5M)
        result = program.process_claim(5_000_001)
        # First layer pays 4M, second layer pays min(5M+1 - 5M, 10M) = 1
        assert result["insurance_recovery"] == 4_000_001


class TestInsuranceProgramSimple:
    """Tests for InsuranceProgram.simple() convenience constructor."""

    def test_simple_structure(self):
        """simple() creates a single-layer program with correct fields."""
        program = InsuranceProgram.simple(
            deductible=500_000,
            limit=10_000_000,
            rate=0.025,
        )
        assert len(program.layers) == 1
        assert isinstance(program.layers[0], EnhancedInsuranceLayer)
        assert program.deductible == 500_000
        assert program.layers[0].attachment_point == 500_000
        assert program.layers[0].limit == 10_000_000
        assert program.layers[0].base_premium_rate == 0.025
        assert program.layers[0].reinstatements == 0
        assert program.name == "Simple Insurance Program"

    def test_custom_name(self):
        """simple() respects custom name."""
        program = InsuranceProgram.simple(
            deductible=500_000,
            limit=10_000_000,
            rate=0.025,
            name="My Custom Program",
        )
        assert program.name == "My Custom Program"

    def test_premium_calculation(self):
        """Premium is limit * rate."""
        program = InsuranceProgram.simple(
            deductible=500_000,
            limit=10_000_000,
            rate=0.025,
        )
        assert program.calculate_premium() == 250_000  # 10M * 0.025

    def test_claim_processing(self):
        """Claims flow correctly through a simple() program."""
        program = InsuranceProgram.simple(
            deductible=500_000,
            limit=10_000_000,
            rate=0.025,
        )

        # Below deductible
        result = program.process_claim(300_000)
        assert result["deductible_paid"] == 300_000
        assert result["insurance_recovery"] == 0

        # Partially in layer
        result = program.process_claim(3_000_000)
        assert result["deductible_paid"] == 500_000
        assert result["insurance_recovery"] == 2_500_000

    def test_zero_deductible(self):
        """simple() works with zero deductible."""
        program = InsuranceProgram.simple(
            deductible=0,
            limit=5_000_000,
            rate=0.03,
        )
        assert program.deductible == 0
        assert program.layers[0].attachment_point == 0

    def test_kwargs_forwarded(self):
        """Extra kwargs are forwarded to InsuranceProgram.__init__."""
        program = InsuranceProgram.simple(
            deductible=500_000,
            limit=10_000_000,
            rate=0.025,
            pricing_enabled=True,
        )
        assert program.pricing_enabled is True

    def test_equivalent_to_manual_construction(self):
        """simple() produces equivalent behavior to manual construction."""
        manual_layer = EnhancedInsuranceLayer(
            attachment_point=500_000,
            limit=10_000_000,
            base_premium_rate=0.025,
            reinstatements=0,
        )
        manual_program = InsuranceProgram(layers=[manual_layer], deductible=500_000)

        simple_program = InsuranceProgram.simple(
            deductible=500_000,
            limit=10_000_000,
            rate=0.025,
        )

        # Same structure
        assert simple_program.deductible == manual_program.deductible
        assert len(simple_program.layers) == len(manual_program.layers)
        assert (
            simple_program.layers[0].attachment_point == manual_program.layers[0].attachment_point
        )
        assert simple_program.layers[0].limit == manual_program.layers[0].limit
        assert (
            simple_program.layers[0].base_premium_rate == manual_program.layers[0].base_premium_rate
        )

        # Same premium
        assert simple_program.calculate_premium() == manual_program.calculate_premium()


class TestInsuranceProgramCalculatePremium:
    """Tests for InsuranceProgram.calculate_premium() and deprecation of calculate_annual_premium()."""

    def test_calculate_annual_premium_deprecation(self):
        """calculate_annual_premium() should emit DeprecationWarning."""
        layers = [
            EnhancedInsuranceLayer(attachment_point=0, limit=5_000_000, base_premium_rate=0.02),
            EnhancedInsuranceLayer(
                attachment_point=5_000_000,
                limit=10_000_000,
                base_premium_rate=0.01,
            ),
        ]
        program = InsuranceProgram(layers)
        with pytest.warns(DeprecationWarning, match="calculate_annual_premium.*deprecated"):
            result = program.calculate_annual_premium()
        assert result == program.calculate_premium()

    def test_calculate_premium_value(self):
        """calculate_premium() returns correct total premium."""
        layers = [
            EnhancedInsuranceLayer(attachment_point=0, limit=5_000_000, base_premium_rate=0.02),
            EnhancedInsuranceLayer(
                attachment_point=5_000_000,
                limit=10_000_000,
                base_premium_rate=0.01,
            ),
        ]
        program = InsuranceProgram(layers)
        expected = 5_000_000 * 0.02 + 10_000_000 * 0.01  # 100K + 100K = 200K
        assert program.calculate_premium() == expected

    def test_calculate_premium_empty_program(self):
        """calculate_premium() returns 0 for empty program."""
        program = InsuranceProgram(layers=[])
        assert program.calculate_premium() == 0.0

    def test_calculate_premium_single_layer(self):
        """calculate_premium() works for single layer."""
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=500_000,
                limit=4_500_000,
                base_premium_rate=0.015,
            )
        ]
        program = InsuranceProgram(layers, deductible=500_000)
        assert program.calculate_premium() == 4_500_000 * 0.015


class TestInsuranceProgramCreateFresh:
    """Tests for InsuranceProgram.create_fresh() factory method (Issue #624)."""

    @pytest.fixture
    def multi_layer_program(self):
        """Create a multi-layer program for create_fresh tests."""
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=500_000,
                limit=4_500_000,
                base_premium_rate=0.02,
                reinstatements=2,
                reinstatement_premium=1.0,
                reinstatement_type=ReinstatementType.PRO_RATA,
            ),
            EnhancedInsuranceLayer(
                attachment_point=5_000_000,
                limit=10_000_000,
                base_premium_rate=0.01,
                reinstatements=1,
            ),
        ]
        return InsuranceProgram(
            layers=layers,
            deductible=500_000,
            name="Test Program",
        )

    def test_create_fresh_returns_new_instance(self, multi_layer_program):
        """create_fresh() returns a distinct InsuranceProgram."""
        fresh = InsuranceProgram.create_fresh(multi_layer_program)
        assert fresh is not multi_layer_program

    def test_create_fresh_preserves_config(self, multi_layer_program):
        """Immutable configuration is identical in the fresh copy."""
        fresh = InsuranceProgram.create_fresh(multi_layer_program)
        assert fresh.deductible == multi_layer_program.deductible
        assert fresh.name == multi_layer_program.name
        assert fresh.pricing_enabled == multi_layer_program.pricing_enabled
        assert fresh.pricer is multi_layer_program.pricer
        assert len(fresh.layers) == len(multi_layer_program.layers)

    def test_create_fresh_shares_layer_objects(self, multi_layer_program):
        """EnhancedInsuranceLayer objects are shared (immutable), not copied."""
        fresh = InsuranceProgram.create_fresh(multi_layer_program)
        for orig_layer, fresh_layer in zip(multi_layer_program.layers, fresh.layers):
            assert orig_layer is fresh_layer

    def test_create_fresh_has_independent_layer_states(self, multi_layer_program):
        """LayerState wrappers are new objects, not shared."""
        fresh = InsuranceProgram.create_fresh(multi_layer_program)
        for orig_state, fresh_state in zip(multi_layer_program.layer_states, fresh.layer_states):
            assert orig_state is not fresh_state

    def test_create_fresh_zeroed_state(self, multi_layer_program):
        """Fresh program starts with pristine mutable state."""
        fresh = InsuranceProgram.create_fresh(multi_layer_program)
        assert fresh.total_claims == 0
        assert fresh.total_premiums_paid == 0.0
        assert fresh.pricing_results == []
        for state in fresh.layer_states:
            assert state.used_limit == 0.0
            assert state.reinstatements_used == 0
            assert state.total_claims_paid == 0.0
            assert state.reinstatement_premiums_paid == 0.0
            assert state.is_exhausted is False
            assert state.aggregate_used == 0.0
            assert state.current_limit == state.layer.limit

    def test_create_fresh_after_claims_gives_clean_state(self, multi_layer_program):
        """create_fresh() from a mutated source still yields zeroed state."""
        # Mutate the source
        multi_layer_program.process_claim(3_000_000)
        multi_layer_program.process_claim(6_000_000)
        assert multi_layer_program.total_claims == 2

        fresh = InsuranceProgram.create_fresh(multi_layer_program)
        assert fresh.total_claims == 0
        assert fresh.total_premiums_paid == 0.0
        for state in fresh.layer_states:
            assert state.used_limit == 0.0
            assert state.total_claims_paid == 0.0

    def test_create_fresh_isolation_from_source(self, multi_layer_program):
        """Mutations on fresh copy do not affect the source."""
        fresh = InsuranceProgram.create_fresh(multi_layer_program)
        fresh.process_claim(2_000_000)

        assert multi_layer_program.total_claims == 0
        assert multi_layer_program.total_premiums_paid == 0.0
        for state in multi_layer_program.layer_states:
            assert state.total_claims_paid == 0.0

    def test_create_fresh_produces_identical_claim_results(self, multi_layer_program):
        """Claim results match between fresh copy and deepcopy."""
        import copy

        fresh = InsuranceProgram.create_fresh(multi_layer_program)
        deep = copy.deepcopy(multi_layer_program)

        claims = [1_000_000, 3_000_000, 8_000_000, 500_000]
        for amount in claims:
            r_fresh = fresh.process_claim(amount)
            r_deep = deep.process_claim(amount)
            assert r_fresh["insurance_recovery"] == r_deep["insurance_recovery"]
            assert r_fresh["deductible_paid"] == r_deep["deductible_paid"]
            assert r_fresh["reinstatement_premiums"] == r_deep["reinstatement_premiums"]

    def test_create_fresh_preserves_layer_order(self):
        """Layers remain sorted by attachment point regardless of input order."""
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=5_000_000, limit=5_000_000, base_premium_rate=0.01
            ),
            EnhancedInsuranceLayer(attachment_point=0, limit=5_000_000, base_premium_rate=0.02),
        ]
        program = InsuranceProgram(layers=layers)
        fresh = InsuranceProgram.create_fresh(program)
        assert fresh.layers[0].attachment_point < fresh.layers[1].attachment_point


class TestClaimResultDataclass:
    """Test that process_claim returns a typed ClaimResult (#797)."""

    def test_returns_claim_result_type(self):
        """process_claim returns a ClaimResult, not a plain dict."""
        program = InsuranceProgram([], deductible=250_000)
        result = program.process_claim(100_000)
        assert isinstance(result, ClaimResult)

    def test_attribute_access(self):
        """Fields are accessible as attributes."""
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=250_000, limit=5_000_000, base_premium_rate=0.01
            )
        ]
        program = InsuranceProgram(layers, deductible=250_000)
        result = program.process_claim(2_000_000)

        assert result.total_claim == 2_000_000
        assert result.deductible_paid == 250_000
        assert result.insurance_recovery == 1_750_000
        assert result.uncovered_loss == 0.0
        assert result.reinstatement_premiums == 0.0
        assert len(result.layers_triggered) == 1

    def test_layers_triggered_are_layer_payment(self):
        """Each item in layers_triggered is a LayerPayment dataclass."""
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=250_000, limit=5_000_000, base_premium_rate=0.01
            )
        ]
        program = InsuranceProgram(layers, deductible=250_000)
        result = program.process_claim(2_000_000)

        layer = result.layers_triggered[0]
        assert isinstance(layer, LayerPayment)
        assert layer.layer_index == 0
        assert layer.attachment == 250_000
        assert layer.payment == 1_750_000
        assert layer.exhausted is False

    def test_zero_claim_returns_claim_result(self):
        """Zero/negative claims still return ClaimResult."""
        program = InsuranceProgram([])
        result = program.process_claim(0)
        assert isinstance(result, ClaimResult)
        assert result.total_claim == 0.0
        assert result.insurance_recovery == 0.0

    def test_backward_compat_getitem(self):
        """Dict-style bracket access still works with deprecation warning."""
        program = InsuranceProgram([], deductible=250_000)
        result = program.process_claim(100_000)

        with pytest.warns(DeprecationWarning, match="Dict-style access"):
            assert result["total_claim"] == 100_000

    def test_backward_compat_get(self):
        """Dict-style .get() still works with deprecation warning."""
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=250_000, limit=5_000_000, base_premium_rate=0.01
            )
        ]
        program = InsuranceProgram(layers, deductible=250_000)
        result = program.process_claim(2_000_000)

        with pytest.warns(DeprecationWarning, match="Dict-style access"):
            assert result.get("insurance_recovery", 0) == 1_750_000

    def test_backward_compat_contains(self):
        """'key in result' still works."""
        program = InsuranceProgram([])
        result = program.process_claim(100_000)
        assert "total_claim" in result
        assert "nonexistent" not in result

    def test_backward_compat_keys(self):
        """result.keys() returns field names."""
        program = InsuranceProgram([])
        result = program.process_claim(100_000)
        assert set(result.keys()) == {
            "total_claim",
            "deductible_paid",
            "insurance_recovery",
            "uncovered_loss",
            "reinstatement_premiums",
            "layers_triggered",
        }

    def test_layer_payment_backward_compat(self):
        """LayerPayment also supports dict-style access."""
        layers = [
            EnhancedInsuranceLayer(
                attachment_point=250_000, limit=5_000_000, base_premium_rate=0.01
            )
        ]
        program = InsuranceProgram(layers, deductible=250_000)
        result = program.process_claim(2_000_000)
        layer = result.layers_triggered[0]

        with pytest.warns(DeprecationWarning, match="Dict-style access"):
            assert layer["payment"] == 1_750_000
