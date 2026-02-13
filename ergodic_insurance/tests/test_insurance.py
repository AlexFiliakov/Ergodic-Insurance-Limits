"""Unit tests for insurance policy structure and claim processing.

This module contains comprehensive tests for the insurance layer system
including layer calculations, claim processing, and configuration loading.
"""

from pathlib import Path

import pytest
import yaml

from ergodic_insurance.insurance import InsuranceLayer, InsurancePolicy


class TestInsuranceLayer:
    """Test suite for InsuranceLayer class."""

    def test_layer_initialization(self):
        """Test insurance layer initialization."""
        layer = InsuranceLayer(attachment_point=1_000_000, limit=5_000_000, rate=0.01)
        assert layer.attachment_point == 1_000_000
        assert layer.limit == 5_000_000
        assert layer.rate == 0.01

    @pytest.mark.parametrize(
        "loss,expected",
        [
            pytest.param(500_000, 0.0, id="below-attachment"),
            pytest.param(1_000_000, 0.0, id="at-attachment"),
            pytest.param(3_000_000, 2_000_000, id="within-layer"),
            pytest.param(10_000_000, 5_000_000, id="exceeds-layer"),
        ],
    )
    def test_calculate_recovery(self, loss, expected):
        """Test recovery calculation for various loss amounts."""
        layer = InsuranceLayer(attachment_point=1_000_000, limit=5_000_000, rate=0.01)
        assert layer.calculate_recovery(loss) == expected

    def test_calculate_premium(self):
        """Test premium calculation."""
        layer = InsuranceLayer(attachment_point=1_000_000, limit=5_000_000, rate=0.015)
        premium = layer.calculate_premium()
        assert premium == 75_000  # 5M * 0.015


class TestInsurancePolicy:
    """Test suite for InsurancePolicy class."""

    @pytest.fixture
    def simple_policy(self):
        """Create a simple test policy with two layers."""
        layers = [
            InsuranceLayer(attachment_point=0, limit=5_000_000, rate=0.02),
            InsuranceLayer(attachment_point=5_000_000, limit=10_000_000, rate=0.01),
        ]
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            return InsurancePolicy(layers=layers, deductible=0)

    @pytest.fixture
    def policy_with_deductible(self):
        """Create a policy with deductible and multiple layers."""
        layers = [
            InsuranceLayer(attachment_point=500_000, limit=4_500_000, rate=0.015),
            InsuranceLayer(attachment_point=5_000_000, limit=20_000_000, rate=0.008),
            InsuranceLayer(attachment_point=25_000_000, limit=25_000_000, rate=0.004),
        ]
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            return InsurancePolicy(layers=layers, deductible=500_000)

    def test_policy_initialization(self, simple_policy):
        """Test policy initialization and layer sorting."""
        assert len(simple_policy.layers) == 2
        assert simple_policy.deductible == 0
        # Layers should be sorted by attachment point
        assert simple_policy.layers[0].attachment_point == 0
        assert simple_policy.layers[1].attachment_point == 5_000_000

    def test_process_claim_zero(self, simple_policy):
        """Test processing zero claim."""
        company_payment, insurance_recovery = simple_policy.process_claim(0)
        assert company_payment == 0
        assert insurance_recovery == 0

    def test_process_claim_negative(self, simple_policy):
        """Test processing negative claim (edge case)."""
        company_payment, insurance_recovery = simple_policy.process_claim(-1000)
        assert company_payment == 0
        assert insurance_recovery == 0

    def test_process_claim_within_first_layer(self, simple_policy):
        """Test claim within first layer coverage."""
        company_payment, insurance_recovery = simple_policy.process_claim(3_000_000)
        assert company_payment == 0  # No deductible
        assert insurance_recovery == 3_000_000  # Fully covered by first layer

    def test_process_claim_across_layers(self, simple_policy):
        """Test claim that spans multiple layers."""
        # 8M claim: 5M from first layer, 3M from second layer
        company_payment, insurance_recovery = simple_policy.process_claim(8_000_000)
        assert company_payment == 0  # No deductible
        assert insurance_recovery == 8_000_000  # Fully covered

    def test_process_claim_exceeds_coverage(self, simple_policy):
        """Test claim that exceeds total coverage."""
        # 20M claim: 5M + 10M = 15M covered, 5M excess
        company_payment, insurance_recovery = simple_policy.process_claim(20_000_000)
        assert company_payment == 5_000_000  # Excess not covered
        assert insurance_recovery == 15_000_000  # Maximum coverage

    def test_process_claim_with_deductible(self, policy_with_deductible):
        """Test claim processing with deductible."""
        # 1M claim: 500K deductible, 500K from insurance
        company_payment, insurance_recovery = policy_with_deductible.process_claim(1_000_000)
        assert company_payment == 500_000  # Deductible
        assert insurance_recovery == 500_000  # From first layer

    def test_process_claim_below_deductible(self, policy_with_deductible):
        """Test claim below deductible amount."""
        company_payment, insurance_recovery = policy_with_deductible.process_claim(300_000)
        assert company_payment == 300_000  # All paid by company
        assert insurance_recovery == 0  # No insurance coverage

    def test_process_large_claim_with_deductible(self, policy_with_deductible):
        """Test large claim across all layers with deductible."""
        # 30M claim: 500K deductible + 4.5M primary + 20M excess1 + 5M excess2
        company_payment, insurance_recovery = policy_with_deductible.process_claim(30_000_000)
        assert company_payment == 500_000  # Just the deductible
        assert insurance_recovery == 29_500_000  # Rest from insurance

    def test_calculate_total_premium_simple(self, simple_policy):
        """Test total premium calculation for simple policy."""
        premium = simple_policy.calculate_premium()
        expected = 5_000_000 * 0.02 + 10_000_000 * 0.01  # 100K + 100K
        assert premium == expected

    def test_calculate_total_premium_complex(self, policy_with_deductible):
        """Test total premium calculation for complex policy."""
        premium = policy_with_deductible.calculate_premium()
        expected = (
            4_500_000 * 0.015
            + 20_000_000 * 0.008  # 67,500
            + 25_000_000 * 0.004  # 160,000  # 100,000
        )
        assert premium == expected

    def test_get_total_coverage(self, simple_policy):
        """Test total coverage calculation."""
        coverage = simple_policy.get_total_coverage()
        assert coverage == 15_000_000  # 5M + 10M

    def test_get_total_coverage_with_deductible(self, policy_with_deductible):
        """Test total coverage with deductible."""
        coverage = policy_with_deductible.get_total_coverage()
        # 50M total - 500K deductible = 49.5M insurance coverage
        assert coverage == 49_500_000

    def test_get_total_coverage_deductible_exceeds_layers(self):
        """get_total_coverage returns 0 when deductible exceeds highest layer exhaust."""
        layers = [
            InsuranceLayer(attachment_point=100_000, limit=200_000, rate=0.01),
        ]
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy(layers=layers, deductible=500_000)
        assert policy.get_total_coverage() == 0.0

    def test_layer_recovery_capped_per_layer(self):
        """Layers overlapping the deductible region produce correct totals."""
        # Layer attaches at 0 (below deductible), so part of its response
        # overlaps with the deductible region.
        layers = [
            InsuranceLayer(attachment_point=0, limit=5_000_000, rate=0.02),
            InsuranceLayer(attachment_point=5_000_000, limit=5_000_000, rate=0.01),
        ]
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy(layers=layers, deductible=1_000_000)

        # 3M claim: deductible=1M, max_recoverable=2M
        # Layer 0 would pay min(3M, 5M)=3M but capped at remaining 2M
        company, insurance = policy.process_claim(3_000_000)
        assert insurance == 2_000_000
        assert company + insurance == 3_000_000

        # Same via calculate_recovery
        recovery = policy.calculate_recovery(3_000_000)
        assert recovery == 2_000_000

    def test_empty_policy(self):
        """Test policy with no layers."""
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy(layers=[], deductible=100_000)

        company_payment, insurance_recovery = policy.process_claim(500_000)
        assert company_payment == 500_000  # Company pays everything
        assert insurance_recovery == 0  # No insurance coverage

        assert policy.calculate_premium() == 0
        assert policy.get_total_coverage() == 0


class TestInsurancePolicyYAML:
    """Test loading insurance policy from YAML configuration."""

    @pytest.fixture
    def yaml_config_path(self, tmp_path):
        """Create a temporary YAML configuration file."""
        config = {
            "deductible": 250_000,
            "layers": [
                {"attachment_point": 250_000, "limit": 2_000_000, "rate": 0.02},
                {"attachment_point": 2_250_000, "limit": 5_000_000, "rate": 0.01},
            ],
        }

        yaml_file = tmp_path / "test_insurance.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(config, f)

        return str(yaml_file)

    def test_load_from_yaml(self, yaml_config_path):
        """Test loading policy from YAML file."""
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy.from_yaml(yaml_config_path)

        assert policy.deductible == 250_000
        assert len(policy.layers) == 2

        # Check first layer
        assert policy.layers[0].attachment_point == 250_000
        assert policy.layers[0].limit == 2_000_000
        assert policy.layers[0].rate == 0.02

        # Check second layer
        assert policy.layers[1].attachment_point == 2_250_000
        assert policy.layers[1].limit == 5_000_000
        assert policy.layers[1].rate == 0.01

    def test_load_default_config(self):
        """Test loading the default insurance configuration."""
        # Get path to default config
        config_path = Path(__file__).parent.parent / "data" / "parameters" / "insurance.yaml"

        if config_path.exists():
            with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
                policy = InsurancePolicy.from_yaml(str(config_path))

            assert policy.deductible == 500_000
            assert len(policy.layers) == 3

            # Verify layers are sorted by attachment point
            for i in range(len(policy.layers) - 1):
                assert policy.layers[i].attachment_point < policy.layers[i + 1].attachment_point

            # Test a sample claim
            company_payment, insurance_recovery = policy.process_claim(10_000_000)
            assert company_payment == 500_000  # Deductible
            assert insurance_recovery == 9_500_000  # Rest from insurance


class TestIntegrationScenarios:
    """Integration tests for realistic insurance scenarios."""

    @pytest.fixture
    def realistic_policy(self):
        """Create a realistic insurance policy."""
        layers = [
            InsuranceLayer(attachment_point=500_000, limit=4_500_000, rate=0.015),
            InsuranceLayer(attachment_point=5_000_000, limit=20_000_000, rate=0.008),
            InsuranceLayer(attachment_point=25_000_000, limit=25_000_000, rate=0.004),
        ]
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            return InsurancePolicy(layers=layers, deductible=500_000)

    def test_small_attritional_losses(self, realistic_policy):
        """Test handling of small frequent losses."""
        losses = [100_000, 200_000, 150_000, 300_000]
        total_company = 0
        total_insurance = 0

        for loss in losses:
            company, insurance = realistic_policy.process_claim(loss)
            total_company += company
            total_insurance += insurance

        # All losses below deductible
        assert total_company == sum(losses)
        assert total_insurance == 0

    def test_medium_operational_loss(self, realistic_policy):
        """Test medium-sized operational loss."""
        loss = 2_000_000
        company, insurance = realistic_policy.process_claim(loss)

        assert company == 500_000  # Deductible
        assert insurance == 1_500_000  # From primary layer

    def test_large_catastrophic_loss(self, realistic_policy):
        """Test large catastrophic loss across multiple layers."""
        loss = 35_000_000
        company, insurance = realistic_policy.process_claim(loss)

        # 500K deductible + 4.5M primary + 20M excess1 + 10M excess2
        assert company == 500_000  # Just deductible
        assert insurance == 34_500_000  # Rest from layers

    def test_loss_exceeding_all_coverage(self, realistic_policy):
        """Test loss that exceeds all insurance coverage."""
        loss = 60_000_000
        company, insurance = realistic_policy.process_claim(loss)

        # 500K deductible + 10M excess (beyond coverage)
        assert company == 10_500_000
        # 4.5M primary + 20M excess1 + 25M excess2
        assert insurance == 49_500_000

    def test_annual_premium_cost(self, realistic_policy):
        """Test annual premium calculation for realistic policy."""
        premium = realistic_policy.calculate_premium()

        # Verify premium is reasonable (0.5-1% of total coverage)
        total_coverage = realistic_policy.get_total_coverage()
        base_premium_rate = premium / total_coverage

        assert 0.005 <= base_premium_rate <= 0.01  # Between 0.5% and 1%
        assert premium == 327_500  # Exact calculation


class TestOverRecoveryGuard:
    """Tests for the over-recovery guard (issue #310).

    Ensures total insurance recovery never exceeds (claim - deductible),
    especially when deductible != first layer attachment point.
    """

    def test_deductible_below_first_attachment(self):
        """Test that over-recovery is prevented when deductible < first layer attachment."""
        # Deductible is 200K but first layer attaches at 500K
        # Without guard: company pays 200K deductible, layer pays from 500K up
        # → gap of 300K is double-counted
        layers = [
            InsuranceLayer(attachment_point=500_000, limit=4_500_000, rate=0.015),
        ]
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy(layers=layers, deductible=200_000)

        # 3M claim: deductible=200K, layer recovery=min(3M-500K, 4.5M)=2.5M
        # Without guard: total = 200K + 2.5M = 2.7M (< 3M, no over-recovery here)
        company, insurance = policy.process_claim(3_000_000)
        assert company + insurance == 3_000_000

        # 1M claim: deductible=200K, max_recoverable=800K
        # layer recovery=min(1M-500K, 4.5M)=500K, cap at 800K → 500K (within cap)
        company, insurance = policy.process_claim(1_000_000)
        assert insurance <= 1_000_000 - 200_000  # Never exceeds claim - deductible
        assert company + insurance == 1_000_000

    def test_deductible_above_first_attachment(self):
        """Test over-recovery when deductible > first layer attachment."""
        # Deductible 1M but first layer attaches at 500K
        # Layer would pay from 500K-5.5M for a claim, but deductible already covers to 1M
        layers = [
            InsuranceLayer(attachment_point=500_000, limit=5_000_000, rate=0.015),
        ]
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy(layers=layers, deductible=1_000_000)

        # 3M claim: deductible=1M, max_recoverable=2M
        # Layer: min(3M-500K, 5M) = 2.5M, but cap at 2M
        company, insurance = policy.process_claim(3_000_000)
        assert insurance <= 2_000_000  # Capped at claim - deductible
        assert company + insurance == 3_000_000

    def test_overlapping_layers_capped(self):
        """Test that overlapping layer configurations don't cause over-recovery."""
        # Two layers that overlap in coverage region
        layers = [
            InsuranceLayer(attachment_point=0, limit=5_000_000, rate=0.02),
            InsuranceLayer(attachment_point=0, limit=5_000_000, rate=0.02),
        ]
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy(layers=layers, deductible=0)

        # 3M claim: each layer would pay 3M, total 6M > 3M claim
        company, insurance = policy.process_claim(3_000_000)
        assert insurance <= 3_000_000  # Capped at claim amount
        assert company + insurance == 3_000_000

    @pytest.mark.parametrize(
        "claim",
        [
            pytest.param(100_000, id="below-deductible"),
            pytest.param(500_000, id="at-deductible"),
            pytest.param(1_000_000, id="within-layer"),
            pytest.param(5_000_000, id="mid-layer"),
            pytest.param(15_000_000, id="exceeds-layers"),
        ],
    )
    def test_recovery_never_exceeds_claim(self, claim):
        """Test that recovery <= claim for various configurations."""
        layers = [
            InsuranceLayer(attachment_point=100_000, limit=10_000_000, rate=0.01),
            InsuranceLayer(attachment_point=200_000, limit=10_000_000, rate=0.01),
        ]
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy(layers=layers, deductible=500_000)

        company, insurance = policy.process_claim(claim)
        assert insurance <= claim, f"Over-recovery for claim {claim}"
        assert insurance <= claim - min(
            claim, 500_000
        ), f"Recovery exceeds (claim - deductible) for claim {claim}"
        assert company + insurance == claim

    def test_calculate_recovery_capped(self):
        """Test calculate_recovery also respects the cap."""
        layers = [
            InsuranceLayer(attachment_point=500_000, limit=5_000_000, rate=0.015),
        ]
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy(layers=layers, deductible=1_000_000)

        # 3M claim: max_recoverable = 3M - 1M = 2M
        # Layer: min(3M - 500K, 5M) = 2.5M → capped to 2M
        recovery = policy.calculate_recovery(3_000_000)
        assert recovery <= 2_000_000

    def test_zero_claim_returns_zero(self):
        """Test zero and negative claims produce zero recovery."""
        layers = [InsuranceLayer(attachment_point=0, limit=5_000_000, rate=0.01)]
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy(layers=layers, deductible=0)

        company, insurance = policy.process_claim(0)
        assert company == 0 and insurance == 0

        company, insurance = policy.process_claim(-100)
        assert company == 0 and insurance == 0

    def test_claim_exactly_at_deductible(self):
        """Test claim exactly equal to deductible."""
        layers = [InsuranceLayer(attachment_point=500_000, limit=5_000_000, rate=0.01)]
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy(layers=layers, deductible=500_000)

        company, insurance = policy.process_claim(500_000)
        assert company == 500_000
        assert insurance == 0

    def test_claim_exactly_at_layer_boundary(self):
        """Test claims at exact layer boundaries."""
        layers = [
            InsuranceLayer(attachment_point=1_000_000, limit=4_000_000, rate=0.015),
            InsuranceLayer(attachment_point=5_000_000, limit=10_000_000, rate=0.008),
        ]
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy(layers=layers, deductible=1_000_000)

        # Claim exactly at first layer exhaust point (5M)
        company, insurance = policy.process_claim(5_000_000)
        assert insurance == 4_000_000  # Full first layer
        assert company == 1_000_000  # Deductible only

        # Claim exactly at second layer attachment (5M)
        # Same as above — second layer attaches at 5M, claim=5M doesn't penetrate
        company2, insurance2 = policy.process_claim(5_000_000)
        assert insurance2 == 4_000_000


class TestFromSimple:
    """Tests for InsurancePolicy.from_simple() convenience constructor."""

    def test_from_simple_structure(self):
        """from_simple() creates a single-layer policy with correct fields."""
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy.from_simple(
                deductible=500_000,
                limit=10_000_000,
                premium_rate=0.025,
            )
        assert len(policy.layers) == 1
        assert policy.deductible == 500_000
        assert policy.layers[0].attachment_point == 500_000
        assert policy.layers[0].limit == 10_000_000
        assert policy.layers[0].rate == 0.025

    def test_equivalent_to_manual_construction(self):
        """from_simple() produces an identical policy to manual 2-step construction."""
        manual_layer = InsuranceLayer(
            attachment_point=500_000,
            limit=10_000_000,
            rate=0.025,
        )
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            manual_policy = InsurancePolicy(layers=[manual_layer], deductible=500_000)

        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            simple_policy = InsurancePolicy.from_simple(
                deductible=500_000,
                limit=10_000_000,
                premium_rate=0.025,
            )

        # Same structure
        assert simple_policy.deductible == manual_policy.deductible
        assert len(simple_policy.layers) == len(manual_policy.layers)
        assert simple_policy.layers[0].attachment_point == manual_policy.layers[0].attachment_point
        assert simple_policy.layers[0].limit == manual_policy.layers[0].limit
        assert simple_policy.layers[0].rate == manual_policy.layers[0].rate

        # Same premium
        assert simple_policy.calculate_premium() == manual_policy.calculate_premium()

        # Same claim processing
        for claim in [100_000, 500_000, 3_000_000, 15_000_000]:
            assert simple_policy.process_claim(claim) == manual_policy.process_claim(claim)

    def test_premium_calculation(self):
        """Premium is limit * premium_rate."""
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy.from_simple(
                deductible=500_000,
                limit=10_000_000,
                premium_rate=0.025,
            )
        assert policy.calculate_premium() == 250_000  # 10M * 0.025

    def test_claim_processing(self):
        """Claims flow correctly through a from_simple() policy."""
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy.from_simple(
                deductible=500_000,
                limit=10_000_000,
                premium_rate=0.025,
            )

        # Below deductible
        company, insurance = policy.process_claim(300_000)
        assert company == 300_000
        assert insurance == 0

        # Partially in layer
        company, insurance = policy.process_claim(3_000_000)
        assert company == 500_000
        assert insurance == 2_500_000

        # Exceeds coverage
        company, insurance = policy.process_claim(15_000_000)
        assert company == 5_000_000  # 500K deductible + 4.5M excess
        assert insurance == 10_000_000

    def test_zero_deductible(self):
        """from_simple() works with zero deductible."""
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy.from_simple(
                deductible=0,
                limit=5_000_000,
                premium_rate=0.03,
            )
        assert policy.deductible == 0
        assert policy.layers[0].attachment_point == 0

        company, insurance = policy.process_claim(2_000_000)
        assert company == 0
        assert insurance == 2_000_000

    def test_kwargs_forwarded(self):
        """Extra kwargs are forwarded to InsurancePolicy.__init__."""
        with pytest.warns(DeprecationWarning, match="InsurancePolicy is deprecated"):
            policy = InsurancePolicy.from_simple(
                deductible=500_000,
                limit=10_000_000,
                premium_rate=0.025,
                pricing_enabled=True,
            )
        assert policy.pricing_enabled is True

    def test_validation_propagates(self):
        """InsuranceLayer validation errors propagate from from_simple()."""
        with pytest.raises(ValueError, match="Limit must be positive"):
            InsurancePolicy.from_simple(
                deductible=500_000,
                limit=0,
                premium_rate=0.025,
            )

        with pytest.raises(ValueError, match="Premium rate must be non-negative"):
            InsurancePolicy.from_simple(
                deductible=500_000,
                limit=10_000_000,
                premium_rate=-0.01,
            )
