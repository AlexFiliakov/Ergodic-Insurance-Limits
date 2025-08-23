"""Unit tests for insurance policy structure and claim processing.

This module contains comprehensive tests for the insurance layer system
including layer calculations, claim processing, and configuration loading.
"""

from pathlib import Path

import pytest
import yaml

from ergodic_insurance.src.insurance import InsuranceLayer, InsurancePolicy


class TestInsuranceLayer:
    """Test suite for InsuranceLayer class."""

    def test_layer_initialization(self):
        """Test insurance layer initialization."""
        layer = InsuranceLayer(attachment_point=1_000_000, limit=5_000_000, rate=0.01)
        assert layer.attachment_point == 1_000_000
        assert layer.limit == 5_000_000
        assert layer.rate == 0.01

    def test_calculate_recovery_below_attachment(self):
        """Test recovery calculation when loss is below attachment point."""
        layer = InsuranceLayer(attachment_point=1_000_000, limit=5_000_000, rate=0.01)
        recovery = layer.calculate_recovery(500_000)
        assert recovery == 0.0

    def test_calculate_recovery_within_layer(self):
        """Test recovery calculation when loss is within layer coverage."""
        layer = InsuranceLayer(attachment_point=1_000_000, limit=5_000_000, rate=0.01)
        # Loss of 3M: 1M attachment + 2M into layer
        recovery = layer.calculate_recovery(3_000_000)
        assert recovery == 2_000_000

    def test_calculate_recovery_exceeds_layer(self):
        """Test recovery calculation when loss exceeds layer limit."""
        layer = InsuranceLayer(attachment_point=1_000_000, limit=5_000_000, rate=0.01)
        # Loss of 10M: 1M attachment + 9M excess (capped at 5M limit)
        recovery = layer.calculate_recovery(10_000_000)
        assert recovery == 5_000_000

    def test_calculate_recovery_at_attachment(self):
        """Test recovery calculation when loss equals attachment point."""
        layer = InsuranceLayer(attachment_point=1_000_000, limit=5_000_000, rate=0.01)
        recovery = layer.calculate_recovery(1_000_000)
        assert recovery == 0.0

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
        return InsurancePolicy(layers=layers, deductible=0)

    @pytest.fixture
    def policy_with_deductible(self):
        """Create a policy with deductible and multiple layers."""
        layers = [
            InsuranceLayer(attachment_point=500_000, limit=4_500_000, rate=0.015),
            InsuranceLayer(attachment_point=5_000_000, limit=20_000_000, rate=0.008),
            InsuranceLayer(attachment_point=25_000_000, limit=25_000_000, rate=0.004),
        ]
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

    def test_empty_policy(self):
        """Test policy with no layers."""
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
        premium_rate = premium / total_coverage

        assert 0.005 <= premium_rate <= 0.01  # Between 0.5% and 1%
        assert premium == 327_500  # Exact calculation
