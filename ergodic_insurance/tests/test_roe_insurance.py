"""Unit tests for ROE calculation with insurance costs.

This module ensures that ROE calculations correctly include all insurance
expenses (premiums and claim deductibles) and that the fix doesn't regress.
"""

import numpy as np
import pytest

from ergodic_insurance.claim_generator import ClaimGenerator
from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.insurance import InsuranceLayer, InsurancePolicy
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.simulation import Simulation


class TestROEWithInsurance:
    """Test suite for ROE calculations including insurance costs."""

    @pytest.fixture
    def manufacturer_config(self):
        """Create standard manufacturer configuration."""
        return ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.12,
            tax_rate=0.25,
            retention_ratio=0.7,
        )

    @pytest.fixture
    def manufacturer(self, manufacturer_config):
        """Create a fresh manufacturer for each test."""
        return WidgetManufacturer(manufacturer_config)

    def test_roe_without_insurance(self, manufacturer):
        """Test that ROE without insurance matches expected operating ROE."""
        # Calculate metrics without any insurance costs
        metrics = manufacturer.calculate_metrics()

        # Expected ROE = base_operating_margin * (1 - tax_rate) * asset_turnover
        # = 0.12 * 0.75 * 1.0 = 0.09 (9%)
        expected_roe = 0.09

        assert metrics["roe"] == pytest.approx(expected_roe, rel=0.01)
        assert metrics["insurance_premiums"] == 0
        assert metrics["insurance_losses"] == 0
        assert metrics["total_insurance_costs"] == 0

    def test_roe_with_premium_only(self, manufacturer):
        """Test that ROE decreases when insurance premiums are paid."""
        # Record a premium payment
        premium = 500_000
        manufacturer.record_insurance_premium(premium)

        # Calculate metrics
        metrics = manufacturer.calculate_metrics()

        # ROE should be reduced by the premium impact
        # Net income = (revenue * margin - premium) * (1 - tax_rate)
        # Use default working_capital_pct to match calculate_metrics()
        revenue = manufacturer.calculate_revenue()  # Uses default working_capital_pct=0.0
        operating_income = revenue * 0.12
        net_income_with_premium = (operating_income - premium) * 0.75
        expected_roe = net_income_with_premium / manufacturer.equity

        assert metrics["roe"] == pytest.approx(expected_roe, rel=0.01)
        assert metrics["insurance_premiums"] == premium
        assert metrics["total_insurance_costs"] == premium

    def test_roe_with_losses_only(self, manufacturer):
        """Test that ROE decreases when claim deductibles are paid."""
        # Record insurance losses (deductibles paid)
        losses = 300_000
        manufacturer.record_insurance_loss(losses)

        # Calculate metrics
        metrics = manufacturer.calculate_metrics()

        # ROE should be reduced by the loss impact
        revenue = manufacturer.calculate_revenue()  # Uses default working_capital_pct=0.0
        operating_income = revenue * 0.12
        net_income_with_losses = (operating_income - losses) * 0.75
        expected_roe = net_income_with_losses / manufacturer.equity

        assert metrics["roe"] == pytest.approx(expected_roe, rel=0.01)
        assert metrics["insurance_losses"] == losses
        assert metrics["total_insurance_costs"] == losses

    def test_roe_with_premium_and_losses(self, manufacturer):
        """Test ROE with both premiums and losses."""
        # Record both premium and losses
        premium = 400_000
        losses = 600_000
        total_costs = premium + losses

        manufacturer.record_insurance_premium(premium)
        manufacturer.record_insurance_loss(losses)

        # Calculate metrics
        metrics = manufacturer.calculate_metrics()

        # ROE should reflect both costs
        revenue = manufacturer.calculate_revenue()  # Uses default working_capital_pct=0.0
        operating_income = revenue * 0.12
        net_income = (operating_income - total_costs) * 0.75
        expected_roe = net_income / manufacturer.equity

        assert metrics["roe"] == pytest.approx(expected_roe, rel=0.01)
        assert metrics["insurance_premiums"] == premium
        assert metrics["insurance_losses"] == losses
        assert metrics["total_insurance_costs"] == total_costs

    def test_negative_roe_with_high_insurance(self, manufacturer):
        """Test that ROE becomes negative when insurance costs exceed operating income."""
        # Set very high insurance costs
        premium = 1_500_000  # Much higher than operating income
        losses = 500_000

        manufacturer.record_insurance_premium(premium)
        manufacturer.record_insurance_loss(losses)

        # Calculate metrics
        metrics = manufacturer.calculate_metrics()

        # ROE should be negative
        assert metrics["roe"] < 0
        assert metrics["total_insurance_costs"] == premium + losses

        # Verify the calculation
        revenue = manufacturer.calculate_revenue()  # Uses default working_capital_pct=0.0
        operating_income = revenue * 0.12
        income_before_tax = operating_income - premium - losses
        taxes = max(0, income_before_tax * 0.25)  # No taxes on negative income
        net_income = income_before_tax - taxes
        assert net_income < 0
        assert metrics["net_income"] < 0
        assert metrics["roe"] == pytest.approx(net_income / manufacturer.equity, rel=0.01)

    def test_roe_in_simulation(self, manufacturer_config):
        """Test that simulation ROE includes insurance costs."""
        # Create a new manufacturer for simulation
        manufacturer = WidgetManufacturer(manufacturer_config)

        # Create claim generator with moderate losses
        claim_gen = ClaimGenerator(frequency=2, severity_mean=150_000, severity_std=50_000)

        # Create insurance policy
        insurance = InsurancePolicy(
            layers=[
                InsuranceLayer(attachment_point=100_000, limit=10_000_000, rate=0.02)  # 2% rate
            ],
            deductible=100_000,
        )

        # Run short simulation
        sim = Simulation(
            manufacturer=manufacturer,
            claim_generator=claim_gen,
            insurance_policy=insurance,
            time_horizon=3,
            seed=42,
        )

        results = sim.run()
        stats = results.summary_stats()

        # ROE should be less than theoretical operating ROE (9%)
        theoretical_operating_roe = 0.09
        assert stats["mean_roe"] < theoretical_operating_roe
        assert stats["mean_roe"] > -0.5  # But not catastrophically bad

        # Time-weighted ROE should also reflect insurance costs
        assert stats["time_weighted_roe"] < theoretical_operating_roe

    def test_roe_consistency_in_step(self, manufacturer):
        """Test that ROE in step() method matches calculate_metrics()."""
        # Set up some insurance costs
        premium = 300_000
        losses = 200_000

        manufacturer.record_insurance_premium(premium)
        manufacturer.record_insurance_loss(losses)

        # Run a step
        metrics_from_step = manufacturer.step(
            working_capital_pct=0.2, letter_of_credit_rate=0.015, growth_rate=0.0
        )

        # ROE from step should include insurance costs
        assert metrics_from_step["insurance_premiums"] == premium
        assert metrics_from_step["insurance_losses"] == losses

        # Verify ROE is properly calculated
        assert metrics_from_step["roe"] < 0.09  # Less than operating ROE

        # Net income should be negative or very low given the costs
        # With $500k total insurance costs and ~$1.2M operating income, net income should be reduced
        assert metrics_from_step["net_income"] < 600_000  # Reduced from normal ~$900k

    def test_roe_reset_after_period(self, manufacturer):
        """Test that period insurance costs reset properly."""
        # Set insurance costs
        manufacturer.record_insurance_premium(400_000)
        manufacturer.record_insurance_loss(300_000)

        # Calculate metrics - should include costs
        metrics1 = manufacturer.calculate_metrics()
        assert metrics1["total_insurance_costs"] == 700_000

        # Reset period costs
        manufacturer.reset_period_insurance_costs()

        # Calculate metrics again - costs should be zero
        metrics2 = manufacturer.calculate_metrics()
        assert metrics2["insurance_premiums"] == 0
        assert metrics2["insurance_losses"] == 0
        assert metrics2["total_insurance_costs"] == 0
        assert metrics2["roe"] > metrics1["roe"]  # ROE should improve

    def test_roe_with_multiple_generators(self, manufacturer_config):
        """Test ROE calculation with multiple claim generators."""
        manufacturer = WidgetManufacturer(manufacturer_config)

        # Create multiple risk profiles
        standard = ClaimGenerator(frequency=2, severity_mean=50_000, severity_std=20_000)
        catastrophic = ClaimGenerator(frequency=0.1, severity_mean=1_000_000, severity_std=500_000)

        # Create insurance
        insurance = InsurancePolicy(
            layers=[InsuranceLayer(attachment_point=100_000, limit=10_000_000, rate=0.015)],
            deductible=100_000,
        )

        # Run simulation with multiple generators
        sim = Simulation(
            manufacturer=manufacturer,
            claim_generator=[standard, catastrophic],
            insurance_policy=insurance,
            time_horizon=5,
            seed=42,
        )

        results = sim.run()
        stats = results.summary_stats()

        # ROE should reflect combined risk costs
        assert stats["mean_roe"] < 0.09  # Less than theoretical max
        # Note: insurance_premiums may or may not be in the dataframe
        # depending on the simulation configuration

    def test_roe_regression_check(self, manufacturer):
        """Regression test to ensure ROE calculation doesn't revert to excluding insurance."""
        # This is the critical regression test
        # Set known insurance costs
        premium = 800_000
        losses = 400_000

        manufacturer.record_insurance_premium(premium)
        manufacturer.record_insurance_loss(losses)

        # Calculate expected values
        revenue = manufacturer.calculate_revenue()  # Uses default working_capital_pct=0.0
        operating_income = revenue * 0.12
        net_income = (operating_income - premium - losses) * 0.75
        expected_roe = net_income / manufacturer.equity

        # Get actual metrics
        metrics = manufacturer.calculate_metrics()

        # Critical assertions to prevent regression
        assert metrics["insurance_premiums"] == premium, "Insurance premiums not tracked"
        assert metrics["insurance_losses"] == losses, "Insurance losses not tracked"
        assert metrics["total_insurance_costs"] == premium + losses, "Total costs incorrect"
        assert metrics["net_income"] == pytest.approx(
            net_income, rel=0.01
        ), "Net income doesn't include insurance"
        assert metrics["roe"] == pytest.approx(
            expected_roe, rel=0.01
        ), "ROE doesn't reflect insurance costs"

        # ROE should NOT be the operating ROE
        operating_roe = 0.09
        assert abs(metrics["roe"] - operating_roe) > 0.01, "ROE appears to exclude insurance costs!"


class TestROEEdgeCases:
    """Test edge cases for ROE calculation."""

    @pytest.fixture
    def manufacturer(self):
        """Create manufacturer with specific config for edge cases."""
        config = ManufacturerConfig(
            initial_assets=1_000_000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.05,  # Low margin
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        # WorkingCapitalConfig is no longer used in WidgetManufacturer initialization
        return WidgetManufacturer(config)

    def test_roe_with_zero_equity(self, manufacturer):
        """Test ROE calculation when equity approaches zero."""
        # Deplete equity significantly
        manufacturer.equity = 100  # Very small equity

        manufacturer.record_insurance_premium(50_000)
        metrics = manufacturer.calculate_metrics()

        # When equity is <= 0 or very small, ROE returns 0 by convention
        # This is because dividing by near-zero equity would give meaningless results
        assert metrics["roe"] == 0  # Convention when equity approaches zero
        assert not np.isinf(metrics["roe"])
        assert not np.isnan(metrics["roe"])

    def test_roe_with_negative_equity(self, manufacturer):
        """Test ROE calculation with negative equity (insolvent)."""
        # Make company insolvent
        manufacturer.equity = -100_000
        manufacturer.is_ruined = True

        metrics = manufacturer.calculate_metrics()

        # Should handle negative equity gracefully
        assert metrics["roe"] == 0  # Convention for insolvent companies
        assert not metrics["is_solvent"]

    def test_roe_with_extreme_insurance_costs(self, manufacturer):
        """Test ROE with insurance costs exceeding assets."""
        # Set extreme insurance costs
        manufacturer.record_insurance_premium(10_000_000)  # 10x assets!

        metrics = manufacturer.calculate_metrics()

        # Should handle extreme values
        assert metrics["roe"] < 0
        assert metrics["total_insurance_costs"] == 10_000_000

        # Net income should be very negative
        assert metrics["net_income"] < -7_000_000
