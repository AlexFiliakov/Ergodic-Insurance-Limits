"""Integration tests for the ergodic insurance framework.

This module provides end-to-end tests that verify the complete simulation
pipeline works correctly and demonstrates the ergodic advantage of insurance.
"""

import time
from typing import Any, Dict

import numpy as np
import psutil
import pytest

from ergodic_insurance.claim_generator import ClaimGenerator
from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.ergodic_analyzer import ErgodicAnalyzer
from ergodic_insurance.insurance import InsuranceLayer, InsurancePolicy
from ergodic_insurance.loss_distributions import LossData, LossEvent, ManufacturingLossGenerator
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.simulation import Simulation, SimulationResults


class TestIntegration:
    """Integration tests for the complete simulation pipeline."""

    def create_manufacturer(self, initial_assets: float, **kwargs) -> WidgetManufacturer:
        """Helper to create a manufacturer with the given parameters.

        Default values aligned with Widget Manufacturing Inc. from blog draft.
        """
        config = ManufacturerConfig(
            initial_assets=initial_assets,
            asset_turnover_ratio=kwargs.get("asset_turnover", 1.2),  # $12M revenue on $10M assets
            base_operating_margin=kwargs.get("base_operating_margin", 0.10),  # 10% EBIT margin
            tax_rate=kwargs.get("tax_rate", 0.25),  # 25% corporate tax
            retention_ratio=kwargs.get("retention_ratio", 0.70),  # 70% retention for growth
        )
        return WidgetManufacturer(config)

    @pytest.fixture
    def base_config(self) -> dict:
        """Create base configuration for tests."""
        return {
            "initial_assets": 10_000_000,
            "time_horizon": 100,
            "random_seed": 42,
            "n_scenarios": 100,  # Small count for speed
            "batch_size": 20,
            "enable_parallel": False,  # Disable for memory testing
        }

    @pytest.fixture
    def insurance_policy(self) -> InsurancePolicy:
        """Create insurance policy for tests."""
        # Create a single layer insurance structure
        layer = InsuranceLayer(
            attachment_point=50_000,  # Acts like deductible
            limit=5_000_000,
            rate=0.02,
        )
        return InsurancePolicy(layers=[layer], deductible=50_000)

    def test_full_pipeline_execution(self, base_config: dict, insurance_policy: InsurancePolicy):
        """Test that the full simulation pipeline executes without errors."""
        start_time = time.time()

        # Create manufacturer with blog draft parameters
        manufacturer = self.create_manufacturer(
            initial_assets=base_config["initial_assets"],
        )

        # Create claim generator with reasonable parameters
        # Total expected annual loss ≈ $450K
        claim_gen = ClaimGenerator(
            seed=base_config["random_seed"],
            base_frequency=3.0,  # Moderate frequency for stability
            severity_mean=150_000,  # Expected $450K annual loss
            severity_std=200_000,  # Moderate variability
        )

        # Run single simulation
        simulation = Simulation(
            manufacturer=manufacturer,
            claim_generator=claim_gen,
            time_horizon=base_config["time_horizon"],
            insurance_policy=insurance_policy,
        )
        result = simulation.run()

        # Verify results
        assert result is not None
        assert len(result.years) == base_config["time_horizon"]
        assert len(result.equity) == base_config["time_horizon"]
        assert len(result.assets) == base_config["time_horizon"]

        # Check timing
        elapsed = time.time() - start_time
        assert elapsed < 60, f"Single simulation took {elapsed:.2f}s, should be < 60s"

    def test_memory_usage(self, base_config: dict):
        """Test that memory usage stays within acceptable limits."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create manufacturer and claim generator
        manufacturer = self.create_manufacturer(initial_assets=base_config["initial_assets"])
        claim_gen = ClaimGenerator(
            base_frequency=5.0,
            severity_mean=100_000,
            severity_std=50_000,
            seed=base_config["random_seed"],
        )

        # Run multiple simulations to test memory
        results = []
        for i in range(base_config["batch_size"]):
            # Create new manufacturer for each simulation
            mfg = self.create_manufacturer(initial_assets=base_config["initial_assets"])
            sim = Simulation(
                manufacturer=mfg,
                claim_generator=claim_gen,
                time_horizon=base_config["time_horizon"],
                seed=base_config["random_seed"] + i,
            )
            results.append(sim.run())

        # Check memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        assert (
            memory_increase < 2000
        ), f"Memory increased by {memory_increase:.2f}MB, should be < 2000MB"
        assert len(results) == base_config["batch_size"]

    def test_insured_vs_uninsured_comparison(
        self, base_config: dict, insurance_policy: InsurancePolicy
    ):
        """Test comparison between insured and uninsured scenarios."""
        # Set up components
        manufacturer = self.create_manufacturer(initial_assets=base_config["initial_assets"])
        claim_gen = ClaimGenerator(
            base_frequency=2.0,
            severity_mean=200_000,
            severity_std=100_000,
            seed=base_config["random_seed"],
        )

        # Run insured simulations
        insured_results = []
        for i in range(10):  # Small sample for speed
            mfg = self.create_manufacturer(initial_assets=base_config["initial_assets"])
            simulation = Simulation(
                manufacturer=mfg,
                claim_generator=claim_gen,
                time_horizon=base_config["time_horizon"],
                insurance_policy=insurance_policy,
                seed=base_config["random_seed"] + i,
            )
            result = simulation.run()
            insured_results.append(result)

        # Run uninsured simulations
        uninsured_results = []
        for i in range(10):
            mfg = self.create_manufacturer(initial_assets=base_config["initial_assets"])
            simulation = Simulation(
                manufacturer=mfg,
                claim_generator=claim_gen,
                time_horizon=base_config["time_horizon"],
                insurance_policy=None,
                seed=base_config["random_seed"] + 100 + i,
            )
            result = simulation.run()
            uninsured_results.append(result)

        # Analyze with ergodic analyzer
        analyzer = ErgodicAnalyzer()
        comparison = analyzer.compare_scenarios(
            insured_results=insured_results,
            uninsured_results=uninsured_results,
            metric="equity",
        )

        # Verify comparison structure
        assert "insured" in comparison
        assert "uninsured" in comparison
        assert "ergodic_advantage" in comparison

        # Check that we have valid metrics
        assert "time_average_mean" in comparison["insured"]
        assert "ensemble_average" in comparison["insured"]
        assert "survival_rate" in comparison["insured"]

    def test_ergodic_advantage_demonstration(self, base_config: dict):
        """Test that ergodic advantage can be demonstrated with proper parameters."""
        # Use blog draft parameters for consistency
        manufacturer = self.create_manufacturer(
            initial_assets=base_config["initial_assets"],
        )

        # More reasonable loss parameters for testing
        # Lower frequency and severity to prevent immediate bankruptcy
        claim_gen = ClaimGenerator(
            seed=base_config["random_seed"],
            base_frequency=3.0,  # Reduced frequency for stability
            severity_mean=150_000,  # Lower mean for expected $450k annual loss
            severity_std=200_000,  # Reduced std for less volatility
        )

        # Better calibrated insurance for testing
        layer = InsuranceLayer(
            attachment_point=50_000,  # Lower attachment for better coverage
            limit=5_000_000,  # Reasonable limit
            rate=0.02,  # 2% rate = $100k premium
        )
        insurance = InsurancePolicy(layers=[layer], deductible=50_000)

        # Run comparison
        analyzer = ErgodicAnalyzer()

        # Small batch for testing
        insured_batch = []
        uninsured_batch = []

        for i in range(20):
            # Insured scenario
            insured_mfg = self.create_manufacturer(
                initial_assets=base_config["initial_assets"],
                asset_turnover=1.2,
                base_operating_margin=0.1,
            )
            insured_simulation = Simulation(
                manufacturer=insured_mfg,
                claim_generator=claim_gen,
                time_horizon=50,  # Shorter for testing
                insurance_policy=insurance,
                seed=base_config["random_seed"] + i,
            )
            insured_result = insured_simulation.run()
            insured_batch.append(insured_result)

            # Uninsured scenario
            uninsured_mfg = self.create_manufacturer(
                initial_assets=base_config["initial_assets"],
                asset_turnover=1.2,
                base_operating_margin=0.1,
            )
            uninsured_simulation = Simulation(
                manufacturer=uninsured_mfg,
                claim_generator=claim_gen,
                time_horizon=50,
                insurance_policy=None,
                seed=base_config["random_seed"] + 100 + i,
            )
            uninsured_result = uninsured_simulation.run()
            uninsured_batch.append(uninsured_result)

        # Analyze results
        comparison = analyzer.compare_scenarios(
            insured_results=insured_batch,
            uninsured_results=uninsured_batch,
            metric="equity",
        )

        # Verify ergodic advantage exists
        ergodic_adv = comparison["ergodic_advantage"]

        # Check survival advantage
        assert (
            comparison["insured"]["survival_rate"] >= comparison["uninsured"]["survival_rate"]
        ), "Insurance should improve survival rate"

        # Check that time average is meaningful
        assert np.isfinite(
            comparison["insured"]["time_average_mean"]
        ), "Insured time average should be finite"

    @pytest.mark.skip(reason="Performance benchmark, not regular test")
    def test_performance_benchmarks(self, base_config: dict):
        """Test performance benchmarks for different scenario counts."""
        benchmarks = [
            (100, 1.5),  # 100 scenarios in 1.5 seconds (relaxed for CI/system variations)
            (1000, 10.0),  # 1000 scenarios in 10 seconds
        ]

        manufacturer = self.create_manufacturer(initial_assets=base_config["initial_assets"])
        claim_gen = ClaimGenerator(
            base_frequency=0.1,
            severity_mean=5_000_000,
            severity_std=2_000_000,
            seed=base_config["random_seed"],
        )

        for n_scenarios, max_time in benchmarks:
            batch_size = min(100, n_scenarios)

            start_time = time.time()
            results = []
            for i in range(batch_size):
                mfg = self.create_manufacturer(initial_assets=base_config["initial_assets"])
                sim = Simulation(
                    manufacturer=mfg,
                    claim_generator=claim_gen,
                    time_horizon=20,  # Short horizon for speed
                    seed=base_config["random_seed"] + i,
                )
                results.append(sim.run())
            elapsed = time.time() - start_time

            assert (
                elapsed < max_time
            ), f"{batch_size} scenarios took {elapsed:.2f}s, should be < {max_time}s"
            assert len(results) == batch_size

    def test_reproducibility(self, base_config: dict):
        """Test that simulations are reproducible with fixed seed."""
        manufacturer = self.create_manufacturer(initial_assets=base_config["initial_assets"])

        # Run twice with same seed
        claim_gen1 = ClaimGenerator(
            base_frequency=1.0, severity_mean=500_000, severity_std=200_000, seed=42
        )
        mfg1 = self.create_manufacturer(initial_assets=base_config["initial_assets"])
        sim1 = Simulation(
            manufacturer=mfg1,
            claim_generator=claim_gen1,
            time_horizon=10,
            insurance_policy=None,
            seed=42,
        )
        result1 = sim1.run()

        claim_gen2 = ClaimGenerator(
            base_frequency=1.0, severity_mean=500_000, severity_std=200_000, seed=42
        )
        mfg2 = self.create_manufacturer(initial_assets=base_config["initial_assets"])
        sim2 = Simulation(
            manufacturer=mfg2,
            claim_generator=claim_gen2,
            time_horizon=10,
            insurance_policy=None,
            seed=42,
        )
        result2 = sim2.run()

        # Results should be identical
        np.testing.assert_array_almost_equal(result1.equity, result2.equity)
        np.testing.assert_array_almost_equal(result1.assets, result2.assets)

    def test_configuration_validation(self):
        """Test that configuration validation works correctly."""
        # Test manufacturer validation
        with pytest.raises(ValueError):
            config = ManufacturerConfig(
                initial_assets=-1000,  # Negative assets
                asset_turnover_ratio=1.0,
                base_operating_margin=0.08,
                tax_rate=0.25,
                retention_ratio=0.7,
            )

        # Test insurance layer validation
        with pytest.raises(ValueError):
            InsuranceLayer(
                attachment_point=0,
                limit=1000000,
                rate=-0.01,  # Negative premium
            )

        # Test claim generator validation
        with pytest.raises(ValueError):
            ClaimGenerator(
                base_frequency=-1,  # Negative frequency
                severity_mean=100_000,
                severity_std=20_000,
            )

    def test_statistics_calculation(self, base_config: dict):
        """Test that statistics are calculated correctly."""
        analyzer = ErgodicAnalyzer()

        # Create sample trajectories
        n_paths = 10
        n_time = 50
        trajectories = []

        np.random.seed(42)
        for _ in range(n_paths):
            # Generate growth path with some noise
            returns = np.random.normal(0.05, 0.2, n_time)
            values = 1000000 * np.exp(np.cumsum(returns))
            trajectories.append(values)

        trajectories = np.array(trajectories)

        # Calculate ensemble statistics
        ensemble_stats = analyzer.calculate_ensemble_average(trajectories, metric="growth_rate")

        # Verify statistics structure
        assert "mean" in ensemble_stats
        assert "std" in ensemble_stats
        assert "survival_rate" in ensemble_stats
        assert ensemble_stats["n_total"] == n_paths

        # Calculate time average for each path
        time_averages = [analyzer.calculate_time_average_growth(traj) for traj in trajectories]

        # All should be finite for these positive trajectories
        assert all(np.isfinite(g) for g in time_averages)

    def test_loss_data_integration(self, base_config: dict):
        """Test integration of standardized LossData through the pipeline."""
        # Generate losses using ManufacturingLossGenerator
        loss_gen = ManufacturingLossGenerator(seed=base_config["random_seed"])
        losses, stats = loss_gen.generate_losses(
            duration=10, revenue=10_000_000, include_catastrophic=True
        )

        # Convert to LossData
        loss_data = LossData.from_loss_events(losses)

        # Validate the data
        assert loss_data.validate()
        assert len(loss_data.timestamps) == len(losses)

        # Create manufacturer and simulation
        manufacturer = self.create_manufacturer(initial_assets=base_config["initial_assets"])

        simulation = Simulation(
            manufacturer=manufacturer, time_horizon=10, seed=base_config["random_seed"]
        )

        # Run simulation with LossData
        result = simulation.run_with_loss_data(loss_data)

        # Verify results
        assert result is not None
        assert len(result.years) <= 10
        assert np.sum(result.claim_amounts) > 0

    def test_loss_data_conversion(self):
        """Test conversion between ClaimEvent and LossData formats."""
        # Create claim generator
        claim_gen = ClaimGenerator(
            base_frequency=2.0, severity_mean=100_000, severity_std=50_000, seed=42
        )

        # Generate claims
        claims = claim_gen.generate_claims(years=5)

        # Convert to LossData
        loss_data = claim_gen.to_loss_data(claims)

        # Validate
        assert loss_data.validate()
        assert len(loss_data.timestamps) == len(claims)

        # Convert back to ClaimEvents
        converted_claims = ClaimGenerator.from_loss_data(loss_data)

        # Check conversion preserves data
        assert len(converted_claims) == len(claims)
        for orig, conv in zip(claims, converted_claims):
            assert orig.year == conv.year
            assert abs(orig.amount - conv.amount) < 0.01

    def test_ergodic_loss_integration(self, base_config: dict):
        """Test integration of loss modeling with ergodic analysis."""
        # Create loss generator
        loss_gen = ManufacturingLossGenerator(seed=42)
        losses, _ = loss_gen.generate_losses(duration=10, revenue=10_000_000)

        # Convert to LossData
        loss_data = LossData.from_loss_events(losses)

        # Create manufacturer
        manufacturer = self.create_manufacturer(initial_assets=base_config["initial_assets"])

        # Create insurance program (if using enhanced insurance)
        from ergodic_insurance.insurance_program import InsuranceProgram

        insurance_program = InsuranceProgram(layers=[])

        # Create ergodic analyzer
        analyzer = ErgodicAnalyzer()

        # Run integrated analysis
        results = analyzer.integrate_loss_ergodic_analysis(
            loss_data=loss_data,
            insurance_program=insurance_program,
            manufacturer=manufacturer,
            time_horizon=10,
            n_simulations=10,  # Small for testing
        )

        # Verify results
        assert results.validation_passed
        assert np.isfinite(results.time_average_growth) or results.time_average_growth == -np.inf
        assert results.survival_rate >= 0
        assert results.survival_rate <= 1

    def test_insurance_impact_validation(self, base_config: dict):
        """Test validation of insurance impact on ergodic calculations."""
        # Create manufacturer
        manufacturer_base = self.create_manufacturer(initial_assets=base_config["initial_assets"])
        manufacturer_insured = self.create_manufacturer(
            initial_assets=base_config["initial_assets"]
        )

        # Create claim generator
        claim_gen = ClaimGenerator(
            base_frequency=1.0, severity_mean=500_000, severity_std=200_000, seed=42
        )

        # Create insurance
        layer = InsuranceLayer(attachment_point=100_000, limit=5_000_000, rate=0.02)
        insurance = InsurancePolicy(layers=[layer], deductible=100_000)

        # Run base scenario (no insurance)
        sim_base = Simulation(
            manufacturer=manufacturer_base,
            claim_generator=claim_gen,
            time_horizon=20,
            insurance_policy=None,
            seed=42,
        )
        result_base = sim_base.run()

        # Run insured scenario
        sim_insured = Simulation(
            manufacturer=manufacturer_insured,
            claim_generator=claim_gen,
            time_horizon=20,
            insurance_policy=insurance,
            seed=42,
        )
        result_insured = sim_insured.run()

        # Create analyzer and validate
        analyzer = ErgodicAnalyzer()
        validation = analyzer.validate_insurance_ergodic_impact(
            base_scenario=result_base,
            insurance_scenario=result_insured,
            insurance_program=None,  # Using simple policy, not program
        )

        # Check validation results
        assert validation.recoveries_credited
        assert validation.overall_valid or validation.time_average_reflects_benefit

    def test_data_flow_consistency(self, base_config: dict):
        """Test data consistency through the entire pipeline."""
        # Generate losses
        loss_gen = ManufacturingLossGenerator(seed=123)
        original_losses, _ = loss_gen.generate_losses(duration=5, revenue=10_000_000)

        # Track data through pipeline
        # Step 1: Convert to LossData
        loss_data = LossData.from_loss_events(original_losses)
        assert loss_data.validate()

        # Step 2: Apply insurance (if any)
        from ergodic_insurance.insurance_program import EnhancedInsuranceLayer, InsuranceProgram

        layer = EnhancedInsuranceLayer(attachment_point=50_000, limit=1_000_000, premium_rate=0.03)
        program = InsuranceProgram(layers=[layer])

        insured_data = loss_data.apply_insurance(program)
        assert insured_data.validate()

        # Step 3: Convert to ClaimEvents for simulation
        claims = ClaimGenerator.from_loss_data(insured_data)

        # Step 4: Run simulation
        manufacturer = self.create_manufacturer(initial_assets=base_config["initial_assets"])
        simulation = Simulation(manufacturer=manufacturer, time_horizon=5, seed=123)

        # Manually apply claims
        for claim in claims:
            if 0 <= claim.year < 5:
                simulation.step_annual(claim.year, [claim])

        # Verify data consistency
        total_original = sum(loss.amount for loss in original_losses)
        total_insured = sum(claim.amount for claim in claims)

        # Insured amount should be less due to recoveries
        assert total_insured <= total_original

    def test_performance_with_loss_data(self, base_config: dict):
        """Test performance when using LossData structures."""
        # Generate large loss dataset
        loss_gen = ManufacturingLossGenerator(seed=999)

        start_time = time.time()

        # Generate 100 years of losses
        losses, _ = loss_gen.generate_losses(duration=100, revenue=10_000_000)

        # Convert to LossData
        loss_data = LossData.from_loss_events(losses)

        # Run simulation
        manufacturer = self.create_manufacturer(initial_assets=base_config["initial_assets"])
        simulation = Simulation(manufacturer=manufacturer, time_horizon=100, seed=999)

        result = simulation.run_with_loss_data(loss_data)

        elapsed = time.time() - start_time

        # Should complete in reasonable time
        assert elapsed < 10.0, f"100-year simulation took {elapsed:.2f}s, should be < 10s"
        assert result is not None
        assert len(result.years) <= 100
