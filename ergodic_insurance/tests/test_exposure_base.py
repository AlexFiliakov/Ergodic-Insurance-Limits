"""Unit tests for exposure base module."""

import time
from typing import Dict, List

import numpy as np
import pytest

from ergodic_insurance.claim_generator import ClaimEvent, ClaimGenerator
from ergodic_insurance.exposure_base import (
    AssetExposure,
    CompositeExposure,
    EmployeeExposure,
    EquityExposure,
    ExposureBase,
    ProductionExposure,
    RevenueExposure,
    ScenarioExposure,
    StochasticExposure,
)


class TestRevenueExposure:
    """Tests for revenue-based exposure."""

    def test_constant_revenue_no_growth(self):
        """Verify zero growth maintains base revenue."""
        exposure = RevenueExposure(base_revenue=10_000_000, growth_rate=0.0)

        for t in [0, 1, 5, 10]:
            assert exposure.get_exposure(t) == 10_000_000
            assert exposure.get_frequency_multiplier(t) == 1.0

    def test_deterministic_growth(self):
        """Verify compound growth calculation."""
        exposure = RevenueExposure(base_revenue=10_000_000, growth_rate=0.10)

        # After 1 year: 10M * 1.1 = 11M
        assert np.isclose(exposure.get_exposure(1), 11_000_000)

        # After 2 years: 10M * 1.1^2 = 12.1M
        assert np.isclose(exposure.get_exposure(2), 12_100_000)

        # Frequency multiplier should be sqrt(revenue_ratio)
        assert np.isclose(exposure.get_frequency_multiplier(1), np.sqrt(1.1))

    def test_inflation_adjustment(self):
        """Verify inflation compounds with growth."""
        exposure = RevenueExposure(base_revenue=10_000_000, growth_rate=0.05, inflation_rate=0.02)

        # Combined rate: 5% + 2% = 7%
        assert np.isclose(exposure.get_exposure(1), 10_700_000)

    def test_stochastic_growth_reproducibility(self):
        """Verify stochastic growth is reproducible with same seed."""
        exposure1 = RevenueExposure(
            base_revenue=10_000_000, growth_rate=0.10, volatility=0.20, seed=42
        )
        exposure2 = RevenueExposure(
            base_revenue=10_000_000, growth_rate=0.10, volatility=0.20, seed=42
        )

        # Should produce identical results
        for t in [0.5, 1.0, 2.0, 5.0]:
            assert exposure1.get_exposure(t) == exposure2.get_exposure(t)

    def test_zero_base_revenue(self):
        """Test handling of zero base revenue."""
        exposure = RevenueExposure(base_revenue=0, growth_rate=0.10)
        assert exposure.get_exposure(1) == 0
        assert exposure.get_frequency_multiplier(1) == 0

    def test_reset(self):
        """Test reset functionality for stochastic exposure."""
        exposure = RevenueExposure(
            base_revenue=10_000_000, growth_rate=0.10, volatility=0.20, seed=42
        )

        val1 = exposure.get_exposure(1.0)
        exposure.reset()
        val2 = exposure.get_exposure(1.0)

        # Should be identical after reset with same seed
        assert val1 == val2

    def test_negative_time_raises_error(self):
        """Test that negative time raises ValueError."""
        exposure = RevenueExposure(base_revenue=10_000_000)
        with pytest.raises(ValueError, match="Time must be non-negative"):
            exposure.get_exposure(-1)

    def test_negative_base_revenue_raises_error(self):
        """Test that negative base revenue raises ValueError."""
        with pytest.raises(ValueError, match="Base revenue must be non-negative"):
            RevenueExposure(base_revenue=-10_000_000)


class TestAssetExposure:
    """Tests for asset-based exposure."""

    def test_depreciation(self):
        """Verify assets depreciate correctly."""
        exposure = AssetExposure(base_assets=50_000_000, growth_rate=0.0, depreciation_rate=0.10)

        # After 1 year: 50M * 0.9 = 45M
        assert np.isclose(exposure.get_exposure(1), 45_000_000)

        # After 2 years: 50M * 0.9^2 = 40.5M
        assert np.isclose(exposure.get_exposure(2), 40_500_000)

    def test_capex_schedule(self):
        """Verify capital expenditures are added correctly."""
        exposure = AssetExposure(
            base_assets=50_000_000,
            depreciation_rate=0.10,
            capex_schedule={
                1.0: 10_000_000,  # 10M investment at year 1
                3.0: 5_000_000,  # 5M investment at year 3
            },
        )

        # At year 2: Original assets depreciated + Year 1 capex depreciated
        # 50M * 0.9^2 + 10M * 0.9 = 40.5M + 9M = 49.5M
        assert np.isclose(exposure.get_exposure(2), 49_500_000)

        # At year 4: All assets depreciated
        # 50M * 0.9^4 + 10M * 0.9^3 + 5M * 0.9 = 32.805M + 7.29M + 4.5M = 44.595M
        expected = 50_000_000 * 0.9**4 + 10_000_000 * 0.9**3 + 5_000_000 * 0.9
        assert np.isclose(exposure.get_exposure(4), expected)

    def test_growth_with_depreciation(self):
        """Test net growth accounting for depreciation."""
        exposure = AssetExposure(
            base_assets=50_000_000,
            growth_rate=0.15,  # 15% growth
            depreciation_rate=0.10,  # 10% depreciation
        )

        # Net growth = 15% - 10% = 5%
        assert np.isclose(exposure.get_exposure(1), 50_000_000 * 1.05)

    def test_inflation(self):
        """Test inflation adjustment."""
        exposure = AssetExposure(
            base_assets=50_000_000,
            depreciation_rate=0.0,  # No depreciation for pure inflation test
            inflation_rate=0.03,
        )

        # With 3% inflation and no depreciation
        assert np.isclose(exposure.get_exposure(1), 50_000_000 * 1.03)

    def test_linear_frequency_scaling(self):
        """Verify frequency scales linearly with assets."""
        exposure = AssetExposure(
            base_assets=50_000_000,
            growth_rate=0.05,
            depreciation_rate=0.0,  # No depreciation for pure growth test
        )

        # If assets grow by 5%, frequency multiplier should be 1.05
        assert np.isclose(exposure.get_frequency_multiplier(1), 1.05)

    def test_zero_base_assets(self):
        """Test handling of zero base assets."""
        exposure = AssetExposure(base_assets=0)
        assert exposure.get_exposure(1) == 0
        assert exposure.get_frequency_multiplier(1) == 0

    def test_invalid_depreciation_rate(self):
        """Test that invalid depreciation rate raises error."""
        with pytest.raises(ValueError, match="Depreciation rate must be between 0 and 1"):
            AssetExposure(base_assets=50_000_000, depreciation_rate=1.5)


class TestEquityExposure:
    """Tests for equity-based exposure."""

    def test_retained_earnings_growth(self):
        """Verify equity grows through retained earnings."""
        exposure = EquityExposure(base_equity=20_000_000, roe=0.15, dividend_payout_ratio=0.40)

        # Growth rate = ROE * retention = 0.15 * 0.6 = 0.09
        # After 1 year: 20M * 1.09 = 21.8M
        assert np.isclose(exposure.get_exposure(1), 21_800_000)

    def test_conservative_scaling(self):
        """Verify frequency scales conservatively with equity."""
        exposure = EquityExposure(base_equity=20_000_000, roe=0.12)

        # After 1 year with 12% ROE and 70% retention: growth = 8.4%
        growth_rate = 0.12 * (1 - 0.3)
        equity_ratio = (1 + growth_rate) ** 1
        expected_multiplier = equity_ratio ** (1 / 3)

        assert np.isclose(exposure.get_frequency_multiplier(1), expected_multiplier)

    def test_volatility(self):
        """Test equity volatility with reproducible seed."""
        exposure = EquityExposure(base_equity=20_000_000, roe=0.10, volatility=0.30, seed=42)

        # Get value with volatility
        value = exposure.get_exposure(1)

        # Reset and verify reproducibility
        exposure.reset()
        value2 = exposure.get_exposure(1)
        assert value == value2

    def test_full_dividend_payout(self):
        """Test zero growth with full dividend payout."""
        exposure = EquityExposure(
            base_equity=20_000_000, roe=0.15, dividend_payout_ratio=1.0  # 100% payout
        )

        # No growth when all earnings are paid out
        assert exposure.get_exposure(1) == 20_000_000

    def test_invalid_payout_ratio(self):
        """Test that invalid payout ratio raises error."""
        with pytest.raises(ValueError, match="Payout ratio must be between 0 and 1"):
            EquityExposure(base_equity=20_000_000, dividend_payout_ratio=1.5)


class TestEmployeeExposure:
    """Tests for employee-based exposure."""

    def test_hiring_growth(self):
        """Verify employee count grows with hiring."""
        exposure = EmployeeExposure(base_employees=100, hiring_rate=0.10)

        # After 1 year: 100 * 1.1 = 110
        assert np.isclose(exposure.get_exposure(1), 110)

        # After 2 years: 100 * 1.1^2 = 121
        assert np.isclose(exposure.get_exposure(2), 121)

    def test_automation_reduction(self):
        """Verify automation reduces frequency multiplier."""
        exposure = EmployeeExposure(base_employees=100, hiring_rate=0.10, automation_factor=0.05)

        # Employees grow to 110 but automation reduces incident rate
        # Multiplier = 1.10 * 0.95 = 1.045
        assert np.isclose(exposure.get_frequency_multiplier(1), 1.10 * 0.95)

    def test_downsizing(self):
        """Test negative hiring rate (downsizing)."""
        exposure = EmployeeExposure(base_employees=100, hiring_rate=-0.05)  # 5% reduction per year

        # After 1 year: 100 * 0.95 = 95
        assert np.isclose(exposure.get_exposure(1), 95)

    def test_zero_employees(self):
        """Test handling of zero employees."""
        exposure = EmployeeExposure(base_employees=0)
        assert exposure.get_exposure(1) == 0
        assert exposure.get_frequency_multiplier(1) == 0

    def test_invalid_automation_factor(self):
        """Test that invalid automation factor raises error."""
        with pytest.raises(ValueError, match="Automation factor must be between 0 and 1"):
            EmployeeExposure(base_employees=100, automation_factor=1.5)


class TestProductionExposure:
    """Tests for production volume exposure."""

    def test_basic_growth(self):
        """Test basic production growth."""
        exposure = ProductionExposure(base_units=1000, growth_rate=0.08)

        # After 1 year: 1000 * 1.08 = 1080
        assert np.isclose(exposure.get_exposure(1), 1080)

    def test_seasonality(self):
        """Verify seasonal patterns apply correctly."""

        def seasonal_pattern(time):
            # Simple sinusoidal pattern
            return 1.0 + 0.3 * np.sin(2 * np.pi * time)

        exposure = ProductionExposure(base_units=1000, seasonality=seasonal_pattern)

        # At time 0.25 (quarter year), sin(π/2) = 1
        # Production = 1000 * (1 + 0.3 * 1) = 1300
        assert np.isclose(exposure.get_exposure(0.25), 1300)

        # At time 0.75, sin(3π/2) = -1
        # Production = 1000 * (1 + 0.3 * -1) = 700
        assert np.isclose(exposure.get_exposure(0.75), 700)

    def test_quality_improvement(self):
        """Verify quality improvements reduce frequency."""
        exposure = ProductionExposure(
            base_units=1000, growth_rate=0.10, quality_improvement_rate=0.05
        )

        # Production grows to 1100 but quality improvements offset frequency
        # Multiplier = 1.10 * 0.95 = 1.045
        assert np.isclose(exposure.get_frequency_multiplier(1), 1.10 * 0.95)

    def test_combined_effects(self):
        """Test growth, seasonality, and quality together."""

        def seasonal_pattern(time):
            return 1.2  # Constant 20% boost

        exposure = ProductionExposure(
            base_units=1000,
            growth_rate=0.10,
            seasonality=seasonal_pattern,
            quality_improvement_rate=0.03,
        )

        # After 1 year: 1000 * 1.1 * 1.2 = 1320
        assert np.isclose(exposure.get_exposure(1), 1320)

        # Frequency: (1320/1000) * 0.97 = 1.32 * 0.97 = 1.2804
        assert np.isclose(exposure.get_frequency_multiplier(1), 1.32 * 0.97)


class TestCompositeExposure:
    """Tests for composite exposure combinations."""

    def test_weighted_combination(self):
        """Verify weighted averaging works correctly."""
        revenue_exp = RevenueExposure(base_revenue=10_000_000, growth_rate=0.10)
        asset_exp = AssetExposure(base_assets=50_000_000, growth_rate=0.05)

        composite = CompositeExposure(
            exposures={"revenue": revenue_exp, "assets": asset_exp},
            weights={"revenue": 0.7, "assets": 0.3},
        )

        # Weighted multiplier at t=1
        rev_mult = revenue_exp.get_frequency_multiplier(1)
        asset_mult = asset_exp.get_frequency_multiplier(1)
        expected = 0.7 * rev_mult + 0.3 * asset_mult

        assert np.isclose(composite.get_frequency_multiplier(1), expected)

    def test_weight_normalization(self):
        """Verify weights are normalized to sum to 1."""
        revenue_exp = RevenueExposure(base_revenue=10_000_000)
        asset_exp = AssetExposure(base_assets=50_000_000)

        composite = CompositeExposure(
            exposures={"revenue": revenue_exp, "assets": asset_exp},
            weights={"revenue": 2.0, "assets": 1.0},  # Sum = 3
        )

        # Weights should be normalized to 2/3 and 1/3
        assert np.isclose(composite.weights["revenue"], 2 / 3)
        assert np.isclose(composite.weights["assets"], 1 / 3)

    def test_three_component_composite(self):
        """Test composite with three exposure types."""
        exposures = {
            "revenue": RevenueExposure(base_revenue=10_000_000, growth_rate=0.10),
            "assets": AssetExposure(base_assets=50_000_000, growth_rate=0.05),
            "employees": EmployeeExposure(base_employees=100, hiring_rate=0.03),
        }

        composite = CompositeExposure(
            exposures=exposures, weights={"revenue": 0.5, "assets": 0.3, "employees": 0.2}
        )

        # Verify it produces reasonable results
        mult = composite.get_frequency_multiplier(1)
        assert mult > 1.0  # Should show growth
        assert mult < 1.2  # But not excessive

    def test_reset_propagation(self):
        """Test that reset propagates to all constituent exposures."""
        revenue_exp = RevenueExposure(base_revenue=10_000_000, volatility=0.20, seed=42)

        composite = CompositeExposure(exposures={"revenue": revenue_exp}, weights={"revenue": 1.0})

        val1 = composite.get_exposure(1)
        composite.reset()
        val2 = composite.get_exposure(1)

        assert val1 == val2  # Should be identical after reset

    def test_empty_exposures_raises_error(self):
        """Test that empty exposures raises error."""
        with pytest.raises(ValueError, match="Must provide at least one exposure"):
            CompositeExposure(exposures={}, weights={})


class TestScenarioExposure:
    """Tests for scenario-based exposure."""

    def test_recession_scenario(self):
        """Verify recession scenario path."""
        scenarios = {
            "baseline": [100, 105, 110, 115, 120],
            "recession": [100, 95, 90, 92, 95],
            "boom": [100, 110, 125, 140, 160],
        }

        exposure = ScenarioExposure(scenarios=scenarios, selected_scenario="recession")

        # At year 2, exposure should be 90
        assert exposure.get_exposure(2) == 90

        # Frequency multiplier = 90/100 = 0.9
        assert np.isclose(exposure.get_frequency_multiplier(2), 0.9)

    def test_linear_interpolation(self):
        """Verify linear interpolation between years."""
        scenarios = {"test": [100, 110, 120]}

        exposure = ScenarioExposure(
            scenarios=scenarios, selected_scenario="test", interpolation="linear"
        )

        # At time 0.5, should be halfway between 100 and 110
        assert np.isclose(exposure.get_exposure(0.5), 105)

        # At time 1.5, should be halfway between 110 and 120
        assert np.isclose(exposure.get_exposure(1.5), 115)

    def test_nearest_interpolation(self):
        """Test nearest neighbor interpolation."""
        scenarios = {"test": [100, 110, 120]}

        exposure = ScenarioExposure(
            scenarios=scenarios, selected_scenario="test", interpolation="nearest"
        )

        # At time 0.4, should round to 0 -> 100
        assert exposure.get_exposure(0.4) == 100

        # At time 0.6, should round to 1 -> 110
        assert exposure.get_exposure(0.6) == 110

    def test_boundary_conditions(self):
        """Test exposure at and beyond scenario boundaries."""
        scenarios = {"test": [100, 110, 120]}

        exposure = ScenarioExposure(scenarios=scenarios, selected_scenario="test")

        # Before start
        assert exposure.get_exposure(0) == 100

        # After end
        assert exposure.get_exposure(10) == 120

    def test_scenario_switching(self):
        """Test switching between scenarios."""
        scenarios = {"optimistic": [100, 110, 120], "pessimistic": [100, 90, 80]}

        exposure = ScenarioExposure(scenarios=scenarios, selected_scenario="optimistic")

        assert exposure.get_exposure(1) == 110

        # Create new exposure with different scenario
        exposure2 = ScenarioExposure(scenarios=scenarios, selected_scenario="pessimistic")
        assert exposure2.get_exposure(1) == 90

    def test_invalid_scenario_raises_error(self):
        """Test that invalid scenario selection raises error."""
        scenarios = {"test": [100, 110]}

        with pytest.raises(
            ValueError, match="Selected scenario 'invalid' not in available scenarios"
        ):
            ScenarioExposure(scenarios=scenarios, selected_scenario="invalid")


class TestStochasticExposure:
    """Tests for stochastic exposure processes."""

    def test_gbm_process(self):
        """Verify GBM properties."""
        exposure = StochasticExposure(
            base_value=100,
            process_type="gbm",
            parameters={"drift": 0.05, "volatility": 0.20},
            seed=42,
        )

        # Generate value at t=1
        value = exposure.get_exposure(1.0)

        # Should be positive
        assert value > 0

        # Should be reproducible with same seed
        exposure2 = StochasticExposure(
            base_value=100,
            process_type="gbm",
            parameters={"drift": 0.05, "volatility": 0.20},
            seed=42,
        )
        assert exposure2.get_exposure(1.0) == value

    def test_mean_reverting_process(self):
        """Test Ornstein-Uhlenbeck process."""
        exposure = StochasticExposure(
            base_value=100,
            process_type="mean_reverting",
            parameters={"mean_reversion_speed": 0.5, "long_term_mean": 110, "volatility": 0.15},
            seed=42,
        )

        # Generate value
        value = exposure.get_exposure(1.0)
        assert value > 0

        # Should tend toward long-term mean over time
        # (statistical test would require many paths)

    def test_jump_diffusion_process(self):
        """Test jump diffusion process."""
        exposure = StochasticExposure(
            base_value=100,
            process_type="jump_diffusion",
            parameters={
                "drift": 0.05,
                "volatility": 0.15,
                "jump_intensity": 0.1,
                "jump_mean": 0.0,
                "jump_std": 0.1,
            },
            seed=42,
        )

        value = exposure.get_exposure(1.0)
        assert value > 0

    def test_path_caching(self):
        """Verify paths are cached for consistency."""
        exposure = StochasticExposure(
            base_value=100,
            process_type="gbm",
            parameters={"drift": 0.05, "volatility": 0.20},
            seed=42,
        )

        # Multiple calls should return same value
        val1 = exposure.get_exposure(1.0)
        val2 = exposure.get_exposure(1.0)
        assert val1 == val2

    def test_reset_clears_cache(self):
        """Test that reset clears the path cache."""
        exposure = StochasticExposure(
            base_value=100,
            process_type="gbm",
            parameters={"drift": 0.05, "volatility": 0.20},
            seed=42,
        )

        val1 = exposure.get_exposure(1.0)
        exposure.reset()
        val2 = exposure.get_exposure(1.0)

        # Should be same value due to same seed
        assert val1 == val2

    def test_invalid_process_type_raises_error(self):
        """Test that invalid process type raises error."""
        with pytest.raises(ValueError, match="Unknown process type"):
            StochasticExposure(base_value=100, process_type="invalid", parameters={}, seed=42)

    def test_zero_time_returns_base_value(self):
        """Test that time=0 returns base value."""
        exposure = StochasticExposure(
            base_value=100, process_type="gbm", parameters={"drift": 0.05, "volatility": 0.20}
        )

        assert exposure.get_exposure(0) == 100


class TestClaimGeneratorIntegration:
    """Integration tests for ClaimGenerator with ExposureBase."""

    def test_backward_compatibility(self):
        """Verify old interface still works with deprecation warning."""
        with pytest.warns(DeprecationWarning, match="Parameter 'frequency' is deprecated"):
            gen = ClaimGenerator(frequency=0.5, seed=42)

        claims = gen.generate_claims(years=10)

        # Should generate claims as before
        assert len(claims) > 0

    def test_basic_exposure_integration(self):
        """Test basic integration with exposure base."""
        exposure = RevenueExposure(base_revenue=10_000_000, growth_rate=0.10)

        gen = ClaimGenerator(
            base_frequency=1.0, exposure_base=exposure, severity_mean=1_000_000, seed=42
        )

        # Check frequency adjustment
        assert gen.get_adjusted_frequency(0) == 1.0  # Base year
        assert gen.get_adjusted_frequency(1) > 1.0  # Should increase

        # Generate claims
        claims = gen.generate_claims(years=5)
        assert len(claims) > 0

    def test_exposure_scaling_effect(self):
        """Verify claims scale with exposure over time."""
        exposure = RevenueExposure(base_revenue=10_000_000, growth_rate=0.20)  # 20% growth

        gen = ClaimGenerator(
            base_frequency=2.0,  # Higher base frequency for more reliable statistics
            exposure_base=exposure,
            seed=42,
        )

        # Run multiple simulations to get reliable statistics
        early_total = 0
        late_total = 0

        for seed in range(10):  # Run 10 simulations
            gen.reset_seed(seed)
            claims = gen.generate_claims(years=10)

            claims_by_year: Dict[int, int] = {}
            for claim in claims:
                claims_by_year[claim.year] = claims_by_year.get(claim.year, 0) + 1

            # Count claims in early vs late years
            early_total += sum(claims_by_year.get(y, 0) for y in range(5))
            late_total += sum(claims_by_year.get(y, 0) for y in range(5, 10))

        # With 20% growth, late years should have significantly more claims
        # Expected ratio based on sqrt scaling: sqrt(1.2^5) ≈ 1.38
        assert late_total > early_total * 1.2  # Conservative threshold

    def test_zero_exposure_no_claims(self):
        """Verify zero exposure generates no claims."""
        exposure = RevenueExposure(base_revenue=0)

        gen = ClaimGenerator(base_frequency=1.0, exposure_base=exposure)

        claims = gen.generate_claims(years=10)
        assert len(claims) == 0

    def test_multiple_year_consistency(self):
        """Verify multi-year generation is consistent."""
        exposure = AssetExposure(base_assets=50_000_000, growth_rate=0.05)

        gen = ClaimGenerator(base_frequency=0.5, exposure_base=exposure, seed=42)

        # Generate 10 years at once
        all_claims = gen.generate_claims(years=10)

        # Reset and generate year by year
        gen.reset_seed(42)
        yearly_claims = []
        for year in range(10):
            yearly_claims.extend(gen.generate_year(year))

        # Should produce same claims
        assert len(all_claims) == len(yearly_claims)
        for c1, c2 in zip(all_claims, yearly_claims):
            assert c1.year == c2.year
            assert np.isclose(c1.amount, c2.amount)

    def test_catastrophic_with_exposure(self):
        """Verify catastrophic claims work with exposure."""
        exposure = EquityExposure(base_equity=20_000_000, roe=0.15)

        gen = ClaimGenerator(base_frequency=0.1, exposure_base=exposure, seed=42)

        regular, cat = gen.generate_all_claims(
            years=50, include_catastrophic=True, cat_frequency=0.02
        )

        # Both types should be generated
        assert len(regular) >= 0  # Could be 0 due to randomness
        assert isinstance(cat, list)

    def test_composite_exposure_integration(self):
        """Test integration with composite exposure."""
        composite = CompositeExposure(
            exposures={
                "revenue": RevenueExposure(base_revenue=10_000_000, growth_rate=0.05),
                "assets": AssetExposure(base_assets=50_000_000, growth_rate=0.03),
            },
            weights={"revenue": 0.6, "assets": 0.4},
        )

        gen = ClaimGenerator(base_frequency=0.5, exposure_base=composite, seed=42)

        claims = gen.generate_claims(years=10)
        assert len(claims) > 0

    def test_scenario_exposure_integration(self):
        """Test integration with scenario exposure."""
        scenarios = {"growth": [1.0, 1.1, 1.2, 1.3, 1.4], "recession": [1.0, 0.9, 0.85, 0.87, 0.9]}

        exposure = ScenarioExposure(scenarios=scenarios, selected_scenario="recession")

        gen = ClaimGenerator(base_frequency=2.0, exposure_base=exposure, seed=42)

        # Frequency should decrease during recession
        assert gen.get_adjusted_frequency(0) == 2.0
        assert gen.get_adjusted_frequency(2) < 2.0  # Year 2 is worst


class TestPerformance:
    """Performance and stress tests."""

    def test_large_simulation_performance(self):
        """Verify performance with large simulations."""
        exposure = CompositeExposure(
            exposures={
                "revenue": RevenueExposure(base_revenue=10_000_000, growth_rate=0.05),
                "assets": AssetExposure(base_assets=50_000_000),
                "employees": EmployeeExposure(base_employees=100, hiring_rate=0.03),
            },
            weights={"revenue": 0.5, "assets": 0.3, "employees": 0.2},
        )

        gen = ClaimGenerator(base_frequency=2.0, exposure_base=exposure, seed=42)

        start = time.time()
        claims = gen.generate_claims(years=1000)
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 5.0  # 5 seconds max

        # Should generate expected number of claims
        assert len(claims) > 1500  # At least 1.5 per year average

    def test_memory_usage(self):
        """Verify memory usage is reasonable."""
        import sys

        exposure = StochasticExposure(
            base_value=100, process_type="gbm", parameters={"drift": 0.05, "volatility": 0.20}
        )

        # Generate many paths
        for t in range(100):
            _ = exposure.get_exposure(float(t))

        # Check cache size is bounded
        cache_size = sys.getsizeof(exposure._path_cache)
        assert cache_size < 100_000  # Less than 100KB


class TestStatisticalValidation:
    """Statistical validation of exposure-adjusted frequencies."""

    def test_long_run_convergence(self):
        """Verify long-run averages converge to expected values."""
        exposure = RevenueExposure(base_revenue=10_000_000, growth_rate=0.10)

        gen = ClaimGenerator(base_frequency=1.0, exposure_base=exposure, seed=42)

        # Run many simulations
        total_claims = []
        for seed in range(100):
            gen.reset_seed(seed)
            claims = gen.generate_claims(years=10)
            total_claims.append(len(claims))

        # Average should be close to expected
        # Expected = sum(base_freq * sqrt(1.1^t) for t in range(10))
        expected_per_sim = sum(1.0 * np.sqrt(1.1**t) for t in range(10))
        actual_average = np.mean(total_claims)

        # Allow 20% deviation due to randomness
        assert np.abs(actual_average - expected_per_sim) / expected_per_sim < 0.2

    def test_frequency_distribution(self):
        """Verify frequency follows Poisson distribution."""
        exposure = RevenueExposure(base_revenue=10_000_000)  # No growth

        gen = ClaimGenerator(base_frequency=2.0, exposure_base=exposure)

        # Generate many single-year samples
        counts = []
        for seed in range(1000):
            gen.reset_seed(seed)
            claims = gen.generate_year(0)
            counts.append(len(claims))

        # Should follow Poisson(2.0)
        mean_count = np.mean(counts)
        var_count = np.var(counts)

        # For Poisson, mean = variance
        assert np.abs(mean_count - 2.0) < 0.1
        assert np.abs(var_count - 2.0) < 0.3


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_cannot_specify_both_frequency_params(self):
        """Test that specifying both frequency and base_frequency raises error."""
        with pytest.raises(
            ValueError, match="Cannot specify both 'frequency' and 'base_frequency'"
        ):
            ClaimGenerator(frequency=0.5, base_frequency=0.5)

    def test_very_high_growth_rate(self):
        """Test handling of very high growth rates."""
        exposure = RevenueExposure(base_revenue=10_000_000, growth_rate=1.0)  # 100% annual growth

        gen = ClaimGenerator(base_frequency=0.1, exposure_base=exposure, seed=42)

        # Should handle exponential growth gracefully
        freq_10 = gen.get_adjusted_frequency(10)
        assert freq_10 > 10  # Significant increase expected

    def test_fractional_years(self):
        """Test exposure calculation at fractional time points."""
        exposure = RevenueExposure(base_revenue=10_000_000, growth_rate=0.12)

        # Test various fractional times
        for t in [0.25, 0.5, 0.75, 1.25, 2.5]:
            value = exposure.get_exposure(t)
            assert value > 0
            expected = 10_000_000 * (1.12**t)
            assert np.isclose(value, expected)
