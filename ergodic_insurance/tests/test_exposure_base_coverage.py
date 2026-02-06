"""Additional coverage tests for exposure_base.py.

Targets specific uncovered lines to improve coverage from 83.66% toward 100%.
Focuses on: FinancialStateProvider protocol methods, reset methods, validation
edge cases, negative time errors, zero-base multiplier returns, cubic
interpolation, ScenarioExposure edge paths, StochasticExposure error paths,
jump diffusion details, and get_frequency_multiplier paths.
"""

from decimal import Decimal
from unittest.mock import Mock

import numpy as np
import pytest

from ergodic_insurance.exposure_base import (
    AssetExposure,
    CompositeExposure,
    EmployeeExposure,
    EquityExposure,
    ExposureBase,
    FinancialStateProvider,
    ProductionExposure,
    RevenueExposure,
    ScenarioExposure,
    StochasticExposure,
)

# ===========================================================================
# Helper: mock FinancialStateProvider
# ===========================================================================


def _make_state_provider(
    current_revenue=Decimal("10_000_000"),
    current_assets=Decimal("50_000_000"),
    current_equity=Decimal("20_000_000"),
    base_revenue=Decimal("10_000_000"),
    base_assets=Decimal("50_000_000"),
    base_equity=Decimal("20_000_000"),
):
    """Create a mock FinancialStateProvider with given values."""
    provider = Mock()
    type(provider).current_revenue = property(lambda self: current_revenue)
    type(provider).current_assets = property(lambda self: current_assets)
    type(provider).current_equity = property(lambda self: current_equity)
    type(provider).base_revenue = property(lambda self: base_revenue)
    type(provider).base_assets = property(lambda self: base_assets)
    type(provider).base_equity = property(lambda self: base_equity)
    return provider


# ===========================================================================
# FinancialStateProvider protocol properties (lines 59, 64, 69, 74, 79, 84)
# ===========================================================================


class TestFinancialStateProviderProtocol:
    """Test that the FinancialStateProvider protocol defines expected properties."""

    def test_protocol_properties_exist(self):
        """Lines 59, 64, 69, 74, 79, 84: all protocol properties are defined."""
        # Verify the protocol defines the expected properties
        provider = _make_state_provider()
        assert hasattr(provider, "current_revenue")
        assert hasattr(provider, "current_assets")
        assert hasattr(provider, "current_equity")
        assert hasattr(provider, "base_revenue")
        assert hasattr(provider, "base_assets")
        assert hasattr(provider, "base_equity")


# ===========================================================================
# RevenueExposure: reset (line 186)
# ===========================================================================


class TestRevenueExposureReset:
    """Test RevenueExposure reset method."""

    def test_reset_is_noop(self):
        """Line 186: reset() is a no-op for state-driven exposure."""
        provider = _make_state_provider()
        exposure = RevenueExposure(state_provider=provider)
        # Should not raise, and exposure should still work afterward
        exposure.reset()
        assert exposure.get_exposure(1.0) == 10_000_000


# ===========================================================================
# AssetExposure: current_assets <= 0 (line 229) and reset (line 236)
# ===========================================================================


class TestAssetExposureEdgeCases:
    """Test AssetExposure edge cases."""

    def test_negative_current_assets_returns_zero_multiplier(self):
        """Line 229: negative current_assets returns 0.0 multiplier."""
        provider = _make_state_provider(current_assets=Decimal("-1_000_000"))
        exposure = AssetExposure(state_provider=provider)
        assert exposure.get_frequency_multiplier(1.0) == 0.0

    def test_zero_current_assets_returns_zero_multiplier(self):
        """Line 229: zero current_assets returns 0.0 multiplier."""
        provider = _make_state_provider(current_assets=Decimal("0"))
        exposure = AssetExposure(state_provider=provider)
        assert exposure.get_frequency_multiplier(1.0) == 0.0

    def test_reset_is_noop(self):
        """Line 236: reset() is a no-op."""
        provider = _make_state_provider()
        exposure = AssetExposure(state_provider=provider)
        exposure.reset()
        assert exposure.get_exposure(1.0) == 50_000_000


# ===========================================================================
# EquityExposure: base_equity == 0 (line 273) and reset (line 282)
# ===========================================================================


class TestEquityExposureEdgeCases:
    """Test EquityExposure edge cases."""

    def test_zero_base_equity_returns_zero_multiplier(self):
        """Line 273: zero base_equity returns 0.0 multiplier."""
        provider = _make_state_provider(base_equity=Decimal("0"))
        exposure = EquityExposure(state_provider=provider)
        assert exposure.get_frequency_multiplier(1.0) == 0.0

    def test_negative_current_equity_returns_zero_multiplier(self):
        """Line 275-276: negative current_equity returns 0.0 multiplier."""
        provider = _make_state_provider(current_equity=Decimal("-5_000_000"))
        exposure = EquityExposure(state_provider=provider)
        assert exposure.get_frequency_multiplier(1.0) == 0.0

    def test_reset_is_noop(self):
        """Line 282: reset() is a no-op."""
        provider = _make_state_provider()
        exposure = EquityExposure(state_provider=provider)
        exposure.reset()
        assert exposure.get_exposure(1.0) == 20_000_000


# ===========================================================================
# EmployeeExposure: validation and edge cases (315, 324, 337)
# ===========================================================================


class TestEmployeeExposureEdgeCases:
    """Test EmployeeExposure validation and edge cases."""

    def test_negative_base_employees_raises(self):
        """Line 315: negative base_employees raises ValueError."""
        with pytest.raises(ValueError, match="Base employees must be non-negative"):
            EmployeeExposure(base_employees=-10)

    def test_negative_time_raises(self):
        """Line 324: negative time raises ValueError."""
        exposure = EmployeeExposure(base_employees=100)
        with pytest.raises(ValueError, match="Time must be non-negative"):
            exposure.get_exposure(-1.0)

    def test_reset_is_noop(self):
        """Line 337: reset() is a no-op."""
        exposure = EmployeeExposure(base_employees=100, hiring_rate=0.05)
        exposure.reset()
        # Should still function correctly after reset
        assert exposure.get_exposure(0) == 100.0

    def test_automation_factor_below_zero_raises(self):
        """Line 316-318: automation_factor < 0 raises ValueError."""
        with pytest.raises(ValueError, match="Automation factor must be between 0 and 1"):
            EmployeeExposure(base_employees=100, automation_factor=-0.1)

    def test_automation_factor_above_one_raises(self):
        """Line 316-318: automation_factor > 1 raises ValueError."""
        with pytest.raises(ValueError, match="Automation factor must be between 0 and 1"):
            EmployeeExposure(base_employees=100, automation_factor=1.5)


# ===========================================================================
# ProductionExposure: validation and edge cases (376, 378, 385, 398, 405)
# ===========================================================================


class TestProductionExposureEdgeCases:
    """Test ProductionExposure validation and edge cases."""

    def test_negative_base_units_raises(self):
        """Line 376: negative base_units raises ValueError."""
        with pytest.raises(ValueError, match="Base units must be non-negative"):
            ProductionExposure(base_units=-100)

    def test_negative_quality_improvement_rate_raises(self):
        """Line 378: quality_improvement_rate < 0 raises ValueError."""
        with pytest.raises(ValueError, match="Quality improvement rate must be between"):
            ProductionExposure(base_units=1000, quality_improvement_rate=-0.1)

    def test_quality_improvement_rate_above_one_raises(self):
        """Line 378: quality_improvement_rate > 1 raises ValueError."""
        with pytest.raises(ValueError, match="Quality improvement rate must be between"):
            ProductionExposure(base_units=1000, quality_improvement_rate=1.5)

    def test_negative_time_raises(self):
        """Line 385: negative time raises ValueError."""
        exposure = ProductionExposure(base_units=1000)
        with pytest.raises(ValueError, match="Time must be non-negative"):
            exposure.get_exposure(-1.0)

    def test_zero_base_units_returns_zero_multiplier(self):
        """Line 398: zero base_units returns 0.0 multiplier."""
        exposure = ProductionExposure(base_units=0)
        assert exposure.get_frequency_multiplier(1.0) == 0.0

    def test_reset_is_noop(self):
        """Line 405: reset() is a no-op."""
        exposure = ProductionExposure(base_units=1000, growth_rate=0.1)
        exposure.reset()
        assert exposure.get_exposure(0) == 1000.0


# ===========================================================================
# CompositeExposure: empty weights and negative sum (440, 444)
# ===========================================================================


class TestCompositeExposureEdgeCases:
    """Test CompositeExposure validation edge cases."""

    def test_empty_weights_raises(self):
        """Line 440: empty weights raises ValueError."""
        exposure = EmployeeExposure(base_employees=100)
        with pytest.raises(ValueError, match="Must provide weights"):
            CompositeExposure(exposures={"emp": exposure}, weights={})

    def test_zero_sum_weights_raises(self):
        """Line 444: weights summing to zero raises ValueError."""
        exposure = EmployeeExposure(base_employees=100)
        with pytest.raises(ValueError, match="Sum of weights must be positive"):
            CompositeExposure(
                exposures={"emp": exposure},
                weights={"emp": 0.0},
            )

    def test_negative_sum_weights_raises(self):
        """Line 444: weights summing to negative raises ValueError."""
        exp1 = EmployeeExposure(base_employees=100)
        exp2 = EmployeeExposure(base_employees=200)
        with pytest.raises(ValueError, match="Sum of weights must be positive"):
            CompositeExposure(
                exposures={"a": exp1, "b": exp2},
                weights={"a": -2.0, "b": 1.0},
            )


# ===========================================================================
# ScenarioExposure: invalid interpolation (510), time < 0 (518),
#   cubic interpolation (536, 542-545), base_exposure None/0 (550)
# ===========================================================================


class TestScenarioExposureEdgeCases:
    """Test ScenarioExposure edge case paths."""

    def test_invalid_interpolation_raises(self):
        """Line 510: invalid interpolation method raises ValueError."""
        scenarios = {"test": [100.0, 110.0, 120.0]}
        with pytest.raises(ValueError, match="Interpolation must be"):
            ScenarioExposure(
                scenarios=scenarios,
                selected_scenario="test",
                interpolation="spline",
            )

    def test_negative_time_raises(self):
        """Line 518: negative time raises ValueError."""
        scenarios = {"test": [100.0, 110.0, 120.0]}
        exposure = ScenarioExposure(scenarios=scenarios, selected_scenario="test")
        with pytest.raises(ValueError, match="Time must be non-negative"):
            exposure.get_exposure(-1.0)

    def test_cubic_interpolation(self):
        """Lines 536, 542-545: cubic interpolation falls back to linear."""
        scenarios = {"test": [100.0, 110.0, 120.0, 130.0]}
        exposure = ScenarioExposure(
            scenarios=scenarios,
            selected_scenario="test",
            interpolation="cubic",
        )
        # At time 1.5, cubic (fallback to linear) should be 115
        assert np.isclose(exposure.get_exposure(1.5), 115.0)

    def test_cubic_interpolation_at_half(self):
        """Lines 542-545: cubic interpolation at time 0.5."""
        scenarios = {"test": [100.0, 200.0, 300.0]}
        exposure = ScenarioExposure(
            scenarios=scenarios,
            selected_scenario="test",
            interpolation="cubic",
        )
        # 0.5 interpolation between 100 and 200 = 150
        assert np.isclose(exposure.get_exposure(0.5), 150.0)

    def test_base_exposure_zero_returns_one_multiplier(self):
        """Line 550: base_exposure == 0 returns multiplier 1.0."""
        scenarios = {"test": [0.0, 10.0, 20.0]}
        exposure = ScenarioExposure(scenarios=scenarios, selected_scenario="test")
        # base_exposure is 0.0 (first element), so multiplier should be 1.0
        assert exposure.get_frequency_multiplier(1.0) == 1.0

    def test_base_exposure_none_returns_one_multiplier(self):
        """Line 549: _base_exposure is None returns 1.0."""
        scenarios = {"test": [100.0, 110.0, 120.0]}
        exposure = ScenarioExposure(scenarios=scenarios, selected_scenario="test")
        # Override _base_exposure to None
        exposure._base_exposure = None
        assert exposure.get_frequency_multiplier(1.0) == 1.0

    def test_exposure_beyond_end_returns_last(self):
        """Scenario exposure beyond path length returns last value."""
        scenarios = {"test": [100.0, 110.0]}
        exposure = ScenarioExposure(scenarios=scenarios, selected_scenario="test")
        assert exposure.get_exposure(10.0) == 110.0


# ===========================================================================
# StochasticExposure: validation (599, 613), jump diffusion details
#   (643-644, 661-662, 679-681), get_frequency_multiplier (687-691)
# ===========================================================================


class TestStochasticExposureEdgeCases:
    """Test StochasticExposure validation and edge case paths."""

    def test_negative_base_value_raises(self):
        """Line 599: negative base_value raises ValueError."""
        with pytest.raises(ValueError, match="Base value must be non-negative"):
            StochasticExposure(
                base_value=-100,
                process_type="gbm",
                parameters={},
            )

    def test_negative_time_raises(self):
        """Line 613: negative time raises ValueError."""
        exposure = StochasticExposure(base_value=100, process_type="gbm", parameters={}, seed=42)
        with pytest.raises(ValueError, match="Time must be non-negative"):
            exposure.get_exposure(-1.0)

    def test_jump_diffusion_at_time_zero(self):
        """Lines 660-661: jump diffusion at t=0 returns base_value."""
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
        assert exposure.get_exposure(0) == 100

    def test_mean_reverting_at_time_zero(self):
        """Lines 642-643: mean_reverting at t=0 returns base_value."""
        exposure = StochasticExposure(
            base_value=100,
            process_type="mean_reverting",
            parameters={
                "mean_reversion_speed": 0.5,
                "long_term_mean": 110,
                "volatility": 0.15,
            },
            seed=42,
        )
        assert exposure.get_exposure(0) == 100

    def test_jump_diffusion_with_jumps(self):
        """Lines 679-681: jump diffusion path with actual jumps."""
        # Use parameters that make jumps very likely
        exposure = StochasticExposure(
            base_value=100,
            process_type="jump_diffusion",
            parameters={
                "drift": 0.05,
                "volatility": 0.15,
                "jump_intensity": 10.0,  # High intensity = many jumps
                "jump_mean": 0.0,
                "jump_std": 0.1,
            },
            seed=42,
        )
        value = exposure.get_exposure(5.0)  # Long time period for jump probability
        assert value > 0

    def test_jump_diffusion_no_jumps(self):
        """Lines 678: jump diffusion with no jumps (n_jumps == 0)."""
        # Very low intensity and short period to minimize jump probability
        exposure = StochasticExposure(
            base_value=100,
            process_type="jump_diffusion",
            parameters={
                "drift": 0.05,
                "volatility": 0.15,
                "jump_intensity": 0.001,  # Very low intensity
                "jump_mean": 0.0,
                "jump_std": 0.1,
            },
            seed=42,
        )
        # Very short time to make jumps unlikely
        value = exposure.get_exposure(0.001)
        assert value > 0

    def test_get_frequency_multiplier_zero_base(self):
        """Lines 687-688: base_value == 0 returns 0.0."""
        exposure = StochasticExposure(
            base_value=0,
            process_type="gbm",
            parameters={"drift": 0.05, "volatility": 0.20},
            seed=42,
        )
        assert exposure.get_frequency_multiplier(1.0) == 0.0

    def test_get_frequency_multiplier_positive_base(self):
        """Lines 689-691: positive base_value returns sqrt(current/base)."""
        exposure = StochasticExposure(
            base_value=100,
            process_type="gbm",
            parameters={"drift": 0.05, "volatility": 0.20},
            seed=42,
        )
        mult = exposure.get_frequency_multiplier(1.0)
        assert mult > 0
        # Should be sqrt(current / base)
        current = exposure.get_exposure(1.0)
        expected = float(np.sqrt(current / 100.0))
        assert np.isclose(mult, expected)


# ===========================================================================
# Integration: Composite with multiple state-driven exposures + reset
# ===========================================================================


class TestCompositeExposureIntegration:
    """Integration tests for CompositeExposure with various sub-exposures."""

    def test_composite_reset_propagates(self):
        """Verify reset propagates through all sub-exposures."""
        exp1 = EmployeeExposure(base_employees=100, hiring_rate=0.05)
        exp2 = ProductionExposure(base_units=10_000, growth_rate=0.08)

        composite = CompositeExposure(
            exposures={"emp": exp1, "prod": exp2},
            weights={"emp": 0.5, "prod": 0.5},
        )

        # Get values before and after reset
        val_before = composite.get_frequency_multiplier(1.0)
        composite.reset()
        val_after = composite.get_frequency_multiplier(1.0)

        # Should be the same since these exposures are stateless
        assert val_before == val_after

    def test_composite_weighted_exposure(self):
        """Test weighted exposure calculation."""
        exp1 = EmployeeExposure(base_employees=100, hiring_rate=0.0)
        exp2 = ProductionExposure(base_units=200, growth_rate=0.0)

        composite = CompositeExposure(
            exposures={"emp": exp1, "prod": exp2},
            weights={"emp": 0.6, "prod": 0.4},
        )

        # At time 0, emp=100, prod=200, weighted=0.6*100+0.4*200=60+80=140
        expected = 0.6 * 100 + 0.4 * 200
        assert np.isclose(composite.get_exposure(0), expected)


# ===========================================================================
# ScenarioExposure: nearest interpolation edge + frequency multiplier
# ===========================================================================


class TestScenarioExposureFrequencyMultiplier:
    """Test ScenarioExposure frequency multiplier with various scenarios."""

    def test_multiplier_with_recession_scenario(self):
        """Test multiplier during recession (values decreasing)."""
        scenarios = {"recession": [100.0, 80.0, 70.0, 75.0]}
        exposure = ScenarioExposure(scenarios=scenarios, selected_scenario="recession")
        mult = exposure.get_frequency_multiplier(2.0)
        assert np.isclose(mult, 0.7)  # 70/100

    def test_multiplier_with_growth_scenario(self):
        """Test multiplier during growth (values increasing)."""
        scenarios = {"growth": [50.0, 75.0, 100.0]}
        exposure = ScenarioExposure(scenarios=scenarios, selected_scenario="growth")
        mult = exposure.get_frequency_multiplier(1.0)
        assert np.isclose(mult, 1.5)  # 75/50

    def test_nearest_interpolation_at_exact_point(self):
        """Nearest interpolation at exact integer time."""
        scenarios = {"test": [100.0, 200.0, 300.0]}
        exposure = ScenarioExposure(
            scenarios=scenarios,
            selected_scenario="test",
            interpolation="nearest",
        )
        assert exposure.get_exposure(1.0) == 200.0


# ===========================================================================
# EmployeeExposure: frequency multiplier with zero employees
# ===========================================================================


class TestEmployeeExposureFrequencyMultiplier:
    """Test EmployeeExposure frequency multiplier edge cases."""

    def test_zero_employees_frequency_returns_zero(self):
        """Line 329-330: zero base_employees returns 0.0 multiplier."""
        exposure = EmployeeExposure(base_employees=0)
        assert exposure.get_frequency_multiplier(1.0) == 0.0

    def test_frequency_multiplier_with_automation(self):
        """Test multiplier calculation with automation factor."""
        exposure = EmployeeExposure(
            base_employees=100,
            hiring_rate=0.0,
            automation_factor=0.1,
        )
        # At time 1: employees = 100 (no hiring), automation = 0.9
        # Multiplier = (100/100) * 0.9 = 0.9
        mult = exposure.get_frequency_multiplier(1.0)
        assert np.isclose(mult, 0.9)
