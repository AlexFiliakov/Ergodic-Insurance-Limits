"""Coverage tests for insurance_pricing.py targeting specific uncovered lines.

Missing lines: 434, 453, 482-484, 503, 602, 648, 653-658, 690-691
"""

import copy
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ergodic_insurance.insurance_pricing import (
    InsurancePricer,
    LayerPricing,
    MarketCycle,
    PricingParameters,
)


def _make_loss_generator(seed=42):
    """Create a real ManufacturingLossGenerator for pricing tests."""
    from ergodic_insurance.loss_distributions import ManufacturingLossGenerator

    return ManufacturingLossGenerator.create_simple(
        frequency=0.5, severity_mean=500_000, severity_std=200_000, seed=seed
    )


def _make_insurance_program():
    """Create a minimal InsuranceProgram for pricing tests."""
    from ergodic_insurance.insurance_program import EnhancedInsuranceLayer, InsuranceProgram

    layer = EnhancedInsuranceLayer(
        attachment_point=100_000,
        limit=5_000_000,
        base_premium_rate=0.03,
    )
    program = InsuranceProgram(layers=[layer], deductible=100_000)
    return program


def _make_insurance_policy():
    """Create a minimal InsurancePolicy for pricing tests."""
    import warnings

    from ergodic_insurance.insurance import InsuranceLayer, InsurancePolicy

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return InsurancePolicy(
            layers=[
                InsuranceLayer(attachment_point=100_000, limit=5_000_000, rate=0.03),
            ],
            deductible=100_000,
        )


class TestPriceInsuranceProgramNoRevenue:
    """Tests for price_insurance_program ValueError when no revenue (line 434)."""

    def test_no_revenue_and_no_exposure_raises(self):
        """Line 434: Neither expected_revenue nor exposure provided raises ValueError."""
        loss_gen = _make_loss_generator()
        pricer = InsurancePricer(
            loss_generator=loss_gen,
            market_cycle=MarketCycle.NORMAL,
            seed=42,
        )
        program = _make_insurance_program()

        with pytest.raises(ValueError, match="Either expected_revenue or exposure"):
            pricer.price_insurance_program(
                program=program,
                expected_revenue=None,
            )


class TestPriceInsuranceProgramStoreResults:
    """Tests for price_insurance_program storing pricing_results (line 453)."""

    def test_pricing_results_stored_on_program(self):
        """Line 453: pricing_results attribute is set on the program."""
        loss_gen = _make_loss_generator()
        pricer = InsurancePricer(
            loss_generator=loss_gen,
            market_cycle=MarketCycle.NORMAL,
            seed=42,
        )
        program = _make_insurance_program()

        priced = pricer.price_insurance_program(
            program=program,
            expected_revenue=10_000_000,
            update_program=True,
        )
        assert hasattr(priced, "pricing_results")
        assert len(priced.pricing_results) == 1
        assert isinstance(priced.pricing_results[0], LayerPricing)


class TestPriceInsurancePolicyCopy:
    """Tests for price_insurance_policy copy branch (lines 482-484)."""

    def test_policy_not_updated_when_copy(self):
        """Lines 482-484: When update_policy=False, original policy is unchanged."""
        loss_gen = _make_loss_generator()
        pricer = InsurancePricer(
            loss_generator=loss_gen,
            market_cycle=MarketCycle.NORMAL,
            seed=42,
        )
        policy = _make_insurance_policy()
        original_rate = policy.layers[0].rate

        with pytest.warns(DeprecationWarning, match="price_insurance_policy.*deprecated"):
            priced_copy = pricer.price_insurance_policy(
                policy=policy,
                expected_revenue=10_000_000,
                update_policy=False,
            )

        # Original should be unchanged
        assert policy.layers[0].rate == original_rate
        # Copy should have updated pricing
        assert hasattr(priced_copy, "pricing_results")
        assert priced_copy is not policy

    def test_policy_updated_in_place(self):
        """When update_policy=True, the original policy is modified."""
        loss_gen = _make_loss_generator()
        pricer = InsurancePricer(
            loss_generator=loss_gen,
            market_cycle=MarketCycle.NORMAL,
            seed=42,
        )
        policy = _make_insurance_policy()

        with pytest.warns(DeprecationWarning, match="price_insurance_policy.*deprecated"):
            priced = pricer.price_insurance_policy(
                policy=policy,
                expected_revenue=10_000_000,
                update_policy=True,
            )
        # Should be the same object
        assert priced is policy


class TestPriceInsurancePolicyStoreResults:
    """Tests for price_insurance_policy storing pricing_results (line 503)."""

    def test_pricing_results_stored_on_policy(self):
        """Line 503: pricing_results attribute is set on the policy."""
        loss_gen = _make_loss_generator()
        pricer = InsurancePricer(
            loss_generator=loss_gen,
            market_cycle=MarketCycle.NORMAL,
            seed=42,
        )
        policy = _make_insurance_policy()

        with pytest.warns(DeprecationWarning, match="price_insurance_policy.*deprecated"):
            pricer.price_insurance_policy(
                policy=policy,
                expected_revenue=10_000_000,
            )
        assert hasattr(policy, "pricing_results")
        assert len(policy.pricing_results) == 1
        assert isinstance(policy.pricing_results[0], LayerPricing)


class TestSimulateCycleTransitionFallback:
    """Tests for simulate_cycle_transition fallback premium (line 602)."""

    def test_fallback_premium_calculation(self):
        """Line 602: When pricing_results missing, use calculate_annual_premium."""
        loss_gen = _make_loss_generator()
        pricer = InsurancePricer(
            loss_generator=loss_gen,
            market_cycle=MarketCycle.NORMAL,
            seed=42,
        )
        program = _make_insurance_program()

        # Mock price_insurance_program to return a program without pricing_results
        def mock_price(program, expected_revenue, market_cycle, update_program):
            """Return program with no pricing_results to trigger fallback."""
            result_program = copy.deepcopy(program)
            # Ensure pricing_results is empty to trigger the fallback path
            result_program.pricing_results = []
            return result_program

        with patch.object(pricer, "price_insurance_program", side_effect=mock_price):
            results = pricer.simulate_cycle_transition(
                program=program,
                expected_revenue=10_000_000,
                years=2,
            )
        assert len(results) == 2
        for r in results:
            assert "total_premium" in r


class TestTransitionCycleNormalAndSoft:
    """Tests for _transition_cycle NORMAL and SOFT transitions (lines 648, 653-658)."""

    @staticmethod
    def _make_pricer_with_fixed_rng(value):
        """Create pricer with a mock RNG that returns a fixed value."""
        pricer = InsurancePricer(market_cycle=MarketCycle.NORMAL, seed=42)
        mock_rng = MagicMock()
        mock_rng.random.return_value = value
        pricer.rng = mock_rng
        return pricer

    def test_normal_to_soft_transition(self):
        """Line 648: NORMAL market transitions to SOFT."""
        # normal_to_hard=0.2, normal_to_soft=0.2, so SOFT when 0.2 <= rand < 0.4
        pricer = self._make_pricer_with_fixed_rng(0.25)
        result = pricer._transition_cycle(
            MarketCycle.NORMAL,
            {"normal_to_hard": 0.2, "normal_to_soft": 0.2},
        )
        assert result == MarketCycle.SOFT

    def test_normal_stays_normal(self):
        """NORMAL market stays NORMAL when rand >= normal_to_hard + normal_to_soft."""
        pricer = self._make_pricer_with_fixed_rng(0.9)
        result = pricer._transition_cycle(
            MarketCycle.NORMAL,
            {"normal_to_hard": 0.2, "normal_to_soft": 0.2},
        )
        assert result == MarketCycle.NORMAL

    def test_soft_to_normal_transition(self):
        """Line 653: SOFT market transitions to NORMAL."""
        pricer = self._make_pricer_with_fixed_rng(0.15)
        result = pricer._transition_cycle(
            MarketCycle.SOFT,
            {"soft_to_normal": 0.3, "soft_to_hard": 0.1},
        )
        assert result == MarketCycle.NORMAL

    def test_soft_to_hard_transition(self):
        """Lines 655-656: SOFT market transitions to HARD."""
        # soft_to_normal=0.3, soft_to_hard=0.1, HARD when 0.3 <= rand < 0.4
        pricer = self._make_pricer_with_fixed_rng(0.35)
        result = pricer._transition_cycle(
            MarketCycle.SOFT,
            {"soft_to_normal": 0.3, "soft_to_hard": 0.1},
        )
        assert result == MarketCycle.HARD

    def test_soft_stays_soft(self):
        """Lines 657-658: SOFT market stays SOFT."""
        pricer = self._make_pricer_with_fixed_rng(0.95)
        result = pricer._transition_cycle(
            MarketCycle.SOFT,
            {"soft_to_normal": 0.3, "soft_to_hard": 0.1},
        )
        assert result == MarketCycle.SOFT


class TestCreateFromConfigInvalidCycle:
    """Tests for create_from_config invalid market cycle fallback (lines 690-691)."""

    def test_invalid_market_cycle_defaults_to_normal(self):
        """Lines 690-691: Invalid market_cycle string falls back to NORMAL."""
        config = {
            "market_cycle": "INVALID_CYCLE",
            "loss_ratio": 0.70,
        }
        pricer = InsurancePricer.create_from_config(config)
        assert pricer.market_cycle == MarketCycle.NORMAL

    def test_valid_market_cycle_from_config(self):
        """Valid market_cycle string is properly parsed."""
        config = {"market_cycle": "HARD"}
        pricer = InsurancePricer.create_from_config(config)
        assert pricer.market_cycle == MarketCycle.HARD

    def test_lowercase_market_cycle_from_config(self):
        """Lowercase market_cycle string is properly uppercased."""
        config = {"market_cycle": "soft"}
        pricer = InsurancePricer.create_from_config(config)
        assert pricer.market_cycle == MarketCycle.SOFT

    def test_config_with_all_parameters(self):
        """Full config with all parameters creates proper pricer."""
        loss_gen = _make_loss_generator()
        config = {
            "loss_ratio": 0.65,
            "expense_ratio": 0.30,
            "profit_margin": 0.08,
            "risk_loading": 0.15,
            "confidence_level": 0.99,
            "simulation_years": 20,
            "min_premium": 5000.0,
            "max_rate_on_line": 0.40,
            "market_cycle": "HARD",
            "seed": 123,
        }
        pricer = InsurancePricer.create_from_config(config, loss_generator=loss_gen)
        assert pricer.parameters.expense_ratio == 0.30
        assert pricer.parameters.profit_margin == 0.08
        assert pricer.market_cycle == MarketCycle.HARD
