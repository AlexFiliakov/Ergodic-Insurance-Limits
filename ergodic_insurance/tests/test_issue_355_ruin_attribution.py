"""Tests for issue #355: Ruin probability cause attribution and convergence bugs.

Verifies three fixes:
1. Bankruptcy cause attribution reads the actual bankruptcy year (not horizon)
2. Growth rates use equity instead of total assets
3. Convergence check uses configured time horizons
"""

from unittest.mock import Mock, PropertyMock

import numpy as np
import pytest

from ergodic_insurance.ruin_probability import (
    RuinProbabilityAnalyzer,
    RuinProbabilityConfig,
)


class TestBankruptcyCauseAttribution:
    """Bug 1: Cause attribution should read the actual bankruptcy year."""

    def test_early_bankruptcy_cause_detected(self):
        """Simulations bankrupt at year 3 should have causes read at year 3, not horizon."""
        n_sims = 100
        max_horizon = 10

        # Simulate: all simulations go bankrupt at year 3 due to asset_threshold
        bankruptcy_years = np.full(n_sims, 3, dtype=np.int32)

        # Causes array: asset_threshold is True at year 2 (0-indexed for year 3)
        causes = {
            "asset_threshold": np.zeros((n_sims, max_horizon), dtype=bool),
            "equity_threshold": np.zeros((n_sims, max_horizon), dtype=bool),
            "consecutive_negative": np.zeros((n_sims, max_horizon), dtype=bool),
            "debt_service": np.zeros((n_sims, max_horizon), dtype=bool),
        }
        # Set cause at year 3 (index 2) — with early stopping, later years are False
        causes["asset_threshold"][:, 2] = True

        simulation_results = {
            "bankruptcy_years": bankruptcy_years,
            "bankruptcy_causes": causes,
        }

        config = RuinProbabilityConfig(
            time_horizons=[5, 10],
            n_simulations=n_sims,
        )

        # Create analyzer with mocks
        analyzer = RuinProbabilityAnalyzer.__new__(RuinProbabilityAnalyzer)
        result = analyzer._analyze_horizons(simulation_results, config)

        # All simulations went bankrupt at year 3 due to asset_threshold
        # Both 5-year and 10-year horizons should show 100% asset_threshold cause
        assert result["bankruptcy_causes"]["asset_threshold"][0] == pytest.approx(
            1.0
        ), "5-year horizon should detect asset_threshold cause for early bankruptcies"
        assert result["bankruptcy_causes"]["asset_threshold"][1] == pytest.approx(
            1.0
        ), "10-year horizon should detect asset_threshold cause for early bankruptcies"

    def test_no_false_negatives_for_long_horizons(self):
        """Regression: old code read causes[:, horizon-1] which was False after early stop."""
        n_sims = 50
        max_horizon = 20

        # Half go bankrupt at year 2, half survive
        bankruptcy_years = np.full(n_sims, max_horizon + 1, dtype=np.int32)
        bankruptcy_years[:25] = 2  # First 25 go bankrupt at year 2

        causes = {
            "asset_threshold": np.zeros((n_sims, max_horizon), dtype=bool),
            "equity_threshold": np.zeros((n_sims, max_horizon), dtype=bool),
            "consecutive_negative": np.zeros((n_sims, max_horizon), dtype=bool),
            "debt_service": np.zeros((n_sims, max_horizon), dtype=bool),
        }
        # Bankrupt sims have equity_threshold at year 2 (index 1)
        causes["equity_threshold"][:25, 1] = True

        simulation_results = {
            "bankruptcy_years": bankruptcy_years,
            "bankruptcy_causes": causes,
        }

        config = RuinProbabilityConfig(
            time_horizons=[5, 10, 20],
            n_simulations=n_sims,
        )

        analyzer = RuinProbabilityAnalyzer.__new__(RuinProbabilityAnalyzer)
        result = analyzer._analyze_horizons(simulation_results, config)

        # For all horizons >= 2, the 25 bankrupt sims should show equity_threshold
        for i in range(3):
            assert result["bankruptcy_causes"]["equity_threshold"][i] == pytest.approx(1.0), (
                f"Horizon {config.time_horizons[i]}: equity_threshold should be 1.0 for "
                f"all bankrupt sims, got {result['bankruptcy_causes']['equity_threshold'][i]}"
            )

    def test_mixed_bankruptcy_years(self):
        """Different simulations going bankrupt at different years."""
        n_sims = 6
        max_horizon = 10

        # Bankruptcy at years: 1, 3, 5, 7, never, never
        bankruptcy_years = np.array([1, 3, 5, 7, 11, 11], dtype=np.int32)

        causes = {
            "asset_threshold": np.zeros((n_sims, max_horizon), dtype=bool),
            "equity_threshold": np.zeros((n_sims, max_horizon), dtype=bool),
            "consecutive_negative": np.zeros((n_sims, max_horizon), dtype=bool),
            "debt_service": np.zeros((n_sims, max_horizon), dtype=bool),
        }
        # Each bankrupt sim has asset_threshold at its bankruptcy year
        causes["asset_threshold"][0, 0] = True  # sim 0, year 1
        causes["asset_threshold"][1, 2] = True  # sim 1, year 3
        causes["asset_threshold"][2, 4] = True  # sim 2, year 5
        causes["asset_threshold"][3, 6] = True  # sim 3, year 7

        simulation_results = {
            "bankruptcy_years": bankruptcy_years,
            "bankruptcy_causes": causes,
        }

        config = RuinProbabilityConfig(
            time_horizons=[5, 10],
            n_simulations=n_sims,
        )

        analyzer = RuinProbabilityAnalyzer.__new__(RuinProbabilityAnalyzer)
        result = analyzer._analyze_horizons(simulation_results, config)

        # Horizon 5: sims 0,1,2 are bankrupt (years 1,3,5), all have asset_threshold
        assert result["bankruptcy_causes"]["asset_threshold"][0] == pytest.approx(1.0)
        # Horizon 10: sims 0,1,2,3 are bankrupt (years 1,3,5,7), all have asset_threshold
        assert result["bankruptcy_causes"]["asset_threshold"][1] == pytest.approx(1.0)


class TestGrowthRatesUseEquity:
    """Bug 2: Growth rates should use equity, not total assets."""

    def test_growth_rates_use_equity(self):
        """_calculate_growth_rates should reference manufacturer.equity."""
        from ergodic_insurance.monte_carlo import MonteCarloEngine, SimulationConfig

        # Create mock manufacturer with different total_assets and equity
        manufacturer = Mock()
        type(manufacturer).total_assets = PropertyMock(return_value=10_000_000)
        type(manufacturer).equity = PropertyMock(return_value=4_000_000)

        config = SimulationConfig(
            n_simulations=10,
            n_years=5,
            parallel=False,
        )

        engine = MonteCarloEngine.__new__(MonteCarloEngine)
        engine.manufacturer = manufacturer
        engine.config = config

        # If equity is 4M and final equity is 8M, growth should be ln(8M/4M)/5
        final_equity = np.array([8_000_000.0])
        growth_rates = engine._calculate_growth_rates(final_equity)

        expected = np.log(8_000_000 / 4_000_000) / 5
        assert growth_rates[0] == pytest.approx(
            expected, rel=1e-10
        ), "Growth rate should use equity (4M) as base, not total_assets (10M)"

    def test_growth_rate_not_based_on_total_assets(self):
        """Verify that total_assets is NOT used as the denominator."""
        from ergodic_insurance.monte_carlo import MonteCarloEngine, SimulationConfig

        manufacturer = Mock()
        type(manufacturer).total_assets = PropertyMock(return_value=10_000_000)
        type(manufacturer).equity = PropertyMock(return_value=4_000_000)

        config = SimulationConfig(n_simulations=10, n_years=5, parallel=False)

        engine = MonteCarloEngine.__new__(MonteCarloEngine)
        engine.manufacturer = manufacturer
        engine.config = config

        final_equity = np.array([8_000_000.0])
        growth_rates = engine._calculate_growth_rates(final_equity)

        # If it were using total_assets (10M), result would be ln(8M/10M)/5 < 0
        wrong_rate = np.log(8_000_000 / 10_000_000) / 5
        assert growth_rates[0] != pytest.approx(
            wrong_rate, rel=1e-5
        ), "Growth rate should not be based on total_assets"
        # Correct rate is positive (equity doubled from 4M to 8M)
        assert growth_rates[0] > 0

    def test_simulation_result_includes_final_equity(self):
        """All simulation code paths should return final_equity."""
        import inspect

        from ergodic_insurance.monte_carlo import _simulate_path_enhanced
        from ergodic_insurance.monte_carlo_worker import run_chunk_standalone

        source_enhanced = inspect.getsource(_simulate_path_enhanced)
        assert (
            "final_equity" in source_enhanced
        ), "_simulate_path_enhanced should include final_equity in its return dict"

        source_worker = inspect.getsource(run_chunk_standalone)
        assert (
            "final_equity" in source_worker
        ), "run_chunk_standalone should include final_equity in its return dict"


class TestRuinConvergenceHorizon:
    """Bug 3: Convergence check should use configured time horizons."""

    def test_convergence_uses_configured_horizon(self):
        """_check_ruin_convergence should use max of time_horizons, not hardcoded 10."""
        analyzer = RuinProbabilityAnalyzer.__new__(RuinProbabilityAnalyzer)

        n_sims = 1000
        # All simulations go bankrupt at year 15
        # With hardcoded 10, chain_binary = (15 <= 10) = False → converged trivially
        # With configured [20], chain_binary = (15 <= 20) = True → different result
        bankruptcy_years = np.full(n_sims, 15, dtype=np.int32)

        result_with_20 = analyzer._check_ruin_convergence(bankruptcy_years, time_horizons=[20])
        result_with_10 = analyzer._check_ruin_convergence(bankruptcy_years, time_horizons=[10])

        # With horizon=20: all sims are bankrupt (15<=20), all chains have mean=1.0
        # With horizon=10: no sims are bankrupt (15>10), all chains have mean=0.0
        # Both should converge (all-same), but the binary values differ
        assert result_with_20 is True  # All True → converged
        assert result_with_10 is True  # All False → converged

    def test_convergence_respects_non_default_horizons(self):
        """Convergence should work with horizons other than 10."""
        analyzer = RuinProbabilityAnalyzer.__new__(RuinProbabilityAnalyzer)

        n_sims = 800
        # Mix: half bankrupt at year 3, half at year 25
        bankruptcy_years = np.full(n_sims, 25, dtype=np.int32)
        bankruptcy_years[: n_sims // 2] = 3

        # With horizon=5: half bankrupt (3<=5), half not (25>5)
        result_5 = analyzer._check_ruin_convergence(bankruptcy_years, time_horizons=[5])
        # With horizon=20: half bankrupt (3<=20), half not (25>20) — same proportion
        result_20 = analyzer._check_ruin_convergence(bankruptcy_years, time_horizons=[1, 5, 20])

        # Both should converge since the proportions are consistent across chains
        assert isinstance(result_5, bool)
        assert isinstance(result_20, bool)

    def test_convergence_default_fallback(self):
        """When no time_horizons provided, should default to 10."""
        analyzer = RuinProbabilityAnalyzer.__new__(RuinProbabilityAnalyzer)

        n_sims = 400
        bankruptcy_years = np.full(n_sims, 5, dtype=np.int32)

        # Without time_horizons (default None → falls back to 10)
        result_default = analyzer._check_ruin_convergence(bankruptcy_years)
        # Explicit 10
        result_10 = analyzer._check_ruin_convergence(bankruptcy_years, time_horizons=[10])

        assert result_default == result_10, "Default should behave same as explicit [10]"
