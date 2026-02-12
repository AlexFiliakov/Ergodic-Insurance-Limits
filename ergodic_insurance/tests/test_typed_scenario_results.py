"""Tests for typed return types from compare_scenarios() and analyze_simulation_batch().

Validates the new ScenarioComparison and BatchAnalysisResults dataclasses
introduced in #713, including typed attribute access, backward-compatible
dict-style access with deprecation warnings, and correct data propagation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from ergodic_insurance.ergodic_analyzer import ErgodicAnalyzer
from ergodic_insurance.ergodic_types import (
    BatchAnalysisResults,
    ConvergenceStats,
    EnsembleAverageStats,
    ErgodicAdvantage,
    ScenarioComparison,
    ScenarioMetrics,
    SurvivalAnalysisStats,
    TimeAverageStats,
)
from ergodic_insurance.simulation import SimulationResults

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def analyzer():
    return ErgodicAnalyzer(convergence_threshold=0.01)


def _make_sim_result(
    n_years=20,
    equity_start=1_000_000,
    growth=0.05,
    insolvency_year=None,
):
    """Build a deterministic SimulationResults."""
    rng = np.random.default_rng(42)
    years = np.arange(n_years)
    equity = equity_start * np.exp(growth * years)
    assets = equity * 1.2
    roe = np.full(n_years, growth)
    revenue = np.full(n_years, 500_000.0)
    net_income = np.full(n_years, 50_000.0)
    claim_counts = np.ones(n_years, dtype=int)
    claim_amounts = rng.lognormal(10, 2, n_years)

    if insolvency_year is not None and insolvency_year < n_years:
        equity[insolvency_year:] = 0.0
        assets[insolvency_year:] = 0.0

    return SimulationResults(
        years=years,
        assets=assets,
        equity=equity,
        roe=roe,
        revenue=revenue,
        net_income=net_income,
        claim_counts=claim_counts,
        claim_amounts=claim_amounts,
        insolvency_year=insolvency_year,
    )


# ===================================================================
# ScenarioComparison return type
# ===================================================================


class TestCompareScenarioReturnType:
    """compare_scenarios() returns ScenarioComparison dataclass."""

    def test_returns_scenario_comparison(self, analyzer):
        insured = [_make_sim_result(growth=0.05) for _ in range(5)]
        uninsured = [_make_sim_result(growth=0.03) for _ in range(5)]

        result = analyzer.compare_scenarios(insured, uninsured)

        assert isinstance(result, ScenarioComparison)
        assert isinstance(result.insured, ScenarioMetrics)
        assert isinstance(result.uninsured, ScenarioMetrics)
        assert isinstance(result.ergodic_advantage, ErgodicAdvantage)

    def test_typed_attribute_access(self, analyzer):
        insured = [_make_sim_result(growth=0.05) for _ in range(5)]
        uninsured = [_make_sim_result(growth=0.03) for _ in range(5)]

        result = analyzer.compare_scenarios(insured, uninsured)

        # Access via attributes (no warnings)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            _ = result.insured.time_average_mean
            _ = result.insured.time_average_median
            _ = result.insured.time_average_std
            _ = result.insured.ensemble_average
            _ = result.insured.survival_rate
            _ = result.insured.n_survived
            _ = result.ergodic_advantage.time_average_gain
            _ = result.ergodic_advantage.ensemble_average_gain
            _ = result.ergodic_advantage.survival_gain
            _ = result.ergodic_advantage.t_statistic
            _ = result.ergodic_advantage.p_value
            _ = result.ergodic_advantage.significant

    def test_dict_access_with_deprecation_warning(self, analyzer):
        insured = [_make_sim_result(growth=0.05) for _ in range(5)]
        uninsured = [_make_sim_result(growth=0.03) for _ in range(5)]

        result = analyzer.compare_scenarios(insured, uninsured)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            insured_metrics = result["insured"]
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

        assert isinstance(insured_metrics, ScenarioMetrics)

    def test_nested_dict_access(self, analyzer):
        """Backward-compatible nested dict access: result["insured"]["survival_rate"]."""
        insured = [_make_sim_result(growth=0.05) for _ in range(5)]
        uninsured = [_make_sim_result(growth=0.03) for _ in range(5)]

        result = analyzer.compare_scenarios(insured, uninsured)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            survival = result["insured"]["survival_rate"]
            # Two deprecation warnings: one for result["insured"], one for ...["survival_rate"]
            assert len(w) == 2

        assert isinstance(survival, float)
        assert 0.0 <= survival <= 1.0

    def test_contains_operator(self, analyzer):
        """The ``in`` operator works without deprecation warnings."""
        insured = [_make_sim_result(growth=0.05) for _ in range(5)]
        uninsured = [_make_sim_result(growth=0.03) for _ in range(5)]

        result = analyzer.compare_scenarios(insured, uninsured)

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert "insured" in result
            assert "uninsured" in result
            assert "ergodic_advantage" in result
            assert "nonexistent" not in result

    def test_significance_fields_always_present(self, analyzer):
        """t_statistic, p_value, significant are always present (not conditionally added)."""
        insured = [_make_sim_result(growth=0.05) for _ in range(5)]
        uninsured = [_make_sim_result(growth=0.03) for _ in range(5)]

        result = analyzer.compare_scenarios(insured, uninsured)

        assert hasattr(result.ergodic_advantage, "t_statistic")
        assert hasattr(result.ergodic_advantage, "p_value")
        assert hasattr(result.ergodic_advantage, "significant")

    def test_significance_nan_when_no_valid_data(self, analyzer):
        """When all paths bankrupt, significance fields are NaN."""
        insured = [np.array([100.0, 0.0]), np.array([200.0, 0.0])]
        uninsured = [np.array([150.0, 0.0]), np.array([300.0, 0.0])]

        result = analyzer.compare_scenarios(insured, uninsured)

        assert np.isnan(result.ergodic_advantage.t_statistic)
        assert np.isnan(result.ergodic_advantage.p_value)
        assert result.ergodic_advantage.significant is False

    def test_with_numpy_arrays(self, analyzer):
        """Works with 2D numpy array inputs."""
        rng = np.random.default_rng(42)
        n_paths, n_time = 20, 50

        insured = np.array(
            [
                1_000_000
                * np.exp(0.04 * np.arange(n_time) + np.cumsum(rng.normal(0, 0.02, n_time)))
                for _ in range(n_paths)
            ]
        )
        uninsured = np.array(
            [
                1_000_000
                * np.exp(0.03 * np.arange(n_time) + np.cumsum(rng.normal(0, 0.05, n_time)))
                for _ in range(n_paths)
            ]
        )

        result = analyzer.compare_scenarios(insured, uninsured, metric="equity")

        assert isinstance(result, ScenarioComparison)
        assert result.insured.survival_rate >= 0.0

    def test_n_survived_is_int(self, analyzer):
        """n_survived should be a Python int, not numpy int."""
        insured = [_make_sim_result(growth=0.05) for _ in range(5)]
        uninsured = [_make_sim_result(growth=0.03) for _ in range(5)]

        result = analyzer.compare_scenarios(insured, uninsured)

        assert isinstance(result.insured.n_survived, int)
        assert not isinstance(result.insured.n_survived, np.integer)
        assert isinstance(result.uninsured.n_survived, int)
        assert not isinstance(result.uninsured.n_survived, np.integer)


# ===================================================================
# BatchAnalysisResults return type
# ===================================================================


class TestAnalyzeSimulationBatchReturnType:
    """analyze_simulation_batch() returns BatchAnalysisResults dataclass."""

    def test_returns_batch_analysis_results(self, analyzer):
        results = [_make_sim_result(growth=0.05) for _ in range(10)]

        analysis = analyzer.analyze_simulation_batch(results, label="Test")

        assert isinstance(analysis, BatchAnalysisResults)
        assert isinstance(analysis.time_average, TimeAverageStats)
        assert isinstance(analysis.ensemble_average, EnsembleAverageStats)
        assert isinstance(analysis.convergence, ConvergenceStats)
        assert isinstance(analysis.survival_analysis, SurvivalAnalysisStats)

    def test_typed_attribute_access(self, analyzer):
        results = [_make_sim_result(growth=0.05) for _ in range(10)]

        analysis = analyzer.analyze_simulation_batch(results, label="Test")

        # Access without warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert analysis.label == "Test"
            assert analysis.n_simulations == 10
            _ = analysis.time_average.mean
            _ = analysis.time_average.median
            _ = analysis.time_average.std
            _ = analysis.time_average.min
            _ = analysis.time_average.max
            _ = analysis.ensemble_average.mean
            _ = analysis.ensemble_average.std
            _ = analysis.ensemble_average.survival_rate
            _ = analysis.ensemble_average.n_survived
            _ = analysis.ensemble_average.n_total
            _ = analysis.convergence.converged
            _ = analysis.convergence.standard_error
            _ = analysis.convergence.threshold
            _ = analysis.survival_analysis.survival_rate
            _ = analysis.survival_analysis.mean_survival_time
            _ = analysis.ergodic_divergence

    def test_dict_access_with_deprecation_warning(self, analyzer):
        results = [_make_sim_result(growth=0.05) for _ in range(10)]

        analysis = analyzer.analyze_simulation_batch(results, label="Test")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            label = analysis["label"]
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

        assert label == "Test"

    def test_nested_dict_access(self, analyzer):
        """Backward-compatible: analysis["time_average"]["mean"]."""
        results = [_make_sim_result(growth=0.05) for _ in range(10)]

        analysis = analyzer.analyze_simulation_batch(results, label="Test")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)
            mean = analysis["time_average"]["mean"]
            assert len(w) == 2

        assert isinstance(mean, float)

    def test_contains_operator(self, analyzer):
        results = [_make_sim_result(growth=0.05) for _ in range(10)]

        analysis = analyzer.analyze_simulation_batch(results, label="Test")

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert "label" in analysis
            assert "n_simulations" in analysis
            assert "time_average" in analysis
            assert "ensemble_average" in analysis
            assert "convergence" in analysis
            assert "survival_analysis" in analysis
            assert "ergodic_divergence" in analysis

    def test_all_bankrupt_batch(self, analyzer):
        results = [_make_sim_result(growth=-0.5, insolvency_year=3) for _ in range(5)]

        analysis = analyzer.analyze_simulation_batch(results, label="Bankrupt")

        assert analysis.convergence.converged is False
        assert analysis.convergence.standard_error == np.inf
        assert np.isnan(analysis.ergodic_divergence)
        assert analysis.time_average.mean == -np.inf

    def test_ensemble_average_n_survived_is_int(self, analyzer):
        results = [_make_sim_result(growth=0.05) for _ in range(10)]

        analysis = analyzer.analyze_simulation_batch(results, label="Test")

        assert isinstance(analysis.ensemble_average.n_survived, int)
        assert not isinstance(analysis.ensemble_average.n_survived, np.integer)
        assert isinstance(analysis.ensemble_average.n_total, int)
        assert not isinstance(analysis.ensemble_average.n_total, np.integer)


# ===================================================================
# _DictAccessMixin
# ===================================================================


class TestDictAccessMixin:
    """Tests for the backward-compatible dict access mixin."""

    def test_getitem_raises_key_error_for_missing(self):
        metrics = ScenarioMetrics(
            time_average_mean=0.05,
            time_average_median=0.04,
            time_average_std=0.01,
            ensemble_average=0.06,
            survival_rate=1.0,
            n_survived=10,
        )

        with pytest.raises(KeyError, match="nonexistent"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                _ = metrics["nonexistent"]

    def test_get_returns_default_for_missing(self):
        metrics = ScenarioMetrics(
            time_average_mean=0.05,
            time_average_median=0.04,
            time_average_std=0.01,
            ensemble_average=0.06,
            survival_rate=1.0,
            n_survived=10,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            val = metrics.get("nonexistent", 42)

        assert val == 42

    def test_keys_values_items(self):
        metrics = ScenarioMetrics(
            time_average_mean=0.05,
            time_average_median=0.04,
            time_average_std=0.01,
            ensemble_average=0.06,
            survival_rate=1.0,
            n_survived=10,
        )

        assert "time_average_mean" in metrics.keys()
        assert 0.05 in metrics.values()
        assert ("survival_rate", 1.0) in metrics.items()

    def test_contains_non_string_returns_false(self):
        metrics = ScenarioMetrics(
            time_average_mean=0.05,
            time_average_median=0.04,
            time_average_std=0.01,
            ensemble_average=0.06,
            survival_rate=1.0,
            n_survived=10,
        )

        assert 42 not in metrics
        assert None not in metrics


# ===================================================================
# Import accessibility
# ===================================================================


class TestImports:
    """New types are importable from expected locations."""

    def test_import_from_ergodic_types(self):
        # pylint: disable=reimported,import-outside-toplevel
        from ergodic_insurance.ergodic_types import BatchAnalysisResults as BAR
        from ergodic_insurance.ergodic_types import ScenarioComparison as SC

        assert BAR is not None
        assert SC is not None

    def test_import_from_ergodic_analyzer(self):
        # pylint: disable=reimported,import-outside-toplevel
        from ergodic_insurance.ergodic_analyzer import BatchAnalysisResults as BAR
        from ergodic_insurance.ergodic_analyzer import ScenarioComparison as SC

        assert BAR is not None
        assert SC is not None

    def test_import_from_top_level(self):
        # pylint: disable=reimported,import-outside-toplevel,no-name-in-module
        from ergodic_insurance import BatchAnalysisResults as BAR
        from ergodic_insurance import ScenarioComparison as SC

        assert BAR is not None
        assert SC is not None
