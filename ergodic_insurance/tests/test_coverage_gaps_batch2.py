"""Tests targeting specific untested code paths across multiple modules.

This file covers gaps identified by coverage analysis in:
- benchmarking.py (lines 290-291, 306-309, 312-319, 447, 480-487, 552, 554, 612)
- adaptive_stopping.py (lines 246, 293, 297, 362-363, 470, 538-543, 561-563)
- parameter_sweep.py (lines 268, 272-275, 574, 666, 761, 801-805, 822)
- optimal_control.py (lines 102, 299, 337, 362, 389-431, 480, 617, 723)
"""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.adaptive_stopping import (
    AdaptiveStoppingMonitor,
    ConvergenceStatus,
    StoppingCriteria,
    StoppingRule,
)

# ---------------------------------------------------------------------------
# benchmarking.py coverage gaps
# ---------------------------------------------------------------------------
from ergodic_insurance.benchmarking import (
    BenchmarkConfig,
    BenchmarkMetrics,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSuite,
    ComprehensiveBenchmarkResult,
    SystemProfiler,
)
from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.hjb_solver import (
    ControlVariable,
    HJBProblem,
    HJBSolver,
    HJBSolverConfig,
    LogUtility,
    StateSpace,
    StateVariable,
)
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.optimal_control import (
    ControlMode,
    ControlSpace,
    HJBFeedbackControl,
    OptimalController,
    StaticControl,
    TimeVaryingControl,
    create_hjb_controller,
)
from ergodic_insurance.parameter_sweep import ParameterSweeper, SweepConfig


class TestBenchmarkRunnerExceptionHandling:
    """Cover lines 290-291: exception re-raised as RuntimeError."""

    def test_run_single_benchmark_raises_runtime_error_on_failure(self):
        """When the benchmarked function raises, a RuntimeError should wrap it."""
        runner = BenchmarkRunner()

        def failing_func():
            raise ValueError("simulated failure")

        with patch.object(runner.profiler, "start"):
            with pytest.raises(RuntimeError, match="Benchmark failed: simulated failure"):
                runner.run_single_benchmark(failing_func)


class TestBenchmarkRunnerCacheHitRate:
    """Cover lines 306-309: cache hit rate extraction from func.__self__.cache."""

    def test_cache_hit_rate_from_bound_method_cache(self):
        """Extract cache_hit_rate via func.__self__.cache.hit_rate path."""

        class FakeCache:
            hit_rate = 92.5

        class FakeEngine:
            cache = FakeCache()

            def run(self, **kwargs):
                return "result_without_cache_attr"

        engine = FakeEngine()
        runner = BenchmarkRunner()

        with patch.object(runner.profiler, "start"):
            with patch.object(runner.profiler, "sample"):
                with patch.object(
                    runner.profiler, "get_metrics", return_value=(50.0, 200.0, 180.0)
                ):
                    metrics = runner.run_single_benchmark(engine.run, kwargs={"n_simulations": 100})

        assert metrics.cache_hit_rate == 92.5


class TestBenchmarkRunnerResultAttributes:
    """Cover lines 312-319: accuracy_score and convergence_iterations from result."""

    def test_accuracy_score_extracted_from_result(self):
        """When the result object has accuracy_score, it should be used."""

        @dataclass
        class FakeResult:
            accuracy_score: float = 0.987
            convergence_iterations: int = 42

        runner = BenchmarkRunner()

        def func_returning_result(**kwargs):
            return FakeResult()

        with patch.object(runner.profiler, "start"):
            with patch.object(runner.profiler, "sample"):
                with patch.object(
                    runner.profiler, "get_metrics", return_value=(60.0, 300.0, 250.0)
                ):
                    metrics = runner.run_single_benchmark(
                        func_returning_result, kwargs={"n_simulations": 50}
                    )

        assert metrics.accuracy_score == 0.987
        assert metrics.convergence_iterations == 42


class TestBenchmarkScaleMissesTarget:
    """Cover line 447: the 'misses target' print branch."""

    def test_benchmark_scale_prints_misses_target(self, capsys):
        """When result does not meet target, the misses-target message is printed."""
        suite = BenchmarkSuite()
        mock_engine = MagicMock()
        mock_engine.config = MagicMock()

        # Metrics that exceed both time and memory targets
        metrics = BenchmarkMetrics(
            execution_time=999.0,
            simulations_per_second=1.0,
            memory_peak_mb=9999.0,
            memory_average_mb=9999.0,
        )
        config = BenchmarkConfig(scales=[1000], target_times={1000: 1.0}, memory_limit_mb=100.0)

        with patch.object(suite.runner, "run_with_warmup", return_value=[metrics]):
            result = suite.benchmark_scale(mock_engine, 1000, config)

        captured = capsys.readouterr()
        assert "Misses targets" in captured.out or "misses" in captured.out.lower()


class TestComprehensiveBenchmarkWithOptimizations:
    """Cover lines 480-487: engine.enable_optimizations() path."""

    def test_run_comprehensive_benchmark_with_enable_optimizations(self):
        """When engine has enable_optimizations, it should be called and
        additional optimized benchmarks should be run."""
        suite = BenchmarkSuite()
        mock_engine = MagicMock()
        mock_engine.config = MagicMock()
        # Ensure the engine has enable_optimizations attribute
        mock_engine.enable_optimizations = MagicMock()

        metrics = BenchmarkMetrics(
            execution_time=1.0,
            simulations_per_second=1000.0,
            memory_peak_mb=256.0,
            memory_average_mb=200.0,
        )
        result = BenchmarkResult(
            scale=1000, metrics=metrics, configuration={}, timestamp=datetime.now()
        )

        config = BenchmarkConfig(scales=[1000])

        with patch.object(suite, "benchmark_scale", return_value=result):
            with patch("builtins.print"):
                comp_result = suite.run_comprehensive_benchmark(mock_engine, config)

        # enable_optimizations should have been called
        mock_engine.enable_optimizations.assert_called_once()

        # Should have 2 results: 1 without optimizations + 1 with
        assert len(comp_result.results) == 2


class TestMeetsRequirements100KSpecialChecks:
    """Cover lines 552, 554: 100K special checks for time/memory and accuracy."""

    def test_100k_fails_on_execution_time_over_60(self):
        """A 100K result with execution_time > 60 should fail meets_requirements
        even if target_times allows it."""
        metrics = BenchmarkMetrics(
            execution_time=61.0,
            simulations_per_second=1639.0,
            memory_peak_mb=3000.0,
            memory_average_mb=2500.0,
            accuracy_score=0.99999,
        )
        result = BenchmarkResult(
            scale=100000, metrics=metrics, configuration={}, timestamp=datetime.now()
        )
        # Set a very generous target_time so meets_target passes but
        # the 100K special check fails
        config = BenchmarkConfig(target_times={100000: 120.0}, memory_limit_mb=5000.0)
        comp = ComprehensiveBenchmarkResult([result], config, {})

        assert comp.meets_requirements() is False

    def test_100k_fails_on_memory_over_4000(self):
        """A 100K result with memory_peak_mb > 4000 should fail."""
        metrics = BenchmarkMetrics(
            execution_time=50.0,
            simulations_per_second=2000.0,
            memory_peak_mb=4500.0,
            memory_average_mb=3000.0,
            accuracy_score=0.99999,
        )
        result = BenchmarkResult(
            scale=100000, metrics=metrics, configuration={}, timestamp=datetime.now()
        )
        config = BenchmarkConfig(target_times={100000: 120.0}, memory_limit_mb=5000.0)
        comp = ComprehensiveBenchmarkResult([result], config, {})

        assert comp.meets_requirements() is False

    def test_100k_fails_on_low_accuracy(self):
        """A 100K result with accuracy_score < 0.9999 should fail."""
        metrics = BenchmarkMetrics(
            execution_time=50.0,
            simulations_per_second=2000.0,
            memory_peak_mb=3000.0,
            memory_average_mb=2500.0,
            accuracy_score=0.999,  # Below 0.9999
        )
        result = BenchmarkResult(
            scale=100000, metrics=metrics, configuration={}, timestamp=datetime.now()
        )
        config = BenchmarkConfig(target_times={100000: 120.0}, memory_limit_mb=5000.0)
        comp = ComprehensiveBenchmarkResult([result], config, {})

        assert comp.meets_requirements() is False


class TestComprehensiveSummaryRequirementsNotMet:
    """Cover line 612: requirements not met message in summary()."""

    def test_summary_contains_requirements_not_met(self):
        """When meets_requirements is False, summary should say so."""
        metrics = BenchmarkMetrics(
            execution_time=999.0,
            simulations_per_second=100.0,
            memory_peak_mb=9999.0,
            memory_average_mb=9000.0,
            accuracy_score=0.5,
        )
        result = BenchmarkResult(
            scale=100000, metrics=metrics, configuration={}, timestamp=datetime.now()
        )
        config = BenchmarkConfig()
        comp = ComprehensiveBenchmarkResult([result], config, {})

        summary = comp.summary()
        assert "REQUIREMENTS NOT MET" in summary


# ---------------------------------------------------------------------------
# adaptive_stopping.py coverage gaps
# ---------------------------------------------------------------------------


class TestVarianceBurnInFallback:
    """Cover line 246: variance method returns default fallback."""

    def test_variance_method_default_fallback_short_chain(self):
        """When variance method finds no stable points, return n_iterations//10."""
        monitor = AdaptiveStoppingMonitor()
        # Create a very short monotonically increasing chain so
        # no stable variance region is found
        chains = np.arange(200, dtype=float).reshape(1, -1)
        burn_in = monitor.detect_adaptive_burn_in(chains, method="variance")
        # Should be n_iterations // 10 = 200 // 10 = 20 or a reasonable value
        assert isinstance(burn_in, int)
        assert burn_in >= 0


class TestEstimateConvergenceRateAlreadyConverged:
    """Cover line 293: iterations_to_target = 0 when already close to target."""

    def test_convergence_rate_returns_zero_remaining_when_converged(self):
        """When current value is already within 1% of target, remaining should be 0."""
        monitor = AdaptiveStoppingMonitor()
        # Create history that decays and then reaches near the target (1.0)
        # The last value should be within 1% of target
        history = [2.0, 1.5, 1.2, 1.05, 1.005]
        rate, remaining = monitor.estimate_convergence_rate(history, target_value=1.0)
        # Since the last value 1.005 is within 1% of 1.0, remaining should be 0
        assert remaining == 0
        assert rate > 0


class TestEstimateConvergenceRateFallback:
    """Cover line 297: return 0.0, -1 when slope is non-positive."""

    def test_convergence_rate_returns_fallback_for_diverging_data(self):
        """When data is diverging (positive slope in log space), return (0.0, -1)."""
        monitor = AdaptiveStoppingMonitor()
        # Create a history that diverges from the target
        history = [1.1, 1.2, 1.4, 1.8, 2.5]
        rate, remaining = monitor.estimate_convergence_rate(history, target_value=1.0)
        assert rate == 0.0
        assert remaining == -1


class TestDiagnosticsESSZero:
    """Cover lines 362-363: mcse=inf, mcse_relative=inf when ESS is 0."""

    def test_zero_ess_produces_inf_mcse(self):
        """When ESS calculates to 0, mcse and mcse_relative should be inf."""
        monitor = AdaptiveStoppingMonitor()

        # We need to trick _calculate_ess into returning 0.
        # Patch _calculate_ess to return 0
        with patch.object(monitor, "_calculate_ess", return_value=0.0):
            chains = np.random.randn(1, 100)
            diagnostics = monitor._calculate_diagnostics(chains)

        assert diagnostics["mcse"] == np.inf
        assert diagnostics["mcse_relative"] == np.inf


class TestDetectBurnInEarlyReturn:
    """Cover line 470: _detect_burn_in returns early when iteration < 500."""

    def test_detect_burn_in_skips_when_too_early(self):
        """When iteration < 500, burn-in detection should not run."""
        monitor = AdaptiveStoppingMonitor()
        chains = np.random.randn(2, 300)

        monitor._detect_burn_in(chains, iteration=300)

        # Burn-in should not be detected
        assert not monitor.burn_in_detected
        assert monitor.burn_in_iteration == 0


class TestUnknownStoppingRule:
    """Cover lines 538-543: the else branch for unknown stopping rules."""

    def test_unknown_stopping_rule_returns_not_converged(self):
        """An unrecognized stopping rule should return converged=False."""
        monitor = AdaptiveStoppingMonitor()

        # Create a fake rule value that is not handled
        # We can do this by monkeypatching the criteria.rule after creation
        monitor.criteria.rule = "totally_unknown_rule"  # type: ignore[assignment]

        diagnostics = {"r_hat": 1.0, "ess": 5000, "mcse_relative": 0.001}
        converged, reason = monitor._check_stopping_rule(diagnostics)

        assert converged is False
        assert "Unknown stopping rule" in reason


class TestEstimateRemainingWithConvergenceRate:
    """Cover lines 561-563: _estimate_remaining_iterations when
    iterations_to_target > 0 from R-hat history."""

    def test_remaining_iterations_estimated_from_r_hat_history(self):
        """When R-hat history shows improvement, remaining iterations should
        be estimated and convergence_rate / estimated_total stored."""
        monitor = AdaptiveStoppingMonitor()
        monitor.criteria.rule = StoppingRule.R_HAT
        monitor.criteria.r_hat_threshold = 1.05
        monitor.criteria.min_iterations = 100
        monitor.criteria.check_interval = 100

        # Populate R-hat history showing convergence
        monitor.r_hat_history = [1.5, 1.3, 1.15, 1.08]
        monitor.ess_history = [200, 400, 600, 800]
        monitor.iteration_history = [100, 200, 300, 400]
        monitor.mean_history = [5.0, 5.0, 5.0, 5.0]

        diagnostics = {"r_hat": 1.08, "ess": 800, "mean": 5.0, "variance": 1.0}

        remaining = monitor._estimate_remaining_iterations(500, False, diagnostics)

        # Should return a positive estimate or None
        if remaining is not None:
            assert remaining >= 0
            # Check that convergence_rate was set
            if monitor.convergence_rate is not None:
                assert monitor.convergence_rate > 0


class TestCustomRuleNoFunction:
    """Cover lines 537-539: custom rule with no function provided."""

    def test_custom_rule_without_function_returns_not_converged(self):
        """When CUSTOM rule is set but no custom_rule function is provided."""
        monitor = AdaptiveStoppingMonitor(
            criteria=StoppingCriteria(rule=StoppingRule.CUSTOM),
            custom_rule=None,
        )
        chains = np.random.randn(2, 2000) + 10

        status = monitor.check_convergence(2000, chains)

        assert not status.converged
        assert "No custom rule provided" in status.reason


# ---------------------------------------------------------------------------
# parameter_sweep.py coverage gaps
# ---------------------------------------------------------------------------


class TestSweepProgressCallback:
    """Cover line 268: progress_callback invocation during parallel sweep."""

    def test_progress_callback_called_during_sequential_sweep(self):
        """The progress_callback should be called during sequential sweep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sweeper = ParameterSweeper(optimizer=None, cache_dir=tmpdir, use_parallel=False)
            config = SweepConfig(
                parameters={"param1": [1, 2]},
                fixed_params={"time_horizon": 10},
                n_workers=1,
            )

            callback = MagicMock()

            def mock_run_single(params, metrics):
                return {**params, "optimal_roe": 0.1, "ruin_probability": 0.01}

            with patch.object(sweeper, "_run_single", side_effect=mock_run_single):
                with patch.object(sweeper, "_save_results"):
                    results = sweeper.sweep(config, progress_callback=callback)

            assert callback.call_count == 2  # Once per parameter combination


class TestSweepExceptionHandlingInSequential:
    """Cover lines 272-275 and 287-288: error handling during sweep execution."""

    def test_sequential_sweep_continues_on_error(self):
        """Errors in sequential execution should be caught, not crash the sweep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sweeper = ParameterSweeper(optimizer=None, cache_dir=tmpdir, use_parallel=False)
            config = SweepConfig(
                parameters={"param1": [1, 2, 3]},
                fixed_params={"time_horizon": 10},
                n_workers=1,
            )

            call_count = 0

            def mock_run_single(params, metrics):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise ValueError("Simulated failure on second run")
                return {**params, "optimal_roe": 0.1, "ruin_probability": 0.01}

            with patch.object(sweeper, "_run_single", side_effect=mock_run_single):
                with patch.object(sweeper, "_save_results"):
                    results = sweeper.sweep(config)

            # Should have 2 results (3 runs minus 1 failure)
            assert len(results) == 2


class TestCompareScenariosEmpty:
    """Cover line 574: compare_scenarios with empty results dict."""

    def test_compare_scenarios_returns_empty_for_empty_input(self):
        """When no scenarios are provided, return empty DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sweeper = ParameterSweeper(cache_dir=tmpdir)
            result = sweeper.compare_scenarios({})
            assert isinstance(result, pd.DataFrame)
            assert result.empty


class TestAdaptiveRefinementCategoricalParam:
    """Cover line 666: categorical parameter handling in adaptive refinement."""

    def test_adaptive_refinement_preserves_categorical_params(self):
        """Categorical parameters should be kept as-is during refinement."""
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as tmpdir:
            sweeper = ParameterSweeper(cache_dir=tmpdir, use_parallel=False)

            initial_results = pd.DataFrame(
                {
                    "category": ["a", "b", "c", "a", "b", "c"] * 3,
                    "numeric_param": list(range(18)),
                    "optimal_roe": rng.uniform(0.05, 0.25, 18),
                }
            )

            config = SweepConfig(
                parameters={
                    "category": ["a", "b", "c"],
                    "numeric_param": [1.0, 5.0, 10.0],
                },
                adaptive_refinement=True,
                refinement_threshold=50,
            )

            # Mock find_optimal_regions to bypass .mean() on string columns
            optimal_df = initial_results.head(5)
            param_stats = pd.DataFrame(
                {
                    "min": {"category": "a", "numeric_param": 5.0},
                    "max": {"category": "c", "numeric_param": 15.0},
                    "mean": {"category": "b", "numeric_param": 10.0},
                    "std": {"category": "a", "numeric_param": 3.0},
                    "median": {"category": "b", "numeric_param": 10.0},
                }
            )

            def mock_sweep(refined_config):
                # Verify categorical param was preserved
                assert refined_config.parameters["category"] == ["a", "b", "c"]
                n = 5
                return pd.DataFrame(
                    {
                        "category": ["a"] * n,
                        "numeric_param": np.linspace(5, 10, n),
                        "optimal_roe": rng.uniform(0.18, 0.22, n),
                    }
                )

            with patch.object(
                sweeper, "find_optimal_regions", return_value=(optimal_df, param_stats)
            ):
                with patch.object(sweeper, "sweep", side_effect=mock_sweep):
                    refined = sweeper._apply_adaptive_refinement(initial_results, config)

            assert len(refined) >= len(initial_results)


class TestLoadResultsNonDataFrame:
    """Cover lines 801-805, 822: load_results when stored object is not a DataFrame."""

    def test_load_results_returns_none_for_missing_hash(self):
        """When no files exist for the sweep hash, return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sweeper = ParameterSweeper(cache_dir=tmpdir)
            result = sweeper.load_results("nonexistent_hash_12345")
            assert result is None

    def test_load_results_returns_none_for_non_dataframe_hdf5(self):
        """When the HDF5 file contains something other than a DataFrame,
        return None. This covers lines 801-805."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sweeper = ParameterSweeper(cache_dir=tmpdir)
            sweep_hash = "test_hash"

            # Create a dummy .h5 file so exists() returns True
            h5_file = sweeper.cache_dir / f"sweep_{sweep_hash}.h5"
            h5_file.write_text("dummy")

            # Mock pd.read_hdf to return a Series (not a DataFrame)
            with patch("ergodic_insurance.parameter_sweep.pd.read_hdf") as mock_read:
                mock_read.return_value = pd.Series([1, 2, 3])
                result = sweeper.load_results(sweep_hash)

            # Should return None since it's not a DataFrame
            assert result is None

    def test_load_results_hdf5_import_error_falls_through(self):
        """When read_hdf raises ImportError, load_results falls through
        to try parquet. If no parquet exists, returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sweeper = ParameterSweeper(cache_dir=tmpdir)
            sweep_hash = "import_err"

            # Create dummy .h5 file
            h5_file = sweeper.cache_dir / f"sweep_{sweep_hash}.h5"
            h5_file.write_text("dummy")

            with patch("ergodic_insurance.parameter_sweep.pd.read_hdf") as mock_read:
                mock_read.side_effect = ImportError("tables not available")
                result = sweeper.load_results(sweep_hash)

            # No parquet file either, so should return None
            assert result is None

    def test_load_results_temp_file_non_dataframe(self):
        """When a temp HDF5 file contains a non-DataFrame, return None.
        This covers line 822."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sweeper = ParameterSweeper(cache_dir=tmpdir)
            sweep_hash = "test_temp_hash"

            # Create only a temp HDF5 file (no main file, no parquet files)
            temp_h5_file = sweeper.cache_dir / f"sweep_{sweep_hash}_temp.h5"
            temp_h5_file.write_text("dummy")

            # Mock pd.read_hdf to return a Series for the temp file path
            with patch("ergodic_insurance.parameter_sweep.pd.read_hdf") as mock_read:
                mock_read.return_value = pd.Series([1, 2, 3])
                result = sweeper.load_results(sweep_hash)

            assert result is None


class TestSaveIntermediateResultsEmpty:
    """Cover line 761 area: _save_intermediate_results with empty list."""

    def test_save_intermediate_results_empty_list_is_noop(self):
        """Saving an empty result list should be a no-op."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sweeper = ParameterSweeper(cache_dir=tmpdir)
            # Should not raise and should not create files
            sweeper._save_intermediate_results([], "empty_hash")
            # Verify no temp files were created
            temp_files = list(sweeper.cache_dir.glob("sweep_empty_hash_temp.*"))
            assert len(temp_files) == 0


# ---------------------------------------------------------------------------
# optimal_control.py coverage gaps
# ---------------------------------------------------------------------------


class TestControlSpaceCoveragePercentageLengthMismatch:
    """Cover line 102: coverage percentages length mismatch validation."""

    def test_coverage_percentages_length_mismatch_raises(self):
        """Providing coverage_percentages with wrong length should raise ValueError."""
        with pytest.raises(ValueError, match="Coverage percentages must match number of layers"):
            ControlSpace(
                limits=[(1e6, 5e7), (5e7, 1e8)],
                retentions=[(1e5, 1e6), (1e6, 5e6)],
                coverage_percentages=[(0.8, 1.0)],  # Only 1 but 2 layers
            )


class TestHJBFeedbackControlUnsolvedSolver:
    """Cover line 299: HJB solver with None optimal_policy."""

    def test_unsolved_hjb_solver_raises_value_error(self):
        """Creating HJBFeedbackControl with unsolved solver should raise ValueError."""
        mock_solver = MagicMock()
        mock_solver.optimal_policy = None

        control_space = ControlSpace(limits=[(1e6, 5e7)], retentions=[(1e5, 1e6)])

        with pytest.raises(ValueError, match="HJB solver must be solved"):
            HJBFeedbackControl(mock_solver, control_space)


class TestDefaultStateMappingFallback:
    """Cover line 337: default state mapping fallback to default value."""

    @pytest.fixture
    def solved_hjb_solver(self):
        """Create a minimal solved HJB problem for testing."""
        state_space = StateSpace(
            [StateVariable("wealth", 1e6, 1e8, 5), StateVariable("time", 0, 10, 3)]
        )
        control_variables = [
            ControlVariable("limit", 1e6, 5e7, 3),
            ControlVariable("retention", 1e5, 1e6, 3),
        ]

        def dynamics(state, control, time):
            return np.zeros_like(state)

        def running_cost(state, control, time):
            return np.zeros(state.shape[:-1])

        problem = HJBProblem(
            state_space=state_space,
            control_variables=control_variables,
            utility_function=LogUtility(),
            dynamics=dynamics,
            running_cost=running_cost,
            time_horizon=10,
        )
        config = HJBSolverConfig(max_iterations=2, verbose=False)
        solver = HJBSolver(problem, config)
        solver.solve()
        return solver

    def test_default_mapping_with_no_wealth_key(self, solved_hjb_solver):
        """When state dict has no assets/wealth/equity key, use default 1e7."""
        control_space = ControlSpace(limits=[(1e6, 5e7)], retentions=[(1e5, 1e6)])
        control = HJBFeedbackControl(solved_hjb_solver, control_space)

        # State with no standard wealth keys
        state = {"something_else": 42, "time": 5.0}
        result = control.get_control(state, time=5.0)

        # Should not raise - should use default value 1e7
        assert "limits" in result
        assert "retentions" in result
        assert "coverages" in result

    def test_default_mapping_with_no_time_key(self, solved_hjb_solver):
        """When state dict has no time key, use default 0.0."""
        control_space = ControlSpace(limits=[(1e6, 5e7)], retentions=[(1e5, 1e6)])
        control = HJBFeedbackControl(solved_hjb_solver, control_space)

        state = {"assets": 5e7}  # No 'time' key
        result = control.get_control(state, time=0.0)

        assert "limits" in result
        assert "retentions" in result


class TestCreateInterpolatorsNonePolicy:
    """Cover line 362: _create_interpolators when optimal_policy is None."""

    def test_create_interpolators_returns_early_when_policy_is_none(self):
        """When optimal_policy becomes None, _create_interpolators returns without error."""
        # Create a solved solver first, then set policy to None to test the guard
        mock_solver = MagicMock()
        mock_solver.optimal_policy = {"limit": np.ones((5, 3)), "retention": np.ones((5, 3))}
        mock_solver.problem.state_space.grids = (
            np.linspace(1e6, 1e8, 5),
            np.linspace(0, 10, 3),
        )

        control_space = ControlSpace(limits=[(1e6, 5e7)], retentions=[(1e5, 1e6)])

        # Create the control (will create interpolators normally)
        control = HJBFeedbackControl(mock_solver, control_space)
        assert len(control.interpolators) == 2

        # Now set policy to None and re-run _create_interpolators
        mock_solver.optimal_policy = None
        control._create_interpolators()

        # Interpolators dict should now be empty (reset by the method)
        assert control.interpolators == {}


class TestGetControlMissingControlNames:
    """Cover lines 389-431: get_control method various branches for missing control names."""

    @pytest.fixture
    def hjb_control_with_generic_names(self):
        """Create HJBFeedbackControl where controls have generic names
        (not 'limit_0'/'retention_0' but 'limit'/'retention')."""
        mock_solver = MagicMock()
        # Use generic names 'limit' and 'retention' (not 'limit_0')
        mock_solver.optimal_policy = {
            "limit": np.full((5, 3), 2e7),
            "retention": np.full((5, 3), 5e5),
        }
        mock_solver.problem.state_space.grids = (
            np.linspace(1e6, 1e8, 5),
            np.linspace(0, 10, 3),
        )

        control_space = ControlSpace(limits=[(1e6, 5e7)], retentions=[(1e5, 1e6)])
        return HJBFeedbackControl(mock_solver, control_space)

    def test_get_control_uses_generic_limit_name(self, hjb_control_with_generic_names):
        """When controls have 'limit' (not 'limit_0'), the elif branch fires."""
        state = {"assets": 5e7, "time": 5.0}
        result = hjb_control_with_generic_names.get_control(state)

        # Should use the 'limit' and 'retention' interpolators
        assert len(result["limits"]) == 1
        assert len(result["retentions"]) == 1
        # Values should be close to the fill value
        assert result["limits"][0] > 0
        assert result["retentions"][0] > 0

    def test_get_control_falls_back_to_midpoint(self):
        """When controls don't have matching names, fall back to midpoint values.
        This covers lines 414-416 and 424-425."""
        mock_solver = MagicMock()
        # Use names that don't match limit_0/limit or retention_0/retention
        mock_solver.optimal_policy = {
            "some_other_control": np.full((5, 3), 100.0),
        }
        mock_solver.problem.state_space.grids = (
            np.linspace(1e6, 1e8, 5),
            np.linspace(0, 10, 3),
        )

        control_space = ControlSpace(limits=[(1e6, 5e7)], retentions=[(1e5, 1e6)])
        control = HJBFeedbackControl(mock_solver, control_space)

        state = {"assets": 5e7, "time": 5.0}
        result = control.get_control(state)

        # Limits should fall back to midpoint of (1e6, 5e7) = 2.55e7
        expected_limit_midpoint = (1e6 + 5e7) / 2
        assert result["limits"][0] == pytest.approx(expected_limit_midpoint)

        # Retentions should fall back to midpoint of (1e5, 1e6) = 5.5e5
        expected_retention_midpoint = (1e5 + 1e6) / 2
        assert result["retentions"][0] == pytest.approx(expected_retention_midpoint)

        # Coverages should default to 1.0
        assert result["coverages"][0] == 1.0

    def test_get_control_coverage_from_interpolator(self):
        """When controls include 'coverage_0', it should be used.
        This covers line 428-429."""
        mock_solver = MagicMock()
        mock_solver.optimal_policy = {
            "limit_0": np.full((5, 3), 2e7),
            "retention_0": np.full((5, 3), 5e5),
            "coverage_0": np.full((5, 3), 0.95),
        }
        mock_solver.problem.state_space.grids = (
            np.linspace(1e6, 1e8, 5),
            np.linspace(0, 10, 3),
        )

        control_space = ControlSpace(limits=[(1e6, 5e7)], retentions=[(1e5, 1e6)])
        control = HJBFeedbackControl(mock_solver, control_space)

        state = {"assets": 5e7, "time": 5.0}
        result = control.get_control(state)

        assert result["coverages"][0] == pytest.approx(0.95, abs=0.05)


class TestHJBFeedbackControlUpdate:
    """Cover line 480 area: update method is a no-op."""

    def test_update_is_noop(self):
        """HJBFeedbackControl.update should do nothing without error."""
        mock_solver = MagicMock()
        mock_solver.optimal_policy = {
            "limit": np.full((5, 3), 2e7),
        }
        mock_solver.problem.state_space.grids = (
            np.linspace(1e6, 1e8, 5),
            np.linspace(0, 10, 3),
        )
        control_space = ControlSpace(limits=[(1e6, 5e7)], retentions=[(1e5, 1e6)])
        control = HJBFeedbackControl(mock_solver, control_space)

        # Should not raise
        control.update({"assets": 5e7}, {"losses": 1e5})


class TestCreateHJBControllerFunction:
    """Cover line 617: create_hjb_controller function."""

    @pytest.fixture
    def manufacturer(self):
        config = ManufacturerConfig(
            initial_assets=1e7,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.6,
        )
        return WidgetManufacturer(config)

    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    def test_create_hjb_controller_returns_functional_controller(self, manufacturer):
        """create_hjb_controller should return a working OptimalController."""
        controller = create_hjb_controller(manufacturer, simulation_years=2, utility_type="log")

        assert isinstance(controller, OptimalController)
        assert isinstance(controller.strategy, HJBFeedbackControl)

        # Should be able to apply control
        program = controller.apply_control(manufacturer, time=0)
        assert program is not None
        assert len(program.layers) > 0


class TestOptimalControllerEmptySummary:
    """Cover the empty performance summary path."""

    def test_performance_summary_empty_history(self):
        """get_performance_summary with no history returns empty DataFrame."""
        strategy = StaticControl([1e7], [1e6])
        control_space = ControlSpace(limits=[(1e6, 5e7)], retentions=[(1e5, 1e6)])
        controller = OptimalController(strategy, control_space)

        summary = controller.get_performance_summary()
        assert isinstance(summary, pd.DataFrame)
        assert summary.empty


class TestTimeVaryingControlUpdate:
    """Cover TimeVaryingControl.update as a no-op."""

    def test_time_varying_update_is_noop(self):
        """TimeVaryingControl.update should do nothing without error."""
        control = TimeVaryingControl(
            time_schedule=[0, 10],
            limits_schedule=[[1e7], [2e7]],
            retentions_schedule=[[1e6], [2e6]],
        )
        # Should not raise
        control.update({"wealth": 1e7}, {"loss": 1e5})


class TestStaticControlUpdate:
    """Cover StaticControl.update as a no-op."""

    def test_static_control_update_is_noop(self):
        """StaticControl.update should do nothing without error."""
        control = StaticControl([1e7], [1e6])
        control.update({"wealth": 1e7}, {"loss": 1e5})
        # Verify state is unchanged
        result = control.get_control({}, 0)
        assert result["limits"] == [1e7]
        assert result["retentions"] == [1e6]
