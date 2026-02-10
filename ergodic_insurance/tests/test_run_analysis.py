"""Tests for the run_analysis quick-start factory function."""

import numpy as np
import pandas as pd
import pytest

from ergodic_insurance._run_analysis import AnalysisResults, run_analysis

# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImport:
    """Verify that run_analysis is accessible from the top-level package."""

    def test_import_from_package(self):
        from ergodic_insurance import (  # noqa: F811  # pylint: disable=no-name-in-module
            run_analysis as ra,
        )

        assert callable(ra)

    def test_import_analysis_results_from_package(self):
        from ergodic_insurance import (  # noqa: F811  # pylint: disable=no-name-in-module
            AnalysisResults as AR,
        )

        assert AR is not None


# ---------------------------------------------------------------------------
# Basic invocation tests (small n_simulations / time_horizon for speed)
# ---------------------------------------------------------------------------


class TestRunAnalysis:
    """Core tests for the run_analysis() factory."""

    def test_returns_analysis_results(self):
        results = run_analysis(
            n_simulations=5,
            time_horizon=5,
            seed=42,
            compare_uninsured=False,
        )
        assert isinstance(results, AnalysisResults)

    def test_insured_results_populated(self):
        results = run_analysis(
            n_simulations=5,
            time_horizon=5,
            seed=42,
            compare_uninsured=False,
        )
        assert len(results.insured_results) == 5

    def test_uninsured_results_when_compare_true(self):
        results = run_analysis(
            n_simulations=5,
            time_horizon=5,
            seed=42,
            compare_uninsured=True,
        )
        assert len(results.uninsured_results) == 5
        assert results.comparison is not None

    def test_uninsured_results_empty_when_compare_false(self):
        results = run_analysis(
            n_simulations=3,
            time_horizon=3,
            seed=0,
            compare_uninsured=False,
        )
        assert len(results.uninsured_results) == 0
        assert results.comparison is None

    def test_reproducible_with_seed(self):
        r1 = run_analysis(
            n_simulations=3,
            time_horizon=5,
            seed=123,
            compare_uninsured=False,
        )
        r2 = run_analysis(
            n_simulations=3,
            time_horizon=5,
            seed=123,
            compare_uninsured=False,
        )
        for a, b in zip(r1.insured_results, r2.insured_results):
            np.testing.assert_array_equal(a.equity, b.equity)

    def test_custom_parameters(self):
        results = run_analysis(
            initial_assets=5_000_000,
            operating_margin=0.10,
            loss_frequency=1.0,
            loss_severity_mean=500_000,
            loss_severity_std=200_000,
            deductible=100_000,
            coverage_limit=5_000_000,
            premium_rate=0.03,
            n_simulations=3,
            time_horizon=5,
            seed=99,
            growth_rate=0.03,
            tax_rate=0.21,
            compare_uninsured=False,
        )
        assert len(results.insured_results) == 3

    def test_default_severity_std_equals_mean(self):
        """When loss_severity_std is omitted it defaults to loss_severity_mean."""
        results = run_analysis(
            loss_severity_mean=2_000_000,
            n_simulations=2,
            time_horizon=3,
            seed=0,
            compare_uninsured=False,
        )
        assert len(results.insured_results) == 2

    def test_default_severity_std_logs_info(self, caplog):
        """Omitting loss_severity_std emits an info-level log (#463)."""
        import logging

        with caplog.at_level(logging.INFO, logger="ergodic_insurance._run_analysis"):
            run_analysis(
                loss_severity_mean=1_000_000,
                n_simulations=2,
                time_horizon=3,
                seed=0,
                compare_uninsured=False,
            )
        assert any(
            "loss_severity_std not provided" in rec.message and "CV=1.0" in rec.message
            for rec in caplog.records
        )

    def test_explicit_severity_std_no_warning(self, caplog):
        """Providing loss_severity_std explicitly should NOT emit the default log."""
        import logging

        with caplog.at_level(logging.INFO, logger="ergodic_insurance._run_analysis"):
            run_analysis(
                loss_severity_mean=1_000_000,
                loss_severity_std=500_000,
                n_simulations=2,
                time_horizon=3,
                seed=0,
                compare_uninsured=False,
            )
        assert not any("loss_severity_std not provided" in rec.message for rec in caplog.records)

    def test_config_preserved(self):
        results = run_analysis(
            initial_assets=7_000_000,
            n_simulations=2,
            time_horizon=3,
            seed=0,
            compare_uninsured=False,
        )
        assert results.config.manufacturer.initial_assets == 7_000_000

    def test_insurance_policy_preserved(self):
        results = run_analysis(
            deductible=250_000,
            coverage_limit=8_000_000,
            premium_rate=0.02,
            n_simulations=2,
            time_horizon=3,
            seed=0,
            compare_uninsured=False,
        )
        assert results.insurance_policy.deductible == 250_000
        assert len(results.insurance_policy.layers) == 1
        assert results.insurance_policy.layers[0].limit == 8_000_000


# ---------------------------------------------------------------------------
# AnalysisResults method tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_results():
    """Run a small analysis once and reuse across the method tests."""
    return run_analysis(
        n_simulations=10,
        time_horizon=10,
        seed=42,
        compare_uninsured=True,
    )


class TestSummary:
    """Tests for AnalysisResults.summary()."""

    def test_returns_string(self, sample_results):
        s = sample_results.summary()
        assert isinstance(s, str)

    def test_contains_key_sections(self, sample_results):
        s = sample_results.summary()
        assert "Insured Scenario" in s
        assert "Uninsured Scenario" in s
        assert "Ergodic Advantage" in s
        assert "Survival Rate" in s

    def test_summary_caching(self, sample_results):
        s1 = sample_results.summary()
        s2 = sample_results.summary()
        assert s1 is s2  # same object, not just equal


class TestToDataFrame:
    """Tests for AnalysisResults.to_dataframe()."""

    def test_returns_dataframe(self, sample_results):
        df = sample_results.to_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self, sample_results):
        df = sample_results.to_dataframe()
        expected = len(sample_results.insured_results) + len(sample_results.uninsured_results)
        assert len(df) == expected

    def test_expected_columns(self, sample_results):
        df = sample_results.to_dataframe()
        for col in [
            "scenario",
            "simulation",
            "survived",
            "final_assets",
            "final_equity",
            "mean_roe",
            "time_weighted_roe",
            "total_claims",
        ]:
            assert col in df.columns, f"Missing column: {col}"

    def test_scenarios_labeled(self, sample_results):
        df = sample_results.to_dataframe()
        assert set(df["scenario"].unique()) == {"insured", "uninsured"}

    def test_insured_only_dataframe(self):
        results = run_analysis(
            n_simulations=3,
            time_horizon=3,
            seed=0,
            compare_uninsured=False,
        )
        df = results.to_dataframe()
        assert set(df["scenario"].unique()) == {"insured"}
        assert len(df) == 3


class TestPlot:
    """Tests for AnalysisResults.plot()."""

    def test_plot_returns_figure(self, sample_results):
        import matplotlib

        matplotlib.use("Agg")
        fig = sample_results.plot(show=False)
        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_insured_only(self):
        import matplotlib

        matplotlib.use("Agg")
        results = run_analysis(
            n_simulations=3,
            time_horizon=3,
            seed=0,
            compare_uninsured=False,
        )
        fig = results.plot(show=False)
        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_simulation(self):
        results = run_analysis(
            n_simulations=1,
            time_horizon=3,
            seed=42,
            compare_uninsured=True,
        )
        assert len(results.insured_results) == 1
        assert len(results.uninsured_results) == 1
        s = results.summary()
        assert "1/1" in s or "100.0%" in s

    def test_no_seed(self):
        """Running without seed should not error (non-reproducible)."""
        results = run_analysis(
            n_simulations=2,
            time_horizon=3,
            seed=None,
            compare_uninsured=False,
        )
        assert len(results.insured_results) == 2

    def test_zero_deductible(self):
        results = run_analysis(
            deductible=0,
            coverage_limit=10_000_000,
            n_simulations=2,
            time_horizon=3,
            seed=0,
            compare_uninsured=False,
        )
        assert results.insurance_policy.deductible == 0

    def test_under_ten_lines(self):
        """Acceptance criterion: complete comparison in <10 lines."""
        # This is exactly the API promised in the issue.
        from ergodic_insurance import (  # noqa: F811  # pylint: disable=no-name-in-module,reimported
            run_analysis as ra,
        )

        results = ra(
            initial_assets=10_000_000,
            operating_margin=0.08,
            loss_frequency=2.5,
            loss_severity_mean=1_000_000,
            deductible=500_000,
            coverage_limit=10_000_000,
            premium_rate=0.025,
            n_simulations=5,
            time_horizon=5,
        )
        text = results.summary()
        assert "Ergodic Advantage" in text
