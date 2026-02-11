"""Tests for issue #380: FFT-based autocorrelation optimization.

Validates that the FFT-based _calculate_autocorrelation produces results
matching the original O(N*L) loop implementation within 1e-10, and that
the performance improvement is at least 3x for chains of length 25K+.
"""

import os
import time

import numpy as np
import pytest

from ergodic_insurance.convergence import ConvergenceDiagnostics


def _autocorrelation_loop(chain: np.ndarray, max_lag: int) -> np.ndarray:
    """Original O(N*L) loop implementation for reference comparison."""
    n = len(chain)
    chain = chain - np.mean(chain)
    c0 = np.dot(chain, chain) / n

    autocorr = np.zeros(max_lag + 1)
    autocorr[0] = 1.0

    for lag in range(1, min(max_lag + 1, n)):
        c_lag = np.dot(chain[:-lag], chain[lag:]) / n
        autocorr[lag] = c_lag / c0 if c0 > 0 else 0

    return autocorr


class TestFFTAutocorrelationAccuracy:
    """Verify FFT results match the loop implementation within 1e-10."""

    @pytest.fixture
    def diagnostics(self):
        return ConvergenceDiagnostics()

    def test_white_noise_matches(self, diagnostics):
        """FFT matches loop for white noise input."""
        rng = np.random.default_rng(42)
        chain = rng.standard_normal(10_000)
        max_lag = 100

        fft_result = diagnostics._calculate_autocorrelation(chain, max_lag)
        loop_result = _autocorrelation_loop(chain, max_lag)

        np.testing.assert_allclose(fft_result, loop_result, atol=1e-10)

    def test_ar1_process_matches(self, diagnostics):
        """FFT matches loop for autocorrelated AR(1) process."""
        rng = np.random.default_rng(123)
        n = 25_000
        phi = 0.9
        chain = np.zeros(n)
        chain[0] = rng.standard_normal()
        for t in range(1, n):
            chain[t] = phi * chain[t - 1] + rng.standard_normal()

        max_lag = 200

        fft_result = diagnostics._calculate_autocorrelation(chain, max_lag)
        loop_result = _autocorrelation_loop(chain, max_lag)

        np.testing.assert_allclose(fft_result, loop_result, atol=1e-10)

    def test_short_chain_matches(self, diagnostics):
        """FFT matches loop for a short chain."""
        rng = np.random.default_rng(7)
        chain = rng.standard_normal(50)
        max_lag = 10

        fft_result = diagnostics._calculate_autocorrelation(chain, max_lag)
        loop_result = _autocorrelation_loop(chain, max_lag)

        np.testing.assert_allclose(fft_result, loop_result, atol=1e-10)

    def test_constant_chain(self, diagnostics):
        """Zero-variance chain: lag-0 is 1.0, all others are 0."""
        chain = np.ones(500)
        max_lag = 10

        result = diagnostics._calculate_autocorrelation(chain, max_lag)

        assert result[0] == 1.0
        assert all(result[i] == 0 for i in range(1, max_lag + 1))

    def test_large_chain_matches(self, diagnostics):
        """FFT matches loop for a large chain (50K samples, 500 lags)."""
        rng = np.random.default_rng(999)
        chain = rng.standard_normal(50_000)
        max_lag = 500

        fft_result = diagnostics._calculate_autocorrelation(chain, max_lag)
        loop_result = _autocorrelation_loop(chain, max_lag)

        np.testing.assert_allclose(fft_result, loop_result, atol=1e-10)

    def test_max_lag_exceeds_chain_length(self, diagnostics):
        """Handles max_lag > chain length gracefully."""
        rng = np.random.default_rng(55)
        chain = rng.standard_normal(20)
        max_lag = 50

        fft_result = diagnostics._calculate_autocorrelation(chain, max_lag)
        loop_result = _autocorrelation_loop(chain, max_lag)

        # Both are truncated to min(max_lag+1, n) = 20
        assert len(fft_result) == len(chain)
        np.testing.assert_allclose(fft_result, loop_result[: len(fft_result)], atol=1e-10)


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Speedup benchmarks are flaky on shared CI runners due to variable CPU performance",
)
class TestFFTAutocorrelationPerformance:
    """Verify >= 3x speedup for chains of length 25K+."""

    @pytest.fixture
    def diagnostics(self):
        return ConvergenceDiagnostics()

    def test_speedup_25k_chain(self, diagnostics):
        """FFT is at least 3x faster than loop for 25K chain."""
        rng = np.random.default_rng(42)
        chain = rng.standard_normal(25_000)
        max_lag = min(25_000 // 4, 1000)

        # Warm up
        diagnostics._calculate_autocorrelation(chain, max_lag)
        _autocorrelation_loop(chain, max_lag)

        # Time FFT
        n_runs = 5
        start = time.perf_counter()
        for _ in range(n_runs):
            diagnostics._calculate_autocorrelation(chain, max_lag)
        fft_time = (time.perf_counter() - start) / n_runs

        # Time loop
        start = time.perf_counter()
        for _ in range(n_runs):
            _autocorrelation_loop(chain, max_lag)
        loop_time = (time.perf_counter() - start) / n_runs

        speedup = loop_time / fft_time
        assert speedup >= 3.0, (
            f"Expected >= 3x speedup, got {speedup:.1f}x "
            f"(FFT: {fft_time*1000:.1f}ms, loop: {loop_time*1000:.1f}ms)"
        )

    def test_speedup_100k_chain(self, diagnostics):
        """FFT is at least 3x faster than loop for 100K chain."""
        rng = np.random.default_rng(42)
        chain = rng.standard_normal(100_000)
        max_lag = min(100_000 // 4, 1000)

        # Warm up
        diagnostics._calculate_autocorrelation(chain, max_lag)
        _autocorrelation_loop(chain, max_lag)

        # Time FFT
        n_runs = 3
        start = time.perf_counter()
        for _ in range(n_runs):
            diagnostics._calculate_autocorrelation(chain, max_lag)
        fft_time = (time.perf_counter() - start) / n_runs

        # Time loop
        start = time.perf_counter()
        for _ in range(n_runs):
            _autocorrelation_loop(chain, max_lag)
        loop_time = (time.perf_counter() - start) / n_runs

        speedup = loop_time / fft_time
        assert speedup >= 3.0, (
            f"Expected >= 3x speedup, got {speedup:.1f}x "
            f"(FFT: {fft_time*1000:.1f}ms, loop: {loop_time*1000:.1f}ms)"
        )


class TestNoLoopOverLags:
    """Verify the implementation has no Python-level loop over lag values."""

    def test_no_loop_in_source(self):
        """The _calculate_autocorrelation method has no loop over lag values."""
        import ast
        import inspect
        import textwrap

        source = inspect.getsource(ConvergenceDiagnostics._calculate_autocorrelation)
        tree = ast.parse(textwrap.dedent(source))

        # Walk the AST looking for for-loops with 'lag' as the iteration variable
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                target = node.target
                if isinstance(target, ast.Name) and target.id == "lag":
                    pytest.fail(
                        "Implementation still contains a Python-level 'for lag in ...' loop"
                    )
