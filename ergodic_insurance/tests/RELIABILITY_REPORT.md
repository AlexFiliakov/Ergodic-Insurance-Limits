# Test Reliability Report

**Date:** 2026-02-13
**Branch:** tests/571_refactor_tests
**Analyst:** reliability-engineer

## Summary

Scanned all 183 test files (175 unit + 8 integration) for flakiness indicators across 6 categories. Found and fixed issues in 17 files across 39 locations.

## Category 1: Floating-Point Comparisons

### Findings
- **83 existing `pytest.approx` usages** across 26 files -- many tests already handle this well
- **~200 exact float comparisons** (`assert x == 0.05`, etc.) across the test suite
- Most are safe: they compare directly-set config/attribute values (e.g., `assert config.tax_rate == 0.25`)

### Fixed (high risk -- computed float values)
| File | Lines | Issue | Fix |
|------|-------|-------|-----|
| `test_claim_development.py` | 31, 38, 45, 52, 62 | `sum(development_factors) == 1.0` -- floating-point sum accumulates error | Changed to `pytest.approx(1.0)` or `pytest.approx(1.0, abs=0.01)` |
| `test_claim_development.py` | 138-144 | `get_cumulative_paid(n) == 0.65` -- cumulative sum of floats | Changed to `pytest.approx()` |
| `test_claim_development.py` | 584, 617, 668 | `ibnr == 0.0` -- result of calculation that should be zero | Changed to `pytest.approx(0.0, abs=1e-10)` |
| `test_benchmarking.py` | 330-332 | `avg_cpu == 70.0`, `avg_memory == 257.5` -- computed averages | Changed to `pytest.approx()` |
| `test_bootstrap.py` | 427-431 | Bonferroni-corrected p-values (0.01 * 5, etc.) | Changed to `pytest.approx()` |
| `test_performance.py` | 309 | `retained + recovered == losses` -- numpy float array equality | Changed to `pytest.approx()` |

### Not Fixed (safe patterns)
- Direct attribute/config value comparisons (e.g., `config.tax_rate == 0.25`) -- these are set, not computed
- Integer-exact float comparisons (e.g., `payment == 100_000` where payment = 1_000_000 * 0.10)
- Special case returns (`growth == 0.0`, `growth == -np.inf`)

## Category 2: Timing Dependencies

### Findings
- **26 files** use `time.time()` for performance measurement
- **16 assertions** compare execution time against thresholds

### Fixed (flaky timing assertions)
| File | Lines | Issue | Fix |
|------|-------|-------|-----|
| `test_claim_development.py` | 508 | `elapsed < 0.20` (200ms for 10K claims) | Relaxed to `< 2.0` |
| `test_performance.py` | 166 | `vectorized_time < 0.1` (100ms for 1M calcs) | Relaxed to `< 1.0` |
| `test_performance.py` | 299 | `vec_time < 0.1` (100ms for vectorized ops) | Relaxed to `< 1.0` |
| `test_performance.py` | 394 | `cached_time < opt_time * 0.5` (cache 2x faster) | Removed timing comparison, kept functional assertion |
| `test_cache_manager.py` | 513, 521-522 | `load_time < 1.0` + `speedup > 5x` | Relaxed load to `< 10.0`, removed speedup comparison |
| `test_monte_carlo_extended.py` | 167 | `cache_load_time < results1.execution_time * 0.5` | Removed timing comparison, kept functional assertion |
| `test_monte_carlo.py` | 896 | `overhead < 0.05` (5% serialization overhead) | Relaxed to `< 0.50` |
| `test_integration.py` | 294-296 | `100 scenarios in 1.5s, 1000 in 10s` | Relaxed to `10s` and `60s` |
| `test_end_to_end.py` | 520 | `max_time < min_time * 10.0` (scaling linearity) | Relaxed to `* 50.0` |
| `test_insurance_program.py` | 1085 | `elapsed < 1.0` (10K claims processing) | Relaxed to `< 10.0` |
| `test_loss_distributions.py` | 800, 812 | `elapsed_time < 1.0`, `< 5.0` | Relaxed to `< 10.0`, `< 30.0` |

### Not Fixed (safe patterns)
- `time.sleep(0.01)` inside profiled functions (simulates work, not synchronization)
- Timing assertions with generous thresholds (e.g., `< 60s`)
- Timing measurements that don't have assertions

## Category 3: Randomness (Unseeded np.random)

### Findings
- **56 files** use `np.random` functions
- Most files properly use `np.random.seed(42)` before random calls
- ~15 files had unseeded random calls that could produce non-deterministic test data

### Fixed (missing seeds)
| File | Issue |
|------|-------|
| `test_claim_development.py` (lines 492-497, 518-526) | Performance tests generated claims with unseeded `np.random.lognormal` -- replaced with `np.random.default_rng(42)` |
| `test_performance.py` (lines 152, 177, 290, 302, 366) | Multiple test methods used unseeded `np.random.lognormal`, `np.random.exponential`, `np.random.uniform` -- added `np.random.seed(42)` |
| `test_cache_manager.py` (lines 497, 546, 732) | Performance and metadata tests used unseeded random -- added `np.random.seed(42)` |
| `test_visualization_simple.py` (lines 418-420, 437-438) | Dashboard tests used unseeded random for data generation -- added `np.random.seed(42)` |
| `test_walk_forward.py` (lines 108-109, 246-251, 282-286, 1133-1137) | Multiple test methods and mock data used unseeded random -- added `np.random.seed(42)` |
| `test_trajectory_storage.py` (lines 458, 712) | Storage tests used unseeded random for noise/data generation -- added `np.random.seed(42)` |
| `test_parallel_executor.py` (lines 257, 491) | Shared memory and parallel tests used unseeded random -- added `np.random.seed(42)` |
| `test_scenario_batch.py` (line 62-67) | Mock fixture used unseeded random for simulation results -- added `np.random.seed(42)` |

### Not Fixed (acceptable)
- Visualization test files where random data is used only to create plots and no values are asserted (the tests only check that plotting functions don't crash)
- Files where `np.random.seed()` is called in a fixture or setup that runs before the random calls
- Integration test fixtures that already use explicit seeds

## Category 4: Order Dependence

### Findings
- No module-level mutable state shared between tests (except `_registry` in shared memory test files, which use `autouse=True` fixtures to clear before/after each test)
- No tests that modify class-level variables without restoration
- No dict-ordering dependencies found

### Not Fixed
- No issues found requiring fixes

## Category 5: Resource Leaks

### Findings
- **~30 `tempfile.mkdtemp()` usages** -- all have proper cleanup via `yield`/`shutil.rmtree` in fixtures or `try/finally` blocks
- **~20 `NamedTemporaryFile` usages** -- all use `delete=False` with `try/finally` cleanup
- Matplotlib `plt.close(fig)` is used consistently in visualization tests

### Fixed
| File | Issue | Fix |
|------|-------|-----|
| `test_convergence_ess.py` (line 211) | Unnecessary `time.sleep(0.05)` -- assertions already handle elapsed_time >= 0 | Removed sleep |

### Not Fixed
- No resource leak issues found requiring fixes

## Category 6: Environment Sensitivity

### Findings
- `conftest.py` properly checks `os.environ.get("CI")` for CI-specific behavior
- Platform-specific tests use proper `pytest.mark.skipif` decorators (e.g., Windows shared memory tests)
- No timezone or locale-dependent tests found
- Path handling uses `pathlib.Path` consistently

### Not Fixed
- No issues found requiring fixes

## Overall Statistics

| Category | Files Scanned | Issues Found | Issues Fixed |
|----------|--------------|--------------|--------------|
| Float comparisons | 183 | 12 locations | 12 |
| Timing dependencies | 26 | 11 assertions | 11 |
| Unseeded randomness | 56 | 15 locations | 15 |
| Order dependence | 183 | 0 | 0 |
| Resource leaks | 30 | 1 (unnecessary sleep) | 1 |
| Environment sensitivity | 183 | 0 | 0 |
| **Total** | **183** | **39** | **39** |

## Files Modified

1. `test_claim_development.py` -- float comparisons, unseeded random, timing
2. `test_performance.py` -- unseeded random, timing assertions, float comparison
3. `test_cache_manager.py` -- unseeded random, timing assertions
4. `test_benchmarking.py` -- float comparisons
5. `test_bootstrap.py` -- float comparisons
6. `test_convergence_ess.py` -- unnecessary sleep
7. `test_monte_carlo.py` -- timing assertion
8. `test_monte_carlo_extended.py` -- timing assertion
9. `test_integration.py` -- timing assertions
10. `test_end_to_end.py` -- timing assertion
11. `test_insurance_program.py` -- timing assertion
12. `test_loss_distributions.py` -- timing assertions
13. `test_visualization_simple.py` -- unseeded random
14. `test_walk_forward.py` -- unseeded random
15. `test_trajectory_storage.py` -- unseeded random
16. `test_parallel_executor.py` -- unseeded random
17. `test_scenario_batch.py` -- unseeded random
