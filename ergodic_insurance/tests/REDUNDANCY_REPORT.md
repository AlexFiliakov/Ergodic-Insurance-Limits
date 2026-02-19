# Test Suite Redundancy Report

**Date:** 2026-02-17
**Branch:** tests/refactor-tests
**Analyst:** redundancy-analyst

## Executive Summary

Systematic analysis of all 192 test files (~116,768 lines, ~5,723 tests) identified and consolidated redundant tests. The test suite was generally well-designed, with coverage companion files (`*_coverage.py`) targeting distinct uncovered lines rather than duplicating existing tests.

**Key finding**: Most `test_X.py` + `test_X_coverage.py` pairs are complementary, not duplicative. The main redundancy clusters are: (1) copy-paste duplicates where coverage files re-test functionality already covered in the primary test file, (2) groups of 3+ tests following identical patterns that should be parametrized, and (3) visualization formatting tests duplicated across multiple files.

| Category | Clusters Found | Tests Consolidated/Removed |
|----------|---------------:|---------------------------:|
| Copy-paste duplicates (main vs coverage files) | 6 | ~25 removed |
| Parametrize candidates (3+ same-pattern tests) | 8 | ~30 consolidated |
| Visualization formatting duplicates | 2 | ~6 removed |
| Visualization comprehensive duplicates | 1 | ~23 removed |
| Coverage gaps batch parametrize | 4 | ~30 consolidated |
| Coverage companion parametrize | 5 | ~25 consolidated |
| **Total** | **26** | **~139** |

**Net result across all changes**: 1,113 lines removed, 883 lines added = **230 net lines reduced** while preserving all test intent and edge case coverage.

---

## Phase 1: Insurance Module Consolidation (Previous Iteration)

### 1.1 Copy-Paste Duplicates Removed

#### `test_insurance_program_coverage.py` :: `TestInsuranceProgramSimpleCoverage`
**Duplicate of:** `test_insurance_program.py` :: `TestInsuranceProgramSimple`

Both classes tested `InsuranceProgram.simple()` with structurally identical tests:
- `test_creates_single_layer` / `test_creates_single_layer_program`
- `test_attachment_equals_deductible` / `test_layer_attachment_equals_deductible`
- `test_layer_has_zero_reinstatements` / `test_no_reinstatements`
- `test_respects_name_kwarg` / `test_custom_name`
- `test_forwards_extra_kwargs` / `test_kwargs_forwarded`
- `test_zero_deductible` / `test_zero_deductible`

**Action:** Removed `TestInsuranceProgramSimpleCoverage` (6 tests). The one extra assertion (`isinstance(..., EnhancedInsuranceLayer)`) was merged into the main test.

#### `test_insurance_program_coverage.py` :: `TestInsuranceProgramCalculatePremiumCoverage`
**Duplicate of:** `test_insurance_program.py` :: `TestInsuranceProgramCalculatePremium`

**Action:** Removed `TestInsuranceProgramCalculatePremiumCoverage` (3 tests).

#### `test_insurance_program_coverage.py` :: `TestCreateStandardProgram`
**Duplicate of:** `test_insurance_program.py` :: `TestInsuranceProgram.test_standard_manufacturing_program`

**Action:** Removed `TestCreateStandardProgram` (1 test).

#### `test_insurance_coverage.py` :: `TestInsurancePolicyFromYaml`
**Duplicate of:** `test_insurance.py` :: `TestInsurancePolicyYAML.test_load_from_yaml`

**Action:** Removed `TestInsurancePolicyFromYaml` (1 test). Cleaned up unused imports.

#### `test_insurance_coverage.py` :: `test_empty_layers_returns_zero`
**Duplicate of:** `test_insurance.py` :: `TestInsurancePolicy.test_empty_policy`

**Action:** Removed `test_empty_layers_returns_zero` (1 test).

### 1.2 Insurance Parametrize Consolidations

- `test_insurance.py` :: `test_calculate_recovery_*` (4 tests -> 1 parametrized)
- `test_insurance.py` :: `TestFromSimple` (4 tests -> 1 combined)
- `test_insurance.py` :: `TestOverRecoveryGuard` (loop -> parametrize with 5 IDs)
- `test_insurance_program.py` :: `test_round_attachment_point` (6 asserts + 4 coverage tests -> 1 parametrized with 12 cases)
- `test_insurance_program.py` :: `TestInsuranceProgramSimple` (7 tests -> 2)
- `test_insurance_program_coverage.py` :: Lookup-table tests (5+5 tests -> 2 parametrized)
- `test_insurance_program.py` :: `test_recovery_never_exceeds_claim` (loop -> parametrize with 7 IDs)
- `test_insurance_program.py` :: `test_calculate_layer_loss` (3 inline cases -> parametrize)

---

## Phase 2: Visualization Duplicate Removal (Current Iteration)

### 2.1 `test_visualization_comprehensive.py` Cleanup

**Problem**: `test_visualization_comprehensive.py` contained tests identical to those in `test_visualization_factory.py`, both testing the same `FigureFactory` and `StyleManager` classes from `visualization_infra`.

**Action**: Removed 23 duplicate tests:
- 8 `TestFigureFactory` methods (test_create_figure, test_create_subplots, test_create_line_plot, test_create_bar_plot, test_create_scatter_plot, test_create_histogram, test_create_heatmap, test_save_figure) -- kept in `test_visualization_factory.py`
- 9 `TestStyleManager` methods (test_get_theme_config, test_update_colors, test_update_fonts, test_create_style_sheet, test_inherit_from, test_get_colors, test_get_fonts, test_get_figure_config, test_get_grid_config)
- 5 `test_get_figsize_*` tests consolidated into 1 parametrized test
- 3 `test_get_dpi_*` tests consolidated into 1 parametrized test

**Lines saved**: ~200 lines

---

## Phase 3: Coverage Gaps Batch Files Consolidation (Current Iteration)

### 3.1 `test_coverage_gaps_batch1.py`
- `TestInsuranceRecoveryDecimalConversion`: 3 type tests -> 1 parametrized (float/int/decimal)
- `TestInsuranceAccountingDecimalConversion`: 5 tests -> 2 parametrized + 1 unique
- `TestStatisticalTestsInvalidAlternative`: 3 tests -> 1 parametrized (ks/anderson/shapiro)
- `TestHierarchicalAggregatorLeafTypes`: 4 leaf tests -> 1 parametrized + 2 unique

### 3.2 `test_coverage_gaps_batch2.py`
- `TestMeetsRequirements100KSpecialChecks`: 3 tests -> 1 parametrized (slow/high_memory/low_accuracy)
- `TestLoadResultsNonDataFrame`: 2 HDF5 tests -> 1 parametrized

### 3.3 `test_coverage_gaps_batch4.py`
- ROE recommendations: 8 tests -> 3 parametrized + 1 standalone + 2 parametrized
- Risk recommendations: 5 tests -> 1 parametrized + 2 standalone
- Comprehensive recommendations: 8 tests -> 1 parametrized + 1 standalone
- Excel reporter fallbacks: 4 tests -> 2 parametrized

**Lines saved**: ~150 lines across batch files

---

## Phase 4: Coverage Companion Files Parametrization (Current Iteration)

### 4.1 `test_bootstrap_analysis_coverage.py`
- `TestBootstrapWorkerStringStatistics`: 5 valid statistic tests -> 1 parametrized (mean/median/std/var/callable)

### 4.2 `test_safe_pickle_coverage.py`
- `TestRestrictedUnpicklerFindClass`:
  - 4 blocked OS/subprocess tests -> 1 parametrized (os_system/subprocess_popen/nt_system/posix_system)
  - 6 blocked builtins tests -> 1 parametrized (exec/eval/getattr/import/open/compile)
  - 3 allowed builtins tests -> 1 parametrized (dict/set/int)

### 4.3 `test_reporting_coverage.py`
- 4 NaN formatting tests -> 1 parametrized (currency/percentage/number/ratio)
- 2 scientific notation tests -> 1 parametrized (large/small)
- 2 currency abbreviation tests -> 1 parametrized (billion/thousand)
- 5 `_get_unit` assertions -> 1 parametrized test with 5 cases
- 6 `validate_results_data` invalid tests -> 1 parametrized with 6 cases

### 4.4 `test_visualization_gaps_coverage.py`
- 5 `_get_preferred_quadrant` tests -> 1 parametrized (sw_to_ne/se_to_nw/nw_to_se/ne_to_sw/center)

---

## Phase 5: Analysis Results (No Action Needed)

### 5.1 Monte Carlo Test Cluster (6 files, 4,723 lines, 178 tests)

| File | Lines | Tests | Classes |
|------|------:|------:|--------:|
| `test_monte_carlo.py` | 1,332 | 48 | 6 |
| `test_monte_carlo_coverage.py` | 1,668 | 69 | 21 |
| `test_monte_carlo_extended.py` | 442 | 18 | 1 |
| `test_monte_carlo_worker_config.py` | 505 | 22 | 8 |
| `test_monte_carlo_parallel.py` | 581 | 14 | 3 |
| `test_monte_carlo_trajectory_storage.py` | 195 | 7 | 1 |

**Finding**: Zero overlapping test names between files. Each tests distinct functionality. No consolidation needed.

### 5.2 Fixture Duplication Analysis

Most-shared fixture names:

| Fixture | Definitions | Verdict |
|---------|------------|---------|
| `manufacturer` | 48 | Different configs per test class -- NOT duplicates |
| `config` | 21 | Different module configs (ManufacturerConfig, BatchConfig, etc.) |
| `engine` | 15 | Different DecisionEngine setups per edge case test |
| `loss_generator` | 11 | Different distribution parameters per test |
| `analyzer` | 8 | Different ErgodicAnalyzer configs |
| `diagnostics` | 6 | Same simple `AdvancedConvergenceDiagnostics()` -- too low-impact to share |

**Finding**: Fixture "duplication" is actually "reuse of names with different implementations." The conftest.py hierarchy properly shares common fixtures via `integration/test_fixtures.py`. No consolidation needed.

### 5.3 Insurance Test Files (`test_insurance.py` vs `test_insurance_program.py`)
Initially appeared to have 8 overlapping test names. Investigation revealed they test **different classes**:
- `test_insurance.py`: Legacy `InsurancePolicy` / `InsuranceLayer` API
- `test_insurance_program.py`: New `InsuranceProgram` / `EnhancedInsuranceLayer` API

**Finding**: NOT redundant. Same test patterns applied to distinct APIs.

### 5.4 Coverage Companion Files Overlap Check

| Primary + Coverage Pair | Overlap | Verdict |
|------------------------|---------|---------|
| `test_visualization_comprehensive.py` + `test_visualization_factory.py` | 12 tests | Addressed (Phase 2) |
| `test_ledger.py` + `test_ledger_coverage.py` | 1 test | Different logic -- kept both |
| All other pairs | 0 tests | Well-designed, complementary |

---

## Verification

All modified test files pass:
```
532 passed, 4 skipped, 6 warnings in 51.03s
```

The 4 skips are Unix-only permission tests on Windows. The warnings are standard matplotlib/numpy deprecation notices.

---

## Files Modified (Complete List)

| File | Phase | Change |
|------|-------|--------|
| `test_visualization_comprehensive.py` | 2 | Removed 23 duplicates, parametrized 8 tests |
| `test_coverage_gaps_batch1.py` | 3 | Parametrized 4 test groups |
| `test_coverage_gaps_batch2.py` | 3 | Parametrized 2 test groups |
| `test_coverage_gaps_batch4.py` | 3 | Parametrized 4 test groups |
| `test_bootstrap_analysis_coverage.py` | 4 | Parametrized 1 test group |
| `test_safe_pickle_coverage.py` | 4 | Parametrized 3 test groups |
| `test_reporting_coverage.py` | 4 | Parametrized 5 test groups |
| `test_visualization_gaps_coverage.py` | 4 | Parametrized 1 test group |
| `test_insurance.py` | 1 | Parametrized + consolidated |
| `test_insurance_coverage.py` | 1 | Removed duplicates |
| `test_insurance_program.py` | 1 | Parametrized + consolidated |
| `test_insurance_program_coverage.py` | 1 | Removed duplicates + parametrized |

---

## Files NOT Modified (Conservative Decisions)

- All integration tests in `ergodic_insurance/tests/integration/` preserved
- All `_coverage.py` files that are complementary preserved unchanged
- All bug-specific test files (`test_*_bug*.py`, `test_issue_*.py`) preserved
- All actuarial/statistical tests preserved (may appear redundant but test different numerical edge cases)
- Fixture duplications left as-is (benign, context-appropriate)
- Decision engine, config, convergence, and Monte Carlo test files preserved (no duplicates found)

---

## Remaining Opportunities (Low Priority)

1. **More parametrize candidates in `test_visualization_gaps_coverage.py`**: Several groups of 3-8 similar tests could be parametrized, but each has unique setup logic with minimal benefit.
2. **For-loop test patterns**: Some files use `for obj in test_objects:` patterns that could use `@pytest.mark.parametrize` for better error reporting, but this is style preference.
3. **`diagnostics` fixture**: Defined identically in 6 convergence test files. Could be moved to conftest.py but scope is narrow.
