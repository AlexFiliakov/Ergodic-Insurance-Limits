# Test Suite Redundancy Report

**Date:** 2026-02-13
**Branch:** tests/571_refactor_tests
**Analyst:** redundancy-analyst

## Executive Summary

Systematic analysis of all 183 test files (~109,400 lines of test code) identified and consolidated redundant tests across the insurance core module group and other areas. The `_coverage.py` files generally target distinct uncovered lines and are NOT duplicates of their primary test files, with specific exceptions noted below.

**Key finding**: Most `test_X.py` + `test_X_coverage.py` pairs are complementary, not duplicative. The main redundancy clusters are: (1) copy-paste duplicates where coverage files re-test functionality already covered in the primary test file, (2) groups of 3+ tests following identical patterns that should be parametrized, and (3) visualization formatting tests duplicated across multiple files.

| Category | Clusters Found | Tests Consolidated/Removed |
|----------|---------------:|---------------------------:|
| Copy-paste duplicates (main vs coverage files) | 6 | ~25 removed |
| Parametrize candidates (3+ same-pattern tests) | 8 | ~30 consolidated |
| Visualization formatting duplicates | 2 | ~6 removed |
| **Total** | **16** | **~61** |

---

## 1. Copy-Paste Duplicates Removed (Implemented)

### 1.1 `test_insurance_program_coverage.py` :: `TestInsuranceProgramSimpleCoverage`
**Duplicate of:** `test_insurance_program.py` :: `TestInsuranceProgramSimple`

Both classes tested `InsuranceProgram.simple()` with structurally identical tests:
- `test_creates_single_layer` / `test_creates_single_layer_program`
- `test_attachment_equals_deductible` / `test_layer_attachment_equals_deductible`
- `test_layer_has_zero_reinstatements` / `test_no_reinstatements`
- `test_respects_name_kwarg` / `test_custom_name`
- `test_forwards_extra_kwargs` / `test_kwargs_forwarded`
- `test_zero_deductible` / `test_zero_deductible`

**Action:** Removed `TestInsuranceProgramSimpleCoverage` (6 tests). The one extra assertion (`isinstance(..., EnhancedInsuranceLayer)`) was merged into the main test.

### 1.2 `test_insurance_program_coverage.py` :: `TestInsuranceProgramCalculatePremiumCoverage`
**Duplicate of:** `test_insurance_program.py` :: `TestInsuranceProgramCalculatePremium`

Both test the `calculate_premium()` alias with:
- Premium-is-alias/matches-annual test (identical logic)
- Premium correct value (same pattern, different amounts)
- Empty layers returns 0 (identical)

**Action:** Removed `TestInsuranceProgramCalculatePremiumCoverage` (3 tests).

### 1.3 `test_insurance_program_coverage.py` :: `TestCreateStandardProgram`
**Duplicate of:** `test_insurance_program.py` :: `TestInsuranceProgram.test_standard_manufacturing_program`

Both test `create_standard_manufacturing_program()` checking deductible=250K, name, and 4 layers.

**Action:** Removed `TestCreateStandardProgram` (1 test).

### 1.4 `test_insurance_program_coverage.py` :: `TestGetTotalCoverageEmpty`
**Duplicate of:** (new) `test_insurance_program.py` :: `TestInsuranceProgram.test_get_total_coverage_empty_layers`

**Action:** Merged into main file, removed from coverage file (1 test).

### 1.5 `test_insurance_coverage.py` :: `TestInsurancePolicyFromYaml`
**Duplicate of:** `test_insurance.py` :: `TestInsurancePolicyYAML.test_load_from_yaml`

Both create a YAML config, load it via `from_yaml`, assert 2 layers and deductible.

**Action:** Removed `TestInsurancePolicyFromYaml` (1 test). Cleaned up unused `yaml`, `os`, `tempfile` imports.

### 1.6 `test_insurance_coverage.py` :: `TestInsurancePolicyGetTotalCoverage.test_empty_layers_returns_zero`
**Duplicate of:** `test_insurance.py` :: `TestInsurancePolicy.test_empty_policy`

Both test empty layers return 0 coverage.

**Action:** Removed `test_empty_layers_returns_zero` (1 test). Kept `test_multi_layer_coverage` which tests a unique scenario.

### 1.7 `test_insurance_program_coverage.py` :: `TestRoundAttachmentPointEdgeCases`
**Merged with:** `test_insurance_program.py` :: `TestInsuranceProgramOptimization.test_round_attachment_point`

Both tested `_round_attachment_point` with overlapping values across different ranges.

**Action:** Merged all cases into a single parametrized test in the main file (12 cases). Removed `TestRoundAttachmentPointEdgeCases` from coverage file.

---

## 2. Parametrize Consolidations (Implemented)

### 2.1 `test_insurance.py` :: `TestInsuranceLayer.test_calculate_recovery_*` (4 tests -> 1 parametrized)

**Before:** 4 separate test functions, each creating the same layer and testing one loss amount:
```python
def test_calculate_recovery_below_attachment(self): ...  # loss=500K, expected=0
def test_calculate_recovery_within_layer(self): ...      # loss=3M, expected=2M
def test_calculate_recovery_exceeds_layer(self): ...     # loss=10M, expected=5M
def test_calculate_recovery_at_attachment(self): ...     # loss=1M, expected=0
```

**After:**
```python
@pytest.mark.parametrize("loss,expected", [
    pytest.param(500_000, 0.0, id="below-attachment"),
    pytest.param(1_000_000, 0.0, id="at-attachment"),
    pytest.param(3_000_000, 2_000_000, id="within-layer"),
    pytest.param(10_000_000, 5_000_000, id="exceeds-layer"),
])
def test_calculate_recovery(self, loss, expected): ...
```

### 2.2 `test_insurance.py` :: `TestFromSimple` (4 tests -> 1 combined)

`test_creates_single_layer_policy`, `test_deductible_set_correctly`, `test_layer_attachment_equals_deductible`, `test_layer_limit_and_rate` all created the exact same policy and asserted one property. Consolidated into `test_from_simple_structure` with all assertions together.

### 2.3 `test_insurance.py` :: `TestOverRecoveryGuard.test_recovery_never_exceeds_claim` (loop -> parametrize)

Converted `for claim in [100K, 500K, 1M, 5M, 15M]` loop to `@pytest.mark.parametrize` with 5 descriptive IDs.

### 2.4 `test_insurance_program.py` :: `test_round_attachment_point` (1 test with 6 asserts + 4 tests in coverage -> 1 parametrized with 12 cases)

Merged all rounding test cases from both files into a single parametrized test covering all ranges: `below-100k-small`, `below-100k-round-10k`, `below-100k-round-up`, `100k-1m-round-50k-low`, `100k-1m-round-50k-mid`, `100k-1m-round-50k-high`, `100k-1m-round-50k-upper`, `1m-10m-round-250k`, `1m-10m-round-250k-mid`, `above-10m-round-1m`, `above-10m-round-1m-up`, `above-10m-round-1m-large`.

### 2.5 `test_insurance_program.py` :: `TestInsuranceProgramSimple` (7 tests -> 2)

Consolidated `test_creates_single_layer_program`, `test_deductible_set_correctly`, `test_layer_attachment_equals_deductible`, `test_layer_limit_and_rate`, `test_default_name`, `test_no_reinstatements` into `test_simple_structure`. Kept `test_custom_name` separate (tests a different code path).

### 2.6 `test_insurance_program_coverage.py` :: Lookup-table tests

`TestGetLayerCapacity` (5 tests -> 1 parametrized with 5 cases):
```python
@pytest.mark.parametrize("attachment,expected", [
    pytest.param(500_000, 5_000_000, id="below-1m"),
    pytest.param(5_000_000, 25_000_000, id="1m-10m"),
    pytest.param(25_000_000, 50_000_000, id="10m-50m"),
    pytest.param(75_000_000, 100_000_000, id="above-50m"),
    pytest.param(float("inf"), 100_000_000, id="infinity-fallback"),
])
def test_layer_capacity(self, attachment, expected): ...
```

`TestGetBasePremiumRate` (5 tests -> 1 parametrized with 5 cases):
```python
@pytest.mark.parametrize("attachment,expected", [
    pytest.param(500_000, 0.015, id="below-1m"),
    pytest.param(3_000_000, 0.010, id="1m-5m"),
    pytest.param(15_000_000, 0.006, id="5m-25m"),
    pytest.param(50_000_000, 0.003, id="above-25m"),
    pytest.param(float("inf"), 0.003, id="infinity-fallback"),
])
def test_base_premium_rate(self, attachment, expected): ...
```

### 2.7 `test_insurance_program.py` :: `test_recovery_never_exceeds_claim_various_amounts` (loop -> parametrize)

Converted `for claim in [0, 100K, 250K, 1M, 5M, 10M, 30M]` loop to `@pytest.mark.parametrize` with 7 descriptive IDs.

### 2.8 `test_insurance_program.py` :: `test_calculate_layer_loss` (3 in-test cases -> parametrize)

Converted inline assertions for below/within/exceeds cases into a parametrized test with 3 cases.

---

## 3. Visualization Formatting Duplicates (Previously Identified)

### 3.1 `test_visualization_simple.py` formatting tests
**Duplicate of:** `test_visualization.py` and `test_visualization_extended.py`

`format_currency`, `format_percentage`, `wsj_formatter`, and `set_wsj_style` tests in `test_visualization_simple.py` are strict subsets of the more comprehensive versions in the other files.

**Action taken (by previous iteration):** Removed `TestVisualizationFormatting` from `test_visualization_simple.py` with a note: "those tests are covered more thoroughly in test_visualization.py and test_visualization_extended.py."

---

## 4. Overlapping Coverage (NOT Modified -- Flagged for Review)

### 4.1 `TestOverRecoveryGuard` in both `test_insurance.py` and `test_insurance_program.py`

These are NOT duplicates despite testing the same concern (issue #310). They test different implementations:
- `test_insurance.py` tests `InsurancePolicy` (deprecated legacy class)
- `test_insurance_program.py` tests `InsuranceProgram` (current class)

**Recommendation:** Keep both until `InsurancePolicy` is fully removed.

### 4.2 `apply_pricing` error tests in `test_insurance_coverage.py` and `test_insurance_program_coverage.py`

Both test `pricing_not_enabled_raises` and `no_pricer_no_generator_raises`, but on different classes (`InsurancePolicy` vs `InsuranceProgram`).

**Recommendation:** Keep both - they test different implementations.

### 4.3 Fixture duplication: `manufacturer` defined ~40 times across test files

The `manufacturer` fixture is defined independently in ~40 test files, each creating a `WidgetManufacturer` with slightly different `ManufacturerConfig` parameters.

**Recommendation:** A shared `default_manufacturer` fixture could be added to conftest.py for the ~15 cases using identical defaults, but this risks coupling unrelated test files. Lower priority.

### 4.4 Coverage-gap batch files (`test_coverage_gaps_batch1-4.py`)

These 4 files (4,246 total lines) contain auto-generated tests to fill coverage gaps. They likely overlap with other test files, but removing them without detailed source-line coverage analysis is risky.

**Recommendation:** Flag for future coverage analysis. Run coverage on base test files first, then selectively remove batch tests that no longer contribute.

---

## 5. Coverage Files Analysis (Complementary, Not Redundant)

| Primary File | Coverage File | Verdict |
|-------------|--------------|---------|
| `test_batch_processor.py` | `test_batch_processor_coverage.py` | Complementary |
| `test_config_manager.py` | `test_config_manager_coverage.py` | Complementary |
| `test_monte_carlo.py` | `test_monte_carlo_coverage.py` | Complementary |
| `test_insurance_pricing.py` | `test_insurance_pricing_coverage.py` | Complementary |
| `test_insurance_program.py` | `test_insurance_program_coverage.py` | Mostly complementary (duplicates removed above) |
| `test_insurance.py` | `test_insurance_coverage.py` | Mostly complementary (duplicates removed above) |
| `test_ledger.py` | `test_ledger_coverage.py` | Complementary |
| `test_risk_metrics.py` | `test_risk_metrics_coverage.py` | Complementary |
| `test_ruin_probability.py` | `test_ruin_probability_coverage.py` | Complementary |
| `test_manufacturer.py` | `test_manufacturer_coverage.py` | Complementary |

---

## 6. Files Modified

| File | Change Type | Detail |
|------|-----------|--------|
| `test_insurance.py` | Parametrize + consolidate | 4 recovery tests -> 1 parametrized; 4 from_simple tests -> 1 combined; loop -> parametrize |
| `test_insurance_coverage.py` | Remove duplicates | Removed `TestInsurancePolicyFromYaml`, `test_empty_layers_returns_zero`; cleaned imports |
| `test_insurance_program.py` | Parametrize + consolidate | 7 simple tests -> 2; rounding tests merged+parametrized; layer_loss parametrized; loop -> parametrize; added empty coverage test |
| `test_insurance_program_coverage.py` | Remove duplicates + parametrize | Removed 4 classes (~25 tests); parametrized _get_layer_capacity and _get_base_premium_rate |

## 7. Verification

All modified files verified with `pytest -x`:
- `test_insurance.py`: 47 passed
- `test_insurance_coverage.py`: 13 passed
- `test_insurance_program.py`: 112 passed
- `test_insurance_program_coverage.py`: 52 passed
- **Total: 225 passed, 0 failed**

## 8. Files NOT Modified (Conservative Decisions)

- All integration tests in `ergodic_insurance/tests/integration/` preserved
- All `_coverage.py` files that are complementary preserved unchanged
- All bug-specific test files (`test_*_bug*.py`, `test_issue_*.py`) preserved
- All actuarial/statistical tests preserved (may appear redundant but test different numerical edge cases)
- Fixture duplications left as-is (benign, context-appropriate)
- Decision engine, config, convergence, and Monte Carlo test files preserved (no duplicates found)
