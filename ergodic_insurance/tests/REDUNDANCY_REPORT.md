# Test Suite Redundancy Report

**Date**: 2026-02-09
**Analyst**: redundancy-analyst
**Branch**: bugfix/360_fix_test_suite

## Executive Summary

After systematic analysis of all 164 test files (156 unit + 8 integration), this report identifies **redundant tests**, **parametrize candidates**, **fixture duplication**, and **overlapping coverage**. Overall, the `_coverage.py` files generally target distinct uncovered lines and are NOT duplicates of their primary test files. However, several clusters of redundancy exist.

**Key finding**: Most `test_X.py` + `test_X_coverage.py` pairs are complementary, not duplicative. The coverage files explicitly target specific uncovered line numbers and test different code paths. The main redundancy clusters are in the visualization formatting tests and fixture definitions.

## 1. Copy-Paste Duplicates

### 1A. Visualization Formatting Tests (REDUNDANT)

**Files involved**: `test_visualization_simple.py`, `test_visualization.py`, `test_visualization_extended.py`

The `format_currency` function is tested redundantly:

| Test | File | What it tests |
|------|------|---------------|
| `test_format_currency` | `test_visualization_simple.py:31` | `format_currency(1000)`, `format_currency(1000000)`, etc. |
| `test_format_currency_basic` | `test_visualization.py:117` | `format_currency(1000)`, `format_currency(1000.50)`, etc. |
| `test_format_currency_abbreviated` | `test_visualization.py:124` | Abbreviated format with K/M/B |
| `test_format_currency_abbreviate_large_numbers` | `test_visualization_extended.py:51` | Abbreviated format with K/M/B (overlaps with above) |
| `test_format_currency_edge_cases` | `test_visualization.py:133` | Edge cases (0, small values, negatives) |

The `format_percentage` function is tested redundantly:

| Test | File | What it tests |
|------|------|---------------|
| `test_format_percentage` | `test_visualization_simple.py:39` | `format_percentage(0.05)`, etc. |
| `test_format_percentage_basic` | `test_visualization.py:149` | Same inputs as above |
| `test_format_percentage_edge_cases` | `test_visualization.py:156` | Edge cases (0, small values) |

The `set_wsj_style` function is tested in both:
- `test_visualization_simple.py:63`
- `test_visualization.py:97`

The `WSJFormatter` is tested redundantly:
- `test_visualization_simple.py:47` (`test_wsj_formatter`) - currency/percentage/number basics
- `test_visualization_extended.py:69` (`test_wsj_formatter_currency_method_edge_cases`) - extends to trillions, negatives
- `test_visualization_extended.py:95` (`test_wsj_formatter_number_method_edge_cases`) - extends to large numbers

**Action**: Consolidate `test_visualization_simple.py::TestVisualizationFormatting` tests into `test_visualization.py`. The simple file's formatting tests are strict subsets. The extended file adds NEW edge cases, so those remain.

### 1B. `test_visualization_simple.py::test_set_wsj_style` vs `test_visualization.py::test_set_wsj_style`

Both call `set_wsj_style()` and check `plt.rcParams`. The `test_visualization.py` version is slightly more detailed (checks `axes.spines.right`). The simple version is redundant.

**Action**: Remove `test_set_wsj_style` from `test_visualization_simple.py`.

## 2. Parametrize Candidates

### 2A. Insurance Program Validation Tests

In `test_insurance_program.py::TestEnhancedInsuranceLayer::test_invalid_parameters` (line 47), four separate `pytest.raises` blocks test different validation errors. These could be a single parametrized test.

**Original** (4 assertions in 1 test):
```python
def test_invalid_parameters(self):
    with pytest.raises(ValueError, match="Attachment point must be non-negative"):
        EnhancedInsuranceLayer(attachment_point=-100, limit=1_000_000, base_premium_rate=0.01)
    with pytest.raises(ValueError, match="Limit must be positive"):
        EnhancedInsuranceLayer(attachment_point=0, limit=-1_000_000, base_premium_rate=0.01)
    with pytest.raises(ValueError, match="Base premium rate must be non-negative"):
        EnhancedInsuranceLayer(attachment_point=0, limit=1_000_000, base_premium_rate=-0.01)
    with pytest.raises(ValueError, match="Reinstatements must be non-negative"):
        EnhancedInsuranceLayer(attachment_point=0, limit=1_000_000, base_premium_rate=0.01, reinstatements=-1)
```

**Consolidated** (parametrized):
```python
@pytest.mark.parametrize("kwargs,match", [
    ({"attachment_point": -100, "limit": 1_000_000, "base_premium_rate": 0.01}, "Attachment point must be non-negative"),
    ({"attachment_point": 0, "limit": -1_000_000, "base_premium_rate": 0.01}, "Limit must be positive"),
    ({"attachment_point": 0, "limit": 1_000_000, "base_premium_rate": -0.01}, "Base premium rate must be non-negative"),
    ({"attachment_point": 0, "limit": 1_000_000, "base_premium_rate": 0.01, "reinstatements": -1}, "Reinstatements must be non-negative"),
], ids=["negative-attachment", "negative-limit", "negative-rate", "negative-reinstatements"])
def test_invalid_parameters(self, kwargs, match):
    with pytest.raises(ValueError, match=match):
        EnhancedInsuranceLayer(**kwargs)
```

**Action**: Parametrize this test.

### 2B. Reinstatement Premium Tests

In `test_insurance_program.py`, three separate tests cover reinstatement premium calculation for PRO_RATA, FULL, and FREE types (lines 71-109). These can be parametrized.

**Action**: Parametrize reinstatement premium tests.

### 2C. Insurance Program Coverage Validation Tests

In `test_insurance_program_coverage.py::TestEnhancedInsuranceLayerValidation`, the hybrid limit validation tests (lines 57-110) test 5 separate validation scenarios. These are good parametrize candidates.

**Action**: Parametrize hybrid validation tests.

### 2D. Market Cycle Tests

In `test_insurance_pricing.py::TestMarketCycle`, two tests check values and names of the same 3 enum members. These can be parametrized.

**Action**: Parametrize market cycle tests.

## 3. Overlapping Coverage

### 3A. Coverage Files Are Generally Complementary (NOT Redundant)

After careful analysis, the `_coverage.py` files consistently target different code paths than their primary test files:

| Primary File | Coverage File | Verdict |
|-------------|--------------|---------|
| `test_batch_processor.py` | `test_batch_processor_coverage.py` | **Complementary** - coverage file targets lines 281-283, 314, 320-323, etc. |
| `test_config_manager.py` | `test_config_manager_coverage.py` | **Complementary** - coverage file targets lines 57-63, 200-204, 216, etc. |
| `test_monte_carlo.py` | `test_monte_carlo_coverage.py` | **Complementary** - coverage file targets helper functions and edge cases |
| `test_insurance_pricing.py` | `test_insurance_pricing_coverage.py` | **Complementary** - coverage file targets lines 434, 453, 482-484, etc. |
| `test_insurance_program.py` | `test_insurance_program_coverage.py` | **Complementary** - coverage file targets hybrid/aggregate limit types |
| `test_ledger.py` | `test_ledger_coverage.py` | **Complementary** - coverage file targets lines 236, 362, 451-460, etc. |
| `test_risk_metrics.py` | `test_risk_metrics_coverage.py` | **Complementary** - coverage file targets weighted edge cases |
| `test_ruin_probability.py` | `test_ruin_probability_coverage.py` | **Complementary** - coverage file targets summary() and edge cases |

### 3B. Decision Engine Files Are Complementary

- `test_decision_engine.py` - Core tests for dataclasses, optimization, decision-making
- `test_decision_engine_edge_cases.py` - CVaR edge cases, empty arrays, extreme values
- `test_decision_engine_scenarios.py` - Real-world business scenarios

**Verdict**: All three test different aspects. No redundancy.

### 3C. Config Files Are Complementary

- `test_config.py` - Config v1 Pydantic model validation (ManufacturerConfig, GrowthConfig, etc.)
- `test_config_v2.py` - ConfigV2 model (ProfileMetadata, preset/module support)
- `test_config_validation.py` - IndustryConfig validation (asset ratios, working capital days)
- `test_config_compat.py` - ConfigTranslator, LegacyConfigAdapter
- `test_config_loader.py` - ConfigLoader file I/O
- `test_config_manager.py` - ConfigManager system (profiles, modules, presets)
- `test_config_manager_coverage.py` - Coverage gaps in ConfigManager
- `test_config_migrator.py` - Config migration
- `test_config_v2_integration.py` - V2 integration tests

**Verdict**: Each file tests a different config module/class. No redundancy.

### 3D. Visualization Files Are Mostly Complementary

- `test_visualization.py` - Core module (imports, colors, formatting, FigureFactory)
- `test_visualization_simple.py` - Plot functions (loss_distribution, return_period, etc.)
- `test_visualization_extended.py` - Extended formatter/plot edge cases
- `test_visualization_comprehensive.py` - Submodules (annotations, batch_plots, export, interactive_plots)
- `test_visualization_factory.py` - StyleManager, FigureFactory from visualization_infra
- `test_visualization_gaps_coverage.py` - Specific uncovered lines in visualization submodules

**Verdict**: Minor overlap in formatting tests (addressed in Section 1A). Otherwise complementary.

## 4. Fixture Duplication

### 4A. `temp_checkpoint_dir` (2 definitions)

- `test_batch_processor.py:175` (class-level fixture)
- `test_batch_processor_coverage.py:37` (module-level fixture)

Both create `tempfile.TemporaryDirectory()` and yield `Path(tmpdir)`. Identical logic.

**Action**: Move to a shared batch processor conftest or leave as-is since they are in different files and both are lightweight. No action needed - the duplication is benign.

### 4B. `mock_components` (4 definitions)

- `test_batch_processor.py:181` - Returns (loss_gen, insurance, manufacturer) mocks
- `test_batch_processor_coverage.py:44` - Identical
- `test_convergence_ess.py:308` - Different mock structure (ESS-specific)
- `test_scenario_batch.py:77` - Similar but with ScenarioManager-specific setup

**Action**: The two batch processor files share identical fixtures. This is benign since they are in separate files and both are lightweight. No action needed.

### 4C. `manufacturer_config` (7 definitions)

- `test_execution_semantics.py:23`
- `test_monte_carlo_coverage.py:54`
- `test_parallel_independence.py:33`
- `test_roe_insurance.py:22`
- `test_strategy_backtester_coverage.py:42`
- `test_simulation_coverage.py:57`
- `test_simulation.py:17`

All create `ManufacturerConfig` with similar but not identical parameters. Each file uses slightly different values appropriate to their specific test scenarios.

**Action**: No consolidation - the configs are intentionally different per test context.

### 4D. `temp_config_dir` (4 definitions)

- `test_config_loader.py:21`
- `test_config_manager.py:20`
- `test_config_manager_coverage.py:76`
- `test_coverage_gaps_batch3.py:536`

Each creates a different directory structure appropriate to the module being tested.

**Action**: No consolidation - structures differ per test context.

## 5. Implementation Plan

### Changes to implement:

1. **Consolidate formatting tests**: Remove duplicate `format_currency`, `format_percentage`, `wsj_formatter`, and `set_wsj_style` tests from `test_visualization_simple.py` (keep the versions in `test_visualization.py` and `test_visualization_extended.py` which are more comprehensive)

2. **Parametrize insurance program validation**: Convert `test_invalid_parameters` in `test_insurance_program.py` to use `@pytest.mark.parametrize`

3. **Parametrize reinstatement premium tests**: Convert PRO_RATA/FULL/FREE tests in `test_insurance_program.py` to parametrize

4. **Parametrize hybrid validation**: Convert hybrid limit validation tests in `test_insurance_program_coverage.py` to parametrize

5. **Parametrize market cycle tests**: Convert enum value/name tests in `test_insurance_pricing.py` to parametrize

## Summary Statistics

| Category | Count | Tests Affected |
|----------|-------|----------------|
| Copy-paste duplicates (formatting) | 6 tests | ~6 removed |
| Parametrize candidates | 4 clusters | ~12 tests consolidated into 4 |
| Overlapping coverage (false alarm) | 0 | 0 |
| Fixture duplication (benign) | ~15 definitions | 0 changed |
| **Total tests to remove/consolidate** | **~18** | |

## Files NOT Modified (Conservative Decisions)

- All integration tests in `ergodic_insurance/tests/integration/` are preserved
- All `_coverage.py` files are preserved (they test distinct code paths)
- All bug-specific test files (`test_*_bug*.py`, `test_issue_*.py`) are preserved
- All actuarial/statistical tests are preserved unchanged
- Fixture duplications are left as-is (benign, context-appropriate)
