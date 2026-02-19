# Test Suite Refactor Survey

**Date:** 2026-02-17
**Branch:** tests/refactor-tests
**Total tests collected:** 5,723
**Total test files:** 192 (184 unit + 8 integration)
**Total lines of test code:** 116,768

## Directory Structure

```
ergodic_insurance/tests/
├── conftest.py                    (shared fixtures, matplotlib backend, mp cache reset)
├── __init__.py
├── integration/                   (8 files)
│   ├── test_claim_development_wrapper.py
│   ├── test_critical_integrations.py
│   ├── test_financial_integration.py
│   ├── test_fixtures.py           (shared fixtures imported by root conftest)
│   ├── test_helpers.py
│   ├── test_insurance_stack.py
│   ├── test_parallel_worker.py
│   └── test_simulation_pipeline.py
└── test_*.py                      (184 unit test files)
```

No conftest.py in integration/ -- shared fixtures are in integration/test_fixtures.py and imported by the top-level conftest.py.

## Shared Fixtures (conftest.py)

The top-level conftest.py:
1. Sets matplotlib to `Agg` backend (non-interactive)
2. Imports 18 fixtures from `integration/test_fixtures.py`:
   - base_manufacturer, basic_insurance_policy, catastrophic_loss_generator
   - claim_development, config_manager, default_config_v2
   - enhanced_insurance_program, gbm_process, high_frequency_loss_generator
   - integration_test_dir, lognormal_volatility, manufacturing_loss_generator
   - mature_manufacturer, mean_reverting_process, monte_carlo_engine
   - multi_layer_insurance, standard_loss_generator, startup_manufacturer
3. Auto-skips `requires_multiprocessing` and `benchmark` tests in CI
4. Auto-closes matplotlib figures after each test (`_close_matplotlib_figures`)
5. Resets `_mp_probe_result` cache for test isolation (`_reset_mp_probe_cache`)
6. Provides `project_root`, `test_data_dir`, `parameters_dir` fixtures

## 14 Largest Test Files (by line count)

| Lines | File |
|------:|------|
| 2,613 | test_hjb_numerical.py |
| 2,062 | test_insurance_program.py |
| 2,021 | test_manufacturer.py |
| 1,987 | test_decision_engine.py |
| 1,873 | test_visualization_gaps_coverage.py |
| 1,742 | test_claim_development.py |
| 1,680 | test_reporting_coverage.py |
| 1,646 | test_monte_carlo_coverage.py |
| 1,528 | test_insurance_pricing.py |
| 1,440 | test_financial_statements.py |
| 1,332 | test_monte_carlo.py |
| 1,326 | test_misc_gaps_coverage.py |
| 1,307 | test_ledger.py |
| 1,241 | test_summary_statistics_coverage.py |

## Notable Patterns

1. **Many *_coverage files**: Files like test_batch_processor_coverage.py, test_bootstrap_analysis_coverage.py, etc. appear to be supplementary coverage tests alongside the main test file — high redundancy risk.

2. **Bug-specific test files**: test_convergence_bug350.py, test_convergence_bug396.py, test_convergence_bug476.py, test_bootstrap_ci_seed_bug400.py, test_std_ddof1_bug478.py, test_autocorrelation_fft_380.py, test_issue_355_ruin_attribution.py, test_issue_357.py — regression tests for specific issues.

3. **test_coverage_gaps_batch*.py files (4 files)**: Named as batch coverage gap fillers — likely auto-generated or bulk-added tests.

4. **Multiple test files for same module**: e.g., test_decision_engine.py + test_decision_engine_edge_cases.py + test_decision_engine_scenarios.py, test_monte_carlo.py + test_monte_carlo_coverage.py + test_monte_carlo_extended.py + test_monte_carlo_parallel.py, etc.

## Source Module Groups

Major testable modules and their test file clusters:

**Insurance core:** test_insurance.py, test_insurance_coverage.py, test_insurance_program.py, test_insurance_program_coverage.py, test_insurance_pricing.py, test_insurance_pricing_coverage.py, test_insurance_accounting.py

**Monte Carlo:** test_monte_carlo.py, test_monte_carlo_coverage.py, test_monte_carlo_extended.py, test_monte_carlo_parallel.py, test_monte_carlo_trajectory_storage.py, test_monte_carlo_worker_config.py

**Financial:** test_financial_statements.py, test_financial_statements_coverage.py, test_cash_flow_statement.py, test_cash_reconciliation.py, test_balance_sheet_*.py, test_ledger.py, test_ledger_coverage.py

**Convergence:** test_convergence_advanced.py, test_convergence_advanced_coverage.py, test_convergence_extended.py, test_convergence_ess.py, test_convergence_plots.py, test_convergence_bug*.py (3 files)

**Decision engine:** test_decision_engine.py, test_decision_engine_edge_cases.py, test_decision_engine_scenarios.py

**Configuration:** test_config.py, test_config_compat.py, test_config_loader.py, test_config_manager.py, test_config_manager_coverage.py, test_config_manager_security.py, test_config_migrator.py, test_config_v2.py, test_config_v2_integration.py, test_config_validation.py

**Visualization:** test_visualization.py, test_visualization_comprehensive.py, test_visualization_extended.py, test_visualization_factory.py, test_visualization_gaps_coverage.py, test_visualization_namespace.py, test_visualization_simple.py, test_executive_visualizations.py, test_convergence_plots.py, test_technical_plots.py, test_sensitivity_visualization.py, test_figure_factory.py

**Manufacturer:** test_manufacturer.py, test_manufacturer_coverage.py, test_manufacturer_methods.py

**Optimization:** test_optimization.py, test_optimization_coverage.py, test_optimizer_config.py, test_business_optimizer.py, test_pareto_frontier.py, test_pareto_frontier_coverage.py

**Risk/Ruin:** test_risk_metrics.py, test_risk_metrics_coverage.py, test_ruin_probability.py, test_ruin_probability_coverage.py

**GPU:** test_gpu_backend.py, test_gpu_cpu_parity.py, test_gpu_mc_engine.py, test_gpu_objective.py

**HJB:** test_hjb_numerical.py, test_hjb_solver.py

**Reporting:** test_reporting_coverage.py, test_table_generation.py, test_insight_extractor.py, test_scenario_comparator.py, test_excel_reporter.py, test_report_generation.py, test_report_builder_security.py

**Infrastructure:** test_parallel_executor.py, test_parallel_executor_coverage.py, test_batch_processor.py, test_batch_processor_coverage.py, test_cache_manager.py, test_trajectory_storage.py, test_progress_monitor.py, test_progress_monitor_coverage.py, test_safe_pickle_coverage.py, test_performance.py, test_performance_optimizer.py

## Recent Test Evolution (last 20 commits touching tests)

Recent commits focus on:
- Computing pure premium as mean annual aggregate
- Working capital facility limits for insolvency detection
- Closing temporary accounts in GAAP closing entries
- Moving MC classmethods off Simulation to standalone functions
- Adding factory methods (from_config, from_company)
- Config validation improvements
- Parameter naming unification
- Adding __repr__, __str__ to result dataclasses
- Sign convention and validation for RiskMetrics
- Depreciation, working capital, NOL carryforward in decision engine

## Notes for Teammates

- Many files have a "base" test file and a "_coverage" companion (e.g., `test_monte_carlo.py` + `test_monte_carlo_coverage.py`). The coverage files were likely generated to fill coverage gaps — prime candidates for redundancy analysis.
- Bug fix test files (test_*_bug*.py, test_issue_*.py) should be preserved but may benefit from consolidation into the main test files.
- The integration/ directory has 8 files including shared fixtures — handle with extra care per project constraints.
- GPU test files may have environment-specific considerations.
- **Be conservative with actuarial/statistical tests** — tests involving distributions, loss models, credibility calculations, or reserve estimates may appear redundant but test different numerical edge cases. When in doubt, keep them and flag for review.
