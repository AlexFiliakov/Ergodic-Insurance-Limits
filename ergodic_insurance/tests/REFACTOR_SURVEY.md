# Test Suite Refactoring Survey

**Date**: 2026-02-09
**Branch**: bugfix/360_fix_test_suite
**Total tests collected**: 4558
**Total test files**: 164 (156 unit + 8 integration)
**Total test lines**: ~97,232

## Directory Structure

```
ergodic_insurance/tests/
  conftest.py                          # Root conftest - imports fixtures from integration/test_fixtures.py
  test_*.py                            # 156 unit test files
  integration/
    conftest.py                        # (none - fixtures in test_fixtures.py)
    test_claim_development_wrapper.py
    test_critical_integrations.py
    test_financial_integration.py
    test_fixtures.py                   # Shared fixtures imported by root conftest
    test_helpers.py
    test_insurance_stack.py
    test_parallel_worker.py
    test_simulation_pipeline.py
```

## Conftest Pattern

The root `conftest.py`:
- Sets matplotlib backend to `Agg`
- Imports 18 fixtures from `integration/test_fixtures.py` (base_manufacturer, basic_insurance_policy, claim_development, config_manager, default_config_v2, etc.)
- Auto-skips `requires_multiprocessing` tests in CI
- Defines `project_root`, `test_data_dir`, `parameters_dir` fixtures

## 15 Largest Test Files (by line count)

| Lines | File |
|-------|------|
| 2232 | test_hjb_numerical.py |
| 1873 | test_visualization_gaps_coverage.py |
| 1756 | test_manufacturer.py |
| 1680 | test_reporting_coverage.py |
| 1634 | test_monte_carlo_coverage.py |
| 1440 | test_financial_statements.py |
| 1323 | test_misc_gaps_coverage.py |
| 1307 | test_ledger.py |
| 1213 | integration/test_critical_integrations.py |
| 1203 | test_insurance_program.py |
| 1167 | test_monte_carlo.py |
| 1143 | test_decision_engine.py |
| 1080 | test_visualization_extended.py |
| 1065 | test_coverage_gaps_batch4.py |

## Test File → Source Module Mapping

### Direct mappings (test_X.py → X.py):
- test_accrual_manager.py → accrual_manager.py
- test_accuracy_validator.py → accuracy_validator.py
- test_adaptive_stopping.py → adaptive_stopping.py
- test_batch_processor.py → batch_processor.py
- test_benchmarking.py → benchmarking.py
- test_bootstrap.py → bootstrap_analysis.py
- test_business_optimizer.py → business_optimizer.py
- test_cache_manager.py → reporting/cache_manager.py
- test_claim_development.py → claim_development.py
- test_config.py → config/ (package)
- test_config_loader.py → config_loader.py
- test_config_manager.py → config_manager.py
- test_config_migrator.py → config_migrator.py
- test_convergence_advanced.py → convergence_advanced.py
- test_convergence_plots.py → convergence_plots.py
- test_decision_engine.py → decision_engine.py
- test_ergodic_analyzer.py → ergodic_analyzer.py
- test_excel_reporter.py → excel_reporter.py
- test_exposure_base.py → exposure_base.py
- test_figure_factory.py → visualization/figure_factory.py
- test_financial_statements.py → financial_statements.py
- test_hjb_solver.py → hjb_solver.py
- test_hjb_numerical.py → hjb_solver.py (numerical tests)
- test_insurance.py → insurance.py
- test_insurance_accounting.py → insurance_accounting.py
- test_insurance_pricing.py → insurance_pricing.py
- test_insurance_program.py → insurance_program.py
- test_ledger.py → ledger.py
- test_loss_distributions.py → loss_distributions.py
- test_manufacturer.py → manufacturer.py
- test_monte_carlo.py → monte_carlo.py
- test_optimal_control.py → optimal_control.py
- test_optimization.py → optimization.py
- test_parallel_executor.py → parallel_executor.py
- test_parameter_sweep.py → parameter_sweep.py
- test_pareto_frontier.py → pareto_frontier.py
- test_performance_optimizer.py → performance_optimizer.py
- test_progress_monitor.py → progress_monitor.py
- test_result_aggregation.py → result_aggregator.py
- test_risk_metrics.py → risk_metrics.py
- test_ruin_probability.py → ruin_probability.py
- test_safe_pickle_coverage.py → safe_pickle.py
- test_scenario_manager.py → scenario_manager.py
- test_sensitivity.py → sensitivity.py
- test_simulation.py → simulation.py
- test_stochastic.py → stochastic_processes.py
- test_trajectory_storage.py → trajectory_storage.py
- test_trends.py → trends.py
- test_visualization.py → visualization/ (package)
- test_walk_forward.py → walk_forward_validator.py

### Coverage gap files (generated to fill coverage):
- test_batch_processor_coverage.py
- test_bootstrap_analysis_coverage.py
- test_config_manager_coverage.py
- test_convergence_advanced_coverage.py
- test_coverage_gaps_batch1.py through batch4.py
- test_coverage_gaps_parallel_executor.py
- test_ergodic_analyzer_coverage.py
- test_exposure_base_coverage.py
- test_financial_statements_coverage.py
- test_insurance_pricing_coverage.py
- test_insurance_program_coverage.py
- test_ledger_coverage.py
- test_loss_distributions_coverage.py
- test_manufacturer_coverage.py
- test_monte_carlo_coverage.py
- test_misc_gaps_coverage.py
- test_optimization_coverage.py
- test_parallel_executor_coverage.py
- test_pareto_frontier_coverage.py
- test_progress_monitor_coverage.py
- test_reporting_coverage.py
- test_result_aggregator_coverage.py
- test_risk_metrics_coverage.py
- test_ruin_probability_coverage.py
- test_sensitivity_coverage.py
- test_simulation_coverage.py
- test_strategy_backtester_coverage.py
- test_summary_statistics_coverage.py
- test_validation_metrics_coverage.py
- test_visualization_gaps_coverage.py

### Bug-specific / feature-specific test files:
- test_bootstrap_ci_seed_bug400.py (Bug #400)
- test_convergence_bug350.py (Bug #350)
- test_convergence_bug396.py (Bug #396)
- test_convergence_bug476.py (Bug #476)
- test_issue_355_ruin_attribution.py (Issue #355)
- test_issue_357.py (Issue #357)
- test_std_ddof1_bug478.py (Bug #478)

### Feature/domain test files:
- test_accrual_integration.py
- test_balance_sheet_classification.py
- test_capex.py
- test_cash_flow_statement.py
- test_cash_reconciliation.py
- test_config_compat.py (deprecated, config_compat module removed)
- test_config_v2.py
- test_config_v2_integration.py
- test_config_validation.py
- test_convergence_ess.py
- test_convergence_extended.py
- test_deep_copy.py
- test_decision_engine_edge_cases.py
- test_decision_engine_scenarios.py
- test_depreciation_tracking.py
- test_dividend_phantom_payments.py
- test_dta_dtl_and_capex.py
- test_dta_valuation_allowance.py
- test_dynamic_premium_scaling.py
- test_end_to_end.py
- test_execution_semantics.py
- test_executive_visualizations.py
- test_going_concern.py
- test_imports.py
- test_industry_configs.py
- test_industry_switching.py
- test_insight_extractor.py
- test_insurance_coverage.py
- test_integration.py
- test_lae_tracking.py
- test_limit_types.py
- test_manufacturer_methods.py
- test_monte_carlo_extended.py
- test_monte_carlo_parallel.py
- test_monte_carlo_trajectory_storage.py
- test_monte_carlo_worker_config.py
- test_negative_cash_reclassification.py
- test_nol_carryforward.py
- test_parallel_independence.py
- test_parameter_combinations.py
- test_performance.py
- test_periodic_ruin_tracking.py
- test_premium_amortization.py
- test_pricing_scenarios.py
- test_properties.py
- test_recovery_accounting.py
- test_report_generation.py
- test_reserve_development.py
- test_retention_calculation.py
- test_roe_insurance.py
- test_run_analysis.py
- test_scenario_batch.py
- test_scenario_comparator.py
- test_sensitivity_visualization.py
- test_setup.py
- test_skipped_slow_tests.py
- test_table_generation.py
- test_tax_handling.py
- test_technical_plots.py
- test_visualization_comprehensive.py
- test_visualization_extended.py
- test_visualization_factory.py
- test_visualization_simple.py
- test_working_capital_calculation.py
- test_working_capital_changes.py

## Recent Test Evolution (last 20 commits touching tests)

Recent work has been primarily bug-fix driven, with each PR adding tests for specific issues:
- Insurance coverage calculation fixes
- Batch processor crash fixes
- Config system fixes (pricing scenarios, deep merge, circular inheritance)
- Financial statement corrections (ASC compliance)
- Risk metrics formula corrections
- Bootstrap CI seed bug
- Convergence fixes

## Key Observations for Refactoring

1. **Heavy coverage-gap pattern**: ~30 files named `*_coverage.py` or `test_coverage_gaps_*` suggest auto-generated tests to fill coverage. These are prime candidates for tautological tests.

2. **Duplicate source coverage**: Many source modules have both a primary test file AND a coverage file (e.g., `test_monte_carlo.py` + `test_monte_carlo_coverage.py`). High redundancy risk.

3. **Large files**: The top 15 files average ~1400 lines. These likely contain parametrize candidates.

4. **Bug-specific files**: 7 files exist for specific bug fixes. Their tests may overlap with the main test files after the bugs were fixed.

5. **Visualization proliferation**: 7 visualization test files that likely have significant overlap.

6. **Config proliferation**: 9 config-related test files.

## File Assignment Guidance for Teammates

Given 164 files, each teammate should focus on the aspects they specialize in across ALL files, but pay special attention to:

- **Tautology Hunter**: Start with `*_coverage.py` and `test_coverage_gaps_*` files — highest probability of tautological tests. Then work through the rest.
- **Redundancy Analyst**: Focus on file pairs like `test_X.py` + `test_X_coverage.py`. Also check config (9 files) and visualization (7 files) clusters.
- **Performance Optimizer**: Start with the 15 largest files. Also check `test_monte_carlo*`, `test_hjb_numerical.py`, and `test_simulation*` for computational overhead.
- **Reliability Engineer**: Focus on anything with Monte Carlo, stochastic, floating-point math, or time-based logic. `test_convergence_*` files are prime candidates.
