# Test Suite Refactor Survey

**Date:** 2026-02-13
**Branch:** tests/571_refactor_tests
**Total tests collected:** 5,347 (plus 2 collection errors from notebook test files)
**Total test files:** 183 (175 unit + 8 integration)
**Total lines of test code:** ~109,400

## Directory Structure

ergodic_insurance/tests/
  conftest.py                  (shared fixtures, imports from integration/test_fixtures.py)
  __init__.py
  test_*.py                    (175 unit test files)
  integration/
    test_claim_development_wrapper.py
    test_critical_integrations.py
    test_financial_integration.py
    test_fixtures.py         (shared fixture definitions used by conftest.py)
    test_helpers.py
    test_insurance_stack.py
    test_parallel_worker.py
    test_simulation_pipeline.py

No conftest.py in integration/ -- shared fixtures are in integration/test_fixtures.py and imported by the top-level conftest.py.

## Shared Fixtures (conftest.py)

The top-level conftest.py:
- Sets matplotlib to Agg backend
- Imports 18 fixtures from integration/test_fixtures.py:
  base_manufacturer, basic_insurance_policy, catastrophic_loss_generator,
  claim_development, config_manager, default_config_v2,
  enhanced_insurance_program, gbm_process, high_frequency_loss_generator,
  integration_test_dir, lognormal_volatility, manufacturing_loss_generator,
  mature_manufacturer, mean_reverting_process, monte_carlo_engine,
  multi_layer_insurance, standard_loss_generator, startup_manufacturer
- Has pytest_collection_modifyitems to auto-skip requires_multiprocessing and benchmark tests in CI
- Provides project_root, test_data_dir, parameters_dir fixtures

## 20 Largest Test Files (by line count)

| Lines | File |
|------:|------|
| 2,613 | test_hjb_numerical.py |
| 2,005 | test_manufacturer.py |
| 1,873 | test_visualization_gaps_coverage.py |
| 1,680 | test_reporting_coverage.py |
| 1,659 | test_monte_carlo_coverage.py |
| 1,631 | test_insurance_program.py |
| 1,440 | test_financial_statements.py |
| 1,423 | test_claim_development.py |
| 1,379 | test_insurance_pricing.py |
| 1,325 | test_misc_gaps_coverage.py |
| 1,307 | test_ledger.py |
| 1,237 | test_decision_engine.py |
| 1,234 | test_walk_forward.py |
| 1,220 | integration/test_critical_integrations.py |
| 1,219 | test_monte_carlo.py |
| 1,081 | test_visualization_extended.py |
| 1,071 | test_simulation_coverage.py |
| 1,065 | test_coverage_gaps_batch4.py |
| 1,023 | test_cache_manager.py |
| 1,015 | test_technical_plots.py |

## Notable Patterns

1. **Many *_coverage files**: Files like test_batch_processor_coverage.py, test_bootstrap_analysis_coverage.py, etc. appear to be supplementary coverage tests alongside the main test file -- high redundancy risk.

2. **Bug-specific test files**: test_convergence_bug350.py, test_convergence_bug396.py, test_convergence_bug476.py, test_bootstrap_ci_seed_bug400.py, test_std_ddof1_bug478.py, test_autocorrelation_fft_380.py -- regression tests for specific issues.

3. **test_coverage_gaps_batch*.py files (4 files)**: Named as batch coverage gap fillers -- likely auto-generated or bulk-added tests.

4. **Multiple test files for same module**: e.g., test_decision_engine.py + test_decision_engine_edge_cases.py + test_decision_engine_scenarios.py, test_monte_carlo.py + test_monte_carlo_coverage.py + test_monte_carlo_extended.py + test_monte_carlo_parallel.py etc.

5. **Test files with no clear source module** (103 of 183): Many test files test cross-cutting concerns or were created for specific features within larger modules.

## Recent Test Evolution (last 20 commits touching tests)

The most recent commits focus on:
- GPU acceleration (test_gpu_backend, test_gpu_mc_engine, test_gpu_objective)
- HJB solver fixes
- Financial statement accounting fixes
- Parallel executor improvements
- Security fixes (safe_pickle, report_builder)
- Progress callback features

## Source Module Groups

Major testable modules and their test file clusters:

Insurance core: test_insurance.py, test_insurance_coverage.py, test_insurance_program.py, test_insurance_program_coverage.py, test_insurance_pricing.py, test_insurance_pricing_coverage.py, test_insurance_accounting.py

Monte Carlo: test_monte_carlo.py, test_monte_carlo_coverage.py, test_monte_carlo_extended.py, test_monte_carlo_parallel.py, test_monte_carlo_trajectory_storage.py, test_monte_carlo_worker_config.py

Financial: test_financial_statements.py, test_financial_statements_coverage.py, test_cash_flow_statement.py, test_cash_reconciliation.py, test_balance_sheet_*.py

Convergence: test_convergence_advanced.py, test_convergence_advanced_coverage.py, test_convergence_extended.py, test_convergence_ess.py, test_convergence_plots.py, test_convergence_bug*.py

Decision engine: test_decision_engine.py, test_decision_engine_edge_cases.py, test_decision_engine_scenarios.py

Configuration: test_config.py, test_config_compat.py, test_config_loader.py, test_config_manager.py, test_config_manager_coverage.py, test_config_manager_security.py, test_config_migrator.py, test_config_v2.py, test_config_v2_integration.py, test_config_validation.py

Visualization: test_visualization.py, test_visualization_comprehensive.py, test_visualization_extended.py, test_visualization_factory.py, test_visualization_gaps_coverage.py, test_visualization_simple.py, test_executive_visualizations.py, test_convergence_plots.py, test_technical_plots.py, test_sensitivity_visualization.py, test_figure_factory.py

Manufacturer: test_manufacturer.py, test_manufacturer_coverage.py, test_manufacturer_methods.py

Optimization: test_optimization.py, test_optimization_coverage.py, test_optimizer_config.py, test_business_optimizer.py, test_pareto_frontier.py, test_pareto_frontier_coverage.py

Ledger: test_ledger.py, test_ledger_coverage.py

Risk/Ruin: test_risk_metrics.py, test_risk_metrics_coverage.py, test_ruin_probability.py, test_ruin_probability_coverage.py

GPU: test_gpu_backend.py, test_gpu_mc_engine.py, test_gpu_objective.py

HJB: test_hjb_numerical.py, test_hjb_solver.py
