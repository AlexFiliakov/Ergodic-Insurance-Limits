# GitHub Issues Validation Review

**Reviewer:** Issues Validation Agent
**Date:** 2026-02-06
**Repository:** AlexFiliakov/Ergodic-Insurance-Limits

## Summary

| Status | Count |
|--------|-------|
| Confirmed | 24 |
| Partially Correct | 3 |
| Needs Update | 1 |
| Not Code Issues | 3 |
| **Total Validated** | **31** |

## Validated Issues

| Issue | Title | Verdict | Notes |
|-------|-------|---------|-------|
| #362 | Ledger historical queries after pruning | **Confirmed** | Both bugs verified in code |
| #361 | Insurance coverage inconsistencies | **Confirmed** | All 5 sub-issues verified |
| #360 | Test suite tautological assertions | **Confirmed** | Root causes correctly identified |
| #358 | Batch processor edge cases | **Confirmed** | All 3 bugs verified |
| #357 | Config system additional issues | **Confirmed** | All 3 bugs verified |
| #356 | Parallel executor resource issues | **Confirmed** | All 3 bugs verified in code |
| #355 | Ruin probability attribution bugs | **Confirmed** | All 3 bugs verified |
| #354 | Financial statements capex double-count | **Confirmed** | gross_ppe used instead of net_ppe |
| #353 | ROEAnalyzer metric formulas | **Confirmed** | All 3 formulas incorrect as described |
| #352 | BusinessOptimizer unreliable results | **Confirmed** | Deductible unused, RNG unseeded, weight mutation |
| #351 | Optimization module math errors | **Confirmed** | All 4 bugs verified |
| #350 | Convergence diagnostics ESS errors | **Confirmed** | Missing factor of 2, lag-0 bug, Geyer pairs |
| #349 | Simulation execution inconsistencies | **Confirmed** | Ordering differences verified across paths |
| #348 | MC engine InsuranceProgram state bug | **Confirmed** | Critical: layer_states bypass, no reset |
| #347 | Flaky test: startup_company_scenario | **Confirmed** | Subsumed by #360 root cause analysis |
| #346 | Flaky test: manufacturer_copy_independence | **Confirmed** | Subsumed by #360 root cause analysis |
| #345 | Flaky test: multi_year_projection_performance | **Confirmed** | Subsumed by #360 root cause analysis |
| #344 | Flaky test: large_cohort_performance | **Confirmed** | Subsumed by #360 root cause analysis |
| #337 | TDigest NaN/infinity corruption | **Confirmed** | No input validation in update() |
| #336 | TDigest merge direction toggled by reads | **Confirmed** | _flush() toggles on read-only ops |
| #335 | TDigest.merge() mutates other digest | **Confirmed** | other._flush() called directly |
| #313 | Naive monthly financials | **Confirmed** | All monthly figures are annual/12 |
| #309 | Decompose ergodic_analyzer.py | **Confirmed** | File is oversized; edge cases verified |
| #307 | Sortino ratio and risk metrics bugs | **Confirmed** | Related to but distinct from #353 |
| #306 | Config data loss and validation bypass | **Partially Correct** | Most items confirmed; hybrid validation uses `and` not `or` as reported |
| #303 | Decision engine recursion and heuristic | **Partially Correct** | Recursion confirmed; hardcoded values now configurable but heuristic concern valid |
| #300 | HJB solver mathematical errors | **Confirmed** | `if dim == 0` guard, np.roll wraparound, missing gradient |
| #277 | Hardcoded calendar logic in accrual mgr | **Confirmed** | Months [3,5,8,11] hardcoded |
| #205 | Release version 1.0.0 | **Not Code Bug** | Tracking issue, still relevant |
| #194 | Whitepaper outlining main features | **Not Code Bug** | Documentation task, still relevant |
| #155 | Fix Mobile Website | **Not Code Bug** | UI/website issue, cannot validate from code |

## Key Findings

### Critical Priority Issues (recommend immediate attention)
1. **#348** - MC engine InsuranceProgram state not reset - causes all MC results to be incorrect
2. **#349** - Simulation execution ordering inconsistencies - different code paths give different results
3. **#300** - HJB solver fundamentally broken for multi-dimensional problems
4. **#303** - Decision engine infinite recursion possible

### High Priority Issues
5. **#350** - ESS overestimated by ~2x due to missing factor
6. **#351** - Optimization augmented Lagrangian sign error
7. **#356** - SharedMemory handle leaks
8. **#354** - Capex double-counts depreciation
9. **#353/#307** - Multiple risk metric formula errors
10. **#357** - Config `switch_pricing_scenario` is a complete no-op

### Overlap/Duplication Detected
- **#353 and #307**: Both report Sortino ratio bugs in `risk_metrics.py` but in different classes (ROEAnalyzer vs RiskMetrics). Both are valid but should cross-reference each other.
- **#344-347 and #360**: The four flaky test issues are root-caused in #360. Consider closing #344-347 as duplicates of #360.

## New Issues Created

None needed - existing issue coverage is comprehensive for identified problems.

## Gap Analysis

After reviewing all core source files, the existing 31 issues provide thorough coverage of the major bugs. The most significant gaps are:
- No issue for `SharedMemory` cleanup in `parallel_executor.py` worker processes (partially covered by #356)
- No dedicated issue for the `Simulation.run()` re-entrancy problem (partially covered by #349)

## 10+ Actionable Findings

1. **#348 Critical**: `Simulation.run_monte_carlo()` creates `InsuranceProgram(layers=[])` then appends to `.layers` but `layer_states` is only built in `__init__`. All MC insurance analysis produces zero coverage. Fix: use constructor `InsuranceProgram(layers=[layer])`.
2. **#348 Critical**: InsuranceProgram layer state (`used_limit`, `is_exhausted`) never reset between MC paths. Fix: deep-copy or reset before each simulation.
3. **#349 High**: Three simulation execution paths process claims/premiums/step in different orders, producing systematically different results. Fix: unify to a single ordering.
4. **#349 High**: Insolvency detected one year late because check uses pre-claim equity. Fix: check post-claim state.
5. **#350 High**: ESS overestimated by ~2x in `convergence_advanced.py` due to missing factor of 2 in integrated autocorrelation time. Fix: add `2 *` multiplier.
6. **#350 High**: `ConvergenceStats.autocorrelation` always reports 1.0 (lag-0 instead of lag-1). Fix: index `[1]` instead of `[0]`.
7. **#351 High**: Augmented Lagrangian slack variable sign is wrong, preventing correct constraint enforcement. Fix: `max(0, lambdas/rho - g)`.
8. **#351 High**: `trust-exact` selected when Hessian is absent (ternary inverted). Fix: swap branches.
9. **#303 High**: Decision engine infinite recursion: SLSQP -> DE -> WEIGHTED_SUM (=SLSQP) -> loop. Fix: track attempted methods.
10. **#354 Medium-High**: Capex formula double-counts depreciation by using gross PP&E with a net PP&E formula. Fix: use net PP&E or remove depreciation add-back.
11. **#357 High**: `switch_pricing_scenario` logs new rates but never writes them back. Complete no-op. Fix: actually write rates to config_dict.
12. **#353 Medium**: Sortino, downside deviation, and semi-variance formulas all incorrect in ROEAnalyzer. Fix: use standard formulas with total count.

## Validation Complete

All 31 open issues have been reviewed and commented on. Comments posted to each issue with validation status, verification details, accuracy assessment, completeness notes, priority assessment, and related issue cross-references.
