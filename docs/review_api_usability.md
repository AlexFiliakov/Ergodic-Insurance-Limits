# API Usability Review - Progress Tracker

**Reviewer**: API Usability Agent
**Date**: 2026-02-07 (Round 2 update)
**Target audience**: Actuaries, CFOs, Risk Managers

## Areas Reviewed

| Area | Status | Issues Found (Round 1) | Issues Found (Round 2) |
|------|--------|------------------------|------------------------|
| Entry points (`__init__.py`) | Reviewed | 2 | 1 |
| Config system (`config.py`, `config_loader.py`, etc.) | Reviewed | 2 | 5 |
| Core workflow (`manufacturer.py`, `simulation.py`) | Reviewed | 2 | 1 |
| Insurance modeling (`insurance.py`, `insurance_program.py`, `insurance_pricing.py`) | Reviewed | 2 | 1 |
| Analysis tools (`risk_metrics.py`, `ergodic_analyzer.py`, `decision_engine.py`) | Reviewed | 1 | 3 |
| Loss distributions (`loss_distributions.py`) | Reviewed | 1 | 0 |
| Visualization (`visualization/`) | Reviewed | 1 | 0 |
| Quick-start API (`_run_analysis.py`) | Reviewed | 0 | 2 |
| Optimization (`optimization.py`) | Reviewed | 0 | 1 |
| Reporting (`excel_reporter.py`, `reporting/`) | Reviewed | 0 | 1 |
| Examples (`examples/`) | Reviewed | 1 | 0 |
| Documentation (`docs/`) | Reviewed | 1 | 0 |

## Summary

### Round 1 (2026-02-06)
- **Issues identified**: 13
- **Critical (blocks basic usage)**: 3
- **Major (significant friction)**: 6
- **Minor (polish/ergonomics)**: 4

### Round 2 (2026-02-07)
- **New issues identified**: 13
- **Priority-high (blocks beta)**: 4
- **Priority-medium (needed for v1.0)**: 6
- **Priority-low (nice-to-have)**: 3

### Combined totals: 26 API usability issues across 2 review rounds

## Round 1 Issues (Previously Filed)

| # | Title | Severity | GitHub Issue |
|---|-------|----------|-------------|
| 1 | Config requires all sub-configs with no defaults -- prohibitive boilerplate | Critical | [#369](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/369) |
| 2 | No quick-start factory functions for common workflows | Critical | [#372](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/372) |
| 3 | Two competing SimulationResults classes create confusion | Major | [#378](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/378) |
| 4 | print() used for warnings instead of Python logging/warnings | Major | [#382](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/382) |
| 5 | InsurancePolicy and InsuranceProgram are confusingly parallel | Major | [#388](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/388) |
| 6 | Examples use stale import paths (src.config instead of ergodic_insurance) | Critical | [#392](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/392) |
| 7 | WidgetManufacturer naming is domain-inappropriate for general actuarial use | Major | [#397](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/397) |
| 8 | OptimizationConstraints defined in two modules with same name | Major | [#401](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/401) |
| 9 | RiskMetrics requires raw arrays instead of accepting SimulationResults | Minor | [#403](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/403) |
| 10 | Missing key actuarial exports from __init__.py | Major | [#405](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/405) |
| 11 | Visualization functions lack consistent return types | Minor | [#407](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/407) |
| 12 | Quick start docs use YAML config schema that doesn't match Config class | Minor | [#409](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/409) |
| 13 | LossEvent has redundant fields (time/timestamp, loss_type/event_type) | Minor | [#410](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/410) |

## Round 2 Issues (Filed 2026-02-07)

| # | Title | Priority | File(s) | GitHub Issue |
|---|-------|----------|---------|-------------|
| 14 | config.py is a 1984-line mega-module with 20+ classes harming discoverability | priority-high | `config.py` | [#458](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/458) |
| 15 | Simulation docstring shows wrong InsurancePolicy constructor signature | priority-high | `simulation.py:432-438` | [#459](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/459) |
| 16 | Config.from_company() silently ignores unsupported industry values | priority-medium | `config.py:753` | [#460](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/460) |
| 17 | RiskMetrics.var() returns different types based on parameter value | priority-high | `risk_metrics.py:83-127` | [#461](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/461) |
| 18 | IndustryConfig uses assert for validation which can be silently disabled | priority-medium | `config.py:1597-1623` | [#462](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/462) |
| 19 | run_analysis() silently defaults loss_severity_std to loss_severity_mean (CV=1.0) | priority-medium | `_run_analysis.py:436-437` | [#463](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/463) |
| 20 | Duplicate ExcelReportConfig class in config.py and excel_reporter.py | priority-medium | `config.py:1507`, `excel_reporter.py:62` | [#465](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/465) |
| 21 | DecisionMetrics.calculate_score() uses hardcoded 20% growth target for normalization | priority-medium | `decision_engine.py:135-145` | [#467](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/467) |
| 22 | TransitionProbabilities uses 9 flat fields instead of matrix interface | priority-low | `config.py:972-1005` | [#469](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/469) |
| 23 | BusinessOptimizerConfig and DecisionEngineConfig are plain dataclasses lacking Pydantic validation | priority-medium | `config.py:78-157` | [#471](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/471) |
| 24 | Config.override() uses dunder notation (manufacturer__tax_rate) which is non-standard | priority-medium | `config.py:841-862` | [#473](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/473) |
| 25 | __all__ exports 40+ names with no namespace grouping making import discovery difficult | priority-medium | `__init__.py:66-130` | [#477](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/477) |
| 26 | AnalysisResults.plot() and to_dataframe() lack return type annotations | priority-low | `_run_analysis.py:234` | [#479](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/479) |
| 27 | ConstraintViolation.__str__ uses emoji characters that break in some terminals | priority-low | `optimization.py:38-45` | [#480](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/480) |
| 28 | No InsurancePolicy.from_simple() convenience constructor for common single-layer case | priority-high | `insurance.py:184-265` | [#485](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/485) |
| 29 | RiskMetrics.__init__ uses print() for NaN/inf removal warning instead of logging | priority-low | `risk_metrics.py:59` | [#487](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues/487) |

## Strengths Observed

Despite the usability issues above, the API has several notable strengths:

1. **Excellent docstrings**: Most classes have comprehensive docstrings with examples, parameter descriptions, and cross-references. The `ErgodicAnalyzer` docstring in particular is outstanding academic documentation.

2. **Strong type hints**: Pydantic models in `config.py` provide excellent validation and type safety. IDE support is good for config objects.

3. **Good actuarial terminology in insurance modules**: Terms like "attachment point", "limit", "deductible", "reinstatement", "layer" are used correctly and consistently within each module.

4. **Sensible parameter validation**: The `InsuranceLayer.__post_init__` validates attachment_point >= 0, limit > 0, rate >= 0. Config validators catch unrealistic values early.

5. **Well-structured loss modeling**: The `LossDistribution` ABC with concrete implementations (Lognormal, Pareto, GPD) follows standard actuarial practice.

6. **Comprehensive visualization**: 30+ plot types covering executive, technical, and interactive needs. WSJ-style formatting shows attention to presentation quality.

7. **Financial rigor**: GAAP-compliant insurance accounting, proper depreciation, Decimal-based currency calculations show deep domain knowledge.

8. **`run_analysis()` quick-start**: The one-call entry point (`run_analysis()`) with `AnalysisResults` container is well-designed and provides `.summary()`, `.to_dataframe()`, and `.plot()` methods that cover the 80% use case cleanly.

9. **`Config.from_company()` factory**: Smart industry-specific defaults derived from basic company info (initial_assets, operating_margin) reduce boilerplate for the common case.

10. **Cross-field validation**: Pydantic model validators catch inconsistencies like time_horizon > max_horizon, stochastic model with zero volatility, and transition probabilities not summing to 1.0.

## Remaining Areas to Cover in Future Reviews

- **Thread safety**: Simulation objects modify manufacturer state in-place; thread safety for parallel Monte Carlo not fully audited
- **Error message quality**: Systematic review of all `ValueError` messages for actionable guidance
- **Deprecation strategy**: `ConfigV2`, `LegacyConfigAdapter`, `config_compat.py` -- migration path clarity
- **Notebook experience**: Jupyter-specific usability (progress bars, inline plots, display repr)
- **Performance API**: `PerformanceOptimizer`, `OptimizationConfig` -- user-facing performance tuning
