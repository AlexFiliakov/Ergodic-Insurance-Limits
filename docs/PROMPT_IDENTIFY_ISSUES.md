# Codebase Review Prompt: Ergodic Insurance Limits

> **Purpose**: Launch a team of Opus 4.6 agents to perform a comprehensive review of the `ergodic-insurance` Python package (v0.9.0) and file implementation-ready GitHub issues.
>
> **Context**: This is an actuarial/risk management framework approaching v1.0. It models insurance purchasing decisions using ergodic theory (time-average growth rates) rather than ensemble averages. The codebase has ~90 production modules, ~150 test files, and 30 open GitHub issues. It is intended for use in financial systems where calculation correctness directly impacts compliance and reporting.

---

## Prompt

Create an agent team to perform a thorough review of this Python package for actuarial and risk management use. Work in the `develop` branch. Spawn **eight** reviewers, each operating independently in their domain:

### Team Composition

1. **Performance Reviewer** — Identifies optimization opportunities across the codebase
2. **Financial Accuracy Reviewer** — Validates financial implementations against GAAP (ASC 740, ASC 944, ASC 450) and actuarial standards
3. **Mathematical Correctness Reviewer** — Verifies statistical, probabilistic, and numerical algorithm implementations
4. **Actuarial Methodology Reviewer** — Evaluates pricing and reserving practices against current actuarial standards and emerging research
5. **API Usability Reviewer** — Evaluates the developer experience for actuarial and risk management professionals
6. **Security Reviewer** — Audits for vulnerabilities including unsafe deserialization, input validation, and dependency risks
7. **Issue Validation Reviewer** — Cross-cutting review of all 30 existing open GitHub issues for accuracy, completeness, and priority correctness
8. **Team Lead / Coordinator** — Orchestrates the team, prevents duplicates, and ensures complete coverage

---

### Scope

**In scope**: All production source code under `ergodic_insurance/` (excluding `.venv/`, `__pycache__/`, and `docs/` build scripts). Approximately 90 modules including:

- Core simulation: `simulation.py`, `monte_carlo.py`, `monte_carlo_worker.py`, `parallel_executor.py`
- Financial: `financial_statements.py`, `insurance_accounting.py`, `tax_handler.py`, `ledger.py`, `accrual_manager.py`, `manufacturer_balance_sheet.py`, `manufacturer_income.py`
- Mathematical: `ergodic_analyzer.py`, `convergence.py`, `convergence_advanced.py`, `ruin_probability.py`, `hjb_solver.py`, `optimal_control.py`, `stochastic_processes.py`, `loss_distributions.py`, `statistical_tests.py`
- Insurance: `insurance.py`, `insurance_pricing.py`, `insurance_program.py`, `claim_development.py`, `claim_liability.py`, `exposure_base.py`
- Optimization: `optimization.py`, `business_optimizer.py`, `pareto_frontier.py`, `parameter_sweep.py`, `sensitivity.py`
- API surface: `config/` package, `decision_engine.py`, `risk_metrics.py`, `reporting/` package, `visualization/` package
- Infrastructure: `safe_pickle.py`, `batch_processor.py`, `trajectory_storage.py`, `walk_forward_validator.py`, `strategy_backtester.py`

**Out of scope**: Test files (unless referenced to validate a production code finding), Jupyter notebooks, example scripts, documentation build scripts.

---

### Pre-Review Requirements

Before filing any new issue, each reviewer **must**:

1. Read all 570+ existing open GitHub issues (`gh issue list --state open --limit 200`)
2. Check that their finding is **not already covered** by an existing issue
3. If a related issue exists, reference it in the new issue body rather than filing a duplicate

---

### Review Instructions Per Domain

#### 1. Performance Reviewer

Review all production modules for:
- **Algorithmic complexity**: O(n^2) or worse patterns that could be reduced (nested loops, repeated list scans, redundant computations)
- **Memory**: Unnecessary copies (`deepcopy` in hot paths), unbounded data accumulation, large intermediate allocations
- **Parallelism**: Underutilized multiprocessing, GIL bottlenecks, serialization overhead in worker dispatch
- **NumPy/SciPy**: Loops that should be vectorized, redundant array creation, suboptimal dtype usage
- **Caching**: Missing cache opportunities, cache invalidation bugs, unbounded cache growth
- **I/O**: Blocking I/O in compute paths, excessive logging in tight loops, inefficient serialization

Label issues with: `performance`, and one of `priority-high`, `priority-medium`, `priority-low`.

#### 2. Financial Accuracy Reviewer

Review all financial modules for:
- **GAAP compliance**: ASC 740 (income taxes, DTA/DTL, NOL carryforward, valuation allowance), ASC 944 (insurance contracts, premium recognition, loss reserves), ASC 450 (contingencies)
- **Double-entry integrity**: Every debit has an equal credit in the ledger; trial balance always zeros
- **Period-end accuracy**: Accruals, amortization schedules, depreciation calculations, working capital changes
- **Cash flow statement**: Operating/investing/financing classification correctness, indirect method reconciliation
- **Insurance accounting**: Premium earned vs. written recognition, loss development, IBNR estimation, LAE allocation
- **Tax calculations**: Effective tax rate computation, temporary vs. permanent differences, deferred tax asset/liability netting
- **Rounding and precision**: Financial amounts should maintain appropriate precision; check for floating-point accumulation errors in monetary calculations
- **Edge cases**: Zero-revenue periods, negative equity, tax loss scenarios, going-concern thresholds

Label issues with: `financial-accuracy`, and one of `priority-high`, `priority-medium`, `priority-low`.

#### 3. Mathematical Correctness Reviewer

Review all mathematical modules for:
- **Ergodic theory**: Time-average growth rate calculations, ensemble vs. time-average comparisons, multiplicative dynamics handling
- **Statistical tests**: Correct test statistic formulas, p-value computations, degrees of freedom, multiple comparison corrections
- **Convergence diagnostics**: Effective sample size, Raftery-Lewis, Geweke, R-hat implementations against textbook definitions
- **Distribution fitting**: Parameter estimation methods, goodness-of-fit tests, tail behavior handling
- **Numerical methods**: HJB solver stability, finite difference schemes, boundary condition handling, convergence criteria
- **Monte Carlo**: Random seed management, variance reduction techniques, confidence interval construction, bias in estimators
- **Ruin theory**: Classical ruin probability formulas, Lundberg exponent, adjustment coefficient correctness
- **Optimization**: Objective function correctness, constraint handling, convergence to valid optima, Pareto frontier computation
- **Edge cases**: Division by zero guards, log of zero/negative, overflow/underflow, degenerate inputs (single observation, zero variance)
- **Population vs. sample statistics**: Verify ddof parameter usage (ddof=0 vs ddof=1) is appropriate for each context

Label issues with: `mathematical-correctness`, and one of `priority-high`, `priority-medium`, `priority-low`.

#### 4. Actuarial Methodology Reviewer

Review all pricing and reserving modules for adherence to sound actuarial practice, current industry standards, and forward-thinking enhancements informed by recent research. Key modules: `insurance_pricing.py`, `claim_development.py`, `claim_liability.py`, `loss_distributions.py`, `exposure_base.py`, `insurance_program.py`, `ruin_probability.py`, `optimal_control.py`, `config/insurance.py`, `config/market.py`.

**Pricing methodology**:
- **Rate adequacy**: Are indicated premiums derived from credible loss experience? Is the loss ratio approach vs. pure premium approach applied correctly?
- **Loss cost trending**: Are trend factors applied to project historical losses to prospective levels? Is the trend period appropriate?
- **Credibility weighting**: Does the model blend experience-rated and manual-rated premiums appropriately? Are credibility standards (Buhlmann, limited fluctuation) implemented correctly?
- **Expense and profit loading**: Are fixed vs. variable expense provisions handled? Is the permissible loss ratio derivation sound?
- **Experience rating**: Does the experience modification factor calculation follow standard practice (e.g., NCCI-style split between primary and excess losses)?
- **Market cycle modeling**: Are underwriting cycle adjustments (hard/soft market) calibrated to realistic parameters? Do pricing scenarios in `PricingScenario`/`PricingScenarioConfig` reflect plausible market conditions?
- **Layer pricing**: For excess layers in `InsuranceProgram`, are increased limits factors (ILF) or loss elimination ratios applied correctly? Are layer correlations accounted for?
- **Frequency-severity independence**: Is the assumption of independence between claim frequency and severity appropriate? If not, are dependencies modeled?

**Reserving methodology**:
- **IBNR estimation**: Are the implemented methods (chain ladder, Bornhuetter-Ferguson, expected loss ratio) applied correctly in `ClaimDevelopment`? Are their limitations documented?
- **Development patterns**: Are claim development factors (CDFs) derived appropriately? Are tail factors included for long-tailed lines? Is the `StochasticClaimDevelopment` model using reasonable distributional assumptions for factor variability?
- **Reserve variability**: Is reserve uncertainty quantified (e.g., Mack method, bootstrapped triangles, stochastic reserving)? Are confidence intervals around reserve estimates provided?
- **Case reserve adequacy**: Are case-level reserves calibrated separately from IBNR? Is there a mechanism for reserve strengthening/weakening over development?
- **LAE reserving**: Are loss adjustment expenses (allocated and unallocated) reserved with appropriate methods? Is the current LAE-to-loss ratio approach actuarially defensible?
- **Salvage and subrogation**: Are recovery expectations modeled and netted appropriately?

**Loss modeling**:
- **Distribution selection**: Are `LognormalLoss`, `ParetoLoss`, and `GeneralizedParetoLoss` appropriate choices for the risk types modeled? Are there lines of business where other distributions (Weibull, Burr, mixed exponential) would be more actuarially appropriate?
- **Tail modeling**: Is extreme value theory (EVT) applied appropriately for large/catastrophic losses? Are GPD thresholds selected with sound methodology (mean excess plots, parameter stability)?
- **Exposure rating**: Are the exposure bases (`RevenueExposure`, `AssetExposure`, `ProductionExposure`, etc.) calibrated to industry exposure curves? Are the frequency-to-exposure relationships realistic?
- **Loss development on a ground-up vs. limited basis**: Is development applied to the correct loss basis for each layer?
- **Parameter estimation**: Are distribution parameters estimated with appropriate methods (MLE, method of moments, minimum distance)? Are parameter uncertainty ranges propagated through the analysis?

**Emerging practices and research opportunities**:
- **Ergodic pricing implications**: Does the framework fully exploit ergodic theory for pricing? Are there opportunities to derive time-average-optimal premium rates that differ from traditional expected-value pricing?
- **Dynamic reserving**: Could the reserving methodology benefit from Bayesian updating as new data arrives each simulation period?
- **Climate/trend risk**: Are secular trends (social inflation, climate risk, litigation funding) considered in trend factors or scenario generation?
- **Machine learning integration points**: Are there areas where ML-based reserving (e.g., individual claim reserving models, DeepTriangle) could enhance the framework as a future extension?
- **Regulatory and standard alignment**: Do methodologies align with relevant Actuarial Standards of Practice (ASOPs), particularly ASOP 23 (Data Quality), ASOP 25 (Credibility), ASOP 36 (Statements of Actuarial Opinion on P&C Loss Reserves), and ASOP 43 (Unpaid Claim Estimates)?

Label issues with: `actuarial-methodology`, and one of `priority-high`, `priority-medium`, `priority-low`. (Create the `actuarial-methodology` label if it doesn't exist.)

#### 5. API Usability Reviewer

Review the public API surface for:
- **Consistency**: Naming conventions, parameter ordering, return type patterns across similar functions
- **Discoverability**: Can an actuary find what they need? Are module/class/method names intuitive for the domain?
- **Error messages**: Are errors actionable? Do they guide the user toward the fix? Are domain-specific constraints explained?
- **Configuration**: Is the config system (config/ package, ConfigManager, presets) intuitive? Are defaults sensible for typical actuarial use?
- **Type safety**: Missing type annotations on public APIs, overly permissive types (Any, dict), inconsistent Optional handling
- **Documentation gaps**: Public methods without docstrings, misleading docstrings, missing parameter/return descriptions
- **Breaking change risks**: APIs that will need breaking changes before v1.0 — identify these now
- **Output format**: Are results returned in formats actuaries expect (DataFrames, structured dicts, named tuples)?
- **Visualization API**: Is the plotting API consistent? Can users easily customize charts for reports?
- **Print statements**: Any `print()` calls that should use `logging` instead

Label issues with: `api-usability`, and one of `priority-high`, `priority-medium`, `priority-low`.

#### 6. Security Reviewer

Review all production modules for:
- **Deserialization**: `safe_pickle.py` — is it actually safe? What classes are allowed? Can the allowlist be bypassed?
- **Input validation**: Are user-provided configs validated before use? Can malformed YAML/JSON configs cause unexpected behavior?
- **Dependency audit**: Review `pyproject.toml` dependencies for known CVEs or pinning issues
- **Code injection**: Any use of `eval()`, `exec()`, `subprocess` with user-controlled input, or `os.system()`
- **Path traversal**: File I/O operations that accept user-provided paths without sanitization
- **Information disclosure**: Verbose error messages that expose internal state, stack traces in production output
- **Resource exhaustion**: Inputs that could trigger unbounded memory/CPU usage (e.g., extremely large simulation configs)
- **Randomness**: Is `secrets` used where cryptographic randomness is needed? Is `random` vs `numpy.random` usage appropriate?

Label issues with: `security`, and one of `priority-high`, `priority-medium`, `priority-low`. (Create the `security` label if it doesn't exist.)

#### 7. Issue Validation Reviewer

For each of the 30 existing open GitHub issues:
- **Verify the problem still exists**: Read the referenced code and confirm the issue is still present
- **Validate the priority**: Is the assigned priority (high/medium/low) appropriate given v1.0 readiness?
- **Check completeness**: Does the issue have enough detail to be actionable? Are code locations still accurate?
- **Cross-reference**: Note if multiple issues are actually the same root cause
- **Add a comment** on each issue with validation findings (e.g., "Validated: issue still present in `risk_metrics.py:47`. Priority seems appropriate." or "Priority should be upgraded to high — this affects core simulation accuracy.")
- **Relabel/reprioritize** if warranted, but **never close** an open issue

Also produce a summary table in the Markdown tracker showing each issue's validation status.

---

### Issue Filing Requirements

Every GitHub issue must include:

```markdown
## Problem Statement
[1-2 sentence description of what's wrong or what's missing]

## v1.0 Impact
[Is this a v1.0 blocker? Why or why not?]

## Affected Code
- `ergodic_insurance/module.py:L42-L58` — [what this code does wrong]
- `ergodic_insurance/other_module.py:L100` — [related location]

## Current Behavior
[What happens now, with a concrete example if possible]

## Expected Behavior
[What should happen instead, with reference to the standard/formula/spec]

## Alternative Solutions Evaluated
1. **[Approach A]**: [description] — Pros: [X]. Cons: [Y].
2. **[Approach B]**: [description] — Pros: [X]. Cons: [Y].
3. **[Approach C]**: [description] — Pros: [X]. Cons: [Y].

## Recommended Approach
[Which alternative and why. Include reasoning about correctness, performance, and maintainability.]

## Acceptance Criteria
- [ ] [Specific, testable criterion 1]
- [ ] [Specific, testable criterion 2]
- [ ] [All existing tests continue to pass]
- [ ] [New test added to prevent regression]
```

**Labels**: Each issue must have exactly:
- One domain label: `performance`, `financial-accuracy`, `mathematical-correctness`, `actuarial-methodology`, `api-usability`, or `security`
- One priority label:
  - `priority-high` — Must be resolved before v1.0. Affects calculation accuracy, compliance, or prevents the framework from being used for robust research and real-world applications.
  - `priority-medium` — Should be resolved to enable new research and applications, but is not central to existing functionality. Can wait until after v1.0.
  - `priority-low` — Nice-to-have improvement for future iterations.

---

### Progress Tracking

Each reviewer must maintain a Markdown file in `docs/reviews/reviews_2026_02_12` documenting their progress:

- **Performance**: `docs/reviews/reviews_2026_02_12REVIEW_PERFORMANCE.md`
- **Financial**: `docs/reviews/reviews_2026_02_12REVIEW_FINANCIAL.md`
- **Mathematical**: `docs/reviews/reviews_2026_02_12REVIEW_MATHEMATICAL.md`
- **Actuarial Methodology**: `docs/reviews/reviews_2026_02_12REVIEW_ACTUARIAL_METHODOLOGY.md`
- **API Usability**: `docs/reviews/reviews_2026_02_12REVIEW_API_USABILITY.md`
- **Security**: `docs/reviews/reviews_2026_02_12REVIEW_SECURITY.md`
- **Issue Validation**: `docs/reviews/reviews_2026_02_12REVIEW_ISSUE_VALIDATION.md`

Each file should contain:

```markdown
# [Domain] Review — Progress Tracker

## Summary
- **Reviewer**: [agent name]
- **Files reviewed**: X / Y
- **Issues filed**: N
- **Issues by priority**: H high, M medium, L low

## Areas Reviewed
| Module | Status | Issues Found | Notes |
|--------|--------|-------------|-------|
| `module.py` | Reviewed | #NNN, #NNN | [brief note] |
| `other.py` | In Progress | — | — |
| `pending.py` | Not Started | — | — |

## Remaining Work
- [List of modules not yet reviewed]

## Cross-Cutting Observations
- [Patterns that span multiple modules]
- [Systemic issues worth noting]
```

---

### Minimum Issue Targets

Each domain reviewer should identify **at least 10 actionable `priority-high` issues**. If fewer than 10 `priority-high` issues are found in any domain, the reviewer must include a section in their Markdown tracker explaining:
- Which specific areas were reviewed
- Why the codebase is already well-optimized in those areas with regard to high priority issues as defined above
- Any borderline findings that were considered but not filed, and why

---

### Coordination Rules

1. **No duplicate issues**: Check existing issues AND coordinate with other reviewers (via the team lead) before filing
2. **Cross-domain findings**: If a performance reviewer finds a mathematical error, report it to the team lead for routing to the math reviewer
3. **Disagreements**: If two reviewers have conflicting assessments (e.g., a financial reviewer says a calculation is correct but the math reviewer disagrees), document both perspectives in the issue
4. **Priority calibration**: The team lead should review all `priority-high` issues before they're filed to ensure consistent calibration

---

### Final Deliverable

When all reviewers complete their work, the team lead produces a summary in `docs/reviews/reviews_2026_02_12REVIEW_SUMMARY.md`:

```markdown
# Codebase Review Summary

## Overview
- **Date**: [date]
- **Version reviewed**: 0.9.0
- **Branch**: develop
- **Total issues filed**: N
- **Priority breakdown**: H high, M medium, L low

## By Domain
| Domain | Issues Filed | High | Medium | Low | Key Finding |
|--------|-------------|------|--------|-----|-------------|
| Performance | N | H | M | L | [one-liner] |
| Financial | N | H | M | L | [one-liner] |
| Mathematical | N | H | M | L | [one-liner] |
| Actuarial Methodology | N | H | M | L | [one-liner] |
| API Usability | N | H | M | L | [one-liner] |
| Security | N | H | M | L | [one-liner] |

## Existing Issue Validation
- **Validated as still open**: N
- **Recommended priority changes**: N
- **Duplicate clusters identified**: N

## v1.0 Readiness Assessment
[Brief assessment of how far the codebase is from v1.0 based on review findings]

## Top 5 Most Critical Issues
1. #NNN — [title] — [why it's critical]
2. ...
```

---

### Important Reminders

- This package is used in production financial systems. Thoroughness matters more than speed.
- Prefer **specific, actionable findings** over general advice. "Consider optimizing X" is not helpful; "Replace O(n^2) loop at `module.py:L42` with vectorized operation, expected 10x speedup for n>1000" is helpful.
- Every claim about incorrectness must reference the authoritative source (GAAP codification, textbook formula, SciPy documentation, etc.).
- Review **all files** in your domain, not just the obvious ones. Infrastructure modules (`batch_processor.py`, `trajectory_storage.py`, `safe_pickle.py`) often harbor subtle issues.
- Analyze all files not just the obvious ones.
