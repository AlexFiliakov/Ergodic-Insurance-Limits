# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- next-version -->

## [0.13.8] - 2026-02-20

### Fixed
- notebooks metadata.widgets issue causing them to not display in GitHub ([`d72ac4d`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/d72ac4d868f0687d60bee302ae633dd1b001e7af))

[0.13.8]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.13.7...v0.13.8

## [0.13.7] - 2026-02-19

[0.13.7]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.13.6...v0.13.7

## [0.13.6] - 2026-02-19

### Fixed
- added sensitivity to pareto analysis ([`b6371ac`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/b6371ac23f5bd09976a7e808bfbd747d7c8d7182))

[0.13.6]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.13.5...v0.13.6

## [0.13.5] - 2026-02-19

[0.13.5]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.13.4...v0.13.5

## [0.13.4] - 2026-02-19

### Changed
- working paper inviting collab, useful prompts to refactor ([`4c44120`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/4c441208c061e14a010afddbc337a8b6020c9b89))
- comprehensive test suite cleanup ([`dddd8a0`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/dddd8a04fa0bc4792ee7214bc83845b4b4a8a3a0))

### Fixed
- pareto analysis and fixed text in hjb example ([`38a6f11`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/38a6f11bb511941fe05115ff00effd6635ba646f))

[0.13.4]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.13.3...v0.13.4

## [0.13.3] - 2026-02-17

### Fixed
- updated hjb example notebook ([`7ce58e6`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/7ce58e6952de44c6fbba719eb070616fc15c7199))
- core hjb solver to deal with large numbers ([`aebe7eb`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/aebe7ebbc7dc1ad0fd9438971df9eb0d04eb21b4))
- allow log hjb solver and recalibrate noetbook ([`ecf4899`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/ecf4899cfc36011d1cbceb20d4ddc810ec16a449))
- recalibrate notebook 07 insurance pricing to produce genuine SIR tradeoff ([`60f265d`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/60f265d050a2a19d35819b9cac39cbc0b10e7c25))
- when retention is higher than certain layers, skip them during program definition. Also new HJB exploration. ([`76dab1d`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/76dab1ded50844a825b1fbc979cfa4e3cc33822c))
- correct ergodic gap calculation in reconciliation notebook 07 ([`336c7b3`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/336c7b325f6facfc5b2ff34b53e2d58b25258548))

[0.13.3]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.13.2...v0.13.3

## [0.13.2] - 2026-02-15

### Fixed
- update README.md ([`7f3ff6b`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/7f3ff6bb1201f753b83979e5edc87ddd2f57af30))
- updated all `pip install` references to PyPI ([`3426227`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/3426227f0170ab1b253e957cb3d8a629241bdc56))

[0.13.2]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.13.1...v0.13.2

## [0.13.1] - 2026-02-15

### Changed
- updated notebooks to run in Colab and fixed README images for PyPI ([`379d9f7`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/379d9f71fff21194acfc61e243bc66540e96035b))

### Fixed
- all notebooks to run on Google Colab ([`c36f282`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/c36f2827efbd16d46ab4bd779a96ec932024b426))

[0.13.1]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.13.0...v0.13.1

## [0.13.0] - 2026-02-15

### Fixed
- skip unknown commits in changelog templates to prevent ParseError crash ([`a51e577`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/a51e5773104ace643711c980d8a812ac8202b8d9))
- disable attestations in PyPI publish to resolve 400 error ([`b9085eb`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/b9085ebc5bded4007c04b00e62269065091c31cf))
- compute pure premium as mean annual aggregate instead of freq*sev ([`4f2ded6`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/4f2ded61d8afadf79665c805b7c24a61a69042b4))

[0.13.0]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.12.0...v0.13.0

## [0.12.0] - 2026-02-15

### Changed
- add PyPI trusted publishing to release workflow (#1514) ([`ec8b08a`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/ec8b08a46b3abd5e53b62f576b1c16d67f0097c0))

[0.12.0]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.11.0...v0.12.0

## [0.11.0] - 2026-02-15

### Added
- GAAP accounting overhaul, GPU acceleration, API ergonomics, and 100+ bug fixes (#1416) ([`c2e89a0`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/c2e89a0968e81cb45f15408e99edf0052b1c66e4))

[0.11.0]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.10.0...v0.11.0

## [0.10.0] - 2026-02-13

### Added
- GPU acceleration, HJB solver fixes, GAAP accounting corrections, and config unification (#1132) ([`d3437aa`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/d3437aa28968daa8af0e496e8145254986d97f9a))

[0.10.0]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.9.0...v0.10.0

## [0.9.0] - 2026-02-11

### Added
- dot-notation overrides, stochastic claims, and perf boosts (#594) ([`d9f9b7b`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/d9f9b7b55fe6d196fa2d63d4da761410df901365))

[0.9.0]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.8.0...v0.9.0

## [0.8.0] - 2026-02-10

### Added

- API cleanup — rename SimulationResults to MonteCarloResults, deprecate InsurancePolicy in favor of InsuranceProgram, and fix validation (#579)

## [0.7.0] - 2026-02-09

### Added

- HJB numerical overhaul, GAAP compliance, and actuarial features (#564)

### Changed

- Skip docs build on semantic-release version-bump commits

## [0.6.0] - 2026-02-08

### Added

- Major API improvements, math corrections, and config refactor (#526)

## [0.5.0] - 2026-02-07

### Added

- Ergodic defaults, quick-start API, NOL tracking, and optimizer fixes (#444)

### Fixed

- Use PAT for semantic-release to bypass branch protection (#509)
- Skip unparseable commits in changelog templates (#457)

## [0.4.3] - 2026-02-06

### Changed

- Remove Decimal arithmetic from worker hot loop for major performance improvement (#368)

## [0.4.2] - 2026-02-06

### Fixed

- Unify simulation execution semantics across all code paths (#349)

## [0.4.1] - 2026-02-06

### Fixed

- CI enhancements — consolidate config, update hooks, fix Monte Carlo state (#426)

### Changed

- Fix broken code examples across 8 documentation files

## [0.4.0] - 2026-02-05

### Added

- Event-sourcing ledger for financial statements (#246)
- Mid-year liquidity detection to prevent blind spot insolvency (#279)
- Generalized Pareto Distribution (GPD) support for extreme events (#198)
- Limited liability to prevent negative equity (#200)
- Ledger pruning option to bound memory usage in simulations
- O(1) balance cache for Ledger class (#259)
- Common Random Numbers (CRN) strategy for simulation reproducibility
- T-digest streaming quantile algorithm (#333)
- Configurable insolvency tolerance replacing $1 equity floor (#208)
- Volatility parameter and SeedSequence for simulations (#317)
- Default to claim liability with letter of credit in Monte Carlo engine (#342)

### Fixed

- Eliminate phantom cash injection and fix cash flow accounting (#319)
- Remove data fabrication from financial statements (#301)
- Correct ledger equity accounting errors (#302)
- Re-seed parallel MC workers to produce distinct loss sequences (#299)
- Resolve split brain architecture between Ledger and metrics_history (#254)
- Remove working capital double-counting in revenue calculation (#244)
- Consolidate tax accrual logic with TaxHandler class (#245)
- Fix dividend phantom payments — track actual dividends considering cash constraints (#239)
- Correct systematic insurance pricing bug in loss calculation formula (#204)
- Fix double counting of insurance premiums in cash flow statement (#212)
- Fix math domain error in time-weighted ROE for total loss scenarios (#211)
- Fix incorrect gross margin calculation in financial statements (#215)
- Standardize Decimal types across financial calculations (#308)
- Convert financial calculations from float to Decimal (#258)
- Make Ledger single source of truth to resolve cash flow divergence (#275)
- Add deep copy support to fix Monte Carlo worker state corruption (#273)
- Replace non-deterministic hash() and insecure pickle with safe alternatives (#312)
- Add over-recovery guard and fix premium double-loading (#310)
- Stop simulation on insolvency, make Simulation re-entrant (#304)
- Migrate deprecated NumPy RNG APIs to modern Generator/default_rng (#311)
- Use per-mille quantile keys to prevent sub-percentile collisions (#334)
- Remove unsafe data estimation from reporting layer (#256)
- Remove hardcoded COGS/SGA breakdown from reporting layer (#255)
- Consolidate config.py and config_v2.py into single configuration module (#236)
- Replace naive seed incrementing with SeedSequence for statistical independence (#233)
- Fix Working Capital Warm-Up Distortion (#232)
- Fix fragile optional imports hiding logic (#214)
- Fix inconsistent claim payment tracking (#213)

### Changed

- Integrate ClaimLiability with ClaimDevelopment using Strategy Pattern (#274)
- Review and update tutorial documentation for accuracy (#220)
- Deprecate ClaimGenerator in favor of ManufacturingLossGenerator (#235)
- Replace simplified ergodic metrics with empirical comparison (#234)
- Remove depreciation fallback logic from reporting layer (#231)

## [0.3.0] - 2025-09-21

### Added

- Dynamic insurance premium scaling based on revenue exposure (#190)
- Periodic ruin probability tracking in MonteCarloEngine (#192)
- Working capital configuration and revenue-scaled premium calculation
- Trend infrastructure for ClaimGenerator — core trends (#181), stochastic types (#182), integration (#183), and test suite (#184)
- Financial Data Configuration Framework (#176)
- Configurable PPE ratio in ManufacturerConfig (#175)
- Accrual and Timing Management System (#174)
- Insurance Premium Accounting Module (#172)
- Per-occurrence and aggregate limit types for insurance layers (#171)
- Proper three-section cash flow statement (#170)
- Enhanced income statement with GAAP expense categorization (#169)
- Enhanced balance sheet with GAAP structure (#162)
- Analytical and simulated statistics properties for ClaimGenerator (#188)
- Loss generation handling for both generate_losses and generate_claims methods

### Fixed

- Correct prepaid insurance tracking using insurance accounting module
- Implement proper accounting equation tracking (#166)

## [0.2.0] - 2025-09-17

### Added

- Exposure Base module for dynamic frequency scaling (#151)
- Insurance pricing module with market cycle support (#123)
- Retention optimization with analytical methods and visualizations
- Insurance tower visualization
- Comprehensive Sphinx documentation system with theory, tutorials, and API reference
- GitHub Pages deployment with Jekyll and MathJax integration
- Tutorial redirect pages for improved navigation

### Fixed

- Prevent insurance premiums from immediately reducing productive assets
- Fix financial bugs in manufacturer module
- Correct equity calculation when processing claims
- Fix inline math vertical alignment in documentation

### Changed

- Refactor exposure bases to use state providers (#153)
- Update insurance cost handling in ROE calculations
- Refactor claim processing to track payments as losses and expenses

## [0.1.0] - 2025-08-29

### Added

- Hamilton-Jacobi-Bellman Solver for optimal insurance control (#82)
- High-Performance Monte Carlo Simulation Engine with parallel execution (#23)
- Comprehensive Risk Metrics Suite for tail risk analysis (#28)
- Constrained Optimization Solver (#73)
- Monte Carlo ruin probability estimation (#72)
- ROE calculation framework (#71)
- Algorithmic Insurance Decision Engine (#31)
- Scenario batch processing framework (#51)
- Walk-Forward Validation System (#54)
- Memory-efficient trajectory storage system (#49)
- Enhanced parallel simulation architecture (#48)
- 3-tier configuration system (#84)
- Visualization suite — ruin cliff (#62), ROE-ruin frontier, premium multiplier, executive figures, convergence, and ergodic theory visualizations
- Caching system for expensive computations (#98)
- Visualization Factory with consistent styling (#58)
- Performance optimization and benchmarking suite (#55)
- Excel report generation for financial statements (#90)
- Automated report generation system (#68)
- Advanced convergence monitoring (#56)
- Parameter sweep utilities (#81)
- Comprehensive sensitivity analysis tools
- Statistical significance testing framework
- Advanced result aggregation framework
- Claim development patterns for cash flow modeling (#27)
- Multi-layer insurance program with reinstatements (#20)
- Enhanced loss distributions for manufacturing risks (#19)
- Basic ergodic analysis framework (#17)
- Simple insurance layer structure (#15)
- Basic stochastic processes (#14)
- ClaimGenerator with comprehensive tests (#12)
- Pareto Frontier visualization and analysis tool (#74)
- Scenario comparison and annotation framework (#70)

[0.8.0]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.4.3...v0.5.0
[0.4.3]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/releases/tag/v0.1.0
