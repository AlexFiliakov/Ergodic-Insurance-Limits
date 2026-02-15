# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- next-version -->

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
