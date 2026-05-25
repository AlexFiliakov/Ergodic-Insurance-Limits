# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- next-version -->

## [0.16.0] - 2026-05-25

### Added
- shift TARGET_LOSS_RATIO to 0.60 (Normal-Hard Market) ([`23507b6`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/23507b6777270963ad43c60b144237b07dd8d9ee))
- hybrid Gauss-Hermite + stratified-Pareto jump-term quadrature ([`6932de1`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/6932de1d1c087f08b171fb572561585a022f55ab))

### Changed
- calibrating optimization no7 notebook ([`6a49fac`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/6a49facc4e2b1a5f8b4b98a0215ddcf10a3b9c56))
- optimization no7 enhancements (incl. boundary conditions) and documentation ([`f038838`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/f038838cbe5e1ce86873c64dadc28bfba1ae11b5))
- optimization no7 notebook config changes, but calibration still in progress ([`68f7ca3`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/68f7ca3c5804d3f712f167950b25d8ee8e4e92a2))
- optimization no7 notebook fixes, but still in progress ([`c5b3e19`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/c5b3e194605500cf3cfc24c29a1b61d80d4e1ffc))
- optimization no07 notebook pip install from `develop` branch instead of PyPI for now ([`bea0c19`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/bea0c1944306274a646a9a76197798a972cd946e))
- fix optimization no7 notebook smooth jump and other patches ([`bc8da7b`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/bc8da7bd15b167f814aae394a90d040fadf6029e))
- optimization no7 notebook, calibrating HJB solver in progress ([`44a2f9b`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/44a2f9ba1557c85ee0f0a282c275793af15b9711))
- optimization no7 notebook enhanced HJB runtime calibration ([`f9e93bf`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/f9e93bf4af810c266ca72857fb4346d4d1eedf72))
- optimized HJB solver in notebook optimization no7 ([`8e0e92d`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/8e0e92de5f39dc45ff8ae2e39d7b74ba2839aef8))
- CLAUDE.md focusing ([`95c027a`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/95c027a4873a12f9fd9c74cec321df38237d9328))

### Fixed
- tighten convergence, document RUIN_PENALTY methodology, add Merton regression test ([`5698db1`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/5698db176be79a693ba5fed3df2d829f2eadb8b2))
- drop double-counted loss drag from sensitivity worker drift ([`8650cf5`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/8650cf54a73f6960d0294a29e08691595e6d098a))
- additional solver control keys ([`26b7ee4`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/26b7ee4a2765f3003e95599b0a0eb9314a0eafa7))

[0.16.0]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.15.0...v0.16.0

## [0.15.0] - 2026-05-16

### Added
- HJB enhancements and optimization #7 notebook parallelization ([`2e9dd73`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/2e9dd73a4cbddaf33bc85900dede6d68578ded70))
- HJB enhancements ([`c443c4f`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/c443c4f468d4d3e5bcc839500eff379675518ffb))

[0.15.0]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.14.0...v0.15.0

## [0.14.0] - 2026-05-16

### Added
- implement jump-aware HJB ([`e7a60fe`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/e7a60fe4d89b87acb1110322d4cf13a5ad68f5ac))

### Changed
- failing accounting tests around recoveries ([`e8b40a7`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/e8b40a768f6499955405094341146040a0196786))
- HJB intermediate enhancement, but fundamental flaw in the setup ([`1e3059f`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/1e3059f72168aa7048182298901f0822b352a315))
- HJB experiment enhance search grid ([`b1c3ab7`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/b1c3ab7a3fda8b244e4dd82f8f5a1285cdb2d42c))
- HJB experiment #7 updates ([`4fae216`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/4fae216f32b8080fdf0d9f68d5437821661b5c28))
- update HJB notebook to use EE terminology rather than EUT ([`1d0e2af`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/1d0e2af0e97cadb4ccd68a34684f495cef59fc99))
- fixed rendering mermaid diagrams ([`de66371`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/de66371ebad8709ed78069be76b6a48f17dea990))
- rec 15 infinite moments experiment ([`f8a49bb`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/f8a49bb97c546345ad4dbb3ebf7951fe31ed24a6))
- rec 15 infinite moments sketch ([`d4ac805`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/d4ac805abfad3d394d6856a158598c0e2661535a))
- opt 03 sensitivity add charts ([`f37441d`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/f37441dbd7da07a73656102105c2098d0b9b0c9c))
- opt update pareto analyses ([`7b97f00`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/7b97f00e7a3458ec9c3cffe4bf3363f36c630a1c))
- rec 11 shadow mean added extra plots ([`4d08d18`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/4d08d18a94efafe2f49b4d8f13e530332ccda964))
- rec 14 expectile study result ([`2c1adee`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/2c1adee97fb88fdc1371223756c05af5962bcbc1))
- rec 14 expectile ([`29b867c`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/29b867cb8da7cec142741375d98913cf3e019601))
- rec 13 risk measure sensitivity experiment results ([`857e9b5`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/857e9b55a42430e10b03d3ad44391c053dcf5da2))
- rec 13 risk measure sensitivity experiment ([`8f4ed2c`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/8f4ed2c4b6de5d0290829bfb2f9d1a2fd611b589))
- opt 12 drags experiment results ([`ff9faed`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/ff9faedd1ed368602e117ae4ab310b828d253e57))
- claude and two experiment sketches ([`12e0a1a`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/12e0a1a59e8667f302bb580a0704d90fea182902))
- opt 11 market cycles experiment completed ([`1adec3b`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/1adec3b9d30f33a1abd94ff49367c0768d121bc0))
- opt 11 market cycles continuing to calibrate experiment setup ([`b50290c`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/b50290c98e0dd856ccbead17f3e9edf38163bbc2))
- opt 11 market cycle tweaking config ([`91feff8`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/91feff8ca20531a77dd4f0c893605427959878b9))
- opt 11 market cycle tweaking experiment, still a bit rough ([`c807723`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/c807723bca1ce7c81b5eaf40ae61ccb60b3ccbfc))
- opt 11 market cycle set up experiment ([`3af2ede`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/3af2ede2dee2bb1b755c34e564a2d915ab9d57da))
- ran early optimization experiments, set up experiment opt 11, hid more .gitignore cache ([`9db6fef`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/9db6feff62645287defad9a0e76c8fc289d8e102))
- recon 11 severity distribution nicer example ([`ee10042`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/ee10042d6cd3069e4063d417547aab9e42d24fcf))
- recon 11 severity distribution example ([`4a98a14`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/4a98a140afb2de7b6327188f3384360b513b6207))
- recon 11 severity distribution still needs validation ([`9ed70cb`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/9ed70cbc012ecbfc94c9bb1a4372d55ea0092cfe))
- 10 hollow tower demand curve results ([`8cb9ced`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/8cb9ceddf611ef58a2c8f55bb132a92eb0b13d19))
- 11 hollow tower demand curve results ([`1a9a2da`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/1a9a2dad50ead898b0ee32df93a71524a02fb14f))
- recon 11 severity distribution study draft ([`740cccb`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/740cccb61dba8891db71f203c5185de558b59857))
- recon 11 severity distribution notebook experiment frame ([`9264305`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/9264305a6e2554aba6a24ed5f05907b2b0037e30))
- recon 11 severity distribution initial notebook ([`a4eb588`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/a4eb588a6d577306d5332dffd34d5ed46edaebe9))
- fixing mypy issues ([`efb74f4`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/efb74f4d7b61ae561726b847c381a3d3cd903f96))
- added valid cell IDs to notebooks as required by the new Jupyter format. ([`06745fe`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/06745fe4aa22c1ba4fa512759154f4268279e7d8))

### Fixed
- clarify insurance receivables and gross loss recognition per ASC 410-30 ([`05a4cb8`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/05a4cb88005b74a2f59d6854fd158922d3fb9a9f))

[0.14.0]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.13.10...v0.14.0

## [0.13.10] - 2026-02-21

### Changed
- optimization 09 results and 10 setup ([`23f9029`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/23f90297fdf7addb9e725f3d2633ebfc4c111724))
- optimization 09 and 10 experiments are set up to run ([`7a8f59c`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/7a8f59c768d7d1ae29b63c65a8cae9964e50a44a))
- 09 tower demand experiment improvements ([`3b66108`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/3b661086d82b87839202f300887519761e146221))
- 10 hollow tower experiment improvements ([`a00e799`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/a00e799db753a1cfc64d54c23e3e787a6b959103))
- 09 tower demand preliminary experiment result ([`a1bbf3d`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/a1bbf3d3962003e2b68389b5c06c83f908f33ddd))
- 09 run and 10 hollow tower setup ([`f9e87b3`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/f9e87b37cb42896a4bc14c50e1886654782533ff))
- 10 hollow tower demand curve initial sketch ([`8e0b743`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/8e0b743274b6b5d8390a24f38c9042596eacd7a1))
- 09 demand curve update and klein bottle exploration ([`7cf2b35`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/7cf2b35128e340897f83101cfda018e579ad6298))
- 09 demand curve updated notebook ([`f8e9b70`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/f8e9b7022d7373cfa95240c831fc7e63e82273b5))
- 09 demand curve initial experiment sketch ([`c88eed4`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/c88eed4f5b2945a0e6fd164a2234f51448406dae))

### Fixed
- opt 09 & 10 minor experiment patches ([`ccb09b6`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/ccb09b65749d48ec453932325ee947c0540eb599))

[0.13.10]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.13.9...v0.13.10

## [0.13.9] - 2026-02-20

### Fixed
- tower experiment results ([`318ff4f`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/318ff4fea055b0aea9f49012967d2d8bba4ff048))
- initial tower experiment sketch ([`16dd97c`](https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/commit/16dd97cac8df5a2acd44016725208813810aae61))

[0.13.9]: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/compare/v0.13.8...v0.13.9

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
