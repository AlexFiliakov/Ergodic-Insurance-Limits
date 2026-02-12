# Prompt: Refactor, Update, and Extend Jupyter Notebooks

## Instructions for Claude Code

Create a Claude Code agent team (Opus 4.6) to refactor, reorganize, update, and extend the Jupyter notebooks in `ergodic_insurance/notebooks/`. This is a large-scale effort touching 30+ numbered notebooks, several unnumbered research notebooks, and a reproducible research subfolder.

**Work on a feature branch** off `develop` (e.g., `feature/notebook-refactor`). Never commit to `main`. Use conventional commit prefixes (`refactor:`, `feat:`, `docs:`). Make a PR to `develop` when done.

---

## Context: What This Project Does

This project implements a framework for assessing insurance purchasing decisions using **ergodic theory** (time-average growth) rather than traditional ensemble (expected value) approaches. The core insight: insurance that appears "expensive" by expected-value metrics can be **optimal** when you evaluate time-average growth, because wealth dynamics are multiplicative.

Key classes and modules:
- `ManufacturerConfig` / `WidgetManufacturer` — Financial model of a widget manufacturer
- `ManufacturingLossGenerator` — Stochastic loss generation (attritional, large, catastrophic)
- `RevenueExposure` / `AssetExposure` / `EquityExposure` — State-driven exposure bases
- `InsurancePolicy` / `InsuranceLayer` / `InsuranceProgram` / `EnhancedInsuranceLayer` — Insurance structures
- `Simulation` / `MonteCarloEngine` / `MonteCarloConfig` — Simulation engines
- `ErgodicAnalyzer` — Compares ensemble vs time-average metrics
- `InsuranceDecisionEngine` — Optimization algorithms (SLSQP, differential evolution, weighted sum)
- Visualization modules: `WSJ_COLORS`, `format_currency`, Plotly + Matplotlib outputs
- Reporting modules: Excel reports, table generation, scenario comparison

---

## Current State of the Notebooks

### Numbered Series (00–30)
Located in `ergodic_insurance/notebooks/`:

| # | File | Current Purpose | Issues |
|---|------|----------------|--------|
| 00 | `00_notebook_executioner.ipynb` | Runs all notebooks sequentially via papermill | Utility, not tutorial |
| 00 | `00_setup_verification.ipynb` | Verify imports and environment | Uses `sys.path.insert` hack |
| 01 | `01_basic_manufacturer.ipynb` | Basic financial model demo | Good structure, clean |
| 02 | `02_long_term_simulation.ipynb` | 1000-year stochastic simulation | Memory-intensive (~19GB) |
| 02L | `02_long_term_simulation_light.ipynb` | Lighter version of 02 | Redundant with 02 |
| 03 | `03_growth_dynamics.ipynb` | Growth dynamics + insurance interaction | |
| 04 | `04_ergodic_demo.ipynb` | Core ergodic insight demonstration | Has debug cells left in, duplicate code blocks (cells 17 and 19 are nearly identical), hardcoded parameters |
| 05 | `05_risk_metrics.ipynb` | Risk metrics (VaR, TVaR, etc.) | |
| 06 | `06_loss_distributions.ipynb` | Loss distribution modeling | |
| 07 | `07_insurance_layers.ipynb` | Insurance layer/program structures | |
| 08 | `08_monte_carlo_analysis.ipynb` | Monte Carlo engine usage | |
| 09 | `09_optimization_results.ipynb` | Optimization algorithm comparison | Large (196KB), uses simulated/hardcoded results in places |
| 10 | `10_sensitivity_analysis.ipynb` | Sensitivity analysis | **Duplicates** notebook 26 |
| 11 | `11_pareto_analysis.ipynb` | Pareto frontier analysis | |
| 12 | `12_hjb_optimal_control.ipynb` | Hamilton-Jacobi-Bellman solver | Advanced/theoretical |
| 13 | `13_walk_forward_validation.ipynb` | Walk-forward backtesting | |
| 14 | `14_visualization_factory_demo.ipynb` | Visualization factory patterns | |
| 15 | `15_roe_ruin_frontier_demo.ipynb` | ROE vs ruin probability frontier | |
| 16 | `16_ruin_cliff_visualization.ipynb` | Ruin cliff / threshold effects | |
| 17 | `17_executive_visualizations_showcase.ipynb` | Executive-level visualizations | **Overlaps** with 18 |
| 18 | `18_executive_visualization_demo.ipynb` | Executive visualization demo | **Overlaps** with 17 |
| 19 | `19_technical_convergence_visualizations.ipynb` | Convergence monitoring plots | **Overlaps** with 22 |
| 20 | `20_ergodic_visualizations.ipynb` | Ergodic-specific visualizations | |
| 21 | `21_insurance_structure_visualizations.ipynb` | Insurance structure diagrams | |
| 22 | `22_advanced_convergence_monitoring.ipynb` | Advanced convergence analysis | **Overlaps** with 19 |
| 23 | `23_table_generation_demo.ipynb` | Table/report generation | |
| 24 | `24_scenario_comparison_demo.ipynb` | Scenario comparison tooling | |
| 25 | `25_excel_reporting.ipynb` | Excel report generation | |
| 26 | `26_sensitivity_analysis.ipynb` | Sensitivity analysis | **Duplicates** notebook 10 |
| 27 | `27_parameter_sweep_demo.ipynb` | Parameter sweep grid search | |
| 28 | `28_retention_optimization_demo.ipynb` | Retention optimization (91KB, large) | |
| 29 | `29_report_generation_demo.ipynb` | Report generation pipeline | |
| 30 | `30_state_driven_exposures.ipynb` | State-driven exposure bases | Good, recent |

### Unnumbered / Research Notebooks
| File | Purpose |
|------|---------|
| `ergodicity_basic_simulation.ipynb` | Basic volatility simulation (31KB) |
| `ergodicity_basic_simulation_parallel.ipynb` | Parallel version (~4KB, thin wrapper) |
| `ergodicity_basic_simulation_results.ipynb` | Results visualization (1.5MB — embedded outputs) |
| `ergodicity_vol_sim_parallel.ipynb` | Volatility simulation with parameter sweeps (38KB) |
| `run_basic_simulation.py` | Python script for simulation |
| `run_basic_simulation_colab.py` | Colab-compatible runner |
| `run_vol_sim_colab.py` | Colab-compatible volatility sim |
| `bootstrap_demo.py` | Bootstrap CI demonstration |
| `axis_formatter.py` | Matplotlib axis formatter utility |

### Reproducible Research Folder
`reproducible_research_2026_02_02_basic_volatility/` — A self-contained research pipeline:
- `0.a.` Run simulation
- `0.b.` Copy data from Google Drive
- `1.` Process results
- `2.a.` EDA space explorer
- `2.b.` EDA paired path browser
- `3.` Generate publication-quality visualizations
- Has its own `README.md` and `VISUAL_SPECIFICATION.md`

### Support Files
- `custom_style_config.yaml` — Visualization style configuration
- `cache/`, `results/`, `reports/`, `exports/`, `excel_reports/`, `table_exports/`, `visualization_exports/`, `comparison_reports/`, `premium_analysis_exports/`, `tutorials/` — Output directories

---

## Target Reorganization

### New Folder Structure
```
ergodic_insurance/notebooks/
├── getting-started/           # Executive/newcomer track
│   ├── 01_setup_verification.ipynb
│   ├── 02_quick_start.ipynb           # NEW: 5-minute "aha" moment
│   └── 03_basic_manufacturer.ipynb    # From current 01
│
├── core/                      # Practitioner track (actuaries, risk managers)
│   ├── 01_loss_distributions.ipynb    # From 06
│   ├── 02_insurance_structures.ipynb  # From 07 + 21 (merged)
│   ├── 03_ergodic_advantage.ipynb     # From 04 (cleaned up)
│   ├── 04_monte_carlo_simulation.ipynb # From 08
│   ├── 05_risk_metrics.ipynb          # From 05
│   ├── 06_long_term_dynamics.ipynb    # From 02 + 03 (merged, with light/full modes)
│   └── 07_growth_dynamics.ipynb       # From 03
│
├── optimization/              # Advanced practitioner track
│   ├── 01_optimization_overview.ipynb  # From 09 (trimmed)
│   ├── 02_sensitivity_analysis.ipynb   # From 10/26 (deduplicated)
│   ├── 03_pareto_analysis.ipynb        # From 11
│   ├── 04_retention_optimization.ipynb # From 28
│   ├── 05_parameter_sweeps.ipynb       # From 27
│   └── 06_state_driven_exposures.ipynb # From 30
│
├── visualization/             # Visualization cookbook
│   ├── 01_visualization_factory.ipynb   # From 14
│   ├── 02_executive_dashboards.ipynb    # From 17+18 (merged)
│   ├── 03_ergodic_visualizations.ipynb  # From 20
│   ├── 04_convergence_monitoring.ipynb  # From 19+22 (merged)
│   ├── 05_ruin_analysis_plots.ipynb     # From 15+16 (merged)
│   └── 06_scenario_comparison.ipynb     # From 24
│
├── reporting/                 # Output generation
│   ├── 01_table_generation.ipynb   # From 23
│   ├── 02_excel_reporting.ipynb    # From 25
│   └── 03_report_generation.ipynb  # From 29
│
├── advanced/                  # Developer/researcher track
│   ├── 01_hjb_optimal_control.ipynb    # From 12
│   ├── 02_walk_forward_validation.ipynb # From 13
│   └── 03_advanced_convergence.ipynb    # NEW: deep convergence theory
│
├── research/                  # Working research notebooks (NOT tutorials)
│   ├── ergodicity_basic_simulation.ipynb
│   ├── ergodicity_basic_simulation_parallel.ipynb
│   ├── ergodicity_basic_simulation_results.ipynb
│   ├── ergodicity_vol_sim_parallel.ipynb
│   ├── run_basic_simulation.py
│   ├── run_basic_simulation_colab.py
│   ├── run_vol_sim_colab.py
│   ├── bootstrap_demo.py
│   └── reproducible_research_2026_02_02_basic_volatility/  # Keep as-is
│
├── _utilities/                # Infrastructure
│   ├── notebook_executioner.ipynb  # From 00
│   ├── axis_formatter.py
│   └── custom_style_config.yaml
│
├── cache/                     # Keep existing output directories
├── results/
├── reports/
└── ... (other output dirs)
```

### Key Merges Required
1. **10 + 26** → `optimization/02_sensitivity_analysis.ipynb` (deduplicate, keep best of both)
2. **17 + 18** → `visualization/02_executive_dashboards.ipynb` (merge overlapping content)
3. **19 + 22** → `visualization/04_convergence_monitoring.ipynb` (merge overlapping content)
4. **15 + 16** → `visualization/05_ruin_analysis_plots.ipynb` (merge ROE-ruin frontier + ruin cliff)
5. **02 + 02L** → `core/06_long_term_dynamics.ipynb` (add toggle for light/full mode in a single notebook)
6. **07 + 21** → `core/02_insurance_structures.ipynb` (merge layer logic + structure visualization)

---

## Quality Standards for Each Notebook

### Narrative Structure
Every refactored notebook MUST follow this template:

```
# Title: Clear, Descriptive Name

## Overview
- What this notebook demonstrates (1-2 sentences)
- Prerequisites (which notebooks to read first)
- Estimated runtime
- Audience indicator: [Executive] / [Practitioner] / [Developer]

## Setup
- Clean imports (no sys.path hacks — use `pip install -e .`)
- Reproducibility: set random seeds
- Configuration: use ManufacturerConfig properly

## Sections 1..N
- Clear markdown headers explaining WHAT and WHY before code
- Each code cell does ONE thing
- Print statements show key results inline
- Visualizations have titles, labels, legends

## Key Takeaways
- Bulleted summary of insights
- Connection to ergodic theory thesis
- Links to related notebooks

## Next Steps
- Where to go next in the notebook series
```

### Code Quality Standards
1. **No `sys.path.insert` hacks** — All notebooks should assume `pip install -e .` was run
2. **No debug/scratch cells** — Remove all `# Debug:` cells, commented-out blocks, and testing artifacts
3. **No duplicate code** — Extract repeated setup patterns into a shared utility or use consistent copy/paste-free patterns
4. **Consistent imports** — Use the same import style across all notebooks
5. **Consistent visualization style** — Use `WSJ_COLORS` and the project's style config consistently. Prefer Plotly for interactive notebooks, Matplotlib for static/publication outputs
6. **Reproducible** — Every notebook with stochastic elements must set a seed. State the seed prominently
7. **Parameterized** — Use variables (not magic numbers) for all key parameters. Define them in a clearly marked "Configuration" cell near the top
8. **Progressive complexity** — Within each folder, notebooks go from simple to complex
9. **Cross-references** — Each notebook should reference related notebooks by their new path
10. **No hardcoded/fake results** — If a cell generates fake results (as in notebook 09's `analyze_algorithm_performance`), replace with actual API calls or clearly mark as "illustrative only"

### Execution Requirements
- All notebooks in `getting-started/`, `core/`, and `optimization/` MUST execute cleanly via papermill in under 2 minutes each
- `visualization/` and `reporting/` notebooks MUST execute cleanly
- `advanced/` notebooks may be marked "manual run only" if they require >5 minutes or special dependencies
- `research/` notebooks are exempt from execution testing
- Update the notebook executioner to reflect the new folder structure and skip known long-runners

---

## Team Structure

Create a team with 5 agents:

### 1. Lead Coordinator (`team-lead`)
- **Type**: general-purpose
- **Role**: Orchestrate the overall effort, maintain the task list, review completed work
- **Responsibilities**:
  - Create the new folder structure
  - Create and assign tasks to other agents
  - Review merged/refactored notebooks for quality
  - Ensure cross-references between notebooks are correct
  - Update the notebook executioner for the new structure
  - Create a final `ergodic_insurance/notebooks/README.md` with the notebook catalog

### 2. Getting Started & Core Refactorer (`core-refactorer`)
- **Type**: general-purpose
- **Responsibilities**:
  - Refactor `getting-started/` notebooks (01-03)
  - Create the NEW `02_quick_start.ipynb` — a 5-minute notebook that demonstrates the ergodic advantage with minimal code
  - Refactor `core/` notebooks (01-07)
  - Execute merge of 02+02L and 07+21
  - Clean up notebook 04 (remove debug cells, deduplicate sections)
  - Ensure all core notebooks execute cleanly

### 3. Optimization & Advanced Refactorer (`optimization-refactorer`)
- **Type**: general-purpose
- **Responsibilities**:
  - Refactor `optimization/` notebooks (01-06)
  - Deduplicate notebooks 10+26 into single sensitivity analysis
  - Clean up notebook 09 (remove fake results, trim to reasonable size)
  - Refactor `advanced/` notebooks (01-03)
  - Move research notebooks to `research/` subfolder
  - Move utility files to `_utilities/`

### 4. Visualization & Reporting Refactorer (`viz-refactorer`)
- **Type**: general-purpose
- **Responsibilities**:
  - Refactor `visualization/` notebooks (01-06)
  - Merge 17+18 into executive dashboards
  - Merge 19+22 into convergence monitoring
  - Merge 15+16 into ruin analysis plots
  - Refactor `reporting/` notebooks (01-03)
  - Ensure consistent use of WSJ_COLORS and project style

### 5. Quality Assurance & Testing (`qa-tester`)
- **Type**: general-purpose
- **Responsibilities**:
  - Run each refactored notebook via papermill
  - Verify all imports resolve correctly (no sys.path hacks)
  - Check that all API calls match the current codebase
  - Verify cross-references between notebooks are valid
  - Test the updated notebook executioner
  - Report broken notebooks back to the responsible agent
  - Create a test matrix documenting pass/fail/skip status for every notebook

---

## Execution Plan

### Phase 1: Setup (Lead)
1. Create feature branch `feature/notebook-refactor` off `develop`
2. Create the new folder structure (empty directories)
3. Create task list and assign work

### Phase 2: File Migration (All agents in parallel)
4. Move/copy existing notebooks to new locations (preserving originals until verified)
5. Move research notebooks to `research/`
6. Move utilities to `_utilities/`

### Phase 3: Refactoring (Agents 2-4 in parallel)
7. Each agent refactors their assigned notebooks:
   - Add proper narrative structure (Overview, Setup, Takeaways, Next Steps)
   - Remove `sys.path.insert` hacks
   - Remove debug cells and commented-out code
   - Parameterize magic numbers
   - Update import paths
   - Fix or replace hardcoded/fake results
   - Add cross-references to related notebooks
8. Execute required merges (6 merge operations)
9. Create the new `02_quick_start.ipynb`

### Phase 4: Testing (Agent 5)
10. Run all notebooks via papermill
11. Report failures to responsible agents
12. Iterate until all required notebooks pass

### Phase 5: Cleanup (Lead)
13. Delete original files that have been migrated (after verification)
14. Update notebook executioner for new structure
15. Create `ergodic_insurance/notebooks/README.md` catalog
16. Review all work, commit, and create PR to `develop`

---

## Important Notes

- **Preserve git history**: Use `git mv` when moving files so rename tracking works
- **Don't bloat the repo**: Clear cell outputs from notebooks before committing (use `jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace`)
- **Don't break the build**: The `tests/` directory has tests that may reference notebook paths — check for any such references
- **Respect the API**: The public API is defined in `ergodic_insurance/__init__.py`. Use only documented imports
- **No emoji in code**: Follow the project style — only use emoji in markdown narrative if it enhances engagement and provides clarity
- **Commit messages**: Use `refactor: reorganize notebook X into folder/Y` format for moves, `docs: improve narrative in folder/Y` for content changes
