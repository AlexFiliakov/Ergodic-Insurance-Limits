# Clauce Code Prompt Planning

Let's discuss and plan a project based on the specification in `Financial Modeling Framework - Spec.md`. Read this file, ask me clarifying questions, and then let's come up with a project roadmap.

## Project Planning Questions

1. Scope & Focus: Are you looking to build a complete simulation framework, or would you prefer to start with a specific component (e.g., the loss modeling, ergodic optimization engine, or insurance layer structuring)?
2. Target Users: Who will use this framework - internal risk managers, actuaries, or is this for broader distribution/SaaS?
3. Data Sources: Do you have historical loss data for calibration, or should we build synthetic data generation capabilities based on the specified parameters?
4. Integration Requirements: Should this integrate with existing financial systems, or operate as a standalone analytical tool?
5. Visualization Needs: How important are interactive dashboards vs. static reports for presenting results?
6. Performance Constraints: The spec mentions 100K-1M Monte Carlo iterations. What's your tolerance for computation time (seconds, minutes, hours)?
7. Priority Features: Which is most critical initially:
- Basic ergodic growth rate calculations
- Multi-layer insurance optimization
- ROE optimization with ruin constraints
- Full Monte Carlo simulation engine
8. Technology Preferences: The existing simone project uses TypeScript. Would you prefer to continue in TypeScript or switch to Python (as mentioned in the spec) for better scientific computing libraries?

### Project Planning Responses

1. I'm looking to build the complete simulation framework and run simulations in order to make some preliminary conclusions.
2. Actuaries who are familiar with Python are the target users. The implementation should be in Python.
3. We should build synthetic data based on the specified parameters. I don't have industry sources yet to calibrate.
4. This is a standalone analytical tool.
5. Compelling static reports are crucial since this will first feed a series of blog posts.
6. No performance constraints, no need to test performance yet.
7. The following order of importance:
- Basic financial model
- Basic ergodic growth rate calculations
- Multi-layer insurance optimization
- ROE optimization with ruin constraints
- Full Monte Carlo simulation engine
8. Definitely prefer Python

## Sprint 01: Foundation

### Sprint Planning Questions

1. Financial Model Specifics

- Should the widget manufacturer start with fixed initial parameters (e.g., exactly $10M assets), or should these
be configurable from the start?
- Do you want to model specific widget types/products, or keep it generic?
- Should we include inventory modeling in this sprint, or just financial aggregates?

2. Time Series Modeling

- What time resolution do you prefer: monthly, quarterly, or annual periods?
- How many years should our base simulation cover (the spec mentions 10 years for optimization, but for basic
modeling)?
- Should we implement stochastic revenue/cost variations in Sprint 1, or start with deterministic growth?

3. Balance Sheet Detail

- Do you need full balance sheet modeling (assets, liabilities, equity) or focus on key metrics (assets, revenue,
operating margin)?
- Should we track debt financing from the start, or add it later?
- How detailed should working capital be (separate AR, inventory, AP vs. aggregate)?

4. Growth Mechanics

- Should growth be purely organic (retained earnings) in Sprint 1, or include debt/equity financing options?
- Do you want capacity constraints based on assets from day one?
- Should we model depreciation and capital expenditure explicitly?

5. Development Approach

- Do you prefer Test-Driven Development (write tests first)?
- Should we create Jupyter notebooks alongside the Python modules for exploration?
- Would you like a simple CLI interface, or just library functions for now?

6. Configuration & Data

- YAML or JSON for configuration files?
- Should we create multiple preset scenarios in Sprint 1 (conservative, baseline, aggressive)?
- Do you want data persistence (save/load simulations) in this sprint?

7. Validation & Output

- What key metrics should we validate against (ROE, asset turnover, etc.)?
- Should Sprint 1 include basic visualizations, or just data structures?
- Any specific output formats needed (CSV, pickle, parquet)?

### Sprint Planning Responses

1. Create specification of financial parameters in `baseline.yaml`. The widgets should be generic products, a single type for now. Just model the financial aggregates.
2. I'd like time resolution to be monthly if possible. If you think this is too granular, we can model annually. I'm wondering if we need to model a long time, such as up to 1000 years. We should start with deterministic growth/cost at first.
3. I need to model the financial dynamics, but the model can be basic as outlined, without including complex accounting such as inventory or current/illiquid asset distinctions. Don't allow going into debt to leverage asset growth, but implement debt financing to support unpaid claims that will be. Don't need complicated working capital, assume there are no receivables or inventory issues.
4. No debt/equity financing beyond claim collateral. No depreciation.
5. I prefer a test-driven approach. We can create Jupyter notebooks for exploration. Library functions called from Jupyter notebooks for now.
6. We can use YAML for configuration. We can create three scenarios and model conservative/baseline/aggressive considerations from the start. We can have data persistence.
7. ROE, ROA, Risk of Ruin are my top metrics. Feel free to contemplate metrics that fill in gaps these 3 metrics leave undiagnosed. Parquet the output files and create Markdown output for summaries and diagnostics.

### Task Implementation

#### CC 1: Ask me clarifying questions then implement the story "Story 1: Project Infrastructure" from `\simone\SPRINT_01_FOUNDATION.md`. Follow the following parameters:
1. Store the project in `root/ergodic_insurance/`. Source code should be in `ergodic_insurance/src/`
2. Use the latest mutually compatible package versions using the `uv-mcp` mcp server. Include optional dependencies for notebooks in requirements.txt
3. Opinionated formatting is fine. Let's use pylint for linting. Gradual typing adoption is fine.
4. I like to use pytest with pytest-cov and pytest-xdist for parallel testing. tests/ at root directory is preferred. 90% test coverage is good.
5. Work with the existing repo structure for git, root is already a git project. I'll take care of .gitignore
6. Document commands in README and keep a minimal setup.py

##### Story Questions

1. Project location: You mentioned root/ergodic_insurance/ but the
sprint doc shows simone/src/ergodic_insurance/. Should I create the
structure at the root level as you specified?
2. Python version: What Python version should I target? This will
affect the package versions I select.
3. Dependencies: The sprint mentions several packages (pydantic for
config validation, pandas for data handling). Should I include data
science packages like numpy, matplotlib for the notebooks?
4. Test coverage: You want 90% coverage - should I set up coverage
reporting to enforce this threshold?

##### Responses

1. I fixed the sprint doc, it should be at the root level.
2. Use Python version 3.12.3
3. Yes include the standard data science packages and anything else that's needed.
4. Yes, set up coverage reporting to enforce this threshold. Create pytest-cov reports using HTML output.

#### Ask me clarifying questions then implement the story "Story 2: Configuration Management" from `\simone\SPRINT_01_FOUNDATION.md`

##### Story Questions

1. Configuration file location: Should the baseline.yaml be in
ergodic_insurance/data/parameters/ as specified in the sprint doc?
2. Pydantic version: Should I use Pydantic v2 (which we have installed)
with its newer syntax for models and validation?
3. Configuration override mechanism: Do you want:
 - CLI argument support for overrides?
 - Environment variable support?
 - Programmatic override via dict/kwargs?
 - Or all of the above?
4. Error handling: For invalid parameters, should the system:
 - Raise detailed validation errors immediately?
 - Provide default fallbacks for some parameters?
 - Log warnings for out-of-range but acceptable values?
5. Additional parameters: The sprint doc shows basic parameters. Should
I also include:
 - Random seed for reproducibility?
 - Output directory paths?
 - Logging configuration?
6. YAML structure: Should I support:
 - Multiple configuration files (e.g., baseline.yaml,
scenarios/high_growth.yaml)?
 - YAML anchors and references for DRY configs?
 - Comments in YAML explaining each parameter?

##### Responses

1. Yes, the file location should be per the sprint doc.
2. Yes use Pydantic v2
3. Support programmatic overrides via dict/kwargs
4. Raise detailed validation errors immediately for invalid parameters
5. Include a random seed for reproducibility. Take an output directory path. Also take logging configuration including path and skipping logging.
6. Provide comments in YAML explaining each parameter. Use anchors and reference for DRY configs. Provide 3 config files (conservative.yaml, baseline.yaml, optimistic.yaml)

#### Ask me clarifying questions then implement the story "Story 3: Core Financial Model" from `\simone\SPRINT_01_FOUNDATION.md`

##### Story Questions

1. Debt Interest Handling: Should interest on debt be calculated and
paid monthly or annually? The config supports both time resolutions.
2. Working Capital: Should working capital be deducted from available
assets for operations, or is it just tracked separately? The sprint doc
mentions it in the debt financing section.
3. Balance Sheet Structure: Should we track:
 - Cash separately from total assets?
 - Different asset categories (current vs. fixed)?
 - Or just total assets as a single value?
4. Growth Mechanism: For the deterministic growth model, should growth
apply to:
 - Revenue directly?
 - Assets (which then drives revenue)?
 - Both independently?
5. Debt Repayment: When debt is taken for claims, what's the repayment
schedule?
 - Pay off as much as possible from operating income each period?
 - Fixed amortization schedule?
 - Interest-only with balloon payment?
6. File Structure: Should the WidgetManufacturer class go in:
 - ergodic_insurance/src/manufacturer.py?
 - ergodic_insurance/src/models/manufacturer.py?
 - Another location?

##### Responses

1. There is no debt, only letters of credit purchased to support collateral. The collateral is explained later.
2. Working capital should be deducted from available assets for operations if it makes sense for modeling insurance and risk of ruin, and doesn't complicate the model.
3. Track assets as a single value, without duration or liquidity or asset type concerns for now.
4. Revenue should grow and net profit should contribute to the assets, but assets should not grow by themselves. Assume assets support the operations on the backend.
5. There is no debt, it's strictly collateral for outstanding claims that is supported by letters of credit for the insurer at a cost to the company. The claim liabilities are paid out over time with a configurable payment pattern over 10 years. Assume about 10% is paid in year 1, then 20% in year 2, 20% in year 3, and then decreasing payments of the remainder over the remaining 7 years. The collateral incurs collateral costs that are paid annually.
6. WidgetManufacturer class should go in `ergodic_insurance/src/models/manufacturer.py`

#### Ask me clarifying questions then implement the story "Story 4: Time Series Evolution" from `\simone\SPRINT_01_FOUNDATION.md`

##### Story Questions

##### Responses

#### Ask me clarifying questions then fix the story "Story 5: Debt Financing for Claims" from `\simone\SPRINT_01_FOUNDATION.md` to only concern the model with the letter of credit underlying the collateral to support large claims for as an annual interest charge.

##### Story Questions

1. Collateral Mechanism: Should Story 5 describe:
 - Letters of credit posted as collateral to guarantee future claim
payments?
 - The collateral amount equals the outstanding claim liability?
 - Or the collateral is some percentage of the claim liability?
2. Letter of Credit Costs:
 - The 1.5% annual rate applies to the total collateral amount?
 - These costs are treated as an operating expense (like interest)?
 - Are there any setup/initial fees for establishing the letter of
credit?
3. Collateral Release:
 - As claims are paid down over the 10-year schedule, is the
collateral reduced proportionally?
 - Or does the full collateral remain until the claim is fully paid?
4. Balance Sheet Treatment:
 - Should the collateral be shown as a restricted asset?
 - Or just tracked separately as an off-balance-sheet commitment with
only the annual cost hitting the P&L?
5. Triggering Conditions:
 - Letters of credit are required when claims exceed available liquid
assets?
 - Is there a minimum cash balance the company must maintain (like the
$100k mentioned)?
6. Title/Description: Should Story 5 be renamed to something like:
 - "Letter of Credit Collateral for Claims"
 - "Collateral Financing for Insurance Claims"
 - "Claim Collateral Management"
 - Or keep the current title but clarify it's about collateral, not
debt?

##### Responses

1. Letter of credit should be posted as collateral to guarantee future claim payments.
2. 1.5% annual rate should be paid monthly on total letters of credit securing the outstanding collateral.
3. Collateral gets reduced as claims are paid, it should basically be equal to the liability (unless we want to model out the net present value of letter of credit interest charges as a liability).
4. We can track collateral as a restricted asset and then have the interest on letter of credit as a liability.
5. Letters of credit are required on unpaid (large) claims that take time to pay out. This is demanded by the insurers regardless of assets. We can say there is no minimum assets, but the company becomes insolvent when its equity reaches 0 or becomes negative, at which point it is considered ruined.
6. You can call Story 5 "Story 5: Claim Collateral Management"

#### Ask me clarifying questions then implement the story "Story 6: Exploration Notebooks" from `\simone\SPRINT_01_FOUNDATION.md`

##### Story Questions

1. Visualization Preferences:
 - Should I use matplotlib/seaborn for static plots?
 - Or would you prefer interactive plots with plotly?
 - Any specific color schemes or styling preferences?
2. Notebook Structure:
 - Should each notebook be self-contained (import and initialize
everything)?
 - Or can they assume the user has run previous notebooks?
 - Should I include markdown cells explaining the financial concepts?
3. Long-term Simulation (Notebook 02):
 - For the 1000-year simulation, should I demonstrate:
     - Effects of different claim frequencies/sizes?
   - Comparison of different growth rates?
   - Impact of varying LoC rates?
 - Should I include memory/performance metrics?
4. Growth Dynamics (Notebook 03):
 - Should this focus on:
     - Revenue growth vs. asset growth relationships?
   - Impact of retention ratio on long-term wealth?
   - Effects of operating margins on sustainability?
   - Interaction between growth and claim events?
5. Sensitivity Analysis:
 - Which parameters are most important to analyze?
     - Operating margin sensitivity?
   - LoC rate sensitivity?
   - Claim size/frequency sensitivity?
   - Growth rate sensitivity?
 - Should I include tornado diagrams or similar visualizations?
6. Claim Scenarios:
 - Should the notebooks demonstrate:
     - Single large claim events?
   - Multiple smaller claims?
   - Periodic claim patterns?
   - Random claim generation for future sprints?
7. Output/Export:
 - Should notebooks save results to CSV/Parquet files?
 - Should they generate any summary reports?

##### Responses

1. Prefer matplotlib for simplicity. Use a Wall Street Journal professional style.
2. The notebooks can assume an order of execution. Include markdown cells explaining financial concepts.
3. Long-Term Simulation: Run 1000-year simulations 100000 times and track the overall performance of assumptions viz a viz the company's chance of ruin at 10-year, 20-year, 50-year, 100-year, 500-year, and 1000-year time horizons (ie, the number of simulated paths that resulted in ruin on or before each of those years), as well as the annual ROE distribution (ie, 100000 distributions of annualized ROEs over the 1000-year paths). I'm mainly interested in the effect of insurance limits and deductibles on risk of ruin and ROE. Include memory and performance metrics.
4. Growth Dynamics: You can perform a similar study as Long-Term Simulation but focusing on Return on Assets, asset fluctuations, and operating margin fluctuations.
5. I'm most interested in sensitivity to deductibles and insurance limits. Also claim frequency/severity sensitivity.
6. For claim scenarios, track annually the top 10 claims to date, top 10 open claims, total open claims, annual paid/incurred claims, and frequency/severity metrics. Create smart reports keeping in mind the volume of simulations, so we probably don't need full actuarial triangles at each year or it will get too unwieldy.
7. Store simulation results in parquet files. Generate summary reports as appropriate, perhaps in Markdown.

## Sprint 02: Ergodic Framework

Ask me clarifying questions then draft the sprint plan for "Sprint 02: Ergodic Framework" to be located in the file `\simone\SPRINT_02_ERGODIC_FRAMEWORK.md`.

### Sprint Planning Questions



### Sprint Planning Responses



## Sprint 03: Loss Modeling

Ask me clarifying questions then draft the sprint plan for "Sprint 03: Loss Modeling" to be located in the file `\simone\SPRINT_03_LOSS_MODELING.md`.

### Sprint Planning Questions



### Sprint Planning Responses



## Sprint 04: Insurance Optimization

### Sprint Planning Questions

### Sprint Planning Responses


## Sprint 05: Constrained Optimization

### Sprint Planning Questions

### Sprint Planning Responses


## Sprint 06: Monte Carlo Engine

### Sprint Planning Questions

### Sprint Planning Responses


## Sprint 07: Reporting & Visualization

### Sprint Planning Questions

### Sprint Planning Responses


## Sprint 08: Validation & Documentation

### Sprint Planning Questions

### Sprint Planning Responses

