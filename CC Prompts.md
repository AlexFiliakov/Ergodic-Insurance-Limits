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



##### Responses



#### Ask me clarifying questions then implement the story "Story 3: Core Financial Model" from `\simone\SPRINT_01_FOUNDATION.md`

##### Story Questions



##### Responses



## Sprint 02: Ergodic Framework

### Sprint Planning Questions

### Sprint Planning Responses


## Sprint 03: Loss Modeling

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

