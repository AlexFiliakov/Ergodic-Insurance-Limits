I'm writing a research paper based on insurance simulation results, and I need to analyze the results and draw conclusions, even though the main goal of the paper is to motivate actuaries to try my framework, whose codebase is in the main directory `ergodic_insurance\`.

The research paper is based on the following simulation:
- `ergodic_insurance\notebooks\results_vol_sim\run_vol_sim_colab.py` configures each simulation
- `ergodic_insurance\notebooks\results_vol_sim\ergodicity_vol_sim_parallel.ipynb` orchestrates the parallel runs

Simulation results are pickled `SimulationResults` objects, so you can look up their interface. The top-level fields consist of:
- 'aggregated_results',
- 'annual_losses',
- 'bootstrap_confidence_intervals',
- 'config',
- 'convergence',
- 'execution_time',
- 'final_assets',
- 'growth_rates',
- 'insurance_recoveries',
- 'metrics',
- 'performance_metrics',
- 'retained_losses',
- 'ruin_probability',
- 'statistical_summary',
- 'summary',
- 'summary_report',
- 'time_series_aggregation'

## Exploratory Data Analysis (EDA) Dashboards needed

I want to start by building out the following Interactive Exploration Tools (Pre-Publication)

### Parameter Space Explorer Dashboard

**What it builds:** A Plotly or Panel dashboard with dropdown selectors for Cap, ATR, and Deductible. Selecting a configuration shows:
- Wealth fan chart
- Growth rate distribution
- Ruin probability curve
- Key metrics (mean growth rate, median terminal wealth, ruin probability at years 10/25/50)

**Why it's useful:** With 36 configurations, I need a fast way to scan for the most interesting cases. The dashboard lets me flip through configurations and identify which ones to feature in the paper.

### Paired Path Browser

**What it builds:** A tool that lets me select a sim_id and see the insured vs. uninsured wealth trajectories side by side for a specific configuration. Filter by: paths with catastrophic events, paths where ruin occurred, paths near the median, etc.

**Why it's useful:** For selecting the specific paths to feature in analyses A3 and C2. The CRN pairing makes individual paths meaningful, but I need a way to find the compelling ones.

### Future Presentation Analytics

The full suite of analytics for consideration is described in `ergodic_insurance\notebooks\results_vol_sim\ANALYSIS_BRAINSTORMING.md`

## My request

I'm asking you to review the parser in `ergodic_insurance\notebooks\results_vol_sim\1. process_vol_sim_results.ipynb` and see if it needs to bring in additional data into the Parquet file or `parsed_params_by_key` cached files to power the two exploratory dashboards above, as well as to support the future presentation analytics in `ergodic_insurance\notebooks\results_vol_sim\ANALYSIS_BRAINSTORMING.md`. Make appropriate edits to the parser. Then, create two notebooks for the EDA dashboards above:
- `ergodic_insurance\notebooks\results_vol_sim\2.a. eda_space_explorer.ipynb` that implements the Parameter Space Explorer Dashboard
- `ergodic_insurance\notebooks\results_vol_sim\2.b. eda_paired_path_browser.ipynb` that implements the Paired Path Browser

Please ask me clarifying questions and implement these enhancements.
