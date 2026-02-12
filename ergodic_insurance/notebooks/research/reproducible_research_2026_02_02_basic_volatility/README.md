# Reproducible Research: Ergodic Insurance Under Volatility

Companion code for the paper **"Ergodicity Economics in Property & Casualty Insurance: A Simulation Framework for Understanding Risk Appetite"** (February 2, 2026).

The full paper is in [`output/publication/ergodicity_basic_simulation_anon_revision_clean.pdf`](output/publication/ergodicity_basic_simulation_anon_revision_clean.pdf).

## Why Rerun This?

Traditional insurance analysis says: minimize expected costs, self-insure when premiums exceed expected losses. This simulation shows that advice is **directionally wrong** for many companies.

The core insight comes from ergodicity economics. Corporate wealth compounds multiplicatively: a 50% loss followed by a 50% gain leaves you at 75%, not 100%. This "Volatility Tax" means that reducing loss variance through insurance can enhance long-term growth even when premiums substantially exceed expected losses. The simulation makes this concrete:

- **Without insurance**, 37.8% of simulated firms go insolvent within 50 years (median time-to-ruin: 20 years).
- **With guaranteed cost insurance** ($0 deductible), insolvency drops to 0.01%, and the insured firms grow *faster* despite paying premiums that exceed expected losses.
- **The optimal deductible ranking reverses** under time-average analysis: the strategy that minimizes expected costs (no insurance) produces the worst actual compound growth.

By rerunning, you can verify these results independently, explore how they change for different company profiles, and stress-test whether the conclusions hold under your own assumptions.

## Key Conclusions

![Ergodic Reversal](output/publication/interesting/volatility_tax_vs_premium_savings.png)

1. **Insurance value compounds over time.** Growth-rate advantages of 50-150 basis points per year translate into substantial terminal wealth differences over multi-decade horizons.
2. **The Volatility Tax overwhelms premium savings.** No Insurance has the highest expected-value growth (255 bps) but the worst actual compound growth (124 bps) due to a -132 bps volatility tax. Guaranteed cost insurance achieves 235 bps actual growth despite having the lowest expected-value growth (208 bps).
3. **Capitalization is the dominant driver.** A $5M company derives fundamentally different insurance value than a $25M company facing identical loss distributions. Peer benchmarking is misleading.
4. **Asset turnover amplifies insurance value.** Capital-efficient companies (higher ATR) generate more revenue per dollar of equity, creating more loss exposure and greater benefit from insurance.

## Parameters to Tweak

The simulation grid is defined in notebook `0.a.` These are the most interesting parameters to vary:

| Parameter | Default Values | What It Controls | Try This |
|-----------|---------------|------------------|----------|
| **Initial Capital** | \$5M, \$10M, \$25M | Company size relative to loss severity. Smaller companies see larger ergodic effects. | Try \$50M or more to find where insurance value vanishes |
| **Asset Turnover Ratio (ATR)** | 0.8, 1.0, 1.2 | Revenue = Assets x ATR. Higher ATR means more loss exposure per dollar of equity. | Try 0.5 (capital-intensive) or 2.0 (asset-light) |
| **Operational Volatility** | 0.15 | Year-to-year revenue volatility via Geometric Brownian Motion. Adds $\sigma^2/2$ drag to growth. | Try 0.0 (deterministic), 0.10, 0.25 to isolate the GBM effect from the loss volatility effect |
| Deductibles | \$0, \$100K, \$250K, \$500K | The insurance decision variable. (\$0 is guaranteed cost, no deductible is self-insured) | Add \$50K, \$750K, or \$1M to find the crossover |
| Loss Ratio | 0.7 | Insurer pricing: premium = expected insured loss / loss ratio. Lower = more expensive insurance. | Try 0.5 (hard market) or 0.9 (soft market) to test price sensitivity. Insurance may provide value even at 0.4. |
| EBITABL | 0.125 | Operating margin before claims and insurance. Higher margins buffer losses better. | Try 0.05 (thin margin) or 0.20 (high margin). |

The simulation runs 250,000 Monte Carlo paths over 50 years for each parameter combination, with Common Random Numbers (CRN) ensuring insured and uninsured scenarios face identical underlying loss events for clean paired comparisons.

## Environment Setup

The simulation is set up to run on **Google Colab** (but can run in any environment with multiple CPU cores, you just have to point to the right directories). The full grid (~36 parameter combinations) costs roughly **$25 in Colab compute** and takes a couple of days. A reduced test grid can run in minutes.

The `ergodic_insurance` package is installed directly from GitHub at the start of notebook `0.a.`:

```
pip install git+https://github.com/AlexFiliakov/Ergodic-Insurance-Limits
```

## Files and Workflow

Run the numbered notebooks in order. Steps 0.a and 0.b run on Google Colab; the rest run locally.

### Step 0: Run Simulations (Google Colab)

| File | Description |
|------|-------------|
| **`run_vol_sim_colab.py`** | Core simulation function. Defines the corporate profile, loss distributions, insurance program, and Monte Carlo engine. Upload this to Google Drive before running `0.a.` |
| **`0.a. ergodicity_vol_sim_parallel.ipynb`** | Main simulation runner. Installs the package, sweeps the parameter grid in parallel using `joblib`, and saves results as pickle files to Google Drive. This is the compute-intensive step. |
| **`0.b. copy_files_from_g_drive.ipynb`** | Utility to copy result pickle files from Google Drive to a local/USB drive, with retry logic and filename sanitization for Windows. |

### Step 1: Process Results (Local)

| File | Description |
|------|-------------|
| **`1. process_vol_sim_results.ipynb`** | Loads all pickle files, parses filenames to extract parameters, computes growth rate statistics (mean, std, quantiles, kurtosis), ruin probabilities, and builds two caches: `cache/all_df.parquet` (summary DataFrame) and `cache/dashboard_cache.pkl` (time-series percentiles, interesting simulation IDs, and CRN pairing maps for the interactive dashboards). |

### Step 2: Explore (Local, Interactive)

| File | Description |
|------|-------------|
| **`2.a. eda_space_explorer.ipynb`** | Interactive Plotly dashboard. Select Cap, ATR, and Deductible to view wealth fan charts, growth rate distributions, ruin probability curves, and retained loss envelopes -- all comparing insured vs. uninsured side by side. |
| **`2.b. eda_paired_path_browser.ipynb`** | Drill into individual CRN-paired simulation paths. Filter by path type (catastrophic, near-ruin, median, best, worst surviving, ruined) to see exactly how insurance changes the trajectory of a single firm facing the same loss events. |

### Step 3: Generate Publication Figures (Local)

| File | Description |
|------|-------------|
| **`3. generate_publication_viz.ipynb`** | Generates all publication-ready figures (300 DPI, Okabe-Ito colorblind-safe palette, serif fonts). Includes: optimal deductible heatmap, wealth fan charts, survival curves, year-by-year growth lift, volatility tax decomposition, peer benchmark comparison, outcome distributions, and more. Outputs to `output/publication/`. |

### Supporting Files

| File | Description |
|------|-------------|
| **`VISUAL_SPECIFICATION.md`** | Consistent visual standards for all figures: color palette, typography, line styles, figure sizing, and chart-specific guidelines. |
| **`cache/`** | Intermediate cached data. Generated by notebook `1.`; consumed by notebooks `2.a.`, `2.b.`, and `3.` |
| **`output/publication/`** | Final publication figures (PNG, 300 DPI). |

## Reproducing from Scratch

1. Upload `run_vol_sim_colab.py` to your Google Drive (`Colab Notebooks/` folder). To run 250,000 scenarios, you need the Pro account or your run will time out after ~12 hours. With Pro, the timeout is extended to 24 hours. The default notebook is set up to run only 1,000 simulations as a test, which will run in a few minutes.
2. Open `0.a. ergodicity_vol_sim_parallel.ipynb` in Colab and run all cells. Results save to Google Drive.
3. Download results to a local directory (or use `0.b.` to copy to USB).
4. Update the `results_dir` path in notebook `1.` to point to your downloaded results.
5. Run notebooks `1.` through `3.` in order.
