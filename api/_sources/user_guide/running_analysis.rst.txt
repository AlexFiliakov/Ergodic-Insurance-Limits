Running Your Analysis
=====================

This section provides a comprehensive walkthrough of performing insurance optimization analysis for your company. We'll cover setting up scenarios, running simulations, and interpreting results.

Part 1: Setting Up Your Analysis Environment
---------------------------------------------

Preparing Your Workspace
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Create a project directory** for your analysis:

.. code-block:: bash

   mkdir my_insurance_analysis
   cd my_insurance_analysis

2. **Copy configuration templates**:

.. code-block:: bash

   # Copy example configurations as starting points
   cp ergodic_insurance/data/parameters/baseline.yaml ./my_baseline.yaml
   cp ergodic_insurance/data/parameters/insurance_structures.yaml ./my_insurance.yaml

3. **Set up output directories**:

.. code-block:: bash

   mkdir results
   mkdir figures
   mkdir reports

Part 2: Defining Your Business Model
-------------------------------------

Step 2.1: Financial Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit your configuration file with company-specific parameters:

.. code-block:: python
   :caption: Setting up your manufacturer model

   from ergodic_insurance.src.manufacturer import WidgetManufacturer
   from ergodic_insurance.src.config import ManufacturerConfig

   # Option 1: Direct instantiation
   manufacturer = WidgetManufacturer(
       starting_assets=10_000_000,
       base_revenue=15_000_000,
       operating_margin=0.08,
       tax_rate=0.25,
       working_capital_pct=0.20,
       dividend_rate=0.30,  # 30% of profits as dividends
       capex_rate=0.05,      # 5% of revenue for capital expenditure
       debt_capacity=0.5     # Can borrow up to 50% of assets
   )

   # Option 2: Using configuration
   from ergodic_insurance.src.config_loader import load_config

   config = load_config('my_baseline.yaml')
   manufacturer = WidgetManufacturer.from_config(config.manufacturer)

Step 2.2: Modeling Revenue Dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose between deterministic and stochastic revenue models:

.. code-block:: python
   :caption: Revenue modeling options

   from ergodic_insurance.src.stochastic_processes import (
       GeometricBrownianMotion,
       LognormalShock,
       MeanRevertingProcess
   )

   # Stable business - deterministic growth
   manufacturer.revenue_shock = None

   # Volatile business - GBM process
   manufacturer.revenue_shock = GeometricBrownianMotion(
       drift=0.06,      # 6% expected growth
       volatility=0.15  # 15% annual volatility
   )

   # Cyclical business - mean reverting
   manufacturer.revenue_shock = MeanRevertingProcess(
       mean_level=1.0,
       reversion_speed=0.3,
       volatility=0.2
   )

Part 3: Configuring Loss Distributions
---------------------------------------

Step 3.1: Using Historical Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have loss history, calibrate distributions:

.. code-block:: python
   :caption: Calibrating from historical losses

   from ergodic_insurance.src.loss_distributions import (
       AttritionalLosses,
       LargeLosses,
       CatastrophicLosses
   )
   import pandas as pd

   # Load your historical data
   loss_history = pd.read_csv('historical_losses.csv')

   # Separate by magnitude
   small_losses = loss_history[loss_history['amount'] < 100_000]
   large_losses = loss_history[
       (loss_history['amount'] >= 100_000) &
       (loss_history['amount'] < 5_000_000)
   ]
   cat_losses = loss_history[loss_history['amount'] >= 5_000_000]

   # Calibrate distributions
   attritional = AttritionalLosses()
   attritional.calibrate(
       frequency=len(small_losses) / years_of_data,
       severity_mean=small_losses['amount'].mean(),
       severity_std=small_losses['amount'].std()
   )

Step 3.2: Industry Benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use industry-standard parameters if historical data is limited:

.. code-block:: python
   :caption: Industry-specific loss configurations

   # Manufacturing industry
   manufacturing_losses = {
       'attritional': {
           'frequency': 4.5,
           'severity_mean': 75_000,
           'severity_cv': 0.8
       },
       'large': {
           'frequency': 0.25,
           'severity_mean': 2_500_000,
           'severity_cv': 1.2
       },
       'catastrophic': {
           'frequency': 0.015,
           'severity_mean': 20_000_000,
           'severity_cv': 0.6
       }
   }

   # Technology industry
   tech_losses = {
       'cyber': {
           'frequency': 0.8,
           'severity_mean': 3_000_000,
           'severity_cv': 1.5
       },
       'business_interruption': {
           'frequency': 0.3,
           'severity_mean': 5_000_000,
           'severity_cv': 1.0
       }
   }

Part 4: Running Simulations
----------------------------

Step 4.1: Baseline Scenario (No Insurance)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, establish your baseline risk:

.. code-block:: python
   :caption: Baseline simulation without insurance

   from ergodic_insurance.src.monte_carlo import MonteCarloEngine
   from ergodic_insurance.src.claim_generator import ClaimGenerator

   # Set up claim generator
   claim_gen = ClaimGenerator(
       frequency=5.0,  # Expected claims per year
       severity_mean=500_000,
       severity_cv=1.2
   )

   # Run baseline simulation
   engine = MonteCarloEngine(
       n_simulations=10_000,
       random_seed=42  # For reproducibility
   )

   baseline_results = engine.run(
       manufacturer=manufacturer,
       claim_generator=claim_gen,
       insurance_program=None,  # No insurance
       n_years=10
   )

   print(f"Baseline 10-year survival: {baseline_results.survival_rate:.1%}")
   print(f"Baseline growth rate: {baseline_results.mean_growth_rate:.2%}")

Step 4.2: Testing Insurance Structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate different insurance configurations:

.. code-block:: python
   :caption: Comparing insurance structures

   from ergodic_insurance.src.insurance_program import InsuranceProgram
   import numpy as np

   # Test different retention levels
   retentions = [50_000, 100_000, 250_000, 500_000, 1_000_000]
   results = {}

   for retention in retentions:
       # Create insurance program
       insurance = InsuranceProgram(
           retention=retention,
           layers=[
               {
                   'name': 'Primary',
                   'limit': 5_000_000,
                   'attachment': retention,
                   'premium_rate': 0.015
               },
               {
                   'name': 'Excess',
                   'limit': 20_000_000,
                   'attachment': retention + 5_000_000,
                   'premium_rate': 0.008
               }
           ]
       )

       # Run simulation
       sim_results = engine.run(
           manufacturer=manufacturer,
           claim_generator=claim_gen,
           insurance_program=insurance,
           n_years=10
       )

       results[retention] = {
           'survival_rate': sim_results.survival_rate,
           'mean_growth': sim_results.mean_growth_rate,
           'total_premium': insurance.annual_premium * 10,
           'ergodic_value': sim_results.ergodic_wealth_multiple
       }

   # Find optimal retention
   optimal = max(results.items(),
                key=lambda x: x[1]['ergodic_value'])
   print(f"Optimal retention: ${optimal[0]:,}")

Step 4.3: Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test sensitivity to key assumptions:

.. code-block:: python
   :caption: Sensitivity analysis

   from ergodic_insurance.src.visualization import create_sensitivity_plot

   # Parameters to test
   sensitivity_params = {
       'loss_frequency': np.linspace(3, 7, 5),
       'loss_severity': np.linspace(0.5, 1.5, 5) * 500_000,
       'revenue_volatility': np.linspace(0.10, 0.30, 5),
       'premium_loading': np.linspace(1.0, 2.0, 5)
   }

   sensitivity_results = {}

   for param_name, param_values in sensitivity_params.items():
       param_results = []

       for value in param_values:
           # Modify parameter
           if param_name == 'loss_frequency':
               claim_gen.frequency = value
           elif param_name == 'loss_severity':
               claim_gen.severity_mean = value
           # ... etc

           # Run simulation
           result = engine.run(
               manufacturer=manufacturer,
               claim_generator=claim_gen,
               insurance_program=optimal_insurance,
               n_years=10
           )

           param_results.append({
               'value': value,
               'survival_rate': result.survival_rate,
               'growth_rate': result.mean_growth_rate
           })

       sensitivity_results[param_name] = param_results

   # Create visualization
   create_sensitivity_plot(sensitivity_results,
                          output_path='figures/sensitivity.png')

Part 5: Analyzing Results
-------------------------

Step 5.1: Ergodic Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare time-average vs ensemble-average performance:

.. code-block:: python
   :caption: Ergodic analysis

   from ergodic_insurance.src.ergodic_analyzer import ErgodicAnalyzer

   analyzer = ErgodicAnalyzer()

   # Calculate ergodic metrics
   ergodic_metrics = analyzer.analyze(
       simulation_results=results,
       time_horizon=10
   )

   print("Ergodic Analysis Results:")
   print(f"Time-Average Growth: {ergodic_metrics['time_avg_growth']:.2%}")
   print(f"Ensemble-Average Growth: {ergodic_metrics['ensemble_avg_growth']:.2%}")
   print(f"Ergodic Gap: {ergodic_metrics['ergodic_gap']:.2%}")
   print(f"Kelly Criterion Insurance: ${ergodic_metrics['kelly_premium']:,}")

Step 5.2: Risk Metrics
~~~~~~~~~~~~~~~~~~~~~~

Calculate comprehensive risk measures:

.. code-block:: python
   :caption: Risk metric calculation

   from ergodic_insurance.src.risk_metrics import RiskMetrics

   risk_calc = RiskMetrics()

   # Calculate various risk measures
   metrics = risk_calc.calculate_all(results.wealth_paths)

   print("\nRisk Metrics:")
   print(f"Value at Risk (95%): ${metrics['var_95']:,.0f}")
   print(f"Conditional VaR (95%): ${metrics['cvar_95']:,.0f}")
   print(f"Maximum Drawdown: {metrics['max_drawdown']:.1%}")
   print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
   print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
   print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")

Step 5.3: Visualization
~~~~~~~~~~~~~~~~~~~~~~~

Create comprehensive visualizations:

.. code-block:: python
   :caption: Creating analysis visualizations

   from ergodic_insurance.src.visualization import (
       plot_wealth_paths,
       plot_survival_curves,
       plot_growth_distribution,
       create_dashboard
   )
   import matplotlib.pyplot as plt

   # Create figure with subplots
   fig, axes = plt.subplots(2, 2, figsize=(15, 12))

   # Wealth paths
   plot_wealth_paths(
       results.wealth_paths,
       ax=axes[0, 0],
       title="Wealth Evolution",
       highlight_percentiles=[5, 50, 95]
   )

   # Survival probability
   plot_survival_curves(
       [baseline_results, insured_results],
       labels=['No Insurance', 'With Insurance'],
       ax=axes[0, 1]
   )

   # Growth rate distribution
   plot_growth_distribution(
       results.growth_rates,
       ax=axes[1, 0],
       show_ergodic=True
   )

   # Insurance efficiency
   axes[1, 1].plot(retentions,
                  [r['ergodic_value'] for r in results.values()])
   axes[1, 1].set_xlabel('Retention Level ($)')
   axes[1, 1].set_ylabel('Ergodic Wealth Multiple')
   axes[1, 1].set_title('Insurance Efficiency Curve')

   plt.tight_layout()
   plt.savefig('figures/analysis_dashboard.png', dpi=300)
   plt.show()

Part 6: Using Analysis Notebooks
---------------------------------

Leverage Pre-Built Notebooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our notebook collection provides ready-to-use analyses:

**Optimization Analysis** (``notebooks/09_optimization_results.ipynb``)
   * Comprehensive optimization across retention and limit combinations
   * 3D surface plots of ergodic value
   * Identifies global optimum

**Sensitivity Analysis** (``notebooks/10_sensitivity_analysis.ipynb``)
   * Tests sensitivity to all major parameters
   * Tornado diagrams for parameter importance
   * Scenario stress testing

**Monte Carlo Deep Dive** (``notebooks/08_monte_carlo_analysis.ipynb``)
   * Convergence analysis
   * Confidence intervals
   * Simulation efficiency

To use these notebooks with your data:

.. code-block:: python
   :caption: Customizing notebooks for your analysis

   # In the notebook, replace the default parameters:

   # Cell 1: Load your configuration
   config_path = '../my_insurance_analysis/my_baseline.yaml'
   config = load_config(config_path)

   # Cell 2: Use your manufacturer
   manufacturer = WidgetManufacturer.from_config(config.manufacturer)

   # Cell 3: Run with your parameters
   # The notebook will handle the rest!

Part 7: Generating Reports
--------------------------

Automated Report Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create professional reports for stakeholders:

.. code-block:: python
   :caption: Generating analysis report

   from ergodic_insurance.src.reporting import ReportGenerator

   # Initialize report generator
   report = ReportGenerator(
       company_name="My Manufacturing Co",
       analysis_date="2025-01-15",
       analyst="Risk Management Team"
   )

   # Add sections
   report.add_executive_summary(
       baseline_results=baseline_results,
       optimal_results=optimal_results,
       recommendations=recommendations
   )

   report.add_risk_analysis(
       risk_metrics=metrics,
       survival_analysis=survival_data,
       sensitivity_results=sensitivity_results
   )

   report.add_recommendation(
       optimal_retention=250_000,
       optimal_limit=25_000_000,
       expected_benefit=3_500_000,
       confidence_level=0.95
   )

   # Generate PDF report
   report.save_pdf('reports/insurance_analysis_2025.pdf')

   # Generate Excel workbook with detailed data
   report.save_excel('reports/analysis_data.xlsx')

Best Practices
--------------

Simulation Guidelines
~~~~~~~~~~~~~~~~~~~~~

1. **Number of Simulations**:

   * Quick exploration: 1,000 simulations
   * Detailed analysis: 10,000 simulations
   * Final recommendations: 100,000 simulations

2. **Time Horizons**:

   * Short-term (cash flow): 1-3 years
   * Medium-term (strategic): 5-10 years
   * Long-term (ergodic): 20-50 years

3. **Convergence Checking**:

.. code-block:: python

   # Check if results have converged
   from ergodic_insurance.src.convergence import check_convergence

   converged = check_convergence(
       results.growth_rates,
       tolerance=0.001,  # 0.1% tolerance
       window=1000       # Check last 1000 simulations
   )

   if not converged:
       print("Warning: Results may not have converged")
       print("Consider increasing simulation count")

Common Pitfalls to Avoid
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Ignoring Correlation**: Losses often correlate with revenue downturns
2. **Static Analysis**: Business parameters change over time
3. **Point Estimates**: Always consider confidence intervals
4. **Over-Optimization**: Leave margin for model uncertainty
5. **Ignoring Liquidity**: Survival requires cash, not just solvency

Next Steps
----------

After completing your analysis:

1. Review results with the :doc:`decision_framework`
2. Compare with :doc:`case_studies` from similar companies
3. Explore :doc:`advanced_topics` for customization
4. Document assumptions and decisions
5. Schedule periodic reviews (quarterly/annually)

Remember: The optimal insurance structure depends on your specific circumstances. Use these tools to inform, not replace, professional judgment.
