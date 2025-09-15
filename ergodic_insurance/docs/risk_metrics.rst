Risk Metrics and Analysis
==========================

This guide covers the risk metrics used throughout the Ergodic Insurance framework.

Overview
--------

The risk metrics module provides comprehensive tools for measuring and analyzing risk in insurance and financial contexts.

Core Metrics
------------

Value at Risk (VaR)
~~~~~~~~~~~~~~~~~~~

Quantile-based risk measure for potential losses:

.. code-block:: python

   from ergodic_insurance.risk_metrics import RiskMetrics

   metrics = RiskMetrics()
   var_95 = metrics.calculate_var(returns, confidence=0.95)
   print(f"95% VaR: ${var_95:,.0f}")

Conditional Value at Risk (CVaR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Expected loss beyond VaR threshold:

.. code-block:: python

   cvar_95 = metrics.calculate_cvar(returns, confidence=0.95)
   print(f"95% CVaR: ${cvar_95:,.0f}")

Maximum Drawdown
~~~~~~~~~~~~~~~~

Largest peak-to-trough decline:

.. code-block:: python

   drawdown = metrics.calculate_max_drawdown(wealth_path)
   print(f"Maximum drawdown: {drawdown:.1%}")

Insurance-Specific Metrics
--------------------------

Loss Ratio
~~~~~~~~~~

Claims paid divided by premiums earned:

.. code-block:: python

   loss_ratio = metrics.calculate_loss_ratio(
       claims_paid=1_500_000,
       premiums_earned=2_000_000
   )
   print(f"Loss ratio: {loss_ratio:.1%}")

Combined Ratio
~~~~~~~~~~~~~~

Loss ratio plus expense ratio:

.. code-block:: python

   combined_ratio = metrics.calculate_combined_ratio(
       loss_ratio=0.75,
       expense_ratio=0.20
   )
   print(f"Combined ratio: {combined_ratio:.1%}")

Solvency Metrics
~~~~~~~~~~~~~~~~

Capital adequacy measures:

.. code-block:: python

   solvency = metrics.calculate_solvency_ratio(
       available_capital=10_000_000,
       required_capital=6_000_000
   )
   print(f"Solvency ratio: {solvency:.1%}")

Ergodic Risk Metrics
--------------------

Time-Average Volatility
~~~~~~~~~~~~~~~~~~~~~~~~

Volatility of growth rates over time:

.. code-block:: python

   time_vol = metrics.calculate_time_volatility(
       wealth_path,
       time_horizon=100
   )
   print(f"Time-average volatility: {time_vol:.2%}")

Bankruptcy Probability
~~~~~~~~~~~~~~~~~~~~~~

Probability of ruin over time horizon:

.. code-block:: python

   bankruptcy_prob = metrics.calculate_bankruptcy_probability(
       simulations,
       bankruptcy_threshold=0
   )
   print(f"Bankruptcy probability: {bankruptcy_prob:.1%}")

Growth-Risk Trade-off
~~~~~~~~~~~~~~~~~~~~~

Sharpe-like ratio for ergodic growth:

.. code-block:: python

   trade_off = metrics.calculate_growth_risk_ratio(
       time_avg_growth=0.05,
       growth_volatility=0.15
   )
   print(f"Growth-risk ratio: {trade_off:.2f}")

Advanced Analysis
-----------------

Tail Risk Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ergodic_insurance.risk_metrics import TailRiskAnalyzer

   analyzer = TailRiskAnalyzer()

   # Analyze tail behavior
   tail_stats = analyzer.analyze_tails(
       data=loss_data,
       threshold_percentile=95
   )

   print(f"Tail index: {tail_stats['tail_index']:.2f}")
   print(f"Expected shortfall: ${tail_stats['expected_shortfall']:,.0f}")

Stress Testing
~~~~~~~~~~~~~~

.. code-block:: python

   from ergodic_insurance.risk_metrics import StressTester

   tester = StressTester()

   # Define stress scenarios
   scenarios = [
       {"name": "Market crash", "equity_shock": -0.30, "claim_multiplier": 2.0},
       {"name": "Catastrophe", "claim_severity": 10_000_000, "frequency": 1},
       {"name": "Recession", "revenue_reduction": 0.25, "duration": 2}
   ]

   # Run stress tests
   results = tester.run_scenarios(
       manufacturer=manufacturer,
       scenarios=scenarios
   )

Risk Attribution
~~~~~~~~~~~~~~~~

.. code-block:: python

   from ergodic_insurance.risk_metrics import RiskAttribution

   attribution = RiskAttribution()

   # Decompose risk sources
   risk_sources = attribution.decompose_risk(
       portfolio=manufacturer.portfolio,
       factors=["operational", "financial", "insurance"]
   )

   for factor, contribution in risk_sources.items():
       print(f"{factor}: {contribution:.1%} of total risk")

Visualization
-------------

Risk Dashboard
~~~~~~~~~~~~~~

.. code-block:: python

   from ergodic_insurance.visualization import plot_risk_dashboard

   plot_risk_dashboard(
       metrics={
           "VaR": var_95,
           "CVaR": cvar_95,
           "Max Drawdown": max_drawdown,
           "Bankruptcy Prob": bankruptcy_prob
       },
       title="Risk Metrics Dashboard"
   )

Distribution Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ergodic_insurance.visualization import plot_loss_distribution

   plot_loss_distribution(
       losses=loss_data,
       var_line=var_95,
       cvar_region=cvar_95,
       title="Loss Distribution with Risk Metrics"
   )

Implementation Details
----------------------

Calculation Methods
~~~~~~~~~~~~~~~~~~~

1. **Historical**: Based on actual data
2. **Parametric**: Assumes distribution (e.g., normal)
3. **Monte Carlo**: Simulation-based
4. **Extreme Value Theory**: For tail modeling

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Use vectorized operations for large datasets
* Cache frequently computed metrics
* Consider approximations for real-time calculations

Best Practices
--------------

1. **Choose appropriate time horizons** for different metrics
2. **Consider correlation** between risk factors
3. **Use multiple metrics** for comprehensive view
4. **Validate models** with backtesting
5. **Document assumptions** clearly

Configuration
-------------

Risk metrics can be configured via YAML:

.. code-block:: yaml

   # data/config/modules/risk_metrics.yaml
   risk_metrics:
     var_confidence: 0.95
     cvar_confidence: 0.95
     time_horizon_years: 10
     bankruptcy_threshold: 0
     stress_test_percentile: 99

See Also
--------

* :doc:`api/risk_metrics` - Risk metrics API reference
* :doc:`theory` - Theoretical background
* :doc:`user_guide/decision_framework` - Using metrics for decisions
* :doc:`examples` - Practical examples
