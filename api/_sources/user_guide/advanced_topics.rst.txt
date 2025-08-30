Advanced Topics
===============

This section covers sophisticated techniques for users who want to customize their analysis beyond the standard framework. Topics include custom loss distributions, correlation modeling, multi-year optimization, and advanced stochastic processes.

Customizing Loss Distributions
-------------------------------

Industry-Specific Loss Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different industries have unique loss patterns. Here's how to customize the distributions for your sector:

.. code-block:: python
   :caption: Custom loss distribution for cyber risks

   from ergodic_insurance.src.loss_distributions import CustomLossDistribution
   import numpy as np
   from scipy import stats

   class CyberLossDistribution(CustomLossDistribution):
       """Model cyber losses with increasing frequency over time."""

       def __init__(self, base_frequency=0.5, growth_rate=0.15):
           self.base_frequency = base_frequency
           self.growth_rate = growth_rate  # Cyber risk growing 15% annually

       def generate_losses(self, year, random_state=None):
           # Frequency increases over time
           current_frequency = self.base_frequency * (1 + self.growth_rate) ** year

           # Number of events (Poisson process)
           n_events = random_state.poisson(current_frequency)

           # Severity follows power law (heavy tail)
           if n_events > 0:
               # Power law: many small, few catastrophic
               alpha = 2.5  # Shape parameter
               x_min = 100_000  # Minimum loss

               # Generate from power law
               u = random_state.uniform(0, 1, n_events)
               losses = x_min * (1 - u) ** (-1 / (alpha - 1))

               return losses
           return np.array([])

Fitting Distributions to Historical Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use your actual loss history to calibrate distributions:

.. code-block:: python
   :caption: Calibrating distributions from data

   from ergodic_insurance.src.loss_distributions import fit_distribution
   import pandas as pd
   from scipy import stats

   # Load your historical losses
   loss_data = pd.read_csv('historical_losses.csv')

   # Separate by magnitude
   small_losses = loss_data[loss_data['amount'] < 100_000]['amount']
   large_losses = loss_data[loss_data['amount'] >= 100_000]['amount']

   # Fit frequency distribution
   years_of_data = loss_data['year'].nunique()
   small_frequency = len(small_losses) / years_of_data
   large_frequency = len(large_losses) / years_of_data

   # Fit severity distributions
   # Test multiple distributions
   distributions = [
       stats.lognorm,
       stats.gamma,
       stats.weibull_min,
       stats.pareto
   ]

   best_fit = None
   best_ks_stat = float('inf')

   for dist in distributions:
       # Fit distribution
       params = dist.fit(large_losses)

       # Kolmogorov-Smirnov test
       ks_stat, p_value = stats.kstest(large_losses,
                                       lambda x: dist.cdf(x, *params))

       if ks_stat < best_ks_stat and p_value > 0.05:
           best_fit = (dist, params)
           best_ks_stat = ks_stat

   print(f"Best fitting distribution: {best_fit[0].name}")
   print(f"Parameters: {best_fit[1]}")

Correlation Modeling
--------------------

Modeling Business Cycle Correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Losses often correlate with economic conditions:

.. code-block:: python
   :caption: Implementing correlation between revenue and losses

   from ergodic_insurance.src.stochastic_processes import CorrelatedShocks

   class CorrelatedRiskModel:
       """Model correlation between business performance and losses."""

       def __init__(self, correlation_matrix):
           """
           correlation_matrix: 2x2 matrix
           [[1.0, rho],
            [rho, 1.0]]
           where rho is correlation between revenue shock and loss shock
           """
           self.correlation_matrix = correlation_matrix

       def generate_shocks(self, n_periods, random_state=None):
           """Generate correlated shocks for revenue and losses."""

           # Generate independent standard normal variables
           if random_state is None:
               random_state = np.random.RandomState()

           independent_shocks = random_state.randn(n_periods, 2)

           # Apply Cholesky decomposition for correlation
           L = np.linalg.cholesky(self.correlation_matrix)
           correlated_shocks = independent_shocks @ L.T

           return {
               'revenue_shocks': correlated_shocks[:, 0],
               'loss_shocks': correlated_shocks[:, 1]
           }

   # Example: 30% correlation between bad revenue and high losses
   correlation_model = CorrelatedRiskModel(
       correlation_matrix=[[1.0, 0.3],
                          [0.3, 1.0]]
   )

   # Use in simulation
   shocks = correlation_model.generate_shocks(n_periods=10)

   for year in range(10):
       revenue_multiplier = 1 + 0.15 * shocks['revenue_shocks'][year]
       loss_multiplier = np.exp(0.3 * shocks['loss_shocks'][year])

       # Bad revenue years have higher losses
       annual_revenue = base_revenue * revenue_multiplier
       annual_losses = base_losses * loss_multiplier

Geographic and Peril Correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Model correlation across locations and perils:

.. code-block:: python
   :caption: Multi-location correlation modeling

   class MultiLocationRisk:
       """Model correlated risks across multiple locations."""

       def __init__(self, locations, correlation_by_distance):
           self.locations = locations
           self.correlation_func = correlation_by_distance

       def build_correlation_matrix(self):
           """Build correlation matrix based on geographic distance."""
           n = len(self.locations)
           corr_matrix = np.eye(n)

           for i in range(n):
               for j in range(i+1, n):
                   # Calculate distance between locations
                   distance = self.calculate_distance(
                       self.locations[i],
                       self.locations[j]
                   )

                   # Correlation decreases with distance
                   correlation = self.correlation_func(distance)
                   corr_matrix[i, j] = correlation
                   corr_matrix[j, i] = correlation

           return corr_matrix

       def calculate_distance(self, loc1, loc2):
           """Calculate distance between two locations."""
           # Simplified Euclidean distance
           return np.sqrt((loc1['lat'] - loc2['lat'])**2 +
                         (loc1['lon'] - loc2['lon'])**2)

   # Example: Correlation decreases exponentially with distance
   locations = [
       {'name': 'Factory A', 'lat': 40.7, 'lon': -74.0},
       {'name': 'Factory B', 'lat': 41.8, 'lon': -87.6},
       {'name': 'Factory C', 'lat': 34.0, 'lon': -118.2}
   ]

   correlation_func = lambda d: np.exp(-d / 500)  # 500 mile correlation length

   multi_location = MultiLocationRisk(locations, correlation_func)
   corr_matrix = multi_location.build_correlation_matrix()

Multi-Year Optimization
-----------------------

Dynamic Insurance Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optimize insurance purchases over multiple years:

.. code-block:: python
   :caption: Multi-year insurance optimization

   class DynamicInsuranceOptimizer:
       """Optimize insurance strategy over multiple years."""

       def __init__(self, planning_horizon=5):
           self.planning_horizon = planning_horizon

       def optimize_multi_year(self, manufacturer, market_conditions):
           """Find optimal insurance path over planning horizon."""

           # State space: (assets, market_cycle, years_remaining)
           states = self.discretize_state_space(manufacturer)

           # Value function: V(state, action) = expected future value
           value_function = {}
           optimal_policy = {}

           # Backward induction (dynamic programming)
           for year in range(self.planning_horizon, 0, -1):
               for state in states:
                   best_value = -np.inf
                   best_action = None

                   # Try different insurance levels
                   for retention in [100_000, 250_000, 500_000, 1_000_000]:
                       for limit in [5_000_000, 15_000_000, 25_000_000]:

                           # Calculate expected value of this action
                           action = (retention, limit)
                           expected_value = self.calculate_expected_value(
                               state, action, year, value_function
                           )

                           if expected_value > best_value:
                               best_value = expected_value
                               best_action = action

                   value_function[(state, year)] = best_value
                   optimal_policy[(state, year)] = best_action

           return optimal_policy

       def calculate_expected_value(self, state, action, year, value_function):
           """Calculate expected value of action in given state."""

           retention, limit = action
           assets, market_state = state

           # Simulate outcomes
           outcomes = []

           for scenario in range(100):  # Sample scenarios
               # Apply insurance for this year
               premium = self.calculate_premium(retention, limit, market_state)

               # Generate losses
               losses = self.generate_losses(market_state)

               # Apply insurance
               retained_loss = min(losses, retention)
               insurance_recovery = min(max(losses - retention, 0), limit)

               # Update assets
               new_assets = assets * (1 + growth_rate) - premium - retained_loss

               # Transition to new market state
               new_market_state = self.market_transition(market_state)

               # Future value (if not final year)
               if year > 1:
                   future_value = value_function.get(
                       ((new_assets, new_market_state), year - 1), 0
                   )
               else:
                   future_value = new_assets

               outcomes.append(future_value)

           return np.mean(outcomes)

Market Cycle Timing
~~~~~~~~~~~~~~~~~~~

Adjust insurance purchases based on market cycles:

.. code-block:: python
   :caption: Market-aware insurance purchasing

   class MarketCycleStrategy:
       """Adjust insurance based on market conditions."""

       def __init__(self):
           self.market_states = ['hard', 'transitioning', 'soft']
           self.transition_matrix = [
               [0.5, 0.4, 0.1],  # From hard market
               [0.3, 0.4, 0.3],  # From transitioning
               [0.1, 0.4, 0.5]   # From soft market
           ]

       def recommend_strategy(self, current_market, company_state):
           """Recommend insurance strategy based on market."""

           if current_market == 'hard':
               # Hard market: prices high, capacity limited
               return {
                   'strategy': 'defensive',
                   'retention': 'increase by 25%',
                   'limit': 'maintain current',
                   'timing': 'lock in multi-year if possible',
                   'alternative': 'consider captive or self-insurance'
               }

           elif current_market == 'soft':
               # Soft market: prices low, capacity abundant
               return {
                   'strategy': 'aggressive',
                   'retention': 'decrease to optimal',
                   'limit': 'increase by 50%',
                   'timing': 'lock in long-term',
                   'alternative': 'buy additional coverage'
               }

           else:  # transitioning
               return {
                   'strategy': 'balanced',
                   'retention': 'maintain',
                   'limit': 'modest increase',
                   'timing': 'short-term contracts',
                   'alternative': 'prepare for market shift'
               }

Advanced Stochastic Processes
------------------------------

Jump Diffusion Models
~~~~~~~~~~~~~~~~~~~~~~

Model sudden shocks alongside continuous volatility:

.. code-block:: python
   :caption: Jump diffusion for catastrophic events

   class JumpDiffusionProcess:
       """Combine continuous volatility with discrete jumps."""

       def __init__(self, drift=0.06, volatility=0.15,
                   jump_intensity=0.1, jump_mean=-0.3, jump_std=0.2):
           self.drift = drift
           self.volatility = volatility
           self.jump_intensity = jump_intensity  # Jumps per year
           self.jump_mean = jump_mean  # Average jump size (negative = down)
           self.jump_std = jump_std

       def simulate(self, S0, T, dt, random_state=None):
           """Simulate jump diffusion process."""

           if random_state is None:
               random_state = np.random.RandomState()

           n_steps = int(T / dt)
           times = np.linspace(0, T, n_steps + 1)
           S = np.zeros(n_steps + 1)
           S[0] = S0

           for i in range(1, n_steps + 1):
               # Brownian motion component
               dW = random_state.randn() * np.sqrt(dt)

               # Jump component
               dN = random_state.poisson(self.jump_intensity * dt)

               if dN > 0:
                   # Jump occurred
                   jump_size = random_state.normal(
                       self.jump_mean,
                       self.jump_std,
                       dN
                   ).sum()
               else:
                   jump_size = 0

               # Combine diffusion and jump
               S[i] = S[i-1] * np.exp(
                   (self.drift - 0.5 * self.volatility**2) * dt +
                   self.volatility * dW +
                   jump_size
               )

           return times, S

   # Example: Revenue with occasional major disruptions
   revenue_process = JumpDiffusionProcess(
       drift=0.06,        # 6% growth
       volatility=0.10,   # 10% regular volatility
       jump_intensity=0.2,  # One jump every 5 years on average
       jump_mean=-0.25,   # 25% average revenue drop when jump occurs
       jump_std=0.15      # Jump size volatility
   )

Regime-Switching Models
~~~~~~~~~~~~~~~~~~~~~~~~

Model different economic regimes:

.. code-block:: python
   :caption: Regime-switching volatility model

   class RegimeSwitchingModel:
       """Switch between different volatility regimes."""

       def __init__(self):
           self.regimes = {
               'stable': {'growth': 0.06, 'volatility': 0.08},
               'volatile': {'growth': 0.04, 'volatility': 0.25},
               'crisis': {'growth': -0.05, 'volatility': 0.40}
           }

           # Transition probabilities (monthly)
           self.transition_matrix = {
               'stable': {'stable': 0.95, 'volatile': 0.04, 'crisis': 0.01},
               'volatile': {'stable': 0.10, 'volatile': 0.85, 'crisis': 0.05},
               'crisis': {'stable': 0.05, 'volatile': 0.25, 'crisis': 0.70}
           }

       def simulate(self, initial_regime, n_periods):
           """Simulate regime switches and returns."""

           current_regime = initial_regime
           regimes = []
           returns = []

           for period in range(n_periods):
               regimes.append(current_regime)

               # Generate return for current regime
               params = self.regimes[current_regime]
               monthly_return = np.random.normal(
                   params['growth'] / 12,
                   params['volatility'] / np.sqrt(12)
               )
               returns.append(monthly_return)

               # Transition to next regime
               probs = self.transition_matrix[current_regime]
               current_regime = np.random.choice(
                   list(probs.keys()),
                   p=list(probs.values())
               )

           return regimes, returns

Alternative Risk Transfer
-------------------------

Parametric Insurance Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Design parametric triggers for automatic payouts:

.. code-block:: python
   :caption: Parametric insurance for business interruption

   class ParametricInsurance:
       """Insurance that pays based on objective triggers."""

       def __init__(self, trigger_type='revenue_drop'):
           self.trigger_type = trigger_type

       def design_revenue_trigger(self, baseline_revenue):
           """Design parametric trigger based on revenue drop."""

           triggers = [
               {
                   'level': 'minor',
                   'threshold': 0.80 * baseline_revenue,  # 20% drop
                   'payout': 1_000_000,
                   'premium_rate': 0.02
               },
               {
                   'level': 'major',
                   'threshold': 0.60 * baseline_revenue,  # 40% drop
                   'payout': 5_000_000,
                   'premium_rate': 0.01
               },
               {
                   'level': 'severe',
                   'threshold': 0.40 * baseline_revenue,  # 60% drop
                   'payout': 15_000_000,
                   'premium_rate': 0.005
               }
           ]

           return triggers

       def calculate_payout(self, actual_revenue, triggers):
           """Calculate parametric payout based on actual results."""

           total_payout = 0

           for trigger in triggers:
               if actual_revenue < trigger['threshold']:
                   total_payout += trigger['payout']

           return total_payout

Captive Insurance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate forming a captive insurance company:

.. code-block:: python
   :caption: Captive insurance feasibility analysis

   class CaptiveAnalysis:
       """Analyze feasibility of captive insurance company."""

       def __init__(self, parent_company):
           self.parent = parent_company

       def evaluate_captive(self, retention_level, premium_volume):
           """Evaluate captive insurance economics."""

           # Initial capital requirements
           initial_capital = max(
               retention_level * 3,  # 3x annual retention
               5_000_000  # Regulatory minimum
           )

           # Operating costs
           annual_costs = {
               'management': 250_000,
               'regulatory': 100_000,
               'actuarial': 75_000,
               'audit': 50_000,
               'reinsurance': premium_volume * 0.60  # 60% ceded
           }

           # Tax benefits (simplified)
           tax_deduction = premium_volume * self.parent.tax_rate

           # Investment income on reserves
           investment_income = initial_capital * 0.04  # 4% return

           # Calculate NPV over 10 years
           cash_flows = []
           for year in range(10):
               # Inflows
               premium_received = premium_volume
               tax_benefit = tax_deduction
               investment_return = investment_income

               # Outflows
               operating_cost = sum(annual_costs.values())
               expected_claims = retention_level * 0.7  # 70% loss ratio

               net_cash_flow = (
                   premium_received +
                   tax_benefit +
                   investment_return -
                   operating_cost -
                   expected_claims
               )

               cash_flows.append(net_cash_flow)

           # Calculate NPV
           discount_rate = 0.08
           npv = -initial_capital
           for i, cf in enumerate(cash_flows):
               npv += cf / (1 + discount_rate) ** (i + 1)

           return {
               'initial_capital': initial_capital,
               'annual_benefit': np.mean(cash_flows),
               'npv': npv,
               'breakeven_year': self.find_breakeven(cash_flows, initial_capital),
               'recommended': npv > 0
           }

Performance Optimization
------------------------

Caching and Parallel Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Speed up large-scale simulations:

.. code-block:: python
   :caption: Optimized simulation with caching

   from functools import lru_cache
   from multiprocessing import Pool
   import pickle
   import hashlib

   class OptimizedMonteCarloEngine:
       """High-performance Monte Carlo engine."""

       def __init__(self, n_cores=None):
           self.n_cores = n_cores or mp.cpu_count()
           self.cache_dir = 'simulation_cache'

       def run_parallel(self, params, n_simulations):
           """Run simulations in parallel."""

           # Check cache first
           cache_key = self.generate_cache_key(params, n_simulations)
           cached_result = self.load_from_cache(cache_key)

           if cached_result is not None:
               print("Loaded from cache")
               return cached_result

           # Split simulations across cores
           chunk_size = n_simulations // self.n_cores
           chunks = [(params, chunk_size, i)
                    for i in range(self.n_cores)]

           # Run in parallel
           with Pool(self.n_cores) as pool:
               results = pool.map(self.run_chunk, chunks)

           # Combine results
           combined = self.combine_results(results)

           # Cache results
           self.save_to_cache(cache_key, combined)

           return combined

       @staticmethod
       def run_chunk(args):
           """Run a chunk of simulations."""
           params, n_sims, seed = args

           # Set unique seed for this chunk
           np.random.seed(seed)

           # Run simulations
           results = []
           for _ in range(n_sims):
               result = run_single_simulation(params)
               results.append(result)

           return results

       def generate_cache_key(self, params, n_simulations):
           """Generate unique cache key for parameters."""

           # Serialize parameters
           param_str = pickle.dumps((params, n_simulations))

           # Generate hash
           return hashlib.md5(param_str).hexdigest()

Next Steps
----------

These advanced topics provide powerful tools for customization:

1. **Start simple** - Use basic framework first
2. **Identify gaps** - Where does standard model fall short?
3. **Add complexity gradually** - One advanced feature at a time
4. **Validate thoroughly** - Backtest against historical data
5. **Document assumptions** - Critical for advanced models

For questions about these advanced topics, consult the :doc:`faq` or review the full API documentation.
