Examples
========

This section provides practical examples of using the Ergodic Insurance Limits framework
for various analysis scenarios.

Basic Simulation
----------------

Run a simple 100-year simulation with baseline parameters:

.. code-block:: python

    from ergodic_insurance import Simulation
    from ergodic_insurance.manufacturer import WidgetManufacturer
    from ergodic_insurance.loss_distributions import ManufacturingLossGenerator
    from ergodic_insurance.config_loader import load_config

    # Load configuration
    config = load_config("baseline")

    # Create components
    manufacturer = WidgetManufacturer(config.manufacturer)
    loss_generator = ManufacturingLossGenerator.create_simple(
        frequency=0.1, severity_mean=5_000_000, seed=42
    )

    # Run simulation
    sim = Simulation(
        manufacturer=manufacturer,
        loss_generator=loss_generator,
        time_horizon=100,
        seed=42
    )

    results = sim.run()

    # Get summary statistics
    stats = results.summary_stats()
    print(f"Final Assets: ${stats['final_assets']:,.0f}")
    print(f"Average ROE: {stats['mean_roe']:.2%}")
    print(f"Ruin Probability: {1 - stats['survived']:.2%}")

Scenario Comparison
-------------------

Compare baseline, conservative, and optimistic scenarios:

.. code-block:: python

    from ergodic_insurance.config_loader import ConfigLoader
    import pandas as pd

    loader = ConfigLoader()
    scenarios = ["baseline", "conservative", "optimistic"]
    results = {}

    for scenario in scenarios:
        config = loader.load_scenario(scenario)

        manufacturer = WidgetManufacturer(config.manufacturer)
        claim_generator = ClaimGenerator.from_config(config)

        sim = Simulation(manufacturer, claim_generator, time_horizon=100, seed=42)
        results[scenario] = sim.run().summary_statistics()

    # Create comparison DataFrame
    df = pd.DataFrame(results).T
    print(df[['final_assets', 'mean_roe', 'ruin_probability']])

Parameter Sensitivity Analysis
------------------------------

Analyze sensitivity to operating margin:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    margins = np.arange(0.04, 0.16, 0.01)
    final_assets = []
    ruin_probs = []

    for margin in margins:
        config = load_config("baseline", overrides={"manufacturer.operating_margin": margin})

        manufacturer = WidgetManufacturer(config.manufacturer)
        claim_generator = ClaimGenerator.from_config(config)

        sim = Simulation(manufacturer, claim_generator, time_horizon=100, seed=42)
        stats = sim.run().summary_statistics()

        final_assets.append(stats['final_assets'])
        ruin_probs.append(stats['ruin_probability'])

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(margins, final_assets, 'b-o')
    ax1.set_xlabel('Operating Margin')
    ax1.set_ylabel('Final Assets ($)')
    ax1.set_title('Final Assets vs Operating Margin')

    ax2.plot(margins, ruin_probs, 'r-o')
    ax2.set_xlabel('Operating Margin')
    ax2.set_ylabel('Ruin Probability')
    ax2.set_title('Ruin Risk vs Operating Margin')

    plt.tight_layout()
    plt.show()

Monte Carlo Analysis
--------------------

Run multiple simulations with different random seeds:

.. code-block:: python

    import numpy as np

    config = load_config("baseline")
    n_simulations = 1000
    seeds = np.arange(n_simulations)

    results = []

    for seed in seeds:
        manufacturer = WidgetManufacturer(config.manufacturer)
        claim_generator = ClaimGenerator.from_config(config, seed=seed)

        sim = Simulation(manufacturer, claim_generator, time_horizon=100, seed=seed)
        stats = sim.run().summary_statistics()
        results.append(stats)

    # Convert to DataFrame for analysis
    mc_results = pd.DataFrame(results)

    print("Monte Carlo Results Summary:")
    print(f"Mean Final Assets: ${mc_results['final_assets'].mean():,.0f}")
    print(f"Std Final Assets: ${mc_results['final_assets'].std():,.0f}")
    print(f"Overall Ruin Probability: {mc_results['ruin_probability'].mean():.2%}")

    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(mc_results['final_assets'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Final Assets ($)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Final Assets ({n_simulations:,} Simulations)')
    plt.show()

Custom Claim Modeling
----------------------

Create a custom claim generator with specific loss patterns:

.. code-block:: python

    # Custom claim generator for high-frequency, low-severity losses
    claim_generator = ClaimGenerator(
        attritional_frequency=8.0,      # 8 claims per year on average
        attritional_severity_params=(25000, 0.6),  # Lower severity
        large_loss_frequency=0.1,       # Rare large losses
        large_loss_severity_params=(10000000, 1.5),  # But very severe
        correlation=0.2,                # Low correlation
        seed=42
    )

    # Generate sample year of claims
    annual_claims = claim_generator.generate_annual_claims(year=1)

    print(f"Number of claims: {len(annual_claims)}")
    print(f"Total claim amount: ${sum(c.amount for c in annual_claims):,.0f}")

    # Get expected loss statistics
    stats = claim_generator.get_loss_statistics()
    print(f"Expected annual loss: ${stats['expected_annual_loss']:,.0f}")

Working with Configuration
--------------------------

Advanced configuration management:

.. code-block:: python

    from ergodic_insurance.config_loader import ConfigLoader
    from pathlib import Path

    loader = ConfigLoader()

    # Load and modify configuration
    base_config = loader.load("baseline")

    # Create a high-growth scenario
    high_growth = base_config.override({
        "growth.annual_growth_rate": 0.12,
        "growth.type": "stochastic",
        "growth.volatility": 0.15,
        "manufacturer.operating_margin": 0.10,
    })

    # Save custom configuration
    output_path = Path("outputs/high_growth_config.yaml")
    high_growth.to_yaml(output_path)

    # Compare with baseline
    differences = loader.compare_configs(base_config, high_growth)
    for param, diff in differences.items():
        print(f"{param}: {diff['config1']} -> {diff['config2']}")

Performance Optimization
------------------------

For large-scale analysis, use these performance tips:

.. code-block:: python

    import time

    # Configure for performance
    config = load_config(
        "baseline",
        overrides={
            "simulation.time_resolution": "annual",  # Use annual steps
            "output.detailed_metrics": False,         # Reduce output detail
            "logging.enabled": False,                 # Disable logging
        },
    )

    # Time a long simulation
    start_time = time.time()

    manufacturer = WidgetManufacturer(config.manufacturer)
    claim_generator = ClaimGenerator.from_config(config)

    # 1000-year simulation
    sim = Simulation(manufacturer, claim_generator, time_horizon=1000)
    results = sim.run()

    elapsed = time.time() - start_time
    print(f"1000-year simulation completed in {elapsed:.2f} seconds")

    # For even better performance, use run_to_dataframe()
    # which skips creating the full SimulationResults object
    df = sim.run_to_dataframe()

Interactive Analysis
--------------------

For interactive exploration, try the Jupyter notebooks:

* ``notebooks/00_setup_verification.ipynb`` - Installation verification
* ``notebooks/01_basic_manufacturer.ipynb`` - Core financial modeling
* ``notebooks/02_long_term_simulation.ipynb`` - Extended time horizon analysis
* ``notebooks/03_growth_dynamics.ipynb`` - Growth rate sensitivity

These notebooks provide step-by-step walkthroughs with visualizations and
detailed explanations of the underlying theory.
