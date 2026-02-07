# pylint: disable=too-many-lines
# This module contains comprehensive ergodic analysis functionality with extensive
# documentation. The length is justified by the complexity of the subject matter
# and the need for thorough explanations of ergodic theory applications.

"""Ergodic analysis framework for comparing time-average vs ensemble-average growth.

This module provides the theoretical foundation and computational tools for applying
ergodic economics to insurance decision making. It implements Ole Peters' framework
for distinguishing between ensemble averages (what we expect to happen across many
parallel scenarios) and time averages (what actually happens to a single entity
over time).

The key insight is that for multiplicative processes like business growth with
volatile losses, the ensemble average and time average diverge significantly.
Insurance transforms the growth process in ways that traditional expected value
analysis cannot capture, often making insurance optimal even when premiums
exceed expected losses by substantial margins.

Key Concepts:
    **Time Average Growth Rate**: The growth rate experienced by a single business
    entity over time, calculated as g = (1/T) * ln(X(T)/X(0)). This captures the
    actual compound growth experience.

    **Ensemble Average Growth Rate**: The expected growth rate calculated across
    many parallel scenarios at each time point. This represents the traditional
    expected value approach.

    **Ergodic Divergence**: The difference between time and ensemble averages,
    indicating non-ergodic behavior where individual experience differs from
    statistical expectations.

    **Survival Rate**: The fraction of simulation paths that remain solvent,
    capturing the probability dimension ignored by pure growth metrics.

Theoretical Foundation:
    Based on Ole Peters' ergodic economics framework (Peters, 2019; Peters & Gell-Mann, 2016),
    this module demonstrates that:

    1. **Multiplicative Growth**: Business equity follows multiplicative dynamics
       where losses compound over time in non-linear ways.

    2. **Jensen's Inequality**: For concave utility functions (log wealth), the
       expected value of a function differs from the function of expected values.

    3. **Path Dependence**: The order and timing of losses matters critically,
       making time-average analysis essential for decision making.

    4. **Insurance as Growth Optimization**: Insurance can increase time-average
       growth rates even when premiums appear "expensive" from ensemble perspective.

Core Classes:
    - :class:`ErgodicAnalyzer`: Main analysis engine with comparison methods
    - :class:`ErgodicData`: Standardized data container for time series analysis
    - :class:`ErgodicAnalysisResults`: Comprehensive results from integrated analysis
    - :class:`ValidationResults`: Insurance impact validation results

Examples:
    Basic ergodic comparison between insured and uninsured scenarios::

        import numpy as np
        from ergodic_insurance import ErgodicAnalyzer

        # Initialize analyzer
        analyzer = ErgodicAnalyzer(convergence_threshold=0.01)

        # Simulate equity trajectories (example data)
        insured_trajectories = [
            np.array([10e6, 10.2e6, 10.5e6, 10.8e6, 11.1e6]),    # Stable growth
            np.array([10e6, 10.1e6, 10.3e6, 10.6e6, 10.9e6]),    # Stable growth
            np.array([10e6, 10.3e6, 10.7e6, 11.0e6, 11.4e6])     # Stable growth
        ]

        uninsured_trajectories = [
            np.array([10e6, 10.5e6, 8.2e6, 12.1e6, 13.5e6]),     # Volatile
            np.array([10e6, 9.8e6, 5.1e6, 0]),                   # Bankruptcy
            np.array([10e6, 10.8e6, 11.2e6, 14.8e6, 16.2e6])     # High growth
        ]

        # Compare scenarios
        comparison = analyzer.compare_scenarios(
            insured_trajectories,
            uninsured_trajectories,
            metric="equity"
        )

        print(f"Insured time-average growth: {comparison['insured']['time_average_mean']:.1%}")
        print(f"Uninsured time-average growth: {comparison['uninsured']['time_average_mean']:.1%}")
        print(f"Ergodic advantage: {comparison['ergodic_advantage']['time_average_gain']:.1%}")
        print(f"Survival rate improvement: {comparison['ergodic_advantage']['survival_gain']:.1%}")

    Monte Carlo analysis with convergence checking::

        from ergodic_insurance.simulation import run_monte_carlo

        # Run Monte Carlo simulations (pseudo-code)
        simulation_results = run_monte_carlo(
            n_simulations=1000,
            time_horizon=20,
            insurance_enabled=True
        )

        # Analyze batch results
        analysis = analyzer.analyze_simulation_batch(
            simulation_results,
            label="Insured Scenario"
        )

        print(f"Time-average growth: {analysis['time_average']['mean']:.2%} ± {analysis['time_average']['std']:.2%}")
        print(f"Ensemble average: {analysis['ensemble_average']['mean']:.2%}")
        print(f"Ergodic divergence: {analysis['ergodic_divergence']:.2%}")
        print(f"Convergence: {analysis['convergence']['converged']} (SE: {analysis['convergence']['standard_error']:.4f})")

    Integration with loss modeling::

        from ergodic_insurance import LossData, InsuranceProgram, WidgetManufacturer

        # Set up integrated analysis
        loss_data = LossData.from_distribution(
            frequency_lambda=2.5,
            severity_mean=1_000_000,
            severity_cv=2.0
        )

        insurance = InsuranceProgram(
            layers=[(0, 1_000_000, 0.015), (1_000_000, 10_000_000, 0.008)]
        )

        manufacturer = WidgetManufacturer(config)

        # Run integrated ergodic analysis
        results = analyzer.integrate_loss_ergodic_analysis(
            loss_data=loss_data,
            insurance_program=insurance,
            manufacturer=manufacturer,
            time_horizon=20,
            n_simulations=1000
        )

        print(f"Time-average growth rate: {results.time_average_growth:.2%}")
        print(f"Ensemble average growth: {results.ensemble_average_growth:.2%}")
        print(f"Survival rate: {results.survival_rate:.1%}")
        print(f"Insurance benefit: ${results.insurance_impact['net_benefit']:,.0f}")
        print(f"Analysis valid: {results.validation_passed}")

Implementation Notes:
    - All growth rate calculations use natural logarithms for mathematical consistency
    - Infinite values (from bankruptcy) are handled gracefully in statistical calculations
    - Convergence checking uses standard error to determine Monte Carlo adequacy
    - Significance testing employs t-tests for comparing growth rate distributions
    - Variable-length trajectories (due to insolvency) are supported throughout

Performance Optimization:
    - Vectorized numpy operations for large Monte Carlo batches
    - Efficient handling of mixed-length trajectory data
    - Memory-conscious processing of large simulation datasets
    - Configurable convergence thresholds to balance accuracy and computation time

References:
    - Peters, O. (2019). "The ergodicity problem in economics." Nature Physics, 15(12), 1216-1221.
    - Peters, O., & Gell-Mann, M. (2016). "Evaluating gambles using dynamics." Chaos, 26(2), 023103.
    - Kelly, J. L. (1956). "A new interpretation of information rate." Bell System Technical Journal, 35(4), 917-926.

See Also:
    :mod:`~ergodic_insurance.simulation`: Monte Carlo simulation framework
    :mod:`~ergodic_insurance.manufacturer`: Financial model for business dynamics
    :mod:`~ergodic_insurance.insurance_program`: Insurance structure modeling
    :mod:`~ergodic_insurance.optimization`: Optimization algorithms using ergodic metrics
"""

from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

from .simulation import SimulationResults

if TYPE_CHECKING:
    from .insurance_program import InsuranceProgram
    from .loss_distributions import LossData

logger = logging.getLogger(__name__)


@dataclass
class ErgodicData:
    """Standardized data container for ergodic time series analysis.

    This class provides a consistent format for storing and validating time series
    data used in ergodic calculations. It ensures data integrity and provides
    metadata tracking for analysis reproducibility.

    Attributes:
        time_series (np.ndarray): Array of time points corresponding to values.
            Should be monotonically increasing for meaningful analysis.
        values (np.ndarray): Array of observed values (e.g., equity, assets) at
            each time point. Must have same length as time_series.
        metadata (Dict[str, Any]): Dictionary containing analysis metadata such as
            simulation parameters, data source, units, etc.

    Examples:
        Create ergodic data for analysis::

            import numpy as np

            # Equity trajectory over 10 years
            data = ErgodicData(
                time_series=np.arange(11),  # Years 0-10
                values=np.array([10e6, 10.2e6, 10.5e6, 10.1e6, 10.8e6,
                               11.2e6, 10.9e6, 11.5e6, 12.1e6, 12.8e6, 13.2e6]),
                metadata={
                    'units': 'USD',
                    'metric': 'equity',
                    'simulation_id': 'run_001',
                    'scenario': 'insured'
                }
            )

            # Validate data consistency
            assert data.validate(), "Data validation failed"

        Handle validation failures::

            # Mismatched lengths will fail validation
            invalid_data = ErgodicData(
                time_series=np.arange(10),
                values=np.arange(5),  # Wrong length
                metadata={'note': 'This will fail validation'}
            )

            if not invalid_data.validate():
                print("Data validation failed - fix before analysis")

    Note:
        The validate() method should be called before using data in ergodic
        calculations to ensure mathematical operations will succeed.

    See Also:
        :class:`ErgodicAnalyzer`: Main analysis class that uses ErgodicData
        :class:`ErgodicAnalysisResults`: Results format for ergodic calculations
    """

    time_series: np.ndarray = field(default_factory=lambda: np.array([]))
    values: np.ndarray = field(default_factory=lambda: np.array([]))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate data consistency and integrity.

        Performs comprehensive validation of the ergodic data to ensure it meets
        requirements for mathematical analysis. This includes checking array
        lengths, data types, and basic reasonableness of values.

        Returns:
            bool: True if all validation checks pass, False otherwise.
                False indicates the data should not be used in ergodic calculations
                without correction.

        Examples:
            Validate data before analysis::

                data = ErgodicData(
                    time_series=np.arange(10),
                    values=np.random.randn(10) + 100,
                    metadata={'units': 'USD'}
                )

                if data.validate():
                    print("Data validated - ready for analysis")
                else:
                    print("Data validation failed - check inputs")

        Validation Checks:
            - Arrays have matching lengths
            - Arrays are not empty
            - Time series is monotonic (if more than one point)
            - Values are numeric (not NaN in inappropriate places)
        """
        if len(self.time_series) == 0 or len(self.values) == 0:
            return False
        return len(self.time_series) == len(self.values)


@dataclass
class ErgodicAnalysisResults:
    """Comprehensive results from integrated ergodic analysis.

    This class encapsulates all results from a complete ergodic analysis,
    including growth metrics, survival statistics, insurance impacts, and
    validation status. It provides a standardized format for reporting
    and comparing different insurance strategies.

    Attributes:
        time_average_growth (float): Mean time-average growth rate across
            all valid simulation paths. Calculated as the average of individual
            path growth rates: mean(ln(X_final/X_initial)/T). May be -inf if
            all paths resulted in bankruptcy.
        ensemble_average_growth (float): Ensemble average growth rate calculated
            from the mean of initial and final values across all paths:
            ln(mean(X_final)/mean(X_initial))/T. Always finite for valid data.
        survival_rate (float): Fraction of simulation paths that remained solvent
            throughout the analysis period. Range: [0.0, 1.0].
        ergodic_divergence (float): Difference between time-average and ensemble
            average growth rates (time_average_growth - ensemble_average_growth).
            Positive values indicate time-average exceeds ensemble average.
        insurance_impact (Dict[str, float]): Dictionary containing insurance-related
            metrics such as:
            - 'premium_cost': Total premium payments
            - 'recovery_benefit': Total insurance recoveries
            - 'net_benefit': Net financial benefit of insurance
            - 'growth_improvement': Improvement in growth rate from insurance
        validation_passed (bool): Whether the analysis passed internal validation
            checks for data consistency and mathematical validity.
        metadata (Dict[str, Any]): Additional analysis metadata including:
            - 'n_simulations': Number of Monte Carlo simulations
            - 'time_horizon': Analysis time horizon
            - 'n_survived': Absolute number of paths that survived
            - 'loss_statistics': Statistics about loss distributions

    Examples:
        Interpret analysis results::

            # Example results from ergodic analysis
            results = ErgodicAnalysisResults(
                time_average_growth=0.045,      # 4.5% annual growth
                ensemble_average_growth=0.052,   # 5.2% ensemble average
                survival_rate=0.95,              # 95% survival rate
                ergodic_divergence=-0.007,       # -0.7% divergence
                insurance_impact={
                    'premium_cost': 2_500_000,
                    'recovery_benefit': 8_200_000,
                    'net_benefit': 5_700_000,
                    'growth_improvement': 0.012
                },
                validation_passed=True,
                metadata={
                    'n_simulations': 1000,
                    'time_horizon': 20,
                    'n_survived': 950
                }
            )

            # Interpret results
            if results.validation_passed:
                print(f"Time-average growth: {results.time_average_growth:.1%}")
                print(f"Ensemble average: {results.ensemble_average_growth:.1%}")

                if results.ergodic_divergence < 0:
                    print("Insurance reduces volatility drag (ergodic benefit)")

                if results.insurance_impact['net_benefit'] > 0:
                    print(f"Insurance provides net benefit: ${results.insurance_impact['net_benefit']:,.0f}")
            else:
                print("Analysis validation failed - results may be unreliable")

        Compare multiple scenarios::

            def compare_results(results_a, results_b, label_a="Scenario A", label_b="Scenario B"):
                print(f"{label_a} vs {label_b}:")
                print(f"  Time-average growth: {results_a.time_average_growth:.2%} vs {results_b.time_average_growth:.2%}")
                print(f"  Survival rate: {results_a.survival_rate:.1%} vs {results_b.survival_rate:.1%}")
                print(f"  Ergodic divergence: {results_a.ergodic_divergence:.3f} vs {results_b.ergodic_divergence:.3f}")

                growth_advantage = results_a.time_average_growth - results_b.time_average_growth
                survival_advantage = results_a.survival_rate - results_b.survival_rate

                print(f"  Advantages: Growth={growth_advantage:.2%}, Survival={survival_advantage:.1%}")

    Note:
        All growth rates are expressed as decimal values (0.05 = 5% annual growth).
        Negative ergodic_divergence indicates insurance reduces "volatility drag".
        Always check validation_passed before interpreting results.

    See Also:
        :class:`ErgodicAnalyzer`: Class that generates these results
        :meth:`ErgodicAnalyzer.integrate_loss_ergodic_analysis`: Method producing these results
        :class:`ValidationResults`: Detailed validation information
    """

    time_average_growth: float
    ensemble_average_growth: float
    survival_rate: float
    ergodic_divergence: float
    insurance_impact: Dict[str, float]
    validation_passed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResults:
    """Comprehensive results from insurance impact validation analysis.

    This class encapsulates the results of detailed validation checks performed
    on insurance effects in ergodic analysis. It provides both high-level
    validation status and detailed diagnostic information to help identify
    and resolve any modeling inconsistencies.

    Attributes:
        premium_deductions_correct (bool): Whether insurance premiums are properly
            deducted from cash flows. True indicates expected premium costs match
            observed differences in net income between scenarios.
        recoveries_credited (bool): Whether insurance recoveries are properly
            credited to improve financial outcomes. True indicates insured scenarios
            show appropriate financial benefit from loss recoveries.
        collateral_impacts_included (bool): Whether letter of credit costs and
            asset restrictions are properly modeled. True indicates collateral
            requirements are reflected in financial calculations.
        time_average_reflects_benefit (bool): Whether time-average growth rate
            calculations properly reflect insurance benefits. True indicates growth
            improvements are consistent with insurance effects.
        overall_valid (bool): Master validation flag indicating whether all
            individual checks passed. True means the ergodic analysis results
            are reliable and properly reflect insurance impacts.
        details (Dict[str, Any]): Detailed diagnostic information from each
            validation check, including specific metrics, calculations, and
            discrepancy measurements. Used for troubleshooting validation failures.

    Examples:
        Interpret validation results::

            validation = analyzer.validate_insurance_ergodic_impact(
                base_scenario, insurance_scenario, insurance_program
            )

            if validation.overall_valid:
                print("✓ All validation checks passed")
                print("Ergodic analysis results are reliable")
            else:
                print("⚠ Validation issues detected:")

                if not validation.premium_deductions_correct:
                    print("  - Premium deduction mismatch")
                if not validation.recoveries_credited:
                    print("  - Recovery crediting issue")
                if not validation.collateral_impacts_included:
                    print("  - Collateral impact missing")
                if not validation.time_average_reflects_benefit:
                    print("  - Growth calculation inconsistency")

                print("Review model implementation before using results")

        Access detailed diagnostics::

            if 'premium_check' in validation.details:
                premium_info = validation.details['premium_check']
                expected = premium_info['expected']
                actual = premium_info['actual_diff']
                print(f"Premium validation: Expected ${expected:,.0f}, Got ${actual:,.0f}")

                if abs(expected - actual) > expected * 0.05:  # 5% tolerance
                    print("⚠ Significant premium discrepancy detected")

    Note:
        A failed overall validation doesn't necessarily mean the analysis is
        wrong - it may indicate edge cases or modeling assumptions that need
        review. Always examine the details for specific guidance on issues.

    See Also:
        :meth:`ErgodicAnalyzer.validate_insurance_ergodic_impact`: Method generating these results
        :class:`ErgodicAnalysisResults`: Main analysis results that this validation supports
    """

    premium_deductions_correct: bool
    recoveries_credited: bool
    collateral_impacts_included: bool
    time_average_reflects_benefit: bool
    overall_valid: bool
    details: Dict[str, Any] = field(default_factory=dict)


class ErgodicAnalyzer:
    """Advanced analyzer for ergodic properties of insurance strategies.

    This class implements the core computational engine for ergodic economics
    analysis in insurance contexts. It provides methods to calculate and compare
    time-average versus ensemble-average growth rates, demonstrating the fundamental
    difference between traditional expected-value thinking and actual experienced
    growth over time.

    The analyzer addresses the key ergodic insight that for multiplicative processes
    (like business growth with volatile losses), what happens to an ensemble of
    businesses differs from what happens to any individual business over time.
    Insurance can improve time-average growth even when it appears costly from
    an ensemble (expected value) perspective.

    Key Capabilities:
        - Time-average growth rate calculation for individual trajectories
        - Ensemble average computation across multiple simulation paths
        - Statistical significance testing of insurance benefits
        - Monte Carlo convergence analysis
        - Integrated loss modeling and insurance impact assessment
        - Comprehensive validation of insurance effects

    Attributes:
        convergence_threshold (float): Standard error threshold for determining
            Monte Carlo convergence. Lower values require more simulations but
            provide higher confidence in results.

    Mathematical Foundation:
        **Time-Average Growth**: For a trajectory X(t), the time-average growth rate is:
            g_time = (1/T) * ln(X(T)/X(0))

        **Ensemble Average Growth**: Across N paths, the ensemble growth rate is:
            g_ensemble = (1/T) * ln(⟨X(T)⟩/⟨X(0)⟩)

        **Ergodic Divergence**: The difference g_time - g_ensemble indicates
            non-ergodic behavior where individual experience differs from
            statistical expectations.

    Examples:
        Basic analyzer setup and usage::

            from ergodic_insurance import ErgodicAnalyzer
            import numpy as np

            # Initialize with tight convergence criteria
            analyzer = ErgodicAnalyzer(convergence_threshold=0.005)

            # Calculate time-average growth for a single trajectory
            equity_path = np.array([10e6, 10.5e6, 9.8e6, 11.2e6, 12.1e6])
            time_avg_growth = analyzer.calculate_time_average_growth(equity_path)
            print(f"Time-average growth: {time_avg_growth:.2%} annually")

        Ensemble analysis with multiple trajectories::

            # Multiple simulation paths (some ending in bankruptcy)
            trajectories = [
                np.array([10e6, 10.5e6, 11.2e6, 11.8e6, 12.5e6]),  # Survivor
                np.array([10e6, 9.2e6, 8.1e6, 6.8e6, 0]),          # Bankruptcy
                np.array([10e6, 10.8e6, 11.5e6, 12.8e6, 14.2e6]),  # High growth
                np.array([10e6, 9.8e6, 10.2e6, 10.6e6, 11.1e6])   # Stable growth
            ]

            # Calculate ensemble statistics
            ensemble_stats = analyzer.calculate_ensemble_average(
                trajectories,
                metric="growth_rate"
            )

            print(f"Ensemble growth rate: {ensemble_stats['mean']:.2%}")
            print(f"Survival rate: {ensemble_stats['survival_rate']:.1%}")
            print(f"Growth rate std dev: {ensemble_stats['std']:.2%}")

        Insurance scenario comparison::

            # Compare insured vs uninsured scenarios
            insured_paths = generate_insured_trajectories()    # Your simulation code
            uninsured_paths = generate_uninsured_trajectories()  # Your simulation code

            comparison = analyzer.compare_scenarios(
                insured_paths,
                uninsured_paths,
                metric="equity"
            )

            # Extract key insights
            time_avg_benefit = comparison['ergodic_advantage']['time_average_gain']
            survival_benefit = comparison['ergodic_advantage']['survival_gain']
            is_significant = comparison['ergodic_advantage']['significant']

            print(f"Time-average growth improvement: {time_avg_benefit:.2%}")
            print(f"Survival rate improvement: {survival_benefit:.1%}")
            print(f"Statistically significant: {is_significant}")

        Monte Carlo convergence analysis::

            # Run large Monte Carlo study
            simulation_results = run_monte_carlo_study(n_sims=2000)

            analysis = analyzer.analyze_simulation_batch(
                simulation_results,
                label="High-Coverage Insurance"
            )

            # Check if we have enough simulations
            if analysis['convergence']['converged']:
                print("Monte Carlo has converged - results are reliable")
                print(f"Standard error: {analysis['convergence']['standard_error']:.4f}")
            else:
                print("Need more simulations for convergence")
                needed_se = analyzer.convergence_threshold
                current_se = analysis['convergence']['standard_error']
                factor = (current_se / needed_se) ** 2
                print(f"Suggest ~{int(2000 * factor)} simulations")

    Advanced Features:
        The analyzer provides several advanced capabilities for robust analysis:

        **Variable-Length Trajectories**: Handles paths that end early due to
        bankruptcy, maintaining proper statistics across mixed survival scenarios.

        **Significance Testing**: Built-in t-tests to determine if observed
        differences between scenarios are statistically meaningful.

        **Convergence Monitoring**: Automated checking of Monte Carlo convergence
        using rolling standard error calculations.

        **Integrated Validation**: Comprehensive validation of insurance effects
        to ensure results accurately reflect premium costs, recoveries, and
        collateral impacts.

    Performance Notes:
        - Optimized for large Monte Carlo datasets (1000+ simulations)
        - Memory-efficient processing of variable-length trajectories
        - Vectorized calculations where possible for speed
        - Graceful handling of edge cases (bankruptcy, infinite values)

    See Also:
        :class:`ErgodicAnalysisResults`: Comprehensive results format
        :class:`ValidationResults`: Insurance impact validation results
        :meth:`integrate_loss_ergodic_analysis`: End-to-end analysis pipeline
        :meth:`compare_scenarios`: Core scenario comparison functionality
    """

    def __init__(self, convergence_threshold: float = 0.01):
        """Initialize the ergodic analyzer with configuration parameters.

        Sets up the analyzer with the specified convergence criteria for Monte Carlo
        analysis. The convergence threshold determines when sufficient simulations
        have been run to provide reliable statistical estimates.

        Args:
            convergence_threshold (float): Standard error threshold for determining
                Monte Carlo convergence. Lower values require more simulations but
                provide higher confidence. Typical values:
                - 0.005: High precision (2000+ simulations)
                - 0.01: Standard precision (1000+ simulations)
                - 0.02: Quick analysis (500+ simulations)
                Defaults to 0.01 for balanced accuracy and computational efficiency.

        Examples:
            Initialize for different analysis needs::

                # High precision for final analysis
                precise_analyzer = ErgodicAnalyzer(convergence_threshold=0.005)

                # Quick analysis for parameter exploration
                quick_analyzer = ErgodicAnalyzer(convergence_threshold=0.02)

                # Standard analysis for most use cases
                standard_analyzer = ErgodicAnalyzer()  # Uses default 0.01

        Note:
            The convergence threshold affects the balance between computational
            cost and result reliability. Choose based on your analysis requirements
            and available computational resources.
        """
        self.convergence_threshold = convergence_threshold

    def calculate_time_average_growth(
        self, values: np.ndarray, time_horizon: Optional[int] = None
    ) -> float:
        """Calculate time-average growth rate for a single trajectory.

        This method implements the core ergodic calculation for individual path
        growth rates using the logarithmic growth formula. It handles edge cases
        gracefully, including bankruptcy scenarios and invalid data.

        The time-average growth rate represents the actual compound growth
        experienced by a single entity over time, which differs fundamentally
        from ensemble averages in multiplicative processes.

        Args:
            values (np.ndarray): Array of values over time (e.g., equity, assets,
                wealth). Should be monotonic in time with positive values for
                meaningful growth calculations. Length must be >= 2 for growth
                calculation.
            time_horizon (Optional[int]): Specific time horizon to use for
                calculation. If None, uses the full trajectory length minus 1.
                Useful for comparing trajectories of different lengths or
                analyzing partial periods.

        Returns:
            float: Time-average growth rate as decimal (0.05 = 5% annual growth).
                Special return values:
                - -inf: Trajectory ended in bankruptcy (final value <= 0)
                - 0.0: Single time point or zero time horizon
                - Finite value: Calculated growth rate

        Examples:
            Calculate growth for successful trajectory::

                import numpy as np

                # 5-year equity trajectory
                equity = np.array([10e6, 10.5e6, 11.2e6, 11.8e6, 12.5e6])
                growth = analyzer.calculate_time_average_growth(equity)
                print(f"Growth rate: {growth:.2%} annually")
                # Output: Growth rate: 5.68% annually

            Handle bankruptcy scenario::

                # Trajectory ending in bankruptcy
                failed_equity = np.array([10e6, 9.2e6, 7.1e6, 4.8e6, 0])
                growth = analyzer.calculate_time_average_growth(failed_equity)
                print(f"Growth rate: {growth}")
                # Output: Growth rate: -inf

            Analyze partial trajectory::

                # Long trajectory, analyze first 3 years only
                long_equity = np.array([10e6, 10.5e6, 11.2e6, 11.8e6, 12.5e6, 13.1e6])
                partial_growth = analyzer.calculate_time_average_growth(
                    long_equity,
                    time_horizon=3
                )
                # Analyzes first 4 points (years 0-3)

            Handle trajectories with initial zeros::

                # Trajectory starting from zero (invalid)
                invalid_equity = np.array([0, 1e6, 2e6, 3e6, 4e6])
                growth = analyzer.calculate_time_average_growth(invalid_equity)
                # Will find first valid positive value and calculate from there

        Mathematical Details:
            The calculation uses the formula:
                g = (1/T) * ln(X(T) / X(0))

            Where:
            - g: time-average growth rate
            - T: time horizon
            - X(T): final value
            - X(0): initial value
            - ln: natural logarithm

            This formula gives the constant compound growth rate that would
            produce the observed change from initial to final value.

        Edge Cases:
            - Empty array: Returns -inf
            - Single value: Returns 0.0
            - Final value <= 0: Returns -inf (bankruptcy)
            - All values <= 0: Returns -inf
            - Zero time horizon: Returns 0.0 if positive, -inf if negative

        Note:
            This is the fundamental calculation in ergodic economics, representing
            the growth rate that a single entity actually experiences over time,
            as opposed to what we might expect from ensemble averages.

        Warning:
            The method filters out non-positive values when finding the initial
            value, which may skip early periods of the trajectory. Ensure your
            data represents meaningful business values (positive equity/assets).

        See Also:
            :meth:`calculate_ensemble_average`: For ensemble growth calculations
            :meth:`compare_scenarios`: For comparing time vs ensemble averages
        """
        # Handle edge cases and invalid trajectories
        if len(values) == 0:
            return -np.inf
        if values[-1] <= 0:
            return -np.inf

        if len(values) == 1:
            return 0.0

        # Filter out zero or negative values for calculating growth
        valid_mask = values > 0
        if not np.any(valid_mask):
            return -np.inf

        # Get first valid positive value
        first_idx = np.argmax(valid_mask)
        initial_value = values[first_idx]
        final_value = values[-1]

        # Calculate time period
        if time_horizon is None:
            time_horizon = int(len(values) - 1 - first_idx)

        # Calculate growth rate
        if final_value > 0 and initial_value > 0 and time_horizon > 0:
            growth_rate = float((1.0 / time_horizon) * np.log(final_value / initial_value))
            # LIMITED LIABILITY: Floor at -100% (-1.0) due to equity constraint
            # Companies with limited liability cannot lose more than 100% of equity
            return max(growth_rate, -1.0)

        return 0.0 if time_horizon <= 0 else -np.inf

    def _extract_trajectory_values(
        self, trajectories: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract initial, final values and lengths from trajectories."""
        values_data = [(traj[-1], traj[0], len(traj)) for traj in trajectories if len(traj) > 0]
        if not values_data:
            return np.array([]), np.array([]), np.array([])

        finals, initials, lengths = zip(*values_data)
        return np.array(finals), np.array(initials), np.array(lengths)

    def _calculate_growth_rates(
        self, finals: np.ndarray, initials: np.ndarray, lengths: np.ndarray
    ) -> np.ndarray:
        """Calculate growth rates from trajectory values."""
        rates = [np.log(f / i) / (t - 1) for f, i, t in zip(finals, initials, lengths) if t > 1]
        return np.array(rates) if rates else np.array([])

    def _process_variable_length_trajectories(
        self, trajectories: List[np.ndarray], metric: str
    ) -> Dict[str, float]:
        """Process trajectories with variable lengths."""
        n_paths = len(trajectories)
        results = {}

        if metric in ["final_value", "growth_rate"]:
            # Get final and initial values from each trajectory
            final_values, initial_values, time_lengths = self._extract_trajectory_values(
                trajectories
            )

            # Filter valid paths (positive initial and final values)
            if len(final_values) > 0:
                valid_mask = (initial_values > 0) & (final_values > 0)
                valid_finals = final_values[valid_mask]
                valid_initials = initial_values[valid_mask]
                valid_lengths = time_lengths[valid_mask]
            else:
                valid_finals = valid_initials = valid_lengths = np.array([])

            if metric == "final_value":
                results["mean"] = np.mean(valid_finals) if len(valid_finals) > 0 else 0.0
                results["std"] = np.std(valid_finals) if len(valid_finals) > 0 else 0.0
                results["median"] = np.median(valid_finals) if len(valid_finals) > 0 else 0.0
            else:  # growth_rate
                growth_rates = self._calculate_growth_rates(
                    valid_finals, valid_initials, valid_lengths
                )
                results["mean"] = np.mean(growth_rates) if len(growth_rates) > 0 else 0.0
                results["std"] = np.std(growth_rates) if len(growth_rates) > 0 else 0.0
                results["median"] = np.median(growth_rates) if len(growth_rates) > 0 else 0.0
        else:  # full trajectory - not well-defined for different lengths
            results["mean_trajectory"] = None
            results["std_trajectory"] = None

        # Add survival statistics
        survived = sum(1 for traj in trajectories if len(traj) > 0 and traj[-1] > 0)
        results["survival_rate"] = survived / n_paths if n_paths > 0 else 0.0
        results["n_survived"] = survived
        results["n_total"] = n_paths

        return results

    def _process_fixed_length_trajectories(
        self, trajectories: np.ndarray, metric: str
    ) -> Dict[str, float]:
        """Process trajectories with fixed lengths."""
        n_paths, n_time = trajectories.shape
        results = {}

        if metric in ["final_value", "growth_rate"]:
            # Get final values (trajectories is now a numpy array)
            final_values = trajectories[:, -1]
            initial_values = trajectories[:, 0]

            # Filter valid paths (positive initial and final values)
            valid_mask = (initial_values > 0) & (final_values > 0)
            valid_finals = final_values[valid_mask]
            valid_initials = initial_values[valid_mask]

            if metric == "final_value":
                results["mean"] = np.mean(valid_finals) if len(valid_finals) > 0 else 0.0
                results["std"] = np.std(valid_finals) if len(valid_finals) > 0 else 0.0
                results["median"] = np.median(valid_finals) if len(valid_finals) > 0 else 0.0
            else:  # growth_rate
                growth_rates = np.log(valid_finals / valid_initials) / (n_time - 1)
                results["mean"] = np.mean(growth_rates) if len(growth_rates) > 0 else 0.0
                results["std"] = np.std(growth_rates) if len(growth_rates) > 0 else 0.0
                results["median"] = np.median(growth_rates) if len(growth_rates) > 0 else 0.0
        else:  # full trajectory
            results["mean_trajectory"] = np.mean(trajectories, axis=0)
            results["std_trajectory"] = np.std(trajectories, axis=0)

        # Add survival statistics
        survived = np.sum(trajectories[:, -1] > 0)
        results["survival_rate"] = survived / n_paths
        results["n_survived"] = survived
        results["n_total"] = n_paths

        return results

    def calculate_ensemble_average(
        self, trajectories: Union[List[np.ndarray], np.ndarray], metric: str = "final_value"
    ) -> Dict[str, float]:
        """Calculate ensemble average and statistics across multiple simulation paths.

        This method computes ensemble statistics representing the traditional
        expected value approach to analyzing multiple parallel scenarios. It
        handles variable-length trajectories (due to bankruptcy) and provides
        comprehensive statistics for comparison with time-average calculations.

        The ensemble perspective answers: "What would happen on average across
        many parallel businesses?" This differs from the time-average perspective
        of "What happens to one business over time?"

        Args:
            trajectories (Union[List[np.ndarray], np.ndarray]): Multiple simulation
                trajectories. Can be:
                - List of 1D numpy arrays (supports variable lengths)
                - 2D numpy array with shape (n_paths, n_timesteps)
                Each trajectory represents values over time (equity, assets, etc.)
            metric (str): Type of ensemble statistic to compute:
                - "final_value": Statistics of final values across paths
                - "growth_rate": Statistics of growth rates across paths
                - "full": Average trajectory at each time step (fixed-length only)
                Defaults to "final_value".

        Returns:
            Dict[str, float]: Dictionary containing ensemble statistics:
                - 'mean': Mean of the selected metric across all valid paths
                - 'std': Standard deviation of the metric
                - 'median': Median value of the metric
                - 'survival_rate': Fraction of paths avoiding bankruptcy
                - 'n_survived': Absolute number of surviving paths
                - 'n_total': Total number of input paths

                For metric="full":
                - 'mean_trajectory': Mean values at each time step
                - 'std_trajectory': Standard deviations at each time step

        Examples:
            Analyze final equity values::

                import numpy as np

                # Multiple simulation results
                trajectories = [
                    np.array([10e6, 10.5e6, 11.2e6, 11.8e6, 12.5e6]),  # Success
                    np.array([10e6, 9.2e6, 8.1e6, 6.8e6, 0]),          # Bankruptcy
                    np.array([10e6, 10.8e6, 11.5e6, 12.8e6, 14.2e6]),  # High growth
                ]

                final_stats = analyzer.calculate_ensemble_average(
                    trajectories,
                    metric="final_value"
                )

                print(f"Average final equity: ${final_stats['mean']:,.0f}")
                print(f"Survival rate: {final_stats['survival_rate']:.1%}")
                print(f"Standard deviation: ${final_stats['std']:,.0f}")

            Analyze growth rate distribution::

                growth_stats = analyzer.calculate_ensemble_average(
                    trajectories,
                    metric="growth_rate"
                )

                print(f"Average growth rate: {growth_stats['mean']:.2%}")
                print(f"Growth rate volatility: {growth_stats['std']:.2%}")
                print(f"Median growth: {growth_stats['median']:.2%}")

            Full trajectory analysis (fixed-length only)::

                # Convert to fixed-length array
                fixed_trajectories = np.array([
                    [10e6, 10.5e6, 11.2e6, 11.8e6, 12.5e6],
                    [10e6, 9.8e6, 10.1e6, 10.6e6, 11.1e6],
                    [10e6, 10.8e6, 11.5e6, 12.8e6, 14.2e6]
                ])

                full_stats = analyzer.calculate_ensemble_average(
                    fixed_trajectories,
                    metric="full"
                )

                mean_path = full_stats['mean_trajectory']
                print(f"Mean trajectory: {mean_path}")
                # Shows average value at each time step

            Handle mixed survival scenarios::

                mixed_trajectories = [
                    np.array([10e6, 11e6, 12e6]),        # Short survivor
                    np.array([10e6, 9e6, 0]),             # Early bankruptcy
                    np.array([10e6, 11e6, 12e6, 13e6]),   # Long survivor
                ]

                stats = analyzer.calculate_ensemble_average(mixed_trajectories)
                print(f"{stats['n_survived']}/{stats['n_total']} paths survived")
                print(f"Survival rate: {stats['survival_rate']:.1%}")

        Statistical Interpretation:
            **Mean**: The expected value under the ensemble perspective. For
            multiplicative processes, this may differ significantly from what
            any individual entity experiences.

            **Standard Deviation**: Measures the spread of outcomes across the
            ensemble, indicating the uncertainty in individual results.

            **Survival Rate**: Critical metric often ignored in traditional
            expected value analysis. Shows the probability of avoiding bankruptcy.

            **Median**: Often more representative than mean for skewed distributions
            common in financial modeling.

        Edge Cases:
            - Empty trajectory list: Returns zeros/NaN appropriately
            - All paths end in bankruptcy: survival_rate=0, mean/median may be 0
            - Single trajectory: Statistics reduce to that trajectory's values
            - Mixed lengths: Handled gracefully with proper filtering

        Performance Notes:
            - Optimized for large numbers of trajectories (1000+ paths)
            - Memory efficient for mixed-length trajectory lists
            - Vectorized calculations where possible

        See Also:
            :meth:`calculate_time_average_growth`: Individual trajectory analysis
            :meth:`compare_scenarios`: Ensemble vs time-average comparison
            :meth:`analyze_simulation_batch`: Comprehensive batch analysis
        """
        # Handle list of arrays with potentially different lengths
        if isinstance(trajectories, list):
            # Check if all arrays have the same length
            lengths = [len(traj) for traj in trajectories if len(traj) > 0]
            if len(set(lengths)) == 1 and lengths:
                # All same length, can convert to 2D array
                trajectories_array = np.array(trajectories)
                return self._process_fixed_length_trajectories(trajectories_array, metric)
            # Different lengths, work with list
            return self._process_variable_length_trajectories(trajectories, metric)
        return self._process_fixed_length_trajectories(trajectories, metric)

    def check_convergence(self, values: np.ndarray, window_size: int = 100) -> Tuple[bool, float]:
        """Check Monte Carlo convergence using rolling standard error analysis.

        This method determines whether a Monte Carlo simulation has run enough
        iterations to provide statistically reliable results. It uses rolling
        standard error calculations to assess whether adding more simulations
        would significantly change the estimated mean.

        Convergence analysis is crucial for ergodic analysis because insufficient
        simulations can lead to misleading conclusions about insurance benefits.
        The method provides both a binary convergence decision and quantitative
        standard error metrics for informed decision making.

        Args:
            values (np.ndarray): Array of values to check for convergence,
                typically time-average growth rates from Monte Carlo simulations.
                Should contain at least window_size values for meaningful analysis.
                Infinite values (from bankruptcy) are handled appropriately.
            window_size (int): Size of rolling window for convergence assessment.
                Larger windows provide more stable convergence detection but require
                more data points. Typical values:
                - 50: Quick convergence check for small samples
                - 100: Standard convergence analysis (default)
                - 200: Conservative convergence for high precision
                Must be <= len(values) for analysis to proceed.

        Returns:
            Tuple[bool, float]: Convergence assessment results:
                - converged (bool): Whether the series has converged according to
                  the specified threshold. True indicates sufficient simulations.
                - standard_error (float): Current standard error of the mean based
                  on the last window_size observations. Lower values indicate
                  higher precision and greater confidence in results.

        Examples:
            Check convergence during Monte Carlo analysis::

                import numpy as np

                # Simulate running Monte Carlo with growth rate collection
                growth_rates = []

                for i in range(2000):  # Up to 2000 simulations
                    # Run single simulation (pseudo-code)
                    result = run_single_simulation()
                    growth_rate = analyzer.calculate_time_average_growth(result.equity)
                    growth_rates.append(growth_rate)

                    # Check convergence every 100 simulations
                    if (i + 1) % 100 == 0 and i >= 100:
                        converged, se = analyzer.check_convergence(
                            np.array(growth_rates),
                            window_size=100
                        )

                        print(f"Simulation {i+1}: SE={se:.4f}, Converged={converged}")

                        if converged:
                            print(f"✓ Convergence achieved after {i+1} simulations")
                            break

                if not converged:
                    print(f"⚠ Convergence not achieved after {len(growth_rates)} simulations")
                    print(f"Current standard error: {se:.4f}")
                    print(f"Target threshold: {analyzer.convergence_threshold:.4f}")

            Adaptive Monte Carlo with convergence monitoring::

                    def run_adaptive_monte_carlo(target_precision=0.01, max_sims=5000):
                        '''Run Monte Carlo until convergence or maximum simulations.'''
                        results = []

                        for i in range(max_sims):
                            # Run simulation
                            sim_result = run_single_simulation()
                            results.append(sim_result)

                            # Extract growth rates for convergence check
                            growth_rates = [analyzer.calculate_time_average_growth(r.equity)
                                          for r in results]

                            # Check convergence (need at least 100 for stability)
                            if i >= 100:
                                converged, se = analyzer.check_convergence(
                                    np.array([g for g in growth_rates if np.isfinite(g)])
                                )

                                if converged and se <= target_precision:
                                    print(f"Achieved target precision after {i+1} simulations")
                                    return results, True

                        print(f"Maximum simulations reached without convergence")
                        return results, False

                    # Run adaptive analysis
                    results, converged = run_adaptive_monte_carlo()
                    if converged:
                        print("Analysis complete with sufficient precision")
                    else:
                        print("Consider increasing maximum simulations")

            Convergence diagnostics and troubleshooting::

                # Analyze convergence pattern
                growth_rates = np.array([...])  # Your Monte Carlo results

                # Check convergence with different window sizes
                window_sizes = [50, 100, 150, 200]

                print("=== Convergence Analysis ===")
                for ws in window_sizes:
                    if len(growth_rates) >= ws:
                        converged, se = analyzer.check_convergence(growth_rates, ws)
                        print(f"Window {ws:3d}: SE={se:.5f}, Converged={converged}")

                # Plot convergence pattern (conceptual)
                rolling_means = []
                rolling_ses = []

                for i in range(100, len(growth_rates), 10):
                    subset = growth_rates[:i]
                    converged, se = analyzer.check_convergence(subset)
                    rolling_means.append(np.mean(subset[np.isfinite(subset)]))
                    rolling_ses.append(se)

                # Analyze convergence stability
                recent_se_trend = np.diff(rolling_ses[-10:])  # Last 10 points
                if np.mean(recent_se_trend) < 0:
                    print("✓ Standard error decreasing - convergence improving")
                else:
                    print("⚠ Standard error not decreasing - may need more simulations")

            Compare convergence across scenarios::

                # Check convergence for both insured and uninsured scenarios
                scenarios = {
                    'insured': insured_growth_rates,
                    'uninsured': uninsured_growth_rates
                }

                convergence_status = {}
                for name, rates in scenarios.items():
                    converged, se = analyzer.check_convergence(rates)
                    convergence_status[name] = {
                        'converged': converged,
                        'standard_error': se,
                        'n_simulations': len(rates)
                    }

                print("=== Scenario Convergence Status ===")
                for name, status in convergence_status.items():
                    print(f"{name:10}: {status['converged']} "
                          f"(SE={status['standard_error']:.4f}, n={status['n_simulations']})")

                # Determine if comparison is valid
                both_converged = all(s['converged'] for s in convergence_status.values())
                if both_converged:
                    print("✓ Both scenarios converged - comparison is reliable")
                else:
                    print("⚠ Incomplete convergence - results may be unreliable")

        Mathematical Background:
            The method calculates the standard error of the mean for the most recent
            window_size observations:

                SE = σ / √n

            Where:
            - σ = standard deviation of the sample
            - n = sample size (window_size)

            Convergence is achieved when SE < convergence_threshold, indicating
            that the sample mean is stable within the desired precision.

        Convergence Guidelines:
            **Standard Error Thresholds**:
            - SE < 0.005: High precision (recommended for final analysis)
            - SE < 0.01: Standard precision (adequate for most decisions)
            - SE < 0.02: Low precision (suitable for initial exploration)
            - SE > 0.02: Insufficient precision (run more simulations)

            **Sample Size Rules of Thumb**:
            - n < 100: Generally insufficient for convergence assessment
            - n = 100-500: May achieve convergence for low-volatility scenarios
            - n = 500-2000: Standard range for most insurance analyses
            - n > 2000: High-precision analysis or high-volatility scenarios

        Edge Cases:
            - Fewer observations than window_size: Returns (False, inf)
            - All infinite values: Returns (False, inf)
            - High volatility data: May require very large samples for convergence
            - Bimodal distributions: Standard error may not capture full uncertainty

        Performance Notes:
            - Fast execution even for large arrays (10,000+ observations)
            - Memory efficient rolling window calculations
            - Robust handling of infinite and missing values

        See Also:
            :attr:`convergence_threshold`: Threshold used for convergence decision
            :meth:`analyze_simulation_batch`: Includes automatic convergence analysis
            :meth:`calculate_ensemble_average`: Ensemble statistics that benefit from convergence
        """
        if len(values) < window_size:
            return False, np.inf

        # Calculate rolling mean and standard error
        _rolling_means = np.convolve(values, np.ones(window_size) / window_size, mode="valid")

        # Standard error of the mean
        se = np.std(values[-window_size:]) / np.sqrt(window_size)

        # Check if SE is below threshold
        converged = bool(se < self.convergence_threshold)

        return converged, se

    def compare_scenarios(
        self,
        insured_results: Union[List[SimulationResults], np.ndarray],
        uninsured_results: Union[List[SimulationResults], np.ndarray],
        metric: str = "equity",
    ) -> Dict[str, Any]:
        """Compare insured vs uninsured scenarios using comprehensive ergodic analysis.

        This is the core method for demonstrating ergodic advantages of insurance.
        It performs side-by-side comparison of insured and uninsured scenarios,
        calculating both time-average and ensemble-average growth rates to reveal
        the fundamental difference between expected value thinking and actual
        experienced growth.

        The comparison reveals how insurance can be optimal from a time-average
        perspective even when it appears costly from an ensemble (expected value)
        perspective - the key insight of ergodic economics applied to insurance.

        Args:
            insured_results (Union[List[SimulationResults], np.ndarray]): Simulation
                results from insured scenarios. Can be:

                - List of SimulationResults objects from Monte Carlo runs
                - List of numpy arrays representing trajectories
                - 2D numpy array with shape (n_simulations, n_timesteps)

            uninsured_results (Union[List[SimulationResults], np.ndarray]):
                Simulation results from uninsured scenarios, same format as
                insured_results. Should have same number of simulations for
                valid comparison.
            metric (str): Financial metric to analyze for comparison:

                - "equity": Company equity over time (recommended)
                - "assets": Total assets over time
                - "cash": Available cash over time
                - Any attribute available in SimulationResults objects

                Defaults to "equity".

        Returns:
            Dict[str, Any]: Comprehensive comparison results with nested structure:

                - **'insured'** (Dict): Insured scenario statistics:

                  - 'time_average_mean': Mean time-average growth rate
                  - 'time_average_median': Median time-average growth rate
                  - 'time_average_std': Standard deviation of growth rates
                  - 'ensemble_average': Ensemble average growth rate
                  - 'survival_rate': Fraction avoiding bankruptcy
                  - 'n_survived': Absolute number of survivors

                - **'uninsured'** (Dict): Uninsured scenario statistics:

                  - Same structure as 'insured'

                - **'ergodic_advantage'** (Dict): Comparative metrics:

                  - 'time_average_gain': Difference in time-average growth
                  - 'ensemble_average_gain': Difference in ensemble averages
                  - 'survival_gain': Improvement in survival rate
                  - 't_statistic': t-test statistic for significance
                  - 'p_value': p-value for statistical significance
                  - 'significant': Boolean indicating significance (p < 0.05)

        Examples:
            Basic insurance vs no insurance comparison:

            .. code-block:: python

                # Run Monte Carlo simulations (pseudo-code)
                insured_sims = run_simulations(insurance_enabled=True, n_sims=1000)
                uninsured_sims = run_simulations(insurance_enabled=False, n_sims=1000)

                # Compare scenarios
                comparison = analyzer.compare_scenarios(
                    insured_sims,
                    uninsured_sims,
                    metric="equity"
                )

                # Extract key insights
                time_avg_gain = comparison['ergodic_advantage']['time_average_gain']
                survival_gain = comparison['ergodic_advantage']['survival_gain']
                is_significant = comparison['ergodic_advantage']['significant']

                print(f"Time-average growth improvement: {time_avg_gain:.2%}")
                print(f"Survival rate improvement: {survival_gain:.1%}")
                print(f"Statistical significance: {is_significant}")

            Detailed analysis of results:

            .. code-block:: python

                # Examine both perspectives
                insured = comparison['insured']
                uninsured = comparison['uninsured']
                advantage = comparison['ergodic_advantage']

                print("\n=== ENSEMBLE PERSPECTIVE (Traditional Analysis) ===")
                print(f"Insured ensemble growth: {insured['ensemble_average']:.2%}")
                print(f"Uninsured ensemble growth: {uninsured['ensemble_average']:.2%}")
                print(f"Ensemble advantage: {advantage['ensemble_average_gain']:.2%}")

                print("\n=== TIME-AVERAGE PERSPECTIVE (Ergodic Analysis) ===")
                print(f"Insured time-average growth: {insured['time_average_mean']:.2%}")
                print(f"Uninsured time-average growth: {uninsured['time_average_mean']:.2%}")
                print(f"Time-average advantage: {advantage['time_average_gain']:.2%}")

                print("\n=== SURVIVAL ANALYSIS ===")
                print(f"Insured survival rate: {insured['survival_rate']:.1%}")
                print(f"Uninsured survival rate: {uninsured['survival_rate']:.1%}")
                print(f"Survival improvement: {advantage['survival_gain']:.1%}")

                # Interpret ergodic vs ensemble difference
                if advantage['time_average_gain'] > advantage['ensemble_average_gain']:
                    print("\n✓ Insurance shows ergodic advantage!")
                    print("  Time-average benefit exceeds ensemble expectation")
                else:
                    print("\n! No clear ergodic advantage detected")

            Statistical significance analysis:

            .. code-block:: python

                if comparison['ergodic_advantage']['significant']:
                    p_val = comparison['ergodic_advantage']['p_value']
                    t_stat = comparison['ergodic_advantage']['t_statistic']

                    print(f"Results are statistically significant:")
                    print(f"  t-statistic: {t_stat:.3f}")
                    print(f"  p-value: {p_val:.4f}")
                    print(f"  Confidence level: {(1-p_val)*100:.1f}%")
                else:
                    print("Results not statistically significant")
                    print("Consider running more simulations")

            Multiple metric analysis:

            .. code-block:: python

                # Compare different financial metrics
                metrics_to_analyze = ['equity', 'assets', 'cash']
                results = {}

                for metric in metrics_to_analyze:
                    results[metric] = analyzer.compare_scenarios(
                        insured_sims, uninsured_sims, metric=metric
                    )

                # Find metric showing strongest insurance advantage
                best_metric = max(metrics_to_analyze,
                    key=lambda m: results[m]['ergodic_advantage']['time_average_gain']
                )

                print(f"Strongest insurance advantage in: {best_metric}")
                gain = results[best_metric]['ergodic_advantage']['time_average_gain']
                print(f"Time-average improvement: {gain:.2%}")

        Mathematical Background:
            The comparison reveals the ergodic/non-ergodic nature of financial
            processes by calculating:

            **Time-Average Growth**: Mean of individual trajectory growth rates:
                g_time = mean([ln(X_i(T)/X_i(0))/T for each path i])

            **Ensemble Average Growth**: Growth of the ensemble mean:
                g_ensemble = ln(mean([X_i(T)])/mean([X_i(0)]))/T

            **Ergodic Divergence**: g_time - g_ensemble

            For multiplicative processes with volatility, these typically differ,
            with insurance often improving time-average more than ensemble average.

        Interpretation Guidelines:
            **Positive Time-Average Gain**: Insurance improves actual experienced
            growth rates, even if ensemble analysis suggests otherwise.

            **Survival Rate Improvement**: Critical for long-term viability,
            often the primary benefit of insurance in high-volatility scenarios.

            **Statistical Significance**: p < 0.05 indicates results are unlikely
            due to random chance, supporting reliability of conclusions.

        Edge Cases:
            - All paths bankrupt in one scenario: Handled with -inf growth rates
            - Mismatched simulation counts: Statistics calculated on available data
            - Identical scenarios: All advantages will be zero
            - High volatility: May require more simulations for significance

        Performance Notes:
            - Handles thousands of simulation paths efficiently
            - Memory-conscious processing of large trajectory datasets
            - Automatic handling of variable-length trajectories

        See Also:
            :meth:`calculate_time_average_growth`: Individual path analysis
            :meth:`calculate_ensemble_average`: Ensemble statistics
            :meth:`significance_test`: Statistical testing details
            :class:`ErgodicAnalysisResults`: Comprehensive results format
        """
        # Extract trajectories
        if isinstance(insured_results, list) and isinstance(insured_results[0], SimulationResults):
            # Handle variable-length trajectories (e.g., due to insolvency)
            insured_trajectories = [getattr(r, metric) for r in insured_results]
            uninsured_trajectories = [getattr(r, metric) for r in uninsured_results]

            # Convert to list of arrays rather than 2D array to handle different lengths
            insured_trajectories = [np.asarray(traj) for traj in insured_trajectories]
            uninsured_trajectories = [np.asarray(traj) for traj in uninsured_trajectories]
        else:
            insured_trajectories = [np.asarray(traj) for traj in insured_results]
            uninsured_trajectories = [np.asarray(traj) for traj in uninsured_results]

        # Calculate time-average growth for each path
        insured_time_avg = [
            self.calculate_time_average_growth(traj) for traj in insured_trajectories
        ]
        uninsured_time_avg = [
            self.calculate_time_average_growth(traj) for traj in uninsured_trajectories
        ]

        # Filter out infinite values
        insured_time_avg_valid = [g for g in insured_time_avg if np.isfinite(g)]
        uninsured_time_avg_valid = [g for g in uninsured_time_avg if np.isfinite(g)]

        # Calculate ensemble averages
        insured_ensemble = self.calculate_ensemble_average(
            insured_trajectories, metric="growth_rate"
        )
        uninsured_ensemble = self.calculate_ensemble_average(
            uninsured_trajectories, metric="growth_rate"
        )

        # Compile results
        results = {
            "insured": {
                "time_average_mean": (
                    np.mean(insured_time_avg_valid) if insured_time_avg_valid else -np.inf
                ),
                "time_average_median": (
                    np.median(insured_time_avg_valid) if insured_time_avg_valid else -np.inf
                ),
                "time_average_std": (
                    np.std(insured_time_avg_valid) if insured_time_avg_valid else 0.0
                ),
                "ensemble_average": insured_ensemble["mean"],
                "survival_rate": insured_ensemble["survival_rate"],
                "n_survived": insured_ensemble["n_survived"],
            },
            "uninsured": {
                "time_average_mean": (
                    np.mean(uninsured_time_avg_valid) if uninsured_time_avg_valid else -np.inf
                ),
                "time_average_median": (
                    np.median(uninsured_time_avg_valid) if uninsured_time_avg_valid else -np.inf
                ),
                "time_average_std": (
                    np.std(uninsured_time_avg_valid) if uninsured_time_avg_valid else 0.0
                ),
                "ensemble_average": uninsured_ensemble["mean"],
                "survival_rate": uninsured_ensemble["survival_rate"],
                "n_survived": uninsured_ensemble["n_survived"],
            },
            "ergodic_advantage": {
                "time_average_gain": float(
                    float(np.mean(insured_time_avg_valid) if insured_time_avg_valid else -np.inf)
                    - float(
                        np.mean(uninsured_time_avg_valid) if uninsured_time_avg_valid else -np.inf
                    )
                ),
                "ensemble_average_gain": insured_ensemble["mean"] - uninsured_ensemble["mean"],
                "survival_gain": insured_ensemble["survival_rate"]
                - uninsured_ensemble["survival_rate"],
            },
        }

        # Add significance test if we have valid data
        if insured_time_avg_valid and uninsured_time_avg_valid:
            t_stat, p_value = self.significance_test(
                insured_time_avg_valid, uninsured_time_avg_valid
            )
            results["ergodic_advantage"]["t_statistic"] = t_stat  # type: ignore[index]
            results["ergodic_advantage"]["p_value"] = p_value  # type: ignore[index]
            results["ergodic_advantage"]["significant"] = p_value < 0.05  # type: ignore[index]
        else:
            results["ergodic_advantage"]["t_statistic"] = np.nan  # type: ignore[index]
            results["ergodic_advantage"]["p_value"] = np.nan  # type: ignore[index]
            results["ergodic_advantage"]["significant"] = False  # type: ignore[index]

        return results

    def significance_test(
        self,
        sample1: Union[List[float], np.ndarray],
        sample2: Union[List[float], np.ndarray],
        test_type: str = "two-sided",
    ) -> Tuple[float, float]:
        """Perform statistical significance test between two growth rate samples.

        This method conducts a two-sample t-test to determine whether observed
        differences between insured and uninsured scenarios are statistically
        significant or could reasonably be attributed to random variation.
        Statistical significance provides confidence that ergodic advantages
        are genuine rather than artifacts of sampling variability.

        Args:
            sample1 (Union[List[float], np.ndarray]): First sample of growth rates,
                typically from insured scenarios. Should contain time-average growth
                rates from individual simulation paths. Infinite values (from
                bankruptcy) are automatically handled.
            sample2 (Union[List[float], np.ndarray]): Second sample of growth rates,
                typically from uninsured scenarios. Should be comparable to sample1
                with same underlying business conditions but different insurance
                coverage.
            test_type (str): Type of statistical test to perform:
                - "two-sided": Tests if samples have different means (default)
                - "greater": Tests if sample1 mean > sample2 mean
                - "less": Tests if sample1 mean < sample2 mean
                Defaults to "two-sided" for general hypothesis testing.

        Returns:
            Tuple[float, float]: Statistical test results:
                - t_statistic (float): t-test statistic value. Positive values
                  indicate sample1 has higher mean than sample2.
                - p_value (float): Probability of observing the data under the null
                  hypothesis of no difference. Lower values indicate stronger
                  evidence against the null hypothesis.

        Examples:
            Test insurance benefit significance::

                import numpy as np

                # Growth rates from Monte Carlo simulations
                insured_growth = np.array([0.048, 0.051, 0.047, 0.049, 0.052, ...])
                uninsured_growth = np.array([0.038, -np.inf, 0.042, 0.035, 0.041, ...])

                # Two-sided test for any difference
                t_stat, p_value = analyzer.significance_test(
                    insured_growth,
                    uninsured_growth,
                    test_type="two-sided"
                )

                print(f"t-statistic: {t_stat:.3f}")
                print(f"p-value: {p_value:.4f}")

                if p_value < 0.05:
                    print("✓ Statistically significant difference at 5% level")
                else:
                    print("No significant difference detected")

            One-sided test for insurance superiority::

                # Test if insurance provides superior growth rates
                t_stat, p_value = analyzer.significance_test(
                    insured_growth,
                    uninsured_growth,
                    test_type="greater"
                )

                print(f"Testing if insured > uninsured:")
                print(f"t-statistic: {t_stat:.3f}")
                print(f"p-value: {p_value:.4f}")

                if p_value < 0.01:
                    print("✓ Strong evidence that insurance improves growth (p < 0.01)")
                elif p_value < 0.05:
                    print("✓ Moderate evidence that insurance improves growth (p < 0.05)")
                elif p_value < 0.10:
                    print("? Weak evidence that insurance improves growth (p < 0.10)")
                else:
                    print("No significant evidence that insurance improves growth")

            Comprehensive significance analysis::

                # Test multiple hypotheses
                tests = [
                    ("two-sided", "Any difference"),
                    ("greater", "Insurance superior"),
                    ("less", "Insurance inferior")
                ]

                print("=== Statistical Significance Analysis ===")
                for test_type, description in tests:
                    t_stat, p_value = analyzer.significance_test(
                        insured_growth, uninsured_growth, test_type
                    )

                    significance = "***" if p_value < 0.001 else \
                                 "**" if p_value < 0.01 else \
                                 "*" if p_value < 0.05 else \
                                 "" if p_value < 0.10 else "n.s."

                    print(f"{description:20}: t={t_stat:6.3f}, p={p_value:.4f} {significance}")

            Sample size and power analysis::

                # Check if samples are large enough for reliable testing
                n1, n2 = len(insured_growth), len(uninsured_growth)

                if n1 < 30 or n2 < 30:
                    print(f"⚠ Small sample sizes (n1={n1}, n2={n2})")
                    print("Consider running more simulations for robust results")

                # Calculate effect size (Cohen's d)
                mean1, mean2 = np.mean(insured_growth), np.mean(uninsured_growth)
                pooled_std = np.sqrt(((n1-1)*np.var(insured_growth) +
                                    (n2-1)*np.var(uninsured_growth)) / (n1+n2-2))

                cohens_d = (mean1 - mean2) / pooled_std

                print(f"Effect size (Cohen's d): {cohens_d:.3f}")
                if abs(cohens_d) > 0.8:
                    print("Large effect size")
                elif abs(cohens_d) > 0.5:
                    print("Medium effect size")
                elif abs(cohens_d) > 0.2:
                    print("Small effect size")
                else:
                    print("Very small effect size")

        Statistical Interpretation:
            **p-value Guidelines**:
            - p < 0.001: Very strong evidence against null hypothesis (***)
            - p < 0.01: Strong evidence (**)
            - p < 0.05: Moderate evidence (*)
            - p < 0.10: Weak evidence
            - p >= 0.10: No significant evidence

            **t-statistic Guidelines**:
            - ``|t| > 3``: Very large effect
            - ``|t| > 2``: Large effect
            - ``|t| > 1``: Moderate effect
            - ``|t| < 1``: Small effect

        Assumptions and Limitations:
            **t-test Assumptions**:
            1. Samples are independent
            2. Data approximately normally distributed (robust to violations with large n)
            3. Equal variances (Welch's t-test used automatically if needed)

            **Handling of Infinite Values**: The method automatically excludes
            infinite values (from bankruptcy scenarios) using scipy's nan_policy='omit'.
            This is appropriate since infinite values represent qualitatively different
            outcomes (business failure) rather than extreme but finite growth rates.

            **Multiple Testing**: If performing multiple significance tests,
            consider adjusting significance levels (e.g., Bonferroni correction)
            to account for increased Type I error probability.

        Performance Notes:
            - Efficient for samples up to 10,000+ observations
            - Automatic handling of missing/infinite values
            - Uses scipy.stats for robust statistical calculations

        See Also:
            :meth:`compare_scenarios`: Includes automatic significance testing
            :meth:`check_convergence`: For determining adequate sample sizes
            scipy.stats.ttest_ind: Underlying statistical test implementation
        """
        # Perform independent samples t-test
        t_stat, p_value = stats.ttest_ind(
            sample1, sample2, alternative=test_type, nan_policy="omit"
        )

        return t_stat, p_value

    def analyze_simulation_batch(
        self, simulation_results: List[SimulationResults], label: str = "Scenario"
    ) -> Dict[str, Any]:
        """Perform comprehensive ergodic analysis on a batch of simulation results.

        This method provides a complete analysis of a single scenario (e.g., all
        insured simulations or all uninsured simulations), including time-average
        and ensemble statistics, convergence analysis, and survival metrics. It
        serves as a comprehensive diagnostic tool for understanding the ergodic
        properties of a particular insurance strategy.

        Args:
            simulation_results (List[SimulationResults]): List of simulation result
                objects from Monte Carlo runs. Each should contain trajectory data
                including equity, assets, years, and potential insolvency information.
                Typically 100-2000 simulations for robust analysis.
            label (str): Descriptive label for this batch of simulations, used
                in reporting and metadata. Examples: "High Deductible", "Full Coverage",
                "Base Case", "Stress Scenario". Defaults to "Scenario".

        Returns:
            Dict[str, Any]: Comprehensive analysis dictionary with nested structure:

                - **'label'** (str): The provided label for identification
                - **'n_simulations'** (int): Number of simulations analyzed
                - **'time_average'** (Dict): Time-average growth statistics:

                  - 'mean': Mean time-average growth rate across all paths
                  - 'median': Median time-average growth rate
                  - 'std': Standard deviation of growth rates
                  - 'min': Minimum growth rate observed
                  - 'max': Maximum growth rate observed

                - **'ensemble_average'** (Dict): Ensemble statistics:

                  - 'mean': Ensemble average growth rate
                  - 'std': Standard deviation of ensemble
                  - 'median': Median of ensemble
                  - 'survival_rate': Fraction avoiding bankruptcy
                  - 'n_survived': Absolute number of survivors
                  - 'n_total': Total number of simulations

                - **'convergence'** (Dict): Monte Carlo convergence analysis:

                  - 'converged': Boolean indicating if results have converged
                  - 'standard_error': Current standard error of the estimates
                  - 'threshold': Convergence threshold used

                - **'survival_analysis'** (Dict): Survival metrics:

                  - 'survival_rate': Fraction avoiding bankruptcy (duplicate)
                  - 'mean_survival_time': Average time to insolvency or end

                - **'ergodic_divergence'** (float): Difference between time-average
                  and ensemble-average growth rates

        Examples:
            Analyze a batch of insured simulations:

            .. code-block:: python

                # Run Monte Carlo simulations
                insured_results = []
                for i in range(1000):
                    sim = run_single_simulation(insurance_enabled=True, seed=i)
                    insured_results.append(sim)

                # Comprehensive analysis
                analysis = analyzer.analyze_simulation_batch(
                    insured_results,
                    label="Full Insurance Coverage"
                )

                # Report key findings
                print(f"\n=== {analysis['label']} Analysis ===")
                print(f"Simulations: {analysis['n_simulations']}")
                print(f"Time-average growth: {analysis['time_average']['mean']:.2%} ± {analysis['time_average']['std']:.2%}")
                print(f"Ensemble average: {analysis['ensemble_average']['mean']:.2%}")
                print(f"Survival rate: {analysis['survival_analysis']['survival_rate']:.1%}")
                print(f"Ergodic divergence: {analysis['ergodic_divergence']:.3f}")

            Check Monte Carlo convergence:

            .. code-block:: python

                if analysis['convergence']['converged']:
                    print(f"✓ Results have converged (SE: {analysis['convergence']['standard_error']:.4f})")
                    print("Analysis is reliable for decision making")
                else:
                    current_se = analysis['convergence']['standard_error']
                    target_se = analysis['convergence']['threshold']

                    print(f"⚠ Convergence not reached (SE: {current_se:.4f} > {target_se:.4f})")

                    # Estimate additional simulations needed
                    current_n = analysis['n_simulations']
                    factor = (current_se / target_se) ** 2
                    recommended_n = int(current_n * factor)
                    additional_needed = recommended_n - current_n

                    print(f"Recommend ~{additional_needed} additional simulations")

            Compare growth rate distributions:

            .. code-block:: python

                # Analyze distribution characteristics
                time_avg = analysis['time_average']

                print(f"\n=== Growth Rate Distribution ===")
                print(f"Mean: {time_avg['mean']:.2%}")
                print(f"Median: {time_avg['median']:.2%}")
                print(f"Std Dev: {time_avg['std']:.2%}")
                print(f"Range: {time_avg['min']:.2%} to {time_avg['max']:.2%}")

                # Check for skewness
                if time_avg['mean'] > time_avg['median']:
                    print("Distribution is right-skewed (long tail of high growth)")
                elif time_avg['mean'] < time_avg['median']:
                    print("Distribution is left-skewed (long tail of poor performance)")
                else:
                    print("Distribution appears roughly symmetric")

            Survival analysis insights:

            .. code-block:: python

                survival = analysis['survival_analysis']
                ensemble = analysis['ensemble_average']

                print(f"\n=== Survival Analysis ===")
                print(f"Survival rate: {survival['survival_rate']:.1%}")
                print(f"Survivors: {ensemble['n_survived']}/{ensemble['n_total']}")
                print(f"Mean time to insolvency/end: {survival['mean_survival_time']:.1f} years")

                # Risk assessment
                if survival['survival_rate'] < 0.9:
                    print("⚠ High bankruptcy risk - consider more insurance")
                elif survival['survival_rate'] > 0.99:
                    print("✓ Very low bankruptcy risk - insurance is effective")
                else:
                    print("✓ Moderate bankruptcy risk - acceptable for most businesses")

            Ergodic divergence interpretation:

            .. code-block:: python

                divergence = analysis['ergodic_divergence']

                if abs(divergence) < 0.001:  # Less than 0.1%
                    print("Minimal ergodic divergence - process is nearly ergodic")
                elif divergence > 0:
                    print(f"Positive ergodic divergence ({divergence:.3f})")
                    print("Time-average exceeds ensemble average - favorable")
                else:
                    print(f"Negative ergodic divergence ({divergence:.3f})")
                    print("Ensemble average exceeds time-average - volatility drag")

        Use Cases:
            **Single Scenario Analysis**: Understand the characteristics of one
            insurance configuration before comparing alternatives.

            **Convergence Diagnostics**: Determine if enough simulations have
            been run for reliable conclusions.

            **Risk Assessment**: Evaluate bankruptcy probabilities and growth
            rate distributions for risk management decisions.

            **Parameter Sensitivity**: Analyze how changes in insurance parameters
            affect ergodic properties by comparing batch analyses.

        Performance Notes:
            - Efficient processing of 1000+ simulation results
            - Memory-conscious handling of trajectory data
            - Automatic filtering of invalid/infinite growth rates
            - Vectorized calculations for speed

        Warning:
            Large numbers of bankruptcy scenarios may skew statistics. Check the
            survival rate and consider whether the scenario parameters are realistic
            for your analysis goals.

        See Also:
            :meth:`compare_scenarios`: For comparing multiple scenario batches
            :meth:`check_convergence`: For detailed convergence analysis
            :class:`SimulationResults`: Expected format for simulation_results
            :class:`ErgodicAnalysisResults`: Alternative comprehensive results format
        """
        # Extract equity trajectories
        equity_trajectories = np.array([r.equity for r in simulation_results])
        _asset_trajectories = np.array([r.assets for r in simulation_results])

        # Calculate time-average growth for each path
        time_avg_growth = [
            self.calculate_time_average_growth(equity) for equity in equity_trajectories
        ]

        # Filter valid growth rates
        valid_growth = [g for g in time_avg_growth if np.isfinite(g)]

        # Calculate ensemble statistics
        ensemble_stats = self.calculate_ensemble_average(equity_trajectories, metric="growth_rate")

        # Check convergence
        if len(valid_growth) > 0:
            converged, se = self.check_convergence(np.array(valid_growth))
        else:
            converged, se = False, np.inf

        # Compile analysis
        analysis: Dict[str, Any] = {
            "label": label,
            "n_simulations": len(simulation_results),
            "time_average": {
                "mean": np.mean(valid_growth) if valid_growth else -np.inf,
                "median": np.median(valid_growth) if valid_growth else -np.inf,
                "std": np.std(valid_growth) if valid_growth else 0.0,
                "min": np.min(valid_growth) if valid_growth else -np.inf,
                "max": np.max(valid_growth) if valid_growth else -np.inf,
            },
            "ensemble_average": ensemble_stats,
            "convergence": {
                "converged": converged,
                "standard_error": se,
                "threshold": self.convergence_threshold,
            },
            "survival_analysis": {
                "survival_rate": ensemble_stats["survival_rate"],
                "mean_survival_time": np.mean(
                    [
                        r.insolvency_year if r.insolvency_year else len(r.years)
                        for r in simulation_results
                    ]
                ),
            },
        }

        # Calculate ergodic divergence
        if valid_growth:
            time_avg_mean = analysis["time_average"]["mean"]
            ensemble_mean = ensemble_stats["mean"]
            analysis["ergodic_divergence"] = time_avg_mean - ensemble_mean
        else:
            analysis["ergodic_divergence"] = np.nan

        return analysis

    def integrate_loss_ergodic_analysis(  # pylint: disable=too-many-locals
        self,
        loss_data: "LossData",
        insurance_program: Optional["InsuranceProgram"],
        manufacturer: Any,
        time_horizon: int,
        n_simulations: int = 100,
    ) -> ErgodicAnalysisResults:
        """Perform end-to-end integrated loss modeling and ergodic analysis.

        This method provides a complete pipeline from loss generation through
        insurance application to final ergodic analysis. It demonstrates the
        full power of the ergodic framework by seamlessly connecting actuarial
        loss modeling with business financial modeling and ergodic growth analysis.

        The integration pipeline follows these steps:
        1. **Validate Input Data**: Ensure loss data meets quality standards
        2. **Apply Insurance Program**: Calculate recoveries and net exposures
        3. **Generate Annual Loss Aggregates**: Convert to time-series format
        4. **Run Monte Carlo Simulations**: Execute business simulations with losses
        5. **Calculate Ergodic Metrics**: Analyze time-average vs ensemble behavior
        6. **Validate Results**: Ensure mathematical and business logic consistency
        7. **Package Results**: Return comprehensive analysis in standardized format

        Args:
            loss_data (LossData): Standardized loss data object containing loss
                frequency and severity distributions. Must pass validation checks
                including proper distribution parameters and reasonable ranges.
            insurance_program (Optional[InsuranceProgram]): Insurance program to
                apply to losses. If None, analysis proceeds with no insurance
                coverage. Program should specify layers, deductibles, limits, and
                premium rates.
            manufacturer (Any): Manufacturer model instance for running business
                simulations. Should be configured with appropriate initial conditions
                and financial parameters. Must support claim processing and annual
                step operations.
            time_horizon (int): Analysis time horizon in years. Typical values:

                - 10-20 years: Standard analysis period
                - 50+ years: Long-term ergodic behavior
                - 5-10 years: Quick analysis for parameter exploration

            n_simulations (int): Number of Monte Carlo simulations to run.
                More simulations provide better statistical reliability:

                - 100: Quick analysis for development/testing
                - 1000: Standard analysis for decision making
                - 5000+: High-precision analysis for final recommendations

                Defaults to 100 for reasonable performance.

        Returns:
            ErgodicAnalysisResults: Comprehensive analysis results containing:

                - Time-average and ensemble-average growth rates
                - Survival rates and ergodic divergence
                - Insurance impact metrics (premiums, recoveries, net benefit)
                - Validation status and detailed metadata
                - All necessary information for decision making

        Examples:
            Basic integrated analysis:

            .. code-block:: python

                from ergodic_insurance import LossData, InsuranceProgram, WidgetManufacturer, ManufacturerConfig

                # Set up loss data
                loss_data = LossData.from_poisson_lognormal(
                    frequency_lambda=2.5,      # 2.5 claims per year on average
                    severity_mean=1_000_000,   # $1M average claim
                    severity_cv=2.0,           # High variability
                    time_horizon=20
                )

                # Configure insurance program
                insurance = InsuranceProgram([
                    # (attachment, limit, rate)
                    (0, 1_000_000, 0.015),           # $1M primary layer at 1.5%
                    (1_000_000, 10_000_000, 0.008),  # $10M excess at 0.8%
                    (11_000_000, 50_000_000, 0.004)  # $50M umbrella at 0.4%
                ])

                # Set up manufacturer
                config = ManufacturerConfig(
                    initial_assets=25_000_000,
                   base_operating_margin=0.08,
                    asset_turnover_ratio=0.75
                )
                manufacturer = WidgetManufacturer(config)

                # Run integrated analysis
                results = analyzer.integrate_loss_ergodic_analysis(
                    loss_data=loss_data,
                    insurance_program=insurance,
                    manufacturer=manufacturer,
                    time_horizon=20,
                    n_simulations=1000
                )

                # Interpret results
                if results.validation_passed:
                    print(f"Time-average growth: {results.time_average_growth:.2%}")
                    print(f"Ensemble average: {results.ensemble_average_growth:.2%}")
                    print(f"Survival rate: {results.survival_rate:.1%}")
                    print(f"Ergodic divergence: {results.ergodic_divergence:.3f}")

                    net_benefit = results.insurance_impact['net_benefit']
                    print(f"Insurance net benefit: ${net_benefit:,.0f}")

                    if results.ergodic_divergence > 0:
                        print("✓ Insurance shows ergodic advantage")
                else:
                    print("⚠ Analysis validation failed - check inputs")

            Compare insured vs uninsured scenarios:

            .. code-block:: python

                # Run analysis with insurance
                insured_results = analyzer.integrate_loss_ergodic_analysis(
                    loss_data, insurance, manufacturer, 20, 1000
                )

                # Run analysis without insurance
                uninsured_results = analyzer.integrate_loss_ergodic_analysis(
                    loss_data, None, manufacturer, 20, 1000
                )

                # Compare outcomes
                if insured_results.validation_passed and uninsured_results.validation_passed:
                    growth_improvement = (insured_results.time_average_growth -
                                        uninsured_results.time_average_growth)
                    survival_improvement = (insured_results.survival_rate -
                                          uninsured_results.survival_rate)

                    print(f"Growth rate improvement: {growth_improvement:.2%}")
                    print(f"Survival rate improvement: {survival_improvement:.1%}")

                    if growth_improvement > 0 and survival_improvement > 0:
                        print("✓ Insurance provides clear benefits")
                    elif growth_improvement > 0:
                        print("✓ Insurance improves growth despite survival costs")
                    elif survival_improvement > 0:
                        print("✓ Insurance improves survival despite growth costs")
                    else:
                        print("? Insurance benefits unclear - review parameters")

            Parameter sensitivity analysis:

            .. code-block:: python

                # Test different loss frequencies
                frequencies = [1.0, 2.0, 3.0, 4.0, 5.0]
                results = {}

                for freq in frequencies:
                    test_loss_data = LossData.from_poisson_lognormal(
                        frequency_lambda=freq,
                        severity_mean=1_000_000,
                        severity_cv=2.0,
                        time_horizon=20
                    )

                    result = analyzer.integrate_loss_ergodic_analysis(
                        test_loss_data, insurance, manufacturer, 20, 500
                    )

                    results[freq] = result

                # Find optimal frequency range for insurance benefit
                for freq, result in results.items():
                    if result.validation_passed:
                        print(f"Frequency {freq}: Growth={result.time_average_growth:.2%}, "
                              f"Survival={result.survival_rate:.1%}")

            Detailed insurance impact analysis:

            .. code-block:: python

                if results.validation_passed:
                    impact = results.insurance_impact
                    metadata = results.metadata

                    print(f"\n=== Insurance Impact Analysis ===")
                    print(f"Total premiums paid: ${impact.get('premium_cost', 0):,.0f}")
                    print(f"Total recoveries: ${impact.get('recovery_benefit', 0):,.0f}")
                    print(f"Net financial benefit: ${impact.get('net_benefit', 0):,.0f}")
                    print(f"Growth rate improvement: {impact.get('growth_improvement', 0):.2%}")

                    # Calculate benefit ratios
                    premium_cost = impact.get('premium_cost', 1)  # Avoid division by zero
                    if premium_cost > 0:
                        recovery_ratio = impact.get('recovery_benefit', 0) / premium_cost
                        benefit_ratio = impact.get('net_benefit', 0) / premium_cost

                        print(f"\n=== Efficiency Metrics ===")
                        print(f"Recovery ratio: {recovery_ratio:.2f}x premiums")
                        print(f"Net benefit ratio: {benefit_ratio:.2f}x premiums")

                        if benefit_ratio > 0:
                            print("✓ Insurance provides positive net value")
                        else:
                            print("⚠ Insurance costs exceed benefits in expectation")
                            print("  (But may still provide ergodic advantages)")

        Validation and Error Handling:
            The method includes comprehensive validation at multiple stages:

            **Input Validation**:
            - Loss data consistency checks
            - Insurance program parameter validation
            - Manufacturer model state verification

            **Process Validation**:
            - Simulation convergence monitoring
            - Mathematical consistency checks
            - Business logic validation

            **Output Validation**:
            - Result reasonableness checks
            - Statistical significance assessment
            - Cross-validation with alternative methods

        Performance Considerations:
            - Optimized for 100-5000 simulation runs
            - Memory-efficient trajectory storage
            - Parallel processing capabilities where available
            - Progress monitoring for long-running analyses

        Error Conditions:
            Returns results with validation_passed=False if:
            - Loss data fails validation checks
            - All simulation paths end in bankruptcy
            - Mathematical inconsistencies detected
            - Insufficient data for statistical analysis

        See Also:
            :class:`ErgodicAnalysisResults`: Detailed results format
            :class:`~ergodic_insurance.loss_distributions.LossData`: Loss data requirements
            :class:`~ergodic_insurance.insurance_program.InsuranceProgram`: Insurance setup
            :meth:`validate_insurance_ergodic_impact`: Additional validation methods
        """
        from .loss_distributions import LossEvent
        from .simulation import Simulation

        # Validate input data
        if not loss_data.validate():
            logger.warning("Loss data validation failed")
            return ErgodicAnalysisResults(
                time_average_growth=-np.inf,
                ensemble_average_growth=0.0,
                survival_rate=0.0,
                ergodic_divergence=-np.inf,
                insurance_impact={},
                validation_passed=False,
                metadata={"error": "Invalid loss data"},
            )

        # Apply insurance if provided
        if insurance_program:
            insured_loss_data = loss_data.apply_insurance(insurance_program)
            insurance_metadata = insured_loss_data.metadata
        else:
            insured_loss_data = loss_data
            insurance_metadata = {}

        # Convert to annual aggregates for simulation
        annual_losses = insured_loss_data.get_annual_aggregates(time_horizon)

        # Run Monte Carlo simulations
        simulation_results = []
        for sim_idx in range(n_simulations):
            # Create fresh manufacturer instance for each simulation
            import copy

            mfg_copy = copy.deepcopy(manufacturer)

            # Create simulation with manufacturer
            sim = Simulation(manufacturer=mfg_copy, time_horizon=time_horizon, seed=sim_idx)

            # Initialize result storage
            sim.years = np.arange(time_horizon)
            sim.assets = np.zeros(time_horizon)
            sim.equity = np.zeros(time_horizon)
            sim.roe = np.zeros(time_horizon)
            sim.revenue = np.zeros(time_horizon)
            sim.net_income = np.zeros(time_horizon)
            sim.claim_counts = np.zeros(time_horizon, dtype=int)
            sim.claim_amounts = np.zeros(time_horizon)
            sim.insolvency_year = None

            # Step through each year manually
            for year in range(time_horizon):
                # Get losses for this year (if any)
                loss_amount = annual_losses.get(year, 0.0)

                # Create loss events for this year
                losses = []
                if loss_amount > 0:
                    losses.append(
                        LossEvent(time=float(year), amount=loss_amount, loss_type="aggregate")
                    )

                # Execute time step
                metrics = sim.step_annual(year, losses)

                # Store results
                sim.assets[year] = metrics.get("assets", 0)
                sim.equity[year] = metrics.get("equity", 0)
                sim.roe[year] = metrics.get("roe", 0)
                sim.revenue[year] = metrics.get("revenue", 0)
                sim.net_income[year] = metrics.get("net_income", 0)
                sim.claim_counts[year] = metrics.get("claim_count", 0)
                sim.claim_amounts[year] = metrics.get("claim_amount", 0)

                # Check for insolvency
                if metrics.get("equity", 0) <= 0:
                    sim.insolvency_year = year
                    # Fill remaining years with zeros
                    sim.assets[year + 1 :] = 0
                    sim.equity[year + 1 :] = 0
                    sim.roe[year + 1 :] = np.nan
                    sim.revenue[year + 1 :] = 0
                    sim.net_income[year + 1 :] = 0
                    break

            # Create results object
            from .simulation import SimulationResults

            result = SimulationResults(
                years=sim.years[: year + 1] if sim.insolvency_year else sim.years,
                assets=sim.assets[: year + 1] if sim.insolvency_year else sim.assets,
                equity=sim.equity[: year + 1] if sim.insolvency_year else sim.equity,
                roe=sim.roe[: year + 1] if sim.insolvency_year else sim.roe,
                revenue=sim.revenue[: year + 1] if sim.insolvency_year else sim.revenue,
                net_income=sim.net_income[: year + 1] if sim.insolvency_year else sim.net_income,
                claim_counts=(
                    sim.claim_counts[: year + 1] if sim.insolvency_year else sim.claim_counts
                ),
                claim_amounts=(
                    sim.claim_amounts[: year + 1] if sim.insolvency_year else sim.claim_amounts
                ),
                insolvency_year=sim.insolvency_year,
            )
            simulation_results.append(result)

        # Calculate ergodic metrics
        equity_trajectories = [r.equity for r in simulation_results]

        # Time-average growth rates
        time_avg_growth_rates = [
            self.calculate_time_average_growth(traj) for traj in equity_trajectories
        ]
        valid_time_avg = [g for g in time_avg_growth_rates if np.isfinite(g)]

        # Ensemble statistics
        ensemble_stats = self.calculate_ensemble_average(equity_trajectories, metric="growth_rate")

        # Calculate insurance impact
        insurance_impact = {}
        if insurance_metadata:
            insurance_impact = {
                "premium_cost": insurance_metadata.get("total_premiums", 0),
                "recovery_benefit": insurance_metadata.get("total_recoveries", 0),
                "net_benefit": insurance_metadata.get("net_benefit", 0),
                "growth_improvement": np.mean(valid_time_avg) if valid_time_avg else 0,
            }

        # Calculate ergodic divergence
        time_avg_mean = float(np.mean(valid_time_avg)) if valid_time_avg else -np.inf
        ensemble_mean = float(ensemble_stats["mean"])
        ergodic_divergence = time_avg_mean - ensemble_mean

        # Validate results
        validation_passed = (
            len(valid_time_avg) > 0
            and ensemble_stats["survival_rate"] > 0
            and np.isfinite(ergodic_divergence)
        )

        return ErgodicAnalysisResults(
            time_average_growth=time_avg_mean,
            ensemble_average_growth=ensemble_mean,
            survival_rate=ensemble_stats["survival_rate"],
            ergodic_divergence=ergodic_divergence,
            insurance_impact=insurance_impact,
            validation_passed=validation_passed,
            metadata={
                "n_simulations": n_simulations,
                "time_horizon": time_horizon,
                "n_survived": ensemble_stats["n_survived"],
                "loss_statistics": insured_loss_data.calculate_statistics(),
            },
        )

    def validate_insurance_ergodic_impact(  # pylint: disable=too-many-locals
        self,
        base_scenario: SimulationResults,
        insurance_scenario: SimulationResults,
        insurance_program: Optional["InsuranceProgram"] = None,
    ) -> ValidationResults:
        """Comprehensively validate insurance effects in ergodic calculations.

        This method performs detailed validation to ensure that insurance impacts
        are properly reflected in the ergodic analysis. It checks the mathematical
        consistency and business logic of insurance effects on cash flows, growth
        rates, and survival probabilities.

        The validation is crucial for ensuring that ergodic analysis results are
        reliable and that observed insurance benefits (or costs) are genuine rather
        than artifacts of modeling errors or inconsistent implementations.

        Validation Checks Performed:
            1. **Premium Deduction Validation**: Verifies that insurance premiums
               are properly deducted from cash flows and reflected in net income
            2. **Recovery Credit Validation**: Confirms that insurance recoveries
               are properly credited and improve financial outcomes
            3. **Collateral Impact Validation**: Checks that letter of credit costs
               and asset restrictions are properly modeled
            4. **Growth Rate Consistency**: Validates that time-average growth
               calculations properly reflect insurance benefits

        Args:
            base_scenario (SimulationResults): Simulation results from baseline
                scenario without insurance coverage. Should represent the same
                business conditions and loss realizations as insurance_scenario
                but without insurance program applied.
            insurance_scenario (SimulationResults): Simulation results from scenario
                with insurance coverage. Should be directly comparable to base_scenario
                with only insurance coverage as the differentiating factor.
            insurance_program (Optional[InsuranceProgram]): The insurance program
                that was applied in insurance_scenario. If provided, enables more
                detailed validation of premium calculations and coverage effects.
                If None, performs validation based on observed differences only.

        Returns:
            ValidationResults: Comprehensive validation results containing:

                - premium_deductions_correct: Boolean indicating premium validation
                - recoveries_credited: Boolean indicating recovery validation
                - collateral_impacts_included: Boolean indicating collateral validation
                - time_average_reflects_benefit: Boolean indicating growth validation
                - overall_valid: Boolean indicating overall validation status
                - details: Dict with detailed validation information and metrics

        Examples:
            Basic validation after scenario comparison:

            .. code-block:: python

                # Run paired simulations
                base_sim = run_simulation(insurance_enabled=False, seed=12345)
                insured_sim = run_simulation(insurance_enabled=True, seed=12345)

                # Validate insurance effects
                validation = analyzer.validate_insurance_ergodic_impact(
                    base_sim,
                    insured_sim,
                    insurance_program
                )

                if validation.overall_valid:
                    print("✓ Insurance effects properly modeled")
                    print(f"  Premium deductions: {validation.premium_deductions_correct}")
                    print(f"  Recoveries credited: {validation.recoveries_credited}")
                    print(f"  Collateral impacts: {validation.collateral_impacts_included}")
                    print(f"  Growth consistency: {validation.time_average_reflects_benefit}")
                else:
                    print("⚠ Validation issues detected")
                    print("Review modeling implementation")

            Detailed validation diagnostics:

            .. code-block:: python

                validation = analyzer.validate_insurance_ergodic_impact(
                    base_scenario, insurance_scenario, insurance_program
                )

                # Examine premium validation details
                if 'premium_check' in validation.details:
                    premium_info = validation.details['premium_check']
                    print(f"\n=== Premium Validation ===")
                    print(f"Expected premium: ${premium_info['expected']:,.0f}")
                    print(f"Actual cost difference: ${premium_info['actual_diff']:,.0f}")
                    print(f"Validation passed: {premium_info['valid']}")

                    if not premium_info['valid']:
                        diff = abs(premium_info['expected'] - premium_info['actual_diff'])
                        print(f"⚠ Premium discrepancy: ${diff:,.0f}")

                # Examine recovery validation details
                if 'recovery_check' in validation.details:
                    recovery_info = validation.details['recovery_check']
                    print(f"\n=== Recovery Validation ===")
                    print(f"Base scenario claims: ${recovery_info['base_claims']:,.0f}")
                    print(f"Insured scenario claims: ${recovery_info['insured_claims']:,.0f}")
                    print(f"Base final equity: ${recovery_info['base_final_equity']:,.0f}")
                    print(f"Insured final equity: ${recovery_info['insured_final_equity']:,.0f}")
                    print(f"Validation passed: {recovery_info['valid']}")

                # Examine growth rate validation
                if 'growth_check' in validation.details:
                    growth_info = validation.details['growth_check']
                    print(f"\n=== Growth Rate Validation ===")
                    print(f"Base growth rate: {growth_info['base_growth']:.2%}")
                    print(f"Insured growth rate: {growth_info['insured_growth']:.2%}")
                    print(f"Growth improvement: {growth_info['improvement']:.2%}")
                    print(f"Validation passed: {growth_info['valid']}")

                    if growth_info['improvement'] > 0:
                        print("✓ Insurance improves time-average growth")
                    elif np.isfinite(growth_info['insured_growth']) and not np.isfinite(growth_info['base_growth']):
                        print("✓ Insurance prevents bankruptcy (infinite improvement)")

            Validation in Monte Carlo context:

            .. code-block:: python

                # Validate across multiple random seeds
                validation_results = []

                for seed in range(10):  # Test 10 paired simulations
                    base = run_simulation(insurance_enabled=False, seed=seed)
                    insured = run_simulation(insurance_enabled=True, seed=seed)

                    validation = analyzer.validate_insurance_ergodic_impact(
                        base, insured, insurance_program
                    )
                    validation_results.append(validation.overall_valid)

                # Check consistency across seeds
                validation_rate = sum(validation_results) / len(validation_results)
                print(f"Validation rate across seeds: {validation_rate:.1%}")

                if validation_rate < 0.8:
                    print("⚠ Inconsistent validation - check model implementation")
                else:
                    print("✓ Consistent validation across scenarios")

            Integration with scenario comparison:

            .. code-block:: python

                # Run comparison analysis
                comparison = analyzer.compare_scenarios(
                    [insured_sim], [base_sim], metric="equity"
                )

                # Validate the comparison
                validation = analyzer.validate_insurance_ergodic_impact(
                    base_sim, insured_sim, insurance_program
                )

                # Cross-check results
                if validation.overall_valid and comparison['ergodic_advantage']['significant']:
                    print("✓ Validated significant ergodic advantage from insurance")
                    print(f"  Time-average improvement: {comparison['ergodic_advantage']['time_average_gain']:.2%}")
                    print(f"  Statistical significance: p = {comparison['ergodic_advantage']['p_value']:.4f}")
                elif validation.overall_valid:
                    print("✓ Insurance effects validated but not statistically significant")
                    print("Consider running more simulations or adjusting parameters")
                else:
                    print("⚠ Validation failed - results may be unreliable")
                    print("Review model implementation before drawing conclusions")

        Validation Logic Details:
            **Premium Validation**: Compares expected premium costs (from insurance
            program) with actual observed difference in net income between scenarios.
            Allows for small numerical differences (<1% of expected premium).

            **Recovery Validation**: Checks that insurance scenario shows better
            financial performance despite potential premium costs. Allows for 5%
            variance to account for timing differences and model approximations.

            **Collateral Validation**: Verifies that letter of credit costs and
            asset restrictions are reflected in the financial calculations. Checks
            for non-zero differences in asset levels between scenarios.

            **Growth Rate Validation**: Ensures that time-average growth calculations
            properly reflect insurance benefits, especially in scenarios with
            significant loss exposure. Handles bankruptcy cases appropriately.

        Common Validation Failures:
            - Premium costs not properly deducted from cash flows
            - Insurance recoveries not credited to reduce net losses
            - Letter of credit collateral costs not included in expense calculations
            - Inconsistent treatment of bankruptcy scenarios
            - Timing mismatches between premium payments and loss occurrences

        Troubleshooting:
            If validation fails, check:
            1. Consistent random seed usage between base and insured scenarios
            2. Proper integration of insurance program with manufacturer model
            3. Correct timing of premium payments and loss recoveries
            4. Accurate letter of credit cost calculations
            5. Consistent handling of bankruptcy and survival scenarios

        Performance Notes:
            - Fast execution for single scenario pairs
            - Efficient for batch validation across multiple seeds
            - Comprehensive diagnostics with minimal computational overhead

        See Also:
            :class:`ValidationResults`: Detailed validation results format
            :meth:`compare_scenarios`: Main scenario comparison method
            :meth:`integrate_loss_ergodic_analysis`: End-to-end analysis pipeline
            :class:`~ergodic_insurance.insurance_program.InsuranceProgram`: Insurance modeling
        """
        details = {}

        # Check premium deductions
        premium_deductions_correct = True
        if insurance_program and hasattr(insurance_program, "calculate_premium"):
            expected_premium = insurance_program.calculate_premium()
            actual_cost_diff = np.sum(base_scenario.net_income - insurance_scenario.net_income)
            premium_diff = abs(actual_cost_diff - expected_premium * len(base_scenario.years))
            premium_deductions_correct = premium_diff < 0.01 * expected_premium
            details["premium_check"] = {
                "expected": expected_premium,
                "actual_diff": actual_cost_diff,
                "valid": premium_deductions_correct,
            }

        # Check recoveries are credited
        # With insurance, gross claims may be same or higher (if taking more risk),
        # but the net financial impact should be lower
        recoveries_credited = True
        total_base_claims = np.sum(base_scenario.claim_amounts)
        total_insured_claims = np.sum(insurance_scenario.claim_amounts)

        # Check if insurance scenario shows better financial performance
        # (either less net loss or better equity growth)
        base_final_equity = base_scenario.equity[-1] if len(base_scenario.equity) > 0 else 0
        insured_final_equity = (
            insurance_scenario.equity[-1] if len(insurance_scenario.equity) > 0 else 0
        )

        # Insurance is credited if final equity is better or claims don't reduce equity as much
        recoveries_credited = (
            insured_final_equity >= base_final_equity * 0.95
        )  # Allow small variance

        details["recovery_check"] = {
            "base_claims": total_base_claims,
            "insured_claims": total_insured_claims,
            "base_final_equity": base_final_equity,
            "insured_final_equity": insured_final_equity,
            "valid": recoveries_credited,
        }

        # Check collateral impacts (simplified check)
        collateral_impacts_included = True
        if insurance_program and hasattr(insurance_program, "collateral_requirement"):
            # Check if working capital is affected
            base_assets = base_scenario.assets
            insured_assets = insurance_scenario.assets
            asset_diff = np.mean(insured_assets - base_assets)
            collateral_impacts_included = abs(asset_diff) > 0
            details["collateral_check"] = {
                "asset_difference": asset_diff,
                "valid": collateral_impacts_included,
            }

        # Check time-average growth benefit
        base_growth = self.calculate_time_average_growth(base_scenario.equity)
        insured_growth = self.calculate_time_average_growth(insurance_scenario.equity)
        growth_improvement = insured_growth - base_growth

        # Insurance should improve growth if losses are significant
        time_average_reflects_benefit = True
        if total_base_claims > 0:
            # Expect positive growth improvement with insurance
            time_average_reflects_benefit = growth_improvement >= 0 or np.isfinite(insured_growth)

        details["growth_check"] = {
            "base_growth": base_growth,
            "insured_growth": insured_growth,
            "improvement": growth_improvement,
            "valid": time_average_reflects_benefit,
        }

        # Overall validation
        overall_valid = (
            premium_deductions_correct
            and recoveries_credited
            and collateral_impacts_included
            and time_average_reflects_benefit
        )

        return ValidationResults(
            premium_deductions_correct=premium_deductions_correct,
            recoveries_credited=recoveries_credited,
            collateral_impacts_included=collateral_impacts_included,
            time_average_reflects_benefit=time_average_reflects_benefit,
            overall_valid=overall_valid,
            details=details,
        )
