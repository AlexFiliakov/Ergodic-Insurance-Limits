"""Comprehensive tests for walk-forward validation system."""

import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.src.config import Config, ManufacturerConfig
from ergodic_insurance.src.insurance import InsuranceLayer, InsurancePolicy
from ergodic_insurance.src.insurance_program import EnhancedInsuranceLayer, InsuranceProgram
from ergodic_insurance.src.manufacturer import WidgetManufacturer
from ergodic_insurance.src.monte_carlo import SimulationConfig
from ergodic_insurance.src.monte_carlo import SimulationResults as MCSimulationResults
from ergodic_insurance.src.simulation import Simulation, SimulationResults
from ergodic_insurance.src.strategy_backtester import (
    AdaptiveStrategy,
    AggressiveFixedStrategy,
    BacktestResult,
    ConservativeFixedStrategy,
    InsuranceStrategy,
    NoInsuranceStrategy,
    OptimizedStaticStrategy,
    StrategyBacktester,
)
from ergodic_insurance.src.validation_metrics import (
    MetricCalculator,
    PerformanceTargets,
    StrategyPerformance,
    ValidationMetrics,
)
from ergodic_insurance.src.walk_forward_validator import (
    ValidationResult,
    ValidationWindow,
    WalkForwardValidator,
    WindowResult,
)


class TestValidationMetrics:
    """Test validation metrics module."""

    def test_validation_metrics_creation(self):
        """Test creating ValidationMetrics."""
        metrics = ValidationMetrics(
            roe=0.15, ruin_probability=0.01, growth_rate=0.08, volatility=0.2, sharpe_ratio=1.5
        )

        assert metrics.roe == 0.15
        assert metrics.ruin_probability == 0.01
        assert metrics.growth_rate == 0.08
        assert metrics.volatility == 0.2
        assert metrics.sharpe_ratio == 1.5

    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = ValidationMetrics(
            roe=0.15, ruin_probability=0.01, growth_rate=0.08, volatility=0.2
        )

        metrics_dict = metrics.to_dict()
        assert metrics_dict["roe"] == 0.15
        assert metrics_dict["ruin_probability"] == 0.01
        assert "sharpe_ratio" in metrics_dict

    def test_metrics_comparison(self):
        """Test comparing two metrics."""
        metrics1 = ValidationMetrics(
            roe=0.15, ruin_probability=0.01, growth_rate=0.08, volatility=0.2
        )
        metrics2 = ValidationMetrics(
            roe=0.10, ruin_probability=0.02, growth_rate=0.06, volatility=0.15
        )

        comparison = metrics1.compare(metrics2)

        assert abs(comparison["roe_diff"] - 0.5) < 0.001  # 50% better
        assert (
            abs(comparison["ruin_probability_diff"] + 0.5) < 0.001
        )  # 50% better (lower is better)

    def test_strategy_performance_degradation(self):
        """Test calculating performance degradation."""
        in_sample = ValidationMetrics(
            roe=0.15, ruin_probability=0.01, growth_rate=0.08, volatility=0.2
        )
        out_sample = ValidationMetrics(
            roe=0.12, ruin_probability=0.015, growth_rate=0.07, volatility=0.22
        )

        performance = StrategyPerformance(
            strategy_name="Test", in_sample_metrics=in_sample, out_sample_metrics=out_sample
        )

        performance.calculate_degradation()

        assert performance.degradation["roe_diff"] < 0  # ROE degraded
        assert performance.overfitting_score > 0  # Some overfitting detected

    def test_metric_calculator(self):
        """Test MetricCalculator functionality."""
        calculator = MetricCalculator(risk_free_rate=0.02)

        returns = np.random.normal(0.08, 0.02, 1000)
        final_assets = np.random.lognormal(16, 1, 1000)

        metrics = calculator.calculate_metrics(
            returns=returns, final_assets=final_assets, initial_assets=10000000, n_years=5
        )

        assert isinstance(metrics, ValidationMetrics)
        assert metrics.roe > 0
        assert 0 <= metrics.win_rate <= 1
        assert metrics.volatility > 0

    def test_performance_targets_evaluation(self):
        """Test performance target evaluation."""
        targets = PerformanceTargets(min_roe=0.10, max_ruin_probability=0.02, min_sharpe_ratio=1.0)

        # Test passing metrics
        good_metrics = ValidationMetrics(
            roe=0.15, ruin_probability=0.01, growth_rate=0.08, volatility=0.1, sharpe_ratio=1.5
        )

        meets_targets, failures = targets.evaluate(good_metrics)
        assert meets_targets
        assert len(failures) == 0

        # Test failing metrics
        bad_metrics = ValidationMetrics(
            roe=0.05, ruin_probability=0.05, growth_rate=0.03, volatility=0.3, sharpe_ratio=0.5
        )

        meets_targets, failures = targets.evaluate(bad_metrics)
        assert not meets_targets
        assert len(failures) > 0


class TestStrategyBacktester:
    """Test strategy backtesting module."""

    def test_no_insurance_strategy(self):
        """Test NoInsuranceStrategy."""
        strategy = NoInsuranceStrategy()
        config = ManufacturerConfig(
            initial_assets=10000000,
            asset_turnover_ratio=1.0,
            operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)

        program = strategy.get_insurance_program(manufacturer)
        assert program is None
        assert strategy.name == "No Insurance"

    def test_conservative_fixed_strategy(self):
        """Test ConservativeFixedStrategy."""
        strategy = ConservativeFixedStrategy(
            primary_limit=5000000, excess_limit=20000000, deductible=50000
        )
        config = ManufacturerConfig(
            initial_assets=10000000,
            asset_turnover_ratio=1.0,
            operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)

        program = strategy.get_insurance_program(manufacturer)
        assert isinstance(program, InsuranceProgram)
        assert len(program.layers) == 3
        assert program.layers[0].attachment_point == 50000

    def test_aggressive_fixed_strategy(self):
        """Test AggressiveFixedStrategy."""
        strategy = AggressiveFixedStrategy(
            primary_limit=2000000, excess_limit=5000000, deductible=250000
        )
        config = ManufacturerConfig(
            initial_assets=10000000,
            asset_turnover_ratio=1.0,
            operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)

        program = strategy.get_insurance_program(manufacturer)
        assert isinstance(program, InsuranceProgram)
        assert len(program.layers) == 2
        assert program.layers[0].attachment_point == 250000

    def test_adaptive_strategy_update(self):
        """Test AdaptiveStrategy adaptation."""
        strategy = AdaptiveStrategy(
            base_deductible=100000, base_primary=3000000, base_excess=10000000, adaptation_window=3
        )

        # Initial state
        assert strategy.current_deductible == 100000
        assert strategy.current_primary == 3000000

        # Update with high losses
        high_losses = np.array([5000000, 6000000])
        strategy.update(high_losses, np.array([1000000]), year=1)

        # Should increase coverage (or at least maintain it due to adjustment limits)
        assert strategy.current_primary >= 3000000

        # Reset
        strategy.reset()
        assert strategy.current_deductible == 100000
        assert len(strategy.loss_history) == 0

    def test_optimized_static_strategy(self):
        """Test OptimizedStaticStrategy with defaults."""
        strategy = OptimizedStaticStrategy(target_roe=0.15, max_ruin_prob=0.01)
        config = ManufacturerConfig(
            initial_assets=10000000,
            asset_turnover_ratio=1.0,
            operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)

        # Should use defaults when not optimized
        program = strategy.get_insurance_program(manufacturer)
        assert isinstance(program, InsuranceProgram)
        assert strategy.optimized_params is not None

    @patch("ergodic_insurance.src.strategy_backtester.MonteCarloEngine")
    def test_strategy_backtester(self, mock_mc_engine):
        """Test StrategyBacktester."""
        # Mock Monte Carlo engine
        mock_results = MCSimulationResults(
            final_assets=np.random.lognormal(16, 1, 100),
            annual_losses=np.random.exponential(100000, (100, 5)),
            insurance_recoveries=np.random.exponential(50000, (100, 5)),
            retained_losses=np.random.exponential(50000, (100, 5)),
            growth_rates=np.random.normal(0.05, 0.02, 100),
            ruin_probability=0.02,
            metrics={"mean_roe": 0.12},
            convergence={},
            execution_time=1.0,
            config=SimulationConfig(n_simulations=100, n_years=5),
        )
        mock_mc_engine.return_value.run.return_value = mock_results

        backtester = StrategyBacktester()
        strategy = NoInsuranceStrategy()
        config = ManufacturerConfig(
            initial_assets=10000000,
            asset_turnover_ratio=1.0,
            operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)
        config = SimulationConfig(n_simulations=100, n_years=5)

        result = backtester.test_strategy(strategy, manufacturer, config)

        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "No Insurance"
        assert isinstance(result.metrics, ValidationMetrics)

    @patch("ergodic_insurance.src.strategy_backtester.MonteCarloEngine")
    def test_multiple_strategies_comparison(self, mock_mc_engine):
        """Test comparing multiple strategies."""
        # Mock results
        mock_results = MCSimulationResults(
            final_assets=np.random.lognormal(16, 1, 100),
            annual_losses=np.random.exponential(100000, (100, 5)),
            insurance_recoveries=np.random.exponential(50000, (100, 5)),
            retained_losses=np.random.exponential(50000, (100, 5)),
            growth_rates=np.random.normal(0.05, 0.02, 100),
            ruin_probability=0.02,
            metrics={"mean_roe": 0.12},
            convergence={},
            execution_time=1.0,
            config=SimulationConfig(n_simulations=100, n_years=5),
        )
        mock_mc_engine.return_value.run.return_value = mock_results

        backtester = StrategyBacktester()
        strategies = [NoInsuranceStrategy(), ConservativeFixedStrategy(), AggressiveFixedStrategy()]
        config = ManufacturerConfig(
            initial_assets=10000000,
            asset_turnover_ratio=1.0,
            operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)
        config = SimulationConfig(n_simulations=100, n_years=5)

        results_df = backtester.test_multiple_strategies(strategies, manufacturer, config)

        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 3
        assert "strategy" in results_df.columns
        assert "roe" in results_df.columns
        assert "ruin_probability" in results_df.columns


class TestWalkForwardValidator:
    """Test walk-forward validation system."""

    def test_window_generation(self):
        """Test validation window generation."""
        validator = WalkForwardValidator(window_size=3, step_size=1, test_ratio=0.3)

        windows = validator.generate_windows(total_years=10)

        assert len(windows) == 8  # (10 - 3) / 1 + 1
        assert windows[0].train_start == 0
        assert windows[0].train_end == 2  # 70% of 3 years
        assert windows[0].test_start == 2
        assert windows[0].test_end == 3

        # Check step size
        assert windows[1].train_start == 1
        assert windows[1].train_end == 3

    def test_validation_window_properties(self):
        """Test ValidationWindow properties."""
        window = ValidationWindow(
            window_id=0, train_start=0, train_end=7, test_start=7, test_end=10
        )

        assert window.get_train_years() == 7
        assert window.get_test_years() == 3
        assert "Train[0-7]" in str(window)

    @patch("ergodic_insurance.src.walk_forward_validator.StrategyBacktester")
    def test_process_window(self, mock_backtester):
        """Test processing a single window."""
        # Setup mocks
        mock_train_result = BacktestResult(
            strategy_name="Test",
            simulation_results=Mock(),
            metrics=ValidationMetrics(
                roe=0.15, ruin_probability=0.01, growth_rate=0.08, volatility=0.2
            ),
            execution_time=1.0,
            config=SimulationConfig(),
        )
        mock_test_result = BacktestResult(
            strategy_name="Test",
            simulation_results=Mock(),
            metrics=ValidationMetrics(
                roe=0.12, ruin_probability=0.015, growth_rate=0.07, volatility=0.22
            ),
            execution_time=1.0,
            config=SimulationConfig(),
        )

        mock_backtester.return_value.test_strategy.side_effect = [
            mock_train_result,
            mock_test_result,
        ]

        validator = WalkForwardValidator()
        window = ValidationWindow(0, 0, 2, 2, 3)
        strategies = [NoInsuranceStrategy()]
        config = ManufacturerConfig(
            initial_assets=10000000,
            asset_turnover_ratio=1.0,
            operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)
        # Config not needed - _process_window will handle defaults

        result = validator._process_window(
            window=window,
            strategies=strategies,
            n_simulations=100,
            manufacturer=manufacturer,
            config=None,
        )

        assert isinstance(result, WindowResult)
        assert "No Insurance" in result.strategy_performances
        assert result.strategy_performances["No Insurance"].overfitting_score > 0

    @patch("ergodic_insurance.src.walk_forward_validator.StrategyBacktester")
    def test_validate_strategies(self, mock_backtester):
        """Test full validation process."""
        # Setup mocks
        mock_metrics = ValidationMetrics(
            roe=0.12, ruin_probability=0.02, growth_rate=0.06, volatility=0.2
        )
        mock_result = BacktestResult(
            strategy_name="Test",
            simulation_results=Mock(),
            metrics=mock_metrics,
            execution_time=1.0,
            config=SimulationConfig(),
        )
        mock_backtester.return_value.test_strategy.return_value = mock_result

        validator = WalkForwardValidator(window_size=3, step_size=2)
        strategies = [NoInsuranceStrategy(), ConservativeFixedStrategy()]

        validation_result = validator.validate_strategies(
            strategies=strategies, n_years=10, n_simulations=100
        )

        assert isinstance(validation_result, ValidationResult)
        assert len(validation_result.window_results) > 0
        assert validation_result.best_strategy is not None
        assert len(validation_result.overfitting_analysis) > 0

    def test_analyze_results(self):
        """Test result analysis."""
        validator = WalkForwardValidator()

        # Create mock window results
        window_results = []
        for i in range(3):
            window = ValidationWindow(i, i * 2, i * 2 + 2, i * 2 + 2, i * 2 + 3)
            window_result = WindowResult(window=window)

            # Add strategy performances
            for strategy_name in ["Strategy1", "Strategy2"]:
                performance = StrategyPerformance(
                    strategy_name=strategy_name,
                    in_sample_metrics=ValidationMetrics(
                        roe=0.15 - i * 0.01,
                        ruin_probability=0.01 + i * 0.005,
                        growth_rate=0.08,
                        volatility=0.2,
                    ),
                    out_sample_metrics=ValidationMetrics(
                        roe=0.12 - i * 0.01,
                        ruin_probability=0.02 + i * 0.005,
                        growth_rate=0.07,
                        volatility=0.25,
                    ),
                )
                performance.calculate_degradation()
                window_result.strategy_performances[strategy_name] = performance

            window_results.append(window_result)

        validation_result = ValidationResult(window_results=window_results)
        strategies = [Mock(name="Strategy1"), Mock(name="Strategy2")]

        validator._analyze_results(validation_result, strategies)

        assert len(validation_result.overfitting_analysis) == 2
        assert len(validation_result.consistency_scores) == 2
        assert not validation_result.strategy_rankings.empty
        assert validation_result.best_strategy in ["Strategy1", "Strategy2"]

    def test_composite_score_calculation(self):
        """Test composite score calculation."""
        validator = WalkForwardValidator()

        metrics = ValidationMetrics(
            roe=0.15, ruin_probability=0.01, growth_rate=0.08, volatility=0.15, sharpe_ratio=1.5
        )

        score = validator._calculate_composite_score(
            metrics=metrics, overfitting_score=0.1, consistency_score=0.9
        )

        assert 0 <= score <= 1
        assert score > 0.5  # Should be good with these metrics

    def test_report_generation(self):
        """Test report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = WalkForwardValidator()

            # Create mock validation result
            validation_result = ValidationResult()
            validation_result.best_strategy = "TestStrategy"
            validation_result.overfitting_analysis = {"TestStrategy": 0.15}
            validation_result.consistency_scores = {"TestStrategy": 0.85}
            validation_result.strategy_rankings = pd.DataFrame(
                [{"strategy": "TestStrategy", "avg_roe": 0.12, "composite_score": 0.75}]
            )

            window = ValidationWindow(0, 0, 2, 2, 3)
            window_result = WindowResult(window=window)
            performance = StrategyPerformance(
                strategy_name="TestStrategy",
                in_sample_metrics=ValidationMetrics(
                    roe=0.15, ruin_probability=0.01, growth_rate=0.08, volatility=0.2
                ),
                out_sample_metrics=ValidationMetrics(
                    roe=0.12, ruin_probability=0.02, growth_rate=0.07, volatility=0.22
                ),
            )
            window_result.strategy_performances["TestStrategy"] = performance
            validation_result.window_results = [window_result]

            # Generate reports
            report_files = validator.generate_report(
                validation_result=validation_result,
                output_dir=temp_dir,
                include_visualizations=True,
            )

            assert "markdown" in report_files
            assert "html" in report_files
            assert "json" in report_files
            assert report_files["markdown"].exists()
            assert report_files["html"].exists()
            assert report_files["json"].exists()

            # Check JSON content
            with open(report_files["json"], "r") as f:
                json_data = json.load(f)
                assert json_data["best_strategy"] == "TestStrategy"
                assert "windows" in json_data

    def test_markdown_report_content(self):
        """Test markdown report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = WalkForwardValidator()

            # Create validation result
            validation_result = ValidationResult()
            validation_result.best_strategy = "BestStrategy"
            validation_result.overfitting_analysis = {
                "Strategy1": 0.1,
                "Strategy2": 0.3,
                "Strategy3": 0.5,
            }
            validation_result.consistency_scores = {
                "Strategy1": 0.9,
                "Strategy2": 0.7,
                "Strategy3": 0.5,
            }
            validation_result.strategy_rankings = pd.DataFrame(
                [
                    {"strategy": "Strategy1", "avg_roe": 0.15, "composite_score": 0.85},
                    {"strategy": "Strategy2", "avg_roe": 0.12, "composite_score": 0.65},
                    {"strategy": "Strategy3", "avg_roe": 0.10, "composite_score": 0.45},
                ]
            )

            md_path = Path(temp_dir) / "test_report.md"
            validator._generate_markdown_report(validation_result, md_path)

            content = md_path.read_text()
            assert "Walk-Forward Validation Report" in content
            assert "BestStrategy" in content
            assert "✓ Low" in content  # For low overfitting
            assert "✗ High" in content  # For high overfitting
            assert "Strategy Rankings" in content

    def test_html_report_structure(self):
        """Test HTML report structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = WalkForwardValidator()

            validation_result = ValidationResult()
            validation_result.best_strategy = "TestStrategy"
            validation_result.strategy_rankings = pd.DataFrame(
                [{"strategy": "TestStrategy", "avg_roe": 0.12}]
            )

            html_path = Path(temp_dir) / "test_report.html"
            validator._generate_html_report(
                validation_result, html_path, include_visualizations=False
            )

            content = html_path.read_text()
            assert "<!DOCTYPE html>" in content
            assert "Walk-Forward Validation Report" in content
            assert "TestStrategy" in content
            assert "<table" in content


class TestIntegration:
    """Integration tests for the walk-forward validation system."""

    @patch("ergodic_insurance.src.strategy_backtester.MonteCarloEngine")
    def test_end_to_end_validation(self, mock_mc_engine):
        """Test complete validation workflow."""
        # Setup mock results
        mock_results = MCSimulationResults(
            final_assets=np.random.lognormal(16, 1, 100),
            annual_losses=np.random.exponential(100000, (100, 5)),
            insurance_recoveries=np.random.exponential(50000, (100, 5)),
            retained_losses=np.random.exponential(50000, (100, 5)),
            growth_rates=np.random.normal(0.05, 0.02, 100),
            ruin_probability=0.02,
            metrics={"mean_roe": 0.12},
            convergence={},
            execution_time=1.0,
            config=SimulationConfig(n_simulations=100, n_years=5),
        )
        mock_mc_engine.return_value.run.return_value = mock_results

        # Create strategies
        strategies = [
            NoInsuranceStrategy(),
            ConservativeFixedStrategy(),
            AggressiveFixedStrategy(),
            AdaptiveStrategy(),
        ]

        # Run validation
        validator = WalkForwardValidator(window_size=3, step_size=2, test_ratio=0.3)

        validation_result = validator.validate_strategies(
            strategies=strategies, n_years=7, n_simulations=100
        )

        # Verify results
        assert isinstance(validation_result, ValidationResult)
        assert validation_result.best_strategy is not None
        assert len(validation_result.window_results) > 0

        # Generate reports
        with tempfile.TemporaryDirectory() as temp_dir:
            report_files = validator.generate_report(
                validation_result=validation_result,
                output_dir=temp_dir,
                include_visualizations=False,
            )

            assert all(Path(f).exists() for f in report_files.values() if isinstance(f, Path))

    def test_performance_targets_integration(self):
        """Test integration with performance targets."""
        targets = PerformanceTargets(min_roe=0.10, max_ruin_probability=0.05, min_sharpe_ratio=1.0)

        validator = WalkForwardValidator(performance_targets=targets)

        # Verify targets are used in evaluation
        metrics = ValidationMetrics(
            roe=0.08,  # Below target
            ruin_probability=0.03,
            growth_rate=0.05,
            volatility=0.2,
            sharpe_ratio=0.8,  # Below target
        )

        meets_targets, failures = targets.evaluate(metrics)
        assert not meets_targets
        assert len(failures) == 2  # ROE and Sharpe failures


# Test fixtures
@pytest.fixture
def sample_validation_result():
    """Create sample validation result for testing."""
    validation_result = ValidationResult()
    validation_result.best_strategy = "TestStrategy"
    validation_result.overfitting_analysis = {"TestStrategy": 0.15}
    validation_result.consistency_scores = {"TestStrategy": 0.85}
    validation_result.strategy_rankings = pd.DataFrame(
        [{"strategy": "TestStrategy", "avg_roe": 0.12, "composite_score": 0.75}]
    )

    window = ValidationWindow(0, 0, 2, 2, 3)
    window_result = WindowResult(window=window)
    performance = StrategyPerformance(
        strategy_name="TestStrategy",
        in_sample_metrics=ValidationMetrics(
            roe=0.15, ruin_probability=0.01, growth_rate=0.08, volatility=0.2
        ),
        out_sample_metrics=ValidationMetrics(
            roe=0.12, ruin_probability=0.02, growth_rate=0.07, volatility=0.22
        ),
    )
    performance.calculate_degradation()
    window_result.strategy_performances["TestStrategy"] = performance
    validation_result.window_results = [window_result]

    return validation_result


@pytest.fixture
def mock_simulation_engine():
    """Create mock simulation engine."""
    engine = Mock(spec=Simulation)
    results = Mock(spec=SimulationResults)
    results.calculate_time_weighted_roe.return_value = 0.12
    results.years = np.arange(5)
    results.assets = np.random.lognormal(16, 0.1, 5)
    results.roe = np.random.normal(0.12, 0.02, 5)
    engine.run_simulation.return_value = results
    return engine
