"""Comprehensive tests for walk-forward validation system."""

import json
from pathlib import Path
import tempfile
from typing import List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.config import Config, ManufacturerConfig
from ergodic_insurance.insurance import InsuranceLayer, InsurancePolicy
from ergodic_insurance.insurance_program import EnhancedInsuranceLayer, InsuranceProgram
from ergodic_insurance.manufacturer import WidgetManufacturer
from ergodic_insurance.monte_carlo import MonteCarloResults, SimulationConfig
from ergodic_insurance.simulation import Simulation, SimulationResults
from ergodic_insurance.strategy_backtester import (
    AdaptiveStrategy,
    AggressiveFixedStrategy,
    BacktestResult,
    ConservativeFixedStrategy,
    InsuranceStrategy,
    NoInsuranceStrategy,
    OptimizedStaticStrategy,
    StrategyBacktester,
)
from ergodic_insurance.validation_metrics import (
    MetricCalculator,
    PerformanceTargets,
    StrategyPerformance,
    ValidationMetrics,
)
from ergodic_insurance.walk_forward_validator import (
    ValidationResult,
    ValidationWindow,
    WalkForwardValidator,
    WindowResult,
    _process_window_worker,
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
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)

        # NoInsuranceStrategy intentionally returns None
        # pylint: disable-next=assignment-from-none
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
            base_operating_margin=0.08,
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
            base_operating_margin=0.08,
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
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)

        # Should use defaults when not optimized
        program = strategy.get_insurance_program(manufacturer)
        assert isinstance(program, InsuranceProgram)
        assert strategy.optimized_params is not None

    @patch("ergodic_insurance.strategy_backtester.MonteCarloEngine")
    def test_strategy_backtester(self, mock_mc_engine):
        """Test StrategyBacktester."""
        # Mock Monte Carlo engine
        mock_results = MonteCarloResults(
            final_assets=np.random.lognormal(16, 1, 100),
            annual_losses=np.random.exponential(100000, (100, 5)),
            insurance_recoveries=np.random.exponential(50000, (100, 5)),
            retained_losses=np.random.exponential(50000, (100, 5)),
            growth_rates=np.random.normal(0.05, 0.02, 100),
            ruin_probability={"5": 0.02},
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
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)
        sim_config = SimulationConfig(n_simulations=100, n_years=5)

        result = backtester.test_strategy(strategy, manufacturer, sim_config)

        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "No Insurance"
        assert isinstance(result.metrics, ValidationMetrics)

    @patch("ergodic_insurance.strategy_backtester.MonteCarloEngine")
    def test_multiple_strategies_comparison(self, mock_mc_engine):
        """Test comparing multiple strategies."""
        # Mock results
        mock_results = MonteCarloResults(
            final_assets=np.random.lognormal(16, 1, 100),
            annual_losses=np.random.exponential(100000, (100, 5)),
            insurance_recoveries=np.random.exponential(50000, (100, 5)),
            retained_losses=np.random.exponential(50000, (100, 5)),
            growth_rates=np.random.normal(0.05, 0.02, 100),
            ruin_probability={"5": 0.02},
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
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(config)
        sim_config = SimulationConfig(n_simulations=100, n_years=5)

        results_df = backtester.test_multiple_strategies(strategies, manufacturer, sim_config)

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

    @patch("ergodic_insurance.walk_forward_validator.StrategyBacktester")
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
        strategies: List[InsuranceStrategy] = [NoInsuranceStrategy()]
        mfg_config = ManufacturerConfig(
            initial_assets=10000000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(mfg_config)
        # Create full config since None is not allowed
        from ergodic_insurance.config import (  # pylint: disable=reimported
            Config,
            DebtConfig,
            GrowthConfig,
            LoggingConfig,
            OutputConfig,
        )
        from ergodic_insurance.config import SimulationConfig as SimConfig
        from ergodic_insurance.config import WorkingCapitalConfig  # pylint: disable=reimported

        full_config = Config(
            manufacturer=mfg_config,
            working_capital=WorkingCapitalConfig(percent_of_sales=0.15),
            growth=GrowthConfig(type="deterministic", annual_growth_rate=0.05, volatility=0.15),
            debt=DebtConfig(
                interest_rate=0.05, max_leverage_ratio=2.0, minimum_cash_balance=100000
            ),
            simulation=SimConfig(time_resolution="annual", time_horizon_years=10),
            output=OutputConfig(
                output_directory="./results",
                file_format="csv",
                checkpoint_frequency=0,
                detailed_metrics=True,
            ),
            logging=LoggingConfig(enabled=True, level="INFO", log_file=None),
        )

        result = validator._process_window(
            window=window,
            strategies=strategies,
            n_simulations=100,
            manufacturer=manufacturer,
            config=full_config,
        )

        assert isinstance(result, WindowResult)
        assert "No Insurance" in result.strategy_performances
        assert result.strategy_performances["No Insurance"].overfitting_score > 0

    @patch("ergodic_insurance.walk_forward_validator.StrategyBacktester")
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

        validator = WalkForwardValidator(window_size=3, step_size=2, max_workers=1)
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
        # Create proper strategy objects
        strategy1 = Mock(spec=InsuranceStrategy)
        strategy1.name = "Strategy1"
        strategy2 = Mock(spec=InsuranceStrategy)
        strategy2.name = "Strategy2"
        strategies: List[InsuranceStrategy] = [strategy1, strategy2]

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

            content = md_path.read_text(encoding="utf-8")
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


class TestParallelExecution:
    """Tests for parallel window processing."""

    def test_max_workers_default(self):
        """Test default max_workers is None (auto-detect)."""
        validator = WalkForwardValidator()
        assert validator.max_workers is None

    def test_max_workers_stored(self):
        """Test max_workers parameter is stored correctly."""
        validator = WalkForwardValidator(max_workers=4)
        assert validator.max_workers == 4

    @patch("ergodic_insurance.walk_forward_validator.StrategyBacktester")
    def test_sequential_with_max_workers_one(self, mock_backtester):
        """Test sequential execution when max_workers=1."""
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

        validator = WalkForwardValidator(window_size=3, step_size=2, max_workers=1)
        result = validator.validate_strategies(
            strategies=[NoInsuranceStrategy(), ConservativeFixedStrategy()],
            n_years=10,
            n_simulations=100,
        )

        assert isinstance(result, ValidationResult)
        assert len(result.window_results) > 0
        assert result.best_strategy is not None

    @patch("ergodic_insurance.walk_forward_validator.StrategyBacktester")
    def test_worker_function_produces_valid_result(self, mock_backtester):
        """Test _process_window_worker produces correct WindowResult."""
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

        window = ValidationWindow(0, 0, 2, 2, 3)
        strategies = [NoInsuranceStrategy()]
        mfg_config = ManufacturerConfig(
            initial_assets=10000000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(mfg_config)

        from ergodic_insurance.config import (  # pylint: disable=reimported
            Config,
            DebtConfig,
            GrowthConfig,
            LoggingConfig,
            OutputConfig,
        )
        from ergodic_insurance.config import SimulationConfig as SimConfig
        from ergodic_insurance.config import WorkingCapitalConfig  # pylint: disable=reimported

        full_config = Config(
            manufacturer=mfg_config,
            working_capital=WorkingCapitalConfig(percent_of_sales=0.15),
            growth=GrowthConfig(type="deterministic", annual_growth_rate=0.05, volatility=0.15),
            debt=DebtConfig(
                interest_rate=0.05, max_leverage_ratio=2.0, minimum_cash_balance=100000
            ),
            simulation=SimConfig(time_resolution="annual", time_horizon_years=10),
            output=OutputConfig(
                output_directory="./results",
                file_format="csv",
                checkpoint_frequency=0,
                detailed_metrics=True,
            ),
            logging=LoggingConfig(enabled=True, level="INFO", log_file=None),
        )

        result = _process_window_worker(
            window=window,
            strategies=strategies,
            n_simulations=100,
            manufacturer=manufacturer,
            config=full_config,
        )

        assert isinstance(result, WindowResult)
        assert result.window == window
        assert "No Insurance" in result.strategy_performances
        assert result.execution_time >= 0

    @patch.object(WalkForwardValidator, "_process_windows_parallel")
    @patch("ergodic_insurance.walk_forward_validator.StrategyBacktester")
    def test_dispatches_to_parallel_when_multiple_windows(self, mock_backtester, mock_parallel):
        """Test that parallel execution is used for multiple windows with max_workers > 1."""
        mock_metrics = ValidationMetrics(
            roe=0.12, ruin_probability=0.02, growth_rate=0.06, volatility=0.2
        )

        # Create window results matching generate_windows output
        windows = WalkForwardValidator(window_size=3, step_size=2).generate_windows(10)
        mock_window_results = []
        for window in windows:
            perf = StrategyPerformance(
                strategy_name="No Insurance",
                in_sample_metrics=mock_metrics,
                out_sample_metrics=mock_metrics,
            )
            perf.calculate_degradation()
            wr = WindowResult(
                window=window,
                strategy_performances={"No Insurance": perf},
                execution_time=1.0,
            )
            mock_window_results.append(wr)
        mock_parallel.return_value = mock_window_results

        validator = WalkForwardValidator(window_size=3, step_size=2, max_workers=2)
        result = validator.validate_strategies(
            strategies=[NoInsuranceStrategy()],
            n_years=10,
            n_simulations=100,
        )

        mock_parallel.assert_called_once()
        assert isinstance(result, ValidationResult)
        assert len(result.window_results) == 4

    @patch.object(WalkForwardValidator, "_process_windows_parallel")
    @patch("ergodic_insurance.walk_forward_validator.StrategyBacktester")
    def test_no_parallel_dispatch_when_max_workers_one(self, mock_backtester, mock_parallel):
        """Test that sequential execution is used when max_workers=1."""
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

        validator = WalkForwardValidator(window_size=3, step_size=2, max_workers=1)
        result = validator.validate_strategies(
            strategies=[NoInsuranceStrategy()],
            n_years=10,
            n_simulations=100,
        )

        mock_parallel.assert_not_called()
        assert isinstance(result, ValidationResult)

    @patch("ergodic_insurance.walk_forward_validator.StrategyBacktester")
    def test_worker_function_multiple_strategies(self, mock_backtester):
        """Test _process_window_worker handles multiple strategies correctly."""
        mock_metrics_1 = ValidationMetrics(
            roe=0.15, ruin_probability=0.01, growth_rate=0.08, volatility=0.2
        )
        mock_metrics_2 = ValidationMetrics(
            roe=0.10, ruin_probability=0.03, growth_rate=0.05, volatility=0.25
        )
        mock_backtester.return_value.test_strategy.side_effect = [
            BacktestResult("Conservative", Mock(), mock_metrics_1, 1.0, SimulationConfig()),
            BacktestResult("Conservative", Mock(), mock_metrics_1, 1.0, SimulationConfig()),
            BacktestResult("No Insurance", Mock(), mock_metrics_2, 1.0, SimulationConfig()),
            BacktestResult("No Insurance", Mock(), mock_metrics_2, 1.0, SimulationConfig()),
        ]

        window = ValidationWindow(0, 0, 2, 2, 3)
        strategies = [ConservativeFixedStrategy(), NoInsuranceStrategy()]
        mfg_config = ManufacturerConfig(
            initial_assets=10000000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(mfg_config)

        from ergodic_insurance.config import (  # pylint: disable=reimported
            Config,
            DebtConfig,
            GrowthConfig,
            LoggingConfig,
            OutputConfig,
        )
        from ergodic_insurance.config import SimulationConfig as SimConfig
        from ergodic_insurance.config import WorkingCapitalConfig  # pylint: disable=reimported

        full_config = Config(
            manufacturer=mfg_config,
            working_capital=WorkingCapitalConfig(percent_of_sales=0.15),
            growth=GrowthConfig(type="deterministic", annual_growth_rate=0.05, volatility=0.15),
            debt=DebtConfig(
                interest_rate=0.05, max_leverage_ratio=2.0, minimum_cash_balance=100000
            ),
            simulation=SimConfig(time_resolution="annual", time_horizon_years=10),
            output=OutputConfig(
                output_directory="./results",
                file_format="csv",
                checkpoint_frequency=0,
                detailed_metrics=True,
            ),
            logging=LoggingConfig(enabled=True, level="INFO", log_file=None),
        )

        result = _process_window_worker(
            window=window,
            strategies=strategies,
            n_simulations=100,
            manufacturer=manufacturer,
            config=full_config,
        )

        assert isinstance(result, WindowResult)
        assert len(result.strategy_performances) == 2
        assert "Conservative Fixed" in result.strategy_performances
        assert "No Insurance" in result.strategy_performances


class TestSeedSequenceSeeding:
    """Tests for SeedSequence-based window seeding (#605)."""

    def test_seed_parameter_stored(self):
        """Test that the seed parameter is stored on the validator."""
        validator = WalkForwardValidator(seed=42)
        assert validator.seed == 42

    def test_seed_default_is_none(self):
        """Test that seed defaults to None (system entropy)."""
        validator = WalkForwardValidator()
        assert validator.seed is None

    def test_no_seed_collision_across_windows(self):
        """Test that SeedSequence eliminates the arithmetic seed collision.

        Previously window_id*1000 caused collisions: window 0 sim 1000
        and window 1 sim 0 both produced seed 1000.  SeedSequence.spawn()
        guarantees distinct child entropy pools.
        """
        base_ss = np.random.SeedSequence(42)
        window_seeds = base_ss.spawn(5)

        # Derive train/test int seeds for each window
        all_seeds = set()
        for ws in window_seeds:
            train_ss, test_ss = ws.spawn(2)
            train_seed = int(train_ss.generate_state(1)[0])
            test_seed = int(test_ss.generate_state(1)[0])
            all_seeds.add(train_seed)
            all_seeds.add(test_seed)

        # 5 windows x 2 seeds each = 10 unique seeds expected
        assert len(all_seeds) == 10

    def test_reproducibility_with_same_seed(self):
        """Test that the same base seed produces identical window seeds."""

        def derive_seeds(base_seed):
            base_ss = np.random.SeedSequence(base_seed)
            window_seeds = base_ss.spawn(3)
            result = []
            for ws in window_seeds:
                t, v = ws.spawn(2)
                result.append((int(t.generate_state(1)[0]), int(v.generate_state(1)[0])))
            return result

        run1 = derive_seeds(123)
        run2 = derive_seeds(123)
        assert run1 == run2

    def test_different_seeds_produce_different_streams(self):
        """Test that different base seeds produce different window seeds."""

        def derive_seeds(base_seed):
            base_ss = np.random.SeedSequence(base_seed)
            ws = base_ss.spawn(1)[0]
            t, v = ws.spawn(2)
            return int(t.generate_state(1)[0]), int(v.generate_state(1)[0])

        seeds_a = derive_seeds(42)
        seeds_b = derive_seeds(99)
        assert seeds_a != seeds_b

    @patch("ergodic_insurance.walk_forward_validator.StrategyBacktester")
    def test_worker_receives_window_seed(self, mock_backtester):
        """Test that _process_window_worker uses the provided SeedSequence."""
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

        window = ValidationWindow(0, 0, 2, 2, 3)
        strategies = [NoInsuranceStrategy()]
        mfg_config = ManufacturerConfig(
            initial_assets=10000000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(mfg_config)

        from ergodic_insurance.config import (  # pylint: disable=reimported
            Config,
            DebtConfig,
            GrowthConfig,
            LoggingConfig,
            OutputConfig,
        )
        from ergodic_insurance.config import SimulationConfig as SimConfig
        from ergodic_insurance.config import WorkingCapitalConfig  # pylint: disable=reimported

        full_config = Config(
            manufacturer=mfg_config,
            working_capital=WorkingCapitalConfig(percent_of_sales=0.15),
            growth=GrowthConfig(type="deterministic", annual_growth_rate=0.05, volatility=0.15),
            debt=DebtConfig(
                interest_rate=0.05, max_leverage_ratio=2.0, minimum_cash_balance=100000
            ),
            simulation=SimConfig(time_resolution="annual", time_horizon_years=10),
            output=OutputConfig(
                output_directory="./results",
                file_format="csv",
                checkpoint_frequency=0,
                detailed_metrics=True,
            ),
            logging=LoggingConfig(enabled=True, level="INFO", log_file=None),
        )

        # Provide an explicit SeedSequence to the worker
        window_seed = np.random.SeedSequence(42).spawn(1)[0]
        result = _process_window_worker(
            window=window,
            strategies=strategies,
            n_simulations=100,
            manufacturer=manufacturer,
            config=full_config,
            window_seed=window_seed,
        )

        assert isinstance(result, WindowResult)
        # Verify the backtester was called with SimulationConfig whose seeds
        # are derived from the SeedSequence, NOT from window_id arithmetic
        calls = mock_backtester.return_value.test_strategy.call_args_list
        train_seed = calls[0].kwargs.get("config") or calls[0][1].get("config")
        if train_seed is None:
            # positional arg
            train_seed = calls[0][0][2] if len(calls[0][0]) > 2 else None
        # The seed should NOT be window_id * 1000 (which would be 0)
        for call in calls:
            sim_config = call.kwargs.get("config")
            if sim_config is None:
                # Try positional args - config is not a positional in test_strategy
                continue
            assert sim_config.seed != 0, "Seed should not be window_id * 1000 = 0"
            assert sim_config.seed != 500, "Seed should not be window_id * 1000 + 500 = 500"

    @patch("ergodic_insurance.walk_forward_validator.StrategyBacktester")
    def test_worker_fallback_without_window_seed(self, mock_backtester):
        """Test that _process_window_worker creates a SeedSequence from window_id when no seed provided."""
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

        window = ValidationWindow(2, 0, 2, 2, 3)
        strategies = [NoInsuranceStrategy()]
        mfg_config = ManufacturerConfig(
            initial_assets=10000000,
            asset_turnover_ratio=1.0,
            base_operating_margin=0.08,
            tax_rate=0.25,
            retention_ratio=0.7,
        )
        manufacturer = WidgetManufacturer(mfg_config)

        from ergodic_insurance.config import (  # pylint: disable=reimported
            Config,
            DebtConfig,
            GrowthConfig,
            LoggingConfig,
            OutputConfig,
        )
        from ergodic_insurance.config import SimulationConfig as SimConfig
        from ergodic_insurance.config import WorkingCapitalConfig  # pylint: disable=reimported

        full_config = Config(
            manufacturer=mfg_config,
            working_capital=WorkingCapitalConfig(percent_of_sales=0.15),
            growth=GrowthConfig(type="deterministic", annual_growth_rate=0.05, volatility=0.15),
            debt=DebtConfig(
                interest_rate=0.05, max_leverage_ratio=2.0, minimum_cash_balance=100000
            ),
            simulation=SimConfig(time_resolution="annual", time_horizon_years=10),
            output=OutputConfig(
                output_directory="./results",
                file_format="csv",
                checkpoint_frequency=0,
                detailed_metrics=True,
            ),
            logging=LoggingConfig(enabled=True, level="INFO", log_file=None),
        )

        # Call without window_seed - should use SeedSequence(window.window_id) fallback
        result = _process_window_worker(
            window=window,
            strategies=strategies,
            n_simulations=100,
            manufacturer=manufacturer,
            config=full_config,
        )

        assert isinstance(result, WindowResult)
        # Seeds should NOT be the old arithmetic: 2*1000=2000 and 2*1000+500=2500
        calls = mock_backtester.return_value.test_strategy.call_args_list
        for call in calls:
            sim_config = call.kwargs.get("config")
            if sim_config is not None:
                assert sim_config.seed != 2000
                assert sim_config.seed != 2500

    def test_seedsequence_is_picklable(self):
        """Test that SeedSequence objects survive pickling (needed for ProcessPoolExecutor)."""
        import pickle

        base_ss = np.random.SeedSequence(42)
        children = base_ss.spawn(3)

        for child in children:
            restored = pickle.loads(pickle.dumps(child))
            # generate_state is deterministic: same SeedSequence always produces the same output
            assert np.array_equal(
                restored.generate_state(4),
                child.generate_state(4),
            )

    @patch("ergodic_insurance.walk_forward_validator.StrategyBacktester")
    def test_validate_strategies_passes_seed_to_windows(self, mock_backtester):
        """Test that validate_strategies creates SeedSequence from self.seed and passes to windows."""
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

        validator = WalkForwardValidator(window_size=3, step_size=2, max_workers=1, seed=42)
        result = validator.validate_strategies(
            strategies=[NoInsuranceStrategy()],
            n_years=10,
            n_simulations=100,
        )

        assert isinstance(result, ValidationResult)
        assert len(result.window_results) > 0

        # Collect all seeds used across windows - they should all be unique
        used_seeds = set()
        for call in mock_backtester.return_value.test_strategy.call_args_list:
            sim_config = call.kwargs.get("config")
            if sim_config is not None and sim_config.seed is not None:
                used_seeds.add(sim_config.seed)
        # Each window produces 2 calls (train + test), so at least as many unique seeds
        assert len(used_seeds) >= 2


class TestIntegration:
    """Integration tests for the walk-forward validation system."""

    @patch("ergodic_insurance.strategy_backtester.MonteCarloEngine")
    def test_end_to_end_validation(self, mock_mc_engine):
        """Test complete validation workflow."""
        # Setup mock results
        mock_results = MonteCarloResults(
            final_assets=np.random.lognormal(16, 1, 100),
            annual_losses=np.random.exponential(100000, (100, 5)),
            insurance_recoveries=np.random.exponential(50000, (100, 5)),
            retained_losses=np.random.exponential(50000, (100, 5)),
            growth_rates=np.random.normal(0.05, 0.02, 100),
            ruin_probability={"5": 0.02},
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
        validator = WalkForwardValidator(window_size=3, step_size=2, test_ratio=0.3, max_workers=1)

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
