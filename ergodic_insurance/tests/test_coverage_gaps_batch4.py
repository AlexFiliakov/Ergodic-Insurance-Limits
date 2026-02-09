"""Tests targeting untested code paths across multiple modules (batch 4).

Covers gaps in:
- business_optimizer.py (recommendation generators, objective evaluation, constraint building)
- walk_forward_validator.py (analysis branches, visualization, heatmap edge cases)
- excel_reporter.py (engine selection fallbacks, reconciliation formatting, pandas engine)
- reporting/report_builder.py (figure caching, data loading, PDF save, template fallback)
- manufacturer_solvency.py (ASC 205-40 going concern multi-factor assessment)

Author: Alex Filiakov
"""

from decimal import Decimal
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, PropertyMock, patch
import warnings

import matplotlib

matplotlib.use("Agg")

# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.business_optimizer import (
    BusinessConstraints,
    BusinessObjective,
    BusinessOptimizationResult,
    BusinessOptimizer,
    OptimalStrategy,
    OptimizationDirection,
)
from ergodic_insurance.claim_development import ClaimDevelopment
from ergodic_insurance.config import ManufacturerConfig
from ergodic_insurance.decimal_utils import ZERO, to_decimal
from ergodic_insurance.excel_reporter import (
    OPENPYXL_AVAILABLE,
    XLSXWRITER_AVAILABLE,
    ExcelReportConfig,
    ExcelReporter,
)
from ergodic_insurance.ledger import AccountName, TransactionType
from ergodic_insurance.manufacturer import ClaimLiability, WidgetManufacturer
from ergodic_insurance.reporting.config import (
    FigureConfig,
    ReportConfig,
    ReportMetadata,
    ReportStyle,
    SectionConfig,
    TableConfig,
)
from ergodic_insurance.reporting.report_builder import ReportBuilder
from ergodic_insurance.validation_metrics import StrategyPerformance, ValidationMetrics
from ergodic_insurance.walk_forward_validator import (
    ValidationResult,
    ValidationWindow,
    WalkForwardValidator,
    WindowResult,
)

# ---------------------------------------------------------------------------
# Module 1: business_optimizer.py
# ---------------------------------------------------------------------------


def _make_mock_manufacturer(
    total_assets: float = 10_000_000,
    equity: float = 4_000_000,
    revenue: float = 5_000_000,
) -> Mock:
    """Create a standard mock manufacturer used across optimizer tests."""
    manufacturer = Mock(spec=WidgetManufacturer)
    manufacturer.total_assets = total_assets
    manufacturer.equity = equity
    manufacturer.liabilities = total_assets - equity
    manufacturer.revenue = revenue
    manufacturer.operating_income = revenue * 0.10
    manufacturer.cash = total_assets * 0.20
    manufacturer.config = Mock()
    manufacturer.calculate_revenue = Mock(return_value=revenue)
    return manufacturer


class TestBusinessOptimizerRecommendations:
    """Test the three recommendation generators in business_optimizer.py.

    Targets untested lines 1086-1197.
    """

    @pytest.fixture
    def optimizer(self) -> BusinessOptimizer:
        return BusinessOptimizer(_make_mock_manufacturer())

    # -- _generate_roe_recommendations --

    def test_roe_recommendations_excellent_roe(self, optimizer: BusinessOptimizer):
        """Line 1086: expected_roe > 0.20 triggers 'Excellent ROE' message."""
        recs = optimizer._generate_roe_recommendations(
            coverage_limit=5_000_000,
            deductible=100_000,
            premium_rate=0.02,
            expected_roe=0.25,
        )
        assert any("Excellent ROE" in r for r in recs)

    def test_roe_recommendations_strong_roe(self, optimizer: BusinessOptimizer):
        """Lines 1089-1090: 0.15 < expected_roe <= 0.20 triggers 'Strong ROE'."""
        recs = optimizer._generate_roe_recommendations(
            coverage_limit=5_000_000,
            deductible=100_000,
            premium_rate=0.02,
            expected_roe=0.18,
        )
        assert any("Strong ROE" in r for r in recs)

    def test_roe_recommendations_below_target(self, optimizer: BusinessOptimizer):
        """Lines 1092-1094: expected_roe <= 0.15 triggers 'ROE below target'."""
        recs = optimizer._generate_roe_recommendations(
            coverage_limit=5_000_000,
            deductible=100_000,
            premium_rate=0.02,
            expected_roe=0.10,
        )
        assert any("ROE below target" in r for r in recs)

    def test_roe_recommendations_high_premium_rate(self, optimizer: BusinessOptimizer):
        """Line 1097: premium_rate > 0.05 triggers 'High premium rate'."""
        recs = optimizer._generate_roe_recommendations(
            coverage_limit=5_000_000,
            deductible=100_000,
            premium_rate=0.08,
            expected_roe=0.18,
        )
        assert any("High premium rate" in r for r in recs)

    def test_roe_recommendations_low_deductible(self, optimizer: BusinessOptimizer):
        """Line 1102: deductible < 50000 triggers 'Low deductible'."""
        recs = optimizer._generate_roe_recommendations(
            coverage_limit=5_000_000,
            deductible=30_000,
            premium_rate=0.02,
            expected_roe=0.18,
        )
        assert any("Low deductible" in r for r in recs)

    def test_roe_recommendations_high_deductible(self, optimizer: BusinessOptimizer):
        """Line 1106: deductible > 500_000 triggers 'High deductible exposes'."""
        recs = optimizer._generate_roe_recommendations(
            coverage_limit=5_000_000,
            deductible=600_000,
            premium_rate=0.02,
            expected_roe=0.18,
        )
        assert any("High deductible" in r for r in recs)

    def test_roe_recommendations_low_coverage_ratio(self, optimizer: BusinessOptimizer):
        """Line 1112: coverage_ratio < 0.5 triggers 'insufficient'."""
        recs = optimizer._generate_roe_recommendations(
            coverage_limit=3_000_000,  # 3M / 10M = 0.3
            deductible=100_000,
            premium_rate=0.02,
            expected_roe=0.18,
        )
        assert any("insufficient" in r.lower() for r in recs)

    def test_roe_recommendations_high_coverage_ratio(self, optimizer: BusinessOptimizer):
        """Line 1114: coverage_ratio > 1.5 triggers 'exceeds actual exposure'."""
        recs = optimizer._generate_roe_recommendations(
            coverage_limit=20_000_000,  # 20M / 10M = 2.0
            deductible=100_000,
            premium_rate=0.02,
            expected_roe=0.18,
        )
        assert any("exceeds" in r.lower() for r in recs)

    # -- _generate_risk_recommendations --

    def test_risk_recommendations_excellent_risk(self, optimizer: BusinessOptimizer):
        """Line 1129: bankruptcy_risk < 0.001 triggers 'Excellent risk profile'."""
        recs = optimizer._generate_risk_recommendations(
            coverage_limit=5_000_000,
            deductible=100_000,
            premium_rate=0.02,
            bankruptcy_risk=0.0005,
        )
        assert any("Excellent risk profile" in r for r in recs)

    def test_risk_recommendations_well_controlled(self, optimizer: BusinessOptimizer):
        """Lines 1130-1131: 0.001 <= risk < 0.01 triggers 'well-controlled'."""
        recs = optimizer._generate_risk_recommendations(
            coverage_limit=5_000_000,
            deductible=100_000,
            premium_rate=0.02,
            bankruptcy_risk=0.005,
        )
        assert any("well-controlled" in r for r in recs)

    def test_risk_recommendations_elevated(self, optimizer: BusinessOptimizer):
        """Line 1133: bankruptcy_risk >= 0.01 triggers 'Elevated bankruptcy risk'."""
        recs = optimizer._generate_risk_recommendations(
            coverage_limit=5_000_000,
            deductible=100_000,
            premium_rate=0.02,
            bankruptcy_risk=0.05,
        )
        assert any("Elevated bankruptcy risk" in r for r in recs)

    def test_risk_recommendations_insufficient_coverage(self, optimizer: BusinessOptimizer):
        """Line 1138: coverage_limit < total_assets * 0.5 triggers tail risk warning."""
        recs = optimizer._generate_risk_recommendations(
            coverage_limit=3_000_000,  # 3M / 10M = 0.3 < 0.5
            deductible=100_000,
            premium_rate=0.02,
            bankruptcy_risk=0.005,
        )
        assert any("tail risks" in r for r in recs)

    def test_risk_recommendations_high_premium_cost(self, optimizer: BusinessOptimizer):
        """Line 1141: premium > 3% of revenue triggers cost warning."""
        # premium_rate * coverage_limit > revenue * 0.03 => 0.05 * 5M = 250_000 > 5M * 0.03 = 150_000
        recs = optimizer._generate_risk_recommendations(
            coverage_limit=5_000_000,
            deductible=100_000,
            premium_rate=0.05,
            bankruptcy_risk=0.005,
        )
        assert any("3% of revenue" in r for r in recs)

    # -- _generate_comprehensive_recommendations --

    def test_comprehensive_exceptional_roe(self, optimizer: BusinessOptimizer):
        """Line 1159: roe > 0.20 triggers 'Exceptional ROE'."""
        recs = optimizer._generate_comprehensive_recommendations(
            coverage_limit=5_000_000,
            deductible=100_000,
            premium_rate=0.02,
            objective_values={"ROE": 0.25, "bankruptcy_risk": 0.005, "growth_rate": 0.10},
        )
        assert any("Exceptional ROE" in r for r in recs)

    def test_comprehensive_low_roe(self, optimizer: BusinessOptimizer):
        """Line 1161: roe < 0.10 triggers 'below industry standards'."""
        recs = optimizer._generate_comprehensive_recommendations(
            coverage_limit=5_000_000,
            deductible=100_000,
            premium_rate=0.02,
            objective_values={"ROE": 0.05, "bankruptcy_risk": 0.005, "growth_rate": 0.10},
        )
        assert any("below industry standards" in r for r in recs)

    def test_comprehensive_high_risk(self, optimizer: BusinessOptimizer):
        """Line 1166: risk > 0.02 triggers 'High bankruptcy risk'."""
        recs = optimizer._generate_comprehensive_recommendations(
            coverage_limit=5_000_000,
            deductible=100_000,
            premium_rate=0.02,
            objective_values={"ROE": 0.15, "bankruptcy_risk": 0.05, "growth_rate": 0.10},
        )
        assert any("High bankruptcy risk" in r for r in recs)

    def test_comprehensive_strong_growth(self, optimizer: BusinessOptimizer):
        """Line 1173: growth > 0.15 triggers 'Strong growth trajectory'."""
        recs = optimizer._generate_comprehensive_recommendations(
            coverage_limit=5_000_000,
            deductible=100_000,
            premium_rate=0.02,
            objective_values={"ROE": 0.15, "bankruptcy_risk": 0.005, "growth_rate": 0.20},
        )
        assert any("Strong growth" in r for r in recs)

    def test_comprehensive_low_insurance_spend(self, optimizer: BusinessOptimizer):
        """Line 1188: premium_to_revenue < 0.01 triggers 'Low insurance spend'."""
        # coverage_limit * premium_rate / revenue < 0.01 => 1M * 0.01 / 5M = 0.002
        recs = optimizer._generate_comprehensive_recommendations(
            coverage_limit=1_000_000,
            deductible=100_000,
            premium_rate=0.01,
            objective_values={"ROE": 0.15, "bankruptcy_risk": 0.005, "growth_rate": 0.10},
        )
        assert any("Low insurance spend" in r for r in recs)

    def test_comprehensive_high_deductible_to_assets(self, optimizer: BusinessOptimizer):
        """Line 1193: deductible_to_assets > 0.05 triggers retention capacity warning."""
        recs = optimizer._generate_comprehensive_recommendations(
            coverage_limit=5_000_000,
            deductible=600_000,  # 600k / 10M = 0.06
            premium_rate=0.02,
            objective_values={"ROE": 0.15, "bankruptcy_risk": 0.005, "growth_rate": 0.10},
        )
        assert any("retention capacity" in r for r in recs)

    def test_comprehensive_uses_lowercase_roe_key(self, optimizer: BusinessOptimizer):
        """Line 1156: objective_values uses lowercase 'roe' key."""
        recs = optimizer._generate_comprehensive_recommendations(
            coverage_limit=5_000_000,
            deductible=100_000,
            premium_rate=0.02,
            objective_values={"roe": 0.25, "bankruptcy_risk": 0.005, "growth_rate": 0.10},
        )
        assert any("Exceptional ROE" in r for r in recs)

    def test_comprehensive_limits_to_five_recommendations(self, optimizer: BusinessOptimizer):
        """Line 1197: result is capped at 5 recommendations."""
        # Trigger as many recommendations as possible
        recs = optimizer._generate_comprehensive_recommendations(
            coverage_limit=1_000_000,  # low coverage ratio
            deductible=600_000,  # high deductible to assets
            premium_rate=0.01,  # low premium
            objective_values={
                "ROE": 0.05,  # low ROE
                "bankruptcy_risk": 0.05,  # high risk
                "growth_rate": 0.20,  # strong growth
            },
        )
        assert len(recs) <= 5


class TestBusinessOptimizerObjectiveAndConstraints:
    """Test objective evaluation and constraint building branches.

    Targets lines 408, 515, 544, 829, 868, 923, 970.
    """

    @pytest.fixture
    def optimizer(self) -> BusinessOptimizer:
        return BusinessOptimizer(_make_mock_manufacturer())

    def test_evaluate_objective_capital_efficiency(self, optimizer: BusinessOptimizer):
        """Line 923: capital_efficiency objective path in _evaluate_objective."""
        result = optimizer._evaluate_objective("capital_efficiency", 5_000_000, 100_000, 0.02, 5)
        assert isinstance(result, float)
        assert result >= 0

    def test_estimate_growth_rate_equity_metric(self, optimizer: BusinessOptimizer):
        """Line 829: metric='equity' path in _estimate_growth_rate."""
        growth = optimizer._estimate_growth_rate(
            coverage_limit=5_000_000,
            deductible=100_000,
            premium_rate=0.02,
            time_horizon=5,
            metric="equity",
        )
        assert isinstance(growth, float)
        assert growth >= 0

    def test_estimate_growth_rate_assets_metric(self, optimizer: BusinessOptimizer):
        """Line 827: metric='assets' path in _estimate_growth_rate."""
        growth = optimizer._estimate_growth_rate(
            coverage_limit=5_000_000,
            deductible=100_000,
            premium_rate=0.02,
            time_horizon=5,
            metric="assets",
        )
        assert isinstance(growth, float)
        assert growth >= 0

    def test_calculate_ergodic_growth_without_analyzer(self, optimizer: BusinessOptimizer):
        """Line 868: no ergodic_analyzer falls back to _estimate_growth_rate."""
        assert optimizer.ergodic_analyzer is None
        growth = optimizer._calculate_ergodic_growth(5_000_000, 100_000, 0.02, 5)
        assert isinstance(growth, float)
        assert growth >= 0

    def test_analyze_time_horizon_default_horizons(self, optimizer: BusinessOptimizer):
        """Line 544: time_horizons=None defaults to [1, 3, 10, 30]."""
        strategies = [
            {
                "name": "Test",
                "coverage_limit": 5_000_000,
                "deductible": 100_000,
                "premium_rate": 0.02,
            }
        ]
        df = optimizer.analyze_time_horizon_impact(strategies=strategies, time_horizons=None)
        assert isinstance(df, pd.DataFrame)
        assert set(df["horizon_years"].unique()) == {1, 3, 10, 30}

    def test_constraint_equality_type(self, optimizer: BusinessOptimizer):
        """Line 970: constraint_type '==' path in _build_constraint_list."""
        objectives = [
            BusinessObjective(
                name="ROE",
                weight=1.0,
                constraint_type="==",
                constraint_value=0.15,
            )
        ]
        constraints = BusinessConstraints()
        constraint_list = optimizer._build_constraint_list(objectives, constraints, 5)
        # Find the equality constraint
        eq_constraints = [c for c in constraint_list if c["type"] == "eq"]
        assert len(eq_constraints) >= 1

    def test_risk_minimization_convergence_warning(self, optimizer: BusinessOptimizer):
        """Line 408: convergence warning from minimize_bankruptcy_risk when minimize fails."""
        with patch("scipy.optimize.minimize") as mock_minimize:
            mock_result = Mock()
            mock_result.success = False
            mock_result.message = "Did not converge"
            mock_result.x = np.array([5_000_000, 100_000, 0.02])
            mock_result.fun = 0.1
            mock_result.nit = 150
            mock_minimize.return_value = mock_result

            strategy = optimizer.minimize_bankruptcy_risk(
                growth_targets={"revenue": 0.10},
                budget_constraint=200_000,
                time_horizon=5,
            )
            assert isinstance(strategy, OptimalStrategy)

    def test_capital_allocation_convergence_warning(self, optimizer: BusinessOptimizer):
        """Line 515: convergence warning from optimize_capital_efficiency when minimize fails."""
        with patch("scipy.optimize.minimize") as mock_minimize:
            mock_result = Mock()
            mock_result.success = False
            mock_result.message = "Did not converge"
            mock_result.x = np.array([0.25, 0.25, 0.25, 0.25]) * 1_000_000
            mock_result.fun = -0.5
            mock_minimize.return_value = mock_result

            allocation = optimizer.optimize_capital_efficiency(
                available_capital=1_000_000,
                investment_opportunities={"growth": 0.25},
            )
            assert isinstance(allocation, dict)


# ---------------------------------------------------------------------------
# Module 2: walk_forward_validator.py
# ---------------------------------------------------------------------------


class TestWalkForwardValidatorGaps:
    """Test untested branches in walk_forward_validator.py.

    Targets lines 363, 401, 407, 432, 482, 701, 721, 847, 860-863, 891, 907, 945.
    """

    @pytest.fixture
    def validator(self) -> WalkForwardValidator:
        return WalkForwardValidator()

    def test_analyze_results_strategy_not_in_initial_list(self, validator: WalkForwardValidator):
        """Lines 400-401, 406-407: strategy in window_result but not in initial strategies list.

        This exercises the branch where a strategy_name found in window_result
        is not already in the strategy_metrics dict built from the strategies list.
        """
        window = ValidationWindow(0, 0, 2, 2, 3)
        window_result = WindowResult(window=window)

        # Add an "ExtraStrategy" that is not in the strategies list
        extra_perf = StrategyPerformance(
            strategy_name="ExtraStrategy",
            in_sample_metrics=ValidationMetrics(
                roe=0.15, ruin_probability=0.01, growth_rate=0.08, volatility=0.2
            ),
            out_sample_metrics=ValidationMetrics(
                roe=0.12, ruin_probability=0.02, growth_rate=0.07, volatility=0.22
            ),
        )
        extra_perf.calculate_degradation()
        window_result.strategy_performances["ExtraStrategy"] = extra_perf

        validation_result = ValidationResult(window_results=[window_result])
        # Provide strategies list that does NOT include ExtraStrategy
        strategy_mock = Mock()
        strategy_mock.name = "OriginalStrategy"
        validator._analyze_results(validation_result, [strategy_mock])

        # ExtraStrategy should still be analyzed
        assert (
            "ExtraStrategy" in validation_result.consistency_scores
            or "ExtraStrategy" in validation_result.overfitting_analysis
        )

    def test_analyze_results_single_roe_consistency(self, validator: WalkForwardValidator):
        """Line 432: single ROE value sets consistency to 1.0 (len(roes) <= 1)."""
        window = ValidationWindow(0, 0, 2, 2, 3)
        window_result = WindowResult(window=window)

        perf = StrategyPerformance(
            strategy_name="SingleWindow",
            in_sample_metrics=ValidationMetrics(
                roe=0.15, ruin_probability=0.01, growth_rate=0.08, volatility=0.2
            ),
            out_sample_metrics=ValidationMetrics(
                roe=0.12, ruin_probability=0.02, growth_rate=0.07, volatility=0.22
            ),
        )
        perf.calculate_degradation()
        window_result.strategy_performances["SingleWindow"] = perf

        validation_result = ValidationResult(window_results=[window_result])
        strategy_mock = Mock()
        strategy_mock.name = "SingleWindow"
        validator._analyze_results(validation_result, [strategy_mock])

        # With a single window, consistency should be 1.0
        assert validation_result.consistency_scores["SingleWindow"] == 1.0

    def test_average_metrics_empty_list(self, validator: WalkForwardValidator):
        """Line 482: _average_metrics with empty list returns zeros."""
        result = validator._average_metrics([])
        assert result.roe == 0
        assert result.ruin_probability == 0
        assert result.growth_rate == 0
        assert result.volatility == 0

    def test_html_report_empty_rankings(self, validator: WalkForwardValidator):
        """Line 701: empty strategy_rankings produces 'No rankings available'."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            validation_result = ValidationResult()
            validation_result.best_strategy = "Test"
            validation_result.strategy_rankings = pd.DataFrame()

            html_path = Path(tmp_dir) / "test.html"
            validator._generate_html_report(
                validation_result, html_path, include_visualizations=False
            )
            content = html_path.read_text()
            assert "No rankings available" in content

    def test_html_report_row_class_warning(self, validator: WalkForwardValidator):
        """Lines 718-721: overfitting_score > 0.2 but < 0.4 produces 'warning' row class."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            validation_result = ValidationResult()
            validation_result.best_strategy = "TestStrat"
            validation_result.strategy_rankings = pd.DataFrame(
                [{"strategy": "TestStrat", "avg_roe": 0.12}]
            )

            window = ValidationWindow(0, 0, 2, 2, 3)
            window_result = WindowResult(window=window)
            perf = StrategyPerformance(
                strategy_name="TestStrat",
                in_sample_metrics=ValidationMetrics(
                    roe=0.15, ruin_probability=0.01, growth_rate=0.08, volatility=0.2
                ),
                out_sample_metrics=ValidationMetrics(
                    roe=0.10, ruin_probability=0.03, growth_rate=0.05, volatility=0.3
                ),
            )
            perf.calculate_degradation()
            # Force overfitting score to be between 0.2 and 0.4 for "warning"
            perf.overfitting_score = 0.3
            window_result.strategy_performances["TestStrat"] = perf
            validation_result.window_results = [window_result]

            html_path = Path(tmp_dir) / "test_warn.html"
            validator._generate_html_report(
                validation_result, html_path, include_visualizations=False
            )
            content = html_path.read_text()
            assert 'class="warning"' in content

    def test_plot_overfitting_empty(self, validator: WalkForwardValidator):
        """Line 847: empty overfitting_analysis returns None."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            validation_result = ValidationResult()
            validation_result.overfitting_analysis = {}
            result = validator._plot_overfitting_analysis(validation_result, Path(tmp_dir))
            assert result is None

    def test_plot_overfitting_colored_bars(self, validator: WalkForwardValidator):
        """Lines 858-863: bars colored green (<0.2), orange (0.2-0.4), red (>=0.4)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            validation_result = ValidationResult()
            validation_result.overfitting_analysis = {
                "GreenStrat": 0.1,
                "OrangeStrat": 0.3,
                "RedStrat": 0.5,
            }
            result = validator._plot_overfitting_analysis(validation_result, Path(tmp_dir))
            assert result is not None
            assert result.exists()

    def test_plot_heatmap_empty_rankings(self, validator: WalkForwardValidator):
        """Line 891: empty strategy_rankings returns None."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            validation_result = ValidationResult()
            validation_result.strategy_rankings = pd.DataFrame()
            result = validator._plot_strategy_ranking_heatmap(validation_result, Path(tmp_dir))
            assert result is None

    def test_plot_heatmap_no_ranking_columns(self, validator: WalkForwardValidator):
        """Lines 905-907, 945: rankings present but no recognized ranking columns => returns None."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            validation_result = ValidationResult()
            # DataFrame has 'strategy' but none of the possible_cols
            validation_result.strategy_rankings = pd.DataFrame(
                [{"strategy": "StratA", "custom_metric": 0.5}]
            )
            result = validator._plot_strategy_ranking_heatmap(validation_result, Path(tmp_dir))
            assert result is None

    def test_process_window_with_optimized_strategy(self, validator: WalkForwardValidator):
        """Line 363: strategy with optimized_params captures optimization params."""
        from ergodic_insurance.monte_carlo import SimulationConfig
        from ergodic_insurance.strategy_backtester import BacktestResult, NoInsuranceStrategy

        mock_metrics = ValidationMetrics(
            roe=0.12, ruin_probability=0.02, growth_rate=0.06, volatility=0.2
        )
        mock_result = BacktestResult(
            strategy_name="Optimized",
            simulation_results=Mock(),
            metrics=mock_metrics,
            execution_time=0.5,
            config=SimulationConfig(),
        )

        with patch.object(validator, "backtester") as mock_bt:
            mock_bt.test_strategy.return_value = mock_result

            strategy = Mock()
            strategy.name = "Optimized"
            strategy.optimized_params = {"deductible": 50000, "limit": 5_000_000}

            window = ValidationWindow(0, 0, 2, 2, 3)
            mfg_config = ManufacturerConfig(
                initial_assets=10_000_000,
                asset_turnover_ratio=1.0,
                base_operating_margin=0.08,
                tax_rate=0.25,
                retention_ratio=0.7,
            )
            manufacturer = WidgetManufacturer(mfg_config)

            from ergodic_insurance.config import Config, DebtConfig
            from ergodic_insurance.config import GrowthConfig as GC
            from ergodic_insurance.config import LoggingConfig, OutputConfig
            from ergodic_insurance.config import SimulationConfig as SimConfig
            from ergodic_insurance.config import WorkingCapitalConfig

            full_config = Config(
                manufacturer=mfg_config,
                working_capital=WorkingCapitalConfig(percent_of_sales=0.15),
                growth=GC(type="deterministic", annual_growth_rate=0.05, volatility=0.15),
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
                strategies=[strategy],
                n_simulations=100,
                manufacturer=manufacturer,
                config=full_config,
            )
            assert "Optimized" in result.optimization_params


# ---------------------------------------------------------------------------
# Module 3: excel_reporter.py
# ---------------------------------------------------------------------------


class TestExcelReporterGaps:
    """Test untested branches in excel_reporter.py.

    Targets lines 140-144, 164-166, 508-509, 572-580, 589, 698, 982, 1044.
    """

    def test_select_engine_auto_openpyxl_fallback(self):
        """Lines 140-141: auto engine with xlsxwriter unavailable falls to openpyxl."""
        config = ExcelReportConfig(engine="auto")
        reporter = ExcelReporter(config)
        with (
            patch("ergodic_insurance.excel_reporter.XLSXWRITER_AVAILABLE", False),
            patch("ergodic_insurance.excel_reporter.OPENPYXL_AVAILABLE", True),
        ):
            reporter._select_engine()
            assert reporter.engine == "openpyxl"

    def test_select_engine_auto_no_libraries(self):
        """Lines 142-144: auto engine with both libraries unavailable falls to pandas."""
        config = ExcelReportConfig(engine="auto")
        reporter = ExcelReporter(config)
        with (
            patch("ergodic_insurance.excel_reporter.XLSXWRITER_AVAILABLE", False),
            patch("ergodic_insurance.excel_reporter.OPENPYXL_AVAILABLE", False),
        ):
            reporter._select_engine()
            assert reporter.engine == "pandas"

    def test_get_pandas_engine_xlsxwriter_only(self):
        """Lines 164-165: pandas fallback with only xlsxwriter available."""
        config = ExcelReportConfig(engine="pandas")
        reporter = ExcelReporter(config)
        reporter.engine = "pandas"
        with (
            patch("ergodic_insurance.excel_reporter.OPENPYXL_AVAILABLE", False),
            patch("ergodic_insurance.excel_reporter.XLSXWRITER_AVAILABLE", True),
        ):
            result = reporter._get_pandas_engine()
            assert result == "xlsxwriter"

    def test_get_pandas_engine_none_available(self):
        """Line 166: no library available returns None."""
        config = ExcelReportConfig(engine="pandas")
        reporter = ExcelReporter(config)
        reporter.engine = "pandas"
        with (
            patch("ergodic_insurance.excel_reporter.OPENPYXL_AVAILABLE", False),
            patch("ergodic_insurance.excel_reporter.XLSXWRITER_AVAILABLE", False),
        ):
            result = reporter._get_pandas_engine()
            assert result is None

    def test_categorize_metric_number_format(self):
        """Line 698: metric that falls through to 'number' format (not % or currency)."""
        reporter = ExcelReporter()
        # This metric name should categorize as "Other" and use number format
        category = reporter._categorize_metric("count_of_simulations")
        assert category == "Other"


# ---------------------------------------------------------------------------
# Module 4: reporting/report_builder.py
# ---------------------------------------------------------------------------


def _make_test_report_class():
    """Create a concrete test subclass of ReportBuilder."""

    class TestReport(ReportBuilder):
        def generate(self) -> Path:
            return self.save("markdown")

    return TestReport


class TestReportBuilderGaps:
    """Test untested branches in report_builder.py.

    Targets lines 80, 185, 220-222, 292, 325, 447, 454.
    """

    def test_template_dir_not_exists_fallback(self):
        """Line 80: template_dir does not exist falls back to empty Environment."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ReportConfig(
                metadata=ReportMetadata(title="Test"),
                sections=[],
                output_dir=Path(tmp_dir) / "output",
                cache_dir=Path(tmp_dir) / "cache",
            )
            TestReport = _make_test_report_class()

            # Patch the template_dir to a non-existent path
            with patch.object(Path, "exists", return_value=False):
                report = TestReport(config)
                # env should still be created (empty Environment)
                assert report.env is not None

    def test_embed_figure_cache_hit(self):
        """Line 185: figure with cache_key that already exists on disk."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ReportConfig(
                metadata=ReportMetadata(title="Test"),
                sections=[],
                output_dir=Path(tmp_dir) / "output",
                cache_dir=Path(tmp_dir) / "cache",
            )
            TestReport = _make_test_report_class()
            report = TestReport(config)

            # Pre-create the cached figure
            cached_fig = report.cache_dir / "test_cache_key.png"
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])
            fig.savefig(cached_fig)
            plt.close(fig)

            fig_config = FigureConfig(
                name="test_fig",
                caption="Test Figure",
                source="nonexistent.png",
                cache_key="test_cache_key",
            )
            result = report._embed_figure(fig_config)
            assert "Test Figure" in result
            assert len(report.figures) == 1

    def test_generate_figure_from_existing_file(self):
        """Lines 220-222: source is an existing file path, copy to cache."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ReportConfig(
                metadata=ReportMetadata(title="Test"),
                sections=[],
                output_dir=Path(tmp_dir) / "output",
                cache_dir=Path(tmp_dir) / "cache",
            )
            TestReport = _make_test_report_class()
            report = TestReport(config)

            # Create a real source image file
            source_file = Path(tmp_dir) / "source_image.png"
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])
            fig.savefig(source_file)
            plt.close(fig)

            fig_config = FigureConfig(
                name="from_file",
                caption="From File",
                source=str(source_file),
            )
            result_path = report._generate_figure(fig_config)
            assert result_path.exists()
            assert result_path.parent == report.cache_dir

    def test_load_table_data_parquet(self):
        """Line 292: load table data from a .parquet file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ReportConfig(
                metadata=ReportMetadata(title="Test"),
                sections=[],
                output_dir=Path(tmp_dir) / "output",
                cache_dir=Path(tmp_dir) / "cache",
            )
            TestReport = _make_test_report_class()
            report = TestReport(config)

            # Create a parquet file
            parquet_path = Path(tmp_dir) / "test_data.parquet"
            df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
            df.to_parquet(parquet_path)

            result = report._load_table_data(str(parquet_path))
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert "A" in result.columns

    def test_compile_report_with_footer(self):
        """Line 325: _generate_footer returns non-empty string so footer is appended."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ReportConfig(
                metadata=ReportMetadata(title="Test"),
                sections=[SectionConfig(title="Sec", level=1, content="Content")],
                output_dir=Path(tmp_dir) / "output",
                cache_dir=Path(tmp_dir) / "cache",
            )
            TestReport = _make_test_report_class()
            report = TestReport(config)

            # Override _generate_footer to return non-empty
            report._generate_footer = lambda: "--- END OF REPORT ---"
            compiled = report.compile_report()
            assert "--- END OF REPORT ---" in compiled

    def test_save_pdf_falls_back_to_html(self):
        """Lines 447-454: save('pdf') with weasyprint unavailable falls back to HTML."""
        pytest.importorskip("markdown2", reason="markdown2 not installed")
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ReportConfig(
                metadata=ReportMetadata(title="PDF Test"),
                sections=[SectionConfig(title="Sec", level=1, content="Content")],
                output_dir=Path(tmp_dir) / "output",
                cache_dir=Path(tmp_dir) / "cache",
            )
            TestReport = _make_test_report_class()
            report = TestReport(config)

            # Make weasyprint import fail
            import builtins

            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "weasyprint":
                    raise ImportError("mocked")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                result_path = report.save("pdf")
                # Should fallback to .html since weasyprint is unavailable
                assert result_path.exists()
                assert result_path.suffix == ".html"

    def test_save_unsupported_format_raises(self):
        """Line 457: unsupported format raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ReportConfig(
                metadata=ReportMetadata(title="Test"),
                sections=[],
                output_dir=Path(tmp_dir) / "output",
                cache_dir=Path(tmp_dir) / "cache",
            )
            TestReport = _make_test_report_class()
            report = TestReport(config)

            with pytest.raises(ValueError, match="Unsupported format"):
                report.save("docx")


# ---------------------------------------------------------------------------
# Module 5: manufacturer_solvency.py
# ---------------------------------------------------------------------------


class TestManufacturerSolvencyGoingConcern:
    """Test ASC 205-40 multi-factor going concern assessment in check_solvency().

    Targets the Tier 2 going concern path in manufacturer_solvency.py
    where breaching N-of-4 indicators triggers insolvency (Issue #489).
    """

    @pytest.fixture
    def config(self) -> ManufacturerConfig:
        return ManufacturerConfig(
            initial_assets=10_000_000,
            asset_turnover_ratio=0.5,  # revenue = 5M
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.7,
        )

    def _make_front_loaded_claim(self, amount: int, year: int = 0) -> ClaimLiability:
        """Create a ClaimLiability with a heavily front-loaded development strategy.

        Uses 60% payment in year 0 so a moderate total amount can still
        produce large debt service while keeping equity positive.
        """
        front_loaded = ClaimDevelopment(
            pattern_name="FRONT_LOADED",
            development_factors=[0.60, 0.10, 0.10, 0.05, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01],
        )
        return ClaimLiability(
            original_amount=to_decimal(amount),
            remaining_amount=to_decimal(amount),
            year_incurred=year,
            development_strategy=front_loaded,
        )

    def test_multi_factor_triggers_insolvency(self, config: ManufacturerConfig):
        """Multi-factor going concern: breaching 2+ indicators triggers insolvency.

        Uses configurable thresholds to create a scenario where DSCR and
        equity ratio are both breached while equity remains positive
        (so the Tier 1 hard stop does not fire first).

        Scenario with 2M assets, 1.8M claim:
        - DSCR = 100K / 1.08M = 0.093 < 1.0 (BREACH)
        - Equity ratio = 200K / 2M = 10% < 15% (BREACH with raised threshold)
        - 2-of-4 default triggers insolvency.
        """
        tight_config = ManufacturerConfig(
            initial_assets=2_000_000,
            asset_turnover_ratio=0.5,  # revenue = 1M
            base_operating_margin=0.10,  # operating income = 100K
            tax_rate=0.25,
            retention_ratio=0.7,
            # Raise equity ratio threshold so 10% equity ratio breaches
            going_concern_min_equity_ratio=0.15,
        )
        manufacturer = WidgetManufacturer(tight_config)
        assert manufacturer.check_solvency()
        assert not manufacturer.is_ruined

        claim = self._make_front_loaded_claim(1_800_000)
        manufacturer.claim_liabilities.append(claim)

        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.INSURANCE_LOSS,
            credit_account=AccountName.CLAIM_LIABILITIES,
            amount=to_decimal(1_800_000),
            transaction_type=TransactionType.INSURANCE_CLAIM,
            description="Test claim liability for going concern assessment",
        )

        # Equity should still be positive (not a Tier 1 hard stop)
        assert (
            manufacturer.equity > ZERO
        ), f"Equity should be positive for this test, got {manufacturer.equity}"

        # Multi-factor assessment should trigger insolvency (2+ indicators breached)
        result = manufacturer.check_solvency()
        assert result is False
        assert manufacturer.is_ruined is True

    def test_single_indicator_breach_remains_solvent(self, config: ManufacturerConfig):
        """Single indicator breach does not trigger insolvency under 2-of-4 default.

        A large claim breaches DSCR but other indicators remain healthy,
        so the company stays solvent.
        """
        manufacturer = WidgetManufacturer(config)

        # Claim of 8M: DSCR = 500K / 4.8M = 0.10 < 1.0 (BREACH)
        # But equity ratio = 2M/10M = 20% > 5% (OK)
        # Current ratio: no current liabilities (claim offsets total) => Inf (OK)
        # Cash runway: high assets keep cash healthy (OK)
        # Only 1-of-4 breached => remains solvent
        claim = self._make_front_loaded_claim(8_000_000)
        manufacturer.claim_liabilities.append(claim)

        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.INSURANCE_LOSS,
            credit_account=AccountName.CLAIM_LIABILITIES,
            amount=to_decimal(8_000_000),
            transaction_type=TransactionType.INSURANCE_CLAIM,
            description="Test claim liability for single indicator breach",
        )

        assert manufacturer.equity > ZERO

        result = manufacturer.check_solvency()
        assert result is True
        assert manufacturer.is_ruined is False

    def test_already_ruined_multi_factor(self, config: ManufacturerConfig):
        """When is_ruined is already True and multi-factor triggers, returns False.

        Tests that a pre-ruined company with 2+ indicator breaches still
        returns False without duplicate warning logging.
        """
        tight_config = ManufacturerConfig(
            initial_assets=2_000_000,
            asset_turnover_ratio=0.5,
            base_operating_margin=0.10,
            tax_rate=0.25,
            retention_ratio=0.7,
            going_concern_min_equity_ratio=0.15,
        )
        manufacturer = WidgetManufacturer(tight_config)

        claim = self._make_front_loaded_claim(1_800_000)
        manufacturer.claim_liabilities.append(claim)

        manufacturer.ledger.record_double_entry(
            date=manufacturer.current_year,
            debit_account=AccountName.INSURANCE_LOSS,
            credit_account=AccountName.CLAIM_LIABILITIES,
            amount=to_decimal(1_800_000),
            transaction_type=TransactionType.INSURANCE_CLAIM,
            description="Test claim liability for already-ruined going concern",
        )

        # Pre-set is_ruined
        manufacturer.is_ruined = True

        result = manufacturer.check_solvency()
        assert result is False
        assert manufacturer.is_ruined is True
