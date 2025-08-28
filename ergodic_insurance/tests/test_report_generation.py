"""Comprehensive tests for the report generation system.

This module tests all components of the automated report generation system
including configuration, table generation, report builders, and validation.
"""

from datetime import datetime
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from ergodic_insurance.src.reporting.config import (
    FigureConfig,
    ReportConfig,
    ReportMetadata,
    ReportStyle,
    SectionConfig,
    TableConfig,
    create_executive_config,
    create_technical_config,
)
from ergodic_insurance.src.reporting.executive_report import ExecutiveReport
from ergodic_insurance.src.reporting.report_builder import ReportBuilder
from ergodic_insurance.src.reporting.table_generator import (
    TableGenerator,
    create_parameter_table,
    create_performance_table,
    create_sensitivity_table,
)
from ergodic_insurance.src.reporting.technical_report import TechnicalReport
from ergodic_insurance.src.reporting.validator import (
    ReportValidator,
    validate_parameters,
    validate_results_data,
)


class TestReportConfig:
    """Test report configuration classes."""

    def test_create_report_metadata(self):
        """Test ReportMetadata creation."""
        metadata = ReportMetadata(
            title="Test Report",
            subtitle="Test Subtitle",
            authors=["Author 1", "Author 2"],
            date=datetime(2024, 1, 1),
            version="1.0.0",
            organization="Test Org",
            confidentiality="Internal",
            keywords=["test", "report"],
            abstract="Test abstract",
        )

        assert metadata.title == "Test Report"
        assert metadata.subtitle == "Test Subtitle"
        assert len(metadata.authors) == 2
        assert metadata.version == "1.0.0"
        assert metadata.confidentiality == "Internal"
        assert len(metadata.keywords) == 2

    def test_create_figure_config(self):
        """Test FigureConfig creation and validation."""
        fig_config = FigureConfig(
            name="test_figure",
            caption="Test Figure",
            source="test.png",
            width=6.5,
            height=4.0,
            dpi=300,
        )

        assert fig_config.name == "test_figure"
        assert fig_config.caption == "Test Figure"
        assert fig_config.width == 6.5
        assert fig_config.height == 4.0
        assert fig_config.dpi == 300

    def test_figure_config_validation(self):
        """Test FigureConfig validation."""
        # Valid image formats
        for ext in [".png", ".jpg", ".pdf", ".svg"]:
            fig = FigureConfig(name="test", caption="Test", source=f"test{ext}")
            assert fig.source == f"test{ext}"

        # Invalid format should raise error
        with pytest.raises(ValueError, match="Invalid figure format"):
            FigureConfig(name="test", caption="Test", source="test.txt")

    def test_create_table_config(self):
        """Test TableConfig creation."""
        table_config = TableConfig(
            name="test_table",
            caption="Test Table",
            data_source="data.csv",
            format="markdown",
            columns=["A", "B", "C"],
            index=False,
            precision=2,
        )

        assert table_config.name == "test_table"
        assert table_config.caption == "Test Table"
        assert table_config.format == "markdown"
        assert len(table_config.columns) == 3
        assert table_config.precision == 2

    def test_create_section_config(self):
        """Test SectionConfig creation with subsections."""
        subsection = SectionConfig(title="Subsection", level=2, content="Subsection content")

        section = SectionConfig(
            title="Main Section",
            level=1,
            content="Main content",
            figures=[],
            tables=[],
            subsections=[subsection],
            page_break=True,
        )

        assert section.title == "Main Section"
        assert section.level == 1
        assert section.page_break is True
        assert len(section.subsections) == 1
        assert section.subsections[0].title == "Subsection"

    def test_create_report_config(self):
        """Test complete ReportConfig creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReportConfig(
                metadata=ReportMetadata(title="Test Report"),
                style=ReportStyle(),
                sections=[
                    SectionConfig(title="Section 1", level=1),
                    SectionConfig(title="Section 2", level=1),
                ],
                template="executive",
                output_formats=["markdown", "html"],
                output_dir=Path(tmpdir) / "output",
                cache_dir=Path(tmpdir) / "cache",
            )

            assert config.metadata.title == "Test Report"
            assert len(config.sections) == 2
            assert config.template == "executive"
            assert len(config.output_formats) == 2
            assert config.output_dir.exists()
            assert config.cache_dir.exists()

    def test_default_configs(self):
        """Test default configuration generators."""
        exec_config = create_executive_config()
        assert exec_config.template == "executive"
        assert exec_config.metadata.subtitle == "Executive Summary"
        assert len(exec_config.sections) > 0

        tech_config = create_technical_config()
        assert tech_config.template == "technical"
        assert tech_config.metadata.subtitle == "Technical Appendix"
        assert len(tech_config.sections) > 0


class TestTableGenerator:
    """Test table generation functionality."""

    def test_create_table_generator(self):
        """Test TableGenerator initialization."""
        gen = TableGenerator(default_format="markdown", precision=3, max_width=60)
        assert gen.default_format == "markdown"
        assert gen.precision == 3
        assert gen.max_width == 60

    def test_generate_markdown_table(self):
        """Test markdown table generation."""
        gen = TableGenerator()
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4.123, 5.456, 6.789]})

        table = gen.generate(df, caption="Test Table", format="markdown")
        assert "Test Table" in table
        assert "|" in table  # Markdown table format
        assert "4.12" in table  # Check precision

    def test_generate_html_table(self):
        """Test HTML table generation."""
        gen = TableGenerator()
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

        table = gen.generate(df, format="html")
        assert "<table" in table or "<td>" in table

    def test_generate_with_column_selection(self):
        """Test table generation with column selection."""
        gen = TableGenerator()
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})

        table = gen.generate(df, columns=["A", "C"])
        assert "3" not in table and "4" not in table  # Column B excluded

    def test_generate_summary_statistics(self):
        """Test summary statistics table generation."""
        gen = TableGenerator()
        df = pd.DataFrame({"X": np.random.randn(100), "Y": np.random.randn(100) * 2 + 5})

        summary = gen.generate_summary_statistics(df)
        assert "Summary Statistics" in summary
        assert "mean" in summary.lower()
        assert "std" in summary.lower()

    def test_generate_decision_matrix(self):
        """Test decision matrix generation."""
        gen = TableGenerator()

        alternatives = ["Option A", "Option B", "Option C"]
        criteria = ["Cost", "Quality", "Time"]
        scores = np.array([[0.8, 0.6, 0.7], [0.5, 0.9, 0.6], [0.7, 0.7, 0.8]])
        weights = [0.4, 0.4, 0.2]

        matrix = gen.generate_decision_matrix(alternatives, criteria, scores, weights)

        assert "Decision Matrix" in matrix
        assert "Option A" in matrix
        assert "Weighted Total" in matrix

    def test_create_performance_table(self):
        """Test performance table creation."""
        results = {
            "roe": 0.18,
            "ruin_prob": 0.01,
            "growth_rate": 0.07,
            "sharpe": 1.2,
            "max_drawdown": 0.15,
        }

        table = create_performance_table(results)
        assert "Performance Metrics" in table
        assert "0.18" in table or "18" in table
        assert "✓" in table or "⚠" in table

    def test_create_parameter_table(self):
        """Test parameter table creation."""
        params = {
            "financial": {"tax_rate": 0.25, "margin": 0.08},
            "simulation": {"years": 10, "paths": 1000},
        }

        table = create_parameter_table(params)
        assert "Model Parameters" in table
        assert "financial" in table.lower() or "Financial" in table
        assert "0.25" in table

    def test_create_sensitivity_table(self):
        """Test sensitivity analysis table creation."""
        base_case = 0.18
        sensitivities = {"Premium": [0.16, 0.18, 0.20], "Volatility": [0.17, 0.18, 0.19]}
        ranges = {"Premium": [0.02, 0.03, 0.04], "Volatility": [0.10, 0.15, 0.20]}

        table = create_sensitivity_table(base_case, sensitivities, ranges)
        assert "Sensitivity Analysis" in table
        assert "Premium" in table
        assert "Change from Base" in table


class TestReportBuilder:
    """Test base report builder functionality."""

    def test_report_builder_initialization(self):
        """Test ReportBuilder initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReportConfig(
                metadata=ReportMetadata(title="Test"),
                sections=[],
                output_dir=Path(tmpdir) / "output",
                cache_dir=Path(tmpdir) / "cache",
            )

            # Create concrete implementation for testing
            class TestReport(ReportBuilder):
                def generate(self):
                    return Path("test.md")

            report = TestReport(config)
            assert report.config == config
            assert report.cache_manager is not None
            assert report.table_generator is not None
            assert len(report.content) == 0
            assert len(report.figures) == 0
            assert len(report.tables) == 0

    def test_build_section(self):
        """Test section building."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReportConfig(
                metadata=ReportMetadata(title="Test"),
                sections=[],
                output_dir=Path(tmpdir) / "output",
                cache_dir=Path(tmpdir) / "cache",
            )

            class TestReport(ReportBuilder):
                def generate(self):
                    return Path("test.md")

            report = TestReport(config)

            section = SectionConfig(
                title="Test Section", level=2, content="Test content", page_break=True
            )

            result = report.build_section(section)
            assert "## Test Section" in result
            assert "Test content" in result
            assert "\\newpage" in result

    def test_compile_report(self):
        """Test report compilation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReportConfig(
                metadata=ReportMetadata(
                    title="Test Report", authors=["Test Author"], date=datetime(2024, 1, 1)
                ),
                sections=[
                    SectionConfig(title="Section 1", level=1, content="Content 1"),
                    SectionConfig(title="Section 2", level=1, content="Content 2"),
                ],
                output_dir=Path(tmpdir) / "output",
                cache_dir=Path(tmpdir) / "cache",
            )

            class TestReport(ReportBuilder):
                def generate(self):
                    return Path("test.md")

            report = TestReport(config)
            compiled = report.compile_report()

            assert "# Test Report" in compiled
            assert "Test Author" in compiled
            assert "# Section 1" in compiled
            assert "Content 1" in compiled
            assert "# Section 2" in compiled
            assert "Content 2" in compiled

    def test_save_markdown(self):
        """Test saving report as markdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReportConfig(
                metadata=ReportMetadata(title="Test Report"),
                sections=[SectionConfig(title="Test", level=1)],
                output_dir=Path(tmpdir),
                cache_dir=Path(tmpdir) / "cache",
            )

            class TestReport(ReportBuilder):
                def generate(self):
                    return self.save("markdown")

            report = TestReport(config)
            path = report.save("markdown")

            assert path.exists()
            assert path.suffix == ".md"
            content = path.read_text()
            assert "Test Report" in content

    def test_save_html(self):
        """Test saving report as HTML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ReportConfig(
                metadata=ReportMetadata(title="Test Report"),
                sections=[SectionConfig(title="Test", level=1)],
                output_dir=Path(tmpdir),
                cache_dir=Path(tmpdir) / "cache",
            )

            class TestReport(ReportBuilder):
                def generate(self):
                    return self.save("html")

            report = TestReport(config)
            path = report.save("html")

            assert path.exists()
            assert path.suffix == ".html"
            content = path.read_text()
            assert "<html>" in content.lower()
            assert "Test Report" in content


class TestExecutiveReport:
    """Test executive report generation."""

    def test_create_executive_report(self):
        """Test ExecutiveReport creation."""
        results = {"roe": 0.18, "ruin_probability": 0.01, "trajectories": np.random.randn(100, 100)}

        with tempfile.TemporaryDirectory() as tmpdir:
            report = ExecutiveReport(results=results, cache_dir=Path(tmpdir))

            assert report.results == results
            assert "roe" in report.key_metrics
            assert report.key_metrics["roe"] == 0.18

    def test_extract_key_metrics(self):
        """Test key metrics extraction."""
        results = {
            "roe": 0.18,
            "roe_baseline": 0.15,
            "ruin_probability": 0.01,
            "ruin_probability_baseline": 0.03,
            "growth_rate": 0.07,
            "optimal_limits": [5e6, 15e6],
            "total_premium": 500000,
            "expected_losses": 200000,
            "trajectories": np.random.randn(100, 100),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            report = ExecutiveReport(results, cache_dir=Path(tmpdir))
            metrics = report.key_metrics

            assert metrics["roe"] == 0.18
            assert metrics["roe_improvement"] == pytest.approx(20.0, rel=0.1)
            assert metrics["ruin_prob"] == 0.01
            assert metrics["ruin_reduction"] == pytest.approx(66.67, rel=0.1)
            assert metrics["premium_ratio"] == 2.5

    def test_generate_decision_matrix(self):
        """Test decision matrix generation."""
        results = {"roe": 0.18}

        with tempfile.TemporaryDirectory() as tmpdir:
            report = ExecutiveReport(results, cache_dir=Path(tmpdir))
            matrix = report.generate_decision_matrix()

            assert isinstance(matrix, pd.DataFrame)
            assert "Weighted Score" in matrix.columns
            assert len(matrix) == 4  # Four alternatives


class TestTechnicalReport:
    """Test technical report generation."""

    def test_create_technical_report(self):
        """Test TechnicalReport creation."""
        results = {
            "trajectories": np.random.randn(100, 100),
            "simulated_losses": np.random.lognormal(10, 2, 1000),
        }

        parameters = {
            "financial": {"tax_rate": 0.25},
            "insurance": {"premium_rate": 0.03},
            "simulation": {"years": 10, "num_simulations": 1000},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            report = TechnicalReport(results=results, parameters=parameters, cache_dir=Path(tmpdir))

            assert report.results == results
            assert report.parameters == parameters
            assert isinstance(report.validation_metrics, dict)

    def test_generate_model_parameters_table(self):
        """Test model parameters table generation."""
        results = {}
        parameters = {
            "financial": {"initial_assets": 10000000, "tax_rate": 0.25},
            "insurance": {"premium_rate": 0.03, "deductible": 50000},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            report = TechnicalReport(results, parameters, cache_dir=Path(tmpdir))
            table = report.generate_model_parameters_table()

            assert isinstance(table, pd.DataFrame)
            assert "Category" in table.columns
            assert "Parameter" in table.columns
            assert "Value" in table.columns
            assert len(table) == 4


class TestReportValidator:
    """Test report validation functionality."""

    def test_validator_initialization(self):
        """Test ReportValidator initialization."""
        config = create_executive_config()
        validator = ReportValidator(config)

        assert validator.config == config
        assert len(validator.errors) == 0
        assert len(validator.warnings) == 0

    def test_validate_structure(self):
        """Test structure validation."""
        # Invalid config - no title
        config = ReportConfig(metadata=ReportMetadata(title=""), sections=[])

        validator = ReportValidator(config)
        is_valid, errors, warnings = validator.validate()

        assert is_valid is False
        assert len(errors) > 0
        assert any("title" in e.lower() for e in errors)

    def test_validate_references(self):
        """Test reference validation."""
        config = ReportConfig(
            metadata=ReportMetadata(title="Test"),
            sections=[
                SectionConfig(
                    title="Section", level=1, content="See Figure: undefined_figure", figures=[]
                )
            ],
        )

        validator = ReportValidator(config)
        is_valid, errors, warnings = validator.validate()

        assert len(warnings) > 0
        assert any("undefined" in w.lower() for w in warnings)

    def test_validate_completeness(self):
        """Test completeness validation."""
        config = ReportConfig(
            metadata=ReportMetadata(title="Test"),
            sections=[
                SectionConfig(
                    title="Empty Section",
                    level=1
                    # No content, figures, or tables
                )
            ],
            template="executive",
        )

        validator = ReportValidator(config)
        is_valid, errors, warnings = validator.validate()

        assert len(warnings) > 0
        assert any("no content" in w.lower() for w in warnings)

    def test_validate_results_data(self):
        """Test results data validation."""
        # Valid data
        valid_results = {
            "roe": 0.18,
            "ruin_probability": 0.01,
            "trajectories": np.random.randn(100, 100),
        }

        is_valid, errors = validate_results_data(valid_results)
        assert is_valid is True
        assert len(errors) == 0

        # Invalid data - missing required key
        invalid_results = {"roe": 0.18, "trajectories": np.random.randn(100, 100)}

        is_valid, errors = validate_results_data(invalid_results)
        assert is_valid is False
        assert len(errors) > 0
        assert any("ruin_probability" in e for e in errors)

    def test_validate_parameters(self):
        """Test parameter validation."""
        # Valid parameters
        valid_params = {
            "financial": {"initial_assets": 10000000, "tax_rate": 0.25},
            "insurance": {"premium_rate": 0.03},
            "simulation": {"years": 10, "num_simulations": 1000},
        }

        is_valid, errors = validate_parameters(valid_params)
        assert is_valid is True
        assert len(errors) == 0

        # Invalid parameters
        invalid_params = {
            "financial": {"initial_assets": -1000},  # Negative assets
            "insurance": {},
            "simulation": {"years": 0},  # Zero years
        }

        is_valid, errors = validate_parameters(invalid_params)
        assert is_valid is False
        assert len(errors) > 0


class TestIntegration:
    """Integration tests for the complete report generation system."""

    def test_end_to_end_executive_report(self):
        """Test complete executive report generation."""
        # Create sample data
        np.random.seed(42)
        results = {
            "roe": 0.185,
            "roe_baseline": 0.155,
            "ruin_probability": 0.008,
            "ruin_probability_baseline": 0.025,
            "growth_rate": 0.072,
            "optimal_limits": [5e6, 15e6, 10e6],
            "total_premium": 450000,
            "expected_losses": 180000,
            "trajectories": np.random.randn(100, 100) * 1000000 + 10000000,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create report
            config = create_executive_config()
            config.output_dir = Path(tmpdir)
            config.cache_dir = Path(tmpdir) / "cache"
            config.output_formats = ["markdown", "html"]

            report = ExecutiveReport(results, config)

            # Validate configuration
            validator = ReportValidator(config)
            is_valid, errors, warnings = validator.validate()
            assert is_valid or len(errors) == 0  # Allow warnings

            # Generate report
            path = report.generate()
            assert path.exists()

            # Check content
            content = path.read_text()
            assert "Insurance Optimization Analysis" in content
            assert "Key Findings" in content
            assert "18.5%" in content or "0.185" in content

    def test_end_to_end_technical_report(self):
        """Test complete technical report generation."""
        # Create sample data
        np.random.seed(42)
        results = {
            "trajectories": np.random.randn(100, 100),
            "simulated_losses": np.random.lognormal(10, 2, 1000),
            "convergence_metrics": {"gelman_rubin": 1.02, "ess": 5234},
        }

        parameters = {
            "financial": {"initial_assets": 10000000, "tax_rate": 0.25},
            "insurance": {"premium_rate": 0.03},
            "simulation": {"years": 10, "num_simulations": 1000, "seed": 42},
            "theoretical_dist": "lognormal",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create report
            config = create_technical_config()
            config.output_dir = Path(tmpdir)
            config.cache_dir = Path(tmpdir) / "cache"
            config.output_formats = ["markdown"]

            report = TechnicalReport(results, parameters, config)

            # Generate report
            path = report.generate()
            assert path.exists()

            # Check content
            content = path.read_text()
            assert "Technical Appendix" in content
            assert "Methodology" in content

    def test_batch_report_generation(self):
        """Test generating multiple reports."""
        scenarios = [
            {"name": "Conservative", "roe": 0.14, "ruin_prob": 0.001},
            {"name": "Balanced", "roe": 0.18, "ruin_prob": 0.008},
            {"name": "Aggressive", "roe": 0.22, "ruin_prob": 0.025},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            generated_reports = []

            for scenario in scenarios:
                results = {
                    "roe": scenario["roe"],
                    "ruin_probability": scenario["ruin_prob"],
                    "trajectories": np.random.randn(100, 100),
                }

                config = create_executive_config()
                config.metadata.title = f"Analysis - {scenario['name']}"
                config.output_dir = Path(tmpdir)
                config.cache_dir = Path(tmpdir) / "cache"
                config.output_formats = ["markdown"]

                report = ExecutiveReport(results, config)
                path = report.generate()
                generated_reports.append(path)

            # Verify all reports were generated
            assert len(generated_reports) == 3
            for path in generated_reports:
                assert path.exists()

    def test_performance_requirements(self):
        """Test that report generation meets performance requirements."""
        import time

        # Create large dataset
        results = {
            "roe": 0.18,
            "ruin_probability": 0.01,
            "trajectories": np.random.randn(1000, 1000),
            "simulated_losses": np.random.lognormal(10, 2, 10000),
        }

        parameters = {
            "financial": {f"param_{i}": i * 0.1 for i in range(20)},
            "insurance": {f"param_{i}": i * 0.01 for i in range(20)},
            "simulation": {"years": 100, "num_simulations": 10000},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_technical_config()
            config.output_dir = Path(tmpdir)
            config.cache_dir = Path(tmpdir) / "cache"
            config.output_formats = ["markdown"]

            start_time = time.time()
            report = TechnicalReport(results, parameters, config)
            path = report.generate()
            end_time = time.time()

            generation_time = end_time - start_time
            assert generation_time < 30  # Must complete within 30 seconds
            assert path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
