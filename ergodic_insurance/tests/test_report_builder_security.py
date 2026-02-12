"""Security regression tests for report_builder.py.

Covers:
- Issue #804: allowlist enforcement for dynamic method dispatch in
  _load_table_data() and _generate_figure().
- Issue #801: HTML escaping and CSS validation in save() HTML output.
"""

import io
from pathlib import Path
import tempfile
from typing import FrozenSet
from unittest.mock import patch as _patch

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from ergodic_insurance.reporting.config import (
    FigureConfig,
    ReportConfig,
    ReportMetadata,
    ReportStyle,
    SectionConfig,
)
from ergodic_insurance.reporting.report_builder import (
    _ALLOWED_FONT_FAMILIES,
    ReportBuilder,
)

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _NoCloseStringIO(io.StringIO):
    """A StringIO that ignores close() so content survives a ``with`` block."""

    def close(self):  # noqa: D102
        pass  # keep buffer open for later getvalue()


def _capture_html_save(report):
    """Run ``report.save('html')`` and return the captured HTML string.

    Works even when the title contains characters that are invalid in
    Windows filenames (e.g. ``<``, ``>``, ``"``).
    """
    captured = _NoCloseStringIO()
    real_open = open

    def _intercepting_open(path, mode="r", **kwargs):
        if "w" in str(mode) and str(path).endswith(".html"):
            return captured
        return real_open(path, mode, **kwargs)

    with _patch("builtins.open", side_effect=_intercepting_open):
        try:
            report.save("html")
        except (OSError, ValueError):
            pass  # filename may still be invalid on Windows

    html_text = captured.getvalue()
    assert html_text, "No HTML content was captured"
    return html_text


def _make_test_report_class(
    *,
    allowed_figure_generators: FrozenSet[str] = frozenset(),
    allowed_table_generators: FrozenSet[str] = frozenset(),
):
    """Create a concrete ReportBuilder subclass with specified allowlists."""

    class _TestReport(ReportBuilder):
        _ALLOWED_FIGURE_GENERATORS = allowed_figure_generators
        _ALLOWED_TABLE_GENERATORS = allowed_table_generators

        def generate(self) -> Path:
            return self.save("markdown")

        # A generator method that *could* be called but should only be
        # callable when it appears in the allowlist.
        def generate_allowed_table(self) -> pd.DataFrame:
            return pd.DataFrame({"x": [1, 2, 3]})

        def generate_forbidden_table(self) -> pd.DataFrame:
            return pd.DataFrame({"secret": [42]})

        def generate_allowed_figure(self, fig_config: FigureConfig) -> plt.Figure:
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])
            return fig

        def generate_forbidden_figure(self, fig_config: FigureConfig) -> plt.Figure:
            fig, ax = plt.subplots()
            ax.plot([4, 5, 6])
            return fig

    return _TestReport


def _make_config(tmp_dir: Path) -> ReportConfig:
    return ReportConfig(
        metadata=ReportMetadata(title="Test"),
        sections=[],
        output_dir=tmp_dir / "output",
        cache_dir=tmp_dir / "cache",
    )


# ===========================================================================
# Issue #804 — allowlist enforcement for dynamic method dispatch
# ===========================================================================


class TestAllowlistTableGenerators:
    """_load_table_data must reject generators not in the allowlist."""

    def test_allowed_table_generator_is_called(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            Report = _make_test_report_class(
                allowed_table_generators=frozenset({"generate_allowed_table"}),
            )
            report = Report(config)
            df = report._load_table_data("generate_allowed_table")
            assert "x" in df.columns

    def test_forbidden_table_generator_returns_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            Report = _make_test_report_class(
                allowed_table_generators=frozenset({"generate_allowed_table"}),
            )
            report = Report(config)
            df = report._load_table_data("generate_forbidden_table")
            # Should get fallback sample data, not the secret column
            assert "secret" not in df.columns
            assert "Column A" in df.columns

    def test_empty_allowlist_blocks_all_generators(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            Report = _make_test_report_class(
                allowed_table_generators=frozenset(),
            )
            report = Report(config)
            df = report._load_table_data("generate_allowed_table")
            # Falls through to sample data
            assert "Column A" in df.columns


class TestAllowlistFigureGenerators:
    """_generate_figure must reject generators not in the allowlist."""

    def test_allowed_figure_generator_is_called(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            Report = _make_test_report_class(
                allowed_figure_generators=frozenset({"generate_allowed_figure"}),
            )
            report = Report(config)
            fig_config = FigureConfig(
                name="test_fig",
                caption="Test",
                source="generate_allowed_figure",
            )
            path = report._generate_figure(fig_config)
            assert path.exists()
            assert path.name == "test_fig.png"

    def test_forbidden_figure_generator_returns_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            Report = _make_test_report_class(
                allowed_figure_generators=frozenset({"generate_allowed_figure"}),
            )
            report = Report(config)
            fig_config = FigureConfig(
                name="forbidden_fig",
                caption="Forbidden",
                source="generate_forbidden_figure",
            )
            path = report._generate_figure(fig_config)
            # Should still produce a file (placeholder), but the forbidden
            # generator should NOT have been called.
            assert path.exists()
            assert path.name == "forbidden_fig.png"

    def test_empty_allowlist_blocks_all_figure_generators(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(Path(tmp))
            Report = _make_test_report_class(
                allowed_figure_generators=frozenset(),
            )
            report = Report(config)
            fig_config = FigureConfig(
                name="blocked_fig",
                caption="Blocked",
                source="generate_allowed_figure",
            )
            path = report._generate_figure(fig_config)
            assert path.exists()  # placeholder produced


class TestSubclassAllowlists:
    """Verify that shipped subclasses define non-empty allowlists."""

    def test_executive_report_has_allowlists(self):
        from ergodic_insurance.reporting.executive_report import ExecutiveReport

        assert len(ExecutiveReport._ALLOWED_FIGURE_GENERATORS) > 0
        assert len(ExecutiveReport._ALLOWED_TABLE_GENERATORS) > 0
        assert "generate_roe_frontier" in ExecutiveReport._ALLOWED_FIGURE_GENERATORS
        assert "generate_decision_matrix" in ExecutiveReport._ALLOWED_TABLE_GENERATORS

    def test_technical_report_has_allowlists(self):
        from ergodic_insurance.reporting.technical_report import TechnicalReport

        assert len(TechnicalReport._ALLOWED_FIGURE_GENERATORS) > 0
        assert len(TechnicalReport._ALLOWED_TABLE_GENERATORS) > 0
        assert "generate_qq_plot" in TechnicalReport._ALLOWED_FIGURE_GENERATORS
        assert "generate_model_parameters_table" in TechnicalReport._ALLOWED_TABLE_GENERATORS


# ===========================================================================
# Issue #801 — HTML escaping and CSS validation in save()
# ===========================================================================


class TestHtmlEscaping:
    """save('html') must escape user-controlled values."""

    @pytest.fixture(autouse=True)
    def _skip_without_markdown2(self):
        pytest.importorskip("markdown2", reason="markdown2 not installed")

    def test_script_injection_in_title_is_escaped(self):
        """A <script> tag in the title must be escaped in the <title> tag.

        Note: the markdown body content is escaped separately (issue #722).
        This test covers the HTML wrapper template (issue #801).
        """
        with tempfile.TemporaryDirectory() as tmp:
            config = ReportConfig(
                metadata=ReportMetadata(title='<script>alert("xss")</script>'),
                sections=[SectionConfig(title="Sec", level=1, content="Hello")],
                output_dir=Path(tmp) / "output",
                cache_dir=Path(tmp) / "cache",
            )
            Report = _make_test_report_class()
            report = Report(config)
            html_text = _capture_html_save(report)

            # Extract the <title> content and verify it is escaped
            import re

            title_match = re.search(r"<title>(.*?)</title>", html_text)
            assert title_match is not None
            title_content = title_match.group(1)
            assert "<script>" not in title_content
            assert "&lt;script&gt;" in title_content

    def test_font_family_injection_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = ReportConfig(
                metadata=ReportMetadata(title="Safe Title"),
                style=ReportStyle(font_family='Arial; } body { background: url("evil")'),
                sections=[SectionConfig(title="Sec", level=1, content="Hello")],
                output_dir=Path(tmp) / "output",
                cache_dir=Path(tmp) / "cache",
            )
            Report = _make_test_report_class()
            report = Report(config)
            path = report.save("html")
            html_text = path.read_text(encoding="utf-8")

            # The malicious font_family should NOT appear; should fall back to Arial
            assert "evil" not in html_text
            assert "font-family: Arial" in html_text

    def test_safe_font_family_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = ReportConfig(
                metadata=ReportMetadata(title="Safe Title"),
                style=ReportStyle(font_family="Georgia"),
                sections=[SectionConfig(title="Sec", level=1, content="Hello")],
                output_dir=Path(tmp) / "output",
                cache_dir=Path(tmp) / "cache",
            )
            Report = _make_test_report_class()
            report = Report(config)
            path = report.save("html")
            html_text = path.read_text(encoding="utf-8")
            assert "font-family: Georgia" in html_text

    def test_numeric_font_size_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = ReportConfig(
                metadata=ReportMetadata(title="Safe Title"),
                style=ReportStyle(font_size=12),
                sections=[SectionConfig(title="Sec", level=1, content="Hello")],
                output_dir=Path(tmp) / "output",
                cache_dir=Path(tmp) / "cache",
            )
            Report = _make_test_report_class()
            report = Report(config)
            path = report.save("html")
            html_text = path.read_text(encoding="utf-8")
            assert "font-size: 12pt" in html_text

    def test_font_family_allowlist_is_comprehensive(self):
        """Verify the allowlist contains common web-safe fonts."""
        assert "Arial" in _ALLOWED_FONT_FAMILIES
        assert "Times New Roman" in _ALLOWED_FONT_FAMILIES
        assert "sans-serif" in _ALLOWED_FONT_FAMILIES
        assert "monospace" in _ALLOWED_FONT_FAMILIES


class TestHtmlEscapingEdgeCases:
    """Edge cases for HTML output sanitization."""

    @pytest.fixture(autouse=True)
    def _skip_without_markdown2(self):
        pytest.importorskip("markdown2", reason="markdown2 not installed")

    def test_title_with_ampersand_is_escaped(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = ReportConfig(
                metadata=ReportMetadata(title="Risk & Return Analysis"),
                sections=[SectionConfig(title="Sec", level=1, content="Hello")],
                output_dir=Path(tmp) / "output",
                cache_dir=Path(tmp) / "cache",
            )
            Report = _make_test_report_class()
            report = Report(config)
            path = report.save("html")
            html_text = path.read_text(encoding="utf-8")
            assert "<title>Risk &amp; Return Analysis</title>" in html_text

    def test_title_with_quotes_is_escaped(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = ReportConfig(
                metadata=ReportMetadata(title='Report "Special"'),
                sections=[SectionConfig(title="Sec", level=1, content="Hello")],
                output_dir=Path(tmp) / "output",
                cache_dir=Path(tmp) / "cache",
            )
            Report = _make_test_report_class()
            report = Report(config)
            html_text = _capture_html_save(report)
            assert "&quot;" in html_text
