"""Shared utilities for reconciliation smoke-test notebooks.

Provides check/assert/display infrastructure, standard factory functions,
and timing utilities used across all reconciliation notebooks.

See GitHub issue #1393 for the full specification.
"""

from contextlib import contextmanager
from decimal import Decimal
import time
from typing import Any, List, Optional, Tuple

from IPython.display import HTML, display
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Check / assertion infrastructure
# ---------------------------------------------------------------------------


class CheckResult:
    """Single pass/fail check with context."""

    def __init__(self, passed: bool, message: str, detail: str = ""):
        self.passed = passed
        self.message = message
        self.detail = detail

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.message}"


class ReconciliationChecker:
    """Accumulates checks and renders a summary table."""

    def __init__(self, section: str = ""):
        self.section = section
        self.checks: List[CheckResult] = []

    # -- recording helpers ---------------------------------------------------

    def check(self, condition: bool, message: str, detail: str = "") -> bool:
        """Record a boolean check. Returns the condition for inline use."""
        self.checks.append(CheckResult(condition, message, detail))
        return condition

    def assert_close(
        self,
        actual: float,
        expected: float,
        tol: float = 0.01,
        message: str = "",
        label_actual: str = "Actual",
        label_expected: str = "Expected",
    ) -> bool:
        """Assert two numbers are within tolerance. Records pass/fail."""
        diff = abs(float(actual) - float(expected))
        passed = diff <= tol
        detail = (
            f"{label_actual}={_fmt(actual)}, {label_expected}={_fmt(expected)}, "
            f"diff={_fmt(diff)}, tol={_fmt(tol)}"
        )
        if not message:
            message = f"{label_actual} ~ {label_expected}"
        self.checks.append(CheckResult(passed, message, detail))
        return passed

    def assert_equal(self, actual: Any, expected: Any, message: str = "") -> bool:
        """Assert exact equality."""
        passed = actual == expected
        detail = f"actual={actual}, expected={expected}"
        if not message:
            message = "Values equal"
        self.checks.append(CheckResult(passed, message, detail))
        return passed

    def assert_greater(self, a: float, b: float, message: str = "") -> bool:
        """Assert a > b."""
        passed = float(a) > float(b)
        detail = f"{_fmt(a)} > {_fmt(b)}"
        if not message:
            message = f"{_fmt(a)} > {_fmt(b)}"
        self.checks.append(CheckResult(passed, message, detail))
        return passed

    def assert_in_range(self, value: float, low: float, high: float, message: str = "") -> bool:
        """Assert low <= value <= high."""
        passed = low <= float(value) <= high
        detail = f"{_fmt(low)} <= {_fmt(value)} <= {_fmt(high)}"
        if not message:
            message = f"Value in [{_fmt(low)}, {_fmt(high)}]"
        self.checks.append(CheckResult(passed, message, detail))
        return passed

    # -- display helpers -----------------------------------------------------

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def summary_counts(self) -> Tuple[int, int]:
        """Return (passed, failed) counts."""
        passed = sum(1 for c in self.checks if c.passed)
        return passed, len(self.checks) - passed

    def display_results(self) -> None:
        """Render an HTML summary table in the notebook."""
        rows = []
        for i, c in enumerate(self.checks, 1):
            icon = "&#10003;" if c.passed else "&#10007;"
            color = "#28a745" if c.passed else "#dc3545"
            rows.append(
                f"<tr>"
                f'<td style="text-align:center;color:{color};font-weight:bold">{icon}</td>'
                f"<td>{c.message}</td>"
                f'<td style="color:#666;font-size:0.9em">{c.detail}</td>'
                f"</tr>"
            )
        header = f"<h3>{self.section}</h3>" if self.section else ""
        passed, failed = self.summary_counts
        total = passed + failed
        summary_color = "#28a745" if failed == 0 else "#dc3545"
        html = f"""
        {header}
        <table style="border-collapse:collapse;width:100%">
        <thead><tr>
            <th style="width:40px"></th>
            <th style="text-align:left">Check</th>
            <th style="text-align:left">Detail</th>
        </tr></thead>
        <tbody>{''.join(rows)}</tbody>
        </table>
        <p style="color:{summary_color};font-weight:bold;font-size:1.1em">
            {passed}/{total} checks passed{' - ALL PASS' if failed == 0 else f' - {failed} FAILED'}
        </p>
        """
        display(HTML(html))


def final_summary(*checkers: ReconciliationChecker) -> None:
    """Display a final PASS/FAIL banner across all checkers."""
    total_passed = sum(c.summary_counts[0] for c in checkers)
    total_failed = sum(c.summary_counts[1] for c in checkers)
    total = total_passed + total_failed
    all_ok = total_failed == 0

    bg = "#d4edda" if all_ok else "#f8d7da"
    fg = "#155724" if all_ok else "#721c24"
    status = "ALL CHECKS PASSED" if all_ok else f"{total_failed} CHECK(S) FAILED"
    html = f"""
    <div style="padding:20px;margin:10px 0;border-radius:8px;background:{bg};
                text-align:center;font-size:1.4em;font-weight:bold;color:{fg}">
        {status}<br>
        <span style="font-size:0.7em;font-weight:normal">{total_passed}/{total} passed</span>
    </div>
    """
    display(HTML(html))

    if not all_ok:
        raise AssertionError(f"Reconciliation failed: {total_failed}/{total} checks did not pass.")


# ---------------------------------------------------------------------------
# Display / formatting helpers
# ---------------------------------------------------------------------------


def section_header(title: str) -> None:
    """Display a styled section header."""
    display(HTML(f'<h2 style="border-bottom:2px solid #4a86c8;padding-bottom:6px">{title}</h2>'))


def _fmt(value: Any) -> str:
    """Format a number for display."""
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, float):
        if abs(value) >= 1e6:
            return f"${value:,.0f}"
        elif abs(value) >= 1:
            return f"{value:,.2f}"
        else:
            return f"{value:.6f}"
    return str(value)


def fmt_dollar(value: float) -> str:
    """Format a dollar amount with commas."""
    return f"${float(value):,.0f}"


def display_df(df: pd.DataFrame, title: str = "") -> None:
    """Display a DataFrame with optional title."""
    if title:
        display(HTML(f"<b>{title}</b>"))
    display(df)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


@contextmanager
def timed_cell(label: str = "Cell"):
    """Context manager that prints execution time."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"  [{label}] completed in {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Standard factory functions for reproducible setups
# ---------------------------------------------------------------------------


def create_standard_manufacturer(
    initial_assets: float = 10_000_000,
    asset_turnover: float = 1.2,
    operating_margin: float = 0.10,
    tax_rate: float = 0.25,
    retention_ratio: float = 0.70,
    **kwargs,
):
    """Create a WidgetManufacturer with standard blog-post parameters."""
    from ergodic_insurance.config import ManufacturerConfig
    from ergodic_insurance.manufacturer import WidgetManufacturer

    config = ManufacturerConfig(
        initial_assets=initial_assets,
        asset_turnover_ratio=asset_turnover,
        base_operating_margin=operating_margin,
        tax_rate=tax_rate,
        retention_ratio=retention_ratio,
        lae_ratio=kwargs.get("lae_ratio", 0.0),
    )
    return WidgetManufacturer(config)


def create_standard_loss_generator(seed: int = 42, **kwargs):
    """Create a ManufacturingLossGenerator with standard parameters."""
    from ergodic_insurance.loss_distributions import ManufacturingLossGenerator

    return ManufacturingLossGenerator(seed=seed, **kwargs)


def create_simple_insurance_program(
    deductible: float = 50_000,
    limit: float = 1_000_000,
    rate: float = 0.02,
    **kwargs,
):
    """Create a simple single-layer InsuranceProgram."""
    from ergodic_insurance.insurance_program import InsuranceProgram

    return InsuranceProgram.simple(
        deductible=deductible,
        limit=limit,
        rate=rate,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Notebook metadata
# ---------------------------------------------------------------------------

NOTEBOOK_METADATA = {
    "issue": "#1393",
    "repo": "https://github.com/AlexFiliakov/Ergodic-Insurance-Limits",
    "series": "Reconciliation Smoke Tests",
}


def notebook_header(number: int, title: str, description: str) -> None:
    """Display a standard notebook header."""
    html = f"""
    <div style="padding:15px;margin-bottom:20px;border-left:4px solid #4a86c8;background:#f8f9fa">
        <h1 style="margin:0 0 5px 0">Reconciliation #{number:02d}: {title}</h1>
        <p style="margin:0;color:#666">{description}</p>
        <p style="margin:5px 0 0 0;font-size:0.85em;color:#999">
            Part of the reconciliation smoke-test suite (Issue {NOTEBOOK_METADATA['issue']})
        </p>
    </div>
    """
    display(HTML(html))
