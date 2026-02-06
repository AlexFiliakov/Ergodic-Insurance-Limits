"""Interactive visualization functions using Plotly.

This module provides functions for creating interactive dashboards
and visualizations for Monte Carlo simulations and analysis results.
"""

from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .core import COLOR_SEQUENCE, WSJ_COLORS


def create_interactive_dashboard(
    results: Union[Dict[str, Any], pd.DataFrame],
    title: str = "Monte Carlo Simulation Dashboard",
    height: int = 600,
    show_distributions: bool = False,
) -> go.Figure:
    """Create interactive Plotly dashboard with WSJ styling.

    Creates a comprehensive interactive dashboard with multiple panels
    showing simulation results, convergence, and risk metrics.

    Args:
        results: Dictionary with simulation results or DataFrame
        title: Dashboard title
        height: Dashboard height in pixels
        show_distributions: Whether to show distribution plots

    Returns:
        Plotly figure with interactive dashboard

    Examples:
        >>> results = {
        ...     "growth_rates": np.random.normal(0.05, 0.02, 1000),
        ...     "losses": np.random.lognormal(10, 2, 1000),
        ...     "metrics": {"var_95": 100000, "var_99": 150000}
        ... }
        >>> fig = create_interactive_dashboard(results)
    """
    # Handle DataFrame input
    if isinstance(results, pd.DataFrame):
        # Convert DataFrame to dictionary format expected by dashboard
        results_dict = {
            "data": results,
            "summary": {
                "mean_assets": (
                    results.get("assets", pd.Series()).mean() if "assets" in results.columns else 0
                ),
                "mean_losses": (
                    results.get("losses", pd.Series()).mean() if "losses" in results.columns else 0
                ),
                "years": results["year"].nunique() if "year" in results.columns else 1,
            },
        }
        results = results_dict
    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Growth Rate Distribution",
            "Loss Exceedance Curve",
            "Convergence Diagnostics",
            "Risk Metrics",
        ),
        specs=[
            [{"type": "histogram"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "bar"}],
        ],
    )

    # WSJ-style layout
    layout_theme = {
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "font": {"family": "Arial, sans-serif", "size": 11, "color": WSJ_COLORS["black"]},
        "title": {"font": {"size": 16, "color": WSJ_COLORS["black"]}},
        "xaxis": {"gridcolor": WSJ_COLORS["light_gray"], "gridwidth": 0.5},
        "yaxis": {"gridcolor": WSJ_COLORS["light_gray"], "gridwidth": 0.5},
        "colorway": COLOR_SEQUENCE,
    }

    # Growth rate histogram
    if "growth_rates" in results:
        fig.add_trace(
            go.Histogram(
                x=results["growth_rates"],
                nbinsx=50,
                marker_color=WSJ_COLORS["blue"],
                opacity=0.7,
                name="Growth Rate",
            ),
            row=1,
            col=1,
        )

    # Loss exceedance curve
    if "losses" in results:
        losses_data = np.asarray(results["losses"])
        sorted_losses = np.sort(losses_data)[::-1]
        exceedance_prob = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)

        fig.add_trace(
            go.Scatter(
                x=sorted_losses / 1e6,
                y=exceedance_prob,
                mode="lines",
                line={"color": WSJ_COLORS["red"], "width": 2},
                name="Exceedance",
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="Loss Amount ($M)", row=1, col=2)
        fig.update_yaxes(title_text="Exceedance Probability", type="log", row=1, col=2)

    # Convergence diagnostics
    if "convergence" in results and isinstance(results["convergence"], dict):
        iterations = results["convergence"].get("iterations", [])
        r_hat = results["convergence"].get("r_hat", [])

        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=r_hat,
                mode="lines+markers",
                line={"color": WSJ_COLORS["green"], "width": 2},
                marker={"size": 6},
                name="R-hat",
            ),
            row=2,
            col=1,
        )

        # Add convergence threshold line
        fig.add_hline(
            y=1.1,
            line_dash="dash",
            line_color=WSJ_COLORS["orange"],
            annotation_text="Convergence Threshold",
            row=2,
            col=1,
        )

        fig.update_xaxes(title_text="Iterations", row=2, col=1)
        fig.update_yaxes(title_text="R-hat Statistic", row=2, col=1)

    # Risk metrics bar chart
    if "metrics" in results and isinstance(results["metrics"], dict):
        metric_names = ["VaR(95%)", "VaR(99%)", "TVaR(99%)", "Expected Shortfall"]
        metric_values = [
            results["metrics"].get("var_95", 0) / 1e6,
            results["metrics"].get("var_99", 0) / 1e6,
            results["metrics"].get("tvar_99", 0) / 1e6,
            results["metrics"].get("expected_shortfall", 0) / 1e6,
        ]

        fig.add_trace(
            go.Bar(
                x=metric_names,
                y=metric_values,
                marker_color=COLOR_SEQUENCE[: len(metric_names)],
                text=[f"${v:.1f}M" for v in metric_values],
                textposition="outside",
                name="Risk Metrics",
            ),
            row=2,
            col=2,
        )
        fig.update_yaxes(title_text="Amount ($M)", row=2, col=2)

    # Update layout
    fig.update_layout(title_text=title, showlegend=False, height=height, **layout_theme)

    # Update all axes
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor=WSJ_COLORS["light_gray"])
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor=WSJ_COLORS["light_gray"])

    return fig


def create_time_series_dashboard(
    data: pd.DataFrame,
    value_col: str,
    time_col: str = "date",
    title: str = "Time Series Analysis",
    height: int = 600,
    show_forecast: bool = False,
) -> go.Figure:
    """Create interactive time series visualization.

    Creates an interactive time series plot with optional forecast bands
    and statistical overlays.

    Args:
        data: DataFrame with time series data
        value_col: Name of value column
        time_col: Name of time column
        title: Plot title
        height: Plot height in pixels
        show_forecast: Whether to show forecast bands

    Returns:
        Plotly figure with time series visualization
    """
    fig = go.Figure()

    # Main time series
    fig.add_trace(
        go.Scatter(
            x=data[time_col],
            y=data[value_col],
            mode="lines",
            name="Actual",
            line={"color": WSJ_COLORS["blue"], "width": 2},
        )
    )

    # Add moving average
    if len(data) > 20:
        ma = data[value_col].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(
                x=data[time_col],
                y=ma,
                mode="lines",
                name="20-period MA",
                line={"color": WSJ_COLORS["orange"], "width": 1, "dash": "dash"},
            )
        )

    # Add forecast if requested
    if show_forecast and f"{value_col}_forecast" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data[time_col],
                y=data[f"{value_col}_forecast"],
                mode="lines",
                name="Forecast",
                line={"color": WSJ_COLORS["green"], "width": 2, "dash": "dot"},
            )
        )

        # Add confidence bands if available
        if f"{value_col}_upper" in data.columns and f"{value_col}_lower" in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data[time_col],
                    y=data[f"{value_col}_upper"],
                    mode="lines",
                    showlegend=False,
                    line={"width": 0},
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=data[time_col],
                    y=data[f"{value_col}_lower"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(0, 128, 199, 0.2)",
                    name="95% CI",
                    line={"width": 0},
                )
            )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=value_col.replace("_", " ").title(),
        height=height,
        hovermode="x unified",
        template="plotly_white",
        font={"family": "Arial, sans-serif"},
    )

    # Add range slider
    fig.update_xaxes(rangeslider_visible=True)

    return fig


def create_correlation_heatmap(
    data: pd.DataFrame,
    title: str = "Correlation Matrix",
    height: int = 600,
    show_values: bool = True,
) -> go.Figure:
    """Create interactive correlation heatmap.

    Creates an interactive heatmap showing correlations between variables
    with customizable color scheme and annotations.

    Args:
        data: DataFrame with variables to correlate
        title: Plot title
        height: Plot height in pixels
        show_values: Whether to show correlation values

    Returns:
        Plotly figure with correlation heatmap
    """
    # Calculate correlation matrix
    corr_matrix = data.corr()

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale=[
                [0, WSJ_COLORS["red"]],
                [0.5, "white"],
                [1, WSJ_COLORS["blue"]],
            ],
            zmin=-1,
            zmax=1,
            text=corr_matrix.values if show_values else None,
            texttemplate="%{text:.2f}" if show_values else None,
            textfont={"size": 10},
            colorbar={"title": "Correlation", "tickmode": "linear", "tick0": -1, "dtick": 0.5},
        )
    )

    # Update layout
    fig.update_layout(
        title=title,
        height=height,
        xaxis={"side": "bottom"},
        yaxis={"side": "left"},
        template="plotly_white",
        font={"family": "Arial, sans-serif"},
    )

    return fig


def create_risk_dashboard(
    risk_metrics: Dict[str, Any],
    title: str = "Risk Analytics Dashboard",
    height: int = 800,
) -> go.Figure:
    """Create comprehensive risk analytics dashboard.

    Creates a multi-panel dashboard showing various risk metrics
    and distributions for comprehensive risk assessment.

    Args:
        risk_metrics: Dictionary containing risk metrics and data
        title: Dashboard title
        height: Dashboard height in pixels

    Returns:
        Plotly figure with risk dashboard
    """
    # Create 3x2 subplot grid
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Value at Risk Distribution",
            "Expected Shortfall Analysis",
            "Risk Contribution by Factor",
            "Stress Test Results",
            "Historical VaR Breaches",
            "Risk Metric Trends",
        ),
        specs=[
            [{"type": "histogram"}, {"type": "bar"}],
            [{"type": "pie"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
    )

    # VaR Distribution (if available)
    if "var_distribution" in risk_metrics:
        fig.add_trace(
            go.Histogram(
                x=risk_metrics["var_distribution"],
                nbinsx=30,
                marker_color=WSJ_COLORS["blue"],
                opacity=0.7,
                name="VaR",
            ),
            row=1,
            col=1,
        )

    # Expected Shortfall
    if "expected_shortfall" in risk_metrics:
        categories = list(risk_metrics["expected_shortfall"].keys())
        values = list(risk_metrics["expected_shortfall"].values())

        fig.add_trace(
            go.Bar(
                x=categories,
                y=values,
                marker_color=WSJ_COLORS["red"],
                name="ES",
            ),
            row=1,
            col=2,
        )

    # Risk Contribution Pie
    if "risk_contribution" in risk_metrics:
        labels = list(risk_metrics["risk_contribution"].keys())
        values = list(risk_metrics["risk_contribution"].values())

        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                marker={"colors": COLOR_SEQUENCE[: len(labels)]},
            ),
            row=2,
            col=1,
        )

    # Stress Test Results
    if "stress_tests" in risk_metrics:
        scenarios = list(risk_metrics["stress_tests"].keys())
        impacts = list(risk_metrics["stress_tests"].values())

        colors = [WSJ_COLORS["green"] if x >= 0 else WSJ_COLORS["red"] for x in impacts]

        fig.add_trace(
            go.Bar(
                x=scenarios,
                y=impacts,
                marker_color=colors,
                name="Impact",
            ),
            row=2,
            col=2,
        )

    # Historical VaR Breaches
    if "var_breaches" in risk_metrics:
        dates = risk_metrics["var_breaches"]["dates"]
        breaches = risk_metrics["var_breaches"]["values"]

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=breaches,
                mode="markers",
                marker={"color": WSJ_COLORS["red"], "size": 10},
                name="Breaches",
            ),
            row=3,
            col=1,
        )

    # Risk Metric Trends
    if "trends" in risk_metrics:
        for metric_name, trend_data in risk_metrics["trends"].items():
            fig.add_trace(
                go.Scatter(
                    x=trend_data["dates"],
                    y=trend_data["values"],
                    mode="lines",
                    name=metric_name,
                ),
                row=3,
                col=2,
            )

    # Update layout
    fig.update_layout(
        title_text=title,
        showlegend=False,
        height=height,
        template="plotly_white",
        font={"family": "Arial, sans-serif"},
    )

    return fig
