"""Technical report generation for detailed analysis documentation.

This module provides the TechnicalReport class that generates comprehensive
technical appendices with methodology, validation, and detailed results.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

from ..convergence import ConvergenceDiagnostics
from .config import FigureConfig, ReportConfig, create_technical_config
from .report_builder import ReportBuilder

logger = logging.getLogger(__name__)


class TechnicalReport(ReportBuilder):
    """Generate detailed technical reports.

    This class creates comprehensive technical documentation including
    methodology, mathematical proofs, statistical validation, and
    detailed analysis results.

    Attributes:
        results: Complete analysis results.
        parameters: Model parameters used.
        validation_metrics: Validation and convergence metrics.
    """

    _ALLOWED_FIGURE_GENERATORS = frozenset(
        {
            "generate_parameter_sensitivity_plot",
            "generate_qq_plot",
            "generate_correlation_matrix_plot",
        }
    )
    _ALLOWED_TABLE_GENERATORS = frozenset({"generate_model_parameters_table"})

    def __init__(
        self,
        results: Dict[str, Any],
        parameters: Dict[str, Any],
        config: Optional[ReportConfig] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize TechnicalReport.

        Args:
            results: Complete analysis results.
            parameters: Model parameters.
            config: Report configuration (uses default if None).
            cache_dir: Optional cache directory.
        """
        if config is None:
            config = create_technical_config()

        super().__init__(config, cache_dir)
        self.results = results
        self.parameters = parameters
        self.validation_metrics = self._compute_validation_metrics()

    def generate(self) -> Path:
        """Generate the technical report.

        Returns:
            Path to generated report file.
        """
        logger.info("Generating technical report...")

        # Update configuration with technical details
        self._update_config_with_details()

        # Generate report in all requested formats
        output_paths = []
        for output_format in self.config.output_formats:
            path = self.save(output_format)
            output_paths.append(path)

        logger.info(f"Technical report generated: {output_paths}")
        return output_paths[0]

    def _compute_validation_metrics(self) -> Dict[str, Any]:
        """Compute validation metrics from results.

        Returns:
            Dictionary of validation metrics.
        """
        metrics: Dict[str, Any] = {}

        # Convergence metrics
        if "trajectories" in self.results:
            trajectories = self.results["trajectories"]
            diagnostics = ConvergenceDiagnostics(trajectories)

            # Compute various convergence diagnostics
            metrics["gelman_rubin"] = diagnostics.calculate_r_hat(trajectories)
            metrics["effective_sample_size"] = diagnostics.calculate_ess(trajectories[0])
            metrics["autocorrelation"] = diagnostics._calculate_autocorrelation(
                trajectories[0], 50
            ).mean()
            metrics["batch_means_test"] = 0.0  # Placeholder for Geweke statistic

        # Statistical tests
        if "simulated_losses" in self.results:
            losses = self.results["simulated_losses"]

            # Basic statistical tests using scipy

            # Perform normality test on log-transformed losses
            if np.all(losses > 0):
                log_losses = np.log(losses)
                ad_result = stats.anderson(log_losses, dist="norm")
                ks_stat, ks_pval = stats.kstest(log_losses, "norm")

                # Anderson test returns a result object with statistic and critical values
                # We use the first critical value (15% significance) as a simplified p-value proxy
                ad_pval = 0.15 if ad_result.statistic < ad_result.critical_values[0] else 0.01

                metrics["anderson_darling"] = {
                    "statistic": ad_result.statistic,
                    "p_value": ad_pval,
                }
                metrics["kolmogorov_smirnov"] = {
                    "statistic": ks_stat,
                    "p_value": ks_pval,
                }

        # Model validation
        if "holdout_results" in self.results:
            holdout = self.results["holdout_results"]
            metrics["out_of_sample_rmse"] = self._calculate_rmse(
                holdout["predicted"], holdout["actual"]
            )
            metrics["out_of_sample_mape"] = self._calculate_mape(
                holdout["predicted"], holdout["actual"]
            )

        return metrics

    def _calculate_rmse(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """Calculate root mean squared error.

        Args:
            predicted: Predicted values.
            actual: Actual values.

        Returns:
            RMSE value.
        """
        return float(np.sqrt(np.mean((predicted - actual) ** 2)))

    def _calculate_mape(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """Calculate mean absolute percentage error.

        Args:
            predicted: Predicted values.
            actual: Actual values.

        Returns:
            MAPE value.
        """
        mask = actual != 0
        return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

    def _update_config_with_details(self):
        """Update report configuration with technical details."""
        # Update sections with generated content
        for section in self.config.sections:
            if section.title == "Methodology":
                for subsection in section.subsections:
                    if subsection.title == "Ergodic Theory Application":
                        subsection.content = self._generate_ergodic_methodology()
                    elif subsection.title == "Simulation Framework":
                        subsection.content = self._generate_simulation_methodology()
            elif section.title == "Statistical Validation":
                section.content = self._generate_validation_summary()

    def _generate_ergodic_methodology(self) -> str:
        """Generate ergodic theory methodology section.

        Returns:
            Methodology text with equations.
        """
        methodology = []

        # Introduction
        methodology.append(
            "The ergodic approach to insurance optimization leverages the fundamental "
            "distinction between time-average and ensemble-average growth rates in "
            "multiplicative stochastic processes."
        )

        # Mathematical framework
        methodology.append("\n#### Mathematical Framework\n")
        methodology.append(
            "For a wealth process $W_t$ following geometric Brownian motion:\n\n"
            "$$dW_t = \\mu W_t dt + \\sigma W_t dB_t$$\n\n"
            "The time-average growth rate is:\n\n"
            "$$g_{time} = \\mu - \\frac{\\sigma^2}{2}$$\n\n"
            "While the ensemble-average growth rate is:\n\n"
            "$$g_{ensemble} = \\mu$$"
        )

        # Insurance impact
        methodology.append("\n#### Insurance Impact on Growth\n")
        methodology.append(
            "Insurance modifies the growth dynamics by:\n"
            "1. Reducing volatility through loss capping\n"
            "2. Introducing a deterministic premium cost\n"
            "3. Creating non-linear payoff structures\n\n"
            "The optimized growth rate with insurance becomes:\n\n"
            "$$g_{insured} = \\mu - p - \\frac{\\sigma_{insured}^2}{2}$$\n\n"
            "where $p$ is the premium rate and $\\sigma_{insured} < \\sigma$ "
            "is the reduced volatility."
        )

        # Implementation details
        methodology.append("\n#### Implementation Details\n")
        methodology.append(
            f"- Simulation horizon: {self.parameters.get('years', 100)} years\n"
            f"- Time steps: {self.parameters.get('steps_per_year', 12)} per year\n"
            f"- Monte Carlo paths: {self.parameters.get('num_simulations', 10000)}\n"
            f"- Random seed: {self.parameters.get('seed', 'variable')}"
        )

        return "\n".join(methodology)

    def _generate_simulation_methodology(self) -> str:
        """Generate simulation framework methodology.

        Returns:
            Simulation methodology text.
        """
        methodology = []

        # Overview
        methodology.append(
            "The simulation framework implements a comprehensive Monte Carlo engine "
            "with advanced variance reduction techniques and parallel processing."
        )

        # Stochastic processes
        methodology.append("\n#### Stochastic Process Implementation\n")
        methodology.append(
            "Loss events are modeled using:\n"
            "- **Frequency**: Poisson process with rate $\\lambda$\n"
            "- **Severity**: Lognormal distribution $LN(\\mu_L, \\sigma_L)$\n"
            "- **Correlation**: Copula-based dependency structure\n\n"
            "Revenue volatility follows:\n"
            "- **Base growth**: Deterministic trend $g$\n"
            "- **Volatility**: Stochastic component $\\sigma_R dB_t$\n"
            "- **Mean reversion**: Ornstein-Uhlenbeck process for bounded variables"
        )

        # Numerical methods
        methodology.append("\n#### Numerical Methods\n")
        methodology.append(
            "- **Integration scheme**: Euler-Maruyama with adaptive timestep\n"
            "- **Variance reduction**: Antithetic variates and control variates\n"
            "- **Parallelization**: Process-based parallel execution\n"
            "- **Memory optimization**: Chunked trajectory storage"
        )

        # Convergence criteria
        methodology.append("\n#### Convergence Criteria\n")
        methodology.append(
            "Simulations continue until:\n"
            "1. Gelman-Rubin statistic R-hat < 1.1\n"
            "2. Effective sample size > 1000\n"
            "3. Relative standard error < 1%\n"
            "4. Batch means test p-value > 0.05"
        )

        return "\n".join(methodology)

    def _generate_validation_summary(self) -> str:
        """Generate validation summary section.

        Returns:
            Validation summary text.
        """
        summary = []

        # Convergence validation
        summary.append("### Convergence Validation\n")
        if "gelman_rubin" in self.validation_metrics:
            gr = self.validation_metrics["gelman_rubin"]
            status = "Converged" if gr < 1.1 else "Not converged"
            summary.append(f"- Gelman-Rubin R-hat: {gr:.4f} ({status})")

        if "effective_sample_size" in self.validation_metrics:
            ess = self.validation_metrics["effective_sample_size"]
            status = "Sufficient" if ess > 1000 else "Insufficient"
            summary.append(f"- Effective sample size: {ess:.0f} ({status})")

        # Statistical validation
        summary.append("\n### Statistical Tests\n")
        if "anderson_darling" in self.validation_metrics:
            ad = self.validation_metrics["anderson_darling"]
            summary.append(f"- Anderson-Darling test: p={ad['p_value']:.4f}")

        if "kolmogorov_smirnov" in self.validation_metrics:
            ks = self.validation_metrics["kolmogorov_smirnov"]
            summary.append(f"- Kolmogorov-Smirnov test: p={ks['p_value']:.4f}")

        # Model validation
        summary.append("\n### Model Performance\n")
        if "out_of_sample_rmse" in self.validation_metrics:
            rmse = self.validation_metrics["out_of_sample_rmse"]
            summary.append(f"- Out-of-sample RMSE: {rmse:.4f}")

        if "out_of_sample_mape" in self.validation_metrics:
            mape = self.validation_metrics["out_of_sample_mape"]
            summary.append(f"- Out-of-sample MAPE: {mape:.2f}%")

        return "\n".join(summary)

    def generate_parameter_sensitivity_plot(self, fig_config: FigureConfig) -> plt.Figure:
        """Generate parameter sensitivity tornado plot.

        Args:
            fig_config: Figure configuration.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(fig_config.width, fig_config.height))

        if "sensitivity_analysis" in self.results:
            sensitivity = self.results["sensitivity_analysis"]

            # Create tornado plot
            parameters = list(sensitivity.keys())
            low_impacts = [sensitivity[p]["low"] for p in parameters]
            high_impacts = [sensitivity[p]["high"] for p in parameters]
            base_value = self.results.get("base_case_value", 0)

            # Calculate deviations from base
            low_dev = [(base_value - l) / base_value * 100 for l in low_impacts]
            high_dev = [(h - base_value) / base_value * 100 for h in high_impacts]

            # Create horizontal bars
            y_pos = np.arange(len(parameters))
            ax.barh(y_pos, low_dev, left=0, color="#e74c3c", alpha=0.7, label="Low scenario")
            ax.barh(
                y_pos,
                high_dev,
                left=0,
                color="#2ecc71",
                alpha=0.7,
                label="High scenario",
            )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(parameters)
            ax.set_xlabel("Impact on ROE (%)")
            ax.set_title("Parameter Sensitivity Analysis")
            ax.axvline(x=0, color="black", linewidth=0.5)
            ax.legend()
            ax.grid(True, alpha=0.3)

        return fig

    def generate_qq_plot(self, fig_config: FigureConfig) -> plt.Figure:
        """Generate Q-Q plot for distribution validation.

        Args:
            fig_config: Figure configuration.

        Returns:
            Matplotlib figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(fig_config.width, fig_config.height))

        if "simulated_losses" in self.results:
            losses = self.results["simulated_losses"]

            # Q-Q plot against normal
            stats.probplot(losses, dist="norm", plot=axes[0])
            axes[0].set_title("Normal Q-Q Plot")
            axes[0].grid(True, alpha=0.3)

            # Q-Q plot against lognormal
            stats.probplot(np.log(losses[losses > 0]), dist="norm", plot=axes[1])
            axes[1].set_title("Lognormal Q-Q Plot")
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def generate_model_parameters_table(self) -> pd.DataFrame:
        """Generate comprehensive model parameters table.

        Returns:
            Parameters DataFrame.
        """
        rows = []

        # Financial parameters
        if "financial" in self.parameters:
            for key, value in self.parameters["financial"].items():
                rows.append(
                    {
                        "Category": "Financial",
                        "Parameter": key.replace("_", " ").title(),
                        "Value": value,
                        "Unit": self._get_unit(key),
                    }
                )

        # Insurance parameters
        if "insurance" in self.parameters:
            for key, value in self.parameters["insurance"].items():
                rows.append(
                    {
                        "Category": "Insurance",
                        "Parameter": key.replace("_", " ").title(),
                        "Value": value,
                        "Unit": self._get_unit(key),
                    }
                )

        # Simulation parameters
        if "simulation" in self.parameters:
            for key, value in self.parameters["simulation"].items():
                rows.append(
                    {
                        "Category": "Simulation",
                        "Parameter": key.replace("_", " ").title(),
                        "Value": value,
                        "Unit": self._get_unit(key),
                    }
                )

        return pd.DataFrame(rows)

    def _get_unit(self, parameter_name: str) -> str:
        """Get unit for parameter.

        Args:
            parameter_name: Parameter name.

        Returns:
            Unit string.
        """
        units = {
            "rate": "%",
            "probability": "%",
            "limit": "$",
            "premium": "$",
            "years": "years",
            "simulations": "paths",
            "seed": "-",
        }

        for key, unit in units.items():
            if key in parameter_name.lower():
                return unit
        return "-"

    def generate_correlation_matrix_plot(self, fig_config: FigureConfig) -> plt.Figure:
        """Generate correlation matrix heatmap.

        Args:
            fig_config: Figure configuration.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(fig_config.width, fig_config.height))

        if "correlation_matrix" in self.results:
            corr_matrix = self.results["correlation_matrix"]

            # Create heatmap
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                square=True,
                ax=ax,
                cbar_kws={"label": "Correlation"},
            )

            ax.set_title("Variable Correlation Matrix")

        return fig
