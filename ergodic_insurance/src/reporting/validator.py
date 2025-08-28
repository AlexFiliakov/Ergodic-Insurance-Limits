"""Report validation and quality control utilities.

This module provides validation functions to ensure report completeness,
accuracy, and quality before generation.
"""

import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .config import FigureConfig, ReportConfig, SectionConfig, TableConfig

logger = logging.getLogger(__name__)


class ReportValidator:
    """Validate report configuration and content.

    This class provides comprehensive validation for report configurations,
    ensuring all references are valid, data is complete, and quality
    standards are met.

    Attributes:
        config: Report configuration to validate.
        errors: List of validation errors.
        warnings: List of validation warnings.
    """

    def __init__(self, config: ReportConfig):
        """Initialize ReportValidator.

        Args:
            config: Report configuration to validate.
        """
        self.config = config
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """Run complete validation suite.

        Returns:
            Tuple of (is_valid, errors, warnings).
        """
        self.errors = []
        self.warnings = []
        self.info = []

        # Run all validation checks
        self._validate_structure()
        self._validate_references()
        self._validate_data_sources()
        self._validate_formatting()
        self._validate_completeness()
        self._validate_quality()

        is_valid = len(self.errors) == 0

        # Log results
        if self.errors:
            for error in self.errors:
                logger.error(f"Validation error: {error}")

        if self.warnings:
            for warning in self.warnings:
                logger.warning(f"Validation warning: {warning}")

        if self.info:
            for info in self.info:
                logger.info(f"Validation info: {info}")

        return is_valid, self.errors, self.warnings

    def _validate_structure(self):
        """Validate report structure and hierarchy."""
        # Check metadata
        if not self.config.metadata.title:
            self.errors.append("Report title is required")

        if not self.config.metadata.authors:
            self.warnings.append("No authors specified")

        # Check sections
        if not self.config.sections:
            self.errors.append("Report must have at least one section")

        # Validate section hierarchy
        self._check_section_hierarchy(self.config.sections)

        # Check output formats
        if not self.config.output_formats:
            self.errors.append("At least one output format must be specified")

        # Validate directories exist
        if not self.config.output_dir.exists():
            self.warnings.append(f"Output directory does not exist: {self.config.output_dir}")

        if not self.config.cache_dir.exists():
            self.warnings.append(f"Cache directory does not exist: {self.config.cache_dir}")

    def _check_section_hierarchy(self, sections: List[SectionConfig], parent_level: int = 0):
        """Check section hierarchy is valid.

        Args:
            sections: List of sections to check.
            parent_level: Parent section level.
        """
        for section in sections:
            # Check level progression
            if section.level <= parent_level:
                self.warnings.append(
                    f"Section '{section.title}' level {section.level} "
                    f"should be greater than parent level {parent_level}"
                )

            # Check title
            if not section.title:
                self.errors.append("Section title cannot be empty")

            # Recursively check subsections
            if section.subsections:
                self._check_section_hierarchy(section.subsections, section.level)

    def _validate_references(self):
        """Validate all figure and table references."""
        # Collect all defined figures and tables
        defined_figures = set()
        defined_tables = set()

        for section in self._iter_sections(self.config.sections):
            for fig in section.figures:
                if fig.name in defined_figures:
                    self.errors.append(f"Duplicate figure name: {fig.name}")
                defined_figures.add(fig.name)

            for table in section.tables:
                if table.name in defined_tables:
                    self.errors.append(f"Duplicate table name: {table.name}")
                defined_tables.add(table.name)

        # Check for references in content
        referenced_figures = set()
        referenced_tables = set()

        for section in self._iter_sections(self.config.sections):
            if section.content:
                # Find figure references
                fig_refs = re.findall(r"Figure[:\s]+(\w+)", section.content)
                referenced_figures.update(fig_refs)

                # Find table references
                table_refs = re.findall(r"Table[:\s]+(\w+)", section.content)
                referenced_tables.update(table_refs)

        # Check for undefined references
        undefined_figures = referenced_figures - defined_figures
        if undefined_figures:
            for fig in undefined_figures:
                self.warnings.append(f"Referenced but undefined figure: {fig}")

        undefined_tables = referenced_tables - defined_tables
        if undefined_tables:
            for table in undefined_tables:
                self.warnings.append(f"Referenced but undefined table: {table}")

        # Check for unused definitions
        unused_figures = defined_figures - referenced_figures
        if unused_figures:
            for fig in unused_figures:
                self.info.append(f"Defined but unreferenced figure: {fig}")

        unused_tables = defined_tables - referenced_tables
        if unused_tables:
            for table in unused_tables:
                self.info.append(f"Defined but unreferenced table: {table}")

    def _iter_sections(self, sections: List[SectionConfig]):
        """Iterate through all sections including subsections.

        Args:
            sections: List of sections.

        Yields:
            Each section and subsection.
        """
        for section in sections:
            yield section
            if section.subsections:
                yield from self._iter_sections(section.subsections)

    def _validate_data_sources(self):
        """Validate data sources for figures and tables."""
        for section in self._iter_sections(self.config.sections):
            # Check figure sources
            for fig in section.figures:
                if isinstance(fig.source, (str, Path)):
                    source_path = Path(fig.source)
                    if not source_path.exists() and not str(fig.source).startswith("generate_"):
                        self.warnings.append(
                            f"Figure source not found: {fig.source} (figure: {fig.name})"
                        )

            # Check table data sources
            for table in section.tables:
                if isinstance(table.data_source, (str, Path)):
                    source_path = Path(table.data_source)
                    if not source_path.exists() and not str(table.data_source).startswith("generate_"):
                        self.warnings.append(
                            f"Table data source not found: {table.data_source} (table: {table.name})"
                        )

    def _validate_formatting(self):
        """Validate formatting parameters."""
        # Check figure dimensions
        for section in self._iter_sections(self.config.sections):
            for fig in section.figures:
                if fig.width > 10 or fig.height > 10:
                    self.warnings.append(
                        f"Figure '{fig.name}' dimensions may be too large: "
                        f"{fig.width}x{fig.height} inches"
                    )

                if fig.dpi < 150:
                    self.warnings.append(
                        f"Figure '{fig.name}' DPI ({fig.dpi}) may be too low for print quality"
                    )

        # Check style parameters
        style = self.config.style
        if style.font_size < 8:
            self.warnings.append(f"Font size {style.font_size}pt may be too small")
        elif style.font_size > 14:
            self.warnings.append(f"Font size {style.font_size}pt may be too large")

        # Check margins
        for margin_name, margin_value in style.margins.items():
            if margin_value < 0.5:
                self.warnings.append(f"Margin '{margin_name}' ({margin_value}in) may be too small")
            elif margin_value > 2:
                self.warnings.append(f"Margin '{margin_name}' ({margin_value}in) may be too large")

    def _validate_completeness(self):
        """Check report completeness."""
        # Check for required sections based on template
        if self.config.template == "executive":
            required_sections = {"Key Findings", "Recommendations"}
            section_titles = {s.title for s in self.config.sections}
            missing = required_sections - section_titles
            if missing:
                for section in missing:
                    self.warnings.append(f"Executive report missing section: {section}")

        elif self.config.template == "technical":
            required_sections = {"Methodology", "Statistical Validation"}
            section_titles = {s.title for s in self.config.sections}
            missing = required_sections - section_titles
            if missing:
                for section in missing:
                    self.warnings.append(f"Technical report missing section: {section}")

        # Check for empty sections
        for section in self._iter_sections(self.config.sections):
            if (  # type: ignore[attr-defined]
                not section.content
                and not section.figures
                and not section.tables
                and not section.subsections
            ):
                self.warnings.append(f"Section '{section.title}' has no content")

    def _validate_quality(self):
        """Perform quality checks on report configuration."""
        # Check caption quality
        for section in self._iter_sections(self.config.sections):
            for fig in section.figures:
                if len(fig.caption) < 10:
                    self.warnings.append(
                        f"Figure '{fig.name}' caption may be too short: '{fig.caption}'"
                    )

            for table in section.tables:
                if len(table.caption) < 10:
                    self.warnings.append(
                        f"Table '{table.name}' caption may be too short: '{table.caption}'"
                    )

        # Check metadata quality
        if self.config.metadata.abstract and len(self.config.metadata.abstract) < 50:
            self.warnings.append("Abstract may be too short")

        if not self.config.metadata.keywords:
            self.warnings.append("No keywords specified for report")

        # Count total content
        total_figures = sum(len(s.figures) for s in self._iter_sections(self.config.sections))
        total_tables = sum(len(s.tables) for s in self._iter_sections(self.config.sections))

        self.info.append(
            f"Report contains {len(self.config.sections)} sections, "
            f"{total_figures} figures, and {total_tables} tables"
        )

        # Check balance
        if total_figures > 20:
            self.warnings.append(f"Report has many figures ({total_figures}), consider reducing")

        if total_tables > 15:
            self.warnings.append(f"Report has many tables ({total_tables}), consider reducing")


def validate_results_data(results: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate results data for report generation.

    Args:
        results: Results dictionary to validate.

    Returns:
        Tuple of (is_valid, error_messages).
    """
    errors = []

    # Check for required keys
    required_keys = ["roe", "ruin_probability", "trajectories"]
    for key in required_keys:
        if key not in results:
            errors.append(f"Missing required results key: {key}")

    # Validate data types and ranges
    if "roe" in results:
        roe = results["roe"]
        if not isinstance(roe, (int, float)):
            errors.append(f"ROE must be numeric, got {type(roe)}")
        elif not -1 <= roe <= 10:
            errors.append(f"ROE value {roe} seems unrealistic")

    if "ruin_probability" in results:
        ruin_prob = results["ruin_probability"]
        if not isinstance(ruin_prob, (int, float)):
            errors.append(f"Ruin probability must be numeric, got {type(ruin_prob)}")
        elif not 0 <= ruin_prob <= 1:
            errors.append(f"Ruin probability must be between 0 and 1, got {ruin_prob}")

    if "trajectories" in results:
        trajectories = results["trajectories"]
        if not isinstance(trajectories, np.ndarray):
            errors.append(f"Trajectories must be numpy array, got {type(trajectories)}")
        elif len(trajectories.shape) not in [1, 2]:
            errors.append(f"Trajectories must be 1D or 2D array, got shape {trajectories.shape}")

    return len(errors) == 0, errors


def validate_parameters(params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate model parameters.

    Args:
        params: Parameters dictionary to validate.

    Returns:
        Tuple of (is_valid, error_messages).
    """
    errors = []

    # Check for required parameter groups
    required_groups = ["financial", "insurance", "simulation"]
    for group in required_groups:
        if group not in params:
            errors.append(f"Missing required parameter group: {group}")

    # Validate financial parameters
    if "financial" in params:
        financial = params["financial"]
        if "initial_assets" in financial:
            if financial["initial_assets"] <= 0:
                errors.append("Initial assets must be positive")

        if "tax_rate" in financial:
            if not 0 <= financial["tax_rate"] <= 1:
                errors.append("Tax rate must be between 0 and 1")

    # Validate simulation parameters
    if "simulation" in params:
        sim = params["simulation"]
        if "years" in sim:
            if sim["years"] <= 0:
                errors.append("Simulation years must be positive")

        if "num_simulations" in sim:
            if sim["num_simulations"] <= 0:
                errors.append("Number of simulations must be positive")

    return len(errors) == 0, errors
