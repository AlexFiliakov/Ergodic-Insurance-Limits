"""Configuration schema for report generation system.

This module defines Pydantic models for configuring various types of reports,
including sections, figures, tables, and formatting options.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, validator


class FigureConfig(BaseModel):
    """Configuration for a figure in a report.

    Attributes:
        name: Figure identifier name.
        caption: Caption text for the figure.
        source: Source file path or generation function.
        width: Figure width in inches.
        height: Figure height in inches.
        dpi: Resolution in dots per inch.
        position: LaTeX-style position hint.
        cache_key: Optional cache key for pre-generated figures.
    """

    name: str
    caption: str
    source: Union[str, Path]
    width: float = Field(default=6.5, ge=1.0, le=10.0)
    height: float = Field(default=4.0, ge=1.0, le=10.0)
    dpi: int = Field(default=300, ge=150, le=600)
    position: str = Field(default="htbp")
    cache_key: Optional[str] = None

    @validator("source")
    def validate_source(cls, v):
        """Validate that source is a valid path or callable name."""
        if isinstance(v, str) and not v.startswith("generate_"):
            path = Path(v)
            if not path.suffix in [".png", ".jpg", ".pdf", ".svg"]:
                raise ValueError(f"Invalid figure format: {path.suffix}")
        return v


class TableConfig(BaseModel):
    """Configuration for a table in a report.

    Attributes:
        name: Table identifier name.
        caption: Caption text for the table.
        data_source: Source of data (file path or function).
        format: Output format for the table.
        columns: List of columns to include.
        index: Whether to include row index.
        precision: Number of decimal places for numeric values.
        style: Table styling options.
    """

    name: str
    caption: str
    data_source: Union[str, Path]
    format: Literal["markdown", "html", "latex"] = "markdown"
    columns: Optional[List[str]] = None
    index: bool = False
    precision: int = Field(default=2, ge=0, le=6)
    style: Dict[str, Any] = Field(default_factory=dict)


class SectionConfig(BaseModel):
    """Configuration for a report section.

    Attributes:
        title: Section title.
        level: Heading level (1-4).
        content: Text content or template name.
        figures: List of figures to include.
        tables: List of tables to include.
        subsections: Nested subsections.
        page_break: Whether to start on new page.
    """

    title: str
    level: int = Field(default=2, ge=1, le=4)
    content: Optional[str] = None
    figures: List[FigureConfig] = Field(default_factory=list)
    tables: List[TableConfig] = Field(default_factory=list)
    subsections: List["SectionConfig"] = Field(default_factory=list)
    page_break: bool = False

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class ReportMetadata(BaseModel):
    """Metadata for a report.

    Attributes:
        title: Report title.
        subtitle: Optional subtitle.
        authors: List of author names.
        date: Report generation date.
        version: Report version.
        organization: Organization name.
        confidentiality: Confidentiality level.
        keywords: List of keywords.
        abstract: Brief abstract or summary.
    """

    title: str
    subtitle: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    date: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
    organization: str = "Ergodic Insurance Analytics"
    confidentiality: Literal["Public", "Internal", "Confidential"] = "Internal"
    keywords: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None


class ReportStyle(BaseModel):
    """Styling configuration for reports.

    Attributes:
        font_family: Main font family.
        font_size: Base font size in points.
        line_spacing: Line spacing multiplier.
        margins: Page margins in inches.
        page_size: Paper size.
        orientation: Page orientation.
        header_footer: Include headers and footers.
        page_numbers: Include page numbers.
        color_scheme: Color scheme for charts.
    """

    font_family: str = "Arial"
    font_size: int = Field(default=11, ge=8, le=14)
    line_spacing: float = Field(default=1.5, ge=1.0, le=2.0)
    margins: Dict[str, float] = Field(
        default_factory=lambda: {"top": 1.0, "bottom": 1.0, "left": 1.0, "right": 1.0}
    )
    page_size: Literal["A4", "Letter", "Legal"] = "Letter"
    orientation: Literal["portrait", "landscape"] = "portrait"
    header_footer: bool = True
    page_numbers: bool = True
    color_scheme: str = "default"


class ReportConfig(BaseModel):
    """Complete configuration for a report.

    Attributes:
        metadata: Report metadata.
        style: Report styling.
        sections: List of report sections.
        template: Template type to use.
        output_formats: List of output formats to generate.
        output_dir: Directory for output files.
        cache_dir: Directory for cached figures.
        debug: Enable debug mode.
    """

    metadata: ReportMetadata
    style: ReportStyle = Field(default_factory=ReportStyle)
    sections: List[SectionConfig]
    template: Literal["executive", "technical", "full", "custom"] = "full"
    output_formats: List[Literal["pdf", "html", "markdown"]] = Field(
        default_factory=lambda: ["pdf"]
    )
    output_dir: Path = Field(default_factory=lambda: Path("reports"))
    cache_dir: Path = Field(default_factory=lambda: Path("reports/cache"))
    debug: bool = False

    @validator("output_dir", "cache_dir")
    def create_directories(cls, v):
        """Ensure output directories exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    def to_yaml(self, path: Optional[Path] = None) -> str:
        """Export configuration to YAML format.

        Args:
            path: Optional path to save YAML file.

        Returns:
            YAML string representation.
        """
        import yaml

        config_dict = self.dict()
        yaml_str = yaml.dump(config_dict, default_flow_style=False)

        if path:
            path.write_text(yaml_str)

        return yaml_str

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ReportConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            ReportConfig instance.
        """
        import yaml

        path = Path(path)
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


# Update forward references for nested models
SectionConfig.model_rebuild()


def create_executive_config() -> ReportConfig:
    """Create default configuration for executive report.

    Returns:
        ReportConfig for executive summary.
    """
    return ReportConfig(
        metadata=ReportMetadata(
            title="Insurance Optimization Analysis",
            subtitle="Executive Summary",
            authors=["Analytics Team"],
            keywords=["insurance", "optimization", "risk", "ergodic"],
        ),
        template="executive",
        sections=[
            SectionConfig(
                title="Key Findings", level=1, content="executive_key_findings.md", page_break=True
            ),
            SectionConfig(
                title="Performance Metrics",
                level=1,
                figures=[
                    FigureConfig(
                        name="roe_frontier",
                        caption="ROE-Ruin Efficient Frontier",
                        source="generate_roe_frontier",
                        width=8.0,
                        height=5.0,
                    )
                ],
                tables=[
                    TableConfig(
                        name="decision_matrix",
                        caption="Decision Matrix",
                        data_source="generate_decision_matrix",
                        format="markdown",
                    )
                ],
            ),
            SectionConfig(title="Recommendations", level=1, content="executive_recommendations.md"),
        ],
    )


def create_technical_config() -> ReportConfig:
    """Create default configuration for technical report.

    Returns:
        ReportConfig for technical appendix.
    """
    return ReportConfig(
        metadata=ReportMetadata(
            title="Insurance Optimization Analysis",
            subtitle="Technical Appendix",
            authors=["Analytics Team"],
            keywords=["methodology", "statistics", "validation", "ergodic"],
        ),
        template="technical",
        sections=[
            SectionConfig(
                title="Methodology",
                level=1,
                content="technical_methodology.md",
                subsections=[
                    SectionConfig(
                        title="Ergodic Theory Application", level=2, content="ergodic_theory.md"
                    ),
                    SectionConfig(
                        title="Simulation Framework", level=2, content="simulation_framework.md"
                    ),
                ],
                page_break=True,
            ),
            SectionConfig(
                title="Statistical Validation",
                level=1,
                tables=[
                    TableConfig(
                        name="convergence_metrics",
                        caption="Convergence Analysis",
                        data_source="generate_convergence_table",
                        format="latex",
                        precision=4,
                    )
                ],
                figures=[
                    FigureConfig(
                        name="convergence_plot",
                        caption="Convergence Diagnostics",
                        source="generate_convergence_plot",
                    )
                ],
            ),
        ],
    )
