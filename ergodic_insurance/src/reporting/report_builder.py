"""Base report builder class for automated report generation.

This module provides the core ReportBuilder class that handles report compilation,
figure embedding, section management, and content generation.
"""

from abc import ABC, abstractmethod
from datetime import datetime
import json
import logging
from pathlib import Path
import pickle
import shutil
from typing import Any, Dict, List, Optional, Union

from jinja2 import Environment, FileSystemLoader, Template
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..reporting.cache_manager import CacheManager
from .config import FigureConfig, ReportConfig, SectionConfig, TableConfig
from .table_generator import TableGenerator

logger = logging.getLogger(__name__)


class ReportBuilder(ABC):
    """Base class for building automated reports.

    This abstract base class provides common functionality for generating
    different types of reports, including section management, figure embedding,
    and template rendering.

    Attributes:
        config: Report configuration object.
        cache_manager: Cache manager for figures and data.
        table_generator: Table generation utility.
        content: Accumulated report content.
        figures: List of generated figures.
        tables: List of generated tables.
    """

    def __init__(self, config: ReportConfig, cache_dir: Optional[Path] = None):
        """Initialize ReportBuilder.

        Args:
            config: Report configuration.
            cache_dir: Optional cache directory override.
        """
        self.config = config
        # Ensure cache_dir is created as a subdirectory of output_dir if not absolute
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        elif config.cache_dir.is_absolute():
            self.cache_dir = config.cache_dir
        else:
            # Make cache_dir relative to output_dir
            self.cache_dir = config.output_dir / "cache"
        
        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        
        from ..reporting.cache_manager import CacheConfig
        cache_config = CacheConfig(cache_dir=self.cache_dir)
        self.cache_manager = CacheManager(cache_config)
        self.table_generator = TableGenerator()

        self.content: List[str] = []
        self.figures: List[Dict[str, Any]] = []
        self.tables: List[Dict[str, Any]] = []

        # Setup template environment
        self.template_dir = Path(__file__).parent / "templates"
        if self.template_dir.exists():
            self.env = Environment(loader=FileSystemLoader(str(self.template_dir)))
        else:
            self.env = Environment()

        # Ensure output directories exist
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def generate(self) -> Path:
        """Generate the complete report.

        Returns:
            Path to generated report file.
        """
        pass

    def build_section(self, section: SectionConfig) -> str:
        """Build a report section.

        Args:
            section: Section configuration.

        Returns:
            Formatted section content.
        """
        content_parts = []

        # Add section heading
        heading = "#" * section.level + " " + section.title
        content_parts.append(heading)
        content_parts.append("")  # Empty line after heading

        # Add text content
        if section.content:
            text = self._load_content(section.content)
            content_parts.append(text)
            content_parts.append("")

        # Add figures
        for fig_config in section.figures:
            fig_content = self._embed_figure(fig_config)
            content_parts.append(fig_content)
            content_parts.append("")

        # Add tables
        for table_config in section.tables:
            table_content = self._embed_table(table_config)
            content_parts.append(table_content)
            content_parts.append("")

        # Process subsections
        for subsection in section.subsections:
            subsection_content = self.build_section(subsection)
            content_parts.append(subsection_content)
            content_parts.append("")

        # Add page break if needed
        if section.page_break:
            content_parts.append("\\newpage")
            content_parts.append("")

        return "\n".join(content_parts)

    def _load_content(self, content_ref: str) -> str:
        """Load text content from file or template.

        Args:
            content_ref: Content reference (file path or template name).

        Returns:
            Loaded content string.
        """
        # Check if it's a template
        if content_ref.endswith(".md"):
            template_path = self.template_dir / content_ref
            if template_path.exists():
                template = self.env.get_template(content_ref)
                return template.render(  # type: ignore[no-any-return]
                    metadata=self.config.metadata, figures=self.figures, tables=self.tables
                )

        # Check if it's a file path
        content_path = Path(content_ref)
        if content_path.exists():
            return content_path.read_text()

        # Return as literal content
        return content_ref

    def _embed_figure(self, fig_config: FigureConfig) -> str:
        """Embed a figure in the report.

        Args:
            fig_config: Figure configuration.

        Returns:
            Figure embedding markdown/latex.
        """
        # Check cache first
        if fig_config.cache_key:
            cached_path = self.cache_dir / f"{fig_config.cache_key}.png"
            if cached_path.exists():
                fig_path = cached_path
            else:
                fig_path = self._generate_figure(fig_config)
        else:
            fig_path = self._generate_figure(fig_config)

        # Store figure info
        self.figures.append(
            {"name": fig_config.name, "path": str(fig_path), "caption": fig_config.caption}
        )

        # Generate markdown for figure with relative path
        # Get relative path from output_dir to cache_dir
        try:
            rel_path = fig_path.relative_to(self.config.output_dir)
        except ValueError:
            # If not relative to output_dir, use absolute path
            rel_path = fig_path
        
        return f"![{fig_config.caption}]({rel_path.as_posix()})\n*Figure: {fig_config.caption}*"

    def _generate_figure(self, fig_config: FigureConfig) -> Path:
        """Generate a figure from configuration.

        Args:
            fig_config: Figure configuration.

        Returns:
            Path to generated figure.
        """
        source = fig_config.source

        # If source is a file path
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            if source_path.exists():
                # Copy to cache directory
                dest_path = self.cache_dir / f"{fig_config.name}.png"
                shutil.copy(source_path, dest_path)
                return dest_path

            # If it's a generation function name
            elif str(source).startswith("generate_"):
                func_name = str(source)
                if hasattr(self, func_name):
                    fig = getattr(self, func_name)(fig_config)
                    dest_path = self.cache_dir / f"{fig_config.name}.png"
                    fig.savefig(dest_path, dpi=fig_config.dpi, bbox_inches="tight")
                    plt.close(fig)
                    return dest_path

        # Fallback: create placeholder figure
        fig, ax = plt.subplots(figsize=(fig_config.width, fig_config.height))
        ax.text(0.5, 0.5, f"Figure: {fig_config.name}", ha="center", va="center", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

        dest_path = self.cache_dir / f"{fig_config.name}.png"
        fig.savefig(dest_path, dpi=fig_config.dpi, bbox_inches="tight")
        plt.close(fig)

        return dest_path

    def _embed_table(self, table_config: TableConfig) -> str:
        """Embed a table in the report.

        Args:
            table_config: Table configuration.

        Returns:
            Formatted table string.
        """
        # Load data
        data = self._load_table_data(table_config.data_source)

        # Generate table
        table_content = self.table_generator.generate(
            data=data,
            caption=table_config.caption,
            columns=table_config.columns,
            index=table_config.index,
            format=table_config.format,
            precision=table_config.precision,
            style=table_config.style,
        )

        # Store table info
        self.tables.append(
            {"name": table_config.name, "caption": table_config.caption, "content": table_content}
        )

        return table_content

    def _load_table_data(self, data_source: Union[str, Path]) -> pd.DataFrame:
        """Load table data from source.

        Args:
            data_source: Data source reference.

        Returns:
            DataFrame with table data.
        """
        # Check if it's a file path
        if isinstance(data_source, (str, Path)):
            source_path = Path(data_source)
            if source_path.exists():
                if source_path.suffix == ".csv":
                    return pd.read_csv(source_path)
                elif source_path.suffix == ".parquet":
                    return pd.read_parquet(source_path)
                elif source_path.suffix == ".json":
                    return pd.read_json(source_path)

            # If it's a generation function name
            elif str(data_source).startswith("generate_"):
                func_name = str(data_source)
                if hasattr(self, func_name):
                    return getattr(self, func_name)()  # type: ignore[no-any-return]

        # Fallback: create sample data
        return pd.DataFrame({"Column A": [1, 2, 3], "Column B": [4, 5, 6], "Column C": [7, 8, 9]})

    def compile_report(self) -> str:
        """Compile all sections into complete report.

        Returns:
            Complete report content as string.
        """
        self.content = []

        # Add metadata header
        header = self._generate_header()
        self.content.append(header)

        # Build all sections
        for section in self.config.sections:
            section_content = self.build_section(section)
            self.content.append(section_content)

        # Add footer if needed
        footer = self._generate_footer()
        if footer:
            self.content.append(footer)

        return "\n".join(self.content)

    def _generate_header(self) -> str:
        """Generate report header with metadata.

        Returns:
            Formatted header string.
        """
        metadata = self.config.metadata
        header_parts = []

        # Title page
        header_parts.append(f"# {metadata.title}")
        if metadata.subtitle:
            header_parts.append(f"## {metadata.subtitle}")
        header_parts.append("")

        # Authors and date
        if metadata.authors:
            header_parts.append(f"**Authors:** {', '.join(metadata.authors)}")
        header_parts.append(f"**Date:** {metadata.date.strftime('%B %d, %Y')}")
        header_parts.append(f"**Version:** {metadata.version}")

        # Organization and confidentiality
        header_parts.append(f"**Organization:** {metadata.organization}")
        header_parts.append(f"**Classification:** {metadata.confidentiality}")
        header_parts.append("")

        # Abstract
        if metadata.abstract:
            header_parts.append("## Abstract")
            header_parts.append(metadata.abstract)
            header_parts.append("")

        # Page break after header
        header_parts.append("\\newpage")
        header_parts.append("")

        return "\n".join(header_parts)

    def _generate_footer(self) -> str:
        """Generate report footer.

        Returns:
            Formatted footer string.
        """
        return ""

    def save(self, format: str = "markdown") -> Path:
        """Save report in specified format.

        Args:
            format: Output format (markdown, html, pdf).

        Returns:
            Path to saved report.
        """
        # Compile report content
        content = self.compile_report()

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.config.metadata.title.replace(' ', '_')}_{timestamp}"

        if format == "markdown":
            output_path = self.config.output_dir / f"{base_name}.md"
            # Write with explicit UTF-8 encoding and handle errors
            with open(output_path, 'w', encoding='utf-8', errors='replace') as f:
                f.write(content)

        elif format == "html":
            import markdown2

            html_content = markdown2.markdown(content, extras=["tables", "fenced-code-blocks"])
            output_path = self.config.output_dir / f"{base_name}.html"

            # Wrap in HTML template
            html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{self.config.metadata.title}</title>
    <style>
        body {{ font-family: {self.config.style.font_family};
                font-size: {self.config.style.font_size}pt;
                line-height: {self.config.style.line_spacing};
                max-width: 900px;
                margin: 0 auto;
                padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        img {{ max-width: 100%; height: auto; }}
        pre {{ background-color: #f4f4f4; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""
            # Write with explicit UTF-8 encoding and handle errors
            with open(output_path, 'w', encoding='utf-8', errors='replace') as f:
                f.write(html_template)

        elif format == "pdf":
            # First convert to HTML, then to PDF
            import markdown2

            html_content = markdown2.markdown(content, extras=["tables", "fenced-code-blocks"])

            # Create temporary HTML file
            temp_html = self.config.output_dir / f"{base_name}_temp.html"
            with open(temp_html, 'w', encoding='utf-8', errors='replace') as f:
                f.write(html_content)

            # Convert to PDF using weasyprint
            output_path = self.config.output_dir / f"{base_name}.pdf"
            try:
                from weasyprint import HTML

                HTML(string=html_content).write_pdf(output_path)
            except ImportError:
                logger.warning("WeasyPrint not available, saving as HTML instead")
                output_path = self.config.output_dir / f"{base_name}.html"
                temp_html.rename(output_path)
            finally:
                if temp_html.exists():
                    temp_html.unlink()

        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Report saved to: {output_path}")
        return output_path
