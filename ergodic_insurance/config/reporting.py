"""Output, logging, and reporting configuration.

Contains configuration classes that control where and how simulation results
are saved, logging behavior, and Excel report generation options.

Since:
    Version 0.9.0 (Issue #458)
"""

from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class OutputConfig(BaseModel):
    """Output and results configuration.

    Controls where and how simulation results are saved, including
    file formats and checkpoint frequencies.
    """

    output_directory: str = Field(default="outputs", description="Directory for saving results")
    file_format: Literal["csv", "parquet", "json"] = Field(
        default="csv", description="Output file format"
    )
    checkpoint_frequency: int = Field(
        ge=0, default=0, description="Save checkpoints every N years (0=disabled)"
    )
    detailed_metrics: bool = Field(default=True, description="Include detailed metrics in output")

    @property
    def output_path(self) -> Path:
        """Get output directory as Path object.

        Returns:
            Path object for the output directory.
        """
        return Path(self.output_directory)


class LoggingConfig(BaseModel):
    """Logging configuration.

    Controls logging behavior including level, output destinations,
    and message formatting.
    """

    enabled: bool = Field(default=True, description="Enable logging")
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )
    log_file: Optional[str] = Field(
        default=None, description="Log file path (None=no file logging)"
    )
    console_output: bool = Field(default=True, description="Log to console")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format",
    )


class ExcelReportConfig(BaseModel):
    """Configuration for Excel report generation.

    This is the canonical definition used throughout the codebase.
    Both ``ExcelReporter`` and the unified config hierarchy (``ConfigV2``)
    use this class.

    Attributes:
        enabled: Whether Excel reporting is enabled.
        output_path: Directory for output files.
        include_balance_sheet: Whether to include balance sheet.
        include_income_statement: Whether to include income statement.
        include_cash_flow: Whether to include cash flow statement.
        include_reconciliation: Whether to include reconciliation sheet.
        include_metrics_dashboard: Whether to include metrics dashboard.
        include_pivot_data: Whether to include pivot-ready data sheet.
        formatting: Custom formatting options.
        engine: Excel engine to use ('xlsxwriter', 'openpyxl', 'auto', 'pandas').
        currency_format: Currency format string.
        decimal_places: Number of decimal places for numbers.
        date_format: Date format string.
    """

    enabled: bool = Field(default=True, description="Whether Excel reporting is enabled")
    output_path: Path = Field(default=Path("./reports"), description="Directory for Excel reports")
    include_balance_sheet: bool = Field(default=True, description="Include balance sheet")
    include_income_statement: bool = Field(default=True, description="Include income statement")
    include_cash_flow: bool = Field(default=True, description="Include cash flow statement")
    include_reconciliation: bool = Field(default=True, description="Include reconciliation report")
    include_metrics_dashboard: bool = Field(default=True, description="Include metrics dashboard")
    include_pivot_data: bool = Field(default=True, description="Include pivot-ready data")
    formatting: Optional[Dict[str, Any]] = Field(
        default=None, description="Custom formatting options"
    )
    engine: str = Field(default="auto", description="Excel engine: xlsxwriter, openpyxl, or auto")
    currency_format: str = Field(default="$#,##0", description="Currency format string")
    decimal_places: int = Field(default=0, ge=0, le=10, description="Number of decimal places")
    date_format: str = Field(default="yyyy-mm-dd", description="Date format string")

    @field_validator("engine")
    @classmethod
    def validate_engine(cls, v: str) -> str:
        """Validate Excel engine selection.

        Args:
            v: Engine name to validate.

        Returns:
            Validated engine name.

        Raises:
            ValueError: If engine is not valid.
        """
        valid_engines = ["xlsxwriter", "openpyxl", "auto", "pandas"]
        if v not in valid_engines:
            raise ValueError(f"Invalid Excel engine: {v}. Must be one of {valid_engines}")
        return v
