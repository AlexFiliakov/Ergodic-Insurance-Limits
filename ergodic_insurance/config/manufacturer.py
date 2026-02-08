"""Business entity configuration including manufacturer, expense, and industry profiles.

Contains the core business model configuration classes: manufacturer financial
parameters, expense ratio breakdowns, depreciation policies, and industry-specific
default profiles (manufacturing, service, retail).

Since:
    Version 0.9.0 (Issue #458)
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ExpenseRatioConfig(BaseModel):
    """Configuration for expense categorization and allocation.

    Defines how revenue translates to expenses with proper GAAP categorization
    between COGS and operating expenses (SG&A).

    Issue #255: COGS and SG&A breakdown ratios are now configurable to allow
    the Manufacturer to calculate these values explicitly, rather than having
    the Reporting layer estimate them with hardcoded ratios.
    """

    gross_margin_ratio: float = Field(
        default=0.15,
        gt=0,
        lt=1,
        description="Gross margin ratio (Revenue - COGS) / Revenue",
    )
    sga_expense_ratio: float = Field(
        default=0.07,
        gt=0,
        lt=1,
        description="SG&A expenses as percentage of revenue",
    )
    manufacturing_depreciation_allocation: float = Field(
        default=0.7,
        ge=0,
        le=1,
        description="Percentage of depreciation allocated to COGS (manufacturing)",
    )
    admin_depreciation_allocation: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="Percentage of depreciation allocated to SG&A (administrative)",
    )

    # COGS breakdown ratios (Issue #255)
    direct_materials_ratio: float = Field(
        default=0.4,
        ge=0,
        le=1,
        description="Direct materials as percentage of COGS (excluding depreciation)",
    )
    direct_labor_ratio: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="Direct labor as percentage of COGS (excluding depreciation)",
    )
    manufacturing_overhead_ratio: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="Manufacturing overhead as percentage of COGS (excluding depreciation)",
    )

    # SG&A breakdown ratios (Issue #255)
    selling_expense_ratio: float = Field(
        default=0.4,
        ge=0,
        le=1,
        description="Selling expenses as percentage of SG&A (excluding depreciation)",
    )
    general_admin_ratio: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="General & Admin as percentage of SG&A (excluding depreciation)",
    )

    @model_validator(mode="after")
    def validate_depreciation_allocation(self):
        """Ensure depreciation allocations sum to 100%."""
        total = self.manufacturing_depreciation_allocation + self.admin_depreciation_allocation
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Depreciation allocations must sum to 100%, got {total*100:.1f}%")
        return self

    @model_validator(mode="after")
    def validate_cogs_breakdown(self):
        """Ensure COGS breakdown ratios sum to 100%."""
        total = (
            self.direct_materials_ratio
            + self.direct_labor_ratio
            + self.manufacturing_overhead_ratio
        )
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"COGS breakdown ratios must sum to 100%, got {total*100:.1f}%")
        return self

    @model_validator(mode="after")
    def validate_sga_breakdown(self):
        """Ensure SG&A breakdown ratios sum to 100%."""
        total = self.selling_expense_ratio + self.general_admin_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"SG&A breakdown ratios must sum to 100%, got {total*100:.1f}%")
        return self

    @property
    def cogs_ratio(self) -> float:
        """Calculate COGS as percentage of revenue."""
        return 1.0 - self.gross_margin_ratio

    @property
    def operating_margin_ratio(self) -> float:
        """Calculate operating margin after all operating expenses."""
        return self.gross_margin_ratio - self.sga_expense_ratio


class DepreciationConfig(BaseModel):
    """Configuration for depreciation and amortization tracking.

    Defines how fixed assets depreciate and prepaid expenses amortize over time.
    """

    ppe_useful_life_years: float = Field(
        default=10,
        gt=0,
        le=50,
        description="Average useful life of PP&E in years for straight-line depreciation",
    )
    prepaid_insurance_amortization_months: int = Field(
        default=12,
        gt=0,
        le=24,
        description="Number of months over which prepaid insurance amortizes",
    )
    initial_accumulated_depreciation: float = Field(
        default=0, ge=0, description="Starting accumulated depreciation balance"
    )

    @property
    def annual_depreciation_rate(self) -> float:
        """Calculate annual depreciation rate."""
        return 1.0 / self.ppe_useful_life_years

    @property
    def monthly_insurance_amortization_rate(self) -> float:
        """Calculate monthly insurance amortization rate."""
        return 1.0 / self.prepaid_insurance_amortization_months


class ManufacturerConfig(BaseModel):
    """Financial parameters for the widget manufacturer.

    This class defines the core financial parameters used to initialize
    and configure a widget manufacturing company in the simulation. All
    parameters are validated to ensure realistic business constraints.

    Attributes:
        initial_assets: Starting asset value in dollars. Must be positive.
        asset_turnover_ratio: Revenue per dollar of assets. Typically 0.5-2.0
            for manufacturing companies.
        base_operating_margin: Core operating margin before insurance costs
            (EBIT before insurance / Revenue). Typically 5-15% for healthy
            manufacturers.
        tax_rate: Corporate tax rate. Typically 20-30% depending on jurisdiction.
        retention_ratio: Portion of earnings retained vs distributed as dividends.
            Higher retention supports faster growth.
        ppe_ratio: Property, Plant & Equipment allocation ratio as fraction of
            initial assets. Defaults based on operating margin if not specified.

    Examples:
        Conservative manufacturer::

            config = ManufacturerConfig(
                initial_assets=5_000_000,
                asset_turnover_ratio=0.6,  # Low turnover
                base_operating_margin=0.05,      # 5% base margin
                tax_rate=0.25,
                retention_ratio=0.9         # High retention
            )

        Aggressive growth manufacturer::

            config = ManufacturerConfig(
                initial_assets=20_000_000,
                asset_turnover_ratio=1.2,  # High turnover
                base_operating_margin=0.12,      # 12% base margin
                tax_rate=0.25,
                retention_ratio=1.0         # Full retention
            )

        Custom PP&E allocation::

            config = ManufacturerConfig(
                initial_assets=15_000_000,
                asset_turnover_ratio=0.9,
                base_operating_margin=0.10,
                tax_rate=0.25,
                retention_ratio=0.8,
                ppe_ratio=0.6  # Override default PP&E allocation
            )

    Note:
        The asset turnover ratio and base operating margin together determine
        the core return on assets (ROA) before insurance costs and taxes.
        Actual operating margins will be lower when insurance costs are included.
    """

    initial_assets: float = Field(
        default=10_000_000, gt=0, description="Starting asset value in dollars"
    )
    asset_turnover_ratio: float = Field(
        default=0.8, gt=0, le=5, description="Revenue per dollar of assets"
    )
    base_operating_margin: float = Field(
        default=0.08,
        gt=-1,
        lt=1,
        description="Core operating margin before insurance costs (EBIT before insurance / Revenue)",
    )
    tax_rate: float = Field(default=0.25, ge=0, le=1, description="Corporate tax rate")
    nol_carryforward_enabled: bool = Field(
        default=True,
        description="Enable NOL carryforward tracking per IRC §172. "
        "When False, losses generate no future tax benefit (legacy behavior).",
    )
    nol_limitation_pct: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="NOL deduction limitation as fraction of taxable income. "
        "Set to 0.80 per IRC §172(a)(2) post-TCJA. "
        "Set to 1.0 for pre-2018 NOLs or non-US jurisdictions.",
    )
    retention_ratio: float = Field(
        default=0.7, ge=0, le=1, description="Portion of earnings retained"
    )
    ppe_ratio: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Property, Plant & Equipment allocation ratio (fraction of initial assets). "
        "If None, defaults based on operating margin: <10%: 0.3, 10-15%: 0.5, >15%: 0.7",
    )
    insolvency_tolerance: float = Field(
        default=10_000,
        gt=0,
        description="Insolvency threshold in dollars. Company is considered insolvent when "
        "equity falls below this level. Default of $10,000 (0.1%% of typical $10M assets) "
        "represents practical insolvency where company cannot maintain operations.",
    )
    expense_ratios: Optional[ExpenseRatioConfig] = Field(
        default=None,
        description="Expense ratio configuration for COGS and SG&A breakdown. "
        "If None, default ratios from ExpenseRatioConfig are used. "
        "(Issue #255: Enables explicit COGS/SG&A calculation in Manufacturer)",
    )

    # Mid-year liquidity configuration (Issue #279)
    premium_payment_month: int = Field(
        default=0,
        ge=0,
        le=11,
        description="Month when annual insurance premium is paid (0-11, where 0=January). "
        "Used for intra-period liquidity estimation to detect mid-year insolvency.",
    )
    revenue_pattern: Literal["uniform", "seasonal", "back_loaded"] = Field(
        default="uniform",
        description="Revenue distribution pattern throughout the year. "
        "'uniform': equal monthly revenue, 'seasonal': higher in Q4, "
        "'back_loaded': 60% in H2. Used for mid-year liquidity estimation.",
    )
    check_intra_period_liquidity: bool = Field(
        default=True,
        description="Whether to check for potential mid-year insolvency by estimating "
        "minimum cash point within each period. When True, the simulation estimates "
        "the lowest cash point and triggers insolvency if it goes negative.",
    )

    # Reserve re-estimation configuration (Issue #470, ASC 944-40-25)
    enable_reserve_development: bool = Field(
        default=False,
        description="Enable stochastic reserve re-estimation per ASC 944-40-25. "
        "When True, claim reserves start as noisy estimates that converge "
        "toward the true ultimate over the claim's life. Default off.",
    )
    reserve_noise_std: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Std dev of initial reserve estimation noise as fraction of "
        "true ultimate (typically 0.15-0.40 depending on line of business). "
        "Noise shrinks proportionally to claim maturity.",
    )

    # Capital expenditure configuration (Issue #543)
    capex_to_depreciation_ratio: float = Field(
        default=1.0,
        ge=0.0,
        le=5.0,
        description="Capital expenditure as a multiple of depreciation expense. "
        "1.0 = maintenance capex (replace depreciated assets). "
        ">1.0 = growth capex (expand capacity). "
        "0.0 = no reinvestment (legacy behavior). "
        "Typical range: 1.0-2.5 for manufacturers "
        "(Damodaran sector data).",
    )

    @model_validator(mode="after")
    def set_default_ppe_ratio(self):
        """Set default PPE ratio based on operating margin if not provided."""
        if self.ppe_ratio is None:
            if self.base_operating_margin < 0.10:
                self.ppe_ratio = 0.3  # Low margin businesses need more working capital
            elif self.base_operating_margin < 0.15:
                self.ppe_ratio = 0.5  # Medium margin can support moderate PP&E
            else:
                self.ppe_ratio = 0.7  # High margin businesses can support more PP&E
        return self

    @field_validator("base_operating_margin")
    @classmethod
    def validate_margin(cls, v: float) -> float:
        """Warn if base operating margin is unusually high or negative.

        Args:
            v: Base operating margin value to validate (as decimal, e.g., 0.1 for 10%).

        Returns:
            float: The validated base operating margin value.

        Note:
            Margins above 30% are flagged as unusual for manufacturing.
            Negative margins indicate unprofitable operations before insurance.
        """
        if v > 0.3:
            print(f"Warning: Base operating margin {v:.1%} is unusually high")
        elif v < 0:
            print(f"Warning: Base operating margin {v:.1%} is negative")
        return v

    @classmethod
    def from_industry_config(cls, industry_config, **kwargs):
        """Create ManufacturerConfig from an IndustryConfig instance.

        Args:
            industry_config: IndustryConfig instance with industry-specific parameters
            **kwargs: Additional parameters to override or supplement

        Returns:
            ManufacturerConfig instance with parameters derived from industry config
        """
        # Map industry config parameters to manufacturer config
        # Use provided kwargs to override any derived values
        config_params = kwargs.copy()

        # Set base operating margin from industry config if not provided
        if "base_operating_margin" not in config_params:
            config_params["base_operating_margin"] = industry_config.operating_margin

        # Set PPE ratio from industry config if not provided
        if "ppe_ratio" not in config_params:
            config_params["ppe_ratio"] = industry_config.ppe_ratio

        # Set other defaults if not provided
        if "initial_assets" not in config_params:
            config_params["initial_assets"] = 10_000_000  # Default $10M
        if "asset_turnover_ratio" not in config_params:
            config_params["asset_turnover_ratio"] = 0.8  # Default 0.8x
        if "tax_rate" not in config_params:
            config_params["tax_rate"] = 0.25  # Default 25%
        if "retention_ratio" not in config_params:
            config_params["retention_ratio"] = 0.7  # Default 70%

        return cls(**config_params)


@dataclass
class IndustryConfig:
    """Base configuration for different industry types.

    This class defines industry-specific financial parameters that determine
    how businesses operate, including working capital needs, margin structures,
    asset composition, and depreciation policies.

    Attributes:
        industry_type: Name of the industry (e.g., 'manufacturing', 'services')

        Working capital ratios:
        days_sales_outstanding: Average collection period for receivables (days)
        days_inventory_outstanding: Average inventory holding period (days)
        days_payables_outstanding: Average payment period to suppliers (days)

        Margin structure:
        gross_margin: Gross profit as percentage of revenue
        operating_expense_ratio: Operating expenses as percentage of revenue

        Asset composition:
        current_asset_ratio: Current assets as fraction of total assets
        ppe_ratio: Property, Plant & Equipment as fraction of total assets
        intangible_ratio: Intangible assets as fraction of total assets

        Depreciation:
        ppe_useful_life: Average useful life of PP&E in years
        depreciation_method: Method for calculating depreciation
    """

    industry_type: str = "manufacturing"

    # Working capital ratios (in days)
    days_sales_outstanding: float = 45
    days_inventory_outstanding: float = 60
    days_payables_outstanding: float = 30

    # Margin structure (as percentages)
    gross_margin: float = 0.35
    operating_expense_ratio: float = 0.25

    # Asset composition (must sum to 1.0)
    current_asset_ratio: float = 0.4
    ppe_ratio: float = 0.5
    intangible_ratio: float = 0.1

    # Depreciation settings
    ppe_useful_life: int = 10  # years
    depreciation_method: str = "straight_line"

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self):
        """Validate that all parameters are within reasonable bounds."""
        # Validate margins
        assert (
            0 <= self.gross_margin <= 1
        ), f"Gross margin must be between 0 and 1, got {self.gross_margin}"
        assert (
            0 <= self.operating_expense_ratio <= 1
        ), f"Operating expense ratio must be between 0 and 1, got {self.operating_expense_ratio}"

        # Validate asset composition
        asset_sum = self.current_asset_ratio + self.ppe_ratio + self.intangible_ratio
        assert abs(asset_sum - 1.0) < 0.01, f"Asset ratios must sum to 1.0, got {asset_sum}"

        # Validate working capital days
        assert self.days_sales_outstanding >= 0, "Days sales outstanding must be non-negative"
        assert (
            self.days_inventory_outstanding >= 0
        ), "Days inventory outstanding must be non-negative"
        assert self.days_payables_outstanding >= 0, "Days payables outstanding must be non-negative"

        # Validate depreciation
        assert self.ppe_useful_life > 0, "PPE useful life must be positive"
        assert self.depreciation_method in [
            "straight_line",
            "declining_balance",
        ], f"Unknown depreciation method: {self.depreciation_method}"

    @property
    def working_capital_days(self) -> float:
        """Calculate net working capital cycle in days."""
        return (
            self.days_sales_outstanding
            + self.days_inventory_outstanding
            - self.days_payables_outstanding
        )

    @property
    def operating_margin(self) -> float:
        """Calculate operating margin (EBIT margin)."""
        return self.gross_margin - self.operating_expense_ratio


class ManufacturingConfig(IndustryConfig):
    """Configuration for manufacturing companies.

    Manufacturing businesses typically have:
    - Significant inventory holdings
    - Moderate to high PP&E requirements
    - Working capital needs for raw materials and WIP
    - Gross margins of 25-40%
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with manufacturing-specific defaults."""
        defaults: Dict[str, Any] = {
            "industry_type": "manufacturing",
            "days_sales_outstanding": 45,
            "days_inventory_outstanding": 60,
            "days_payables_outstanding": 30,
            "gross_margin": 0.35,
            "operating_expense_ratio": 0.25,
            "current_asset_ratio": 0.4,
            "ppe_ratio": 0.5,
            "intangible_ratio": 0.1,
            "ppe_useful_life": 10,
            "depreciation_method": "straight_line",
        }
        # Override defaults with any provided kwargs
        defaults.update(kwargs)
        super().__init__(**defaults)


class ServiceConfig(IndustryConfig):
    """Configuration for service companies.

    Service businesses typically have:
    - Minimal or no inventory
    - Lower PP&E requirements
    - Faster cash conversion cycles
    - Higher gross margins but also higher operating expenses
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with service-specific defaults."""
        defaults: Dict[str, Any] = {
            "industry_type": "services",
            "days_sales_outstanding": 30,
            "days_inventory_outstanding": 0,  # No inventory for services
            "days_payables_outstanding": 20,
            "gross_margin": 0.60,
            "operating_expense_ratio": 0.45,
            "current_asset_ratio": 0.6,
            "ppe_ratio": 0.2,  # Less capital intensive
            "intangible_ratio": 0.2,  # More intangibles (brand, IP)
            "ppe_useful_life": 5,
            "depreciation_method": "straight_line",
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


class RetailConfig(IndustryConfig):
    """Configuration for retail companies.

    Retail businesses typically have:
    - High inventory turnover
    - Moderate PP&E (stores, fixtures)
    - Fast cash collection (often immediate)
    - Lower gross margins but efficient operations
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with retail-specific defaults."""
        defaults: Dict[str, Any] = {
            "industry_type": "retail",
            "days_sales_outstanding": 5,  # Mostly cash/credit card sales
            "days_inventory_outstanding": 45,
            "days_payables_outstanding": 35,
            "gross_margin": 0.30,
            "operating_expense_ratio": 0.22,
            "current_asset_ratio": 0.5,
            "ppe_ratio": 0.4,
            "intangible_ratio": 0.1,
            "ppe_useful_life": 7,
            "depreciation_method": "straight_line",
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
