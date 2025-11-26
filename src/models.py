from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd


@dataclass
class Fund:
    ticker: str
    name: str
    manager: str
    asset_class: str
    region: str
    vehicle_type: str
    expense_ratio: float

    is_tax_efficient: bool = True
    dividend_yield: float = 0.0
    qualified_dividend_ratio: float = 1.0
    cap_gain_distribution_yield: float = 0.0
    income_character: str = "mostly_dividends"

    allowed_accounts: List[str] = field(default_factory=lambda: ["Taxable", "TaxDeferred", "TaxFree"])

    historical_returns: Optional[pd.Series] = None
    factor_loadings: Dict[str, float] = field(default_factory=dict)
    alpha: Optional[float] = None
    regression_stats: Optional[Dict] = None


@dataclass
class Account:
    name: str
    type: str  # "Taxable", "TaxDeferred", "TaxFree"
    current_value: float
    available_funds: List[str]
    priority_for_tax_inefficient_assets: int = 1


@dataclass
class InvestorProfile:
    federal_ordinary_rate: float
    federal_ltcg_qualdiv_rate: float
    state_income_rate: float
    uses_municipal_bonds: bool
    time_horizon_years: int
    risk_tolerance: Optional[str] = None


@dataclass
class TargetPortfolioSpec:
    equity_weight: float
    bond_weight: float
    alts_weight: float

    us_equity_share_of_equity: float
    intl_equity_share_of_equity: float
    reit_share_of_equity: Optional[float]

    factor_list: List[str]
    factor_targets: Dict[str, float]

    min_weight_per_fund: float
    max_weight_per_fund: float
    min_weight_per_manager: Optional[Dict[str, float]] = None
    max_weight_per_manager: Optional[Dict[str, float]] = None
    min_weight_per_region: Optional[Dict[str, float]] = None
    max_weight_per_region: Optional[Dict[str, float]] = None

    max_number_of_funds: Optional[int] = None
    weight_floor: float = 0.0


def default_target_portfolio_spec() -> TargetPortfolioSpec:
    """
    Provide a reasonable default TargetPortfolioSpec aligned with requirements.
    """
    return TargetPortfolioSpec(
        equity_weight=0.8,
        bond_weight=0.1,
        alts_weight=0.1,
        us_equity_share_of_equity=0.6,
        intl_equity_share_of_equity=0.4,
        reit_share_of_equity=0.05,
        factor_list=["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"],
        factor_targets={"MKT_RF": 1.0, "SMB": 0.2, "HML": 0.2, "RMW": 0.10, "CMA": 0.05, "MOM": 0.2},
        min_weight_per_fund=0.0,
        max_weight_per_fund=0.50,
        min_weight_per_manager=None,
        max_weight_per_manager={"Vanguard": 0.7, "Avantis": 0.6, "DFA": 0.6, "AQR": 0.5},
        min_weight_per_region=None,
        max_weight_per_region=None,
        max_number_of_funds=None,
        weight_floor=0.01,
    )
