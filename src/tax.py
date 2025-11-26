import logging
from typing import List

import pandas as pd

from src.models import Account, Fund, InvestorProfile

logger = logging.getLogger(__name__)


def estimate_tax_drag_per_fund(fund: Fund, investor: InvestorProfile, account_type: str) -> float:
    """
    Estimate annual tax drag (decimal, e.g., 0.005 = 50 bps) for a fund in a given account type.
    """
    account_type = account_type.lower()
    if account_type == "taxable":
        qualified_rate = investor.federal_ltcg_qualdiv_rate + investor.state_income_rate
        ordinary_rate = investor.federal_ordinary_rate + investor.state_income_rate
        div_tax = fund.dividend_yield * (
            fund.qualified_dividend_ratio * qualified_rate
            + (1 - fund.qualified_dividend_ratio) * ordinary_rate
        )
        cg_tax = fund.cap_gain_distribution_yield * qualified_rate
        return float(div_tax + cg_tax)
    if account_type in {"taxdeferred", "tax_deferred", "taxfree", "tax_free"}:
        return 0.0
    raise ValueError(f"Unsupported account_type: {account_type}")


def build_tax_drag_table(funds: List[Fund], investor: InvestorProfile, accounts: List[Account]) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by [Fund, AccountType] with tax_drag values.
    Respects account.available_funds if provided (skips unavailable tickers).
    """
    rows = []
    for account in accounts:
        for fund in funds:
            if account.available_funds and account.available_funds != ["ANY"] and fund.ticker not in account.available_funds:
                continue
            drag = estimate_tax_drag_per_fund(fund, investor, account.type)
            rows.append(
                {
                    "Fund": fund.ticker,
                    "Account": account.name,
                    "AccountType": account.type,
                    "tax_drag": drag,
                }
            )
    if not rows:
        logger.warning("No rows generated for tax drag table")
        return pd.DataFrame(columns=["Fund", "Account", "AccountType", "tax_drag"]).set_index(["Fund", "AccountType"])
    df = pd.DataFrame(rows)
    return df.set_index(["Fund", "AccountType"]).sort_index()


def estimate_after_tax_expected_return(
    fund: Fund, expected_pre_tax_return: float, tax_drag: float
) -> float:
    """
    Approximate after-tax expected return for a given account type.
    """
    return float(expected_pre_tax_return - tax_drag)
