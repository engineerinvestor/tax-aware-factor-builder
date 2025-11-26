import pandas as pd

from src.models import Account, Fund, InvestorProfile
from src.tax import build_tax_drag_table, estimate_after_tax_expected_return, estimate_tax_drag_per_fund


def _sample_fund(div_yield=0.02, qdr=0.8, cg_yield=0.01):
    return Fund(
        ticker="F1",
        name="Fund1",
        manager="Vanguard",
        asset_class="US_Equity",
        region="US",
        vehicle_type="ETF",
        expense_ratio=0.001,
        dividend_yield=div_yield,
        qualified_dividend_ratio=qdr,
        cap_gain_distribution_yield=cg_yield,
    )


def _investor():
    return InvestorProfile(
        federal_ordinary_rate=0.32,
        federal_ltcg_qualdiv_rate=0.15,
        state_income_rate=0.05,
        uses_municipal_bonds=False,
        time_horizon_years=20,
    )


def test_estimate_tax_drag_per_fund_taxable():
    fund = _sample_fund()
    investor = _investor()
    drag = estimate_tax_drag_per_fund(fund, investor, "Taxable")
    qualified_rate = investor.federal_ltcg_qualdiv_rate + investor.state_income_rate
    ordinary_rate = investor.federal_ordinary_rate + investor.state_income_rate
    expected = fund.dividend_yield * (fund.qualified_dividend_ratio * qualified_rate + (1 - fund.qualified_dividend_ratio) * ordinary_rate) + fund.cap_gain_distribution_yield * qualified_rate
    assert abs(drag - expected) < 1e-9


def test_build_tax_drag_table_respects_available_funds():
    fund = _sample_fund()
    funds = [fund]
    investor = _investor()
    accounts = [
        Account(name="Taxable", type="Taxable", current_value=100000, available_funds=["F1"]),
        Account(name="401k", type="TaxDeferred", current_value=200000, available_funds=["NONE"]),
    ]
    df = build_tax_drag_table(funds, investor, accounts)
    assert ("F1", "Taxable") in df.index
    assert ("F1", "TaxDeferred") not in df.index
    assert df.loc[("F1", "Taxable"), "tax_drag"] > 0


def test_estimate_after_tax_expected_return():
    fund = _sample_fund()
    result = estimate_after_tax_expected_return(fund, expected_pre_tax_return=0.06, tax_drag=0.01)
    assert abs(result - 0.05) < 1e-12
