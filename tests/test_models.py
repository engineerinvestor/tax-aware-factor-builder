from src.models import Account, Fund, InvestorProfile, TargetPortfolioSpec


def test_models_instantiation():
    fund = Fund(
        ticker="TEST",
        name="Test Fund",
        manager="Vanguard",
        asset_class="US_Equity",
        region="US",
        vehicle_type="ETF",
        expense_ratio=0.001,
    )
    account = Account(name="Taxable", type="Taxable", current_value=100000.0, available_funds=["ANY"])
    investor = InvestorProfile(
        federal_ordinary_rate=0.32,
        federal_ltcg_qualdiv_rate=0.15,
        state_income_rate=0.05,
        uses_municipal_bonds=False,
        time_horizon_years=20,
    )
    target = TargetPortfolioSpec(
        equity_weight=0.8,
        bond_weight=0.15,
        alts_weight=0.05,
        us_equity_share_of_equity=0.7,
        intl_equity_share_of_equity=0.3,
        reit_share_of_equity=0.05,
        factor_list=["MKT", "SMB", "HML"],
        factor_targets={"MKT": 1.0, "SMB": 0.2, "HML": 0.1},
        min_weight_per_fund=0.0,
        max_weight_per_fund=0.5,
    )

    assert fund.ticker == "TEST"
    assert account.type == "Taxable"
    assert investor.time_horizon_years == 20
    assert target.factor_targets["MKT"] == 1.0
