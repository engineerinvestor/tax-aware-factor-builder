import numpy as np

from src.models import Fund, TargetPortfolioSpec
from src.reporting import format_run_summary


def test_format_run_summary_includes_key_fields():
    funds = [
        Fund(
            ticker="A",
            name="Fund A",
            manager="Vanguard",
            asset_class="US_Equity",
            region="US",
            vehicle_type="ETF",
            expense_ratio=0.001,
            factor_loadings={"MKT": 1.0},
        )
    ]
    target = TargetPortfolioSpec(
        equity_weight=0.7,
        bond_weight=0.25,
        alts_weight=0.05,
        us_equity_share_of_equity=0.6,
        intl_equity_share_of_equity=0.4,
        reit_share_of_equity=None,
        factor_list=["MKT"],
        factor_targets={"MKT": 0.9},
        min_weight_per_fund=0.0,
        max_weight_per_fund=1.0,
    )
    weights = np.array([1.0])
    portfolio_factors = {"MKT": 1.0}
    breakdown = {
        "asset_class": {"US_Equity": 1.0},
        "manager": {"Vanguard": 1.0},
        "region": {"US": 1.0},
    }
    summary = format_run_summary(
        config={
            "data_mode": "csv",
            "start_date": "2020-01-01",
            "frequency": "M",
            "factor_data_path": "path/to/factors.csv",
            "weight_floor": 0.01,
        },
        target_spec=target,
        weights=weights,
        funds=funds,
        portfolio_factors=portfolio_factors,
        asset_breakdown=breakdown,
        tax_stats={"optimized_bps": 12.3, "naive_bps": 15.3},
    )
    assert "Data mode: csv" in summary
    assert "MKT: target 0.900" in summary
    assert "Weight floor: 0.01" in summary
    assert "Tax drag" in summary
