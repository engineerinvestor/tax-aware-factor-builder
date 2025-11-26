import numpy as np
import pandas as pd

from src.factors import (
    apply_factor_estimation_to_all_funds,
    compute_portfolio_asset_class_breakdown,
    compute_portfolio_factor_loadings,
    estimate_factor_loadings,
)
from src.models import Fund


def test_estimate_factor_loadings_recovers_betas():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    factor_list = ["MKT_RF", "SMB", "HML"]
    true_alpha = 0.001
    true_betas = np.array([1.1, 0.3, -0.2])

    factors = rng.normal(0, 0.02, size=(len(dates), len(factor_list)))
    rf = np.full(len(dates), 0.0001)
    noise = rng.normal(0, 0.001, size=len(dates))
    excess_returns = true_alpha + factors @ true_betas + noise
    fund_returns = pd.Series(excess_returns + rf, index=dates)

    factor_df = pd.DataFrame(factors, index=dates, columns=factor_list)
    factor_df["RF"] = rf

    result = estimate_factor_loadings(fund_returns, factor_df, factor_list)
    for beta_name, beta_val in result["betas"].items():
        assert abs(beta_val - dict(zip(factor_list, true_betas))[beta_name]) < 0.1
    assert abs(result["alpha"] - true_alpha) < 0.0015
    assert result["n_obs"] == len(dates)


def test_apply_factor_estimation_to_all_funds_sets_attributes():
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    factor_list = ["MKT_RF", "SMB"]
    factor_df = pd.DataFrame({"MKT_RF": 0.01, "SMB": 0.02, "RF": 0.0001}, index=dates)

    returns = pd.Series(0.03, index=dates)
    fund = Fund(
        ticker="T1",
        name="Test1",
        manager="Vanguard",
        asset_class="US_Equity",
        region="US",
        vehicle_type="ETF",
        expense_ratio=0.001,
        historical_returns=returns,
    )
    apply_factor_estimation_to_all_funds([fund], factor_df, factor_list)
    assert fund.factor_loadings
    assert fund.alpha is not None
    assert fund.regression_stats is not None
    assert "r2" in fund.regression_stats


def test_portfolio_factor_and_asset_class_helpers():
    funds = [
        Fund(
            ticker="A",
            name="A",
            manager="Vanguard",
            asset_class="US_Equity",
            region="US",
            vehicle_type="ETF",
            expense_ratio=0.001,
            factor_loadings={"MKT": 1.0, "SMB": 0.5},
        ),
        Fund(
            ticker="B",
            name="B",
            manager="Avantis",
            asset_class="Intl_Equity",
            region="Global",
            vehicle_type="ETF",
            expense_ratio=0.001,
            factor_loadings={"MKT": 0.9, "SMB": 0.1},
        ),
    ]
    weights = np.array([0.6, 0.4])
    factor_list = ["MKT", "SMB"]

    portfolio_betas = compute_portfolio_factor_loadings(weights, funds, factor_list)
    assert abs(portfolio_betas["MKT"] - (0.6 * 1.0 + 0.4 * 0.9)) < 1e-6
    assert abs(portfolio_betas["SMB"] - (0.6 * 0.5 + 0.4 * 0.1)) < 1e-6

    breakdown = compute_portfolio_asset_class_breakdown(weights, funds)
    assert breakdown["asset_class"]["US_Equity"] == 0.6
    assert breakdown["asset_class"]["Intl_Equity"] == 0.4
    assert breakdown["manager"]["Vanguard"] == 0.6
    assert breakdown["manager"]["Avantis"] == 0.4


def test_apply_factor_estimation_respects_regional_factor_sets():
    dates = pd.date_range("2023-01-31", periods=6, freq="ME")
    factor_list = ["MKT_RF"]
    us_mkt = pd.Series([0.01, 0.02, 0.015, 0.005, -0.01, 0.0], index=dates)
    intl_mkt = pd.Series([0.02, 0.01, -0.005, 0.03, 0.0, -0.01], index=dates)
    us_factors = pd.DataFrame({"MKT_RF": us_mkt, "RF": 0.001}, index=dates)
    ex_us_factors = pd.DataFrame({"MKT_RF": intl_mkt, "RF": 0.001}, index=dates)

    us_returns = pd.Series(us_factors["RF"] + 1.0 * us_factors["MKT_RF"], index=dates)
    intl_returns = pd.Series(ex_us_factors["RF"] + 0.5 * ex_us_factors["MKT_RF"], index=dates)

    funds = [
        Fund(
            ticker="US",
            name="US Fund",
            manager="Manager",
            asset_class="US_Equity",
            region="US",
            vehicle_type="ETF",
            expense_ratio=0.0,
            historical_returns=us_returns,
        ),
        Fund(
            ticker="INTL",
            name="Intl Fund",
            manager="Manager",
            asset_class="Intl_Equity",
            region="Developed ex-US",
            vehicle_type="ETF",
            expense_ratio=0.0,
            historical_returns=intl_returns,
        ),
    ]

    factor_map = {"us": us_factors, "developed_ex_us": ex_us_factors}
    apply_factor_estimation_to_all_funds(funds, factor_map, factor_list)

    assert abs(funds[0].factor_loadings["MKT_RF"] - 1.0) < 1e-6
    assert abs(funds[1].factor_loadings["MKT_RF"] - 0.5) < 1e-6
