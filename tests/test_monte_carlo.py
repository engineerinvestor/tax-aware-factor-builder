import numpy as np
import pandas as pd

from src.models import Account, Fund
from src.monte_carlo import (
    simulate_after_tax_wealth,
    simulate_factor_premia,
    simulate_portfolio_paths,
)


def test_simulate_factor_premia_shape_and_stats():
    factor_list = ["MKT_RF", "SMB"]
    mu = np.array([0.01, 0.005])
    cov = np.array([[0.0004, 0.0], [0.0, 0.0001]])
    sims = simulate_factor_premia(T=12, n_sims=100, factor_list=factor_list, mu=mu, cov=cov)
    assert sims.shape == (100, 12, len(factor_list))
    # Rough sanity on mean closeness
    approx_mean = sims.mean(axis=(0, 1))
    assert np.allclose(approx_mean, mu, atol=0.01)


def test_simulate_portfolio_paths():
    factor_list = ["MKT_RF", "SMB"]
    premia = np.zeros((10, 5, len(factor_list)))
    premia[:, :, 0] = 0.01
    premia[:, :, 1] = 0.0
    betas = {"MKT_RF": 1.0, "SMB": 0.5}
    rf = 0.001
    paths = simulate_portfolio_paths(premia, betas, rf)
    assert paths.shape == (10, 5)
    assert np.allclose(paths, 0.011)


def test_simulate_after_tax_wealth_shapes_and_sign():
    funds = [
        Fund(
            ticker="F1",
            name="Fund1",
            manager="Vanguard",
            asset_class="US_Equity",
            region="US",
            vehicle_type="ETF",
            expense_ratio=0.001,
        ),
        Fund(
            ticker="F2",
            name="Fund2",
            manager="Avantis",
            asset_class="Bond",
            region="US",
            vehicle_type="ETF",
            expense_ratio=0.001,
        ),
    ]
    accounts = [
        Account(name="Taxable", type="Taxable", current_value=100000, available_funds=["ANY"]),
        Account(name="401k", type="TaxDeferred", current_value=50000, available_funds=["ANY"]),
    ]
    tax_idx = pd.MultiIndex.from_tuples([("F1", "Taxable"), ("F1", "TaxDeferred"), ("F2", "Taxable"), ("F2", "TaxDeferred")], names=["Fund", "AccountType"])
    tax_drag_table = pd.DataFrame({"tax_drag": [0.01, 0.0, 0.02, 0.0]}, index=tax_idx)
    alloc = np.array([[60000, 0], [40000, 50000]])
    portfolio_returns = np.full((20, 10), 0.01)  # 1% per period
    wealth = simulate_after_tax_wealth(150000, alloc, tax_drag_table, funds, accounts, portfolio_returns)
    assert wealth.shape == (20,)
    assert np.all(wealth > 0)
