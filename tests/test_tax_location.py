import numpy as np
import pandas as pd

from src.models import Account, Fund
from src.tax_location import (
    build_naive_allocation,
    build_tax_location_problem,
    solve_tax_location_problem,
    summarize_tax_drag,
)


def _funds():
    return [
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


def _accounts():
    return [
        Account(name="Taxable", type="Taxable", current_value=100000, available_funds=["ANY"]),
        Account(name="401k", type="TaxDeferred", current_value=50000, available_funds=["ANY"]),
    ]


def _tax_table():
    idx = pd.MultiIndex.from_tuples([("F1", "Taxable"), ("F1", "TaxDeferred"), ("F2", "Taxable"), ("F2", "TaxDeferred")], names=["Fund", "AccountType"])
    return pd.DataFrame({"tax_drag": [0.01, 0.0, 0.02, 0.0]}, index=idx)


def test_build_tax_location_problem_shapes():
    funds = _funds()
    accounts = _accounts()
    tax_table = _tax_table()
    w_target = np.array([0.6, 0.4])
    problem, x = build_tax_location_problem(w_target, funds, accounts, tax_table)
    assert x.shape == (len(funds), len(accounts))
    assert len(problem.constraints) > 0


def test_solve_tax_location_problem_prefers_tax_deferred():
    funds = _funds()
    accounts = _accounts()
    tax_table = _tax_table()
    w_target = np.array([0.6, 0.4])
    allocation = solve_tax_location_problem(w_target, funds, accounts, tax_table)
    opt_summary = summarize_tax_drag(allocation, tax_table, funds, accounts)
    opt_total = (opt_summary["tax_drag"] * opt_summary["%Total"]).sum()

    naive = build_naive_allocation(w_target, accounts)
    naive_summary = summarize_tax_drag(naive, tax_table, funds, accounts)
    naive_total = (naive_summary["tax_drag"] * naive_summary["%Total"]).sum()

    assert opt_total <= naive_total + 1e-6


def test_build_naive_allocation_dimensions():
    accounts = _accounts()
    w_target = np.array([0.6, 0.4])
    naive = build_naive_allocation(w_target, accounts)
    assert naive.shape == (2, 2)
    assert abs(naive.sum() - sum(a.current_value for a in accounts)) < 1e-6


def test_summarize_tax_drag():
    funds = _funds()
    accounts = _accounts()
    tax_table = _tax_table()
    allocation = np.array([[60000, 0], [40000, 50000]])
    df = summarize_tax_drag(allocation, tax_table, funds, accounts)
    assert not df.empty
    assert set(df.columns) == {"Account", "Fund", "DollarAmount", "%Account", "%Total", "tax_drag"}
