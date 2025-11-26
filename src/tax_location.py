import logging
from typing import List

import cvxpy as cp
import numpy as np
import pandas as pd

from src.models import Account, Fund

logger = logging.getLogger(__name__)


def build_tax_location_problem(
    w_target: np.ndarray,
    funds: List[Fund],
    accounts: List[Account],
    tax_drag_table: pd.DataFrame,
):
    """
    Build LP to allocate fund holdings across accounts minimizing tax drag.
    x[f,a] in dollars.
    """
    n_f = len(funds)
    n_a = len(accounts)
    total_portfolio_value = sum(a.current_value for a in accounts)
    if total_portfolio_value <= 0:
        raise ValueError("Total portfolio value must be positive")

    x = cp.Variable((n_f, n_a))

    constraints = []

    # Account capacity constraints
    for j, account in enumerate(accounts):
        constraints.append(cp.sum(x[:, j]) == account.current_value)

    # Fund target weights constraints
    for i, fund in enumerate(funds):
        constraints.append(cp.sum(x[i, :]) == w_target[i] * total_portfolio_value)

    # Non-negativity and availability
    constraints.append(x >= 0)
    for i, fund in enumerate(funds):
        for j, account in enumerate(accounts):
            if account.available_funds and account.available_funds != ["ANY"] and fund.ticker not in account.available_funds:
                constraints.append(x[i, j] == 0)

    # Build tax drag matrix aligned with funds/accounts
    tax_matrix = np.zeros((n_f, n_a))
    for i, fund in enumerate(funds):
        for j, account in enumerate(accounts):
            key = (fund.ticker, account.type)
            if key in tax_drag_table.index:
                tax_matrix[i, j] = tax_drag_table.loc[key, "tax_drag"]
            else:
                logger.warning("Missing tax drag for %s in %s; assuming 0", fund.ticker, account.type)
                tax_matrix[i, j] = 0.0

    objective = cp.Minimize(cp.sum(cp.multiply(x / total_portfolio_value, tax_matrix)))
    problem = cp.Problem(objective, constraints)
    return problem, x


def solve_tax_location_problem(
    w_target: np.ndarray,
    funds: List[Fund],
    accounts: List[Account],
    tax_drag_table: pd.DataFrame,
) -> np.ndarray:
    problem, x = build_tax_location_problem(w_target, funds, accounts, tax_drag_table)
    installed = {s.upper() for s in cp.installed_solvers()}
    preferred = ["ECOS", "OSQP", "SCS"]
    solver = next((s for s in preferred if s in installed), None)
    solve_kwargs = {"verbose": False}
    if solver:
        if solver == "OSQP":
            solve_kwargs["max_iter"] = 100000
        else:
            solve_kwargs["max_iters"] = 100000
        problem.solve(solver=solver, **solve_kwargs)
    else:
        problem.solve(verbose=False)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        # Try a backup solver if available
        backup = next((s for s in installed if s not in preferred), None)
        if backup:
            problem.solve(solver=backup, verbose=False)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        logger.warning("Tax location optimization failed (%s); falling back to naive allocation", problem.status)
        return build_naive_allocation(w_target, accounts)

    opt_allocation = x.value

    # Fallback: if optimized drag is not better than naive, return naive allocation
    naive = build_naive_allocation(w_target, accounts)
    try:
        opt_total = _compute_total_drag(opt_allocation, tax_drag_table, funds, accounts)
        naive_total = _compute_total_drag(naive, tax_drag_table, funds, accounts)
        if opt_total > naive_total:
            logger.warning("Optimized tax drag higher than naive; using naive allocation")
            return naive
    except Exception:
        logger.warning("Could not compute total drag for comparison; returning optimized allocation")
    return opt_allocation


def build_naive_allocation(w_target: np.ndarray, accounts: List[Account]) -> np.ndarray:
    """
    Pro-rata allocation by account value shares.
    Returns matrix x[f,a] in dollars.
    """
    total_value = sum(a.current_value for a in accounts)
    if total_value <= 0:
        raise ValueError("Total portfolio value must be positive")
    account_shares = np.array([a.current_value / total_value for a in accounts])
    return np.outer(w_target, account_shares * total_value)


def _compute_total_drag(allocation_matrix: np.ndarray, tax_drag_table: pd.DataFrame, funds: List[Fund], accounts: List[Account]) -> float:
    summary = summarize_tax_drag(allocation_matrix, tax_drag_table, funds, accounts)
    return float((summary["tax_drag"] * summary["%Total"]).sum())


def summarize_tax_drag(allocation_matrix: np.ndarray, tax_drag_table: pd.DataFrame, funds: List[Fund], accounts: List[Account]) -> pd.DataFrame:
    """
    Returns DataFrame with Account | Fund | DollarAmount | %Account | %Total | tax_drag
    """
    total_value = allocation_matrix.sum()
    rows = []
    for i, fund in enumerate(funds):
        for j, account in enumerate(accounts):
            dollars = allocation_matrix[i, j]
            if dollars <= 0:
                continue
            drag_key = (fund.ticker, account.type)
            tax_drag = tax_drag_table.loc[drag_key, "tax_drag"] if drag_key in tax_drag_table.index else 0.0
            rows.append(
                {
                    "Account": account.name,
                    "Fund": fund.ticker,
                    "DollarAmount": dollars,
                    "%Account": dollars / account.current_value if account.current_value else 0.0,
                    "%Total": dollars / total_value if total_value else 0.0,
                    "tax_drag": tax_drag,
                }
            )
    return pd.DataFrame(rows)
