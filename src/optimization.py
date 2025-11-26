import logging

import cvxpy as cp
import numpy as np

from src.models import Fund, TargetPortfolioSpec

logger = logging.getLogger(__name__)


def _group_indexes(funds: list[Fund], predicate) -> list[int]:
    return [i for i, f in enumerate(funds) if predicate(f)]


def build_pre_tax_optimization(funds: list[Fund], target_spec: TargetPortfolioSpec):
    """
    Build an optimization problem (cvxpy) to find fund weights.
    """
    n = len(funds)
    w = cp.Variable(n)

    constraints = [
        cp.sum(w) == 1.0,
        w >= target_spec.min_weight_per_fund,
        w <= target_spec.max_weight_per_fund,
    ]

    # Asset class constraints (minimums)
    us_idx = _group_indexes(funds, lambda f: f.asset_class == "US_Equity")
    intl_idx = _group_indexes(funds, lambda f: f.asset_class == "Intl_Equity")
    bond_idx = _group_indexes(funds, lambda f: f.asset_class == "Bond")
    alt_idx = _group_indexes(funds, lambda f: f.asset_class == "Alt")
    reit_idx = _group_indexes(funds, lambda f: f.asset_class == "REIT")

    equity_total = target_spec.equity_weight
    if us_idx:
        constraints.append(
            cp.sum(cp.hstack([w[i] for i in us_idx])) >= equity_total * target_spec.us_equity_share_of_equity
        )
    if intl_idx:
        constraints.append(
            cp.sum(cp.hstack([w[i] for i in intl_idx])) >= equity_total * target_spec.intl_equity_share_of_equity
        )
    if bond_idx:
        constraints.append(cp.sum(cp.hstack([w[i] for i in bond_idx])) >= target_spec.bond_weight)
    if alt_idx:
        constraints.append(cp.sum(cp.hstack([w[i] for i in alt_idx])) >= target_spec.alts_weight)
    if reit_idx and target_spec.reit_share_of_equity is not None:
        constraints.append(cp.sum(cp.hstack([w[i] for i in reit_idx])) >= equity_total * target_spec.reit_share_of_equity)

    # Manager constraints
    if target_spec.min_weight_per_manager:
        for manager, value in target_spec.min_weight_per_manager.items():
            idxs = _group_indexes(funds, lambda f, m=manager: f.manager == m)
            if idxs:
                constraints.append(cp.sum(cp.hstack([w[i] for i in idxs])) >= value)
    if target_spec.max_weight_per_manager:
        for manager, value in target_spec.max_weight_per_manager.items():
            idxs = _group_indexes(funds, lambda f, m=manager: f.manager == m)
            if idxs:
                constraints.append(cp.sum(cp.hstack([w[i] for i in idxs])) <= value)

    # Region constraints
    if target_spec.min_weight_per_region:
        for region, value in target_spec.min_weight_per_region.items():
            idxs = _group_indexes(funds, lambda f, r=region: f.region == r)
            if idxs:
                constraints.append(cp.sum(cp.hstack([w[i] for i in idxs])) >= value)
    if target_spec.max_weight_per_region:
        for region, value in target_spec.max_weight_per_region.items():
            idxs = _group_indexes(funds, lambda f, r=region: f.region == r)
            if idxs:
                constraints.append(cp.sum(cp.hstack([w[i] for i in idxs])) <= value)

    # Factor objective
    factor_list = target_spec.factor_list
    target_vec = np.array([target_spec.factor_targets.get(f, 0.0) for f in factor_list])
    factor_matrix = np.zeros((len(factor_list), n))
    for j, fund in enumerate(funds):
        missing = [f for f in factor_list if f not in fund.factor_loadings]
        if missing:
            logger.warning("Fund %s missing factor loadings for %s; using zeros", fund.ticker, ",".join(missing))
        for i, factor in enumerate(factor_list):
            factor_matrix[i, j] = fund.factor_loadings.get(factor, 0.0)
    portfolio_betas = factor_matrix @ w
    objective = cp.Minimize(cp.sum_squares(portfolio_betas - target_vec))

    problem = cp.Problem(objective, constraints)
    return problem, w


def solve_pre_tax_optimization(funds: list[Fund], target_spec: TargetPortfolioSpec) -> np.ndarray:
    """
    Solve the optimization and return optimal weights w.
    """
    problem, w = build_pre_tax_optimization(funds, target_spec)
    installed = {s.upper() for s in cp.installed_solvers()}
    preferred = ["ECOS", "OSQP", "SCS", "CLARABEL"]
    solver = next((s for s in preferred if s in installed), None)
    problem.solve(solver=solver, verbose=False)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise ValueError(f"Optimization failed with status {problem.status}")
    weights = w.value
    if target_spec.weight_floor and target_spec.weight_floor > 0:
        weights = _apply_weight_floor(weights, target_spec.weight_floor)
    return weights


def _apply_weight_floor(weights: np.ndarray, floor: float) -> np.ndarray:
    """
    Zero out weights with absolute value below `floor` and renormalize the remainder.
    If all weights fall below the floor, return the original weights to avoid divide-by-zero.
    """
    mask = np.abs(weights) >= floor
    if not mask.any():
        logger.warning("All weights below floor %.6f; returning unadjusted weights", floor)
        return weights
    adjusted = np.where(mask, weights, 0.0)
    total = adjusted.sum()
    if total <= 0:
        logger.warning("Non-positive weight sum after floor applied; returning unadjusted weights")
        return weights
    return adjusted / total
