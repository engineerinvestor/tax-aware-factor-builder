import logging
from typing import Mapping

import numpy as np
import pandas as pd

from src.models import Fund

logger = logging.getLogger(__name__)


def _normalize_region(region: str) -> str:
    """
    Map user-provided region strings to canonical keys.
    """
    key = region.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "usa": "us",
        "united_states": "us",
        "north_america": "us",
        "global": "developed",
        "world": "developed",
        "intl": "developed_ex_us",
        "international": "developed_ex_us",
        "developed_ex-us": "developed_ex_us",
        "em": "emerging",
        "emerging_markets": "emerging",
    }
    return aliases.get(key, key)


def _select_factor_data_for_fund(fund: Fund, factor_data: pd.DataFrame | Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Choose the factor dataset matching the fund's region when multiple are provided.
    Falls back to a sensible default if no direct match is found.
    """
    if not isinstance(factor_data, Mapping):
        return factor_data

    region_key = _normalize_region(fund.region)
    candidates = [region_key]
    if region_key.startswith("developed") and "developed" in factor_data:
        candidates.append("developed")
    if "developed" not in candidates:
        candidates.append("developed")
    candidates.extend(["us", "emerging"])

    for key in candidates:
        if key in factor_data:
            return factor_data[key]

    return next(iter(factor_data.values()))


def _align_returns_and_factors(
    fund_returns: pd.Series, factor_data: pd.DataFrame, factor_list: list[str]
) -> tuple[pd.Series, pd.DataFrame]:
    if "RF" in factor_data.columns:
        factors = factor_data[factor_list + ["RF"]]
    else:
        factors = factor_data[factor_list]
    aligned = pd.concat([fund_returns, factors], axis=1, join="inner").dropna()
    if aligned.empty:
        return pd.Series(dtype=float), pd.DataFrame()
    aligned_returns = aligned.iloc[:, 0]
    aligned_factors = aligned[factor_list + (["RF"] if "RF" in aligned.columns else [])]
    return aligned_returns, aligned_factors


def estimate_factor_loadings(
    fund_returns: pd.Series, factor_data: pd.DataFrame, factor_list: list[str]
) -> dict:
    """
    Regress excess fund returns on selected factor returns.
    Returns dict with alpha, betas, r2, tstats, n_obs.
    """
    aligned_returns, aligned_factors = _align_returns_and_factors(fund_returns, factor_data, factor_list)
    if aligned_returns.empty:
        raise ValueError("No overlapping data between fund returns and factor data")

    excess = (
        aligned_returns - aligned_factors["RF"]
        if "RF" in aligned_factors.columns
        else aligned_returns
    )
    X = aligned_factors[factor_list].to_numpy()
    y = excess.to_numpy()

    # Add intercept
    X_with_const = np.column_stack([np.ones(len(X)), X])
    beta_hat, residuals, rank, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
    alpha = beta_hat[0]
    betas = beta_hat[1:]

    n = len(y)
    p = X_with_const.shape[1]
    if residuals.size == 0:
        residuals_sum = np.sum((y - X_with_const @ beta_hat) ** 2)
    else:
        residuals_sum = residuals[0]
    sigma2 = residuals_sum / max(n - p, 1)

    xtx_inv = np.linalg.pinv(X_with_const.T @ X_with_const)
    se_beta = np.sqrt(np.diag(sigma2 * xtx_inv))
    tstats_values = beta_hat / se_beta

    ss_total = np.sum((y - y.mean()) ** 2)
    r2 = 1 - residuals_sum / ss_total if ss_total != 0 else 0.0

    return {
        "alpha": float(alpha),
        "betas": {factor_list[i]: float(betas[i]) for i in range(len(factor_list))},
        "r2": float(r2),
        "tstats": {
            "alpha": float(tstats_values[0]),
            **{factor_list[i]: float(tstats_values[i + 1]) for i in range(len(factor_list))},
        },
        "n_obs": int(n),
    }


def apply_factor_estimation_to_all_funds(
    funds: list[Fund], factor_data: pd.DataFrame | Mapping[str, pd.DataFrame], factor_list: list[str]
) -> None:
    """
    Estimate and store factor loadings + regression stats in each Fund.
    Supports either a single factor DataFrame or a region-keyed mapping of DataFrames.
    """
    for fund in funds:
        if fund.historical_returns is None:
            logger.warning("Skipping factor estimation for %s (no historical_returns)", fund.ticker)
            continue
        try:
            factors_for_fund = _select_factor_data_for_fund(fund, factor_data)
            result = estimate_factor_loadings(fund.historical_returns, factors_for_fund, factor_list)
            fund.factor_loadings = result["betas"]
            fund.alpha = result["alpha"]
            fund.regression_stats = {
                "r2": result["r2"],
                "tstats": result["tstats"],
                "n_obs": result["n_obs"],
            }
        except ValueError:
            logger.warning("Skipping factor estimation for %s (no overlapping dates)", fund.ticker)
            continue


def compute_portfolio_factor_loadings(
    weights: np.ndarray, funds: list[Fund], factor_list: list[str]
) -> dict[str, float]:
    """
    Weighted sum of fund factor loadings.
    """
    if len(weights) != len(funds):
        raise ValueError("Weights length must match funds length")
    portfolio = {f: 0.0 for f in factor_list}
    for w, fund in zip(weights, funds):
        for f in factor_list:
            portfolio[f] += w * fund.factor_loadings.get(f, 0.0)
    return portfolio


def compute_portfolio_asset_class_breakdown(weights: np.ndarray, funds: list[Fund]) -> dict[str, dict[str, float]]:
    """
    Compute share of total weight per asset_class, manager, and region.
    """
    if len(weights) != len(funds):
        raise ValueError("Weights length must match funds length")
    breakdown = {
        "asset_class": {},
        "manager": {},
        "region": {},
    }
    for w, fund in zip(weights, funds):
        breakdown["asset_class"][fund.asset_class] = breakdown["asset_class"].get(fund.asset_class, 0.0) + w
        breakdown["manager"][fund.manager] = breakdown["manager"].get(fund.manager, 0.0) + w
        breakdown["region"][fund.region] = breakdown["region"].get(fund.region, 0.0) + w
    return breakdown
